# Phase 7 — Generic Reflective ReAct Agent

## Goal

Phase 7 turns the conversational core into a domain-agnostic agent.

RAG is no longer part of agent control flow. It is one optional tool capability composed by the
application, exactly like any future calculator, web, calendar, GitHub, or business tool.

The governing rule is:

> **ReAct decides and acts; the runtime verifies; reflection corrects bounded failures.**

## Runtime flow

```text
user
  |
  v
ConversationAgent
  |
  v
actor model
  |
  +----------------------------> final answer
  |
  v
tool call
  |
  v
ToolRuntime
  |
  +--> deterministic call validation
  +--> request-context binding
  +--> tool execution
  +--> deterministic result verification
  |
  +-- valid -------------------> observation --> actor model
  |
  +-- retryable failure
          |
          v
      Reflection
          |
          v
   corrective observation
          |
          v
      actor model
```

The model never needs an explicit domain router. It decides whether to answer or call one of the
injected tools. A tool observation always returns to the actor model, which may answer or select
another tool.

## 7.1 Generic ConversationAgent

`ConversationAgent` depends on the generic `Tool` and `ToolCallingModel` ports. It does not import
`KnowledgeSearch`, `KnowledgeCatalog`, `SearchDocumentsTool`, or `ListDocumentsTool`.

Document capabilities are assembled in `app.py` and injected into the agent.

## 7.2 Real sequential ReAct loop

The canonical loop is now:

```text
model -> tool -> observation -> model -> [tool -> observation -> model] -> answer
```

A tool result does not terminate reasoning automatically. The next model turn decides whether the
result is sufficient.

The default tool-bearing round limit is `3`.

## 7.3 Injected generic tools

The constructor accepts `tools: list[Tool]`. No tool name is required by the core.

A fake or future capability can be injected without modifying `ConversationAgent` as long as it
implements:

```python
class Tool(Protocol):
    name: str
    description: str

    def schema(self) -> dict[str, Any]: ...
    def invoke(self, arguments: dict[str, Any]) -> ToolResult: ...
```

## 7.4 Tool runtime

`ToolRuntime` owns execution. It receives a `ToolCall`, validates it, binds `QueryContext` for the
execution scope, invokes the tool, and returns a `ToolExecution`.

The agent does not execute tool implementations directly.

## 7.5 Deterministic verifier

`ToolVerifier` validates conditions software can know deterministically, including:

- unknown tool names;
- required arguments;
- unexpected arguments when `additionalProperties` is false;
- basic JSON argument types;
- malformed/unsupported root tool schemas;
- explicit tool failure results;
- tool execution exceptions.

A verifier result declares whether the failure is retryable. This keeps validation outside the
model prompt.

## 7.6 Triggered reflection

Reflection is not a permanent second pass. It is invoked only when deterministic verification
reports a retryable failure and the reflection budget is still available.

The reflector receives only:

- the current user goal;
- the failed tool call;
- the failure result;
- available tool descriptors.

It returns one short correction that is attached to the failed tool observation. The ReAct actor
then decides the corrected next action.

The default reflection budget is `1` per user turn.

## 7.7 Bounded retry

Two independent bounds prevent agent loops:

```text
MAX_TOOL_ROUNDS = 3
MAX_REFLECTIONS = 1
```

Exceeding the tool-round budget fails the current turn without committing partial conversation
state. The previous conversation remains available.

## 7.8 Multi-tool evaluation

Deterministic tests prove that the same agent can:

```text
lookup_rate
  -> observation
  -> calculator
  -> observation
  -> answer
```

The test tools have unrelated names and schemas. This is intentional: the contract tests the ReAct
core rather than RAG behavior.

Additional cases cover invalid arguments, reflection, corrected retry, and bounded failure.

## 7.9 Multi-turn evaluation

A deterministic conversation test proves that the public answer from one turn is retained as
visible history and can drive tool selection in the next turn:

```text
user:      What is 10 + 20?
assistant: 30
user:      And double that?
assistant: calculator("30 * 2") -> 60
```

Stale tool protocol messages from prior turns remain excluded from future prompts; visible user and
assistant conversation is preserved.

## Removed RAG-first behavior

Phase 7 intentionally removes two previous behaviors from the conversational core:

1. every user turn no longer performs a background relevance probe;
2. a scoped `QueryContext` no longer forces `search_documents` when the model chooses a direct
   response.

`QueryContext` is still available to tools while they execute, so document retrieval can continue to
apply system/environment/API-version filters after the model actually selects that capability.

## Response and citation compatibility

The existing segmented public response contract remains in place during this phase to avoid mixing
agent-loop changes with an API contract migration.

When tools return source objects, citation IDs are validated against those admitted sources. Direct
and non-evidence statements use no citations.

This structured final-response pass is deliberately separate from the ReAct control loop and can be
optimized in a later phase.

## Acceptance matrix

| Case | Expected behavior |
|---|---|
| Direct answer | zero tools, no retrieval side effect |
| One tool | model -> tool -> model -> answer |
| Two tools | model -> tool A -> model -> tool B -> model -> answer |
| Unknown tool | deterministic failure, at most one reflection |
| Invalid arguments | tool not invoked, reflection may correct and retry |
| Tool exception | normalized failure observation, bounded reflection |
| Round overflow | turn fails without unbounded execution |
| Scoped context | context reaches selected tool only; never forces a tool |
| Multi-turn follow-up | visible prior answer informs the next ReAct decision |
| Safety block | deterministic guard runs before model and tools |

## Physical validation

The deterministic tests establish the architecture and loop semantics independently of model
quality. The separate physical evaluation on the trusted `rag-e2e` runner is still required to
measure how reliably the configured local Qwen model chooses and chains real tools.
