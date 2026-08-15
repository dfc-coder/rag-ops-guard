# Conversational Grounding v2

## Goal

Make grounding decisions explicit and deterministic instead of delegating every turn-level RAG
choice to the local 2B generation model.

The generation model remains responsible for understanding and producing language. The application
owns whether an internal fact must be retrieved, whether recent evidence can be reused, and how an
elliptical follow-up is resolved into a standalone retrieval query.

## Components

### `ConversationState`

Stored per thread independently from the raw LangChain message history.

It tracks:

- successful turn index;
- current grounded topic/system/environment;
- last turn policy;
- last grounded standalone query;
- active `EvidenceWindow`;
- whether the thread currently has reusable grounding.

Message history and grounding state are committed together only after a successful turn. A timeout,
cancel, truncation, or backend error commits neither.

### `TurnPolicyEngine`

Classifies each turn into one of four policies:

| Policy | Meaning | Tool behavior |
|---|---|---|
| `DIRECT` | general chat, coding, social, general definitions | no RAG tool |
| `REUSE_EVIDENCE` | summarize/rephrase/explain/format evidence already retrieved | no new retrieval |
| `RETRIEVE` | a new internal operational fact is required | deterministic `search_knowledge` |
| `LIST_KNOWLEDGE` | explicit request for available documentation | deterministic `list_knowledge` |

The model does not decide these transitions.

### `FollowupResolver`

When a short follow-up depends on a prior internal topic, it combines the last grounded query with
the current user turn before retrieval.

Example:

```text
previous grounded query: Cuantos reintentos permite Calypso?
current turn: Y despues del tercero?

resolved retrieval query:
Cuantos reintentos permite Calypso. Follow-up: Y despues del tercero?
```

If the user explicitly names a new internal target, the current message is used directly rather than
carrying the previous topic forward.

### `EvidenceWindow`

A successful `search_knowledge` result is retained for a small number of successful turns. The
window contains the exact admitted source text used by the agent.

It is reused only for transformations such as:

- summarize;
- make shorter;
- explain;
- translate;
- reformat;
- provide an example based on the existing evidence.

A request for a new internal fact triggers retrieval instead of reuse.

The evidence window expires by successful turn count, not wall-clock time. The default is four
subsequent turns.

## Deterministic tool execution

`RETRIEVE` and `LIST_KNOWLEDGE` are represented as synthetic, valid LangChain tool calls generated
by application policy. LangGraph's normal `ToolNode` executes them, so tool activity remains visible
and auditable in the message history.

The local Qwen generation model therefore cannot accidentally skip RAG for a turn that policy has
classified as requiring internal evidence.

## Turn examples

```text
"Hola"
  -> DIRECT

"Escribe una función Fibonacci en C"
  -> DIRECT

"Cuantos reintentos permite Calypso?"
  -> RETRIEVE

"Y despues del tercero?"
  -> RETRIEVE
  -> standalone query carries Calypso/retry context

"Resumilo en una linea"
  -> REUSE_EVIDENCE

"Que es exponential backoff?"
  -> DIRECT

"Que documentacion hay?"
  -> LIST_KNOWLEDGE
```

## Transactional behavior

For each turn:

```text
snapshot message history + ConversationState
        |
        v
TurnPolicyEngine
        |
        v
LangGraph / optional ToolNode / generation
        |
   +----+----+
   |         |
success    failure
   |         |
commit      discard
messages    current turn
+ state     state unchanged
```

This preserves the recovery guarantees introduced by the streaming-resilience work.

## Observability

Each turn logs:

```text
thread_id
turn
policy
reason
topic
grounded
retrieval_query
```

Successful commits also log the resulting grounded state and number of evidence sources. This makes
"why did the agent use RAG?" and "why did it reuse evidence?" inspectable without exposing internal
plumbing in the client UI.

## Acceptance contract

`make chainlit-gate` must validate all of the following against the target local runtime:

1. direct chat/code stays tool-free and streams;
2. initial Calypso fact uses `RETRIEVE` and answers three retries;
3. `ConversationState` stores an evidence window;
4. `Y despues del tercero?` uses `RETRIEVE` again and answers Treasury Integrations;
5. `Resumilo en una linea` uses `REUSE_EVIDENCE` with zero new tool calls;
6. failures/truncations commit neither message history nor grounding state;
7. Chainlit still imports and runs on top of the same agent API.

## Non-goals

This version does not add durable history across application restarts, a second classifier LLM, a
new embedding/reranking model, new retrieval thresholds, or a periodic "RAG every N turns" rule.
Grounding is decided from semantics and explicit thread state, never from message count alone.
