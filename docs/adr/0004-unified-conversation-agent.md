# ADR-0004: One ConversationAgent for every client

## Status

Accepted for the next client beta freeze candidate.

## Context

The repository accumulated two application-level orchestration systems:

- Chainlit/Gradio called `ReactAgent` directly.
- REST `/v1/query` called `graph.conversational_agent.ConversationalAgent` through `query_workflow()`.

Those paths shared retrieval infrastructure but owned different routing, memory, safety and response
contracts. A fix validated in one client could therefore be absent from another client.

The newer ReAct path also stopped binding tools to the generation model. A deterministic
`TurnPolicyEngine` fabricated `search_knowledge` and `list_knowledge` tool calls before generation,
which made the graph ReAct-shaped without giving the model actual tool choice.

## Decision

`rag_ops_guard.agent.conversation.ConversationAgent` is the single application agent.

```text
Chainlit ---------\
Gradio -----------+--> app.conversation_agent() --> ConversationAgent --> tools
REST /v1/query ---/
```

`app.query_workflow()` remains only as a compatibility name and returns the same cached
`ConversationAgent` object. It must never construct another workflow.

The canonical agent:

1. runs deterministic `SafetyGuard` before model/tool execution;
2. binds `search_documents` and `list_documents` to the local generation model;
3. lets the model decide whether a document tool is needed;
4. executes tools through the LangGraph `agent -> tools -> agent` loop;
5. treats retrieval admission as deterministic post-retrieval evidence policy;
6. converts unsupported document retrieval into deterministic `insufficient_evidence` before another
   model generation can invent the missing corpus fact;
7. commits conversation history only after successful completion of a turn;
8. exposes the same `invoke`/`stream` behavior to every client adapter.

## Grounding contract

The response contract now distinguishes:

- `answered`: a grounded knowledge answer; knowledge-route citations are required;
- `answered_ungrounded`: normal model/chat/code/catalog output; citations are forbidden;
- `insufficient_evidence`: a corpus-specific request was searched but could not be supported;
- `safety_blocked`: deterministic safety stopped the turn before model/tool execution;
- `error`: recoverable runtime failure.

This makes a direct code answer carrying unrelated document citations invalid by contract.

## Retrieval calibration

The old `positive/negative` calibration dataset mixed two different application behaviors under
`negative`. The v2 dataset keeps three semantic classes:

- `grounded`;
- `in_domain_unanswerable`;
- `out_of_domain`.

The retrieval admission floor intentionally treats the final two classes as non-admissible. Their
application behavior is not separated by another scope classifier: the model's tool decision separates
normal direct questions from document-dependent questions, while post-retrieval evidence admission
separates supported from unsupported corpus claims.

## Architecture fitness rule

`tests/architecture/test_single_query_pipeline.py` fails if a runtime entrypoint imports either legacy
orchestrator or if the canonical agent stops binding tools to the model.

Legacy modules may remain temporarily for historical tests/migration support, but no production client
or handler may depend on them.

## Beta freeze gate

`scripts/beta_freeze_gate.sh` is the freeze proof bundle. It requires:

- repository formatting/lint/type checks;
- single-pipeline architecture tests;
- response-contract tests;
- adversarial deterministic guards;
- local Floci/Qwen/OpenVINO runtime and corpus integrity;
- calibrated post-retrieval admission;
- direct code/general knowledge with zero document tools;
- grounded Calypso retrieval with citations;
- grounded conversational follow-up;
- unsupported SAP corpus fact -> `insufficient_evidence`;
- catalog tool use;
- deterministic secret-extraction block;
- Chainlit import against the canonical application agent.

The branch is not eligible to freeze until repository CI is green and this gate passes on the target
Fedora/Tiger Lake machine. ADR-0001 through ADR-0003 remain historical design records; this decision
supersedes their routing architecture for runtime clients.
