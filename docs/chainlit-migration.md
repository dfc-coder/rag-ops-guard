# Chainlit UI

Chainlit is the primary local client for the canonical `ConversationAgent`.

## Invariants

Changing the UI must not create another agent or retrieval pipeline.

```text
Chainlit ───────┐
Gradio ─────────┼──> rag_ops_guard.app.conversation_agent()
REST /v1/query ─┘                 |
                                  v
                           ConversationAgent
```

The UI may render status, streamed text, sources and failure states. It does not own routing, safety, retrieval, memory or citation policy.

## Runtime profile

Target local split:

- CPU: Python orchestration, Floci, BM25/RRF and Qwen3-4B generation/tool calling.
- Intel Iris Xe/OpenVINO: Qwen3 embedding and reranker workloads.

`make beta-react` starts the local dependencies and runs Chainlit on port 8000 by default.

## Streaming contract

The application core exposes `ConversationAgent.stream()`. Chainlit consumes its framework-neutral events:

- `status`
- `token` where available
- `done`
- `error`

Tool execution remains owned by the core. The UI must never infer a route from text or fabricate a document tool call.

## Grounding/source rendering

Final answer segments carry validated citations. Chainlit renders source metadata from the terminal event but does not decide whether a claim is grounded.

`search_documents` source text is untrusted document data. Rendering it in the UI does not change that security boundary.

## Failure behavior

A failed/partial turn does not commit corrupted conversation history. The UI may show the error, but transactional history behavior is implemented by `ConversationAgent`.

## Migration status

The migration is complete at the architecture level:

- Chainlit and Gradio resolve the same singleton application core as REST.
- legacy `ReactAgent`, semantic router/gate and `graph/` orchestration have been removed.
- no UI-specific RAG policy exists.

The remaining release prerequisites are target-machine Qwen3-4B/OpenVINO validation and real 10-case judge-human calibration; they are not UI migration tasks.
