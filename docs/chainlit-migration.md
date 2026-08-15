# Chainlit migration

## Decision

Chainlit is the client-facing UI candidate for the ReAct beta. Gradio remains available as a developer/debug fallback during the migration window.

The agent and RAG architecture do not change:

- Qwen 3.5 2B generation on CPU
- LangGraph/ReAct orchestration
- Floci + BM25/RRF
- OpenVINO embeddings and reranking on Intel Iris Xe
- transactional conversation memory in `ReactAgent`

## Client UI contract

The Chainlit surface provides:

- immediate activity feedback
- token streaming
- normal chat and direct-code responses without forced RAG
- RAG activity shown as a compact tool step
- exact evidence used by `search_knowledge` exposed as side-panel source cards
- environment selection (`any`, `production`, `staging`)
- recoverable errors with Retry
- Stop support without committing an interrupted turn
- current-chat history and multi-turn agent memory
- New Chat through Chainlit's native UI
- local runtime diagnostics on demand
- dark/light theme and responsive layout

### History scope

The local beta intentionally keeps history at the active-session/application-memory level. Durable browsing/resume of conversations across application restarts is not enabled in this migration because Chainlit requires a persistence data layer plus authentication for that experience. That can be added independently without changing the ReAct/RAG core.

## Retrieval regression fixed before migration

The Spanish short query `¿Cuántos reintentos permite Calypso?` exposed a false abstention even though `Calypso Timeout Runbook` contains the answer.

`ResilientKnowledgeSearch` now:

1. runs the normal instruction-wrapped knowledge retrieval path;
2. logs dense, lexical, fused, reranker and admitted stages;
3. only when that path has no admitted support, retries recall once with the raw user query;
4. still applies the same resolver and reranker admission rules;
5. never lowers the relevance threshold or bypasses fail-closed named-target checks.

This fallback is a recall repair, not a relaxation of grounding.

## Migration gate

Run on the target Fedora/Tiger Lake machine:

```bash
make chainlit-gate
```

The headless gate validates:

- corpus/vector integrity
- streaming and transactional rollback regressions
- short direct-code streaming with zero RAG tools
- exact Calypso retrieval under the default `any environment` context
- final answer contains the retry count
- follow-up preserves the Treasury Integrations escalation
- Chainlit application syntax/import configuration

Then run the UI candidate:

```bash
make chainlit-beta
```

Open `http://127.0.0.1:8001` and manually confirm:

1. direct code streams normally;
2. Calypso returns three retries and displays source cards;
3. `¿Y después del tercero?` returns Treasury Integrations;
4. Stop interrupts a long generation without corrupting prior memory;
5. a recoverable failure exposes Retry;
6. environment settings work;
7. Local Status reports the four local runtime components.

## Promotion

After the gate passes, the branch is already wired so:

```bash
make beta-react
```

launches Chainlit on `127.0.0.1:8000`.

Gradio remains available during the transition with:

```bash
make gradio-react
```

Shutdown remains unchanged:

```bash
make local-down
```
