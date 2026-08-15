# Chainlit migration / unified beta

## Decision

Chainlit is the client-facing UI for the next beta, but it owns no agent policy.

All clients resolve the same application core:

```text
Chainlit ---------\
Gradio -----------+--> rag_ops_guard.app.conversation_agent()
REST /v1/query ---/
                         |
                         v
                  ConversationAgent
                         |
                 LangGraph ReAct loop
                         |
             search_documents / list_documents
```

The UI is an adapter: it renders status, streamed tokens, terminal response metadata and the exact
sources returned by the canonical agent. It does not inspect private agent memory or run a second
retrieval pass.

## Runtime

- Qwen 3.5 2B generation on CPU
- LangGraph orchestration on CPU
- Floci + BM25/RRF
- OpenVINO embeddings and reranking on Intel Iris Xe
- transactional per-thread in-process conversation memory
- deterministic safety before model/tool execution
- deterministic post-retrieval evidence admission

## Grounding behavior

The model is bound to two document tools:

- `search_documents(query)`
- `list_documents()`

Normal chat, public knowledge and self-contained code should use zero document tools. Corpus-specific
questions use `search_documents`; unsupported corpus facts end as `insufficient_evidence` with no
citations. Grounded answers expose the exact admitted chunks as source cards.

## Freeze validation

Run on the target Fedora/Tiger Lake machine:

```bash
make chainlit-gate
```

`make chainlit-gate` delegates to `scripts/beta_freeze_gate.sh` and validates the single-pipeline
architecture, response contract, local runtime, retrieval admission, direct/no-tool behavior, grounded
Calypso behavior, unsupported corpus behavior, safety and Chainlit import.

Then run:

```bash
make chainlit-beta
```

Manual checks before freezing:

1. a short Python coding request streams without document tools or citations;
2. `¿Cuántos reintentos permite Calypso?` uses documents, answers three and shows source cards;
3. `¿Y después del tercero?` re-queries documents and answers Treasury Integrations;
4. a SAP-specific fact absent from the corpus returns insufficient evidence rather than Calypso data;
5. Stop does not corrupt the previous successful conversation;
6. Retry works after a recoverable model/backend failure;
7. environment filtering still affects document retrieval.

## Scope boundary for this beta

This freeze unifies orchestration and tool/RAG behavior. It intentionally does **not** replace the
existing Markdown/YAML ingestion contract with multi-format PDF/DOCX/HTML ingestion. Generic ingestion
is the next isolated migration after this client beta is frozen; mixing it into the orchestration
freeze would make failures harder to attribute.
