# RAG Ops Guard

Local-first conversational ReAct agent with optional document RAG.

The product is **not** an Ops-specific router. It is one conversational agent that can answer directly or choose document tools when the request depends on an ingested corpus.

## Current architecture

```text
Chainlit ───────┐
Gradio ─────────┤
REST /v1/query ─┼──> app.conversation_agent()
                │            │
                │            v
                │     ConversationAgent
                │       - SafetyGuard
                │       - conversation memory
                │       - ReAct/tool loop
                │       - segmented response contract
                │            │
                │            v
                │       Tool registry
                │       ├─ search_documents
                │       └─ list_documents
                │            │
                │            v
                │     hybrid retrieval
                │       dense + BM25 + RRF
                │       domain relevance (pre-policy)
                │       optional governance resolver
                │       reranker/evidence admission
                │       grounded relevance (post-policy)
                │
                └──────── same core
```

There is no second conversational pipeline and no `src/rag_ops_guard/graph/` package. The canonical owners are `src/rag_ops_guard/app.py` and `src/rag_ops_guard/agent/conversation.py`.

## Response contract

Every public answer is segmented. Grounding is derived from validated citations, not from a router label.

- `answered_grounded`: every segment has admitted citations.
- `answered_mixed`: grounded and ungrounded segments coexist.
- `answered_ungrounded`: no segment has a citation.
- `clarification_required`, `safety_blocked`, `error`: non-answer outcomes.

A citation is accepted only when its chunk ID came from admitted `search_documents` evidence in the current turn.

## Document ingestion

Supported now:

- Markdown (`.md`)
- plain text (`.txt`)

YAML front matter is optional. Without it, ingestion derives a deterministic document identity, title and version. With front matter, richer governance metadata such as status, authority, environment and supersession is preserved. Governance rules apply only when those fields exist; generic documents do not need fake Ops metadata.

## Retrieval correctness

Two different relevance signals are measured:

1. `domain_relevance`: corpus affinity on raw fused candidates, before governance resolution.
2. `grounded_relevance`: evidence support after resolver/target checks.

Both floors are calibrated from three classes in `evaluation/datasets/retrieval-calibration-v2.json`:

- `grounded`
- `in_domain_unanswerable`
- `out_of_domain`

A low domain score cannot produce admitted grounded evidence even when the grounded floor is numerically lower.

## Security boundary

Deterministic secret/policy-bypass checks run before the relevance probe, model and tools.

Retrieved document content is marked `UNTRUSTED_DOCUMENT_DATA`. It is preserved as evidence but must never be followed as an instruction. Only admitted chunks enter the tool observation, preventing rejected chunks from becoming model context or citations.

## Local model split

- generation/runtime/judge: **Qwen3-4B**, llama.cpp CPU
- embeddings: Qwen3-Embedding-0.6B, OpenVINO on Intel iGPU for the target profile
- reranker: Qwen3-Reranker-0.6B, OpenVINO on Intel iGPU
- object/vector infrastructure: Floci + S3/S3 Vectors-compatible adapters

The generation model artifact is pinned by SHA256 in `scripts/download_models.py`.

## Evaluation

Blocking deterministic gates cover:

- status/segment contract
- citation validity and segment integrity
- architecture fitness and zero unreachable pipeline code
- deterministic/adversarial security
- unit/property/integration tests
- strict mypy
- dependency/security audit
- CDK test/build/synth

Ruff formatting/lint remains visible but is **advisory during the architecture pivot**; it does not block architecture work.

RAGAS adds mean and per-case gates, but it may become a release gate only after exactly 10 real human-labelled cases are compared with the same Qwen3-4B judge. If agreement is 6/10 or lower, RAGAS is informational and deterministic gates remain authoritative. The repository deliberately does not fabricate that calibration result.

## Evaluation fixtures are not product logic

The AcmePay/Payments/Calypso corpus exists to exercise conflicts, deprecated documents, missing evidence, security and prompt injection. Those names must not appear in product control-flow code.

## Useful commands

```bash
make setup
make models
make local-up
make test
make types
make beta-react
make eval
make eval-judge-calibrate
make release-check
```

`make lint` is available as a strict manual style check. `make ci` and `make release-check` use advisory lint plus blocking correctness checks.

## Release caveat

Hosted CI can validate software contracts, but two target-machine gates remain external to this repository state:

1. run the full Qwen3-4B + OpenVINO local gate on the target Fedora/Tiger Lake machine;
2. produce the real 10-case human/judge calibration artifact.

Do not claim final release readiness until those two measurements pass.
