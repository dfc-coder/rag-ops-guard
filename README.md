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

Both floors are calibrated from three labelled classes in `evaluation/datasets/retrieval-calibration-v2.json`:

- `grounded`
- `in_domain_unanswerable`
- `out_of_domain`

A low domain score cannot produce admitted grounded evidence even when the grounded floor is numerically lower. Explicit named-target anchors are applied per evidence candidate, so a document mentioning the requested target cannot authorize unrelated candidates that omit it.

Physical relevance calibration is fail-closed against the declared acceptance policy: **zero false positives and at least 80% recall** for both domain and grounded signals. Score overlap is allowed when a threshold still satisfies that measured policy; grounded cases whose expected evidence does not survive candidate selection still fail. The subsequent deterministic validator requires at least one labelled expected evidence title for grounded cases and zero admitted grounded evidence for in-domain-unanswerable/out-of-domain cases. Additional admitted context is reported diagnostically and its precision is measured by RAGAS rather than being treated as an undocumented exhaustive allow-list.

## Security boundary

Deterministic secret/policy-bypass checks run before the relevance probe, model and tools.

Retrieved document content is marked `UNTRUSTED_DOCUMENT_DATA`. It is preserved as evidence but must never be followed as an instruction. Only admitted chunks enter the tool observation, preventing rejected chunks from becoming model context or citations.

## Local model split

- generation/runtime: **Unsloth Qwen3.5-0.8B `UD-Q4_K_XL` Dynamic 2.0** on llama.cpp CPU
- embeddings: Qwen3-Embedding-0.6B, OpenVINO on Intel iGPU for the physical profile
- reranker: Qwen3-Reranker-0.6B seq-cls, OpenVINO on Intel iGPU with the required Qwen relevance template
- object/vector infrastructure: Floci + S3/S3 Vectors-compatible adapters
- RAGAS judge: configurable independently; local OpenAI-compatible or external OpenAI-compatible API

The generation artifact and SHA256 are pinned in `scripts/download_models.py`. The runtime alias includes provider/model/quant identity so a materially different generation artifact cannot silently reuse an old judge calibration.

The canonical Qwen3.5 agent profile is explicitly non-thinking: 16K runtime context, `temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`, `presence_penalty=1.5`, `repeat_penalty=1.0`, Jinja tool templates, unified KV, and Q8_0 K/V cache. Thinking is disabled through the Qwen chat-template argument rather than the legacy Qwen3 reasoning switch.

Qwen3.5-0.8B is the **generation/tool-calling model**, not the embedding or reranking model. Retrieval keeps specialized embedding/reranker models because replacing them with a causal generation model would remove the measured retrieval contracts.

Changing the RAGAS judge provider/model, judge prompt or evaluation dataset invalidates the calibrated judge policy and requires recalibration.

## Evaluation

Blocking deterministic gates cover:

- status/segment contract, including four explicit `answered_mixed` cases whose grounded and ungrounded claims must stay in the correct segment type;
- citation validity and segment integrity;
- architecture fitness and zero unreachable pipeline code;
- deterministic/adversarial security;
- unit/property/integration tests;
- strict mypy;
- dependency/security audit;
- CDK test/build/synth.

Ruff formatting/lint remains visible but is **advisory during the architecture pivot**; it does not block architecture work.

Golden executes the agent once and writes `artifacts/evaluation/golden-samples.json`. RAGAS reuses exactly those responses; it does not regenerate them. RAGAS evaluates grounded segments with a reference answer and their cited contexts. For mixed responses, the grounded segment is evaluated against its grounded subquestion; the accompanying general calculation/example remains an uncited Golden concern. Direct/ungrounded answers remain covered by deterministic Golden/status/safety contracts rather than being forced into context metrics that do not apply to them.

RAGAS has bounded judge/embedding requests, bounded retries/workers and a hard suite wall timeout. It writes `ragas-results.json` and `ragas.json` before release-policy enforcement. A stale judge policy is ignored only in explicit measurement mode so a changed judge can be measured and recalibrated; strict evaluation still rejects stale identity.

`make physical-eval-measure` is the complete pre-human physical path. It prepares the runtime, calibrates/validates retrieval, provisions and ingests through the API, runs Golden once, runs RAGAS in measurement mode, and creates `artifacts/evaluation/judge-human-review.json` with exactly 10 deterministic review cases. The script leaves every `human_pass` as `null`; only a real human may set those ten booleans.

After those exactly 10 human decisions, `make eval-judge-calibrate` creates a policy bound to the exact judge provider/model/prompt and the complete base+mixed evaluation datasets. Strict `make physical-eval` remains **fail-closed** until that policy exists and matches the active evaluator. A low-agreement policy can intentionally keep RAGAS informational, but the policy artifact itself is still required by strict release evaluation.

## Evaluation fixtures are not product logic

The AcmePay/Payments/Calypso corpus exists to exercise conflicts, deprecated documents, missing evidence, security and prompt injection. Those names must not appear in product control-flow code.

## Useful commands

```bash
make setup
make test
make types
make beta-react
make physical-eval-measure
make eval-judge-calibrate
make physical-eval
```

`make physical-eval-measure` is the single command for physical measurement before the genuinely human calibration step. Measurement success is evidence that the software/runtime/evaluation pipeline completed; it is not a fabricated claim of external release readiness.

## Physical validation

Hosted CI validates software contracts, including importing/compiling/testing the evaluation runner with its real optional dependencies. The trusted `rag-e2e` runner validates the API/Floci/local-model path from a clean checkout. The canonical physical profile uses Unsloth Qwen3.5-0.8B Dynamic 2.0 for generation and OpenVINO embeddings/reranking, and calibrates labelled relevance floors before validating retrieval admission.

A release is not RAGAS-complete until the physical measurements, exactly ten human labels, calibrated judge policy and strict physical evaluation all exist for the same evaluator identity.
