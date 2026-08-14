# Demo Readiness SDD

## Objective

A client demo is ready only when the same local runtime used by Gradio passes a fail-closed behavioral gate with real Floci, Qwen embeddings, the learned Qwen relevance grader, and the Qwen generation model.

A green deterministic CI gate is necessary but not sufficient.

## Scope

The bounded conversational agent uses a two-stage RAG path: broad hybrid retrieval followed by a dedicated learned relevance grader. Production must scale to new systems through corpus/index updates, not entity-specific code.

Production MUST NOT contain entity allowlists, product-specific `if` statements, name/acronym comparisons, CamelCase heuristics, or hand-tuned corpus-wide relevance thresholds.

## Architectural invariants

### A. Input and UI safety

1. Short turns such as `ok`, `si`, `no`, and `hi` are valid.
2. Whitespace-only input is invalid.
3. Gradio never exposes Python tracebacks or raw backend exceptions.
4. Client demo mode does not expose a free-form `system` filter.
5. Clearing visible chat deletes the LangGraph thread state.
6. Client demo mode uses a pinned Gradio version.

### B. Routing and retrieval

1. Narrow control utterances (`chat`, `capabilities`, `catalog`, `out_of_scope`) are normalized and exact-matched before retrieval.
2. The control fast-path is data-driven and contains no product/system conditions.
3. Free-form utterances that do not exact-match a control intent enter `uncertain`; semantic route similarity is telemetry only.
4. An `uncertain` turn probes the corpus with the raw query.
5. Candidate retrieval is `dense + BM25 -> RRF -> EvidenceResolver`.
6. `EvidenceResolver` owns active status, structured context, version, authority, and supersedes.
7. Resolved candidates are graded by Qwen3-Reranker-0.6B using its learned yes/no relevance objective.
8. Candidate scores are ranking/diagnostic signals. Admission is the reranker's model-native yes/no classification, not a corpus-tuned similarity threshold.
9. The legacy vector/lexical blended score is diagnostic only.
10. Cross-language queries are judged by the multilingual learned grader rather than token overlap.
11. A free-form turn resolves to `knowledge` only when at least one resolved candidate receives a learned relevant grade.
12. Dense nearest-neighbor similarity alone is never evidence.
13. Contextual retrieval may expand candidate recall, but relevance grading uses the original current question.
14. Startup executes a functional real reranker probe, not only `/health`.
15. Production does not prefilter candidates by names, acronyms, casing, product identifiers, or manually maintained entity rules.

### C. Grounding ownership

1. Evidence sufficiency is decided before generation from learned-grader-admitted evidence.
2. Once grounded generation starts, the generation model does not choose `answered` versus `insufficient_evidence`.
3. Grounded generation returns answer text only; status and citation IDs are application-owned.
4. Generation uses only `ADMITTED_EVIDENCE_JSON`.
5. Citations are attached deterministically from admitted evidence.
6. No admitted evidence produces `insufficient_evidence` before generation.
7. Generation transport/schema failure is a runtime failure, not evidence insufficiency.

### D. Conversational memory

1. Retrieval never concatenates raw chat history.
2. Grounded generation never receives free-form assistant history.
3. Contextual rewriting uses only the last successful grounded query and validated source titles.
4. Trusted focus is scoped to QueryContext.
5. Failed grounded turns clear trusted focus.
6. Catalog/capabilities/out-of-scope turns clear trusted operational focus; lightweight chat may preserve it.
7. Follow-up contextualization is attempted only after the current-question pass has no learned relevant evidence and trusted focus exists.
8. Rewrite output is plain text; failed/invalid rewrite falls back to the current question unchanged.
9. A rewritten query may improve recall, but the learned grader judges candidates against the original current question.
10. Topic changes are handled by contextualization plus learned relevance grading, never product-name comparisons.

### E. Knowledge-base integrity

1. Startup verifies every repository document against its stored manifest digest.
2. Startup verifies every manifest-referenced vector exists.
3. One surviving vector is insufficient to declare readiness.
4. Changed/incomplete corpus triggers reseeding/reingestion.
5. Reingestion deletes stale chunk objects.

### F. Safety

1. Direct secret extraction and explicit policy bypass are blocked before retrieval/generation.
2. Retrieved evidence is data, never instructions.
3. Adversarial tests target the current `ConversationalAgent`.
4. The real demo gate includes an indirect prompt-injection case.

### G. Demo gate truthfulness

1. A scenario is never labeled `PASS` before its assertions succeed.
2. Pre-assertion diagnostics use `CHECK` or retrieval diagnostics.
3. The only global success signal is `DEMO READY: all real-runtime client scenarios passed`.

### H. Learned retrieval validation

1. `evaluation/datasets/retrieval-calibration-v1.json` is a labeled behavioral validation dataset containing multilingual positives and realistic same-domain hard negatives.
2. Validation runs the same `dense + BM25 -> RRF -> EvidenceResolver -> Qwen relevance grader` path used by production.
3. Every positive case must admit at least one expected document.
4. Every negative/hard-negative case must admit no document.
5. Validation aggregates all failures instead of tuning behavior around one query.
6. There is no derived global admission threshold or calibration artifact.
7. New systems/products are supported through corpus and learned retrieval behavior, not new comparison rules.
8. `make ui`, `make demo-ready`, `make demo-client`, and benchmarks run learned retrieval validation first.

## Client-demo behavioral contract

`make demo-client` stops before Gradio if any case fails:

1. `Hola` -> conversational response.
2. `Que haces?` -> capabilities.
3. `Que documentacion tienes disponible?` -> active KB catalog.
4. `que pasa con sendgrid?` -> grounded SendGrid evidence.
5. `Cual es el objetivo de Calypso Payments API?` -> grounded Calypso/Payments evidence.
6. `Cuantos reintentos permite Calypso?` -> three retries and Payment Retry Policy evidence.
7. `Y despues del tercero?` -> Treasury Integrations through contextual retrieval.
8. Switch to SendGrid then `Y si falla?` -> SendGrid, not stale Calypso.
9. Missing SAP fact -> abstain; following turn cannot resurrect older Calypso focus.
10. Clear thread -> dependent follow-up cannot recover prior focus.
11. `ok`, `si`, `no`, `hi` -> no exception.
12. France capital -> no internal grounded answer/citations.
13. Direct credential/policy-bypass request -> safety blocked.
14. Indirect prompt injection -> remains grounded in authoritative evidence.

## Test strategy

### Unit / deterministic contracts

- Query normalization and short inputs.
- Exact control routing without embedding the query.
- Free-form semantic collisions enter retrieval-first `uncertain`.
- Dense + BM25 + RRF + resolver behavior.
- Qwen reranker response parsing preserves candidate indexes.
- Qwen model-native yes/no admission behavior.
- Malformed/incomplete reranker response rejection.
- Contextual candidate retrieval is graded against the original current question.
- Failed contextual rewrite falls back to the current question.
- Learned relevant evidence cannot be vetoed by model-owned status.
- Deterministic citation attachment.
- Focus invalidation and thread deletion.
- Safety, corpus integrity, stale chunk deletion, safe UI errors.

### Integration

- Floci `GetVectors` corpus completeness.
- Existing S3/S3 Vectors integration.
- Local llama.cpp reranker accepts a real `/v1/rerank` request with Qwen3-Reranker-0.6B and distinguishes a supported document from a same-domain wrong-target document.

### Real local validation and behavioral gate

`make retrieval-validate` executes the labeled multilingual positive/hard-negative dataset against the real local retrieval stack. It exits non-zero if any positive expected document is not admitted or any hard negative is admitted.

`scripts/demo_ready.py` then runs `query_workflow()` with real Floci, Qwen3-Embedding-0.6B, Qwen3-Reranker-0.6B, and Qwen3.5-2B. Any assertion/runtime error exits non-zero.

The self-hosted `demo-validation` workflow executes this exact `make demo-ready` path.

`make demo-client` is the recommended client entrypoint because it validates retrieval and the full behavioral contract before launching Gradio.

## Validation rule

The change is not client-ready from deterministic CI alone. Final acceptance requires successful `make demo-ready` on the same local runtime used for the client session, and deterministic CI must be green for that exact commit.

No failed query may be fixed by adding entity comparisons, one-off regexes, or a hand-selected universal relevance cutoff.

## Runtime model profile

- Generation/contextualization: Qwen3.5-2B Q4_K_M.
- Embeddings: Qwen3-Embedding-0.6B Q8_0.
- Learned relevance: Qwen3-Reranker-0.6B Q4_K_M using llama.cpp native reranking/classifier output.

## Non-goals

- Multi-agent orchestration.
- Entity-specific routing/retrieval rules.
- Surface-form entity comparison for topic switching/admission.
- Handmade semantic/lexical relevance as production admission.
- Per-query relevance threshold tuning.
