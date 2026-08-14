# Demo Readiness SDD

## Objective

A client demo is considered ready only when the same local runtime used by Gradio passes a fail-closed behavioral gate with real Floci, the real embedding model, the real multilingual cross-encoder reranker, and the real Qwen generation model.

A green deterministic CI gate is necessary but not sufficient for a client demo.

## Scope

This specification hardens the bounded conversational agent and local demo path. Retrieval uses a standard two-stage pattern: broad candidate retrieval followed by a dedicated multilingual cross-encoder reranker. Reranker admission is calibrated from labeled positive and negative examples; no fixed hand-written score cutoff is allowed in production.

The production retrieval path MUST remain scalable across new systems and products. It MUST NOT contain entity allowlists, product-specific `if` statements, token-name comparisons, CamelCase/acronym heuristics, or other surface-form rules that must grow as the corpus grows.

## Architectural invariants

### A. Input and UI safety

1. Short conversational turns such as `ok`, `si`, `no`, and `hi` are valid requests.
2. Whitespace-only input is invalid.
3. Gradio must never expose a Python traceback or raw backend exception to the client.
4. Client demo mode must not expose a free-form `system` filter. API callers may still provide structured context.
5. Clearing the visible conversation must delete the LangGraph thread state, not only the browser history.
6. Client demo mode must use a pinned Gradio version.

### B. Routing and retrieval

1. Registered narrow control utterances (`chat`, `capabilities`, `catalog`, `out_of_scope`) are normalized for case, accents, punctuation and whitespace and resolved exactly before retrieval.
2. The exact control fast-path is data-driven from the route-example set; it must not contain entity-specific conditions such as product or system names.
3. Direct product-role questions such as `¿Qué haces?`, including normalized variants such as `QUE HACES!!!`, belong to `capabilities` and must not enter RAG.
4. Free-form utterances that do not match a narrow control intent MUST enter `uncertain`; semantic route similarity is telemetry only and MUST NOT prevent corpus retrieval.
5. An `uncertain` turn probes the corpus with the raw user query, not with an instruction that presupposes an integration-operations intent.
6. `uncertain` is not a user-facing terminal route.
7. Candidate retrieval is `dense + BM25 -> RRF -> EvidenceResolver`.
8. `EvidenceResolver` remains authoritative for active status, context, version, authority, and supersedes before relevance classification.
9. Resolved candidates are scored and ordered by a dedicated multilingual cross-encoder reranker.
10. Reranker scores are ranking signals, not calibrated probabilities. A literal cutoff such as `0.4` or `0.5` MUST NOT be introduced as a production support rule without labeled calibration evidence.
11. The legacy blended score derived from vector distance and lexical overlap is diagnostic only and MUST NOT decide `knowledge` versus fallback or `answered` versus `insufficient_evidence`.
12. A Spanish question against an English document MUST be judged by the multilingual cross-encoder rather than by token-language overlap.
13. A free-form turn resolves to `knowledge` only when at least one resolved candidate passes the currently valid calibrated admission threshold.
14. Dense-only nearest-neighbor similarity is never sufficient evidence by itself.
15. A clear knowledge retrieval may use the operational embedding instruction; raw disambiguation probes remain instruction-free.
16. The reranker service is a required runtime dependency and local startup must execute a functional `/v1/rerank` probe, not only a health check.
17. Production retrieval MUST NOT prefilter candidates by comparing query entity names, acronyms, title case, CamelCase, product names, or system identifiers. New systems must be supported by indexing and learned retrieval/reranking, not code changes.

### C. Grounding ownership

1. Evidence sufficiency is an application decision made before grounded generation from calibrated cross-encoder-admitted evidence.
2. Once a turn enters grounded generation, the generation model MUST NOT choose `answered` versus `insufficient_evidence`.
3. The grounded generation schema contains answer text only; it does not contain status or citation identifiers.
4. The generation model may synthesize only from `ADMITTED_EVIDENCE_JSON` and may not use external knowledge.
5. Grounded citations are attached by the application from the admitted evidence bundle, not invented or selected as internal IDs by the generation model.
6. Citation attachment preserves admitted ranking and emits at most one chunk citation per logical document/version.
7. No calibrated-admitted evidence produces `insufficient_evidence` before any grounded generation call.
8. A generation transport/schema failure is not evidence insufficiency; the demo gate must fail closed rather than silently treating a model veto as a valid abstention.

### D. Conversational memory and contextual retrieval

1. Retrieval never concatenates raw chat history.
2. Grounded generation never receives free-form assistant history.
3. Contextual rewriting may use only the last successful grounded query and validated source titles.
4. Trusted focus is scoped to the QueryContext that produced it.
5. A failed grounded knowledge turn clears trusted focus so a later follow-up cannot fall back to an unrelated older topic.
6. Explicit out-of-scope, catalog, and capabilities turns clear trusted operational focus. Lightweight greetings/thanks may preserve it.
7. A follow-up is contextualized only when the first retrieval/reranking pass has no admitted support and trusted focus exists.
8. The contextualizer returns one plain-text standalone query. Structured JSON output is not required for query rewriting.
9. If contextualization fails, times out, returns empty output, or returns invalid output, the safe fallback is the current question unchanged. The fallback MUST NOT concatenate previous focus into the query.
10. A contextualized query may improve candidate recall, but the cross-encoder MUST score those candidates against the original current user question. Conversation history therefore cannot redefine the relevance question.
11. Topic changes are handled by the contextualizer plus the learned reranker and hard-negative calibration. Production code MUST NOT compare named systems to decide whether a topic changed.

### E. Knowledge-base integrity

1. Local startup must verify every repository knowledge document against its stored manifest digest.
2. Local startup must verify every vector key referenced by those manifests exists in S3 Vectors.
3. One surviving vector is not sufficient to declare the corpus ready.
4. A changed or incomplete corpus triggers deterministic reseeding/reingestion.
5. Reingesting a changed document deletes stale chunk objects left by the previous version of the same logical document/version.

### F. Safety

1. Direct secret-extraction and explicit policy-bypass intent is blocked before retrieval/generation.
2. Retrieved evidence is never treated as instructions.
3. The current `ConversationalAgent`, not a legacy workflow, is the subject of adversarial tests.
4. The real-model demo gate includes an indirect prompt-injection scenario from the repository corpus.

### G. Demo gate truthfulness

1. A turn is never labeled `PASS` before its scenario-specific assertions have succeeded.
2. Runtime observations may be printed as `CHECK`/diagnostic output before assertions.
3. The only global success signal is `DEMO READY: all real-runtime client scenarios passed` after every required assertion succeeds.

### H. Data-driven reranker calibration

1. Production MUST NOT contain a manually selected reranker support threshold.
2. `evaluation/datasets/retrieval-calibration-v1.json` contains labeled multilingual positive and negative queries representative of both supported behavior and realistic hard negatives from the same operational domain.
3. Calibration runs the same `dense + BM25 -> RRF -> EvidenceResolver -> cross-encoder` path as production, with admission temporarily disabled only to observe raw reranker rankings and scores.
4. Every positive calibration query MUST retrieve at least one labeled expected document within the configured context window. Failure is a retrieval/ranking defect and MUST stop calibration.
5. For positive samples, calibration records the reranker score of an expected document. For negative samples, it records the highest candidate score, representing the strongest false-positive pressure.
6. A production threshold is derived only when `max(negative scores) < min(positive scores)`. The threshold is the midpoint of that observed separation interval.
7. If positive and negative distributions overlap, calibration MUST fail. The implementation MUST NOT choose a compromise threshold, lower a cutoff, or add entity-specific exceptions.
8. The calibration artifact is bound to the reranker model, the complete knowledge-base fingerprint, and the labeled dataset fingerprint.
9. Runtime MUST fail closed when the calibration artifact is missing, stale, malformed, created for another reranker model, another corpus, or another calibration dataset.
10. `make ui`, `make demo-ready`, `make demo-client`, and runtime benchmarks MUST run calibration before constructing the production knowledge search.
11. `make demo-ready` recalibrates on every invocation. Demo correctness must not depend on a previously cached calibration artifact.
12. Score distributions and the resulting separation gap are diagnostic evidence and must be visible in calibration output; they are not hidden constants.

## Client-demo behavioral contract

`make demo-client` MUST stop before launching Gradio if any of these fail:

1. `Hola` -> conversational response; no raw error; language is not mixed with an English help sentence.
2. `Que haces?` -> capabilities; no grounded citations required.
3. `Que documentacion tienes disponible?` -> catalog containing real active KB entries.
4. `que pasa con sendgrid?` -> grounded answer with SendGrid evidence even if a control intent has a higher raw embedding similarity.
5. `Cual es el objetivo de Calypso Payments API?` -> grounded answer with Calypso/Payments API evidence.
6. `Cuantos reintentos permite Calypso?` -> three retries and active Payment Retry Policy evidence.
7. `Y despues del tercero?` in the same thread -> Treasury Integrations, using contextual retrieval.
8. Topic switch to SendGrid followed by `Y si falla?` -> resolves against SendGrid, not the prior Calypso topic.
9. After a failed/missing SAP fact, a follow-up must not resurrect an older Calypso focus.
10. Clearing the thread removes backend memory; the same dependent follow-up after clear must not recover the previous answer.
11. `ok`, `si`, `no`, `hi` -> no exception.
12. `Cual es la capital de Francia?` -> no internal grounded answer/citations.
13. Direct credential/policy-bypass request -> `safety_blocked` with no citations.
14. Indirect prompt-injection case -> no instruction-following leak; answer remains grounded in authoritative operational evidence.

## Test strategy

### Unit / deterministic contracts

- QueryRequest normalization and short inputs.
- Normalized exact-match control routing without embedding the query.
- Free-form semantic control collisions always enter the retrieval-first `uncertain` path.
- Raw-query uncertain probe.
- Dense + BM25 fusion and resolver behavior.
- Cross-encoder response parsing preserves original candidate indexes.
- Cross-encoder malformed-response rejection.
- Contextual candidate retrieval is reranked against the original current question.
- Plain-text contextual rewrite returns current question on invalid/empty output.
- Data-driven threshold derivation from separated labeled distributions.
- Calibration refuses overlapping positive/negative distributions.
- Runtime calibration loader rejects wrong model, stale corpus, and stale dataset fingerprints.
- Cross-encoder support cannot be vetoed by a model-owned `insufficient_evidence` status.
- Grounded drafting uses an answer-only structured schema.
- Grounded source IDs are deterministic and valid for the admitted evidence bundle.
- Focus context compatibility and focus invalidation.
- Thread deletion.
- Safety guard on current agent.
- Corpus manifest/vector completeness.
- Stale chunk deletion on reingestion.
- Safe UI error formatting helpers.

### Integration

- Floci `GetVectors` supports corpus completeness checks.
- Existing S3/S3 Vectors integration remains green.
- llama.cpp reranker service must accept a real `/v1/rerank` request with the configured BGE reranker and score every supplied document.

### Real local calibration and behavioral gate

`make retrieval-calibrate` runs the labeled multilingual calibration dataset against the real local retrieval/reranker stack and writes `.local/reranker-calibration.json` only when the labeled populations are separable.

`scripts/demo_ready.py` then uses `query_workflow()` directly after `models + local-up + local-data + retrieval-calibrate`. It uses real Floci, Qwen embeddings, the multilingual cross-encoder reranker, and Qwen generation. It uses fresh unique thread IDs and asserts the client-demo behavioral contract. Any assertion or runtime error exits non-zero.

The gate prints turn observations as `CHECK`, never as `PASS` before validation. Only the final `DEMO READY` line represents complete success.

The self-hosted `demo-validation` workflow executes this exact `make demo-ready` path rather than legacy demo/workflow assertions.

`make demo-client` is the only recommended entry point for a client-facing demo. It runs calibration and the real behavioral gate first and launches Gradio only on success.

## Validation rule

The hardening change is not declared client-ready from deterministic CI alone. The final acceptance signal is a successful `make demo-ready` execution on the same local runtime that will be used for the client session.

The deterministic CI suite must also be green for the exact commit used by that local runtime, including formatter, lint, strict typing, unit, property, Floci integration, security, and CDK gates.

No retrieval cutoff may be changed in response to a single failed query. A cutoff change requires a changed labeled calibration dataset or retrieval/reranker behavior, followed by a complete recalibration and demo gate run.

## Non-goals for this hardening pass

- Postgres/Redis conversation persistence.
- Multi-agent orchestration.
- Generation-model upgrade.
- New vector database.
- Entity-specific routing or retrieval rules.
- Surface-form entity comparison for topic switching or candidate admission.
- Using the legacy handmade semantic/lexical relevance blend as a production admission gate.
- Manually tuning a universal reranker cutoff from an individual client query.
