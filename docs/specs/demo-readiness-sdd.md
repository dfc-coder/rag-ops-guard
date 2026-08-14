# Demo Readiness SDD

## Objective

A client demo is considered ready only when the same local runtime used by Gradio passes a fail-closed behavioral gate with real Floci, the real embedding model and the real Qwen generation model.

A green deterministic CI gate is necessary but not sufficient for a client demo.

## Scope

This specification hardens the bounded conversational agent and local demo path. It does not change the generation model, Floci, llama.cpp, retrieval topology or long-term persistence architecture.

## Architectural invariants

### A. Input and UI safety

1. Short conversational turns such as `ok`, `si`, `no`, and `hi` are valid requests.
2. Whitespace-only input is invalid.
3. Gradio must never expose a Python traceback or raw backend exception to the client.
4. Client demo mode must not expose a free-form `system` filter. API callers may still provide structured context.
5. Clearing the visible conversation must delete the LangGraph thread state, not only the browser history.
6. Client demo mode must use a pinned Gradio version.

### B. Routing and retrieval

1. Registered narrow control utterances (`chat`, `capabilities`, `catalog`, `out_of_scope`) are normalized for case, accents, punctuation and whitespace and resolved exactly before semantic routing.
2. The exact control fast-path is data-driven from the route-example set; it must not contain entity-specific conditions such as product or system names.
3. Direct product-role questions such as `¿Qué haces?`, including normalized variants such as `QUE HACES!!!`, belong to `capabilities` and must not enter RAG.
4. Free-form utterances that do not match a narrow control intent MUST enter `uncertain`; semantic similarity alone is advisory telemetry and MUST NOT directly choose `chat`, `capabilities`, `catalog`, `knowledge`, or `out_of_scope`.
5. An `uncertain` turn probes the corpus with the raw user query, not with an instruction that presupposes an integration-operations intent.
6. `uncertain` is not a user-facing terminal route.
7. The post-retrieval resolver decides `knowledge` versus a control fallback using admitted evidence, lexical support, relevance, and semantic route scores.
8. A semantic collision in which `capabilities` or another control class scores above `knowledge` MUST NOT prevent an entity/system query from reaching retrieval.
9. If admitted evidence is relevant and has an informative lexical anchor, the turn may resolve to `knowledge` even when a control class had the highest embedding similarity.
10. Dense-only nearest-neighbor similarity is not sufficient to convert a control/meta question into `knowledge`.
11. Relevance and lexical support are computed only from evidence that survived deterministic admission/version/context rules.
12. Rejected dense candidates must not increase the final relevance or lexical-support signal.
13. A clear knowledge retrieval uses the operational retrieval instruction only after the agent has resolved the turn to the knowledge path; raw disambiguation probes remain instruction-free.

### C. Grounding ownership

1. Evidence sufficiency is a deterministic application decision made before grounded generation from admitted evidence and the configured relevance gate.
2. Once a turn enters grounded generation, the generation model MUST NOT choose `answered` versus `insufficient_evidence`.
3. The grounded generation schema contains answer text only; it does not contain status or citation identifiers.
4. The generation model may synthesize only from `ADMITTED_EVIDENCE_JSON` and may not use external knowledge.
5. Grounded citations are attached by the application from the admitted evidence bundle, not invented or selected as internal IDs by the generation model.
6. Citation attachment preserves admitted ranking and emits at most one chunk citation per logical document/version.
7. Low relevance or no admitted evidence produces `insufficient_evidence` before any grounded generation call.
8. A generation transport/schema failure is not evidence insufficiency; the demo gate must still fail closed rather than silently treating a model veto as a valid abstention.

### D. Trusted conversational memory

1. Retrieval never concatenates raw chat history.
2. Grounded generation never receives free-form assistant history.
3. Follow-up rewriting may use only the last successful grounded query and validated source titles.
4. Trusted focus is scoped to the QueryContext that produced it.
5. A failed grounded knowledge turn clears trusted focus so a later follow-up cannot fall back to an unrelated older topic.
6. Explicit out-of-scope, catalog, and capabilities turns clear trusted operational focus. Lightweight greetings/thanks may preserve it.

### E. Knowledge-base integrity

1. Local startup must verify every repository knowledge document against its stored manifest digest.
2. Local startup must verify every vector key referenced by those manifests exists in S3 Vectors.
3. One surviving vector is not sufficient to declare the corpus ready.
4. A changed or incomplete corpus triggers deterministic reseeding/reingestion.
5. Reingesting a changed document deletes stale chunk objects left by the previous version of the same logical document/version.

### F. Safety

1. Direct secret-extraction and explicit policy-bypass intent is blocked deterministically before retrieval/generation.
2. Retrieved evidence is never treated as instructions.
3. The current `ConversationalAgent`, not a legacy workflow, is the subject of adversarial tests.
4. The real-model demo gate includes an indirect prompt-injection scenario from the repository corpus.

### G. Demo gate truthfulness

1. A turn is never labeled `PASS` before its scenario-specific assertions have succeeded.
2. Runtime observations may be printed as `CHECK`/diagnostic output before assertions.
3. The only global success signal is `DEMO READY: all real-runtime client scenarios passed` after every required assertion succeeds.

## Client-demo behavioral contract

`make demo-client` MUST stop before launching Gradio if any of these fail:

1. `Hola` -> conversational response; no raw error; language is not mixed with an English help sentence.
2. `Que haces?` -> capabilities; no grounded citations required.
3. `Que documentacion tienes disponible?` -> catalog containing real active KB entries.
4. `que pasa con sendgrid?` -> grounded answer with SendGrid evidence even if a control intent has a higher raw embedding similarity.
5. `Cual es el objetivo de Calypso Payments API?` -> grounded answer with Calypso/Payments API evidence.
6. `Cuantos reintentos permite Calypso?` -> three retries and active Payment Retry Policy evidence.
7. `Y despues del tercero?` in the same thread -> Treasury Integrations, using trusted follow-up context.
8. Topic switch to SendGrid followed by `Y si falla?` -> resolves against SendGrid, not the prior Calypso topic.
9. After a failed/missing SAP fact, a follow-up must not resurrect an older Calypso focus.
10. Clearing the thread removes backend memory; the same dependent follow-up after clear must not recover the previous answer.
11. `ok`, `si`, `no`, `hi` -> no exception.
12. `Cual es la capital de Francia?` -> no internal grounded answer/citations.
13. Direct credential/policy-bypass request -> `safety_blocked` with no citations.
14. Indirect prompt-injection case -> no instruction-following leak; answer remains grounded in authoritative operational evidence.

## Test strategy

### Unit / deterministic

- QueryRequest normalization and short inputs.
- Normalized exact-match control routing without embedding the query.
- Free-form semantic control collisions always enter the retrieval-first `uncertain` path.
- Semantic scores remain available for post-retrieval fallback but cannot short-circuit retrieval.
- Direct capabilities intent coverage.
- Relevance uses only admitted evidence.
- Raw-query uncertain probe.
- Dense-only probe cannot override a leading control intent.
- Admitted lexical anchor may confirm knowledge when the router is uncertain.
- Admitted evidence above the gate cannot be vetoed by a model-owned `insufficient_evidence` status.
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

### Real local behavioral gate

`scripts/demo_ready.py` uses `query_workflow()` directly after `models + local-up + local-data`. It uses fresh unique thread IDs and asserts the client-demo behavioral contract. Any assertion or runtime error exits non-zero.

The gate prints turn observations as `CHECK`, never as `PASS` before validation. Only the final `DEMO READY` line represents complete success.

`make demo-client` is the only recommended entry point for a client-facing demo. It runs the real behavioral gate first and launches Gradio only on success.

## Validation rule

The hardening change is not declared client-ready from deterministic CI alone. The final acceptance signal is a successful `make demo-ready` execution on the same local runtime that will be used for the client session.

The deterministic CI suite must also be green for the exact commit used by that local runtime, including formatter, lint, strict typing, unit, property, Floci integration, security, and CDK gates.

## Non-goals for this hardening pass

- Postgres/Redis conversation persistence.
- Multi-agent orchestration.
- Model upgrade.
- New vector database.
- Entity-specific routing rules.
- Arbitrary threshold tuning without measured evidence.
