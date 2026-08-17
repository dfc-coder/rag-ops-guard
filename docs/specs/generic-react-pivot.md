# Generic ReAct Pivot — executable specification

Status: **U0–U6 implemented; external release measurements still pending**.

This document is the canonical SDD contract for replacing the legacy routed RAG application with a generic conversational ReAct agent using optional document tools.

## Replacement rules

- **R-1 — Replace, do not stack.** Each unit deletes the mechanism it replaces in the same PR. `git revert` is the rollback mechanism; dead runtime paths are not.
- **R-2 — Non-additive pipeline budget.** The unreachable runtime budget decreases through the pivot: U1a 2105, U1 1580, U2 1212, U3 1108, U4 638, U5 638, U6 0.
- **R-3 — Zero unreachable runtime modules at completion.** Canonical roots are `rag_ops_guard.app`, `rag_ops_guard.agent.conversation`, and `rag_ops_guard.handlers.query`; Lambda ingest/health entrypoints are excluded from conversational ownership.

## U1a — Tool and tool-calling model ports

- **SPEC-1a.1** — A `Tool` port exposes `name`, `description`, `schema()` and `invoke()`.
- **SPEC-1a.2** — Tools return typed `ToolResult`, never serialized JSON strings as their application contract.
- **SPEC-1a.3** — A `ToolCallingModel` port exposes `bind_tools`, `invoke`, and `invoke_structured`.
- **SPEC-1a.4** — `agent/conversation.py` imports neither LangChain nor LangGraph. Framework dependencies live in adapters.
- **SPEC-1a.5** — `search_documents` performs retrieval and reports observations; it does not decide whether an answer is grounded.

Status: implemented. Legacy router/agent/graph orchestration was removed instead of retained as a fallback.

## U1 — Segmented response contract

- **SPEC-1.1** — A response is an ordered sequence of segments. Each segment is grounded with at least one citation or ungrounded with zero citations.
- **SPEC-1.2** — Every grounded citation references a chunk admitted during that turn. Invented citations are validation errors.
- **SPEC-1.3** — An ungrounded segment carrying citations is unrepresentable.
- **SPEC-1.4** — Public response status is derived from segments, not assigned independently.
- **SPEC-1.5** — `QueryResponse.citations` is derived as the union of segment citations.
- **SPEC-1.6** — Segments originate from structured generation; finished prose is not split afterward.

Public answer states: `answered_grounded`, `answered_mixed`, `answered_ungrounded`, `clarification_required`, `safety_blocked`, `error`. `insufficient_evidence` is retired.

Status: implemented.

## U2 — Generic ingestion

- **SPEC-2.1** — Plain Markdown or text without front matter ingests successfully. Title is inferred; identity is deterministic.
- **SPEC-2.2** — Initial generic formats are `.md` and `.txt`. PDF, DOCX and HTML are deferred.
- **SPEC-2.3** — Valid rich front matter is preserved and can enable governance policy.
- **SPEC-2.4** — Governance resolution is optional and applies only where relevant metadata exists.
- **SPEC-2.5** — Present but malformed front matter fails loudly.

Status: implemented. Generic documents are storable, listable and searchable without fake Ops metadata.

## U3 — Direct and indirect prompt injection

- **SPEC-3.1** — Adversarial tests target the canonical `ConversationAgent` prompt/boundary.
- **SPEC-3.2** — Retrieved evidence is explicitly marked `UNTRUSTED_DOCUMENT_DATA`.
- **SPEC-3.3** — Instructions embedded in documents do not change agent behavior.
- **SPEC-3.4** — An injected document may still be cited for legitimate factual content.
- **SPEC-3.5** — Direct secret extraction is `safety_blocked` before probe, model or tool execution.
- **SPEC-3.6** — Indirect injection coverage includes documents without front matter.
- **SPEC-3.7** — Injected content cannot cause output/citation of a non-admitted document.

Status: implemented. Hosted CI runs the full `tests/adversarial` directory.

## U4 — Two measured relevance signals

- **SPEC-4.1** — `domain_relevance` is measured on raw pre-policy candidates; `grounded_relevance` is measured after resolver/target checks.
- **SPEC-4.2** — `domain_relevance < DOMAIN_FLOOR` prohibits admitted grounded evidence but does not prohibit a normal ungrounded answer.
- **SPEC-4.3** — Relevance floors come from calibration measurements, never guessed constants.
- **SPEC-4.4** — Every safe first turn performs a silent non-tool probe needed for a trace/score, including turns answered ungrounded.
- **SPEC-4.5** — Explicit named-target anchors are applied per evidence candidate; one candidate mentioning the requested target cannot authorize unrelated candidates that omit it.
- **SPEC-4.6** — Deterministic labelled retrieval validation requires at least one expected evidence title for `grounded` cases and zero admitted evidence for `in_domain_unanswerable`/`out_of_domain`. Additional admitted context is diagnostic here and is evaluated by context-precision metrics rather than an undocumented exhaustive title allow-list.

Calibration classes are `grounded`, `in_domain_unanswerable`, and `out_of_domain`. Calibration chooses thresholds from labelled measurements and fails closed unless both signals achieve **zero false positives and at least 80% recall**. Score distributions may overlap when a threshold still satisfies that declared policy; perfect class separation is not an independent requirement.

Status: implemented, including the final domain-floor admission invariant.

## U5 — Evaluation/RAGAS alignment

- **SPEC-5.1** — Strict release evaluation enforces per-case floors in addition to aggregate means. Current per-case floors: faithfulness `0.60`, context precision `0.30`.
- **SPEC-5.2** — RAGAS evaluates grounded segments with reference answers and the contexts cited by those segments. Direct/ungrounded answer quality, status, safety and citation integrity remain deterministic Golden concerns and are not forced into context metrics that do not apply.
- **SPEC-5.3** — Golden executes the agent once. RAGAS reuses the exact resulting `golden-samples.json`; it never regenerates the answers being judged.
- **SPEC-5.4** — Judge requests, embedding requests, RAGAS retries/workers and the whole RAGAS suite are time-bounded. A backend cannot leave physical evaluation waiting indefinitely.
- **SPEC-5.5** — The runtime generation model and the RAGAS judge are independently configurable. A judge policy is valid only for the exact provider/model/prompt/dataset identity it calibrated. Agreement uses exactly 10 real human labels:
  - 9–10/10: RAGAS may gate at aggregate faithfulness floor `0.85` plus per-case floor;
  - 7–8/10: use a lower threshold derived from measured labels plus per-case floor;
  - <=6/10: RAGAS is informational and cannot block release.
- **SPEC-5.6** — Automatic physical validation runs RAGAS in explicit measurement mode before human calibration. Strict `physical-eval` remains fail-closed until a matching `judge-policy.json` exists; successful measurement must never be mislabeled as external release readiness.

Status: mechanism implemented. The repository deliberately does **not** claim a human-agreement score before a real target-machine run and ten real labels exist.

## U6 — Evaluation corpus and final replacement

- **SPEC-6.1** — Golden statuses use the segmented contract; legacy `answered`/`insufficient_evidence` expectations are removed.
- **SPEC-6.2** — `retrieval-calibration-v2.json` is the single three-class calibration and deterministic retrieval-validation dataset.
- **SPEC-6.3** — AcmePay/Payments/Calypso are evaluation fixtures, never the product domain or routing logic.
- **SPEC-6.4** — Golden does not run a second dense-only retrieval implementation. Required/forbidden source assertions validate the evidence actually cited by the canonical agent response.

Final closure conditions:

- obsolete legacy `scripts/react_*.py` shims removed;
- `src/rag_ops_guard/graph/` physically absent;
- legacy structured chat adapter/ports removed;
- LangGraph not a direct runtime dependency and its old ADR explicitly superseded;
- unreachable pipeline budget exactly `0`;
- pipeline ownership exactly `{app, agent.conversation}`;
- README/architecture/technology docs describe the actual system;
- golden runner validates deterministic `segment_integrity`.

Status: implemented on U6; merge requires hosted correctness CI green.

## Architecture fitness gates

Every implementation PR uses behavioral tests plus architecture fitness. Final U6 fitness requires:

```python
PIPELINE_OWNERS == {"app", "agent.conversation"}
UNREACHABLE_BUDGET_LINES == 0
not Path("src/rag_ops_guard/graph").exists()
```

Citation validation must remain reachable from the canonical path. Tests against deleted/dead modules do not count.

## CI policy

Style-only failures must not stop architectural replacement work.

- Ruff format/lint: advisory in hosted CI and release-check workflow.
- strict mypy: blocking.
- unit coverage, architecture fitness and adversarial security: blocking.
- property + Floci integration: blocking.
- evaluation runner import/compile/contracts with the real eval extra: blocking.
- dependency/security audit: blocking.
- CDK test/build/synth: blocking.
- deterministic evaluation gates: blocking when the physical evaluation environment is available.

A strict manual `make lint` remains available.

## Definition of done

Software pivot DoD:

- one canonical `ConversationAgent`;
- model-driven optional tools;
- generic Markdown/text ingestion;
- segmented citation contract;
- untrusted-document security boundary;
- double relevance + three-class calibration;
- zero unreachable pipeline code;
- no legacy graph/router path;
- one configured runtime generation model and an explicitly identified/calibrated judge;
- hosted correctness/security/evaluation-contract CI green.

External release DoD, deliberately not fabricated:

1. run the target physical measurement gate with Qwen3.5-0.8B generation plus OpenVINO embeddings/reranker;
2. produce real RAGAS scores and exactly ten human labels;
3. calibrate judge agreement against the exact judge identity and apply the resulting policy;
4. run strict `physical-eval` with that policy;
5. only then claim final release readiness.
