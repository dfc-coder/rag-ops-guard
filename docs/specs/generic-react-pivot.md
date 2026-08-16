# Generic ReAct Pivot — executable specification

Status: Accepted direction after the failed physical freeze gate on `716f3f4`.

This document is the versioned SDD source for the implementation units below. Tests and implementation commits cite these IDs.

## Replacement rules

- **R-1 — Replace, do not stack.** Each unit deletes the mechanism it replaces in the same PR. `git revert` is the rollback mechanism; dead runtime paths are not.
- **R-2 — Non-additive pipeline budget.** Reachable/unreachable line budgets in `agent/`, `graph/`, and `retrieval/` never increase without explicit review justification. The unreachable baseline measured at `716f3f4` is 2590 lines and must decrease with each replacement unit.
- **R-3 — Zero unreachable runtime modules at completion.** The canonical roots are `rag_ops_guard.app`, `rag_ops_guard.agent.conversation`, and `rag_ops_guard.handlers.query`; Lambda entrypoints `handlers.ingest` and `handlers.health` are excluded.

## U1a — Tool and tool-calling model ports

- **SPEC-1a.1** — A `Tool` port exposes `name`, `description`, `schema()` and `invoke()`.
- **SPEC-1a.2** — Tools return typed `ToolResult`, never serialized JSON strings as their application contract.
- **SPEC-1a.3** — A `ToolCallingModel` port exposes `bind_tools`, `invoke`, and `invoke_structured`.
- **SPEC-1a.4** — `agent/conversation.py` imports neither LangChain nor LangGraph. Framework dependencies live in adapters.
- **SPEC-1a.5** — `search_documents` performs retrieval and reports observations; it does not decide whether an answer is grounded.

U1a replacement budget target: unreachable pipeline code <= 2105 lines. `ports/interfaces.py` becomes reachable; `agent/semantic_gate.py`, `agent/semantic_router.py`, and `agent/router.py` are removed when their replacement is complete.

## U1 — Segmented response contract

- **SPEC-1.1** — A response is an ordered sequence of segments. Each segment is grounded with at least one citation or ungrounded with zero citations.
- **SPEC-1.2** — Every citation in a grounded segment references a chunk admitted during that turn. Invented citations are validation errors.
- **SPEC-1.3** — An ungrounded segment carrying citations is unrepresentable.
- **SPEC-1.4** — Public response status is derived from segments, not assigned independently.
- **SPEC-1.5** — `QueryResponse.citations` is derived as the union of citations from grounded segments.
- **SPEC-1.6** — Segments originate from structured generation; the system does not split finished prose into segments after generation.

Public answer states: `answered_grounded`, `answered_mixed`, `answered_ungrounded`, `clarification_required`, `safety_blocked`, `error`. `insufficient_evidence` is retired.

U1 replacement budget target: unreachable pipeline code <= 1580 lines. `graph/conversational_agent.py` and `graph/state.py` are removed; citation validation is on the canonical path.

## U2 — Generic ingestion

- **SPEC-2.1** — Plain Markdown or text without front matter ingests successfully. Title is inferred; document identity is deterministic from content/source.
- **SPEC-2.2** — Initial generic ingestion formats are `.md` and `.txt` only. PDF, DOCX and HTML are deferred.
- **SPEC-2.3** — Valid rich front matter is preserved and enables governance policy.
- **SPEC-2.4** — Governance resolution (`supersedes`, `authority`, `effective_date`, `environment`) is optional and only applies where relevant metadata exists.
- **SPEC-2.5** — Present but malformed front matter fails loudly; it never silently degrades to a generic document.

The AcmePay corpus must ingest equivalently after U2. U2 replacement budget target: unreachable pipeline code <= 1212 lines; `agent/grounding.py` is removed when replaced.

## U3 — Direct and indirect prompt injection

- **SPEC-3.1** — Adversarial tests target the canonical prompt used by `agent/conversation.py`; tests against dead prompts do not count.
- **SPEC-3.2** — Retrieved evidence is delivered to the model explicitly delimited as untrusted data and escaped so document content cannot break the delimiter.
- **SPEC-3.3** — Instructions embedded in ingested documents do not change agent behavior; this is verified end-to-end.
- **SPEC-3.4** — A document containing injection text can still be cited as evidence without executing its instructions.
- **SPEC-3.5** — Direct secret extraction is `safety_blocked` before generation or tool execution.
- **SPEC-3.6** — Indirect-injection coverage includes documents without front matter.
- **SPEC-3.7** — An injected document cannot cause output or citation of another document that was not admitted in the turn.

Required deterministic gates: `prompt_injection_pass_rate: 1.00` and `critical_safety_pass_rate: 1.00`. U3 replacement budget target: unreachable pipeline code <= 1108 lines; obsolete `graph/prompts.py` is removed.

## U4 — Two measured relevance signals

- **SPEC-4.1** — `domain_relevance` is measured on raw pre-policy candidates; `grounded_relevance` is measured on admitted post-policy evidence.
- **SPEC-4.2** — `domain_relevance < DOMAIN_FLOOR` prohibits grounded segments, but does not prohibit a normal ungrounded answer.
- **SPEC-4.3** — Relevance floors come from calibration measurements, never guessed constants.
- **SPEC-4.4** — Every first turn performs the relevance search needed to produce a trace/score, including turns ultimately answered ungrounded.

Calibration uses `grounded`, `in_domain_unanswerable`, and `out_of_domain`. If `out_of_domain` and `in_domain_unanswerable` cannot be separated by the chosen signal, calibration stops and U4 is redesigned; no threshold is invented through overlap. Forced-search p95 regression >300 ms invalidates the design.

U4 replacement budget target: unreachable pipeline code <= 638 lines; `graph/workflow.py` and `graph/timed_workflow.py` are removed.

## U5 — Evaluation/RAGAS alignment

- **SPEC-5.1** — Release evaluation enforces per-case floors in addition to aggregate means.
- **SPEC-5.2** — Faithfulness is evaluated only on grounded segments.
- **SPEC-5.3** — Qwen3-4B remains the judge; agreement with 10 human labels is measured and release thresholds are derived from that agreement. If agreement is <=6/10, RAGAS becomes informational rather than a release gate.

Thresholds live in `evaluation/thresholds.yaml`, including `ragas_per_case.faithfulness` and `ragas_per_case.context_precision`.

## U6 — Evaluation corpus and final replacement

- **SPEC-6.1** — Golden statuses are remapped to the segmented contract: grounded answers -> `answered_grounded`; former insufficient-evidence cases -> `answered_mixed` or `answered_ungrounded` according to expected content.
- **SPEC-6.2** — `retrieval-calibration-v2.json` remains the three-class calibration dataset.
- **SPEC-6.3** — AcmePay is documented and treated as an evaluation suite, never as the product domain.

At U6 completion, obsolete `agent/react_agent.py` and legacy `scripts/react_*.py` are removed, unreachable pipeline budget is 0, the `graph/` runtime package no longer exists, and pipeline ownership is `{app, agent.conversation}`.

## Architecture fitness gates

Every implementation PR runs:

1. the unit's behavioral tests;
2. `test_unreachable_code_only_shrinks` using the current decreasing budget;
3. a canonical-path citation-validator assertion;
4. the existing single-pipeline ownership gate;
5. no `skip`/`xfail` introduced to make the unit green.

A spec without a test/gate is incomplete. A test against a module outside the canonical path is not evidence of coverage.