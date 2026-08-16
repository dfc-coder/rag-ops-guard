# ADR-0005: Mixed responses replace hard abstention

## Status

Accepted for the generic-agent pivot.

## Context

The original product treated document grounding as the reason for the application to exist. Its effective invariant was:

> If document evidence is insufficient, do not answer.

That rule is appropriate for a narrow corpus-bound copilot, but it conflicts with the new product: a generic conversational ReAct agent where document RAG is one capability among others.

A generic agent must be able to answer ordinary questions, use non-document tools, and combine document-backed information with model/general explanation without pretending that every sentence came from the corpus.

The physical beta-freeze gate on `agent/beta-unified-react-core` also demonstrated that forcing one binary retrieval-admission threshold to protect every application behavior is too coarse. The zero-false-admission threshold reached `0.786295`, but grounded recall dropped to `0.333` (3 true positives, 6 false negatives). A release gate correctly rejected that tradeoff.

## Decision

Retire the hard-abstention invariant and replace it with:

> No claim may be presented as grounded unless it is supported by evidence admitted in that turn.

Responses may contain a mixture of grounded and ungrounded segments.

The public response contract is therefore segment-based:

- a grounded segment has one or more citations;
- an ungrounded segment has zero citations;
- citations must refer to chunks admitted in the current turn;
- response status is derived from the segment set;
- top-level citations are derived from grounded segments.

The target public statuses are:

```text
answered_grounded
answered_mixed
answered_ungrounded
clarification_required
safety_blocked
error
```

`insufficient_evidence` is retired as a final public answer status. Lack of document support may be reported explicitly in an ungrounded segment. It must never be disguised as a grounded claim.

Segments must ultimately come from structured generation/representation. Splitting free-form prose after generation is not sufficient evidence of grounding provenance.

## Alternatives considered

### Keep hard abstention

Rejected. It makes the document corpus a global scope gate and prevents the agent from behaving as a general ReAct system.

### Keep response-level grounded/ungrounded only

Rejected. A single boolean/status cannot represent a response that combines a cited document fact with an uncited explanation.

### Allow citations anywhere and rely on prompt instructions

Rejected. Citation integrity is a data-contract invariant, not a best-effort prompting concern.

## Consequences

Positive:

- normal conversation no longer depends on corpus coverage;
- grounded claims remain auditable;
- mixed answers become representable without laundering model knowledge as document evidence;
- RAG evaluation can focus faithfulness on grounded segments rather than penalizing legitimate ungrounded content.

Costs:

- the response model becomes more structured;
- generation must support or emulate reliable structured output;
- citation validation must move from whole-response checks to segment-level integrity;
- datasets and RAGAS evaluation require status/metric realignment.

## Migration impact

U1a verifies the tool/model abstraction and structured-output capability.

U1 implements the segmented response contract.

U5 realigns RAGAS so faithfulness applies only to grounded segments and adds per-case release floors.

This ADR supersedes the hard-abstention portions of earlier retrieval-gating decisions. Historical ADRs remain useful records of the previous product phase.
