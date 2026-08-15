# ADR-0003: Post-retrieval correctness boundary

## Status

Proposed for the Chainlit grounding consolidation.

## Context

The conversational client needs to decide when internal evidence is required without turning a pre-retrieval intent classifier into the primary correctness mechanism. The V4/V5 semantic gate improved routing latency but most ambiguous turns intentionally failed closed to retrieval, so routing accuracy did not reliably predict user-visible correctness.

The retrieval pipeline already has stronger evidence: actual corpus candidates plus learned query-document relevance scores. That is the appropriate place to decide whether internal evidence is admissible.

## Decision

Correctness is enforced in this order:

1. `SafetyGuard` runs before semantic routing and blocks explicit secret extraction or policy bypass deterministically.
2. The semantic gate is only a high-precision optimization for bypassing retrieval on clearly direct turns. Ambiguous decisions continue to retrieve.
3. Retrieval performs dense + lexical recall and deterministic document-policy resolution.
4. The learned reranker scores resolved candidates against the effective retrieval intent.
5. `retrieval_min_relevance` is the explicit admission floor. Candidates below it cannot become evidence or citations regardless of document authority.
6. The admission floor is calibrated with labeled positive and negative retrieval cases, including off-domain coding prompts.
7. Generation can use only admitted evidence for new internal operational facts.

## Consequences

- False retrieval is primarily a latency cost instead of a citation-correctness failure.
- Off-domain queries can retrieve candidates internally but must still produce `supported=false` when no candidate clears the calibrated admission floor.
- Safety regressions are testable on the exact `ReactAgent` path used by Chainlit.
- Routing thresholds and retrieval-admission thresholds are separate controls with separate evaluation datasets.
- Query rewriting or descontextualization can evolve independently later without becoming a policy controller.

## Validation

`make chainlit-gate` must cover deterministic safety, retrieval admission calibration, direct-path smoke tests, conversational RAG smoke tests, and Chainlit import/configuration before promotion. Repository CI must also pass on the final consolidation commit before the PR is taken out of draft.