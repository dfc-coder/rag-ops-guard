# ADR 0002: Retrieval-gated semantic control for conversational grounding

- Status: proposed
- Date: 2026-08-15

## Context

Conversational Grounding v2 encoded natural-language meaning in phrase/entity/regex rules. V3
removed those rules but put a generative Qwen 3.5 2B resolver on every turn. The Fedora/Tiger Lake
gate showed that the V3 router was both inconsistent and expensive: correlated fields could disagree,
an explicit `xarlatan` subject could be rewritten as `Calypso`, and warm routing took roughly
4–5 seconds before retrieval or answer generation.

The repository already has a simpler bounded pattern in `graph/conversational_agent.py`: use a cheap
semantic decision, probe retrieval for uncertain knowledge turns, and let deterministic evidence
admission decide whether an internal fact is supported.

## Decision

V4 uses only three semantic actions in the hot path:

- `direct`: no new private/internal operational evidence is required.
- `retrieve`: a new private/internal operational fact or verification is required.
- `catalog`: list the available internal knowledge sources.

The embedding gate may also emit `uncertain`. `uncertain` is not a user intent; it is a control state
that always fails closed to `retrieve`.

The gate uses the existing OpenVINO embedding service. It receives the literal current user request
plus two typed booleans indicating whether grounded context and compatible evidence exist. It does
not receive previous entity names or generated rewrites.

There is exactly one abstract semantic description per action. Dialogue examples, entity names,
pronoun catalogs, language-specific prefixes, and regex routing remain forbidden in production code.
Examples belong only in evaluation data.

Retrieval query construction is deterministic. When grounded context exists, the trusted stable root
is concatenated with the literal current turn for recall. The literal current turn is always passed
separately as the reranker query. No generative model may invent or rewrite the retrieval subject.

Pure transformations such as translation, summarization, or code generation over visible conversation
history use the normal direct path. They do not need a separate evidence-reuse intent.

Qwen remains the answer generator. It is not a routing model.

## Invariants

- Internal facts are answered only from admitted current-turn evidence.
- Uncertain routing never falls through to direct generation.
- The current literal turn is always preserved for reranking.
- Evidence TTL, environment/API-version compatibility, authority, abstention, and transactional
  commit/rollback remain deterministic.
- No production dialogue/entity catalogs or regex conversational routing.
- A new language phrasing may change evaluation data or learned weights; it must not add a source-code
  branch.

## Consequences

The routing surface is smaller than V3, contradictory structured-output states disappear, and the
multi-second generative-router call leaves the hot path. The beta may over-retrieve on genuinely
ambiguous turns; this is an intentional safety trade-off. Routing quality and latency are measured
against a frozen evaluation set before promotion.
