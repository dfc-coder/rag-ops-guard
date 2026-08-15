# ADR-0001: Semantic routing; no dialogue catalogues in production policy

Status: Accepted

Date: 2026-08-15

## Context

The conversational grounding layer accumulated phrase lists, entity lists, stopword lists and regexes
in order to classify user turns such as follow-ups, transformations, internal facts and topic switches.
That approach is locally convenient but does not scale with language, phrasing, new domains or new
entities. It also couples domain vocabulary to application control flow and creates repeated failures
when a semantically equivalent user message is worded differently.

## Decision

Natural-language meaning MUST NOT be implemented as growing Python catalogues, prefix/suffix tables,
entity allowlists, stopword exceptions or regex-based dialogue routing.

The target architecture is:

```text
user message + compact conversation state
                  |
                  v
        semantic structured resolver
                  |
                  v
            TurnDecision
                  |
                  v
      deterministic grounding controller
          |          |          |
        direct     retrieve    reuse
                     |
                     v
              evidence controls
                     |
                     v
                 generator
```

The semantic resolver owns language understanding and emits a small validated structured contract.
The deterministic controller owns only system invariants.

A representative contract is:

```text
requires_grounding: bool
relation_to_context: same | new | none
operation: answer | transform
standalone_query: string | null
```

The exact schema may evolve, but its meaning must remain semantic and domain-independent.

## Deterministic code MAY enforce

- evidence is required before asserting internal operational facts;
- unsupported retrieval causes abstention rather than invention;
- environment/system/API-version compatibility;
- EvidenceWindow TTL and invalidation;
- transactional commit and rollback;
- timeout/cancel/truncation recovery;
- source authority, relevance and admission rules;
- security and prompt-injection boundaries;
- schema validation and safe fallback when semantic routing fails.

## Deterministic code MUST NOT encode

- phrases such as `y si`, `what happens`, `translate it`, etc. as routing rules;
- lists of known internal systems/entities to decide whether a question is internal;
- language-specific follow-up prefixes or pronoun catalogues;
- regexes whose purpose is to infer conversational intent/entity semantics;
- one-off exceptions added because a regression phrase failed.

Entities are data, not code. If a user asks about a previously unseen system, retrieval determines
whether evidence exists. The application does not need a source-code entry for that entity.

## Migration / ratchet

The current v2 implementation still contains legacy heuristic catalogues. They are frozen technical
debt, not an approved extension point.

`architecture/grounding-policy-guard.json` records the maximum legacy budget. The architecture test
`tests/architecture/test_grounding_policy_guard.py` enforces a ratchet:

- existing catalogues may shrink or disappear;
- their literal/regex counts may never increase;
- no new top-level dialogue/entity string catalogue may be introduced;
- the declared target is zero language literals and zero regex dialogue patterns.

Conversational Grounding v3 must drive those budgets to zero as the semantic resolver replaces the
legacy router.

## Regression policy

Natural-language examples belong in evaluation/test datasets, not in production routing code.

A newly discovered phrase may be added as an eval case to measure semantic routing quality, but fixing
the eval must not require adding that phrase/entity to an application rule list.

## Consequences

Positive:

- routing behavior scales across phrasing, language and domain entities;
- conversation policy becomes testable independently from natural-language interpretation;
- new systems do not require code changes;
- regressions become router-quality/evidence-quality problems rather than endless heuristic patches;
- application safety remains deterministic.

Trade-offs:

- semantic routing is probabilistic and must use schema validation, confidence/fallback behavior and
  evaluation datasets;
- a small local model may require measurement or replacement if it cannot reliably satisfy the
  structured routing contract;
- the v2 heuristic layer must be removed incrementally rather than extended.
