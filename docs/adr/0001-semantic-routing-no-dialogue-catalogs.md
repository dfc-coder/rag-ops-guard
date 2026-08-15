# ADR-0001: Semantic routing; no dialogue catalogues in production policy

Status: Accepted / Implemented by Conversational Grounding v3

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

The architecture is:

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

The v3 contract is:

```text
requires_grounding: bool
relation_to_context: same | new | none
operation: answer | transform | catalog
standalone_query: string | null
```

The schema may evolve, but its meaning must remain semantic and domain-independent.

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

## V3 implementation / ratchet

Conversational Grounding v3 removes the legacy heuristic router from production code.

`architecture/grounding-policy-guard.json` now declares a hard zero budget:

- maximum dialogue/entity string catalogues: zero;
- maximum regex dialogue patterns: zero.

`tests/architecture/test_grounding_policy_guard.py` scans both production routing modules so the
heuristic router cannot be reintroduced under a different file:

- `src/rag_ops_guard/agent/semantic_router.py`;
- `src/rag_ops_guard/agent/grounding.py`.

Natural-language examples are stored in evaluation data and can grow without changing production
routing rules.

## Regression policy

Natural-language examples belong in evaluation/test datasets, not in production routing code.

A newly discovered phrasing may be added as an eval case to measure semantic routing quality. Fixing a
failed eval may change the semantic contract/prompt/model or improve contextual state, but MUST NOT add
the phrase or entity as an application routing rule.

## Safe failure policy

The semantic resolver is probabilistic; the evidence controller is not. If structured routing fails
because of timeout, malformed output or model/tool incompatibility, the application fails closed to a
grounded retrieval attempt. With an existing grounded topic, the fallback query combines the stable
root with the current request. Without one, the current request itself is retrieved. Unsupported
retrieval then follows the normal abstention path.

This fallback deliberately optimizes safety over latency.

## Consequences

Positive:

- routing behavior scales across phrasing, language and domain entities;
- conversation policy is testable independently from natural-language interpretation;
- new systems do not require code changes;
- regressions become semantic-router/evidence-quality problems rather than endless heuristic patches;
- application safety remains deterministic;
- production language-heuristic budget is mechanically kept at zero.

Trade-offs:

- semantic routing adds one small structured model call per user turn;
- routing quality must be measured with eval datasets instead of assumed from unit rules;
- a small local model may need a better prompt or a dedicated router model if measured semantic
  accuracy is insufficient;
- resolver failure may cause a conservative extra retrieval before abstention.
