# Conversational Grounding v3

## Purpose

V3 removes natural-language routing heuristics from production code. The application no longer
contains phrase lists, entity allowlists, pronoun tables, capitalization rules, or regexes that try to
infer conversational meaning.

## Runtime flow

```text
current user message
+ compact conversation state
          |
          v
SemanticTurnResolver (Qwen, forced structured tool call)
          |
          v
TurnDecision
  - requires_grounding
  - relation_to_context: same | new | none
  - operation: answer | transform | catalog
  - standalone_query
          |
          v
GroundingController (deterministic)
          |
    +-----+------+----------------+
    |            |                |
  DIRECT    REUSE_EVIDENCE     RETRIEVE
                                  |
                                  v
                         hybrid RAG + reranker
                                  |
                                  v
                              generator
```

## Semantic resolver responsibility

The resolver interprets language and emits one validated `TurnDecision`. It does not answer the user
and does not execute retrieval. A forced tool call is used instead of parsing free-form JSON.

The resolver receives only compact state needed for reference resolution:

- grounded topic/root query;
- whether compatible evidence is active;
- previous successful user/assistant exchange;
- current system/environment/API-version filters.

It does not receive entity catalogs.

## Deterministic controller responsibility

The controller does not understand language. It enforces invariants:

- a semantic decision requiring grounding always routes through retrieval;
- a same-context transform reuses active compatible evidence;
- expired/incompatible evidence is refreshed before a grounded transform;
- a new subject does not inherit evidence from the previous subject;
- knowledge-catalog requests route to the catalog tool;
- environment/system/API-version changes invalidate evidence reuse;
- unsupported retrieval cannot be replaced by model invention;
- message history and grounding state still commit transactionally.

## Safe resolver failure

If the semantic resolver times out, returns malformed structured output, or otherwise fails, routing
fails closed. The controller performs a grounded retrieval using the current message and, when a
stable grounded root exists, includes that root in the fallback query. If retrieval cannot support the
request, the existing abstention behavior applies.

This fallback may be slower than a correct semantic route, but it does not introduce language
heuristics or permit unsupported internal claims.

## Zero-heuristic guardrail

`architecture/grounding-policy-guard.json` is ratcheted to:

```text
max dialogue/entity string catalogs = 0
max regex dialogue patterns         = 0
```

`tests/architecture/test_grounding_policy_guard.py` scans both production routing modules:

- `agent/semantic_router.py`
- `agent/grounding.py`

Natural-language examples belong in `tests/evals/semantic_turn_cases.jsonl`; adding a new phrasing to
the eval cannot justify adding a new production routing rule.

## Validation layers

`make chainlit-gate` now validates six layers:

1. compile, Ruff, mypy, architecture guard and deterministic unit tests;
2. local Qwen/OpenVINO runtime and corpus integrity;
3. real local-Qwen semantic routing evaluation;
4. direct chat/code streaming;
5. full conversational grounding + retrieval + evidence reuse;
6. Chainlit import/config.

The semantic-router smoke is intentionally separate from RAG generation so a routing-quality failure
cannot be hidden behind a fluent final answer.
