# Conversational Grounding v2

## Goal

Grounding decisions are application policy, not a probabilistic side effect of the local 2B model.
The generation model produces language; the application decides whether a turn is direct, needs new
internal evidence, can reuse evidence, or asks for the knowledge catalog.

## Core state

`ConversationState` is stored per thread independently from raw chat history. It tracks the stable
grounded topic/root query, last retrieval outcome, and an `EvidenceWindow` containing the exact
admitted source text used by the agent.

Message history and grounding state commit together only after a successful turn. Timeout, cancel,
truncation, or backend failure commits neither.

## Turn policy

| Policy | Meaning | Tool behavior |
|---|---|---|
| `DIRECT` | general chat/code/social/general definitions | no RAG tool |
| `REUSE_EVIDENCE` | transform evidence already active for the same context | no new retrieval |
| `RETRIEVE` | new internal fact or evidence refresh required | deterministic `search_knowledge` |
| `LIST_KNOWLEDGE` | explicit documentation catalog request | deterministic `list_knowledge` |

There is no periodic “RAG every N turns” rule.

## Stable follow-up resolution

A contextual follow-up is resolved from a stable root query, not from the previous rewritten query.
This prevents query growth across long conversations.

```text
root:      Cuantos reintentos permite Calypso?
turn 2:    Y despues del tercero?
retrieval: Cuantos reintentos permite Calypso. Y despues del tercero?

turn 3:    Y quien interviene?
retrieval: Cuantos reintentos permite Calypso. Y quien interviene?
```

The second query never contains the first follow-up. Synthetic labels such as `Follow-up:` are not
inserted because they can be mistaken for explicit entity anchors by fail-closed retrieval guards.

If a new target is explicit, including an unknown capitalized operational target such as `Xarlatan`,
the previous topic is not inherited.

## Recall query vs ranking query

Contextual retrieval uses two inputs:

- `retrieval_query`: stable topic + current follow-up, used for dense/BM25 recall;
- `ranking_query`: the literal current user turn, also supplied to the reranker.

This prevents the previous question from overwhelming the new intent while still retaining the
entity/topic needed for recall.

## EvidenceWindow invariants

Evidence is reusable only when all of these are true:

1. the window is within its successful-turn TTL;
2. the current `system`, `environment`, and `api_version` filters match those used when evidence was
   retrieved;
3. the most recent retrieval was not unsupported;
4. the user asks for a transformation rather than a new internal fact.

If evidence expires or the user changes environment/API filters, a transformation re-retrieves
instead of silently reusing stale evidence.

If a contextual retrieval returns `supported=false`, the old evidence is not overwritten, but an
ambiguous subsequent transformation is not allowed to treat that stale evidence as the answer to the
failed new fact.

## Expired evidence is not forgotten topic

The evidence cache may expire while the grounded topic remains known. Therefore an elliptical turn
such as `Y despues?` after expiry triggers `RETRIEVE`; it does not become a direct model answer.
This keeps internal facts grounded regardless of conversation length.

## Retrieval safety guard

The explicit-anchor guard remains fail-closed, but only real entity-like tokens can veto retrieval.
Sentence-initial temporal words and structural/synthetic tokens do not become required anchors.
Examples covered by regression tests:

```text
Cuantos reintentos permite Calypso. Y despues del tercero? -> anchor: Calypso
Despues del tercero, que pasa?                            -> no entity anchor
Calypso retries?                                          -> anchor: Calypso
Cuantos retries permite Xarlatan?                         -> anchor: Xarlatan
```

The guard checks the standalone retrieval query, while the reranker may additionally receive the
literal current turn.

## Retrieval failures

`search_knowledge` catches retrieval backend failures and converts them to a structured
`supported=false` tool result. This fails closed without exposing infrastructure details or letting a
backend exception turn into an invented internal fact.

For same-topic follow-ups, previously retrieved evidence may be supplied as fallback context to the
model. It can be used only when it explicitly answers the current question; otherwise the assistant
must abstain.

## Acceptance matrix

The targeted unit gate covers:

- initial internal fact -> `RETRIEVE`;
- generic retry/code request -> `DIRECT`;
- code depending on an internal fact -> `RETRIEVE`;
- contextual follow-up -> `RETRIEVE`;
- repeated follow-ups do not grow the root query;
- known and unknown explicit topic switches do not inherit the previous query;
- active compatible evidence -> `REUSE_EVIDENCE`;
- expired evidence -> re-retrieval;
- environment change -> re-retrieval;
- API-version change -> re-retrieval;
- unsupported retrieval does not overwrite previous evidence and blocks stale ambiguous reuse;
- social messages preserve the topic;
- unrelated general definitions clear the old topic;
- synthetic/capitalization anchor false positives are rejected;
- timeout/truncation rollback remains transactional.

The hardware smoke additionally validates:

1. initial Calypso retrieval admits an authority >= 90 source containing the three-retry rule;
2. the exact contextual query plus literal ranking query admits evidence containing the Treasury
   Integrations escalation before generation runs;
3. streamed first answer uses `policy=retrieve` and a tool call;
4. `Y despues del tercero?` re-grounds and answers Treasury Integrations;
5. the stable root query remains the original Calypso question after the follow-up;
6. `Resumilo en una linea` uses `policy=reuse_evidence` with zero new tool calls;
7. Chainlit imports/runs on the same agent API.

## Non-goals

This work does not add durable history across application restarts, a second classifier LLM, new
embedding/reranking models, relaxed retrieval thresholds, or a message-count-based refresh rule.
