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
truncation, or unexpected graph/backend failure commits neither.

## Turn policy

| Policy | Meaning | Tool behavior |
|---|---|---|
| `DIRECT` | general chat/code/social/general definitions | no RAG tool |
| `REUSE_EVIDENCE` | transform active evidence for the same context | no new retrieval |
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

Known, unknown-capitalized, and supported lowercase operational target forms are treated as explicit
topic switches and do not inherit the previous query.

## Recall query vs ranking query

Contextual retrieval uses two inputs:

- `retrieval_query`: stable topic + current follow-up, used for dense/BM25 recall;
- `ranking_query`: the literal current user turn, also supplied to the reranker.

This keeps the entity/topic needed for recall while preventing the previous question from drowning
out the new intent (`after the third`, `who handles it`, `manual limit`, etc.).

## EvidenceWindow invariants

Evidence is reusable only when all of these are true:

1. the window is within its successful-turn TTL;
2. the current `system`, `environment`, and `api_version` filters match those used when evidence was
   retrieved;
3. the most recent retrieval was not unsupported;
4. the user asks for a transformation rather than a new internal fact.

If evidence expires or the user changes context filters, a transformation refreshes evidence instead
of silently reusing stale data. Topic/root state survives evidence expiry so an elliptical internal
follow-up still triggers `RETRIEVE` regardless of conversation length.

If a contextual retrieval returns `supported=false`, the previous evidence window is not destroyed,
but the failed retrieval is recorded and ambiguous transformations cannot silently treat old evidence
as the answer to the newly requested fact.

## Historical tool-evidence isolation

Raw `ToolMessage` evidence is not a durable source of truth. Before generation, tool calls/results
from older turns are removed from the model prompt while normal human/final-assistant conversational
history is retained. Current-turn tool protocol remains intact.

This prevents an expired production source from remaining visible to the model after the active
EvidenceWindow expires or after the user changes to staging/API-version filters. Reusable internal
evidence can only re-enter the prompt through the policy-controlled active `EvidenceWindow`.

## Retrieval safety guard

The explicit-anchor guard remains fail-closed, but only real entity-like tokens can veto retrieval.
Sentence-initial temporal words and structural/synthetic tokens do not become required anchors.
Lowercase targets in supported operational syntax remain fail-closed anchors.

```text
Cuantos reintentos permite Calypso. Y despues del tercero? -> anchor: Calypso
Despues del tercero, que pasa?                            -> no entity anchor
Calypso retries?                                          -> anchor: Calypso
Cuantos retries permite Xarlatan?                         -> anchor: Xarlatan
cuantos retries permite xarlatan?                         -> anchor: xarlatan
```

The guard checks the standalone retrieval query, while the reranker may additionally receive the
literal current turn.

## Relevance vs source authority

The learned reranker still decides whether evidence is relevant. Authority never rescues a candidate
that the reranker marked irrelevant.

For relevant candidates whose reranker scores are effectively tied (within a small bounded band),
document authority is used to choose the limited context slots before falling back to pure reranker
order. This prevents a low-authority vendor note from crowding an authority-100 canonical policy out
of a small context window merely because its relevance score is marginally higher.

This rule is specifically defensive against adversarial/supporting documents and does not lower any
relevance/admission threshold.

## Retrieval failures

`search_knowledge` catches retrieval backend failures and converts them to a structured
`supported=false` tool result with a machine-readable reason. Backend exception details are logged,
not exposed to the model/user.

Unexpected tool/protocol failures are not converted into arbitrary model-readable tool text; they
bubble to the existing transactional turn boundary so the previous conversation/grounding state is
preserved.

For same-topic follow-ups, active previous evidence may be supplied as explicit fallback context. It
can be used only when it directly supports the current question; otherwise the assistant must
abstain.

## Follow-up coverage

The policy handles short, long, causal, implicit-operational, and code follow-ups. Examples include:

```text
Y despues?
Que pasa luego?
Y si falla?
Quien interviene?
Cuando se escala?
Por que?
Como funciona eso?
Hay algun limite manual?
Cual es el procedimiento siguiente?
What happens next?
And after the third?
Who handles it?
How does that work?
```

A transformation word does not mask a new fact request: `Explicame que pasa despues del tercero`
retrieves. Pure transformations such as `Resumilo`, `Reformula`, `Translate it`, `Dame un ejemplo`,
or `Ponelo en una tabla` reuse active evidence.

Contextual code follows the same grounding contract: code based on `eso/that` reuses active evidence;
if the evidence expired it refreshes the stable root first; code asking for a new internal fact
retrieves with the literal coding request retained as ranking intent. Unrelated code remains direct.

## Acceptance matrix

The fast preflight runs compile checks, Ruff, strict mypy on the changed runtime modules, and a broad
unit matrix before starting Qwen/OpenVINO. It covers:

- initial internal fact -> `RETRIEVE`;
- general code/retry examples -> `DIRECT`;
- named/unknown internal code -> grounded behavior;
- contextual and causal follow-ups -> `RETRIEVE`;
- long and implicit operational follow-ups -> `RETRIEVE`;
- repeated follow-ups never grow the stable root query;
- known, unknown-capitalized, and inferred-lowercase topic switches do not inherit the prior topic;
- pure transformations -> `REUSE_EVIDENCE`;
- transformation wording plus a new fact -> `RETRIEVE`;
- expired evidence -> fresh retrieval while preserving topic;
- environment/API-version changes invalidate evidence reuse;
- unsupported retrieval preserves but cannot ambiguously reuse previous evidence;
- social/catalog side trips preserve topic as intended;
- unrelated general definitions/code clear or bypass old grounding as intended;
- contextual code remains grounded;
- synthetic/capitalization anchor false positives are rejected;
- lowercase inferred targets remain fail-closed;
- near-tied relevant context selection prefers authoritative sources without rescuing irrelevant ones;
- retrieval backend errors fail closed without leaking infrastructure details;
- old ToolMessages cannot leak stale evidence into new model prompts;
- timeout/truncation/stream failures remain transactional.

The hardware smoke additionally validates, before relying on final LLM wording:

1. initial Calypso retrieval admits an authority >= 90 source containing the three-retry rule;
2. the exact contextual retrieval plus literal ranking query admits evidence containing the Treasury
   Integrations escalation;
3. streamed first answer uses `policy=retrieve` and a tool call;
4. `Y despues del tercero?` re-grounds and answers Treasury Integrations;
5. the stable root query remains the original Calypso question after the follow-up;
6. `Resumilo en una linea` uses `policy=reuse_evidence` with zero new tool calls;
7. Chainlit imports/runs on the same agent API.

## Gate order

`make chainlit-gate` intentionally fails cheaply first:

```text
[1/5] compile + Ruff + mypy + grounding/retrieval/streaming unit matrix
[2/5] local Qwen/OpenVINO runtime + corpus integrity
[3/5] direct chat/code streaming
[4/5] real OpenVINO/Qwen conversational grounding smoke
[5/5] Chainlit import/config
```

This makes most regressions visible before model startup and makes a hardware retrieval failure
observable before the generation model can hide it behind fluent output.

## Non-goals

This work does not add durable history across application restarts, a second classifier LLM, new
embedding/reranking models, relaxed retrieval thresholds, or a message-count-based refresh rule.
