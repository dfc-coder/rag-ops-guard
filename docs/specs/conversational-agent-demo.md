# Conversational Ops Agent — demo specification

## Goal

Provide a bounded conversational agent that separates conversation memory from retrieval state, answers product/catalog questions from deterministic application data, and uses grounded hybrid retrieval for operational knowledge questions.

## Constraints

- Keep Qwen3.5-0.8B, llama.cpp CPU T8, Floci, S3 Vectors and grounded-answer sampling unchanged.
- Normal knowledge turns use one Qwen answer generation. A second, low-token rewrite call is allowed only when first-pass retrieval is below the relevance threshold and a prior grounded focus exists.
- No entity-specific routing rules (`Calypso`, `SendGrid`, etc.).
- Keep version/authority/supersedes and citation validation deterministic.
- Demo memory is thread-scoped and in-memory; persistent production memory is out of scope.

## Target flow

```text
User
  -> MessagesState + InMemorySaver
  -> confidence-aware semantic route
       chat         -> deterministic-sampling light chat
       capabilities -> deterministic application capabilities
       catalog      -> deterministic active-document catalog
       out_of_scope -> deterministic scope response
       knowledge    -> current question only -> hybrid retrieval
       uncertain    -> current question only -> hybrid retrieval
                          relevant -> knowledge
                          weak     -> best non-knowledge control intent

knowledge retrieval
  -> dense top-k + BM25 top-k
  -> RRF fusion
  -> EvidenceResolver
  -> generic rerank
  -> relevance gate
       relevant -> grounded answer
       weak + grounded focus -> standalone query rewrite
                                  -> retrieve again
                                  -> relevance gate
       weak -> insufficient evidence
  -> answer + citations
```

## Contracts

### Semantic router

Input: current user message only.
Output: a route decision with `route`, best similarity score, margin and per-route scores.

The router scores individual bilingual intent examples instead of mandatory class centroids. It may abstain as `uncertain` when minimum score or margin is not met. It does not contain domain/entity rules.

`uncertain` is not a terminal user-facing route. The agent resolves it against the real knowledge corpus. If hybrid retrieval returns relevant admitted evidence, the final route becomes `knowledge`. Otherwise the agent selects the strongest non-knowledge control intent from the router scores. This prevents close embedding scores from suppressing evidence for named systems and services.

### Deterministic product tools

`capabilities` describes only capabilities implemented by the application. `catalog` lists active documents from the real chunk/object corpus and does not ask the LLM to invent what is available.

### search_knowledge

First-pass input is always the current question, never concatenated conversation history. Retrieval remains the single production path:

`Dense + BM25 -> RRF -> EvidenceResolver -> rerank -> top context`.

The result includes a deterministic relevance score combining semantic distance and lexical coverage.

### Trusted conversation focus

Raw assistant text is never used as grounded retrieval context. After a successful knowledge answer, the state stores only:

- the retrieval query that produced the answer;
- titles of the validated cited sources.

That grounded focus may be used for a later follow-up.

### Conditional standalone-query rewrite

A rewrite is attempted only when:

1. the current turn is routed to knowledge or reaches retrieval through `uncertain`;
2. first-pass retrieval relevance is below the configured threshold; and
3. a trusted grounded focus exists from a previous successful knowledge answer.

The rewrite receives only the current question, prior grounded query and validated source titles. It does not receive free-form assistant history, does not answer the question, and is used only if the rewritten retrieval improves relevance.

### Generation

- Chat route is restricted to greetings/thanks/light conversation and uses the deterministic local sampler.
- Capabilities, catalog and out-of-scope responses are deterministic.
- Grounded generation receives only the latest user question and admitted evidence; assistant conversation history is not passed into the grounding prompt.
- Grounded `answered` responses require validated citations.
- User-facing output follows the language of the current user question.

## Observability

Each turn exposes/traces:

- original semantic route, confidence, margin and route scores;
- final resolved route;
- original question;
- retrieval query;
- rewritten query when used;
- dense, BM25 and admitted titles;
- relevance score;
- route/search/rewrite/generation timings;
- citations.

## Demo acceptance cases

1. `Hola` -> `chat`; response stays in Spanish for Spanish input.
2. `¿Qué puedes hacer?` -> `capabilities`, no grounded generation and no invented product claims.
3. `¿Qué documentación tienes disponible?` -> `catalog`, listing the active KB documents actually present.
4. `¿Qué pasa con SendGrid?` may initially be semantically uncertain, but relevant KB evidence resolves the final route to `knowledge`.
5. `¿Qué haces?` may initially be semantically uncertain; if KB evidence is weak, the final route falls back to `capabilities` when that is the strongest control intent.
6. `¿Cuál es el objetivo de Calypso Payments API?` -> `knowledge`, using only that current question for first-pass retrieval.
7. A previous casual/meta turn must never be concatenated into a later retrieval query.
8. Follow-up `¿Y qué pasa después del tercero?` may use the previous grounded knowledge focus; if first-pass relevance is weak it is rewritten into a standalone query and retrieved again.
9. Free-form assistant replies are never passed into grounded generation.
10. An unrelated question such as `¿Cuál es la capital de Francia?` does not force internal-document retrieval when the semantic route is confidently out-of-scope.
11. A deprecated policy never wins over its active superseding document.
12. Citations map only to admitted evidence.
13. LangSmith, when enabled, traces routing, retrieval, optional rewrite and generation under the same request/thread metadata.

## Demo non-goals

- Multi-agent orchestration.
- Long-term user memory.
- Postgres/Redis checkpointer.
- OpenSearch/Elasticsearch.
- LLM relevance grading.
- Model-size or inference-backend changes.

## Validation

The implementation is validated as one integrated `develop` state. Unit, property, Floci integration, security, CDK, formatting, linting and strict typing must all pass before the demo is considered ready for live manual validation.
