# Conversational Ops Agent — demo specification

## Goal

Replace the stateless RAG hot path with a bounded conversational agent that can answer normal conversational/meta questions without retrieval and can use a grounded knowledge-search capability for operational questions.

## Constraints

- Keep Qwen3.5-0.8B, llama.cpp CPU T8, Floci, S3 Vectors and the current sampling profile unchanged.
- At most one Qwen generation per user turn.
- No entity-specific routing rules (`Calypso`, `SendGrid`, etc.).
- Keep version/authority/supersedes and citation validation deterministic.
- Demo memory is thread-scoped and in-memory; persistent production memory is out of scope.

## Target flow

```text
User
  -> MessagesState + InMemorySaver
  -> semantic route: chat | knowledge
       chat      -> one Qwen generation
       knowledge -> search_knowledge
                    -> dense top-k
                    + BM25 top-k
                    -> RRF fusion
                    -> EvidenceResolver
                    -> generic rerank
                    -> top context
                    -> one grounded Qwen generation
  -> answer + citations
```

## Contracts

### Semantic router

Input: recent conversation + current user message.
Output: `chat` or `knowledge`.
The router uses embeddings and intent exemplars, not domain/entity string rules and not another LLM call.

### search_knowledge

Input: contextualized query + `QueryContext`.
Output: admitted evidence ordered by relevance.
It is the single retrieval implementation used by the runtime and later by evaluation.

### Conversation

The caller supplies a `thread_id`. Messages in the same thread are retained by a LangGraph checkpointer. Different threads are isolated.

### Generation

- Chat route: conversational/product-level response, no operational facts invented.
- Knowledge route: answer only from admitted evidence and return evidence refs for citations.
- User-facing output follows the language of the current user question.

## Demo acceptance cases

1. `Hola, ¿qué haces?` routes to chat and performs no knowledge search.
2. `¿Qué me puedes contar de SendGrid?` routes to knowledge and retrieves the SendGrid runbook/landscape when available.
3. `¿Cuántos reintentos permite Calypso en la Payment API?` returns the active three-retry policy.
4. Follow-up `¿Y qué pasa después del tercero?` in the same thread retains the previous subject and retrieves the escalation evidence.
5. A deprecated policy never wins over its active superseding document.
6. Citations map only to admitted evidence.
7. LangSmith, when enabled, traces route, knowledge search and generation under the same request/thread metadata.

## Demo non-goals

- Multi-agent orchestration.
- Long-term user memory.
- Postgres/Redis checkpointer.
- OpenSearch/Elasticsearch.
- LLM query rewriting or LLM relevance grading.
- Model-size or inference-backend changes.
