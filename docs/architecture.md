# Architecture

## Canonical ownership

Only two modules own the conversational pipeline:

- `rag_ops_guard.app`: dependency composition and the singleton `conversation_agent()` factory.
- `rag_ops_guard.agent.conversation`: conversation state, deterministic safety boundary, non-routing relevance probe, model-driven tool loop, streaming and response assembly.

Chainlit, Gradio and REST are adapters to that same core. There is no `rag_ops_guard.graph` package and no alternate `ReactAgent`/router workflow.

## Turn flow

```text
User request
    |
    v
SafetyGuard  ---- blocked ----> safety_blocked
    |
    v
silent relevance probe
(domain_relevance + grounded_relevance telemetry; never routes the turn)
    |
    v
Qwen3-4B tool-calling model
    |
    +------ direct response --------------------+
    |                                           |
    +------ search_documents                    |
    |          |                                |
    |          v                                |
    |      dense + BM25                         |
    |          |                                |
    |          v                                |
    |         RRF                               |
    |          |                                |
    |          +--> domain_relevance            |
    |          |                                |
    |          v                                |
    |   optional governance resolver            |
    |          |                                |
    |          v                                |
    |   reranker / anchor checks                |
    |          |                                |
    |          +--> grounded_relevance          |
    |          |                                |
    |          v                                |
    |     admitted evidence --------------------+
    |
    +------ list_documents ---------------------+
                                                |
                                                v
                                      structured segments
                                                |
                                                v
                                     citation validation
                                                |
                                                v
                                      QueryResponse status
```

The model decides whether document tools are required. No deterministic semantic router fabricates tool calls.

## Relevance semantics

`domain_relevance` measures affinity to the corpus before governance resolution. `grounded_relevance` measures support after resolver/target checks. The admission threshold is constrained by both floors: corpus affinity below the domain floor cannot produce grounded evidence.

The non-tool probe runs on every safe turn so direct answers still produce relevance telemetry; it is not inserted into conversation history and does not count as a tool call.

## Document model

Generic ingestion requires only derived/basic document identity. Markdown and text documents may omit YAML front matter. Optional governance metadata activates status/environment/version/authority/supersession rules when present.

Current pipeline:

```text
.md / .txt
    -> parse_document
    -> DocumentMetadata + body
    -> MarkdownChunker
    -> embeddings
    -> S3 Vectors-compatible store
```

## Trust boundary

1. User secret-extraction/policy-bypass patterns are checked before model and retrieval.
2. Retrieved text is marked `UNTRUSTED_DOCUMENT_DATA` and treated only as source data.
3. Only admitted chunks enter the model's document observation.
4. Grounded segments may cite only admitted chunk IDs from the current turn.
5. The public status is derived from validated segment citations.

## Infrastructure boundary

Runtime code depends on ports for object storage, vectors, embeddings, reranking and tool calling. Local adapters use Floci, llama.cpp and OpenVINO-compatible endpoints. AWS/CDK infrastructure remains outside conversational decision logic.

## Architecture fitness

`tests/architecture/test_pivot_replacement.py` enforces:

- canonical owners are `app` + `agent.conversation`;
- the old `graph/` package is absent;
- the core has no LangChain/LangGraph imports;
- unreachable pipeline code is exactly zero;
- citation validation remains reachable from the canonical path.

Ruff is advisory. Architecture, mypy, unit/property/integration, adversarial security and dependency/security checks are blocking.
