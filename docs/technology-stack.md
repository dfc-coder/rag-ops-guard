# Technology Stack — v0.1.0-beta.1

This document freezes the Beta 1 stack. Changes require an ADR.

## Runtime

| Layer | Choice | Beta 1 role |
|---|---|---|
| Language | Python 3.12 | RAG application, Lambda handlers, tests and evaluation |
| Package manager | uv 0.12.3 | Reproducible Python dependency resolution |
| Validation | Pydantic v2 | API/domain/LLM structured contracts |
| AWS SDK | boto3 | S3, S3 Vectors, Lambda/API provisioning against Floci |
| IaC | AWS CDK v2 + TypeScript | Synthesizable target AWS architecture |
| Node | Node.js 24 | CDK toolchain |

## AI

### Generation

- Runtime: `llama.cpp` CPU-only.
- Container: `ghcr.io/ggml-org/llama.cpp:server-b9445` pinned to the repository's multi-arch manifest digest.
- Model: `Qwen3-4B-Q4_K_M.gguf`.
- Model SHA256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- Context: 8192.
- Threads: 8.
- Parallel slots: 1.
- Maximum generated tokens: 256.
- Qwen mode: `/no_think` for operational RAG prompts.

### Embeddings

- Runtime: independent `llama.cpp` server.
- Model: `Qwen3-Embedding-0.6B-Q8_0.gguf`.
- SHA256: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439`.
- Dimensions: 1024.
- Pooling: `last`.
- Application performs L2 normalization before vector storage/query.

## RAG framework

- LangChain: OpenAI-compatible adapters for the local llama.cpp HTTP interfaces, document/prompt abstractions and structured outputs.
- LangGraph: explicit state machine for query analysis, retrieval, evidence resolution, abstention, generation and citation validation.
- LangSmith: optional tracing/evaluation profile. It is not required to execute Beta 1 offline.
- RAGAS: release evaluation metrics, executed with the local models rather than a hosted model API.

## AWS-compatible local runtime

Floci `1.5.34` emulates:

- S3 — document/chunk/manifest storage.
- S3 Vectors — vector index and similarity retrieval.
- Lambda — query and ingestion handlers.
- API Gateway v2 — HTTP API.
- IAM — local role/control-plane validation.
- CloudWatch — target logging/metrics integration.

Floci Bedrock Runtime is deliberately not used because its model responses are stubs rather than real inference.

## Vector store

S3 Vectors configuration:

- vector bucket: `rag-ops-guard-vectors-local`;
- index: `ops-knowledge-v1`;
- data type: `float32`;
- dimensions: `1024`;
- distance metric: `cosine`;
- initial retrieval Top K: `8`;
- context after deterministic resolution: maximum `5` chunks;
- no hard vector-distance cutoff in Beta 1 until calibrated against the golden dataset.

## Testing

- pytest — unit/integration/regression tests.
- Hypothesis — property-based input and invariant testing.
- pytest-cov — 80% minimum application coverage.
- synthetic embeddings — deterministic Floci/S3 Vectors CI.
- real Qwen E2E — trusted release runner only.
- RAGAS — AI quality metrics.
- LangSmith — optional experiment/tracing evidence.

## Intentionally excluded from Beta 1

FastAPI, LlamaIndex, Chroma, FAISS, Pinecone, Qdrant, pgvector, Redis, PostgreSQL, reranker models, hybrid search, PDFs/OCR, frontend, authentication, conversation memory, tool-calling agents and real AWS deployment.
