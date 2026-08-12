# MVP Technical Implementation Plan — v0.1.0-beta.1

## Beta definition

Beta 1 is accepted when a fresh clone can run the complete operational RAG locally with no AWS account and no hosted inference API.

Expected happy path:

```bash
make doctor
make setup
make models
make local-up
make local-provision
make seed
make smoke
```

## Deliverables

1. Reproducible dependency locks for Python and TypeScript.
2. AcmePay sample operational knowledge corpus.
3. Typed document metadata schema and Markdown parser.
4. Deterministic chunking and stable chunk IDs.
5. Qwen3 local embeddings over llama.cpp.
6. Floci S3 + S3 Vectors persistence/retrieval.
7. Idempotent ingestion Lambda and `/v1/ingest` endpoint.
8. LangGraph query workflow and `/v1/query` endpoint.
9. Deterministic version/evidence resolution.
10. Grounded answer generation and citation validation.
11. Abstention, ambiguity and safety routes.
12. 30-case golden dataset and adversarial subset.
13. pytest, Hypothesis and Floci integration suites.
14. Real-model E2E validation on a trusted 8-thread runner.
15. RAGAS release evaluation artifacts.
16. Optional LangSmith tracing profile.
17. CDK-synthesizable target AWS architecture.
18. GitHub CI and release quality gates.

## Development slices

### Slice 1 — Foundation

Repository structure, typed domain, ports/adapters, test harness, docs, dependency locks and CI.

### Slice 2 — Local runtime

Floci, llama.cpp generation and embedding services, pinned images/models and model checksum verification.

### Slice 3 — Ingestion and retrieval

Markdown metadata, chunking, embedding generation, S3 storage, S3 Vectors indexing, manifests and idempotency.

### Slice 4 — Query graph

Query analysis, semantic retrieval, metadata/version resolver, abstention, grounded generation and citations.

### Slice 5 — Evaluation and release

Golden/adversarial datasets, property tests, RAGAS, optional LangSmith experiments, clean-clone reproduction and release gate.

## Explicit non-goals

No frontend, auth, PDF/OCR, agent tools, MCP, reranker, hybrid search, multi-tenancy, conversation history or real AWS deployment in Beta 1.
