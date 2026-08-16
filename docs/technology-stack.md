# Technology Stack

## Runtime

| Layer | Technology | Role |
|---|---|---|
| Language | Python 3.12 | Application/runtime code |
| Validation | Pydantic 2 | Request, document, tool and segmented-response contracts |
| Generation + judge | Qwen3-4B Q4_K_M via llama.cpp | Conversational reasoning, tool calling and calibrated RAGAS judge |
| Tool adapter | LangChain OpenAI-compatible client | Infrastructure adapter only; the core uses framework-neutral ports |
| Embeddings | Qwen3-Embedding-0.6B | Dense retrieval |
| Reranker | Qwen3-Reranker-0.6B | Learned relevance grading |
| Accelerator target | OpenVINO / Intel Iris Xe | Embedding and reranking workloads on the target laptop |
| Object/vector services | Floci + S3/S3 Vectors-compatible APIs | Local AWS-compatible storage |
| UI | Chainlit primary, Gradio fallback | Thin clients over the same ConversationAgent |
| API | Lambda-compatible handlers | `/v1/query`, ingest and health transports |

There is no LangGraph orchestration package in the product. `ConversationAgent` owns the ReAct loop directly through `ToolCallingModel`, `Tool`, and `ToolResult` ports.

## Retrieval

The retrieval pipeline combines:

- dense vector candidates;
- BM25 lexical candidates;
- reciprocal-rank fusion;
- `domain_relevance` on raw candidates;
- optional document-governance resolution;
- learned reranking and explicit-target checks;
- `grounded_relevance` after policy resolution;
- evidence admission constrained by both relevance floors.

Calibration uses the three-class dataset `evaluation/datasets/retrieval-calibration-v2.json` (`grounded`, `in_domain_unanswerable`, `out_of_domain`).

## Ingestion

Current generic formats are Markdown and plain text. YAML front matter is optional. Rich Ops metadata remains supported as optional governance metadata rather than a required product schema.

## Evaluation and quality gates

Blocking software gates:

- strict mypy;
- unit tests with >=80% coverage;
- architecture fitness and zero unreachable pipeline code;
- full adversarial security suite;
- property tests;
- Floci integration tests;
- dependency/security audit;
- CDK test/build/synth;
- deterministic evaluation metrics such as citation validity and segment integrity.

Ruff format/lint is advisory during the architecture pivot. A strict manual `make lint` command remains available.

RAGAS uses the same Qwen3-4B model as runtime. It can gate release only after a real 10-case human calibration establishes sufficient agreement. The repository fails closed instead of fabricating that measurement.

## Model artifacts

`scripts/download_models.py` pins model files by SHA256. Generation uses the official `Qwen3-4B-Q4_K_M.gguf`; embedding/reranking models have independent pinned artifacts or OpenVINO preparation flows.

## Infrastructure

- AWS CDK v2 for deployment definition.
- Floci for local AWS-compatible services.
- Podman/Docker-compatible compose for local containers.
- GitHub Actions for hosted correctness/security gates.
- LangSmith is optional observability/evaluation tooling, not a runtime dependency for answer correctness.
