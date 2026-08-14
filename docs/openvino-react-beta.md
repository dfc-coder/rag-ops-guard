# OpenVINO ReAct beta

This beta keeps orchestration and generation on CPU while moving the two repeated RAG inference workloads to the Intel iGPU.

```text
CPU
├── Python / LangGraph
├── ReAct orchestration
├── BM25 / RRF
├── Floci / S3 vectors
└── Qwen 3.5 2B generation (llama.cpp)

Intel Iris Xe / OpenVINO Model Server
├── Qwen3 Embedding 0.6B
└── Qwen3 Reranker 0.6B
```

The ReAct agent exposes RAG as `search_knowledge` and documentation inventory as `list_knowledge`. Conversation state is stored by `thread_id` using the LangGraph checkpointer. Additional tools can be added to the tool list without changing the RAG implementation.

## First run

From the repository root:

```bash
git switch fix/self-hosted-ci-isolation
git pull --ff-only

ls -l /dev/dri/render*
stat -c '%g %n' /dev/dri/render*

make local-down
make openvino-models
make openvino-up
make openvino-smoke
make react-smoke
make beta-react
```

`openvino-models` is the expensive first-run step. It downloads/prepares the embedding and reranking models into `~/.cache/rag-ops-guard/openvino-models`. Later runs reuse that cache.

The browser UI is served at `http://127.0.0.1:8000`.

## What each command validates

`make openvino-up` starts OpenVINO Model Server with `/dev/dri` exposed and waits until both `/v3/embeddings` and `/v3/rerank` respond.

`make openvino-smoke` prints measured embedding/reranking latency against the Intel GPU backend.

`make react-smoke` checks three product behaviors without Gradio:

1. a greeting must not call RAG;
2. a Calypso retry question must call RAG and answer the documented retry count;
3. a follow-up must preserve conversational context and answer the Treasury escalation.

`make beta-react` launches the interactive conversational agent.

## Runtime endpoints

- Floci: `127.0.0.1:4566`
- Qwen 3.5 2B generation: `127.0.0.1:8080`
- OpenVINO embeddings/reranker: `127.0.0.1:8083/v3`
- Gradio: `127.0.0.1:8000`

The legacy llama.cpp embedding and reranking containers are intentionally stopped in this beta.

## Separate vector index

The OpenVINO path uses `ops-knowledge-openvino-v1` instead of the legacy vector index. This forces the corpus to be embedded with the same backend used at query time and prevents vectors from different embedding runtimes from being mixed accidentally.

## Useful diagnostics

```bash
make openvino-status
podman ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
podman logs --tail 100 rag-ops-ovms-rag
podman logs --tail 100 rag-ops-llama-gen
```

GPU activity can be inspected with Intel GPU tooling when installed, for example `intel_gpu_top`.

## Stop

```bash
make local-down
```
