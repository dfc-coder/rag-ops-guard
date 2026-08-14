# Functional ReAct Beta Baseline

Status: **client-validated functional beta**

Frozen code commit:

```text
a16c1cc5b66fc7c83890760c5658e16699ca1264
```

Frozen baseline branch:

```text
baseline/beta-react-v0.1
```

Integration commit on `develop`:

```text
0b86fd9dca05b89c090308f9b6818e6632542391
```

## Runtime architecture

CPU:

- Python / LangGraph orchestration
- conversational ReAct loop with thread-level memory
- BM25 / RRF
- Floci
- Qwen 3.5 2B generation through llama.cpp

Intel Iris Xe / OpenVINO Model Server:

- Qwen3 Embedding 0.6B
- Qwen3 Reranker 0.6B

RAG is exposed to the agent as a tool rather than forcing every conversational turn through retrieval.

## Start the functional beta

From the repository root:

```bash
make beta-react
```

`beta-react` verifies the generation model, starts Floci and the CPU generation server, starts the OpenVINO iGPU backend, ensures the OpenVINO vector index is valid, and launches the Gradio ReAct UI.

The UI is served at:

```text
http://127.0.0.1:8000
```

## Stop the complete runtime

```bash
make local-down
```

This stops the OpenVINO runtime and the Podman Compose services used by the beta.

## Normal verification

Optional runtime inspection:

```bash
podman ps
podman stats
sudo intel_gpu_top
```

The expected main containers are:

```text
rag-ops-floci
rag-ops-llama-gen
rag-ops-ovms-rag
```

The legacy CPU embedding and reranking containers are not part of the primary ReAct beta runtime.

## Baseline policy

The commit `a16c1cc5b66fc7c83890760c5658e16699ca1264` is the recovery point for the client-validated beta.

Rules from this point forward:

1. Do not rewrite or repurpose `baseline/beta-react-v0.1`.
2. New feature work starts from this baseline or from `develop` after the baseline merge.
3. Hardware, model, retrieval, and agent-loop changes must be incremental and independently measurable.
4. Do not combine SYCL experiments, model swaps, retrieval-policy changes, and agent features in the same change.
5. Preserve `make beta-react` and `make local-down` as the stable operator contract.
6. If a future change breaks the beta, compare against the frozen commit before introducing additional fixes.

## Validated behavior

The functional beta has been exercised for:

- ordinary conversation without unnecessary RAG calls;
- RAG tool use for internal operational questions;
- multi-turn conversational continuity;
- grounded Calypso retry answers;
- contextual follow-up after the third retry;
- fail-closed behavior when the knowledge base does not support an answer;
- CPU generation with OpenVINO embedding/reranking on Intel Iris Xe;
- local vector-index rebuild using the matching OpenVINO embedding backend.
