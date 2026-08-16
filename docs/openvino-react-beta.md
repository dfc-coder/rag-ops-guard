# Local Qwen3-4B + OpenVINO profile

This is the target local hardware profile after the generic ReAct pivot.

## Workload split

CPU:

- Python `ConversationAgent` orchestration and memory
- Floci / local AWS-compatible services
- BM25 + reciprocal-rank fusion
- Qwen3-4B Q4_K_M generation and tool calling through llama.cpp

Intel Iris Xe / OpenVINO:

- Qwen3-Embedding-0.6B
- Qwen3-Reranker-0.6B

The retrieval probe and the document tool share the same embedding/reranking infrastructure; the probe is telemetry and never decides whether the model calls a tool.

## Model preparation

```bash
make generation-model
make openvino-models
```

The generation download is SHA256-verified by `scripts/download_models.py`. OpenVINO assets are cached under the configured OpenVINO model directory.

## Start local runtime

```bash
make local-core-up
make openvino-up
make local-data
```

Then run the primary UI:

```bash
make beta-react
```

Chainlit, Gradio and REST all resolve the same `rag_ops_guard.app.conversation_agent()`.

## Correctness gates

Before treating the target-machine profile as release-ready, run:

```bash
make types
make test
make test-integration
make chainlit-gate
make eval
```

Ruff/formatting are advisory for the architecture pivot and are not a reason to block this hardware validation.

The evaluation path also requires:

1. double-relevance calibration on the target corpus/runtime;
2. RAGAS with the same Qwen3-4B runtime/judge;
3. exactly ten real human labels;
4. `make eval-judge-calibrate` to produce the judge policy.

If judge-human agreement is <=6/10, RAGAS remains informational and deterministic citation/segment/security gates stay authoritative.

## What hosted CI cannot prove

Hosted CI verifies software contracts, but it does not prove Intel Iris Xe/OpenVINO behavior or the real Qwen3-4B latency/accuracy profile on the target Fedora/Tiger Lake machine. Those checks remain explicit external release gates.
