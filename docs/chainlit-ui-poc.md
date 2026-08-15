# Chainlit UI proof of concept

This experiment replaces only the browser UI. The local architecture remains unchanged:

- Qwen 3.5 2B generation on CPU
- LangGraph/ReAct orchestration in Python
- OpenVINO embeddings and reranking on Intel Iris Xe
- Floci/vector storage and BM25/RRF retrieval
- `search_knowledge` remains the RAG tool

## Run

```bash
cd ~/Documents/projects/rag-ops-guard
git fetch origin
git switch experiment/chainlit-ui
git pull --ff-only
bash scripts/run_chainlit_beta.sh
```

Open:

```text
http://127.0.0.1:8001
```

The existing Gradio UI can stay on port 8000 for visual comparison, although only one active generation should be tested at a time because llama.cpp currently runs with one parallel slot.

## What to test

### Direct chat/code

```text
Write a short Perl function to encode text using a Caesar cipher.
```

Expected: immediate activity feedback, visible token streaming, no knowledge-base tool call.

### RAG

```text
¿Cuántos reintentos permite Calypso?
```

Expected: the activity step changes to knowledge-base lookup and the grounded answer is streamed afterwards.

### Recoverable long response

Ask for a response long enough to hit the generation limit. The partial answer should stay visible and the UI should offer **Reintentar**. The failed turn must not be committed to agent memory.

### Environment filter

Use Chat Settings to switch between **Cualquier ambiente**, **Producción**, and **Staging**. This value is passed into the existing `QueryContext` used by RAG.

## Stop

Stop Chainlit with `Ctrl+C` in its terminal. Stop the shared local services with:

```bash
make local-down
```

## Scope

This is an isolated UI experiment. It does not modify the frozen beta baseline, model selection, OpenVINO backend, retrieval thresholds, vector index, or generation threads.
