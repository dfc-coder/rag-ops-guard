# ADR 005 — Qwen3.5 0.8B generation model

## Decision

Beta 1 uses `unsloth/Qwen3.5-0.8B-GGUF`, file `Qwen3.5-0.8B-Q4_K_M.gguf`, for local grounded generation through llama.cpp.

The embedding model remains `Qwen3-Embedding-0.6B-Q8_0.gguf`; changing the generator does not require reindexing the knowledge base.

## Why

The 0.8B Q4_K_M generator is substantially smaller than the previous 4B generator, reducing download size, startup cost and CPU latency for the local beta. The change is accepted only if real-model E2E, structured output, safety behavior, prompt-cache reuse and accuracy evaluation remain measurable and acceptable.

## Integrity

Generation model SHA256:

`bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`

## Compatibility

The OpenAI-compatible service alias remains `qwen3-4b-rag` for this beta so existing application configuration and test fixtures do not change as part of the runtime-model experiment. The alias is an API identifier, not the physical GGUF model identity.

## Consequences

Accuracy baselines must be regenerated for Qwen3.5-0.8B. A smaller model may improve latency while reducing routing or grounded-generation accuracy, so the evaluation gate is the deciding signal rather than model size alone.
