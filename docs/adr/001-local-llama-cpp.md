# ADR 001 — Local llama.cpp inference

## Decision

Beta 1 runs generation and embeddings locally with two independent llama.cpp servers on CPU.

## Why

The beta must be reproducible without cloud inference credentials while still exercising real models rather than stubs. Separate servers avoid swapping models on every query and give generation/embedding adapters stable endpoints.

## Models

- Qwen3-4B Q4_K_M for generation.
- Qwen3-Embedding-0.6B Q8_0, 1024 dimensions, for retrieval embeddings.

## Consequences

CPU latency is expected and reported rather than hidden. Concurrency is intentionally one slot per model in Beta 1.
