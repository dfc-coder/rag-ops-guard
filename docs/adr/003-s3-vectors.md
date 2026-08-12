# ADR 003 — S3 Vectors

## Decision

Use S3 Vectors as the only vector store for Beta 1.

## Configuration

1024-dimensional float32 vectors, cosine distance, Top K 8. Full chunk text remains in S3 and vector metadata stores its S3 pointer.

## Why

This keeps the architecture AWS-native and serverless-compatible while avoiding a second database solely for the portfolio beta.
