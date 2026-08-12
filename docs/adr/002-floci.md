# ADR 002 — Floci for local AWS compatibility

## Decision

Use Floci 1.5.34 for S3, S3 Vectors, Lambda, API Gateway, IAM and CloudWatch-compatible local integration.

## Why

The project should exercise AWS SDK contracts locally without requiring an AWS account. Floci's Bedrock stub is not used for AI quality tests.
