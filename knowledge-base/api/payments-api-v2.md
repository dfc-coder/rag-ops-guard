---
id: payments-api-v2
logical_id: payments-api
title: Payments API v2
version: "2.0"
status: active
effective_date: 2026-06-01
system: payments
environment: production
document_type: api
authority: 95
supersedes:
  - payments-api-v1
---
# Payments API v2
Current payment submission endpoint `/payments/v2`.

## Idempotency
Every payment submission requires an `Idempotency-Key`. Reusing the same key returns the previously created transaction rather than creating a duplicate.

## Retry Contract
Transient Calypso timeouts may be retried by the automated client up to three times with exponential backoff. Manual resubmission requires a transaction-status check first.
