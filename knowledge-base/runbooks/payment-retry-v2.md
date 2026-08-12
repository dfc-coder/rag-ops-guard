---
id: payment-retry-policy-v2
logical_id: payment-retry-policy
title: Payment Retry Policy
version: "2.0"
status: active
effective_date: 2026-06-01
system: payments
environment: production
document_type: runbook
authority: 100
supersedes:
  - payment-retry-policy-v1
---
# Retry Policy
Transient Calypso timeouts may be retried automatically a maximum of three times with exponential backoff.

## Manual Action
Before manually resubmitting a timed-out payment, query the existing transaction by transaction ID or idempotency key. Do not submit a new payment when the previous transaction status is unknown.

## Escalation
After the third automated retry fails, create an operational alert and escalate to Treasury Integrations.
