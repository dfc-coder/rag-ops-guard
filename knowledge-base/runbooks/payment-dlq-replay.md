---
id: payment-dlq-replay-v1
logical_id: payment-dlq-replay
title: Payment DLQ Replay Runbook
version: "1.0"
status: active
effective_date: 2026-04-10
system: payments
environment: production
document_type: runbook
authority: 100
supersedes: []
---
# Payment DLQ Replay
Bulk replay of the entire payment dead-letter queue is prohibited.

## Safe Procedure
Review each message's idempotency key and transaction state. Replay only messages confirmed not to have produced a successful downstream transaction. For more than 20 messages, obtain Payments Platform approval before replay.
