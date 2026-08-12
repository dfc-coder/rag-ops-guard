---
id: duplicate-payment-postmortem-v1
logical_id: duplicate-payment-postmortem
title: Duplicate Payment Replay Postmortem
version: "1.0"
status: active
effective_date: 2026-03-20
system: payments
environment: production
document_type: postmortem
authority: 85
supersedes: []
---
# Root Cause
A full DLQ replay ignored transaction state and idempotency checks.

## Prevention
Never replay the entire payment DLQ as a single operational action. Verify each transaction and preserve the original idempotency key.
