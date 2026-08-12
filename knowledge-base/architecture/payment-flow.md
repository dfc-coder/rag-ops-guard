---
id: payment-flow-v2
logical_id: payment-flow
title: Payment Processing Flow
version: "2.0"
status: active
effective_date: 2026-05-20
system: payments
environment: production
document_type: architecture
authority: 85
supersedes: []
---
# Payment Processing Flow
Salesforce submits payment instructions to MuleSoft, which calls Payments API v2. Payments API persists the idempotency key before calling Calypso.

## Timeout Semantics
A network timeout between Payments API and Calypso does not prove that Calypso rejected the transaction. The transaction status must be checked before manual resubmission.
