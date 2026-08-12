---
id: calypso-timeout-runbook-v2
logical_id: calypso-timeout-runbook
title: Calypso Timeout Runbook
version: "2.0"
status: active
effective_date: 2026-06-15
system: payments
environment: production
document_type: runbook
authority: 100
supersedes: []
---
# Calypso Timeout
A submission timeout is an unknown outcome, not a confirmed failure.

## Procedure
1. Record the transaction ID and idempotency key.
2. Check the Calypso transaction-status endpoint.
3. If no successful transaction exists, allow the automated retry policy to continue.
4. Do not exceed three automated retries.
5. If the third retry fails, alert Treasury Integrations.
