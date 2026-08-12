---
id: calypso-api-v3
logical_id: calypso-api
title: Calypso Integration API
version: "3.0"
status: active
effective_date: 2026-05-01
system: calypso
environment: production
document_type: api
authority: 90
supersedes: []
---
# Calypso Integration API
The Calypso adapter accepts payment instructions from Payments API.

## Transaction Status
`GET /transactions/{transactionId}` returns the processing state. A timeout from the submission endpoint must be followed by this status lookup before an operator attempts a manual replay.
