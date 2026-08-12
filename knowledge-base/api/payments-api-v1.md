---
id: payments-api-v1
logical_id: payments-api
title: Payments API v1
version: "1.0"
status: deprecated
effective_date: 2024-09-01
system: payments
environment: production
document_type: api
authority: 70
supersedes: []
---
# Payments API v1
Legacy endpoint `/payments/v1`.

## Retry
Clients historically retried transient downstream failures up to five times. This API version is deprecated and must not define current production behavior.
