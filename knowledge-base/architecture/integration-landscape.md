---
id: integration-landscape-v1
logical_id: integration-landscape
title: AcmePay Integration Landscape
version: "1.0"
status: active
effective_date: 2026-01-10
system: payments
environment: all
document_type: architecture
authority: 80
supersedes: []
---
# Integration Landscape
AcmePay uses MuleSoft as the integration layer between Salesforce, the Payments API, SAP, Calypso, Stripe, and SendGrid.

## Ownership
The Integration Operations team owns MuleSoft runtime incidents and routing failures. Payments Platform owns payment authorization and idempotency behavior. Treasury Integrations owns Calypso connectivity.
