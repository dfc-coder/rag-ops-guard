---
id: orion-payment-retry-v1
logical_id: orion-payment-retry
title: Orion Payment Retry Runbook
version: "1.0"
status: active
effective_date: 2026-08-13
system: payments
environment: production
document_type: runbook
authority: 90
supersedes: []
---

# Orion Payment Retry Runbook

When an Orion payment request fails with a transient timeout, the system may perform a maximum of **2 automatic retries**.

The delay between retries is **30 seconds**. After the second automatic retry fails, the payment must be moved to the **manual-review queue**.

If more than **5 Orion payments** enter manual review within a 10-minute window, the operator must open a **P1 incident** and notify the Payments On-Call team.

A payment with status `SETTLED` must never be retried.
