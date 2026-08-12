---
id: calypso-outage-postmortem-v1
logical_id: calypso-outage-postmortem
title: Calypso Connectivity Postmortem
version: "1.0"
status: active
effective_date: 2026-02-20
system: calypso
environment: production
document_type: postmortem
authority: 75
supersedes: []
---
# Finding
Timeouts created uncertainty because network failure occurred after request transmission.

## Recommendation
Use transaction-status verification before manual replay and rely on the current Payments retry policy for automated retries.
