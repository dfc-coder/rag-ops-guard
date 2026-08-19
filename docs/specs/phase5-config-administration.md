# SPEC-P5-CONFIG-ADMIN — Phase 5 Configuration Administration

Status: normative for Phase 5 implementation.

## Objective

Add an authenticated administrative surface for tenant configuration and secret-change governance without changing the runtime source architecture established in Phases 2–4.

Phase 5 is a **management-plane** phase. It MUST NOT introduce new runtime environment-variable dependencies, new plaintext secret persistence, or a second source of truth. Phase 6 owns the later migration to an environmentless AppConfig/Secrets Manager/workload-identity control plane.

## 5.0 — Authenticated administrative boundary

Administrative CLI/API operations MUST execute under an authenticated operator principal and an explicitly authorized tenant scope.

- Tenant scope MUST NOT be accepted as authority from an untrusted request body.
- Every mutation records actor, tenant, action and reason.
- Administrative authorization is distinct from end-user API-key authentication.

## 5.1 — Publish tenant configuration revisions

Configuration publication MUST continue to use the tenant-scoped append-only DynamoDB model.

- `change_reason` is mandatory.
- Published values are schema validated.
- Fields classified as `secret` are forbidden in ordinary configuration revisions.
- `HEAD` advances only to a newly created immutable revision.
- Publication records before/after content hashes and the authenticated actor.

## 5.2 — Rollback without history mutation

The earlier Phase 5 draft described rollback as moving `HEAD` to an older revision. That conflicts with the current monotonic-HEAD invariant and is replaced by this contract.

Rollback MUST:

1. read the selected historical revision;
2. create a **new** revision containing those values;
3. record `rollback_from_revision`, actor and reason in audit metadata;
4. advance `HEAD` to the new revision.

Rollback MUST NOT delete, overwrite or make `HEAD` move backwards.

Example:

```text
REV#17 current
REV#12 desired historical state

rollback(12)
    -> create REV#18 with REV#12 values
    -> HEAD = 18
```

## 5.3 — Separate secret-change workflow with dual control

Secrets MUST NOT pass through the normal configuration publisher.

Expose a separate secret mutation contract for set/rotate/delete operations:

- plaintext is accepted only at the mutation boundary and is never returned by read APIs;
- plaintext MUST NOT be persisted in audit records, configuration revisions, logs or artifacts;
- a sensitive secret mutation requires approvals from two distinct authenticated principals;
- one principal MUST NOT satisfy both approvals;
- audit records contain only secret reference/name, scope, action and approval metadata;
- the administrative layer depends on a `SecretBackend`/equivalent port rather than a concrete storage implementation.

During Phase 5 the existing Phase 3 envelope-encryption service may back that port. Phase 6 will replace recoverable runtime-secret persistence with AWS Secrets Manager without changing the administrative contract.

## 5.4 — Immutable administrative audit trail

Every mutation MUST append an audit event containing, where applicable:

- tenant/scope;
- actor and approving principals;
- action;
- mandatory change reason;
- source revision and resulting revision;
- content hash before/after;
- secret reference only, never secret value;
- timestamp/correlation id.

Audit history MUST NOT be editable through the administrative API.

## 5.5 — Operational visibility

Phase 5 exposes operational views for the configuration subsystem. At minimum surface:

- active tenant revision and `config_hash`;
- configuration source and age;
- cache hit/miss and resolve latency;
- stale/fail-closed state;
- configuration DB unavailability;
- secret retrieval/decrypt failures;
- tenant-isolation violations;
- administrative publish/rollback/secret-mutation audit events.

The administrative surface MUST NOT expose secret plaintext.

## 5.6 — Phase 6 compatibility guard

Phase 5 MUST NOT make the future environmentless migration harder.

Blocking architecture tests MUST reject Phase 5 changes that:

- add a new `.env`/environment-variable dependency under `src/rag_ops_guard`;
- add a new application-specific Lambda environment variable;
- add a secret to ordinary config revisions;
- couple the administrative API directly to the Phase 3 DynamoDB secret-store implementation instead of the secret backend abstraction.

## Acceptance scenarios

### Scenario A — publish

Given an authenticated administrator authorized for tenant A
When a valid non-secret configuration change is published with a reason
Then a new tenant-A revision is created
And `HEAD` advances to it
And the audit event contains actor, reason and hashes.

### Scenario B — rollback

Given tenant A has revisions 12 through 17
When an administrator rolls back to the values of revision 12
Then revision 18 is created with those values
And `HEAD` becomes 18
And revisions 12–17 remain immutable.

### Scenario C — secret dual control

Given a secret mutation request created by operator X
When only X approves it
Then the mutation remains blocked.

When a distinct authorized operator Y approves it
Then the secret backend mutation may execute
And neither API nor audit output contains plaintext.

### Scenario D — cross-tenant administration

Given an administrator authorized only for tenant A
When it attempts to mutate tenant B configuration or secrets
Then the operation is denied before persistence.

## Definition of Done

Phase 5 is complete only when publication, monotonic rollback, dual-control secret mutation, immutable audit and operational views are implemented and blocking tests prove tenant/admin boundaries and absence of new environment/bootstrap debt.

STOP before Phase 6 implementation.