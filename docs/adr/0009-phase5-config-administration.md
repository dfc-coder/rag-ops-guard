# ADR 0009 — Phase 5 configuration administration boundary

Status: Accepted

## Context

Phases 1–4 established append-only tenant configuration, encrypted recoverable secrets, authenticated tenant identity and tenant-scoped runtime isolation. Phase 5 adds an operator-facing management plane without changing the runtime configuration source or starting the Phase 6 environmentless migration.

Three constraints shape the design:

1. configuration `HEAD` is monotonic, so rollback cannot point it backwards;
2. ordinary config publication rejects secret fields, so secret changes require a separate workflow;
3. Phase 6 will replace the Phase 3 recoverable-secret backend, so the management plane must not depend on its concrete implementation.

## Decision

1. Configuration administration is tenant-authorized before persistence.
2. Publish creates a new immutable revision and requires `change_reason`.
3. Rollback reads an old revision, republishes its values as a new revision and advances `HEAD`; historical revisions are never mutated or deleted.
4. Secret set/rotate/delete uses a separate two-person workflow. Requester and approver must be distinct authenticated principals.
5. Secret approval state stores only metadata and a SHA-256 digest used to prove that the approver supplied the same secret payload. Plaintext is never persisted in the approval or audit records.
6. Secret execution depends on a narrow `SecretBackend` port. During Phase 5 the Phase 3 envelope-encryption implementation is an adapter behind that port.
7. Administrative audit is append-only and contains actor, approver when applicable, reason, tenant, revision/hash metadata and secret references only.
8. CDK declares the Phase 5 KMS administrative boundary and AWS operational dashboard. Phase 5 does not add new application environment variables or environment reads.
9. Phase 6 remains a separate architectural change and is not partially implemented here.

## Consequences

- Rollback preserves the append-only history and monotonic `HEAD` invariant.
- A compromised single administrator cannot execute a recoverable-secret mutation alone.
- The management plane can switch from the Phase 3 encrypted DynamoDB secret backend to Secrets Manager in Phase 6 without changing its public workflow.
- Existing Phase 4 runtime/bootstrap debt remains visible but frozen; its removal belongs exclusively to Phase 6.
- Local Floci validates application/storage behavior. AWS IAM semantics remain subject to the real-AWS isolation gate already required by ADR 0008.
