# SPEC-P4-MULTITENANCY — Phase 4 Real Tenant Isolation

Status: normative for Phase 4 implementation.

## Objective

Move RAG Ops Guard from a single shared corpus/runtime to real tenant isolation while preserving the Phase 3 encrypted-config contract and the Python 3.13 + canonical-CDK infrastructure model.

The tenant boundary is an authenticated runtime boundary, not request metadata. `tenant_id` MUST be derived from a verified API key and MUST NOT be accepted from a query or ingest request body.

## Required implementation order

### 4.0 — Canonical key layout

Introduce one `KeyLayout` abstraction as the only place that constructs tenant-scoped S3 keys and prefixes.

Canonical tenant prefix:

```text
t/{tenant_id}/
```

At minimum it owns raw/document, chunk and manifest key construction. Application code MUST NOT hand-build tenant storage paths.

### 4.1 — Immutable request identity

Introduce an immutable `RequestContext` carrying the authenticated principal and `tenant_id`.

Authentication contract:

1. read API key from the transport/header boundary;
2. resolve the candidate tenant credential record;
3. verify the presented token with Argon2id;
4. derive `tenant_id` from the verified credential record;
5. pass `RequestContext` explicitly into application dependencies and handlers.

The body is never an authority for `tenant_id`.

Tenant-token hashing parameters are fixed for this phase:

- Argon2id;
- memory: 64 MiB;
- iterations: 3;
- parallelism: 4.

Cleartext tokens MUST NOT be logged or persisted.

### 4.2 — Tenant/config-scoped dependency container

Replace global singleton-style dependency caches with a bounded application `Container` keyed by:

```text
(tenant_id, config_hash)
```

The container cache MUST be LRU-bounded to 32 generations. Warm dependencies for tenant A MUST never be returned to tenant B, even when both use the same `config_hash`.

### 4.3 — Tenant-scoped object storage

Canonical chunk layout:

```text
t/{tenant_id}/chunks/{logical_id}/{version}/chunk-{chunk_index:03d}.json
```

Raw/document and manifest data MUST also live below `t/{tenant_id}/...`.

Ingestion MUST reject unprefixed or cross-tenant S3 keys. A migration helper MUST exist for the previous single-tenant layout; implicit fallback to unscoped keys is forbidden in normal runtime reads.

### 4.4 — One vector index per tenant

Tenant isolation in S3 Vectors MUST be structural: one index per tenant.

Do not rely on vector metadata filters as the security boundary.

The canonical tenant index name MUST be derived deterministically from the configured base index and tenant id. The local Floci compatibility adapter may materialize the index because Floci 1.6.0 does not materialize `AWS::S3Vectors::*` through CloudFormation, but CDK remains the source of truth for the declared vector topology.

### 4.5 — AWS administrative-plane enforcement

CDK MUST model tenant-scoped IAM/DynamoDB access using `dynamodb:LeadingKeys` for the tenant administrative/data plane where applicable.

`sts:AssumeRole` session-tag based tenant roles are explicitly deferred until the first external tenant. That deferral MUST be documented and MUST NOT weaken the application/storage isolation gates in this phase.

Floci validation is not sufficient evidence for AWS `dynamodb:LeadingKeys`; a real-AWS isolation gate is required before onboarding the first external tenant.

### 4.6 — Blocking tenant-isolation gate

CI MUST contain a blocking `ci/tenant-isolation` gate.

The deterministic test suite MUST seed at least two tenants and prove zero cross-tenant access for:

- chunks/object keys;
- effective/config records;
- warm dependency containers;
- encrypted-secret AAD scope;
- vector index selection.

It MUST also prove that ingestion rejects unprefixed and wrong-tenant S3 keys.

No Phase 4 isolation test may use `skip` or `xfail`.

## Encrypted secret/AAD contract

Phase 3 encryption remains mandatory. Tenant-scoped encrypted values MUST bind ciphertext to tenant scope with AAD equivalent to:

```text
scope || key_name || dek_version
```

Ciphertext created for one scope MUST fail authentication/decryption under another scope's AAD.

## BDD acceptance scenarios

### Scenario A — tenant is derived from authentication

Given API key A maps to tenant A
When a query body contains no tenant metadata
Then the runtime context tenant is A
And the request succeeds within tenant A's scope.

When the body attempts to provide `tenant_id=B`
Then validation rejects the body or ignores it as non-authoritative
And tenant A remains the runtime scope.

### Scenario B — object isolation

Given tenant A and tenant B contain different chunks
When tenant A queries or enumerates storage
Then no key below `t/B/` is returned or read.

### Scenario C — warm-container isolation

Given tenant A and tenant B share the same effective `config_hash`
When both build warm dependencies
Then their container entries are distinct
And neither tenant receives the other's object store/vector store/agent graph.

### Scenario D — vector isolation

Given vectors exist for A and B
When tenant A performs retrieval
Then only A's vector index is queried
And vector metadata filtering is not used as the tenant boundary.

### Scenario E — AAD isolation

Given a secret encrypted with tenant A scope
When decryption is attempted with tenant B scope
Then authentication/decryption fails.

### Scenario F — ingestion path isolation

Given tenant A is authenticated
When ingest receives an unprefixed key or a key below `t/B/`
Then the request is rejected before document parsing or persistence.

## Delivery process

- This spec is merged before Phase 4 implementation.
- Each implementation unit 4.0–4.6 is developed from the current `develop` head.
- The first implementation commit for a unit introduces failing/red contract tests and cites `SPEC-P4-MULTITENANCY`.
- Implementation then makes those tests green.
- No stacked implementation PRs.
- No `skip`/`xfail` for isolation acceptance tests.

## Local and AWS validation

Local physical validation continues to use the canonical path:

```text
CDK
 ├── Floci local
 └── AWS real
```

The same Python 3.13 Lambda package is used in both targets.

Local Floci proves application, Lambda, S3, S3 Vectors compatibility, config, and request isolation behavior. Real AWS validation is additionally required for IAM condition semantics such as `dynamodb:LeadingKeys` before any external tenant is admitted.
