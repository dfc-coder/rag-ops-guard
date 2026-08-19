# SPEC-P6-ENVIRONMENTLESS-CONTROL-PLANE — Phase 6 Workload-Identity Control Plane

Status: normative after Phase 5.

## Objective

Remove the application runtime's dependency on `.env`, application-owned environment variables and environment-derived `Settings`.

The runtime MUST discover configuration and secrets through AWS service APIs under workload identity. The Python application MUST execute through the same code path in local Floci and real AWS.

```text
                         canonical CDK
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
         AppConfig      Secrets Manager     DynamoDB
      platform config   recoverable secrets tenants/config
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Lambda Python 3.13
                              |
                       workload identity
                              |
                        authenticated tenant
                              |
                    STS tenant-scoped session
                              |
                  S3 / DynamoDB / S3 Vectors
```

## Environmentless invariant

For this specification, **environmentless** means:

1. `src/rag_ops_guard` contains no `os.environ`, `os.getenv`, `BaseSettings`, `.env` loader or equivalent application configuration read.
2. CDK does not define application-specific Lambda `Environment.Variables`.
3. `.env`/`.env.example` are not part of the supported application runtime or operator configuration contract.
4. AWS credentials, region and endpoint routing are supplied by the workload/platform and consumed only through the standard AWS SDK provider chain; application code MUST NOT read or branch on those environment variables directly.
5. Floci may inject standard AWS SDK variables such as `AWS_ENDPOINT_URL` into its Lambda container. This is platform behavior, not application configuration, and the Python application MUST remain unaware of it.
6. The application MUST NOT branch on `local`, `aws`, Floci hostnames, account-specific resource names or environment-specific URLs.

Stable protocol identifiers such as the AppConfig application/environment/profile names are code-level contract constants, not deploy-time environment configuration.

## 6.0 — AppConfig control plane created by canonical CDK

CDK remains the single infrastructure source of truth and MUST create the AppConfig control plane for both local Floci and real AWS.

Canonical identifiers:

```text
application:             rag-ops-guard
environment:             runtime
configuration profile:   control-plane
```

The hosted configuration payload MUST be schema-versioned and contain non-secret platform discovery data such as:

- config table identity;
- tenant credential table identity;
- document bucket identity;
- vector bucket/base-index identity;
- model/embedding/reranker service descriptors;
- references to recoverable secrets, never secret values;
- cache/fail-closed policy needed before tenant config is resolved.

The application starts an AppConfigData session using the canonical names and retrieves the active payload. No resource identifier in this payload may also be supplied through an application environment variable.

### 6.0 entry gate

Before implementation proceeds, integration tests MUST prove that the pinned Floci version supports the exact AppConfig/AppConfigData operations needed by the runtime and that the same client contract works against AWS SDK interfaces without application-side endpoint overrides.

## 6.1 — `Settings` becomes a pure validated model

`Settings` MUST stop inheriting from `pydantic_settings.BaseSettings`.

Use a pure validation model (`pydantic.BaseModel` or equivalent) whose values are explicitly supplied by the control-plane/config resolvers.

Required consequences:

- no implicit environment lookup;
- no `env_file`;
- no `SecretStr` value sourced from process environment;
- construction is deterministic from explicit inputs;
- existing cross-field invariants remain enforced;
- the effective tenant config hash continues to identify the immutable behavioral generation.

`pydantic-settings` MUST leave the application runtime dependency set if nothing outside tooling requires it.

## 6.2 — Zero environment reads under `src/rag_ops_guard`

All direct environment access MUST be removed from application source:

```text
os.environ
os.getenv
BaseSettings
SettingsConfigDict(env_file=...)
```

Environment-aware behavior belongs to platform tooling outside `src/rag_ops_guard`, and MUST NOT alter application semantics.

The runtime obtains:

- platform discovery from AppConfigData;
- tenant identity from authenticated credentials;
- tenant behavior from the tenant-scoped config store;
- recoverable secrets from Secrets Manager;
- AWS authorization from workload/STS credentials.

## 6.3 — Lambda CDK without custom environment variables

The query and ingest Lambda resources MUST contain no application-specific environment variables.

The current `commonEnvironment`-style injection is removed, including values such as resource names, model URLs, config source, config hash, table names and application environment labels.

AWS/Lambda- or Floci-managed standard runtime variables are outside this restriction; CDK/application code MUST NOT create them as an application configuration mechanism.

CDK tests MUST assert that synthesized query/ingest functions contain no application `Environment.Variables` block.

## 6.4 — Secrets Manager replaces recoverable runtime secrets

All secrets that the application must later recover in plaintext MUST be stored in AWS Secrets Manager and protected by IAM/KMS controls.

Examples include external API credentials, observability tokens and other third-party credentials.

Requirements:

- secret plaintext MUST NOT appear in AppConfig, DynamoDB config revisions, CloudFormation/CDK output, `.env`, logs or artifacts;
- AppConfig/tenant config contains only a logical secret reference when one is needed;
- secret retrieval is authorized at runtime and cached only in memory with a bounded lifetime;
- secret rotation MUST NOT require changing application code or Lambda environment variables;
- tenant/scope isolation MUST be enforceable through IAM resource policy, role policy and/or tags;
- secret values MUST never be returned by the Phase 5 administrative read surface.

### Migration from Phase 3

The Phase 3 envelope-encryption implementation remains valid historical/security work, but it MUST cease being the authoritative runtime store for recoverable secrets after this cutover.

Provide an explicit migration command that:

1. reads/decrypts an existing Phase 3 secret at the operator boundary;
2. writes it directly to Secrets Manager;
3. records only migration metadata/reference in audit;
4. never persists plaintext to disk or configuration history;
5. verifies the new secret before marking the old record migrated.

Do not silently fall back to the old secret store after migration.

## 6.5 — Tenant API keys remain Argon2id verifier records in DynamoDB

Tenant API keys are intentionally **not** moved to Secrets Manager because the runtime does not need to recover their original secret value.

The current verifier contract remains:

```text
presented key = key_id.secret
          |
          v
rag-ops-tenants lookup by key_id
          |
          v
Argon2id.verify(secret, stored_hash)
          |
          v
RequestContext(tenant_id from verified record)
```

Only `key_id`, `tenant_id`, Argon2id hash and lifecycle metadata are persisted. Cleartext API-key secrets MUST NOT be recoverable.

## 6.6 — STS tenant-scoped sessions and ABAC

This unit closes the STS deferral recorded in ADR 0008.

The shared Lambda execution role is a minimal bootstrap identity. After API-key authentication derives a trusted `tenant_id`, the runtime MUST assume the tenant data role using AWS STS and trusted session tags.

Canonical session attribute:

```text
tenant_id=<authenticated tenant id>
```

Resource authorization MUST use that trusted principal/session attribute where AWS supports it, including:

- DynamoDB tenant partition access, including `dynamodb:LeadingKeys` where applicable;
- S3 `t/{tenant_id}/...` prefixes;
- tenant-scoped Secrets Manager access;
- other tenant-scoped resources supported by resource-level IAM/ABAC.

The caller MUST NOT be able to choose an arbitrary session `tenant_id`. The session tag is created only from the authenticated `RequestContext`.

Trust policies MUST explicitly authorize the required `sts:AssumeRole`/`sts:TagSession` flow and prevent untrusted tag substitution.

A real-AWS blocking security gate MUST prove both allowed and denied cross-tenant operations before an external tenant is admitted. Local Floci behavior is useful integration evidence but is not sufficient proof of AWS IAM condition semantics.

## 6.7 — Identical Python application code in Floci and AWS

The application path MUST be identical:

```python
boto3.client("appconfigdata")
boto3.client("dynamodb")
boto3.client("secretsmanager")
boto3.client("sts")
```

Application code MUST NOT provide `endpoint_url` based on environment and MUST NOT contain `if local`, `if aws`, hostname rewrites or Floci-specific branches.

In real AWS, the SDK uses the Lambda workload identity and normal AWS endpoints.

In local Lambda execution, Floci is responsible for AWS SDK routing/credentials. Local launch/deployment tooling may configure Floci itself, but that configuration MUST remain outside `src/rag_ops_guard` and outside the application's Lambda custom environment.

The same packaged Python 3.13 Lambda asset MUST be used by both targets.

## 6.8 — Blocking CI anti-regression gate

Add blocking `ci/environmentless-control-plane` checks and make them dependencies of `ci/gate`.

The gate MUST fail if any of the following regressions appear:

1. `os.environ`, `os.getenv`, `BaseSettings` or `.env` loading under `src/rag_ops_guard`;
2. application-specific Lambda `Environment.Variables` in synthesized CDK;
3. recoverable plaintext secrets in AppConfig payloads, config revisions, CloudFormation, repository files or test artifacts;
4. runtime dependency on `.env`/`.env.example`;
5. application-side AWS `endpoint_url` overrides or `local/aws` behavioral branches;
6. tenant API keys persisted as recoverable plaintext;
7. tenant-scoped clients created without the authenticated tenant session after the STS cutover;
8. skipped/xfail security acceptance tests.

CI MUST additionally prove:

- AppConfig payload schema validation;
- AppConfigData resolution through Floci;
- Secrets Manager create/retrieve/rotate integration through Floci;
- Phase 4 tenant-isolation tests remain green;
- Phase 5 administration tests remain green;
- CDK synth uses the same infrastructure definition for local/AWS;
- the real-AWS STS/ABAC isolation gate exists and is required before external-tenant release.

## Required implementation order

```text
6.0 AppConfig control plane
 ↓
6.1 explicit Settings model
 ↓
6.2 remove application env reads
 ↓
6.3 remove Lambda custom env
 ↓
6.4 Secrets Manager cutover
 ↓
6.5 preserve/hash-only tenant API keys
 ↓
6.6 STS + ABAC
 ↓
6.7 prove identical Floci/AWS code path
 ↓
6.8 make all invariants blocking in CI
```

Each unit starts from current `develop`, begins with a red contract test, turns green, and does not stack implementation PRs unless explicitly approved.

## Acceptance scenarios

### Scenario A — environmentless cold start

Given a freshly deployed Lambda with no custom application environment variables
When it cold-starts
Then it uses the AWS SDK workload identity
And retrieves `rag-ops-guard/runtime/control-plane` from AppConfigData
And constructs validated runtime configuration without reading process environment.

### Scenario B — no `.env` dependency

Given the repository contains no runtime `.env`
When the physical local stack is provisioned and the Lambda is invoked through Floci
Then query and ingest resolve their complete application configuration through the AWS-compatible control plane.

### Scenario C — recoverable secret

Given tenant A configuration references secret X
When tenant A runtime resolves X
Then Secrets Manager returns it under authorized credentials
And neither AppConfig, DynamoDB config nor logs contain the plaintext.

### Scenario D — API-key verifier

Given tenant A API key is registered
Then DynamoDB stores an Argon2id verifier only
And authentication derives tenant A
And the original secret cannot be retrieved from the store.

### Scenario E — STS/ABAC isolation

Given an authenticated tenant A request
When the runtime assumes its tenant-scoped session
Then access to tenant A resources succeeds
And direct access to tenant B DynamoDB/S3/secret resources is denied by AWS authorization even if application code attempts it.

### Scenario F — parity

Given the same Lambda ZIP
When deployed through canonical CDK to Floci and AWS
Then no Python application source change or environment-specific branch is required.

## Definition of Done

Phase 6 is complete only when the application is environmentless by the invariant above, AppConfig and Secrets Manager are authoritative for platform discovery/recoverable secrets, API keys remain hash-only, tenant data-plane access uses authenticated STS sessions, the same Python code runs against Floci/AWS, and all anti-regression/security gates are blocking.

No external tenant may be considered AWS-security-ready until the real-AWS STS/ABAC/LeadingKeys isolation gate passes.
