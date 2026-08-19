# ADR 0008 — Phase 4 tenant IAM boundary

Status: Accepted

## Context

Phase 4 introduces authenticated tenant isolation while the query and ingest Lambdas remain shared execution functions. The application derives `tenant_id` from a verified API key and scopes object keys, config partitions, warm dependency containers and vector indexes with that identity.

DynamoDB supports `dynamodb:LeadingKeys`, but a shared Lambda execution role cannot express a different static leading-key value for every request. Dynamic IAM enforcement at the Lambda request boundary therefore requires an STS role-assumption design with trusted session tags.

## Decision

1. CDK creates one tenant configuration administrative role per declared tenant.
2. Each administrative role's item-level config permissions are conditioned with `ForAllValues:StringEquals` on `dynamodb:LeadingKeys = TENANT#{tenant_id}`.
3. `dynamodb:DescribeTable` is granted separately because it is not an item-key operation.
4. Runtime query/ingest isolation is enforced now by authenticated `RequestContext`, tenant config partitions, `KeyLayout`, tenant-specific S3 Vectors indexes and the bounded tenant/config dependency container.
5. Per-request `sts:AssumeRole` with trusted tenant session tags is deferred until the first external tenant. It is not simulated in Floci.
6. Before the first external tenant is admitted, a real-AWS isolation gate must prove that a tenant administrative role cannot read another tenant's config partition and that the allowed tenant partition succeeds.

## Consequences

The local/Floci gate validates the application and storage isolation model but is not accepted as evidence for AWS IAM condition semantics. CDK remains the source of truth for both targets. No external tenant may be onboarded until the real-AWS LeadingKeys gate passes.
