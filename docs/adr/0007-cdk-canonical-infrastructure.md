# ADR 0007 — CDK is the canonical infrastructure definition

Status: Accepted

Date: 2026-08-19

## Context

The repository historically had two infrastructure definitions:

- `infra/cdk/` described the AWS target, but used a placeholder Lambda asset;
- `scripts/local/provision.py` independently created S3, DynamoDB, IAM, Lambda and API Gateway resources against Floci.

That duplication allowed local and AWS topology to drift. The Python 3.12 → 3.13 migration exposed the same class of problem because runtime declarations existed independently in the project, local Lambda packaging and CDK.

REQ-001 Phase 3 (encrypted configuration secrets) is already integrated on `develop`. The next planned product phase is Phase 4; this ADR is an infrastructure/runtime convergence change and does not introduce Phase-4 tenant semantics.

## Decision

AWS CDK is the single infrastructure model for both targets:

```text
                  infra/cdk
                     CDK
                  /       \
                 v         v
          local target    aws target
              Floci        AWS
```

The same stack defines S3, S3 Vectors, DynamoDB, IAM, Lambda and API Gateway. Target-specific endpoint values are configuration, not separate infrastructure implementations.

The real Lambda application package produced by `scripts/package_lambda.sh` is deployed by CDK to both targets. The old `infra/cdk/lambda-placeholder` asset is removed.

`.python-version` is the operational Python runtime source of truth. `uv`, Lambda packaging, CDK runtime selection, mypy/Ruff, `doctor` and CI must derive from or assert alignment with that file.

## Floci compatibility boundary

Floci 1.6.0 exposes the S3 Vectors API but does not currently materialize `AWS::S3Vectors::*` CloudFormation resources. Therefore `scripts/local/provision.py` remains as a deliberately narrow compatibility adapter:

1. read names and dimensions from the deployed CDK stack outputs;
2. materialize only the missing S3 Vectors bucket/index through Floci's S3 Vectors API;
3. inject local-only observability secrets without putting them in a CloudFormation template;
4. write the Floci API data-plane URL from the CDK stack output.

Runtime connectivity is validated separately by `scripts/local/connectivity.py` after the Phase-3 configuration revision exists.

It must not create S3 buckets, DynamoDB tables, IAM roles, Lambda functions or API Gateway resources.

The S3 Vectors bridge is temporary and should be deleted when the pinned Floci version can materialize those CloudFormation resource types.

## Consequences

Positive:

- local and AWS topology share one reviewable definition;
- runtime upgrades cannot silently diverge between local packaging and AWS CDK;
- the real Lambda handlers are exercised locally through the same CDK topology intended for AWS;
- infrastructure drift becomes a CI failure instead of a runtime surprise.

Trade-offs:

- local provisioning now requires Node/CDK plus `cdklocal` in addition to Python tooling;
- Floci still needs one explicit S3 Vectors compatibility bridge;
- external AWS inference endpoints remain fail-closed until explicit HTTPS endpoints are supplied.

## Validation

This decision is accepted only while these gates remain green:

- architecture/runtime-source contracts;
- CDK test/build/synth;
- Floci integration deployed through CDK;
- Phase-3 config/secret integration tests;
- physical connectivity, retrieval, Golden and adversarial gates.

Phase 4 can proceed only after this convergence branch is green against the current `develop` head.
