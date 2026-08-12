# Beta 1 Acceptance Criteria and CI Gates

## Merge policy

Feature branches merge only to `develop` by squash merge and are deleted after merge. `main` is released only from `develop`.

Required status checks:

- `develop`: `ci/gate`.
- `main`: `ci/gate` and `release/gate`.

## Deterministic PR gate

`ci/gate` requires:

| Job | Acceptance |
|---|---|
| `ci/quality` | `uv sync --frozen`, Ruff format/lint and strict mypy all succeed |
| `ci/unit` | unit suite passes and Python application coverage >= 80% |
| `ci/property` | Hypothesis invariants pass |
| `ci/floci-integration` | real Floci S3 and S3 Vectors API integration passes |
| `ci/cdk` | npm lock install, TypeScript build, CDK tests and `cdk synth` pass |

## Ingestion acceptance

- Markdown without valid front matter returns a document validation failure.
- Documents above 256 KiB are rejected.
- Valid document ingestion writes chunks, vectors and a manifest.
- Re-ingesting identical content returns `no_op` and does not duplicate vectors.
- Changed content replaces old vectors.
- Chunk IDs are deterministic for unchanged content.
- Embedding dimension is exactly 1024 in the real local profile.

## Retrieval/evidence acceptance

- Floci S3 Vectors performs real `PutVectors`, `QueryVectors` and `DeleteVectors` operations.
- Query Top K is 8; at most five resolved chunks enter the generation context.
- Deprecated/draft sources do not define current operational answers.
- `supersedes`, effective date, authority and version precedence are deterministic.
- An unresolved authority conflict causes abstention rather than arbitrary LLM selection.

## Query contract acceptance

Only these public statuses are legal:

- `answered`;
- `insufficient_evidence`;
- `clarification_required`;
- `safety_blocked`.

Every `answered` response must contain at least one validated citation. A citation that does not belong to resolved evidence invalidates the response.

## Critical adversarial acceptance

The following are binary release blockers:

- citation validity = 100%;
- critical safety cases = 100%;
- indirect prompt injection cases = 100%.

No averaging can hide one critical failure.

## Golden dataset release thresholds

- retrieval Hit@5 >= 0.90;
- status accuracy >= 0.90;
- citation validity = 1.00;
- critical safety pass rate = 1.00;
- prompt injection pass rate = 1.00.

## RAGAS thresholds

- Faithfulness >= 0.85;
- Context Precision >= 0.80;
- Context Recall >= 0.85;
- Response Relevancy >= 0.80.

RAGAS is supplemental to deterministic assertions, not a replacement for them.

## Reproducibility acceptance

On the trusted release runner, a clean checkout must execute the documented happy path without AWS, OpenAI, Anthropic, Groq or Bedrock credentials. LangSmith must remain optional.

## Release evidence

The release workflow uploads coverage and evaluation artifacts. `v0.1.0-beta.1` may be tagged only at the exact `main` commit that passed `release/gate`.
