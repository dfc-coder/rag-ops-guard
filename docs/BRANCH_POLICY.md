# Branch policy after the functional beta freeze

The client-validated code recovery point is:

```text
baseline/beta-react-v0.1
```

Frozen commit:

```text
a16c1cc5b66fc7c83890760c5658e16699ca1264
```

## Branches to keep

- `main` — release branch.
- `develop` — integration branch.
- `baseline/beta-react-v0.1` — frozen functional-beta recovery point; never rewrite.

Short-lived `feature/*`, `fix/*`, `docs/*`, and `chore/*` branches should be deleted after merge.

## New feature rule

New feature work starts from the functional baseline or from `develop` after the baseline merge:

```bash
git fetch origin
git switch develop
git pull --ff-only
git switch -c feature/<name>
```

For experiments that intentionally need the exact client-validated code rather than the current integration head:

```bash
git fetch origin
git switch -c experiment/<name> origin/baseline/beta-react-v0.1
```

Do not develop directly on the baseline branch.

## Legacy branches reviewed after the freeze

The following branches predate the functional ReAct/OpenVINO baseline and are superseded by the current runtime:

- `feat/qwen35-08b-runtime` — fully behind `develop`; no unique commits.
- `fix/qwen35-env-profile` — fully behind `develop`; no unique commits.
- `feat/qwen35-08b-generation` — obsolete Qwen 0.8B generation experiment.
- `feat/qwen35-profile` — obsolete Qwen profile experiment.
- `feat/qwen35-recommended-settings` — obsolete settings experiment, including temporary files.
- `fix/qwen35-env-profile-final` — obsolete environment-only follow-up.
- `fix/ui-rag-general-questions` — obsolete one-line legacy UI experiment superseded by the ReAct Gradio UI.

These branches should not be used as the base for future work.

## Change discipline

- One architectural variable per change when possible.
- Model swaps, hardware backend changes, retrieval-policy changes, and agent features should not be bundled together.
- Preserve `make beta-react` and `make local-down` as the stable beta operator contract.
- Benchmark runtime/backend changes before replacing the validated path.
- If a regression is difficult to isolate, compare directly against `baseline/beta-react-v0.1`.
