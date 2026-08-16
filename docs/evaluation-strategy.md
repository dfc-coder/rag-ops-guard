# Evaluation Strategy

RAG Ops Guard uses three complementary evaluation layers.

## 1. Deterministic software assertions

pytest/Hypothesis validate parsing, chunk IDs, idempotency, version precedence, citation/segment integrity, safety boundaries and API contracts. These are the strongest release gates because their results are deterministic.

## 2. Ground-truth benchmark

`evaluation/datasets/golden-v1.json` contains the operational evaluation corpus with expected status, expected/forbidden sources and required/forbidden facts. AcmePay/operations examples are evaluation fixtures, not product-domain routing rules.

The release runner produces status/source/retrieval/citation/safety and prompt-injection metrics.

## 3. Model-based RAG metrics

RAGAS is complementary evidence, not independent ground truth. SPEC-5 imposes three rules:

1. A metric must pass its aggregate mean floor and, where configured, every per-case floor. Current per-case floors live in `evaluation/thresholds.yaml`: faithfulness `0.60` and context precision `0.30`.
2. Faithfulness receives only response segments carrying validated citations. Ungrounded/general segments are intentionally excluded from that metric.
3. The evaluator is the same Qwen3-4B model used by runtime. A second judge model is forbidden.

### Judge-human calibration

Before RAGAS may gate a release:

1. Run RAGAS on the target runtime to produce `artifacts/evaluation/ragas-results.json`.
2. A human manually reviews exactly 10 cases and creates `evaluation/datasets/judge-human-10.json` as a JSON list of `{ "case_id": "...", "human_pass": true|false }` entries.
3. Run `scripts/calibrate_ragas_judge.py`. It compares the human labels with Qwen3-4B faithfulness judgments and writes `artifacts/evaluation/judge-policy.json`.
4. Run RAGAS again with that policy present.

Policy from the measured agreement:

- 9-10/10: RAGAS may gate; the mean faithfulness floor is `0.85` plus the per-case floor.
- 7-8/10: RAGAS may gate, but its faithfulness mean cutoff is derived from the labelled sample and must be lower than `0.85`; per-case floors remain active.
- 6/10 or less: RAGAS becomes informational and cannot fail the release. Deterministic `citation_validity`, `segment_integrity`, security and architecture gates remain blocking.

The calibration artifact is intentionally not fabricated or committed without a real target-machine run and real human labels.

## Current runtime prerequisite

The pivot specification requires Qwen3-4B for both runtime and judge. If `LLM_MODEL` is not a Qwen3-4B model, `run_ragas.py` and `calibrate_ragas_judge.py` fail explicitly instead of silently evaluating with a different/smaller judge.

LangSmith remains optional for trace visualization and experiments when credentials are supplied.
