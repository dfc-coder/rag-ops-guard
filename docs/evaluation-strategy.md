# Evaluation Strategy

RAG Ops Guard uses three complementary evaluation layers.

## 1. Deterministic software assertions

pytest/Hypothesis validate parsing, chunk IDs, idempotency, version precedence, graph routing, citation integrity and API contracts. These results must be deterministic.

## 2. Ground-truth RAG benchmark

`evaluation/datasets/golden-v1.json` contains 30 operational questions with expected status, expected/forbidden sources and required/forbidden facts. It includes normal, missing-evidence, ambiguity, obsolete/conflicting documentation, unsafe-operation, prompt-injection and safety cases.

The release runner produces:

- status accuracy;
- source hit rate;
- citation validity;
- safety pass rate;
- prompt-injection pass rate.

## 3. Model-based RAG metrics

RAGAS evaluates real local Qwen responses. The same local model family is intentionally used for generation/evaluation in Beta 1 to keep the project offline, so RAGAS values are labelled as local-model evaluator results and are never treated as independent ground truth.

LangSmith is optional and provides trace visualization, datasets and experiments when credentials are supplied.
