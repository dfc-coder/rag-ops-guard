from __future__ import annotations

import json
from pathlib import Path

from rag_ops_guard.app import knowledge_search
from rag_ops_guard.domain.models import QueryContext

DATASET_PATH = Path("evaluation/datasets/retrieval-calibration-v1.json")


def main() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    search = knowledge_search()
    failures: list[str] = []

    for sample in samples:
        query = str(sample["query"])
        label = str(sample["label"])
        expected_titles = {str(title) for title in sample.get("expected_titles", [])}
        result = search.search(query, QueryContext(), query_mode="probe")
        admitted_titles = [item.chunk.title for item in result.admitted]

        if label == "positive":
            passed = bool(expected_titles.intersection(admitted_titles))
        elif label == "negative":
            passed = not result.admitted
        else:
            raise RuntimeError(f"unknown retrieval validation label: {label}")

        status = "PASS" if passed else "FAIL"
        print(
            f"RETRIEVAL {status} {sample['id']}: "
            f"supported={result.supported} titles={admitted_titles} "
            f"scores={result.reranker_scores}"
        )
        if not passed:
            failures.append(
                f"{sample['id']}: label={label} expected={sorted(expected_titles)} "
                f"admitted={admitted_titles}"
            )

    if failures:
        joined = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"retrieval validation failed:\n{joined}")

    print("RETRIEVAL READY: all labeled positive and hard-negative cases passed")


if __name__ == "__main__":
    main()
