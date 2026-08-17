from __future__ import annotations

import json
from pathlib import Path

from rag_ops_guard.app import knowledge_search
from rag_ops_guard.domain.models import QueryContext

# Calibration and deterministic admission validation must use the same labelled corpus.
DATASET_PATH = Path("evaluation/datasets/retrieval-calibration-v2.json")


def validation_passes(
    case_class: str,
    *,
    expected_titles: set[str],
    admitted_titles: list[str],
) -> bool:
    """Validate the deterministic retrieval contract, not RAG context precision.

    Grounded cases require at least one expected evidence title to survive admission.
    In-domain-unanswerable and out-of-domain cases must admit no grounded evidence.

    Additional admitted evidence is diagnostic here, not an automatic failure: context
    precision is an evaluation metric and is measured later by RAGAS. Treating
    ``expected_titles`` as an exhaustive allow-list made the validator stricter than the
    labelled dataset actually specifies and duplicated the context-precision gate.
    """
    admitted = set(admitted_titles)
    if case_class == "grounded":
        return bool(expected_titles.intersection(admitted))
    if case_class in {"in_domain_unanswerable", "out_of_domain"}:
        return not admitted
    raise RuntimeError(f"unknown retrieval validation class: {case_class}")


def main() -> None:
    samples = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    search = knowledge_search()
    failures: list[str] = []
    report_rows: list[dict[str, object]] = []

    for sample in samples:
        query = str(sample["query"])
        case_class = str(sample["class"])
        expected_titles = {str(title) for title in sample.get("expected_titles", [])}
        result = search.search(query, QueryContext(), query_mode="probe")
        admitted_titles = [item.chunk.title for item in result.admitted]
        admitted_title_set = set(admitted_titles)
        matched_titles = sorted(expected_titles.intersection(admitted_title_set))
        extra_titles = sorted(admitted_title_set.difference(expected_titles))

        passed = validation_passes(
            case_class,
            expected_titles=expected_titles,
            admitted_titles=admitted_titles,
        )
        status = "PASS" if passed else "FAIL"
        print(
            f"RETRIEVAL {status} {sample['id']}: class={case_class} "
            f"domain={result.domain_relevance:.6f} grounded={result.grounded_relevance:.6f} "
            f"supported={result.supported} matched={matched_titles} extras={extra_titles} "
            f"titles={admitted_titles}"
        )
        report_rows.append(
            {
                "case_id": str(sample["id"]),
                "class": case_class,
                "passed": passed,
                "domain_relevance": result.domain_relevance,
                "grounded_relevance": result.grounded_relevance,
                "supported": result.supported,
                "expected_titles": sorted(expected_titles),
                "matched_titles": matched_titles,
                "extra_titles": extra_titles,
                "admitted_titles": admitted_titles,
            }
        )
        if not passed:
            failures.append(
                f"{sample['id']}: class={case_class} expected={sorted(expected_titles)} "
                f"admitted={admitted_titles}"
            )

    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    (output / "retrieval-validation.json").write_text(
        json.dumps(report_rows, indent=2), encoding="utf-8"
    )

    if failures:
        joined = "\n".join(f"- {failure}" for failure in failures)
        raise RuntimeError(f"retrieval validation failed:\n{joined}")

    print("RETRIEVAL READY: all labelled cases satisfy required-evidence admission")


if __name__ == "__main__":
    main()
