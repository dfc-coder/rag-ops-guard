from __future__ import annotations

import pytest

from scripts.prepare_judge_human_review import build_review, select_review_rows


def _golden(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "question": f"question {case_id}",
        "ragas_question": f"grounded question {case_id}",
        "payload": {
            "status": "answered_grounded",
            "segments": [
                {
                    "text": f"grounded response {case_id}",
                    "citations": [{"chunk_id": case_id, "s3_key": f"chunks/{case_id}.json"}],
                }
            ],
            "citations": [{"chunk_id": case_id, "s3_key": f"chunks/{case_id}.json"}],
        },
    }


def test_human_review_selects_exactly_ten_cases_deterministically() -> None:
    golden = [_golden(f"c{index:02d}") for index in range(12)]
    scores = [
        {"case_id": f"c{index:02d}", "faithfulness": index / 20}
        for index in range(12)
    ]

    first = select_review_rows(golden, scores)
    second = select_review_rows(golden, scores)

    assert len(first) == 10
    assert [pair[1]["case_id"] for pair in first] == [pair[1]["case_id"] for pair in second]
    assert len({str(pair[1]["case_id"]) for pair in first}) == 10


def test_human_review_never_fabricates_human_labels() -> None:
    golden = [_golden(f"c{index:02d}") for index in range(10)]
    scores = [
        {"case_id": f"c{index:02d}", "faithfulness": 0.9}
        for index in range(10)
    ]
    pairs = select_review_rows(golden, scores)

    review = build_review(pairs, context_loader=lambda key: f"evidence:{key}")

    assert len(review) == 10
    assert all(row["human_pass"] is None for row in review)
    assert all(str(row["question"]).startswith("grounded question") for row in review)
    assert all(row["evidence"] for row in review)


def test_human_review_rejects_fewer_than_ten_scored_cases() -> None:
    golden = [_golden(f"c{index:02d}") for index in range(9)]
    scores = [
        {"case_id": f"c{index:02d}", "faithfulness": 0.9}
        for index in range(9)
    ]

    with pytest.raises(ValueError, match="at least 10"):
        select_review_rows(golden, scores)
