from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langsmith import Client

BASE_DATASET = Path("evaluation/datasets/golden-v1.json")
MIXED_DATASET = Path("evaluation/datasets/golden-mixed-v1.json")


def _cases() -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    for path in (BASE_DATASET, MIXED_DATASET):
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise SystemExit(f"{path} must contain a JSON list")
        rows.extend(row for row in loaded if isinstance(row, dict))
    result = {str(row["question"]): str(row["id"]) for row in rows}
    if len(result) != 34:
        raise SystemExit(f"expected exactly 34 unique Golden questions, got {len(result)}")
    return result


def _find_question(value: Any) -> str | None:
    if isinstance(value, dict):
        direct = value.get("question")
        if isinstance(direct, str):
            return direct
        body = value.get("body")
        if isinstance(body, str):
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                parsed = None
            question = _find_question(parsed)
            if question:
                return question
        for nested in value.values():
            question = _find_question(nested)
            if question:
                return question
    elif isinstance(value, list):
        for nested in value:
            question = _find_question(nested)
            if question:
                return question
    return None


def _start_time() -> datetime:
    raw = os.environ.get("GOLDEN_LANGSMITH_START", "").strip()
    if not raw:
        raise SystemExit("GOLDEN_LANGSMITH_START is required")
    parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc)


def main() -> None:
    if os.environ.get("LANGSMITH_TRACING", "").casefold() != "true":
        raise SystemExit("LangSmith tracing must be enabled for Golden acceptance")
    if not os.environ.get("LANGSMITH_API_KEY", "").strip():
        raise SystemExit("LANGSMITH_API_KEY is required for Golden acceptance")

    expected = _cases()
    project = os.environ.get("LANGSMITH_PROJECT", "rag-ops-guard-physical-golden")
    client = Client()
    start = _start_time()
    deadline = time.monotonic() + 120
    matched: dict[str, str] = {}
    errored: list[str] = []

    while time.monotonic() < deadline:
        matched.clear()
        errored.clear()
        runs = client.list_runs(
            project_name=project,
            start_time=start,
            is_root=True,
        )
        for run in runs:
            if run.name != "rag_query":
                continue
            question = _find_question(run.inputs)
            if question not in expected:
                continue
            if run.error:
                errored.append(expected[question])
                continue
            matched[expected[question]] = str(run.id)
        if len(matched) == 34 and not errored:
            break
        time.sleep(5)

    missing = sorted(set(expected.values()) - set(matched))
    if missing or errored:
        raise SystemExit(
            "LangSmith Golden trace verification failed: "
            f"matched={len(matched)}/34 missing={missing} errored={sorted(set(errored))}"
        )

    print(f"LANGSMITH GOLDEN READY: 34/34 root traces in project {project}")


if __name__ == "__main__":
    main()
