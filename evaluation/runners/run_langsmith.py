from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import httpx
from langsmith import Client

DATASET_PATH = Path("evaluation/datasets/golden-v1.json")
DATASET_PREFIX = "rag-ops-guard-golden-v1"


def api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


def load_cases() -> list[dict[str, Any]]:
    payload = json.loads(DATASET_PATH.read_text())
    if not isinstance(payload, list) or not payload:
        raise SystemExit("golden dataset must be a non-empty JSON array")
    return [dict(case) for case in payload]


def dataset_name() -> str:
    digest = hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest()[:12]
    return f"{DATASET_PREFIX}-{digest}"


def ensure_dataset(client: Client, cases: list[dict[str, Any]]) -> str:
    name = dataset_name()
    matches = list(client.list_datasets(dataset_name=name))
    if len(matches) > 1:
        raise SystemExit(f"multiple LangSmith datasets found with exact name {name}")

    if matches:
        dataset = matches[0]
        examples = list(client.list_examples(dataset_id=dataset.id))
        if len(examples) != len(cases):
            raise SystemExit(
                f"existing dataset {name} has {len(examples)} examples; expected {len(cases)}"
            )
        return dataset.name

    dataset = client.create_dataset(
        dataset_name=name,
        description=(
            "Versioned RAG Ops Guard Beta 1 golden dataset. The content hash in the dataset "
            "name changes only when the repository benchmark changes."
        ),
    )
    examples = []
    for case in cases:
        examples.append(
            {
                "inputs": {
                    "question": case["question"],
                    "context": case.get("context", {}),
                },
                "outputs": {
                    "expected_status": case["expected_status"],
                    "reference_answer": case.get("reference_answer", ""),
                    "expected_source_ids": case.get("expected_source_ids", []),
                    "forbidden_source_ids": case.get("forbidden_source_ids", []),
                    "required_facts": case.get("required_facts", []),
                    "forbidden_facts": case.get("forbidden_facts", []),
                },
                "metadata": {
                    "case_id": case["id"],
                    "category": case["category"],
                    "dataset_version": "golden-v1",
                },
            }
        )
    client.create_examples(dataset_id=dataset.id, examples=examples)
    return dataset.name


def target(inputs: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(
        f"{api_url()}/v1/query",
        json={
            "question": inputs["question"],
            "context": inputs.get("context", {}),
        },
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "status": payload.get("status"),
        "answer": payload.get("answer") or "",
        "citations": payload.get("citations", []),
    }


def _source_identities(citations: list[dict[str, Any]]) -> set[str]:
    identities: set[str] = set()
    for citation in citations:
        logical_id = str(citation.get("logical_id", ""))
        version = str(citation.get("version", ""))
        if logical_id:
            identities.add(logical_id)
            major = version.split(".", maxsplit=1)[0]
            if major:
                identities.add(f"{logical_id}-v{major}")
    return identities


def status_match(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> bool:
    del inputs
    return outputs.get("status") == reference_outputs.get("expected_status")


def fact_contract(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> bool:
    del inputs
    answer = str(outputs.get("answer") or "").lower()
    required = [str(value).lower() for value in reference_outputs.get("required_facts", [])]
    forbidden = [str(value).lower() for value in reference_outputs.get("forbidden_facts", [])]
    return all(value in answer for value in required) and all(
        value not in answer for value in forbidden
    )


def source_contract(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> bool:
    del inputs
    citations = outputs.get("citations") or []
    identities = _source_identities([dict(value) for value in citations])
    expected = {str(value) for value in reference_outputs.get("expected_source_ids", [])}
    forbidden = {str(value) for value in reference_outputs.get("forbidden_source_ids", [])}
    return expected.issubset(identities) and forbidden.isdisjoint(identities)


def main() -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise SystemExit("LANGSMITH_API_KEY is required")

    client = Client()
    cases = load_cases()
    name = ensure_dataset(client, cases)
    commit = os.environ.get("GITHUB_SHA", "local")[:12]
    results = client.evaluate(
        target,
        data=name,
        evaluators=[status_match, fact_contract, source_contract],
        experiment_prefix=f"rag-ops-guard-beta1-{commit}",
        max_concurrency=1,
    )
    print(results)


if __name__ == "__main__":
    main()
