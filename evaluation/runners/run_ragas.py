from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import boto3
import httpx
import yaml
from botocore.config import Config
from openai import AsyncOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)


def _api_url() -> str:
    configured = os.environ.get("RAG_API_URL")
    if configured:
        return configured.rstrip("/")
    path = Path(".local/api-url")
    if not path.exists():
        raise SystemExit("RAG_API_URL missing and .local/api-url not found")
    return path.read_text().strip().rstrip("/")


def _s3() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL", "http://localhost:4566"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
        config=Config(s3={"addressing_style": "path"}),
    )


def _context_text(s3: Any, bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    payload = json.loads(response["Body"].read().decode("utf-8"))
    return str(payload["text"])


def _collect_dataset() -> EvaluationDataset:
    cases = json.loads(Path("evaluation/datasets/golden-v1.json").read_text())
    api = _api_url()
    s3 = _s3()
    bucket = os.environ.get("S3_DOCUMENT_BUCKET", "rag-ops-guard-docs-local")
    samples: list[dict[str, object]] = []

    for case in cases:
        if case["expected_status"] != "answered":
            continue
        response = httpx.post(
            f"{api}/v1/query",
            json={"question": case["question"], "context": case.get("context", {})},
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "answered":
            raise SystemExit(
                f"RAGAS dataset collection expected answered for {case['id']}, "
                f"got {payload.get('status')}"
            )
        contexts = [
            _context_text(s3, bucket, str(citation["s3_key"]))
            for citation in payload.get("citations", [])
        ]
        samples.append(
            {
                "user_input": case["question"],
                "retrieved_contexts": contexts,
                "response": payload["answer"],
                "reference": case["reference_answer"],
            }
        )
    return EvaluationDataset.from_list(samples)


def main() -> None:
    thresholds = yaml.safe_load(Path("evaluation/thresholds.yaml").read_text())["ragas"]

    llm_client = AsyncOpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8080/v1"),
        api_key="local",
    )
    embedding_client = AsyncOpenAI(
        base_url=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8081/v1"),
        api_key="local",
    )
    evaluator_llm = llm_factory(
        os.environ.get("LLM_MODEL", "qwen35-0.8b-rag"),
        client=llm_client,
        temperature=0.0,
        system_prompt=(
            "You are evaluating an integration-operations RAG system. "
            "Judge only the provided question, response, reference, and contexts. /no_think"
        ),
    )
    evaluator_embeddings = OpenAIEmbeddings(
        client=embedding_client,
        model=os.environ.get("EMBEDDING_MODEL", "qwen3-embedding-0.6b"),
    )
    result = evaluate(
        dataset=_collect_dataset(),
        metrics=[
            Faithfulness(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
            ResponseRelevancy(),
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    frame = result.to_pandas()
    summary = {
        "faithfulness": float(frame["faithfulness"].mean()),
        "context_precision": float(frame["llm_context_precision_with_reference"].mean()),
        "context_recall": float(frame["context_recall"].mean()),
        "response_relevancy": float(frame["answer_relevancy"].mean()),
        "evaluator": "Qwen3.5-0.8B-Q4_K_M via llama.cpp",
        "embedding_evaluator": "Qwen3-Embedding-0.6B-Q8_0 via llama.cpp",
        "samples": len(frame),
    }
    output = Path("artifacts/evaluation")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_json(output / "ragas-results.json", orient="records", indent=2)
    (output / "ragas.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    failures = [
        f"{name}: {summary[name]:.3f} < {float(threshold):.3f}"
        for name, threshold in thresholds.items()
        if float(summary[name]) < float(threshold)
    ]
    if failures:
        raise SystemExit("RAGAS thresholds failed:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
