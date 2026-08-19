from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError

from rag_ops_guard.local_credentials import LocalCredentialError, require_local_api_key

ROOT = Path(__file__).resolve().parents[2]
API_FILE = ROOT / ".local" / "api-url"
QUERY_FUNCTION = "rag-ops-guard-query"
DEFAULT_ENDPOINT = "http://localhost:4566"
DEFAULT_REGION = "us-east-1"
PROBE = {"__rag_ops_probe__": "connectivity"}


def _env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(_env_file(ROOT / ".env"))
    if API_FILE.is_file():
        env["RAG_API_URL"] = API_FILE.read_text(encoding="utf-8").strip().rstrip("/")
    return env


def _client(service: str, env: dict[str, str]) -> Any:
    return boto3.client(
        service,
        endpoint_url=env.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT),
        region_name=env.get("AWS_REGION", DEFAULT_REGION),
        aws_access_key_id=env.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _lambda_environment(env: dict[str, str]) -> dict[str, str]:
    response = _client("lambda", env).get_function_configuration(FunctionName=QUERY_FUNCTION)
    raw = response.get("Environment", {}).get("Variables", {})
    return {str(key): str(value) for key, value in raw.items()}


def _expected_topology(env: dict[str, str]) -> dict[str, str]:
    ovms = env.get("OVMS_CONTAINER_NAME", "rag-ops-ovms-rag")
    return {
        "AWS_ENDPOINT_URL": "http://floci:4566",
        "LLM_BASE_URL": "http://llama-gen:8080/v1",
        "EMBEDDING_BASE_URL": f"http://{ovms}:8000/v3",
        "RERANKER_BASE_URL": f"http://{ovms}:8000/v3",
    }


def _topology_problems(env: dict[str, str], lambda_env: dict[str, str]) -> list[str]:
    problems: list[str] = []
    for key, expected in _expected_topology(env).items():
        actual = lambda_env.get(key, "<missing>")
        if actual.rstrip("/") != expected.rstrip("/"):
            problems.append(f"{key}={actual} expected={expected}")
    return problems


def _api_id(api_url: str) -> str:
    marker = "/execute-api/"
    if marker not in api_url:
        raise RuntimeError(f"unsupported Floci API URL: {api_url}")
    suffix = api_url.split(marker, 1)[1]
    api_id = suffix.split("/", 1)[0].strip()
    if not api_id:
        raise RuntimeError(f"cannot resolve API id from {api_url}")
    return api_id


def _api_route_integration_problem(env: dict[str, str], api_url: str) -> str | None:
    api_id = _api_id(api_url)
    api = _client("apigatewayv2", env)
    routes = api.get_routes(ApiId=api_id).get("Items", [])
    route = next((item for item in routes if item.get("RouteKey") == "POST /v1/query"), None)
    if not isinstance(route, dict):
        return "POST /v1/query route is missing"
    target = str(route.get("Target") or "")
    if not target.startswith("integrations/"):
        return f"POST /v1/query has invalid target {target!r}"
    integration_id = target.split("/", 1)[1]
    integrations = api.get_integrations(ApiId=api_id).get("Items", [])
    integration = next(
        (item for item in integrations if str(item.get("IntegrationId")) == integration_id), None
    )
    if not isinstance(integration, dict):
        return f"API integration {integration_id!r} is missing"
    uri = str(integration.get("IntegrationUri") or "")
    expected = f":function:{QUERY_FUNCTION}"
    if expected not in uri:
        return f"POST /v1/query targets {uri!r}, expected Lambda {QUERY_FUNCTION}"
    return None


def _decode_proxy(payload: dict[str, Any], *, source: str) -> dict[str, Any]:
    status = payload.get("statusCode")
    body = payload.get("body")
    if not isinstance(body, str):
        raise RuntimeError(f"{source} returned proxy body {body!r}")
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{source} returned non-JSON body {body[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{source} returned non-object body {parsed!r}")
    if status != 200:
        raise RuntimeError(f"{source} connectivity probe returned statusCode={status}: {parsed!r}")
    return parsed


def _invoke_lambda_connectivity(env: dict[str, str], api_key: str) -> dict[str, Any]:
    event = {"headers": {"x-api-key": api_key}, "body": json.dumps(PROBE)}
    response = _client("lambda", env).invoke(
        FunctionName=QUERY_FUNCTION,
        Payload=json.dumps(event).encode("utf-8"),
    )
    raw = response["Payload"].read()
    if response.get("FunctionError"):
        raise RuntimeError(
            f"direct Lambda connectivity probe failed: {raw[:1000].decode(errors='replace')}"
        )
    try:
        proxy = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"direct Lambda probe returned non-JSON payload {raw[:500]!r}") from exc
    if not isinstance(proxy, dict):
        raise RuntimeError(f"direct Lambda probe returned {proxy!r}")
    return _decode_proxy(proxy, source="direct Lambda")


def _invoke_api_connectivity(
    api_url: str,
    env: dict[str, str],
    api_key: str,
) -> dict[str, Any]:
    del env
    response = httpx.post(
        f"{api_url.rstrip('/')}/v1/query",
        json=PROBE,
        headers={"x-api-key": api_key},
        timeout=120.0,
    )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"API connectivity probe returned HTTP {response.status_code} non-JSON body"
        ) from exc
    if response.status_code != 200 or not isinstance(payload, dict):
        raise RuntimeError(
            f"API connectivity probe returned HTTP {response.status_code}: {payload!r}"
        )
    return payload


def _validate_probe(payload: dict[str, Any], *, source: str, expected_hash: str) -> list[str]:
    problems: list[str] = []
    if payload.get("probe") != "runtime_connectivity":
        problems.append(f"{source}: wrong probe identity {payload.get('probe')!r}")
    if payload.get("function_name") != QUERY_FUNCTION:
        problems.append(
            f"{source}: function identity {payload.get('function_name')!r} != {QUERY_FUNCTION!r}"
        )
    if payload.get("config_hash") != expected_hash:
        problems.append(
            f"{source}: config_hash={payload.get('config_hash')!r} expected={expected_hash!r}"
        )
    if payload.get("ok") is not True:
        problems.append(f"{source}: runtime connectivity probe failed")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        problems.append(f"{source}: missing dependency checks")
        return problems
    for name in ("dynamodb", "s3", "s3vectors", "llm", "embeddings", "reranker"):
        check = checks.get(name)
        if not isinstance(check, dict):
            problems.append(f"{source}: missing {name} check")
        elif check.get("ok") is not True:
            detail = f"{check.get('error_type', 'error')} {check.get('error', '')}".rstrip()
            problems.append(f"{source}: {name} failed: {detail}")
    return problems


def _print_probe(payload: dict[str, Any]) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        return
    for name in ("dynamodb", "s3", "s3vectors", "llm", "embeddings", "reranker"):
        check = checks.get(name)
        if not isinstance(check, dict):
            continue
        if check.get("ok") is True:
            endpoint = check.get("endpoint")
            detail = f" @ {endpoint}" if endpoint else ""
            print(f"OK   {name:<12}{detail}")
        else:
            error_type = check.get("error_type", "error")
            error = check.get("error", "")
            print(f"FAIL {name:<12} {error_type}: {error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the authenticated physical data plane")
    parser.add_argument("--tenant-id", default="default")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    env = runtime_env()
    api_url = env.get("RAG_API_URL", "").strip()
    problems: list[str] = []

    print("\nRAG OPS GUARD - PHYSICAL CONNECTIVITY")
    print("=" * 60)
    if not api_url:
        print("FAIL API          .local/api-url missing")
        return 2
    try:
        api_key = require_local_api_key(str(args.tenant_id))
    except LocalCredentialError as exc:
        print(f"FAIL auth         {exc}")
        return 2

    try:
        lambda_env = _lambda_environment(env)
    except (BotoCoreError, ClientError, OSError) as exc:
        print(f"FAIL Lambda       {type(exc).__name__}: {exc}")
        return 2

    topology = _topology_problems(env, lambda_env)
    if topology:
        problems.extend(f"Lambda topology mismatch: {item}" for item in topology)
        for item in topology:
            print(f"FAIL topology     {item}")
    else:
        print("OK   topology     Floci + Qwen + OpenVINO endpoints match Lambda runtime")

    try:
        route_problem = _api_route_integration_problem(env, api_url)
    except Exception as exc:
        route_problem = f"{type(exc).__name__}: {exc}"
    if route_problem:
        problems.append(f"API route mismatch: {route_problem}")
        print(f"FAIL API route    {route_problem}")
    else:
        print(f"OK   API route    POST /v1/query -> {QUERY_FUNCTION}")

    expected_hash = lambda_env.get("CONFIG_HASH", "")
    try:
        direct = _invoke_lambda_connectivity(env, api_key)
    except Exception as exc:
        problems.append(f"direct Lambda runtime connectivity probe failed: {exc}")
        print(f"FAIL Lambda probe {type(exc).__name__}: {exc}")
    else:
        direct_problems = _validate_probe(direct, source="Lambda", expected_hash=expected_hash)
        problems.extend(direct_problems)
        print("OK   Lambda probe  authenticated dependency calls executed from Lambda runtime")
        _print_probe(direct)

    try:
        api_payload = _invoke_api_connectivity(api_url, env, api_key)
    except Exception as exc:
        problems.append(f"API connectivity probe failed: {exc}")
        print(f"FAIL API probe    {type(exc).__name__}: {exc}")
    else:
        api_problems = _validate_probe(api_payload, source="API", expected_hash=expected_hash)
        problems.extend(api_problems)
        if not api_problems:
            print("OK   API probe     API Gateway reached the same authenticated Lambda runtime")

    print("=" * 60)
    if problems:
        print("PHYSICAL CONNECTIVITY FAILED")
        for problem in problems:
            print(f"- {problem}")
        return 2
    print("PHYSICAL CONNECTIVITY READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
