from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import boto3
import httpx
from botocore.exceptions import BotoCoreError, ClientError

ROOT = Path(__file__).resolve().parents[2]
API_FILE = ROOT / ".local" / "api-url"
GOLDEN_FILES = (
    ROOT / "evaluation" / "datasets" / "golden-v1.json",
    ROOT / "evaluation" / "datasets" / "golden-mixed-v1.json",
)
QUERY_FUNCTION = "rag-ops-guard-query"
DEFAULT_ENDPOINT = "http://localhost:4566"
DEFAULT_REGION = "us-east-1"


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
    """Load local config deterministically and ignore stale shell RAG_API_URL values."""
    env = os.environ.copy()
    env.update(_env_file(ROOT / ".env"))
    env.update(_env_file(ROOT / ".local" / "relevance-floors.env"))
    if API_FILE.is_file():
        env["RAG_API_URL"] = API_FILE.read_text(encoding="utf-8").strip().rstrip("/")
    else:
        env.pop("RAG_API_URL", None)
    return env


def _run_text(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return ""
    return result.stdout.strip()


def _git_state() -> tuple[str, str]:
    branch = _run_text(["git", "branch", "--show-current"]) or "unknown"
    commit = _run_text(["git", "rev-parse", "--short", "HEAD"]) or "unknown"
    return branch, commit


def _container_states() -> dict[str, str]:
    output = _run_text(["podman", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"])
    states: dict[str, str] = {}
    for line in output.splitlines():
        if "\t" not in line:
            continue
        name, state = line.split("\t", 1)
        states[name.strip()] = state.strip()
    return states


def _runner_active() -> bool:
    output = _run_text(["pgrep", "-af", "Runner.Listener"])
    return bool(output.strip())


def _lambda_client(env: dict[str, str]) -> Any:
    return boto3.client(
        "lambda",
        endpoint_url=env.get("AWS_ENDPOINT_URL", DEFAULT_ENDPOINT),
        region_name=env.get("AWS_REGION", DEFAULT_REGION),
        aws_access_key_id=env.get("AWS_ACCESS_KEY_ID", "test"),
        aws_secret_access_key=env.get("AWS_SECRET_ACCESS_KEY", "test"),
    )


def _lambda_environment(env: dict[str, str]) -> dict[str, str] | None:
    try:
        response = _lambda_client(env).get_function_configuration(FunctionName=QUERY_FUNCTION)
    except (BotoCoreError, ClientError, OSError):
        return None
    raw = response.get("Environment", {}).get("Variables", {})
    return {str(key): str(value) for key, value in raw.items()}


def _physical_model(env: dict[str, str]) -> str | None:
    port = env.get("LLAMA_GEN_HOST_PORT", "8080")
    try:
        response = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=2.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    model = data[0].get("id")
    return str(model) if model else None


def _golden_count() -> int:
    total = 0
    for path in GOLDEN_FILES:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, list):
            total += len(payload)
    return total


def _last_summary() -> str | None:
    path = ROOT / "artifacts" / "evaluation" / "summary.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cases = payload.get("cases")
    passed = payload.get("passed")
    accuracy = payload.get("case_accuracy")
    if isinstance(cases, int) and isinstance(passed, int):
        return f"{passed}/{cases} passed"
    if isinstance(cases, int) and isinstance(accuracy, (int, float)):
        return f"{cases} cases, accuracy={accuracy:.1%}"
    return None


def _ok(label: str, detail: str) -> None:
    print(f"OK   {label:<12} {detail}")


def _warn(label: str, detail: str) -> None:
    print(f"WARN {label:<12} {detail}")


def _fail(label: str, detail: str) -> None:
    print(f"FAIL {label:<12} {detail}")


def status() -> int:
    env = runtime_env()
    branch, commit = _git_state()
    containers = _container_states()
    lambda_env = _lambda_environment(env)
    physical_model = _physical_model(env)
    local_model = env.get("LLM_MODEL", "<unset>")
    api_url = env.get("RAG_API_URL")

    print("\nRAG OPS GUARD - LOCAL STATUS")
    print("=" * 52)
    _ok("Git", f"{branch} @ {commit}")

    if _runner_active():
        _warn("GH runner", "RUNNING - stop it during local physical/Golden evaluation")
    else:
        _ok("GH runner", "stopped")

    expected = {
        "Floci": env.get("FLOCI_CONTAINER_NAME", "rag-ops-floci"),
        "LLM": env.get("LLAMA_GEN_CONTAINER_NAME", "rag-ops-llama-gen"),
        "OpenVINO": env.get("OVMS_CONTAINER_NAME", "rag-ops-ovms-rag"),
    }
    for label, name in expected.items():
        state = containers.get(name)
        if state and state.casefold().startswith("up"):
            _ok(label, f"{name} - {state}")
        elif state:
            _fail(label, f"{name} - {state}")
        else:
            _fail(label, f"{name} not found")

    if api_url:
        _ok("API", api_url)
    else:
        _fail("API", ".local/api-url missing; run make physical-ready")

    if lambda_env is None:
        _fail("Lambda", f"{QUERY_FUNCTION} unavailable")
    else:
        _ok("Lambda", QUERY_FUNCTION)
        lambda_model = lambda_env.get("LLM_MODEL", "<unset>")
        if lambda_model == local_model:
            _ok("Model cfg", lambda_model)
        else:
            _warn("Model cfg", f".env={local_model} | Lambda={lambda_model}")
        tracing = lambda_env.get("LANGSMITH_TRACING", "false")
        project = lambda_env.get("LANGSMITH_PROJECT", "<unset>")
        if tracing.casefold() == "true":
            _ok("LangSmith", f"enabled - {project}")
        else:
            _warn("LangSmith", f"disabled - {project}")

    if physical_model:
        if physical_model == local_model:
            _ok("Model real", physical_model)
        else:
            _warn("Model real", f"server={physical_model} | .env={local_model}")
    else:
        _fail("Model real", "llama.cpp /v1/models unavailable")

    _ok("Golden", f"{_golden_count()} cases available")
    last = _last_summary()
    if last:
        _ok("Last run", last)

    print("=" * 52)
    print("Commands: make status | make golden CASE=<id> | make golden-all | make logs")
    return 0


def _preflight(env: dict[str, str]) -> list[str]:
    problems: list[str] = []
    if _runner_active():
        problems.append("GitHub self-hosted runner is active and can steal CPU from Qwen")
    if not env.get("RAG_API_URL"):
        problems.append(".local/api-url is missing")
    containers = _container_states()
    for key, default in (
        ("FLOCI_CONTAINER_NAME", "rag-ops-floci"),
        ("LLAMA_GEN_CONTAINER_NAME", "rag-ops-llama-gen"),
        ("OVMS_CONTAINER_NAME", "rag-ops-ovms-rag"),
    ):
        name = env.get(key, default)
        state = containers.get(name, "")
        if not state.casefold().startswith("up"):
            problems.append(f"container {name} is not running")
    if _lambda_environment(env) is None:
        problems.append(f"Lambda {QUERY_FUNCTION} is unavailable")
    return problems


def _classify_failure(output: str, elapsed: float) -> str:
    lower = output.casefold()
    if "http 502" in lower:
        if elapsed < 5:
            return "API/Floci: request failed before model execution"
        return "timeout/data-plane: API or Lambda was cut during a long model execution"
    if "timed out during agent-query" in lower or "hard wall timeout" in lower:
        return "LLM/runtime timeout"
    if "status=error" in lower:
        return "agent/model: application returned domain status=error"
    if "evaluation thresholds failed" in lower:
        return "quality gate: suite ran, but one or more Golden thresholds were not met"
    if "connection refused" in lower or "connecterror" in lower:
        return "runtime connectivity: one local service is unreachable"
    return "runner/test failure - inspect the last lines above"


def golden(case_id: str | None, all_cases: bool, force: bool) -> int:
    env = runtime_env()
    problems = _preflight(env)
    if problems and not force:
        print("\nPRE-FLIGHT FAILED")
        for problem in problems:
            print(f"- {problem}")
        print("\nRun `make status` for the complete state.")
        print("Use FORCE=1 only if you intentionally want to ignore these warnings.")
        return 2

    if all_cases:
        label = f"ALL {_golden_count()} GOLDEN CASES"
        command = [sys.executable, "evaluation/runners/run_golden.py"]
    else:
        selected = case_id or "retry-count-1"
        label = f"GOLDEN CASE {selected}"
        command = [sys.executable, "evaluation/runners/run_golden.py", "--case-id", selected]

    print(f"\nRUN  {label}")
    print(f"API       {env.get('RAG_API_URL', '<missing>')}")
    lambda_env = _lambda_environment(env) or {}
    print(f"MODEL     {lambda_env.get('LLM_MODEL', env.get('LLM_MODEL', '<unknown>'))}")
    print(
        "LANGSMITH "
        f"{lambda_env.get('LANGSMITH_TRACING', 'false')} / "
        f"{lambda_env.get('LANGSMITH_PROJECT', '<unset>')}"
    )
    print("RESULTS   artifacts/evaluation/")
    print("-" * 72)

    started = perf_counter()
    lines: list[str] = []
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip())
    return_code = process.wait()
    elapsed = perf_counter() - started

    print("-" * 72)
    if return_code == 0:
        print(f"PASS {label} ({elapsed:.1f}s)")
        print("Next: `make golden-all`" if not all_cases else "Results: artifacts/evaluation/summary.json")
        return 0

    reason = _classify_failure("\n".join(lines), elapsed)
    print(f"FAIL {label} ({elapsed:.1f}s)")
    print(f"CAUSE     {reason}")
    print("NEXT      make status")
    print("LOGS      make logs")
    return return_code or 1


def logs() -> int:
    env = runtime_env()
    names = [
        env.get("FLOCI_CONTAINER_NAME", "rag-ops-floci"),
        env.get("LLAMA_GEN_CONTAINER_NAME", "rag-ops-llama-gen"),
        env.get("OVMS_CONTAINER_NAME", "rag-ops-ovms-rag"),
    ]
    print("\nRECENT LOCAL RUNTIME LOGS (last 10 minutes)")
    for name in names:
        print(f"\n===== {name} =====")
        output = _run_text(["podman", "logs", "--since", "10m", "--tail", "120", name])
        if output:
            print(output)
        else:
            print("<no logs or container unavailable>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG Ops Guard local operational console")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show what is running and the active local configuration")

    golden_parser = subparsers.add_parser("golden", help="Run one Golden case or all 34")
    golden_parser.add_argument("--case", dest="case_id")
    golden_parser.add_argument("--all", action="store_true", dest="all_cases")
    golden_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("logs", help="Show recent Floci/LLM/OpenVINO logs")
    args = parser.parse_args()

    if args.command == "status":
        return status()
    if args.command == "golden":
        return golden(args.case_id, args.all_cases, args.force)
    if args.command == "logs":
        return logs()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
