from __future__ import annotations

import os
import sys
from pathlib import Path

LANGSMITH_KEYS = (
    "LANGSMITH_API_KEY",
    "LANGSMITH_WORKSPACE_ID",
    "LANGSMITH_ENDPOINT",
    "LANGSMITH_PROJECT",
)


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in LANGSMITH_KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def _candidate_files() -> list[Path]:
    paths: list[Path] = []
    explicit = os.environ.get("RAG_OPS_LANGSMITH_ENV_FILE", "").strip()
    if explicit:
        paths.append(Path(explicit).expanduser())
    paths.extend(
        [
            Path.home() / ".config/rag-ops-guard/langsmith.env",
            Path.home() / ".config/rag-ops-guard/.env",
            Path.home() / "Documents/projects/rag-ops-guard/.env",
        ]
    )
    return paths


def resolve_langsmith_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in _candidate_files():
        for key, value in _parse_env_file(path).items():
            values.setdefault(key, value)

    for key in LANGSMITH_KEYS:
        current = os.environ.get(key, "").strip()
        if current:
            values[key] = current

    repo_key = os.environ.get("REPO_LANGSMITH_API_KEY", "").strip()
    if repo_key:
        values["LANGSMITH_API_KEY"] = repo_key
    repo_workspace = os.environ.get("REPO_LANGSMITH_WORKSPACE_ID", "").strip()
    if repo_workspace:
        values["LANGSMITH_WORKSPACE_ID"] = repo_workspace

    values.setdefault("LANGSMITH_PROJECT", "rag-ops-guard-physical-golden")
    values.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    values["LANGSMITH_TRACING"] = "true"
    return values


def write_github_environment(path: Path, values: dict[str, str]) -> None:
    api_key = values.get("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        searched = ", ".join(str(path) for path in _candidate_files())
        raise SystemExit(
            "Golden acceptance requires LANGSMITH_API_KEY via repository secret, runner "
            f"environment, RAG_OPS_LANGSMITH_ENV_FILE, or one of: {searched}"
        )
    print(f"::add-mask::{api_key}")
    with path.open("a", encoding="utf-8") as handle:
        for key in (
            "LANGSMITH_TRACING",
            "LANGSMITH_PROJECT",
            "LANGSMITH_ENDPOINT",
            "LANGSMITH_API_KEY",
            "LANGSMITH_WORKSPACE_ID",
        ):
            value = values.get(key, "").strip()
            if value:
                handle.write(f"{key}={value}\n")
    print(f"LangSmith tracing configured for project {values['LANGSMITH_PROJECT']}")


def main() -> None:
    github_env = os.environ.get("GITHUB_ENV", "").strip()
    if not github_env:
        raise SystemExit("GITHUB_ENV is required")
    write_github_environment(Path(github_env), resolve_langsmith_environment())


if __name__ == "__main__":
    sys.exit(main())
