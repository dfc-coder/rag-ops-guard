from __future__ import annotations

import os
import subprocess


def run_make(target: str) -> None:
    subprocess.run(["make", target], check=True)


def main() -> None:
    for target in ("models", "local-up", "demo-prepare"):
        run_make(target)
    print("RAG Ops Guard Web UI: http://127.0.0.1:8000", flush=True)
    os.execvp(
        "uv",
        [
            "uv",
            "run",
            "--with",
            "fastapi>=0.116,<1",
            "--with",
            "uvicorn>=0.35,<1",
            "--with",
            "python-multipart>=0.0.20,<1",
            "python",
            "scripts/web_ui.py",
        ],
    )


if __name__ == "__main__":
    main()
