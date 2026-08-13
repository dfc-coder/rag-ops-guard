from __future__ import annotations

from pathlib import Path

from rag_ops_guard.app import ingestion_service


def main() -> None:
    service = ingestion_service()
    for path in sorted(Path("knowledge-base").rglob("*.md")):
        key = f"raw/{path.relative_to('knowledge-base').as_posix()}"
        response = service.ingest(key)
        print(f"{key}: {response.status} ({response.chunks} chunks)")


if __name__ == "__main__":
    main()
