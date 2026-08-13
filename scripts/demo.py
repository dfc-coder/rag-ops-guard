from __future__ import annotations

import argparse
import json

from rag_ops_guard.app import query_workflow
from rag_ops_guard.domain.models import QueryContext, QueryRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one real RAG Ops Guard beta query")
    parser.add_argument("question")
    parser.add_argument("--system")
    parser.add_argument("--environment", choices=["production", "staging"])
    args = parser.parse_args()

    request = QueryRequest(
        question=args.question,
        context=QueryContext(system=args.system, environment=args.environment),
    )
    response = query_workflow().invoke(request)
    print(json.dumps(response.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
