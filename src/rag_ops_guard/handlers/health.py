from __future__ import annotations

import json
from typing import Any


def handler(event: dict[str, Any], context: object) -> dict[str, Any]:
    del event, context
    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body": json.dumps({"status": "ok"}),
    }
