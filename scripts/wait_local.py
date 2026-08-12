from __future__ import annotations

import time

import httpx


SERVICES = {
    "floci": "http://127.0.0.1:4566/",
    "llama-gen": "http://127.0.0.1:8080/health",
    "llama-embed": "http://127.0.0.1:8081/health",
}


def wait_for(name: str, url: str, timeout_seconds: int = 180) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not started"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=3.0)
            if response.status_code < 500:
                print(f"{name}: ready ({response.status_code})")
                return
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            last_error = str(exc)
        time.sleep(2)
    raise SystemExit(f"{name} did not become ready: {last_error}")


def main() -> None:
    for name, url in SERVICES.items():
        wait_for(name, url)


if __name__ == "__main__":
    main()
