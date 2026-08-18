from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL_DIR = ROOT / ".local"
PID_FILE = LOCAL_DIR / "podman-api.pid"
LOG_FILE = LOCAL_DIR / "podman-api.log"
DEFAULT_SOCKET = Path(
    os.environ.get(
        "PODMAN_API_SOCKET",
        f"/run/user/{os.getuid()}/rag-ops-guard/podman-api.sock",
    )
)


def _pid() -> int | None:
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _ping(path: Path, timeout: float = 0.5) -> bool:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
        client.sendall(
            b"GET /_ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        data = client.recv(4096)
        return b"200 OK" in data and b"OK" in data
    except OSError:
        return False
    finally:
        client.close()


def start(path: Path) -> int:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _ping(path):
        print(f"Podman API bridge: ready ({path})")
        return 0

    previous = _pid()
    if _alive(previous):
        os.killpg(previous, signal.SIGTERM)
        time.sleep(0.2)

    PID_FILE.unlink(missing_ok=True)
    path.unlink(missing_ok=True)

    log = LOG_FILE.open("ab", buffering=0)
    process = subprocess.Popen(
        ["podman", "system", "service", "--time=0", f"unix://{path}"],
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(f"{process.pid}\n", encoding="utf-8")

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"Podman API service exited with code {process.returncode}; see {LOG_FILE}"
            )
        if _ping(path):
            try:
                path.chmod(0o600)
            except OSError:
                pass
            print(f"Podman API bridge: ready ({path})")
            return 0
        time.sleep(0.1)

    os.killpg(process.pid, signal.SIGTERM)
    raise RuntimeError(f"Podman API service did not become ready; see {LOG_FILE}")


def stop(path: Path) -> int:
    pid = _pid()
    if _alive(pid):
        assert pid is not None
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and _alive(pid):
            time.sleep(0.05)
        if _alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    PID_FILE.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    print("Podman API bridge: stopped")
    return 0


def status(path: Path) -> int:
    if _ping(path):
        print(f"Podman API bridge: ready ({path})")
        return 0
    print(f"Podman API bridge: unavailable ({path})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage the project-scoped rootless Podman API Unix socket used by Floci."
    )
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    args = parser.parse_args()

    if args.action == "start":
        return start(args.socket)
    if args.action == "stop":
        return stop(args.socket)
    return status(args.socket)


if __name__ == "__main__":
    raise SystemExit(main())
