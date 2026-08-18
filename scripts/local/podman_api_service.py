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


def _owned_process(pid: int | None, path: Path) -> bool:
    if pid is None:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return False
    return "podman system service" in cmdline and f"unix://{path}" in cmdline


def _ping(path: Path, timeout: float = 0.5) -> bool:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
        client.sendall(
            b"GET /_ping HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        response = bytearray()
        while len(response) < 16384:
            chunk = client.recv(4096)
            if not chunk:
                break
            response.extend(chunk)
        return b"200 OK" in response and response.rstrip().endswith(b"OK")
    except OSError:
        return False
    finally:
        client.close()


def _terminate_owned(pid: int | None, path: Path) -> None:
    if not _owned_process(pid, path):
        return
    assert pid is not None
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if not _owned_process(pid, path):
            return
        time.sleep(0.05)
    if _owned_process(pid, path):
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def start(path: Path) -> int:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    if _ping(path):
        try:
            path.chmod(0o660)
        except OSError:
            pass
        print(f"Podman API service: ready ({path})")
        return 0

    previous = _pid()
    _terminate_owned(previous, path)
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
                # Floci's official entrypoint drops to uid 1001 / group 0 and
                # normalizes the mounted socket group. Group rw is therefore
                # intentional; the socket remains inside the user's runtime dir.
                path.chmod(0o660)
            except OSError:
                pass
            print(f"Podman API service: ready ({path})")
            return 0
        time.sleep(0.1)

    _terminate_owned(process.pid, path)
    raise RuntimeError(f"Podman API service did not become ready; see {LOG_FILE}")


def stop(path: Path) -> int:
    _terminate_owned(_pid(), path)
    PID_FILE.unlink(missing_ok=True)
    path.unlink(missing_ok=True)
    print("Podman API service: stopped")
    return 0


def status(path: Path) -> int:
    if _ping(path):
        print(f"Podman API service: ready ({path})")
        return 0
    print(f"Podman API service: unavailable ({path})")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage the project-scoped rootless Podman Docker-compatible API."
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
