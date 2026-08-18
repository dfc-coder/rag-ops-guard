from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

MIN_CPU_THREADS = 8
MIN_MEMORY_GIB = 8
MIN_FREE_DISK_GIB = 8
REQUIRED_PORTS = (
    ("FLOCI_HOST_PORT", 4566),
    ("LLAMA_GEN_HOST_PORT", 8080),
    ("LLAMA_EMBED_HOST_PORT", 8081),
    ("LLAMA_RERANK_HOST_PORT", 8082),
)
REQUIRED_COMMANDS = ("uv", "node", "npm", "podman", "git")


def gib(value: int) -> float:
    return value / (1024**3)


def physical_memory_bytes() -> int:
    if hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            return pages * page_size
        except (OSError, ValueError):
            pass
    raise RuntimeError("unable to determine physical memory")


def command_version(command: str, *args: str) -> str:
    completed = subprocess.run(
        [command, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip().splitlines()[0]


def command_output(command: str, *args: str) -> str:
    completed = subprocess.run(
        [command, *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return (completed.stdout or completed.stderr).strip()


def assert_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is already in use")


def main() -> None:
    failures: list[str] = []

    if not ((3, 12) <= sys.version_info[:2] < (3, 14)):
        failures.append(f"Python 3.12 or 3.13 required; found {sys.version.split()[0]}")
    print(f"python: {sys.version.split()[0]}")

    for command in REQUIRED_COMMANDS:
        if shutil.which(command) is None:
            failures.append(f"missing command: {command}")
            continue
        try:
            print(f"{command}: {command_version(command, '--version')}")
        except subprocess.CalledProcessError as exc:
            failures.append(f"{command} failed its version check: {exc}")

    if shutil.which("podman") is not None:
        try:
            print(f"podman_compose: {command_version('podman', 'compose', 'version')}")
        except subprocess.CalledProcessError as exc:
            failures.append(f"podman compose is unavailable: {exc}")
        try:
            rootless = command_output(
                "podman", "info", "--format", "{{.Host.Security.Rootless}}"
            ).casefold()
            if rootless not in {"true", "false"}:
                raise RuntimeError(f"unexpected rootless value: {rootless!r}")
            print(f"podman_rootless: {rootless}")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            failures.append(f"podman info is unavailable: {exc}")

    # The physical runtime intentionally does not depend on the systemd-activated
    # $XDG_RUNTIME_DIR/podman/podman.sock. `make up` creates a project-scoped
    # `podman system service --time=0` socket and verifies /_ping before Floci starts.
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    if not runtime_dir.is_dir():
        failures.append(f"runtime directory does not exist: {runtime_dir}")
    elif not os.access(runtime_dir, os.W_OK | os.X_OK):
        failures.append(f"runtime directory is not writable: {runtime_dir}")
    else:
        print(f"podman_api_runtime_dir: {runtime_dir}")

    cpu_threads = os.cpu_count() or 0
    print(f"cpu_threads: {cpu_threads}")
    if cpu_threads < MIN_CPU_THREADS:
        failures.append(f"at least {MIN_CPU_THREADS} logical CPU threads are required")

    try:
        memory_gib = gib(physical_memory_bytes())
        print(f"memory_gib: {memory_gib:.1f}")
        if memory_gib < MIN_MEMORY_GIB:
            failures.append(f"at least {MIN_MEMORY_GIB} GiB RAM is required")
    except RuntimeError as exc:
        failures.append(str(exc))

    disk_gib = gib(shutil.disk_usage(Path.cwd()).free)
    print(f"free_disk_gib: {disk_gib:.1f}")
    if disk_gib < MIN_FREE_DISK_GIB:
        failures.append(f"at least {MIN_FREE_DISK_GIB} GiB free disk is required")

    for env_name, default_port in REQUIRED_PORTS:
        raw_port = os.environ.get(env_name, str(default_port))
        try:
            port = int(raw_port)
            if not 1 <= port <= 65535:
                raise ValueError
        except ValueError:
            failures.append(f"{env_name} must be a valid TCP port; found {raw_port!r}")
            continue
        try:
            assert_port_available(port)
            print(f"port_{port}: available ({env_name})")
        except RuntimeError as exc:
            failures.append(str(exc))

    if failures:
        print("\ndoctor: FAILED", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("doctor: PASS")


if __name__ == "__main__":
    main()
