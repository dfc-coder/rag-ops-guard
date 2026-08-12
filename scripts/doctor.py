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
REQUIRED_PORTS = (4566, 8080, 8081)
REQUIRED_COMMANDS = ("uv", "node", "npm", "docker", "git")


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
            args = ("compose", "version") if command == "docker" else ("--version",)
            print(f"{command}: {command_version(command, *args)}")
        except subprocess.CalledProcessError as exc:
            failures.append(f"{command} failed its version check: {exc}")

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

    for port in REQUIRED_PORTS:
        try:
            assert_port_available(port)
            print(f"port_{port}: available")
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
