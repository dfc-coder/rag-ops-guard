from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOCI_VERSION = "1.6.0"
FLOCI_SOURCE_COMMIT = "e0aab2e27d896772847517a29cd4025203ddc4f8"
FLOCI_REPOSITORY = "https://github.com/floci-io/floci.git"
DEFAULT_IMAGE = f"localhost/rag-ops-floci-jvm:{FLOCI_VERSION}"


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd or ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def image_matches(image: str) -> bool:
    result = _run(
        [
            "podman",
            "image",
            "inspect",
            image,
            "--format",
            '{{ index .Labels "org.opencontainers.image.revision" }}',
        ]
    )
    return result.returncode == 0 and result.stdout.strip() == FLOCI_SOURCE_COMMIT


def ensure(image: str) -> int:
    if image_matches(image):
        print(f"Floci JVM image: ready ({image} @ {FLOCI_SOURCE_COMMIT[:8]})")
        return 0

    print(
        f"Floci JVM image: building {image} from upstream {FLOCI_VERSION} "
        f"({FLOCI_SOURCE_COMMIT[:8]})"
    )
    with tempfile.TemporaryDirectory(prefix="rag-ops-floci-") as raw:
        checkout = Path(raw) / "floci"
        clone = _run(["git", "init", str(checkout)])
        if clone.returncode != 0:
            raise RuntimeError(clone.stdout.strip() or "git init failed")

        remote = _run(["git", "remote", "add", "origin", FLOCI_REPOSITORY], cwd=checkout)
        if remote.returncode != 0:
            raise RuntimeError(remote.stdout.strip() or "git remote add failed")

        fetch = _run(
            ["git", "fetch", "--depth=1", "origin", FLOCI_SOURCE_COMMIT],
            cwd=checkout,
        )
        if fetch.returncode != 0:
            raise RuntimeError(fetch.stdout.strip() or "git fetch Floci source failed")

        checkout_result = _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
        if checkout_result.returncode != 0:
            raise RuntimeError(checkout_result.stdout.strip() or "git checkout failed")

        actual = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
        if actual.returncode != 0 or actual.stdout.strip() != FLOCI_SOURCE_COMMIT:
            raise RuntimeError("Floci source commit verification failed")

        build = subprocess.run(
            [
                "podman",
                "build",
                "--pull=missing",
                "--build-arg",
                f"VERSION={FLOCI_VERSION}",
                "--label",
                f"org.opencontainers.image.version={FLOCI_VERSION}",
                "--label",
                f"org.opencontainers.image.revision={FLOCI_SOURCE_COMMIT}",
                "--label",
                f"org.opencontainers.image.source={FLOCI_REPOSITORY}",
                "-t",
                image,
                "-f",
                "docker/Dockerfile",
                ".",
            ],
            cwd=checkout,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(f"Floci JVM image build failed with exit code {build.returncode}")

    if not image_matches(image):
        raise RuntimeError("Floci JVM image provenance verification failed after build")
    print(f"Floci JVM image: ready ({image} @ {FLOCI_SOURCE_COMMIT[:8]})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/verify the pinned Floci JVM image.")
    parser.add_argument("action", choices=("ensure", "status"))
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()

    if args.action == "ensure":
        return ensure(args.image)
    if image_matches(args.image):
        print(f"Floci JVM image: ready ({args.image} @ {FLOCI_SOURCE_COMMIT[:8]})")
        return 0
    print(f"Floci JVM image: missing or wrong provenance ({args.image})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
