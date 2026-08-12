from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str


MODELS = (
    Model(
        filename="Qwen3-4B-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/"
            "Qwen3-4B-Q4_K_M.gguf?download=true"
        ),
        sha256="7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
    ),
    Model(
        filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
        url=(
            "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/"
            "Qwen3-Embedding-0.6B-Q8_0.gguf?download=true"
        ),
        sha256="06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(model: Model, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / model.filename
    if target.exists() and sha256(target) == model.sha256:
        print(f"verified {model.filename}")
        return
    if target.exists():
        target.unlink()

    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(model.url, headers={"User-Agent": "rag-ops-guard/0.1"})
    print(f"downloading {model.filename}")
    with urllib.request.urlopen(request) as response, temporary.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            received += len(chunk)
            if total:
                print(f"  {received / total:.1%}", end="\r")
    print()
    temporary.replace(target)
    actual = sha256(target)
    if actual != model.sha256:
        target.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA256 mismatch for {model.filename}: expected {model.sha256}, got {actual}"
        )
    print(f"verified {model.filename}")


def main() -> int:
    directory = Path(os.environ.get("MODEL_DIR", ".models"))
    for model in MODELS:
        download(model, directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
