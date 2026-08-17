from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CHUNK_SIZE = 8 * 1024 * 1024
DEFAULT_ATTEMPTS = 5
DEFAULT_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Model:
    filename: str
    url: str
    sha256: str


MODELS = (
    Model(
        filename="Qwen3.5-0.8B-UD-Q4_K_XL.gguf",
        url=(
            "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/"
            "Qwen3.5-0.8B-UD-Q4_K_XL.gguf?download=true"
        ),
        sha256="3177ebd67afe4438374da19e690bc1b98756f7e0fea9240e1be404336156a7b5",
    ),
    Model(
        filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
        url=(
            "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/"
            "Qwen3-Embedding-0.6B-Q8_0.gguf?download=true"
        ),
        sha256="06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
    ),
    Model(
        filename="Qwen3-Reranker-0.6B-Q4_K_M.gguf",
        url=(
            "https://huggingface.co/Voodisss/"
            "Qwen3-Reranker-0.6B-GGUF-llama_cpp/resolve/main/"
            "Qwen3-Reranker-0.6B-Q4_K_M.gguf?download=true"
        ),
        sha256="c04f5f5657c52e04538c455e8c62817db3d3b795b39e9f547f8581510445f075",
    ),
)


class DownloadIntegrityError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def default_model_dir() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "rag-ops-guard" / "models"


def selected_models() -> tuple[Model, ...]:
    requested = {
        item.strip() for item in os.environ.get("MODEL_FILES", "").split(",") if item.strip()
    }
    if not requested:
        return MODELS

    available = {model.filename: model for model in MODELS}
    unknown = requested.difference(available)
    if unknown:
        raise SystemExit(f"unknown MODEL_FILES entries: {', '.join(sorted(unknown))}")
    return tuple(model for model in MODELS if model.filename in requested)


def _download_once(model: Model, temporary: Path, timeout_seconds: int) -> None:
    offset = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "rag-ops-guard/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    request = urllib.request.Request(model.url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = getattr(response, "status", response.getcode())
        resume = offset > 0 and status == 206
        if not resume:
            offset = 0

        content_length = int(response.headers.get("Content-Length", "0"))
        total = offset + content_length if content_length else 0
        received = offset
        mode = "ab" if resume else "wb"

        with temporary.open(mode) as handle:
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if total:
                    print(f"  {received / total:.1%}", end="\r", flush=True)
    print()


def download(model: Model, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / model.filename
    temporary = target.with_suffix(target.suffix + ".part")

    if target.exists() and sha256(target) == model.sha256:
        print(f"verified cached {model.filename}")
        return
    target.unlink(missing_ok=True)

    if temporary.exists() and sha256(temporary) == model.sha256:
        temporary.replace(target)
        print(f"verified resumed {model.filename}")
        return

    attempts = int(os.environ.get("MODEL_DOWNLOAD_ATTEMPTS", str(DEFAULT_ATTEMPTS)))
    timeout_seconds = int(os.environ.get("MODEL_DOWNLOAD_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS)))

    for attempt in range(1, attempts + 1):
        partial_size = temporary.stat().st_size if temporary.exists() else 0
        action = "resuming" if partial_size else "downloading"
        print(f"{action} {model.filename} (attempt {attempt}/{attempts})")
        try:
            _download_once(model, temporary, timeout_seconds)
            actual = sha256(temporary)
            if actual != model.sha256:
                temporary.unlink(missing_ok=True)
                raise DownloadIntegrityError(
                    f"SHA256 mismatch for {model.filename}: expected {model.sha256}, got {actual}"
                )
            temporary.replace(target)
            print(f"verified {model.filename}")
            return
        except urllib.error.HTTPError as error:
            if error.code == 416:
                temporary.unlink(missing_ok=True)
            failure: Exception = error
        except (urllib.error.URLError, OSError, DownloadIntegrityError) as error:
            failure = error

        if attempt == attempts:
            raise RuntimeError(
                f"failed to download {model.filename} after {attempts} attempts"
            ) from failure

        delay = min(2 ** (attempt - 1), 16)
        print(f"download interrupted: {failure}; retrying in {delay}s")
        time.sleep(delay)


def main() -> int:
    directory = Path(os.environ.get("MODEL_DIR", default_model_dir())).expanduser().resolve()
    print(f"model cache: {directory}")
    for model in selected_models():
        download(model, directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
