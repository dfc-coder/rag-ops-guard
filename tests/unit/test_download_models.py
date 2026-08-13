from __future__ import annotations

import hashlib
import io
import urllib.error
import urllib.request
from pathlib import Path

from scripts import download_models


class FakeResponse:
    def __init__(self, payload: bytes, *, status: int = 200, interrupt: bool = False) -> None:
        self._payload = io.BytesIO(payload)
        self._interrupt = interrupt
        self._reads = 0
        self.status = status
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self, _size: int) -> bytes:
        self._reads += 1
        if self._interrupt and self._reads == 2:
            raise urllib.error.URLError("temporary network failure")
        if self._interrupt and self._reads == 1:
            return self._payload.read(4)
        return self._payload.read()


def model_for(payload: bytes) -> download_models.Model:
    return download_models.Model(
        filename="model.gguf",
        url="https://example.invalid/model.gguf",
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_valid_cached_model_skips_network(tmp_path: Path, monkeypatch) -> None:
    payload = b"already cached"
    model = model_for(payload)
    (tmp_path / model.filename).write_bytes(payload)

    def fail_urlopen(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network should not be used for a validated cache hit")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    download_models.download(model, tmp_path)

    assert (tmp_path / model.filename).read_bytes() == payload


def test_interrupted_download_resumes_from_partial_file(tmp_path: Path, monkeypatch) -> None:
    payload = b"abcdefgh"
    model = model_for(payload)
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, **_kwargs: object) -> FakeResponse:
        requests.append(request)
        if len(requests) == 1:
            return FakeResponse(payload, interrupt=True)
        assert request.headers.get("Range") == "bytes=4-"
        return FakeResponse(payload[4:], status=206)

    monkeypatch.setenv("MODEL_DOWNLOAD_ATTEMPTS", "2")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(download_models.time, "sleep", lambda _seconds: None)

    download_models.download(model, tmp_path)

    assert (tmp_path / model.filename).read_bytes() == payload
    assert not (tmp_path / f"{model.filename}.part").exists()
    assert len(requests) == 2
