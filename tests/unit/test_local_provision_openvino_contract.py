from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_local_provision_preserves_physical_openvino_backend() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    block = makefile.split("local-provision: package-lambda", maxsplit=1)[1].split(
        "\nseed:", maxsplit=1
    )[0]

    assert "CONFIG_SOURCE=db" in block
    assert "$(OPENVINO_BACKEND_ENV)" in block
    assert "$(LAMBDA_OPENVINO_ENV)" in block
