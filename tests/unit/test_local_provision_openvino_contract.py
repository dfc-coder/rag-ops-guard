from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_local_infra_deploys_canonical_cdk_and_preserves_openvino_topology() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    block = makefile.split("local-infra: package-lambda", maxsplit=1)[1].split(
        "\nlocal-provision:", maxsplit=1
    )[0]

    assert "RAG_OPS_INFRA_TARGET=local" in block
    assert "$(CDK_LOCAL) deploy $(CDK_LOCAL_STACK)" in block
    assert "scripts/local/provision.py" in block

    stack = (ROOT / "infra/cdk/lib/rag-ops-guard-stack.ts").read_text(encoding="utf-8")
    assert "http://rag-ops-ovms-rag:8000/v3" in stack
    assert "S3_VECTOR_INDEX" in stack


def test_local_provisioner_is_only_a_floci_compatibility_bridge() -> None:
    provision = (ROOT / "scripts/local/provision.py").read_text(encoding="utf-8")

    assert "materialize_floci_s3_vectors" in provision
    assert 'client("cloudformation").describe_stacks' in provision
    assert "create_function" not in provision
    assert "create_api" not in provision
    assert "create_table" not in provision
    assert "create_bucket" not in provision
