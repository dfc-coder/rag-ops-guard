from rag_ops_guard.config import Settings
from scripts import download_models


def test_runtime_default_is_qwen3_4b() -> None:
    """SPEC-5.3: the judge is not a second model; runtime itself is Qwen3-4B."""
    assert Settings(_env_file=None).llm_model == "qwen3-4b-rag"


def test_model_manifest_contains_official_qwen3_4b_quant() -> None:
    """SPEC-5.3"""
    models = {model.filename: model for model in download_models.MODELS}
    model = models["Qwen3-4B-Q4_K_M.gguf"]
    assert "Qwen/Qwen3-4B-GGUF" in model.url
    assert model.sha256 == "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"
