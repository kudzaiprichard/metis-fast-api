"""
Backend-side adapter that builds an ``InferenceConfig`` from the app's
existing YAML-driven configs (``src.configs.model`` and ``src.configs.gemini``)
and constructs an ``InferenceEngine`` from it.

``src/shared/inference`` is treated as a frozen third-party library — this
module is the single seam where backend configuration is translated into the
engine's own config shape. Import only from here when wiring the engine into
the FastAPI app.
"""
from __future__ import annotations

from pathlib import Path

from src.configs import model as model_config
from src.configs import gemini as gemini_config
from src.shared.inference import InferenceConfig, InferenceEngine


def build_inference_config() -> InferenceConfig:
    """
    Translate ``src.configs.model`` + ``src.configs.gemini`` into an
    ``InferenceConfig``. The base model directory from ``model.path`` is
    combined with ``model.model_file`` / ``model.pipeline_file`` to form the
    absolute artefact paths the engine loads.
    """
    base_path = Path(model_config.path)
    model_path = base_path / model_config.model_file
    pipeline_path = base_path / model_config.pipeline_file

    api_key = gemini_config.api_key
    llm_enabled = bool(api_key)
    llm_provider = "gemini" if llm_enabled else "none"

    return InferenceConfig.load(
        model_path=model_path,
        pipeline_path=pipeline_path,
        llm_enabled=llm_enabled,
        llm_provider=llm_provider,
        llm_api_key=api_key or None,
        llm_model_name=gemini_config.model_name,
        llm_temperature=gemini_config.temperature,
    )


def build_inference_engine() -> InferenceEngine:
    """Construct a fully-initialised ``InferenceEngine`` from app config."""
    return InferenceEngine.from_config(build_inference_config())


__all__ = ["build_inference_config", "build_inference_engine"]
