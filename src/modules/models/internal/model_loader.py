import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from src.configs import model as model_config
from src.modules.models.internal.neural_bandit import NeuralThompson
from src.modules.models.internal.feature_engineering import FeaturePipeline
from src.modules.models.internal.explainability import ExplainabilityExtractor

logger = logging.getLogger(__name__)

# Static model hyperparameters (must match training configuration)
DEFAULT_HIDDEN_DIMS = [128, 64]
DEFAULT_NOISE_VARIANCE = 0.25


@dataclass
class ModelBundle:
    """All components needed for inference, bundled together."""
    name: str
    pipeline: FeaturePipeline
    model: NeuralThompson
    extractor: ExplainabilityExtractor
    model_path: str = ""
    input_dim: int = 0
    hidden_dims: list = field(default_factory=list)
    noise_variance: float = 0.25


class ModelRegistry:
    """
    In-memory registry of loaded model bundles.

    Models are loaded once and served from memory.
    Supports multiple named versions side by side.
    """

    def __init__(self):
        self._models: dict[str, ModelBundle] = {}

    def load(
        self,
        name: str,
        model_file: str,
        pipeline_file: str,
        base_path: Optional[str] = None,
    ) -> ModelBundle:
        """Load a model bundle into memory under the given name."""
        if name in self._models:
            logger.info("Model '%s' already loaded, skipping", name)
            return self._models[name]

        base_path = base_path or model_config.path

        # Load pipeline first
        pipeline_path = Path(base_path) / pipeline_file
        if not pipeline_path.exists():
            raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
        pipeline = FeaturePipeline.load(str(pipeline_path))
        logger.info("Pipeline loaded: %s", pipeline_path)

        # Derive input_dim from pipeline
        input_dim = len(pipeline.features)
        logger.info("Input dim derived from pipeline: %d", input_dim)

        # Load model
        model_path = Path(base_path) / model_file
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = NeuralThompson(
            input_dim=input_dim,
            hidden_dims=DEFAULT_HIDDEN_DIMS,
            noise_variance=DEFAULT_NOISE_VARIANCE,
        )
        model.load(str(model_path))
        logger.info("Model loaded: %s", model_path)

        # Create extractor
        extractor = ExplainabilityExtractor(model)

        bundle = ModelBundle(
            name=name,
            pipeline=pipeline,
            model=model,
            extractor=extractor,
            model_path=str(model_path),
            input_dim=input_dim,
            hidden_dims=DEFAULT_HIDDEN_DIMS,
            noise_variance=DEFAULT_NOISE_VARIANCE,
        )
        self._models[name] = bundle
        logger.info("Model '%s' registered", name)
        return bundle

    def get(self, name: str) -> ModelBundle:
        """Get a loaded model by name. Raises KeyError if not found."""
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not loaded. Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def clone_fresh(self, name: str, reset_posterior: bool = True) -> NeuralThompson:
        """
        Create a fresh model instance from a registered bundle.

        Args:
            name: registered model name
            reset_posterior: if True, resets posterior to prior (default).
                             if False, keeps the learned posterior from the checkpoint.
        """
        bundle = self.get(name)

        model = NeuralThompson(
            input_dim=bundle.input_dim,
            hidden_dims=DEFAULT_HIDDEN_DIMS,
            noise_variance=DEFAULT_NOISE_VARIANCE,
        )
        model.load(bundle.model_path)

        if reset_posterior:
            model.reset_posterior()

        logger.info(
            "Model cloned from '%s' (posterior %s)",
            name, "reset" if reset_posterior else "preserved",
        )
        return model

    def remove(self, name: str) -> None:
        """Unload a model from memory."""
        if name in self._models:
            del self._models[name]
            logger.info("Model '%s' removed", name)

    def list_models(self) -> list[str]:
        """Return names of all loaded models."""
        return list(self._models.keys())

    def is_loaded(self, name: str) -> bool:
        return name in self._models


# Single global instance
registry = ModelRegistry()