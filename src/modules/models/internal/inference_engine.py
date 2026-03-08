import logging
from typing import Dict

from src.configs import gemini as gemini_config
from src.modules.models.internal.model_loader import registry
from src.modules.models.internal.llm_explainer import LLMExplainer

logger = logging.getLogger(__name__)


def predict(context: Dict, model_name: str = "default") -> Dict:
    """
    Run prediction for a single patient using a specific model.

    Args:
        context: dict with the 16 clinical features
        model_name: which loaded model to use

    Returns:
        Extraction payload with keys: patient, decision, safety, fairness
    """
    bundle = registry.get(model_name)

    x = bundle.pipeline.transform_single(context)
    payload = bundle.extractor.extract(context, x)

    logger.info(
        "Prediction [%s]: %s (confidence: %d%%, safety: %s)",
        model_name,
        payload["decision"]["recommended_treatment"],
        payload["decision"]["confidence_pct"],
        payload["safety"]["status"],
    )
    return payload


def predict_with_explanation(context: Dict, model_name: str = "default") -> Dict:
    """Run prediction + LLM explanation for a single patient."""
    payload = predict(context, model_name)
    payload["explanation"] = explain(payload)
    return payload


def explain(payload: Dict) -> Dict:
    """Generate LLM explanation from an existing prediction payload."""
    explainer = LLMExplainer(
        api_key=gemini_config.api_key,
        model_name=gemini_config.model_name,
        temperature=gemini_config.temperature,
    )
    explanation = explainer.explain(payload)
    logger.info(
        "Explanation generated for %s",
        payload["decision"]["recommended_treatment"],
    )
    return explanation


def predict_batch(contexts: list[Dict], model_name: str = "default") -> list[Dict]:
    """Run predictions for multiple patients."""
    bundle = registry.get(model_name)

    payloads = []
    for ctx in contexts:
        x = bundle.pipeline.transform_single(ctx)
        payload = bundle.extractor.extract(ctx, x)
        payloads.append(payload)

    logger.info("Batch prediction [%s]: %d patients", model_name, len(payloads))
    return payloads


def predict_batch_with_explanations(
    contexts: list[Dict], model_name: str = "default"
) -> list[Dict]:
    """Run predictions + LLM explanations for multiple patients."""
    payloads = predict_batch(contexts, model_name)

    explainer = LLMExplainer(
        api_key=gemini_config.api_key,
        model_name=gemini_config.model_name,
        temperature=gemini_config.temperature,
    )
    explanations = explainer.explain_batch(payloads)

    for payload, explanation in zip(payloads, explanations):
        payload["explanation"] = explanation

    logger.info("Batch + explain [%s]: %d patients", model_name, len(payloads))
    return payloads