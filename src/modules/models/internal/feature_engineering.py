"""
Feature Engineering for Diabetes Contextual Bandits
Copied from ML project — import paths updated for FastAPI integration.

Only change: src.data_generator → src.modules.models.internal.constants
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import logging
logger = logging.getLogger(__name__)

from src.modules.models.internal.constants import (
    CONTEXT_FEATURES, CONTINUOUS_FEATURES, BINARY_FEATURES,
    INTERACTION_FEATURES, ALL_FEATURES, N_TREATMENTS, TREATMENT_TO_IDX,
)


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["age_x_ckd"] = out["age"] * out["ckd"]
    out["bmi_x_nafld"] = out["bmi"] * out["nafld"]
    out["cvd_x_egfr"] = out["cvd"] * out["egfr"]
    out["cpeptide_x_hba1c"] = out["c_peptide"] * out["hba1c_baseline"]
    out["duration_x_hba1c"] = out["diabetes_duration"] * out["hba1c_baseline"]
    out["fg_x_bmi"] = out["fasting_glucose"] * out["bmi"] / 100.0
    out["tg_hdl_ratio"] = out["triglycerides"] / out["hdl"].clip(lower=1.0)
    out["renal_risk"] = (
        (out["egfr"] < 60).astype(float)
        + (out["egfr"] < 30).astype(float)
        + out["ckd"]
        + out["hypertension"]
    )
    out["severity_score"] = (
        (out["hba1c_baseline"] - 7.0).clip(lower=0) / 3.0
        + (1.0 - out["c_peptide"].clip(upper=2.0) / 2.0)
        + out["diabetes_duration"] / 30.0
    )
    return out


class FeaturePipeline:
    def __init__(self, scale: bool = True, add_interactions: bool = True, features: Optional[List[str]] = None):
        self.scale = scale
        self.add_interactions = add_interactions
        self.features = features or ALL_FEATURES
        self.scaler: Optional[StandardScaler] = None
        self._continuous_idx: Optional[List[int]] = None
        self._fitted = False

    def _get_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.add_interactions:
            df = compute_interaction_features(df)
        missing = [f for f in self.features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing features in dataframe: {missing}")
        return df[self.features].copy()

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        feat_df = self._get_feature_matrix(df)
        if self.scale:
            self._continuous_idx = [
                i for i, f in enumerate(self.features) if f in CONTINUOUS_FEATURES + INTERACTION_FEATURES
            ]
            self.scaler = StandardScaler()
            self.scaler.fit(feat_df.iloc[:, self._continuous_idx])
        self._fitted = True
        logger.info(f"FeaturePipeline fitted: {len(self.features)} features, scale={self.scale}")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        feat_df = self._get_feature_matrix(df)
        X = feat_df.values.astype(np.float64)
        if self.scale and self.scaler is not None:
            X[:, self._continuous_idx] = self.scaler.transform(X[:, self._continuous_idx])
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)

    def transform_single(self, context: Dict) -> np.ndarray:
        row_df = pd.DataFrame([context])
        return self.transform(row_df).flatten()

    def save(self, path: str = "models/feature_pipeline.joblib") -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted pipeline.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "scale": self.scale, "add_interactions": self.add_interactions,
            "features": self.features, "scaler": self.scaler,
            "_continuous_idx": self._continuous_idx, "_fitted": self._fitted,
        }
        joblib.dump(state, path)
        logger.info(f"FeaturePipeline saved to {path}")

    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        state = joblib.load(path)
        pipe = cls(scale=state["scale"], add_interactions=state["add_interactions"], features=state["features"])
        pipe.scaler = state["scaler"]
        pipe._continuous_idx = state["_continuous_idx"]
        pipe._fitted = state["_fitted"]
        logger.info(f"FeaturePipeline loaded from {path}: {len(pipe.features)} features, scale={pipe.scale}")
        return pipe