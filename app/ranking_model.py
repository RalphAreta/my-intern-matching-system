"""
Supervised ranking model that scores internship suitability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils import ensure_directory, get_model_path


@dataclass
class RankingConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 42


class SupervisedRankingModel:
    """Wrapper around a RandomForest regressor used for ranking."""

    feature_columns: List[str] = [
        "matched_skill_count",
        "missing_skill_count",
        "skill_similarity",
        "cf_score",
        "required_skill_count",
        "preferred_skill_count",
        "freq_score",
        "company_score",
        "title_score",
    ]

    def __init__(self, config: RankingConfig | None = None):
        self.config = config or RankingConfig()
        self.model = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

    def fit(self, training_df: pd.DataFrame) -> None:
        """Train the ranking model."""
        if training_df.empty:
            raise ValueError("Training dataset is empty. Cannot fit ranking model.")
        X = training_df[self.feature_columns]
        y = training_df["label"]
        self.model.fit(X, y)

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores for the provided feature dataframe."""
        if feature_df.empty:
            return np.array([])
        return self.model.predict(feature_df[self.feature_columns])

    def save(self, filename: str = "ranking_model.pkl") -> None:
        """Persist the trained model to disk."""
        path = get_model_path(filename)
        ensure_directory(path.parent)
        joblib.dump(self, path)

    @staticmethod
    def load(filename: str = "ranking_model.pkl") -> "SupervisedRankingModel":
        """Load model from disk."""
        path = get_model_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Ranking model artifact not found: {path}")
        return joblib.load(path)


