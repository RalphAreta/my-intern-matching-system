"""
High-level orchestration for training and serving internship recommendations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from cf_model import CollaborativeFilteringModel
from preprocessing import DatasetBundle, build_training_samples, load_dataset_bundle
from ranking_model import SupervisedRankingModel
from utils import (
    calculate_match_percentage,
    get_dataset_path,
    get_model_path,
    parse_user_skills,
)


class InternshipRecommendationSystem:
    """Trains, persists, and serves hybrid CF + ranking recommendations."""

    def __init__(
        self,
        internships_file: Path | None = None,
        resumes_file: Path | None = None,
    ):
        self.internships_path = internships_file or get_dataset_path("internship_requirements_1000.csv")
        self.resumes_path = resumes_file or get_dataset_path("resume_dataset_1000.csv")

        self.dataset_bundle: DatasetBundle = load_dataset_bundle(self.internships_path, self.resumes_path)
        self.cf_model: CollaborativeFilteringModel | None = self._try_load_cf()
        self.ranking_model: SupervisedRankingModel | None = self._try_load_ranking()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train_models(self, force_retrain: bool = False) -> dict:
        """Train CF + ranking models. Retrains if artifacts missing or forced."""
        training_summary = {}

        if force_retrain or self.cf_model is None:
            self.cf_model = CollaborativeFilteringModel()
            self.cf_model.fit(self.dataset_bundle.internships)
            self.cf_model.save()
            training_summary["cf_model_trained"] = True
        else:
            training_summary["cf_model_trained"] = False

        if self.cf_model is None:
            raise RuntimeError("Collaborative filtering model unavailable after training attempt.")

        if force_retrain or self.ranking_model is None:
            training_df = build_training_samples(
                resumes_df=self.dataset_bundle.resumes,
                score_callback=self.cf_model.score_internships,
                company_frequency=self.dataset_bundle.company_frequency,
                title_frequency=self.dataset_bundle.title_frequency,
            )
            if training_df.empty:
                raise RuntimeError("Unable to create training samples for ranking model.")

            self.ranking_model = SupervisedRankingModel()
            self.ranking_model.fit(training_df)
            self.ranking_model.save()
            training_summary["ranking_model_trained"] = True
            training_summary["ranking_samples"] = len(training_df)
        else:
            training_summary["ranking_model_trained"] = False

        return training_summary

    def recommend(self, user_input: str, top_n: int = 5) -> List[dict]:
        """Generate ranked internship recommendations for the provided skills."""
        user_skills = parse_user_skills(user_input)
        if not user_skills:
            return []

        if self.cf_model is None or self.ranking_model is None:
            self.train_models(force_retrain=False)

        assert self.cf_model is not None
        assert self.ranking_model is not None

        scored_internships = self.cf_model.score_internships(user_skills)
        if not scored_internships:
            return []

        feature_df = self._build_feature_frame(scored_internships)
        ranking_scores = self.ranking_model.predict(feature_df)
        feature_df["ranking_score"] = ranking_scores

        cf_min, cf_max = feature_df["cf_score_raw"].min(), feature_df["cf_score_raw"].max()
        if cf_max - cf_min > 0:
            feature_df["cf_score_norm"] = (feature_df["cf_score_raw"] - cf_min) / (cf_max - cf_min)
        else:
            feature_df["cf_score_norm"] = 0.0

        feature_df["final_score"] = 0.5 * feature_df["cf_score_norm"] + 0.5 * feature_df["ranking_score"]
        feature_df.sort_values(by="final_score", ascending=False, inplace=True)

        internships_df = self.dataset_bundle.internships
        recommendations = []
        for _, row in feature_df.head(top_n).iterrows():
            internship_row = internships_df.iloc[int(row["index"])]
            recommendation = {
                "internship_title": internship_row["Internship_Title"],
                "company": internship_row["Company_Name"],
                "location": internship_row.get("Location", "N/A"),
                "cf_score": round(row["cf_score_raw"], 3),
                "ranking_score": round(row["ranking_score"], 3),
                "final_score": round(row["final_score"], 3),
                "matched_skills": row["matched_skills"],
                "missing_skills": row["missing_skills"],
                "required_skills": row["required_skills"],
                "preferred_skills": row["preferred_skills"],
                "minimum_experience": internship_row.get("Minimum_Experience", "0"),
                "match_percentage": calculate_match_percentage(row["cosine_similarity"]),
            }
            recommendations.append(recommendation)
        return recommendations

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_feature_frame(self, scored_records: List[dict]) -> pd.DataFrame:
        """Assemble ranking features from scored CF outputs."""
        company_freq = self.dataset_bundle.company_frequency
        title_freq = self.dataset_bundle.title_frequency

        frame = pd.DataFrame(scored_records)
        frame["company_score"] = frame["company"].map(lambda name: company_freq.get(name, 0.0))
        frame["title_score"] = frame["internship_title"].map(lambda title: title_freq.get(title, 0.0))
        frame["matched_skill_count"] = frame["matched_skills"].apply(len)
        frame["missing_skill_count"] = frame["missing_skills"].apply(len)
        frame["skill_similarity"] = frame["cosine_similarity"]
        frame["cf_score"] = frame["cf_score_raw"]
        return frame

    def _try_load_cf(self) -> CollaborativeFilteringModel | None:
        """Load CF model artifact if available."""
        model_path = get_model_path("cf_apriori.pkl")
        if not model_path.exists():
            return None
        return CollaborativeFilteringModel.load("cf_apriori.pkl")

    def _try_load_ranking(self) -> SupervisedRankingModel | None:
        """Load ranking model artifact if available."""
        model_path = get_model_path("ranking_model.pkl")
        if not model_path.exists():
            return None
        return SupervisedRankingModel.load("ranking_model.pkl")

