"""
Collaborative Filtering model powered by Apriori association rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd
import joblib
from mlxtend.frequent_patterns import apriori, association_rules

from utils import ensure_directory, get_model_path, normalize_skills, skills_to_vector


@dataclass
class CFConfig:
    min_support: float = 0.05
    min_lift: float = 1.0


class CollaborativeFilteringModel:
    """Encapsulates Apriori-based collaborative filtering logic."""

    def __init__(self, config: CFConfig | None = None):
        self.config = config or CFConfig()
        self.skill_vocabulary: List[str] = []
        self.internship_vectors: np.ndarray | None = None
        self.rules_table: List[dict] = []
        self.skill_frequency: dict[str, float] = {}
        self.internships_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def fit(self, internships_df: pd.DataFrame) -> None:
        """Train Apriori frequent itemsets on internship skill requirements."""
        self.internships_df = internships_df.copy()
        self.internships_df["Required_Skill_List"] = self.internships_df["Required_Skill_List"].apply(normalize_skills)
        self.internships_df["Preferred_Skill_List"] = self.internships_df["Preferred_Skill_List"].apply(normalize_skills)

        self.skill_vocabulary = sorted(
            {
                skill
                for skills in self.internships_df["Required_Skill_List"]
                for skill in skills
            }
        )

        if not self.skill_vocabulary:
            self.internship_vectors = np.array([])
            self.rules_table = []
            self.skill_frequency = {}
            return

        transactions = self._encode_transactions(self.internships_df["Required_Skill_List"])
        frequent_itemsets = apriori(
            transactions,
            min_support=self.config.min_support,
            use_colnames=True,
        )

        if frequent_itemsets.empty:
            self.rules_table = []
        else:
            rules_df = association_rules(
                frequent_itemsets,
                metric="lift",
                min_threshold=self.config.min_lift,
            )
            # Persist lift scores for quick lookups
            self.rules_table = [
                {
                    "antecedent": set(rule["antecedents"]),
                    "consequent": set(rule["consequents"]),
                    "lift": float(rule["lift"]),
                }
                for _, rule in rules_df.iterrows()
            ]

        self.skill_frequency = (
            transactions.sum(axis=0) / max(1, len(transactions))
        ).to_dict()

        self.internship_vectors = np.vstack(
            [skills_to_vector(skills, self.skill_vocabulary) for skills in self.internships_df["Required_Skill_List"]]
        )

    def _encode_transactions(self, transactions: Sequence[Sequence[str]]) -> pd.DataFrame:
        """Binary encode transactions for Apriori consumption."""
        encoded = pd.DataFrame(0, index=range(len(transactions)), columns=self.skill_vocabulary, dtype=int)
        for idx, skills in enumerate(transactions):
            for skill in skills:
                encoded.at[idx, skill] = 1
        return encoded

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------
    def score_internships(self, user_skills: Sequence[str]) -> List[dict]:
        """Score all internships for the provided user skill list."""
        if self.internships_df is None or self.internship_vectors is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        user_vector = skills_to_vector(user_skills, self.skill_vocabulary)
        user_set = set(user_skills)

        results: List[dict] = []
        for idx, row in self.internships_df.iterrows():
            req_skills = row["Required_Skill_List"]
            preferred_skills = row["Preferred_Skill_List"]
            internship_vector = self.internship_vectors[idx]

            cf_score = self._compute_cf_score(user_set, set(req_skills))
            freq_score = self._compute_frequency_score(req_skills)

            matched = sorted(user_set.intersection(req_skills))
            missing = sorted(set(req_skills) - user_set)

            cosine_sim = 0.0
            if np.any(user_vector) and np.any(internship_vector):
                cosine_sim = float(
                    np.dot(user_vector, internship_vector)
                    / (np.linalg.norm(user_vector) * np.linalg.norm(internship_vector))
                )

            results.append(
                {
                    "index": idx,
                    "internship_title": row["Internship_Title"],
                    "company": row["Company_Name"],
                    "location": row.get("Location", "N/A"),
                    "minimum_experience": row.get("Minimum_Experience", "0"),
                    "required_skills": req_skills,
                    "preferred_skills": preferred_skills,
                    "matched_skills": matched,
                    "missing_skills": missing,
                    "required_skill_count": len(req_skills),
                    "preferred_skill_count": len(preferred_skills),
                    "cf_score_raw": cf_score if cf_score > 0 else len(matched),
                    "freq_score": freq_score,
                    "cosine_similarity": cosine_sim,
                }
            )
        return results

    def _compute_cf_score(self, user_set: set, internship_set: set) -> float:
        """Sum lifts of applicable association rules."""
        if not self.rules_table:
            return 0.0
        score = 0.0
        for rule in self.rules_table:
            if rule["antecedent"].issubset(user_set) and rule["consequent"].issubset(internship_set):
                score += rule["lift"]
        return score

    def _compute_frequency_score(self, skills: Sequence[str]) -> float:
        """Average frequency of skills within the dataset."""
        if not skills:
            return 0.0
        return float(
            sum(self.skill_frequency.get(skill, 0.0) for skill in skills) / len(skills)
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save(self, filename: str = "cf_apriori.pkl") -> None:
        """Serialize the trained model."""
        path = get_model_path(filename)
        ensure_directory(path.parent)
        joblib.dump(self, path)

    @staticmethod
    def load(filename: str = "cf_apriori.pkl") -> "CollaborativeFilteringModel":
        """Load a serialized model instance."""
        path = get_model_path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        return joblib.load(path)


