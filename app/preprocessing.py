"""
Preprocessing helpers for preparing datasets and engineered features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import pandas as pd

from utils import (
    extract_skills_from_internship,
    normalize_skills,
)


@dataclass
class DatasetBundle:
    """Container that keeps the raw dataframes and derived statistics."""

    internships: pd.DataFrame
    resumes: pd.DataFrame
    company_frequency: Dict[str, float]
    title_frequency: Dict[str, float]


def load_dataset_bundle(internships_path: str, resumes_path: str) -> DatasetBundle:
    """Load CSV datasets and augment with helper statistics."""
    internships_df = pd.read_csv(internships_path).fillna("")
    resumes_df = pd.read_csv(resumes_path).fillna("")

    internships_df["Required_Skill_List"] = internships_df["Required_Skills"].apply(extract_skills_from_internship)
    internships_df["Preferred_Skill_List"] = internships_df["Preferred_Skills"].apply(extract_skills_from_internship)

    # Frequency encodings for categorical metadata
    company_frequency = (
        internships_df["Company_Name"]
        .value_counts(normalize=True)
        .to_dict()
    )
    title_frequency = (
        internships_df["Internship_Title"]
        .value_counts(normalize=True)
        .to_dict()
    )

    return DatasetBundle(
        internships=internships_df,
        resumes=resumes_df,
        company_frequency=company_frequency,
        title_frequency=title_frequency,
    )


def build_training_samples(
    resumes_df: pd.DataFrame,
    score_callback: Callable[[list], list],
    company_frequency: Dict[str, float],
    title_frequency: Dict[str, float],
    max_samples_per_resume: int = 30,
) -> pd.DataFrame:
    """
    Convert resume/internship matches into supervised training samples.

    Args:
        resumes_df: DataFrame of candidate resumes
        score_callback: Callable returning scored internships for the provided skills
        company_frequency: Frequency encoding for company names
        title_frequency: Frequency encoding for internship titles
        max_samples_per_resume: Cap to limit dataset size

    Returns:
        pd.DataFrame ready for model training
    """
    samples = []
    for resume in resumes_df.itertuples():
        user_skills = normalize_skills(resume.Skills)
        if not user_skills:
            continue

        scored = score_callback(user_skills)
        if not scored:
            continue

        scored.sort(key=lambda rec: (rec["cf_score_raw"], rec["cosine_similarity"]), reverse=True)

        limited_records = scored[:max_samples_per_resume]
        for record in limited_records:
            matched_count = len(record["matched_skills"])
            required_count = max(1, record["required_skill_count"])
            coverage_ratio = matched_count / required_count
            label = 1 if coverage_ratio >= 0.6 else 0

            samples.append(
                {
                    "matched_skill_count": matched_count,
                    "missing_skill_count": len(record["missing_skills"]),
                    "skill_similarity": record["cosine_similarity"],
                    "cf_score": record["cf_score_raw"],
                    "required_skill_count": record["required_skill_count"],
                    "preferred_skill_count": record["preferred_skill_count"],
                    "freq_score": record.get("freq_score", 0.0),
                    "company_score": company_frequency.get(record["company"], 0.0),
                    "title_score": title_frequency.get(record["internship_title"], 0.0),
                    "label": label,
                }
            )

    return pd.DataFrame(samples)


