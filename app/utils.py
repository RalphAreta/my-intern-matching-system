"""
Shared utilities for data preparation, vector math, and filesystem helpers.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_project_root() -> Path:
    """Return the absolute project root path."""
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS"))
    return PROJECT_ROOT


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not exist and return the Path object."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_dataset_path(filename: str) -> Path:
    """Resolve dataset paths, preferring the working directory if available."""
    cwd_candidate = Path.cwd() / "datasets" / filename
    if cwd_candidate.exists():
        return cwd_candidate
    return get_project_root() / "datasets" / filename


def get_model_path(filename: str) -> Path:
    """Resolve model artifact paths relative to the current working directory."""
    models_dir = ensure_directory(Path.cwd() / "models")
    return models_dir / filename


def normalize_skills(skills_string: str | Iterable[str]) -> List[str]:
    """
    Normalize any collection (or comma-separated string) of skills.

    Args:
        skills_string: Comma-separated skills or iterable of skills

    Returns:
        Ordered list of lowercase, de-duplicated skills
    """
    if not skills_string:
        return []

    if isinstance(skills_string, str):
        raw_tokens = [token.strip() for token in skills_string.split(",")]
    else:
        raw_tokens = [str(token).strip() for token in skills_string]

    seen = set()
    normalized = []
    for token in raw_tokens:
        token = token.lower()
        if token and token not in seen:
            seen.add(token)
            normalized.append(token)
    return normalized


def parse_user_skills(user_input: str) -> List[str]:
    """Convert raw user text into a normalized skill list."""
    return normalize_skills(user_input)


def extract_skills_from_internship(skills_str: str) -> List[str]:
    """Normalize skill strings sourced from internship datasets."""
    if pd.isna(skills_str) or not skills_str:
        return []
    return normalize_skills(skills_str)


def calculate_match_percentage(similarity_score: float) -> float:
    """Convert 0-1 similarity into a rounded percentage."""
    similarity_score = similarity_score or 0.0
    return round(similarity_score * 100, 2)


def get_matched_skills(user_skills: Sequence[str], required_skills: Sequence[str]) -> List[str]:
    """Return overlapping skills between user profile and internship."""
    return sorted(set(user_skills).intersection(required_skills))


def get_missing_skills(user_skills: Sequence[str], required_skills: Sequence[str]) -> List[str]:
    """Return required skills absent from the user profile."""
    return sorted(set(required_skills) - set(user_skills))


def safe_divide(numerator: float, denominator: float) -> float:
    """Guarded division to avoid ZeroDivisionError."""
    return float(numerator) / float(denominator) if denominator else 0.0


def skills_to_vector(skills: Sequence[str], vocabulary: Sequence[str]) -> np.ndarray:
    """Encode a skill list into a binary numpy vector using the given vocabulary."""
    vector = np.zeros(len(vocabulary), dtype=float)
    skill_set = set(skills)
    for idx, skill in enumerate(vocabulary):
        if skill in skill_set:
            vector[idx] = 1.0
    return vector


def cosine_score(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    if not np.any(vec_a) or not np.any(vec_b):
        return 0.0
    return float(cosine_similarity(vec_a.reshape(1, -1), vec_b.reshape(1, -1))[0][0])

