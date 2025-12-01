"""
Utility functions for the Internship Recommendation System.
Contains helper functions for data processing and skill normalization.
"""

import pandas as pd


def normalize_skills(skills_string):
    """
    Normalize skills string by converting to lowercase and removing extra spaces.
    
    Args:
        skills_string (str): Comma-separated skills string
        
    Returns:
        list: List of normalized skill strings
    """
    if not skills_string or not isinstance(skills_string, str):
        return []
    
    # Split by comma and clean each skill
    skills = [skill.strip().lower() for skill in skills_string.split(',')]
    # Remove empty strings
    skills = [skill for skill in skills if skill]
    return skills


def parse_user_skills(user_input):
    """
    Parse user input skills from comma-separated text.
    
    Args:
        user_input (str): User input string with comma-separated skills
        
    Returns:
        list: List of normalized skills
    """
    if not user_input:
        return []
    
    return normalize_skills(user_input)


def extract_skills_from_internship(required_skills_str):
    """
    Extract and normalize skills from internship required skills string.
    
    Args:
        required_skills_str (str): Required skills string from CSV
        
    Returns:
        list: List of normalized skills
    """
    if pd.isna(required_skills_str) or not required_skills_str:
        return []
    
    return normalize_skills(str(required_skills_str))


def calculate_match_percentage(similarity_score):
    """
    Convert cosine similarity score (0-1) to percentage (0-100).
    
    Args:
        similarity_score (float): Cosine similarity score between 0 and 1
        
    Returns:
        float: Match percentage between 0 and 100
    """
    return round(similarity_score * 100, 2)


def get_matched_skills(user_skills, required_skills):
    """
    Get the intersection of user skills and required skills.
    
    Args:
        user_skills (list): List of user skills
        required_skills (list): List of required skills
        
    Returns:
        list: List of matched skills
    """
    user_skills_set = set(user_skills)
    required_skills_set = set(required_skills)
    return list(user_skills_set.intersection(required_skills_set))


def get_missing_skills(user_skills, required_skills):
    """
    Get required skills that the user doesn't have.
    
    Args:
        user_skills (list): List of user skills
        required_skills (list): List of required skills
        
    Returns:
        list: List of missing skills
    """
    user_skills_set = set(user_skills)
    required_skills_set = set(required_skills)
    return list(required_skills_set - user_skills_set)

