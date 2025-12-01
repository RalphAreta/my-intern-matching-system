"""
Collaborative Filtering Recommendation Engine for Internship Matching.
Uses Cosine Similarity to match user skills with internship requirements.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    parse_user_skills,
    extract_skills_from_internship,
    calculate_match_percentage,
    get_matched_skills,
    get_missing_skills
)


class InternshipRecommender:
    """
    Main recommendation engine class that handles skill-based matching
    using collaborative filtering with cosine similarity.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize the recommender with internship dataset.
        
        Args:
            dataset_path (str): Path to the internship requirements CSV file
        """
        self.dataset_path = dataset_path
        self.internships_df = None
        self.load_dataset()
    
    def load_dataset(self):
        """
        Load the internship requirements dataset from CSV.
        """
        try:
            self.internships_df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(self.internships_df)} internships from dataset.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        except Exception as e:
            raise Exception(f"Error loading dataset: {str(e)}")
    
    def get_all_unique_skills(self):
        """
        Extract all unique skills from all internships in the dataset.
        
        Returns:
            list: Sorted list of all unique skills
        """
        all_skills = set()
        
        for idx, row in self.internships_df.iterrows():
            required_skills = extract_skills_from_internship(row['Required_Skills'])
            all_skills.update(required_skills)
        
        return sorted(list(all_skills))
    
    def create_skill_vector(self, skills_list, skill_vocabulary):
        """
        Create a binary vector representation of skills.
        
        Args:
            skills_list (list): List of skills to convert to vector
            skill_vocabulary (list): Complete vocabulary of all possible skills
            
        Returns:
            numpy.ndarray: Binary vector where 1 indicates skill presence
        """
        vector = np.zeros(len(skill_vocabulary))
        skill_set = set(skills_list)
        
        for i, skill in enumerate(skill_vocabulary):
            if skill in skill_set:
                vector[i] = 1
        
        return vector
    
    def recommend(self, user_skills_input, top_n=5):
        """
        Recommend top N internships based on user skills using cosine similarity.
        
        Args:
            user_skills_input (str): Comma-separated string of user skills
            top_n (int): Number of top recommendations to return (default: 5)
            
        Returns:
            list: List of dictionaries containing recommendation details
        """
        # Parse and normalize user skills
        user_skills = parse_user_skills(user_skills_input)
        
        if not user_skills:
            return []
        
        # Get all unique skills from dataset to create vocabulary
        skill_vocabulary = self.get_all_unique_skills()
        
        if not skill_vocabulary:
            return []
        
        # Create user skill vector
        user_vector = self.create_skill_vector(user_skills, skill_vocabulary)
        user_vector = user_vector.reshape(1, -1)
        
        # Calculate similarity for each internship
        recommendations = []
        
        for idx, row in self.internships_df.iterrows():
            # Extract required skills for this internship
            required_skills = extract_skills_from_internship(row['Required_Skills'])
            
            if not required_skills:
                continue
            
            # Create internship skill vector
            internship_vector = self.create_skill_vector(required_skills, skill_vocabulary)
            internship_vector = internship_vector.reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(user_vector, internship_vector)[0][0]
            
            # Calculate matched and missing skills
            matched_skills = get_matched_skills(user_skills, required_skills)
            missing_skills = get_missing_skills(user_skills, required_skills)
            
            # Create recommendation dictionary
            recommendation = {
                'internship_title': row['Internship_Title'],
                'company': row['Company_Name'],
                'location': row.get('Location', 'N/A'),
                'similarity_score': similarity,
                'match_percentage': calculate_match_percentage(similarity),
                'required_skills': required_skills,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'minimum_experience': row.get('Minimum_Experience', 'N/A')
            }
            
            recommendations.append(recommendation)
        
        # Sort by similarity score (descending) and return top N
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return recommendations[:top_n]

