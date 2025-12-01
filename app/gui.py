"""
Tkinter GUI for the Internship Recommendation System.
Provides a user-friendly interface for skill input and recommendation display.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import sys

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommender import InternshipRecommender


class InternshipRecommendationGUI:
    """
    Main GUI class for the Internship Recommendation System.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Internship Recommendation System")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Initialize recommender
        # Get the project root directory (parent of app directory)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(project_root, 'datasets', 'internship_requirements_1000.csv')
        try:
            self.recommender = InternshipRecommender(dataset_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")
            self.recommender = None
        
        self.setup_ui()
    
    def setup_ui(self):
        """
        Set up the user interface components.
        """
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Internship Recommendation System",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Skills input section
        skills_label = ttk.Label(
            main_frame,
            text="Enter your skills (comma-separated):",
            font=("Arial", 10)
        )
        skills_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 5))
        
        # Skills text entry
        self.skills_entry = ttk.Entry(main_frame, width=80, font=("Arial", 10))
        self.skills_entry.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.skills_entry.insert(0, "e.g., Python, JavaScript, React, SQL, Flask")
        
        # Bind focus events to clear placeholder
        self.skills_entry.bind("<FocusIn>", self.clear_placeholder)
        self.skills_entry.bind("<FocusOut>", self.restore_placeholder)
        
        # Recommend button
        self.recommend_button = ttk.Button(
            main_frame,
            text="Get Recommendations",
            command=self.get_recommendations,
            width=30
        )
        self.recommend_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Results section
        results_label = ttk.Label(
            main_frame,
            text="Recommendations:",
            font=("Arial", 12, "bold")
        )
        results_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(20, 5))
        
        # Scrollable results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Canvas and scrollbar for results
        canvas = tk.Canvas(results_frame, bg="white")
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        self.canvas = canvas
        
        # Bind mousewheel to canvas
        canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def clear_placeholder(self, event):
        """
        Clear placeholder text when entry is focused.
        """
        if self.skills_entry.get() == "e.g., Python, JavaScript, React, SQL, Flask":
            self.skills_entry.delete(0, tk.END)
    
    def restore_placeholder(self, event):
        """
        Restore placeholder text if entry is empty.
        """
        if not self.skills_entry.get():
            self.skills_entry.insert(0, "e.g., Python, JavaScript, React, SQL, Flask")
    
    def _on_mousewheel(self, event):
        """
        Handle mousewheel scrolling for canvas.
        """
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def get_recommendations(self):
        """
        Get recommendations based on user input and display them.
        """
        # Clear previous results
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        # Get user input
        user_skills = self.skills_entry.get().strip()
        
        # Validate input
        if not user_skills or user_skills == "e.g., Python, JavaScript, React, SQL, Flask":
            messagebox.showwarning("Warning", "Please enter your skills before getting recommendations.")
            return
        
        if not self.recommender:
            messagebox.showerror("Error", "Recommendation engine not initialized.")
            return
        
        try:
            # Get recommendations
            recommendations = self.recommender.recommend(user_skills, top_n=5)
            
            if not recommendations:
                no_results_label = ttk.Label(
                    self.scrollable_frame,
                    text="No recommendations found. Please check your skills input.",
                    font=("Arial", 10),
                    foreground="red"
                )
                no_results_label.pack(pady=20)
                return
            
            # Display recommendations
            for idx, rec in enumerate(recommendations, 1):
                self.display_recommendation(rec, idx)
        
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
    
    def display_recommendation(self, recommendation, rank):
        """
        Display a single recommendation in the GUI.
        
        Args:
            recommendation (dict): Recommendation dictionary
            rank (int): Rank of the recommendation (1-5)
        """
        # Main frame for each recommendation
        rec_frame = ttk.LabelFrame(
            self.scrollable_frame,
            text=f"#{rank} - {recommendation['internship_title']}",
            padding="10"
        )
        rec_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Company and Location
        company_label = ttk.Label(
            rec_frame,
            text=f"Company: {recommendation['company']} | Location: {recommendation['location']}",
            font=("Arial", 10, "bold")
        )
        company_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Match Score
        match_color = self.get_match_color(recommendation['match_percentage'])
        match_label = ttk.Label(
            rec_frame,
            text=f"Match Score: {recommendation['match_percentage']}%",
            font=("Arial", 11, "bold"),
            foreground=match_color
        )
        match_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Separator
        ttk.Separator(rec_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Required Skills
        req_skills_frame = ttk.Frame(rec_frame)
        req_skills_frame.pack(fill=tk.X, pady=5)
        
        req_label = ttk.Label(
            req_skills_frame,
            text="Required Skills:",
            font=("Arial", 9, "bold")
        )
        req_label.pack(anchor=tk.W)
        
        req_skills_text = ", ".join(recommendation['required_skills'])
        req_skills_display = ttk.Label(
            req_skills_frame,
            text=req_skills_text,
            font=("Arial", 9),
            wraplength=800
        )
        req_skills_display.pack(anchor=tk.W, padx=(20, 0))
        
        # Matched Skills
        matched_frame = ttk.Frame(rec_frame)
        matched_frame.pack(fill=tk.X, pady=5)
        
        matched_label = ttk.Label(
            matched_frame,
            text="✓ Matched Skills:",
            font=("Arial", 9, "bold"),
            foreground="green"
        )
        matched_label.pack(anchor=tk.W)
        
        if recommendation['matched_skills']:
            matched_skills_text = ", ".join(recommendation['matched_skills'])
            matched_skills_display = ttk.Label(
                matched_frame,
                text=matched_skills_text,
                font=("Arial", 9),
                foreground="green",
                wraplength=800
            )
            matched_skills_display.pack(anchor=tk.W, padx=(20, 0))
        else:
            no_match_label = ttk.Label(
                matched_frame,
                text="None",
                font=("Arial", 9),
                foreground="gray"
            )
            no_match_label.pack(anchor=tk.W, padx=(20, 0))
        
        # Missing Skills
        missing_frame = ttk.Frame(rec_frame)
        missing_frame.pack(fill=tk.X, pady=5)
        
        missing_label = ttk.Label(
            missing_frame,
            text="✗ Missing Skills:",
            font=("Arial", 9, "bold"),
            foreground="red"
        )
        missing_label.pack(anchor=tk.W)
        
        if recommendation['missing_skills']:
            missing_skills_text = ", ".join(recommendation['missing_skills'])
            missing_skills_display = ttk.Label(
                missing_frame,
                text=missing_skills_text,
                font=("Arial", 9),
                foreground="red",
                wraplength=800
            )
            missing_skills_display.pack(anchor=tk.W, padx=(20, 0))
        else:
            complete_label = ttk.Label(
                missing_frame,
                text="None - You have all required skills!",
                font=("Arial", 9),
                foreground="green"
            )
            complete_label.pack(anchor=tk.W, padx=(20, 0))
        
        # Minimum Experience
        exp_frame = ttk.Frame(rec_frame)
        exp_frame.pack(fill=tk.X, pady=(5, 0))
        
        exp_label = ttk.Label(
            exp_frame,
            text=f"Minimum Experience: {recommendation['minimum_experience']} year(s)",
            font=("Arial", 9, "italic")
        )
        exp_label.pack(anchor=tk.W)
    
    def get_match_color(self, percentage):
        """
        Get color based on match percentage.
        
        Args:
            percentage (float): Match percentage
            
        Returns:
            str: Color name
        """
        if percentage >= 70:
            return "green"
        elif percentage >= 40:
            return "orange"
        else:
            return "red"

