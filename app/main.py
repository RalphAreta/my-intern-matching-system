"""
Main entry point for the Internship Recommendation System.
Launches the Tkinter GUI application.
"""

import tkinter as tk
import os
import sys

# Add app directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import InternshipRecommendationGUI


def main():
    """
    Main function to launch the application.
    """
    root = tk.Tk()
    app = InternshipRecommendationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

