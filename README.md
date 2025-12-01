# Internship Recommendation System (Desktop App)

A Python desktop application built with Tkinter that recommends internship opportunities using collaborative filtering based on skill matching.

## Features

- **Skill-Based Matching**: Enter your skills as comma-separated text
- **Collaborative Filtering**: Uses Cosine Similarity to match your skills with internship requirements
- **Top 5 Recommendations**: Displays the best matching internships ranked by similarity
- **Detailed Breakdown**: Shows:
  - Match Score (percentage)
  - Required Skills
  - Matched Skills (skills you have)
  - Missing Skills (skills you need to learn)
  - Company and Location information
  - Minimum Experience requirements

## Project Structure

```
my-intern-matching-system/
├── app/
│   ├── main.py              # Application entry point
│   ├── gui.py               # Tkinter GUI implementation
│   ├── recommender.py       # Collaborative filtering recommendation engine
│   └── utils.py             # Utility functions for data processing
├── datasets/
│   ├── internship_requirements_1000.csv
│   └── resume_dataset_1000.csv
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── build_exe.bat           # Script to build Windows EXE
```

## Installation

1. **Install Python 3.8 or higher**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

1. **From the project root directory:**
   ```bash
   python app/main.py
   ```

2. **Enter your skills** in the text box (comma-separated), for example:
   ```
   Python, JavaScript, React, SQL, Flask, HTML, CSS
   ```

3. **Click "Get Recommendations"** to see the top 5 matching internships

4. **Review the results** which show:
   - Rank and internship title
   - Company name and location
   - Match score percentage
   - Required skills breakdown
   - Your matched skills (highlighted in green)
   - Missing skills (highlighted in red)

## Algorithm Details

The recommendation system uses **Collaborative Filtering with Cosine Similarity**:

1. **Skill Vectorization**: Converts user skills and internship requirements into binary vectors
2. **Similarity Calculation**: Computes cosine similarity between user skill vector and each internship's required skills vector
3. **Ranking**: Sorts internships by similarity score (highest to lowest)
4. **Skill Analysis**: Identifies matched and missing skills for each recommendation

### Formula
```
Cosine Similarity = (A · B) / (||A|| × ||B||)
```
Where A is the user skill vector and B is the internship requirement vector.

## Building Executable (Windows)

To create a standalone Windows executable:

1. **Install PyInstaller:**
   ```bash
   pip install pyinstaller
   ```

2. **Run the build script:**
   ```bash
   build_exe.bat
   ```

   Or manually:
   ```bash
   pyinstaller --noconsole --onefile app/main.py
   ```

3. **Find the executable** in the `dist/` folder

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- tkinter (usually included with Python)

## Error Handling

The application includes error handling for:
- Empty skills input
- Missing dataset files
- Invalid data formats
- File loading errors

## Notes

- Skills are case-insensitive and normalized (lowercase, trimmed)
- The system matches exact skill names
- Match scores range from 0% to 100%
- Color coding: Green (≥70%), Orange (40-69%), Red (<40%)

## License

This project is provided as-is for educational and demonstration purposes.

