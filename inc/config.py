from pathlib import Path
import os

# GitHub settings
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is required")
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}

# Gemini API settings
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Directory settings
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 5  # Number of repositories to process