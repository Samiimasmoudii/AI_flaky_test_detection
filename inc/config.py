from pathlib import Path
import os

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')  # Get token from environment variable
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {}
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 5