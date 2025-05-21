from pathlib import Path
import os
from dotenv import load_dotenv
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Get the project root directory (parent of inc directory)
ROOT_DIR = Path(__file__).parent.parent

# Load environment variables from .env file in project root
env_path = ROOT_DIR / '.env'
if env_path.exists():
    logging.info(f"Found .env file at {env_path.absolute()}")
    load_dotenv(env_path)
else:
    logging.warning(f"No .env file found at {env_path.absolute()}")

# GitHub settings
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
logging.info(f"GITHUB_TOKEN loaded: {'Yes' if GITHUB_TOKEN else 'No'}")

# Gemini API settings
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
logging.info(f"GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'}")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Directory settings
RESULTS_DIR = Path("results")
SAMPLE_SIZE = 1600 # Number of repositories to process