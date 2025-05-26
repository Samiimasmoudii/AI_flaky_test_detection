import difflib
from pathlib import Path
from .config import RESULTS_DIR
from openai import OpenAI
import google.generativeai as genai
import os
import re
import csv
import time
import logging
from datetime import datetime
from typing import Optional

# Model configurations
OPENAI_MODEL = "o4-mini"
GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"

def setup_logging(log_dir="logs"):
    """Setup logging configuration and save logs to /logs folder"""
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamp and log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'analysis_{timestamp}.log')
    
    # Remove any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging with both file and console handlers
    handlers = [
        logging.FileHandler(log_file, encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
    
    # Set format for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
    
    # Set logging level
    logging.root.setLevel(logging.INFO)
    
    # Log initial messages
    logging.info("=" * 80)
    logging.info("Starting new analysis session")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 80)
    
    return log_file

import evaluate
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

FLAKINESS_TYPES = {
    "": "No specific flakiness type provided",
    "OD": "Order-Dependent flaky tests - Tests that pass/fail depending on the order of execution",
    "OD-Brit": "Order-Dependent Brittle tests - Tests that need to run in a specific order to pass",
    "OD-Vic": "Order-Dependent Victim tests - Tests that fail when run after specific tests",
    "ID": "Implementation-Dependent Tests - Tests that depend on specific implementation details",
    "ID-HtF": "Implementation-Dependent tests that are hard to fix due to complex dependencies",
    "NIO": "Non-Idempotent-Outcome Tests - Tests that pass in the first run but fail in the second",
    "NOD": "Non-Deterministic tests - Tests with non-deterministic outcomes",
    "NDOD": "Non-Deterministic Order-Dependent tests - Tests that fail non-deterministically with different failure rates in different orders",
    "NDOI": "Non-Deterministic Order-Independent tests - Tests that fail non-deterministically with similar failure rates in all orders",
    "UD": "Unknown Dependency tests - Tests that pass and fail in a test suite or in isolation",
    "OSD": "Operating System Dependent tests - Tests that pass and fail depending on the operating system",
    "TZD": "Time Zone Dependent tests - Tests that fail in machines on different time zones"
}

def is_valid_diff(diff_text: str) -> bool:
    """Check if the diff output is valid and non-empty."""
    if not diff_text:
        return False
    lines = diff_text.strip().splitlines()
    return any(line.startswith(('+', '-')) for line in lines)

def run_openai_with_retry(prompt: str, max_retries: int = 3, delay: int = 6) -> Optional[str]:
    """Run OpenAI with retry logic and rate limiting."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    logging.info(f"Attempting OpenAI API call with model {OPENAI_MODEL}")
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Waiting {delay} seconds before retry {attempt + 1}")
                time.sleep(delay)
            
            logging.debug("Sending request to OpenAI API...")
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a Python testing expert. Your task is to fix flaky tests by providing a unified diff format patch."},
                    {"role": "user", "content": prompt}
                ],
                
                max_completion_tokens=20000
            )
            logging.debug("Received response from OpenAI API")
            
            diff_text = response.choices[0].message.content
            
            if not diff_text or not is_valid_diff(diff_text):
                logging.warning(f"Empty or invalid diff received from OpenAI (attempt {attempt + 1})")
                continue
            
            logging.info("Successfully received valid diff from OpenAI")
            return diff_text
            
        except Exception as e:
            logging.error(f"OpenAI API call failed (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"All {max_retries} attempts failed for OpenAI API call")
                raise Exception(f"OpenAI failed after {max_retries} attempts: {str(e)}")
    
    return None

def run_gemini_with_retry(prompt: str, max_retries: int = 3, delay: int = 6) -> Optional[str]:
    """Run Gemini with retry logic and rate limiting."""
    logging.info(f"Attempting Gemini API call with model {GEMINI_MODEL}")
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                logging.info(f"Waiting {delay} seconds before retry {attempt + 1}")
                time.sleep(delay)
            
            logging.debug("Sending request to Gemini API...")
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(prompt)
            logging.debug("Received response from Gemini API")
            
            if not response.text or not is_valid_diff(response.text):
                logging.warning(f"Empty or invalid diff received from Gemini (attempt {attempt + 1})")
                continue
            
            logging.info("Successfully received valid diff from Gemini")
            return response.text
            
        except Exception as e:
            logging.error(f"Gemini API call failed (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logging.error(f"All {max_retries} attempts failed for Gemini API call")
                raise Exception(f"Gemini failed after {max_retries} attempts: {str(e)}")
    
    return None

def get_next_test_number(directory: Path, model_name: str) -> int:
    """Find the next available test number by checking existing folders."""
    # Format base model name (without test number)
    base_name = model_name.lower().replace('-', '_').replace('.', '_')
    
    # Find all existing test folders for this model
    existing_folders = [d.name for d in directory.iterdir() if d.is_dir() and d.name.startswith(base_name)]
    
    if not existing_folders:
        return 0
        
    # Extract test numbers from folder names
    test_numbers = []
    for folder in existing_folders:
        try:
            # Extract the number after "Test"
            test_num = int(folder.split('Test')[-1])
            test_numbers.append(test_num)
        except (ValueError, IndexError):
            continue
    
    # Return next available number
    return max(test_numbers + [-1]) + 1

def format_model_folder_name(model_name: str, test_number: int) -> str:
    """Format the model folder name according to the convention."""
    formatted_name = model_name.lower().replace('-', '_').replace('.', '_')
    return f"{formatted_name}_Test{test_number:02d}"

def save_model_output(filepath: Path, model_name: str, response: str) -> None:
    """Save model output to the appropriate folder structure."""
    try:
        # Get the next available test number
        test_number = get_next_test_number(filepath.parent, model_name)
        
        # Create model-specific output directory with test number
        folder_name = format_model_folder_name(model_name, test_number)
        output_dir = filepath.parent / folder_name
        output_dir.mkdir(exist_ok=True)
        
        # Save the response to a file
        output_file = output_dir / "diff.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response)
        
        logging.info(f"Saved model output to {output_file}")
    except Exception as e:
        logging.error(f"Error saving model output: {e}")

def run_openai_on_file(filepath: str, category: str = "") -> str:
    """Process a file using OpenAI."""
    try:
        logging.info(f"Processing file: {filepath}")
        logging.info(f"Category: {category}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            logging.debug(f"Successfully read file content ({len(content)} characters)")
        
        flakiness_desc = FLAKINESS_TYPES.get(category, FLAKINESS_TYPES[""])
        logging.info(f"Flakiness type: {category} - {flakiness_desc}")
        
        prompt = create_prompt(category, flakiness_desc, content)
        logging.debug(f"Created prompt ({len(prompt)} characters)")
        
        response = run_openai_with_retry(prompt)
        
        if response is None:
            logging.error(f"Failed to get OpenAI response for {os.path.basename(filepath)}")
            return ""
            
        logging.info(f"Successfully processed {os.path.basename(filepath)} with OpenAI")
        logging.debug(f"Response length: {len(response)} characters")
        
        # Save the response
        save_model_output(Path(filepath), OPENAI_MODEL, response)
        
        return response
        
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(filepath)} with OpenAI: {e}")
        return ""

def run_gemini_on_file(filepath: str, category: str = "") -> str:
    """Process a file using Gemini."""
    try:
        logging.info("Configuring Gemini API...")
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment variables")
            return ""
            
        genai.configure(api_key=api_key)
        logging.info("Successfully configured Gemini API")
        
        logging.info(f"Processing file: {filepath}")
        logging.info(f"Category: {category}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            logging.debug(f"Successfully read file content ({len(content)} characters)")
        
        flakiness_desc = FLAKINESS_TYPES.get(category, FLAKINESS_TYPES[""])
        logging.info(f"Flakiness type: {category} - {flakiness_desc}")
        
        prompt = create_prompt(category, flakiness_desc, content)
        logging.debug(f"Created prompt ({len(prompt)} characters)")
        
        response = run_gemini_with_retry(prompt)
        
        if response is None:
            logging.error(f"Failed to get Gemini response for {os.path.basename(filepath)}")
            return ""
            
        logging.info(f"Successfully processed {os.path.basename(filepath)} with Gemini")
        logging.debug(f"Response length: {len(response)} characters")
        
        # Save the response
        save_model_output(Path(filepath), GEMINI_MODEL, response)
        
        return response
        
    except Exception as e:
        logging.error(f"Error processing {os.path.basename(filepath)} with Gemini: {e}")
        return ""

def create_prompt(category: str, flakiness_desc: str, content: str) -> str:
    """Create a standardized prompt for both models."""
    return f"""
    The following test file exhibits flaky behavior categorized as:

Type: {category}
Description: {flakiness_desc}

Step 1: Identify the root cause of the flakiness based on the category.
Step 2: Fix the code using known patterns that address this type of flakiness.
Step 3: Output ONLY the fix as a unified diff.

Here is the test file:
<code>
{content}
</code>

Please provide the fixed version of the code that addresses any flaky test patterns. Output only the diff like this example:
--- before.py
+++ after.py
@@ -10,6 +10,8 @@
   old line
+  new line
   old line"""

# Utility functions for comparison and analysis
def compare_fixes(llm_fix: str, dev_patch: str) -> str:
    d = difflib.unified_diff(dev_patch.splitlines(), llm_fix.splitlines(), lineterm="")
    return "\n".join(d)
def extract_patch_blocks(content: str) -> tuple[str, str]:
    dev_patch, llm_patch = "", ""
    dev_start = content.find("--- before.py")
    llm_start = content.find("```diff")
    
    if dev_start != -1 and llm_start != -1:
        dev_patch = content[dev_start:llm_start].strip()
        llm_patch = content[llm_start:].strip("```diff\n").strip("```").strip()
    
    return dev_patch, llm_patch

def extract_changed_lines(diff_text: str) -> tuple[set, set]:
    added = set()
    removed = set()
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.add(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.add(line[1:].strip())
    return removed, added

def compare_dev_llm(dev_patch: str, llm_patch: str) -> tuple[int, int, int, int]:
    dev_removed, dev_added = extract_changed_lines(dev_patch)
    llm_removed, llm_added = extract_changed_lines(llm_patch)

    dev_lines = dev_removed | dev_added
    llm_lines = llm_removed | llm_added

    matched_lines = dev_lines & llm_lines
    score = round(100 * len(matched_lines) / max(len(dev_lines), 1))
    return len(dev_lines), len(llm_lines), len(matched_lines), score

def count_added_removed_lines(diff_text: str) -> tuple[int, int]:
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed

def compute_bleu_score(reference_text: str, candidate_text: str) -> float:
    result = bleu_metric.compute(predictions=[candidate_text], references=[[reference_text]])
    return round(result["bleu"], 4)

def compute_rouge_score(reference_text: str, candidate_text: str) -> float:
    result = rouge_metric.compute(predictions=[candidate_text], references=[reference_text])
    return round(result["rougeL"], 4)
