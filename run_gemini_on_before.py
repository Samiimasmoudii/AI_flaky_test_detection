import logging
from datetime import datetime
import os
import shutil
from pathlib import Path
import google.generativeai as genai
import re
import time
from dotenv import load_dotenv
from inc.config import RESULTS_DIR
from inc.processor import run_llm_on_before_file, compare_fixes, MODEL_NAME

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
required_env_vars = ['GITHUB_TOKEN', 'GEMINI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

MAX_RETRIES = 3
RATE_LIMIT_DELAY = 6  # 6 seconds between requests (10 requests per minute)

def setup_logging():
    """Setup logging configuration and save logs to /logs folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'analysis_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def get_next_test_number(test_case_dir, model_name):
    """Get the next test number for a given model in a test case directory"""
    pattern = f"{model_name}_Test(\\d+)"
    max_number = 0
    
    if test_case_dir.exists():
        for item in test_case_dir.iterdir():
            if item.is_dir():
                match = re.match(pattern, item.name)
                if match:
                    test_num = int(match.group(1))
                    max_number = max(max_number, test_num)
    
    return max_number + 1

def create_model_test_folder(test_case_dir, model_name):
    """Create a new test folder for the model with incremental numbering"""
    next_number = get_next_test_number(test_case_dir, model_name)
    test_folder_name = f"{model_name}_Test{next_number:02d}"
    test_folder = test_case_dir / test_folder_name
    test_folder.mkdir(parents=True, exist_ok=True)
    return test_folder

def process_result_folder(folder_path):
    """Process a single result folder containing before.py and after.py"""
    logging.info(f"Processing folder: {folder_path}")
    
    before_file = folder_path / "before.py"
    
    if not before_file.exists():
        logging.warning(f"Skipping {folder_path} - missing before.py")
        return
    
    try:
        # Get the category from the parent folder name
        category = folder_path.parent.name
        
        # Create a new test folder for this run
        model_test_folder = create_model_test_folder(folder_path, MODEL_NAME.replace(".", "_"))
        logging.info(f"Created test folder: {model_test_folder}")
        
        # Add rate limiting delay between files
        time.sleep(RATE_LIMIT_DELAY)
        
        # Run Gemini analysis on the before file
        llm_fix = run_llm_on_before_file(before_file, category)
        
        if not llm_fix:
            logging.error(f"❌ Failed to get valid fix for {before_file}")
            return
            
        # Save the LLM suggestion
        (model_test_folder / "llm_suggestion.txt").write_text(llm_fix)
        
        # Copy the developer patch for reference if it exists
        dev_patch_file = folder_path / "developer_patch.diff"
        if dev_patch_file.exists():
            dev_patch = dev_patch_file.read_text()
            # Save developer patch at test case level if it doesn't exist
            if not (folder_path / "developer_patch.diff").exists():
                (folder_path / "developer_patch.diff").write_text(dev_patch)
            
            # Compare the fixes
            comparison = compare_fixes(llm_fix, dev_patch)
            # Save the comparison in model test folder
            (model_test_folder / "comparison.txt").write_text(comparison)
        
        # Create after.py in the model test folder by applying the LLM suggestion
        try:
            # Read the original before.py content
            before_content = before_file.read_text()
            
            # TODO: Parse the LLM suggestion diff and apply it to create after.py
            # For now, just copy the before.py as a placeholder
            (model_test_folder / "after.py").write_text(before_content)
            
        except Exception as e:
            logging.error(f"Failed to create after.py: {str(e)}")
        
        # Save metadata about this test run
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "category": category,
            "success": True
        }
        metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        (model_test_folder / "metadata.txt").write_text(metadata_str)
        
        logging.info(f"✅ Successfully processed {folder_path} to {model_test_folder}")
        
    except Exception as e:
        logging.error(f"❌ Error processing {folder_path}: {str(e)}")
        # Save error metadata
        if 'model_test_folder' in locals():
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "category": category if 'category' in locals() else "unknown",
                "success": False,
                "error": str(e)
            }
            metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            (model_test_folder / "metadata.txt").write_text(metadata_str)
            
        with open("logs/error_log.txt", "a") as f:
            f.write(f"{datetime.now().isoformat()} - {folder_path}: {str(e)}\n")

def main():
    setup_logging()
    logging.info(f"Starting results analysis using {MODEL_NAME}")
    
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    
    # Process each category folder
    for category_dir in RESULTS_DIR.iterdir():
        if not category_dir.is_dir():
            continue
            
        logging.info(f"Processing category: {category_dir.name}")
        
        # Process each test case folder
        for test_case_dir in category_dir.iterdir():
            if not test_case_dir.is_dir():
                continue
                
            process_result_folder(test_case_dir)
    
    logging.info("Analysis completed")

if __name__ == "__main__":
    main() 