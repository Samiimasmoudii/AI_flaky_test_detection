import logging
from datetime import datetime
import os
from pathlib import Path
import google.generativeai as genai
from inc.config import RESULTS_DIR
from inc.processor import run_llm_on_before_file, compare_fixes

MODEL_NAME = "Gemini2.5"  # Can be changed for different models

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

def get_next_test_number(folder_path, model_name):
    """Get the next test number for a given model"""
    pattern = f"{model_name}_Test(\\d+)"
    max_number = 0
    
    if folder_path.exists():
        for item in folder_path.iterdir():
            if item.is_dir():
                match = re.match(pattern, item.name)
                if match:
                    test_num = int(match.group(1))
                    max_number = max(max_number, test_num)
    
    return max_number + 1

def create_model_test_folder(base_folder, model_name):
    """Create a new test folder for the model with incremental numbering"""
    next_number = get_next_test_number(base_folder, model_name)
    test_folder_name = f"{model_name}_Test{next_number:02d}"
    test_folder = base_folder / test_folder_name
    test_folder.mkdir(parents=True, exist_ok=True)
    return test_folder

def process_result_folder(folder_path):
    """Process a single result folder containing before.py and after.py"""
    logging.info(f"Processing folder: {folder_path}")
    
    before_file = folder_path / "before.py"
    after_file = folder_path / "after.py"
    
    if not before_file.exists() or not after_file.exists():
        logging.warning(f"Skipping {folder_path} - missing before.py or after.py")
        return
    
    try:
        # Get the category from the parent folder name
        category = folder_path.parent.name
        
        # Create a new test folder for this run
        model_test_folder = create_model_test_folder(folder_path, MODEL_NAME)
        logging.info(f"Created test folder: {model_test_folder}")
        
        # Run Gemini analysis on the before file
        llm_fix = run_llm_on_before_file(before_file, category)
        
        # Save the LLM suggestion
        (model_test_folder / "llm_suggestion.txt").write_text(llm_fix)
        
        # Copy the developer patch for reference
        dev_patch = (folder_path / "developer_patch.diff").read_text()
        (model_test_folder / "developer_patch.diff").write_text(dev_patch)
        
        # Compare the fixes
        comparison = compare_fixes(llm_fix, dev_patch)
        
        # Save the comparison
        (model_test_folder / "comparison.txt").write_text(comparison)
        
        # Save metadata about this test run
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "category": category,
        }
        metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        (model_test_folder / "metadata.txt").write_text(metadata_str)
        
        logging.info(f"✅ Successfully processed {folder_path} to {model_test_folder}")
        
    except Exception as e:
        logging.error(f"❌ Error processing {folder_path}: {str(e)}")
        with open("logs/error_log.txt", "a") as f:
            f.write(f"{folder_path}: {str(e)}\n")

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