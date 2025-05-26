import logging
from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from inc.config import RESULTS_DIR
from inc.processor import (
    run_openai_on_file,
    setup_logging,
    OPENAI_MODEL as MODEL_NAME
)

def main():
    # Initialize logging first
    log_file = setup_logging()
    
    try:
        # Load environment variables from .env file
        logging.info("Loading environment variables from .env file...")
        load_dotenv()
        logging.info("Environment variables loaded")
        
        # Check for required environment variables
        logging.info("Checking required environment variables...")
        required_env_vars = ['GITHUB_TOKEN', 'OPENAI_API_KEY']
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        logging.info("All required environment variables found")
        
        if not RESULTS_DIR.exists():
            logging.error(f"Results directory not found: {RESULTS_DIR}")
            raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
        
        # Read the manual comparison Excel file
        excel_path = Path("Manual_Comparison.xlsx")
        if not excel_path.exists():
            logging.error(f"Manual comparison file not found: {excel_path}")
            raise FileNotFoundError(f"Manual comparison file not found: {excel_path}")
        
        logging.info("Reading test cases from Excel file...")
        df = pd.read_excel(excel_path)
        test_cases_to_process = set(df.iloc[:, 0].dropna().astype(str).tolist())
        logging.info(f"Found {len(test_cases_to_process)} test cases to process")
        
        # Collect all files that will be processed
        logging.info("\nCollecting files to process...")
        files_to_process = []
        for category_dir in RESULTS_DIR.iterdir():
            if not category_dir.is_dir():
                continue
                
            for test_case_dir in category_dir.iterdir():
                if not test_case_dir.is_dir():
                    continue
                    
                if test_case_dir.name in test_cases_to_process:
                    before_file = test_case_dir / "before.py"
                    if before_file.exists():
                        files_to_process.append({
                            'category': category_dir.name,
                            'test_case': test_case_dir.name,
                            'path': before_file
                        })

        # Print summary of files to be processed
        logging.info("\n=== Files to be processed ===")
        logging.info(f"Total number of files: {len(files_to_process)}")
        logging.info("\nFiles list:")
        for idx, file_info in enumerate(files_to_process, 1):
            logging.info(f"{idx}. Category: {file_info['category']}")
            logging.info(f"   Test Case: {file_info['test_case']}")
            logging.info(f"   Path: {file_info['path']}\n")

        # Ask for confirmation
        user_input = input("\nDo you want to proceed with processing these files? (Type 'ok' to continue): ")
        
        if user_input.lower() != 'ok':
            logging.info("Operation cancelled by user")
            return
        
        # Process the files
        logging.info("\nStarting file processing...")
        for idx, file_info in enumerate(files_to_process, 1):
            logging.info(f"\nProcessing file {idx}/{len(files_to_process)}")
            logging.info(f"Test case: {file_info['test_case']}")
            run_openai_on_file(str(file_info['path']), file_info['category'])
        
        logging.info("\nAnalysis completed")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 