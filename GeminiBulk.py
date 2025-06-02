import logging
from datetime import datetime
from inc.config import RESULTS_DIR, SAMPLE_SIZE
from inc.data_loader import load_and_clean_csv
from inc.github_utils import get_changed_files, download_file_content
from inc.processor import save_versions, run_llm_on_before_file, compare_fixes
import os
from pathlib import Path
import pandas as pd

def setup_logging():
    """Setup logging configuration and save logs to /logs folder"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)  # Create logs directory if it doesn't exist
    log_path = os.path.join(log_dir, f'processing_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def ensure_directories():
    """Ensure all required directories exist"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def process_repo(row):
    """Process a single repository entry"""
    url, sha, test_path, category = row['Project URL'], row['SHA Detected'], row['Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)'], row['Category']
    owner, repo = url.rstrip("/").split("/")[-2:]
    
    logging.info(f"Processing repository: {owner}/{repo}")
    logging.info(f"Commit SHA: {sha}")
    logging.info(f"Test path: {test_path}")
    
    try:
        changed_files, parent_sha = get_changed_files(owner, repo, sha)
        logging.info(f"Found {len(changed_files)} changed files in commit")
        
        files_processed = 0
        for file in changed_files:
            file_path = file.get("filename", "")
            test_file = test_path.split("::")[0]
                
            # Debug print what we're comparing
            logging.info(f"🔍Comparing:")
            logging.info(f"  Test file: {test_file}")
            logging.info(f"  Changed file: {file_path}")
            test_name = Path(test_file).name
            changed_name = Path(file_path).name
            if test_name == changed_name:
                logging.info(f"✨ Match found! Processing: {file_path}")
                before = download_file_content(owner, repo, file_path, parent_sha)
                after = download_file_content(owner, repo, file_path, sha)
                
                outdir = RESULTS_DIR / category / test_path.replace("/", "_").replace("::", "_")
                outdir.mkdir(parents=True, exist_ok=True)
                
                save_versions(outdir, before, after)
                llm_fix = run_llm_on_before_file(outdir / "before.py",category)
                (outdir / "LLM_suggestion.txt").write_text(llm_fix)


                dev_patch = (outdir / "developer_patch.diff").read_text()
                comparison = compare_fixes(llm_fix, dev_patch)
                
                (outdir / "comparison.txt").write_text(comparison)
                files_processed += 1
                logging.info(f"✅ Successfully processed file {file_path}")
                
        logging.info(f"Completed processing {files_processed} files for {owner}/{repo}")
                
    except Exception as e:
        logging.error(f"❌ Error processing {url}@{sha}: {str(e)}")
        with open("logs/error_log.txt", "a") as f:
            f.write(f"{url}@{sha}: {str(e)}\n")

def main():
    setup_logging()
    logging.info("Starting bulk processing")
    ensure_directories()
    
    CSV_PATH = "raw_data_csv/py-data.csv"
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")
    
    df = load_and_clean_csv(CSV_PATH)
    logging.info(f"Loaded CSV with {len(df)} entries")
    
    sample_df = df.sample(SAMPLE_SIZE, random_state=42)
    logging.info(f"Processing {SAMPLE_SIZE} samples")
    
    processed_count = 0
    for _, row in sample_df.iterrows():
        processed_count += 1
        logging.info(f"Processing item {processed_count}/{SAMPLE_SIZE}")
        process_repo(row)
    
    logging.info("Bulk processing completed")
    logging.info(f"Processed {processed_count} repositories")

if __name__ == "__main__":
    main()