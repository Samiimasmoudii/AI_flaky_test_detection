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
from inc.processor import run_gemini_on_file, run_openai_on_file, compare_fixes, GEMINI_MODEL, OPENAI_MODEL

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables based on chosen LLM
def check_required_env_vars(selected_llm):
    """Check for required environment variables based on selected LLM"""
    base_vars = ['GITHUB_TOKEN']
    
    if selected_llm == 'openai':
        required_vars = base_vars + ['OPENAI_API_KEY']
    elif selected_llm == 'gemini':
        required_vars = base_vars + ['GEMINI_API_KEY']
    else:  # 'both'
        required_vars = base_vars + ['OPENAI_API_KEY', 'GEMINI_API_KEY']
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables for {selected_llm}: {', '.join(missing_vars)}")

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

def choose_llm():
    """Let user choose which LLM to use"""
    print("\n" + "="*60)
    print("LLM MODEL SELECTION")
    print("="*60)
    
    # Check which APIs are available
    available_llms = []
    
    if os.getenv('OPENAI_API_KEY'):
        available_llms.append(('openai', OPENAI_MODEL))
        print(f"1. OpenAI ({OPENAI_MODEL})")
    
    if os.getenv('GEMINI_API_KEY'):
        available_llms.append(('gemini', GEMINI_MODEL))
        print(f"2. Gemini ({GEMINI_MODEL})")
    
    if len(available_llms) > 1:
        print(f"3. Both models (run comparison)")
    
    if not available_llms:
        print("❌ No LLM API keys found! Please set OPENAI_API_KEY and/or GEMINI_API_KEY")
        return None, None
    
    print("="*60)
    
    while True:
        try:
            if len(available_llms) == 1:
                # Only one LLM available, auto-select
                llm_choice, model_name = available_llms[0]
                print(f"✅ Auto-selected: {llm_choice.upper()} ({model_name}) - only API key available")
                return llm_choice, model_name
            
            max_choice = 2 + (1 if len(available_llms) > 1 else 0)
            choice = input(f"Choose LLM (1-{max_choice}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                return None, None
            
            choice_num = int(choice)
            
            if choice_num == 1 and len(available_llms) >= 1:
                return available_llms[0]
            elif choice_num == 2 and len(available_llms) >= 2:
                return available_llms[1]
            elif choice_num == 3 and len(available_llms) > 1:
                return 'both', 'both'
            else:
                print(f"❌ Please enter a number between 1 and {max_choice}")
        
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None, None

def choose_file_type():
    """Let user choose which file type to process"""
    print("\n" + "="*60)
    print("FILE TYPE SELECTION")
    print("="*60)
    
    # Check what data is available
    python_dir = RESULTS_DIR / 'python_data'
    java_dir = RESULTS_DIR / 'java_data'
    
    available_types = []
    if python_dir.exists():
        python_count = sum(1 for category_dir in python_dir.iterdir() 
                          if category_dir.is_dir() 
                          for test_dir in category_dir.iterdir() 
                          if test_dir.is_dir())
        available_types.append(('python', python_count))
        print(f"1. Python tests ({python_count:,} available)")
    
    if java_dir.exists():
        java_count = sum(1 for category_dir in java_dir.iterdir() 
                        if category_dir.is_dir() 
                        for test_dir in category_dir.iterdir() 
                        if test_dir.is_dir())
        available_types.append(('java', java_count))
        print(f"2. Java tests ({java_count:,} available)")
    
    if len(available_types) > 1:
        print(f"3. All types ({sum(count for _, count in available_types):,} total)")
    
    if not available_types:
        print("❌ No test data found in results directory!")
        return None
    
    print("="*60)
    
    while True:
        try:
            if len(available_types) == 1:
                # Only one type available, auto-select
                file_type = available_types[0][0]
                print(f"✅ Auto-selected: {file_type.upper()} (only type available)")
                return file_type
            
            choice = input(f"Choose file type (1-{2 + (1 if len(available_types) > 1 else 0)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                return None
            
            choice_num = int(choice)
            
            if choice_num == 1 and len(available_types) >= 1:
                return available_types[0][0]
            elif choice_num == 2 and len(available_types) >= 2:
                return available_types[1][0]
            elif choice_num == 3 and len(available_types) > 1:
                return 'all'
            else:
                print(f"❌ Please enter a number between 1 and {2 + (1 if len(available_types) > 1 else 0)}")
        
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None

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

def run_llm_analysis(before_file, category, llm_choice):
    """Run LLM analysis based on the chosen model"""
    if llm_choice == 'openai':
        return run_openai_on_file(str(before_file), category)
    elif llm_choice == 'gemini':
        return run_gemini_on_file(str(before_file), category)
    else:
        raise ValueError(f"Unknown LLM choice: {llm_choice}")

def get_model_info(llm_choice):
    """Get model name and safe folder name for the chosen LLM"""
    if llm_choice == 'openai':
        model_name = OPENAI_MODEL
        safe_name = model_name.replace(".", "_").replace("-", "_")
    elif llm_choice == 'gemini':
        model_name = GEMINI_MODEL
        safe_name = model_name.replace(".", "_").replace("-", "_")
    else:
        raise ValueError(f"Unknown LLM choice: {llm_choice}")
    
    return model_name, safe_name

def process_result_folder(folder_path, file_type, llm_choice):
    """Process a single result folder containing before.py/java and after.py/java"""
    logging.info(f"Processing folder: {folder_path}")
    
    # Determine file extension based on file type
    file_ext = '.py' if file_type == 'python' else '.java'
    before_file = folder_path / f"before{file_ext}"
    
    if not before_file.exists():
        logging.warning(f"Skipping {folder_path} - missing before{file_ext}")
        return False
    
    # Check if before file is empty
    try:
        before_content = before_file.read_text(encoding='utf-8').strip()
        if not before_content:
            logging.warning(f"Skipping {folder_path} - before{file_ext} is empty")
            return False
    except Exception as e:
        logging.warning(f"Skipping {folder_path} - error reading before{file_ext}: {e}")
        return False
    
    try:
        # Get the category from the parent folder name
        category = folder_path.parent.name
        
        # Get model information
        model_name, safe_model_name = get_model_info(llm_choice)
        
        # Create a new test folder for this run
        model_test_folder = create_model_test_folder(folder_path, safe_model_name)
        logging.info(f"Created test folder: {model_test_folder}")
        
        # Add rate limiting delay between files
        time.sleep(RATE_LIMIT_DELAY)
        
        # Run LLM analysis on the before file
        llm_fix = run_llm_analysis(before_file, category, llm_choice)
        
        if not llm_fix:
            logging.error(f"❌ Failed to get valid fix for {before_file}")
            return False
            
        # Save the LLM suggestion
        (model_test_folder / "llm_suggestion.txt").write_text(llm_fix, encoding='utf-8')
        
        # Copy the developer patch for reference if it exists
        dev_patch_file = folder_path / "developer_patch.diff"
        if dev_patch_file.exists():
            dev_patch = dev_patch_file.read_text(encoding='utf-8')
            # Save developer patch at test case level if it doesn't exist
            if not (folder_path / "developer_patch.diff").exists():
                (folder_path / "developer_patch.diff").write_text(dev_patch, encoding='utf-8')
            
            # Compare the fixes
            comparison = compare_fixes(llm_fix, dev_patch)
            # Save the comparison in model test folder
            (model_test_folder / "comparison.txt").write_text(comparison, encoding='utf-8')
        
        # Create after file in the model test folder by applying the LLM suggestion
        try:
            # TODO: Parse the LLM suggestion diff and apply it to create after file
            # For now, just copy the before file as a placeholder
            (model_test_folder / f"after{file_ext}").write_text(before_content, encoding='utf-8')
            
        except Exception as e:
            logging.error(f"Failed to create after{file_ext}: {str(e)}")
        
        # Save metadata about this test run
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "llm_provider": llm_choice,
            "category": category,
            "file_type": file_type,
            "file_extension": file_ext,
            "success": True
        }
        metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
        (model_test_folder / "metadata.txt").write_text(metadata_str, encoding='utf-8')
        
        logging.info(f"✅ Successfully processed {folder_path} to {model_test_folder}")
        return True
        
    except Exception as e:
        logging.error(f"❌ Error processing {folder_path}: {str(e)}")
        # Save error metadata
        if 'model_test_folder' in locals():
            model_name, _ = get_model_info(llm_choice) if llm_choice in ['openai', 'gemini'] else ("unknown", "unknown")
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "llm_provider": llm_choice,
                "category": category if 'category' in locals() else "unknown",
                "file_type": file_type,
                "success": False,
                "error": str(e)
            }
            metadata_str = "\n".join(f"{k}: {v}" for k, v in metadata.items())
            (model_test_folder / "metadata.txt").write_text(metadata_str, encoding='utf-8')
            
        with open("logs/error_log.txt", "a", encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()} - {folder_path}: {str(e)}\n")
        return False

def process_file_type(file_type, llm_choice):
    """Process all test folders for a specific file type using the chosen LLM"""
    data_dir = RESULTS_DIR / f'{file_type}_data'
    
    if not data_dir.exists():
        logging.warning(f"Data directory not found: {data_dir}")
        return 0, 0
    
    processed_count = 0
    success_count = 0
    
    model_name, _ = get_model_info(llm_choice)
    logging.info(f"Processing {file_type.upper()} tests from: {data_dir} using {model_name}")
    
    # Process each category folder
    for category_dir in data_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        logging.info(f"Processing category: {category_dir.name}")
        
        # Process each test case folder
        for test_case_dir in category_dir.iterdir():
            if not test_case_dir.is_dir():
                continue
                
            processed_count += 1
            if process_result_folder(test_case_dir, file_type, llm_choice):
                success_count += 1
            
            # Progress update every 10 files
            if processed_count % 10 == 0:
                success_rate = (success_count / processed_count) * 100
                logging.info(f"Progress: {processed_count} processed, {success_count} successful ({success_rate:.1f}%)")
    
    return processed_count, success_count

def main():
    setup_logging()
    
    if not RESULTS_DIR.exists():
        raise FileNotFoundError(f"Results directory not found: {RESULTS_DIR}")
    
    # Let user choose LLM
    llm_choice, model_name = choose_llm()
    if not llm_choice:
        return
    
    # Check required environment variables based on choice
    check_required_env_vars(llm_choice)
    
    # Let user choose file type
    selected_file_type = choose_file_type()
    if not selected_file_type:
        return
    
    if llm_choice == 'both':
        logging.info(f"Starting results analysis using BOTH models for {selected_file_type.upper()} files")
        # TODO: Implement both model comparison
        print("⚠️  Both model comparison not yet implemented. Please choose a single model.")
        return
    else:
        logging.info(f"Starting results analysis using {model_name} for {selected_file_type.upper()} files")
    
    total_processed = 0
    total_success = 0
    
    if selected_file_type == 'all':
        # Process both Python and Java
        for file_type in ['python', 'java']:
            data_dir = RESULTS_DIR / f'{file_type}_data'
            if data_dir.exists():
                processed, success = process_file_type(file_type, llm_choice)
                total_processed += processed
                total_success += success
                logging.info(f"{file_type.upper()} summary: {success}/{processed} successful")
    else:
        # Process selected file type
        total_processed, total_success = process_file_type(selected_file_type, llm_choice)
    
    # Final summary
    if total_processed > 0:
        success_rate = (total_success / total_processed) * 100
        logging.info(f"\n{'='*60}")
        logging.info(f"ANALYSIS COMPLETED")
        logging.info(f"{'='*60}")
        logging.info(f"Total files processed: {total_processed:,}")
        logging.info(f"Successful: {total_success:,} ({success_rate:.1f}%)")
        logging.info(f"Failed: {total_processed - total_success:,}")
        logging.info(f"Model used: {model_name}")
        logging.info(f"LLM Provider: {llm_choice.upper()}")
        logging.info(f"File type(s): {selected_file_type.upper()}")
    else:
        logging.warning("No files were processed!")

if __name__ == "__main__":
    main() 