import logging
from datetime import datetime
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from inc.config import RESULTS_DIR
from inc.processor import setup_logging
import re

GEMINI_MODEL = "gemini-2.5-flash-preview-04-17"
ANALYSIS_DIR = Path("analysis_results")

def read_file_content(file_path: Path) -> str:
    """Read and return file content, return empty string if file doesn't exist."""
    try:
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            logging.warning(f"File not found: {file_path}")
            return ""
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return ""

def find_diff_file(model_dir: Path) -> Path:
    """Find the diff file in the model directory, checking for various possible names."""
    possible_names = ["diff.txt", "diff (1).txt", "diff (2).txt"]
    for name in possible_names:
        diff_file = model_dir / name
        if diff_file.exists():
            return diff_file
    return model_dir / "diff.txt"  # Return default path if none found

def discover_available_models_and_tests(results_dir: Path) -> dict:
    """Discover all available models and test numbers across all test cases."""
    models_tests = {}
    
    for category_dir in results_dir.iterdir():
        if not category_dir.is_dir():
            continue
            
        for test_case_dir in category_dir.iterdir():
            if not test_case_dir.is_dir():
                continue
                
            for model_dir in test_case_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                # Parse model name and test number using regex
                # Pattern: modelname_Test## or modelname_test##
                match = re.match(r'^(.+?)_[Tt]est(\d+)$', model_dir.name)
                if match:
                    model_name = match.group(1)
                    test_number = match.group(2)
                    
                    if model_name not in models_tests:
                        models_tests[model_name] = set()
                    models_tests[model_name].add(test_number)
    
    # Convert sets to sorted lists
    for model in models_tests:
        models_tests[model] = sorted(list(models_tests[model]))
    
    return models_tests

def select_model_and_test() -> tuple:
    """Let user select which model and test number to analyze."""
    available = discover_available_models_and_tests(RESULTS_DIR)
    
    if not available:
        print("No model test results found!")
        return None, None
    
    print("\nAvailable models and their test numbers:")
    print("=" * 50)
    for model, tests in available.items():
        print(f"Model: {model}")
        print(f"  Available tests: {', '.join(tests)}")
        print()
    
    # Select model
    print("Available models:")
    models = list(available.keys())
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    
    while True:
        try:
            choice = int(input("\nSelect model number: ")) - 1
            if 0 <= choice < len(models):
                selected_model = models[choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Select test number
    available_tests = available[selected_model]
    print(f"\nAvailable test numbers for {selected_model}: {', '.join(available_tests)}")
    
    while True:
        test_number = input("Enter test number (e.g., 00, 01, 02): ").strip()
        if test_number in available_tests:
            break
        else:
            print(f"Test number '{test_number}' not available. Available: {', '.join(available_tests)}")
    
    return selected_model, test_number

def analyze_with_gemini(before_code: str, dev_patch: str, llm_patch: str, model_name: str, test_number: str, category: str) -> dict:
    """Use Gemini to analyze and compare the fixes."""
    prompt = f"""
    As an expert in analyzing Python test code and flaky test fixes, analyze how well the LLM's fix matches the developer's solution.
    The test is categorized as: {category}
    Model: {model_name}, Test: {test_number}
    
    Here is the original flaky test code:
    ```python
    {before_code}
    ```
    
    Here is the developer's fix (this is the correct solution):
    ```diff
    {dev_patch}
    ```
    
    Here is the {model_name}'s proposed fix (Test {test_number}):
    ```diff
    {llm_patch}
    ```
    
    Compare the LLM's fix to the developer's solution and rate how well it matches:
    
    1. Result Rating (choose ONE):
       "X" - The LLM's fix is completely different from the developer's solution or misses the core issue
       "-" - The LLM's fix partially matches the developer's solution but misses some key aspects
       "+" - The LLM's fix successfully replicates the developer's solution or achieves the same outcome
    
    2. Brief Summary (explain how the LLM's fix compares to the developer's solution)
    
    Format your response exactly like this:
    RATING: [X/-/+]
    SUMMARY: [Your brief explanation comparing LLM's fix to developer's solution]
    """
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        # Parse the response
        lines = response.text.strip().split('\n')
        rating = next((line.split(':')[1].strip() for line in lines if line.startswith('RATING:')), 'X')
        summary = next((line.split(':')[1].strip() for line in lines if line.startswith('SUMMARY:')), 'Analysis failed')
        
        return {
            'rating': rating,
            'summary': summary
        }
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return {
            'rating': 'X',
            'summary': f"Error analyzing fixes: {str(e)}"
        }

def save_analysis_results(analyses: list, model_name: str, test_number: str, timestamp: str) -> None:
    """Save all analyses to a single directory with timestamp."""
    try:
        # Create analysis directory with timestamp
        output_dir = ANALYSIS_DIR / f"{model_name}_Test{test_number}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Excel-like summary
        df_data = []
        for analysis in analyses:
            df_data.append({
                'Test Case': analysis['test_case'],
                'Category': analysis['category'],
                'Dev Fix': analysis['dev_fix_summary'],
                f'{model_name}_Test{test_number}': analysis.get('rating', ''),
                'Summary': analysis['summary']
            })
        
        df = pd.DataFrame(df_data)
        excel_file = output_dir / f"analysis_{model_name}_Test{test_number}_{timestamp}.xlsx"
        df.to_excel(excel_file, index=False)
        
        # Also save detailed text report
        text_file = output_dir / f"detailed_analysis_{model_name}_Test{test_number}_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Report: {model_name} Test {test_number}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for analysis in analyses:
                f.write(f"Test Case: {analysis['test_case']}\n")
                f.write(f"Category: {analysis['category']}\n")
                f.write(f"Developer Fix: {analysis['dev_fix_summary']}\n")
                f.write(f"Rating: {analysis.get('rating', 'N/A')}\n")
                f.write(f"Summary: {analysis['summary']}\n")
                f.write("-" * 80 + "\n\n")
        
        logging.info(f"Analysis results saved to {output_dir}")
        print(f"\nResults saved to: {output_dir}")
    except Exception as e:
        logging.error(f"Error saving analysis results: {e}")

def main():
    # Initialize logging
    log_file = setup_logging()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Load environment variables
        logging.info("Loading environment variables...")
        load_dotenv()
        
        # Check for Gemini API key
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        # Read the Excel file
        excel_path = Path("Manual_Comparison.xlsx")
        if not excel_path.exists():
            raise FileNotFoundError(f"Manual comparison file not found: {excel_path}")
        
        df = pd.read_excel(excel_path)
        test_cases = set(df.iloc[:, 0].dropna().astype(str).tolist())
        logging.info(f"Found {len(test_cases)} test cases to analyze")
        
        # Let user select model and test number once
        selected_model, selected_test = select_model_and_test()
        if not selected_model or not selected_test:
            print("No valid selection made. Exiting.")
            return
        
        print(f"\nAnalyzing: {selected_model} Test {selected_test}")
        print("=" * 50)
        
        # Collect all files that will be processed
        logging.info("\nCollecting files to process...")
        files_to_process = []
        for category_dir in RESULTS_DIR.iterdir():
            if not category_dir.is_dir():
                continue
                
            for test_case_dir in category_dir.iterdir():
                if not test_case_dir.is_dir() or test_case_dir.name not in test_cases:
                    continue
                
                before_file = test_case_dir / "before.py"
                dev_patch_file = test_case_dir / "developer_patch.diff"
                
                # Look for the specific model and test combination
                target_folder = f"{selected_model}_Test{selected_test}"
                model_test_dir = test_case_dir / target_folder
                
                if before_file.exists() and dev_patch_file.exists() and model_test_dir.exists():
                    files_to_process.append({
                        'category': category_dir.name,
                        'test_case': test_case_dir.name,
                        'path': test_case_dir,
                        'model_dir': model_test_dir
                    })

        print(f"\nFound {len(files_to_process)} test cases with {selected_model} Test {selected_test}")
        
        if not files_to_process:
            print("No matching test cases found!")
            return
        
        # Process the files and collect analyses
        logging.info("\nStarting analysis...")
        analyses = []
        
        for idx, file_info in enumerate(files_to_process, 1):
            print(f"\nProcessing {idx}/{len(files_to_process)}: {file_info['test_case']}")
            logging.info(f"Analyzing {file_info['test_case']}")
            
            test_dir = file_info['path']
            before_code = read_file_content(test_dir / "before.py")
            dev_patch = read_file_content(test_dir / "developer_patch.diff")
            
            # Read LLM patch
            llm_patch_file = find_diff_file(file_info['model_dir'])
            llm_patch = read_file_content(llm_patch_file)
            
            if not llm_patch:
                logging.warning(f"No LLM patch found for {file_info['test_case']}")
                continue
            
            # Analyze with Gemini
            result = analyze_with_gemini(before_code, dev_patch, llm_patch, 
                                       selected_model, selected_test, 
                                       file_info['category'])
            
            analysis = {
                'category': file_info['category'],
                'test_case': file_info['test_case'],
                'dev_fix_summary': dev_patch.split('\n')[0] if dev_patch else "No developer fix found",
                'rating': result['rating'],
                'summary': result['summary']
            }
            
            analyses.append(analysis)
            print(f"  Result: {result['rating']} - {result['summary'][:50]}...")
        
        # Save all analyses
        save_analysis_results(analyses, selected_model, selected_test, timestamp)
        
        # Print summary
        total = len(analyses)
        plus_count = sum(1 for a in analyses if a['rating'] == '+')
        minus_count = sum(1 for a in analyses if a['rating'] == '-')
        x_count = sum(1 for a in analyses if a['rating'] == 'X')
        
        print(f"\n" + "=" * 50)
        print(f"ANALYSIS SUMMARY for {selected_model} Test {selected_test}")
        print(f"=" * 50)
        print(f"Total test cases: {total}")
        print(f"Successful (+): {plus_count} ({plus_count/total*100:.1f}%)")
        print(f"Partial (-): {minus_count} ({minus_count/total*100:.1f}%)")
        print(f"Failed (X): {x_count} ({x_count/total*100:.1f}%)")
        
        logging.info("\nAnalysis completed")
        logging.info("=" * 80)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 