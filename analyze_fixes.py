"""
 Analysis Script with LLMs and CodeBLEU Support

This script analyzes LLM-generated fixes against developer fixes using either:
1. Gemini LLM Analysis (qualitative comparison)
2. CodeBLEU Metric Analysis (quantitative similarity score with 4 components)

SETUP:
1. Install dependencies: pip install -r requirements.txt
2. For Gemini analysis: Set GEMINI_API_KEY in .env file
3. For CodeBLEU analysis: Install codebleu library (pip install codebleu)

CodeBLEU COMPONENTS (equal weights 0.25 each):
- BLEU: n-gram overlap
- BLEUweight: gives more weight to important syntax tokens (assert, public, etc.)
- MatchAST: measures syntactic tree similarity (AST)
- MatchDF: measures semantic similarity using data-flow graphs

USAGE:
python analyze_fixes.py

The script will guide you through:
1. Choosing analysis method (Gemini or CodeBLEU)
2. Selecting file type (Python or Java tests)
3. Choosing which model and test numbers to analyze
4. Optional Excel filtering of test cases

RESULTS:
- Individual test results saved to analysis_results/
- Excel files with detailed breakdowns
- Text files with comprehensive analysis
- CodeBLEU scores include all 4 component scores
"""

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

# CodeBLEU imports
try:
    from codebleu import calc_codebleu
    CODEBLEU_AVAILABLE = True
except ImportError:
    CODEBLEU_AVAILABLE = False

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

def detect_available_file_types() -> dict:
    """Detect available file types (py_data, java_data) and count test cases."""
    file_types = {}
    
    for data_type in ['py_data', 'java_data']:
        data_dir = RESULTS_DIR / data_type
        if data_dir.exists():
            test_count = sum(1 for category_dir in data_dir.iterdir() 
                           if category_dir.is_dir() 
                           for test_dir in category_dir.iterdir() 
                           if test_dir.is_dir())
            if test_count > 0:
                file_types[data_type] = test_count
    
    return file_types

def choose_file_type() -> str:
    """Let user choose which file type to analyze."""
    available_types = detect_available_file_types()
    
    if not available_types:
        print("❌ No test data found in results directory!")
        return None
    
    print("\n" + "="*60)
    print("FILE TYPE SELECTION FOR ANALYSIS")
    print("="*60)
    
    type_list = list(available_types.keys())
    for i, (file_type, count) in enumerate(available_types.items(), 1):
        display_name = file_type.replace('_data', '').upper()
        print(f"{i}. {display_name} tests ({count:,} available)")
    
    print("="*60)
    
    while True:
        try:
            if len(available_types) == 1:
                selected = type_list[0]
                display_name = selected.replace('_data', '').upper()
                print(f"✅ Auto-selected: {display_name} (only type available)")
                return selected
            
            choice = input(f"Choose file type (1-{len(available_types)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                return None
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(available_types):
                return type_list[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(available_types)}")
        
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return None

def get_test_numbering_info(file_type: str) -> dict:
    """Get test numbering information based on file type."""
    if file_type == 'java_data':
        return {
            'start_number': 10,
            'format': '{:02d}',  # Still use 2-digit format: 10, 11, 12
            'file_ext': '.java',
            'language': 'Java'
        }
    else:  # py_data
        return {
            'start_number': 0,
            'format': '{:02d}',   # 00, 01, 02
            'file_ext': '.py',
            'language': 'Python'
        }

def find_diff_file(model_dir: Path) -> Path:
    """Find the diff file in the model directory, checking for various possible names."""
    possible_names = ["diff.txt", "diff (1).txt", "diff (2).txt", "llm_suggestion.txt"]
    for name in possible_names:
        diff_file = model_dir / name
        if diff_file.exists():
            return diff_file
    return model_dir / "llm_suggestion.txt"  # Return default path if none found

def discover_available_models_and_tests(results_dir: Path, file_type: str) -> dict:
    """Discover all available models and test numbers for the selected file type."""
    models_tests = {}
    data_dir = results_dir / file_type
    
    if not data_dir.exists():
        return models_tests
    
    numbering_info = get_test_numbering_info(file_type)
    
    for category_dir in data_dir.iterdir():
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
                    
                    # For Java, only include test numbers >= 10
                    # For Python, only include test numbers < 10
                    test_num_int = int(test_number)
                    if file_type == 'java_data' and test_num_int >= 10:
                        if model_name not in models_tests:
                            models_tests[model_name] = set()
                        models_tests[model_name].add(test_number)
                    elif file_type == 'py_data' and test_num_int < 10:
                        if model_name not in models_tests:
                            models_tests[model_name] = set()
                        models_tests[model_name].add(test_number)
    
    # Convert sets to sorted lists
    for model in models_tests:
        models_tests[model] = sorted(list(models_tests[model]))
    
    return models_tests

def select_model_and_test(file_type: str) -> tuple:
    """Let user select which model and test number(s) to analyze."""
    available = discover_available_models_and_tests(RESULTS_DIR, file_type)
    numbering_info = get_test_numbering_info(file_type)
    
    if not available:
        print(f"No model test results found for {file_type}!")
        return None, None
    
    print(f"\nAvailable Tested models and their test numbers for {numbering_info['language']}:")
    print("=" * 60)
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
    
    # Select test number(s) with option for all tests
    available_tests = available[selected_model]
    print(f"\nAvailable test numbers for {selected_model}: {', '.join(available_tests)}")
    print("\nTest selection options:")
    print("1. Select a specific test number")
    print("2. Select ALL tests for this file type")
    
    while True:
        try:
            selection_choice = int(input("\nChoose selection type (1 or 2): "))
            if selection_choice == 1:
                # Original single test selection
                while True:
                    test_number = input(f"Enter test number (e.g., {', '.join(available_tests[:3])}): ").strip()
                    if test_number in available_tests:
                        return selected_model, [test_number]  # Return as list for consistency
                    else:
                        print(f"Test number '{test_number}' not available. Available: {', '.join(available_tests)}")
            elif selection_choice == 2:
                # Select all tests
                print(f"✅ Selected ALL tests for {selected_model}: {', '.join(available_tests)}")
                return selected_model, list(available_tests)  # Return all tests as list
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number (1 or 2).")

def calculate_codebleu_score(dev_patch: str, llm_patch: str, language: str) -> dict:
    """Calculate CodeBLEU score between developer and LLM patches."""
    if not CODEBLEU_AVAILABLE:
        return {'rating': 'FAIL', 'analysis': 'CodeBLEU library not available. Install with: pip install codebleu'}
    
    try:
        # Map language names to CodeBLEU supported languages
        lang_mapping = {'Python': 'python', 'Java': 'java', 'C': 'c', 'C++': 'cpp', 'C#': 'c_sharp', 'JavaScript': 'javascript', 'PHP': 'php', 'Go': 'go', 'Ruby': 'ruby', 'Rust': 'rust'}
        codebleu_lang = lang_mapping.get(language, 'python')
        
        # Extract code from patches (remove diff markers)
        def extract_code_from_patch(patch: str) -> str:
            lines = patch.split('\n')
            code_lines = []
            for line in lines:
                if line.startswith('+') and not line.startswith('+++'):
                    code_lines.append(line[1:])  # Remove + prefix
                elif not line.startswith('-') and not line.startswith('@@') and not line.startswith('diff ') and not line.startswith('index ') and not line.startswith('---') and not line.startswith('+++'):
                    code_lines.append(line)
            return '\n'.join(code_lines).strip()
        
        dev_code = extract_code_from_patch(dev_patch)
        llm_code = extract_code_from_patch(llm_patch)
        
        if not dev_code.strip() or not llm_code.strip():
            return {'rating': 'FAIL', 'analysis': 'Empty patch content'}
        
        # Tokenize code for CodeBLEU (split by whitespace and basic tokens)
        def simple_tokenize(code):
            return code.split()
        
        dev_tokens = simple_tokenize(dev_code)
        llm_tokens = simple_tokenize(llm_code)
        
        # Debug: log what we're sending to CodeBLEU
        logging.info(f"CodeBLEU input - Language: {codebleu_lang}")
        logging.info(f"Dev tokens: {len(dev_tokens)}, LLM tokens: {len(llm_tokens)}")
        
        # Calculate CodeBLEU with equal weights (0.25 each)
        result = calc_codebleu(references=[dev_tokens], predictions=[llm_tokens], lang=codebleu_lang, weights=(0.25, 0.25, 0.25, 0.25))
        
        codebleu_score = result['codebleu']
        rating = 'SUCCESS' if codebleu_score >= 0.7 else 'PARTIAL' if codebleu_score >= 0.4 else 'FAIL'
        
        analysis = f"CodeBLEU Score: {codebleu_score:.4f}\n"
        analysis += f"• BLEU (n-gram): {result['ngram_match_score']:.4f}\n"
        analysis += f"• BLEU-weighted: {result['weighted_ngram_match_score']:.4f}\n" 
        analysis += f"• AST Match: {result['syntax_match_score']:.4f}\n"
        analysis += f"• DataFlow Match: {result['dataflow_match_score']:.4f}"
        
        return {'rating': rating, 'analysis': analysis}
        
    except Exception as e:
        logging.error(f"CodeBLEU error: {e}")
        return {'rating': 'FAIL', 'analysis': f"CodeBLEU calculation failed: {str(e)}"}

def analyze_with_gemini(before_code: str, dev_patch: str, llm_patch: str, model_name: str, test_number: str, category: str, language: str) -> dict:
    """Use Gemini to analyze and compare the fixes with simplified criteria."""
    prompt = f"""
    Compare what the LLM did versus what the developer did to fix this flaky {language} test.
    
    Category: {category}
    Model: {model_name}, Test: {test_number}
    
    Developer's fix:
    ```diff
    {dev_patch}
    ```
    
    LLM's fix:
    ```diff
    {llm_patch}
    ```
    
    Compare the two fixes using bullet points:
    
    • What did the developer change?
    • What did the LLM change?
    • How similar are they?
    
    Rate the LLM's performance:
    - SUCCESS: LLM did what the developer did (70%+ similarity in approach/changes)
    - PARTIAL: LLM did about half the changes the developer made
    - FAIL: LLM's approach is completely different or misses the main issue
    
    Format your response exactly like this:
    RATING: [SUCCESS/PARTIAL/FAIL]
    ANALYSIS:
    • Developer changed: [brief description]
    • LLM changed: [brief description]
    • Similarity: [brief comparison]
    """
    
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        # Parse the response
        text = response.text.strip()
        lines = text.split('\n')
        
        rating = 'FAIL'  # default
        for line in lines:
            if line.startswith('RATING:'):
                rating = line.split(':')[1].strip()
                break
        
        # Extract analysis section
        analysis_start = text.find('ANALYSIS:')
        if analysis_start != -1:
            analysis = text[analysis_start + 9:].strip()
        else:
            analysis = 'Analysis not found in response'
        
        return {
            'rating': rating,
            'analysis': analysis
        }
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return {
            'rating': 'FAIL',
            'analysis': f"Error analyzing fixes: {str(e)}"
        }

def save_analysis_results(analyses: list, model_name: str, test_number: str, timestamp: str, file_type: str) -> None:
    """Save all analyses to a single directory with timestamp."""
    try:
        language = get_test_numbering_info(file_type)['language']
        
        # Create analysis directory with timestamp and language
        output_dir = ANALYSIS_DIR / f"{language}_{model_name}_Test{test_number}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Excel-like summary
        df_data = []
        for analysis in analyses:
            df_data.append({
                'Test Case': analysis['test_case'],
                'Category': analysis['category'],
                'Language': language,
                'Dev Fix Summary': analysis['dev_fix_summary'],
                f'{model_name}_Test{test_number}': analysis.get('rating', ''),
                'Analysis': analysis['analysis']
            })
        
        df = pd.DataFrame(df_data)
        excel_file = output_dir / f"analysis_{language}_{model_name}_Test{test_number}_{timestamp}.xlsx"
        df.to_excel(excel_file, index=False)
        
        # Also save detailed text report
        text_file = output_dir / f"detailed_analysis_{language}_{model_name}_Test{test_number}_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Analysis Report: {language} - {model_name} Test {test_number}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for analysis in analyses:
                f.write(f"Test Case: {analysis['test_case']}\n")
                f.write(f"Category: {analysis['category']}\n")
                f.write(f"Developer Fix: {analysis['dev_fix_summary']}\n")
                f.write(f"Rating: {analysis.get('rating', 'N/A')}\n")
                f.write(f"Analysis:\n{analysis['analysis']}\n")
                f.write("-" * 80 + "\n\n")
        
        logging.info(f"Analysis results saved to {output_dir}")
        print(f"Results saved to: {output_dir}")
    except Exception as e:
        logging.error(f"Error saving analysis results: {e}")

def save_combined_analysis_results(all_analyses: list, model_name: str, test_numbers: list, timestamp: str, file_type: str) -> None:
    """Save combined analysis results for all tests to a single directory."""
    try:
        language = get_test_numbering_info(file_type)['language']
        
        # Create combined analysis directory with timestamp and language
        test_list = "_".join(test_numbers)
        output_dir = ANALYSIS_DIR / f"{language}_{model_name}_AllTests_{test_list}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Excel-like summary with all tests
        df_data = []
        for analysis in all_analyses:
            df_data.append({
                'Test Case': analysis['test_case'],
                'Test Number': analysis['test_number'],
                'Category': analysis['category'],
                'Language': language,
                'Dev Fix Summary': analysis['dev_fix_summary'],
                f'{model_name}_Rating': analysis.get('rating', ''),
                'Analysis': analysis['analysis']
            })
        
        df = pd.DataFrame(df_data)
        excel_file = output_dir / f"combined_analysis_{language}_{model_name}_AllTests_{timestamp}.xlsx"
        df.to_excel(excel_file, index=False)
        
        # Also save detailed text report
        text_file = output_dir / f"combined_detailed_analysis_{language}_{model_name}_AllTests_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"Combined Analysis Report: {language} - {model_name} All Tests {', '.join(test_numbers)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group by test number for better organization
            analyses_by_test = {}
            for analysis in all_analyses:
                test_num = analysis['test_number']
                if test_num not in analyses_by_test:
                    analyses_by_test[test_num] = []
                analyses_by_test[test_num].append(analysis)
            
            for test_num in sorted(analyses_by_test.keys()):
                f.write(f"TEST {test_num} RESULTS\n")
                f.write("=" * 40 + "\n\n")
                
                for analysis in analyses_by_test[test_num]:
                    f.write(f"Test Case: {analysis['test_case']}\n")
                    f.write(f"Category: {analysis['category']}\n")
                    f.write(f"Developer Fix: {analysis['dev_fix_summary']}\n")
                    f.write(f"Rating: {analysis.get('rating', 'N/A')}\n")
                    f.write(f"Analysis:\n{analysis['analysis']}\n")
                    f.write("-" * 40 + "\n\n")
                
                f.write("\n")
        
        logging.info(f"Combined analysis results saved to {output_dir}")
        print(f"\nCombined results saved to: {output_dir}")
    except Exception as e:
        logging.error(f"Error saving combined analysis results: {e}")

def choose_excel_filtering() -> bool:
    """Let user choose whether to use Excel file filtering or process all test cases."""
    excel_path = Path("/analysis_results/Java_o4_mini_Test10_20250602_154303/analysis_Java_o4_mini_Test10_20250602_154303.xlsx")
    
    if not excel_path.exists():
        print("📄 No Manual_Comparison.xlsx file found - will process all available test cases.")
        return False
    
    print("\n" + "="*60)
    print("EXCEL FILE FILTERING OPTIONS")
    print("="*60)
    print(f"📄 Found Excel file: {excel_path}")
    
    # Try to read and show info about the Excel file
    try:
        df = pd.read_excel(excel_path)
        test_case_count = len(df.iloc[:, 0].dropna())
        print(f"📊 Excel file contains {test_case_count} test cases")
    except Exception as e:
        print(f"⚠️  Could not read Excel file: {e}")
        print("Will proceed with option to process all test cases.")
    
    print("\nFiltering options:")
    print("1. Use Excel file filtering (only process test cases listed in Excel)")
    print("2. Process ALL available test cases (ignore Excel file)")
    print("="*60)
    
    while True:
        try:
            choice = input("Choose filtering option (1 or 2): ").strip()
            
            if choice == '1':
                print("✅ Using Excel file filtering")
                return True
            elif choice == '2':
                print("✅ Processing ALL available test cases")
                return False
            else:
                print("❌ Please enter 1 or 2")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            return False

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
        
        # Choose file type first
        selected_file_type = choose_file_type()
        if not selected_file_type:
            return
        
        # Choose analysis method
        print("\n" + "="*60)
        print("ANALYSIS METHOD SELECTION")
        print("="*60)
        methods = []
        if os.getenv('GEMINI_API_KEY'):
            methods.append(('gemini', 'Gemini LLM Analysis'))
        if CODEBLEU_AVAILABLE:
            methods.append(('codebleu', 'CodeBLEU Metric Analysis'))
        
        if not methods:
            print("❌ No analysis methods available!")
            print("   - Set GEMINI_API_KEY for Gemini analysis")
            print("   - Install codebleu for CodeBLEU analysis")
            return
            
        for i, (_, desc) in enumerate(methods, 1):
            print(f"{i}. {desc}")
        
        while True:
            try:
                if len(methods) == 1:
                    analysis_method = methods[0][0]
                    print(f"✅ Auto-selected: {methods[0][1]}")
                    break
                choice = int(input("Choose analysis method (1-{}): ".format(len(methods))))
                if 1 <= choice <= len(methods):
                    analysis_method = methods[choice-1][0]
                    break
                print("Invalid choice.")
            except ValueError:
                print("Please enter a number.")
        
        numbering_info = get_test_numbering_info(selected_file_type)
        language = numbering_info['language']
        file_ext = numbering_info['file_ext']
        
        print(f"\n🔍 Analyzing {language} tests")
        print(f"📁 Looking in: {RESULTS_DIR / selected_file_type}")
        
        # Choose whether to use Excel filtering
        use_excel_filtering = choose_excel_filtering()
        
        # Read the Excel file only if user chose to use filtering
        test_cases = set()
        if use_excel_filtering:
            excel_path = Path("Manual_Comparison.xlsx")
            if excel_path.exists():
                try:
                    df = pd.read_excel(excel_path)
                    test_cases = set(df.iloc[:, 0].dropna().astype(str).tolist())
                    logging.info(f"Using Excel filtering with {len(test_cases)} test cases")
                    print(f"📊 Will filter to {len(test_cases)} test cases from Excel file")
                except Exception as e:
                    logging.error(f"Error reading Excel file: {e}")
                    print(f"⚠️  Error reading Excel file: {e}")
                    print("Proceeding without Excel filtering...")
                    test_cases = set()
            else:
                print("⚠️  Excel file not found - proceeding without filtering")
                test_cases = set()
        else:
            print("📊 Processing ALL available test cases (no Excel filtering)")
        
        # Let user select model and test number
        selected_model, selected_tests = select_model_and_test(selected_file_type)
        if not selected_model or not selected_tests:
            print("No valid selection made. Exiting.")
            return
        
        print(f"\n🔍 Analyzing: {language} - {selected_model} Test(s) {', '.join(selected_tests)}")
        print(f"🔬 Method: {analysis_method.upper()}")
        print("=" * 60)
        
        # Process each selected test
        all_analyses = []
        all_files_processed = 0
        
        for selected_test in selected_tests:
            print(f"\n{'='*80}")
            print(f"🔍 PROCESSING: {language} - {selected_model} Test {selected_test}")
            print(f"{'='*80}")
            
            # Collect all files that will be processed for this test
            logging.info(f"\nCollecting files to process for Test {selected_test}...")
            files_to_process = []
            data_dir = RESULTS_DIR / selected_file_type
            
            print(f"🔍 Scanning directory: {data_dir}")
            
            for category_dir in data_dir.iterdir():
                if not category_dir.is_dir():
                    continue
                    
                print(f"  📁 Category: {category_dir.name}")
                    
                for test_case_dir in category_dir.iterdir():
                    if not test_case_dir.is_dir():
                        continue
                    
                    print(f"    📂 Test case: {test_case_dir.name}")
                    
                    # If we have Excel file filtering enabled, filter by test cases in it
                    if use_excel_filtering and test_cases and test_case_dir.name not in test_cases:
                        print(f"      ⏭️  Skipping (not in Excel filter)")
                        continue
                    elif use_excel_filtering and test_cases:
                        print(f"      ✅ Included (in Excel filter)")
                    elif not use_excel_filtering:
                        print(f"      ✅ Included (no Excel filtering)")
                    else:
                        print(f"      ✅ Included (Excel filtering disabled or empty)")
                    
                    before_file = test_case_dir / f"before{file_ext}"
                    dev_patch_file = test_case_dir / "developer_patch.diff"
                    
                    print(f"      🔍 Looking for:")
                    print(f"        📄 before file: {before_file}")
                    print(f"        📄 dev patch: {dev_patch_file}")
                    print(f"        📄 before exists: {before_file.exists()}")
                    print(f"        📄 dev patch exists: {dev_patch_file.exists()}")
                    
                    # Look for the specific model and test combination
                    target_folder = f"{selected_model}_Test{selected_test}"
                    model_test_dir = test_case_dir / target_folder
                    
                    print(f"        📁 model dir: {model_test_dir}")
                    print(f"        📁 model dir exists: {model_test_dir.exists()}")
                    
                    if before_file.exists() and dev_patch_file.exists() and model_test_dir.exists():
                        print(f"        ✅ MATCH! Adding to processing list")
                        files_to_process.append({
                            'category': category_dir.name,
                            'test_case': test_case_dir.name,
                            'path': test_case_dir,
                            'model_dir': model_test_dir,
                            'test_number': selected_test
                        })
                    else:
                        print(f"        ❌ SKIP: Missing required files/dirs")

            print(f"\n📊 Found {len(files_to_process)} {language} test cases with {selected_model} Test {selected_test}")
            all_files_processed += len(files_to_process)
            
            if not files_to_process:
                print(f"No matching test cases found for Test {selected_test}!")
                continue
            
            # Process the files and collect analyses for this test
            logging.info(f"\nStarting analysis for Test {selected_test}...")
            test_analyses = []
            
            for idx, file_info in enumerate(files_to_process, 1):
                print(f"\nProcessing {idx}/{len(files_to_process)}: {file_info['test_case']} (Test {selected_test})")
                logging.info(f"Analyzing {file_info['test_case']} for Test {selected_test}")
                
                test_dir = file_info['path']
                before_code = read_file_content(test_dir / f"before{file_ext}")
                dev_patch = read_file_content(test_dir / "developer_patch.diff")
                
                # Read LLM patch
                llm_patch_file = find_diff_file(file_info['model_dir'])
                llm_patch = read_file_content(llm_patch_file)
                
                if not llm_patch:
                    logging.warning(f"No LLM patch found for {file_info['test_case']} Test {selected_test}")
                    continue
                
                # Analyze with selected method
                if analysis_method == 'codebleu':
                    result = calculate_codebleu_score(dev_patch, llm_patch, language)
                else:
                    result = analyze_with_gemini(before_code, dev_patch, llm_patch, 
                                               selected_model, selected_test, 
                                               file_info['category'], language)
                
                analysis = {
                    'category': file_info['category'],
                    'test_case': file_info['test_case'],
                    'test_number': selected_test,
                    'dev_fix_summary': dev_patch.split('\n')[0] if dev_patch else "No developer fix found",
                    'rating': result['rating'],
                    'analysis': result['analysis']
                }
                
                test_analyses.append(analysis)
                all_analyses.append(analysis)
                print(f"  Result: {result['rating']}")
            
            # Save analyses for this specific test
            save_analysis_results(test_analyses, selected_model, selected_test, timestamp, selected_file_type)
            
            # Print summary for this test
            total_test = len(test_analyses)
            success_count_test = sum(1 for a in test_analyses if a['rating'] == 'SUCCESS')
            partial_count_test = sum(1 for a in test_analyses if a['rating'] == 'PARTIAL')
            fail_count_test = sum(1 for a in test_analyses if a['rating'] == 'FAIL')
            
            print(f"\n" + "-" * 60)
            print(f"SUMMARY for {language} - {selected_model} Test {selected_test}")
            print(f"-" * 60)
            print(f"Test cases: {total_test}")
            print(f"✅ SUCCESS: {success_count_test} ({success_count_test/total_test*100:.1f}%)")
            print(f"⚠️  PARTIAL: {partial_count_test} ({partial_count_test/total_test*100:.1f}%)")
            print(f"❌ FAIL: {fail_count_test} ({fail_count_test/total_test*100:.1f}%)")
        
        # Print overall summary for all tests
        if len(selected_tests) > 1:
            total_overall = len(all_analyses)
            success_count_overall = sum(1 for a in all_analyses if a['rating'] == 'SUCCESS')
            partial_count_overall = sum(1 for a in all_analyses if a['rating'] == 'PARTIAL')
            fail_count_overall = sum(1 for a in all_analyses if a['rating'] == 'FAIL')
            
            print(f"\n" + "=" * 80)
            print(f"OVERALL SUMMARY for {language} - {selected_model} All Tests {', '.join(selected_tests)}")
            print(f"=" * 80)
            print(f"Total test cases processed: {total_overall}")
            print(f"Total files processed: {all_files_processed}")
            if total_overall > 0:
                print(f"✅ SUCCESS: {success_count_overall} ({success_count_overall/total_overall*100:.1f}%)")
                print(f"⚠️  PARTIAL: {partial_count_overall} ({partial_count_overall/total_overall*100:.1f}%)")
                print(f"❌ FAIL: {fail_count_overall} ({fail_count_overall/total_overall*100:.1f}%)")
        
        logging.info(f"\nAnalysis completed for all {len(selected_tests)} tests")
        logging.info("=" * 80)
        
        # Save combined analysis results for all tests
        save_combined_analysis_results(all_analyses, selected_model, selected_tests, timestamp, selected_file_type)
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 