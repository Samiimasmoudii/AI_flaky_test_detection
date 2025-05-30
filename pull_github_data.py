import logging
import os
from pathlib import Path
from inc.config import RESULTS_DIR, SAMPLE_SIZE
from inc.github_utils import process_repo, process_matched_file, GitHubAccessError
from inc.processor import setup_logging
import pandas as pd

def get_available_csv_files():
    """Get list of available CSV files in raw_data_csv directory"""
    csv_dir = Path("raw_data_csv")
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")
    
    csv_files = list(csv_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")
    
    return csv_files

def choose_csv_file():
    """Display available CSV files and let user choose"""
    csv_files = get_available_csv_files()
    
    print("\n" + "="*60)
    print("AVAILABLE CSV FILES")
    print("="*60)
    
    for i, csv_file in enumerate(csv_files, 1):
        file_size = csv_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        
        try:
            df = pd.read_csv(csv_file)
            row_count = len(df)
            print(f"{i}. {csv_file.name}")
            print(f"   Size: {size_mb:.1f} MB | Rows: {row_count:,}")
        except Exception as e:
            print(f"{i}. {csv_file.name}")
            print(f"   Size: {size_mb:.1f} MB | Error reading file: {str(e)}")
        print()
    
    print("="*60)
    
    while True:
        try:
            choice = input(f"Choose a CSV file (1-{len(csv_files)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                exit(0)
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(csv_files):
                selected_file = csv_files[choice_num - 1]
                print(f"\n✅ Selected: {selected_file.name}")
                return selected_file
            else:
                print(f"❌ Please enter a number between 1 and {len(csv_files)}")
        
        except ValueError:
            print("❌ Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)

def detect_file_type(csv_path):
    """Detect if the CSV contains Python or Java tests"""
    try:
        df = pd.read_csv(csv_path, nrows=10)
        columns = df.columns.tolist()
        
        if 'Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)' in columns:
            return 'python'
        elif 'Fully-Qualified Test Name (packageName.ClassName.methodName)' in columns:
            return 'java'
        else:
            # Try to detect based on test name patterns
            test_col = None
            for col in columns:
                if 'test' in col.lower() and 'name' in col.lower():
                    test_col = col
                    break
            
            if test_col:
                sample_tests = df[test_col].dropna().head(5)
                if any('::' in str(test) for test in sample_tests):
                    return 'python'
                elif any('.' in str(test) and not '::' in str(test) for test in sample_tests):
                    return 'java'
        
        return 'unknown'
    except Exception as e:
        logging.warning(f"Could not detect file type: {e}")
        return 'unknown'

def filter_dataset(df):
    """Filter dataset to only keep reliable test entries based on IDoFT status definitions"""
    # Based on IDoFT documentation: https://github.com/TestingResearchIllinois/idoft
    # Only keep tests with these reliable statuses 
    reliable_statuses = ['Opened', 'Accepted', 'DeveloperFixed']
    
    # Exclude problematic statuses that indicate tests/repos with issues
    problematic_statuses = [
        'Deleted',          # Test removed from repository
        'MovedOrRenamed',   # Test has different name (older SHA)
        'RepoArchived',     # Repository is archived
        'RepoDeleted',      # Repository no longer exists
        'Unmaintained',     # Repository inactive for 2+ years
        'Blank'             # Not yet inspected (as requested)
    ]
    
    initial_count = len(df)
    
    # Check if Status column exists
    if 'Status' not in df.columns:
        print("⚠️  No 'Status' column found - skipping filtering")
        return df
    
    # Show status distribution before filtering
    print(f"\n📊 Status distribution before filtering:")
    status_counts = df['Status'].value_counts()
    for status, count in status_counts.head(10).items():
        print(f"   {status}: {count:,}")
    
    # Filter to only keep reliable statuses
    df_filtered = df[df['Status'].isin(reliable_statuses)]
    
    print(f"\n📊 Dataset filtering results (IDoFT-based):")
    print(f"   Original entries: {initial_count:,}")
    print(f"   After filtering: {len(df_filtered):,}")
    print(f"   Removed: {initial_count - len(df_filtered):,} unreliable entries")
    print(f"   ✅ Kept statuses: {', '.join(reliable_statuses)}")
    print(f"   ❌ Excluded statuses: {', '.join(problematic_statuses)}")
    
    if len(df_filtered) == 0:
        print("❌ No entries remain after filtering! Check if Status column has the expected values.")
        print("Available Status values:", df['Status'].value_counts().to_dict())
    
    return df_filtered

def interactive_file_matching(failed_entries, file_type):
    """Allow user to manually select matching files for failed entries"""
    if not failed_entries:
        return 0
    
    print(f"\n{'='*60}")
    print(f"🤔 INTERACTIVE FILE MATCHING")
    print(f"{'='*60}")
    print(f"Found {len(failed_entries)} entries where automatic matching failed.")
    print(f"You can now manually select the correct files for each test.")
    print(f"This will help improve the dataset completeness!")
    
    proceed = input(f"\nProceed with interactive matching? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Skipping interactive matching.")
        return 0
    
    success_count = 0
    
    for i, entry in enumerate(failed_entries, 1):
        row = entry['row']
        available_files = entry['available_test_files']
        expected_file = entry['expected_test_file']
        
        owner, repo = row['Project URL'].rstrip("/").split("/")[-2:]
        
        if file_type == 'python':
            test_name = row.get('Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)', '')
        else:
            test_name = row.get('Fully-Qualified Test Name (packageName.ClassName.methodName)', '')
        
        print(f"\n{'='*60}")
        print(f"Entry {i}/{len(failed_entries)}: {owner}/{repo}")
        print(f"{'='*60}")
        print(f"Test name: {test_name}")
        print(f"Expected: {expected_file}")
        print(f"Category: {row['Category']}")
        
        if not available_files:
            print("❌ No test files available in this commit.")
            continue
        
        print(f"\nAvailable {file_type} files:")
        for j, file_path in enumerate(available_files, 1):
            print(f"  {j}. {file_path}")
        
        print(f"  0. Skip this entry")
        print(f"  q. Quit interactive matching")
        
        while True:
            try:
                choice = input(f"\nSelect file (1-{len(available_files)}, 0 to skip, q to quit): ").strip().lower()
                
                if choice == 'q':
                    print(f"Quitting interactive matching. Processed {success_count} additional entries.")
                    return success_count
                
                if choice == '0':
                    print("Skipping this entry.")
                    break
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_files):
                    selected_file_path = available_files[choice_num - 1]
                    
                    # Find the file object in changed_files
                    selected_file = None
                    for file_obj in entry['changed_files']:
                        if file_obj.get("filename", "") == selected_file_path:
                            selected_file = file_obj
                            break
                    
                    if selected_file:
                        print(f"Processing {selected_file_path}...")
                        try:
                            success = process_matched_file(selected_file, row, entry['parent_sha'], file_type)
                            if success:
                                success_count += 1
                                print(f"✅ Successfully processed! ({success_count} additional successes)")
                            else:
                                print(f"❌ Failed to process file (download/save error)")
                        except Exception as e:
                            print(f"❌ Error processing file: {str(e)}")
                            logging.error(f"Error in interactive processing: {str(e)}")
                    else:
                        print(f"❌ Could not find file object for {selected_file_path}")
                    break
                else:
                    print(f"❌ Please enter a number between 1 and {len(available_files)}")
            
            except ValueError:
                print("❌ Please enter a valid number, 0, or 'q'")
            except KeyboardInterrupt:
                print(f"\n\nInterrupted. Processed {success_count} additional entries.")
                return success_count
    
    print(f"\n✅ Interactive matching complete! Processed {success_count} additional entries.")
    return success_count

def main():
    setup_logging()
    logging.info("Starting GitHub data pull")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Let user choose CSV file
    try:
        selected_csv = choose_csv_file()
    except (FileNotFoundError, KeyboardInterrupt) as e:
        print(f"Error: {e}")
        return
    
    # Detect file type
    file_type = detect_file_type(selected_csv)
    print(f"\n🔍 Detected file type: {file_type.upper()}")
    
    if file_type == 'unknown':
        print("⚠️  Could not automatically detect file type. Assuming Java format.")
        file_type = 'txt'
    
    # Load and filter CSV data
    df = pd.read_csv(selected_csv)
    df = filter_dataset(df)  # Filter to only keep reliable entries
    
    if len(df) == 0:
        print("❌ No data to process after filtering. Exiting.")
        return
    
    if file_type == 'python':
        required_columns = ['Project URL', 'SHA Detected', 'Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)']
    else:  # java
        required_columns = ['Project URL', 'SHA Detected', 'Fully-Qualified Test Name (packageName.ClassName.methodName)']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return
    
    df = df.dropna(subset=required_columns)
    
    print(f"\n📊 Total entries to process: {len(df):,}")
    print(f"🚀 Processing all {len(df):,} repositories...")
    print(f"📁 Results will be saved to: {RESULTS_DIR / f'{file_type}_data'}")
    print(f"📂 Directory structure: {file_type}_data/{{category}}/{{test_name}}/")
    print(f"   - before.{file_type.replace('python', 'py')}")
    print(f"   - after.{file_type.replace('python', 'py')}")
    print(f"   - developer_patch.diff")
    print(f"   - metadata.txt")
    
    # Track success/failure statistics
    success_count = 0
    failed_entries = []  # Store entries that failed matching for interactive processing
    error_counts = {
        'github_access': 0,
        'file_not_found': 0,
        'network_errors': 0,
        'other_errors': 0
    }
    
    categories_found = set()
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        print(f"\n[{i}/{len(df)}] Processing: {row['Project URL'].split('/')[-2:]}")
        
        try:
            result = process_repo(row, file_type)
            
            if result is True:  # Success
                success_count += 1
                categories_found.add(row['Category'])
                print(f"✅ Success ({success_count}/{i}) - Category: {row['Category']}")
            elif isinstance(result, dict) and result.get('status') == 'no_match':
                # Store for interactive processing later
                failed_entries.append(result)
                print(f"⚠️  No matching files found (saved for interactive review)")
                error_counts['file_not_found'] += 1
            else:
                print(f"⚠️  Processing failed")
                error_counts['file_not_found'] += 1
                
        except GitHubAccessError as e:
            error_counts['github_access'] += 1
            print(f"🚫 GitHub access error: {str(e)}")
            
        except ConnectionError as e:
            error_counts['network_errors'] += 1
            print(f"🌐 Network error: {str(e)}")
            
        except Exception as e:
            error_counts['other_errors'] += 1
            print(f"❌ Unexpected error: {str(e)}")
            logging.error(f"Unexpected error processing item {i}: {str(e)}")
        
        # Show running statistics every 50 items
        if i % 50 == 0 or i == len(df):
            success_rate = (success_count / i) * 100
            print(f"\n📊 Progress: {i}/{len(df)} | Success rate: {success_rate:.1f}% ({success_count} successful)")
            if categories_found:
                print(f"📂 Categories found: {', '.join(sorted(categories_found))}")
    
    # Automatic processing complete - now do interactive matching
    print(f"\n🎯 Automatic processing complete!")
    
    if failed_entries:
        print(f"📋 Found {len(failed_entries)} entries that could benefit from manual file selection.")
        interactive_successes = interactive_file_matching(failed_entries, file_type)
        success_count += interactive_successes
        error_counts['file_not_found'] -= interactive_successes
    
    # Final summary
    total_errors = sum(error_counts.values())
    success_rate = (success_count / len(df)) * 100
    
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total repositories processed: {len(df):,}")
    print(f"✅ Successful: {success_count:,} ({success_rate:.1f}%)")
    print(f"❌ Failed: {total_errors:,} ({(total_errors/len(df)*100):.1f}%)")
    print(f"\nError breakdown:")
    print(f"  🚫 GitHub access issues: {error_counts['github_access']:,}")
    print(f"  📄 No matching files: {error_counts['file_not_found']:,}")
    print(f"  🌐 Network errors: {error_counts['network_errors']:,}")
    print(f"  ⚠️  Other errors: {error_counts['other_errors']:,}")
    
    if categories_found:
        print(f"\n📂 Categories with data:")
        for category in sorted(categories_found):
            category_path = RESULTS_DIR / f'{file_type}_data' / category
            if category_path.exists():
                test_count = len(list(category_path.iterdir()))
                print(f"   📁 {category}: {test_count:,} tests")
    
    print(f"\n📁 Results saved to: {RESULTS_DIR / f'{file_type}_data'}")

if __name__ == "__main__":
    main() 