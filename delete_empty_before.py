import os
import shutil
from pathlib import Path
import logging
from inc.config import RESULTS_DIR

def is_empty_file(file_path):
    """Check if a file is empty or contains only whitespace"""
    try:
        if not file_path.exists():
            return True
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return len(content) == 0
    except Exception as e:
        logging.warning(f"Error reading file {file_path}: {e}")
        return True  # Consider unreadable files as empty

def find_empty_before_files(file_type='all'):
    """Find all test directories with empty before files
    
    Args:
        file_type: 'python', 'java', or 'all' to specify which file types to check
    
    Returns:
        dict: Statistics about empty files found
    """
    empty_dirs = []
    stats = {
        'python': {'total': 0, 'empty': 0, 'empty_dirs': []},
        'java': {'total': 0, 'empty': 0, 'empty_dirs': []}
    }
    
    # Determine which data directories to check
    data_dirs = []
    if file_type in ['python', 'all']:
        python_dir = RESULTS_DIR / 'py_data'
        if python_dir.exists():
            data_dirs.append(('python', python_dir))
    
    if file_type in ['java', 'all']:
        java_dir = RESULTS_DIR / 'java_data'
        if java_dir.exists():
            data_dirs.append(('java', java_dir))
    
    for lang, data_dir in data_dirs:
        logging.info(f"Scanning {lang} data directory: {data_dir}")
        
        # Walk through category directories
        for category_dir in data_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            # Walk through test directories
            for test_dir in category_dir.iterdir():
                if not test_dir.is_dir():
                    continue
                
                stats[lang]['total'] += 1
                
                # Check for empty before file
                file_ext = '.py' if lang == 'python' else '.java'
                before_file = test_dir / f'before{file_ext}'
                
                if is_empty_file(before_file):
                    stats[lang]['empty'] += 1
                    stats[lang]['empty_dirs'].append(test_dir)
                    empty_dirs.append({
                        'path': test_dir,
                        'language': lang,
                        'category': category_dir.name,
                        'test_name': test_dir.name,
                        'before_file': before_file
                    })
    
    return empty_dirs, stats

def delete_empty_before_directories(file_type='all', dry_run=True):
    """Delete test directories that have empty before files
    
    Args:
        file_type: 'python', 'java', or 'all' to specify which file types to clean
        dry_run: If True, only show what would be deleted without actually deleting
    
    Returns:
        dict: Summary of deletion results
    """
    empty_dirs, stats = find_empty_before_files(file_type)
    
    if not empty_dirs:
        print("✅ No test directories with empty before files found!")
        return {'deleted': 0, 'failed': 0, 'total_found': 0}
    
    print(f"\n📊 Found {len(empty_dirs)} test directories with empty before files:")
    
    # Group by language and category for reporting
    by_category = {}
    for entry in empty_dirs:
        lang = entry['language']
        category = entry['category']
        key = f"{lang}_{category}"
        if key not in by_category:
            by_category[key] = []
        by_category[key].append(entry)
    
    for key, entries in by_category.items():
        lang, category = key.split('_', 1)
        print(f"  📁 {lang.upper()} - {category}: {len(entries)} empty tests")
    
    if dry_run:
        print(f"\n🔍 DRY RUN MODE - No files will be deleted")
        print(f"Run with dry_run=False to actually delete these directories")
        return {'deleted': 0, 'failed': 0, 'total_found': len(empty_dirs)}
    
    # Confirm deletion
    print(f"\n⚠️  WARNING: This will permanently delete {len(empty_dirs)} test directories!")
    confirm = input("Type 'DELETE' to confirm: ").strip()
    
    if confirm != 'DELETE':
        print("❌ Deletion cancelled.")
        return {'deleted': 0, 'failed': 0, 'total_found': len(empty_dirs)}
    
    # Perform deletion
    deleted_count = 0
    failed_count = 0
    
    print(f"\n🗑️  Deleting {len(empty_dirs)} directories...")
    
    for i, entry in enumerate(empty_dirs, 1):
        test_dir = entry['path']
        try:
            if test_dir.exists():
                shutil.rmtree(test_dir)
                deleted_count += 1
                print(f"  [{i}/{len(empty_dirs)}] ✅ Deleted: {entry['language']}/{entry['category']}/{entry['test_name']}")
            else:
                print(f"  [{i}/{len(empty_dirs)}] ⚠️  Already gone: {entry['test_name']}")
                
        except Exception as e:
            failed_count += 1
            print(f"  [{i}/{len(empty_dirs)}] ❌ Failed: {entry['test_name']} - {str(e)}")
            logging.error(f"Failed to delete {test_dir}: {str(e)}")
    
    # Clean up empty category directories
    cleanup_empty_categories(file_type)
    
    return {
        'deleted': deleted_count,
        'failed': failed_count,
        'total_found': len(empty_dirs)
    }

def cleanup_empty_categories(file_type='all'):
    """Remove empty category directories after cleanup"""
    data_dirs = []
    if file_type in ['python', 'all']:
        python_dir = RESULTS_DIR / 'python_data'
        if python_dir.exists():
            data_dirs.append(python_dir)
    
    if file_type in ['java', 'all']:
        java_dir = RESULTS_DIR / 'java_data'
        if java_dir.exists():
            data_dirs.append(java_dir)
    
    removed_categories = []
    
    for data_dir in data_dirs:
        for category_dir in data_dir.iterdir():
            if category_dir.is_dir():
                # Check if category directory is empty
                try:
                    if not any(category_dir.iterdir()):
                        category_dir.rmdir()
                        removed_categories.append(category_dir.name)
                        print(f"🧹 Removed empty category directory: {category_dir.name}")
                except Exception as e:
                    logging.warning(f"Could not remove empty category {category_dir}: {e}")
    
    return removed_categories

def analyze_empty_before_files(file_type='all'):
    """Analyze and report on empty before files without deleting anything"""
    empty_dirs, stats = find_empty_before_files(file_type)
    
    print(f"\n📊 EMPTY BEFORE FILES ANALYSIS")
    print(f"{'='*50}")
    
    total_tests = 0
    total_empty = 0
    
    for lang in ['python', 'java']:
        if stats[lang]['total'] > 0:
            empty_pct = (stats[lang]['empty'] / stats[lang]['total']) * 100
            print(f"\n🐍 {lang.upper()} Tests:")
            print(f"   Total tests: {stats[lang]['total']:,}")
            print(f"   Empty before files: {stats[lang]['empty']:,} ({empty_pct:.1f}%)")
            print(f"   Usable tests: {stats[lang]['total'] - stats[lang]['empty']:,}")
            
            total_tests += stats[lang]['total']
            total_empty += stats[lang]['empty']
    
    if total_tests > 0:
        overall_empty_pct = (total_empty / total_tests) * 100
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"   Total tests: {total_tests:,}")
        print(f"   Empty before files: {total_empty:,} ({overall_empty_pct:.1f}%)")
        print(f"   Usable tests: {total_tests - total_empty:,}")
        
        if total_empty > 0:
            print(f"\n💡 Recommendation: Run delete_empty_before_directories() to clean up {total_empty:,} unusable test cases")
    else:
        print("No test data found in results directory.")
    
    # Show breakdown by category
    if empty_dirs:
        print(f"\n📂 BREAKDOWN BY CATEGORY:")
        by_category = {}
        for entry in empty_dirs:
            lang = entry['language']
            category = entry['category']
            key = f"{lang}_{category}"
            if key not in by_category:
                by_category[key] = 0
            by_category[key] += 1
        
        for key, count in sorted(by_category.items(), key=lambda x: x[1], reverse=True):
            lang, category = key.split('_', 1)
            print(f"   {lang.upper()}/{category}: {count:,} empty tests")
    
    return stats

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean up test directories with empty before files')
    parser.add_argument('--file-type', choices=['python', 'java', 'all'], default='all',
                       help='File type to process (default: all)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze without deleting anything')
    parser.add_argument('--delete', action='store_true',
                       help='Actually delete files (default is dry run)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if args.analyze_only:
        analyze_empty_before_files(args.file_type)
    else:
        dry_run = not args.delete
        result = delete_empty_before_directories(args.file_type, dry_run)
        
        print(f"\n📊 CLEANUP SUMMARY:")
        print(f"   Found: {result['total_found']:,} directories with empty before files")
        print(f"   Deleted: {result['deleted']:,}")
        print(f"   Failed: {result['failed']:,}")
        
        if dry_run and result['total_found'] > 0:
            print(f"\n💡 To actually delete files, run with --delete flag")

if __name__ == "__main__":
    main()
