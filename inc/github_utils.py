import requests
import base64
import difflib
import logging
from pathlib import Path
from .config import GITHUB_API, HEADERS, RESULTS_DIR
import time
import re

def is_test_file(file_path, file_type='java'):
    """Check if a file is actually a test file (not production code)"""
    if not file_path:
        return False
    
    if file_type == 'python':
        # For Python: check if it's in test directories or has test in name
        return (
            '/test' in file_path.lower() or 
            file_path.lower().startswith('test') or
            'test_' in file_path.lower() or
            '_test.' in file_path.lower()
        )
    else:  # Java
        # For Java: check if it's in src/test/java or has Test in class name
        return (
            '/src/test/java/' in file_path or
            file_path.endswith('Test.java') or
            file_path.endswith('Tests.java') or
            'Test' in file_path.split('/')[-1]  # Test in filename
        )

class GitHubAccessError(Exception):
    """Custom exception for GitHub access issues"""
    pass

def handle_github_response(response, repo_info="repository"):
    """Handle GitHub API response and provide detailed error information"""
    if response.status_code == 200:
        return True
    
    error_messages = {
        404: f"Repository not found or deleted: {repo_info}",
        403: f"Access forbidden (private repo or rate limit): {repo_info}",
        401: f"Authentication failed: {repo_info}",
        451: f"Repository unavailable for legal reasons: {repo_info}",
        500: f"GitHub server error: {repo_info}",
        502: f"GitHub bad gateway: {repo_info}",
        503: f"GitHub service unavailable: {repo_info}"
    }
    
    error_msg = error_messages.get(response.status_code, f"GitHub API error {response.status_code}: {repo_info}")
    
    # Check if it's a rate limit issue
    if response.status_code == 403:
        remaining = response.headers.get('X-RateLimit-Remaining', '0')
        reset_time = response.headers.get('X-RateLimit-Reset')
        
        if remaining == '0' and reset_time:
            # This is definitely a rate limit error
            import datetime
            reset_timestamp = int(reset_time)
            reset_datetime = datetime.datetime.fromtimestamp(reset_timestamp)
            wait_seconds = reset_timestamp - time.time()
            
            if wait_seconds > 0:
                print(f"\n⏰ Rate limit exceeded! Waiting {wait_seconds:.0f} seconds until {reset_datetime.strftime('%H:%M:%S')}...")
                time.sleep(wait_seconds + 5)  # Add 5 seconds buffer
                print("✅ Rate limit reset - continuing...")
                return "retry"  # Signal that we should retry the request
            
        error_msg += f" (Rate limit remaining: {remaining})"
    
    raise GitHubAccessError(error_msg)

def get_changed_files(owner, repo, sha):
    """Get changed files with robust error handling"""
    repo_info = f"{owner}/{repo}@{sha}"
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
            response = requests.get(url, headers=HEADERS, timeout=30)
            
            result = handle_github_response(response, repo_info)
            if result == "retry" and attempt < max_retries - 1:
                continue  # Retry the request
            elif result == "retry":
                raise GitHubAccessError(f"Rate limit issues persist for: {repo_info}")
            
            data = response.json()
            files = data.get("files", [])
            
            # Check if commit has parent (some commits might not have parents)
            parents = data.get("parents", [])
            if not parents:
                raise GitHubAccessError(f"Commit {sha} has no parent commits in {owner}/{repo}")
            
            parent_sha = parents[0]["sha"]
            return files, parent_sha
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logging.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                time.sleep(2)
                continue
            raise GitHubAccessError(f"Timeout accessing commit: {repo_info}")
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logging.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                time.sleep(2)
                continue
            raise GitHubAccessError(f"Network error accessing commit: {repo_info}")
        except requests.exceptions.RequestException as e:
            raise GitHubAccessError(f"Request error accessing commit {repo_info}: {str(e)}")
        except (KeyError, ValueError) as e:
            raise GitHubAccessError(f"Invalid response format for commit {repo_info}: {str(e)}")
        except GitHubAccessError:
            raise  # Re-raise our custom errors

def download_file_content(owner, repo, filepath, sha):
    """Download file content with robust error handling"""
    repo_info = f"{owner}/{repo}:{filepath}@{sha}"
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{filepath}?ref={sha}"
            response = requests.get(url, headers=HEADERS, timeout=30)
            
            result = handle_github_response(response, repo_info)
            if result == "retry" and attempt < max_retries - 1:
                continue  # Retry the request
            elif result == "retry":
                logging.warning(f"Rate limit issues persist for file: {repo_info}")
                return None
            
            data = response.json()
            content = data.get("content")
            
            if not content:
                logging.warning(f"Empty content for file: {repo_info}")
                return ""
            
            return base64.b64decode(content).decode("utf-8")
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                logging.warning(f"Timeout downloading file on attempt {attempt + 1}, retrying...")
                time.sleep(2)
                continue
            logging.warning(f"Timeout downloading file: {repo_info}")
            return None
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                logging.warning(f"Connection error downloading file on attempt {attempt + 1}, retrying...")
                time.sleep(2)
                continue
            logging.warning(f"Network error downloading file: {repo_info}")
            return None
        except requests.exceptions.RequestException as e:
            logging.warning(f"Request error downloading file {repo_info}: {str(e)}")
            return None
        except (ValueError, UnicodeDecodeError) as e:
            logging.warning(f"Error decoding file content {repo_info}: {str(e)}")
            return None
        except GitHubAccessError as e:
            logging.warning(f"GitHub access error for file {repo_info}: {str(e)}")
            return None

def save_file_versions(output_dir, before_content, after_content, file_type='java'):
    """Save before and after versions of files and create diff"""
    file_ext = '.py' if file_type == 'python' else '.java'
    
    with open(output_dir / f"before{file_ext}", "w", encoding="utf-8") as f:
        f.write(before_content or "")
    
    with open(output_dir / f"after{file_ext}", "w", encoding="utf-8") as f:
        f.write(after_content or "")
    
    if before_content and after_content:
        diff = difflib.unified_diff(
            before_content.splitlines(keepends=True),
            after_content.splitlines(keepends=True),
            fromfile=f'before{file_ext}',
            tofile=f'after{file_ext}'
        )
        with open(output_dir / "developer_patch.diff", "w", encoding="utf-8") as f:
            f.writelines(diff)
    
    logging.info(f"Saved file versions to {output_dir}")

def extract_test_file_path(test_name, module_path, file_type='java'):
    """Extract the test file path from the test name and module path"""
    if not test_name or test_name == ".":
        return None
    
    if file_type == 'python':
        if '::' in test_name:
            file_path = test_name.split('::')[0]
            if not file_path.endswith('.py'):
                file_path += '.py'
            return file_path
        else:
            return test_name if test_name.endswith('.py') else f"{test_name}.py"
    
    else:  # Java format
        parts = test_name.split('.')
        if len(parts) < 2:
            return None
        
        class_parts = parts[:-1]  # Remove method name
        file_path = '/'.join(class_parts) + '.java'
        
        if module_path and module_path != ".":
            file_path = f"{module_path}/src/test/java/{file_path}"
        else:
            file_path = f"src/test/java/{file_path}"
        
        return file_path

def find_pull_request_for_commit(owner, repo, sha):
    """Find pull request(s) that contain the given commit"""
    try:
        # Search for PRs that contain this commit
        url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}/pulls"
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            pulls = response.json()
            if pulls:
                # Return the first (usually most relevant) PR
                pr = pulls[0]
                logging.info(f"🔗 Found PR #{pr['number']} for commit {sha[:7]}: {pr['title']}")
                return pr
        elif response.status_code == 404:
            logging.debug(f"No PR found for commit {sha[:7]}")
        else:
            logging.warning(f"Error finding PR for commit {sha[:7]}: {response.status_code}")
        
        return None
        
    except Exception as e:
        logging.warning(f"Error searching for PR for commit {sha[:7]}: {str(e)}")
        return None

def get_pull_request_commits(owner, repo, pr_number):
    """Get all commits from a pull request"""
    try:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
        response = requests.get(url, headers=HEADERS, timeout=30)
        
        if response.status_code == 200:
            commits = response.json()
            commit_shas = [commit['sha'] for commit in commits]
            logging.info(f"📋 PR #{pr_number} has {len(commit_shas)} commits")
            return commit_shas
        else:
            logging.warning(f"Error getting PR commits: {response.status_code}")
            return []
            
    except Exception as e:
        logging.warning(f"Error getting PR commits: {str(e)}")
        return []

def search_test_files_in_commits(owner, repo, commit_shas, test_name, file_type):
    """Search for test files across multiple commits"""
    all_changed_files = []
    file_extension = '.py' if file_type == 'python' else '.java'
    
    logging.info(f"🔍 Searching across {len(commit_shas)} commits for test files...")
    
    for i, sha in enumerate(commit_shas):
        try:
            changed_files, parent_sha = get_changed_files(owner, repo, sha)
            # Filter for actual test files, not just files with the right extension
            test_files = [f for f in changed_files 
                         if f.get("filename", "").endswith(file_extension) 
                         and is_test_file(f.get("filename", ""), file_type)]
            
            if test_files:
                logging.info(f"  Commit {i+1}/{len(commit_shas)} ({sha[:7]}): {len(test_files)} test {file_extension} files")
                for file_info in test_files:
                    file_info['commit_sha'] = sha
                    file_info['parent_sha'] = parent_sha
                all_changed_files.extend(test_files)
            
        except GitHubAccessError as e:
            logging.warning(f"  Commit {i+1}/{len(commit_shas)} ({sha[:7]}): {str(e)}")
            continue
        except Exception as e:
            logging.warning(f"  Commit {i+1}/{len(commit_shas)} ({sha[:7]}): Error - {str(e)}")
            continue
    
    logging.info(f"🎯 Found {len(all_changed_files)} total test {file_extension} files across all commits")
    return all_changed_files

def find_test_in_files(owner, repo, all_test_files, test_name, file_type):
    """Search for test method in available test files by downloading and checking content"""
    if file_type != 'java':
        return None
    
    # Extract method name from fully qualified test name
    parts = test_name.split('.')
    if len(parts) < 2:
        return None
    
    method_name = parts[-1]  # Last part is the method name
    logging.info(f"🔍 Searching for method '{method_name}' in available test files...")
    
    for file_info in all_test_files:
        file_path = file_info.get("filename", "")
        if not file_path.endswith('.java'):
            continue
            
        commit_sha = file_info.get('commit_sha')
        if not commit_sha:
            continue
            
        try:
            # Download the file content to search for the test method
            content = download_file_content(owner, repo, file_path, commit_sha)
            if content and method_name in content:
                # Check if it's actually a test method (not just a string occurrence)
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if (line.startswith('@Test') or 
                        'public void ' + method_name in line or
                        'void ' + method_name + '(' in line):
                        logging.info(f"🎯 Found method '{method_name}' in {file_path}")
                        return file_info
        except Exception as e:
            logging.debug(f"Error checking {file_path}: {str(e)}")
            continue
    
    return None

def process_repo(row, file_type='java'):
    """Process a single repository entry with enhanced matching strategies"""
    url = row['Project URL']
    sha = row['SHA Detected']
    category = row['Category']
    
    # Handle different CSV formats
    if file_type == 'python':
        test_name = row.get('Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)', '')
        module_path = None
    else:  # Java
        test_name = row.get('Fully-Qualified Test Name (packageName.ClassName.methodName)', '')
        module_path = row.get('Module Path', '')
    
    owner, repo = url.rstrip("/").split("/")[-2:]
    repo_info = f"{owner}/{repo}"
    
    logging.info(f"Processing {repo_info} - {test_name}")
    
    try:
        # Strategy 1: Try the original commit first
        try:
            changed_files, parent_sha = get_changed_files(owner, repo, sha)
            logging.info(f"🔍 Original commit {sha[:7]}: Found {len(changed_files)} changed files")
        except GitHubAccessError as e:
            logging.error(f"⚠️  Skipping {repo_info}: {str(e)}")
            raise
        
        file_extension = '.py' if file_type == 'python' else '.java'
        # Filter for actual test files, not just files with the right extension
        available_test_files = [f.get("filename", "") for f in changed_files 
                               if f.get("filename", "").endswith(file_extension) 
                               and is_test_file(f.get("filename", ""), file_type)]
        
        # Strategy 2: If no test files in original commit, search in associated PR
        all_test_files = []
        
        if available_test_files:
            logging.info(f"✅ Found {len(available_test_files)} test {file_extension} files in original commit")
            # Add commit info to files - only actual test files
            for file_info in changed_files:
                filename = file_info.get("filename", "")
                if filename.endswith(file_extension) and is_test_file(filename, file_type):
                    file_info['commit_sha'] = sha
                    file_info['parent_sha'] = parent_sha
                    all_test_files.append(file_info)
        else:
            logging.info(f"⚠️  No test {file_extension} files in original commit, searching in PR...")
            
            # Find associated pull request
            pr_info = find_pull_request_for_commit(owner, repo, sha)
            
            if pr_info:
                pr_number = pr_info['number']
                commit_shas = get_pull_request_commits(owner, repo, pr_number)
                
                if commit_shas:
                    all_test_files = search_test_files_in_commits(owner, repo, commit_shas, test_name, file_type)
                    available_test_files = [f.get("filename", "") for f in all_test_files]
            
            if not all_test_files:
                logging.warning(f"No test {file_extension} files found in commit or associated PR")
        
        if not all_test_files:
            return {
                'status': 'no_files',
                'row': row,
                'changed_files': [],
                'available_test_files': [],
                'expected_test_file': None,
                'parent_sha': parent_sha if 'parent_sha' in locals() else None,
                'file_type': file_type
            }
        
        logging.info(f"🔍 Total available test {file_extension} files: {available_test_files}")
        
        # Extract expected test file path
        expected_test_file = extract_test_file_path(test_name, module_path, file_type)
        if not expected_test_file:
            logging.warning(f"Could not extract test file path from: {test_name}")
            return False
        
        logging.info(f"🎯 Looking for: {expected_test_file}")
        
        # Strategy 3: Find matching test file using multiple approaches
        test_file = None
        
        # Approach 3a: Exact matching
        for file in all_test_files:
            file_path = file.get("filename", "")
            
            if file_path.endswith(file_extension):
                # Check if this matches our expected test file
                if file_type == 'python':
                    # For Python: exact path match or filename match
                    match_found = (file_path == expected_test_file or 
                                 file_path.endswith(expected_test_file.split('/')[-1]))
                else:  # Java
                    # For Java: check if expected path ends with the actual file path
                    match_found = expected_test_file.endswith(file_path.split('/')[-1])
                
                if match_found:
                    test_file = file
                    commit_sha = file.get('commit_sha', sha)
                    logging.info(f"✨ Exact match found: {file_path} in commit {commit_sha[:7]}")
                    break
        
        # Approach 3b: Handle common typos in CSV data
        if not test_file and file_type == 'java':
            # Extract class name from test name (handle typos)
            parts = test_name.split('.')
            if len(parts) >= 2:
                csv_class_name = parts[-2]  # Class name from CSV (may have typos)
                
                # Try to find similar class names in available files
                for file in all_test_files:
                    file_path = file.get("filename", "")
                    if file_path.endswith('.java'):
                        actual_class_name = file_path.split('/')[-1].replace('.java', '')
                        
                        # Check for common typos: double letters, single vs double letters
                        if (csv_class_name.lower().replace('ll', 'l') == actual_class_name.lower() or
                            csv_class_name.lower().replace('l', 'll') == actual_class_name.lower() or
                            csv_class_name.lower().replace('tt', 't') == actual_class_name.lower() or
                            csv_class_name.lower().replace('t', 'tt') == actual_class_name.lower() or
                            csv_class_name.lower().replace('ss', 's') == actual_class_name.lower() or
                            csv_class_name.lower().replace('s', 'ss') == actual_class_name.lower()):
                            
                            test_file = file
                            commit_sha = file.get('commit_sha', sha)
                            logging.info(f"🔧 Typo-corrected match: {csv_class_name} -> {actual_class_name} in {file_path} (commit {commit_sha[:7]})")
                            break
        
        # Approach 3c: Search for test method in available test files
        if not test_file and file_type == 'java':
            logging.info(f"🔍 No class match found, searching for test method in available files...")
            test_file = find_test_in_files(owner, repo, all_test_files, test_name, file_type)
        
        # Approach 3d: Partial class name matching (for renamed tests)
        if not test_file and file_type == 'java':
            parts = test_name.split('.')
            if len(parts) >= 2:
                csv_class_name = parts[-2].lower()
                
                logging.info(f"🔍 Trying partial class name matching for '{csv_class_name}'...")
                
                for file in all_test_files:
                    file_path = file.get("filename", "")
                    if file_path.endswith('.java'):
                        actual_class_name = file_path.split('/')[-1].replace('.java', '').lower()
                        
                        # Check if either name contains the other (for renamed tests)
                        if (csv_class_name in actual_class_name or 
                            actual_class_name in csv_class_name or
                            any(word in actual_class_name for word in csv_class_name.split('test') if word) or
                            any(word in csv_class_name for word in actual_class_name.split('test') if word)):
                            
                            test_file = file
                            commit_sha = file.get('commit_sha', sha)
                            logging.info(f"🔧 Partial match found: {csv_class_name} ~ {actual_class_name} in {file_path} (commit {commit_sha[:7]})")
                            break
        
        if not test_file:
            logging.warning(f"⚠️  No match found for {test_name} in {repo_info}")
            logging.warning(f"Available files: {available_test_files}")
            
            # Return failure info for interactive processing later
            return {
                'status': 'no_match',
                'row': row,
                'changed_files': all_test_files,
                'available_test_files': available_test_files,
                'expected_test_file': expected_test_file,
                'parent_sha': parent_sha if 'parent_sha' in locals() else None,
                'file_type': file_type
            }
        
        # Process the matched file
        success = process_matched_file(test_file, row, test_file.get('parent_sha'), file_type)
        return True if success else False
                
    except GitHubAccessError:
        raise  # Re-raise GitHub access errors to be handled by main script
    except Exception as e:
        logging.error(f"❌ Unexpected error processing {repo_info}: {str(e)}")
        raise  # Re-raise unexpected errors to be handled by main script

def process_matched_file(test_file, row, parent_sha, file_type):
    """Process a matched file and save it"""
    url = row['Project URL']
    sha = row['SHA Detected']
    category = row['Category']
    
    if file_type == 'python':
        test_name = row.get('Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)', '')
        module_path = None
    else:  # Java
        test_name = row.get('Fully-Qualified Test Name (packageName.ClassName.methodName)', '')
        module_path = row.get('Module Path', '')
    
    owner, repo = url.rstrip("/").split("/")[-2:]
    repo_info = f"{owner}/{repo}"
    file_path = test_file.get("filename", "")
    
    # Download files with error handling
    before_content = download_file_content(owner, repo, file_path, parent_sha)
    after_content = download_file_content(owner, repo, file_path, sha)
    
    # Check if we got at least one version
    if before_content is None and after_content is None:
        logging.warning(f"⚠️  Could not download any version of {file_path} from {repo_info}")
        return False
    
    if before_content is None:
        logging.warning(f"Could not download before version of {file_path}")
        before_content = ""
    if after_content is None:
        logging.warning(f"Could not download after version of {file_path}")
        after_content = ""
    
    # Create output directory with proper categorization
    safe_test_name = test_name.replace("/", "_").replace("::", "_").replace(".", "_").replace("[", "_").replace("]", "_")
    
    # Clean up the test name to remove file extensions and path separators
    safe_test_name = re.sub(r'[<>:"/\\|?*]', '_', safe_test_name)
    
    # Create directory structure: results/{file_type}_data/{category}/{safe_test_name}
    outdir = RESULTS_DIR / f"{file_type}_data" / category / safe_test_name
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save files and metadata
    save_file_versions(outdir, before_content, after_content, file_type)
    
    metadata = {
        "repository": f"{owner}/{repo}",
        "commit_sha": sha,
        "parent_sha": parent_sha,
        "test_file": file_path,
        "test_name": test_name,
        "category": category,
        "project_url": url,
        "file_type": file_type,
        "status": row.get('Status', 'Unknown'),
        "owner_repo_sha": f"{owner}_{repo}_{sha[:7]}"
    }
    
    if file_type == 'java' and module_path:
        metadata["module_path"] = module_path
    
    with open(outdir / "metadata.txt", "w", encoding="utf-8") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    logging.info(f"✅ Successfully processed {file_path} -> {outdir}")
    return True
