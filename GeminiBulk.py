# Flaky Test Patch Comparator Notebook (Optimized for Single File Diffs)

import os
import base64
import requests
import difflib
import pandas as pd
from pathlib import Path

# ========== STEP 1: Load Data ==========
CSV_PATH = "/mnt/data/py-data (1).csv"
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=['Project URL', 'SHA Detected'])
df = df[~df['Notes'].astype(str).str.contains("deleted", case=False)]

# ========== STEP 2: GitHub API Utilities ==========
GITHUB_API = "https://api.github.com"
HEADERS = {}  # Optional: {'Authorization': 'token YOUR_GITHUB_TOKEN'}

# ========== STEP 3: Setup Output Directory ==========
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ========== STEP 4: Download File Versions ==========
def get_changed_files(owner, repo, sha):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to get commit data for {owner}/{repo}@{sha}")
    commit_data = response.json()
    files = commit_data.get("files", [])
    parent_sha = commit_data["parents"][0]["sha"] if commit_data["parents"] else None
    return files, parent_sha

def download_file_content(owner, repo, path, sha):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}?ref={sha}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return None
    content = r.json().get("content")
    return base64.b64decode(content).decode("utf-8") if content else None

# ========== STEP 5: Save Files ==========
def save_file_versions(output_dir, filename, before, after):
    with open(output_dir / "before.py", "w", encoding="utf-8") as f:
        f.write(before or "")
    with open(output_dir / "after.py", "w", encoding="utf-8") as f:
        f.write(after or "")
    with open(output_dir / "developer_patch.diff", "w", encoding="utf-8") as f:
        diff = difflib.unified_diff((before or "").splitlines(), (after or "").splitlines(), fromfile='before.py', tofile='after.py', lineterm="")
        f.write("\n".join(diff))

# ========== STEP 6: Run LLM Placeholder ==========
def run_llm_on_file(file_path):
    return "LLM fix placeholder"

# ========== STEP 7: Compare ==========
def compare_fixes(llm_fix, developer_patch):
    d = difflib.unified_diff(
        developer_patch.splitlines(),
        llm_fix.splitlines(),
        lineterm=""
    )
    return "\n".join(d)

# ========== STEP 8: Process Entries ==========
SAMPLE_SIZE = 5

for idx, row in df.sample(SAMPLE_SIZE, random_state=42).iterrows():
    url, sha, test_path, category = row['Project URL'], row['SHA Detected'], row['Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)'], row['Category']
    print(f"Processing {url} @ {sha}")

    try:
        parts = url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]

        changed_files, parent_sha = get_changed_files(owner, repo, sha)
        if not parent_sha:
            continue

        # Heuristic match: test path file
        test_file_path = test_path.split("::")[0]

        # Create output directory based on category
        safe_test = test_path.replace("/", "_").replace("::", "_")
        output_dir = RESULTS_DIR / category / safe_test
        output_dir.mkdir(parents=True, exist_ok=True)

        for file in changed_files:
            filepath = file.get("filename")
            if test_file_path.endswith(filepath) or filepath.endswith(test_file_path):
                before = download_file_content(owner, repo, filepath, parent_sha)
                after = download_file_content(owner, repo, filepath, sha)

                save_file_versions(output_dir, filepath, before, after)
                llm_fix = run_llm_on_file(output_dir / "before.py")

                with open(output_dir / "developer_patch.diff") as f:
                    dev_patch = f.read()

                comparison = compare_fixes(llm_fix, dev_patch)
                with open(output_dir / "comparison.txt", "w") as f:
                    f.write(comparison)

        # Also download test file
        test_before = download_file_content(owner, repo, test_file_path, parent_sha)
        test_after = download_file_content(owner, repo, test_file_path, sha)
        with open(output_dir / "test_before.py", "w") as f:
            f.write(test_before or "")
        with open(output_dir / "test_after.py", "w") as f:
            f.write(test_after or "")

    except Exception as e:
        print(f"Error processing {url}@{sha}: {e}")
