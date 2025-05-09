from pathlib import Path
import base64
import requests
import difflib
# ========== STEP 2: GitHub API Utilities ==========
GITHUB_API = "https://api.github.com"
HEADERS = {}  # Optional: Add token like {'Authorization': 'token YOUR_GITHUB_TOKEN'}

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