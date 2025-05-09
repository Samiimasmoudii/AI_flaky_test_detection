import requests
import base64
from config import GITHUB_API, HEADERS

def get_changed_files(owner, repo, sha):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/commits/{sha}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch commit: {response.status_code}")
    data = response.json()
    return data.get("files", []), data["parents"][0]["sha"]

def download_file_content(owner, repo, filepath, sha):
    url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{filepath}?ref={sha}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code != 200:
        return None
    return base64.b64decode(r.json().get("content", "")).decode("utf-8")
