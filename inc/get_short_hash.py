def get_short_hash(full_hash):
    """Returns the first 7 characters of the commit hash"""
    return full_hash[:7]

# In the main processing loop:
short_hash = get_short_hash(commit_hash)
print(f"Repo: {repo_url}")
print(f"Commit: {short_hash} (full: {commit_hash})")
print(f"GitHub commit URL: {repo_url}/commit/{short_hash}")