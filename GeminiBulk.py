from config import RESULTS_DIR, SAMPLE_SIZE
from data_loader import load_and_clean_csv
from github_utils import get_changed_files, download_file_content
from processor import save_versions, run_llm_on_file, compare_fixes

import os

CSV_PATH = "py-data.csv"
df = load_and_clean_csv(CSV_PATH)

for _, row in df.sample(SAMPLE_SIZE, random_state=42).iterrows():
    try:
        url, sha, test_path, category = row['Project URL'], row['SHA Detected'], row['Pytest Test Name (PathToFile::TestClass::TestMethod or PathToFile::TestMethod)'], row['category']
        owner, repo = url.rstrip("/").split("/")[-2:]
        changed_files, parent_sha = get_changed_files(owner, repo, sha)

        for file in changed_files:
            file_path = file.get("filename", "")
            if test_path.split("::")[0].endswith(file_path):
                before = download_file_content(owner, repo, file_path, parent_sha)
                after = download_file_content(owner, repo, file_path, sha)

                outdir = RESULTS_DIR / category / test_path.replace("/", "_").replace("::", "_")
                outdir.mkdir(parents=True, exist_ok=True)

                save_versions(outdir, before, after)
                llm_fix = run_llm_on_file(outdir / "before.py")
                dev_patch = (outdir / "developer_patch.diff").read_text()
                comparison = compare_fixes(llm_fix, dev_patch)

                (outdir / "comparison.txt").write_text(comparison)

    except Exception as e:
        print(f"Error processing {url}@{sha}: {e}")
