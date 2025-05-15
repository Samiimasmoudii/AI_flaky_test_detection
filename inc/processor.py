import difflib
from pathlib import Path
from .config import RESULTS_DIR
import google.generativeai as genai
import os
import re
import csv


FLAKINESS_TYPES = {
    "": "No specific flakiness type provided",
    "OD": "Order-Dependent flaky tests - Tests that pass/fail depending on the order of execution",
    "OD-Brit": "Order-Dependent Brittle tests - Tests that need to run in a specific order to pass",
    "OD-Vic": "Order-Dependent Victim tests - Tests that fail when run after specific tests",
    "ID": "Implementation-Dependent Tests - Tests that depend on specific implementation details",
    "ID-HtF": "Implementation-Dependent tests that are hard to fix due to complex dependencies",
    "NIO": "Non-Idempotent-Outcome Tests - Tests that pass in the first run but fail in the second",
    "NOD": "Non-Deterministic tests - Tests with non-deterministic outcomes",
    "NDOD": "Non-Deterministic Order-Dependent tests - Tests that fail non-deterministically with different failure rates in different orders",
    "NDOI": "Non-Deterministic Order-Independent tests - Tests that fail non-deterministically with similar failure rates in all orders",
    "UD": "Unknown Dependency tests - Tests that pass and fail in a test suite or in isolation",
    "OSD": "Operating System Dependent tests - Tests that pass and fail depending on the operating system",
    "TZD": "Time Zone Dependent tests - Tests that fail in machines on different time zones"
}

def save_versions(outdir, before, after):
    (outdir / "before.py").write_text(before or "", encoding="utf-8")
    (outdir / "after.py").write_text(after or "", encoding="utf-8")
    diff = difflib.unified_diff(
        (before or "").splitlines(),
        (after or "").splitlines(),
        fromfile="before.py",
        tofile="after.py",
        lineterm=""
    )
    (outdir / "developer_patch.diff").write_text("\n".join(diff), encoding="utf-8")

def run_llm_on_file(filepath, category=""):
    try:
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get flakiness description
        flakiness_desc = FLAKINESS_TYPES.get(category, FLAKINESS_TYPES[""])
        
        # Create prompt with flakiness category
        prompt = f"""You are a Python testing expert. The following test file has {category} flaky behavior:
{flakiness_desc}

Generate ONLY the code fix in diff format (do not explain the changes):

<code>
{content}
</code>


Please provide the fixed version of the code that addresses any flaky test patterns.Output only the diff like this example:
--- before.py
+++ after.py
@@ -10,6 +10,8 @@
    old line
+   new line
    old line"""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error running LLM on file {filepath}: {e}")
        return ""
    
    
def compare_fixes(llm_fix, dev_patch):
    d = difflib.unified_diff(dev_patch.splitlines(), llm_fix.splitlines(), lineterm="")
    return "\n".join(d)





## Comparison and CSV generation


def extract_patch_blocks(content: str):
    dev_patch, llm_patch = "", ""
    # Extract with simple split — adjust this as per your actual file format
    dev_start = content.find("--- before.py")
    llm_start = content.find("```diff")
    
    if dev_start != -1 and llm_start != -1:
        dev_patch = content[dev_start:llm_start].strip()
        llm_patch = content[llm_start:].strip("```diff\n").strip("```").strip()
    
    return dev_patch, llm_patch

def extract_changed_lines(diff_text: str):
    added = set()
    removed = set()
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.add(line[1:].strip())  # actual added line content
        elif line.startswith("-") and not line.startswith("---"):
            removed.add(line[1:].strip())  # actual removed line content
    return removed, added


def compare_dev_llm(dev_patch: str, llm_patch: str):
    dev_removed, dev_added = extract_changed_lines(dev_patch)
    llm_removed, llm_added = extract_changed_lines(llm_patch)

    dev_lines = dev_removed | dev_added
    llm_lines = llm_removed | llm_added

    matched_lines = dev_lines & llm_lines
    score = round(100 * len(matched_lines) / max(len(dev_lines), 1))
    return len(dev_lines), len(llm_lines), len(matched_lines), score


def collect_results(base_dir="results", output_csv="llm_patch_scores.csv"):
    rows = []
    for comp_file in Path(base_dir).rglob("comparison.txt"):
        relative = comp_file.relative_to(base_dir)
        test_category = relative.parts[0]
        fix_category = relative.parts[1]
        # Get flakiness type (assumed encoded in fix_category or .txt file)
        flakiness = ""
        for code in FLAKINESS_TYPES:
            if code and code in fix_category:
                flakiness = code
                break

        # Read content
        content = comp_file.read_text()
        dev_patch, llm_patch = extract_patch_blocks(content)
        dev_total, llm_total, matched, score = compare_dev_llm(dev_patch, llm_patch)

        rows.append({
            "Test Category": test_category,
            "Fix Category": fix_category,
            "Flakiness Type": flakiness,
            "Dev Lines Changed": dev_total,
            "LLM Lines Changed": llm_total,
            "Matched Lines": matched,
            "Match Score (%)": score
        })

    # Save CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved patch match scores to {output_csv}")
