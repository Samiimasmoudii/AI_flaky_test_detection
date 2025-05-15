import difflib
from pathlib import Path
from .config import RESULTS_DIR
import google.generativeai as genai
import os
import re
import csv



## In case We're using natural language toolkit instead of Hugging face 'evaluate' library
#import nltk
#from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
#from nltk.tokenize import word_tokenize
#from rouge_score import rouge_scorer
#nltk.download('punkt')

import evaluate
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


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
    for category_dir in Path(base_dir).iterdir():
        if not category_dir.is_dir():
            continue

        flakiness_type = category_dir.name

        for test_dir in category_dir.iterdir():
            if not test_dir.is_dir():
                continue

            dev_patch_file = test_dir / "developer_patch.diff"
            llm_file = test_dir / "LLM_suggestion.txt"

            if not dev_patch_file.exists() or not llm_file.exists():
                continue

            dev_patch = dev_patch_file.read_text(encoding="utf-8").strip()
            llm_patch = llm_file.read_text(encoding="utf-8").strip()

            bleu = compute_bleu_score(dev_patch, llm_patch)
            rouge = compute_rouge_score(dev_patch, llm_patch)
            dev_added, dev_removed = count_added_removed_lines(dev_patch)
            llm_added, llm_removed = count_added_removed_lines(llm_patch)

            rows.append({
                "Test Category": flakiness_type,
                "File Path": test_dir.name,
                "BLEU Score": bleu,
                "ROUGE-L Score": rouge,
                "Dev Changed": f"+{dev_added} ; -{dev_removed}",
                "LLM Changed": f"+{llm_added} ; -{llm_removed}"
            })

    if rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"✅ Saved BLEU/ROUGE scores and line counts to {output_csv}")
    else:
        print("⚠️ No valid results found in the given structure.")


def count_added_removed_lines(diff_text: str):
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed



def compute_bleu_score(reference_text: str, candidate_text: str) -> float:
    results = bleu_metric.compute(
        predictions=[candidate_text],
        references=[[reference_text]]
    )
    return round(results["bleu"], 4)


def compute_rouge_score(reference_text: str, candidate_text: str) -> float:
    results = rouge_metric.compute(
        predictions=[candidate_text],
        references=[reference_text]
    )
    return round(results["rougeL"], 4)