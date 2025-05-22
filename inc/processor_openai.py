import difflib
from pathlib import Path
from .config import RESULTS_DIR
import openai
import os
import re
import csv
import time
from typing import Optional

MODEL_NAME = "gpt-4"  # The OpenAI model name

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

def is_valid_diff(diff_text: str) -> bool:
    """Check if the diff output is valid and non-empty."""
    if not diff_text:
        return False
    lines = diff_text.strip().splitlines()
    return any(line.startswith(('+', '-')) for line in lines)

def run_llm_with_retry(prompt: str, max_retries: int = 3, delay: int = 6) -> Optional[str]:
    """Run OpenAI with retry logic and rate limiting."""
    for attempt in range(max_retries):
        try:
            # Rate limiting delay
            if attempt > 0:
                time.sleep(delay)
            
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a Python testing expert. Your task is to fix flaky tests by providing a unified diff format patch."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=2000
            )
            
            diff_text = response.choices[0].message.content
            
            if not diff_text or not is_valid_diff(diff_text):
                print(f"Attempt {attempt + 1}: Empty or invalid diff received, retrying...")
                continue
                
            return diff_text
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)  # Wait before retry
            else:
                raise Exception(f"Failed after {max_retries} attempts: {str(e)}")
    
    return None

def run_llm_on_before_file(filepath, category=""):
    try:
        # Configure OpenAI API
        openai.api_key = os.getenv('OPENAI_API_KEY')
        
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get flakiness description
        flakiness_desc = FLAKINESS_TYPES.get(category, FLAKINESS_TYPES[""])
        
        # Create prompt with flakiness category
        prompt = f"""
        The following test file exhibits flaky behavior categorized as:

Type: {category}
Description: {flakiness_desc}

Step 1: Identify the root cause of the flakiness based on the category.
Step 2: Fix the code using known patterns that address this type of flakiness.
Step 3: Output ONLY the fix as a unified diff.

Here is the test file:
<code>
{content}
</code>

Please provide the fixed version of the code that addresses any flaky test patterns. Output only the diff like this example:
--- before.py
+++ after.py
@@ -10,6 +10,8 @@
    old line
+   new line
    old line"""
        
        # Get response from OpenAI with retry logic
        response = run_llm_with_retry(prompt)
        
        if response is None:
            print(f"Failed to get valid response for {filepath} after all retries")
            return ""
            
        return response
        
    except Exception as e:
        print(f"Error running LLM on file {filepath}: {e}")
        return ""

def compare_fixes(llm_fix, dev_patch):
    d = difflib.unified_diff(dev_patch.splitlines(), llm_fix.splitlines(), lineterm="")
    return "\n".join(d)

def extract_patch_blocks(content: str):
    dev_patch, llm_patch = "", ""
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

def count_added_removed_lines(diff_text: str):
    added = sum(1 for line in diff_text.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_text.splitlines() if line.startswith("-") and not line.startswith("---"))
    return added, removed

def compute_bleu_score(reference_text: str, candidate_text: str) -> float:
    result = bleu_metric.compute(predictions=[candidate_text], references=[[reference_text]])
    return result["bleu"]

def compute_rouge_score(reference_text: str, candidate_text: str) -> float:
    result = rouge_metric.compute(predictions=[candidate_text], references=[reference_text])
    return result["rougeL"] 