import difflib
from pathlib import Path
from .config import RESULTS_DIR
import google.generativeai as genai
import os

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
