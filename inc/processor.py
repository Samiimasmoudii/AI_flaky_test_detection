import difflib
from pathlib import Path
from config import RESULTS_DIR
import google.generativeai as genai
import os


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

def run_llm_on_file(filepath):
    try:
        # Configure Gemini API
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        model = genai.GenerativeModel('gemini-2.5-flash-preview-04-17')
        
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create prompt for fixing flaky tests
        prompt = f"""Analyze this test file and suggest fixes for potential flaky test issues:
        
{content}

Please provide the fixed version of the code that addresses any flaky test patterns."""
        
        # Get response from Gemini
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error running LLM on file {filepath}: {e}")
        return ""
    
    
def compare_fixes(llm_fix, dev_patch):
    d = difflib.unified_diff(dev_patch.splitlines(), llm_fix.splitlines(), lineterm="")
    return "\n".join(d)
