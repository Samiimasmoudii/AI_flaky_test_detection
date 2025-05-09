import difflib
from pathlib import Path
from config import RESULTS_DIR

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
    return "LLM fix placeholder"

def compare_fixes(llm_fix, dev_patch):
    d = difflib.unified_diff(dev_patch.splitlines(), llm_fix.splitlines(), lineterm="")
    return "\n".join(d)
