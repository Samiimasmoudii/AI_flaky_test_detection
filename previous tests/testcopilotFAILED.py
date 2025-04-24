import os
import subprocess

# Step 1: List files in the directory
flaky_dir = "./flaky_code"
print(f"Listing files in the {flaky_dir} folder:")
files = os.listdir(flaky_dir)
for f in files:
    print(f)

# Step 2: Ask user for a filename
filename = input("Enter the file name you want to analyze: ").strip()
file_path = os.path.join(flaky_dir, filename)

# Step 3: Read the file contents
if not os.path.exists(file_path):
    print(f"File {file_path} does not exist!")
    exit(1)

with open(file_path, "r") as f:
    file_content = f.read()
    num_lines = len(file_content.splitlines())
    print(f"✅ Loaded file '{file_path}' successfully with {num_lines} lines.")

# Step 4: Build prompt for GitHub Copilot CLI
prompt = f"""
You are an expert in Python software testing.

The following Python file has flaky tests or unstable behavior.

<begin code>
{file_content}
<end code>

Please help me with:
1. Diagnosing the cause of the flakiness.
2. Suggesting fixes.
3. Improving test reliability.
"""

# Step 5: Use GitHub Copilot CLI to analyze the flaky code
try:
    print("Analyzing the flaky code using GitHub Copilot CLI...")
    result = subprocess.run(
        ["github-copilot-cli", "what-the-shell", prompt],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        print("✅ Analysis complete. Here are the suggestions:")
        print(result.stdout.strip())
    else:
        print("❌ Failed to analyze the code. Error:")
        print(result.stderr.strip())

except FileNotFoundError:
    print("❌ GitHub Copilot CLI is not installed or not in the PATH.")
    print("Please install it using: npm install -g @githubnext/github-copilot-cli")