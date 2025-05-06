import os
from llama_cpp import Llama

# Step 1: List files in the flaky_code directory
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

# Step 4: Build prompt
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

# Step 5: Load the LLaMA 3 model and generate response
print("🧠 Loading LLaMA 3 model...")
llm = Llama(
    model_path="inc/models/Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf",
    n_ctx=20000,           # Adjust context if you have memory issues
    n_threads=4,
    verbose=False
)

print("🤖 Analyzing the flaky code using LLaMA 3...")
response = llm(prompt, max_tokens=512)

print("✅ Analysis complete. Here are the suggestions:\n")
print(response["choices"][0]["text"].strip())
