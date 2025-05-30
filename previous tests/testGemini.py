import os
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Step 1: List files in the flaky_code directory
flaky_dir = "./flaky_code_NIO"
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
the Cause of the flakiness is Non-Idempotent-Outcome Tests as defined in ICSE’22 work. Tests that pass in the first run but fail in the second.

<begin code>
{file_content}
<end code>

Please help me with:
1. Diagnosing the cause of the flakiness.
2. Suggesting fixes.
3. Improving test reliability.

Make the answer short and concise by telling exactly which lines are causing the flakiness and how to fix them.
"""

# Step 5: Load the LLaMA 3 model and generate response
print("🧠 Loading Gemini model...")



response = client.models.generate_content(
    model="gemini-2.5-flash-preview-04-17",
    contents=prompt,
)



print("🤖 Analyzing the flaky code using Google Gemini 2.5 Flash...")



print(response.text)
print("\n\n✅ Full response received.")



