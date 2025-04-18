import os
import subprocess
from llama_cpp import Llama

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

# Step 4: Build prompt for the model
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

# Step 5: Ask the user to input their additional prompt
user_input = input("Enter your prompt (default: Press Enter ): ").strip()
if not user_input:
    user_input = ""  # Use default value if no input is provided

# Combine the base prompt with the user's input
user_prompt = prompt + "\n" + user_input

# Step 6: Load the Llama model
llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf",n_ctx=2048)

# Step 7: Generate output based on the user's prompt
output = llm(user_prompt)

print(output['choices'][0]['text'].strip ())
## OUTPUT Home » News » Eat, Drink and Be Merry nobody can tell you that you can’t.A new restaurant in the heart of the city.# Step 8: Print the model's response
if isinstance(output, dict) and 'choices' in output:
    print(output['choices'][0]['text'].strip())
else:
    print("Unexpected output format from the model:", output)


#### OUTPUT EXCEEDS MAX TOKENS
#Model Used:
#llama-2-7b.Q4_K_M.gguf via llama-cpp-python

#Context Window Configured:
#512 tokens (n_ctx=512)
#ERROR : ValueError: Requested tokens (13060) exceed context window of 512


