import os
import subprocess
from llama_cpp import Llama

# List files in the /flaky_code folder
print("Listing files in the /flaky_code folder:")
subprocess.run("dir flaky_code" if os.name == "nt" else "ls flaky_code", shell=True)

# Ask the user to choose a file
file_choice = input("Enter the file name you want to use: ")

# Ask the user to input their prompt
user_prompt = input("Enter your prompt: ")

# Load the Llama model
llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf")

# Generate output based on the user's prompt

output = llm(user_prompt, max_tokens=256)

print(output['choices'][0]['text'].strip ())
# Save the output to a file
# print(output) # If you want to see the full output and logs

#test with Prompt : You an expert in software testing, this fie causes flaky test, please help me to fix it. 
#output :  hopefully, you can fix it.
#I had a similar problem. I have a class that extends a super class. In the super class, I use the Thread.sleep(5000) method. But in the child class I use Thread.sleep(500000) and it works.
#I have the same problem. I think the problem is when you call the methods from the main thread.
#The solution is to use the TestNG. After this, the tests will start to be executed sequencially.
#sounds like a forum post or example dialogue, means:

#The model is interpreting it as a chat history or online post.

#It's responding as if it's in a Q&A forum, not analyzing the code.