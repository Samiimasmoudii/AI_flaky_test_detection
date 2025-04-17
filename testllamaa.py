from llama_cpp import Llama

llm = Llama(model_path="models/llama-2-7b.Q4_K_M.gguf")
output = llm("Q: What is the capital of France?\nA:", max_tokens=32)
print(output)
