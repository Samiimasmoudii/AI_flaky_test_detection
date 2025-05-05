from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="TheBloke/LLaMA-Pro-8B-Instruct-GGUF",
    filename="llama-pro-8b-instruct.Q5_K_M.gguf",
    local_dir="models"
)
