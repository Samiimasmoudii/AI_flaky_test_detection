from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="Meta-Llama-3.1-8B-Instruct-Q4_K_L.gguf",
    local_dir="models"
)
