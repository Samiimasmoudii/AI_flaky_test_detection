#!/bin/bash

# Exit immediately on error
set -e

echo "🔧 Setting up your LLaMA environment..."

# Create virtual environment
python3 -m venv .venv
echo "✅ Virtual environment created."

# Activate virtual environment
source .venv/bin/activate
echo "✅ Virtual environment activated."

# Install llama-cpp-python
pip install --upgrade pip
pip install llama-cpp-python
echo "✅ llama-cpp-python installed."

# Create models directory and download the model
mkdir -p models
cd models

# ✅ Updated: Download LLaMA 3 8B Instruct model in GGUF format
echo "⬇️  Downloading LLaMA 3 8B model..."
wget https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8B-Instruct.Q4_K_M.gguf

cd ..

echo "✅ Setup complete! You’re ready to run LLaMA 3 8B."
