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

# Download the LLaMA model
echo "⬇️  Downloading LLaMA model..."
wget https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf
cd ..

echo "✅ Setup complete! You’re ready to run LLaMA."
