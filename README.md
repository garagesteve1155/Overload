# ai_splitter
Run Mistral 7b Instruct on less than 4gb of RAM

This repository provides a minimal and memory-efficient way to run the **Mistral-7B Instruct v3.0** model on **low-resource devices** (e.g. CPUs with limited RAM). It loads model weights layer-by-layer directly from disk, avoiding the need to hold the entire model in memory at once.

## 🔧 Features

- Layer-by-layer loading from `safetensors` model shards
- Currently known to be compatible with Mistral-7B Instruct v3.0 from Hugging Face
- CPU-only, no GPU required
- Minimal dependencies and clean CLI interface

## 📦 Requirements

- Python 3.8+
- See `requirements.txt` for packages

## 🚀 Usage

1. Download the [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) files into the `./mistral_7b_instruct/` folder (The mistral_7b_instruct folder itself should be in whichever folder you have the ai_splitter python script). You need the numbered .safetensors files, plus all the other non-weight files

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Run ai_splitter.py in a terminal window
