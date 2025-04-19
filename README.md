Overload

Run Mistral 7b Instruct on less than 4gb of RAM

This repository provides a minimal and memory-efficient way to run the Mistral-7B Instruct v3.0 model on low-resource devices (e.g. CPUs with limited RAM). It loads model weights layer-by-layer directly from disk, avoiding the need to hold the entire model in memory at once. The goal of this project is to allow anyone to run even 100B+ models on almost any modern-ish (Last 10 years or so) computer.

🔧 Features

- Layer-by-layer loading/inference from safetensors model shards

- Optional automatic preloading of layers into available RAM and/or GPU VRAM

- GPU acceleration supported (with user-selectable VRAM safety buffer + RAM safety buffer)

- Currently known to be compatible with Mistral-7B Instruct v3.0 from Hugging Face (More and larger models coming soon)

- CPU-only mode still supported, no GPU required

- Minimal dependencies and clean CLI interface



📦 Requirements

Python 3.8+

4GB of RAM Minimum (More, or added gpu, directly decreases time per token)

See requirements.txt for packages


🚀 Usage

Download the Mistral-7B-Instruct-v0.3 files into the ./mistral_7b_instruct/ folder (The mistral_7b_instruct folder itself should be in whichever folder you have the Overload python script). You need the numbered .safetensors files, plus all the other non-weight files

Install dependencies:

pip install -r requirements.txt

Run overload.py in a terminal window

When prompted, enter how much GPU VRAM and system RAM to leave unused for safety/overload. If you don’t have a GPU, it won't ask about VRAM.
