#!/bin/bash
set -e

# full setup script for Overload on Raspberry Pi 5 (4GB)

# 1) update OS and install build tools
sudo apt update && sudo apt upgrade -y
sudo apt install -y wget git build-essential

# 2) install Miniforge (arm64 build) if not already installed
if [ ! -d "$HOME/miniforge" ]; then
    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -O /tmp/Miniforge3-Linux-aarch64.sh
    bash /tmp/Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge
fi

# 3) activate conda in this shell
eval "$($HOME/miniforge/bin/conda shell.bash hook)"

# 4) create and activate environment if not exists
if ! conda info --envs | grep -q "^overload"; then
    conda create -n overload python=3.10 -y
fi
conda activate overload
pip install --upgrade pip setuptools wheel

# 5) pin numpy < 2
conda install -y -c conda-forge "numpy<2"

# 6) install PyTorch 2.1+ for aarch64
pip install --no-cache-dir -f https://cf.torch.kmtea.eu/whl/stable.html torch

# 7) install requirements except torch
grep -viE '(^|\s)(torch|pytorch)(\s|$)' requirements.txt > reqs_no_torch.txt
pip install --no-deps -r reqs_no_torch.txt

# 8) install/upgrade key libs with correct versions
pip install --upgrade "transformers==4.36.2" safetensors sentencepiece protobuf

# 9) show versions
python -c "import sys, torch, numpy as np, transformers, google.protobuf; print('py',sys.version.split()[0],'torch',torch.__version__,'numpy',np.__version__,'transformers',transformers.__version__,'protobuf',google.protobuf.__version__)"

echo
echo "=============================================================="
echo "Overload setup complete."
echo "To start using it, create a subfolder named mistral_7b_instruct inside this folder,"
echo "download the files for Mistral 7B-Instruct V0.3 into that folder, then each time you want to run Overload, open a terminal, navigate to the folder where overload.py is store, and do these terminal commands:"
echo "  eval \"\$($HOME/miniforge/bin/conda shell.bash hook)\""
echo "  conda activate overload"
echo "  python overload.py"
echo "=============================================================="
