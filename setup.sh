#!/bin/bash
# setup.sh - Full automation for g5.xlarge (NVIDIA A10G 24GB)
set -e

echo "=== EMG GRU+CTC Setup ==="

# 1. Create conda environment
echo "Creating conda environment..."
conda create -n emg python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate emg

# 2. Install PyTorch with CUDA 11.8
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Download Gaddy dataset from Zenodo
echo "Downloading Gaddy EMG dataset..."
mkdir -p data/raw && cd data/raw

# EMG data (~3GB)
if [ ! -f "emg_data.tar.gz" ]; then
    wget -nc https://zenodo.org/record/4064408/files/emg_data.tar.gz
fi

# Text alignments (~50MB)
if [ ! -f "text_alignments.tar.gz" ]; then
    wget -nc https://zenodo.org/record/4064408/files/text_alignments.tar.gz
fi

# Extract
echo "Extracting data..."
tar -xzf emg_data.tar.gz
tar -xzf text_alignments.tar.gz

cd ../..

# 5. Create normalizers (need to run after data is downloaded)
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. conda activate emg"
echo "  2. python train.py"
echo ""
