#!/bin/bash
# setup.sh - Full automation for g5.xlarge (NVIDIA A10G 24GB)
# Includes S3 caching for faster subsequent downloads
set -e

echo "=== EMG GRU+CTC Setup ==="

S3_BUCKET="river-data-prod-us-east-1"
S3_PREFIX="datasets/gaddy-emg"
DATA_DIR="data/raw"

# 1. Create conda environment
echo "Creating conda environment..."
conda create -n emg python=3.10 -y || true
eval "$(conda shell.bash hook)"
conda activate emg

# 2. Install PyTorch with CUDA 11.8
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# 4. Install AWS CLI if not present
if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    pip install awscli
fi

# 5. Download Gaddy dataset (try S3 first, fallback to Zenodo)
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

download_from_s3() {
    echo "Attempting to download from S3 cache..."
    if aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/" &> /dev/null; then
        echo "S3 cache found. Downloading..."
        aws s3 sync "s3://${S3_BUCKET}/${S3_PREFIX}/" . --quiet
        return 0
    else
        echo "S3 cache not found."
        return 1
    fi
}

download_from_zenodo() {
    echo "Downloading from Zenodo..."

    # EMG data (~3GB)
    if [ ! -f "emg_data.tar.gz" ]; then
        echo "Downloading emg_data.tar.gz..."
        wget -q --show-progress https://zenodo.org/record/4064408/files/emg_data.tar.gz
    fi

    # Text alignments (~50MB)
    if [ ! -f "text_alignments.tar.gz" ]; then
        echo "Downloading text_alignments.tar.gz..."
        wget -q --show-progress https://zenodo.org/record/4064408/files/text_alignments.tar.gz
    fi

    # Testset file
    if [ ! -f "testset_largedev.json" ]; then
        echo "Downloading testset_largedev.json..."
        wget -q https://raw.githubusercontent.com/dgaddy/silent_speech/main/testset_largedev.json
    fi

    # Extract
    echo "Extracting data..."
    if [ ! -d "emg_data" ]; then
        tar -xzf emg_data.tar.gz
    fi
    if [ ! -d "text_alignments" ]; then
        tar -xzf text_alignments.tar.gz
    fi
}

upload_to_s3() {
    echo "Uploading to S3 cache for future use..."
    if aws s3 ls "s3://${S3_BUCKET}/" &> /dev/null; then
        # Upload extracted data (faster than tar.gz for subsequent downloads)
        aws s3 sync emg_data "s3://${S3_BUCKET}/${S3_PREFIX}/emg_data/" --quiet &
        aws s3 sync text_alignments "s3://${S3_BUCKET}/${S3_PREFIX}/text_alignments/" --quiet &
        aws s3 cp testset_largedev.json "s3://${S3_BUCKET}/${S3_PREFIX}/testset_largedev.json" --quiet &
        wait
        echo "S3 upload complete."
    else
        echo "Warning: Cannot access S3 bucket. Skipping upload."
    fi
}

# Try S3 first, fallback to Zenodo
if ! download_from_s3; then
    download_from_zenodo

    # Upload to S3 for next time (run in background)
    upload_to_s3 &
fi

cd ../..

# 6. Verify data structure
echo ""
echo "Verifying data structure..."
if [ -d "$DATA_DIR/emg_data" ] && [ -d "$DATA_DIR/text_alignments" ]; then
    echo "Data directories found."
    echo "  - emg_data: $(find $DATA_DIR/emg_data -name "*.npy" | wc -l) EMG files"
    echo "  - text_alignments: $(find $DATA_DIR/text_alignments -name "*.TextGrid" | wc -l) alignment files"
else
    echo "ERROR: Data directories not found!"
    exit 1
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. conda activate emg"
echo "  2. wandb login  # Optional: for experiment tracking"
echo "  3. python train.py"
echo ""
echo "To disable wandb: python train.py wandb.enabled=false"
echo ""
