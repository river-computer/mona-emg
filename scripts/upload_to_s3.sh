#!/bin/bash
# Upload Gaddy EMG dataset to S3 for faster future downloads
set -e

S3_BUCKET="river-data-prod-us-east-1"
S3_PREFIX="datasets/gaddy-emg"
DATA_DIR="data/raw"

echo "=== Uploading Gaddy EMG dataset to S3 ==="
echo "Bucket: s3://${S3_BUCKET}/${S3_PREFIX}/"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "ERROR: AWS credentials not configured."
    echo "Run: aws configure"
    exit 1
fi

# Check data exists
if [ ! -f "${DATA_DIR}/emg_data.tar.gz" ]; then
    echo "ERROR: ${DATA_DIR}/emg_data.tar.gz not found"
    exit 1
fi

# Upload tarred files
echo "Uploading emg_data.tar.gz (~3GB)..."
aws s3 cp "${DATA_DIR}/emg_data.tar.gz" "s3://${S3_BUCKET}/${S3_PREFIX}/emg_data.tar.gz"

echo "Uploading text_alignments.tar.gz..."
aws s3 cp "${DATA_DIR}/text_alignments.tar.gz" "s3://${S3_BUCKET}/${S3_PREFIX}/text_alignments.tar.gz"

if [ -f "${DATA_DIR}/testset_largedev.json" ]; then
    echo "Uploading testset_largedev.json..."
    aws s3 cp "${DATA_DIR}/testset_largedev.json" "s3://${S3_BUCKET}/${S3_PREFIX}/testset_largedev.json"
fi

echo ""
echo "=== Upload complete ==="
echo "Files available at: s3://${S3_BUCKET}/${S3_PREFIX}/"
aws s3 ls "s3://${S3_BUCKET}/${S3_PREFIX}/"
