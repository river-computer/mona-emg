#!/usr/bin/env python3
"""
Upload model weights to S3 bucket river-weights in us-east-1

Usage:
    python scripts/upload_weights.py                          # Upload best model
    python scripts/upload_weights.py --checkpoint path/to/ckpt # Upload specific checkpoint
    python scripts/upload_weights.py --prefix my-experiment   # Custom S3 prefix
    python scripts/upload_weights.py --dry-run                # See what would upload
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)


S3_BUCKET = "river-weights"
S3_REGION = "us-east-1"
S3_PREFIX = "emg-phoneme-ctc"

DEFAULT_CHECKPOINT_DIR = "checkpoints"


def get_s3_client():
    """Create S3 client for us-east-1."""
    return boto3.client('s3', region_name=S3_REGION)


def upload_file(s3_client, local_path: str, s3_key: str, dry_run: bool = False) -> bool:
    """Upload a single file to S3."""
    if dry_run:
        print(f"  [DRY RUN] Would upload: {local_path} -> s3://{S3_BUCKET}/{s3_key}")
        return True

    try:
        print(f"  Uploading: {local_path} -> s3://{S3_BUCKET}/{s3_key}")
        s3_client.upload_file(local_path, S3_BUCKET, s3_key)
        return True
    except ClientError as e:
        print(f"  ERROR: {e}")
        return False


def find_checkpoints(checkpoint_dir: str) -> dict:
    """Find available checkpoints."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}

    if not checkpoint_dir.exists():
        return checkpoints

    # Best model
    best_model = checkpoint_dir / "best_model.pt"
    if best_model.exists():
        checkpoints['best'] = str(best_model)

    # Epoch checkpoints
    for ckpt in sorted(checkpoint_dir.glob("checkpoint_epoch*.pt")):
        epoch = ckpt.stem.replace("checkpoint_epoch", "")
        checkpoints[f'epoch_{epoch}'] = str(ckpt)

    return checkpoints


def upload_checkpoint(
    s3_client,
    checkpoint_path: str,
    prefix: str,
    tag: str = "latest",
    dry_run: bool = False,
) -> bool:
    """Upload a checkpoint with associated files."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    success = True

    # Upload checkpoint
    s3_key = f"{prefix}/{tag}/{checkpoint_path.name}"
    if not upload_file(s3_client, str(checkpoint_path), s3_key, dry_run):
        success = False

    # Also upload to timestamped folder for versioning
    s3_key_versioned = f"{prefix}/{timestamp}/{checkpoint_path.name}"
    if not upload_file(s3_client, str(checkpoint_path), s3_key_versioned, dry_run):
        success = False

    # Upload config if exists
    config_path = Path("config.yaml")
    if config_path.exists():
        s3_key = f"{prefix}/{tag}/config.yaml"
        upload_file(s3_client, str(config_path), s3_key, dry_run)
        s3_key_versioned = f"{prefix}/{timestamp}/config.yaml"
        upload_file(s3_client, str(config_path), s3_key_versioned, dry_run)

    # Upload normalizer if exists
    normalizer_path = Path("checkpoints/normalizer.npz")
    if normalizer_path.exists():
        s3_key = f"{prefix}/{tag}/normalizer.npz"
        upload_file(s3_client, str(normalizer_path), s3_key, dry_run)
        s3_key_versioned = f"{prefix}/{timestamp}/normalizer.npz"
        upload_file(s3_client, str(normalizer_path), s3_key_versioned, dry_run)

    return success


def list_remote_weights(s3_client, prefix: str):
    """List weights already in S3."""
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET,
            Prefix=prefix,
            Delimiter='/'
        )

        print(f"\nExisting weights in s3://{S3_BUCKET}/{prefix}/:")

        # List "folders"
        if 'CommonPrefixes' in response:
            for cp in response['CommonPrefixes']:
                print(f"  {cp['Prefix']}")
        else:
            print("  (none)")

    except ClientError as e:
        print(f"Could not list S3: {e}")


def main():
    parser = argparse.ArgumentParser(description='Upload model weights to S3')
    parser.add_argument('--checkpoint', type=str, help='Path to specific checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help='Checkpoint directory')
    parser.add_argument('--prefix', type=str, default=S3_PREFIX,
                        help=f'S3 prefix (default: {S3_PREFIX})')
    parser.add_argument('--tag', type=str, default='latest',
                        help='Tag for this upload (default: latest)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be uploaded without uploading')
    parser.add_argument('--list', action='store_true',
                        help='List existing weights in S3')
    args = parser.parse_args()

    s3_client = get_s3_client()

    # List existing weights
    if args.list:
        list_remote_weights(s3_client, args.prefix)
        return 0

    print("=" * 60)
    print("UPLOAD WEIGHTS TO S3")
    print("=" * 60)
    print(f"Bucket: s3://{S3_BUCKET}")
    print(f"Prefix: {args.prefix}")
    print(f"Tag: {args.tag}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    # Find or use specified checkpoint
    if args.checkpoint:
        checkpoints = {'specified': args.checkpoint}
    else:
        checkpoints = find_checkpoints(args.checkpoint_dir)

    if not checkpoints:
        print(f"ERROR: No checkpoints found in {args.checkpoint_dir}")
        return 1

    print(f"Found checkpoints:")
    for name, path in checkpoints.items():
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"  {name}: {path} ({size_mb:.1f} MB)")
    print()

    # Upload best model by default, or specified checkpoint
    if args.checkpoint:
        checkpoint_to_upload = args.checkpoint
    elif 'best' in checkpoints:
        checkpoint_to_upload = checkpoints['best']
    else:
        # Upload most recent epoch checkpoint
        checkpoint_to_upload = list(checkpoints.values())[-1]

    print(f"Uploading: {checkpoint_to_upload}")
    success = upload_checkpoint(
        s3_client,
        checkpoint_to_upload,
        args.prefix,
        args.tag,
        args.dry_run,
    )

    print()
    if success:
        print("=" * 60)
        print("Upload complete!")
        print(f"  s3://{S3_BUCKET}/{args.prefix}/{args.tag}/")
        print("=" * 60)
    else:
        print("Upload failed!")
        return 1

    # Show what's there now
    if not args.dry_run:
        list_remote_weights(s3_client, args.prefix)

    return 0


if __name__ == "__main__":
    sys.exit(main())
