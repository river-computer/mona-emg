#!/usr/bin/env python3
"""
Create OpenAI fine-tuning dataset from EMG phoneme predictions.

Converts test_results.csv (from export_results.py) to OpenAI fine-tuning JSONL format.
Stratifies samples by PER to ensure coverage of different error levels.

Usage:
    python create_finetune_dataset.py --input test_results.csv
    python create_finetune_dataset.py --input test_results.csv --n-samples 150
"""

import argparse
import csv
import json
import random
from pathlib import Path


SYSTEM_PROMPT = """You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format, space-separated. Common phonemes:
- Vowels: aa, ae, ah, ao, aw, ay, eh, er, ey, ih, iy, ow, oy, uh, uw
- Consonants: b, ch, d, dh, f, g, hh, jh, k, l, m, n, ng, p, r, s, sh, t, th, v, w, y, z, zh
- Silence: sil

Output only the decoded English sentence, nothing else."""


def load_test_results(csv_path: str) -> list[dict]:
    """Load test results from CSV file."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip samples without ground truth text
            if not row.get('text', '').strip():
                continue

            samples.append({
                'trial_id': row.get('trial_id', ''),
                'phonemes': row['prediction'],  # predicted phonemes
                'ground_truth_text': row['text'],
                'ground_truth_phonemes': row['ground_truth'],
                'per': float(row.get('per', 0)),
                'confidence': float(row.get('confidence', 0)),
                'is_silent': row.get('is_silent', 'False') == 'True',
            })

    return samples


def stratified_sample(samples: list[dict], n_samples: int, seed: int = 42) -> list[dict]:
    """
    Stratified sampling by PER.

    Strata:
        - Low PER: 0-15%
        - Medium PER: 15-30%
        - High PER: 30%+
    """
    random.seed(seed)

    # Define strata (adjusted for EMG which has higher PER)
    low = [s for s in samples if s['per'] < 0.15]
    medium = [s for s in samples if 0.15 <= s['per'] < 0.30]
    high = [s for s in samples if s['per'] >= 0.30]

    print(f"Stratum sizes:")
    print(f"  Low PER (0-15%):    {len(low):,} samples")
    print(f"  Medium PER (15-30%): {len(medium):,} samples")
    print(f"  High PER (30%+):    {len(high):,} samples")

    # Sample from each stratum
    n_per_stratum = n_samples // 3
    remainder = n_samples % 3

    selected = []

    # Sample from each stratum
    for i, (stratum, name) in enumerate([(low, 'low'), (medium, 'medium'), (high, 'high')]):
        n = n_per_stratum + (1 if i < remainder else 0)
        if len(stratum) < n:
            print(f"  Warning: {name} stratum has only {len(stratum)} samples, using all")
            selected.extend(stratum)
        else:
            selected.extend(random.sample(stratum, n))

    random.shuffle(selected)
    return selected


def create_finetune_example(sample: dict) -> dict:
    """Create a single fine-tuning example in OpenAI format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample['phonemes']},
            {"role": "assistant", "content": sample['ground_truth_text']}
        ]
    }


def main():
    parser = argparse.ArgumentParser(description='Create OpenAI fine-tuning dataset')
    parser.add_argument('--input', type=str, default='test_results.csv',
                        help='Input CSV file from export_results.py')
    parser.add_argument('--output', type=str, default='finetune_train.jsonl',
                        help='Output JSONL file for training')
    parser.add_argument('--output-test', type=str, default='finetune_test.jsonl',
                        help='Output JSONL file for testing')
    parser.add_argument('--n-train', type=int, default=50,
                        help='Number of training samples (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--stratify', action='store_true',
                        help='Use stratified sampling by PER')
    args = parser.parse_args()

    print(f"{'='*60}")
    print("CREATE OPENAI FINE-TUNING DATASET")
    print(f"{'='*60}")
    print(f"Input: {args.input}")

    # Load data
    print(f"\nLoading data...")
    all_samples = load_test_results(args.input)
    print(f"Loaded {len(all_samples):,} samples")

    if len(all_samples) == 0:
        print("ERROR: No samples found!")
        return 1

    # Calculate PER statistics
    pers = [s['per'] for s in all_samples]
    print(f"\nPER statistics:")
    print(f"  Min:    {min(pers)*100:.2f}%")
    print(f"  Max:    {max(pers)*100:.2f}%")
    print(f"  Mean:   {sum(pers)/len(pers)*100:.2f}%")
    print(f"  Median: {sorted(pers)[len(pers)//2]*100:.2f}%")

    # Shuffle and split: n_train for training, rest for testing
    random.seed(args.seed)
    shuffled = all_samples.copy()
    random.shuffle(shuffled)

    n_train = min(args.n_train, len(shuffled))

    # Sample training data
    if args.stratify:
        print(f"\nSelecting {n_train} stratified samples for training...")
        train_samples = stratified_sample(shuffled, n_train, args.seed)
        train_ids = {s['trial_id'] for s in train_samples}
        test_samples = [s for s in shuffled if s['trial_id'] not in train_ids]
    else:
        train_samples = shuffled[:n_train]
        test_samples = shuffled[n_train:]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_samples):,} samples")
    print(f"  Test:  {len(test_samples):,} samples")

    # Create output directory if needed
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Write train JSONL
    print(f"\nWriting train set to {args.output}...")
    with open(args.output, 'w') as f:
        for sample in train_samples:
            example = create_finetune_example(sample)
            f.write(json.dumps(example) + '\n')

    # Write test JSONL
    print(f"Writing test set to {args.output_test}...")
    with open(args.output_test, 'w') as f:
        for sample in test_samples:
            example = create_finetune_example(sample)
            f.write(json.dumps(example) + '\n')

    # Show sample examples
    print(f"\n{'='*60}")
    print("Sample training examples:")
    print(f"{'='*60}")
    for i, sample in enumerate(train_samples[:3]):
        print(f"\nExample {i+1} (PER: {sample['per']*100:.1f}%):")
        print(f"  Phonemes: {sample['phonemes'][:70]}...")
        print(f"  Text:     {sample['ground_truth_text']}")

    # Show JSONL format
    print(f"\n{'='*60}")
    print("JSONL format preview (first line):")
    print(f"{'='*60}")
    example = create_finetune_example(train_samples[0])
    print(json.dumps(example, indent=2)[:500] + "...")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    print(f"Train file: {args.output} ({len(train_samples)} samples)")
    print(f"Test file:  {args.output_test} ({len(test_samples)} samples)")
    print(f"\nTo upload for fine-tuning:")
    print(f"  openai api files.create -f {args.output} -p fine-tune")
    print(f"\nOr with Python:")
    print(f"  from openai import OpenAI")
    print(f"  client = OpenAI()")
    print(f"  client.files.create(file=open('{args.output}', 'rb'), purpose='fine-tune')")


if __name__ == '__main__':
    main()
