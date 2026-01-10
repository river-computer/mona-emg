#!/usr/bin/env python3
"""
Evaluation script for GRU + CTC phoneme model.

Computes Phoneme Error Rate (PER) using edit distance.

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pt
    python evaluate.py --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataset import EMGPhonemeDataset, collate_fn
from data.preprocessing import PHONEME_INVENTORY, NUM_PHONEMES
from model.gru_ctc import GRUCTCModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def edit_distance(pred: List[int], target: List[int]) -> Tuple[int, int, int, int]:
    """
    Compute edit distance between two sequences.

    Returns:
        (distance, substitutions, insertions, deletions)
    """
    m, n = len(pred), len(target)

    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )

    # Backtrack to count operations
    i, j = m, n
    subs, ins, dels = 0, 0, 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred[i-1] == target[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return dp[m][n], subs, ins, dels


def compute_per(
    predictions: List[List[int]],
    targets: List[List[int]]
) -> float:
    """
    Compute Phoneme Error Rate.

    PER = (S + I + D) / N

    where:
    - S = substitutions
    - I = insertions
    - D = deletions
    - N = total phonemes in reference

    Args:
        predictions: List of predicted phoneme sequences
        targets: List of target phoneme sequences

    Returns:
        per: Phoneme Error Rate (0 to 1+)
    """
    total_distance = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        dist, _, _, _ = edit_distance(pred, target)
        total_distance += dist
        total_length += len(target)

    if total_length == 0:
        return 0.0

    return total_distance / total_length


def compute_per_detailed(
    predictions: List[List[int]],
    targets: List[List[int]]
) -> dict:
    """
    Compute detailed PER metrics.

    Returns dict with:
    - per: Overall PER
    - substitution_rate: S/N
    - insertion_rate: I/N
    - deletion_rate: D/N
    - correct: 1 - PER (accuracy)
    """
    total_subs = 0
    total_ins = 0
    total_dels = 0
    total_length = 0

    for pred, target in zip(predictions, targets):
        dist, subs, ins, dels = edit_distance(pred, target)
        total_subs += subs
        total_ins += ins
        total_dels += dels
        total_length += len(target)

    if total_length == 0:
        return {'per': 0.0, 'substitution_rate': 0.0, 'insertion_rate': 0.0, 'deletion_rate': 0.0}

    per = (total_subs + total_ins + total_dels) / total_length

    return {
        'per': per,
        'substitution_rate': total_subs / total_length,
        'insertion_rate': total_ins / total_length,
        'deletion_rate': total_dels / total_length,
        'correct': max(0, 1 - per),
        'total_phonemes': total_length,
        'total_predictions': sum(len(p) for p in predictions),
    }


def decode_and_evaluate(
    model: GRUCTCModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Decode all samples and compute PER.

    Args:
        model: Trained GRUCTCModel
        dataloader: DataLoader with test/val data
        device: torch device

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_texts = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            emg_features = batch['emg_features'].to(device)
            emg_lengths = batch['emg_lengths'].to(device)
            phoneme_seq = batch['phoneme_seq']
            phoneme_lengths = batch['phoneme_lengths']

            # Decode
            predictions, _ = model.decode_greedy(emg_features, emg_lengths)

            # Collect targets
            for i, length in enumerate(phoneme_lengths):
                target = phoneme_seq[i, :length].numpy().tolist()
                all_targets.append(target)
                all_predictions.append(predictions[i])
                all_texts.append(batch['text'][i])

    # Compute metrics
    metrics = compute_per_detailed(all_predictions, all_targets)

    return {
        **metrics,
        'predictions': all_predictions,
        'targets': all_targets,
        'texts': all_texts,
    }


def phonemes_to_string(phoneme_ids: List[int]) -> str:
    """Convert phoneme IDs to readable string."""
    return ' '.join(PHONEME_INVENTORY[p] for p in phoneme_ids)


def print_examples(
    predictions: List[List[int]],
    targets: List[List[int]],
    texts: List[str],
    n: int = 5
) -> None:
    """Print example predictions."""
    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS")
    print("="*80)

    indices = np.random.choice(len(predictions), min(n, len(predictions)), replace=False)

    for idx in indices:
        pred = predictions[idx]
        target = targets[idx]
        text = texts[idx]

        pred_str = phonemes_to_string(pred)
        target_str = phonemes_to_string(target)

        dist, subs, ins, dels = edit_distance(pred, target)
        per = dist / len(target) if target else 0

        print(f"\nText: {text[:80]}...")
        print(f"Target:     {target_str}")
        print(f"Predicted:  {pred_str}")
        print(f"PER: {per*100:.1f}% (S={subs}, I={ins}, D={dels})")


def compute_confusion_matrix(
    predictions: List[List[int]],
    targets: List[List[int]]
) -> np.ndarray:
    """
    Compute phoneme confusion matrix.

    Note: This is approximate since CTC doesn't give exact alignment.
    We use a simple heuristic based on sequence position.
    """
    confusion = np.zeros((NUM_PHONEMES, NUM_PHONEMES), dtype=np.int64)

    for pred, target in zip(predictions, targets):
        # Simple positional alignment (not perfect)
        min_len = min(len(pred), len(target))
        for i in range(min_len):
            confusion[pred[i], target[i]] += 1

    return confusion


def print_top_confusions(confusion: np.ndarray, n: int = 10) -> None:
    """Print most common phoneme confusions."""
    print("\n" + "="*80)
    print("TOP PHONEME CONFUSIONS")
    print("="*80)

    # Find off-diagonal elements
    confusions = []
    for i in range(NUM_PHONEMES):
        for j in range(NUM_PHONEMES):
            if i != j and confusion[i, j] > 0:
                confusions.append((confusion[i, j], PHONEME_INVENTORY[i], PHONEME_INVENTORY[j]))

    confusions.sort(reverse=True)

    print(f"\n{'Predicted':<10} {'Actual':<10} {'Count':<10}")
    print("-" * 30)
    for count, pred_ph, target_ph in confusions[:n]:
        print(f"{pred_ph:<10} {target_ph:<10} {count:<10}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate GRU+CTC model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'], help='Evaluation split')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to print')
    args = parser.parse_args()

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(checkpoint['config'])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = GRUCTCModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_phonemes=cfg.model.num_phonemes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    # Create dataset
    logger.info(f"Loading {args.split} dataset...")
    dataset = EMGPhonemeDataset(
        data_root=cfg.data.data_root,
        split=args.split,
        testset_file=cfg.data.testset_file,
        text_align_dir=cfg.data.text_align_dir,
        normalizer_path=cfg.data.normalizer_path,
        use_silent=cfg.data.use_silent,
        use_voiced=cfg.data.use_voiced,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Evaluate
    logger.info("Running evaluation...")
    results = decode_and_evaluate(model, dataloader, device)

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Samples: {len(results['predictions'])}")
    print(f"Total phonemes (reference): {results['total_phonemes']}")
    print(f"Total phonemes (predicted): {results['total_predictions']}")
    print()
    print(f"Phoneme Error Rate (PER): {results['per']*100:.2f}%")
    print(f"  - Substitution Rate: {results['substitution_rate']*100:.2f}%")
    print(f"  - Insertion Rate: {results['insertion_rate']*100:.2f}%")
    print(f"  - Deletion Rate: {results['deletion_rate']*100:.2f}%")
    print(f"Phoneme Accuracy: {results['correct']*100:.2f}%")

    # Print examples
    print_examples(
        results['predictions'],
        results['targets'],
        results['texts'],
        n=args.num_examples
    )

    # Confusion matrix
    confusion = compute_confusion_matrix(results['predictions'], results['targets'])
    print_top_confusions(confusion)

    return results


if __name__ == "__main__":
    main()
