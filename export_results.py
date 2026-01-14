#!/usr/bin/env python3
"""
Export test results to CSV with confidence scores.

Usage:
    python export_results.py --checkpoint checkpoints/best_model.pt --split test
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataset import EMGPhonemeDataset, collate_fn
from data.preprocessing import PHONEME_INVENTORY
from model.gru_ctc import GRUCTCModel
from evaluate import edit_distance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def phonemes_to_string(phoneme_ids: list) -> str:
    """Convert phoneme IDs to readable string."""
    return ' '.join(PHONEME_INVENTORY[p] for p in phoneme_ids)


def decode_with_confidence(model, x, lengths):
    """
    Greedy CTC decoding with confidence scores.

    Returns:
        predictions: List of decoded phoneme sequences
        confidences: List of confidence scores (avg probability of decoded phonemes)
        raw_predictions: List of frame-level predictions
    """
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)  # (B, T, C)
        probs = F.softmax(logits, dim=-1)  # (B, T, C)
        log_probs = F.log_softmax(logits, dim=-1)

        raw_preds = logits.argmax(dim=-1)  # (B, T)

        predictions = []
        confidences = []
        raw_predictions = []

        if lengths is None:
            lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)

        blank_id = model.num_phonemes

        for i, length in enumerate(lengths):
            raw_pred = raw_preds[i, :length].cpu().numpy()
            frame_probs = probs[i, :length].cpu().numpy()
            raw_predictions.append(raw_pred)

            # Collapse: remove blanks and consecutive duplicates
            pred = []
            pred_probs = []
            prev = -1

            for t, p in enumerate(raw_pred):
                if p != blank_id and p != prev:
                    pred.append(int(p))
                    pred_probs.append(frame_probs[t, p])
                prev = p

            predictions.append(pred)

            # Confidence: average probability of decoded phonemes
            if pred_probs:
                conf = np.mean(pred_probs)
            else:
                conf = 0.0
            confidences.append(conf)

    return predictions, confidences, raw_predictions


def main():
    parser = argparse.ArgumentParser(description='Export test results to CSV')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['dev', 'test'], help='Evaluation split')
    parser.add_argument('--output', type=str, default='test_results.csv', help='Output CSV file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()

    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(checkpoint['config'])

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
    model.eval()
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

    # Collect results
    results = []
    trial_id = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing"):
            emg_features = batch['emg_features'].to(device)
            emg_lengths = batch['emg_lengths'].to(device)
            phoneme_seq = batch['phoneme_seq']
            phoneme_lengths = batch['phoneme_lengths']
            texts = batch['text']
            is_silent = batch['is_silent']

            # Decode with confidence
            predictions, confidences, _ = decode_with_confidence(model, emg_features, emg_lengths)

            # Process each sample
            for i, length in enumerate(phoneme_lengths):
                target = phoneme_seq[i, :length].numpy().tolist()
                pred = predictions[i]
                conf = confidences[i]

                # Compute PER for this sample
                dist, subs, ins, dels = edit_distance(pred, target)
                per = dist / len(target) if target else 0.0

                results.append({
                    'trial_id': trial_id,
                    'text': texts[i],
                    'is_silent': is_silent[i],
                    'prediction': phonemes_to_string(pred),
                    'ground_truth': phonemes_to_string(target),
                    'per': per,
                    'confidence': conf,
                    'num_pred_phonemes': len(pred),
                    'num_target_phonemes': len(target),
                    'substitutions': subs,
                    'insertions': ins,
                    'deletions': dels,
                })
                trial_id += 1

    # Write CSV
    logger.info(f"Writing results to {args.output}")
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Summary
    avg_per = np.mean([r['per'] for r in results])
    avg_conf = np.mean([r['confidence'] for r in results])
    logger.info(f"Exported {len(results)} samples")
    logger.info(f"Average PER: {avg_per*100:.2f}%")
    logger.info(f"Average confidence: {avg_conf*100:.2f}%")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
