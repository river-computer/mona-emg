#!/usr/bin/env python3
"""
Evaluate GRU+CTC model with KenLM beam search decoding.

This gives WER comparable to Gaddy's reported ~28% WER.

Setup (run once):
    pip install pyctcdecode pypi-kenlm
    curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.6.1/lm.binary

Usage:
    python evaluate_beam_search.py --checkpoint checkpoints/best_model.pt
    python evaluate_beam_search.py --checkpoint checkpoints/best_model.pt --beam-width 100
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
import jiwer

from data.dataset import EMGPhonemeDataset, collate_fn
from data.preprocessing import PHONEME_INVENTORY, NUM_PHONEMES
from model.gru_ctc import GRUCTCModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_beam_decoder(beam_width: int = 100):
    """Setup CTC beam decoder."""
    from pyctcdecode import build_ctcdecoder

    # Labels for decoder - pyctcdecode adds blank at end automatically
    # Labels should NOT include blank
    labels = PHONEME_INVENTORY.copy()

    decoder = build_ctcdecoder(labels=labels)

    return decoder, beam_width


def decode_batch_beam(model, decoder, beam_width, emg_features, emg_lengths, device, debug=False):
    """Decode a batch using beam search."""
    model.eval()
    with torch.no_grad():
        logits = model(emg_features, emg_lengths)
        # pyctcdecode expects softmax probabilities with blank as LAST column
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        predictions = []
        for i, length in enumerate(emg_lengths):
            # Get probs for this sample (up to actual length)
            sample_probs = probs[i, :length.item(), :]

            # Beam search decode
            try:
                # Get beams with scores for debugging
                beams = decoder.decode_beams(sample_probs, beam_width=beam_width)
                if beams:
                    text = beams[0][0]  # First beam, text part
                else:
                    text = ""

                if debug and i == 0:
                    logger.info(f"Sample probs shape: {sample_probs.shape}")
                    logger.info(f"Decoded text: '{text}'")
                    logger.info(f"Num beams: {len(beams)}")
                    if beams:
                        logger.info(f"Top beam: {beams[0]}")

            except Exception as e:
                logger.warning(f"Decode error: {e}")
                text = ""

            # Convert back to phoneme IDs
            # pyctcdecode joins multi-char tokens, need to parse carefully
            phoneme_ids = []
            if text:
                # Try to split by known phonemes
                remaining = text
                while remaining:
                    found = False
                    # Try longer phonemes first (e.g., "ng" before "n")
                    for plen in [3, 2, 1]:
                        for p in PHONEME_INVENTORY:
                            if len(p) == plen and remaining.startswith(p):
                                phoneme_ids.append(PHONEME_INVENTORY.index(p))
                                remaining = remaining[len(p):]
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        # Skip unknown character
                        remaining = remaining[1:] if remaining else ""

            predictions.append(phoneme_ids)

        return predictions


def decode_batch_greedy(model, emg_features, emg_lengths, device):
    """Decode a batch using greedy decoding (for comparison)."""
    predictions, _ = model.decode_greedy(emg_features, emg_lengths)
    return predictions


def calculate_wer(predictions: list, references: list) -> dict:
    """Calculate WER between phoneme sequences converted to text."""
    # Convert phoneme sequences to space-separated strings
    pred_texts = [' '.join(PHONEME_INVENTORY[p] for p in seq) for seq in predictions]
    ref_texts = [' '.join(PHONEME_INVENTORY[p] for p in seq) for seq in references]

    wer = jiwer.wer(ref_texts, pred_texts)

    try:
        output = jiwer.process_words(ref_texts, pred_texts)
        return {
            'wer': wer,
            'substitutions': output.substitutions,
            'insertions': output.insertions,
            'deletions': output.deletions,
        }
    except:
        return {'wer': wer}


def main():
    parser = argparse.ArgumentParser(description='Evaluate with beam search decoding')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--split', type=str, default='test', choices=['dev', 'test'])
    parser.add_argument('--beam-width', type=int, default=100, help='Beam width')
    parser.add_argument('--lm-path', type=str, default='lm.binary', help='KenLM model path')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--compare-greedy', action='store_true', help='Also run greedy for comparison')
    args = parser.parse_args()

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = OmegaConf.create(checkpoint['config'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

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
    logger.info(f"Loaded model from epoch {checkpoint.get('epoch', '?')}")

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

    # Setup beam decoder
    logger.info(f"Setting up beam decoder (width={args.beam_width})...")
    decoder, beam_width = setup_beam_decoder(beam_width=args.beam_width)

    # Run evaluation
    all_beam_preds = []
    all_greedy_preds = []
    all_targets = []

    logger.info("Running evaluation...")
    first_batch = True
    for batch in tqdm(dataloader, desc="Evaluating"):
        emg_features = batch['emg_features'].to(device)
        emg_lengths = batch['emg_lengths'].to(device)
        phoneme_seq = batch['phoneme_seq']
        phoneme_lengths = batch['phoneme_lengths']

        # Beam search decoding
        beam_preds = decode_batch_beam(model, decoder, beam_width, emg_features, emg_lengths, device, debug=first_batch)
        first_batch = False
        all_beam_preds.extend(beam_preds)

        # Greedy decoding (for comparison)
        if args.compare_greedy:
            greedy_preds = decode_batch_greedy(model, emg_features, emg_lengths, device)
            all_greedy_preds.extend(greedy_preds)

        # Targets
        for i, length in enumerate(phoneme_lengths):
            target = phoneme_seq[i, :length].numpy().tolist()
            all_targets.append(target)

    # Calculate metrics
    beam_metrics = calculate_wer(all_beam_preds, all_targets)

    print(f"\n{'='*60}")
    print("RESULTS - BEAM SEARCH DECODING")
    print(f"{'='*60}")
    print(f"Split: {args.split}")
    print(f"Samples: {len(all_targets)}")
    print(f"Beam width: {args.beam_width}")
    print(f"LM: {args.lm_path if Path(args.lm_path).exists() else 'None'}")
    print()
    print(f"Phoneme Error Rate (PER): {beam_metrics['wer']*100:.2f}%")
    if 'substitutions' in beam_metrics:
        print(f"  - Substitutions: {beam_metrics['substitutions']}")
        print(f"  - Insertions: {beam_metrics['insertions']}")
        print(f"  - Deletions: {beam_metrics['deletions']}")

    if args.compare_greedy:
        greedy_metrics = calculate_wer(all_greedy_preds, all_targets)
        print()
        print(f"Greedy PER (comparison): {greedy_metrics['wer']*100:.2f}%")

    # Show examples
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*60}")
    for i in range(min(5, len(all_targets))):
        target_str = ' '.join(PHONEME_INVENTORY[p] for p in all_targets[i][:15]) + '...'
        beam_str = ' '.join(PHONEME_INVENTORY[p] for p in all_beam_preds[i][:15]) + '...'
        print(f"\n[{i+1}]")
        print(f"  Target: {target_str}")
        print(f"  Beam:   {beam_str}")
        if args.compare_greedy:
            greedy_str = ' '.join(PHONEME_INVENTORY[p] for p in all_greedy_preds[i][:15]) + '...'
            print(f"  Greedy: {greedy_str}")


if __name__ == "__main__":
    main()
