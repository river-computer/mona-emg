#!/usr/bin/env python3
"""
Training script for GRU + CTC phoneme model.

Usage:
    python train.py
    python train.py --config custom_config.yaml
    python train.py training.batch_size=64 model.hidden_dim=512
"""

import os
import sys
import random
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm

from data.dataset import EMGPhonemeDataset, collate_fn
from data.preprocessing import FeatureNormalizer, PHONEME_INVENTORY
from model.gru_ctc import GRUCTCModel, compute_ctc_loss
from evaluate import compute_per, decode_and_evaluate


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_scheduler(optimizer, cfg, total_steps: int):
    """Create learning rate scheduler."""
    warmup_steps = cfg.training.lr_warmup_steps

    if cfg.training.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

        # Warmup then cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.training.lr_min
        )

        class CombinedScheduler:
            def __init__(self, warmup, cosine, warmup_steps):
                self.warmup = warmup
                self.cosine = cosine
                self.warmup_steps = warmup_steps
                self.step_count = 0

            def step(self):
                if self.step_count < self.warmup_steps:
                    self.warmup.step()
                else:
                    self.cosine.step()
                self.step_count += 1

            def get_last_lr(self):
                if self.step_count < self.warmup_steps:
                    return self.warmup.get_last_lr()
                return self.cosine.get_last_lr()

        return CombinedScheduler(warmup_scheduler, cosine_scheduler, warmup_steps)

    else:
        # Step scheduler
        from torch.optim.lr_scheduler import StepLR, LambdaLR

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        return warmup_scheduler


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    optimizer.zero_grad()
    accumulation_steps = cfg.training.gradient_accumulation

    for batch_idx, batch in enumerate(pbar):
        # Move to device
        emg_features = batch['emg_features'].to(device)
        emg_lengths = batch['emg_lengths'].to(device)
        phoneme_seq = batch['phoneme_seq'].to(device)
        phoneme_lengths = batch['phoneme_lengths'].to(device)

        # Forward pass
        loss = compute_ctc_loss(
            model, emg_features, emg_lengths, phoneme_seq, phoneme_lengths
        )

        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if cfg.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Logging
            if global_step % cfg.training.log_every == 0:
                lr = scheduler.get_last_lr()[0]
                writer.add_scalar('train/loss', loss.item() * accumulation_steps, global_step)
                writer.add_scalar('train/lr', lr, global_step)

        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    avg_loss = total_loss / num_batches
    return avg_loss, global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    cfg,
    device: torch.device,
) -> dict:
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            emg_features = batch['emg_features'].to(device)
            emg_lengths = batch['emg_lengths'].to(device)
            phoneme_seq = batch['phoneme_seq'].to(device)
            phoneme_lengths = batch['phoneme_lengths'].to(device)

            # Loss
            loss = compute_ctc_loss(
                model, emg_features, emg_lengths, phoneme_seq, phoneme_lengths
            )
            total_loss += loss.item()
            num_batches += 1

            # Decode
            predictions, _ = model.decode_greedy(emg_features, emg_lengths)

            # Collect targets
            for i, length in enumerate(phoneme_lengths):
                target = phoneme_seq[i, :length].cpu().numpy().tolist()
                all_targets.append(target)
                all_predictions.append(predictions[i])

    avg_loss = total_loss / num_batches
    per = compute_per(all_predictions, all_targets)

    return {
        'loss': avg_loss,
        'per': per,
    }


def main():
    # Load config
    base_cfg = OmegaConf.load('config.yaml')
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_cfg, cli_cfg)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    # Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{cfg.experiment.name}_{timestamp}"
    writer = SummaryWriter(log_dir=f"{cfg.experiment.tensorboard_dir}/{run_name}")

    # Create datasets
    logger.info("Loading datasets...")

    # First create train dataset to compute normalizer
    train_dataset = EMGPhonemeDataset(
        data_root=cfg.data.data_root,
        split="train",
        testset_file=cfg.data.testset_file,
        text_align_dir=cfg.data.text_align_dir,
        use_silent=cfg.data.use_silent,
        use_voiced=cfg.data.use_voiced,
        cache_features=cfg.data.cache_features,
    )

    # Compute or load normalizer
    normalizer_path = Path(cfg.data.normalizer_path)
    if normalizer_path.exists():
        logger.info(f"Loading normalizer from {normalizer_path}")
        normalizer = FeatureNormalizer.load(str(normalizer_path))
    else:
        logger.info("Computing normalizer...")
        normalizer = train_dataset.compute_normalizer(num_samples=100)
        normalizer_path.parent.mkdir(parents=True, exist_ok=True)
        normalizer.save(str(normalizer_path))
        logger.info(f"Saved normalizer to {normalizer_path}")

    train_dataset.normalizer = normalizer

    # Dev dataset
    dev_dataset = EMGPhonemeDataset(
        data_root=cfg.data.data_root,
        split="dev",
        testset_file=cfg.data.testset_file,
        text_align_dir=cfg.data.text_align_dir,
        normalizer_path=str(normalizer_path),
        use_silent=cfg.data.use_silent,
        use_voiced=cfg.data.use_voiced,
        cache_features=cfg.data.cache_features,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Dev samples: {len(dev_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # Create model
    logger.info("Creating model...")
    model = GRUCTCModel(
        input_dim=cfg.model.input_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_phonemes=cfg.model.num_phonemes,
        dropout=cfg.model.dropout,
        bidirectional=cfg.model.bidirectional,
    ).to(device)

    logger.info(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    total_steps = len(train_loader) * cfg.training.max_epochs // cfg.training.gradient_accumulation
    scheduler = get_scheduler(optimizer, cfg, total_steps)

    # Training loop
    best_per = float('inf')
    global_step = 0

    logger.info("Starting training...")

    for epoch in range(1, cfg.training.max_epochs + 1):
        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, cfg, device, epoch, writer, global_step
        )
        logger.info(f"Epoch {epoch} - Train loss: {train_loss:.4f}")
        writer.add_scalar('epoch/train_loss', train_loss, epoch)

        # Validate
        if epoch % cfg.training.val_every == 0:
            val_metrics = validate(model, val_loader, cfg, device)
            logger.info(
                f"Epoch {epoch} - Val loss: {val_metrics['loss']:.4f}, "
                f"PER: {val_metrics['per']*100:.2f}%"
            )
            writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
            writer.add_scalar('epoch/val_per', val_metrics['per'], epoch)

            # Save best model
            if val_metrics['per'] < best_per:
                best_per = val_metrics['per']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_per': val_metrics['per'],
                    'val_loss': val_metrics['loss'],
                    'config': OmegaConf.to_container(cfg),
                }, checkpoint_dir / 'best_model.pt')
                logger.info(f"Saved best model with PER: {best_per*100:.2f}%")

        # Regular checkpoint
        if epoch % cfg.training.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': OmegaConf.to_container(cfg),
            }, checkpoint_dir / f'checkpoint_epoch{epoch}.pt')

    writer.close()
    logger.info(f"Training complete. Best PER: {best_per*100:.2f}%")


if __name__ == "__main__":
    main()
