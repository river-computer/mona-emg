"""
BiGRU + CTC Model for EMG → Phoneme prediction.

Architecture:
- Input: (B, T, 112) EMG features
- BiGRU: 5 layers, 768 hidden (bidirectional = 1536 total)
- Output projection: 1536 → 49 (48 phonemes + blank)
- CTC loss for training

Designed for NVIDIA A10G (24GB) on AWS g5.xlarge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GRUCTCModel(nn.Module):
    """
    Bidirectional GRU with CTC output for phoneme prediction.

    Model size: ~30M parameters with default config.
    """

    def __init__(
        self,
        input_dim: int = 112,
        hidden_dim: int = 768,
        num_layers: int = 5,
        num_phonemes: int = 48,
        dropout: float = 0.4,
        bidirectional: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (112 for Gaddy EMG)
            hidden_dim: GRU hidden dimension
            num_layers: Number of GRU layers
            num_phonemes: Number of phoneme classes (excluding blank)
            dropout: Dropout rate between GRU layers
            bidirectional: Use bidirectional GRU
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_phonemes = num_phonemes
        self.num_classes = num_phonemes + 1  # +1 for CTC blank
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Input projection (optional, helps with gradient flow)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        # Main GRU stack
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection
        gru_output_dim = hidden_dim * self.num_directions
        self.output_proj = nn.Linear(gru_output_dim, self.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Output projection
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, T, input_dim) input features
            lengths: (B,) sequence lengths for packing

        Returns:
            logits: (B, T, num_classes) output logits
        """
        batch_size, seq_len, _ = x.shape

        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = self.input_dropout(x)

        # Pack for variable length sequences
        if lengths is not None:
            # Sort by length for packing (required by pack_padded_sequence)
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            x_sorted = x[sort_idx]

            # Pack
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted,
                lengths_sorted.cpu(),
                batch_first=True,
                enforce_sorted=True
            )

            # GRU
            output_packed, _ = self.gru(x_packed)

            # Unpack
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output_packed,
                batch_first=True,
                total_length=seq_len
            )

            # Unsort
            _, unsort_idx = sort_idx.sort()
            output = output[unsort_idx]
        else:
            output, _ = self.gru(x)

        # Output projection
        logits = self.output_proj(output)

        return logits

    def get_log_probs(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get log probabilities for CTC loss.

        Args:
            x: (B, T, input_dim) input features
            lengths: (B,) sequence lengths

        Returns:
            log_probs: (T, B, num_classes) log probabilities (transposed for CTC)
        """
        logits = self.forward(x, lengths)
        log_probs = F.log_softmax(logits, dim=-1)
        # Transpose for CTC: (B, T, C) -> (T, B, C)
        log_probs = log_probs.transpose(0, 1)
        return log_probs

    def decode_greedy(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[list, list]:
        """
        Greedy CTC decoding.

        Args:
            x: (B, T, input_dim) input features
            lengths: (B,) sequence lengths

        Returns:
            predictions: List of (seq_len,) arrays with predicted phoneme indices
            raw_predictions: List of (T,) arrays with frame-level predictions
        """
        logits = self.forward(x, lengths)  # (B, T, C)

        # Argmax decoding
        raw_preds = logits.argmax(dim=-1)  # (B, T)

        predictions = []
        raw_predictions = []

        if lengths is None:
            lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)

        for i, length in enumerate(lengths):
            raw_pred = raw_preds[i, :length].cpu().numpy()
            raw_predictions.append(raw_pred)

            # Collapse: remove blanks and consecutive duplicates
            pred = []
            prev = -1
            blank_id = self.num_phonemes  # Blank is last class

            for p in raw_pred:
                if p != blank_id and p != prev:
                    pred.append(int(p))
                prev = p

            predictions.append(pred)

        return predictions, raw_predictions

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def compute_ctc_loss(
    model: GRUCTCModel,
    emg_features: torch.Tensor,
    emg_lengths: torch.Tensor,
    phoneme_seq: torch.Tensor,
    phoneme_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Compute CTC loss.

    Args:
        model: GRUCTCModel instance
        emg_features: (B, T, 112) input features
        emg_lengths: (B,) input sequence lengths
        phoneme_seq: (B, S) target phoneme sequences
        phoneme_lengths: (B,) target sequence lengths

    Returns:
        loss: Scalar CTC loss
    """
    # Get log probabilities
    log_probs = model.get_log_probs(emg_features, emg_lengths)  # (T, B, C)

    # CTC loss expects:
    # - log_probs: (T, B, C)
    # - targets: (B, S) or (sum(target_lengths),)
    # - input_lengths: (B,)
    # - target_lengths: (B,)
    loss = F.ctc_loss(
        log_probs=log_probs,
        targets=phoneme_seq,
        input_lengths=emg_lengths,
        target_lengths=phoneme_lengths,
        blank=model.num_phonemes,  # Blank is last class
        reduction='mean',
        zero_infinity=True,  # Prevent inf gradients
    )

    return loss


if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GRUCTCModel(
        input_dim=112,
        hidden_dim=768,
        num_layers=5,
        num_phonemes=48,
        dropout=0.4,
    ).to(device)

    print(f"Model parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 4
    seq_len = 200
    x = torch.randn(batch_size, seq_len, 112).to(device)
    lengths = torch.tensor([200, 180, 150, 100]).to(device)

    logits = model(x, lengths)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test CTC loss
    phoneme_seq = torch.randint(0, 48, (batch_size, 30)).to(device)
    phoneme_lengths = torch.tensor([30, 28, 25, 20]).to(device)

    loss = compute_ctc_loss(model, x, lengths, phoneme_seq, phoneme_lengths)
    print(f"CTC loss: {loss.item():.4f}")

    # Test decoding
    preds, raw_preds = model.decode_greedy(x, lengths)
    print(f"Decoded lengths: {[len(p) for p in preds]}")
