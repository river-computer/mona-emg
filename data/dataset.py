"""
EMG Phoneme Dataset for CTC training.

Handles both voiced and silent EMG data from the Gaddy dataset.
For silent data, uses DTW alignment to transfer phoneme labels from paired voiced utterances.
"""

import json
import os
import re
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .preprocessing import (
    load_emg_utterance,
    read_phonemes,
    get_phoneme_sequence,
    FeatureNormalizer,
    PHONEME_INVENTORY,
    NUM_PHONEMES,
    FRAME_RATE,
)


class EMGPhonemeDataset(Dataset):
    """
    Dataset for EMG → Phoneme CTC training.

    Each sample contains:
    - emg_features: (T, 112) EMG features
    - phoneme_seq: (S,) collapsed phoneme sequence for CTC
    - phoneme_frames: (T,) frame-level phoneme labels
    - text: Original text
    - session_id: Recording session index
    """

    def __init__(
        self,
        data_root: str = "data/raw",
        split: str = "train",  # train, dev, test
        testset_file: str = "data/raw/testset_largedev.json",
        text_align_dir: str = "data/raw/text_alignments",
        normalizer_path: Optional[str] = None,
        use_silent: bool = True,
        use_voiced: bool = True,
        cache_features: bool = True,
    ):
        """
        Args:
            data_root: Root directory containing emg_data/
            split: Data split (train, dev, test)
            testset_file: JSON file with dev/test indices
            text_align_dir: Directory with TextGrid alignments
            normalizer_path: Path to saved normalizer (None to compute)
            use_silent: Include silent EMG data
            use_voiced: Include voiced EMG data
            cache_features: Cache extracted features in memory
        """
        self.data_root = Path(data_root)
        self.split = split
        self.text_align_dir = Path(text_align_dir)
        self.cache_features = cache_features
        self._cache = {}

        # Load test/dev split
        if os.path.exists(testset_file):
            with open(testset_file) as f:
                testset_json = json.load(f)
                self.devset = set(tuple(x) for x in testset_json['dev'])
                self.testset = set(tuple(x) for x in testset_json['test'])
        else:
            self.devset = set()
            self.testset = set()

        # Find all data directories
        self.directories = []
        session_idx = 0

        if use_silent:
            silent_dir = self.data_root / "emg_data" / "silent_parallel_data"
            if silent_dir.exists():
                for session_dir in sorted(os.listdir(silent_dir)):
                    self.directories.append({
                        'path': silent_dir / session_dir,
                        'session_idx': session_idx,
                        'silent': True,
                    })
                    session_idx += 1

        if use_voiced:
            for voiced_subdir in ["voiced_parallel_data", "nonparallel_data"]:
                voiced_dir = self.data_root / "emg_data" / voiced_subdir
                if voiced_dir.exists():
                    for session_dir in sorted(os.listdir(voiced_dir)):
                        self.directories.append({
                            'path': voiced_dir / session_dir,
                            'session_idx': session_idx,
                            'silent': False,
                        })
                        session_idx += 1

        # Build example index
        self.examples = []
        self.voiced_data_map = {}  # Map (book, sentence_idx) -> (dir_info, file_idx)

        for dir_info in self.directories:
            dir_path = dir_info['path']
            for fname in os.listdir(dir_path):
                m = re.match(r'(\d+)_info.json', fname)
                if m is not None:
                    file_idx = int(m.group(1))
                    with open(dir_path / fname) as f:
                        info = json.load(f)

                    # Skip boundary clips
                    if info['sentence_index'] < 0:
                        continue

                    book_loc = (info['book'], info['sentence_index'])

                    # Determine split
                    in_dev = book_loc in self.devset
                    in_test = book_loc in self.testset

                    include = False
                    if split == "train" and not in_dev and not in_test:
                        include = True
                    elif split == "dev" and in_dev:
                        include = True
                    elif split == "test" and in_test:
                        include = True

                    if include:
                        self.examples.append({
                            'dir_info': dir_info,
                            'file_idx': file_idx,
                            'text': info['text'],
                            'book_loc': book_loc,
                        })

                    # Track voiced data for silent alignment
                    if not dir_info['silent']:
                        self.voiced_data_map[book_loc] = (dir_info, file_idx)

        # Sort and shuffle (with fixed seed for reproducibility)
        self.examples.sort(key=lambda x: (x['dir_info']['session_idx'], x['file_idx']))
        random.seed(42)
        random.shuffle(self.examples)

        # Load or create normalizer
        if normalizer_path and os.path.exists(normalizer_path):
            self.normalizer = FeatureNormalizer.load(normalizer_path)
        else:
            self.normalizer = None

        self.num_sessions = session_idx

    def __len__(self) -> int:
        return len(self.examples)

    def _load_features(self, dir_path: Path, file_idx: int) -> np.ndarray:
        """Load and preprocess EMG features."""
        cache_key = (str(dir_path), file_idx)
        if self.cache_features and cache_key in self._cache:
            return self._cache[cache_key]

        features = load_emg_utterance(str(dir_path), file_idx)

        # Normalize
        if self.normalizer is not None:
            features = self.normalizer.normalize(features)
            # Apply tanh compression for outliers (from Gaddy)
            features = 8 * np.tanh(features / 8)

        if self.cache_features:
            self._cache[cache_key] = features

        return features

    def _load_phonemes(
        self,
        dir_info: dict,
        file_idx: int,
        book_loc: tuple,
        max_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load phoneme labels.

        For voiced data: directly from TextGrid
        For silent data: from paired voiced utterance via alignment
        """
        session_name = dir_info['path'].name

        if not dir_info['silent']:
            # Voiced: direct alignment
            tg_path = self.text_align_dir / session_name / f"{session_name}_{file_idx}_audio.TextGrid"
            if tg_path.exists():
                phone_frames = read_phonemes(str(tg_path), max_len)
            else:
                # No alignment available - use silence
                phone_frames = np.full(max_len, PHONEME_INVENTORY.index('sil'), dtype=np.int64)
        else:
            # Silent: get phonemes from paired voiced utterance
            if book_loc in self.voiced_data_map:
                voiced_dir_info, voiced_idx = self.voiced_data_map[book_loc]
                voiced_session = voiced_dir_info['path'].name
                tg_path = self.text_align_dir / voiced_session / f"{voiced_session}_{voiced_idx}_audio.TextGrid"
                if tg_path.exists():
                    # Load voiced phonemes
                    voiced_phones = read_phonemes(str(tg_path))
                    # Simple resampling to match silent length
                    # (More sophisticated: use DTW alignment)
                    indices = np.linspace(0, len(voiced_phones) - 1, max_len).astype(int)
                    phone_frames = voiced_phones[indices]
                else:
                    phone_frames = np.full(max_len, PHONEME_INVENTORY.index('sil'), dtype=np.int64)
            else:
                phone_frames = np.full(max_len, PHONEME_INVENTORY.index('sil'), dtype=np.int64)

        # Get collapsed sequence for CTC
        phone_seq = get_phoneme_sequence(phone_frames)

        return phone_frames, phone_seq

    def __getitem__(self, idx: int) -> Dict:
        """Get a single example."""
        example = self.examples[idx]
        dir_info = example['dir_info']
        file_idx = example['file_idx']

        # Load EMG features
        emg_features = self._load_features(dir_info['path'], file_idx)

        # Load phoneme labels
        phone_frames, phone_seq = self._load_phonemes(
            dir_info, file_idx, example['book_loc'], emg_features.shape[0]
        )

        return {
            'emg_features': torch.from_numpy(emg_features).float(),
            'phoneme_seq': torch.from_numpy(phone_seq).long(),
            'phoneme_frames': torch.from_numpy(phone_frames).long(),
            'text': example['text'],
            'session_id': dir_info['session_idx'],
            'is_silent': dir_info['silent'],
            'file_idx': file_idx,
        }

    def compute_normalizer(self, num_samples: int = 100) -> FeatureNormalizer:
        """Compute feature normalizer from dataset samples."""
        samples = []
        indices = list(range(min(num_samples, len(self))))
        random.shuffle(indices)

        for idx in indices[:num_samples]:
            example = self.examples[idx]
            features = load_emg_utterance(
                str(example['dir_info']['path']),
                example['file_idx']
            )
            samples.append(features)

        self.normalizer = FeatureNormalizer(samples, share_scale=False)
        return self.normalizer


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.

    Pads sequences to same length within batch.
    """
    # Get lengths
    emg_lengths = [x['emg_features'].shape[0] for x in batch]
    phoneme_lengths = [x['phoneme_seq'].shape[0] for x in batch]

    # Pad EMG features
    emg_padded = pad_sequence(
        [x['emg_features'] for x in batch],
        batch_first=True,
        padding_value=0.0
    )

    # Pad phoneme sequences
    phoneme_padded = pad_sequence(
        [x['phoneme_seq'] for x in batch],
        batch_first=True,
        padding_value=0  # Will be ignored by CTC
    )

    # Pad frame-level phonemes
    frames_padded = pad_sequence(
        [x['phoneme_frames'] for x in batch],
        batch_first=True,
        padding_value=PHONEME_INVENTORY.index('sil')
    )

    return {
        'emg_features': emg_padded,  # (B, T, 112)
        'emg_lengths': torch.tensor(emg_lengths, dtype=torch.long),  # (B,)
        'phoneme_seq': phoneme_padded,  # (B, S)
        'phoneme_lengths': torch.tensor(phoneme_lengths, dtype=torch.long),  # (B,)
        'phoneme_frames': frames_padded,  # (B, T)
        'text': [x['text'] for x in batch],
        'session_ids': torch.tensor([x['session_id'] for x in batch], dtype=torch.long),
        'is_silent': [x['is_silent'] for x in batch],
    }


class SizeAwareSampler(torch.utils.data.Sampler):
    """
    Sampler that creates batches with similar total sequence length.
    More efficient than fixed batch size for variable-length sequences.
    """

    def __init__(self, dataset: EMGPhonemeDataset, max_batch_length: int = 128000):
        """
        Args:
            dataset: EMGPhonemeDataset instance
            max_batch_length: Maximum total frames per batch
        """
        self.dataset = dataset
        self.max_batch_length = max_batch_length

        # Pre-compute lengths
        self.lengths = []
        for ex in dataset.examples:
            info_path = ex['dir_info']['path'] / f"{ex['file_idx']}_info.json"
            with open(info_path) as f:
                info = json.load(f)
            # Approximate length from chunks
            length = sum(emg_len for emg_len, _, _ in info.get('chunks', [[1000, 0, 0]]))
            self.lengths.append(length)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        batch = []
        batch_length = 0

        for idx in indices:
            length = self.lengths[idx]

            if length > self.max_batch_length:
                # Single example too long - yield anyway
                if batch:
                    yield batch
                yield [idx]
                batch = []
                batch_length = 0
                continue

            if batch_length + length > self.max_batch_length:
                yield batch
                batch = []
                batch_length = 0

            batch.append(idx)
            batch_length += length

        if batch:
            yield batch

    def __len__(self):
        # Approximate
        return len(self.dataset) // 32


if __name__ == "__main__":
    # Test dataset loading
    dataset = EMGPhonemeDataset(
        data_root="data/raw",
        split="train",
        use_silent=True,
        use_voiced=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Num sessions: {dataset.num_sessions}")
    print(f"Num phonemes: {NUM_PHONEMES}")

    # Test single example
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample:")
        print(f"  EMG shape: {sample['emg_features'].shape}")
        print(f"  Phoneme seq shape: {sample['phoneme_seq'].shape}")
        print(f"  Text: {sample['text'][:50]}...")

        # Test collate
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))
        print(f"\nBatch:")
        print(f"  EMG shape: {batch['emg_features'].shape}")
        print(f"  EMG lengths: {batch['emg_lengths']}")
        print(f"  Phoneme seq shape: {batch['phoneme_seq'].shape}")
        print(f"  Phoneme lengths: {batch['phoneme_lengths']}")
