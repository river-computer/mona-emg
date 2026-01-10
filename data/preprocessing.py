"""
EMG preprocessing pipeline adapted from Gaddy's silent_speech repository.

Signal processing chain:
1. Notch filter at 60Hz harmonics (power line interference)
2. Highpass filter at 2Hz (drift removal)
3. Resample to 516.79 Hz
4. Extract 112 features (8 channels x 14 features each)
"""

import numpy as np
import scipy.signal
import librosa
from pathlib import Path


# Phoneme inventory from Gaddy (48 classes)
PHONEME_INVENTORY = [
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'axr', 'ay',
    'b', 'ch', 'd', 'dh', 'dx', 'eh', 'el', 'em', 'en', 'er',
    'ey', 'f', 'g', 'hh', 'hv', 'ih', 'iy', 'jh',
    'k', 'l', 'm', 'n', 'nx', 'ng', 'ow', 'oy',
    'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw',
    'v', 'w', 'y', 'z', 'zh', 'sil'
]

NUM_PHONEMES = len(PHONEME_INVENTORY)  # 48
EMG_SAMPLE_RATE = 1000  # Original EMG sampling rate
FEATURE_SAMPLE_RATE = 516.79  # Downsampled rate for feature extraction
FRAME_RATE = 86.133  # Frame rate after feature extraction (~FEATURE_SAMPLE_RATE / 6)


def notch(signal: np.ndarray, freq: float, sample_frequency: float) -> np.ndarray:
    """Apply notch filter at specified frequency."""
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal: np.ndarray, freq: float, sample_frequency: float) -> np.ndarray:
    """Apply notch filter at frequency and its harmonics (1-7)."""
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def remove_drift(signal: np.ndarray, fs: float) -> np.ndarray:
    """Remove drift with 2Hz highpass filter (3rd order Butterworth)."""
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def subsample(signal: np.ndarray, new_freq: float, old_freq: float) -> np.ndarray:
    """Resample signal to new frequency using linear interpolation."""
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    return np.interp(sample_times, times, signal)


def apply_to_all_channels(function, signal_array: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Apply function to each channel of multi-channel signal."""
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, axis=1)


def double_average(x: np.ndarray) -> np.ndarray:
    """Apply double moving average filter (9-point window, applied twice)."""
    assert len(x.shape) == 1
    f = np.ones(9) / 9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w


def get_emg_features(emg_data: np.ndarray) -> np.ndarray:
    """
    Extract 112 features from 8-channel EMG data.

    For each channel (8 total), extracts:
    - w_h: Mean of smoothed signal (1)
    - p_w: RMS of smoothed signal (1)
    - p_r: RMS of residual signal (1)
    - z_p: Zero-crossing rate of residual (1)
    - r_h: Mean absolute value of residual (1)
    - s: Spectral features from STFT (9)
    = 14 features per channel

    Total: 8 channels x 14 features = 112 features

    Args:
        emg_data: (time, 8) array at FEATURE_SAMPLE_RATE (516.79 Hz)

    Returns:
        (frames, 112) array at FRAME_RATE (~86.133 Hz)
    """
    # Center each channel
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)

    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:, i]

        # Smoothed signal (double average)
        w = double_average(x)

        # Residual (high frequency content)
        p = x - w
        r = np.abs(p)

        # Frame-based features (frame_length=16, hop_length=6)
        # This gives ~86 Hz frame rate from 516.79 Hz input
        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = librosa.feature.rms(y=w, frame_length=16, hop_length=6, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(y=r, frame_length=16, hop_length=6, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

        # Spectral features (9 bins from n_fft=16)
        s = np.abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))
        # s shape: (9, frames) - transpose to (frames, 9)

        # Stack hand-crafted features: (frames, 5)
        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        # Append spectral features: (frames, 9)
        frame_features.append(s.T)

    # Concatenate all features: (frames, 112)
    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)


def load_emg_utterance(
    base_dir: str,
    index: int,
    include_context: bool = True
) -> np.ndarray:
    """
    Load and preprocess a single EMG utterance.

    Args:
        base_dir: Directory containing EMG files
        index: Utterance index
        include_context: If True, load neighboring utterances for filtering context

    Returns:
        emg_features: (frames, 112) preprocessed EMG features
    """
    base_dir = Path(base_dir)

    # Load raw EMG (1000 Hz, 8 channels)
    raw_emg = np.load(base_dir / f'{index}_emg.npy')

    if include_context:
        # Load context for better filtering at boundaries
        before_path = base_dir / f'{index-1}_emg.npy'
        after_path = base_dir / f'{index+1}_emg.npy'

        if before_path.exists():
            raw_emg_before = np.load(before_path)
        else:
            raw_emg_before = np.zeros([0, raw_emg.shape[1]])

        if after_path.exists():
            raw_emg_after = np.load(after_path)
        else:
            raw_emg_after = np.zeros([0, raw_emg.shape[1]])

        # Concatenate for filtering
        x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], axis=0)
    else:
        x = raw_emg
        raw_emg_before = np.zeros([0, raw_emg.shape[1]])

    # Apply notch filter at 60Hz harmonics
    x = apply_to_all_channels(notch_harmonics, x, 60, EMG_SAMPLE_RATE)

    # Remove drift with highpass filter
    x = apply_to_all_channels(remove_drift, x, EMG_SAMPLE_RATE)

    # Crop back to original utterance
    if include_context:
        x = x[raw_emg_before.shape[0]:x.shape[0] - (len(raw_emg_after) if len(raw_emg_after) > 0 else 0), :]

    # Resample to feature extraction rate
    x = apply_to_all_channels(subsample, x, FEATURE_SAMPLE_RATE, EMG_SAMPLE_RATE)

    # Extract features
    emg_features = get_emg_features(x)

    return emg_features


class FeatureNormalizer:
    """Normalizes features to zero mean and unit variance."""

    def __init__(self, feature_samples: list = None, share_scale: bool = False):
        """
        Args:
            feature_samples: List of (time, features) arrays to compute stats from
            share_scale: If True, use single scale for all features
        """
        if feature_samples is not None:
            feature_samples = np.concatenate(feature_samples, axis=0)
            self.feature_means = feature_samples.mean(axis=0, keepdims=True)
            if share_scale:
                self.feature_stddevs = feature_samples.std()
            else:
                self.feature_stddevs = feature_samples.std(axis=0, keepdims=True)
        else:
            self.feature_means = None
            self.feature_stddevs = None

    def normalize(self, sample: np.ndarray) -> np.ndarray:
        """Normalize sample to zero mean and unit variance."""
        if self.feature_means is None:
            return sample
        sample = sample - self.feature_means
        sample = sample / (self.feature_stddevs + 1e-8)
        return sample

    def inverse(self, sample: np.ndarray) -> np.ndarray:
        """Inverse normalization."""
        if self.feature_means is None:
            return sample
        sample = sample * self.feature_stddevs
        sample = sample + self.feature_means
        return sample

    def save(self, path: str):
        """Save normalizer to file."""
        np.savez(path, means=self.feature_means, stddevs=self.feature_stddevs)

    @classmethod
    def load(cls, path: str) -> 'FeatureNormalizer':
        """Load normalizer from file."""
        data = np.load(path)
        norm = cls()
        norm.feature_means = data['means']
        norm.feature_stddevs = data['stddevs']
        return norm


def read_phonemes(textgrid_path: str, max_len: int = None) -> np.ndarray:
    """
    Read phoneme labels from TextGrid file.

    Args:
        textgrid_path: Path to TextGrid file
        max_len: Maximum length (in frames at FRAME_RATE)

    Returns:
        phone_ids: (frames,) array of phoneme indices
    """
    import string
    from praatio import textgrid as tgio

    tg = tgio.openTextgrid(textgrid_path, includeEmptyIntervals=True)

    # Get phones tier
    phones_tier = tg.getTier('phones')

    # Initialize array
    end_time = phones_tier.entries[-1].end
    phone_ids = np.zeros(int(end_time * FRAME_RATE) + 1, dtype=np.int64)
    phone_ids[:] = -1
    phone_ids[-1] = PHONEME_INVENTORY.index('sil')

    for interval in phones_tier.entries:
        phone = interval.label.lower()

        # Normalize silence markers
        if phone in ['', 'sp', 'spn']:
            phone = 'sil'

        # Remove stress markers (numbers at end)
        if phone and phone[-1] in string.digits:
            phone = phone[:-1]

        ph_id = PHONEME_INVENTORY.index(phone)
        start_frame = int(interval.start * FRAME_RATE)
        end_frame = int(interval.end * FRAME_RATE)
        phone_ids[start_frame:end_frame] = ph_id

    assert (phone_ids >= 0).all(), 'Missing aligned phones'

    if max_len is not None:
        phone_ids = phone_ids[:max_len]

    return phone_ids


def get_phoneme_sequence(phone_ids: np.ndarray) -> np.ndarray:
    """
    Convert frame-level phoneme IDs to collapsed phoneme sequence.

    Removes consecutive duplicates to get the actual phoneme sequence
    (as needed for CTC targets).

    Args:
        phone_ids: (frames,) array of frame-level phoneme indices

    Returns:
        phoneme_seq: (seq_len,) array of collapsed phoneme sequence
    """
    # Remove consecutive duplicates
    seq = []
    prev = -1
    for ph in phone_ids:
        if ph != prev:
            seq.append(ph)
            prev = ph
    return np.array(seq, dtype=np.int64)
