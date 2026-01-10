#!/usr/bin/env python3
"""
Validate Gaddy EMG dataset before training.

Checks:
1. Data directories exist
2. EMG files can be loaded
3. TextGrid alignments exist and can be parsed
4. Phoneme labels are valid
5. Feature extraction works
6. Dataset can be iterated

Usage:
    python scripts/validate_data.py
    python scripts/validate_data.py --num_samples 50
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import (
    PHONEME_INVENTORY,
    NUM_PHONEMES,
    FRAME_RATE,
    load_emg_utterance,
    read_phonemes,
    get_phoneme_sequence,
)


def check_directories(data_root: str) -> dict:
    """Check required directories exist."""
    data_root = Path(data_root)

    results = {
        'data_root': str(data_root),
        'exists': data_root.exists(),
        'emg_data': None,
        'text_alignments': None,
        'testset_file': None,
    }

    # EMG data
    emg_dir = data_root / "emg_data"
    if emg_dir.exists():
        subdirs = list(emg_dir.iterdir())
        results['emg_data'] = {
            'path': str(emg_dir),
            'subdirs': [d.name for d in subdirs if d.is_dir()],
        }

    # Text alignments
    align_dir = data_root / "text_alignments"
    if align_dir.exists():
        sessions = [d.name for d in align_dir.iterdir() if d.is_dir()]
        results['text_alignments'] = {
            'path': str(align_dir),
            'num_sessions': len(sessions),
            'sessions': sessions[:5],  # First 5
        }

    # Testset file
    testset_path = data_root / "testset_largedev.json"
    if testset_path.exists():
        with open(testset_path) as f:
            testset = json.load(f)
        results['testset_file'] = {
            'path': str(testset_path),
            'dev_samples': len(testset.get('dev', [])),
            'test_samples': len(testset.get('test', [])),
        }

    return results


def check_emg_files(data_root: str, num_samples: int = 10) -> dict:
    """Check EMG files can be loaded."""
    data_root = Path(data_root)
    results = {
        'checked': 0,
        'success': 0,
        'failed': [],
        'shapes': [],
    }

    emg_dir = data_root / "emg_data"
    if not emg_dir.exists():
        return results

    # Find EMG files
    emg_files = list(emg_dir.rglob("*_emg.npy"))[:num_samples]

    for emg_file in emg_files:
        results['checked'] += 1
        try:
            emg = np.load(emg_file)
            results['success'] += 1
            results['shapes'].append(emg.shape)
        except Exception as e:
            results['failed'].append({
                'file': str(emg_file),
                'error': str(e),
            })

    if results['shapes']:
        shapes = np.array(results['shapes'])
        results['shape_stats'] = {
            'channels': int(shapes[0, 1]) if len(shapes[0]) > 1 else 1,
            'min_samples': int(shapes[:, 0].min()),
            'max_samples': int(shapes[:, 0].max()),
            'mean_samples': float(shapes[:, 0].mean()),
        }

    return results


def check_alignments(data_root: str, num_samples: int = 10) -> dict:
    """Check TextGrid alignments can be parsed."""
    data_root = Path(data_root)
    results = {
        'checked': 0,
        'success': 0,
        'failed': [],
        'phoneme_counts': Counter(),
        'unknown_phonemes': set(),
    }

    align_dir = data_root / "text_alignments"
    if not align_dir.exists():
        results['error'] = f"Alignments directory not found: {align_dir}"
        return results

    # Find TextGrid files
    tg_files = list(align_dir.rglob("*.TextGrid"))[:num_samples]

    if not tg_files:
        results['error'] = "No TextGrid files found"
        return results

    for tg_file in tg_files:
        results['checked'] += 1
        try:
            phone_ids = read_phonemes(str(tg_file))
            results['success'] += 1

            # Count phonemes
            for ph_id in phone_ids:
                if 0 <= ph_id < NUM_PHONEMES:
                    results['phoneme_counts'][PHONEME_INVENTORY[ph_id]] += 1
        except ValueError as e:
            if "not in list" in str(e):
                results['unknown_phonemes'].add(str(e))
            results['failed'].append({
                'file': str(tg_file),
                'error': str(e),
            })
        except Exception as e:
            results['failed'].append({
                'file': str(tg_file),
                'error': str(e),
            })

    results['unknown_phonemes'] = list(results['unknown_phonemes'])
    results['top_phonemes'] = results['phoneme_counts'].most_common(10)
    del results['phoneme_counts']  # Not JSON serializable

    return results


def check_feature_extraction(data_root: str, num_samples: int = 5) -> dict:
    """Check full feature extraction pipeline."""
    data_root = Path(data_root)
    results = {
        'checked': 0,
        'success': 0,
        'failed': [],
        'feature_shapes': [],
    }

    emg_dir = data_root / "emg_data"
    if not emg_dir.exists():
        return results

    # Find session directories
    for subdir in ['voiced_parallel_data', 'silent_parallel_data', 'nonparallel_data']:
        session_dir = emg_dir / subdir
        if not session_dir.exists():
            continue

        for session in list(session_dir.iterdir())[:2]:  # First 2 sessions
            if not session.is_dir():
                continue

            # Find info files
            info_files = list(session.glob("*_info.json"))[:num_samples]

            for info_file in info_files:
                idx = int(info_file.stem.split('_')[0])
                results['checked'] += 1

                try:
                    features = load_emg_utterance(str(session), idx)
                    results['success'] += 1
                    results['feature_shapes'].append({
                        'session': session.name,
                        'idx': idx,
                        'shape': features.shape,
                    })
                except Exception as e:
                    results['failed'].append({
                        'session': session.name,
                        'idx': idx,
                        'error': str(e),
                    })

    if results['feature_shapes']:
        shapes = [s['shape'] for s in results['feature_shapes']]
        results['feature_stats'] = {
            'expected_features': 112,
            'actual_features': shapes[0][1] if shapes else None,
            'min_frames': min(s[0] for s in shapes),
            'max_frames': max(s[0] for s in shapes),
        }

    return results


def check_dataset_loading(data_root: str, num_samples: int = 5) -> dict:
    """Check full dataset loading."""
    results = {
        'success': False,
        'train_size': 0,
        'dev_size': 0,
        'samples': [],
        'error': None,
    }

    try:
        from data.dataset import EMGPhonemeDataset, collate_fn
        from torch.utils.data import DataLoader

        # Load train dataset (without normalizer for speed)
        train_dataset = EMGPhonemeDataset(
            data_root=data_root,
            split="train",
            use_silent=True,
            use_voiced=True,
            cache_features=False,
        )
        results['train_size'] = len(train_dataset)

        # Load dev dataset
        dev_dataset = EMGPhonemeDataset(
            data_root=data_root,
            split="dev",
            use_silent=True,
            use_voiced=True,
            cache_features=False,
        )
        results['dev_size'] = len(dev_dataset)

        # Try loading some samples
        for i in range(min(num_samples, len(train_dataset))):
            try:
                sample = train_dataset[i]
                results['samples'].append({
                    'idx': i,
                    'emg_shape': tuple(sample['emg_features'].shape),
                    'phoneme_seq_len': len(sample['phoneme_seq']),
                    'text_preview': sample['text'][:50] + '...' if len(sample['text']) > 50 else sample['text'],
                    'is_silent': sample['is_silent'],
                })
            except Exception as e:
                results['samples'].append({
                    'idx': i,
                    'error': str(e),
                })

        # Test dataloader
        loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
        batch = next(iter(loader))
        results['batch_test'] = {
            'emg_shape': tuple(batch['emg_features'].shape),
            'emg_lengths': batch['emg_lengths'].tolist(),
            'phoneme_seq_shape': tuple(batch['phoneme_seq'].shape),
            'phoneme_lengths': batch['phoneme_lengths'].tolist(),
        }

        results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()

    return results


def main():
    parser = argparse.ArgumentParser(description='Validate EMG dataset')
    parser.add_argument('--data_root', type=str, default='data/raw', help='Data root directory')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to check')
    args = parser.parse_args()

    print("=" * 60)
    print("EMG DATASET VALIDATION")
    print("=" * 60)

    # 1. Check directories
    print("\n[1/5] Checking directories...")
    dir_results = check_directories(args.data_root)
    print(f"  Data root: {dir_results['data_root']} {'✓' if dir_results['exists'] else '✗'}")

    if dir_results['emg_data']:
        print(f"  EMG data: ✓ ({', '.join(dir_results['emg_data']['subdirs'])})")
    else:
        print("  EMG data: ✗ NOT FOUND")

    if dir_results['text_alignments']:
        print(f"  Alignments: ✓ ({dir_results['text_alignments']['num_sessions']} sessions)")
    else:
        print("  Alignments: ✗ NOT FOUND")

    if dir_results['testset_file']:
        tf = dir_results['testset_file']
        print(f"  Testset: ✓ (dev={tf['dev_samples']}, test={tf['test_samples']})")
    else:
        print("  Testset: ✗ NOT FOUND")

    # 2. Check EMG files
    print(f"\n[2/5] Checking EMG files ({args.num_samples} samples)...")
    emg_results = check_emg_files(args.data_root, args.num_samples)
    print(f"  Loaded: {emg_results['success']}/{emg_results['checked']}")
    if emg_results.get('shape_stats'):
        stats = emg_results['shape_stats']
        print(f"  Channels: {stats['channels']}")
        print(f"  Samples: {stats['min_samples']}-{stats['max_samples']} (mean: {stats['mean_samples']:.0f})")
    if emg_results['failed']:
        print(f"  Failed: {len(emg_results['failed'])}")
        for f in emg_results['failed'][:3]:
            print(f"    - {f['error']}")

    # 3. Check alignments
    print(f"\n[3/5] Checking TextGrid alignments ({args.num_samples} samples)...")
    align_results = check_alignments(args.data_root, args.num_samples)
    if 'error' in align_results:
        print(f"  ERROR: {align_results['error']}")
    else:
        print(f"  Parsed: {align_results['success']}/{align_results['checked']}")
        if align_results['top_phonemes']:
            top = ', '.join(f"{p}:{c}" for p, c in align_results['top_phonemes'][:5])
            print(f"  Top phonemes: {top}")
        if align_results['unknown_phonemes']:
            print(f"  Unknown phonemes: {align_results['unknown_phonemes']}")
        if align_results['failed']:
            print(f"  Failed: {len(align_results['failed'])}")
            for f in align_results['failed'][:3]:
                print(f"    - {f['error']}")

    # 4. Check feature extraction
    print(f"\n[4/5] Checking feature extraction...")
    feat_results = check_feature_extraction(args.data_root, 3)
    print(f"  Extracted: {feat_results['success']}/{feat_results['checked']}")
    if feat_results.get('feature_stats'):
        stats = feat_results['feature_stats']
        print(f"  Features: {stats['actual_features']} (expected: {stats['expected_features']})")
        print(f"  Frames: {stats['min_frames']}-{stats['max_frames']}")
    if feat_results['failed']:
        print(f"  Failed: {len(feat_results['failed'])}")
        for f in feat_results['failed'][:3]:
            print(f"    - {f['session']}/{f['idx']}: {f['error']}")

    # 5. Check dataset loading
    print(f"\n[5/5] Checking dataset loading...")
    ds_results = check_dataset_loading(args.data_root, 3)
    if ds_results['success']:
        print(f"  Train samples: {ds_results['train_size']}")
        print(f"  Dev samples: {ds_results['dev_size']}")
        if ds_results.get('batch_test'):
            bt = ds_results['batch_test']
            print(f"  Batch EMG shape: {bt['emg_shape']}")
            print(f"  Batch phoneme shape: {bt['phoneme_seq_shape']}")
        print("\n  Sample previews:")
        for s in ds_results['samples']:
            if 'error' in s:
                print(f"    [{s['idx']}] ERROR: {s['error']}")
            else:
                silent = "silent" if s['is_silent'] else "voiced"
                print(f"    [{s['idx']}] {s['emg_shape']} → {s['phoneme_seq_len']} phones ({silent})")
                print(f"         \"{s['text_preview']}\"")
    else:
        print(f"  ERROR: {ds_results['error']}")
        if ds_results.get('traceback'):
            print(ds_results['traceback'])

    # Summary
    print("\n" + "=" * 60)
    all_good = (
        dir_results['exists'] and
        dir_results['emg_data'] and
        dir_results['text_alignments'] and
        emg_results['success'] > 0 and
        align_results.get('success', 0) > 0 and
        feat_results['success'] > 0 and
        ds_results['success']
    )

    if all_good:
        print("STATUS: ✓ Data validated successfully!")
        print(f"Ready to train with {ds_results['train_size']} training samples.")
    else:
        print("STATUS: ✗ Data validation failed. Fix issues above.")
    print("=" * 60)

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
