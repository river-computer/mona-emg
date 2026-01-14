#!/usr/bin/env python3
"""
Convert phoneme predictions to text using LLM and calculate WER.

Takes EMG model phoneme predictions, uses Claude/OpenAI to decode to text,
and computes Word Error Rate against ground truth.

Usage:
    # Zero-shot with Claude
    python phoneme_to_text.py --input test_results.csv --provider anthropic --model claude-sonnet-4-20250514

    # Few-shot with OpenAI
    python phoneme_to_text.py --input test_results.csv --provider openai --model gpt-4o --mode few-shot

    # Test with limit
    python phoneme_to_text.py --input test_results.csv --provider anthropic --model claude-sonnet-4-20250514 --limit 10
"""

import argparse
import json
import csv
import time
import random
from pathlib import Path
from tqdm import tqdm

try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False

# Lazy imports for providers
openai_client = None
anthropic_client = None


SYSTEM_PROMPT_ZERO_SHOT = """You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format, space-separated. Common phonemes:
- Vowels: aa, ae, ah, ao, aw, ay, eh, er, ey, ih, iy, ow, oy, uh, uw
- Consonants: b, ch, d, dh, f, g, hh, jh, k, l, m, n, ng, p, r, s, sh, t, th, v, w, y, z, zh
- Silence: sil

Output only the decoded English sentence, nothing else. Do not include phonemes or explanations."""


SYSTEM_PROMPT_FEW_SHOT = """You are a phoneme-to-text decoder. Convert the input phoneme sequence to the most likely English sentence.

Phonemes are in ARPABET format, space-separated.

Here are some examples:

{examples}

Now convert the following phoneme sequence. Output only the decoded English sentence, nothing else."""


def get_openai_client():
    global openai_client
    if openai_client is None:
        from openai import OpenAI
        openai_client = OpenAI()
    return openai_client


def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()
    return anthropic_client


def load_test_results(csv_path: str) -> list[dict]:
    """Load test results from CSV."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('text', '').strip():
                continue

            samples.append({
                'trial_id': row.get('trial_id', ''),
                'phonemes': row['prediction'],  # predicted phonemes
                'ground_truth_phonemes': row['ground_truth'],
                'ground_truth_text': row['text'],
                'per': float(row.get('per', 0)),
                'confidence': float(row.get('confidence', 0)),
                'is_silent': row.get('is_silent', 'False') == 'True',
            })
    return samples


def select_few_shot_examples(samples: list[dict], n_shots: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Select few-shot examples and return (examples, remaining_samples)."""
    random.seed(seed)

    # Prefer high-confidence, low-PER examples
    sorted_samples = sorted(samples, key=lambda x: (x['per'], -x['confidence']))

    # Take top examples
    examples = sorted_samples[:n_shots]
    example_ids = {e['trial_id'] for e in examples}
    remaining = [s for s in samples if s['trial_id'] not in example_ids]

    return examples, remaining


def format_few_shot_examples(examples: list[dict]) -> str:
    """Format examples for the prompt."""
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"Example {i}:")
        formatted.append(f"Input: {ex['phonemes']}")
        formatted.append(f"Output: {ex['ground_truth_text']}")
        formatted.append("")
    return "\n".join(formatted)


def call_openai(model: str, system_prompt: str, user_content: str, max_retries: int = 10) -> tuple[str, dict]:
    """Call OpenAI API with retry logic."""
    from openai import RateLimitError, APIError, APITimeoutError

    client = get_openai_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=256,
                temperature=0,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return response.choices[0].message.content.strip(), {
                'time_ms': round(elapsed_ms, 2),
                'tokens_in': response.usage.prompt_tokens,
                'tokens_out': response.usage.completion_tokens,
            }

        except RateLimitError:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            tqdm.write(f"Rate limit. Waiting {wait:.1f}s...")
            time.sleep(wait)
        except (APIError, APITimeoutError) as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            tqdm.write(f"API error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    return "ERROR: Max retries exceeded", {'time_ms': 0, 'tokens_in': 0, 'tokens_out': 0}


def call_anthropic(model: str, system_prompt: str, user_content: str, max_retries: int = 10) -> tuple[str, dict]:
    """Call Anthropic API with retry logic."""
    import anthropic

    client = get_anthropic_client()

    for attempt in range(max_retries):
        try:
            start_time = time.perf_counter()
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return response.content[0].text.strip(), {
                'time_ms': round(elapsed_ms, 2),
                'tokens_in': response.usage.input_tokens,
                'tokens_out': response.usage.output_tokens,
            }

        except anthropic.RateLimitError:
            wait = min(60, 2 ** attempt) + random.uniform(0, 1)
            tqdm.write(f"Rate limit. Waiting {wait:.1f}s...")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = 2 ** attempt + random.uniform(0, 1)
            tqdm.write(f"API error: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)

    return "ERROR: Max retries exceeded", {'time_ms': 0, 'tokens_in': 0, 'tokens_out': 0}


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    # Lowercase
    text = text.lower()
    # Remove punctuation except apostrophes
    text = ''.join(c for c in text if c.isalnum() or c.isspace() or c == "'")
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def calculate_wer(predictions: list[str], references: list[str]) -> dict:
    """Calculate WER and related metrics."""
    if not JIWER_AVAILABLE:
        # Simple WER calculation
        total_errors = 0
        total_words = 0
        for pred, ref in zip(predictions, references):
            pred_words = pred.split()
            ref_words = ref.split()
            total_words += len(ref_words)
            # Simple word-level comparison (not true edit distance)
            total_errors += abs(len(pred_words) - len(ref_words))
        return {'wer': total_errors / total_words if total_words > 0 else 0}

    # Use jiwer for proper WER
    measures = jiwer.compute_measures(references, predictions)
    return {
        'wer': measures['wer'],
        'mer': measures['mer'],
        'wil': measures['wil'],
        'substitutions': measures['substitutions'],
        'insertions': measures['insertions'],
        'deletions': measures['deletions'],
        'hits': measures['hits'],
    }


def run_inference(provider: str, model: str, system_prompt: str, samples: list[dict]) -> list[dict]:
    """Run inference on all samples."""
    results = []

    call_fn = call_openai if provider == 'openai' else call_anthropic

    for sample in tqdm(samples, desc=f"Running inference ({provider})"):
        prediction, stats = call_fn(model, system_prompt, sample['phonemes'])

        results.append({
            'trial_id': sample['trial_id'],
            'phonemes': sample['phonemes'],
            'predicted_text': prediction,
            'ground_truth_text': sample['ground_truth_text'],
            'ground_truth_phonemes': sample['ground_truth_phonemes'],
            'per': sample['per'],
            'confidence': sample['confidence'],
            'is_silent': sample['is_silent'],
            **stats,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Convert phonemes to text and calculate WER')
    parser.add_argument('--input', type=str, default='test_results.csv',
                        help='Input CSV file from export_results.py')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'anthropic'],
                        help='API provider')
    parser.add_argument('--model', type=str, required=True,
                        help='Model ID (e.g., gpt-4o, claude-sonnet-4-20250514)')
    parser.add_argument('--mode', type=str, default='zero-shot', choices=['zero-shot', 'few-shot'],
                        help='Inference mode')
    parser.add_argument('--n-shots', type=int, default=5,
                        help='Number of few-shot examples')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Auto-generate output filename
    if args.output is None:
        model_short = args.model.split('/')[-1].replace(':', '-')[:20]
        args.output = f"wer_results_{args.provider}_{model_short}_{args.mode}.csv"

    print(f"{'='*60}")
    print("PHONEME TO TEXT CONVERSION")
    print(f"{'='*60}")
    print(f"Provider: {args.provider}")
    print(f"Model:    {args.model}")
    print(f"Mode:     {args.mode}")
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")

    # Load data
    print(f"\nLoading data...")
    all_samples = load_test_results(args.input)
    print(f"Loaded {len(all_samples):,} samples")

    # Handle few-shot examples
    if args.mode == 'few-shot':
        few_shot_examples, samples = select_few_shot_examples(all_samples, args.n_shots, args.seed)
        print(f"Selected {len(few_shot_examples)} few-shot examples (lowest PER)")
        print(f"Remaining samples: {len(samples)}")

        examples_text = format_few_shot_examples(few_shot_examples)
        system_prompt = SYSTEM_PROMPT_FEW_SHOT.format(examples=examples_text)
    else:
        samples = all_samples
        system_prompt = SYSTEM_PROMPT_ZERO_SHOT

    # Apply limit
    if args.limit:
        samples = samples[:args.limit]
        print(f"Limited to {len(samples)} samples")

    if len(samples) == 0:
        print("ERROR: No samples to process!")
        return 1

    # Run inference
    print(f"\nRunning inference on {len(samples):,} samples...")
    start_time = time.time()
    results = run_inference(args.provider, args.model, system_prompt, samples)
    total_time = time.time() - start_time

    # Calculate WER
    print(f"\nCalculating WER...")
    predictions = [normalize_text(r['predicted_text']) for r in results]
    references = [normalize_text(r['ground_truth_text']) for r in results]

    # Add normalized text to results
    for r, pred_norm, ref_norm in zip(results, predictions, references):
        r['predicted_text_normalized'] = pred_norm
        r['ground_truth_text_normalized'] = ref_norm
        # Per-sample WER
        if JIWER_AVAILABLE:
            try:
                r['sample_wer'] = jiwer.wer(ref_norm, pred_norm)
            except:
                r['sample_wer'] = 1.0
        else:
            r['sample_wer'] = 0.0

    wer_metrics = calculate_wer(predictions, references)

    # Save results
    print(f"\nSaving results to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'trial_id', 'phonemes', 'predicted_text', 'ground_truth_text',
        'predicted_text_normalized', 'ground_truth_text_normalized',
        'sample_wer', 'per', 'confidence', 'is_silent',
        'time_ms', 'tokens_in', 'tokens_out'
    ]

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    # Summary stats
    total_tokens_in = sum(r['tokens_in'] for r in results)
    total_tokens_out = sum(r['tokens_out'] for r in results)
    avg_time_ms = sum(r['time_ms'] for r in results) / len(results) if results else 0
    errors = sum(1 for r in results if r['predicted_text'].startswith('ERROR'))
    avg_per = sum(r['per'] for r in results) / len(results) if results else 0

    # Cost estimation
    if args.provider == 'openai':
        if 'gpt-4o' in args.model:
            cost_in = total_tokens_in / 1_000_000 * 2.50
            cost_out = total_tokens_out / 1_000_000 * 10.00
        else:
            cost_in = total_tokens_in / 1_000_000 * 5.00
            cost_out = total_tokens_out / 1_000_000 * 15.00
    else:  # anthropic
        if 'sonnet' in args.model:
            cost_in = total_tokens_in / 1_000_000 * 3.00
            cost_out = total_tokens_out / 1_000_000 * 15.00
        elif 'haiku' in args.model:
            cost_in = total_tokens_in / 1_000_000 * 0.25
            cost_out = total_tokens_out / 1_000_000 * 1.25
        else:  # opus
            cost_in = total_tokens_in / 1_000_000 * 15.00
            cost_out = total_tokens_out / 1_000_000 * 75.00

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Samples processed: {len(results):,}")
    print(f"API errors:        {errors}")
    print(f"Total time:        {total_time:.1f}s ({avg_time_ms:.0f}ms/sample)")
    print(f"Total tokens:      {total_tokens_in:,} in, {total_tokens_out:,} out")
    print(f"Est. cost:         ${cost_in + cost_out:.2f}")
    print()
    print(f"Avg PER (phoneme): {avg_per*100:.2f}%")
    print(f"WER (word):        {wer_metrics['wer']*100:.2f}%")
    if 'substitutions' in wer_metrics:
        print(f"  - Substitutions: {wer_metrics['substitutions']}")
        print(f"  - Insertions:    {wer_metrics['insertions']}")
        print(f"  - Deletions:     {wer_metrics['deletions']}")
        print(f"  - Hits:          {wer_metrics['hits']}")
    print(f"{'='*60}")

    # Sample results
    print("\nSample results:")
    for i, r in enumerate(results[:5]):
        print(f"\n[{r['trial_id']}] PER: {r['per']*100:.1f}% | WER: {r['sample_wer']*100:.1f}%")
        print(f"  Phonemes:    {r['phonemes'][:50]}...")
        print(f"  Predicted:   {r['predicted_text'][:60]}")
        print(f"  Ground truth: {r['ground_truth_text'][:60]}")

    # Save summary
    summary_file = args.output.replace('.csv', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'provider': args.provider,
            'model': args.model,
            'mode': args.mode,
            'n_samples': len(results),
            'avg_per': avg_per,
            'wer': wer_metrics['wer'],
            'total_time_s': total_time,
            'total_tokens_in': total_tokens_in,
            'total_tokens_out': total_tokens_out,
            'est_cost': cost_in + cost_out,
        }, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
