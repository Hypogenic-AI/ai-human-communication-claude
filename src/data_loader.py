"""
Data loader for AI-to-Human Communication experiments.
Loads and samples from FeedSum dataset.
"""

import json
import random
from pathlib import Path
from datasets import load_from_disk
import numpy as np

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_feedsum(data_dir: str = "datasets/feedsum/data"):
    """Load the FeedSum dataset from disk."""
    dataset = load_from_disk(data_dir)
    return dataset


def sample_diverse_documents(dataset, n_samples: int = 100, min_doc_length: int = 100):
    """
    Sample diverse documents from FeedSum test set.
    Ensures variety in:
    - Document source/domain
    - Document length

    Args:
        dataset: HuggingFace dataset (test split)
        n_samples: Number of samples to select
        min_doc_length: Minimum document length in characters

    Returns:
        List of sample dictionaries
    """
    # Get test split if available, otherwise use train
    if 'test' in dataset:
        data = dataset['test']
    else:
        data = dataset['train']

    # Filter by minimum length
    valid_indices = []
    for i in range(len(data)):
        doc = data[i]
        if len(doc.get('document', '')) >= min_doc_length:
            valid_indices.append(i)

    # Group by source if available
    source_groups = {}
    for idx in valid_indices:
        source = data[idx].get('source', 'unknown')
        if source not in source_groups:
            source_groups[source] = []
        source_groups[source].append(idx)

    print(f"Found {len(valid_indices)} valid documents across {len(source_groups)} sources")
    print(f"Sources: {list(source_groups.keys())}")

    # Sample proportionally from each source
    samples = []
    samples_per_source = max(1, n_samples // len(source_groups))

    for source, indices in source_groups.items():
        n_from_source = min(len(indices), samples_per_source)
        sampled_indices = random.sample(indices, n_from_source)
        for idx in sampled_indices:
            samples.append(data[idx])

        if len(samples) >= n_samples:
            break

    # If we need more samples, take randomly from remaining
    if len(samples) < n_samples:
        all_remaining = [idx for idx in valid_indices if data[idx] not in samples]
        additional = random.sample(all_remaining, min(n_samples - len(samples), len(all_remaining)))
        for idx in additional:
            samples.append(data[idx])

    # Shuffle final sample
    random.shuffle(samples)

    return samples[:n_samples]


def get_document_stats(samples):
    """Compute statistics about sampled documents."""
    doc_lengths = [len(s.get('document', '') or '') for s in samples]
    summary_lengths = [len(s.get('summary', '') or '') for s in samples]
    ref_lengths = [len(s.get('human_reference', '') or '') for s in samples if s.get('human_reference')]

    # Source distribution
    sources = {}
    for s in samples:
        src = s.get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1

    # Feedback score distribution (if available)
    faithfulness_scores = []
    completeness_scores = []
    conciseness_scores = []

    for s in samples:
        feedback = s.get('feedback-c4', {})
        if feedback:
            faithfulness_scores.append(feedback.get('faithfulness_score', 0))
            completeness_scores.append(feedback.get('completeness_score', 0))
            conciseness_scores.append(feedback.get('conciseness_score', 0))

    stats = {
        'n_samples': len(samples),
        'doc_length': {
            'mean': np.mean(doc_lengths),
            'std': np.std(doc_lengths),
            'min': min(doc_lengths),
            'max': max(doc_lengths)
        },
        'summary_length': {
            'mean': np.mean(summary_lengths),
            'std': np.std(summary_lengths),
            'min': min(summary_lengths),
            'max': max(summary_lengths)
        },
        'reference_length': {
            'mean': np.mean(ref_lengths) if ref_lengths else 0,
            'std': np.std(ref_lengths) if ref_lengths else 0,
            'count': len(ref_lengths)
        },
        'sources': sources,
        'feedback_scores': {
            'faithfulness': {
                'mean': np.mean(faithfulness_scores) if faithfulness_scores else None,
                'std': np.std(faithfulness_scores) if faithfulness_scores else None
            },
            'completeness': {
                'mean': np.mean(completeness_scores) if completeness_scores else None,
                'std': np.std(completeness_scores) if completeness_scores else None
            },
            'conciseness': {
                'mean': np.mean(conciseness_scores) if conciseness_scores else None,
                'std': np.std(conciseness_scores) if conciseness_scores else None
            }
        }
    }

    return stats


def save_samples(samples, output_path: str = "results/sampled_documents.json"):
    """Save sampled documents to JSON."""
    # Convert to serializable format
    serializable = []
    for s in samples:
        item = {}
        for k, v in s.items():
            if isinstance(v, (dict, list, str, int, float, bool, type(None))):
                item[k] = v
            else:
                item[k] = str(v)
        serializable.append(item)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")


def load_samples(input_path: str = "results/sampled_documents.json"):
    """Load sampled documents from JSON."""
    with open(input_path, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test data loading
    print("Loading FeedSum dataset...")
    dataset = load_feedsum()
    print(f"Dataset: {dataset}")

    print("\nSampling diverse documents...")
    samples = sample_diverse_documents(dataset, n_samples=100)

    print("\nComputing statistics...")
    stats = get_document_stats(samples)
    print(f"\nDataset Statistics:")
    print(f"  Number of samples: {stats['n_samples']}")
    print(f"  Document length: {stats['doc_length']['mean']:.0f} +/- {stats['doc_length']['std']:.0f} chars")
    print(f"  Summary length: {stats['summary_length']['mean']:.0f} +/- {stats['summary_length']['std']:.0f} chars")
    print(f"  Sources: {stats['sources']}")

    if stats['feedback_scores']['faithfulness']['mean'] is not None:
        print(f"\nFeedback scores (existing summaries):")
        print(f"  Faithfulness: {stats['feedback_scores']['faithfulness']['mean']:.3f}")
        print(f"  Completeness: {stats['feedback_scores']['completeness']['mean']:.3f}")
        print(f"  Conciseness: {stats['feedback_scores']['conciseness']['mean']:.3f}")

    # Save samples
    save_samples(samples)

    # Print example
    print("\n" + "="*80)
    print("EXAMPLE DOCUMENT:")
    print("="*80)
    example = samples[0]
    print(f"Source: {example.get('source', 'unknown')}")
    print(f"Document (first 500 chars):\n{example.get('document', '')[:500]}...")
    print(f"\nHuman Reference:\n{example.get('human_reference', 'N/A')[:300]}...")
