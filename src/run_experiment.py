"""
Main experiment runner for AI-to-Human Communication research.
Orchestrates data loading, summary generation, evaluation, and analysis.
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_feedsum, sample_diverse_documents, get_document_stats, save_samples, load_samples
from generate_summaries import generate_all_summaries, load_summaries
from evaluate import evaluate_all_summaries, run_pairwise_comparisons
from analyze import load_results, save_analysis_results


def run_full_experiment(n_documents: int = 100, skip_generation: bool = False,
                        skip_evaluation: bool = False, skip_analysis: bool = False):
    """
    Run the full experimental pipeline.

    Args:
        n_documents: Number of documents to sample
        skip_generation: If True, use existing summaries
        skip_evaluation: If True, use existing evaluations
        skip_analysis: If True, skip analysis phase
    """
    start_time = time.time()
    print("=" * 60)
    print("AI-TO-HUMAN COMMUNICATION EXPERIMENT")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Documents: {n_documents}")
    print(f"  Skip generation: {skip_generation}")
    print(f"  Skip evaluation: {skip_evaluation}")
    print()

    # Phase 1: Data Preparation
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PREPARATION")
    print("=" * 60)

    samples_path = "results/sampled_documents.json"

    if os.path.exists(samples_path):
        print(f"Loading existing samples from {samples_path}")
        samples = load_samples(samples_path)
        print(f"Loaded {len(samples)} samples")
    else:
        print("Loading FeedSum dataset...")
        dataset = load_feedsum()
        print("Sampling diverse documents...")
        samples = sample_diverse_documents(dataset, n_samples=n_documents)

        print("\nDataset statistics:")
        stats = get_document_stats(samples)
        print(f"  Documents: {stats['n_samples']}")
        print(f"  Avg document length: {stats['doc_length']['mean']:.0f} chars")
        print(f"  Sources: {list(stats['sources'].keys())}")

        save_samples(samples, samples_path)

    # Phase 2: Summary Generation
    print("\n" + "=" * 60)
    print("PHASE 2: SUMMARY GENERATION")
    print("=" * 60)

    summaries_path = "results/summaries/all_summaries.json"

    if skip_generation and os.path.exists(summaries_path):
        print(f"Loading existing summaries from {summaries_path}")
        summaries = load_summaries(summaries_path)
    else:
        print(f"Generating summaries for {len(samples)} documents...")
        summaries = generate_all_summaries(samples)

    print(f"Generated summaries for {len(summaries['summaries'])} documents")

    # Phase 3: Evaluation
    print("\n" + "=" * 60)
    print("PHASE 3: EVALUATION")
    print("=" * 60)

    evaluations_path = "results/evaluations/all_evaluations.json"

    if skip_evaluation and os.path.exists(evaluations_path):
        print(f"Loading existing evaluations from {evaluations_path}")
        with open(evaluations_path, "r") as f:
            evaluations = json.load(f)
    else:
        print("Evaluating summaries with LLM-as-judge...")
        evaluations = evaluate_all_summaries(summaries)

        # Run pairwise comparisons
        print("\nRunning pairwise comparisons...")
        comparison_pairs = [
            ("dense_prose", "bullet_points"),
            ("dense_prose", "progressive_hierarchy"),
            ("formal_technical", "conversational")
        ]
        pairwise = run_pairwise_comparisons(summaries, comparison_pairs,
                                            n_samples=min(50, len(samples)))

    # Phase 4: Analysis
    if not skip_analysis:
        print("\n" + "=" * 60)
        print("PHASE 4: ANALYSIS")
        print("=" * 60)

        print("Running statistical analysis...")
        results = load_results()
        analysis = save_analysis_results(results)

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"\nOutput files:")
    print(f"  - results/sampled_documents.json")
    print(f"  - results/summaries/all_summaries.json")
    print(f"  - results/evaluations/all_evaluations.json")
    print(f"  - results/evaluations/pairwise_comparisons.json")
    print(f"  - results/evaluation_scores.csv")
    print(f"  - results/strategy_stats.csv")
    print(f"  - results/figures/*.png")
    print(f"  - results/analysis_summary.txt")


def run_quick_test(n_documents: int = 5):
    """Run a quick test with fewer documents."""
    print("RUNNING QUICK TEST MODE")
    print(f"Testing with {n_documents} documents")

    # Sample fresh documents for test
    print("\nLoading dataset...")
    dataset = load_feedsum()
    samples = sample_diverse_documents(dataset, n_samples=n_documents)
    save_samples(samples, "results/sampled_documents.json")

    # Run experiment
    run_full_experiment(n_documents=n_documents)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI-to-Human Communication experiment")
    parser.add_argument("--n-documents", type=int, default=100, help="Number of documents to process")
    parser.add_argument("--test", action="store_true", help="Run quick test with 5 documents")
    parser.add_argument("--skip-generation", action="store_true", help="Skip summary generation")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis")

    args = parser.parse_args()

    if args.test:
        run_quick_test(n_documents=5)
    else:
        run_full_experiment(
            n_documents=args.n_documents,
            skip_generation=args.skip_generation,
            skip_evaluation=args.skip_evaluation,
            skip_analysis=args.skip_analysis
        )
