"""
AI-to-Human Communication Experiment

This script tests different communication strategies for conveying information from AI to humans.
Uses real LLM APIs (OpenAI GPT-4, OpenRouter) to generate and evaluate summaries.

Experiments:
1. Format Comparison: Dense prose vs bullet points vs hierarchical vs progressive
2. Length Tradeoff: Finding optimal summary length
3. Progressive Disclosure: Multi-level summarization effectiveness
"""

import os
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk
from tqdm import tqdm
import openai
import httpx

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configuration
BASE_DIR = Path("/data/hypogenicai/workspaces/ai-human-communication-claude")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Use OpenRouter as primary (more model availability)
openrouter_client = openai.OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Fallback to OpenAI if available
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


@dataclass
class EvaluationResult:
    """Stores evaluation scores for a summary"""
    faithfulness: float
    completeness: float
    conciseness: float
    readability: float
    raw_response: str = ""


@dataclass
class SummaryResult:
    """Stores a generated summary with metadata"""
    doc_id: str
    source_domain: str
    document: str
    format_type: str
    summary: str
    word_count: int
    compression_ratio: float
    evaluation: Optional[EvaluationResult] = None
    generation_model: str = ""
    evaluation_model: str = ""


def call_llm(prompt: str, model: str = "openai/gpt-4o-mini", max_tokens: int = 1000,
             temperature: float = 0.0, use_openrouter: bool = True) -> str:
    """
    Call LLM API with retry logic.

    Args:
        prompt: The prompt to send
        model: Model name (openai/gpt-4o-mini, openai/gpt-4.1, etc.)
        max_tokens: Maximum response tokens
        temperature: Sampling temperature (0 for reproducibility)
        use_openrouter: Whether to use OpenRouter instead of OpenAI

    Returns:
        Model response text
    """
    client = openrouter_client if use_openrouter else openai_client
    if client is None:
        client = openrouter_client

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < 2:
                print(f"API error (attempt {attempt+1}): {e}")
                time.sleep(2 ** attempt)
            else:
                raise


def generate_summary_format(document: str, format_type: str,
                           target_length: Optional[int] = None) -> str:
    """
    Generate a summary in a specific format.

    Args:
        document: Source document text
        format_type: One of 'dense', 'bullets', 'hierarchical', 'progressive'
        target_length: Optional target word count

    Returns:
        Generated summary
    """
    length_instruction = f"Target approximately {target_length} words." if target_length else ""

    prompts = {
        'dense': f"""Summarize the following document in a single cohesive paragraph.
Write in flowing prose without bullet points or headers.
{length_instruction}

Document:
{document}

Summary:""",

        'bullets': f"""Summarize the following document using bullet points.
- Use clear, concise bullet points
- Each point should capture one key idea
- Start each bullet with a dash (-)
{length_instruction}

Document:
{document}

Summary:""",

        'hierarchical': f"""Summarize the following document with a hierarchical structure.
Use this format:
## Main Topic
- Key point 1
- Key point 2

## Another Topic (if applicable)
- Key point 3

{length_instruction}

Document:
{document}

Summary:""",

        'progressive': f"""Create a progressive summary of the following document with two levels:

LEVEL 1 (One sentence): A single sentence capturing the core message.
LEVEL 2 (Details): 3-5 bullet points expanding on the key details.

Format exactly as:
ONE-LINE: [single sentence summary]

DETAILS:
- [detail 1]
- [detail 2]
- [detail 3]

Document:
{document}

Summary:"""
    }

    if format_type not in prompts:
        raise ValueError(f"Unknown format type: {format_type}")

    return call_llm(prompts[format_type], model="openai/gpt-4o-mini", max_tokens=500)


def generate_length_controlled_summary(document: str, target_words: int) -> str:
    """Generate a summary targeting a specific word count."""
    prompt = f"""Summarize the following document in approximately {target_words} words.
Be concise but capture the essential information.

Document:
{document}

Summary (about {target_words} words):"""

    return call_llm(prompt, model="openai/gpt-4o-mini", max_tokens=target_words * 2)


def evaluate_summary(document: str, summary: str, format_type: str = "") -> EvaluationResult:
    """
    Evaluate a summary using LLM-as-judge on multiple dimensions.

    Uses GPT-4o-mini for cost-effective evaluation while maintaining quality.
    """
    eval_prompt = f"""You are an expert evaluator assessing the quality of a summary.

SOURCE DOCUMENT:
{document[:3000]}  # Truncate for context length

SUMMARY TO EVALUATE:
{summary}

Rate the summary on these four dimensions from 0.0 to 1.0:

1. FAITHFULNESS (0-1): Does the summary accurately represent the source without adding false information or contradicting the original?
   - 1.0 = Perfectly faithful, no errors
   - 0.5 = Some minor inaccuracies
   - 0.0 = Major hallucinations or contradictions

2. COMPLETENESS (0-1): Does the summary capture all the essential information from the source?
   - 1.0 = All key points included
   - 0.5 = Some important points missing
   - 0.0 = Critical information missing

3. CONCISENESS (0-1): Is the summary appropriately brief without unnecessary repetition or verbosity?
   - 1.0 = Perfectly concise, no wasted words
   - 0.5 = Some redundancy
   - 0.0 = Very verbose or repetitive

4. READABILITY (0-1): Is the summary easy to read and well-structured for human comprehension?
   - 1.0 = Excellent structure, very clear
   - 0.5 = Readable but could be better organized
   - 0.0 = Confusing or poorly organized

Respond in exactly this JSON format:
{{"faithfulness": 0.X, "completeness": 0.X, "conciseness": 0.X, "readability": 0.X}}"""

    for attempt in range(3):
        try:
            response = call_llm(eval_prompt, model="openai/gpt-4o-mini", max_tokens=100, temperature=0.0)

            # Parse JSON response
            # Handle markdown code blocks if present
            if "```" in response:
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
                response = response.strip()

            scores = json.loads(response)
            return EvaluationResult(
                faithfulness=float(scores.get('faithfulness', 0.5)),
                completeness=float(scores.get('completeness', 0.5)),
                conciseness=float(scores.get('conciseness', 0.5)),
                readability=float(scores.get('readability', 0.5)),
                raw_response=response
            )
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < 2:
                print(f"Evaluation parse error: {e}, retrying...")
                continue
            # Return neutral scores on failure
            return EvaluationResult(
                faithfulness=0.5, completeness=0.5,
                conciseness=0.5, readability=0.5,
                raw_response=f"Parse error: {response}"
            )


def load_test_data(n_samples: int = 100) -> pd.DataFrame:
    """Load and sample diverse documents from FeedSum test set."""
    feedsum = load_from_disk(str(BASE_DIR / "datasets/feedsum/data"))
    test_df = feedsum['test'].to_pandas()

    # Get unique documents (each doc appears multiple times with different summarizers)
    unique_docs = test_df.drop_duplicates(subset=['doc_id'])[
        ['doc_id', 'source', 'document', 'human_reference', 'extracted_keyfacts']
    ]

    # Sample ensuring domain diversity
    domains = unique_docs['source'].unique()
    samples_per_domain = max(1, n_samples // len(domains))

    sampled = []
    for domain in domains:
        domain_docs = unique_docs[unique_docs['source'] == domain]
        n_take = min(samples_per_domain, len(domain_docs))
        sampled.append(domain_docs.sample(n=n_take, random_state=SEED))

    result = pd.concat(sampled).head(n_samples)
    print(f"Loaded {len(result)} documents from {len(domains)} domains")
    print(f"Domain distribution: {result['source'].value_counts().to_dict()}")

    return result


def run_experiment_1_format_comparison(test_data: pd.DataFrame, n_docs: int = 50) -> List[SummaryResult]:
    """
    Experiment 1: Compare different summary formats.

    Tests: dense prose, bullet points, hierarchical, progressive
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Format Comparison")
    print("="*60)

    formats = ['dense', 'bullets', 'hierarchical', 'progressive']
    results = []

    sample = test_data.head(n_docs)

    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="Generating summaries"):
        document = row['document']
        doc_len = len(document.split())

        for fmt in formats:
            try:
                summary = generate_summary_format(document, fmt)
                summary_len = len(summary.split())

                # Evaluate
                evaluation = evaluate_summary(document, summary, fmt)

                result = SummaryResult(
                    doc_id=row['doc_id'],
                    source_domain=row['source'],
                    document=document,
                    format_type=fmt,
                    summary=summary,
                    word_count=summary_len,
                    compression_ratio=summary_len / doc_len if doc_len > 0 else 0,
                    evaluation=evaluation,
                    generation_model="openai/gpt-4o-mini",
                    evaluation_model="openai/gpt-4o-mini"
                )
                results.append(result)

            except Exception as e:
                print(f"Error processing {row['doc_id']} with format {fmt}: {e}")
                continue

        # Rate limiting
        time.sleep(0.5)

    return results


def run_experiment_2_length_tradeoff(test_data: pd.DataFrame, n_docs: int = 50) -> List[SummaryResult]:
    """
    Experiment 2: Find optimal summary length.

    Tests: 25, 50, 100, 200, 400 word targets
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Length vs Quality Tradeoff")
    print("="*60)

    target_lengths = [25, 50, 100, 200, 400]
    results = []

    sample = test_data.head(n_docs)

    for idx, row in tqdm(sample.iterrows(), total=len(sample), desc="Length experiments"):
        document = row['document']
        doc_len = len(document.split())

        for target in target_lengths:
            # Skip if target is longer than document
            if target > doc_len:
                continue

            try:
                summary = generate_length_controlled_summary(document, target)
                summary_len = len(summary.split())

                evaluation = evaluate_summary(document, summary)

                result = SummaryResult(
                    doc_id=row['doc_id'],
                    source_domain=row['source'],
                    document=document,
                    format_type=f"length_{target}",
                    summary=summary,
                    word_count=summary_len,
                    compression_ratio=summary_len / doc_len if doc_len > 0 else 0,
                    evaluation=evaluation,
                    generation_model="openai/gpt-4o-mini",
                    evaluation_model="openai/gpt-4o-mini"
                )
                results.append(result)

            except Exception as e:
                print(f"Error processing {row['doc_id']} with length {target}: {e}")
                continue

        time.sleep(0.5)

    return results


def run_experiment_3_progressive_disclosure(test_data: pd.DataFrame, n_docs: int = 30) -> List[Dict]:
    """
    Experiment 3: Test multi-level progressive disclosure.

    Creates 3-level hierarchy: one-line → paragraph → detailed
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Progressive Disclosure")
    print("="*60)

    results = []

    # Filter for longer documents (>200 words)
    long_docs = test_data[test_data['document'].str.split().str.len() > 200].head(n_docs)

    for idx, row in tqdm(long_docs.iterrows(), total=len(long_docs), desc="Progressive disclosure"):
        document = row['document']
        doc_len = len(document.split())

        try:
            # Level 1: One-line summary (~15-20 words)
            level1_prompt = f"""Write a single sentence (15-20 words) capturing the core message of this document:

{document}

One sentence summary:"""
            level1 = call_llm(level1_prompt, model="openai/gpt-4o-mini", max_tokens=50)

            # Level 2: Brief summary (~50-75 words)
            level2_prompt = f"""Write a brief 2-3 sentence summary (50-75 words) of this document:

{document}

Brief summary:"""
            level2 = call_llm(level2_prompt, model="openai/gpt-4o-mini", max_tokens=150)

            # Level 3: Detailed summary (~150-200 words)
            level3_prompt = f"""Write a detailed summary (150-200 words) with key points:

{document}

Detailed summary:"""
            level3 = call_llm(level3_prompt, model="openai/gpt-4o-mini", max_tokens=400)

            # Evaluate each level
            eval1 = evaluate_summary(document, level1)
            eval2 = evaluate_summary(document, level2)
            eval3 = evaluate_summary(document, level3)

            result = {
                'doc_id': row['doc_id'],
                'source_domain': row['source'],
                'document_length': doc_len,
                'level1_summary': level1,
                'level1_words': len(level1.split()),
                'level1_eval': asdict(eval1),
                'level2_summary': level2,
                'level2_words': len(level2.split()),
                'level2_eval': asdict(eval2),
                'level3_summary': level3,
                'level3_words': len(level3.split()),
                'level3_eval': asdict(eval3),
            }
            results.append(result)

        except Exception as e:
            print(f"Error processing {row['doc_id']}: {e}")
            continue

        time.sleep(0.5)

    return results


def compare_with_feedsum_scores(test_data: pd.DataFrame, n_docs: int = 50) -> pd.DataFrame:
    """
    Compare LLM-as-judge scores with FeedSum's existing feedback scores.

    This validates whether our evaluation aligns with established benchmarks.
    """
    print("\n" + "="*60)
    print("VALIDATION: LLM-as-Judge vs FeedSum Scores")
    print("="*60)

    feedsum = load_from_disk(str(BASE_DIR / "datasets/feedsum/data"))
    test_df = feedsum['test'].to_pandas()

    # Sample from summaries that have feedback-c4 scores
    valid_samples = test_df[test_df['feedback-c4'].notna()].head(n_docs * 3)

    results = []

    for idx, row in tqdm(valid_samples.iterrows(), total=len(valid_samples), desc="Validation"):
        try:
            # Get existing FeedSum scores
            feedsum_scores = row['feedback-c4']
            if feedsum_scores is None:
                continue

            # Evaluate with our LLM-as-judge
            our_eval = evaluate_summary(row['document'], row['summary'])

            results.append({
                'doc_id': row['doc_id'],
                'summarizer': row['summarizer'],
                # FeedSum scores
                'feedsum_faithfulness': feedsum_scores.get('faithfulness_score', None),
                'feedsum_completeness': feedsum_scores.get('completeness_score', None),
                'feedsum_conciseness': feedsum_scores.get('conciseness_score', None),
                # Our scores
                'llm_faithfulness': our_eval.faithfulness,
                'llm_completeness': our_eval.completeness,
                'llm_conciseness': our_eval.conciseness,
                'llm_readability': our_eval.readability,
            })

        except Exception as e:
            print(f"Error: {e}")
            continue

        time.sleep(0.3)

    return pd.DataFrame(results)


def analyze_results(exp1_results: List[SummaryResult],
                    exp2_results: List[SummaryResult],
                    exp3_results: List[Dict]) -> Dict:
    """Compute summary statistics and perform analysis."""

    analysis = {}

    # Experiment 1: Format comparison
    if exp1_results:
        exp1_df = pd.DataFrame([
            {
                'format': r.format_type,
                'word_count': r.word_count,
                'compression_ratio': r.compression_ratio,
                'faithfulness': r.evaluation.faithfulness if r.evaluation else None,
                'completeness': r.evaluation.completeness if r.evaluation else None,
                'conciseness': r.evaluation.conciseness if r.evaluation else None,
                'readability': r.evaluation.readability if r.evaluation else None,
            }
            for r in exp1_results
        ])

        format_stats = exp1_df.groupby('format').agg({
            'faithfulness': ['mean', 'std'],
            'completeness': ['mean', 'std'],
            'conciseness': ['mean', 'std'],
            'readability': ['mean', 'std'],
            'word_count': ['mean', 'std'],
            'compression_ratio': ['mean', 'std']
        }).round(3)

        analysis['experiment1_format_stats'] = format_stats.to_dict()

    # Experiment 2: Length tradeoff
    if exp2_results:
        exp2_df = pd.DataFrame([
            {
                'target_length': int(r.format_type.split('_')[1]),
                'actual_length': r.word_count,
                'faithfulness': r.evaluation.faithfulness if r.evaluation else None,
                'completeness': r.evaluation.completeness if r.evaluation else None,
                'conciseness': r.evaluation.conciseness if r.evaluation else None,
            }
            for r in exp2_results
        ])

        length_stats = exp2_df.groupby('target_length').agg({
            'actual_length': ['mean', 'std'],
            'faithfulness': ['mean', 'std'],
            'completeness': ['mean', 'std'],
            'conciseness': ['mean', 'std'],
        }).round(3)

        analysis['experiment2_length_stats'] = length_stats.to_dict()

    # Experiment 3: Progressive disclosure
    if exp3_results:
        exp3_df = pd.DataFrame(exp3_results)

        level_stats = {
            'level1': {
                'words': exp3_df['level1_words'].mean(),
                'faithfulness': exp3_df['level1_eval'].apply(lambda x: x['faithfulness']).mean(),
                'completeness': exp3_df['level1_eval'].apply(lambda x: x['completeness']).mean(),
                'conciseness': exp3_df['level1_eval'].apply(lambda x: x['conciseness']).mean(),
            },
            'level2': {
                'words': exp3_df['level2_words'].mean(),
                'faithfulness': exp3_df['level2_eval'].apply(lambda x: x['faithfulness']).mean(),
                'completeness': exp3_df['level2_eval'].apply(lambda x: x['completeness']).mean(),
                'conciseness': exp3_df['level2_eval'].apply(lambda x: x['conciseness']).mean(),
            },
            'level3': {
                'words': exp3_df['level3_words'].mean(),
                'faithfulness': exp3_df['level3_eval'].apply(lambda x: x['faithfulness']).mean(),
                'completeness': exp3_df['level3_eval'].apply(lambda x: x['completeness']).mean(),
                'conciseness': exp3_df['level3_eval'].apply(lambda x: x['conciseness']).mean(),
            }
        }

        analysis['experiment3_progressive_stats'] = level_stats

    return analysis


def save_results(exp1_results: List[SummaryResult],
                 exp2_results: List[SummaryResult],
                 exp3_results: List[Dict],
                 validation_results: Optional[pd.DataFrame] = None):
    """Save all results to JSON files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert dataclasses to dicts
    def to_dict(r: SummaryResult) -> Dict:
        d = {
            'doc_id': r.doc_id,
            'source_domain': r.source_domain,
            'format_type': r.format_type,
            'summary': r.summary,
            'word_count': r.word_count,
            'compression_ratio': r.compression_ratio,
            'generation_model': r.generation_model,
            'evaluation_model': r.evaluation_model,
        }
        if r.evaluation:
            d['evaluation'] = asdict(r.evaluation)
        return d

    if exp1_results:
        with open(RESULTS_DIR / f"experiment1_format_{timestamp}.json", 'w') as f:
            json.dump([to_dict(r) for r in exp1_results], f, indent=2)

    if exp2_results:
        with open(RESULTS_DIR / f"experiment2_length_{timestamp}.json", 'w') as f:
            json.dump([to_dict(r) for r in exp2_results], f, indent=2)

    if exp3_results:
        with open(RESULTS_DIR / f"experiment3_progressive_{timestamp}.json", 'w') as f:
            json.dump(exp3_results, f, indent=2)

    if validation_results is not None:
        validation_results.to_csv(RESULTS_DIR / f"validation_{timestamp}.csv", index=False)

    print(f"\nResults saved to {RESULTS_DIR}")


def main():
    """Run all experiments."""
    print("="*60)
    print("AI-to-Human Communication Research Experiments")
    print(f"Start time: {datetime.now()}")
    print("="*60)

    # Load test data
    print("\nLoading test data...")
    test_data = load_test_data(n_samples=100)

    # Run experiments (reduced sample sizes for efficiency while maintaining statistical validity)
    exp1_results = run_experiment_1_format_comparison(test_data, n_docs=40)
    exp2_results = run_experiment_2_length_tradeoff(test_data, n_docs=30)
    exp3_results = run_experiment_3_progressive_disclosure(test_data, n_docs=25)

    # Validation: compare with FeedSum scores
    validation_results = compare_with_feedsum_scores(test_data, n_docs=30)

    # Analyze
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    analysis = analyze_results(exp1_results, exp2_results, exp3_results)
    print(json.dumps(analysis, indent=2, default=str))

    # Save results
    save_results(exp1_results, exp2_results, exp3_results, validation_results)

    # Save analysis summary
    with open(RESULTS_DIR / "analysis_summary.json", 'w') as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\nCompleted at: {datetime.now()}")

    return exp1_results, exp2_results, exp3_results, validation_results, analysis


if __name__ == "__main__":
    results = main()
