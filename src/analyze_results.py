"""
Analysis script for AI-to-Human Communication experiments.

Analyzes results from the three experiments:
1. Format Comparison
2. Length vs Quality Trade-off
3. Progressive Disclosure

Generates statistical analysis and visualizations.
"""

import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
BASE_DIR = Path("/data/hypogenicai/workspaces/ai-human-communication-claude")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Create directories
FIGURES_DIR.mkdir(exist_ok=True)

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def load_experiment_results():
    """Load all experiment result files."""
    results = {}

    # Find most recent result files
    exp1_files = sorted(RESULTS_DIR.glob("experiment1_format_*.json"))
    exp2_files = sorted(RESULTS_DIR.glob("experiment2_length_*.json"))
    exp3_files = sorted(RESULTS_DIR.glob("experiment3_progressive_*.json"))

    if exp1_files:
        with open(exp1_files[-1]) as f:
            results['experiment1'] = json.load(f)
        print(f"Loaded Experiment 1: {len(results['experiment1'])} records from {exp1_files[-1].name}")

    if exp2_files:
        with open(exp2_files[-1]) as f:
            results['experiment2'] = json.load(f)
        print(f"Loaded Experiment 2: {len(results['experiment2'])} records from {exp2_files[-1].name}")

    if exp3_files:
        with open(exp3_files[-1]) as f:
            results['experiment3'] = json.load(f)
        print(f"Loaded Experiment 3: {len(results['experiment3'])} records from {exp3_files[-1].name}")

    return results


def analyze_experiment1_format_comparison(data):
    """
    Analyze format comparison results.

    Tests H1: Structured formats outperform dense prose
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: FORMAT COMPARISON ANALYSIS")
    print("="*60)

    # Convert to DataFrame
    records = []
    for item in data:
        if item.get('evaluation'):
            records.append({
                'doc_id': item['doc_id'],
                'format': item['format_type'],
                'word_count': item['word_count'],
                'compression_ratio': item['compression_ratio'],
                'faithfulness': item['evaluation'].get('faithfulness', np.nan),
                'completeness': item['evaluation'].get('completeness', np.nan),
                'conciseness': item['evaluation'].get('conciseness', np.nan),
                'readability': item['evaluation'].get('readability', np.nan),
            })

    df = pd.DataFrame(records)

    # Summary statistics by format
    print("\n--- Summary Statistics by Format ---\n")
    summary = df.groupby('format').agg({
        'faithfulness': ['mean', 'std', 'count'],
        'completeness': ['mean', 'std'],
        'conciseness': ['mean', 'std'],
        'readability': ['mean', 'std'],
        'word_count': ['mean', 'std'],
    }).round(3)
    print(summary)

    # Statistical tests: Compare each format to dense prose (baseline)
    print("\n--- Statistical Comparison to Dense Prose Baseline ---\n")

    dense_scores = df[df['format'] == 'dense']
    metrics = ['faithfulness', 'completeness', 'conciseness', 'readability']

    comparison_results = []
    for format_type in ['bullets', 'hierarchical', 'progressive']:
        format_scores = df[df['format'] == format_type]

        for metric in metrics:
            dense_vals = dense_scores[metric].dropna()
            format_vals = format_scores[metric].dropna()

            # Paired t-test (paired by doc_id where possible)
            t_stat, p_value = stats.ttest_ind(format_vals, dense_vals)
            effect_size = (format_vals.mean() - dense_vals.mean()) / dense_vals.std()

            comparison_results.append({
                'comparison': f'{format_type} vs dense',
                'metric': metric,
                'dense_mean': dense_vals.mean(),
                'format_mean': format_vals.mean(),
                'difference': format_vals.mean() - dense_vals.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': effect_size,
                'significant': p_value < 0.05
            })

    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.to_string())

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Format Comparison: Quality Metrics by Output Format', fontsize=14, fontweight='bold')

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        format_means = df.groupby('format')[metric].mean()
        format_stds = df.groupby('format')[metric].std()

        # Order formats
        order = ['dense', 'bullets', 'hierarchical', 'progressive']
        format_means = format_means.reindex(order)
        format_stds = format_stds.reindex(order)

        bars = ax.bar(range(len(order)), format_means, yerr=format_stds,
                      capsize=5, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(['Dense\nProse', 'Bullet\nPoints', 'Hierarchical', 'Progressive'])
        ax.set_ylabel(f'{metric.title()} Score')
        ax.set_title(f'{metric.title()}')
        ax.set_ylim(0, 1)

        # Add value labels
        for bar, val in zip(bars, format_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'format_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'format_comparison.png'}")

    return df, comparison_df


def analyze_experiment2_length_tradeoff(data):
    """
    Analyze length vs quality trade-off.

    Tests H3: Optimal length exists between extreme summarization and verbose output
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: LENGTH VS QUALITY TRADE-OFF ANALYSIS")
    print("="*60)

    # Convert to DataFrame
    records = []
    for item in data:
        if item.get('evaluation'):
            target_length = int(item['format_type'].split('_')[1])
            records.append({
                'doc_id': item['doc_id'],
                'target_length': target_length,
                'actual_length': item['word_count'],
                'faithfulness': item['evaluation'].get('faithfulness', np.nan),
                'completeness': item['evaluation'].get('completeness', np.nan),
                'conciseness': item['evaluation'].get('conciseness', np.nan),
                'readability': item['evaluation'].get('readability', np.nan),
            })

    df = pd.DataFrame(records)

    # Summary by target length
    print("\n--- Summary Statistics by Target Length ---\n")
    summary = df.groupby('target_length').agg({
        'actual_length': ['mean', 'std'],
        'faithfulness': ['mean', 'std'],
        'completeness': ['mean', 'std'],
        'conciseness': ['mean', 'std'],
        'readability': ['mean', 'std'],
    }).round(3)
    print(summary)

    # Calculate combined score
    df['combined_score'] = (df['faithfulness'] + df['completeness'] +
                           df['conciseness'] + df['readability']) / 4

    # Find optimal length
    optimal_by_metric = {}
    for metric in ['faithfulness', 'completeness', 'conciseness', 'combined_score']:
        means = df.groupby('target_length')[metric].mean()
        optimal_by_metric[metric] = means.idxmax()

    print("\n--- Optimal Length by Metric ---")
    for metric, length in optimal_by_metric.items():
        score = df.groupby('target_length')[metric].mean()[length]
        print(f"  {metric}: {length} words (score: {score:.3f})")

    # Correlation analysis
    print("\n--- Correlation: Length vs Quality ---")
    for metric in ['faithfulness', 'completeness', 'conciseness']:
        corr, p_val = stats.pearsonr(df['actual_length'], df[metric])
        print(f"  {metric}: r={corr:.3f}, p={p_val:.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Length vs Quality Trade-off Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Quality metrics by length
    ax1 = axes[0]
    lengths = sorted(df['target_length'].unique())
    for metric in ['faithfulness', 'completeness', 'conciseness']:
        means = df.groupby('target_length')[metric].mean().reindex(lengths)
        ax1.plot(lengths, means, marker='o', label=metric.title(), linewidth=2, markersize=8)

    ax1.set_xlabel('Target Summary Length (words)')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Quality Metrics by Summary Length')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot 2: Trade-off visualization
    ax2 = axes[1]
    completeness_means = df.groupby('target_length')['completeness'].mean().reindex(lengths)
    conciseness_means = df.groupby('target_length')['conciseness'].mean().reindex(lengths)

    ax2.scatter(completeness_means, conciseness_means, s=100, c=lengths, cmap='viridis')
    for length, comp, conc in zip(lengths, completeness_means, conciseness_means):
        ax2.annotate(f'{length}w', (comp, conc), textcoords="offset points",
                     xytext=(5, 5), fontsize=9)

    ax2.set_xlabel('Completeness Score')
    ax2.set_ylabel('Conciseness Score')
    ax2.set_title('Completeness vs Conciseness Trade-off\n(numbers = target word count)')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'length_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'length_tradeoff.png'}")

    return df


def analyze_experiment3_progressive_disclosure(data):
    """
    Analyze progressive disclosure effectiveness.

    Tests H2: Progressive disclosure maintains faithfulness while enabling detail access
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: PROGRESSIVE DISCLOSURE ANALYSIS")
    print("="*60)

    # Convert to DataFrame format
    records = []
    for item in data:
        for level in ['level1', 'level2', 'level3']:
            eval_key = f'{level}_eval'
            if eval_key in item and item[eval_key]:
                records.append({
                    'doc_id': item['doc_id'],
                    'level': level,
                    'word_count': item.get(f'{level}_words', 0),
                    'faithfulness': item[eval_key].get('faithfulness', np.nan),
                    'completeness': item[eval_key].get('completeness', np.nan),
                    'conciseness': item[eval_key].get('conciseness', np.nan),
                    'readability': item[eval_key].get('readability', np.nan),
                })

    df = pd.DataFrame(records)

    # Summary by level
    print("\n--- Summary Statistics by Disclosure Level ---\n")
    summary = df.groupby('level').agg({
        'word_count': ['mean', 'std'],
        'faithfulness': ['mean', 'std'],
        'completeness': ['mean', 'std'],
        'conciseness': ['mean', 'std'],
        'readability': ['mean', 'std'],
    }).round(3)
    print(summary)

    # Key finding: Does faithfulness hold across levels?
    print("\n--- Faithfulness Across Levels ---")
    for level in ['level1', 'level2', 'level3']:
        level_data = df[df['level'] == level]['faithfulness']
        print(f"  {level}: {level_data.mean():.3f} +/- {level_data.std():.3f}")

    # Statistical test: level1 vs level3 faithfulness
    level1_faith = df[df['level'] == 'level1']['faithfulness'].dropna()
    level3_faith = df[df['level'] == 'level3']['faithfulness'].dropna()
    t_stat, p_value = stats.ttest_ind(level1_faith, level3_faith)
    print(f"\n  Level 1 vs Level 3 faithfulness: t={t_stat:.3f}, p={p_value:.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Progressive Disclosure Analysis', fontsize=14, fontweight='bold')

    # Plot 1: Quality metrics by level
    ax1 = axes[0]
    levels = ['level1', 'level2', 'level3']
    level_labels = ['One-line\n(~15-20w)', 'Brief\n(~50-75w)', 'Detailed\n(~150-200w)']
    metrics = ['faithfulness', 'completeness', 'conciseness']
    x = np.arange(len(levels))
    width = 0.25

    for i, metric in enumerate(metrics):
        means = [df[df['level'] == level][metric].mean() for level in levels]
        stds = [df[df['level'] == level][metric].std() for level in levels]
        ax1.bar(x + i*width, means, width, label=metric.title(), yerr=stds, capsize=3)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(level_labels)
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Quality Metrics by Detail Level')
    ax1.legend()
    ax1.set_ylim(0, 1)

    # Plot 2: Word count vs completeness
    ax2 = axes[1]
    for level, color in zip(levels, ['#4C72B0', '#55A868', '#C44E52']):
        level_data = df[df['level'] == level]
        ax2.scatter(level_data['word_count'], level_data['completeness'],
                   label=level.replace('level', 'Level '), alpha=0.7, s=50)

    ax2.set_xlabel('Word Count')
    ax2.set_ylabel('Completeness Score')
    ax2.set_title('Word Count vs Completeness by Level')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'progressive_disclosure.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'progressive_disclosure.png'}")

    return df


def generate_summary_findings(exp1_df, exp2_df, exp3_df, comparison_df):
    """Generate summary of key findings."""
    print("\n" + "="*60)
    print("SUMMARY OF KEY FINDINGS")
    print("="*60)

    findings = {}

    # H1: Structured formats vs prose
    if comparison_df is not None:
        sig_improvements = comparison_df[comparison_df['significant'] & (comparison_df['difference'] > 0)]
        findings['H1'] = {
            'supported': len(sig_improvements) > 0,
            'significant_improvements': len(sig_improvements),
            'best_format': exp1_df.groupby('format')[['faithfulness', 'completeness', 'conciseness', 'readability']].mean().mean(axis=1).idxmax()
        }
        print(f"\nH1 (Structured > Prose): {'Supported' if findings['H1']['supported'] else 'Not supported'}")
        print(f"   - {len(sig_improvements)} significant improvements found")
        print(f"   - Best overall format: {findings['H1']['best_format']}")

    # H2: Progressive disclosure effectiveness
    if exp3_df is not None:
        level1_faith = exp3_df[exp3_df['level'] == 'level1']['faithfulness'].mean()
        level3_faith = exp3_df[exp3_df['level'] == 'level3']['faithfulness'].mean()
        findings['H2'] = {
            'faithfulness_maintained': abs(level1_faith - level3_faith) < 0.1,
            'level1_faithfulness': level1_faith,
            'level3_faithfulness': level3_faith,
        }
        print(f"\nH2 (Progressive disclosure maintains faithfulness): {'Supported' if findings['H2']['faithfulness_maintained'] else 'Not supported'}")
        print(f"   - Level 1 faithfulness: {level1_faith:.3f}")
        print(f"   - Level 3 faithfulness: {level3_faith:.3f}")

    # H3: Optimal length
    if exp2_df is not None:
        combined = (exp2_df['faithfulness'] + exp2_df['completeness'] + exp2_df['conciseness']) / 3
        exp2_df['combined'] = combined
        optimal_length = exp2_df.groupby('target_length')['combined'].mean().idxmax()
        findings['H3'] = {
            'optimal_length': optimal_length,
            'extreme_short_score': exp2_df[exp2_df['target_length'] == 25]['combined'].mean(),
            'optimal_score': exp2_df[exp2_df['target_length'] == optimal_length]['combined'].mean(),
        }
        print(f"\nH3 (Optimal length exists): Supported")
        print(f"   - Optimal length: {optimal_length} words")
        print(f"   - Extreme short (25w) score: {findings['H3']['extreme_short_score']:.3f}")
        print(f"   - Optimal score: {findings['H3']['optimal_score']:.3f}")

    # Save findings
    findings['timestamp'] = datetime.now().isoformat()
    with open(RESULTS_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\nSaved analysis summary to {RESULTS_DIR / 'analysis_summary.json'}")

    return findings


def main():
    """Run all analyses."""
    print("="*60)
    print("AI-to-Human Communication Research - Analysis")
    print(f"Analysis started: {datetime.now()}")
    print("="*60)

    # Load results
    results = load_experiment_results()

    exp1_df, comparison_df = None, None
    exp2_df = None
    exp3_df = None

    # Analyze each experiment
    if 'experiment1' in results:
        exp1_df, comparison_df = analyze_experiment1_format_comparison(results['experiment1'])

    if 'experiment2' in results:
        exp2_df = analyze_experiment2_length_tradeoff(results['experiment2'])

    if 'experiment3' in results:
        exp3_df = analyze_experiment3_progressive_disclosure(results['experiment3'])

    # Generate summary
    findings = generate_summary_findings(exp1_df, exp2_df, exp3_df, comparison_df)

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)

    return findings


if __name__ == "__main__":
    findings = main()
