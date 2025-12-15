"""
Visualization module for AI-to-Human Communication experiments.

Creates publication-quality figures for the research report.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

BASE_DIR = Path("/data/hypogenicai/workspaces/ai-human-communication-claude")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


def load_latest_results() -> Dict:
    """Load the most recent results from each experiment."""
    results = {}

    # Find latest files for each experiment
    for prefix in ['experiment1_format', 'experiment2_length', 'experiment3_progressive']:
        files = list(RESULTS_DIR.glob(f"{prefix}_*.json"))
        if files:
            latest = max(files, key=lambda x: x.stat().st_mtime)
            with open(latest) as f:
                results[prefix.split('_')[0] + '_' + prefix.split('_')[1]] = json.load(f)
            print(f"Loaded {latest.name}")

    # Validation CSV
    val_files = list(RESULTS_DIR.glob("validation_*.csv"))
    if val_files:
        latest_val = max(val_files, key=lambda x: x.stat().st_mtime)
        results['validation'] = pd.read_csv(latest_val)
        print(f"Loaded {latest_val.name}")

    return results


def plot_format_comparison(results: List[Dict], save: bool = True) -> plt.Figure:
    """
    Plot comparing different summary formats on evaluation metrics.

    Creates a grouped bar chart showing faithfulness, completeness, conciseness,
    and readability for each format type.
    """
    df = pd.DataFrame(results)

    # Extract evaluation scores
    df['faithfulness'] = df['evaluation'].apply(lambda x: x['faithfulness'] if x else None)
    df['completeness'] = df['evaluation'].apply(lambda x: x['completeness'] if x else None)
    df['conciseness'] = df['evaluation'].apply(lambda x: x['conciseness'] if x else None)
    df['readability'] = df['evaluation'].apply(lambda x: x['readability'] if x else None)

    # Aggregate by format
    metrics = ['faithfulness', 'completeness', 'conciseness', 'readability']
    format_order = ['dense', 'bullets', 'hierarchical', 'progressive']

    agg_data = df.groupby('format_type')[metrics].agg(['mean', 'std']).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Bar chart of mean scores
    ax1 = axes[0]
    x = np.arange(len(format_order))
    width = 0.2
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    for i, metric in enumerate(metrics):
        means = [agg_data[agg_data['format_type'] == fmt][(metric, 'mean')].values[0]
                 for fmt in format_order if fmt in agg_data['format_type'].values]
        stds = [agg_data[agg_data['format_type'] == fmt][(metric, 'std')].values[0]
                for fmt in format_order if fmt in agg_data['format_type'].values]

        ax1.bar(x + i*width, means, width, label=metric.capitalize(),
                color=colors[i], yerr=stds, capsize=3, alpha=0.8)

    ax1.set_xlabel('Summary Format')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Quality Metrics by Summary Format')
    ax1.set_xticks(x + width*1.5)
    ax1.set_xticklabels([f.capitalize() for f in format_order])
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)

    # Plot 2: Word count distribution by format
    ax2 = axes[1]
    word_counts = [df[df['format_type'] == fmt]['word_count'].values for fmt in format_order
                   if fmt in df['format_type'].values]
    format_labels = [f.capitalize() for f in format_order if f in df['format_type'].values]

    bp = ax2.boxplot(word_counts, labels=format_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(word_counts)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_xlabel('Summary Format')
    ax2.set_ylabel('Word Count')
    ax2.set_title('Summary Length by Format')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "format_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved format_comparison.png")

    return fig


def plot_length_tradeoff(results: List[Dict], save: bool = True) -> plt.Figure:
    """
    Plot showing quality metrics vs summary length.

    Creates line plots showing how faithfulness, completeness, and conciseness
    change with target summary length.
    """
    df = pd.DataFrame(results)

    # Extract target length
    df['target_length'] = df['format_type'].apply(lambda x: int(x.split('_')[1]))
    df['faithfulness'] = df['evaluation'].apply(lambda x: x['faithfulness'] if x else None)
    df['completeness'] = df['evaluation'].apply(lambda x: x['completeness'] if x else None)
    df['conciseness'] = df['evaluation'].apply(lambda x: x['conciseness'] if x else None)

    # Aggregate by target length
    agg = df.groupby('target_length').agg({
        'word_count': ['mean', 'std'],
        'faithfulness': ['mean', 'std'],
        'completeness': ['mean', 'std'],
        'conciseness': ['mean', 'std']
    }).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Quality vs length tradeoff
    ax1 = axes[0]
    lengths = agg['target_length'].values
    metrics = ['faithfulness', 'completeness', 'conciseness']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    markers = ['o', 's', '^']

    for metric, color, marker in zip(metrics, colors, markers):
        means = agg[(metric, 'mean')].values
        stds = agg[(metric, 'std')].values
        ax1.errorbar(lengths, means, yerr=stds, label=metric.capitalize(),
                     color=color, marker=marker, capsize=3, linewidth=2, markersize=8)

    ax1.set_xlabel('Target Summary Length (words)')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Quality Metrics vs Summary Length')
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, 450)

    # Plot 2: Composite score (weighted average)
    ax2 = axes[1]

    # Compute composite score emphasizing balanced quality
    composite = 0.35 * agg[('faithfulness', 'mean')] + \
                0.35 * agg[('completeness', 'mean')] + \
                0.30 * agg[('conciseness', 'mean')]

    ax2.plot(lengths, composite, 'ko-', linewidth=2, markersize=10)
    ax2.fill_between(lengths, composite - 0.05, composite + 0.05, alpha=0.2, color='gray')

    # Highlight optimal region
    optimal_idx = composite.idxmax()
    optimal_length = lengths[optimal_idx]
    optimal_score = composite.iloc[optimal_idx]
    ax2.axvline(x=optimal_length, color='red', linestyle='--', alpha=0.7)
    ax2.annotate(f'Optimal: ~{optimal_length} words',
                 xy=(optimal_length, optimal_score),
                 xytext=(optimal_length + 50, optimal_score - 0.05),
                 fontsize=10, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    ax2.set_xlabel('Target Summary Length (words)')
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Optimal Summary Length\n(0.35×Faith + 0.35×Complete + 0.30×Concise)')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, 450)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "length_tradeoff.png", dpi=150, bbox_inches='tight')
        print(f"Saved length_tradeoff.png")

    return fig


def plot_progressive_disclosure(results: List[Dict], save: bool = True) -> plt.Figure:
    """
    Plot showing quality at different progressive disclosure levels.

    Compares one-line, brief, and detailed summaries.
    """
    df = pd.DataFrame(results)

    # Extract metrics for each level
    levels = ['level1', 'level2', 'level3']
    level_names = ['One-line (~20 words)', 'Brief (~70 words)', 'Detailed (~175 words)']
    metrics = ['faithfulness', 'completeness', 'conciseness']

    level_data = {level: {
        'words': df[f'{level}_words'].mean(),
        'words_std': df[f'{level}_words'].std(),
        **{m: df[f'{level}_eval'].apply(lambda x: x[m]).mean() for m in metrics},
        **{f'{m}_std': df[f'{level}_eval'].apply(lambda x: x[m]).std() for m in metrics}
    } for level in levels}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Radar/bar chart of metrics by level
    ax1 = axes[0]
    x = np.arange(3)
    width = 0.25
    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for i, metric in enumerate(metrics):
        means = [level_data[level][metric] for level in levels]
        stds = [level_data[level][f'{metric}_std'] for level in levels]
        ax1.bar(x + i*width, means, width, label=metric.capitalize(),
                color=colors[i], yerr=stds, capsize=3, alpha=0.8)

    ax1.set_xlabel('Summary Level')
    ax1.set_ylabel('Score (0-1)')
    ax1.set_title('Quality by Progressive Disclosure Level')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(level_names, fontsize=9)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)

    # Plot 2: Information density (completeness per word)
    ax2 = axes[1]
    word_counts = [level_data[level]['words'] for level in levels]
    completeness = [level_data[level]['completeness'] for level in levels]

    # Calculate information density
    density = [c / (w/100) for c, w in zip(completeness, word_counts)]  # normalized to per 100 words

    bars = ax2.bar(range(3), density, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(level_names, fontsize=9)
    ax2.set_ylabel('Completeness per 100 Words')
    ax2.set_title('Information Density by Summary Level')

    # Add word count annotations
    for i, (bar, wc) in enumerate(zip(bars, word_counts)):
        ax2.annotate(f'{wc:.0f} words',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords='offset points',
                     ha='center', fontsize=9)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "progressive_disclosure.png", dpi=150, bbox_inches='tight')
        print(f"Saved progressive_disclosure.png")

    return fig


def plot_validation_correlation(validation_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """
    Plot correlation between LLM-as-judge scores and FeedSum scores.

    Validates that our evaluation aligns with established benchmarks.
    """
    # Filter rows with valid scores
    df = validation_df.dropna()

    if len(df) < 10:
        print("Insufficient validation data for correlation plot")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    dimensions = [
        ('feedsum_faithfulness', 'llm_faithfulness', 'Faithfulness'),
        ('feedsum_completeness', 'llm_completeness', 'Completeness'),
        ('feedsum_conciseness', 'llm_conciseness', 'Conciseness')
    ]

    for ax, (feedsum_col, llm_col, title) in zip(axes, dimensions):
        x = df[feedsum_col].values
        y = df[llm_col].values

        # Filter valid pairs
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]

        if len(x) < 5:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.scatter(x, y, alpha=0.5, edgecolors='none', s=30)

        # Correlation
        r, p = stats.pearsonr(x, y)

        # Regression line
        z = np.polyfit(x, y, 1)
        p_line = np.poly1d(z)
        x_line = np.linspace(min(x), max(x), 100)
        ax.plot(x_line, p_line(x_line), 'r-', linewidth=2, alpha=0.7)

        ax.set_xlabel(f'FeedSum {title}')
        ax.set_ylabel(f'LLM-as-Judge {title}')
        ax.set_title(f'{title}\nr={r:.3f}, p={p:.4f}')

        # Perfect correlation reference
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "validation_correlation.png", dpi=150, bbox_inches='tight')
        print(f"Saved validation_correlation.png")

    return fig


def plot_domain_analysis(results: List[Dict], save: bool = True) -> plt.Figure:
    """
    Plot showing how quality varies across different content domains.
    """
    df = pd.DataFrame(results)

    # Extract metrics
    df['faithfulness'] = df['evaluation'].apply(lambda x: x['faithfulness'] if x else None)
    df['completeness'] = df['evaluation'].apply(lambda x: x['completeness'] if x else None)
    df['readability'] = df['evaluation'].apply(lambda x: x['readability'] if x else None)

    # Aggregate by domain
    domain_stats = df.groupby('source_domain').agg({
        'faithfulness': 'mean',
        'completeness': 'mean',
        'readability': 'mean',
        'word_count': 'mean'
    }).reset_index()

    if len(domain_stats) < 2:
        print("Insufficient domain diversity for analysis")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by average quality
    domain_stats['avg_quality'] = (domain_stats['faithfulness'] +
                                   domain_stats['completeness'] +
                                   domain_stats['readability']) / 3
    domain_stats = domain_stats.sort_values('avg_quality', ascending=True)

    domains = domain_stats['source_domain'].values
    y = np.arange(len(domains))
    height = 0.25

    colors = ['#2ecc71', '#3498db', '#9b59b6']
    for i, (metric, color) in enumerate(zip(['faithfulness', 'completeness', 'readability'], colors)):
        ax.barh(y + i*height, domain_stats[metric].values, height,
                label=metric.capitalize(), color=color, alpha=0.8)

    ax.set_yticks(y + height)
    ax.set_yticklabels(domains)
    ax.set_xlabel('Score (0-1)')
    ax.set_title('Summary Quality by Content Domain')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)

    plt.tight_layout()

    if save:
        plt.savefig(FIGURES_DIR / "domain_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved domain_analysis.png")

    return fig


def create_summary_table(results: Dict) -> str:
    """Create a markdown table summarizing key findings."""

    if 'experiment1_format' not in results:
        return "No results available"

    df = pd.DataFrame(results['experiment1_format'])
    df['faithfulness'] = df['evaluation'].apply(lambda x: x['faithfulness'] if x else None)
    df['completeness'] = df['evaluation'].apply(lambda x: x['completeness'] if x else None)
    df['conciseness'] = df['evaluation'].apply(lambda x: x['conciseness'] if x else None)
    df['readability'] = df['evaluation'].apply(lambda x: x['readability'] if x else None)

    summary = df.groupby('format_type').agg({
        'faithfulness': ['mean', 'std'],
        'completeness': ['mean', 'std'],
        'conciseness': ['mean', 'std'],
        'readability': ['mean', 'std'],
        'word_count': 'mean'
    }).round(3)

    md = "| Format | Faithfulness | Completeness | Conciseness | Readability | Avg Words |\n"
    md += "|--------|--------------|--------------|-------------|-------------|----------|\n"

    for fmt in ['dense', 'bullets', 'hierarchical', 'progressive']:
        if fmt in summary.index:
            row = summary.loc[fmt]
            md += f"| {fmt.capitalize()} | "
            md += f"{row[('faithfulness', 'mean')]:.2f}±{row[('faithfulness', 'std')]:.2f} | "
            md += f"{row[('completeness', 'mean')]:.2f}±{row[('completeness', 'std')]:.2f} | "
            md += f"{row[('conciseness', 'mean')]:.2f}±{row[('conciseness', 'std')]:.2f} | "
            md += f"{row[('readability', 'mean')]:.2f}±{row[('readability', 'std')]:.2f} | "
            md += f"{row[('word_count', 'mean')]:.0f} |\n"

    return md


def generate_all_figures(results: Dict = None):
    """Generate all figures from experiment results."""

    if results is None:
        results = load_latest_results()

    if 'experiment1_format' in results:
        print("\nGenerating format comparison plot...")
        plot_format_comparison(results['experiment1_format'])

        print("\nGenerating domain analysis plot...")
        plot_domain_analysis(results['experiment1_format'])

    if 'experiment2_length' in results:
        print("\nGenerating length tradeoff plot...")
        plot_length_tradeoff(results['experiment2_length'])

    if 'experiment3_progressive' in results:
        print("\nGenerating progressive disclosure plot...")
        plot_progressive_disclosure(results['experiment3_progressive'])

    if 'validation' in results:
        print("\nGenerating validation correlation plot...")
        plot_validation_correlation(results['validation'])

    print(f"\nAll figures saved to {FIGURES_DIR}")

    return results


if __name__ == "__main__":
    results = generate_all_figures()
