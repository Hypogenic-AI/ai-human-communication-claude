"""
Statistical analysis and visualization for AI-to-Human Communication experiments.
"""

import json
import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100


def load_results(summaries_path: str = "results/summaries/all_summaries.json",
                 evaluations_path: str = "results/evaluations/all_evaluations.json",
                 pairwise_path: str = "results/evaluations/pairwise_comparisons.json"):
    """Load all results files."""
    results = {}

    with open(summaries_path, "r") as f:
        results["summaries"] = json.load(f)

    with open(evaluations_path, "r") as f:
        results["evaluations"] = json.load(f)

    if os.path.exists(pairwise_path):
        with open(pairwise_path, "r") as f:
            results["pairwise"] = json.load(f)
    else:
        results["pairwise"] = None

    return results


def build_evaluation_dataframe(results: dict) -> pd.DataFrame:
    """Build DataFrame from evaluation results."""
    rows = []

    for doc_data in results["evaluations"]["evaluations"]:
        doc_id = doc_data["doc_id"]
        source = doc_data.get("source", "unknown")

        for strategy, eval_data in doc_data["strategies"].items():
            if "error" in eval_data:
                continue

            row = {
                "doc_id": doc_id,
                "source": source,
                "strategy": strategy,
                "faithfulness": eval_data.get("faithfulness", {}).get("score"),
                "completeness": eval_data.get("completeness", {}).get("score"),
                "conciseness": eval_data.get("conciseness", {}).get("score"),
                "clarity": eval_data.get("clarity", {}).get("score"),
                "overall": eval_data.get("overall", {}).get("score")
            }
            rows.append(row)

    return pd.DataFrame(rows)


def build_summary_stats_dataframe(results: dict) -> pd.DataFrame:
    """Build DataFrame with summary statistics (word counts, etc.)."""
    rows = []

    for doc_data in results["summaries"]["summaries"]:
        doc_id = doc_data["doc_id"]
        source = doc_data.get("source", "unknown")
        doc_length = doc_data.get("document_length", 0)

        for strategy, strat_data in doc_data["strategies"].items():
            if strat_data.get("error"):
                continue

            row = {
                "doc_id": doc_id,
                "source": source,
                "strategy": strategy,
                "word_count": strat_data.get("word_count", 0),
                "char_count": strat_data.get("char_count", 0),
                "document_length": doc_length,
                "compression_ratio": strat_data.get("char_count", 0) / max(doc_length, 1)
            }
            rows.append(row)

    return pd.DataFrame(rows)


def compute_strategy_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each strategy across all metrics."""
    metrics = ["faithfulness", "completeness", "conciseness", "clarity", "overall"]

    stats_list = []
    for strategy in df["strategy"].unique():
        strat_df = df[df["strategy"] == strategy]
        row = {"strategy": strategy, "n": len(strat_df)}

        for metric in metrics:
            values = strat_df[metric].dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) > 0 else None
            row[f"{metric}_std"] = values.std() if len(values) > 0 else None
            row[f"{metric}_n"] = len(values)

        stats_list.append(row)

    return pd.DataFrame(stats_list)


def run_anova_tests(df: pd.DataFrame) -> dict:
    """Run one-way ANOVA for each metric across strategies."""
    metrics = ["faithfulness", "completeness", "conciseness", "clarity", "overall"]
    results = {}

    for metric in metrics:
        # Group by strategy
        groups = []
        for strategy in df["strategy"].unique():
            values = df[df["strategy"] == strategy][metric].dropna().values
            if len(values) > 0:
                groups.append(values)

        if len(groups) >= 2:
            f_stat, p_value = stats.f_oneway(*groups)
            results[metric] = {
                "F_statistic": float(f_stat),
                "p_value": float(p_value),
                "significant": bool(p_value < 0.05)
            }
        else:
            results[metric] = {"error": "Not enough groups"}

    return results


def run_pairwise_ttests(df: pd.DataFrame, baseline: str = "dense_prose") -> dict:
    """Run pairwise t-tests comparing each strategy to baseline."""
    metrics = ["faithfulness", "completeness", "conciseness", "clarity", "overall"]
    strategies = [s for s in df["strategy"].unique() if s != baseline]

    results = {}
    baseline_df = df[df["strategy"] == baseline]

    for strategy in strategies:
        strat_df = df[df["strategy"] == strategy]
        results[strategy] = {}

        for metric in metrics:
            # Get paired samples (same doc_ids)
            common_docs = set(baseline_df["doc_id"]) & set(strat_df["doc_id"])

            baseline_vals = baseline_df[baseline_df["doc_id"].isin(common_docs)][metric].dropna()
            strat_vals = strat_df[strat_df["doc_id"].isin(common_docs)][metric].dropna()

            if len(baseline_vals) > 1 and len(strat_vals) > 1:
                # Match by doc_id for paired test
                baseline_dict = baseline_df.set_index("doc_id")[metric].to_dict()
                strat_dict = strat_df.set_index("doc_id")[metric].to_dict()

                paired_baseline = []
                paired_strat = []
                for doc_id in common_docs:
                    if doc_id in baseline_dict and doc_id in strat_dict:
                        b_val = baseline_dict[doc_id]
                        s_val = strat_dict[doc_id]
                        if pd.notna(b_val) and pd.notna(s_val):
                            paired_baseline.append(b_val)
                            paired_strat.append(s_val)

                if len(paired_baseline) > 1:
                    t_stat, p_value = stats.ttest_rel(paired_strat, paired_baseline)
                    effect_size = (np.mean(paired_strat) - np.mean(paired_baseline)) / np.std(paired_baseline) if np.std(paired_baseline) > 0 else 0

                    results[strategy][metric] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                        "effect_size_d": float(effect_size),
                        "mean_diff": float(np.mean(paired_strat) - np.mean(paired_baseline)),
                        "n_pairs": int(len(paired_baseline))
                    }
                else:
                    results[strategy][metric] = {"error": "Not enough paired samples"}
            else:
                results[strategy][metric] = {"error": "Not enough samples"}

    return results


def analyze_pairwise_comparisons(results: dict) -> dict:
    """Analyze pairwise comparison results."""
    if not results.get("pairwise"):
        return None

    pair_results = {}

    for comparison in results["pairwise"]["comparisons"]:
        for pair_key, pair_data in comparison.get("pairs", {}).items():
            if pair_key not in pair_results:
                pair_results[pair_key] = {"A_wins": 0, "B_wins": 0, "ties": 0, "total": 0}

            winner = pair_data.get("winner")
            if winner == "A":
                pair_results[pair_key]["A_wins"] += 1
            elif winner == "B":
                pair_results[pair_key]["B_wins"] += 1
            elif winner == "tie":
                pair_results[pair_key]["ties"] += 1

            pair_results[pair_key]["total"] += 1

    # Calculate win rates
    for pair_key in pair_results:
        total = pair_results[pair_key]["total"]
        if total > 0:
            pair_results[pair_key]["A_win_rate"] = pair_results[pair_key]["A_wins"] / total
            pair_results[pair_key]["B_win_rate"] = pair_results[pair_key]["B_wins"] / total

    return pair_results


def plot_strategy_comparison(df: pd.DataFrame, output_dir: str = "results/figures"):
    """Create comparison plots for all strategies."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics = ["faithfulness", "completeness", "conciseness", "clarity", "overall"]

    # Bar plot of mean scores by strategy
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))

    for i, metric in enumerate(metrics):
        strategy_means = df.groupby("strategy")[metric].mean().sort_values(ascending=False)
        strategy_stds = df.groupby("strategy")[metric].std()

        ax = axes[i]
        bars = ax.bar(range(len(strategy_means)), strategy_means.values)
        ax.errorbar(range(len(strategy_means)), strategy_means.values,
                    yerr=strategy_stds[strategy_means.index].values,
                    fmt='none', color='black', capsize=3)

        ax.set_xticks(range(len(strategy_means)))
        ax.set_xticklabels([s.replace('_', '\n') for s in strategy_means.index], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("Score (1-5)")
        ax.set_title(metric.capitalize())
        ax.set_ylim(1, 5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_comparison_bars.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Heatmap of all scores
    pivot_df = df.groupby("strategy")[metrics].mean()
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=1, vmax=5)
    plt.title("Mean Scores by Strategy and Metric")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/strategy_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {output_dir}/")


def plot_word_count_distribution(summary_df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot word count distribution by strategy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    strategy_order = summary_df.groupby("strategy")["word_count"].median().sort_values().index
    sns.boxplot(data=summary_df, x="strategy", y="word_count", order=strategy_order)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Word Count")
    plt.title("Summary Length by Strategy")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/word_count_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_length_vs_quality(eval_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: str = "results/figures"):
    """Plot relationship between summary length and quality scores."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Merge dataframes
    merged = eval_df.merge(summary_df[["doc_id", "strategy", "word_count"]], on=["doc_id", "strategy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter: word count vs overall quality
    ax1 = axes[0]
    for strategy in merged["strategy"].unique():
        strat_data = merged[merged["strategy"] == strategy]
        ax1.scatter(strat_data["word_count"], strat_data["overall"], alpha=0.5, label=strategy, s=20)

    ax1.set_xlabel("Word Count")
    ax1.set_ylabel("Overall Quality Score")
    ax1.set_title("Summary Length vs Quality")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Average by length bin
    ax2 = axes[1]
    merged["length_bin"] = pd.cut(merged["word_count"], bins=[0, 25, 50, 100, 200, 1000])
    bin_means = merged.groupby("length_bin", observed=True)[["completeness", "conciseness", "overall"]].mean()
    bin_means.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Word Count Range")
    ax2.set_ylabel("Mean Score")
    ax2.set_title("Quality Metrics by Summary Length")
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_vs_quality.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_summary_report(results: dict, eval_df: pd.DataFrame, summary_df: pd.DataFrame) -> str:
    """Generate text summary of analysis results."""
    report = []
    report.append("=" * 60)
    report.append("EXPERIMENTAL RESULTS SUMMARY")
    report.append("=" * 60)

    # Basic stats
    n_docs = len(eval_df["doc_id"].unique())
    n_strategies = len(eval_df["strategy"].unique())
    report.append(f"\nDataset: {n_docs} documents, {n_strategies} strategies")
    report.append(f"Total evaluations: {len(eval_df)}")

    # Strategy rankings
    report.append("\n" + "-" * 40)
    report.append("STRATEGY RANKINGS (by Overall Score)")
    report.append("-" * 40)

    strategy_stats = compute_strategy_stats(eval_df)
    strategy_stats = strategy_stats.sort_values("overall_mean", ascending=False)

    for _, row in strategy_stats.iterrows():
        report.append(f"  {row['strategy']}: {row['overall_mean']:.2f} +/- {row['overall_std']:.2f}")

    # ANOVA results
    report.append("\n" + "-" * 40)
    report.append("ANOVA RESULTS (Strategy Differences)")
    report.append("-" * 40)

    anova = run_anova_tests(eval_df)
    for metric, result in anova.items():
        if "error" not in result:
            sig = "*" if result["significant"] else ""
            report.append(f"  {metric}: F={result['F_statistic']:.2f}, p={result['p_value']:.4f} {sig}")

    # Pairwise comparisons
    pairwise = analyze_pairwise_comparisons(results)
    if pairwise:
        report.append("\n" + "-" * 40)
        report.append("PAIRWISE COMPARISONS")
        report.append("-" * 40)
        for pair, data in pairwise.items():
            parts = pair.split("_vs_")
            if len(parts) == 2:
                report.append(f"  {parts[0]} vs {parts[1]}:")
                report.append(f"    {parts[0]} wins: {data['A_wins']} ({data['A_win_rate']*100:.1f}%)")
                report.append(f"    {parts[1]} wins: {data['B_wins']} ({data['B_win_rate']*100:.1f}%)")
                report.append(f"    Ties: {data['ties']}")

    # Word count stats
    report.append("\n" + "-" * 40)
    report.append("SUMMARY LENGTH BY STRATEGY")
    report.append("-" * 40)

    word_stats = summary_df.groupby("strategy")["word_count"].agg(["mean", "std", "min", "max"])
    for strategy, row in word_stats.iterrows():
        report.append(f"  {strategy}: {row['mean']:.0f} +/- {row['std']:.0f} words (range: {row['min']:.0f}-{row['max']:.0f})")

    return "\n".join(report)


def save_analysis_results(results: dict, output_dir: str = "results"):
    """Run full analysis and save all results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build dataframes
    eval_df = build_evaluation_dataframe(results)
    summary_df = build_summary_stats_dataframe(results)

    # Save dataframes
    eval_df.to_csv(f"{output_dir}/evaluation_scores.csv", index=False)
    summary_df.to_csv(f"{output_dir}/summary_stats.csv", index=False)

    # Compute and save strategy stats
    strategy_stats = compute_strategy_stats(eval_df)
    strategy_stats.to_csv(f"{output_dir}/strategy_stats.csv", index=False)

    # ANOVA
    anova_results = run_anova_tests(eval_df)
    with open(f"{output_dir}/anova_results.json", "w") as f:
        json.dump(anova_results, f, indent=2)

    # Pairwise t-tests
    ttest_results = run_pairwise_ttests(eval_df)
    with open(f"{output_dir}/pairwise_ttests.json", "w") as f:
        json.dump(ttest_results, f, indent=2)

    # Pairwise comparisons analysis
    pairwise = analyze_pairwise_comparisons(results)
    if pairwise:
        with open(f"{output_dir}/pairwise_analysis.json", "w") as f:
            json.dump(pairwise, f, indent=2)

    # Generate plots
    plot_strategy_comparison(eval_df, f"{output_dir}/figures")
    plot_word_count_distribution(summary_df, f"{output_dir}/figures")
    plot_length_vs_quality(eval_df, summary_df, f"{output_dir}/figures")

    # Generate summary report
    report = generate_summary_report(results, eval_df, summary_df)
    print(report)

    with open(f"{output_dir}/analysis_summary.txt", "w") as f:
        f.write(report)

    return {
        "eval_df": eval_df,
        "summary_df": summary_df,
        "strategy_stats": strategy_stats,
        "anova": anova_results,
        "pairwise_ttests": ttest_results,
        "pairwise_comparisons": pairwise
    }


if __name__ == "__main__":
    print("Loading results...")
    results = load_results()

    print("\nRunning analysis...")
    analysis = save_analysis_results(results)

    print("\nAnalysis complete!")
