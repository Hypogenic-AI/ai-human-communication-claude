# AI-to-Human Communication Research

Research project investigating optimal formats and strategies for AI systems to communicate information effectively to humans.

## Key Findings

- **Structured formats win decisively**: Bullet points beat dense prose 90% of the time; progressive hierarchy wins 95%
- **Progressive disclosure is optimal**: TL;DR + Key Points + Detailed summary tied for highest overall quality (4.85/5)
- **Formal style preferred**: Technical tone beats conversational 80% of the time for informational content
- **Completeness vs conciseness trade-off confirmed**: Extreme brevity (36 words) sacrifices completeness; optimal balance at 130-200 words
- **All strategies maintain high faithfulness**: No significant difference in accuracy across formats (F=0.83, p=0.54)

## Quick Start

### Setup Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv add datasets numpy pandas scipy matplotlib seaborn openai httpx tenacity rouge-score
```

### Run Experiments
```bash
# Set API key (OpenRouter or OpenAI)
export OPENROUTER_API_KEY="your-key"

# Run full experiment pipeline
python src/run_experiment.py --n-documents 100

# Or run individual components:
python src/data_loader.py           # Sample documents
python src/generate_summaries.py    # Generate summaries
python src/evaluate.py              # LLM-as-judge evaluation
python src/analyze.py               # Statistical analysis
```

### View Results
- **Full Report**: [REPORT.md](REPORT.md)
- **Visualizations**: `results/figures/` directory
- **Raw Data**: `results/` directory

## Project Structure

```
.
├── REPORT.md                    # Full research report with findings
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature synthesis
├── resources.md                 # Resource catalog
├── src/
│   ├── run_experiment.py        # Main experiment orchestrator
│   ├── data_loader.py           # FeedSum data loading and sampling
│   ├── prompts.py               # Communication strategy prompts
│   ├── generate_summaries.py    # Summary generation via LLM API
│   ├── evaluate.py              # LLM-as-judge evaluation
│   └── analyze.py               # Statistical analysis and visualization
├── results/
│   ├── summaries/               # Generated summaries (140 total)
│   ├── evaluations/             # Evaluation scores and comparisons
│   ├── figures/                 # Visualization outputs
│   ├── evaluation_scores.csv    # Tabular evaluation data
│   └── strategy_stats.csv       # Summary statistics by strategy
├── datasets/                    # FeedSum and other datasets
├── papers/                      # Reference papers (11 PDFs)
└── code/                        # Reference implementations
```

## Research Summary

### Strategies Tested

| Strategy | Description | Avg Words | Overall Score |
|----------|-------------|-----------|---------------|
| Structured Sections | Headers + bullet points | 205 | **4.85** |
| Progressive Hierarchy | TL;DR + bullets + detailed | 273 | **4.85** |
| Formal Technical | Professional tone | 246 | 4.70 |
| Dense Prose | Standard paragraph | 154 | 4.60 |
| Bullet Points | Key points list | 132 | 4.55 |
| Conversational | Casual, engaging tone | 245 | 4.40 |
| Concise | 1-2 sentences | 36 | 4.15 |

### Pairwise Comparison Results

| Comparison | Winner | Win Rate |
|------------|--------|----------|
| Dense Prose vs. Bullet Points | Bullet Points | **90%** |
| Dense Prose vs. Progressive | Progressive | **95%** |
| Formal vs. Conversational | Formal | **80%** |

### Hypothesis Testing

| Hypothesis | Supported? | Evidence |
|------------|-----------|----------|
| H1: Structured > Prose | **Yes** | 90-95% pairwise win rates |
| H2: Length trade-off exists | **Yes** | ANOVA p<0.001 for completeness and conciseness |
| H3: Progressive disclosure optimal | **Yes** | Tied for highest score, 95% win rate |
| H4: Style preferences exist | **Partial** | Formal preferred 80%, but not significant overall |

### Methodology
- **Data**: FeedSum dataset (7 domains, 20 documents, 140 summaries)
- **Model**: GPT-4.1 via OpenRouter API
- **Evaluation**: LLM-as-judge on faithfulness, completeness, conciseness, clarity, overall
- **Statistics**: One-way ANOVA, paired t-tests, Cohen's d effect sizes

## Practical Recommendations

For AI system designers:
1. **Use progressive disclosure** - Present TL;DR first, then expandable details
2. **Structure information** - Use bullet points, headers, clear sections
3. **Target 130-200 words** for balanced summaries
4. **Use formal tone** for informational content
5. **Maintain faithfulness** - Structured formats don't sacrifice accuracy

## Files Generated

| File | Description |
|------|-------------|
| `results/summaries/all_summaries.json` | 140 generated summaries |
| `results/evaluations/all_evaluations.json` | LLM-judge scores |
| `results/evaluations/pairwise_comparisons.json` | Head-to-head comparisons |
| `results/figures/*.png` | Visualization outputs |
| `results/analysis_summary.txt` | Statistical summary |

## Citation

If you use this research:
```
AI-to-Human Communication Research (2025)
Research on optimal communication formats for AI-generated outputs.
```

## License

Research code and analysis released for academic use.

---

*Generated: December 14, 2025*
