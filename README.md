# AI-to-Human Communication Research

Research project investigating optimal formats and strategies for AI systems to communicate information effectively to humans.

## Key Findings

- **Progressive disclosure wins**: Summary formats with TL;DR + expandable details significantly outperform dense prose on readability (p=0.007) and conciseness (p=0.05)
- **Optimal length is ~50 words**: Balances completeness with conciseness; shorter loses info, longer shows diminishing returns
- **Faithfulness maintained across levels**: Multi-level summaries maintain >0.8 faithfulness even at extreme compression
- **Best format overall**: Progressive (TL;DR + Key Points + Details)

## Quick Start

### Setup Environment
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install datasets numpy pandas scipy matplotlib seaborn openai httpx python-dotenv tenacity tqdm
```

### Run Experiments
```bash
# Set API key
export OPENROUTER_API_KEY="your-key"

# Run main experiments
python src/experiment.py

# Run analysis
python src/analyze_results.py
```

### View Results
- **Full Report**: [REPORT.md](REPORT.md)
- **Figures**: `figures/` directory
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
│   ├── experiment.py            # Main experiment script
│   ├── analyze_results.py       # Analysis and visualization
│   ├── config.py                # Configuration settings
│   └── llm_client.py            # LLM API utilities
├── results/
│   ├── experiment1_format_*.json     # Format comparison results
│   ├── experiment2_length_*.json     # Length trade-off results
│   ├── experiment3_progressive_*.json # Progressive disclosure results
│   └── analysis_summary.json         # Summary of findings
├── figures/
│   ├── format_comparison.png         # Format comparison visualization
│   ├── length_tradeoff.png           # Length vs quality plot
│   └── progressive_disclosure.png    # Progressive disclosure analysis
├── datasets/                    # FeedSum and other datasets
├── papers/                      # Reference papers (PDFs)
└── code/                        # Reference implementations
```

## Research Summary

### Hypothesis Testing Results

| Hypothesis | Status | Key Evidence |
|------------|--------|--------------|
| H1: Structured > Prose | Supported | Progressive format shows significant improvements |
| H2: Progressive maintains quality | Supported | Faithfulness >0.8 across all levels |
| H3: Optimal length exists | Supported | 50 words optimal; extremes underperform |

### Methodology
- **Data**: FeedSum dataset (7 domains, 80+ documents)
- **Model**: GPT-4o-mini via OpenRouter API
- **Evaluation**: LLM-as-judge on faithfulness, completeness, conciseness, readability
- **Statistics**: Independent t-tests with Bonferroni correction

## Practical Recommendations

For AI system designers:
1. **Use progressive disclosure** - Present TL;DR first, then expandable details
2. **Target 50 words** for initial summaries
3. **Use visual hierarchy** - Bullet points, headers, clear sections
4. **Maintain faithfulness** - Even short summaries should be accurate

## Citation

If you use this research, please cite:
```
AI-to-Human Communication Research (2025)
Research on optimal communication formats for AI-generated outputs.
https://github.com/[repository]
```

## License

Research code and analysis released for academic use.

---

*Generated: December 14, 2025*
