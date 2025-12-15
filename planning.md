# Research Plan: AI-to-Human Communication Strategies

## Research Question

**How can AI systems communicate large volumes of information more effectively to humans, and which communication strategies improve human understanding compared to traditional dense output?**

Specifically:
1. Do progressive disclosure strategies (hierarchical summaries) improve comprehension compared to flat summaries?
2. Do structured formats (bullet points, sections) outperform prose-style summaries?
3. Is there an optimal level of conciseness that balances information completeness with cognitive load?

## Background and Motivation

AI systems, particularly research agents, generate dense, comprehensive outputs that can overwhelm human readers. While AI benefits from dense information, humans prefer concise, well-structured communication. This mismatch creates a bottleneck in human-AI collaboration, particularly in:

- Research contexts where agents read many papers and generate lengthy reports
- Onboarding scenarios where humans need to understand complex AI-generated analysis
- Decision-making contexts where humans must verify AI conclusions

The literature identifies several promising approaches:
1. **Progressive Disclosure** (Springer & Whittaker, 2018): Start simple, reveal complexity progressively
2. **Cognitive Ergonomics** (CogErgLLM, 2024): Design for reduced mental workload
3. **Multi-dimensional Optimization** (FeedSum, 2024): Balance faithfulness, completeness, and conciseness
4. **Human-like Communication** (Human-Like-DPO): Natural conversational style vs. formal AI style

## Hypothesis Decomposition

**H1 (Structure)**: Structured formats (headers, bullets) will receive higher human preference ratings than prose-style summaries for the same content.

**H2 (Length/Conciseness)**: There is a non-linear relationship between summary length and perceived quality - very short summaries sacrifice completeness, very long ones increase cognitive load.

**H3 (Progressive Disclosure)**: Hierarchical summaries (one-sentence TL;DR + paragraph + full) will be preferred over single-length summaries.

**H4 (Style)**: Human-like conversational style will be preferred for certain contexts but not others (e.g., casual updates vs. formal research reports).

## Proposed Methodology

### Approach: Comparative LLM Experiment with LLM-as-Judge Evaluation

We will use real LLM APIs (GPT-4.1/Claude) to generate summaries using different communication strategies, then evaluate using:
1. Multi-dimensional LLM-as-judge scoring (faithfulness, completeness, conciseness, clarity)
2. Automated metrics (ROUGE, BERTScore) for baseline comparison
3. Pairwise preference rankings between strategies

### Experimental Steps

#### Step 1: Data Selection and Preparation
- Sample 100 diverse documents from FeedSum test set (covering multiple domains)
- Extract documents with human reference summaries for ground truth
- Ensure mix of document lengths and complexity levels

#### Step 2: Define Communication Strategies to Test

**Baseline Strategies:**
1. **Zero-shot Dense**: Standard LLM summary without formatting constraints
2. **Zero-shot Concise**: Short summary (1-2 sentences max)

**Structured Strategies:**
3. **Bullet Point Summary**: Key points as bullet list
4. **Structured Sections**: Headers for Context/Key Findings/Implications
5. **Progressive Hierarchy**:
   - Level 1: One-sentence TL;DR
   - Level 2: 3-5 bullet points
   - Level 3: Detailed paragraph

**Style Strategies:**
6. **Formal Technical**: Professional, objective tone
7. **Conversational**: Natural, engaging tone (informed by Human-Like-DPO)

#### Step 3: Generate Summaries Using Real LLM APIs
- Use GPT-4.1 (or Claude Sonnet 4.5) via API
- Apply each strategy to all 100 documents
- Total: 7 strategies x 100 documents = 700 summaries
- Set temperature=0.3 for consistency, seed for reproducibility

#### Step 4: Multi-dimensional LLM Evaluation
Using a separate LLM call (GPT-4) as evaluator, score each summary on:
- **Faithfulness** (1-5): No hallucinated or incorrect information
- **Completeness** (1-5): Captures all key information from source
- **Conciseness** (1-5): Appropriately brief without unnecessary content
- **Clarity** (1-5): Easy to understand, well-organized
- **Overall Quality** (1-5): Holistic assessment

#### Step 5: Pairwise Preference Comparison
For a subset (n=50), run pairwise comparisons:
- Progressive vs. Flat summary
- Structured vs. Prose
- Conversational vs. Formal
Using LLM-as-judge to select preferred option with explanation

#### Step 6: Statistical Analysis
- Aggregate scores by strategy
- Run statistical tests (ANOVA, post-hoc Tukey HSD)
- Calculate effect sizes
- Analyze interaction between document type and strategy effectiveness

### Baselines

1. **Human Reference**: Gold-standard summaries from FeedSum
2. **Zero-shot LLM**: Unstructured default summary
3. **FeedSum Feedback Scores**: Use pre-computed faithfulness/completeness/conciseness scores where available

### Evaluation Metrics

**Primary Metrics (LLM-as-Judge):**
- Faithfulness Score (1-5)
- Completeness Score (1-5)
- Conciseness Score (1-5)
- Clarity Score (1-5)
- Overall Quality Score (1-5)

**Secondary Metrics (Automated):**
- ROUGE-1, ROUGE-2, ROUGE-L (vs human reference)
- BERTScore F1

**Preference Metrics:**
- Pairwise win rate for each strategy
- Bradley-Terry ranking model coefficients

### Statistical Analysis Plan

1. **One-way ANOVA** for each metric across 7 strategies
2. **Post-hoc Tukey HSD** for pairwise comparisons
3. **Effect size (Cohen's d)** for meaningful differences
4. **Pearson correlation** between automated metrics and LLM-judge scores
5. **Chi-square test** for preference win rates
6. Significance level: p < 0.05, with Bonferroni correction for multiple comparisons

## Expected Outcomes

### If Hypotheses Supported:
- Structured formats (H1): Clarity and Overall scores 0.5+ points higher than prose
- Conciseness trade-off (H2): U-shaped curve in quality vs. length plot
- Progressive disclosure (H3): Higher Overall preference in pairwise comparisons (>60%)
- Style context-dependence (H4): Conversational wins for informal content, formal wins for technical

### If Hypotheses Refuted:
- No significant difference between strategies would suggest the problem lies elsewhere (e.g., content selection, not presentation)
- Results would still be valuable for establishing that format/structure have minimal impact

## Timeline and Milestones

| Phase | Tasks | Estimated Duration |
|-------|-------|-------------------|
| Setup | Environment, data loading, EDA | 15-20 min |
| Implementation | Prompts, API calls, evaluation pipeline | 60-90 min |
| Experimentation | Generate 700 summaries, run evaluations | 60-90 min |
| Analysis | Statistical tests, visualizations | 30-45 min |
| Documentation | REPORT.md, README.md | 20-30 min |

## Potential Challenges

1. **API Rate Limits**: Mitigate with exponential backoff and caching
2. **Cost Management**: Estimate ~$30-50 for 700 summaries + evaluations; acceptable
3. **LLM-Judge Reliability**: Validate with subset of human annotations or self-consistency
4. **Document Diversity**: Ensure FeedSum sample covers multiple domains

## Success Criteria

1. **Statistical Significance**: At least one strategy shows significant improvement (p < 0.05) over baseline
2. **Practical Significance**: Effect size (Cohen's d) > 0.3 for top strategies
3. **Consistency**: Results hold across different document types
4. **Actionable Insights**: Clear recommendations for AI-to-human communication design

## Resource Requirements

**APIs:**
- OpenAI API (GPT-4.1) for generation and evaluation
- OR OpenRouter API for model access

**Python Libraries:**
- datasets (HuggingFace)
- openai
- numpy, pandas
- scipy (statistics)
- matplotlib, seaborn (visualization)
- rouge-score, bert-score

**Data:**
- FeedSum test set (~1,400 samples, sample 100)
- Pre-downloaded in datasets/feedsum/

**Estimated Costs:**
- Generation: 700 summaries x ~500 tokens avg = 350K tokens input, ~100K output = ~$5
- Evaluation: 700 x 5 dimensions x ~200 tokens = ~$10
- Pairwise: 50 x 3 comparisons x ~300 tokens = ~$2
- Total estimated: ~$17-25

## File Structure

```
workspace/
├── planning.md                 # This file
├── src/
│   ├── config.py              # API keys, constants
│   ├── data_loader.py         # Load and sample FeedSum
│   ├── prompts.py             # All prompt templates
│   ├── generate_summaries.py  # Generate summaries with strategies
│   ├── evaluate.py            # LLM-as-judge evaluation
│   ├── analyze.py             # Statistical analysis
│   └── utils.py               # Helper functions
├── results/
│   ├── summaries/             # Generated summaries by strategy
│   ├── evaluations/           # Evaluation scores
│   ├── figures/               # Visualization outputs
│   └── metrics.json           # Aggregated results
├── REPORT.md                  # Final research report
└── README.md                  # Project overview
```
