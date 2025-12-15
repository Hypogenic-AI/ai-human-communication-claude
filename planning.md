# Research Plan: AI-to-Human Communication Methods

## Research Question

**How can AI systems communicate large volumes of information to humans more effectively?**

Specifically, we investigate: Given the same source content, which communication strategies (dense comprehensive vs. hierarchical/structured vs. adaptive) best enable humans to understand and trust AI-generated information?

## Background and Motivation

AI systems like research agents can process thousands of papers, code repositories, and datasets, but their outputs are often overwhelming for humans. A 100-page research report with thousands of lines of code is impractical for human consumption. This creates a fundamental asymmetry:

- **AI input preference**: Dense, comprehensive information
- **Human processing capacity**: Limited cognitive bandwidth, preference for concise, well-structured information

The gap between AI capability and human comprehension threatens the utility of AI systems. Without effective AI-to-human communication:
1. Humans cannot verify AI work (trust issue)
2. Valuable AI insights are missed (utility issue)
3. Collaboration between humans and AI breaks down (workflow issue)

### Key Insights from Literature Review

From the 11 papers reviewed:

1. **Progressive Disclosure** (Springer & Whittaker, 2018): Simple initial feedback helps users build mental models before complexity
2. **Cognitive Ergonomics** (CogErgLLM, 2024): Optimize for mental workload, guide attention, facilitate learning
3. **Multi-Dimensional Quality** (FeedSum, 2024): Three key dimensions - faithfulness, completeness, conciseness
4. **LLM-as-Evaluator** (Nguyen et al., 2024): LLM evaluation aligns better with human judgment than ROUGE/BERTScore

## Hypothesis Decomposition

**Main Hypothesis**: Different communication strategies vary in effectiveness for human comprehension, and the optimal strategy depends on the tradeoff between information density and cognitive load.

### Sub-hypotheses:

**H1**: Structured summaries (with hierarchical sections and bullet points) will be preferred over and perform better than dense prose summaries.

**H2**: Progressive disclosure (summary → details on demand) will enable faster comprehension than providing all information upfront.

**H3**: Extreme summarization sacrifices too much information; optimal summaries balance completeness with conciseness.

**H4**: LLM-as-judge evaluation correlates with the three quality dimensions (faithfulness, completeness, conciseness) from FeedSum.

## Proposed Methodology

### Approach

We will conduct an **automated experimental study** comparing different AI-to-human communication strategies using:
1. Real documents from FeedSum dataset (diverse domains)
2. Real LLM APIs (Claude, GPT-4) to generate different communication formats
3. LLM-as-judge evaluation for automated quality assessment
4. Multi-dimensional metrics aligned with human preferences

### Why This Approach?

- **Real LLMs, not simulations**: Ensures findings generalize to actual AI systems
- **Automated evaluation**: Scalable, reproducible, and (per literature) well-correlated with human judgment
- **Multi-dimensional metrics**: Captures tradeoffs between completeness and conciseness
- **Diverse test set**: FeedSum covers 7 domains (news, medical, dialogue, etc.)

### Experimental Steps

#### Experiment 1: Communication Format Comparison

**Goal**: Compare different output formats for the same source content

**Conditions** (4 formats):
1. **Dense Prose**: Full paragraph summary (baseline)
2. **Bullet Points**: Key points as bulleted list
3. **Hierarchical**: Headers with nested bullet points
4. **Progressive**: One-line summary + expandable details

**Procedure**:
1. Sample 100 diverse documents from FeedSum test set
2. Generate all 4 formats using Claude/GPT-4 with controlled prompts
3. Evaluate each format using LLM-as-judge on:
   - Faithfulness (no hallucination)
   - Completeness (key info preserved)
   - Conciseness (no unnecessary verbosity)
   - Readability (ease of comprehension)
4. Compute preference rankings

#### Experiment 2: Length vs. Quality Tradeoff

**Goal**: Find optimal compression ratio for different use cases

**Conditions** (5 length targets):
- ~25 words (extreme summary)
- ~50 words (short summary)
- ~100 words (medium summary)
- ~200 words (detailed summary)
- ~400 words (comprehensive summary)

**Procedure**:
1. Sample 100 documents from FeedSum
2. Generate summaries at each length target
3. Evaluate faithfulness, completeness, conciseness at each length
4. Analyze tradeoff curves

#### Experiment 3: Progressive Disclosure Simulation

**Goal**: Test if progressive revelation improves information access

**Conditions**:
1. **Flat**: All information presented at once
2. **Two-level**: Summary + one level of detail
3. **Three-level**: Summary → subsections → full details

**Procedure**:
1. Sample 50 longer documents (>500 words)
2. Generate hierarchical summaries at 3 granularity levels
3. Evaluate whether each level maintains faithfulness
4. Measure information preservation at each level

### Baselines

1. **Human Reference**: FeedSum provides human-written summaries
2. **Existing Summarizers**: FeedSum includes outputs from BART, T5, GPT-4-turbo, Mistral
3. **Simple Truncation**: First N words/sentences of source

### Evaluation Metrics

#### Primary Metrics (LLM-as-Judge)

1. **Faithfulness Score** (0-1): Does the summary accurately reflect the source?
   - Prompt: "Rate how faithfully this summary represents the source, with no added or contradicting information"

2. **Completeness Score** (0-1): Are key points included?
   - Prompt: "Rate how completely this summary captures the essential information"

3. **Conciseness Score** (0-1): Is it appropriately brief?
   - Prompt: "Rate how concise this summary is - no redundant or unnecessary information"

4. **Readability Score** (0-1): Is it easy to comprehend?
   - Prompt: "Rate how readable and well-structured this summary is for human comprehension"

#### Secondary Metrics

- Length (word count)
- Compression ratio (summary length / source length)
- Key fact coverage (using extracted_keyfacts from FeedSum)

### Statistical Analysis Plan

1. **Comparison tests**: Paired t-tests or Wilcoxon signed-rank tests for within-document comparisons
2. **Effect sizes**: Cohen's d for practical significance
3. **Correlation analysis**: Pearson/Spearman for metric relationships
4. **Multiple comparison correction**: Bonferroni correction for multiple conditions
5. **Significance level**: α = 0.05

## Expected Outcomes

### If H1 is supported:
- Structured formats (bullets, hierarchical) will have higher readability and similar/better completeness than dense prose
- Users prefer structured output over paragraphs

### If H2 is supported:
- Progressive disclosure maintains high-level understanding while preserving access to details
- Multi-level summaries enable efficient information triage

### If H3 is supported:
- There exists an optimal length range (likely 50-150 words) balancing completeness and conciseness
- Extreme summarization (<25 words) significantly loses faithfulness/completeness

### If H4 is supported:
- LLM-as-judge scores will show strong correlation with FeedSum's multi-dimensional feedback
- Validates using LLM evaluation for AI communication quality

## Timeline and Milestones

| Phase | Tasks | Estimated Duration |
|-------|-------|-------------------|
| Setup | Environment, data loading, API configuration | Complete |
| Planning | This document | Complete |
| Experiment 1 | Format comparison | ~60 min |
| Experiment 2 | Length tradeoffs | ~60 min |
| Experiment 3 | Progressive disclosure | ~45 min |
| Analysis | Statistics, visualizations | ~30 min |
| Documentation | REPORT.md, figures | ~30 min |

## Potential Challenges

1. **API Rate Limits**: Mitigate by batching requests, caching responses
2. **Evaluation Variability**: Run multiple evaluation passes, report variance
3. **Domain Differences**: Analyze results per-domain in FeedSum
4. **Prompt Sensitivity**: Test multiple prompt variants, document final prompts

## Success Criteria

1. **Minimum**: Complete all 3 experiments with statistical analysis
2. **Target**: Clear evidence for/against each hypothesis with effect sizes
3. **Stretch**: Actionable guidelines for AI-to-human communication design

## Resource Requirements

- **Compute**: CPU sufficient (API-based)
- **API Budget**: ~$50-100 for ~5000-10000 API calls
- **Storage**: <1GB for results and cached responses
- **Time**: ~4-5 hours total execution

## Reproducibility

- Random seed: 42 for all sampling
- All prompts documented in code
- API model versions recorded (claude-sonnet-4-20250514, gpt-4.1)
- Results saved as JSON with timestamps
