# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project on AI-to-Human Communication. The research hypothesis is that AI and humans consume information differently - AI benefits from dense comprehensive input while humans prefer concise communication. These resources support experiments to identify and test methods that make AI-to-human communication more effective.

**Total resources gathered:**
- Papers downloaded: 11
- Datasets downloaded: 3
- Repositories cloned: 3

---

## Papers

Total papers downloaded: **11**

| # | Title | Authors | Year | File | Key Focus |
|---|-------|---------|------|------|-----------|
| 1 | Progressive Disclosure: Designing for Effective Transparency | Springer, Whittaker | 2018 | papers/1811.02164_Progressive_Disclosure_Transparency.pdf | User-centric transparency, progressive reveal |
| 2 | Task Supportive Human-LLM Interaction | Wang et al. | 2024 | papers/2402.06170_Task_Supportive_Human_LLM.pdf | Cognitive load reduction, prompt support |
| 3 | Survey on Automatic Text Summarization with LLM | Zhang et al. | 2024 | papers/2403.02901_Text_Summarization_LLM_Survey.pdf | Comprehensive ATS survey |
| 4 | Quality Evaluation Methods for Text Summarization | Nguyen et al. | 2024 | papers/2407.00747_Quality_Eval_Summarization.pdf | LLM vs traditional eval metrics |
| 5 | CogErgLLM: Cognitive Ergonomics | Wasi, Islam | 2024 | papers/2407.02885_CogErgLLM_Cognitive_Ergonomics.pdf | Cognitive ergonomics framework |
| 6 | Agentic Information Retrieval | Zhang et al. | 2024 | papers/2410.09713_Agentic_Information_Retrieval.pdf | Dynamic IR with LLM agents |
| 7 | FeedSum: LLM-generated Feedback for Summarization | Song et al. | 2024 | papers/2410.13116_FeedSum_LLM_Summary_Feedback.pdf | Multi-dimensional summary feedback |
| 8 | CARE: Collaborative AI Research Environment | Lalor et al. | 2024 | papers/2410.24032_CARE_Collaborative_Assistant.pdf | Human-AI research collaboration |
| 9 | Reference-Free Evaluation Metrics | - | 2025 | papers/2501.12011_Reference_Free_Eval_Metrics.pdf | Evaluation without references |
| 10 | Cognitive Load in Streaming LLM Output | - | 2025 | papers/2504.17999_Cognitive_Load_Streaming_LLM.pdf | Cognitive load during streaming |
| 11 | Generative Interfaces for LLM Systems | - | 2025 | papers/2508.19227_Generative_Interfaces_LLM.pdf | LLM interface design patterns |

See `papers/README.md` for detailed paper descriptions.

---

## Datasets

Total datasets downloaded: **3**

| # | Name | Source | Size | Task | Location | Status |
|---|------|--------|------|------|----------|--------|
| 1 | FeedSum | HuggingFace (DISLab) | 125K train + 1.4K test | Summary quality feedback | datasets/feedsum/ | Downloaded |
| 2 | CNN/DailyMail | HuggingFace | 11.5K test samples | News summarization | datasets/cnn_dailymail/ | Test set downloaded |
| 3 | Human-Like-DPO | HuggingFace (HumanLLMs) | 10.9K samples | Human-like vs formal responses | datasets/human_like_dpo/ | Downloaded |

See `datasets/README.md` for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: **3**

| # | Name | URL | Purpose | Location |
|---|------|-----|---------|----------|
| 1 | SumLLM | github.com/yixinL7/SumLLM | Summarization with LLM references | code/SumLLM/ |
| 2 | llm-summarization-evaluation | github.com/haythemtellili/llm-summarization-evaluation | LLM summarization + evaluation | code/llm-summarization-evaluation/ |
| 3 | saga-llm-evaluation | github.com/Sagacify/saga-llm-evaluation | LLM evaluation metrics library | code/saga-llm-evaluation/ |

See `code/README.md` for detailed repository descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search**: Used arXiv, Semantic Scholar, and Google Scholar with keywords:
   - "AI human communication"
   - "LLM text summarization"
   - "Progressive disclosure transparency"
   - "Cognitive ergonomics LLM"
   - "Human-AI interaction"
   - "Summarization evaluation metrics"

2. **Dataset Search**: Searched HuggingFace Datasets with focus on:
   - Summarization datasets with quality feedback
   - Human preference data for LLM output
   - Dialogue/conversation datasets

3. **Code Search**: Searched GitHub for:
   - LLM summarization implementations
   - Evaluation metric libraries
   - Human preference optimization code

### Selection Criteria

- **Papers**: Prioritized recent work (2024-2025) on AI-to-human communication, with focus on cognitive load, transparency, summarization, and evaluation methods.
- **Datasets**: Selected datasets with human preference annotations or quality feedback, suitable for training/evaluating human-preferred communication.
- **Code**: Selected repositories with evaluation metrics and summarization training code that can be adapted for experiments.

### Challenges Encountered

1. **OpenAI summarize_from_feedback**: Dataset uses legacy HuggingFace format incompatible with current library version. Alternative: OpenAI provides direct download from Azure blob storage.
2. **Paper availability**: Some papers from 2025 may have limited detail in abstracts; full reading may be needed during experimentation.

### Gaps and Workarounds

- **Gap**: No dedicated dataset for AI research output communication.
- **Workaround**: Can create synthetic dataset using FeedSum/CNN-DailyMail as base, or generate research-like outputs from existing summarization data.

---

## Recommendations for Experiment Design

### 1. Primary Datasets

**FeedSum** (Recommended):
- Multi-dimensional feedback (faithfulness, completeness, conciseness)
- Large scale (125K+ samples)
- Diverse domains (news, medical, dialogue)
- Direct alignment with human preference optimization

**CNN/DailyMail** (Secondary):
- Standard benchmark for comparison
- News domain (structured information)

**Human-Like-DPO** (For style optimization):
- Comparison between formal and conversational responses
- Can inform communication style experiments

### 2. Recommended Baselines

From literature and code:
- Zero-shot LLM summarization (GPT-4, Claude)
- Fine-tuned summarization models (BART, T5, Pegasus)
- SummLlama models (trained on FeedSum)

### 3. Evaluation Metrics

**Automated**:
- ROUGE-1/2/L (lexical overlap)
- BERTScore (semantic similarity)
- G-Eval (LLM-based evaluation)
- Custom multi-dimensional metrics (faithfulness, completeness, conciseness)

**Human Evaluation**:
- Comprehension accuracy (quiz-based)
- Time to understanding
- Self-reported cognitive load (Likert scale)
- Preference rankings

### 4. Experimental Approaches to Test

Based on literature review:

1. **Progressive Disclosure**: Compare full output vs. hierarchical reveal
2. **Structured Formatting**: Test bullet points, headers, sections vs. prose
3. **Adaptive Length**: Short summary vs. detailed explanation vs. user-controlled
4. **Multi-Dimensional Optimization**: Train for faithfulness+completeness+conciseness balance
5. **Interactive Elements**: Allow users to request clarification/expansion

### 5. Code to Adapt/Reuse

- **saga-llm-evaluation**: For comprehensive evaluation metrics
- **SumLLM**: For training summarization models with LLM references
- **llm-summarization-evaluation**: For document summarization pipeline

---

## Quick Start Guide

### Loading Datasets

```python
from datasets import load_from_disk

# Load FeedSum (largest, most relevant)
feedsum = load_from_disk("datasets/feedsum/data")
print(f"FeedSum: {feedsum}")

# Load CNN/DailyMail test set
cnn_dm = load_from_disk("datasets/cnn_dailymail/test")
print(f"CNN/DM: {cnn_dm}")

# Load Human-Like-DPO
human_like = load_from_disk("datasets/human_like_dpo/data")
print(f"Human-Like: {human_like}")
```

### Using Evaluation Metrics

```python
# From saga-llm-evaluation (see code/saga-llm-evaluation/README.md)
# Provides G-Eval, SelfCheck-GPT, and other LLM-based metrics

# From SumLLM (see code/SumLLM/README.md)
# Provides GPT scoring and ranking utilities
```

---

## File Structure

```
workspace/
├── papers/                          # 11 PDF papers
│   └── README.md                    # Paper descriptions
├── datasets/                        # 3 datasets
│   ├── .gitignore                   # Exclude data from git
│   ├── README.md                    # Dataset descriptions + download instructions
│   ├── feedsum/
│   │   ├── data/                    # HuggingFace dataset
│   │   └── samples.json             # 5 sample records
│   ├── cnn_dailymail/
│   │   ├── test/                    # Test split only
│   │   └── samples.json             # 5 sample records
│   └── human_like_dpo/
│       ├── data/                    # HuggingFace dataset
│       └── samples.json             # 5 sample records
├── code/                            # 3 cloned repositories
│   ├── README.md                    # Repository descriptions
│   ├── SumLLM/
│   ├── llm-summarization-evaluation/
│   └── saga-llm-evaluation/
├── literature_review.md             # Comprehensive literature synthesis
├── resources.md                     # This file
└── .resource_finder_complete        # Completion marker
```
