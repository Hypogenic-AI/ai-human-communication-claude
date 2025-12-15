# Cloned Repositories

This directory contains code repositories relevant to AI-to-Human Communication research, specifically for summarization training and evaluation.

---

## Repository 1: SumLLM

### Overview
- **URL**: https://github.com/yixinL7/SumLLM
- **Purpose**: Training summarization models with LLM references
- **Paper**: "On Learning to Summarize with Large Language Models as References"
- **Location**: `code/SumLLM/`

### Key Files
- `main.py` - Main training script
- `main_mle.py` - MLE (maximum likelihood estimation) training
- `test.py` - Evaluation script
- `gpt_score.py` - GPT-based scoring
- `gpt_rank.py` - GPT-based ranking
- `model.py` - Model architecture
- `data_utils.py` - Data loading utilities
- `config.py` - Configuration
- `requirements.txt` - Dependencies

### Requirements
- Python 3.8
- PyTorch 1.12.1
- Transformers 4.21.2

### Usage
```bash
cd code/SumLLM

# Install dependencies
pip install -r requirements.txt

# Training (see README.md for full options)
python main.py --config config.py

# Evaluation
python test.py --model_path <path>

# GPT scoring
python gpt_score.py --input <summaries>
```

### Notes
- Useful for training custom summarization models
- GPT scoring/ranking can be used for evaluation
- CNN/DM data already organized in `cnndm/` directory

---

## Repository 2: llm-summarization-evaluation

### Overview
- **URL**: https://github.com/haythemtellili/llm-summarization-evaluation
- **Purpose**: LLM-based document summarization with quality evaluation
- **Location**: `code/llm-summarization-evaluation/`

### Key Files
- `src/` - Source code directory
- `service/` - API service code
- `requirements.txt` - Dependencies
- `setup.py` - Package installation

### Features
- Document summarization using LLMs
- Quality evaluation based on research paper methodology
- Packaged as installable Python module

### Usage
```bash
cd code/llm-summarization-evaluation

# Install
pip install -r requirements.txt
pip install -e .

# See src/ for implementation details
```

### Notes
- Good starting point for summarization pipeline
- Evaluation inspired by academic research
- Can be containerized (dockerfile included)

---

## Repository 3: saga-llm-evaluation

### Overview
- **URL**: https://github.com/Sagacify/saga-llm-evaluation
- **Purpose**: Comprehensive LLM evaluation metrics library
- **Location**: `code/saga-llm-evaluation/`

### Key Files
- `saga_llm_evaluation/` - Main library
- `notebooks/` - Example notebooks
- `tests/` - Test suite
- `docs/` - Documentation
- `pyproject.toml` - Poetry-based dependencies

### Metrics Provided

**Embedding-based Metrics**:
- BERTScore
- Sentence embeddings similarity

**Language Model-based Metrics**:
- Perplexity-based evaluation

**LLM-based Metrics**:
- **G-Eval**: Uses LLMs with chain-of-thought for NLG evaluation
  - Text summarization evaluation
  - Dialogue generation evaluation
- **SelfCheck-GPT**: Zero-shot fact-checking for hallucination detection

### Usage
```bash
cd code/saga-llm-evaluation

# Install with Poetry
poetry install

# Or with pip
pip install -e .

# See notebooks/ for examples
```

### Example
```python
from saga_llm_evaluation import GEval, SelfCheckGPT

# G-Eval for summarization quality
evaluator = GEval(task="summarization")
score = evaluator.evaluate(document, summary)

# SelfCheck for hallucination detection
checker = SelfCheckGPT()
hallucination_score = checker.check(response, context)
```

### Notes
- Most comprehensive evaluation toolkit
- G-Eval is highly recommended for summarization evaluation
- SelfCheck useful for detecting hallucinations
- Well-documented with notebooks

---

## Other Relevant Repositories (Not Cloned)

### DeepEval
- **URL**: https://github.com/confident-ai/deepeval
- **Purpose**: LLM evaluation framework (Pytest-style)
- **Features**: G-Eval, hallucination detection, answer relevancy, RAGAS
- **Install**: `pip install deepeval`

### RAGAS
- **URL**: https://github.com/explodinggradients/ragas
- **Purpose**: RAG application evaluation
- **Features**: Automatic test data generation, LLM-based metrics
- **Install**: `pip install ragas`

### Ariadne
- **URL**: https://github.com/athina-ai/ariadne
- **Purpose**: Summarization and RAG evaluation
- **Features**: Hallucination detection, contradiction detection
- **Install**: `pip install ariadne-eval`

---

## Recommended Usage Order

### For Training Summarizers
1. Use **SumLLM** for model training with LLM references
2. Train on FeedSum dataset for multi-dimensional optimization

### For Evaluation
1. Use **saga-llm-evaluation** for G-Eval and comprehensive metrics
2. Use **llm-summarization-evaluation** for document-level evaluation
3. Compare with ROUGE/BERTScore baselines

### For Research Experiments
1. Adapt SumLLM training for custom feedback dimensions
2. Use saga-llm-evaluation metrics for automated evaluation
3. Combine with human evaluation for validation

---

## Directory Structure

```
code/
├── README.md                        # This file
├── SumLLM/                          # Summarization training
│   ├── main.py
│   ├── gpt_score.py
│   └── ...
├── llm-summarization-evaluation/    # Summarization + eval
│   ├── src/
│   ├── service/
│   └── ...
└── saga-llm-evaluation/             # Evaluation metrics
    ├── saga_llm_evaluation/
    ├── notebooks/
    └── ...
```
