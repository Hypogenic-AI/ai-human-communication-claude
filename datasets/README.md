# Downloaded Datasets

This directory contains datasets for the AI-to-Human Communication research project.
**Data files are NOT committed to git** due to size. Follow the download instructions below.

---

## Dataset 1: FeedSum

### Overview
- **Source**: [HuggingFace: DISLab/FeedSum](https://huggingface.co/datasets/DISLab/FeedSum)
- **Size**: 125,388 train + 1,400 test samples (~164 MB local)
- **Format**: HuggingFace Dataset
- **Task**: Summarization with multi-dimensional quality feedback
- **License**: Research use
- **Paper**: [Learning to Summarize from LLM-generated Feedback](https://arxiv.org/abs/2410.13116)

### Features
- `doc_id`: Document identifier
- `source`: Source dataset (CNN/DM, XSum, etc.)
- `document`: Full input document
- `summarizer`: Model that generated the summary
- `summary`: Generated summary text
- `feedback-c4`: Fine-grained feedback scores
  - `faithfulness_score` (0-1): Summary is factually accurate
  - `completeness_score` (0-1): Summary includes key information
  - `conciseness_score` (0-1): Summary is appropriately brief
- `feedback-c3`, `feedback-c2`, `feedback-c1`: Alternative feedback formats
- `human_reference`: Human-written reference summary
- `extracted_keyfacts`: List of key facts from document

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# Download and save
dataset = load_dataset("DISLab/FeedSum")
dataset.save_to_disk("datasets/feedsum/data")
```

**Loading the dataset (after download):**
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/feedsum/data")
print(f"Train: {len(dataset['train'])} samples")
print(f"Test: {len(dataset['test'])} samples")

# Access a sample
sample = dataset['train'][0]
print(f"Document: {sample['document'][:200]}...")
print(f"Summary: {sample['summary']}")
print(f"Faithfulness: {sample['feedback-c4']['faithfulness_score']}")
```

### Sample Data
See `feedsum/samples.json` for example records.

### Notes
- Multi-dimensional feedback is key for training human-preferred summarizers
- Covers 7 domains: News, Lifestyle, Report, Medical, Daily Life, Interview, Meeting
- feedback-c4 provides percentage scores (0-1 scale)

---

## Dataset 2: CNN/DailyMail

### Overview
- **Source**: [HuggingFace: cnn_dailymail](https://huggingface.co/datasets/cnn_dailymail)
- **Size**: 11,490 test samples (~30 MB local, test split only)
- **Format**: HuggingFace Dataset
- **Task**: News article summarization
- **License**: Apache 2.0
- **Full dataset**: 287K train, 13K validation, 11.5K test

### Features
- `article`: Full news article text (~781 tokens average)
- `highlights`: Human-written summary (~56 tokens average, 3.75 sentences)
- `id`: Article identifier

### Download Instructions

**Download test set only (recommended for evaluation):**
```python
from datasets import load_dataset

# Download test split
dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
dataset.save_to_disk("datasets/cnn_dailymail/test")
```

**Download full dataset:**
```python
from datasets import load_dataset

# Download all splits
dataset = load_dataset("cnn_dailymail", "3.0.0")
dataset.save_to_disk("datasets/cnn_dailymail/full")
```

**Loading the dataset (after download):**
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/cnn_dailymail/test")
print(f"Test samples: {len(dataset)}")

# Access a sample
sample = dataset[0]
print(f"Article: {sample['article'][:300]}...")
print(f"Summary: {sample['highlights']}")
```

### Sample Data
See `cnn_dailymail/samples.json` for example records.

### Notes
- Standard benchmark dataset for news summarization
- Good for comparison with published results
- Articles are relatively long (~780 words)

---

## Dataset 3: Human-Like-DPO

### Overview
- **Source**: [HuggingFace: HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset)
- **Size**: 10,884 samples (~15 MB local)
- **Format**: HuggingFace Dataset
- **Task**: DPO training for human-like vs formal responses
- **License**: Research use

### Features
- `prompt`: Conversational question (natural dialogue style)
- `chosen`: Human-like response (natural, conversational)
- `rejected`: Formal response (structured, professional AI style)

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
dataset.save_to_disk("datasets/human_like_dpo/data")
```

**Loading the dataset (after download):**
```python
from datasets import load_from_disk

dataset = load_from_disk("datasets/human_like_dpo/data")
print(f"Samples: {len(dataset['train'])}")

# Access a sample
sample = dataset['train'][0]
print(f"Prompt: {sample['prompt']}")
print(f"Chosen (human-like): {sample['chosen']}")
print(f"Rejected (formal): {sample['rejected']}")
```

### Sample Data
See `human_like_dpo/samples.json` for example records.

### Notes
- 256 topics covered
- Useful for training models to communicate in more natural/human-like style
- Can inform experiments on formal vs conversational AI communication

---

## Additional Datasets (Not Downloaded)

These datasets may be useful but were not downloaded:

### OpenAI Summarize-from-Feedback
- **Source**: Azure Blob Storage (legacy format)
- **Description**: RLHF training data for summarization
- **Download**: See [OpenAI's repository](https://github.com/openai/summarize-from-feedback)

### XSum (Extreme Summarization)
- **Source**: [HuggingFace: xsum](https://huggingface.co/datasets/xsum)
- **Description**: One-sentence summaries of BBC articles
- **Use**: Alternative summarization benchmark

### SamSum (Dialogue Summarization)
- **Source**: [HuggingFace: samsum](https://huggingface.co/datasets/samsum)
- **Description**: Messenger-like conversation summaries
- **Use**: Dialogue summarization experiments

---

## Directory Structure

```
datasets/
├── .gitignore          # Excludes data files from git
├── README.md           # This file
├── feedsum/
│   ├── data/           # HuggingFace dataset (125K+ samples)
│   └── samples.json    # 5 example records
├── cnn_dailymail/
│   ├── test/           # Test split only (11.5K samples)
│   └── samples.json    # 5 example records
└── human_like_dpo/
    ├── data/           # HuggingFace dataset (10.9K samples)
    └── samples.json    # 5 example records
```

---

## Quick Start

```python
from datasets import load_from_disk

# Load all datasets
feedsum = load_from_disk("datasets/feedsum/data")
cnn_dm = load_from_disk("datasets/cnn_dailymail/test")
human_like = load_from_disk("datasets/human_like_dpo/data")

print("Datasets loaded:")
print(f"  FeedSum: {feedsum}")
print(f"  CNN/DM: {cnn_dm}")
print(f"  Human-Like: {human_like}")
```
