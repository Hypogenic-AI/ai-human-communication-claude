"""
Configuration for AI-to-Human Communication experiments.
"""
import os
import random
import numpy as np

# Random seed for reproducibility
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

# API Configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model to use (via OpenRouter)
# Using GPT-4.1 for generation and evaluation
GENERATION_MODEL = "openai/gpt-4.1"
EVALUATION_MODEL = "openai/gpt-4.1"

# Experiment parameters
NUM_SAMPLES = 100  # Number of documents to sample
TEMPERATURE = 0.0  # Zero temperature for reproducibility
MAX_TOKENS_GENERATION = 1024
MAX_TOKENS_EVALUATION = 256

# Communication formats to test
FORMATS = [
    "dense_prose",       # Full paragraph summary
    "bullet_points",     # Key points as bullets
    "hierarchical",      # Headers with nested bullets
    "progressive_2level", # Summary + details
    "progressive_3level"  # TL;DR + Key points + Details
]

# Evaluation dimensions
EVAL_DIMENSIONS = [
    "faithfulness",
    "completeness",
    "conciseness",
    "readability"
]

# Paths
DATA_PATH = "datasets/feedsum/data"
RESULTS_PATH = "results"
FIGURES_PATH = "figures"
