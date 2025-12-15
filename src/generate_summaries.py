"""
Generate summaries using different communication strategies via real LLM APIs.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts import STRATEGIES, get_prompt


# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default to GPT-4.1 via OpenAI or OpenRouter
MODEL = "gpt-4.1"  # Can also use "openai/gpt-4.1" for OpenRouter
TEMPERATURE = 0.3
MAX_TOKENS = 1000


def get_client():
    """Get OpenAI client configured for OpenRouter or direct OpenAI."""
    if OPENROUTER_API_KEY:
        # Use OpenRouter
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        return client, "openai/gpt-4.1"  # OpenRouter model format
    elif OPENAI_API_KEY:
        # Use OpenAI directly
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return client, "gpt-4.1"
    else:
        raise ValueError("No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
def call_llm(client, model: str, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
    """Call LLM with retry logic."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def truncate_document(document: str, max_chars: int = 8000) -> str:
    """Truncate long documents to fit within context limits."""
    if len(document) > max_chars:
        return document[:max_chars] + "... [truncated]"
    return document


def generate_summary(client, model: str, strategy_key: str, document: str) -> dict:
    """Generate a summary using specified strategy."""
    # Truncate if needed
    truncated_doc = truncate_document(document)

    # Get prompt for strategy
    prompt = get_prompt(strategy_key, truncated_doc)

    # Call LLM
    start_time = time.time()
    summary = call_llm(client, model, prompt)
    elapsed = time.time() - start_time

    return {
        "strategy": strategy_key,
        "summary": summary.strip(),
        "word_count": len(summary.split()),
        "char_count": len(summary),
        "generation_time": elapsed,
        "truncated": len(document) > 8000
    }


def generate_all_summaries(samples: list, output_dir: str = "results/summaries") -> dict:
    """Generate summaries for all samples using all strategies."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client, model = get_client()
    print(f"Using model: {model}")

    results = {
        "model": model,
        "temperature": TEMPERATURE,
        "strategies": list(STRATEGIES.keys()),
        "summaries": []
    }

    total_docs = len(samples)
    total_strategies = len(STRATEGIES)
    total_calls = total_docs * total_strategies

    print(f"Generating {total_calls} summaries ({total_docs} documents x {total_strategies} strategies)")

    for doc_idx, sample in enumerate(samples):
        doc_id = sample.get("doc_id", f"doc_{doc_idx}")
        document = sample.get("document", "")
        source = sample.get("source", "unknown")
        human_ref = sample.get("human_reference", "")

        doc_summaries = {
            "doc_id": doc_id,
            "source": source,
            "document_length": len(document),
            "human_reference": human_ref,
            "strategies": {}
        }

        for strategy_key in STRATEGIES.keys():
            try:
                result = generate_summary(client, model, strategy_key, document)
                doc_summaries["strategies"][strategy_key] = result
                print(f"  [{doc_idx+1}/{total_docs}] {strategy_key}: {result['word_count']} words")
            except Exception as e:
                print(f"  [{doc_idx+1}/{total_docs}] {strategy_key}: ERROR - {str(e)}")
                doc_summaries["strategies"][strategy_key] = {
                    "strategy": strategy_key,
                    "summary": None,
                    "error": str(e)
                }

        results["summaries"].append(doc_summaries)

        # Save intermediate results every 10 documents
        if (doc_idx + 1) % 10 == 0:
            with open(f"{output_dir}/summaries_partial.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved intermediate results ({doc_idx+1}/{total_docs})")

    # Save final results
    output_path = f"{output_dir}/all_summaries.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved all summaries to {output_path}")
    return results


def load_summaries(input_path: str = "results/summaries/all_summaries.json") -> dict:
    """Load previously generated summaries."""
    with open(input_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_samples

    # Load sampled documents
    print("Loading sampled documents...")
    samples = load_samples("results/sampled_documents.json")
    print(f"Loaded {len(samples)} documents")

    # Generate summaries (subset for testing)
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    if test_mode:
        print("TEST MODE: Processing only first 5 documents")
        samples = samples[:5]

    results = generate_all_summaries(samples)

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for strategy_key in STRATEGIES.keys():
        word_counts = []
        for doc in results["summaries"]:
            strat_result = doc["strategies"].get(strategy_key, {})
            if strat_result.get("word_count"):
                word_counts.append(strat_result["word_count"])

        if word_counts:
            avg_words = sum(word_counts) / len(word_counts)
            print(f"{strategy_key}: avg {avg_words:.1f} words ({len(word_counts)} generated)")
