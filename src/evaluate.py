"""
LLM-as-Judge evaluation for summary quality.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts import get_evaluation_prompt, get_pairwise_prompt


# Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EVAL_MODEL = "gpt-4.1"
TEMPERATURE = 0.1  # Lower temperature for evaluation


def get_client():
    """Get OpenAI client configured for OpenRouter or direct OpenAI."""
    if OPENROUTER_API_KEY:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )
        return client, "openai/gpt-4.1"
    elif OPENAI_API_KEY:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        return client, "gpt-4.1"
    else:
        raise ValueError("No API key found")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=60))
def call_llm(client, model: str, prompt: str, max_tokens: int = 500) -> str:
    """Call LLM with retry logic."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content


def parse_evaluation_response(response: str) -> dict:
    """Parse JSON evaluation response from LLM."""
    # Try to extract JSON from response
    try:
        # Look for JSON block
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # Return default if parsing fails
    return {
        "faithfulness": {"score": None, "justification": "Parse error"},
        "completeness": {"score": None, "justification": "Parse error"},
        "conciseness": {"score": None, "justification": "Parse error"},
        "clarity": {"score": None, "justification": "Parse error"},
        "overall": {"score": None, "justification": "Parse error"},
        "raw_response": response
    }


def parse_pairwise_response(response: str) -> dict:
    """Parse JSON pairwise comparison response."""
    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    return {"winner": None, "reason": "Parse error", "raw_response": response}


def truncate_for_eval(text: str, max_chars: int = 4000) -> str:
    """Truncate text for evaluation context."""
    if len(text) > max_chars:
        return text[:max_chars] + "... [truncated]"
    return text


def evaluate_summary(client, model: str, document: str, summary: str) -> dict:
    """Evaluate a single summary using LLM-as-judge."""
    # Truncate for context limits
    doc_truncated = truncate_for_eval(document)
    sum_truncated = truncate_for_eval(summary, max_chars=2000)

    prompt = get_evaluation_prompt(doc_truncated, sum_truncated)

    start_time = time.time()
    response = call_llm(client, model, prompt, max_tokens=600)
    elapsed = time.time() - start_time

    result = parse_evaluation_response(response)
    result["evaluation_time"] = elapsed

    return result


def compare_summaries(client, model: str, document: str, summary_a: str, summary_b: str) -> dict:
    """Pairwise comparison of two summaries."""
    doc_truncated = truncate_for_eval(document)
    sum_a_truncated = truncate_for_eval(summary_a, max_chars=1500)
    sum_b_truncated = truncate_for_eval(summary_b, max_chars=1500)

    prompt = get_pairwise_prompt(doc_truncated, sum_a_truncated, sum_b_truncated)

    start_time = time.time()
    response = call_llm(client, model, prompt, max_tokens=300)
    elapsed = time.time() - start_time

    result = parse_pairwise_response(response)
    result["comparison_time"] = elapsed

    return result


def evaluate_all_summaries(summaries_data: dict, output_dir: str = "results/evaluations") -> dict:
    """Evaluate all generated summaries."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client, model = get_client()
    print(f"Using model for evaluation: {model}")

    results = {
        "eval_model": model,
        "temperature": TEMPERATURE,
        "evaluations": []
    }

    total_docs = len(summaries_data["summaries"])
    strategies = summaries_data["strategies"]
    total_evals = total_docs * len(strategies)

    print(f"Evaluating {total_evals} summaries ({total_docs} docs x {len(strategies)} strategies)")

    for doc_idx, doc_data in enumerate(summaries_data["summaries"]):
        doc_id = doc_data["doc_id"]

        # Need original document for evaluation
        # Load from samples
        samples = load_samples_for_eval()
        original_doc = None
        for s in samples:
            if s.get("doc_id") == doc_id:
                original_doc = s.get("document", "")
                break

        if not original_doc:
            print(f"Warning: Could not find original document for {doc_id}")
            continue

        doc_evals = {
            "doc_id": doc_id,
            "source": doc_data.get("source"),
            "strategies": {}
        }

        for strategy_key, strat_data in doc_data["strategies"].items():
            summary = strat_data.get("summary")
            if not summary:
                doc_evals["strategies"][strategy_key] = {"error": "No summary"}
                continue

            try:
                eval_result = evaluate_summary(client, model, original_doc, summary)
                doc_evals["strategies"][strategy_key] = eval_result

                # Extract scores for logging
                scores = {
                    "f": eval_result.get("faithfulness", {}).get("score"),
                    "co": eval_result.get("completeness", {}).get("score"),
                    "cn": eval_result.get("conciseness", {}).get("score"),
                    "cl": eval_result.get("clarity", {}).get("score"),
                    "o": eval_result.get("overall", {}).get("score")
                }
                print(f"  [{doc_idx+1}/{total_docs}] {strategy_key}: {scores}")

            except Exception as e:
                print(f"  [{doc_idx+1}/{total_docs}] {strategy_key}: ERROR - {str(e)}")
                doc_evals["strategies"][strategy_key] = {"error": str(e)}

        results["evaluations"].append(doc_evals)

        # Save intermediate every 10 docs
        if (doc_idx + 1) % 10 == 0:
            with open(f"{output_dir}/evaluations_partial.json", "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved intermediate evaluations ({doc_idx+1}/{total_docs})")

    # Save final
    output_path = f"{output_dir}/all_evaluations.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved all evaluations to {output_path}")
    return results


def run_pairwise_comparisons(summaries_data: dict,
                             comparison_pairs: list,
                             n_samples: int = 50,
                             output_dir: str = "results/evaluations") -> dict:
    """
    Run pairwise comparisons between specified strategy pairs.

    Args:
        summaries_data: Generated summaries
        comparison_pairs: List of tuples (strategy_a, strategy_b)
        n_samples: Number of documents to compare
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    client, model = get_client()
    print(f"Using model for comparisons: {model}")

    results = {
        "eval_model": model,
        "comparison_pairs": comparison_pairs,
        "comparisons": []
    }

    # Sample documents
    docs = summaries_data["summaries"][:n_samples]
    samples = load_samples_for_eval()

    for doc_idx, doc_data in enumerate(docs):
        doc_id = doc_data["doc_id"]

        # Find original document
        original_doc = None
        for s in samples:
            if s.get("doc_id") == doc_id:
                original_doc = s.get("document", "")
                break

        if not original_doc:
            continue

        doc_comparisons = {"doc_id": doc_id, "pairs": {}}

        for strategy_a, strategy_b in comparison_pairs:
            summary_a = doc_data["strategies"].get(strategy_a, {}).get("summary")
            summary_b = doc_data["strategies"].get(strategy_b, {}).get("summary")

            if not summary_a or not summary_b:
                doc_comparisons["pairs"][f"{strategy_a}_vs_{strategy_b}"] = {"error": "Missing summary"}
                continue

            try:
                result = compare_summaries(client, model, original_doc, summary_a, summary_b)
                doc_comparisons["pairs"][f"{strategy_a}_vs_{strategy_b}"] = result
                print(f"  [{doc_idx+1}/{n_samples}] {strategy_a} vs {strategy_b}: {result.get('winner')}")
            except Exception as e:
                doc_comparisons["pairs"][f"{strategy_a}_vs_{strategy_b}"] = {"error": str(e)}

        results["comparisons"].append(doc_comparisons)

    # Save results
    output_path = f"{output_dir}/pairwise_comparisons.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved pairwise comparisons to {output_path}")
    return results


def load_samples_for_eval(path: str = "results/sampled_documents.json") -> list:
    """Load original samples for evaluation."""
    with open(path, "r") as f:
        return json.load(f)


def load_evaluations(path: str = "results/evaluations/all_evaluations.json") -> dict:
    """Load saved evaluations."""
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from generate_summaries import load_summaries

    # Load generated summaries
    print("Loading generated summaries...")
    summaries = load_summaries()
    print(f"Loaded summaries for {len(summaries['summaries'])} documents")

    # Check if we should run in test mode
    test_mode = os.getenv("TEST_MODE", "false").lower() == "true"
    if test_mode:
        print("TEST MODE: Evaluating only first 5 documents")
        summaries["summaries"] = summaries["summaries"][:5]

    # Evaluate all summaries
    evaluations = evaluate_all_summaries(summaries)

    # Run pairwise comparisons for key hypotheses
    print("\n" + "="*60)
    print("PAIRWISE COMPARISONS")
    print("="*60)

    comparison_pairs = [
        ("dense_prose", "bullet_points"),           # H1: Structure
        ("dense_prose", "progressive_hierarchy"),   # H3: Progressive
        ("formal_technical", "conversational")      # H4: Style
    ]

    pairwise_results = run_pairwise_comparisons(
        summaries,
        comparison_pairs,
        n_samples=min(50, len(summaries["summaries"]))
    )
