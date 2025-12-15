"""
LLM client for generation and evaluation using OpenRouter.
"""
import os
import json
import time
from typing import Optional
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get('OPENROUTER_API_KEY', '')
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30)
)
def call_llm(
    prompt: str,
    model: str = "openai/gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    system_prompt: Optional[str] = None
) -> str:
    """
    Call LLM via OpenRouter with retry logic.

    Args:
        prompt: User prompt
        model: Model identifier for OpenRouter
        temperature: Sampling temperature (0 for deterministic)
        max_tokens: Maximum tokens to generate
        system_prompt: Optional system prompt

    Returns:
        Generated text response
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API Error: {e}")
        raise


def generate_summary_format(
    document: str,
    format_type: str,
    model: str = "openai/gpt-4.1"
) -> str:
    """
    Generate a summary in a specific format.

    Args:
        document: Source document to summarize
        format_type: One of: dense_prose, bullet_points, hierarchical,
                     progressive_2level, progressive_3level
        model: Model to use

    Returns:
        Formatted summary
    """

    format_prompts = {
        "dense_prose": """Summarize the following document in a single coherent paragraph.
Include all key information in flowing prose. Do not use bullet points or headers.

Document:
{document}

Write a comprehensive paragraph summary:""",

        "bullet_points": """Summarize the following document using bullet points.
Extract the key information and present each point as a separate bullet.
Keep each bullet concise but informative.

Document:
{document}

Key points (as bullet points):""",

        "hierarchical": """Summarize the following document using a hierarchical structure.
Create 2-3 main headers with bullet points under each.
Organize the information logically by topic.

Document:
{document}

Hierarchical summary:""",

        "progressive_2level": """Summarize the following document in two levels:
1. A brief executive summary (1-2 sentences capturing the essence)
2. Detailed key points expanding on the summary

Format as:
EXECUTIVE SUMMARY:
[1-2 sentence overview]

DETAILS:
[Bullet points with more information]

Document:
{document}

Two-level summary:""",

        "progressive_3level": """Summarize the following document in three levels of detail:
1. TL;DR - A single sentence capturing the core message
2. KEY POINTS - 3-5 main takeaways
3. FULL ANALYSIS - Detailed breakdown with all important information

Format as:
TL;DR:
[One sentence]

KEY POINTS:
[3-5 bullets]

FULL ANALYSIS:
[Detailed summary]

Document:
{document}

Three-level summary:"""
    }

    prompt = format_prompts[format_type].format(document=document)

    return call_llm(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=1024
    )


def evaluate_summary(
    document: str,
    summary: str,
    dimension: str,
    model: str = "openai/gpt-4.1"
) -> float:
    """
    Evaluate a summary on a specific dimension using LLM-as-judge.

    Args:
        document: Original source document
        summary: Summary to evaluate
        dimension: One of: faithfulness, completeness, conciseness, readability
        model: Model to use for evaluation

    Returns:
        Score from 0.0 to 1.0
    """

    eval_prompts = {
        "faithfulness": """Evaluate the faithfulness of this summary.
Faithfulness means the summary accurately represents the source without adding false information or contradicting the original.

Source Document:
{document}

Summary to Evaluate:
{summary}

Rate the faithfulness from 0 to 10 where:
0 = Completely unfaithful (contains major hallucinations or contradictions)
5 = Partially faithful (some accurate content but notable errors)
10 = Perfectly faithful (all information accurate and traceable to source)

Respond with ONLY a number from 0 to 10.""",

        "completeness": """Evaluate the completeness of this summary.
Completeness means the summary captures all essential information and key points from the source.

Source Document:
{document}

Summary to Evaluate:
{summary}

Rate the completeness from 0 to 10 where:
0 = Missing all key information
5 = Captures some key points but misses important ones
10 = Captures all essential information

Respond with ONLY a number from 0 to 10.""",

        "conciseness": """Evaluate the conciseness of this summary.
Conciseness means the summary is appropriately brief without unnecessary verbosity or redundancy.

Summary to Evaluate:
{summary}

Rate the conciseness from 0 to 10 where:
0 = Extremely verbose with excessive redundancy
5 = Moderate verbosity, could be more concise
10 = Perfectly concise, no wasted words

Respond with ONLY a number from 0 to 10.""",

        "readability": """Evaluate the readability of this summary.
Readability means the summary is well-structured, easy to understand, and quickly scannable.

Summary to Evaluate:
{summary}

Rate the readability from 0 to 10 where:
0 = Very difficult to read and understand
5 = Moderately readable, some effort required
10 = Excellent readability, easy to scan and comprehend

Respond with ONLY a number from 0 to 10."""
    }

    prompt = eval_prompts[dimension].format(document=document, summary=summary)

    response = call_llm(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=10
    )

    # Parse numeric response
    try:
        score = float(response.strip())
        return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
    except ValueError:
        # Try to extract number from response
        import re
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        if numbers:
            score = float(numbers[0])
            return min(max(score / 10.0, 0.0), 1.0)
        return 0.5  # Default to middle if parsing fails


def simulate_human_preference(
    document: str,
    summaries: dict,
    model: str = "openai/gpt-4.1"
) -> dict:
    """
    Simulate human preference by asking LLM to rank summaries.

    Args:
        document: Original source document
        summaries: Dict mapping format_type to summary text
        model: Model to use

    Returns:
        Dict with rankings and reasoning
    """

    summary_text = "\n\n".join([
        f"FORMAT {i+1} ({name}):\n{text}"
        for i, (name, text) in enumerate(summaries.items())
    ])

    format_names = list(summaries.keys())

    prompt = f"""You are a busy professional who needs to quickly understand information from documents.
You are evaluating different summary formats to determine which is most effective for human comprehension.

Original Document:
{document}

Here are {len(summaries)} different summary formats of the same document:

{summary_text}

As a busy reader, rank these formats from BEST (1) to WORST ({len(summaries)}) based on:
- How quickly you can grasp the key information
- How easy it is to scan and find specific details
- How well it helps you decide if you need more information
- Overall usefulness for efficient information consumption

Respond in this exact JSON format:
{{
    "rankings": {{
        "FORMAT_NAME": RANK_NUMBER,
        ...
    }},
    "best_format": "FORMAT_NAME",
    "reasoning": "Brief explanation"
}}

Where FORMAT_NAME is one of: {format_names}"""

    response = call_llm(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=512
    )

    # Parse JSON response
    try:
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            return json.loads(json_match.group())
    except json.JSONDecodeError:
        pass

    # Fallback
    return {"rankings": {}, "best_format": "unknown", "reasoning": response}
