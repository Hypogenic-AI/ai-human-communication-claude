"""
Prompt templates for different AI-to-Human communication strategies.
Each strategy tests a different approach to presenting information.
"""

# Strategy 1: Dense Prose (Baseline)
DENSE_PROSE_PROMPT = """Summarize the following document in a single coherent paragraph.
Include all key information and maintain accuracy. Write in complete sentences.

Document:
{document}

Summary:"""

# Strategy 2: Concise (Extreme brevity)
CONCISE_PROMPT = """Provide a very brief summary of the following document in 1-2 sentences maximum.
Focus only on the most essential point.

Document:
{document}

Summary:"""

# Strategy 3: Bullet Points
BULLET_POINTS_PROMPT = """Summarize the following document as a bullet-point list.
Each bullet should be a key point. Use 3-7 bullets depending on document complexity.

Document:
{document}

Summary:
•"""

# Strategy 4: Structured Sections
STRUCTURED_SECTIONS_PROMPT = """Summarize the following document using a structured format with clear sections.

Document:
{document}

Summary:

**Main Topic:** [One sentence describing what this is about]

**Key Points:**
• [Point 1]
• [Point 2]
• [Point 3]

**Conclusion/Implication:** [One sentence on significance or takeaway]"""

# Strategy 5: Progressive Hierarchy (3-level)
PROGRESSIVE_HIERARCHY_PROMPT = """Summarize the following document at three levels of detail:

Document:
{document}

Summary:

**TL;DR (One sentence):**
[Single sentence capturing the essence]

**Key Points (3-5 bullets):**
• [Key point 1]
• [Key point 2]
• [Key point 3]

**Detailed Summary:**
[One paragraph with full context and details]"""

# Strategy 6: Formal Technical
FORMAL_TECHNICAL_PROMPT = """Provide a professional, objective summary of the following document.
Use formal academic tone, precise language, and avoid colloquialisms.

Document:
{document}

Summary:"""

# Strategy 7: Conversational
CONVERSATIONAL_PROMPT = """Summarize the following document in a natural, conversational way.
Imagine you're explaining this to a friend. Be engaging but accurate.

Document:
{document}

Summary:"""


# All strategies with metadata
STRATEGIES = {
    "dense_prose": {
        "name": "Dense Prose",
        "category": "baseline",
        "prompt": DENSE_PROSE_PROMPT,
        "description": "Standard paragraph summary without formatting"
    },
    "concise": {
        "name": "Concise",
        "category": "baseline",
        "prompt": CONCISE_PROMPT,
        "description": "Extremely brief 1-2 sentence summary"
    },
    "bullet_points": {
        "name": "Bullet Points",
        "category": "structured",
        "prompt": BULLET_POINTS_PROMPT,
        "description": "Key points as bullet list"
    },
    "structured_sections": {
        "name": "Structured Sections",
        "category": "structured",
        "prompt": STRUCTURED_SECTIONS_PROMPT,
        "description": "Organized with headers and sections"
    },
    "progressive_hierarchy": {
        "name": "Progressive Hierarchy",
        "category": "structured",
        "prompt": PROGRESSIVE_HIERARCHY_PROMPT,
        "description": "Three levels: TL;DR, bullets, detailed"
    },
    "formal_technical": {
        "name": "Formal Technical",
        "category": "style",
        "prompt": FORMAL_TECHNICAL_PROMPT,
        "description": "Professional, academic tone"
    },
    "conversational": {
        "name": "Conversational",
        "category": "style",
        "prompt": CONVERSATIONAL_PROMPT,
        "description": "Natural, friendly tone"
    }
}


# LLM-as-Judge Evaluation Prompts
EVALUATION_PROMPT_TEMPLATE = """You are evaluating the quality of a summary. Rate the following dimensions on a scale of 1-5.

**Source Document:**
{document}

**Summary to Evaluate:**
{summary}

Rate each dimension and provide a brief justification:

1. **Faithfulness** (1-5): Does the summary accurately represent the source without adding false information?
   - 1: Major factual errors or hallucinations
   - 3: Mostly accurate with minor issues
   - 5: Completely faithful to the source

2. **Completeness** (1-5): Does the summary capture the key information?
   - 1: Missing most important points
   - 3: Captures main idea but misses some details
   - 5: Captures all essential information

3. **Conciseness** (1-5): Is the summary appropriately brief without unnecessary content?
   - 1: Very redundant or verbose
   - 3: Some unnecessary content
   - 5: Perfectly concise

4. **Clarity** (1-5): Is the summary easy to understand and well-organized?
   - 1: Confusing or poorly structured
   - 3: Understandable but could be clearer
   - 5: Crystal clear and well-organized

5. **Overall Quality** (1-5): Holistic assessment of the summary quality
   - 1: Poor quality
   - 3: Acceptable
   - 5: Excellent

Respond in the following JSON format:
{{
    "faithfulness": {{"score": <1-5>, "justification": "<brief reason>"}},
    "completeness": {{"score": <1-5>, "justification": "<brief reason>"}},
    "conciseness": {{"score": <1-5>, "justification": "<brief reason>"}},
    "clarity": {{"score": <1-5>, "justification": "<brief reason>"}},
    "overall": {{"score": <1-5>, "justification": "<brief reason>"}}
}}"""


PAIRWISE_COMPARISON_PROMPT = """Compare these two summaries of the same document and select which one is better.

**Source Document:**
{document}

**Summary A:**
{summary_a}

**Summary B:**
{summary_b}

Consider:
- Which is easier to understand?
- Which captures the key information better?
- Which has a more appropriate length?
- Which is better organized?

Respond in JSON format:
{{
    "winner": "A" or "B" or "tie",
    "reason": "<brief explanation of why this summary is better>"
}}"""


def get_prompt(strategy_key: str, document: str) -> str:
    """Get formatted prompt for a strategy."""
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_key}")
    return STRATEGIES[strategy_key]["prompt"].format(document=document)


def get_evaluation_prompt(document: str, summary: str) -> str:
    """Get formatted evaluation prompt."""
    return EVALUATION_PROMPT_TEMPLATE.format(document=document, summary=summary)


def get_pairwise_prompt(document: str, summary_a: str, summary_b: str) -> str:
    """Get formatted pairwise comparison prompt."""
    return PAIRWISE_COMPARISON_PROMPT.format(
        document=document,
        summary_a=summary_a,
        summary_b=summary_b
    )
