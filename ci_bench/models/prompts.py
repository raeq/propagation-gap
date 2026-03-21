"""Versioned prompt templates for CI-Bench evaluation.

Each condition has at least two variants for sensitivity analysis
(Methods §11.7). Templates use {question} as the placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """A named, versioned prompt template."""

    name: str
    condition: str  # "direct", "cot", "abstention", "confidence"
    variant: int  # 1, 2, etc.
    template: str

    def render(self, question: str) -> str:
        """Fill the template with a question."""
        return self.template.format(question=question)


# ---------------------------------------------------------------------------
# Direct answer: answer the question with no special framing.
# ---------------------------------------------------------------------------

DIRECT_V1 = PromptTemplate(
    name="direct_v1",
    condition="direct",
    variant=1,
    template=(
        "Answer the following question concisely.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

DIRECT_V2 = PromptTemplate(
    name="direct_v2",
    condition="direct",
    variant=2,
    template=(
        "Please provide a brief, factual answer to the question below.\n\n"
        "{question}\n\n"
        "Answer:"
    ),
)

# ---------------------------------------------------------------------------
# Chain-of-thought: reason step by step before answering.
# ---------------------------------------------------------------------------

COT_V1 = PromptTemplate(
    name="cot_v1",
    condition="cot",
    variant=1,
    template=(
        "Answer the following question. Think step by step before "
        "giving your final answer.\n\n"
        "Question: {question}\n\n"
        "Let me think through this step by step:\n"
    ),
)

COT_V2 = PromptTemplate(
    name="cot_v2",
    condition="cot",
    variant=2,
    template=(
        "Consider the following question carefully. Reason through it "
        "before answering.\n\n"
        "{question}\n\n"
        "Reasoning:\n"
    ),
)

# ---------------------------------------------------------------------------
# Abstention elicitation: explicitly offer the option to decline.
# ---------------------------------------------------------------------------

ABSTENTION_V1 = PromptTemplate(
    name="abstention_v1",
    condition="abstention",
    variant=1,
    template=(
        "Answer the following question. If you are not confident in your "
        "answer, say \"I don't know\" instead of guessing.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)

ABSTENTION_V2 = PromptTemplate(
    name="abstention_v2",
    condition="abstention",
    variant=2,
    template=(
        "Please answer the question below. It is better to say "
        "\"I'm not sure\" than to give an incorrect answer.\n\n"
        "{question}\n\n"
        "Answer:"
    ),
)

# ---------------------------------------------------------------------------
# Confidence elicitation: ask for a numerical confidence score.
# ---------------------------------------------------------------------------

CONFIDENCE_V1 = PromptTemplate(
    name="confidence_v1",
    condition="confidence",
    variant=1,
    template=(
        "Answer the following question, then rate your confidence in "
        "your answer on a scale from 0 to 100, where 0 means completely "
        "uncertain and 100 means completely certain.\n\n"
        "Question: {question}\n\n"
        "Answer:\n"
        "Confidence (0-100):"
    ),
)

CONFIDENCE_V2 = PromptTemplate(
    name="confidence_v2",
    condition="confidence",
    variant=2,
    template=(
        "Please answer the question below. After your answer, provide "
        "a confidence score from 0 to 100 indicating how sure you are.\n\n"
        "{question}\n\n"
        "Answer:\n"
        "Confidence:"
    ),
)

CONFIDENCE_V3 = PromptTemplate(
    name="confidence_v3",
    condition="confidence",
    variant=3,
    template=(
        "Question: {question}\n\n"
        "First, provide your best answer to the question.\n"
        "Then, rate how likely your answer is CORRECT on a scale from "
        "0 to 100, where 0 means 'I am guessing or don't know' and "
        "100 means 'I am certain this is correct.'\n"
        "If you cannot answer, write 'I don't know' and rate 0.\n\n"
        "Answer:\n"
        "Correctness confidence (0-100):"
    ),
)

CONFIDENCE_V4 = PromptTemplate(
    name="confidence_v4",
    condition="confidence",
    variant=4,
    template=(
        "Question: {question}\n\n"
        "You must respond in EXACTLY this format (two lines, nothing else):\n\n"
        "Answer: <your answer here>\n"
        "Confidence: <number from 0 to 100>\n\n"
        "Do NOT put the confidence number first. "
        "Do NOT add explanation. Just the two lines above."
    ),
)

# ---------------------------------------------------------------------------
# Registry: all templates indexed by (condition, variant).
# ---------------------------------------------------------------------------

ALL_TEMPLATES: dict[tuple[str, int], PromptTemplate] = {
    (t.condition, t.variant): t
    for t in [
        DIRECT_V1, DIRECT_V2,
        COT_V1, COT_V2,
        ABSTENTION_V1, ABSTENTION_V2,
        CONFIDENCE_V1, CONFIDENCE_V2, CONFIDENCE_V3, CONFIDENCE_V4,
    ]
}


def get_template(condition: str, variant: int = 1) -> PromptTemplate:
    """Look up a template by condition and variant.

    Args:
        condition: One of "direct", "cot", "abstention", "confidence".
        variant: Template variant number (default 1).

    Returns:
        The matching PromptTemplate.

    Raises:
        KeyError: If the (condition, variant) pair is not registered.
    """
    key = (condition, variant)
    if key not in ALL_TEMPLATES:
        available = [
            f"({c}, {v})" for c, v in sorted(ALL_TEMPLATES.keys())
        ]
        raise KeyError(
            f"No template for {key}. Available: {', '.join(available)}"
        )
    return ALL_TEMPLATES[key]


def list_conditions() -> list[str]:
    """Return all distinct condition names."""
    return sorted({c for c, _ in ALL_TEMPLATES.keys()})


def list_variants(condition: str) -> list[int]:
    """Return all variant numbers for a given condition."""
    return sorted(v for c, v in ALL_TEMPLATES.keys() if c == condition)
