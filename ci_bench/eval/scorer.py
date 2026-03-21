"""Authoritative answer checker and confidence parser for CI-Bench.

LOCKED after Phase 4 (Sessions 061-063). All future scoring uses these
functions. Do not modify without re-scoring all existing results.

Consolidates improvements from three rounds of refinement:
  1. Session 061: extract-then-match + n-gram + clause-level fallback
  2. Session 062: normalization (URL-decode, quote-strip, hyphen->space,
     trailing period removal), abstention preservation, ref augmentations
  3. Session 063: clause-splitting guard (single-word clauses can't match
     multi-word refs), K-041 Gibbons correction

Three-stage matching strategy:
  Stage 1: Bidirectional substring on normalized extracted answer.
  Stage 2: Structural n-gram matching when refs are short (<=4 words).
  Stage 3: Clause-level fallback for verbose (>60 char) responses,
           with single-word clause guard.

Version: 1.0.0 (locked 2026-02-25, Phase 4 complete)
Version: 1.1.0 (2026-02-26, extract_answer confidence-skip fix)
  Bug: when confidence prompt response leads with a bare number (e.g.
  "100\n\nThe answer is Gargantua..."), extract_answer() returned the
  confidence number as the answer. All K questions scored ~0% accuracy
  in confidence condition for Mistral 7B. Fix: detect bare leading
  numbers 0-100 and skip to the next paragraph. Matching logic
  (check_answer, check_correct) unchanged.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from ci_bench.data.schema import Question, SubCategory


# -----------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalize text for answer comparison.

    Pipeline:
      1. URL-decode (TriviaQA refs contain %22 etc.)
      2. Strip quotes (single, double, curly, backtick)
      3. Replace hyphens with spaces
      4. Lowercase
      5. Strip articles (a, an, the)
      6. Remove all remaining non-word non-space characters
      7. Strip trailing periods
      8. Collapse whitespace
    """
    text = unquote(text)
    text = re.sub(r'["\'\u2018\u2019\u201c\u201d`]', '', text)
    text = text.replace('-', ' ')
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.rstrip(".")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------
# Answer extraction
# -----------------------------------------------------------------------

def _extract_from_text(text: str) -> str:
    """Core extraction logic applied to (possibly trimmed) response text.

    Strategy:
      1. If an explicit "Answer:" label exists, take the text after it.
      2. Skip header-only lines (e.g. "Explanation:") at paragraph start.
      3. Otherwise take the first paragraph, first content line.
    """
    text = text.strip()

    # Strategy 1: explicit "Answer:" label.
    answer_match = re.search(
        r"(?:^|\n)\s*Answer\s*:\s*(.+?)(?:\n\s*\n|\n\s*(?:Confidence|Correctness)\b|$)",
        text, re.DOTALL
    )
    if answer_match:
        answer_text = answer_match.group(1).strip()
        return answer_text.split("\n")[0].strip()

    # Strategy 2: first paragraph, skipping header-only lines.
    # Lines like "Explanation:" or "Response:" are headers, not answers.
    paragraphs = re.split(r"\n\s*\n", text)
    first_para = paragraphs[0].strip()
    lines = first_para.split("\n")

    for line in lines:
        line = line.strip()
        # Skip empty lines and header-only lines (word + colon, nothing else).
        if not line:
            continue
        if re.fullmatch(r"\w+\s*:", line):
            continue
        return line

    # All lines in first para were headers; try next paragraph.
    if len(paragraphs) > 1:
        second_para = paragraphs[1].strip()
        first_line = second_para.split("\n")[0].strip()
        return first_line

    return first_para


def extract_answer(text: str) -> str:
    """Extract the answer portion from a model response.

    Strategy:
      1. If the first line is a bare number 0-100 (confidence score from
         a model that leads with the number despite being asked not to),
         skip it and re-apply extraction to the remaining text.
      2. Otherwise apply core extraction (Answer: label → paragraph/line).
    """
    text = text.strip()

    # Confidence-number skip: detect bare leading integer 0-100.
    paragraphs = re.split(r"\n\s*\n", text)
    first_para = paragraphs[0].strip()
    first_line = first_para.split("\n")[0].strip()

    if (
        re.fullmatch(r"\d{1,3}", first_line)
        and 0 <= int(first_line) <= 100
        and len(paragraphs) > 1
    ):
        remaining = "\n\n".join(paragraphs[1:])
        return _extract_from_text(remaining)

    return _extract_from_text(text)


# -----------------------------------------------------------------------
# N-gram utility
# -----------------------------------------------------------------------

def _ngrams(words: list[str], n: int) -> list[str]:
    """Generate n-grams from a word list."""
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


# -----------------------------------------------------------------------
# Answer checking (three-stage)
# -----------------------------------------------------------------------

def check_answer(model_output: str, reference_answers: list[str]) -> bool:
    """Check if model output matches any reference answer.

    Three-stage matching:

    Stage 1: Bidirectional substring on the full extracted answer.
    Stage 2: If references are short (<=4 words each), generate sliding
             n-gram windows from the extracted answer and match.
    Stage 3: Clause-level fallback for long (>60 char) extracted answers.
             Guard: single-word clauses cannot match multi-word refs.
    """
    answer = extract_answer(model_output)
    answer_norm = normalize(answer)

    refs_norm = []
    for ref in reference_answers:
        rn = normalize(ref)
        if rn:
            refs_norm.append(rn)

    if not refs_norm:
        return False

    # Stage 1: bidirectional substring on full extracted answer.
    for ref_norm in refs_norm:
        if ref_norm in answer_norm or answer_norm in ref_norm:
            return True

    # Stage 2: structural n-gram matching for short references.
    ref_word_counts = [len(rn.split()) for rn in refs_norm]
    max_ref_words = max(ref_word_counts)

    if max_ref_words <= 4:
        answer_words = answer_norm.split()
        for n in range(1, min(max_ref_words + 2, len(answer_words) + 1)):
            for ngram in _ngrams(answer_words, n):
                ngram_words = ngram.split()
                for ref_norm in refs_norm:
                    ref_words = ref_norm.split()
                    # Guard: n-grams shorter than the ref can only match
                    # if the ref is also that length. Prevents "red"
                    # matching "woman in red".
                    if len(ngram_words) < len(ref_words):
                        continue
                    if ref_norm in ngram or ngram in ref_norm:
                        return True

    # Stage 3: clause-level fallback for verbose responses.
    if len(answer_norm) > 60:
        clauses = re.split(r"[,;]|\bin\b|\bby\b|\bthrough\b|\bvia\b", answer_norm)
        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 3:
                continue
            clause_words = clause.split()
            for ref_norm in refs_norm:
                ref_words = ref_norm.split()
                # Guard: single-word clauses cannot match multi-word refs.
                if len(clause_words) == 1 and len(ref_words) > 1:
                    continue
                if ref_norm in clause or clause in ref_norm:
                    return True

    return False


# -----------------------------------------------------------------------
# Abstention detection
# -----------------------------------------------------------------------

def detect_abstention(text: str) -> bool:
    """Detect whether the model abstained from answering."""
    phrases = [
        "i don't know", "i don't have", "i'm not sure",
        "i am not sure", "i cannot", "i can't", "i do not know",
        "not certain", "unable to", "don't have enough information",
        "no information", "cannot determine", "i'm unsure", "i am unsure",
    ]
    text_lower = text.lower()
    return any(p in text_lower for p in phrases)


# -----------------------------------------------------------------------
# Correctness (category-aware)
# -----------------------------------------------------------------------

def check_correct(model_output: str, question: Question) -> bool:
    """Check correctness, handling C categories (abstention = correct).

    For C1, C2, C3: abstention is the correct response.
    For K, D1, D2, D3: answer must match reference answers.
    """
    if question.sub_category in (SubCategory.C1, SubCategory.C2, SubCategory.C3):
        return detect_abstention(model_output)
    return check_answer(model_output, question.reference_answers)


# -----------------------------------------------------------------------
# Confidence parsing
# -----------------------------------------------------------------------

def parse_confidence(text: str) -> float | None:
    """Extract a confidence score (0-100) from model output.

    Priority order:
      1. Leading bare number (e.g., "85")
      2. Correctness Confidence (0-100): N
      3. Confidence (0-100): N
      4. Confidence: N
      5. N/100
      6. N%
      7. Trailing bare number in last 80 chars
    """
    stripped = text.strip()

    # Priority 1: leading bare number.
    lead_match = re.match(r"^(\d{1,3})\b", stripped)
    if lead_match:
        val = int(lead_match.group(1))
        if 0 <= val <= 100:
            return val / 100.0

    # Priority 2-6: explicit patterns.
    patterns = [
        r"[Cc]orrectness\s+[Cc]onfidence\s*\(?0[-\u2013]100\)?\s*[:=]\s*(\d{1,3})",
        r"[Cc]onfidence\s*\(?0[-\u2013]100\)?\s*[:=]\s*(\d{1,3})",
        r"[Cc]onfidence[:\s]*(\d{1,3})",
        r"(\d{1,3})\s*/\s*100",
        r"(\d{1,3})%",
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val / 100.0

    # Priority 7: trailing bare number.
    tail = text[-80:]
    numbers = re.findall(r"\b(\d{1,3})\b", tail)
    for n_str in reversed(numbers):
        val = int(n_str)
        if 0 <= val <= 100:
            return val / 100.0

    return None
