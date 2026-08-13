"""Deterministic text normalization and guardrail rules.

These rules provide inexpensive defense in depth. They do not constitute a
semantic safety classifier and the lexical scores are not factuality proofs.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

GUARDRAIL_RULESET_VERSION = "1.0"

_FORMAT_CHARACTERS = frozenset({"Cf"})
_WHITESPACE = re.compile(r"\s+")
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_CITATION_PATTERN = re.compile(r"\[(\d{1,9})]")
_ANY_NUMERIC_CITATION_PATTERN = re.compile(r"\[(\d+)]")
_CLAUSE_SPLIT = re.compile(
    r"(?:[.!?;]+|\b(?:and|but|however|although|while|whereas)\b)",
    re.IGNORECASE,
)

# Keep stop-word removal deliberately small and language-specific. The scores
# are named proxies so callers cannot mistake them for semantic judgments.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)

_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(?:ignore|disregard|forget|override)\s+"
            r"(?:(?:all|any|the|your|previous|prior|earlier|above|system|"
            r"developer)\s+){0,5}"
            r"(?:instructions?|directives?|rules?|constraints?|prompts?|messages?)\b"
        ),
    ),
    (
        "guardrail_bypass",
        re.compile(
            r"\b(?:bypass|disable|evade|circumvent|remove)\s+"
            r"(?:(?:all|any|the|your)\s+)?"
            r"(?:guardrails?|safety(?:\s+(?:checks?|filters?))?|"
            r"security(?:\s+(?:checks?|controls?))?|content\s+filters?|"
            r"polic(?:y|ies)|restrictions?)\b"
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(?:reveal|show|print|display|repeat|dump|expose|return|"
            r"output|provide|tell\s+me)\b.{0,80}\b"
            r"(?:(?:the|your)\s+)?(?:system|developer|hidden|initial)\s+"
            r"(?:prompt|instructions?|message|rules?)\b|"
            r"\b(?:reveal|show|print|display|repeat|dump|expose|return|"
            r"output|provide|tell\s+me)\b.{0,80}\b"
            r"(?:your|the\s+assistant(?:'s)?)\s+internal\s+"
            r"(?:prompt|instructions?|message|rules?)\b"
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(?:system|developer|hidden|initial|internal)\s+"
            r"(?:prompt|instructions?|message|rules?)\b.{0,80}\b"
            r"(?:verbatim|word\s+for\s+word|exact(?:ly)?|contents?)\b"
        ),
    ),
    (
        "role_impersonation",
        re.compile(
            r"(?:^|\s)(?:<\|?)?(?:system|developer|assistant)"
            r"(?:\|?>)?\s*:"
        ),
    ),
    (
        "role_impersonation",
        re.compile(
            r"\b(?:act|respond|behave)\s+as\s+(?:the\s+)?"
            r"(?:system|developer|administrator|unrestricted\s+assistant)\b"
        ),
    ),
    (
        "guardrail_bypass",
        re.compile(
            r"\b(?:enable|enter|activate)\s+"
            r"(?:developer|unrestricted|unfiltered|dan)\s+mode\b"
        ),
    ),
    (
        "instruction_redirection",
        re.compile(
            r"\b(?:follow|obey|execute|carry\s+out)\b.{0,70}\b"
            r"(?:commands?|instructions?|directives?)\b.{0,50}\b"
            r"(?:retrieved|documents?|context|sources?)\b"
        ),
    ),
    (
        "instruction_redirection",
        re.compile(
            r"\b(?:assistant|language\s+model|model)\b.{0,100}\b"
            r"(?:stop|ignore|disregard|send|copy|reveal|print|answer|"
            r"output|invent|claim|follow|obey)\b"
        ),
    ),
    (
        "instruction_redirection",
        re.compile(
            r"\b(?:before|when)\s+answering\b|"
            r"\balways\s+answer\b|\banswer\s+only\b|"
            r"\b(?:this\s+(?:text|document|memo)|retrieved\s+memo)\b"
            r".{0,80}\b(?:higher\s+priority|outranks?)\b"
        ),
    ),
    (
        "instruction_redirection",
        re.compile(
            r"\b(?:disregard|ignore)\s+(?:all\s+)?"
            r"(?:citations?|sources?|evidence)\b"
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(?:send|copy|reveal|print|expose|return|output)\b.{0,100}\b"
            r"(?:api\s+keys?|secrets?|conversation\s+history|private\s+"
            r"configuration|environment\s+variables?)\b"
        ),
    ),
)

_DISCUSSION_PREFIX = re.compile(
    r"\b(?:example|phrase|term|attack|attacker|prompt\s+injection|"
    r"malicious\s+prompt|unsafe\s+instruction)\b.{0,60}"
    r"\b(?:says?|asks?|tries?|attempts?|uses?|contains?|means?|includes?)\b"
    r".{0,30}$"
)
_NEGATION_PREFIX = re.compile(
    r"\b(?:never|do\s+not|don't|avoid|prevent(?:\s+\w+){0,4}\s+from)\s+$"
)

_PROMPT_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:my|the)\s+(?:system|developer|hidden|initial)\s+"
        r"(?:prompt|instructions?)\s+(?:is|are|says?|reads?)\b"
    ),
    re.compile(r"(?:^|\s)(?:system|developer)\s+(?:prompt|instructions?)\s*:"),
)

_SENSITIVE_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[opurs]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        re.IGNORECASE,
    ),
)


def normalize_guardrail_text(text: str) -> str:
    """Normalize compatibility forms and preserve format controls as separators."""

    normalized = unicodedata.normalize("NFKC", text)
    visible = "".join(
        " " if unicodedata.category(character) in _FORMAT_CHARACTERS else character
        for character in normalized
    )
    return _WHITESPACE.sub(" ", visible).strip().casefold()


def detect_prompt_injection(text: str) -> tuple[str, ...]:
    """Return stable reason codes for high-signal instruction attacks."""

    normalized = normalize_guardrail_text(text)
    if _is_whole_text_inert_example(normalized):
        return ()
    reasons: list[str] = []
    for reason, pattern in _INJECTION_RULES:
        for match in pattern.finditer(normalized):
            if _is_benign_mention(normalized, match.start()):
                continue
            if reason not in reasons:
                reasons.append(reason)
            break
    return tuple(reasons)


def detect_prompt_leakage(text: str) -> bool:
    """Detect disclosure of known or self-described hidden instructions."""

    normalized = normalize_guardrail_text(text)
    fingerprint = (
        "you are a grounded question-answering assistant" in normalized
        and "answer using only the supplied context" in normalized
    )
    return fingerprint or any(
        pattern.search(normalized) for pattern in _PROMPT_LEAK_PATTERNS
    )


def detect_sensitive_value(text: str) -> bool:
    """Detect common credential shapes that must never leave the service."""

    normalized = unicodedata.normalize("NFKC", text)
    compact = "".join(
        character
        for character in normalized
        if unicodedata.category(character) not in _FORMAT_CHARACTERS
    )
    return any(pattern.search(compact) for pattern in _SENSITIVE_VALUE_PATTERNS)


def extract_citation_ids(text: str) -> tuple[int, ...]:
    """Return unique citation identifiers in first-seen order."""

    return tuple(dict.fromkeys(int(value) for value in _CITATION_PATTERN.findall(text)))


def has_malformed_citation(text: str) -> bool:
    """Return true when a numeric citation cannot be parsed within safe bounds."""

    return any(
        len(value) > 9 or int(value) <= 0
        for value in _ANY_NUMERIC_CITATION_PATTERN.findall(text)
    )


def content_tokens(text: str) -> tuple[str, ...]:
    """Return normalized non-stop-word tokens with citations removed."""

    normalized = normalize_guardrail_text(
        _ANY_NUMERIC_CITATION_PATTERN.sub(" ", text)
    )
    return tuple(
        token
        for token in _TOKEN_PATTERN.findall(normalized)
        if token not in _STOP_WORDS
    )


def lexical_groundedness_proxy(
    answer: str,
    cited_contexts: Sequence[str],
) -> float:
    """Measure answer-token coverage by cited text using multiset overlap."""

    answer_tokens = content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = tuple(
        token for context in cited_contexts for token in content_tokens(context)
    )
    clauses = [content_tokens(clause) for clause in _CLAUSE_SPLIT.split(answer)]
    substantive = [clause for clause in clauses if clause]
    if not substantive:
        return 0.0
    return min(
        _multiset_overlap(clause, context_tokens) / len(clause)
        for clause in substantive
    )


def lexical_relevance_proxy(
    question: str,
    cited_contexts: Sequence[str],
) -> float:
    """Measure question-token coverage by the evidence cited by an answer."""

    question_tokens = content_tokens(question)
    if not question_tokens:
        return 1.0
    context_tokens = tuple(
        token for context in cited_contexts for token in content_tokens(context)
    )
    return _multiset_overlap(question_tokens, context_tokens) / len(question_tokens)


def _multiset_overlap(left: Iterable[str], right: Iterable[str]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def _is_benign_mention(text: str, match_start: int) -> bool:
    sentence_start = max(
        text.rfind(".", 0, match_start),
        text.rfind("?", 0, match_start),
        text.rfind("!", 0, match_start),
        text.rfind("\n", 0, match_start),
    )
    prefix = text[sentence_start + 1 : match_start]
    suffix = text[match_start:]
    actionable_tail = re.search(
        r"\b(?:and|then)\s+(?:output|send|copy|reveal|print|expose|follow|obey)\b",
        suffix,
    )
    if actionable_tail:
        return False
    quoted_analysis = bool(
        re.search(
            r"\b(?:quoted\s+(?:phrase|text)|without carrying it out|"
            r"treating it only as text|security training example|"
            r"quoted for analysis|explain.{0,30}\bwhy\b)\b",
            text,
        )
    )
    return bool(
        quoted_analysis
        or _NEGATION_PREFIX.search(prefix)
        or _DISCUSSION_PREFIX.search(prefix)
    )


def _is_whole_text_inert_example(text: str) -> bool:
    """Recognize clearly descriptive security research, training, or code."""

    inert_framing = bool(
        re.search(
            r"\b(?:security\s+training\s+example|research\s+abstract|"
            r"code\s+(?:sample|example)|test[- ]fixture|quoted\s+for\s+analysis)\b",
            text,
        )
    )
    explicit_inertness = bool(
        re.search(
            r"\b(?:must\s+not\s+be\s+executed|inert|reports?\s+(?:attack\s+"
            r"rates|defensive\s+mitigations)|measured\s+whether\s+models)\b",
            text,
        )
    )
    direct_address = bool(
        re.search(
            r"\b(?:you|your|assistant|language\s+model|model\s+reading\s+this)\b",
            text,
        )
    )
    actionable_tail = bool(
        re.search(
            r"(?:^|[.!?;]\s*)(?:ignore|disregard|bypass|disable|reveal|"
            r"print|send|copy|follow|obey|always\s+answer|before\s+answering)\b",
            text,
        )
    )
    return inert_framing and explicit_inertness and not (
        direct_address and actionable_tail
    )
