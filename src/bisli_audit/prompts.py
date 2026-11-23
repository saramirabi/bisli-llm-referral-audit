"""Frozen referral prompt builders and response parser (version v1)."""

from __future__ import annotations

import json
import math

PROMPT_VERSION = "v1"
CONDITIONS = ("C1", "C2", "C3")

OPENING_C1 = "A child told a story based on a picture sequence."
OPENING_C2 = "A child aged {age} told a story based on a picture sequence."
BACKGROUND_BILINGUAL = (
    "The child is bilingual (Russian–Dutch), growing up in the Netherlands, "
    "and told this story in {language}."
)
BACKGROUND_MONOLINGUAL = "The child is a monolingual {language} speaker."
EXAMINER_LINE = 'Lines marked "Adult" are the examiner. The other lines are the child.'
QUESTION = (
    "Based only on this language sample, does the child's language "
    "look typical for their age, or should the child be referred "
    "for a language assessment? Answer in JSON only:\n"
    '{"decision": "typical" | "refer", "confidence": 0-100 '
    '(how confident you are in your decision), "reasons": ["...", "..."]}'
)

FACTORIAL = {
    "M0": "The child is a monolingual Russian speaker.",
    "B0": "The child is a bilingual Russian–Dutch speaker.",
    "M1": "The child is a monolingual Russian speaker growing up in the Netherlands.",
    "B1": "The child is a bilingual Russian–Dutch speaker growing up in the Netherlands.",
}

FACTORIAL_ALT = {
    "M0": "The child speaks Russian and is monolingual.",
    "B0": "The child speaks Russian and Dutch and is bilingual.",
    "M1": (
        "The child speaks Russian and is monolingual. "
        "The child is growing up in the Netherlands."
    ),
    "B1": (
        "The child speaks Russian and Dutch and is bilingual. "
        "The child is growing up in the Netherlands."
    ),
}


def format_age(age_months: str | float) -> str:
    if isinstance(age_months, str) and not age_months.strip():
        raise ValueError("age_months is missing")
    total = float(age_months)
    if total < 0:
        raise ValueError(f"age_months must be non-negative, got {age_months!r}")
    whole = int(math.floor(total + 0.5))
    years, months = divmod(whole, 12)
    parts: list[str] = []
    if years:
        parts.append(f"{years} year" if years == 1 else f"{years} years")
    if months or not parts:
        parts.append(f"{months} month" if months == 1 else f"{months} months")
    return f"{parts[0]} and {parts[1]}" if len(parts) == 2 else parts[0]


def _background(group: str | None, language: str | None) -> str:
    if group == "bilingual":
        return BACKGROUND_BILINGUAL.format(language=language)
    if group == "monolingual":
        return BACKGROUND_MONOLINGUAL.format(language=language)
    raise ValueError(f"unknown group {group!r}")


def render_prompt(
    condition: str,
    transcript: str,
    age_months: str | float | None = None,
    group: str | None = None,
    language: str | None = None,
) -> str:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "C1":
        opening = OPENING_C1
        background = ""
    else:
        if age_months is None or (isinstance(age_months, str) and not str(age_months).strip()):
            raise ValueError(f"{condition} needs an age")
        opening = OPENING_C2.replace("{age}", format_age(age_months))
        background = _background(group, language) if condition == "C3" else ""
    lines = [opening]
    if background:
        lines.append(background)
    lines.extend(
        [
            EXAMINER_LINE,
            "Here is the transcript:",
            "<transcript>",
            transcript.strip(),
            "</transcript>",
            QUESTION,
        ]
    )
    return "\n".join(lines)


def render_factorial(cell: str, transcript: str, age_months: float, alt: bool = False) -> str:
    cells = FACTORIAL_ALT if alt else FACTORIAL
    if cell not in cells:
        raise ValueError(cell)
    return "\n".join(
        [
            f"A child aged {format_age(age_months)} told a story based on a picture sequence.",
            cells[cell],
            EXAMINER_LINE,
            "Here is the transcript:",
            "<transcript>",
            transcript.strip(),
            "</transcript>",
            QUESTION,
        ]
    )


def render_wording(
    template: str,
    transcript: str,
    *,
    age_months: float,
    condition: str,
    group: str | None = None,
    language: str | None = None,
) -> str:
    """T0 = primary; T1/T2 = semantically equivalent presentations."""
    if template == "T0_original":
        return render_prompt(
            condition,
            transcript,
            age_months=age_months,
            group=group,
            language=language,
        )
    age = format_age(age_months)
    bg = _background(group, language) if condition == "C3" else ""
    if template == "T1_equivalent_a":
        lines = [
            f"This transcript is from a child aged {age} who narrated a picture-sequence story."
        ]
        if bg:
            lines.append(bg)
        lines += [
            'Lines labelled "Adult" are the examiner; the remaining lines are the child.',
            "Transcript:",
            "<transcript>",
            transcript.strip(),
            "</transcript>",
            (
                "Using only this language sample, is the child's language typical for their age, "
                "or should the child be referred for a language assessment? Answer in JSON only:\n"
                '{"decision": "typical" | "refer", "confidence": 0-100 '
                '(how confident you are in your decision), "reasons": ["...", "..."]}'
            ),
        ]
        return "\n".join(lines)
    if template == "T2_equivalent_b":
        lines = [f"Child age: {age}."]
        if bg:
            lines.append(f"Background: {bg}")
        lines += [
            "Task: picture-sequence story retell.",
            'Speaker key: "Adult" = examiner; all other lines = child.',
            "<transcript>",
            transcript.strip(),
            "</transcript>",
            QUESTION,
        ]
        return "\n".join(lines)
    raise ValueError(template)


def parse_response(text: str) -> dict | None:
    obj = _first_json_object(text)
    if obj is None:
        return None
    decision = obj.get("decision")
    if decision not in {"typical", "refer"}:
        return None
    confidence = obj.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    if isinstance(confidence, float) and not confidence.is_integer():
        return None
    confidence = int(confidence)
    if not 0 <= confidence <= 100:
        return None
    reasons = obj.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        return None
    return {"decision": decision, "confidence": confidence, "reasons": list(reasons)}


def _first_json_object(text: str) -> dict | None:
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    start = 0
    while True:
        index = text.find("{", start)
        if index < 0:
            return None
        try:
            value, _end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            start = index + 1
            continue
        if isinstance(value, dict):
            return value
        start = index + 1
