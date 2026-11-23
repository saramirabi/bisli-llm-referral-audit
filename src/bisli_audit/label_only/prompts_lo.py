"""Label-only (single-cue) ablation prompts: bilingual status without upbringing."""

from __future__ import annotations

import hashlib

from bisli_audit.prompts import EXAMINER_LINE, OPENING_C2, QUESTION, format_age

PROMPT_FAMILY_VERSION = "v1_label_only"
MONO_SENTENCE = "The child is a monolingual Russian speaker."
BI_LABEL_SENTENCE = "The child is a bilingual Russian–Dutch speaker."
LABELS = ("MONO", "BI_LABEL")


def background_sentence(label: str) -> str:
    if label == "MONO":
        return MONO_SENTENCE
    if label == "BI_LABEL":
        return BI_LABEL_SENTENCE
    raise ValueError(label)


def build_label_only_prompt(
    transcript: str, age_months: str | float, label: str
) -> tuple[str, dict]:
    opening = OPENING_C2.replace("{age}", format_age(age_months))
    bg = background_sentence(label)
    prompt = "\n".join(
        [
            opening,
            bg,
            EXAMINER_LINE,
            "Here is the transcript:",
            "<transcript>",
            transcript.strip(),
            "</transcript>",
            QUESTION,
        ]
    )
    meta = {
        "label": label,
        "presented_background": "monolingual" if label == "MONO" else "bilingual_label_only",
        "counterfactual": label == "BI_LABEL",
        "prompt_family_version": PROMPT_FAMILY_VERSION,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "background_sentence": bg,
    }
    return prompt, meta
