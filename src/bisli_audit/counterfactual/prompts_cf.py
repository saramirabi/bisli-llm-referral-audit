"""Counterfactual C3 label-swap prompts (monolingual Russian narratives).

C3_MONO presents the accurate monolingual background.
C3_BI_CF presents the bilingual background on the same transcript and age.
True diagnosis and true group never appear in the prompt.
"""

from __future__ import annotations

import hashlib

from bisli_audit.prompts import BACKGROUND_BILINGUAL, BACKGROUND_MONOLINGUAL, render_prompt

PROMPT_FAMILY_VERSION = "v1_cf_label_swap"
LABELS = ("C3_MONO", "C3_BI_CF")

TEMPLATE_HASHES = {
    "C3_monolingual": "7ff6322e1d022d7c1566d95c2655102d1b4f87192d667dbb954c182dd55d5ade",
    "C3_bilingual": "5396a27e35a959eddfb5fa5e193fd702d87a30a9c7cd7abd8c1a491e00c9b28a",
}


def background_sentence(presented_background: str, language: str = "Russian") -> str:
    if presented_background == "monolingual":
        return BACKGROUND_MONOLINGUAL.format(language=language)
    if presented_background == "bilingual":
        return BACKGROUND_BILINGUAL.format(language=language)
    raise ValueError(presented_background)


def build_cf_prompt(
    transcript: str,
    age_months: str | float,
    label: str,
    language: str = "Russian",
) -> tuple[str, dict]:
    if label == "C3_MONO":
        group = "monolingual"
        presented = "monolingual"
        counterfactual = False
    elif label == "C3_BI_CF":
        group = "bilingual"
        presented = "bilingual"
        counterfactual = True
    else:
        raise ValueError(label)

    prompt = render_prompt(
        "C3",
        transcript,
        age_months=age_months,
        group=group,
        language=language,
    )
    template_key = f"C3_{group}"
    meta = {
        "label": label,
        "presented_background": presented,
        "counterfactual": counterfactual,
        "prompt_family_version": PROMPT_FAMILY_VERSION,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "template_hash": TEMPLATE_HASHES[template_key],
        "template_key": template_key,
        "matches_accurate_c3_render": group == "monolingual",
    }
    return prompt, meta


def extract_transcript_span(prompt: str) -> str:
    start = prompt.find("<transcript>")
    end = prompt.find("</transcript>")
    if start < 0 or end < 0:
        return ""
    return prompt[start + len("<transcript>") : end]


def prompt_diff_ok(prompt_mono: str, prompt_bi: str, language: str = "Russian") -> dict:
    """Check that only the background sentence differs between the pair."""
    tx_m = extract_transcript_span(prompt_mono)
    tx_b = extract_transcript_span(prompt_bi)
    mono_bg = background_sentence("monolingual", language)
    bi_bg = background_sentence("bilingual", language)
    swapped = prompt_mono.replace(mono_bg, bi_bg, 1)
    return {
        "transcript_spans_identical": tx_m == tx_b and bool(tx_m),
        "mono_contains_mono_bg": mono_bg in prompt_mono,
        "mono_lacks_bi_bg": bi_bg not in prompt_mono,
        "bi_contains_bi_bg": bi_bg in prompt_bi,
        "bi_lacks_mono_bg": mono_bg not in prompt_bi,
        "swap_mono_bg_yields_bi_prompt": swapped == prompt_bi,
    }
