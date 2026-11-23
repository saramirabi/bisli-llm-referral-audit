"""LM Studio OpenAI-compatible client (localhost only)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from bisli_audit.prompts import parse_response

DEFAULT_BASE = "http://localhost:1234/v1"
DEFAULT_MODELS_API = "http://localhost:1234/api/v0/models"
SETTINGS = {"temperature": 0, "seed": 42, "max_tokens": 8192}
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}, "ttl": 300}

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)
_DECISION_RE = re.compile(
    r'["\u201c\u201d]?\s*decision\s*["\u201c\u201d]?\s*:\s*["\u201c\u201d\'](typical|refer)["\u201c\u201d\']',
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r'["\u201c\u201d]?\s*confidence\s*["\u201c\u201d]?\s*:\s*(\d{1,3})',
    re.IGNORECASE,
)


def load_info(model: str, models_api: str = DEFAULT_MODELS_API) -> dict:
    with urllib.request.urlopen(models_api, timeout=30) as resp:
        data = json.loads(resp.read()).get("data", [])
    for item in data:
        if item.get("id") == model:
            return {
                k: item.get(k)
                for k in (
                    "id",
                    "state",
                    "loaded_context_length",
                    "max_context_length",
                    "quantization",
                    "arch",
                    "compatibility_type",
                )
            }
    return {"id": model, "state": "absent"}


def chat(
    model: str,
    prompt: str,
    *,
    base_url: str = DEFAULT_BASE,
    settings: dict | None = None,
    extra: dict | None = None,
    timeout: float = 900,
) -> tuple[str, dict, dict]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **(settings or SETTINGS),
        **(extra or EXTRA),
    }
    # GPT-OSS: keep reasoning short when using the instruct chat template.
    if "gpt-oss" in model.lower() and "reasoning_effort" not in body:
        body["reasoning_effort"] = "low"

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    text = _message_text(((out.get("choices") or [{}])[0].get("message") or {}).get("content"))
    return text, out, body


def strip_reasoning(text: str) -> str:
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    cleaned = _THINK_TAG.sub("", cleaned)
    return cleaned.strip()


def parse_with_status(text: str) -> dict:
    stripped = strip_reasoning(text)
    parsed = parse_response(stripped)
    if parsed is not None:
        return {
            "parse_status": "ok",
            "decision": parsed["decision"],
            "confidence": parsed["confidence"],
            "reasons": parsed["reasons"],
            "parsed": parsed,
            "error": None,
        }

    decision = None
    confidence = None
    match = _DECISION_RE.search(stripped or "")
    if match:
        decision = match.group(1).lower()
    conf_match = _CONFIDENCE_RE.search(stripped or "")
    if conf_match:
        value = int(conf_match.group(1))
        if 0 <= value <= 100:
            confidence = value

    if decision in {"typical", "refer"} and confidence is not None:
        return {
            "parse_status": "partial_parse",
            "decision": decision,
            "confidence": confidence,
            "reasons": None,
            "parsed": None,
            "error": "salvaged decision and confidence",
        }
    return {
        "parse_status": "failed",
        "decision": decision,
        "confidence": confidence,
        "reasons": None,
        "parsed": None,
        "error": "could not parse decision/confidence",
    }


def _message_text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text") or ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)
