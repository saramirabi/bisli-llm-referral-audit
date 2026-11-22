"""LM Studio OpenAI-compatible chat client (localhost only)."""

from __future__ import annotations

import json
import urllib.request

DEFAULT_BASE = "http://localhost:1234/v1"
DEFAULT_MODELS_API = "http://localhost:1234/api/v0/models"
SETTINGS = {"temperature": 0, "seed": 42, "max_tokens": 8192}
EXTRA = {"chat_template_kwargs": {"enable_thinking": False}, "ttl": 300}


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
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], out, body
