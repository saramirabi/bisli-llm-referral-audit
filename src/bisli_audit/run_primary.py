"""Accurate-disclosure experiment: C1/C2/C3 against local LM Studio."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bisli_audit.client import EXTRA, SETTINGS, chat, load_info, parse_with_status
from bisli_audit.prompts import CONDITIONS, PROMPT_VERSION, render_prompt

REQUIRED_CONTEXT = 16384
MAX_ATTEMPTS = 2


def eligible(row) -> list[str]:
    age = getattr(row, "age_months", None)
    if age is None or (isinstance(age, float) and pd.isna(age)) or str(age).strip() == "":
        return ["C1"]
    return list(CONDITIONS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--allow-context-8192", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = load_info(args.model)
    ctx = info.get("loaded_context_length")
    ok_ctx = ctx == REQUIRED_CONTEXT or (
        args.allow_context_8192 and ctx == 8192 and "aya" in args.model.lower()
    )
    if info.get("state") != "loaded" or not ok_ctx:
        sys.exit(f"STOP: load check failed for {args.model}: {info}")

    cohort = pd.read_csv(args.cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in cohort.itertuples():
        transcript = (args.transcripts / f"{row.item_id}.txt").read_text(encoding="utf-8").strip()
        for cond in eligible(row):
            if cond not in args.conditions:
                continue
            prompt = render_prompt(
                cond,
                transcript,
                age_months=getattr(row, "age_months", None),
                group=getattr(row, "group", None),
                language=getattr(row, "language", None),
            )
            jobs.append((row, cond, prompt))

    print(f"prompt_version={PROMPT_VERSION} planned={len(jobs)} model={args.model}")
    if args.dry_run:
        return

    with args.out.open("a", encoding="utf-8") as fh:
        for row, cond, prompt in jobs:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                text, raw, _ = chat(args.model, prompt)
                status = parse_with_status(text)
                rec = {
                    "experiment": "disclosure",
                    "item_id": row.item_id,
                    "child_id": getattr(row, "child_id", None) or getattr(row, "person_id", None),
                    "group": getattr(row, "group", None),
                    "language": getattr(row, "language", None),
                    "diagnosis": getattr(row, "diagnosis", None),
                    "model": args.model,
                    "condition": cond,
                    "prompt_version": PROMPT_VERSION,
                    "attempt": attempt,
                    "decision": status["decision"],
                    "confidence": status["confidence"],
                    "parse_status": status["parse_status"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "raw_text": text,
                    "latency_s": round(time.time() - t0, 3),
                    "settings": SETTINGS,
                    "extra": EXTRA,
                    "load_info": info,
                    "usage": raw.get("usage"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                if status["parse_status"] == "ok":
                    break


if __name__ == "__main__":
    main()
