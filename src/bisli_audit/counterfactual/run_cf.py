"""Run C3_MONO / C3_BI_CF counterfactual pairs on monolingual Russian narratives."""

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
from bisli_audit.counterfactual.prompts_cf import LABELS, build_cf_prompt

REQUIRED_CONTEXT = 16384
MAX_ATTEMPTS = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
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
        tx = (args.transcripts / f"{row.item_id}.txt").read_text(encoding="utf-8").strip()
        for label in LABELS:
            prompt, meta = build_cf_prompt(tx, float(row.age_months), label)
            jobs.append((row, label, prompt, meta))

    print(f"planned={len(jobs)} labels={LABELS}")
    if args.dry_run:
        return

    with args.out.open("a", encoding="utf-8") as fh:
        for row, label, prompt, meta in jobs:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                text, raw, _ = chat(args.model, prompt)
                status = parse_with_status(text)
                rec = {
                    "experiment": "counterfactual_c3",
                    "item_id": row.item_id,
                    "child_id": getattr(row, "child_id", None),
                    "diagnosis": getattr(row, "diagnosis", None),
                    "model": args.model,
                    "arm": label,
                    "attempt": attempt,
                    "decision": status["decision"],
                    "confidence": status["confidence"],
                    "parse_status": status["parse_status"],
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "prompt_meta": meta,
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
