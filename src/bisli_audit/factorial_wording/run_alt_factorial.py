"""Factorial cells with alternative separate-sentence wording (paper: MedGemma SLI 244)."""

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
from bisli_audit.prompts import FACTORIAL_ALT, render_factorial

REQUIRED_CONTEXT = 16384
MAX_ATTEMPTS = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--order", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = load_info(args.model)
    if info.get("state") != "loaded" or info.get("loaded_context_length") != REQUIRED_CONTEXT:
        sys.exit(f"STOP: load {args.model} with context {REQUIRED_CONTEXT}: {info}")

    cohort = pd.read_csv(args.cohort).set_index("item_id")
    order = pd.read_csv(args.order)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    jobs = []
    for o in order.itertuples():
        row = cohort.loc[o.item_id]
        tx = (args.transcripts / f"{o.item_id}.txt").read_text(encoding="utf-8").strip()
        for pos, cell in enumerate(str(o.order).split(">")):
            prompt = render_factorial(cell, tx, float(row.age_months), alt=True)
            jobs.append((o.item_id, cell, pos, prompt, row))

    print("alt cells:", FACTORIAL_ALT)
    print(f"planned={len(jobs)} model={args.model}")
    if args.dry_run:
        return

    with args.out.open("a", encoding="utf-8") as fh:
        for item_id, cell, pos, prompt, row in jobs:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                text, raw, _ = chat(args.model, prompt)
                status = parse_with_status(text)
                rec = {
                    "experiment": "factorial_alt_separate_sentences",
                    "item_id": item_id,
                    "child_id": getattr(row, "child_id", None) or getattr(row, "person_id", None),
                    "diagnosis": getattr(row, "diagnosis", None),
                    "model": args.model,
                    "arm": cell,
                    "factorial_cell": cell,
                    "order_position": pos,
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
