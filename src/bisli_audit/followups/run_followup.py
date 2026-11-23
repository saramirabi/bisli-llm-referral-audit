"""Follow-up experiments: wording robustness, generation stability, factorial cells."""

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
from bisli_audit.prompts import render_factorial, render_wording

REQUIRED_CONTEXT = 16384
MAX_ATTEMPTS = 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "experiment",
        choices=["prompt_robustness", "generation_stability", "factorial_background"],
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--order", type=Path, default=None, help="Factorial order CSV")
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
    jobs: list[tuple[dict, str]] = []

    if args.experiment == "factorial_background":
        if args.order is None:
            sys.exit("--order required for factorial_background")
        order = pd.read_csv(args.order)
        order = order[order.model == args.model] if "model" in order.columns else order
        by_id = cohort.set_index("item_id")
        for o in order.itertuples():
            row = by_id.loc[o.item_id]
            tx = (args.transcripts / f"{o.item_id}.txt").read_text(encoding="utf-8").strip()
            for pos, cell in enumerate(str(o.order).split(">")):
                meta = {
                    "experiment": args.experiment,
                    "item_id": o.item_id,
                    "child_id": getattr(row, "child_id", None),
                    "diagnosis": getattr(row, "diagnosis", None),
                    "model": args.model,
                    "arm": cell,
                    "factorial_cell": cell,
                    "order_position": pos,
                }
                prompt = render_factorial(cell, tx, float(row.age_months), alt=False)
                jobs.append((meta, prompt))
    else:
        templates = (
            ["T0_original", "T1_equivalent_a", "T2_equivalent_b"]
            if args.experiment == "prompt_robustness"
            else ["T0_original"]
        )
        reps = 1 if args.experiment == "prompt_robustness" else 3
        for rep in range(reps):
            for row in cohort.itertuples():
                tx = (args.transcripts / f"{row.item_id}.txt").read_text(encoding="utf-8").strip()
                for template in templates:
                    for cond in ["C2", "C3"]:
                        meta = {
                            "experiment": args.experiment,
                            "item_id": row.item_id,
                            "child_id": getattr(row, "child_id", None),
                            "group": getattr(row, "group", None),
                            "language": getattr(row, "language", None),
                            "diagnosis": getattr(row, "diagnosis", None),
                            "model": args.model,
                            "arm": cond,
                            "condition": cond,
                            "template": template,
                            "repetition": rep,
                        }
                        prompt = render_wording(
                            template,
                            tx,
                            age_months=float(row.age_months),
                            condition=cond,
                            group=getattr(row, "group", None),
                            language=getattr(row, "language", None),
                        )
                        jobs.append((meta, prompt))

    print(f"planned={len(jobs)} model={args.model} experiment={args.experiment}")
    if args.dry_run:
        return

    with args.out.open("a", encoding="utf-8") as fh:
        for meta, prompt in jobs:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                text, raw, _ = chat(args.model, prompt)
                status = parse_with_status(text)
                rec = {
                    **meta,
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
