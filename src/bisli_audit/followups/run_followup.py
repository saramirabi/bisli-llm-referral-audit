"""Follow-up experiments: wording robustness, generation stability, factorial background.

Expects local LM Studio at http://localhost:1234 with one model loaded.
Transcripts and cohort tables are supplied by the caller (BiSLI data are not
bundled; see README).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bisli_audit.client import EXTRA, SETTINGS, chat, load_info
from bisli_audit.prompts import (
    QUESTION,
    format_age,
    parse_response,
    render_factorial,
    render_prompt,
)

REQUIRED_CONTEXT = 16384
MAX_ATTEMPTS = 2


def background(row, cond: str) -> str | None:
    if cond != "C3":
        return None
    from bisli_audit.prompts import BACKGROUND_BILINGUAL, BACKGROUND_MONOLINGUAL

    tmpl = BACKGROUND_BILINGUAL if row.group == "bilingual" else BACKGROUND_MONOLINGUAL
    return tmpl.format(language=row.language)


def render_wording(template: str, row, cond: str, transcript: str) -> str:
    age = format_age(row.age_months)
    bg = background(row, cond)
    if template == "T0_original":
        return render_prompt(
            cond,
            transcript,
            age_months=row.age_months,
            group=row.group,
            language=row.language,
        )
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
            transcript,
            "</transcript>",
            QUESTION.replace(
                "Based only on this language sample, does the child's language look typical",
                "Using only this language sample, is the child's language typical",
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
            transcript,
            "</transcript>",
            QUESTION,
        ]
        return "\n".join(lines)
    raise ValueError(template)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "experiment",
        choices=["prompt_robustness", "generation_stability", "factorial_background"],
    )
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True, help="CSV with item metadata")
    ap.add_argument("--transcripts", type=Path, required=True, help="Directory of {item_id}.txt")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--order", type=Path, default=None, help="Factorial order CSV")
    ap.add_argument("--allow-context-8192", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = load_info(args.model)
    ctx = info.get("loaded_context_length")
    ok_ctx = ctx == REQUIRED_CONTEXT or (
        args.allow_context_8192 and ctx == 8192 and "aya" in args.model
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
        order = order[order.model == args.model]
        by_id = cohort.set_index("item_id")
        for o in order.itertuples():
            row = by_id.loc[o.item_id]
            tx = (args.transcripts / f"{o.item_id}.txt").read_text(encoding="utf-8").strip()
            for pos, cell in enumerate(str(o.order).split(">")):
                meta = {
                    "experiment": args.experiment,
                    "item_id": o.item_id,
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
                            "model": args.model,
                            "arm": cond,
                            "template": template,
                            "repetition": rep,
                        }
                        prompt = render_wording(template, row, cond, tx)
                        jobs.append((meta, prompt))

    print(f"planned={len(jobs)} model={args.model} load={info}")
    if args.dry_run:
        return

    with args.out.open("a", encoding="utf-8") as fh:
        for meta, prompt in jobs:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                text, raw, _body = chat(args.model, prompt)
                parsed = parse_response(text)
                rec = {
                    **meta,
                    "attempt": attempt,
                    "decision": None if parsed is None else parsed["decision"],
                    "confidence": None if parsed is None else parsed["confidence"],
                    "parse_status": "ok" if parsed else "invalid",
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
                if parsed:
                    break


if __name__ == "__main__":
    main()
