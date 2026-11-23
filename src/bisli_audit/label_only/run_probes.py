"""Single-cue ablation probes: M0 vs full C3 bilingual sentence, and M0 vs B0.

On Russian monolingual narratives (paper: 516 transcripts).
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

from bisli_audit.client import EXTRA, SETTINGS, chat, load_info, parse_with_status
from bisli_audit.label_only.prompts_lo import build_label_only_prompt
from bisli_audit.prompts import BACKGROUND_BILINGUAL, render_factorial, render_prompt

REQUIRED_CONTEXT = 16384
MAX_ATTEMPTS = 2
ARMS = ("M0", "B0", "C3_FULL_BI")


def build_prompt(arm: str, transcript: str, age_months: float) -> str:
    if arm == "M0":
        return render_factorial("M0", transcript, age_months, alt=False)
    if arm == "B0":
        prompt, _ = build_label_only_prompt(transcript, age_months, "BI_LABEL")
        return prompt
    if arm == "C3_FULL_BI":
        # Full accurate-style bilingual C3 sentence on a monolingual Russian transcript.
        return render_prompt(
            "C3",
            transcript,
            age_months=age_months,
            group="bilingual",
            language="Russian",
        )
    raise ValueError(arm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = load_info(args.model)
    if info.get("state") != "loaded" or info.get("loaded_context_length") != REQUIRED_CONTEXT:
        if not (info.get("loaded_context_length") == 8192 and "aya" in args.model.lower()):
            sys.exit(f"STOP: load check failed for {args.model}: {info}")

    cohort = pd.read_csv(args.cohort)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for row in cohort.itertuples():
        tx = (args.transcripts / f"{row.item_id}.txt").read_text(encoding="utf-8").strip()
        for arm in ARMS:
            jobs.append((row, arm, build_prompt(arm, tx, float(row.age_months))))

    print(f"planned={len(jobs)} arms={ARMS}")
    assert BACKGROUND_BILINGUAL  # imported for documentation parity with paper
    if args.dry_run:
        return

    with args.out.open("a", encoding="utf-8") as fh:
        for row, arm, prompt in jobs:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                t0 = time.time()
                text, raw, _ = chat(args.model, prompt)
                status = parse_with_status(text)
                rec = {
                    "experiment": "ablation_probes",
                    "item_id": row.item_id,
                    "child_id": getattr(row, "child_id", None),
                    "diagnosis": getattr(row, "diagnosis", None),
                    "model": args.model,
                    "arm": arm,
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
