"""Child-clustered bootstrap contrasts for paired C2→C3 referral decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from bisli_audit.analysis.cluster_bootstrap import cluster_boot_ci

SEED = 42
B = 10_000


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def prefer_ok(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.sort_values("attempt")
    out = []
    for _, g in df.groupby(keys):
        ok = g[g.parse_status == "ok"]
        out.append(ok.iloc[-1] if len(ok) else g.iloc[-1])
    return pd.DataFrame(out)


def pair_c2_c3(df: pd.DataFrame) -> pd.DataFrame:
    ok = prefer_ok(df[df.parse_status == "ok"], ["item_id", "condition"])
    wide = ok.pivot_table(index="item_id", columns="condition", values="decision", aggfunc="first")
    meta_cols = [c for c in ("child_id", "person_id", "group", "diagnosis", "language") if c in ok.columns]
    meta = ok.drop_duplicates("item_id").set_index("item_id")[meta_cols]
    if "child_id" not in meta.columns and "person_id" in meta.columns:
        meta = meta.rename(columns={"person_id": "child_id"})
    out = wide.join(meta)
    for col in ("C2", "C3"):
        if col in out.columns:
            out[col] = (out[col].astype(str).str.lower() == "refer").astype(float)
    return out.dropna(subset=["C2", "C3"])


def contrast(paired: pd.DataFrame) -> dict:
    paired = paired.copy()
    paired["delta"] = paired["C3"] - paired["C2"]
    est, lo, hi = cluster_boot_ci(
        paired, paired["delta"], cluster_col="child_id", n_boot=B, seed=SEED
    )
    return {
        "estimate_pp": est * 100,
        "ci_low_pp": lo * 100,
        "ci_high_pp": hi * 100,
        "n_children": int(paired["child_id"].nunique()),
        "n_narratives": int(len(paired)),
        "rate_c2": float(paired["C2"].mean()),
        "rate_c3": float(paired["C3"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    paired = pair_c2_c3(load_jsonl(args.jsonl))
    result = {"contrast": "C2_to_C3", "overall": contrast(paired), "bootstrap": {"B": B, "seed": SEED}}
    if "group" in paired.columns:
        result["by_group"] = {
            str(g): contrast(sub) for g, sub in paired.groupby("group")
        }
    if "diagnosis" in paired.columns:
        result["by_diagnosis"] = {
            str(g): contrast(sub) for g, sub in paired.groupby("diagnosis")
        }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
