"""Child-clustered factorial contrasts for complete M0/B0/M1/B1 quadruples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bisli_audit.analysis.cluster_bootstrap import cluster_boot_ci

CELLS = ["M0", "B0", "M1", "B1"]
B = 10_000
SEED = 42


def is_refer(x) -> float:
    s = str(x).lower()
    if s == "refer":
        return 1.0
    if s in {"typical", "no_refer"}:
        return 0.0
    return np.nan


def load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def prefer_ok(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("attempt")
    out = []
    for _, g in df.groupby(["item_id", "arm"]):
        ok = g[g.parse_status == "ok"]
        out.append(ok.iloc[-1] if len(ok) else g.iloc[-1])
    return pd.DataFrame(out)


def wide_table(df: pd.DataFrame) -> pd.DataFrame:
    ok = prefer_ok(df)
    ok = ok[ok.parse_status == "ok"].copy()
    ok["refer"] = ok["decision"].map(is_refer)
    cell_col = "factorial_cell" if "factorial_cell" in ok.columns else "arm"
    wide = ok.pivot_table(index="item_id", columns=cell_col, values="refer", aggfunc="first")
    for c in CELLS:
        if c not in wide.columns:
            wide[c] = np.nan
    child_col = "child_id" if "child_id" in ok.columns else "person_id"
    meta = ok.drop_duplicates("item_id").set_index("item_id")[[child_col]].rename(
        columns={child_col: "child_id"}
    )
    wide = wide.join(meta).dropna(subset=CELLS)
    wide["simple_B_N0"] = wide["B0"] - wide["M0"]
    wide["simple_B_N1"] = wide["B1"] - wide["M1"]
    wide["interaction"] = wide["simple_B_N1"] - wide["simple_B_N0"]
    return wide


def summarize(wide: pd.DataFrame, col: str) -> dict:
    est, lo, hi = cluster_boot_ci(
        wide, wide[col], cluster_col="child_id", n_boot=B, seed=SEED
    )
    return {
        "estimate_pp": est * 100,
        "ci_low_pp": lo * 100,
        "ci_high_pp": hi * 100,
        "n_children": int(wide["child_id"].nunique()),
        "n_narratives": int(len(wide)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    args = ap.parse_args()
    wide = wide_table(load_jsonl(args.jsonl))
    out = {
        "n_narratives": int(len(wide)),
        "n_children": int(wide.child_id.nunique()),
        "rates": {c: float(wide[c].mean()) for c in CELLS},
        "refer_counts": {c: int(wide[c].sum()) for c in CELLS},
        "simple_B_N0": summarize(wide, "simple_B_N0"),
        "simple_B_N1": summarize(wide, "simple_B_N1"),
        "interaction": summarize(wide, "interaction"),
        "bootstrap": {"B": B, "seed": SEED},
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
