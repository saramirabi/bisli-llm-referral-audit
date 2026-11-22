"""Child-clustered factorial contrasts for complete M0/B0/M1/B1 quadruples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CELLS = ["M0", "B0", "M1", "B1"]
B = 10_000
SEED = 42


def is_refer(x) -> int:
    s = str(x).lower()
    if s == "refer":
        return 1
    if s in {"typical", "no_refer"}:
        return 0
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


def child_boot(wide: pd.DataFrame, col: str) -> dict:
    children = wide["child_id"].dropna().unique()
    rng = np.random.default_rng(SEED)
    point = float(wide[col].mean())
    boots = []
    for _ in range(B):
        draw = rng.choice(children, size=len(children), replace=True)
        parts = [wide[wide.child_id == c] for c in draw]
        boots.append(float(pd.concat(parts)[col].mean()))
    arr = np.asarray(boots, dtype=float)
    return {
        "estimate_pp": point * 100,
        "ci_low_pp": float(np.percentile(arr, 2.5)) * 100,
        "ci_high_pp": float(np.percentile(arr, 97.5)) * 100,
        "n_children": int(len(children)),
        "n_narratives": int(len(wide)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jsonl", type=Path, required=True)
    args = ap.parse_args()
    wide = wide_table(load_jsonl(args.jsonl))
    rates = {c: float(wide[c].mean()) for c in CELLS}
    counts = {c: int(wide[c].sum()) for c in CELLS}
    out = {
        "n_narratives": int(len(wide)),
        "n_children": int(wide.child_id.nunique()),
        "rates": rates,
        "refer_counts": counts,
        "simple_B_N0": child_boot(wide, "simple_B_N0"),
        "simple_B_N1": child_boot(wide, "simple_B_N1"),
        "interaction": child_boot(wide, "interaction"),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
