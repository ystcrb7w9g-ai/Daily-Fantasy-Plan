"""
Weekly review: what each contest looked like, how our inputs and
simulation held up against what actually happened, and how our entries
did. Run it on the contest-standings exports every Monday; each run
appends one row per contest to a results log, so real ROI by contest
type and model accuracy accumulate over the season.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .history import player_table
from .sportsgameodds import normalize_name


def _key(name: str, position) -> str:
    k = normalize_name(name)
    return "DST|" + k.split()[-1] if str(position).upper() == "DST" and k else k


def _entries(raw: pd.DataFrame) -> pd.DataFrame:
    e = raw[raw["EntryId"].notna() & raw["Points"].notna()].copy()
    e["EntryId"] = e["EntryId"].astype("int64").astype(str)
    e["Rank"] = pd.to_numeric(e["Rank"], errors="coerce")
    e["n_entries"] = pd.to_numeric(e["EntryName"].astype(str).str.extract(r"\(\d+/(\d+)\)")[0],
                                   errors="coerce").fillna(1)
    return e


def contest_scorecard(raw: pd.DataFrame, our_ids: set[str] | None = None, payout=None) -> dict:
    """
    Field composition and difficulty, plus our entries' finishes. `payout`
    (contest.PayoutCurve) prices our ranks; ties share the prize the way
    the rank window does in the contest sim.
    """
    e = _entries(raw)
    n = len(e)
    pts = e["Points"].astype(float).sort_values(ascending=False).to_numpy()
    top1 = e["Rank"] <= max(1, int(0.01 * n))
    heavy = e["n_entries"] >= 150
    casual = e["n_entries"] <= 3
    out = {
        "entries": n, "accounts": int(e["EntryName"].astype(str).str.replace(r"\s*\(\d+/\d+\)$", "", regex=True).nunique()),
        "max_entry_share": round(float(heavy.mean()), 4), "casual_share": round(float(casual.mean()), 4),
        "max_entry_top1_rate": round(float(top1[heavy].mean()), 4) if heavy.any() else np.nan,
        "casual_top1_rate": round(float(top1[casual].mean()), 4) if casual.any() else np.nan,
        "winning_score": float(pts[0]), "top1_score": float(pts[max(0, int(0.01 * n) - 1)]),
        "top20_score": float(pts[int(0.2 * n)]), "median_score": float(np.median(pts)),
    }
    out["pro_edge"] = round(out["max_entry_top1_rate"] / out["casual_top1_rate"], 2) \
        if out["casual_top1_rate"] and np.isfinite(out["max_entry_top1_rate"]) else np.nan
    mine = e[e["EntryId"].isin(our_ids or set())]
    out["our_entries"] = int(len(mine))
    if len(mine):
        out["our_best_rank"] = int(mine["Rank"].min())
        out["our_best_pct"] = round(float(mine["Rank"].min() / n), 5)
        out["our_median_pct"] = round(float((mine["Rank"] / n).median()), 4)
        out["our_top1"] = int((mine["Rank"] <= max(1, int(0.01 * n))).sum())
        out["our_best_points"] = float(mine["Points"].astype(float).max())
        if payout is not None:
            ranks = e["Rank"].astype(int)
            ties = ranks.value_counts()
            won = 0.0
            for r in mine["Rank"].astype(int):
                won += float(payout.average_payout(np.array([r - 1.0]), np.array([float(ties[r])]))[0])
            out["our_fees"] = round(payout.entry_fee * len(mine), 2)
            out["our_winnings"] = round(won, 2)
            out["our_profit"] = round(won - payout.entry_fee * len(mine), 2)
    return out


def actual_points(raw_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Every drafted player's actual DK points and %Drafted (max over the given contests), by name key."""
    rows = {}
    for raw in raw_list:
        for r in player_table(raw).itertuples(index=False):
            k = _key(r.name, r.position)
            prev = rows.get(k)
            rows[k] = {"key": k, "fpts": float(r.fpts) if pd.notna(r.fpts) else np.nan,
                       "actual_own": max(float(r.own), prev["actual_own"] if prev else 0.0)}
    return pd.DataFrame(list(rows.values()))


def grade_inputs(pool: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Projection and ownership accuracy (provided `own`, and `own_model` if present) on matched players."""
    cols = ["name", "position", "proj"] + [c for c in ("own", "own_model") if c in pool]
    p = pool[cols].assign(key=[_key(n, ps) for n, ps in zip(pool["name"], pool["position"])])
    m = p.merge(actual, on="key")
    m = m[m["proj"].notna() & m["fpts"].notna()]
    out = {"matched": int(len(m)),
           "proj_corr": round(float(np.corrcoef(m["proj"], m["fpts"])[0, 1]), 3),
           "proj_bias": round(float((m["proj"] - m["fpts"]).mean()), 2),
           "proj_bias_by_pos": {k: round(float((g["proj"] - g["fpts"]).mean()), 2) for k, g in m.groupby("position")}}
    for col in ("own", "own_model"):
        if col in m and m[col].notna().any():
            o = pd.to_numeric(m[col], errors="coerce").fillna(0)
            o = o / 100.0 if o.max() > 1.5 else o
            out[f"{col}_corr"] = round(float(np.corrcoef(o, m["actual_own"])[0, 1]), 3)
            out[f"{col}_mae"] = round(float((o - m["actual_own"]).abs().mean()), 4)
    return out


def sim_calibration(pool: pd.DataFrame, actual: pd.DataFrame, scores: np.ndarray) -> dict:
    """
    Did real scores land where the sim said? For a well-calibrated sim, 80%
    of players finish inside their simulated 10-90% range, 50% inside
    25-75%, 15% above their 85th percentile (the `ceiling` the sim is
    fitted to) and 50% above their median. Too few outside the ranges means
    the sim is too wide; too many, too narrow.
    """
    keys = np.array([_key(n, ps) for n, ps in zip(pool["name"], pool["position"])])
    fpts = pd.Series(actual.set_index("key")["fpts"]).reindex(keys).to_numpy()
    ok = np.isfinite(fpts)
    q = np.percentile(scores[:, ok], [10, 25, 50, 75, 85, 90], axis=0)
    a = fpts[ok]
    out = {"players": int(ok.sum()),
           "inside_10_90": round(float(((a >= q[0]) & (a <= q[5])).mean()), 3),
           "inside_25_75": round(float(((a >= q[1]) & (a <= q[3])).mean()), 3),
           "above_p85": round(float((a > q[4]).mean()), 3),
           "above_median": round(float((a > q[2]).mean()), 3)}
    pos = pool["position"].to_numpy()[ok]
    out["above_p85_by_pos"] = {p: round(float((a[pos == p] > q[4][pos == p]).mean()), 3)
                               for p in ("QB", "RB", "WR", "TE", "DST") if (pos == p).any()}
    return out


def append_log(path: str, rows: list[dict]) -> None:
    """Append flat result rows to a CSV log (created with a header on first use)."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)
