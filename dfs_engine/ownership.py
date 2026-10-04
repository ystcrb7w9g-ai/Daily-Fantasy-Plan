"""
Projected-ownership sanity checks and a fallback estimator.

Both lean on real field data: `data/field_profile.json` holds, per
position, the average ownership of the most-owned, 2nd-most-owned, ...
player across three 2026 Millionaire Makers (built by `field-study`
from DraftKings contest-standings exports).

* `check_ownership` compares a slate's ownership projections to those
  curves and flags what looks off (totals not near 900%, a position far
  flatter or chalkier than real fields).
* `estimate_ownership` is a fallback when no ownership projection is
  available: rank players within each position by a value score
  (points per $1k, projection, team implied total) and give the k-th
  ranked player the real fields' k-th-ranked ownership, rescaled to the
  slate. It captures how concentrated ownership is, but not news or
  narrative, and the value score's weights are judgment, not fitted
  (the contest exports have no salaries or pre-lock projections) -- a
  published projection should be preferred, or blended with this.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .history import load_profile

PROFILE_PATH = Path(__file__).resolve().parents[1] / "data" / "field_profile.json"
POSITIONS = ("QB", "RB", "WR", "TE", "DST")
BASE_SLOTS = {"QB": 1.0, "RB": 2.0, "WR": 3.0, "TE": 1.0, "DST": 1.0}


def _profile(profile: dict | None) -> dict:
    return profile if profile is not None else load_profile(str(PROFILE_PATH))


def position_totals(profile: dict | None = None) -> dict[str, float]:
    """Expected ownership total per position: base slots + its share of FLEX."""
    flex = _profile(profile)["flex_split"]
    return {p: BASE_SLOTS[p] + flex.get(p, 0.0) for p in POSITIONS}


def _zscore(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0.0


def value_score(df: pd.DataFrame) -> pd.Series:
    """Within-position chalk score: value (pts per $1k) first, then projection, then team total."""
    out = pd.Series(0.0, index=df.index)
    for p, g in df.groupby("position"):
        score = 1.0 * _zscore(g["proj"] / (g["salary"] / 1000.0)) + 0.5 * _zscore(g["proj"])
        if "team_total" in g and g["team_total"].notna().all():
            score = score + 0.25 * _zscore(g["team_total"])
        out[g.index] = score
    return out


def estimate_ownership(df: pd.DataFrame, profile: dict | None = None) -> pd.Series:
    """Fraction-owned estimate per player (players without a projection get 0)."""
    prof = _profile(profile)
    totals = position_totals(prof)
    own = pd.Series(0.0, index=df.index)
    has_proj = df["proj"].notna() & (df["proj"] > 0)
    score = value_score(df[has_proj])
    for p in POSITIONS:
        idx = score[df.loc[has_proj, "position"] == p].sort_values(ascending=False).index
        if not len(idx):
            continue
        curve = np.asarray(prof["own_curves"][p], dtype=float)
        tail_n = max(0, len(idx) - len(curve))
        tail = curve[-1] * 0.7 ** np.arange(1, tail_n + 1)  # beyond the curve: fading long tail
        vals = np.concatenate([curve, tail])[: len(idx)]
        own[idx] = vals * totals[p] / vals.sum()
    return own.clip(upper=0.95)


def check_ownership(df: pd.DataFrame, profile: dict | None = None) -> list[str]:
    """Human-readable warnings about a slate's ownership column vs real Milly fields."""
    prof = _profile(profile)
    totals = position_totals(prof)
    own = df["own"].fillna(0).astype(float)
    if own.max() > 1.5:
        own = own / 100.0
    notes = []
    total = own.sum()
    if not 7.5 <= total <= 10.5:
        notes.append(f"ownership sums to {total:.0%} (should be ~900% for a full Classic slate)")
    for p in POSITIONS:
        o = own[df["position"] == p].sort_values(ascending=False).to_numpy()
        if not len(o):
            continue
        # Real-Milly chalk level, rescaled to this slate's number of players
        # at the position (fewer options -> naturally more concentrated).
        curve = np.asarray(prof["own_curves"][p], dtype=float)[: len(o)]
        expected_top = curve[0] * totals[p] / curve.sum()
        if abs(o.sum() / totals[p] - 1) > 0.25:
            notes.append(f"{p} ownership sums to {o.sum():.0%}; real fields put ~{totals[p]:.0%} there")
        if o[0] < 0.5 * expected_top:
            notes.append(f"top {p} is only {o[0]:.0%} owned; real Millys suggest ~{expected_top:.0%} "
                         "for the chalkiest on a slate this size -- projection may be too flat")
        elif o[0] > 1.8 * expected_top:
            notes.append(f"top {p} is {o[0]:.0%} owned vs ~{expected_top:.0%} in real Millys "
                         "on a slate this size")
    return notes
