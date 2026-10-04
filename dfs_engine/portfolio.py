"""
Build a portfolio of N lineups from simulated trials, with:
    - tiered exposure caps (ownership-tiered, DST specially tightened)
    - uniqueness constraints (no two lineups >= K overlapping players)

Strategy: draw one simulated trial per lineup slot, solve the MILP-optimal
lineup for that trial, and greedily accept it if it satisfies the running
exposure caps and uniqueness threshold; otherwise resolve against a
shrinking eligible pool. This mirrors the practical "Optimal%" portfolio
construction pattern (build from many independently-optimal trials,
then diversify) used by SaberSim-style tools.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .optimize import solve_classic, solve_showdown, LineupResult


def default_exposure_cap(own: float) -> float:
    """Tiered default max-exposure cap as a function of projected ownership."""
    if own >= 0.30:
        return 0.55
    if own >= 0.15:
        return 0.65
    if own >= 0.05:
        return 0.75
    return 0.85


def build_portfolio(
    df: pd.DataFrame,
    scores: np.ndarray,
    n_lineups: int,
    fmt: str = "classic",
    min_unique: int = 2,
    dst_cap: float = 0.22,
    max_attempts_per_slot: int = 40,
    seed: int | None = None,
) -> list[LineupResult]:
    """
    Parameters
    ----------
    df : player pool (same frame used to generate `scores`)
    scores : (n_trials, n_players) simulated score matrix from simulate.py
    n_lineups : how many lineups to build
    fmt : "classic" or "showdown"
    min_unique : minimum number of differing players required vs. every
                 already-accepted lineup
    dst_cap : max exposure specifically for DST (classic only)
    """
    rng = np.random.default_rng(seed)
    n_trials, n_players = scores.shape
    solver = solve_classic if fmt == "classic" else solve_showdown
    roster_size = 9 if fmt == "classic" else 6

    exposure_count = Counter()
    accepted: list[LineupResult] = []
    accepted_sets: list[set[int]] = []

    exposure_caps = df["own"].apply(default_exposure_cap).to_numpy()
    is_dst = (df["position"] == "DST").to_numpy()

    trial_order = rng.permutation(n_trials)
    trial_cursor = 0

    while len(accepted) < n_lineups and trial_cursor < n_trials:
        excluded: set[int] = set()
        result = None

        for _attempt in range(max_attempts_per_slot):
            if trial_cursor >= n_trials:
                break
            trial_idx = trial_order[trial_cursor]
            points = scores[trial_idx].copy()

            cap_mask = np.ones(n_players, dtype=bool)
            for pid in range(n_players):
                cap = dst_cap if is_dst[pid] else exposure_caps[pid]
                if exposure_count[pid] >= cap * max(n_lineups, 1):
                    cap_mask[pid] = False
            cap_mask[list(excluded)] = False

            candidate = solver(
                df, points,
                excluded_ids=excluded | set(np.where(~cap_mask)[0].tolist()),
            )
            trial_cursor += 1

            if candidate is None:
                continue

            cand_set = set(candidate.player_ids)
            if all(len(cand_set ^ prev) >= 2 * min_unique for prev in accepted_sets):
                result = candidate
                break
            else:
                # force diversification: exclude the most-repeated player
                overlap_counts = Counter()
                for prev in accepted_sets:
                    for pid in cand_set & prev:
                        overlap_counts[pid] += 1
                if overlap_counts:
                    worst = overlap_counts.most_common(1)[0][0]
                    excluded.add(worst)

        if result is None:
            break  # combinatorial ceiling reached — stop rather than loop forever

        accepted.append(result)
        accepted_sets.append(set(result.player_ids))
        for pid in result.player_ids:
            exposure_count[pid] += 1

    return accepted


def exposure_report(df: pd.DataFrame, lineups: list[LineupResult]) -> pd.DataFrame:
    n = len(lineups)
    counts = Counter()
    for lu in lineups:
        for pid in lu.player_ids:
            counts[pid] += 1
    rows = []
    for pid, cnt in counts.most_common():
        row = df.loc[pid]
        rows.append({
            "name": row["name"], "team": row["team"], "position": row["position"],
            "salary": row["salary"], "own_proj": row["own"],
            "exposure_n": cnt, "exposure_pct": cnt / n if n else 0.0,
        })
    return pd.DataFrame(rows)
