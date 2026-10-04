"""
Per-player outcome distribution report (Stokastic-style projections table).

Summarizes the simulated score matrix from `simulate.py` into the columns
DFS sim tools show next to each player: median, ceiling percentiles,
standard deviation, and salary-relative boom / bust rates. Optionally
merges in Optimal% / leverage from `diagnostics.py`.

Boom and bust are defined relative to salary (DraftKings "value"):
    boom  = score >= boom_mult * salary / 1000   (default 4x, a GPP-winning pace)
    bust  = score <  bust_mult * salary / 1000   (default 2x, below cash pace)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def player_outcomes(
    df: pd.DataFrame,
    scores: np.ndarray,
    boom_mult: float = 4.0,
    bust_mult: float = 2.0,
    optimal_counts: np.ndarray | None = None,
    optimal_trials: int | None = None,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : player pool used to generate `scores`
    scores : (n_trials, n_players) simulated fantasy points
    optimal_counts / optimal_trials : optional output of
        `run_optimal_pct_chunk`, to add `optimal_pct` and `leverage`
    """
    value = df["salary"].to_numpy(dtype=float) / 1000.0
    p50, p85, p99 = np.percentile(scores, [50, 85, 99], axis=0)

    out = df[["name", "team", "opp", "position", "salary", "own", "proj"]].copy()
    if "ceiling" in df.columns:
        out["ceiling_input"] = df["ceiling"]
    out["sim_mean"] = scores.mean(axis=0)
    out["sim_median"] = p50
    out["sim_p85"] = p85
    out["sim_p99"] = p99
    out["sim_std"] = scores.std(axis=0)
    out["boom_pct"] = (scores >= boom_mult * value[None, :]).mean(axis=0)
    out["bust_pct"] = (scores < bust_mult * value[None, :]).mean(axis=0)
    out["pts_per_k"] = out["sim_mean"] / value

    if optimal_counts is not None:
        out["optimal_pct"] = optimal_counts / max(optimal_trials or 0, 1)
        out["leverage"] = out["optimal_pct"] - out["own"]

    num_cols = out.select_dtypes("number").columns.drop(["salary"])
    out[num_cols] = out[num_cols].round(3)
    sort_col = "leverage" if "leverage" in out.columns else "boom_pct"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)
