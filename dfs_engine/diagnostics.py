"""
Optimal% diagnostic (SaberSim / Dave Bergman style leverage analysis).

For each of many independent simulated trials, solve the MILP-optimal
lineup for that trial, then tally how often each player appears in the
optimal lineup across trials. Comparing Optimal% to projected Own% gives
a genuine leverage signal: a player who is optimal far more often than
he's owned is underpriced/under-owned (positive leverage); the reverse
is a "chalk trap."

This is intentionally checkpointable: run in chunks and accumulate
counts into an .npz file so a 50,000-trial run can span multiple
tool-call / process boundaries without losing progress.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .optimize import solve_classic, solve_showdown


def run_optimal_pct_chunk(
    df: pd.DataFrame,
    scores_chunk: np.ndarray,
    fmt: str = "classic",
) -> np.ndarray:
    """
    Run the Optimal% diagnostic over one chunk of trials.
    Returns an integer count array of shape (n_players,): how many
    trials in this chunk had each player in the optimal lineup.
    """
    n_players = scores_chunk.shape[1]
    counts = np.zeros(n_players, dtype=int)
    solver = solve_classic if fmt == "classic" else solve_showdown

    for trial_points in scores_chunk:
        result = solver(df, trial_points)
        if result is not None:
            counts[result.player_ids] += 1

    return counts


def save_checkpoint(path: str, counts: np.ndarray, trials_done: int) -> None:
    np.savez(path, counts=counts, trials_done=trials_done)


def load_checkpoint(path: str) -> tuple[np.ndarray, int]:
    data = np.load(path)
    return data["counts"], int(data["trials_done"])


def leverage_report(df: pd.DataFrame, counts: np.ndarray, trials_done: int) -> pd.DataFrame:
    optimal_pct = counts / max(trials_done, 1)
    out = df[["name", "team", "position", "salary", "own", "proj", "ceiling"]].copy()
    out["optimal_pct"] = optimal_pct
    out["leverage"] = out["optimal_pct"] - out["own"]
    return out.sort_values("leverage", ascending=False).reset_index(drop=True)
