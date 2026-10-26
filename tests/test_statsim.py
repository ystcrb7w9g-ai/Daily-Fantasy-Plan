import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine.simulate import simulate_player_scores
from dfs_engine.statsim import stat_game_scores
from test_sim_tools import _multi_game_pool


def _showdown_with_k():
    rows = []
    for team, opp, tt in (("AAA", "BBB", 26.0), ("BBB", "AAA", 21.0)):
        for pos, sal, proj in (("QB", 11000, 21.0), ("RB", 9000, 16.0), ("RB", 4000, 6.0), ("WR", 9500, 17.0),
                               ("WR", 6000, 10.0), ("TE", 5000, 8.0), ("K", 4000, 8.0), ("DST", 3800, 6.5)):
            rows.append({"name": f"{team} {pos}{sal}", "team": team, "opp": opp, "position": pos, "salary": sal,
                         "proj": proj, "own": 0.3, "team_total": tt, "game_total": 47.0, "spread": 0.0})
    df = pd.DataFrame(rows)
    df["player_id"] = df.index
    return df


def _corr(sc, a, b):
    return np.corrcoef(sc[:, a], sc[:, b])[0, 1]


def test_stat_engine_box_scores_move_teammates_and_opponents_like_real_games():
    df = _showdown_with_k()
    raw = stat_game_scores(df, 20000, np.random.default_rng(3))
    ix = {n: i for i, n in enumerate(df["name"])}
    assert raw.shape == (20000, len(df)) and (raw[:, df["position"] != "DST"] >= -3).all()
    # means track projections (the box scores are built from each player's share of his team's volume)
    ratio = raw.mean(axis=0) / df["proj"].to_numpy()
    assert np.all((ratio > 0.6) & (ratio < 1.5))
    qb, wr, oqb = ix["AAA QB11000"], ix["AAA WR9500"], ix["BBB QB11000"]
    assert _corr(raw, qb, wr) > 0.25          # his passing yards and TDs are the receivers'
    assert _corr(raw, qb, oqb) > 0.05         # shared game environment
    assert _corr(raw, ix["AAA DST3800"], oqb) < -0.25   # a defense feeds on the opposing offense
    assert _corr(raw, ix["AAA K4000"], ix["BBB DST3800"]) < -0.1


def test_stat_engine_through_simulate_fits_each_player_to_proj_and_ceiling():
    df = _multi_game_pool()
    sc = simulate_player_scores(df, n_trials=20000, seed=5, engine="stat")
    assert np.allclose(sc.mean(axis=0) / df["proj"].to_numpy(), 1.0, atol=0.08)
    p85 = np.quantile(sc, 0.85, axis=0)
    assert np.allclose(p85 / df["ceiling"].to_numpy(), 1.0, atol=0.12)
    import pytest
    with pytest.raises(ValueError):
        simulate_player_scores(df, n_trials=100, seed=5, engine="nope")


def test_stat_engine_run_pass_tilt_default_off_and_splits_qb_rb_booms():
    df = _showdown_with_k()
    ix = {n: i for i, n in enumerate(df["name"])}
    a = stat_game_scores(df, 20000, np.random.default_rng(9))
    b = stat_game_scores(df, 20000, np.random.default_rng(9), {"td_mix_sd": 0.0, "share_tilt": 0.0})
    assert np.array_equal(a, b)
    c = stat_game_scores(df, 20000, np.random.default_rng(9), {"td_mix_sd": 0.9, "share_tilt": 0.06})
    qb, rb = ix["AAA QB11000"], ix["AAA RB9000"]

    def co_boom(sc, q=0.9):
        return ((sc[:, qb] >= np.quantile(sc[:, qb], q)) & (sc[:, rb] >= np.quantile(sc[:, rb], q))).mean() / (1 - q) ** 2
    assert co_boom(c) < co_boom(a) - 0.2
