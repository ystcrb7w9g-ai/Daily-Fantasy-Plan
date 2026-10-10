import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import contest as cs
from dfs_engine import history
from dfs_engine import review as rv
from dfs_engine.simulate import simulate_player_scores
from test_history import _standings_file


def test_contest_scorecard_field_and_our_results():
    raw = history.read_standings(_standings_file(tempfile.mkdtemp()))
    sc = rv.contest_scorecard(raw, {"1000"}, cs.PayoutCurve(np.array([50.0, 50.0, 0.0, 0.0]), 20.0))
    assert sc["entries"] == 4 and sc["winning_score"] == 200.0 and sc["median_score"] == 175.0
    assert sc["our_entries"] == 1 and sc["our_best_rank"] == 1
    # entries 1 and 2 tie at 200 points (both Rank 1 in DK's export) and split ranks 1-2
    assert sc["our_winnings"] == 50.0 and sc["our_profit"] == 30.0


def test_grade_inputs_and_actual_points():
    raw = history.read_standings(_standings_file(tempfile.mkdtemp()))
    actual = rv.actual_points([raw])
    pool = pd.DataFrame({"name": ["Josh Allen", "Joe Burrow", "James Cook", "Bills"],
                         "position": ["QB", "QB", "RB", "DST"], "proj": [28.0, 22.0, 18.0, 7.0],
                         "own": [40.0, 60.0, 80.0, 50.0]})
    g = rv.grade_inputs(pool, actual)
    assert g["matched"] == 4 and g["proj_corr"] > 0.9
    assert abs(g["proj_bias"] - np.mean([28 - 30, 22 - 20, 18 - 20, 7 - 8])) < 1e-9
    assert g["own_mae"] < 0.11


def test_sim_calibration_on_its_own_outcomes():
    from test_ownership_model import _six_game_pool
    df = _six_game_pool(1)
    df["name"] = [f"Player {chr(65 + i // 26)}{chr(65 + i % 26)}" for i in df.index]
    scores = simulate_player_scores(df, n_trials=3000, seed=1)
    truth = simulate_player_scores(df, n_trials=2000, seed=99)[7]  # (calibration needs many trials)
    actual = pd.DataFrame({"key": [rv._key(n, p) for n, p in zip(df["name"], df["position"])], "fpts": truth})
    cal = rv.sim_calibration(df, actual, scores)
    assert cal["players"] == len(df)
    assert 0.68 < cal["inside_10_90"] < 0.92 and 0.05 < cal["above_p85"] < 0.27
    med = actual.assign(fpts=np.median(scores, axis=0))
    assert rv.sim_calibration(df, med, scores)["inside_25_75"] == 1.0


def test_append_log_accumulates():
    path = os.path.join(tempfile.mkdtemp(), "sub", "log.csv")
    rv.append_log(path, [{"week": 1, "x": 1.0}])
    rv.append_log(path, [{"week": 2, "x": 2.0, "y": 3}])
    log = pd.read_csv(path)
    assert list(log["week"]) == [1, 2] and np.isnan(log.loc[0, "y"])
