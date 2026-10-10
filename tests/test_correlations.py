import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfs_engine import correlations as corr
from dfs_engine.history import load_profile
from dfs_engine.simulate import simulate_player_scores

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_dk_points_scoring_and_bonuses():
    s = pd.DataFrame({"passing_yards": [300.0], "passing_tds": [2], "passing_interceptions": [1],
                      "rushing_yards": [20.0], "receptions": [0], "rushing_fumbles_lost": [1]})
    # 12 + 8 - 1 + 3 (300-yd bonus) + 2 - 1
    assert abs(corr.dk_points(s).iloc[0] - 23.0) < 1e-9


def _weekly_csv(path, seed=0):
    """Fake nflverse weekly file: QB tied to his WR1, RBs splitting work."""
    rng = np.random.default_rng(seed)
    rows = []
    for week in range(1, 13):
        for team, opp in (("AAA", "BBB"), ("BBB", "AAA")):
            pass_yds = rng.normal(240, 60)
            wr1 = pass_yds * 0.4 + rng.normal(0, 15)
            carries = rng.normal(100, 20)
            share = rng.uniform(0.5, 0.9)
            base = {"season": 2024, "week": week, "season_type": "REG", "team": team, "opponent_team": opp, "rushing_tds": 0}
            rows += [dict(base, player_id=f"{team}qb", position="QB", passing_yards=pass_yds, passing_tds=1),
                     dict(base, player_id=f"{team}wr1", position="WR", receiving_yards=wr1, receptions=5),
                     dict(base, player_id=f"{team}wr2", position="WR", receiving_yards=pass_yds * 0.2, receptions=3),
                     dict(base, player_id=f"{team}rb1", position="RB", rushing_yards=carries * share),
                     dict(base, player_id=f"{team}rb2", position="RB", rushing_yards=carries * (1 - share)),
                     dict(base, player_id=f"{team}te", position="TE", receiving_yards=30.0, receptions=3)]
    pd.DataFrame(rows).fillna(0).to_csv(path, index=False)


def test_measure_finds_roles_and_correlations():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "w.csv")
    _weekly_csv(p)
    prof = corr.measure([p])
    assert prof["team_games"] == 24
    assert prof["same_team"]["QB1~WR1"]["corr"] > 0.5
    assert prof["same_team"]["RB1~RB2"]["corr"] < 0.0
    assert set(prof["role_means"]) >= {"QB1", "RB1", "RB2", "WR1", "WR2", "TE1"}


def test_shipped_profile_and_sim_defaults_agree():
    prof = load_profile(os.path.join(ROOT, "data", "correlation_profile.json"))
    df = corr.synthetic_slate(prof["role_means"], n_games=4)
    sim = corr.simulated_correlations(df, simulate_player_scores(df, n_trials=3000, seed=5))
    assert corr.correlation_error(prof, sim) < 0.1
    st = sim["same_team"]
    assert st["QB1~WR1"]["corr"] > 0.25 and st["RB1~RB2"]["corr"] < 0.0
    assert st["WR1~WR2"]["corr"] < st["QB1~WR1"]["corr"] - 0.2
    assert sim["opponent"]["QB1~opp QB1"]["corr"] > 0.05


def test_fit_simulation_prefers_closer_knobs():
    prof = load_profile(os.path.join(ROOT, "data", "correlation_profile.json"))
    best, err, results = corr.fit_simulation(prof, {"qb_catcher_link": [0.0, 0.85]}, n_trials=1500)
    assert best["qb_catcher_link"] == 0.85 and err == min(r[1] for r in results)


def test_simulate_handles_qbs_at_one_salary():
    prof = load_profile(os.path.join(ROOT, "data", "correlation_profile.json"))
    df = corr.synthetic_slate(prof["role_means"], n_games=2)
    assert simulate_player_scores(df, n_trials=50, seed=1).shape == (50, len(df))


def test_profiles_drive_qb_links_like_real_games():
    prof = load_profile(os.path.join(ROOT, "data", "correlation_profile.json"))
    df = corr.hetero_slate(prof)
    sc = simulate_player_scores(df, n_trials=3000, seed=2)
    het = corr.simulated_heterogeneity(df, sc, prof)
    rush = [b["corr"] for b in het["QB1~WR1 by QB1 rush_share"]]
    wr = [b["corr"] for b in het["QB1~WR1 by WR1 tgt_pg"]]
    rb = [b["corr"] for b in het["QB1~RB1 by RB1 tgt_pg"]]
    assert rush[0] > rush[2] + 0.05          # pocket passers tie to their WR1 more than running QBs
    assert wr[2] > wr[0] + 0.05              # high-target WR1s tie to their QB more
    assert rb[2] > rb[0] and max(rb) < 0.25  # pass-catching backs a little, never like a WR
    assert corr.heterogeneity_error(prof, het) < 0.08
    # without profiles the sim falls back to role-only links (no heterogeneity)
    flat = df.drop(columns=["tgt_pg", "rush_share"])
    het0 = corr.simulated_heterogeneity(df, simulate_player_scores(flat, n_trials=3000, seed=2), prof)
    r0 = [b["corr"] for b in het0["QB1~WR1 by QB1 rush_share"]]
    assert abs(r0[0] - r0[2]) < abs(rush[0] - rush[2])


def test_attach_profiles_season_to_date():
    from dfs_engine import ownership_model as om
    weekly = pd.DataFrame({"season": [2026, 2026, 2026], "week": [1, 2, 3], "season_type": "REG",
                           "player_display_name": ["Run Qb"] * 3, "rushing_yards": [100.0, 0.0, 50.0],
                           "rushing_tds": [0, 0, 0], "targets": [0, 0, 0], "carries": [10, 0, 5],
                           "passing_yards": [100.0, 250.0, 0.0]})
    df = pd.DataFrame({"name": ["Run QB"], "position": ["QB"]})
    out = om.attach_profiles(df, weekly, 2026, 3)  # weeks 1-2 only
    assert abs(out.loc[0, "car_pg"] - 5.0) < 1e-9
    assert abs(out.loc[0, "rush_share"] - 10.0 / (10.0 + 4.0 + 10.0 + 3.0)) < 1e-9  # 100-yd bonus is not rushing share
