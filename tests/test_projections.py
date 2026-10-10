import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dfs_engine import projections as pj

TEAMS = [("AAA", "BBB"), ("CCC", "DDD")]


def _games(seasons=(2024, 2025), weeks=8):
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            for i, (h, a) in enumerate(TEAMS):
                total = 50.0 if i == 0 else 38.0
                rows.append({"season": s, "week": w, "game_type": "REG", "home_team": h, "away_team": a,
                             "total_line": total, "spread_line": 3.0, "home_score": total / 2 + 2, "away_score": total / 2 - 2})
    return pd.DataFrame(rows)


def _weekly(seasons=(2024, 2025), weeks=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for w in range(1, weeks + 1):
            for h, a in TEAMS:
                for team in (h, a):
                    base = {"season": s, "week": w, "season_type": "REG", "team": team}
                    rows.append(dict(base, player_display_name=f"Star Wr {team}", position="WR",
                                     targets=10, receptions=7, receiving_yards=rng.normal(95, 15)))
                    rows.append(dict(base, player_display_name=f"Depth Wr {team}", position="WR",
                                     targets=3, receptions=2, receiving_yards=rng.normal(25, 8)))
                    rows.append(dict(base, player_display_name=f"Quarter Back {team}", position="QB", attempts=33,
                                     passing_yards=rng.normal(250, 40), passing_tds=1.0))
                    rows.append(dict(base, player_display_name=f"Run Back {team}", position="RB", carries=16,
                                     rushing_yards=rng.normal(70, 15), targets=3, receptions=2, receiving_yards=15.0))
                    rows.append(dict(base, player_display_name=f"Tight End {team}", position="TE", targets=5,
                                     receptions=4, receiving_yards=rng.normal(40, 10)))
                    rows.append(dict(base, player_display_name=f"Line Backer {team}", position="LB",
                                     def_sacks=float(rng.poisson(2.5)), def_interceptions=float(rng.poisson(0.8))))
    return pd.DataFrame(rows).fillna(0)


def test_asof_features_use_only_earlier_weeks():
    pw = pj.player_weeks(_weekly())
    f1 = pj.asof_features(pw, 2025, 5)
    bumped = pw.copy()
    bumped.loc[(bumped.season == 2025) & (bumped.week >= 5), "dk"] += 50  # the week itself and later
    f2 = pj.asof_features(bumped, 2025, 5)
    assert np.allclose(f1["stab_dk"], f2["stab_dk"]) and np.allclose(f1["l3_dk"], f2["l3_dk"])
    # week 1 leans entirely on last season; later weeks shrink toward it
    w1 = pj.asof_features(pw, 2025, 1)
    assert np.allclose(w1["stab_dk"], w1["prev_dk"])


def test_fit_and_project_rank_usage_and_dst_matchups():
    weekly, games = _weekly(), _games()
    skill, dst = pj.training_rows(weekly, games)
    assert set(skill["position"]) == {"QB", "RB", "WR", "TE"} and len(dst) > 0
    model = pj.fit(skill, dst)
    pool = pd.DataFrame({
        "name": ["Star Wr AAA", "Depth Wr AAA", "Star Wr CCC", "Line Backer AAA"],
        "position": ["WR", "WR", "WR", "DST"], "team": ["AAA", "AAA", "CCC", "AAA"], "opp": ["BBB", "BBB", "DDD", "BBB"],
        "proj": [18.0, 6.0, 17.0, 7.0]})
    pool.loc[3, "name"] = "Aaa"  # DSTs aren't matched by name
    ours = pj.project(pool, {"weekly": weekly, "games": games}, 2025, 8, model)
    assert ours[0] > ours[1] + 5  # usage carries through
    assert ours[0] > ours[2]      # same usage, higher Vegas total (50 vs 38)
    assert np.isfinite(ours[3])
    b = pj.blend(pool, ours, 0.25, 0.5)
    assert abs(b[0] - (0.75 * 18.0 + 0.25 * ours[0])) < 1e-9 and abs(b[3] - (0.5 * 7.0 + 0.5 * ours[3])) < 1e-9
    assert pj.blend(pool, pd.Series(np.nan, index=pool.index), 0.25, 0.5).equals(pool["proj"])


def test_shipped_projection_model():
    m = pj.load(str(pj.MODEL_PATH))
    assert set(m["positions"]) == set(pj.SKILL) and m["features"] == pj.FEATURES
    assert m["seasons"][0] <= 2021 and m["dst"]["n"] > 1000
