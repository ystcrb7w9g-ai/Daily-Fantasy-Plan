import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import history, ownership
from dfs_engine import ownership_model as om
from test_contest import _value_owned_pool

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _games():
    return pd.DataFrame({"season": [2026, 2026], "week": [5, 5], "game_type": ["REG", "REG"],
                         "home_team": ["ARI", "LA"], "away_team": ["DET", "BUF"],
                         "spread_line": [-5.5, 3.0], "total_line": [54.5, 54.5]})


def test_schedule_lines_signs_and_totals():
    lines = om.schedule_lines(_games(), 2026, 5).set_index("team")
    assert lines.loc["DET", "spread"] == -5.5 and lines.loc["ARI", "spread"] == 5.5  # DET favored
    assert lines.loc["DET", "team_total"] == 30.0 and lines.loc["ARI", "team_total"] == 24.5
    assert lines.loc["BUF", "opp"] == "LAR"  # nflverse "LA" -> DK "LAR"
    df = pd.DataFrame({"name": ["a", "b"], "team": ["LA", "DET"], "team_total": [np.nan, 99.0]})
    out = om.attach_schedule(df, _games(), 2026, 5)
    assert out.loc[0, "team"] == "LAR" and out.loc[0, "team_total"] == 28.75  # home, favored by 3
    assert out.loc[1, "team_total"] == 99.0  # existing values kept


def _weekly():
    rows = []
    for wk, pts in ((3, 10.0), (4, 30.0)):
        rows.append({"season": 2026, "week": wk, "season_type": "REG", "player_display_name": "Star Back",
                     "position": "RB", "team": "DET", "rushing_yards": pts * 10})
        rows.append({"season": 2026, "week": wk, "season_type": "REG", "player_display_name": "Backup Back",
                     "position": "RB", "team": "DET", "rushing_yards": 20.0})
    return pd.DataFrame(rows)


def test_recency_usage_and_vacated_volume():
    df = pd.DataFrame({"name": ["Backup Back", "Star Back", "Lions"], "team": ["DET", "DET", "DET"],
                       "position": ["RB", "RB", "DST"]})
    inj = pd.DataFrame({"season": [2026], "week": [5], "team": ["DET"], "position": ["RB"],
                        "full_name": ["Star Back"], "report_status": ["Out"]})
    out = om.prepare(df, {"weekly": _weekly(), "injuries": inj}, 2026, 5)
    assert out.loc[1, "last_dk"] == 30.0 + 3  # 300 yds + 100-yd bonus
    assert out.loc[1, "avg_dk"] == (13.0 + 33.0) / 2  # 100+ yd games carry the 3-pt bonus
    assert out.loc[0, "vacated"] == out.loc[1, "avg_dk"]  # the backup inherits the starter's volume
    assert np.isnan(out.loc[2, "last_dk"])


TRUE_MODEL = {p: {"value": 1.5, "team_total": 0.7} for p in om.POSITIONS}


def _six_game_pool(seed):
    parts = []
    for j in range(3):
        d = _value_owned_pool(seed=10 * seed + j)
        ren = {t: f"{t[:2]}{j}" for t in ("AAA", "BBB", "CCC", "DDD")}
        parts.append(d.assign(team=d["team"].map(ren), opp=d["opp"].map(ren), name=d["name"] + f"_{j}"))
    return pd.concat(parts, ignore_index=True)


def _slates(n=4, seed=0):
    """Six-game pools whose 'actual' ownership comes from a known model (plus noise)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n):
        df = _six_game_pool(seed + s)
        own = om.predict(df, TRUE_MODEL) * np.exp(rng.normal(0, 0.1, len(df)))
        out.append(df.assign(actual_own=own))
    return out


def test_fit_recovers_drivers_and_predict_keeps_totals():
    weeks = _slates()
    model = om.fit(weeks, l2=0.01)
    for p in ("RB", "WR"):
        assert model[p]["value"] > 0.5 and model[p]["team_total"] > 0.2
    test = _slates(1, seed=99)[0]
    pred = om.predict(test, model)
    totals = ownership.position_totals()
    for p in ("QB", "RB", "WR"):
        k = test["position"] == p
        assert abs(pred[k].sum() - totals[p]) < 0.02 or pred[k].max() >= 0.95  # (0.95 cap)
        assert np.corrcoef(pred[k], test.loc[k, "actual_own"])[0, 1] > 0.9


def test_predict_puts_deep_players_in_a_small_tail():
    df = _value_owned_pool().reset_index(drop=True)
    model = om.fit(_slates(), l2=0.01)
    pred = om.predict(df, model)
    n_games = df["team"].nunique() / 2
    wr = df[df["position"] == "WR"]
    k = int(round(om.UNIVERSE_PER_GAME["WR"] * n_games))
    if len(wr) > k:
        deep = wr.nsmallest(len(wr) - k, "proj").index
        assert pred[deep].sum() <= om.TAIL_SHARE * ownership.position_totals()["WR"] + 1e-9


def test_cross_validate_scores_model_and_published():
    weeks = [w.assign(own=w["actual_own"] * 100) for w in _slates()]
    res = om.cross_validate(weeks, l2=0.01)
    assert len(res) == 4
    assert all(r["published"]["corr"] > 0.999 for r in res)  # 'published' = the truth here
    assert all(r["ours"]["corr"] > 0.8 for r in res)


def test_estimate_ownership_prefers_shipped_v2_model():
    prof = history.load_profile(os.path.join(ROOT, "data", "field_profile.json"))
    assert set(prof["own_model_v2"]["coef"]) == {"QB", "RB", "WR", "TE", "DST"}
    df = _value_owned_pool()
    est = ownership.estimate_ownership(df, prof)
    assert np.allclose(est, om.predict(df, prof["own_model_v2"]["coef"], ownership.position_totals(prof)))
