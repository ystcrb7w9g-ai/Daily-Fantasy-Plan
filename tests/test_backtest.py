import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import backtest as bt, contest as cs
from dfs_engine.simulate import simulate_player_scores
from test_sim_tools import _multi_game_pool

TINY = ["--candidates", "40", "--gen-trials", "300", "--eval-trials", "400", "--holdout-trials", "200",
        "--field-size", "1500", "--field-optimizer-solves", "40", "--seed", "3"]


def _lineup_text(df, rows):
    d = df.iloc[list(rows)]
    by = {p: list(d.loc[d["position"] == p, "name"]) for p in ("QB", "RB", "WR", "TE", "DST")}
    slots = [("QB", by["QB"][0]), ("RB", by["RB"][0]), ("RB", by["RB"][1]), ("WR", by["WR"][0]),
             ("WR", by["WR"][1]), ("WR", by["WR"][2]), ("TE", by["TE"][0])]
    flex = (by["RB"][2:] + by["WR"][3:] + by["TE"][1:])[0]
    slots += [("FLEX", flex), ("DST", by["DST"][0])]
    return " ".join(f"{s} {n}" for s, n in slots)


def _fake_contest(tmp, n_entries=3000):
    """A past 'contest': a pool, a field of entries with real-looking points, a DK standings export."""
    df = _multi_game_pool(4)
    # real-looking names: the standings matcher normalizes names (digits and underscores drop out)
    words = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]
    df["name"] = [f"{t.title()} {p.title()} {words[k]}" if p != "DST" else f"Defense {t.title()}"
                  for t, p, k in zip(df["team"], df["position"], df.groupby(["team", "position"]).cumcount())]
    pool_path = os.path.join(tmp, "pool.csv")
    df.drop(columns="player_id").to_csv(pool_path, index=False)
    field = cs.generate_field(df, n_entries, seed=9)
    actual = simulate_player_scores(df, n_trials=1, seed=21)[0].round(2)
    pts = field @ actual
    order = np.argsort(-pts)
    lines = [_lineup_text(df, np.flatnonzero(field[i])) for i in order]
    ent = pd.DataFrame({"Rank": np.arange(1, n_entries + 1), "EntryId": np.arange(n_entries) + 10_000,
                        "EntryName": [f"user{i}" for i in range(n_entries)], "TimeRemaining": 0,
                        "Points": pts[order].round(2), "Lineup": lines})
    own = (field > 0).mean(axis=0)
    players = pd.DataFrame({"Player": df["name"], "Roster Position": df["position"],
                            "%Drafted": [f"{100 * o:.2f}%" for o in own], "FPTS": actual})
    raw = pd.concat([ent, pd.DataFrame({"": [None] * n_entries})], axis=1)
    raw = pd.concat([raw.reset_index(drop=True), players.reindex(range(n_entries)).reset_index(drop=True)], axis=1)
    stand_path = os.path.join(tmp, "standings.csv")
    raw.to_csv(stand_path, index=False)
    man = os.path.join(tmp, "slates.csv")
    pd.DataFrame([{"label": "T1", "pool": "pool.csv", "standings": "standings.csv", "entry_fee": 10,
                   "prize_pool": 25000, "top_prizes": "5000;2000;1000"}]).to_csv(man, index=False)
    return df, actual, pts, man


def test_anchored_payouts_and_finishes_split_ties():
    t = bt.anchored_payouts(1000, 10.0, 8500.0, [2000.0, 800.0])
    assert abs(t.sum() - 8500.0) < 1.0 and t[0] == 2000.0 and (np.diff(t[:220]) <= 1e-9).all()
    assert (t[:220] >= 20.0 - 1e-9).all() and (t[220:] == 0).all()
    ranks, prizes = bt.place_in_contest(np.array([100.0, 90.0, 90.0, 50.0]), np.array([95.0, 90.0, 10.0]),
                                        np.array([40.0, 30.0, 20.0, 10.0, 0.0]))
    assert ranks.tolist() == [2, 2, 5]
    assert prizes[0] == 30.0 and abs(prizes[1] - 20.0) < 1e-9 and prizes[2] == 0.0  # 3-way tie for 2nd-4th


def test_backtest_replays_dk_run_and_grades_against_real_standings():
    with tempfile.TemporaryDirectory() as tmp:
        df, actual, pts, man = _fake_contest(tmp)
        spec = bt.read_manifest(man)[0]
        slate = bt.load_slate(spec)
        assert slate.fmt == "classic" and slate.size == 3000
        assert np.mean([lu is not None for lu in slate.lineups]) > 0.99
        assert np.allclose(slate.actual, actual, atol=0.01)
        r = bt.run_slate(spec, bt.default_args(TINY), n_lineups=(1, 5), n_sample=2000,
                         keep=os.path.join(tmp, "T1.pkl"))
        g = r["ranking"]
        assert 0.0 <= g["auc_top1"] <= 1.0 and g["real_top1"] > 0 and len(g["by_ownership_quintile"]) == 5
        assert 0.0 < g["predicted_top1_mean"] < 0.2
        p1, p5 = r["replay"]["picks"][1], r["replay"]["picks"][5]
        assert p1["n"] == 1 and p5["n"] == 5 and 1 <= p5["best_finish"] <= 3001 and p5["cashes"] <= 5
        assert p5["fees"] == 50.0 and p5["won"] >= 0
        f = r["field"]
        assert abs(f["real"]["proj_mean"] - f["ours"]["proj_mean"]) < 15 and "collision_x1e4" in f["ours"]
        assert os.path.exists(os.path.join(tmp, "T1.pkl"))
        tables = bt.summarize([r], (1, 5))
        assert len(tables["picks"]) == 2 and len(tables["ranking"]) == 1
        bt.save([r], os.path.join(tmp, "out"))
        assert os.path.exists(os.path.join(tmp, "out", "backtest.json"))


def test_fit_sharpen_recovers_how_chalky_a_contest_was():
    from dfs_engine.ownership import sharpen_ownership
    df = _multi_game_pool(6)
    own = df["own"] / df["own"].sum() * 9
    df["own"] = own
    real = sharpen_ownership(own, df["position"], 1.3).to_numpy()
    fit = bt.fit_sharpen(df, real)
    assert abs(fit["best"] - 1.3) < 0.06 and fit["rmse_best"] < fit["rmse_at_1"]
