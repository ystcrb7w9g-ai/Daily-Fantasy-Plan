import os
import sys
import tempfile
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import contest as cs
from dfs_engine import history, ownership
from dfs_engine.simulate import simulate_player_scores
from test_contest import _value_owned_pool

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _standings_file(d, zipped=False):
    """Tiny contest-standings export in DraftKings' layout."""
    lineups = [
        "DST Bills  FLEX Tee Higgins QB Josh Allen RB James Cook RB Bijan Robinson TE Dalton Kincaid "
        "WR Khalil Shakir WR Keon Coleman WR Ja'Marr Chase",           # QB + 3 own (Shakir, Coleman, Kincaid)
        "DST Bills  FLEX Tee Higgins QB Josh Allen RB James Cook RB Bijan Robinson TE Dalton Kincaid "
        "WR Khalil Shakir WR Keon Coleman WR Ja'Marr Chase",           # exact duplicate
        "DST Falcons  FLEX James Cook QB Joe Burrow RB Bijan Robinson RB Derrick Henry TE Dalton Kincaid "
        "WR Ja'Marr Chase WR Khalil Shakir WR Keon Coleman",           # QB + 1 own (Chase)
        "DST Falcons  FLEX Derrick Henry QB Joe Burrow RB Bijan Robinson RB James Cook TE Travis Kelce "
        "WR Khalil Shakir WR Keon Coleman WR Puka Nacua",              # QB + 0 own
    ]
    points = [200.0, 200.0, 150.0, 100.0]
    players = [("Josh Allen", "QB", "50.00%", 30), ("Joe Burrow", "QB", "50.00%", 20),
               ("James Cook", "RB", "75.00%", 20), ("Bijan Robinson", "RB", "100.00%", 15),
               ("Derrick Henry", "RB", "25.00%", 9), ("James Cook", "FLEX", "25.00%", 20),
               ("Derrick Henry", "FLEX", "25.00%", 9), ("Tee Higgins", "FLEX", "50.00%", 9), ("Tee Higgins", "WR", "0.00%", 9),
               ("Ja'Marr Chase", "WR", "75.00%", 12), ("Khalil Shakir", "WR", "100.00%", 14),
               ("Keon Coleman", "WR", "100.00%", 8), ("Puka Nacua", "WR", "25.00%", 7),
               ("Dalton Kincaid", "TE", "75.00%", 11), ("Travis Kelce", "TE", "25.00%", 6),
               ("Bills ", "DST", "50.00%", 8), ("Falcons ", "DST", "50.00%", 4)]
    n = max(len(lineups), len(players))
    rows = []
    for i in range(n):
        e = ([i + 1, 1000 + i, f"user{i} (1/1)", 0, points[i], lineups[i]] if i < len(lineups)
             else [""] * 6)
        p = list(players[i]) if i < len(players) else [""] * 4
        rows.append(e + [""] + p)
    df = pd.DataFrame(rows, columns=["Rank", "EntryId", "EntryName", "TimeRemaining", "Points",
                                     "Lineup", "", "Player", "Roster Position", "%Drafted", "FPTS"])
    df = df.replace("", np.nan)
    path = os.path.join(d, "contest-standings-1.csv")
    df.to_csv(path, index=False)
    if zipped:
        zp = os.path.join(d, "contest-standings-1.zip")
        with zipfile.ZipFile(zp, "w") as z:
            z.write(path, "contest-standings-1.csv")
        return zp
    return path


def test_parse_lineup_handles_dst_and_flex():
    lu = history.parse_lineup("DST Bills  FLEX Tee Higgins QB Josh Allen RB James Cook")
    assert lu == [("DST", "Bills"), ("FLEX", "Tee Higgins"), ("QB", "Josh Allen"), ("RB", "James Cook")]


def test_profile_standings_reads_zip_and_measures_field():
    d = tempfile.mkdtemp()
    path = _standings_file(d, zipped=True)
    team_of = {"josh allen": "BUF", "khalil shakir": "BUF", "keon coleman": "BUF", "dalton kincaid": "BUF",
               "james cook": "BUF", "joe burrow": "CIN", "jamarr chase": "CIN", "tee higgins": "CIN"}
    p = history.profile_standings(path, team_of)
    assert p["entries"] == 4
    assert p["score_median"] == 175.0 and p["score_first"] == 200.0
    assert p["entries_in_duplicates"] == 0.5
    assert p["qb_stack_mix"] == [0.25, 0.25, 0.5]  # 2+ bucket holds the two Allen QB+3 builds
    assert abs(p["flex_split"]["RB"] - 0.5) < 1e-9 and abs(p["flex_split"]["WR"] - 0.5) < 1e-9
    assert p["own_curves"]["RB"][:3] == [1.0, 1.0, 0.5]  # FLEX rows folded into player totals


def test_combine_profiles_averages():
    a = {"file": "a", "ratio_top0.1": 1.6, "ratio_top1": 1.5, "ratio_top20": 1.2, "entries_in_duplicates": 0.1,
         "flex_split": {"RB": 0.4, "WR": 0.4, "TE": 0.2}, "qb_stack_mix": [0.2, 0.5, 0.3],
         "own_curves": {p: [0.2, 0.1] for p in history.POSITIONS}}
    b = dict(a, file="b", ratio_top1=1.7, qb_stack_mix=[0.2, 0.6, 0.2],
             own_curves={p: [0.4, 0.1] for p in history.POSITIONS})
    c = history.combine_profiles([a, b])
    assert abs(c["ratio_top1"] - 1.6) < 1e-9
    assert c["qb_stack_mix"] == [0.2, 0.55, 0.25]
    assert c["own_curves"]["QB"] == [0.3, 0.1]


def test_shipped_profile_matches_engine_defaults():
    prof = history.load_profile(os.path.join(ROOT, "data", "field_profile.json"))
    assert np.allclose(prof["qb_stack_mix"], cs.DEFAULT_STACK_MIX, atol=0.01)
    assert np.allclose([prof["ratio_top0.1"], prof["ratio_top1"], prof["ratio_top20"]],
                       cs.HISTORICAL_SCORE_RATIOS, atol=0.002)


# --- ownership -------------------------------------------------------------

def test_estimate_ownership_follows_value_and_position_totals():
    df = _value_owned_pool()
    df["own"] = np.nan
    est = ownership.estimate_ownership(df)
    totals = ownership.position_totals()
    for p, total in totals.items():
        assert abs(est[df["position"] == p].sum() - total) < 0.03
    rb = df[df["position"] == "RB"].assign(est=est, value=df["proj"] / df["salary"])
    assert rb.sort_values("est").iloc[-1]["value"] >= rb["value"].quantile(0.75)
    assert est.max() <= 0.95


def test_check_ownership_flags_flat_and_bad_totals():
    df = _value_owned_pool()
    df["own"] = 9.0 / len(df)  # perfectly flat
    notes = ownership.check_ownership(df)
    assert any("too flat" in n for n in notes)
    df["own"] = ownership.estimate_ownership(df)
    assert ownership.check_ownership(df) == []
    df["own"] = df["own"] * 0.5
    assert any("sums to" in n for n in ownership.check_ownership(df))


# --- spread calibration ----------------------------------------------------

def test_calibrate_spread_moves_toward_target():
    df = _value_owned_pool()
    df["ceiling"] = np.nan
    field = cs.generate_field(df, 2000, seed=1)
    sim = lambda d, n, seed, k: simulate_player_scores(d, n_trials=n, seed=seed, spread_scale=k)  # noqa: E731
    narrow = (1.3, 1.25, 1.1)
    wide = (1.9, 1.7, 1.3)
    k_narrow, r_narrow, _ = cs.calibrate_spread(df, field, sim, target=narrow, n_trials=150, n_iter=5)
    k_wide, r_wide, _ = cs.calibrate_spread(df, field, sim, target=wide, n_trials=150, n_iter=5)
    assert k_narrow < k_wide
    assert r_narrow[1] < r_wide[1]


def test_calibrate_spread_skips_when_all_ceilings_given():
    df = _value_owned_pool()
    field = cs.generate_field(df, 1000, seed=2)
    sim = lambda d, n, seed, k: simulate_player_scores(d, n_trials=n, seed=seed, spread_scale=k)  # noqa: E731
    k, ratios, _ = cs.calibrate_spread(df, field, sim, n_trials=100)
    assert k is None and len(ratios) == 3


def test_sharpen_ownership_keeps_totals_and_raises_chalk():
    own = pd.Series([0.40, 0.20, 0.10, 0.30, 0.30])
    pos = pd.Series(["RB", "RB", "RB", "QB", "QB"])
    s = ownership.sharpen_ownership(own, pos, 1.5)
    assert abs(s[pos == "RB"].sum() - 0.70) < 1e-9 and abs(s[pos == "QB"].sum() - 0.60) < 1e-9
    assert s[0] > 0.40 and s[2] < 0.10
    assert np.allclose(ownership.sharpen_ownership(own, pos, 1.0), own)


def test_evaluate_ownership_scores_projection_file():
    actual = pd.DataFrame({"name": ["A Guy", "B Guy", "C Guy", "D Guy"],
                           "own": [0.50, 0.30, 0.10, 0.10], "fpts": [20.0, 10.0, 5.0, 8.0]})
    proj = pd.DataFrame({"name": ["A Guy", "B Guy", "C Guy", "D Guy", "Nobody"],
                         "position": ["RB"] * 5, "proj": [18.0, 12.0, 6.0, 7.0, 3.0],
                         "own": [40.0, 30.0, 15.0, 15.0, 1.0]})
    r = ownership.evaluate_ownership(proj, actual)
    assert r["matched"] == 4 and r["unmatched"] == 1
    assert r["own_corr"] > 0.9 and abs(r["own_mae"] - 0.05) < 1e-9
    assert r["gamma_mae"][1.3] < r["gamma_mae"][1.0]  # under-projected chalk -> sharpening helps


def test_fit_ownership_model_recovers_value_driven_ownership():
    rng = np.random.default_rng(0)
    weeks = []
    for w in range(3):
        df = _value_owned_pool(seed=w)[["name", "position", "salary", "proj"]].copy()
        X = ownership.model_features(df)
        true = np.exp(-3.0 + 1.0 * X["value"].astype(float) + rng.normal(0, 0.1, len(df)))
        weeks.append(df.assign(actual_own=true))
    model = ownership.fit_ownership_model(weeks)
    assert set(model) == {"QB", "RB", "WR", "TE", "DST"}
    test = _value_owned_pool(seed=9)
    prof = history.load_profile(os.path.join(ROOT, "data", "field_profile.json"))
    est = ownership.estimate_ownership(test, dict(prof, own_model=model))
    value = test["proj"] / test["salary"]
    for p in ("RB", "WR"):
        k = test["position"] == p
        assert np.corrcoef(est[k], value[k])[0, 1] > 0.8
        assert abs(est[k].sum() - ownership.position_totals(prof)[p]) < 0.02


def test_estimate_ownership_falls_back_to_curves_without_model():
    prof = history.load_profile(os.path.join(ROOT, "data", "field_profile.json"))
    prof.pop("own_model", None)
    df = _value_owned_pool()
    est = ownership.estimate_ownership(df, prof)
    assert abs(est[df["position"] == "QB"].sum() - 1.0) < 0.02
