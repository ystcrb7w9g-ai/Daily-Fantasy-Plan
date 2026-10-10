import csv
import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import contest as cs
from dfs_engine import dk
from dfs_engine import lateswap as ls
from dfs_engine.simulate import simulate_player_scores
from test_contest import _value_owned_pool

EARLY, LATE = "2026-10-11 13:00", "2026-10-11 16:25"


def _pool():
    df = _value_owned_pool().reset_index(drop=True)
    df["dk_id"] = (40000000 + df.index).astype(str)
    df["name"] = [f"Player {chr(65 + i // 26)}{chr(65 + i % 26)}" for i in df.index]  # names join on letters
    df["game_time"] = np.where(df["team"].isin(["AAA", "BBB"]), EARLY, LATE)
    df["player_id"] = df.index
    return df


def _lineup(df):
    from dfs_engine.optimize import solve_classic
    lu = solve_classic(df, df["proj"].to_numpy())
    return dk.assign_slots(lu.player_ids, df)


def _standings(df, so_far, field_ids, our_ids):
    name = dict(zip(df["dk_id"], df["name"]))
    pos = dict(zip(df["dk_id"], df["position"]))
    rows = []
    for k, ids in enumerate(field_ids + [our_ids]):
        rows.append({"Rank": k + 1, "EntryId": 9000 + k if k < len(field_ids) else 1234,
                     "EntryName": f"u{k}", "Points": 0,
                     "Lineup": " ".join(f"{s} {name[i]}" for s, i in zip(dk.CLASSIC_SLOTS, ids))})
    players = pd.DataFrame({"Player": df["name"], "Roster Position": df["position"],
                            "%Drafted": "10.00%", "FPTS": so_far})
    out = pd.DataFrame(rows)
    out = pd.concat([out, players.iloc[: len(out)].reset_index(drop=True)], axis=1)
    return out


def test_read_current_lineups_parses_ids_and_locks():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "e.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(dk.ENTRY_COLUMNS + dk.CLASSIC_SLOTS)
        w.writerow(["1234", "Milly", "77", "$20", "Josh Allen (111111) (LOCKED)", "222222"] + ["333333"] * 7)
    cur = ls.read_current_lineups(path, dk.CLASSIC_SLOTS)
    assert cur.loc[0, "slot_0"] == "111111" and bool(cur.loc[0, "locked_0"])
    assert cur.loc[0, "slot_1"] == "222222" and not bool(cur.loc[0, "locked_1"])


def test_remaining_fraction():
    t = pd.Series(pd.to_datetime([EARLY, LATE]))
    now = pd.Timestamp("2026-10-11 14:37")
    frac = ls.remaining_fraction(t, now)
    assert abs(frac[0] - (1 - 97 / ls.GAME_MINUTES)) < 1e-9 and frac[1] == 1.0
    assert ls.remaining_fraction(t, now, assume_final=True)[0] == 0.0


def test_complete_lineup_keeps_fixed_and_rules():
    df = _pool()
    ids = _lineup(df)
    idx = [int(i) - 40000000 for i in ids]
    open_k = [k for k, j in enumerate(idx) if df.at[j, "game_time"] == LATE]
    fixed = [j for k, j in enumerate(idx) if k not in open_k]
    late = np.flatnonzero(df["game_time"].to_numpy() == LATE)
    fill = ls.complete_lineup(df, df["proj"].to_numpy(), fixed, [ls.dk_slot(k) for k in open_k], late)
    full = list(idx)
    for k, j in zip(open_k, fill):
        full[k] = j
    assert sorted(full) == sorted(idx)  # proj-optimal lineup is its own best completion
    assert ls.complete_lineup(df, df["proj"].to_numpy(), fixed, [ls.dk_slot(k) for k in open_k], late,
                              salary_cap=int(df.loc[fixed, "salary"].sum()) + 100) is None


def test_late_swap_end_to_end_keeps_locked_and_beats_or_keeps():
    df = _pool()
    ours = _lineup(df)
    started = df["game_time"].to_numpy() == EARLY
    truth = simulate_player_scores(df, n_trials=1, seed=4)[0]
    so_far = np.where(started, truth, 0.0)
    field = cs.generate_field(df, 400, seed=5)
    field_ids = [dk.assign_slots(list(np.flatnonzero(f)), df) for f in field]
    raw = _standings(df, so_far, field_ids, ours)

    slate = ls.live_slate(df, [raw], pd.Timestamp("2026-10-11 16:20"), assume_final=True)
    assert np.allclose(slate.actual[started], truth[started])
    pos = dict(zip(df["dk_id"], df["position"]))
    current = pd.DataFrame([{"entry_id": "1234", "contest_id": "77",
                             **{f"slot_{k}": i for k, i in enumerate(ours)},
                             **{f"locked_{k}": bool(started[int(i) - 40000000]) for k, i in enumerate(ours)}}])
    W, off, n = ls.field_from_standings(raw, slate, {"1234"})
    assert n == 400 and (W.sum(axis=1) == 9).all()
    gen = ls.live_scores(slate, simulate_player_scores(df, n_trials=300, seed=6))
    ev = ls.live_scores(slate, simulate_player_scores(df, n_trials=600, seed=7))
    plans = ls.plan_entries(current, slate, gen, n_solves=25, seed=8)
    assert plans[0].keeps_current and len(plans[0].options) > 1
    res = ls.choose(plans, slate, ev, {"77": (W, off)}, (W, off), {"77": cs.gpp_payout_curve(5000, 20.0)})
    r = res[0]
    assert r["new_exp_payout"] >= r["cur_exp_payout"] - 1e-9
    new_ids = ls.upload_ids(plans[0], r["choice"], slate.pool)
    for k in range(9):
        if plans[0].locked[k]:
            assert new_ids[k] == ours[k]
    players = df.assign(game=np.where(df["game_time"] == EARLY, "AAA@BBB", "CCC@DDD"))
    dk.validate_upload_lineup(new_ids, players)
    assert pos[new_ids[8]] == "DST"


def test_add_rostered_players_backfills_keep_only():
    df = _pool()
    extra = pd.DataFrame([{"dk_id": "49999999", "name": "Ghost DST", "position": "DST", "team": "CCC",
                           "opp": "DDD", "salary": 2500, "avg_ppg": 8.0, "game_time": pd.Timestamp(LATE)}])
    current = pd.DataFrame([{"entry_id": "1", "contest_id": "7", "slot_8": "49999999"}])
    out = ls.add_rostered_players(df, current, extra)
    g = out[out["dk_id"] == "49999999"].iloc[0]
    assert g["proj"] == 8.0 and bool(g["keep_only"]) and g["team_total"] == df[df.team == "CCC"]["team_total"].iloc[0]
    assert not out["keep_only"].iloc[: len(df)].any()
