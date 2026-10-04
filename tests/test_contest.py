import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from dfs_engine import contest as cs
from dfs_engine.data import load_player_pool
from dfs_engine.optimize import SALARY_CAP, StackRules, LineupResult
from dfs_engine.simulate import simulate_player_scores
from test_sim_tools import _multi_game_pool

POOL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_classic_pool.csv")


def _value_owned_pool(seed=0):
    """Multi-game pool whose ownership tracks value, like a real slate."""
    df = _multi_game_pool(seed)
    slots = {"QB": 1.0, "RB": 2.4, "WR": 3.4, "TE": 1.2, "DST": 1.0}
    value = df["proj"] / (df["salary"] / 1000)
    z = (value - value.groupby(df["position"]).transform("mean")) / value.groupby(df["position"]).transform("std")
    raw = np.exp(1.0 * z)
    own = raw / raw.groupby(df["position"]).transform("sum") * df["position"].map(slots)
    df["own"] = own.clip(upper=0.6)
    df["own"] *= 9 / df["own"].sum()
    return df


# --- payouts ---------------------------------------------------------------

def test_gpp_payout_curve_shape():
    pc = cs.gpp_payout_curve(10_000, 20.0, rake=0.15, paid_frac=0.2, first_frac=0.15)
    assert abs(pc.payouts.sum() - 10_000 * 20 * 0.85) < 1e-6
    assert pc.paid_places == 2_000
    assert abs(pc.payouts[0] / pc.payouts.sum() - 0.15) < 0.01
    assert np.all(np.diff(pc.payouts[:2_000]) <= 1e-9)  # non-increasing
    assert abs(pc.payouts[1_999] - 40.0) < 1e-6  # min cash = 2x fee


def test_payout_window_average_splits_ties():
    pc = cs.PayoutCurve(np.array([100.0, 50.0, 10.0, 0.0]), entry_fee=10)
    avg = pc.average_payout(np.array([0.0, 0.0, 3.0]), np.array([1.0, 2.0, 1.0]))
    assert np.allclose(avg, [100.0, 75.0, 0.0])


# --- candidates ------------------------------------------------------------

def test_generate_candidates_distinct_and_rule_abiding():
    df = _value_owned_pool()
    scores = simulate_player_scores(df, n_trials=300, seed=1)
    rules = StackRules(qb_stack=2)
    cands = cs.generate_candidates(df, scores, 25, stack_rules=rules, seed=1)
    keys = {cs.lineup_key(c.player_ids) for c in cands}
    assert len(cands) == 25 and len(keys) == 25
    for c in cands:
        lu = df.loc[c.player_ids]
        qb = lu[lu["position"] == "QB"].iloc[0]
        assert ((lu["team"] == qb["team"]) & lu["position"].isin(["WR", "TE"])).sum() >= 2
        assert lu["salary"].sum() <= SALARY_CAP


# --- field -----------------------------------------------------------------

def test_generate_field_is_legal_and_matches_ownership():
    df = _value_owned_pool()
    field = cs.generate_field(df, 3000, seed=2)
    pos = df["position"].to_numpy()
    assert field.shape == (3000, len(df))
    assert (field.sum(axis=1) == 9).all()
    sal = field @ df["salary"].to_numpy()
    assert (sal <= SALARY_CAP).all() and (sal >= 49_000).all()
    for p, lo, hi in (("QB", 1, 1), ("DST", 1, 1), ("RB", 2, 3), ("WR", 3, 4), ("TE", 1, 2)):
        n = field[:, pos == p].sum(axis=1)
        assert ((n >= lo) & (n <= hi)).all()
    realized = (field > 0).mean(axis=0)
    # A 4-team pool has few teammates, so the forced stack mix and the
    # ownership targets compete; full slates match to ~0.2% mean error.
    assert np.abs(realized - df["own"].to_numpy()).mean() < 0.015
    stacked = np.array([
        ((df.loc[np.where(r > 0)[0], "team"] == df.loc[np.where((r > 0) & (pos == "QB"))[0][0], "team"])
         & df.loc[np.where(r > 0)[0], "position"].isin(["WR", "TE"])).sum()
        for r in field
    ])
    mix = [(stacked == 0).mean(), (stacked == 1).mean(), (stacked >= 2).mean()]
    assert np.allclose(mix, cs.DEFAULT_STACK_MIX, atol=0.02)


def test_generate_field_showdown_small_pool_lowers_floor():
    df = load_player_pool(POOL_PATH)
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        field = cs.generate_field(df, 500, fmt="showdown", seed=3)
    assert (np.isclose(field, 1.5).sum(axis=1) == 1).all()  # one captain
    assert ((field > 0).sum(axis=1) == 6).all()
    assert (field @ df["salary"].to_numpy() <= SALARY_CAP).all()


# --- contest sim -----------------------------------------------------------

def test_field_lineups_score_like_the_field():
    """A lineup drawn from the same field process should average ~ -rake ROI, ~1% top-1%."""
    df = _value_owned_pool()
    scores = simulate_player_scores(df, n_trials=1500, seed=4)
    field = cs.generate_field(df, 4000, seed=5)
    probe = cs.generate_field(df, 400, seed=6)
    pc = cs.gpp_payout_curve(20_000, 10.0, rake=0.15, first_frac=0.1)
    res = cs.simulate_contest(probe, field, scores, pc)
    assert abs(res.metrics["top1_pct"].mean() - 0.01) < 0.004
    assert abs(res.metrics["cash_pct"].mean() - 0.22) < 0.03
    assert -0.45 < res.metrics["roi"].mean() < 0.15


def test_optimized_candidates_beat_the_field():
    df = _value_owned_pool()
    gen = simulate_player_scores(df, n_trials=300, seed=7)
    ev = simulate_player_scores(df, n_trials=1500, seed=8)
    cands = cs.generate_candidates(df, gen, 30, seed=7)
    field = cs.generate_field(df, 4000, seed=9)
    pc = cs.gpp_payout_curve(20_000, 10.0)
    res = cs.simulate_contest(cs.lineups_to_weights(cands, len(df)), field, ev, pc)
    assert res.metrics["top1_pct"].mean() > 0.01
    assert res.payouts.shape == (1500, 30)


def test_exact_duplicates_split_prizes():
    df = _value_owned_pool()
    scores = simulate_player_scores(df, n_trials=400, seed=10)
    field = cs.generate_field(df, 2000, seed=11)
    pc = cs.gpp_payout_curve(10_000, 10.0)
    lineup = field[:1].copy()
    solo = cs.simulate_contest(lineup, field[1:], scores, pc)
    duped = cs.simulate_contest(lineup, np.vstack([field[1:], np.repeat(lineup, 50, axis=0)]), scores, pc)
    assert duped.metrics["exp_dupes"].iloc[0] > 0
    assert duped.metrics["exp_payout"].iloc[0] < solo.metrics["exp_payout"].iloc[0]


# --- portfolio selection ---------------------------------------------------

def _graded(seed=12, n_cand=40):
    df = _value_owned_pool()
    gen = simulate_player_scores(df, n_trials=400, seed=seed)
    ev = simulate_player_scores(df, n_trials=800, seed=seed + 1)
    cands = cs.generate_candidates(df, gen, n_cand, seed=seed)
    field = cs.generate_field(df, 3000, seed=seed + 2)
    res = cs.simulate_contest(cs.lineups_to_weights(cands, len(df)), field, ev, cs.gpp_payout_curve(20_000, 10.0))
    return df, cands, res


def test_select_portfolio_respects_caps_and_uniqueness():
    df, cands, res = _graded()
    for obj in ("roi", "top1"):
        chosen = cs.select_portfolio(df, cands, res, 10, objective=obj, min_unique=2, max_exposure=0.6)
        assert len(chosen) == len(set(chosen)) and len(chosen) > 0
        sets = [set(cands[i].player_ids) for i in chosen]
        for i in range(len(sets)):
            for j in range(i):
                assert len(sets[i] ^ sets[j]) >= 4
        counts = np.zeros(len(df))
        for s in sets:
            counts[list(s)] += 1
        assert counts.max() <= max(1, int(0.6 * 10))


def test_select_portfolio_roi_picks_best_first_and_top1_covers_more():
    df, cands, res = _graded()
    roi_pick = cs.select_portfolio(df, cands, res, 8, objective="roi", min_unique=1)
    assert roi_pick[0] == int(np.argmax(res.payouts.mean(axis=0)))
    top_pick = cs.select_portfolio(df, cands, res, 8, objective="top1", min_unique=1)
    s_roi = cs.portfolio_summary(res, roi_pick, 10.0)
    s_top = cs.portfolio_summary(res, top_pick, 10.0)
    assert s_top["p_any_top1"] >= s_roi["p_any_top1"] - 1e-9


def test_optimizer_slice_adds_realistic_duplication_and_keeps_ownership():
    from collections import Counter
    df = _value_owned_pool()
    plain = cs.generate_field(df, 3000, seed=21)
    sliced = cs.generate_field(df, 3000, seed=21, optimizer_share=0.1, optimizer_solves=60,
                               optimizer_noise=0.15)
    def in_dupes(f):
        c = np.array(list(Counter(map(bytes, (f * 2).astype(np.uint8))).values()))
        return c[c >= 2].sum() / len(f)
    assert len(sliced) == 3000
    assert in_dupes(sliced) >= in_dupes(plain) + 0.08
    own = df["own"].to_numpy()
    assert np.abs((sliced > 0).mean(axis=0) - own).mean() < 0.02
    sal = sliced @ df["salary"].to_numpy()
    assert (sal <= 50_000).all() and (sliced.sum(axis=1) == 9).all()


def test_duplicated_field_lineups_raise_candidate_dupes():
    df = _value_owned_pool()
    field = cs.generate_field(df, 3000, seed=22, optimizer_share=0.1, optimizer_solves=40,
                              optimizer_noise=0.1)
    from collections import Counter
    top = max(Counter(map(bytes, (field * 2).astype(np.uint8))).items(), key=lambda kv: kv[1])
    chalk = np.frombuffer(top[0], dtype=np.uint8).astype(np.float32)[None, :] / 2
    scores = simulate_player_scores(df, n_trials=200, seed=23)
    res = cs.simulate_contest(chalk, field, scores, cs.gpp_payout_curve(30_000, 10.0))
    assert res.metrics["exp_dupes"].iloc[0] >= top[1] * (30_000 - 1) / 3000 - 1e-6


def test_flexible_stack_rules_expand_and_keep_caps():
    rules = StackRules(qb_stack=2, bring_back=1, max_per_game=4, max_per_team=3, max_vs_dst=0)
    opts = cs.flexible_stack_rules(rules)
    assert [(r.qb_stack, r.bring_back) for r in opts] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert all(r.max_per_game == 4 and r.max_per_team == 3 and r.max_vs_dst == 0 for r in opts)


def test_flex_candidates_mix_structures_and_share_cap_limits_game_stacks():
    import pandas as pd
    a, b = _value_owned_pool(seed=0), _value_owned_pool(seed=1)
    ren = {"AAA": "EEE", "BBB": "FFF", "CCC": "GGG", "DDD": "HHH"}
    b["team"], b["opp"] = b["team"].map(ren), b["opp"].map(ren)
    b["name"] = b["name"] + "_b"
    df = pd.concat([a, b], ignore_index=True)
    df["own"] = df["own"] / 2
    df["player_id"] = df.index
    rules = cs.flexible_stack_rules(StackRules(qb_stack=2, bring_back=1, max_per_game=4))
    gen = simulate_player_scores(df, n_trials=300, seed=31)
    cands = cs.generate_candidates(df, gen, 40, stack_rules=rules, seed=31)
    sizes = [cs.max_game_stack(df, c) for c in cands]
    assert max(sizes) <= 4 and min(sizes) < 4  # not every lineup is a 4-man game stack
    ev = simulate_player_scores(df, n_trials=400, seed=32)
    field = cs.generate_field(df, 2000, seed=33)
    res = cs.simulate_contest(cs.lineups_to_weights(cands, len(df)), field, ev, cs.gpp_payout_curve(20_000, 10.0))
    chosen = cs.select_portfolio(df, cands, res, 10, min_unique=1, max_game_stack_share=0.3)
    assert len(chosen) > 0
    assert sum(cs.max_game_stack(df, cands[i]) >= 4 for i in chosen) <= 3
