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

    # Single entry (a $20 Milly): a 0.5 share must still allow the best lineup when it's a game
    # stack -- flooring 0.5 x 1 to zero used to ban every 4-man game stack there.
    big = next(i for i, s in enumerate(sizes) if s >= 4)
    pay = np.zeros((50, len(cands)), dtype=np.float32)
    pay[:, big] = 100.0
    rigged = cs.ContestResult(pay, res.metrics, pay > 0)
    kw = dict(min_unique=1, tiered_caps=False, dst_cap=1.0, objective="roi")
    assert cs.select_portfolio(df, cands, rigged, 1, max_game_stack_share=0.5, **kw) == [big]
    assert big not in cs.select_portfolio(df, cands, rigged, 1, max_game_stack_share=0.0, **kw)


# --- cheap-player cap / soft same-team RB rule ------------------------------

def test_cheap_cap_limits_punt_exposure():
    df, cands, res = _graded(n_cand=60)
    cheap_line = 4000
    chosen = cs.select_portfolio(df, cands, res, 10, min_unique=1, tiered_caps=False, dst_cap=1.0,
                                 cheap_salary=cheap_line, cheap_cap=0.3)
    assert len(chosen) > 0
    counts = np.zeros(len(df))
    for i in chosen:
        counts[cands[i].player_ids] += 1
    cheap = df["salary"].to_numpy() < cheap_line
    assert counts[cheap].max() <= 3


def _showdown_pool():
    import pandas as pd
    rows = []
    for team, opp in (("ATL", "NO"), ("NO", "ATL")):
        for pos, sal, proj in (("QB", 10000, 20), ("RB", 9000, 18), ("RB", 4000, 9), ("WR", 8000, 15),
                               ("WR", 5000, 9), ("TE", 4000, 7), ("DST", 3500, 6)):
            rows.append({"name": f"{team} {pos}{sal}", "team": team, "opp": opp, "position": pos,
                         "salary": sal, "proj": float(proj), "own": 6 / 14, "team_total": 22.5,
                         "game_total": 45.0, "spread": 0.0})
    df = pd.DataFrame(rows)
    df["player_id"] = df.index
    return df


def test_soft_showdown_rules_add_rb_pairs_and_share_cap_limits_them():
    df = _showdown_pool()
    strict, soft = cs.soft_showdown_rules()
    assert strict.no_same_team_rbs and not soft.no_same_team_rbs
    scores = simulate_player_scores(df, n_trials=600, seed=41)
    only_strict = cs.generate_candidates(df, scores, 30, fmt="showdown", stack_rules=strict, seed=41)
    assert not any(cs.has_same_team_rbs(df, c) for c in only_strict)
    both = cs.generate_candidates(df, scores, 60, fmt="showdown", stack_rules=[strict, soft],
                                  seed=41, max_solves=400)
    pairs = [cs.has_same_team_rbs(df, c) for c in both]
    assert any(pairs) and not all(pairs)
    field = cs.generate_field(df, 1500, fmt="showdown", seed=42, min_salary=40_000)
    res = cs.simulate_contest(cs.lineups_to_weights(both, len(df)), field,
                              simulate_player_scores(df, n_trials=600, seed=43), cs.gpp_payout_curve(5_000, 10.0))
    chosen = cs.select_portfolio(df, both, res, 10, min_unique=1, tiered_caps=False, dst_cap=1.0,
                                 max_rb_pair_share=0.2)
    assert len(chosen) > 0 and sum(pairs[i] for i in chosen) <= 2


def test_emax_objective_starts_at_best_ev_and_grows_best_entry_payout():
    df, cands, res = _graded()
    chosen = cs.select_portfolio(df, cands, res, 6, objective="emax", min_unique=1)
    assert chosen[0] == int(np.argmax(res.payouts.mean(axis=0)))
    best = [res.payouts[:, chosen[:k]].max(axis=1).mean() for k in range(1, len(chosen) + 1)]
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(best, best[1:]))
    roi_pick = cs.select_portfolio(df, cands, res, 6, objective="roi", min_unique=1)
    assert best[-1] >= res.payouts[:, roi_pick].max(axis=1).mean() - 1e-6


def test_auto_objective_by_contest_size():
    import argparse
    from dfs_engine.cli import _objective
    a = argparse.Namespace(objective="auto")
    assert _objective(a, 150_000) == "top1" and _objective(a, 500) == "roi"
    assert _objective(argparse.Namespace(objective="emax"), 150_000) == "emax"


def test_field_structure_matches_real_milly_rates():
    from test_ownership_model import _six_game_pool
    df = _six_game_pool(0)
    df["own"] = df["own"] / df["own"].sum() * 9
    df["player_id"] = df.index
    plain = cs.structure_rates(cs.generate_field(df, 3000, seed=3, structure=None), df)
    field = cs.generate_field(df, 3000, seed=3)
    r = cs.structure_rates(field, df)
    for k, target in cs.DEFAULT_FIELD_STRUCTURE.items():
        assert abs(r[k] - target) < abs(plain[k] - target) + 1e-9
        assert abs(r[k] - target) < 0.06, (k, r[k], target)
    own = np.clip(df["own"].to_numpy(), 1e-4, 0.95)
    assert np.abs((field > 0).mean(axis=0) - own).mean() < 0.015


# --- per-player exposure limits / imported lineups --------------------------

def _counts(df, cands, chosen):
    c = np.zeros(len(df), dtype=int)
    for i in chosen:
        c[cands[i].player_ids] += 1
    return c


def test_player_max_exposure_caps_and_bans():
    df, cands, res = _graded(n_cand=60)
    free = cs.select_portfolio(df, cands, res, 10, min_unique=1, tiered_caps=False, dst_cap=1.0)
    top = int(np.argmax(_counts(df, cands, free)))
    assert _counts(df, cands, free)[top] >= 4
    capped, banned = df.copy(), df.copy()
    capped["max_exp"] = np.nan
    capped.loc[top, "max_exp"] = 0.2
    banned["max_exp"] = np.nan
    banned.loc[top, "max_exp"] = 0.0
    kw = dict(min_unique=1, tiered_caps=False, dst_cap=1.0)
    assert _counts(df, cands, cs.select_portfolio(capped, cands, res, 10, **kw))[top] <= 2
    assert _counts(df, cands, cs.select_portfolio(banned, cands, res, 10, **kw))[top] == 0


def test_player_min_exposure_is_met_when_lineups_exist():
    df, cands, res = _graded(n_cand=80)
    free = cs.select_portfolio(df, cands, res, 10, min_unique=1)
    have = np.array([sum(i in c.player_ids for c in cands) for i in range(len(df))])
    used = _counts(df, cands, free)
    target = int(np.flatnonzero((used == 0) & (have >= 6) & (df["position"] != "DST").to_numpy())[0])
    want = df.copy()
    want["min_exp"] = np.nan
    want.loc[target, "min_exp"] = 0.5  # 50% given as a fraction ...
    assert _counts(df, cands, cs.select_portfolio(want, cands, res, 10, min_unique=1))[target] >= 5
    want.loc[target, "min_exp"] = 30.0  # ... or as a percentage
    want["min_exp"] = want["min_exp"].astype(float)
    assert _counts(df, cands, cs.select_portfolio(want, cands, res, 10, min_unique=1))[target] >= 3
    want.loc[target, "min_exp"] = 0.6  # 60% of 3 rounds up to 2 even under a 40% global cap (1 of 3)
    picks = cs.select_portfolio(want, cands, res, 3, min_unique=1, max_exposure=0.4)
    assert _counts(df, cands, picks)[target] >= 2


def test_add_locked_candidates_builds_lineups_around_a_player():
    df = _value_owned_pool()
    gen = simulate_player_scores(df, n_trials=300, seed=51)
    cands = cs.generate_candidates(df, gen, 10, seed=51)
    cheap = int(df[df["position"] == "WR"]["proj"].idxmin())
    before = len(cands)
    added = cs.add_locked_candidates(df, gen, cands, [cheap], 6, seed=51)
    assert added > 0 and len(cands) == before + added
    assert all(cheap in c.player_ids for c in cands[before:])
    none = cs.generate_candidates(df, gen, 10, seed=52, excluded_ids={cheap})
    assert all(cheap not in c.player_ids for c in none)


def test_candidates_from_ids_keeps_rule_abiding_lineups_only():
    from dfs_engine.optimize import satisfies_stack_rules
    df = _value_owned_pool()
    df["dk_id"] = (70000 + df.index).astype(str)
    gen = simulate_player_scores(df, n_trials=300, seed=61)
    stack = StackRules(qb_stack=2, max_per_team=3)
    good = cs.generate_candidates(df, gen, 6, seed=61, stack_rules=stack)
    loose = cs.generate_candidates(df, gen, 30, seed=62)
    bad = [c for c in loose if not satisfies_stack_rules(df, c.player_ids, stack)]
    assert good and bad and all(satisfies_stack_rules(df, c.player_ids, stack) for c in good)
    ids = lambda c: [df.at[i, "dk_id"] for i in c.player_ids]  # noqa: E731
    out_player = [ids(good[0])[:-1] + ["99999"]]
    short = [ids(good[0])[:8] + [ids(good[0])[0]]]  # a repeated player is not a roster
    kept, n = cs.candidates_from_ids(df, [ids(c) for c in good] + [ids(bad[0])] + out_player + short,
                                     stack_rules=[stack], existing=good[:1])
    assert n["kept"] == len(good) - 1 and n["duplicate"] == 1
    assert n["breaks_rules"] == 1 and n["not_in_pool"] == 1 and n["invalid"] == 1
    assert [sorted(k.player_ids) for k in kept] == [sorted(c.player_ids) for c in good[1:]]


# --- field targets / salary spend (2026 Week 1-5 Millys) ---------------------

def test_field_defaults_match_the_shipped_field_structure():
    import json
    path = os.path.join(os.path.dirname(__file__), "..", "data", "field_structure.json")
    mix, rates = cs.field_targets(json.load(open(path)))
    assert np.allclose(mix, cs.DEFAULT_STACK_MIX, atol=0.002)
    for k, v in rates.items():
        assert abs(v - cs.DEFAULT_FIELD_STRUCTURE[k]) < 0.002, k


def test_field_targets_weight_recent_weeks_more():
    wk = lambda qb2, bb: {"all": {"qb_stack_0_1_2_3plus": [0.2, 0.8 - qb2, qb2, 0.0], "bring_back_0_1_2plus": [1 - bb, bb, 0.0],  # noqa: E731
                                  "qb_with_own_rb": 0.2, "rb_with_own_dst": 0.15, "any_vs_own_dst": 0.07}}
    mix, rates = cs.field_targets({"weeks": {"week1": wk(0.2, 0.4), "week2": wk(0.4, 0.6)}})
    assert 0.3 < mix[2] < 0.4 and 0.5 < rates["bring_back"] < 0.6  # closer to week 2 than the plain mean


def test_classic_field_spends_salary_like_real_fields():
    df = _value_owned_pool()
    sal = cs.generate_field(df, 4000, seed=3) @ df["salary"].to_numpy(dtype=float)
    # real Millys: ~50% at exactly $50,000, 85% at $49,800+, 96% at $49,500+
    assert 0.35 < (sal == 50_000).mean() < 0.65 and (sal >= 49_800).mean() > 0.75 and (sal >= 49_500).mean() > 0.9
    flat = cs.generate_field(df, 4000, seed=3, min_salary=49_000) @ df["salary"].to_numpy(dtype=float)
    assert flat.min() >= 49_000 and (flat >= 49_800).mean() < (sal >= 49_800).mean()  # an explicit floor still wins
