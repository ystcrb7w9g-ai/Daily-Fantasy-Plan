"""
Command-line entry points for the dfs_engine pipeline.

Examples
--------
Build 150 Classic lineups from a player pool CSV:

    python -m dfs_engine.cli build --pool data/week4_players.csv \\
        --n-lineups 150 --trials 50000 --fmt classic \\
        --out lineups.csv --exposure-out exposure.csv

Run the Optimal% leverage diagnostic only:

    python -m dfs_engine.cli diagnose --pool data/week4_players.csv \\
        --trials 50000 --fmt classic --out leverage.csv

Player outcome table (median / p85 / p99 / boom% / bust% / Optimal%):

    python -m dfs_engine.cli outcomes --pool data/week4_players.csv \\
        --trials 20000 --optimal-trials 2000 --out outcomes.csv

Contest sim + portfolio selection (candidates -> field -> sim -> pick):

    python -m dfs_engine.cli contest --pool data/week4_players.csv \\
        --n-lineups 150 --candidates 3000 --contest-size 200000 --entry-fee 20 \\
        --qb-stack 2 --bring-back 1 --objective roi --out portfolio.csv

Stacking rules (Classic) work on build / diagnose / outcomes / contest:

    --qb-stack 2 --bring-back 1 --max-vs-dst 0 --max-per-team 4

Pull SportsGameOdds lines/props and blend them into a pool:

    export SPORTSGAMEODDS_API_KEY=...
    python -m dfs_engine.cli sgo-fetch --out data/week4_sgo_events.json
    python -m dfs_engine.cli sgo-enrich --pool data/week4_players.csv \\
        --events data/week4_sgo_events.json --weight 0.5 \\
        --out data/week4_players_sgo.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import load_player_pool, validate_showdown_pool
from .simulate import simulate_player_scores
from .optimize import StackRules
from .outcomes import player_outcomes
from . import contest as cs
from .portfolio import build_portfolio, exposure_report
from .diagnostics import run_optimal_pct_chunk, leverage_report
from . import sportsgameodds as sgo


def _lineups_to_dataframe(df: pd.DataFrame, lineups, fmt: str) -> pd.DataFrame:
    rows = []
    for i, lu in enumerate(lineups):
        names = df.loc[lu.player_ids, "name"].tolist()
        row = {"lineup_id": i + 1, "salary_used": lu.salary_used,
               "projected_points": round(lu.projected_points, 2)}
        if fmt == "showdown" and lu.captain_id is not None:
            row["captain"] = df.loc[lu.captain_id, "name"]
            flex = [n for pid, n in zip(lu.player_ids, names) if pid != lu.captain_id]
            for j, n in enumerate(flex, start=1):
                row[f"flex_{j}"] = n
        else:
            for j, n in enumerate(names, start=1):
                row[f"player_{j}"] = n
        rows.append(row)
    return pd.DataFrame(rows)


def _stack_rules(args: argparse.Namespace) -> StackRules | None:
    rules = StackRules(
        qb_stack=args.qb_stack,
        stack_positions=tuple(p.strip().upper() for p in args.stack_positions.split(",")),
        bring_back=args.bring_back,
        max_vs_dst=args.max_vs_dst,
        max_per_team=args.max_per_team,
    )
    if not rules.is_active():
        return None
    if args.fmt != "classic":
        print("Note: stacking rules apply to Classic only; ignoring for Showdown.")
        return None
    return rules


def _simulate(df: pd.DataFrame, args: argparse.Namespace, n_trials: int, seed) -> np.ndarray:
    return simulate_player_scores(
        df, n_trials=n_trials, seed=seed,
        calibrate_ceiling=not args.no_ceiling_calibration, ceiling_pct=args.ceiling_pct,
    )


def _add_sim_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ceiling-pct", type=float, default=0.85,
                   help="percentile the `ceiling` column represents (calibrates each player's spread)")
    p.add_argument("--no-ceiling-calibration", action="store_true",
                   help="ignore `ceiling` and use the default fixed-fraction spread")


def _add_stack_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--qb-stack", type=int, default=0, help="min same-team pass catchers with the QB")
    p.add_argument("--stack-positions", default="WR,TE", help="positions counting toward --qb-stack")
    p.add_argument("--bring-back", type=int, default=0, help="min players from the QB's opponent")
    p.add_argument("--max-vs-dst", type=int, default=None,
                   help="max offensive players facing your DST (0 = never)")
    p.add_argument("--max-per-team", type=int, default=None, help="max players from one team")


def cmd_build(args: argparse.Namespace) -> None:
    df = load_player_pool(args.pool)
    if args.fmt == "showdown":
        validate_showdown_pool(df)

    rules = _stack_rules(args)
    print(f"Loaded {len(df)} players. Simulating {args.trials} correlated trials...")
    scores = _simulate(df, args, args.trials, args.seed)

    print(f"Building {args.n_lineups} lineups ({args.fmt})...")
    lineups = build_portfolio(
        df, scores, n_lineups=args.n_lineups, fmt=args.fmt,
        min_unique=args.min_unique, dst_cap=args.dst_cap, seed=args.seed,
        stack_rules=rules,
    )
    print(f"Built {len(lineups)}/{args.n_lineups} lineups "
          f"({'hit combinatorial ceiling' if len(lineups) < args.n_lineups else 'full portfolio'}).")

    out_df = _lineups_to_dataframe(df, lineups, args.fmt)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote lineups to {args.out}")

    if args.exposure_out:
        exp_df = exposure_report(df, lineups)
        exp_df.to_csv(args.exposure_out, index=False)
        print(f"Wrote exposure report to {args.exposure_out}")


def cmd_diagnose(args: argparse.Namespace) -> None:
    df = load_player_pool(args.pool)
    if args.fmt == "showdown":
        validate_showdown_pool(df)

    rules = _stack_rules(args)
    n_players = len(df)
    counts = np.zeros(n_players, dtype=int)
    done = 0
    chunk_size = args.chunk_size

    print(f"Running Optimal% diagnostic: {args.trials} trials in chunks of {chunk_size}...")
    while done < args.trials:
        this_chunk = min(chunk_size, args.trials - done)
        scores = _simulate(df, args, this_chunk, (args.seed or 0) + done)
        counts += run_optimal_pct_chunk(df, scores, fmt=args.fmt, stack_rules=rules)
        done += this_chunk
        print(f"  {done}/{args.trials} trials complete")

    report = leverage_report(df, counts, done)
    report.to_csv(args.out, index=False)
    print(f"Wrote leverage report to {args.out}")


def cmd_outcomes(args: argparse.Namespace) -> None:
    df = load_player_pool(args.pool)
    if args.fmt == "showdown":
        validate_showdown_pool(df)
    rules = _stack_rules(args)

    print(f"Simulating {args.trials} correlated trials for {len(df)} players...")
    scores = _simulate(df, args, args.trials, args.seed)

    counts, n_opt = None, 0
    if args.optimal_trials:
        n_opt = min(args.optimal_trials, args.trials)
        print(f"Solving optimal lineups on {n_opt} trials for Optimal%...")
        counts = run_optimal_pct_chunk(df, scores[:n_opt], fmt=args.fmt, stack_rules=rules)

    report = player_outcomes(
        df, scores, boom_mult=args.boom_mult, bust_mult=args.bust_mult,
        optimal_counts=counts, optimal_trials=n_opt,
    )
    report.to_csv(args.out, index=False)
    print(f"Wrote player outcome report to {args.out}")


def _parse_stack_mix(text: str):
    if text.strip().lower() == "none":
        return None
    mix = tuple(float(x) for x in text.split(","))
    if len(mix) != 3:
        raise SystemExit("--field-stack-mix needs three comma-separated shares (0, 1, 2+ stacked)")
    return mix


def cmd_contest(args: argparse.Namespace) -> None:
    df = load_player_pool(args.pool)
    if args.fmt == "showdown":
        validate_showdown_pool(df)
    rules = _stack_rules(args)
    seed = args.seed

    if args.payouts:
        payout = cs.load_payout_csv(args.payouts, args.entry_fee, args.contest_size)
    else:
        payout = cs.gpp_payout_curve(
            args.contest_size, args.entry_fee, rake=args.rake, paid_frac=args.paid_frac,
            first_frac=args.first_frac, min_cash_mult=args.min_cash_mult,
        )
    print(f"Contest: {payout.contest_size:,} entries, ${payout.entry_fee:g} fee, "
          f"{payout.paid_places:,} paid, 1st ${payout.payouts[0]:,.0f}")

    # Candidates and grading use independent sims so lineups aren't graded
    # on the same outcomes they were optimized for.
    print(f"Simulating {args.gen_trials} generation + {args.eval_trials} evaluation "
          f"+ {args.holdout_trials} holdout trials...")
    gen_scores = _simulate(df, args, args.gen_trials, seed)
    eval_scores = _simulate(df, args, args.eval_trials, None if seed is None else seed + 1)
    holdout_scores = _simulate(df, args, args.holdout_trials, None if seed is None else seed + 3)

    print(f"Generating {args.candidates} candidate lineups...")
    candidates = cs.generate_candidates(
        df, gen_scores, args.candidates, fmt=args.fmt, stack_rules=rules,
        seed=seed, progress=True,
    )
    print(f"  {len(candidates)} distinct candidates")

    print(f"Sampling a {args.field_size:,}-lineup field from ownership...")
    field = cs.generate_field(
        df, args.field_size, fmt=args.fmt, min_salary=args.field_min_salary,
        stack_mix=_parse_stack_mix(args.field_stack_mix), seed=None if seed is None else seed + 2,
    )

    print("Simulating contest...")
    cand_w = cs.lineups_to_weights(candidates, len(df))
    result = cs.simulate_contest(cand_w, field, eval_scores, payout)

    chosen = cs.select_portfolio(
        df, candidates, result, args.n_lineups, objective=args.objective,
        min_unique=args.min_unique, dst_cap=args.dst_cap, max_exposure=args.max_exposure,
    )
    print(f"Selected {len(chosen)}/{args.n_lineups} lineups (objective={args.objective}).")

    # Re-grade the chosen lineups on fresh sims: selecting the best of
    # thousands on one sim set overstates them (winner's curse).
    holdout = None
    if chosen:
        holdout = cs.simulate_contest(cand_w[chosen], field, holdout_scores, payout)
        sel = cs.portfolio_summary(result, chosen, payout.entry_fee)
        hold = cs.portfolio_summary(holdout, list(range(len(chosen))), payout.entry_fee)
        print(f"  selection sims: ROI {sel['roi']:+.1%}, P(any top-1%) {sel['p_any_top1']:.1%}")
        print(f"  holdout sims:   ROI {hold['roi']:+.1%}, P(any top-1%) {hold['p_any_top1']:.1%}, "
              f"P(profit) {hold['p_profit']:.1%}")
        print("  (ROI vs an ownership-sampled field is optimistic; use it to rank, not to bank.)")

    lineup_df = _lineups_to_dataframe(df, [candidates[i] for i in chosen], args.fmt)
    metrics = result.metrics.iloc[chosen].reset_index(drop=True).round(4)
    if holdout is not None:
        metrics["holdout_roi"] = holdout.metrics["roi"].round(4).to_numpy()
        metrics["holdout_top1_pct"] = holdout.metrics["top1_pct"].round(4).to_numpy()
    pd.concat([lineup_df, metrics], axis=1).to_csv(args.out, index=False)
    print(f"Wrote portfolio to {args.out}")

    if args.candidates_out:
        all_df = _lineups_to_dataframe(df, candidates, args.fmt).drop(columns="lineup_id")
        all_df = pd.concat([all_df, result.metrics.round(4)], axis=1)
        all_df["selected"] = False
        all_df.loc[chosen, "selected"] = True
        all_df.sort_values("roi", ascending=False).to_csv(args.candidates_out, index=False)
        print(f"Wrote {len(candidates)} graded candidates to {args.candidates_out}")

    if args.exposure_out and chosen:
        exp_df = exposure_report(df, [candidates[i] for i in chosen])
        field_own = pd.Series((field > 0).mean(axis=0), index=df["name"])
        exp_df["field_own"] = exp_df["name"].map(field_own).round(4)
        exp_df.to_csv(args.exposure_out, index=False)
        print(f"Wrote exposure report to {args.exposure_out}")


def _fetch_sgo(args: argparse.Namespace) -> list[dict]:
    events = sgo.fetch_events(
        api_key=args.api_key, league_id=args.league,
        starts_after=args.starts_after, starts_before=args.starts_before,
        bookmaker_id=args.bookmaker,
    )
    print(f"Fetched {len(events)} events from SportsGameOdds.")
    return events


def cmd_sgo_fetch(args: argparse.Namespace) -> None:
    events = _fetch_sgo(args)
    sgo.save_events(events, args.out)
    print(f"Wrote raw events to {args.out}")
    if args.lines_out:
        sgo.extract_game_lines(events).to_csv(args.lines_out, index=False)
        print(f"Wrote game lines to {args.lines_out}")
    if args.baseline_out:
        baseline = sgo.props_to_baseline(sgo.extract_player_props(events))
        baseline.to_csv(args.baseline_out, index=False)
        print(f"Wrote {len(baseline)} player baselines to {args.baseline_out}")


def cmd_sgo_enrich(args: argparse.Namespace) -> None:
    events = sgo.load_events(args.events) if args.events else _fetch_sgo(args)
    lines = sgo.extract_game_lines(events)
    baseline = sgo.props_to_baseline(sgo.extract_player_props(events))
    pool = pd.read_csv(args.pool, encoding="utf-8-sig")

    df, report = sgo.apply_sgo_baseline(
        pool,
        lines=None if args.no_lines else lines,
        baseline=None if args.no_props else baseline,
        weight=args.weight,
        update_lines=not args.keep_pool_lines,
    )
    df.to_csv(args.out, index=False)

    print(f"Game lines updated for {len(report.teams_updated)} teams"
          + (f"; no SGO line for: {', '.join(report.teams_unmatched)}" if report.teams_unmatched else ""))
    print(f"SGO prop baseline applied to {report.players_matched} players (weight={args.weight})")
    if report.players_unmatched:
        print(f"  no usable SGO props for: {', '.join(report.players_unmatched)}")
    missing = df["proj"].isna().sum()
    if missing:
        print(f"  WARNING: {missing} players still have no proj; fill them before `build`.")
    print(f"Wrote enriched pool to {args.out}")


def _add_sgo_fetch_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--api-key", default=None, help=f"defaults to ${sgo.API_KEY_ENV}")
    p.add_argument("--league", default="NFL")
    p.add_argument("--starts-after", default=None, help="ISO timestamp, e.g. 2026-10-04T00:00:00Z")
    p.add_argument("--starts-before", default=None)
    p.add_argument("--bookmaker", default=None,
                   help="restrict to bookmakerID(s), e.g. draftkings,fanduel")


def main() -> None:
    parser = argparse.ArgumentParser(prog="dfs_engine")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Simulate + build an optimized lineup portfolio")
    p_build.add_argument("--pool", required=True)
    p_build.add_argument("--n-lineups", type=int, default=150)
    p_build.add_argument("--trials", type=int, default=50_000)
    p_build.add_argument("--fmt", choices=["classic", "showdown"], default="classic")
    p_build.add_argument("--min-unique", type=int, default=2)
    p_build.add_argument("--dst-cap", type=float, default=0.22)
    p_build.add_argument("--seed", type=int, default=None)
    p_build.add_argument("--out", required=True)
    p_build.add_argument("--exposure-out", default=None)
    _add_sim_args(p_build)
    _add_stack_args(p_build)
    p_build.set_defaults(func=cmd_build)

    p_diag = sub.add_parser("diagnose", help="Run the Optimal%% leverage diagnostic")
    p_diag.add_argument("--pool", required=True)
    p_diag.add_argument("--trials", type=int, default=50_000)
    p_diag.add_argument("--chunk-size", type=int, default=5_000)
    p_diag.add_argument("--fmt", choices=["classic", "showdown"], default="classic")
    p_diag.add_argument("--seed", type=int, default=None)
    p_diag.add_argument("--out", required=True)
    _add_sim_args(p_diag)
    _add_stack_args(p_diag)
    p_diag.set_defaults(func=cmd_diagnose)

    p_out = sub.add_parser("outcomes", help="Per-player median/ceiling/boom/bust (+ Optimal%%) report")
    p_out.add_argument("--pool", required=True)
    p_out.add_argument("--trials", type=int, default=20_000)
    p_out.add_argument("--optimal-trials", type=int, default=0,
                       help="also solve this many trials for Optimal%%/leverage (0 = skip)")
    p_out.add_argument("--fmt", choices=["classic", "showdown"], default="classic")
    p_out.add_argument("--boom-mult", type=float, default=4.0, help="boom = score >= mult x salary/1000")
    p_out.add_argument("--bust-mult", type=float, default=2.0, help="bust = score < mult x salary/1000")
    p_out.add_argument("--seed", type=int, default=None)
    p_out.add_argument("--out", required=True)
    _add_sim_args(p_out)
    _add_stack_args(p_out)
    p_out.set_defaults(func=cmd_outcomes)

    p_con = sub.add_parser("contest", help="Candidate pool -> field sim -> contest sim -> portfolio")
    p_con.add_argument("--pool", required=True)
    p_con.add_argument("--fmt", choices=["classic", "showdown"], default="classic")
    p_con.add_argument("--n-lineups", type=int, default=150)
    p_con.add_argument("--candidates", type=int, default=2000, help="candidate lineups to generate")
    p_con.add_argument("--gen-trials", type=int, default=5000, help="sims used to build candidates")
    p_con.add_argument("--eval-trials", type=int, default=5000, help="independent sims used to grade them")
    p_con.add_argument("--holdout-trials", type=int, default=3000,
                       help="fresh sims to re-grade the chosen portfolio (honest ROI estimate)")
    p_con.add_argument("--field-size", type=int, default=20000,
                       help="sampled field lineups standing in for the contest")
    p_con.add_argument("--field-min-salary", type=int, default=None,
                       help="default $49,000 Classic / $47,000 Showdown (auto-lowered for tiny pools)")
    p_con.add_argument("--field-stack-mix", default="0.25,0.40,0.35",
                       help="field share with 0,1,2+ of the QB's own WR/TE ('none' to disable)")
    p_con.add_argument("--contest-size", type=int, default=100_000)
    p_con.add_argument("--entry-fee", type=float, default=20.0)
    p_con.add_argument("--rake", type=float, default=0.15)
    p_con.add_argument("--paid-frac", type=float, default=0.22)
    p_con.add_argument("--first-frac", type=float, default=0.20, help="share of prize pool to 1st")
    p_con.add_argument("--min-cash-mult", type=float, default=2.0)
    p_con.add_argument("--payouts", default=None,
                       help="real payout CSV (rank_min,rank_max,payout); overrides the stylized curve")
    p_con.add_argument("--objective", choices=["roi", "top1"], default="roi")
    p_con.add_argument("--min-unique", type=int, default=2)
    p_con.add_argument("--dst-cap", type=float, default=0.22)
    p_con.add_argument("--max-exposure", type=float, default=1.0)
    p_con.add_argument("--seed", type=int, default=None)
    p_con.add_argument("--out", required=True)
    p_con.add_argument("--candidates-out", default=None)
    p_con.add_argument("--exposure-out", default=None)
    _add_sim_args(p_con)
    _add_stack_args(p_con)
    p_con.set_defaults(func=cmd_contest)

    p_fetch = sub.add_parser("sgo-fetch", help="Download SportsGameOdds events (lines + props) to JSON")
    _add_sgo_fetch_args(p_fetch)
    p_fetch.add_argument("--out", required=True, help="raw events JSON path")
    p_fetch.add_argument("--lines-out", default=None, help="optional game-lines CSV")
    p_fetch.add_argument("--baseline-out", default=None, help="optional player prop-baseline CSV")
    p_fetch.set_defaults(func=cmd_sgo_fetch)

    p_enrich = sub.add_parser("sgo-enrich", help="Blend SportsGameOdds lines/props into a player pool CSV")
    _add_sgo_fetch_args(p_enrich)
    p_enrich.add_argument("--pool", required=True)
    p_enrich.add_argument("--events", default=None,
                          help="events JSON from sgo-fetch (omit to fetch live)")
    p_enrich.add_argument("--weight", type=float, default=0.5,
                          help="weight on the SGO baseline when blending with pool proj (0-1)")
    p_enrich.add_argument("--keep-pool-lines", action="store_true",
                          help="only fill blank team_total/game_total/spread instead of overwriting")
    p_enrich.add_argument("--no-lines", action="store_true", help="don't touch Vegas line columns")
    p_enrich.add_argument("--no-props", action="store_true", help="don't touch proj/ceiling")
    p_enrich.add_argument("--out", required=True)
    p_enrich.set_defaults(func=cmd_sgo_enrich)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
