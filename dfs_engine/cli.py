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
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .data import load_player_pool, validate_showdown_pool
from .simulate import simulate_player_scores
from .portfolio import build_portfolio, exposure_report
from .diagnostics import run_optimal_pct_chunk, leverage_report


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


def cmd_build(args: argparse.Namespace) -> None:
    df = load_player_pool(args.pool)
    if args.fmt == "showdown":
        validate_showdown_pool(df)

    print(f"Loaded {len(df)} players. Simulating {args.trials} correlated trials...")
    scores = simulate_player_scores(df, n_trials=args.trials, seed=args.seed)

    print(f"Building {args.n_lineups} lineups ({args.fmt})...")
    lineups = build_portfolio(
        df, scores, n_lineups=args.n_lineups, fmt=args.fmt,
        min_unique=args.min_unique, dst_cap=args.dst_cap, seed=args.seed,
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

    n_players = len(df)
    counts = np.zeros(n_players, dtype=int)
    done = 0
    chunk_size = args.chunk_size

    print(f"Running Optimal% diagnostic: {args.trials} trials in chunks of {chunk_size}...")
    while done < args.trials:
        this_chunk = min(chunk_size, args.trials - done)
        scores = simulate_player_scores(df, n_trials=this_chunk, seed=(args.seed or 0) + done)
        counts += run_optimal_pct_chunk(df, scores, fmt=args.fmt)
        done += this_chunk
        print(f"  {done}/{args.trials} trials complete")

    report = leverage_report(df, counts, done)
    report.to_csv(args.out, index=False)
    print(f"Wrote leverage report to {args.out}")


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
    p_build.set_defaults(func=cmd_build)

    p_diag = sub.add_parser("diagnose", help="Run the Optimal%% leverage diagnostic")
    p_diag.add_argument("--pool", required=True)
    p_diag.add_argument("--trials", type=int, default=50_000)
    p_diag.add_argument("--chunk-size", type=int, default=5_000)
    p_diag.add_argument("--fmt", choices=["classic", "showdown"], default="classic")
    p_diag.add_argument("--seed", type=int, default=None)
    p_diag.add_argument("--out", required=True)
    p_diag.set_defaults(func=cmd_diagnose)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
