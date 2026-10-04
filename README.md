# dfs-engine

A correlated Monte Carlo + MILP optimal-lineup pipeline for DraftKings NFL
large-field GPP tournaments (Millionaire Maker style), supporting both
**Classic** and **Showdown** (single-game) contest formats.

This is the productionized version of a pipeline originally built
iteratively (week by week, across Weeks 1–3 of the 2026 season) as ad-hoc
scripts. This repo consolidates that work into a tested, reusable package.

## Why this isn't "independent noise per player"

Most naive DFS projection tools simulate each player's fantasy score with
independent random noise. That misses how NFL games actually work: a big
game for a team's QB usually means a big game for his pass-catchers too,
and a bad script for the opposing defense. This engine instead simulates
at the **game** and **team** level first, then derives player scores from
shared, correlated factors — the same basic approach real sim-based tools
(e.g. SaberSim) use:

1. **Game total** is drawn once per trial, with variance around the
   Vegas-implied total.
2. **Team-level offense factor** splits that game total between the two
   teams and correlates every one of a team's players together.
3. **Target-share competition**: same-team WR/TE are *negatively*
   correlated with each other (one's big game partly comes at a
   teammate's expense), so the model doesn't let you "stack" a whole
   receiving corps as if they were independent.
4. **Asymmetric game script**: a leading team's RBs get a volume boost;
   a trailing team's QB/WR/TE get a volume boost.
5. **Right-skewed residual noise** (Gaussian + shifted-exponential blend)
   models real boom risk (long TDs, garbage-time volume) better than a
   symmetric normal distribution.
6. **QB-tier variance scaling**: cheaper/less-proven QBs get higher
   variance as a proxy for backup/game-manager bust risk.

## Install

```bash
pip install -r requirements.txt
```

## Usage

### Build a lineup portfolio

```bash
python -m dfs_engine.cli build \
    --pool data/sample_classic_pool.csv \
    --n-lineups 150 \
    --trials 50000 \
    --fmt classic \
    --out lineups.csv \
    --exposure-out exposure.csv
```

For a Showdown (single-game) slate, pass `--fmt showdown` with a pool CSV
containing exactly two teams.

### Run the Optimal% leverage diagnostic

This is the "SaberSim-style" ownership leverage calculation: how often a
player appears in the MILP-optimal lineup across many independent
simulated trials, compared to his projected ownership.

```bash
python -m dfs_engine.cli diagnose \
    --pool data/sample_classic_pool.csv \
    --trials 50000 \
    --chunk-size 5000 \
    --fmt classic \
    --out leverage.csv
```

Chunked execution lets a large trial count (e.g. 50,000) run across
multiple invocations without needing to hold everything in memory or
time out — rerun with a higher `--trials` value and the same `--seed`
offset pattern to extend a prior run (see `dfs_engine/diagnostics.py`
for the lower-level `save_checkpoint` / `load_checkpoint` helpers if you
want to checkpoint across process restarts).

## Player pool CSV format

Required columns: `name, team, opp, position, salary, proj, own, ceiling,
team_total, game_total, spread`.

- `position` must be one of `QB, RB, WR, TE, DST`.
- `own` (projected ownership) can be given as a fraction (`0.18`) or a
  percentage (`18.0`) — the loader auto-detects and normalizes to a
  fraction.
- `team_total` / `game_total` / `spread` come from the Vegas lines for
  that player's game and drive the correlated simulation.

See `data/sample_classic_pool.csv` for a worked example (one game,
13 players).

## Project layout

```
dfs_engine/
  data.py         # CSV loading + validation
  simulate.py     # correlated Monte Carlo player-score simulation
  optimize.py     # MILP solver (Classic + Showdown)
  portfolio.py    # N-lineup portfolio builder (exposure caps, uniqueness)
  diagnostics.py  # Optimal% leverage diagnostic
  cli.py          # command-line entry points
tests/
  test_engine.py  # pytest-style tests
  run_tests.py    # standalone runner (no pytest dependency)
data/
  sample_classic_pool.csv
```

## Testing

```bash
python3 tests/run_tests.py
```

(or `pytest tests/` if pytest is available in your environment)

## Known limitations

- **Exposure/portfolio construction** is a greedy heuristic (draw a
  trial, solve, accept if it satisfies caps/uniqueness), not a globally
  optimal portfolio solve. On very small or thin player pools this can
  hit a genuine combinatorial ceiling (fewer unique lineups available
  than requested) — the builder detects this and stops rather than
  looping forever.
- **This engine does not generate projections, ownership, or Vegas
  lines itself.** It expects those as input (from a projections source
  such as a paid provider, or your own model) in the player pool CSV.
- **ROI/EV against a real massive-field contest** (hundreds of thousands
  of entries) is not implemented here. Estimating absolute payout
  probabilities against a field that large requires realistically
  modeling the *skill and strategy distribution* of the entire field,
  which is a much harder problem than this engine solves; relative
  lineup ranking (by simulated equity / Optimal%) is the more reliable
  signal this pipeline can give you.
