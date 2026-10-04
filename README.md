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
7. **Ceiling calibration**: each player's distribution is then rescaled
   so his simulated 85th percentile lands on the `ceiling` column while
   his mean stays on `proj` (Stokastic/SaberSim-style median + ceiling
   fitting). It's a per-player affine rescale, so the correlations from
   steps 1–5 are preserved. Change the percentile with `--ceiling-pct`,
   or turn it off with `--no-ceiling-calibration`.
8. **Default ceilings fitted to real contests**: players without a
   ceiling get a position default. Its width (`--spread-scale`) is chosen
   so a simulated field's score spread matches three real 2026
   Millionaire Makers: top 0.1% / top 1% / top 20% at 1.66× / 1.51× /
   1.19× the median score. `contest` and `dk-run` re-fit it on every
   slate.

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

### Player outcome table (median / ceiling / boom / bust / Optimal%)

```bash
python -m dfs_engine.cli outcomes \
    --pool data/sample_classic_pool.csv \
    --trials 20000 --optimal-trials 2000 \
    --out outcomes.csv
```

One row per player with `sim_mean`, `sim_median`, `sim_p85`, `sim_p99`,
`sim_std`, `boom_pct` (score ≥ 4× salary/1000), `bust_pct`
(< 2× salary/1000), and `pts_per_k`. With `--optimal-trials N` it also
solves N trials for `optimal_pct` and `leverage` (Optimal% − Own%).
Thresholds: `--boom-mult` / `--bust-mult`.

### Stacking rules (Classic)

`build`, `diagnose` and `outcomes` accept hard correlation constraints,
applied to every MILP solve:

| Flag | Meaning |
|---|---|
| `--qb-stack N` | at least N same-team players from `--stack-positions` (default `WR,TE`) with your QB |
| `--bring-back N` | at least N players from your QB's opponent |
| `--max-vs-dst N` | at most N offensive players facing your DST (`0` = never) |
| `--max-per-team N` | at most N players from any one team |

```bash
python -m dfs_engine.cli build --pool data/week5_players.csv \
    --n-lineups 150 --qb-stack 2 --bring-back 1 --max-vs-dst 0 --out lineups.csv
```

Rules are ignored (with a note) for Showdown. On a very small pool they
can make every lineup infeasible. The bundled one-game sample, for
example, can't satisfy `--max-vs-dst` at all, and the builder then
reports 0 lineups.

### Contest sim + portfolio selection (SaberSim-style)

`build` picks lineups that are optimal in individual simulations. `contest`
goes further, the way SaberSim and Stokastic's contest sims do: it grades
lineups against a simulated **field** under the contest's **payout
structure**, then picks the portfolio from those results.

```bash
python -m dfs_engine.cli contest \
    --pool data/week5_players_sgo.csv \
    --n-lineups 150 --candidates 3000 \
    --contest-size 200000 --entry-fee 20 \
    --qb-stack 2 --bring-back 1 --objective roi \
    --out portfolio.csv --candidates-out candidates.csv --exposure-out exposure.csv
```

What it does:

1. **Candidates.** Builds `--candidates` distinct lineups. Each one is
   MILP-optimal for a blend of the median projection and one simulated
   outcome, with a random blend weight, so the pool runs from safe builds to
   boom-or-bust ones. Stacking rules apply.
2. **Field.** Samples `--field-size` opponent lineups from projected
   ownership. All are salary-legal and at least `--field-min-salary`
   (default $49k Classic, $47k Showdown). They're resampled so that 25% /
   40% / 35% pair the QB with 0 / 1 / 2+ of his own pass catchers
   (`--field-stack-mix`). The sampling weights are then calibrated so the
   field's ownership matches your `own` column.
3. **Contest sim.** Scores candidates and field on a *separate* set of
   simulations (`--eval-trials`), so lineups aren't graded on the outcomes
   they were built from. Each sampled field lineup stands for
   `contest_size / field_size` real entries. A finish therefore covers a
   window of ranks, and the lineup is paid the average prize across it.
   Ties and duplicates widen the window, which splits prizes the way
   DraftKings does.
4. **Portfolio.** Greedy selection under the same ownership-tiered
   exposure caps as `build`, plus `--max-exposure` and `--min-unique`.
   - `--objective roi` takes the highest expected payout first.
   - `--objective top1` makes each pick maximize the share of simulations
     where *some* entry finishes top 1%. That favors lineups that win in
     different outcomes, so it's better for diversified mass-multi-entry.
5. **Holdout.** The chosen portfolio is re-graded on fresh simulations
   (`--holdout-trials`), so the reported ROI isn't inflated by having
   picked the luckiest of thousands.

Payouts: by default a stylized top-heavy curve (`--rake`, `--paid-frac`,
`--first-frac`, `--min-cash-mult`). For a real contest, pass its payout
table: `--payouts payouts.csv` with columns `rank_min,rank_max,payout`.

Each lineup in `portfolio.csv` gets `exp_payout`, `roi`, `top1_pct`,
`cash_pct`, `win_pct`, `exp_dupes`, `holdout_roi` and `holdout_top1_pct`.
`exposure.csv` adds `field_own` (the simulated field's realized
ownership) next to your exposure.

**Read the ROI as a ranking, not a forecast.** The field is sampled from
ownership; it isn't a model of real opponents' skill. And candidates are
graded by the same projections they were optimized for, so absolute ROI
comes out optimistic, as in every sim tool. The comparisons between
lineups (ROI, top-1%, duplication) are the useful signal.

Runtime: candidate generation dominates at ~40 ms per MILP solve, or
~115 ms with stacking rules on a full slate. So `--candidates 3000` takes
about 2–6 minutes. Field sampling and the contest sim take seconds.

### DraftKings entry file → upload file

Download your entries from DraftKings (Lineups → Edit Entries → Download →
`DKEntries.csv`). That file holds every entry you've reserved *and* the
slate's full player list with DraftKings IDs. The `dk-*` commands turn it
into an upload-ready file:

```bash
# 1. Pool with DK IDs + kickoff times, merged with your projections
#    (and optionally hand-entered Vegas lines instead of step 2)
python -m dfs_engine.cli dk-pool --entries DKEntries.csv \
    --projections my_projections.csv [--lines lines.csv] --out data/main_pool.csv

# 2. Vegas lines (+ prop baseline) from SportsGameOdds
python -m dfs_engine.cli sgo-enrich --pool data/main_pool.csv \
    --events data/main_sgo.json --out data/main_pool_sgo.csv

# 3. Contest size/payout estimates (edit to match the DK lobby)
python -m dfs_engine.cli dk-contests --entries DKEntries.csv --out contests.csv

# 4. Optimize every entry and write the upload file
python -m dfs_engine.cli dk-run --pool data/main_pool_sgo.csv \
    --entries DKEntries.csv --contests contests.csv \
    --qb-stack 2 --bring-back 1 \
    --upload-out dk_upload.csv --report-out entry_report.csv --exposure-out exposure.csv
```

Then upload `dk_upload.csv` on DraftKings' Edit Entries page.

- **Projections input.** A CSV export or a table copied from a website and
  saved as text (tab-separated is fine) both work. Headers are matched
  loosely: `Player`/`Name`, `Pos`, `Team`, `Proj`/`Projected Points`/`FPTS`,
  `Own`/`Ownership`/`Blended Ownership`, `Ceiling`. Values like `18.5%` and
  `$7,700` are parsed. `dk-pool` reports how many players matched, which
  names didn't, and the ownership total (a full slate is ~900%).
- **Manual Vegas lines.** `--lines` takes a CSV of `team,spread,total`
  (negative spread = favorite). One team per game is enough; the
  opponent's spread and both team totals are derived.
- **One candidate pool and one field are shared** by every contest in the
  file. Each contest is priced under its own size and payout curve, and
  gets its own portfolio for the number of entries you hold in it.
- **Contest estimates.** `dk-contests` parses the prize pool from each
  contest name ("$2.75M", "[$1M to 1st]") and estimates size as prize
  pool ÷ (fee × (1 − rake)). Freerolls have no fee, so their size is a
  marked guess. Edit `contest_size`, `first_frac` and `paid_frac`, or set
  `payouts_file` to a `rank_min,rank_max,payout` CSV.
- **The upload file** uses player IDs only (as DraftKings asks) and copies
  the entry columns verbatim. Every lineup is re-checked against DK's
  Classic rules: salary, slots, and at least 2 games.
- **FLEX for late swap.** FLEX holds the latest-kickoff player of the
  position with an extra body, which keeps late-swap options open.
- **Reusing lineups across contests.** By default each contest takes its
  own best lineups, so the same lineup can appear in several contests.
  `--unique-across-contests` forbids that; the highest-fee contest picks
  first.
- **Missing players.** Players without a projection are dropped at load,
  and those without ownership count as 0%. `dk-run` warns if most
  ownership is missing, since the simulated field depends on it.

### Learning from past contests (`field-study`)

DraftKings' contest-standings export (contest page → Export Lineups, a
`contest-standings-<id>.csv`, often zipped) holds every entry's lineup
and score, plus each player's actual `%Drafted` and points. `field-study`
turns those files into `data/field_profile.json`:

```bash
python -m dfs_engine.cli field-study contest-standings-*.zip \
    --entries DKEntries.csv --out data/field_profile.json
```

`--entries` supplies a current DraftKings player list, used to map names
to teams for the stacking stats. The profile keeps aggregates only (no
entries or usernames). From three 2026 Millionaire Makers (Weeks 1–3,
162k–831k entries each):

| Field behavior | Measured | Used for |
|---|---|---|
| Score at top 0.1% / 1% / 20% ÷ median | 1.66 / 1.51 / 1.19 | fitting the simulation's spread |
| QB with 0 / 1 / 2+ of his own WR/TE | 19% / 53% / 28% | the field sampler's stack mix |
| FLEX filled by RB / WR / TE | 42% / 33% / 25% | ownership position totals |
| Most-owned RB / WR / TE / QB / DST | ~45% / 26% / 26% / 12% / 19% | ownership checks + estimator |
| Entries in a duplicated lineup | 6–10% | known gap (see limitations) |

**Ownership checks and fallback.** `dk-pool` and `dk-run` compare your
ownership column to these real fields, rescaled to the slate's size. They
flag totals far from 900% and positions that look too flat or too chalky.
`dk-pool --estimate-own` fills missing ownership from a fallback model:
players are ranked within each position by value, projection and team
total, and the k-th ranked player gets the real fields' k-th-ranked
ownership. `--own-blend 0.3` mixes 30% of that estimate into provided
ownership, and `--own-sharpen 1.2` makes the provided chalk chalkier. The
exports contain no salaries or pre-lock projections, so the ranking
weights are judgment, not fitted. Prefer a published ownership
projection; see `own-eval` below for how they compared on Week 2.

### Scoring an ownership source (`own-eval`)

```bash
python -m dfs_engine.cli own-eval --projections week2_projections.csv \
    --standings contest-standings-<week2 id>.zip
```

`own-eval` matches a past week's projection file to that week's actual
results. It reports ownership correlation and average error (overall and
by position), projection bias, and the error at several chalk-sharpening
settings.

On Week 2 of 2026 (148 players, GoingFor2-based file):
- **Ownership:** correlation 0.85, average error 2.5 points, well
  calibrated, but the heaviest chalk was under-projected (Bijan 38% vs
  47% actual).
- **Sharpening:** `--own-sharpen 1.2` lowered the error slightly (2.48 →
  2.42) and held up when each position was left out in turn (4 of 5).
- **Our fallback estimator:** clearly worse (correlation 0.58), and every
  blend into GoingFor2 made it worse. Keep `--own-blend` at 0 when a real
  ownership projection is available.
- **Projections vs actual points:** correlation 0.55; RBs ran +3.3 points
  and QBs +2.3 points high.

One week is a small sample; score more weeks before trusting these
settings.

### Pull in SportsGameOdds lines + prop-based baseline projections

[SportsGameOdds](https://sportsgameodds.com) aggregates sportsbook odds and
publishes a vig-free consensus ("fair") line for every market. The
`sgo-*` commands use it two ways:

1. **Vegas lines** — the consensus game total, spread and team totals
   overwrite `game_total` / `spread` / `team_total` for every matched team,
   so the correlated sim is anchored to the current market.
2. **Market-implied baseline projection** — each player's full-game props
   (passing/rushing/receiving yards, receptions, pass/rush/rec/anytime TDs,
   INTs) are converted to an expected DraftKings score: yardage props are
   modeled as log-normal around the line (so the mean sits a bit above the
   median line, and 100/300-yard bonuses are priced from the same
   distribution), and count props as Poisson, using the vig-free over
   probability. The result is blended into `proj`:
   `proj = (1 - weight) * proj + weight * sgo_proj` (default `weight=0.5`).
   A blank `proj` is filled with `sgo_proj` outright, and `ceiling` is
   scaled by the same ratio.

```bash
export SPORTSGAMEODDS_API_KEY=your_key

# 1. Snapshot the slate's events (lines + props) to JSON
python -m dfs_engine.cli sgo-fetch \
    --starts-after 2026-10-04T00:00:00Z --starts-before 2026-10-06T12:00:00Z \
    --out data/week5_sgo_events.json \
    --lines-out data/week5_lines.csv --baseline-out data/week5_sgo_baseline.csv

# 2. Blend into your player pool, then build as usual
python -m dfs_engine.cli sgo-enrich \
    --pool data/week5_players.csv --events data/week5_sgo_events.json \
    --weight 0.5 --out data/week5_players_sgo.csv
python -m dfs_engine.cli build --pool data/week5_players_sgo.csv ...
```

Omit `--events` to fetch live inside `sgo-enrich`. Useful flags:
`--keep-pool-lines` (only fill blank line columns), `--no-lines`,
`--no-props`, `--bookmaker draftkings` (restrict odds to one or more books).
The enriched CSV keeps audit columns (`proj_input`, `sgo_proj`,
`sgo_markets`, `sgo_imputed`, `proj_source`) so you can see where your
numbers and the market disagree. Try it offline with the bundled sample:

```bash
python -m dfs_engine.cli sgo-enrich --pool data/sample_classic_pool.csv \
    --events data/sample_sgo_events.json --out enriched.csv
```

Notes:
- Players are matched by normalized name (accents, punctuation and
  Jr./III suffixes stripped), tie-broken by team. Unmatched players and
  DSTs keep their input projection, and the command lists them.
- A player needs at least one core market (QB: pass yards; RB: rush or
  rec yards; WR/TE: rec yards or receptions) to get a baseline. When a TD
  or INT market is missing it is imputed from yardage (listed in
  `sgo_imputed`), but players with thin prop coverage — e.g. an RB with
  only a rushing-yards line — will be understated, so check
  `sgo_markets` before leaning on a low `sgo_proj`.
- Fumbles, return TDs and 2-pt conversions aren't priced by props, so
  the baseline is slightly conservative.

## Player pool CSV format

Required columns: `name, team, opp, position, salary, proj, own, ceiling,
team_total, game_total, spread`.

- `position` must be one of `QB, RB, WR, TE, DST`.
- Rows with a blank or zero `proj` are dropped at load, so a full DK player
  list can be used directly; a blank `own` counts as 0%.
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
  simulate.py     # correlated Monte Carlo player-score simulation (+ ceiling calibration)
  optimize.py     # MILP solver (Classic + Showdown, DK game/team rules) + stacking rules
  outcomes.py     # per-player median/p85/p99/boom/bust/Optimal% report
  contest.py      # candidates, ownership-sampled field, contest sim, portfolio selection
  dk.py           # DraftKings entry file: parse, pool + projections merge, slots, upload
  history.py      # contest-standings exports -> field profile (score spread, stacks, ownership)
  ownership.py    # ownership sanity checks + fallback estimator from real-field curves
  portfolio.py    # N-lineup portfolio builder (exposure caps, uniqueness)
  diagnostics.py  # Optimal% leverage diagnostic
  sportsgameodds.py # SportsGameOdds fetch, game lines, prop -> DK baseline
  cli.py          # command-line entry points
tests/
  test_engine.py  # pytest-style tests
  test_sportsgameodds.py
  test_sim_tools.py  # ceiling calibration, outcomes report, stacking rules
  test_contest.py    # payouts, field sampler, contest sim, portfolio selection
  test_dk.py         # entry-file parsing, matching, DK roster rules, upload, dk-run end to end
  test_history.py    # standings parsing, field profile, ownership estimator, spread fitting
  run_tests.py    # standalone runner (no pytest dependency)
data/
  sample_classic_pool.csv
  sample_sgo_events.json  # SGO /events snapshot matching the sample pool
  field_profile.json      # aggregate field behavior from three 2026 Millys
```

## Testing

```bash
python3 tests/run_tests.py
```

(or `pytest tests/` if pytest is available in your environment)

## Known limitations

- **Exposure/portfolio construction** (both `build` and `contest`) is a greedy heuristic (draw a
  trial, solve, accept if it satisfies caps/uniqueness), not a globally
  optimal portfolio solve. On very small or thin player pools this can
  hit a genuine combinatorial ceiling (fewer unique lineups available
  than requested) — the builder detects this and stops rather than
  looping forever.
- **This engine does not generate ownership itself**, and its own
  projections come only from the SportsGameOdds prop baseline above. For
  best results supply your own `proj` (from a paid provider or your
  model) and use SGO as the market baseline to blend against.
- **The simulated field duplicates too little.** Only ~1% of its entries
  share a lineup, versus 6–10% in real Millys, where chalk and
  optimizer-built lineups cluster. Prize splitting for chalky lineups is
  therefore underestimated, which slightly favors chalk.
- **Contest-sim ROI is optimistic in absolute terms.** The `contest`
  field is sampled from ownership with a stacking mix; it doesn't model
  the real field's skill or strategy distribution. Use simulated ROI /
  top-1% to *rank* lineups, not as an expected return. Our own entries
  also don't compete with each other in the sim.
