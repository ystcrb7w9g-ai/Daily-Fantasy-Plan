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
3. **Target-share and carry competition**: same-team WR/TE are partly
   *negatively* tied to each other (one's big game partly comes at a
   teammate's expense), and so are same-team RBs. The QB's own noise is
   tied to his pass catchers' (his passing line *is* their production),
   so QB~WR is strongly correlated while WR~WR is only weakly correlated.
4. **Asymmetric game script**: a team that out-scores its opponent by
   more than the spread implied runs more (RB volume up, passers down);
   the trailing team passes more. Both teams' passers also share a
   per-game **shootout** shock, which makes QB~opposing QB positive.
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
8. **Correlations measured from real games**: the knobs behind steps 1–4
   are fitted (`correlations.fit_simulation`) so the simulation's
   role-pair correlations match real DK scores from 2021–25 (nflverse
   weekly stats, `data/correlation_profile.json`):

   | pair | real | sim |
   |---|---|---|
   | QB1 ~ WR1 / WR2 / TE1 | +0.38 / +0.37 / +0.32 | +0.41 / +0.29 / +0.25 |
   | QB1 ~ RB1 | +0.09 | +0.05 |
   | RB1 ~ RB2 | −0.07 | −0.13 |
   | WR1 ~ WR2 | +0.09 | +0.07 |
   | QB1 ~ opp QB1 | +0.19 | +0.10 |
   | WR1 ~ opp WR1 | +0.11 | +0.12 |

   Before this fit, every pair in the sim was correlated at +0.3 to +0.5,
   because the team factor swamped everything else. That over-rewarded
   stacking whole teams. Refresh it each season with:
   `python -m dfs_engine.cli corr-measure --weekly stats_player_week_2021.csv ... --out data/correlation_profile.json`
   ([nflverse weekly files](https://github.com/nflverse/nflverse-data/releases/tag/stats_player)).
9. **Default ceilings fitted to real contests**: players without a
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
| `--max-per-game N` | at most N players from any one game (5 stops whole-game stacks) |
| `--flex-stacks` | `contest`/`dk-run`: treat `--qb-stack`/`--bring-back` as the *maximum*; candidates span QB+1 … QB+N, with and without the runback, and the contest sim picks the structure |
| `--max-game-stack-share X` | `contest`/`dk-run`: at most share X of a contest's lineups with 4+ players from one game |

Example (what we'd use next week): `--qb-stack 2 --bring-back 1 --flex-stacks
--max-per-game 4 --max-per-team 3 --max-vs-dst 0 --max-game-stack-share 0.5`.
Without `--flex-stacks`, QB+2 plus a runback forces every lineup to carry
four players from one game.

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
2. **Field.** Samples `--field-size` opponent lineups. 6% of them
   (`--field-optimizer-share`) are an **optimizer slice**: lineups that
   are optimal for your projections plus per-user noise, repeated in
   proportion to how often each comes out optimal. Popular builds
   therefore duplicate the way real fields do. With this slice, a
   simulated 165k-entry Milly has 6.7% of entries in duplicated lineups
   and a top lineup at ~60–140 copies, versus 5.7–10.2% and 69–346 in real
   2026 Millys. The rest are sampled from projected ownership, all
   salary-legal and at least `--field-min-salary` (default $49k Classic,
   $47k Showdown). They're resampled so that 19% / 53% / 28% pair the QB
   with 0 / 1 / 2+ of his own pass catchers, as measured in real Millys
   (`--field-stack-mix`). The sampling weights are then calibrated so the
   whole field's ownership, optimizer slice included, matches your `own`
   column.
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
   - `--objective emax` makes each pick maximize the expected payout of
     the portfolio's *best* entry. This is the objective from Bergman et
     al.: entries should play off each other.
   - `--objective auto` is the default: `top1` for contests of 10,000+
     entries, `roi` below that.

     Why: if our projections were exactly right, `roi` would be optimal,
     because payouts add up. They aren't right, so I graded 20-lineup
     150k-entry portfolios on four sets of "wrong-projection" worlds,
     where every player's true mean was off from our projection by about
     25%:

     | objective | ROI, same model | ROI, wrong projections (avg) | P(any top 1%), wrong projections |
     |---|---|---|---|
     | roi | +165% | +191% (range +130 to +317) | 33.9% |
     | top1 | +151% | +189% (range +155 to +231) | **37.0%** |
     | emax | +141% | +184% (range +108 to +303) | 33.4% |

     All three come out at about the same ROI. `top1` gives more top-1%
     finishes and swings less from world to world, so it's the default for
     big GPPs. `emax` didn't beat it. The absolute ROIs are optimistic,
     as always.
   - **Punt cap.** Any player salaried under `--cheap-salary` (default
     $4,000 Classic, $3,000 Showdown; `0` turns it off) appears in at most
     `--cheap-cap` (default 40%) of the lineups. One punt who scores zero
     can't sink the whole portfolio.
   - If the caps leave entries empty, a relaxed pass fills them. It keeps
     the punt, RB-pair and game-stack share caps.
5. **Holdout.** The chosen portfolio is re-graded on fresh simulations
   (`--holdout-trials`), so the reported ROI isn't inflated by having
   picked the luckiest of thousands.

**Showdown correlation rules** (`--fmt showdown`, `--showdown-rules`):

- `strict` forbids:
  - two RBs from one team;
  - a QB with none of his pass catchers;
  - a WR/TE captain without his QB;
  - two DSTs.
- `soft` (default) builds candidates with the strict rules *and* a variant
  that allows a same-team RB pair. RB pairs are then limited to
  `--max-rb-pair-share` (default 20%) of the lineups. Real games support
  this: RB1 and RB2 scores are correlated at −0.07 overall. But in the 7%
  of games where an offense scores 5+ TDs, both RBs reach 15 DK points
  18% of the time, versus 4% otherwise. That was the MNF-winning Bijan +
  Brian Robinson Jr. build.
- `off` applies no correlation rules.

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

### Late swap (`dk-late-swap`)

Once the early games are under way, you know things the field didn't at
first lock:

- every locked player's points so far;
- the field's *actual* ownership;
- every opponent's lineup.

`dk-late-swap` uses all three to re-optimize each entry's slots that
haven't kicked off yet.

```bash
# ~10-15 min before the next lock: download your entries (DK > Upcoming >
# Edit Entries > download) and the contest's standings export, then
python -m dfs_engine.cli dk-late-swap --entries DKEntries_live.csv \
    --pool week5_pool.csv --standings contest-standings-<milly>.zip \
    --contests contests.csv --upload-out late_swap_upload.csv --report-out late_swap.csv
```

How it works:

1. **Locked players are fixed.** DK marks them "(LOCKED)", and any
   player whose game has kicked off is treated the same way. A locked
   player scores his actual points plus, for games still in progress, a
   simulated remainder. The game clock is estimated from kickoff time;
   pass `--assume-final` once those games are over.
2. **Options.** For each entry the command tries up to `--solves`
   completions of the open slots (salary cap and slot eligibility
   respected), plus the option of keeping the current lineup.
3. **Scoring.** Every option is ranked against the contest's *real*
   lineups from the standings export (your own entries excluded).
   - A standings file is matched to its contest through your entry IDs.
   - A file with none of your entries becomes the field for contests
     without their own export.
4. **Choice.** The best option by expected payout (`--objective top1`
   for top-1% odds) wins. It must beat keeping the lineup by
   `--min-gain` × the entry fee (default 2%), and two entries in one
   contest never end up identical.
5. **Missing players.** Rostered players missing from the pool are
   scored at their DK season average and are never swapped *in*.

The report shows each swap (out → in), points so far, and expected
payout and top-1% odds before and after. Upload the file on DK's Edit
Entries page. Locked players keep their IDs, so DK accepts the file as
is.

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
| Entries in a duplicated lineup | 6–10% (top lineup 69–346 copies) | the field's optimizer slice |

**Ownership checks and fallback.** `dk-pool` and `dk-run` compare your
ownership column to these real fields, rescaled to the slate's size. They
flag totals far from 900% and positions that look too flat or too chalky.
`dk-pool --estimate-own` fills missing ownership from our own ownership
model (see "Our ownership model" below) and writes it to an `own_model`
column next to the provided `own`, listing where the two disagree most.
`--own-blend 0.3` mixes 30% of that estimate into provided ownership, and
`--own-sharpen 1.2` makes the provided chalk chalkier. Over three scored
weeks, neither improved a published projection, so both are off by
default.

### Scoring an ownership source (`own-eval`)

```bash
python -m dfs_engine.cli own-eval --projections week2_projections.csv \
    --standings contest-standings-<week2 id>.zip
```

`own-eval` matches a past week's projection file to that week's actual
results. It reports ownership correlation and average error (overall and
by position), projection bias, and the error at several chalk-sharpening
settings.

Results for 2026 Weeks 1–3 (140–151 players per week, GoingFor2-based files):

| | Week 1 | Week 2 | Week 3 |
|---|---|---|---|
| Ownership correlation / avg error | 0.90 / 2.0 pts | 0.85 / 2.5 | 0.85 / 2.4 |
| Projection vs actual points (correlation) | 0.54 | 0.55 | 0.57 |
| RB projection bias | −3.8 | +3.3 | +0.6 |

- **Use GoingFor2 ownership as-is.** Chalk sharpening helped Weeks 2–3 but
  hurt Week 1, so over three weeks `--own-sharpen 1.0` (off) is best.
  A fitted correction on top of GoingFor2 also made it slightly worse.
- **Projection biases flip sign week to week,** so there's nothing
  consistent to correct.

### Our ownership model (`own-fit`, `dk-pool --estimate-own --week N`)

`dfs_engine/ownership_model.py` projects ownership from information that
every DFS player can see before lock. Per position it fits a softmax
model of how real %Drafted is split. The inputs are:

- points per $1k, projection and salary (all z-scored within the position);
- the rank of each player's value at his position;
- the Vegas team total and expected margin;
- last week's DK points;
- season-to-date DK points per game per $1k (the FPPG the DK lobby shows);
- the DK points per game of teammates in his position group who are ruled
  Out or Doubtful (the volume a backup inherits).

Lines, box scores and injury reports come from free
[nflverse](https://github.com/nflverse) files. They download to
`data/nflverse/` on first use, which is git-ignored; add
`--refresh-nflverse` once final injury designations are out. Players
beyond the realistic pool (≈2 QB, 2 RB, 4.2 WR, 1.6 TE and 2 DST per
game, by projection) share a 4% tail.

Leave-one-week-out on the 2026 Week 1–3 Millionaire Makers:

| | Week 1 | Week 2 | Week 3 | avg |
|---|---|---|---|---|
| **Ours** — correlation / avg error | 0.76 / 3.0 pts | 0.75 / 3.0 | 0.63 / 3.6 | 0.71 / 3.2 |
| Old log-linear fallback | | | | 0.65 / 3.5 |
| GoingFor2 (published) | 0.90 / 2.0 | 0.85 / 2.5 | 0.85 / 2.4 | 0.87 / 2.3 |

GoingFor2 is still clearly better: it reacts to news, which a model
built only on box scores can't. Mixing ours into it made it slightly worse, so
keep GoingFor2's numbers as `own` and use ours:

- as the fallback when no ownership projection is available;
- as a leverage scan: the `own_model` column, plus the "we expect
  MORE / LESS" list `dk-pool` prints, shows where the two disagree.

Optimal% from our sim, last week's targets + carries, and lighter
regularization were also tested. None helped on held-out weeks.

Refit after every week of contest results. Each `--week` takes the
projections file you used (with salary and that source's ownership, so
the run also scores it), the Milly contest-standings export, and the NFL
week number:

```bash
python -m dfs_engine.cli own-fit \
    --week week1_projections.csv contest-standings-<w1>.zip 1 \
    --week week2_projections.csv contest-standings-<w2>.zip 2 \
    --week week3_projections.csv contest-standings-<w3>.zip 3
```

Use it on a slate:

```bash
python -m dfs_engine.cli dk-pool --entries DKEntries.csv --projections goingfor2.tsv \
    --estimate-own --week 5 --refresh-nflverse --out week5_pool.csv
```

`--week` also fills any missing Vegas lines from the nflverse schedule.

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
  correlations.py # real player-pair correlations (nflverse) + fitting the sim to them
  ownership_model.py # our ownership projections: nflverse inputs, softmax fit, leave-one-week-out
  lateswap.py     # late swap: live standings -> actual points/ownership/field -> re-optimized open slots
  cli.py          # command-line entry points
tests/
  test_engine.py  # pytest-style tests
  test_sportsgameodds.py
  test_sim_tools.py  # ceiling calibration, outcomes report, stacking rules
  test_contest.py    # payouts, field sampler, contest sim, portfolio selection
  test_dk.py         # entry-file parsing, matching, DK roster rules, upload, dk-run end to end
  test_history.py    # standings parsing, field profile, ownership estimator, spread fitting
  test_correlations.py # DK scoring from nflverse, measured vs simulated correlations
  test_ownership_model.py # schedule lines, recency/injury inputs, softmax fit, CV
  test_lateswap.py   # live entry parsing, game clock, completions, end-to-end swap
  run_tests.py    # standalone runner (no pytest dependency)
data/
  sample_classic_pool.csv
  sample_sgo_events.json  # SGO /events snapshot matching the sample pool
  field_profile.json      # aggregate field behavior from three 2026 Millys
  correlation_profile.json # role-pair DK score correlations, 2021-25 regular seasons
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
- **Contest-sim ROI is optimistic in absolute terms.** The `contest`
  field is sampled from ownership with a stacking mix; it doesn't model
  the real field's skill or strategy distribution. Use simulated ROI /
  top-1% to *rank* lineups, not as an expected return. Our own entries
  also don't compete with each other in the sim.
