# dfs-engine

A correlated Monte Carlo + MILP optimal-lineup pipeline for DraftKings NFL
large-field GPP tournaments (Millionaire Maker style), supporting both
**Classic** and **Showdown** (single-game) contest formats.

This is the productionized version of a pipeline originally built
iteratively (week by week, across Weeks 1–3 of the 2026 season) as ad-hoc
scripts. This repo consolidates that work into a tested, reusable package.

## How this compares with the paid tools

The sites (SaberSim, Stokastic, FantasyLabs) and the published research
(Hunter, Vielma & Zaman 2016; Haugh & Singal 2021) share one process:
project, simulate games, model the field, simulate the contest, pick a
portfolio, then test what works on many past slates. The table maps each
step to what this engine does. Their internals aren't public; their side
comes from their exports and descriptions.

| Step | The paid tools | Here |
|---|---|---|
| Projections, ownership | Their own models, updated with news all morning | Yours (SaberSim, GoingFor2, DFF), optionally blended with ours; fixed when you run |
| Game simulation | Play-by-play (SaberSim): each sim one consistent game | Box scores per game with DK bonuses, kickers and DSTs (`--engine stat`, the Showdown default); factor model with a TD layer for Classic |
| Candidate lineups | Lineups that win sims; leverage and salary-leftover builds | Optimal for projection + one sim; `--leftover-share` / `--own-penalty-share` add contrarian builds |
| Opponent field | Shaped like the contest: size, buy-in, chalkiness | Ownership-sampled, matched to real fields' stacking, salary and duplication; per-contest `field_sharpen`; `--field-own-sd` for ownership error |
| Ranking and portfolio | ROI / top-finish rates; portfolio across contests | Top-1% coverage (default), ROI, best-entry, ceiling; `--global-exposure` caps players across contests |
| Testing what works | Years of slates | `backtest` on every past contest you keep |

Data and access gaps code can't close: real-time news, live prop markets,
and years of contest history.

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
   variance as a proxy for backup/game-manager bust risk. With the TD
   model (the default) a QB's own noise is also cut to 0.4x a receiver's:
   his line is mostly his team's passing game, so opposing QBs move
   together like real ones do (+0.20).
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
   stacking whole teams.

   **Correlations also depend on the player** (the "similar historical
   pairs" idea from Bergman et al.). With `dk-pool --week N`, each
   player gets his season targets per game and, for QBs, the share of
   points from rushing. The QB's link to his receivers then follows
   those profiles, fitted to real terciles (low / mid / high):

   | pair, split by | real | sim |
   |---|---|---|
   | QB1~WR1 by QB rushing share | +0.46 / +0.42 / +0.33 | +0.45 / +0.34 / +0.34 |
   | QB1~WR1 by WR1 targets | +0.30 / +0.40 / +0.43 | +0.27 / +0.40 / +0.45 |
   | QB1~TE1 by TE1 targets | +0.28 / +0.26 / +0.39 | +0.22 / +0.37 / +0.48 |
   | QB1~RB1 by RB1 targets | +0.04 / +0.10 / +0.14 | +0.08 / +0.09 / +0.12 |

   Stacking a running QB (Josh Allen, Lamar Jackson) buys less
   correlation than stacking a pocket passer, and pass-catching backs
   belong in QB stacks a little. Without those columns the sim falls back
   to role-only links. Refresh it each season with:
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

### Weekly workflow

| when | command | what you need |
|---|---|---|
| Thu–Sat | `dk-pool --entries DKEntries.csv --projections goingfor2.tsv --estimate-own --week N --proj-blend 0.25 --dst-blend 0.5` | entry file, GoingFor2 paste, (DFF) |
| or, with SaberSim | `dk-pool --entries DKEntries.csv --projections NFL_..._DK_Main.csv --out pool.csv`, then `dk-run ... --extra-lineups <SaberSim entries export>` | SaberSim projections export (+ its lineups) |
| Thu–Sat | `dk-contests`, then `dk-run --pool ... --flex-stacks --qb-stack 2 --bring-back 1 --max-per-game 4 --max-per-team 3 --max-vs-dst 0 --max-game-stack-share 0.5` | the pool above |
| Sun ~11:30 ET | re-run `dk-pool --refresh-nflverse` + `dk-run` if inactives change things | |
| Sun ~3:45 ET | `dk-late-swap --entries DKEntries_live.csv --standings <live export>` | live entry file + standings export |
| Mon | `review`, `field-structure`, `own-fit` with the new week | standings exports |


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
   salary-legal. Classic lineups spend like real ones: each draws its
   salary floor from $50k / $49.8k / $49.5k / $49k (40 / 40 / 15 / 5%;
   `--field-min-salary` sets one fixed floor instead; Showdown uses $47k).
   They're resampled so that 15% / 51% / 34% pair the QB with 0 / 1 / 2+
   of his own pass catchers (`--field-stack-mix`). That is the real Milly
   mix, recency-weighted over Weeks 1–5 (`contest.field_targets`). The
   sampling weights are then calibrated so the whole field's ownership,
   optimizer slice included, matches your `own` column.

   The field also copies how real lineups are built. In 160k real
   Milly lineups from Weeks 1–4 (`field-structure`,
   `data/field_structure.json`):

   | share of lineups with... | real Millys (4-week avg) | our old field | field now |
   |---|---|---|---|
   | a bring-back (player from the QB's opponent) | 43% | 22–26% | 40–42% |
   | the QB's own RB | 18% | 9–10% | 18–19% |
   | an RB with his own DST | 16% | 7–9% | 13–14% |
   | a player facing the lineup's own DST | 7% | 21–22% | 8% |
   | 4+ players from one game (not targeted) | 27–29% | 14–18% | 24–27% |

   Sampling boosts for those players are calibrated alongside ownership.
   Matching ownership alone wasn't enough. The realistic field re-ranks
   candidates: expected payouts correlate only 0.74 with the old field's,
   and 8 of the top 20 candidates change. Applied on slates of 6+ games.

   Week 5 (40k real Milly lineups vs a 20k simulated field on the same
   SaberSim pool) showed two more gaps, now fixed. The old $49k floor left
   40% of the field under $49.5k; real lineups do that 4% of the time.
   Real fields also stack more as the season goes: bring-backs ran
   44 / 44 / 41 / 44 / 58% in Weeks 1–5. The targets now weight the
   recent weeks most (half-life 2 weeks).

   | Week 5 Milly | real | old field | field now |
   |---|---|---|---|
   | QB + 2 or more of his pass catchers | 42% | 27% | 33% |
   | a bring-back | 58% | 42% | 47% |
   | 4+ players from one game | 41% | 27% | 31% |
   | the QB's own RB / an RB with his own DST | 14% / 16% | 18% / 16% | 15% / 16% |
   | a player facing the lineup's own DST | 8% | 8% | 9% |
   | average salary used | $49,875 | $49,555 | $49,883 |
   | at exactly $50k / at $49.5k or more | 50% / 96% | 15% / 60% | 52% / 97% |

   Stacking still falls short of Week 5, partly by design: a recency
   weight, not one week's spike. Real fields stack on purpose. Drawn
   independently, even the *actual* Week 5 ownership gives only 18% QB+2.
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
   - `--objective ceiling` ignores the field: each pick maximizes the
     sims in which some entry is the *best candidate* of that sim, like
     SaberSim's "Sim Optimals". `--objective blend` mixes the two,
     (1 − w) × top1 + w × ceiling with `--ceiling-weight w` (default
     0.5). Not the default: in both Showdowns we replayed it lowered the
     20-lineup portfolios' median score (see the Showdown section).
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

- `light` (default) forbids only two DSTs; the sims decide the stacks.
- `strict` forbids:
  - two RBs from one team;
  - a QB with none of his pass catchers;
  - a WR/TE captain without his QB;
  - two DSTs.
- `soft` builds candidates with the strict rules *and* a variant
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
  Classic rules: salary, slots, and at least 2 games (Showdown: see below).
- **FLEX for late swap.** FLEX holds the latest-kickoff player of the
  position with an extra body, which keeps late-swap options open.
- **Reusing lineups across contests.** By default each contest takes its
  own best lineups, so the same lineup can appear in several contests.
  `--unique-across-contests` forbids that; the highest-fee contest picks
  first.
- **One exposure budget for the whole slate.** By default the exposure
  caps (`--max-exposure`, the ownership tiers, the punt cap) apply contest
  by contest, so one player can top every contest's list. With
  `--global-exposure` they apply to all your entries together, the way
  the sites' portfolio tools treat the slate as one bet.
- **A field per contest type.** Small or high-stakes contests tend to play
  chalkier than projected. Add a `field_sharpen` column to the contests
  CSV (ownership raised to that power within each position; above 1 =
  chalkier) and that contest gets its own field. `backtest` prints the
  power that fit each past contest's real ownership: the 2026 Millys and
  primetime Showdowns ranged 0.85-1.15, so they keep the default of 1.
- **Which game simulation.** `--engine auto` (the default) plays
  Showdowns out as box scores (the stat engine below) and simulates
  Classic slates with the factor model; `--engine stat` / `--engine
  factor` forces either. The Backtest section has how each did.
- **More options** (all off by default; see the Backtest section for how
  they did): `--leftover-share` / `--own-penalty-share` (candidates that
  leave salary unused / fade the chalk), `--field-own-sd` (a field built
  from several ownership draws, for ownership error).
- **Missing players.** Players without a projection are dropped at load,
  and those without ownership count as 0%. `dk-run` warns if most
  ownership is missing, since the simulated field depends on it.

### SaberSim exports (`dk-pool --projections`, `dk-run --extra-lineups`)

`dk-pool` recognizes SaberSim's DK projections export (`DFS ID`, `SS Proj`,
`My Proj`, `My Own`, `dk_25_percentile` ... columns) and matches it to the
slate by DK ID:

- `proj` = **My Proj** (SaberSim's number unless you edited it), `own` =
  **My Own** (sums to 900%; `Adj Own`, ~945%, is ignored).
- A projection of **0** is SaberSim saying the player won't play -- often
  a questionable player it expects inactive -- so he's dropped, as are
  `Status` O/D/IR players.
- **Saber Team / Saber Total** replace Vegas as `team_total` / `game_total`
  (spread derived from them); `--week` only fills teams they don't cover.
- **dk_25 ... dk_99 percentiles** become `p25 p50 p75 p85 p95 p99`, and
  `ceiling` = p85. The simulator maps each player's correlated draws onto
  those quantiles at the same rank (`simulate.map_to_percentiles`): who
  booms together is still our game/team/stack model, the shape is
  SaberSim's. A p85-only fit can't carry their fat right tail -- on Week 5
  their WR p99 was 3.3x projection vs our 2.7x, the same thinness our
  Weeks 1-4 tail check found (real players beat our p95 9.9% of the time,
  not 5%).
- **Min Exp / Max Exp** become `min_exp` / `max_exp` (fractions).

Per-player exposure works with any pool: `min_exp` / `max_exp` columns, or
`dk-run --min-exp "Jahmyr Gibbs=30%" --max-exp "Brock Bowers=0"` (repeatable;
flags win over columns). A player's own max replaces the default caps,
max 0 bans him (and keeps him out of every candidate), and minimums are
filled first, from extra candidates built around each min-exposure player.

`dk-run --extra-lineups FILE` (repeatable) adds lineups from any DK-layout
file -- SaberSim's entries export, an old upload, a DKEntries download --
to the candidate pool. Each must be a valid roster of projected players
that meets the stack rules; the contest sim then plays them only where they
beat our own candidates, and the run prints how many it took per contest.

```bash
python -m dfs_engine.cli dk-pool --entries DKEntries.csv \
    --projections NFL_2026-10-11-100pm_DK_Main.csv --out data/pool_saber.csv
python -m dfs_engine.cli dk-run --pool data/pool_saber.csv --entries DKEntries.csv \
    --contests contests.csv --extra-lineups entries_dk_nfl_2026-10-11_100pm_main.csv \
    --flex-stacks --qb-stack 2 --bring-back 1 --max-per-game 4 --max-per-team 3 \
    --max-vs-dst 0 --max-game-stack-share 0.5 --max-exposure 0.4 \
    --eval-trials 50000 --holdout-trials 50000 --upload-out dk_upload.csv
```

Week 5 head-to-head (each portfolio graded in 40,000 fresh sims on SaberSim's
inputs; note the grader is our own contest sim, so read it as a strong hint,
not proof -- the Monday `review` is the real test):

| P(any top-1%) | SaberSim's lineups | our engine on SaberSim's inputs | + SaberSim's lineups as candidates |
|---|---|---|---|
| mini-MAX (6) | 10.3% | 11.4% | 11.4% |
| Quarter Jukebox (20) | 26.4% | 32.6% | 32.9% |
| Dime (20) | 23.3% | 33.1% | 32.5% |

SaberSim's single Milly lineup was the best one tested (3.0% top-1% vs our
2.5%) -- and that exposed a bug: `--max-game-stack-share 0.5` floored
0.5 x 1 entry to zero, banning every 4-man game stack in a single-entry
contest. Fixed: any positive share allows at least one lineup.

### Shared touchdowns (TD model, on by default)

The base simulator moves DK points with symmetric, correlated noise. Real
teammates co-move through shared stat events instead -- a TD pass scores for
the QB (4) and his receiver (6) at once -- so their dependence sits in the
boom games. Scored relative to expectation (season average x the game's Vegas
team total), 2021-25 games show it; the production knobs fit the average
correlations but not where they come from:

| 2021-25, relative to expectation | real | old knobs (`--classic-sim`) | first TD model | Oct 12 refit | QB refit (now) |
|---|---|---|---|---|---|
| QB1 top-5% game -> a WR/TE teammate top-10% | 86% | 66% | 76% | 76% | 83% |
| QB1~WR2 / QB1~TE1 correlation | 0.35 / 0.29 | 0.22 / 0.36 | 0.31 / 0.30 | 0.29 / 0.29 | 0.34 / 0.36 |
| QB1~opp QB1 correlation | 0.20 | 0.11 | 0.15 | 0.12 | 0.20 |
| QB1 + opp QB1 both top-10% (x chance) | 1.69x | | | 1.55x | 1.73x |
| RB1~RB2 correlation | -0.03 | -0.13 | -0.03 | -0.04 | -0.05 |
| QB1 + WR1 + WR2 all top-15% (x chance) | 3.9x | 3.6x | 4.5x | 3.7x | 4.1x |
| RMS error, all 21 teammate / 49 opponent role pairs | | 0.066 / 0.065 | 0.065 / 0.070 | 0.030 / 0.043 | 0.031 / 0.036 |
| fit loss (pairs + boom/bust lifts + team totals) | | 2.16 | 2.16 | 1.30 | 1.15 |

The TD model adds a team TD layer to every simulating command: each team's
passing and rushing TD counts are Poisson with rates scaled by its simulated
points (and a TD environment shared with its opponent); passing TDs go to the
top QB plus a receiver by weight, rushing TDs to the RBs / QB; the layer is
centered so means stay on `proj`. `simulate.TD_MODEL` holds all thirteen
correlation knobs, refit together. We scored 240 random draws, then ran a
local search around the best, on all role pairs plus the boom/bust lifts.
That search fit the pairs best (loss 1.11), but it ranked Week 4 lineups
worst. So its QB-catcher link and TD-layer knobs are pulled back toward the
first TD model, which costs a little fit (loss 1.30). `--classic-sim` goes
back to the old knobs.

**QB refit (Oct 13).** Two Showdowns in a row paid lineups with both QBs,
and the Oct 12 knobs had opposing QBs at 0.12 (real 0.20). None of the
game-level knobs could fix that: raising them made opposing *teams* move
together too much (0.26 already, real 0.19). The cause was the QB's own
noise. He carried as much as a receiver, although his line is the sum of
his team's passing game. `qb_noise_mult` 0.4 (his own noise vs a
receiver's), a QB-catcher link of 0.55 and slightly smaller game shocks
(`shootout_sd` 0.163, `td_game_sd` 0.275) fix the QB pairs and lower the
fit loss (1.26 -> 1.15 on the same seeds; it also pulls the opposing
team link down to 0.24). QB1~TE1 overshoots (0.36 vs 0.29). The ceiling
fit restores each QB's spread, so his range is unchanged; only who he
booms with moves.

It also ranks *real* lineups at least as well: best on Week 5, a tie on Week 4.
We took 40k sampled real Milly lineups and simulated the slate. In each
sim, a lineup scores a hit if it finishes in the top 1% of those 40k. The
AUC measures how well each lineup's hit rate picks out the lineups that
really finished top 1% (0.5 is a coin flip):

| AUC for a real top-1% finish | Week 5 (SaberSim pool) | Week 4 (public projections) |
|---|---|---|
| projected points alone, no sim | 0.468 | 0.452 |
| old knobs | 0.632 | 0.570 |
| first TD model | 0.643 | 0.578 |
| pure pair refit | 0.645 | 0.564 |
| Oct 12 refit | **0.649** | 0.569 |
| QB refit (now) | 0.641 | 0.570 |

The QB refit is a wash on AUC (rerun with more sims: Week 5 0.648 -> 0.641,
Week 4 0.563 -> 0.570), but it concentrates real winners at the top: of
the 5% of real lineups it rates best, 1.41% finished top 1% on Week 5
(1.17% before) and 0.97% on Week 4 (0.78%). On the 77,561 real Week 5 SNF
Showdown entries (with the Showdown field's QB pairing), its top-1% AUC
is 0.575 (0.558 before).

Week 4 differences are within noise (345 top-1% lineups). Still not modeled:
yardage and the 300/100-yard bonuses (in 70-90% of real boom games), and
starters' near-zero games (real p5 is 0.22-0.25x expectation for RB1/WR1;
ours is 0). SaberSim percentiles fix the marginal shapes when you have them.
On Week 5 they were the best-calibrated shapes we have: RMS 0.034 across
p5-p99, vs 0.055 for our own. Default curves fit to Weeks 1-4 did worse out
of sample (0.104), so none ship.

### Stat engine (`--engine stat`, the Showdown default)

SaberSim simulates plays; the factor model above simulates fantasy points.
The stat engine (`dfs_engine/statsim.py`; the default for Showdowns,
`--engine stat` for any simulating command) plays every game out as box
scores instead, then scores them the DraftKings way, 300/100-yard bonuses
included:

- each team gets dropbacks, sacks and carries from a shared game pace, and
  a team that trails passes more;
- targets and carries are split by each player's share, with week-to-week
  noise in the shares; catches, yards per catch and per carry follow
  2021-25 rates by position;
- a QB's passing yards are exactly his receivers' yards, and every passing
  TD is one receiver's TD; TD counts rise with the Vegas total and the
  offense's efficiency that game;
- turnovers hand the other team short fields; kickers score the drives
  that stall; a DST's points come from the opponent's sacks, interceptions,
  fumbles and points.

Usage comes from the pool: SaberSim's stat projections (`dk-pool` keeps
their catches, carries and TDs per player) when present, otherwise targets
per game or each projection split by position. The draws are then fit to
each player's projection and ceiling (or SaberSim's percentiles) exactly
as in the factor model, so the box scores decide who booms together and
the inputs decide by how much.

Fit against the same 2021-25 targets as the TD model (3 seeds x 12k
trials; `statsim.STAT_KNOBS`):

| relative to expectation | real | factor model (TD model) | stat engine |
|---|---|---|---|
| fit loss (pairs + boom/bust lifts + team totals) | | 1.15 | **0.97** |
| RMS error, 21 teammate / 49 opponent role pairs | | **0.031** / 0.036 | 0.034 / **0.028** |
| opposing team totals correlation | 0.19 | 0.25 | **0.18** |
| QB1 ~ opp QB1 | 0.20 | **0.20** | 0.16 |
| QB1 top-5% game -> a WR/TE teammate top-10% | 86% | 83% | **84%** |
| team-total spread (sd / mean) | 0.33 | 0.30 | **0.31** |

A stronger shared game shock (`game_score_sd` 0.235 instead of 0.203)
brings QB ~ opposing QB to the real 0.20 at a worse fit overall (loss
1.03). In the backtest it ranked both Showdowns' real entries worse (AUC
0.647 / 0.553 vs 0.697 / 0.609), so the best fit stays.

The factor model leaves kickers and defenses almost independent of
everything; real ones aren't. Measured on 2,698 real team-games and on the
two Showdown pools:

| correlation | real 2021-25 | factor model | stat engine |
|---|---|---|---|
| DST ~ opposing QB | -0.43 | -0.02 | -0.45 |
| DST ~ opposing RB1 / WR1 | -0.31 / -0.21 | -0.01 / -0.01 | -0.24 / -0.13 |
| DST ~ own RB1 | +0.07 | -0.01 | +0.08 |
| K ~ own QB | +0.10 | +0.02 | +0.09 |
| K ~ own DST | +0.27 | 0.00 | +0.18 |
| K ~ opposing DST | -0.37 | -0.01 | -0.26 |
| DST ~ opposing DST | -0.31 | 0.00 | -0.23 |
| RMS gap over 12 K/DST pairs | | 0.24 | **0.09** |

In the backtest (the seven contests in the Backtest section, same inputs
and seeds; the last column adds SaberSim's stat projections to the pool,
which only the Week 5 slates have):

| contest | ranking AUC: factor model | stat engine | stat + SaberSim stats |
|---|---|---|---|
| W1 / W2 / W3 / W4 Milly | 0.526 / 0.822 / 0.660 / 0.522 | 0.533 / 0.833 / 0.633 / 0.520 | |
| W5 Milly | **0.639** | 0.623 | 0.621 |
| SNF Showdown | 0.593 | 0.666 | **0.697** |
| MNF Showdown | 0.574 | 0.591 | **0.609** |

- **Showdowns:** it ranked both contests' real entries better, and on
  SNF 2.5% of the real top-1% entries landed in the 5% we rated best
  (0.5% with the factor model). Our 150-entry picks did about the same
  (the two contests: 87-91 cashes, 5-6 top-1% finishes, ROI -42% to
  -43% for every version). It's the Showdown default (`--engine auto`).
- **Classic:** even on ranking (mean AUC 0.628 vs 0.634 over the five
  Millys); its picks cashed more (249 vs 232 of 750) but finished top 1%
  less (16 vs 25), so Classic keeps the factor model.

### Showdown entry files (`dk-pool` / `dk-run` on a single game)

The same `dk-pool` → `dk-run` steps work on a DraftKings **Showdown**
`DKEntries.csv` (CPT + 5 FLEX). The format is read from the file:

```bash
python -m dfs_engine.cli dk-pool --entries DKEntries.csv \
    --projections NFL_..._DK_SHOWDOWN_NO--ATL.csv --out sd_pool.csv
python -m dfs_engine.cli dk-run --pool sd_pool.csv --entries DKEntries.csv \
    --extra-lineups lineups_dk_nfl_showdown_....csv --eval-trials 20000 --holdout-trials 10000 \
    --upload-out sd_upload.csv --report-out sd_report.csv
```

- **Captain rows.** DraftKings lists every player twice, as a CPT row
  (its own ID, 1.5x salary) and a FLEX row. SaberSim's export does the
  same, each row with its own ownership. The pool keeps one row per
  player: projection, percentiles and `own` (FLEX ownership, ~500%) come
  from the FLEX row; `cpt_own` (~100%) and `cpt_dk_id` from the CPT row.
  The upload puts the captain's CPT ID first and is re-checked against
  DK's rules: CPT + 5 FLEX IDs, one player per slot, salary, both teams.
- **The field picks captains by captain ownership.** Before, captains
  were drawn by overall ownership. In the Week 5 SNF Showdown that made
  Bijan a 12% captain in our field, when SaberSim projected 26% and he
  was really 33%. That made our Bijan-captain lineup look far less
  popular than it was.
- **Duplication like a real Showdown.** 25% of the field comes from
  2,000 noisy optimizer solves (Classic: 6% from 500), and salary floors
  follow real Showdown spending. On Week 5 SNF (78k entries):

  | per 78k entries | real | old field | field now |
  |---|---|---|---|
  | copies of the most-played lineup | 415 | 22 | 600-680 |
  | copies of our entry / yours | 206 / 63 | 7 / 5 | ~160 / 61-70 |
  | duplication concentration (collision prob. x1e4) | 5.3 | 0.5 | 5.6-5.7 |
  | lineups at $49.5k+ / $48k+ | 57% / 93% | 9% / 31% | 59% / 91% |
  | captain share: Bijan / London / Huntley | 33 / 17 / 6% | 12 / 11 / 12% | 26 / 20 / 9% |

  (The field's captain shares follow SaberSim's captain projections,
  26 / 20 / 9%. Two seeds each.)

- **Rules.** `--showdown-rules light` (default): at most one DST,
  nothing else. The correlation rules of `soft` / `strict` (a QB needs
  one of his WR/TE, a WR/TE captain needs his QB, no two RBs from one
  team) threw out the lineups that won. In the Week 5 SNF Showdown only
  3% of the top-0.1% lineups met them all (39% of the top 1%, 58% of the
  field): 72% had a QB without one of his own WR/TE (the winner: CPT
  Huntley with Henry, London, Pitts and both kickers), and 45% had two
  RBs from one team. Classic stacking flags and `--max-game-stack-share`
  are ignored, with a note. Each pick is printed with its team split,
  how often it was the best candidate in a sim, and expected copies.
- **QB pairing.** Real Showdown fields pair the two starting QBs less
  often than independent picks would. Sunday, 15.1% of lineups played
  both, vs 18.0% if picked independently: an odds ratio of 0.61 (1 =
  independent), where our field had 0.67. The field is now calibrated to
  0.61 (`SHOWDOWN_QB_PAIR_ODDS`). It moves little. Sunday's simulated
  field went from 24% to 23% both-QB. Monday's stayed at 52%: QBs
  projected at 79% and 69% rostered force at least 48% to play both. The
  rest of Sunday's gap (23% vs 15%) is SaberSim's QB ownership: 47.6% /
  42.0% projected FLEX for Huntley / Penix, vs 39.2% / 36.2% real.
- **Which lineups the field copies (Week 6 TNF).** Our field matched the
  real duplication overall (92% of 84k entries in a duplicated lineup;
  the top two lineups 703 / 655 copies real, 668 / 602 ours) but not
  *which* lineups: copies of the lineups our engine would play came out
  at about half the real count (our TNF entry: 37 real, 4 expected; the
  chalk alternative: 292 real, 22-35 expected). One cause: real players
  stack the captain's team harder. Across SNF / MNF / TNF, 12.6-20.9% of
  lineups were 5-1 (captain's team 5, other 1) and 9.5-12.3% were 2-4;
  our fields had 11-13% and ~16%. The field now matches the three-contest
  mean (`SHOWDOWN_CPT_TEAM_MIX`) and drops the one-team lineups DK doesn't
  allow (0.5-0.8% of our draws had them). Fit on two contests and tested
  on the third, the copy counts of the lineups we'd play moved closer to
  real in all three (log-scale error 1.07 -> 0.96, rank correlation 0.51
  -> 0.56), and the Showdown backtests ranked real entries a little
  better (AUC 0.666 / 0.591 / 0.774 -> 0.670 / 0.605 / 0.786). Tried and
  dropped: more optimizer noise (better on our candidates, but it halves
  the copies of the most-played lineups), an optimizer slice tilted
  toward players owned above their value (worse), and a lower QB-pairing
  target (no gain: SNF and TNF paired QBs at our 0.6 odds ratio; their
  extra both-QB lineups came from SaberSim's QB captain ownership, 13-20%
  projected vs 9.5-14% real). The copy estimates for lineups we'd play
  are still about half the real count (Thursday's two: 6 and 33 expected
  now), so a duplicated pick's sim ROI is still optimistic; for a single
  Showdown entry take the top-1% pick (`auto`), which doesn't depend on
  copies.
- **Ceiling vs the field.** `--objective ceiling` / `blend` (see Contest
  sim) rank lineups wholly or partly by how often they are the best
  lineup in a sim, the way SaberSim does. In the replays below that
  didn't help, so `auto` stays `top1`.
- **`--extra-lineups`** takes SaberSim's Showdown lineups export (CPT +
  5 FLEX IDs) or any DK Showdown upload. Those lineups join the
  candidate pool and are played only where our contest sim prefers them.
- **Use 20,000+ evaluation sims for a single entry.** With 5,000, the
  top candidates' top-1% rates (~2.8%) sit within noise of each other,
  and different seeds pick different lineups.

Replay of Week 5 SNF with the same SaberSim inputs. Hindsight on one
game, so it says the field now behaves sensibly, not that the edge is
proven:

| lineup | points | finish of 78,431 | copies: sim expected / real |
|---|---|---|---|
| old engine (entered Sunday): CPT Bijan, London, Penix, Folk, ATL DST, J. Hill | 92.7 | 30,890 (top 39%) | ~7 (old field) / 204 |
| SaberSim's 2nd-best SaberScore (entered) | 103.2 | 6,522 (top 8%) | 61-70 / 63 |
| new engine, 3 of 4 runs (incl. 20k sims): CPT Flowers, Bijan, Huntley, Folk, Loop, ATL DST | 98.7 | 15,079 (top 19%) | 39-51 / 42 |
| new engine, 1 run (5k sims): CPT Huntley, Henry, Flowers, Andrews, Folk, ATL DST | 103.5 | 6,149 (top 8%) | ~4 / 3 |

The expected copies printed for each pick tracked the real field's.

Replays of both Showdowns after the Oct 13 changes. Same SaberSim
inputs, with SaberSim's lineups in the candidate pool and 20k sims.
Sunday's finishes are exact; Monday's are estimated from a simulated
field scored on the real results. Cash lines were about 97.7 Sunday and
138.3 Monday.

| | SNF single entry | SNF 20 lineups: median / cashing | MNF single entry | MNF 20 lineups: median / cashing |
|---|---|---|---|---|
| old engine | 98.7 (15,079th) | 97.1 / 6 | 106.3 (~80,700th) | 122.1 / 3 |
| + QB refit | 98.7 (15,079th) | 97.7 / 7 | 105.9 (~81,000th) | 125.6 / 4 |
| + light rules and QB pairing (default now) | 97.7 (17,226th) | 97.7 / 7 | 105.9 (~81,000th) | 125.6 / 4 |
| `blend`, ceiling weight 0.25 | 104.4 (5,331st) | 96.3 / 6 | 105.9 | 122.0 / 4 |
| `blend`, ceiling weight 0.5 | 96.3 (20,873rd) | 96.3 / 7 | 125.1 (~47,200th) | 118.9 / 2 |
| `ceiling` | 85.5 (52,375th) | 94.9 / 5 | 125.1 (~47,200th) | 122.1 / 2 |
| SaberSim | your entry: 103.2 (6,522nd) | | | your 20: 132.7 / 7 |

- The QB refit is the one change that helped both portfolios: +0.6 and
  +3.5 median points and one more cashing lineup in each game. The light
  rules and QB pairing barely moved these picks.
- Weighting ceiling helped some single entries but lowered every
  20-lineup portfolio's median. On Sunday's 77,561 real entries, ranking
  by "how often it's the best lineup in a sim" picked out the real top-1%
  finishers *worse* than chance (AUC 0.49, vs 0.575 for the top-1% chance
  against the field).
- SaberSim's 20 lineups still beat ours on Monday. Two games settle
  little; Monday's standings are the next test.

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

### Backtest on past contests (`backtest`)

The sites tune on hundreds of slates; a change that helps one game proves
little. `backtest` replays `dk-run` (the same code: `cli.model_slate` and
the per-contest pick) on every past contest you keep, with the inputs you
had at lock, and grades it against the real standings:

```bash
python -m dfs_engine.cli backtest --manifest slates.csv --out-dir outputs/backtest \
    --qb-stack 2 --bring-back 1 --flex-stacks --max-per-game 4 --max-per-team 3 \
    --max-vs-dst 0 --max-game-stack-share 0.5 --max-exposure 0.4   # your usual dk-run flags
```

`slates.csv` has one past contest per row: `label, pool, standings,
entry_fee, prize_pool, top_prizes` (known prizes for 1st, 2nd, ...,
separated by `;`), and optionally `extra_lineups` (another tool's lineups,
e.g. SaberSim's export, graded head to head and offered as candidates),
`actuals` (a name,actual CSV for a contest without standings) and
`contest_size`. Paths are relative to the manifest. Per contest it reports:

1. **Ranking.** Our sims rank the contest's *real* entries (40k sampled):
   AUC for a real top-1% finish, and how many real top-1% entries land in
   the 1% / 5% we rate best. This is the most data-rich test: tens of
   thousands of real lineups per slate. It also scores our top-1% chances
   as forecasts: *log skill* and *Brier skill* (0 = no better than giving
   every entry 1%, 1 = perfect; log skill weighs putting probability on the
   lineups that really won) and the *calibration slope* (1 = right, below
   1 = our chances are spread too far apart).
2. **Field.** Our simulated field next to the real one: projection,
   ownership, salary, stacking, duplication (chance two entries are the
   same lineup), and *field strength*: how often real entries beat our
   field's top-1% line (1% if our field is as strong as the real one).
3. **Picks** for 1 / 20 / 150 entries, scored on the real results:
   finishes (ties split), cashes, top-1% finishes, estimated winnings from
   a DK-style table anchored on the known top prizes, sim ROI vs real. Two
   steadier versions for comparing settings: winnings with every prize
   above 11th place cut to 11th's (one lucky 1st can't swing them), and the
   picks' median finish (% of the field).
4. **Head to head:** the extra lineups' real results, and where our sim
   ranked them.

`--keep-models DIR` also saves each slate's fitted model (field, sims,
candidates) for trying other settings on it.

**Comparing two settings** (`backtest-compare`). Run the backtest twice,
the same seed and one change apart, then:

```bash
python -m dfs_engine.cli backtest-compare --a outputs/bt_base --b outputs/bt_variant --out compare.csv
```

For each score above it prints B minus A on every slate both runs share,
the mean, a 90% interval from resampling whole slates (a contest is the
independent unit, not a lineup: every lineup in it shares the same game
results), how many slates went each way, and a verdict that stays
"unclear" unless the interval excludes zero. Comma-separate folders to
combine runs (e.g. one per slate when a setting is fit leave-one-out).
A correction learned from real results has to be fit on the *other*
slates before it's tested on one, or the test sees the answers.

Our seven contests (the 2026 Week 1-5 Millionaire Makers, the Week 5 SNF
and MNF Showdowns), with the factor model throughout (what every test
below is compared with; Showdowns now default to the stat engine, see
that section):

| contest | ranking AUC | real top-1% in our best 5% | field strength (real is 1%) | 150 entries: cashes / top-1% / est. ROI |
|---|---|---|---|---|
| W1 Milly (832k) | 0.526 | 0.9% | 0.9% | 36 / 3 / -23% |
| W2 Milly (173k) | 0.822 | 6.7% | 1.0% | 79 / 13 / +58% |
| W3 Milly | 0.660 | 1.6% | 0.9% | 35 / 2 / -53% |
| W4 Milly | 0.522 | 0.2% | 1.1% | 41 / 3 / -45% |
| W5 Milly | 0.639 | 1.0% | 1.1% | 41 / 4 / -45% |
| SNF Showdown | 0.593 | 0.5% | 1.2% | 56 / 3 / -26% |
| MNF Showdown | 0.574 | 0.0% | 1.1% | 33 / 2 / -54% |

- **Our picks beat the average entry but still lose money.** Over the
  seven contests our 150-lineup portfolios cashed 31% of the time (22% is
  average) and finished top 1% at 2.9% (1% is average), and still lost 28%
  of fees. A Milly pays about 36% of its pool to 1st; our best finish in
  1,050 entries was 57th. The sim's ROI (+13% to +550%) averages over rare
  1st places, which a handful of slates seldom includes.
- **The field isn't the problem.** Real entries beat our field's top-1%
  line 0.9-1.2% of the time, and scored on the real results our field
  reaches the real field's top scores.
- **Contest chalkiness.** The power on projected ownership that best fits
  each contest's real ownership was 0.85-1.15 for the Millys and 0.95-1.0
  for the Showdowns: these big contests played like their projections.
  For contests that don't, a `field_sharpen` column in the contests CSV
  gives them their own field (above 1 = chalkier); `backtest` prints the
  fitted power for each contest you add.
- **Week 6 TNF Showdown** (84k entries; stat engine, added after the
  table): ranking AUC 0.774 (projection alone 0.740), 7.1% of the 5% we
  rated best finished top 1%. 150 entries: 39 cashes, 4 top-1%, best
  finish 14th, est. ROI +47%; SaberSim's 500 lineups: 22% cashed, best
  finish 38th, est. -41%. The single entry we played was a judgment call
  away from the engine's top-1% pick, chasing a less-duplicated lineup;
  it finished 75,190th (the engine's pick: 40,406th; neither cashed).
- **Week 7 Milly** (159k entries; added after the table): a chalk-bust
  week. Ranking AUC 0.575 (projection alone 0.530), and only 0.3% of the
  5% we rated best finished top 1%. 150 entries: 26 cashes, no top-1%,
  est. -65%; SaberSim's 5,034 lineups: 12% cashed, 0.4% top-1%, est.
  -76%. Even the 150-entry players had a top-1% rate of 0.64% vs 1.11% for
  casual players. Our real entries (191 lineups across four contests,
  scored on the Milly's points): 18% above its cash line, 1% top-1%.
- **Week 7 SNF Showdown** (78k entries): ranking AUC 0.749, but
  projection alone did better (0.839); 24.8% of the 1% we rated best
  finished top 1%, many of them copies of the 2nd-best score. The
  engine's top two candidates were a coin flip (top-1% 2.80% vs 2.70%):
  #1 scored 77.4 (48,122nd; 45 real copies vs ~4 expected), #2 scored
  117.06, the second-best score of the night (313 real copies; 196
  expected). The entry actually played (built by hand, rated 0.66% by
  our sim) finished 3,273rd and cashed. 150 entries: 28 cashes, 2
  top-1%, est. -33%; SaberSim's 5,000 lineups: 14% cashed, 0.2% top-1%.
- **Week 7 MNF Showdown** (87k entries): ranking AUC 0.593 (projection
  alone 0.517), and our field matched the real one closely (duplication
  7.0 vs 6.7 x1e-4, mean salary $49.2k both). The single-entry pick (CPT
  McCaffrey, full salary) scored 95.5 (39,817th; 223 real copies vs ~141
  expected); the under-$48,500 pick also sent scored 111.2 (4,392nd,
  cashed); the entry played (built by hand) finished 47,275th. The winner
  (CPT Daniels, 26 copies) was among our candidates, rated 684th of 2,297.
  150 entries: 35 cashes, 2 top-1%, best finish 360th, est. -51%;
  SaberSim's 348 lineups: 13% cashed, 4 top-1%, est. -70%. Max-entry
  accounts finished top 1% at 0.85% vs 1.02% for casual players.
- **Week 8 TNF Showdown** (73.5k entries; a $1.25M pool assumed from the
  field size): ranking AUC 0.630 (projection alone 0.702), and none of
  the 5% we rated best finished top 1%. Both lineups sent scored about
  100: the engine's pick (CPT Swift, five Bears) 49,844th with 83 real
  copies vs ~13 expected, the Patriots-win build 48,273rd with 40 vs ~4.
  The entry played (built by hand) finished 73,090th. The winner (CPT
  Burden with both QBs, 46 copies) was among our candidates, rated 94th
  of 2,000. 150 entries: 50 cashes, 4 top-1%, best finish 47th, est.
  -13%. Max-entry accounts finished top 1% at 1.33% vs 0.73% for casual
  players.
- **Week 8 Milly** (147k entries; $2.5M): a contrarian week. The winner
  (225.4) and the next four all had the Bucs DST (27.0 at 4.6% owned) and
  Tony Pollard (39.5 at 4.4%); the chalk busted (Golden 9.9 at 39%).
  Max-entry accounts finished top 1% at 3.9x the casual rate (4.6x in the
  Pocket Cup qualifier). Ranking AUC 0.536, and 0.7% of the 5% we rated
  best finished top 1%. 150 entries: 49 cashes, 6 top-1%, est. -35%. The
  entry played (the best-rated lineup of a 12,421-lineup pool) scored
  94.3 (123,621st); the hand-picked chalky lineup with two contrarian
  Jets scored 135.7 (would have cashed). Bigger candidate pool (10,000
  extra lineups, rated in the same sims) vs the usual 2,422, both 191
  lineups scored on the Milly: 62 vs 52 above the cash line, but 7 vs 9
  top-1%, 0 vs 2 top-0.1%, best 180.2 vs 197.4. More cashes, a weaker top
  end: one week, so not a verdict.
- **Week 8 SNF Showdown** (KC @ SEA, $15, 78,431 entries): we entered the
  balanced alternative (CPT Kelce; 3-3, both QBs, no DST) on my call over
  the engine's pick, citing its expected copies and the 5-1/DST blind spot:
  68.78 points, 49,485th, $0. The engine's pick (CPT Smith-Njigba, Darnold,
  Worthy, Wilson, Myers, Seahawks DST: a Seattle 5-1 with its DST) scored
  110.03, 311th (top 0.4%), about $30 per entry: 247 real copies (we
  expected 255). Our 10th-rated lineup won outright (114.93, tied 149 ways,
  about $3,240 each). Seattle 5-1 builds with the DST were 4.5% of entries
  and 19% of them finished top 1%; the Chiefs busted (Walker, 26.5%
  captained, 9.7; Mahomes 7.0; Worthy 4.1). Backtest (seed 7, usual flags):
  1 entry 110.0 (311th); 20 entries about $3,361 on $300 (the winner among
  them); 150 entries $4,531 on $2,250, 44 cashes; ranking AUC 0.866, and 43%
  of the real lineups it rated in its top 1% finished top 1%. The Showdown
  build correction, fit on the other six Showdowns, would have picked the
  lineup we entered and cut 150-entry cashes 44 -> 31, top-1% 3 -> 1. With
  seven Showdowns the sim's overrating of DST lineups shrinks to x0.72
  (6 of 7 contests, t -1.1) and 5-1 builds to x0.94 (no consistent miss);
  RB captains still trail (x0.59, 6 of 7). Lesson: the engine's single
  pick stands unless a correction has passed the backtest. The refit
  Showdown sim on SNF8: same single pick, AUC 0.876 vs 0.866, 150 entries
  39 vs 44 cashes, 3 vs 3 top-1%.
- **Week 8 MNF Showdown** (DAL @ PHI, $20, 73,529 entries; Hurts out): we
  entered the engine's long-sim pick (20,000 + 10,000 sims, seed 11), CPT
  Javonte Williams, Lamb, Prescott, Aubrey, Cowboys DST, Darius Cooper (a
  Dallas 5-1): 105.13, 17,799th (top 24%), about a point under the
  estimated cash line, $0; 194 real copies (we expected 96). The winners
  (126.73, two entries) had the same Dallas 5-1 shape with Makai Lemon at
  captain (4.8% captained) and Flournoy, where we had the DST (4.0) and
  Cooper (2.7, 36.5% owned). Our 20-lineup set for this contest (built, not
  entered): 6 cashes, best 4,601st, about $240 on $400. 20-lineup sets are
  now 0 for 5 in the $20 Monday/Thursday Showdowns (-38%). Backtest (seed
  7, usual flags): 1 entry cashed (8,877th; a different pick, CPT Williams
  with Prescott, Pickens, Goedert, Cowboys, Lemon); 20 entries $320 on $400
  (8 cashes); 150 entries $1,240 on $3,000 (31 cashes, best 1,458th).
  SaberSim's ownership was nearly exact (corr 0.99); projections corr 0.88
  (RB -3.4, TE +2.5); 88% of players inside the sim's 10-90% band by the
  old measure (re-measured below). Max-entry accounts finished top 1% at
  0.59% vs casuals' 1.43%.
- **Week 9 TNF Showdown** (CAR @ GB, $20, 73,529 entries; plus our first
  small-field test, the $0.25 Quarter Jukebox: 9,512 entries from 2,092
  accounts, none with 150 entries). We entered a less-duplicated
  alternative to the engine's pick (CPT Golden for CPT Jacobs; sim copies
  15 vs 55): 102.87, 23,768th, $0. The engine's pick would have scored
  103.22 (22,920th). Real copies were 58 and 237, about 4x what the sim
  expected. Winners (125.47, 33 entries): CPT Hubbard (12.5% captained),
  Love, Jacobs, Golden, Kraft, Mitchell Evans ($1,600, 5.3% owned).
  Jukebox, our 12 long-sim lineups: 5 cashes, best 407th, about $2.56 on
  $3. Backtest (seed 7, usual flags): $20 contest 1 entry $0, 20 entries
  $200 on $400, 150 entries $1,280 on $3,000 (2 top-1%); Jukebox 20
  entries $2.08 on $5, 150 entries $20 on $37.50 (35 cashes, best 42nd).
  20-lineup sets are now 0 for 6 in the $20 Monday/Thursday Showdowns.
  Projections corr 0.89, ownership 0.97; 93% of players inside the sim's
  10-90% band by the old measure (re-measured below).
- **Sim width, re-measured** (all 17 contests; each slate's cached sims vs
  real points). The weekly check in `review` counts a player as inside his
  10-90% range whenever he scores 0 and the sim gave 0 that often, so
  backups inflate it. With ties spread over their probability (randomized
  PIT), Classic is on target: 7 Millys, 1,498 players, 79% inside the
  10-90% range, 10% below the 10th percentile, 11% above the 90th. The
  Showdown sim misses at the top only. 9 games, 142 player-games with
  under a 10% chance of 0: 92% inside, 15.5% above the 85th percentile
  (15% expected), but 2.8% above the 90th (10%) and none above the 95th
  (7 expected). 8 of 9 games had fewer above the 90th than expected, and
  none had a player above the 95th, including the 71-point BUF @ LA game.
  Real scores bunch between the sim's 75th and 90th percentiles (27.5% vs
  15%). The top end is SaberSim's own p95/p99, which the sim maps onto.
  Same direction as the game-script study's 13-31% excess build upside.
  A Showdown top-end trim is untested; `review` still prints the old
  measure.
- **"Cracking the DFS Code" (Kevin "Geek" Alan, 2026), tested on our
  data.** His study: 376 Classic Sunday main-slate GPPs with 10,000+ entries
  (2022-25), the top 100 lineups of each (34,856), full-field actual
  ownership and closing lines. With no full-field lineups, his build stats
  describe winners without a field baseline; his leverage figures (top-100
  usage / field ownership) are the fair comparison. We checked each
  Appendix A rule on our 7 Millys' full fields (top-1% and top-0.1% rates vs
  the field, also within projection fifths) and by re-picking the engine's
  saved candidates with the rule enforced (seed 7, usual flags), graded on
  the real standings. Base, 150 entries a contest: 29 top-1%, 6 top-0.1%,
  329 cashes, capped ROI +29%. None passed:
  - *Ownership band 110-135% (Milly):* the field's edge inside it is mostly
    projection (1.06x at equal projection; under 90% total ownership 0.73x).
    Enforced: 31 / 10 / 282 cashes (945 lineups: Week 3 had 45 qualifying
    candidates); 20 entries 4 vs 6 top-1%. Cap only (under 135%): 37 / 9 /
    303, but its 20-lineup set lost Week 2's $100k. With DraftKings'
    %Drafted, 2026 top-100 lineups sum to a median ~125% (weekly 93-155%),
    inside his 116-132% for 2022-25.
  - *A sub-10% RB/WR/TE:* the engine already has one in 97% of lineups.
  - *Stacking:* naked QBs 0.61x; QB+2 1.33x vs QB+1 0.86x (the same at
    equal projection); QB+3 2.5x on 2% of entries. Our top-1% finishers
    were 45% QB+1, 41% QB+2, 9% naked, 5% QB+3, close to his 49/38/10.
    Forcing QB+2: 30 / 5 / 293.
  - *Second cluster* (an RB plus the other side's WR, outside the QB's
    game): 20 / 4 / 265, and lost the $100k.
  - *RB floor:* lineups with an RB under $5,000 finished top 1% at 0.3-0.5x
    (the same at equal projection), but the engine already almost never
    plays RBs under $4,500. Barring RBs under $5,000 (full re-run): 33
    top-1%, 315 cashes, capped ROI +15%; unclear.
  - *WR punt cap:* 27 / 6 / 317. *TEs under $5K:* 24 / 4 / 324. *Double TE
    only with two sub-$5K TEs:* 26 / 5 / 347 (more cashes in 6 of 7, ROI
    without Week 2 -42% vs -46%, fewer top-1%). *No RB from a team implied
    under 20:* 24 / 7 / 301.
  - *Game selection:* stacks from the 2nd-3rd game totals 1.54x vs the
    highest total 0.59x (below 1 in 5 of 7 weeks). Barring QBs implied for
    26+: 27 / 0 / 310, and lost Week 2's $100k (a 26+ stack).
  - *2-3 QBs in 20 lineups:* the engine uses 8-12; its top 3 only: 4 vs 6
    top-1%, 56 vs 50 cashes.
  - *Projections:* no SaberSim bias for his groups within noise (home TEs
    +1.4 vs road -1.4 points in Weeks 7-8; cheap QBs -0.4; favored vs
    underdog DSTs +0.4 vs +0.5).
  - *DSTs (his most robust finding: home favorites 1.18, positive every
    season; underdogs 0.86; sub-5% "darts" 0.95, 10-20% owned 1.15).* Ours,
    home/away from the nflverse 2026 schedule: home favorites 0.91 (above 1
    in 3 of 7 weeks), road favorites 1.13, underdogs 1.00. The engine plays
    home favorites at the field's rate (35%) but dart DSTs far more often
    (55% vs 33%).
  - *QBs by role:* home underdogs 1.24 (mostly Week 1, 6.0), road
    underdogs 1.23, favorites 0.82-0.87. 2026 top-100 vs his 2025: QB under
    $5K 21% (22%), $7K+ 2% (2.7%), underdog QB 48% (41%), double TE 41%
    (26%), WR in the FLEX 26% (32%), bring-back 57% (55%). The engine: 11%,
    5%, 35%, 49%, 26%, 62%.
  - *Tilts* (the saved models re-ranked with tilted sims, the field
    unchanged): home-favorite DSTs x1.15 (x1.21 vs offenses implied under
    20), underdog DSTs x0.90: home-favorite share 35% -> 49%, darts 55% ->
    53%; 150 entries 24 / 2 / 311, capped ROI +9%. Home-underdog QBs x1.10:
    26 / 7 / 314. QBs under $5K x1.10: 24 / 3 / 312 (worse on every
    measure). All three: 21 / 3 / 289. None adopted. The DST and QB-role
    checks re-run as contests arrive. Scripts in the scratchpad (`book/`).
- **Reverse-engineering the regulars** (13 contests, 2,286,625 entries from
  222,286 accounts, every entry mapped to its lineup; 596 regulars with
  20+ entries in 5+ contests of a format: 484 Classic, 246 Showdown; our
  engine's 150 backtest picks scored as one more account). Scripts in the
  scratchpad (`rev/`, `own_cal/`, `tail/`); the full write-up is the
  "How the Regulars Build" report.
  - *Style repeats, results don't.* Odd vs even contests, rank correlation
    across regulars: salary left 0.89 / 0.87 (Classic / Showdown),
    concentration 0.90 / 0.81, ownership 0.84 / 0.66, projection
    0.79 / 0.62; average finish 0.23 / 0.29, capped ROI 0.16 / 0.12,
    top-1% rate 0.10 / 0.06.
  - *What they weigh* (choice model: each regular's lineups vs 40 random
    real lineups from the same contest). Classic: at equal projection the
    typical regular leans toward ownership (+0.32); our engine is at
    -1.34, more anti-crowd than all but 5 of 482, and chases projection
    harder than all but 7. Regulars' ownership weight correlates +0.38
    with their capped ROI and +0.50 with their points.
  - *Edge split.* An account's points edge over the field = picking
    higher-projected players + picking players who beat projections.
    Classic regulars +0.14 field sd = +0.09 + +0.06; the projection part
    repeats (0.73 between halves), the beat-projection part doesn't
    (0.00; Showdown -0.04). Accounts best at beating projections in the
    other contests led no better in the held-out one. No regular shows a
    lasting information edge. Our engine: +0.24 from projections, -0.06
    from beating them (4 of 7 contests negative); it leans into value
    plays harder than 99.8% of regulars, and value plays fell short of
    projections (-0.11 sd per sd of value, t -2.2; mostly our own
    projections in Weeks 1-4).
  - *Habits that pay* (within-contest slopes averaged over contests).
    Our sim's rating of an account's lineups is the strongest predictor of
    its real results: Showdown top-1% +0.38 points per sd (6 of 6, t 8.0),
    Classic top-5% +1.2 (7 of 7). Ownership: Classic capped ROI +8 per sd
    (6 of 7), Showdown top-1% +0.39 (6 of 6). Bad: no QB stack (Classic,
    0 of 7), salary left (Showdown capped ROI -9 per sd, 0 of 6), RB
    captains and DSTs (Showdown). Jointly, Classic ownership still pays
    after our sim's rating and projection (top-5% t 2.9); projection
    beyond that costs (t -2.0).
  - *Same menu, different pickers* (each rule picks 150 of the lineups
    really entered, pre-lock information only, 40% cap; a pick earns what
    that lineup earned per entry). Classic capped ROI: our engine's picks
    +21%, our sim's top-0.1% rating +16%, regulars' style rules -23% to
    -28%, real regulars -35%, field -48%; fade-ownership -59%, contrarian
    style -77%. Without Week 2 the order flips: style rules -29% to -34%,
    our sim's rating -39%, our engine -46%. Showdown: highest ownership -1%,
    our sim's rating -4%, style rules -9% to -13%, our engine -21%, real
    regulars and field -45%.
  - *Where the sim and real contests disagree* (every real lineup, real
    top-1% odds vs the sim's, per contest). Classic: crowd lean (ownership
    minus what projection implies) x1.33 per sd (6 of 7); by decile the
    real/sim ratio runs 0.74 -> 1.75; at equal sim projection those lineups
    scored +3.1 points per sd (6 of 7, t 3.4). QB with his own RB x0.74
    (7 of 7). Showdown: DST x0.53 (6 of 6), 5-1 x0.67, no QB x0.72, each
    $1k left x0.84 (six Showdowns; with Week 8 SNF: DST x0.72, 6 of 7;
    5-1 x0.94; RB captain x0.59, 6 of 7).
  - *Joint booms vs real games* (nflverse 2021-25, 1,076 team-games, both
    players in their top 10%, x independent): QB with own RB1 real 0.74,
    Classic sim 1.48, Showdown sim 2.00; RB1 with WR1 0.56 / 1.08 / 1.12;
    QB with WR1 2.97 / 2.80 / 3.19; QB with TE1 3.53 / 2.77 / 2.74. Classic
    sim defenses are nearly independent of the offense they face (QB vs
    opposing DST -0.01; real -0.41, Showdown sim -0.48): every team-level
    swing comes through the TD layer, which DSTs aren't in.
  - *Fixes.* Crowd-lean correction for Classic (each candidate's top-1%
    hits scaled by exp(c x lean), c fit on the other contests, plus about
    1,000 crowd-leaning candidates per slate): 150 entries top-1% 38 vs 29
    (better in 5 of 7), cashes 362 vs 329, capped ROI +52% vs +29%, mostly
    Week 2; 20 entries slightly worse. Promising, not adopted; re-test each
    week. In Showdown the same correction ranked real entries worse in 4
    of 6 (rejected). New sim knobs, all off by default (the old sim draw
    for draw): `dst_td_cost`, `run_pass_sd`, `run_pass_yards`,
    `run_pass_script` (factor model); `td_mix_sd`, `share_tilt`,
    `rush_eff_corr` (stat engine). Classic sim refits, fit to real games
    only, then backtested on all 7 Classic contests (seed 7, usual flags):
    the DST link alone (1.77 DST points per opposing TD) left the ranking
    unchanged and the picks slightly worse (150 entries: 23 vs 29 top-1%,
    capped ROI +15% vs +29%). The full refit (loss in real-game standard
    errors 6.3 -> 1.9; DST pairs match, QB~opp DST -0.41) ranks real lineups
    better (log and Brier skill better in 6 of 7, AUC +0.007 in 5 of 7), but
    its picks were no better: top-1% 30 vs 29, cashes 336 vs 329, capped ROI
    +20% vs +29% (Week 2; -44% vs -46% without it). Its best-rated 5% held
    no more real winners (1.78% vs 1.86% top 1%). DST link plus
    `run_pass_yards` 0.3: the biggest ranking gain (AUC +0.016, 5 of 7;
    2.02% of its best-rated 5% finished top 1%), worse picks (cashes 316 vs
    329, capped ROI lower in 6 of 7). None adopted; the knobs stay for
    testing on new contests. Real QB~RB1 co-booms are hard to reach: in the
    stat engine even fully separate rushing efficiency only takes them from
    1.84x to 1.77x (real 0.74x). Showdown refits on the same real-game
    targets, backtested on all 6 Showdowns: re-tuning the old knobs (loss
    3.69 -> 3.01) ranked worse (log skill worse in 5 of 6, Brier 6 of 6) and
    took 150-entry top-1% from 20 to 14. With `rush_eff_corr` 0.01,
    `share_tilt` 0.06, `td_mix_sd` 0.10 (loss 3.04; QB~RB1 co-booms 1.88x ->
    1.65x): AUC +0.013 (4 of 6, unclear), more real winners in its best-rated
    1% (6.6% vs 5.9%), but top-1% 15 vs 20; capped ROI +18% vs -19%, almost
    all from MNF7. Not adopted.
- **The regulars share the Showdown blind spot** (six Showdowns; accounts
  with 100+ entries vs single-entry players, each build's top-1% rate
  relative to its own group's average). Regulars play more 5-1 builds
  (21% of their lineups vs 13%) and more DSTs (28% vs 19%), and their 5-1
  lineups finish top 1% at 0.44x their average (below 1x in all six), DST
  lineups at 0.46x (5 of 6), RB captains at 0.65x; both-QB lineups at
  1.26x. SaberSim's own lineup exports lean the same way: 14-21% are 5-1
  and 19-42% carry a DST. The common sim tools seem to make one team's
  blowout (with its DST) too likely and shootouts too rare. Regulars'
  lineups are far less duplicated than casual players' (median 9 copies
  vs 32). Next: measure real within-game joint scoring on nflverse
  history and fix the game model there, so the six Showdowns stay a
  clean test.
- **Game-script refit of the Showdown stat sim** (nflverse 2021-25: real
  DST and K scores rebuilt from team defensive stats and kicking by
  distance). The pairwise links were close, but with the same marginals
  the sim gives every stacked Showdown build 13-31% more 99th-percentile
  upside than real games' joint ranks (538 complete games; 90% bootstrap
  ranges all above 1), DST and one-sided builds the most. A knob search
  with the old targets plus K / DST rank correlations and these build
  ceilings cut the excess from 1.20x to 1.14x, at the cost of a weaker
  opposing-QB link (0.12 vs 0.16-0.20 real). Six Showdowns, seed 7: ranking
  AUC +0.006 (better in 4 of 6, the one clear result), 150 lineups 237 vs
  227 cashes, 20 vs 20 top-1%, capped ROI the same. Not adopted; the sim's
  shared game shock ties moderate correlation to extreme co-booms, which
  real football doesn't, so the fix is likely structural.
- **Which contest types our edge holds up in** (all 13 contests; for each
  virtual contest the engine re-picks 20 lineups with its own objective and
  caps, and each lineup's real score is priced exactly against opponents
  drawn from that contest's real entries; stylized DK payouts, 15% rake,
  11% for the double-up). Median ROI per entry, contests in profit: as
  played (Milly / $1M+ Showdowns) -29%, 3/13; 10,000-entry GPP -9%, 6/13;
  1,000-entry +13%, 7/13; 100-entry +19%, 7/13; 1,000-entry against
  single-entry players only +52%, 7/13; against 100+-entry accounts only
  -10%, 4/13; 100-entry double-up +17%, 10/13 (all 6 Showdowns). The
  three contests never tuned on show the same order (100-entry and
  casual fields +21%, double-up +19%, the big contests -11% to -35%). The
  edge is over casual players; against the max-entry regulars it is about
  gone, and the contests we have been playing are full of them (17% of the
  Milly's entries, 74% of the Pocket Cup's). The double-up figure is
  flattering: real cash-game fields build for a floor and are sharper
  than the tournament entries used as opponents here.
- **We undercount copies of the lineups we like** (W6 TNF, W7 MNF, W8
  TNF; our ~2,000 candidates matched to real entries). Total copies were
  right (0.92-1.05x expected), but our 10 best-rated lineups drew
  2.0-6.1x the copies we expected, ranks 11-50 1.5-2.4x, and ranks past
  1,000 0.6-0.7x. By projection, lineups within 2 points of the best drew
  0.2-0.7x the expected copies and those 10+ points back about 1.5x:
  real fields spread their copies more evenly than ours. The engine
  favors lineups it thinks are rare, so it picks the ones it undercounts
  most, and its top-1% odds for them are too high. SaberSim's own
  lineups aren't the cause: none of our top 50 were in its pool. (Copies
  don't change the chance of a top-1% finish, since copies tie; they cut
  what a top finish pays, and the top-1% pick rule barely sees them.)
  Re-measured on all 8 Showdowns: our 10 best-rated lineups drew 2.3x the
  expected copies (above 1 in all 8), ranks 11-50 1.9x, 51-200 1.5x,
  201-1,000 1.1x, past 1,000 0.7x; within 2 points of the best projection
  0.6x, 10+ points back 1.5x; all candidates 0.97-1.15x. A noisier
  optimizer slice in the simulated field (projection noise 0.30 instead of
  0.18) fixes the projection pattern (0-2 points back 1.0x) but not the
  favorite-lineup one (still 2.4x): real players' sim-based tools seem to
  build the same high-ceiling lineups we do. Upper bound: pricing every
  candidate with its real copy count did not improve the picks. Top-1%
  objective: 19 vs 23 top-1% at 150 entries, unchanged at 1 and 20.
  Expected payout with real copies picked lineups with about 4 copies and
  won $9.1k vs $19.3k. Payout capped at 11th place: 27 vs 23 top-1% and
  345 vs 302 cashes, but 2 vs 6 top-0.1% and $16.8k vs $19.3k. So no copy
  model: being unique doesn't pay by itself, because lineups nobody plays
  are mostly worse. The noisier field with the usual top-1% objective:
  150 entries capped ROI -11% vs -25% (better in 7 of 8), top-1% 24 vs 23,
  cashes 312 vs 302 (and a TNF8 1st place); 20 entries capped ROI -6% vs
  +1%; 1 entry one fewer cash. Not made the default, since it gains
  nothing at the 1-20 entries we play; `--field-optimizer-noise 0.3`
  turns it on. Scripts in the scratchpad (`dupes2/`).
- **The sim overrates one-sided Showdown builds** (six Showdowns, ~17,000
  of our backtest candidates, the sim's top-1% chance vs real finishes).
  5-1 team splits finished top 1% at 0.32x the sim's rate (short in all
  six), lineups with a DST at 0.60x, lineups with no QB at 0.18x; 3-3
  splits beat it (1.30x), as did lineups with both QBs (1.22x). Real
  fields agree: lineups with a DST finished top 1% at 0.38x the field
  average (below it in all six), 5-1 builds at 0.63x. Both TNF lineups we
  sent were 5-1 with a DST. A correction by split, DST and QB count,
  fitted on the other five contests and applied to each in turn (top
  lineups by corrected chance, no exposure caps): 150 lineups 281 cashes
  vs 230, 27 top-1% vs 18, ROI with prizes capped at 11th -13% vs -27%;
  20 lineups 44 cashes vs 26, capped -11% vs -27%. On TNF8 alone its
  pick scored 134.0 (1,515th) and the winning lineup was its 3rd choice.
  It lost badly in the Week 6 TNF, the one game where 5-1 builds won (16
  cashes vs 50). The full test, through the engine's own picks and
  exposure caps (each candidate's simulated top-1% hits scaled by its
  leave-one-out factor), shrinks the gain. At half strength (the square
  root of the factor): 150 lineups 248 cashes vs 227 (better in 5 of 6
  contests, the only clear result), 23 top-1% vs 20, capped ROI -14% vs
  -19%; 20 lineups 40 cashes vs 33, capped ROI unchanged; the single
  pick no clearer (3 cashes vs 2, but worse in 5 of 6). It ranked the
  real entries slightly better (log skill and AUC up in 4-5 of 6). Full
  strength was mixed, with much worse rankings on the Week 6 TNF. The
  quick test overstated it: without caps its top lineups piled into one
  game script, which paid off on TNF8. Not adopted; re-test as Showdowns
  come in, and look for the cause in the game sim, which makes one
  team's blowout (with its DST) too likely.
- **What real winners looked like** (every entry of the ten contests we
  could map, with the projections and ownership we had at lock; W1-W4
  map only 65-81% of entries, the rest 97-99%):
  - *Showdown:* lineups at $48.5k or less finished top 1% at 0.3-0.6x
    the average rate and won less than the field in all four games. WR
    captains beat the field in all four; RB captains trailed in all four.
  - *Both formats:* the least-owned fifth of lineups finished top 1% at
    about half the average rate (Classic 0.5x, Showdown 0.3x), even next
    to lineups with the same projection. Ownership barely predicts a
    player's points beyond his projection (correlation 0.03), so the
    field's contrarian lineups are mostly weak ones.
  - *Classic:* QB + 2 receivers finished top 1% at 1.5x the average vs
    1.0x for QB + 1 (5 of 6 weeks); 150-entry accounts at 1.4x,
    single-entry players at 0.84x.
- **Our engine vs the experienced regulars** (the 160 accounts in 8+ of
  the 11 contests with 800+ entries; our 150-lineup backtest picks scored
  the same way, each contest weighted equally). Millys: our winnings with
  prizes capped at 11th rank 3rd of 156, top-1% rate 11th, top-0.1% rate
  7th, cash rate 33rd. Showdowns: top-1% rate 6th of 154, top-0.1% 4th,
  capped winnings 19th, cash rate 49th. That flatters us: the engine was
  tuned on these contests, and the Milly figure leans on W2. The
  regulars' own results barely repeat: split the contests into two
  halves and the rank correlation between halves is 0.00-0.16 for capped
  winnings, top-1% and top-0.1% rates, so which regulars look best over 11
  contests is mostly luck. Habits that went with better capped winnings in
  both halves: in Millys, higher projection, chalkier lineups, more
  different QBs across the 150, a TE at FLEX; in Showdowns, full salary,
  higher projection, chalkier and more-copied lineups, QB captains (and,
  for top-0.1% finishes, fewer TE captains). Ours match or beat the
  regulars' median on all of them except Milly ownership (our lineups sit
  at the 49th ownership percentile, theirs at the 63rd).
- **Tried, not adopted:**
  - *A ranking learned from past results* (our sim's top-1% chance plus
    projection, ownership and salary; trained on four Millys, tested on
    the fifth): AUC 0.634 -> 0.584, and fewer real winners in the top 5%.
    The weights flipped sign week to week.
  - *Down-weighting the most contrarian lineups.* Our sim does over-rate
    the least-owned fifth (0.73% predicted top-1% vs 0.44% real, in 5 of 7
    contests), but correcting for it was neutral: 150 entries, 339 vs 321
    cashes, 28 vs 30 top-1%, the same winnings.
  - *Contrarian candidates* (`--leftover-share 0.25 --own-penalty-share
    0.25`): 150 entries, 331 cashes but 19 top-1% finishes (vs 30), and
    $1.1k less won. Off by default.
  - *Ownership uncertainty* (`--field-own-sd 0.3`: the field built from
    four ownership draws): the Millys about even (mean AUC 0.640 vs
    0.634), both Showdowns worse (0.528 / 0.524 vs 0.593 / 0.574). 150
    entries: 334 cashes, 30 top-1% finishes (the same), $0.2k less won.
    Off by default.
  - *More sims* (`--eval-trials 50000 --holdout-trials 50000` instead of
    5,000 + 3,000, with the same field and candidates; Showdowns on the
    stat engine): mean ranking AUC 0.630 vs 0.632, no contest moving more
    than 0.03. 150 entries: 322 vs 323 cashes, 26 vs 31 top-1% finishes,
    $0.5k less won; the 1-entry pick was the same lineup in 5 of 7
    contests. It took 7-10 minutes per contest instead of 2.5-4, so 5,000
    stays the default. The sim's ROI for a single Milly lineup stays
    unreliable either way: the same Week 4 lineup read -16% at 5,000 and
    +366% at 50,000, from a few extra simulated 1st places.
  - *Looser or stricter Classic stacking* (six Millys, seed 7, the usual
    flags otherwise): `--max-per-game 5`, `--max-per-team 4
    --max-per-game 5`, and strict QB + 2 with a bring-back (no
    `--flex-stacks`), prompted by the five-player game stacks that won
    the two shootout weeks. 150 entries: 280 cashes / 23 top-1% for the
    current rules vs 266 / 23, 272 / 23, and 197 / 21 for strict, which
    fills only 112 lineups a slate under the 40% exposure cap. Totals
    hinge on one W2 2nd place ($100k) that only the current rules and
    `--max-per-game 5` found. The current rules stay.
  - *Showdown QB ownership correction* (`--own-scale QB=F --cpt-own-scale
    QB=F`). SaberSim projected the two QBs above their real ownership in
    all four Showdowns (FLEX 0.84-0.98 of projected, captain 0.71-0.94).
    Each game's factors were fit on the other three, shrunk toward no
    correction (FLEX 0.91-0.95, captain 0.76-0.84), and tested on the
    held-out game with `backtest-compare`. Forecast scores rose in 3 of 4
    (log skill +0.003, AUC +0.012), but fewer real top-1% entries landed in
    the 5% we rated best (3 of 4), and the picks were mixed (150 entries:
    +2.5 cashes but -1 top-1% finish a slate). Unclear on four games: the
    flags stay, off by default, for a re-test as Showdowns come in.
  - *Wider WR outcomes* (`--pos-spread WR=K`). Real WR scores spread
    wider than our sims: above the sim's 85th percentile in 15-25% of
    cases in all ten contests, and below its median about 55% of the time
    in all six Millys (other positions mixed; whole games boomed together
    about as often as simulated, 1.07x). Each contest's K was fit on the
    other nine (1.08-1.15) and tested on it. The Millys' forecast scores
    improved slightly (log skill +0.001, better in 4 of 6), but the
    150-entry sets cashed 4.5 fewer times a slate (fewer in 5 of 6) and
    finished lower; the Showdowns were slightly worse on the forecast
    scores. Not adopted: a sim that matches players' ranges better can
    still pick worse lineups.
  - *Tail pricing* (`--tail-pricing`). The field is a 20k sample of a
    160k contest, so a lineup that beats the whole sample is paid an
    average of the top ranks however far ahead it is. On synthetic fields
    an exponential tail on each sim's top 0.5% of sample scores cut that
    error severalfold. On real contest models priced against an exact
    full-size field, it didn't: total expected winnings 0.67x / 0.62x of
    exact (W5 Milly / SNF Showdown) vs 0.94x / 0.90x for the flat average,
    and the chance of a 1st place 0.6% / 2.4% vs 1.6% / 6.6% exact (flat
    average 1.4% / 5.0%). Real lineup scores thin out faster at the top
    than the curve assumes. It ranked Milly lineups slightly better (0.975
    vs 0.949 rank agreement) and Showdown lineups worse (0.851 vs 0.955).
    A full-size field sample is the accurate option (a second full-size
    draw matched the first at 1.00-1.01x, rank agreement 0.98), but takes
    2-3x as long and changes nothing under the default top-1% objective,
    whose top-1% line the 20k sample already gets right (1.16% vs 1.16%).
    Off by default.
  - *Recalibrating the sim's top-1% chances.* Fit on the other nine
    contests' real entries (360k) and tested on the held-out one, with each
    lineup's simulated top-1% hits scaled to its calibrated chance before
    picking. On the sim's chance alone, the fitted slope was 0.72-1.09:
    the sim is slightly over-confident. With lineup ownership added, its
    coefficient was positive in all ten fits: more-owned lineups finish top
    1% more often than the sim says (x1.12-1.28 the odds per standard
    deviation of ownership), the clearest sign yet that the sim over-rates
    contrarian lineups. Neither version forecast the held-out contests
    better (log skill -0.004 / -0.006; better in 5 / 4 of 10). With
    ownership, 150-entry sets cashed 3.5 more times a contest (more in 6
    of 10) but finished top 1% no more often, and 20-entry sets finished
    top 1% less (0.5 fewer a contest, fewer in 4, more in none). Not
    adopted: the tilt buys cashes, not the top finishes that pay. On the
    Millys alone, to close the ownership gap with the regulars: 150-entry
    sets 301 vs 280 cashes, the same 23 top-1% finishes, 3 vs 5 top-0.1%
    finishes, capped winnings +88% vs +41%; 20-entry sets 2 vs 5 top-1%
    finishes.
  - *Higher exposure caps* (`--max-exposure` 0.6 / 0.8 / 1.0 instead of
    0.4, re-picked from the same saved models), prompted by a 150-max
    account in all ten contests whose most-used player was in 65% of its
    Milly lineups (ours 37%). In the Millys the cap rarely binds: the
    top-1% coverage objective spreads exposure on its own, and the picks
    were identical in 5 of 6. In the Showdowns, where it binds, looser caps
    were worse in 3 of 4. 150 entries x 10 contests: 416 cashes and 35
    top-1% finishes vs 422 and 37 at 0.4. The cap stays at 0.4.

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

### Monday review (`review`)

```bash
python -m dfs_engine.cli review --standings contest-standings-*.zip --pool week5_pool.csv \
    --entries DKEntries.csv --contests contests.csv --week 5
```

For every contest it prints a **scorecard**:

- field size and the share of entries from max-entry accounts;
- the **pro edge**: their top-1% rate divided by casual players' rate;
- the winning, top-1%, top-20% and median scores;
- your entries' finishes and approximate winnings.

It then **grades the inputs** (projections, provided ownership,
`own_model`) against actual points and %Drafted. Last, it checks **the
sim against reality**: how often real scores landed inside the sim's
10–90% range (should be 80%) and above its 85th percentile (should be
15%), overall and by position. Each contest adds a row to
`outputs/results_log.csv`, which is git-ignored because it holds your
results. Over a season that log gives real ROI by contest type.

Weeks 1–3 with the sim's default ceilings; Week 4 with the pool we
actually played:

| | Week 1 | Week 2 | Week 3 | Week 4 | target |
|---|---|---|---|---|---|
| inside sim 10–90% | 78% | 86% | 79% | 76% | 80% |
| above sim 85th pct | 24% | 11% | 19% | 16% | 15% |
| pro edge in the Milly (top-1% rate, max-entry vs casual) | x1.9 | x2.8 | x2.2 | x1.9 | |

The ranges are about right on average. Tight ends beat their 85th
percentile 23% of the time in Weeks 1–3 but only 8% in Week 4, so the
TE ceilings stay as they are.

### Our own projections (`proj-fit`, `dk-pool --proj-blend`)

A thorough review of 2026 Weeks 1–4 (664 projected player-weeks scored
against actual DK points) and five seasons of nflverse games led to a
projection model built only from free data. What the review found:

- **Vegas implied team totals are unbiased and well calibrated** (2,848
  team-games, 2021–26). Actual scoring ran +0.25 pts over implied
  overall, and within 1 pt of implied in every bucket from 16 to 30.
  A team's score still swings ±9 pts around its implied total.
- **Each implied point is worth ~2.7 DK points to the offense.** Team
  QB+RB+WR+TE output ≈ 26.3 + 2.73 × implied total. The split barely
  moves with the spread: QB ~20%, RB ~26%, WR ~39%, TE ~14–15%.
  Favorites lean a little more QB/RB.
- **The public projections are unbiased and well spread overall.** They
  averaged 0.3 pts high, with a slope of actual on projection of 0.97.
  Their misses don't correlate with Vegas totals, spread, salary, last
  week's points, season averages or ownership (all |r| < 0.05), and
  re-fitting them per position made held-out error worse. Simple
  corrections won't beat them.
- **Their weak spots:**
  - QBs: correlation 0.25. One projected point is worth only 0.73
    actual points, so they spread QBs too far apart.
  - DSTs: correlation 0.11, with 0.39 actual pts per projected pt.
  - The $5.5–7k tier: correlation 0.25.
- **In Week 4 the sources were close:** DFF 5.71 average error, GoingFor2
  5.75, props 5.93 (+0.85 bias), DK season averages 6.27. Averaging the
  three didn't beat any single one.

**The model (`dfs_engine/projections.py`).** For QB/RB/WR/TE it runs a
ridge regression per position. All features come from games before the
week being projected:

- season-to-date averages, shrunk toward last season;
- last-3-game averages of DK points, targets and carries;
- QB pass attempts;
- the player's share of team DK points × the Vegas-implied team output;
- implied total and spread.

DSTs use the opponent's implied total, own spread, season sack rate and
home field. Opponent-defense adjustments added nothing, so they're left
out.

| test | season average alone | our model | public projections |
|---|---|---|---|
| 2024–25 holdout, same-position correlation QB / RB / WR / TE | 0.46 / 0.65 / 0.59 / 0.56 | 0.50 / 0.67 / 0.60 / 0.58 | — |
| 2026 Weeks 1–4, 556 skill player-weeks: avg error / same-position corr | | 6.04 / 0.412 | 5.97 / 0.438 |
| same, **25% ours + 75% public** | | **5.93 / 0.446** (better in 3 weeks, tied 1) | |
| 2026 DSTs (100): avg error / corr | | 3.85 / **0.24** | 3.91 / 0.11 |

With no news and no paid data, ours comes within 0.07 pts of average
error of the public projections, and it beat them in Week 4. Regressing
actual points on both gives actual ≈ 0.75 × public + 0.26 × ours, so
ours adds information the public numbers lack.
`dk-pool --week N --proj-blend 0.25 --dst-blend 0.5` blends it in. The
provided projection is kept as `proj_public` and ours as `proj_model`,
and `review` grades both every week. Refit after each season with
`proj-fit`, which downloads the nflverse files and prints a holdout
check on the last season.

**Where to get better next:** QB projections are weak everywhere (0.25
for public and ours alike). The likely sources of improvement:

- sportsbook props (pass yards and TD lines) once SportsGameOdds is
  reachable;
- snap shares and red-zone usage from nflverse play-by-play;
- late news: inactives, and backups stepping into a starter's role.

### How a real field built its lineups (`field-structure`)

```bash
python -m dfs_engine.cli field-structure --standings contest-standings-<id>.zip --week 4
```

For every entrant group (all, casual players with 1–3 entries,
150-entry max-entry accounts) it prints:

- QB stack sizes and bring-backs;
- the most players from one game;
- QB + RB and RB + DST pairings;
- players facing the lineup's own DST.

Teams come from nflverse rosters. Week 1–3 Millys show how the pros
build differently from the casual crowd:

| | casual | max-entry |
|---|---|---|
| QB + 2 or more of his pass catchers | 15–21% | 35–43% |
| at least one bring-back | 33–35% | 44–49% |
| 4+ players from one game | 18–19% | 32–34% |
| a player facing own DST | 7–11% | 3–4% |

Run it on each week's export and add the week's "all" row to
`data/field_structure.json`. `contest.field_targets` turns the file into the
recency-weighted stack mix and structure rates. Copy them into
`DEFAULT_STACK_MIX` / `DEFAULT_FIELD_STRUCTURE`; a test checks they match.

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

Leave-one-week-out on the 2026 Week 1–4 Millionaire Makers. Each week
is scored by a model fit on the other three. Week 4 was the first truly
unseen week: the shipped model had been fit on Weeks 1–3 before its
results existed.

| | Week 1 | Week 2 | Week 3 | Week 4 | avg |
|---|---|---|---|---|---|
| **Ours** — correlation / avg error | 0.77 / 3.0 pts | 0.75 / 3.1 | 0.63 / 3.5 | 0.79 / 1.9 | 0.73 / 2.9 |
| GoingFor2 (published) | 0.90 / 2.0 | 0.85 / 2.5 | 0.85 / 2.4 | 0.92 / 1.3 | 0.88 / 2.05 |
| 10% ours + 90% GoingFor2 | 0.90 / 2.0 | 0.86 / 2.4 | 0.85 / 2.4 | 0.92 / 1.3 | 0.883 / 2.03 |
| Old log-linear fallback (Weeks 1–3) | | | | | 0.65 / 3.5 |

GoingFor2 is still clearly better: it reacts to news, which a model
built only on box scores can't. The 10% mix is a hair better on
average but lost in Week 3, so it stays off until it holds up every
week. Keep GoingFor2's numbers as `own` and use ours:

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
- Optional: `p25 p50 p75 p85 p95 p99` (the player's own outcome
  percentiles; sims are mapped onto them), `min_exp` / `max_exp`
  (per-player exposure limits, fractions or percentages).

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
  dk.py           # DraftKings entry file (Classic + Showdown): parse, pool + projections merge, slots, upload
  history.py      # contest-standings exports -> field profile (score spread, stacks, ownership)
  ownership.py    # ownership sanity checks + fallback estimator from real-field curves
  portfolio.py    # N-lineup portfolio builder (exposure caps, uniqueness)
  diagnostics.py  # Optimal% leverage diagnostic
  sportsgameodds.py # SportsGameOdds fetch, game lines, prop -> DK baseline
  correlations.py # real player-pair correlations (nflverse) + fitting the sim to them
  ownership_model.py # our ownership projections: nflverse inputs, softmax fit, leave-one-week-out
  lateswap.py     # late swap: live standings -> actual points/ownership/field -> re-optimized open slots
  field_model.py  # real-field lineup structure (stacks, bring-backs, DST conflicts) from standings
  review.py       # Monday review: contest scorecards, input grades, sim calibration, results log
  projections.py  # our projections from nflverse usage + Vegas lines (skill ridge + DST model)
  cli.py          # command-line entry points
tests/
  test_engine.py  # pytest-style tests
  test_sportsgameodds.py
  test_sim_tools.py  # ceiling calibration, provided percentiles, outcomes report, stacking rules
  test_contest.py    # payouts, field sampler, contest sim, portfolio selection, exposure limits
  test_dk.py         # entry-file parsing, matching, SaberSim exports, lineup files, upload, dk-run end to end
  test_history.py    # standings parsing, field profile, ownership estimator, spread fitting
  test_correlations.py # DK scoring from nflverse, measured vs simulated correlations
  test_ownership_model.py # schedule lines, recency/injury inputs, softmax fit, CV
  test_lateswap.py   # live entry parsing, game clock, completions, end-to-end swap
  test_field_model.py # lineup structure features
  test_review.py     # scorecards, grading, calibration, results log
  test_projections.py # no look-ahead features, fit/project ranking, blend, shipped model
  run_tests.py    # standalone runner (no pytest dependency)
data/
  sample_classic_pool.csv
  sample_sgo_events.json  # SGO /events snapshot matching the sample pool
  field_profile.json      # aggregate field behavior from three 2026 Millys
  correlation_profile.json # role-pair DK score correlations, 2021-25 regular seasons
  field_structure.json    # real Milly lineup structure by entrant group, Weeks 1-5
  projection_model.json   # our projection model (fit on 2021-25 nflverse + Vegas)
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
