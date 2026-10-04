"""
SportsGameOdds (SGO) integration: Vegas lines + player-prop baseline projections.

SportsGameOdds (https://sportsgameodds.com) aggregates odds from many
sportsbooks and exposes a vig-free consensus ("fair") line for every
market. This module pulls that data and turns it into the two things the
rest of the engine needs:

1. **Game lines** -- game total, spread, and team totals, which feed the
   correlated simulation in `simulate.py` (columns `game_total`,
   `spread`, `team_total`).
2. **A market-implied fantasy baseline** -- each player's prop lines
   (passing/rushing/receiving yards, receptions, TDs, INTs) converted into
   an expected DraftKings fantasy score. This is blended with (or fills
   in for) the `proj` column of a player pool.

API notes (v2, https://api.sportsgameodds.com/v2):
    * Auth is an `x-api-key` header. The key is read from the
      `SPORTSGAMEODDS_API_KEY` environment variable by default.
    * `GET /events?leagueID=NFL&oddsAvailable=true` returns
      `{"success": ..., "data": [Event, ...], "nextCursor": ...}`.
    * `event["odds"]` is keyed by oddID:
      `{statID}-{statEntityID}-{periodID}-{betTypeID}-{sideID}`, e.g.
      `points-all-game-ou-over` (game total) or
      `receiving_yards-STEFON_DIGGS_1_NFL-game-ou-over` (player prop).
    * Each odd carries consensus `fairOdds`/`fairOverUnder`/`fairSpread`
      (vig removed) and `bookOdds`/`bookOverUnder`/`bookSpread` (with vig).
      We prefer fair values and fall back to book values.

The fetch layer uses only the standard library, so no extra dependency
is needed. Raw events can be saved to JSON (`save_events`) and reloaded
later (`load_events`) for reproducible / offline runs.
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm, poisson

BASE_URL = "https://api.sportsgameodds.com/v2"
API_KEY_ENV = "SPORTSGAMEODDS_API_KEY"

# SGO teamIDs look like "BUFFALO_BILLS_NFL"; map to DraftKings abbreviations.
TEAM_ID_TO_ABBR = {
    "ARIZONA_CARDINALS_NFL": "ARI", "ATLANTA_FALCONS_NFL": "ATL",
    "BALTIMORE_RAVENS_NFL": "BAL", "BUFFALO_BILLS_NFL": "BUF",
    "CAROLINA_PANTHERS_NFL": "CAR", "CHICAGO_BEARS_NFL": "CHI",
    "CINCINNATI_BENGALS_NFL": "CIN", "CLEVELAND_BROWNS_NFL": "CLE",
    "DALLAS_COWBOYS_NFL": "DAL", "DENVER_BRONCOS_NFL": "DEN",
    "DETROIT_LIONS_NFL": "DET", "GREEN_BAY_PACKERS_NFL": "GB",
    "HOUSTON_TEXANS_NFL": "HOU", "INDIANAPOLIS_COLTS_NFL": "IND",
    "JACKSONVILLE_JAGUARS_NFL": "JAX", "KANSAS_CITY_CHIEFS_NFL": "KC",
    "LAS_VEGAS_RAIDERS_NFL": "LV", "LOS_ANGELES_CHARGERS_NFL": "LAC",
    "LOS_ANGELES_RAMS_NFL": "LAR", "MIAMI_DOLPHINS_NFL": "MIA",
    "MINNESOTA_VIKINGS_NFL": "MIN", "NEW_ENGLAND_PATRIOTS_NFL": "NE",
    "NEW_ORLEANS_SAINTS_NFL": "NO", "NEW_YORK_GIANTS_NFL": "NYG",
    "NEW_YORK_JETS_NFL": "NYJ", "PHILADELPHIA_EAGLES_NFL": "PHI",
    "PITTSBURGH_STEELERS_NFL": "PIT", "SAN_FRANCISCO_49ERS_NFL": "SF",
    "SEATTLE_SEAHAWKS_NFL": "SEA", "TAMPA_BAY_BUCCANEERS_NFL": "TB",
    "TENNESSEE_TITANS_NFL": "TEN", "WASHINGTON_COMMANDERS_NFL": "WAS",
}

# Alternate abbreviations seen in other data sources -> canonical DK form.
ABBR_ALIASES = {"JAC": "JAX", "WSH": "WAS", "LA": "LAR", "OAK": "LV", "SD": "LAC", "STL": "LAR"}

# DraftKings NFL Classic offensive scoring, per unit of each SGO statID.
DK_POINTS = {
    "passing_yards": 0.04,
    "passing_touchdowns": 4.0,
    "passing_interceptions": -1.0,
    "rushing_yards": 0.1,
    "rushing_touchdowns": 6.0,
    "receiving_yards": 0.1,
    "receiving_receptions": 1.0,
    "receiving_touchdowns": 6.0,
    "touchdowns": 6.0,  # anytime TD (rushing + receiving)
}

# DK 100/300-yard bonuses: statID -> (threshold, bonus points).
DK_BONUSES = {
    "passing_yards": (300.0, 3.0),
    "rushing_yards": (100.0, 3.0),
    "receiving_yards": (100.0, 3.0),
}

# Yardage props are modeled as log-normal around the line; this is the
# assumed coefficient of variation of a player's single-game yardage.
YARDAGE_CV = {"passing_yards": 0.30, "rushing_yards": 0.55, "receiving_yards": 0.65}

COUNT_STATS = {
    "passing_touchdowns", "passing_interceptions", "rushing_touchdowns",
    "receiving_receptions", "receiving_touchdowns", "touchdowns",
}

# Rough yards-per-TD rates used only when a player has a yardage prop but
# no TD market (so the baseline doesn't silently drop TD equity).
IMPUTE_YARDS_PER_TD = {
    "passing_yards": ("passing_touchdowns", 150.0),
    "rushing_yards": ("rushing_touchdowns", 120.0),
    "receiving_yards": ("receiving_touchdowns", 160.0),
}
IMPUTE_QB_INTERCEPTIONS = 0.75

# A player needs at least one of these markets to get an SGO baseline.
CORE_MARKETS = {
    "QB": {"passing_yards"},
    "RB": {"rushing_yards", "receiving_yards"},
    "WR": {"receiving_yards", "receiving_receptions"},
    "TE": {"receiving_yards", "receiving_receptions"},
}

# Used to fill a missing `ceiling` from `proj` (matches the sample pool's ratios).
CEILING_MULT = {"QB": 1.55, "RB": 1.85, "WR": 1.9, "TE": 1.9, "DST": 2.0}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_events(
    api_key: str | None = None,
    league_id: str = "NFL",
    starts_after: str | None = None,
    starts_before: str | None = None,
    odds_available: bool = True,
    bookmaker_id: str | None = None,
    page_limit: int = 50,
    max_pages: int = 20,
    base_url: str = BASE_URL,
    timeout: float = 30.0,
) -> list[dict]:
    """
    Fetch upcoming events (with odds) from the SGO `/events` endpoint,
    following `nextCursor` pagination. Returns the raw event dicts.
    """
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"No SportsGameOdds API key: pass api_key or set ${API_KEY_ENV}."
        )

    params: dict[str, str] = {
        "leagueID": league_id,
        "oddsAvailable": str(odds_available).lower(),
        "started": "false",
        "limit": str(page_limit),
    }
    if starts_after:
        params["startsAfter"] = starts_after
    if starts_before:
        params["startsBefore"] = starts_before
    if bookmaker_id:
        params["bookmakerID"] = bookmaker_id

    events: list[dict] = []
    cursor = None
    for _ in range(max_pages):
        if cursor:
            params["cursor"] = cursor
        url = f"{base_url.rstrip('/')}/events?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"x-api-key": api_key, "accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("success") is False:
            raise RuntimeError(f"SportsGameOdds error: {payload.get('error') or payload}")
        events.extend(payload.get("data") or [])
        cursor = payload.get("nextCursor")
        if not cursor:
            break
    return events


def save_events(events: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=1)


def load_events(path: str) -> list[dict]:
    """Load events saved by `save_events` (or a raw `/events` response body)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("data", [])
    return data


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _num(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def american_to_prob(odds) -> float | None:
    """Convert American odds ("-110", "+120", 150) to implied probability."""
    o = _num(odds)
    if o is None or o == 0:
        return None
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def team_abbr(team_id: str | None, names: dict | None = None) -> str | None:
    if team_id and team_id in TEAM_ID_TO_ABBR:
        return TEAM_ID_TO_ABBR[team_id]
    short = (names or {}).get("short")
    if short:
        return ABBR_ALIASES.get(short.upper(), short.upper())
    return None


def normalize_abbr(abbr: str) -> str:
    abbr = str(abbr).strip().upper()
    return ABBR_ALIASES.get(abbr, abbr)


_SUFFIX_RE = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation/suffixes for fuzzy name joins."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", " "))
    s = _SUFFIX_RE.sub("", s)
    return " ".join(s.split())


def _line(odd: dict, kind: str) -> float | None:
    """kind: 'OverUnder' or 'Spread'. Prefer the fair consensus line."""
    v = _num(odd.get(f"fair{kind}"))
    return v if v is not None else _num(odd.get(f"book{kind}"))


def _over_prob(over: dict | None, under: dict | None) -> float | None:
    """Vig-free probability of the over/yes side."""
    if over is not None:
        p = american_to_prob(over.get("fairOdds"))
        if p is not None:
            return p
    po = american_to_prob(over.get("bookOdds")) if over else None
    pu = american_to_prob(under.get("bookOdds")) if under else None
    if po is not None and pu is not None and po + pu > 0:
        return po / (po + pu)
    if po is not None:
        return po
    if pu is not None:
        return 1.0 - pu
    return None


# ---------------------------------------------------------------------------
# Game lines
# ---------------------------------------------------------------------------

def extract_game_lines(events: Iterable[dict]) -> pd.DataFrame:
    """
    One row per team per game: team, opp, game_total, spread (that team's,
    negative = favored), team_total, starts_at, event_id.
    """
    rows = []
    for ev in events:
        teams = ev.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        h = team_abbr(home.get("teamID"), home.get("names"))
        a = team_abbr(away.get("teamID"), away.get("names"))
        if not h or not a:
            continue
        odds = ev.get("odds") or {}

        total = _line(odds.get("points-all-game-ou-over", {}), "OverUnder")
        home_spread = _line(odds.get("points-home-game-sp-home", {}), "Spread")
        if home_spread is None:
            away_spread = _line(odds.get("points-away-game-sp-away", {}), "Spread")
            home_spread = -away_spread if away_spread is not None else None
        if total is None or home_spread is None:
            continue

        home_tt = _line(odds.get("points-home-game-ou-over", {}), "OverUnder")
        away_tt = _line(odds.get("points-away-game-ou-over", {}), "OverUnder")
        # Implied team totals from total/spread when team-total markets are absent.
        if home_tt is None:
            home_tt = total / 2.0 - home_spread / 2.0
        if away_tt is None:
            away_tt = total / 2.0 + home_spread / 2.0

        starts_at = (ev.get("status") or {}).get("startsAt")
        for team, opp, spread, tt in ((h, a, home_spread, home_tt), (a, h, -home_spread, away_tt)):
            rows.append({
                "event_id": ev.get("eventID"), "starts_at": starts_at,
                "team": team, "opp": opp, "game_total": total,
                "spread": spread, "team_total": tt,
            })
    return pd.DataFrame(rows, columns=[
        "event_id", "starts_at", "team", "opp", "game_total", "spread", "team_total",
    ])


# ---------------------------------------------------------------------------
# Player props
# ---------------------------------------------------------------------------

def extract_player_props(events: Iterable[dict]) -> pd.DataFrame:
    """
    Long table of full-game player over/under (and yes/no) props:
    sgo_player_id, name, team, stat_id, line, p_over.
    Only statIDs that score in DraftKings NFL Classic are kept.
    """
    rows = []
    for ev in events:
        odds = ev.get("odds") or {}
        players = ev.get("players") or {}
        seen: set[tuple[str, str]] = set()
        for odd_id, odd in odds.items():
            parts = odd_id.split("-")
            if len(parts) != 5:
                continue
            stat_id, entity, period, bet_type, side = parts
            if stat_id not in DK_POINTS or period != "game" or entity in ("home", "away", "all"):
                continue
            if (bet_type, side) not in (("ou", "over"), ("yn", "yes")):
                continue
            if (entity, stat_id) in seen:
                continue
            seen.add((entity, stat_id))

            opp_side = "under" if bet_type == "ou" else "no"
            opposing = odds.get(odd.get("opposingOddID") or f"{stat_id}-{entity}-{period}-{bet_type}-{opp_side}")
            line = _line(odd, "OverUnder") if bet_type == "ou" else 0.5
            if line is None:
                continue
            p_over = _over_prob(odd, opposing)

            player = players.get(entity, {})
            team_id = player.get("teamID")
            rows.append({
                "sgo_player_id": entity,
                "name": player.get("name") or _name_from_player_id(entity),
                "team": team_abbr(team_id) if team_id else None,
                "stat_id": stat_id,
                "line": line,
                "p_over": 0.5 if p_over is None else float(np.clip(p_over, 0.02, 0.98)),
            })
    return pd.DataFrame(rows, columns=["sgo_player_id", "name", "team", "stat_id", "line", "p_over"])


def _name_from_player_id(player_id: str) -> str:
    # "STEFON_DIGGS_1_NFL" -> "Stefon Diggs"
    parts = [p for p in player_id.split("_") if p and not p.isdigit() and p != "NFL"]
    return " ".join(p.capitalize() for p in parts)


def expected_count(line: float, p_over: float) -> float:
    """Poisson mean lambda such that P(X > line) == p_over."""
    k = math.floor(line)  # P(X > k + 0.5) == P(X >= k + 1)
    f = lambda lam: poisson.sf(k, lam) - p_over  # noqa: E731
    return float(brentq(f, 1e-6, 100.0))


def lognormal_params(line: float, p_over: float, cv: float) -> tuple[float, float]:
    """(mu, sigma) of log-yardage such that P(X > line) == p_over."""
    sigma = math.sqrt(math.log1p(cv ** 2))
    mu = math.log(max(line, 0.5)) + sigma * norm.ppf(p_over)
    return mu, sigma


@dataclass
class PlayerBaseline:
    name: str
    team: str | None
    sgo_player_id: str
    sgo_proj: float
    markets: list[str] = field(default_factory=list)
    imputed: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)


def _player_baseline(pid: str, grp: pd.DataFrame) -> PlayerBaseline:
    by_stat = {r.stat_id: (r.line, r.p_over) for r in grp.itertuples()}
    comps: dict[str, float] = {}
    imputed: list[str] = []
    means: dict[str, float] = {}

    for stat, (line, p) in by_stat.items():
        if stat in YARDAGE_CV:
            mu, sigma = lognormal_params(line, p, YARDAGE_CV[stat])
            means[stat] = math.exp(mu + sigma ** 2 / 2.0)
            comps[stat] = DK_POINTS[stat] * means[stat]
            thresh, bonus = DK_BONUSES[stat]
            comps[f"{stat}_bonus"] = bonus * float(norm.sf((math.log(thresh) - mu) / sigma))
        elif stat in COUNT_STATS:
            means[stat] = expected_count(line, p)

    # Avoid double-counting: anytime TD covers rush + rec TDs.
    if "touchdowns" in means and ("rushing_touchdowns" in means or "receiving_touchdowns" in means):
        del means["touchdowns"]
    has_any_td = "touchdowns" in means
    for yard_stat, (td_stat, ypt) in IMPUTE_YARDS_PER_TD.items():
        if yard_stat in means and td_stat not in means and not (has_any_td and td_stat != "passing_touchdowns"):
            means[td_stat] = means[yard_stat] / ypt
            imputed.append(td_stat)
    if "passing_yards" in means and "passing_interceptions" not in means:
        means["passing_interceptions"] = IMPUTE_QB_INTERCEPTIONS
        imputed.append("passing_interceptions")

    for stat, m in means.items():
        if stat not in YARDAGE_CV:
            comps[stat] = DK_POINTS[stat] * m

    first = grp.iloc[0]
    return PlayerBaseline(
        name=first["name"], team=first["team"], sgo_player_id=pid,
        sgo_proj=round(sum(comps.values()), 2),
        markets=sorted(by_stat), imputed=sorted(imputed), components=comps,
    )


def props_to_baseline(props: pd.DataFrame) -> pd.DataFrame:
    """
    Convert player props into an expected DraftKings fantasy score per player.

    Yardage props are treated as log-normal (right-skewed, so the mean sits
    above the median line); count props (TDs, receptions, INTs) as Poisson.
    The 100/300-yard bonuses are priced from the same yardage distribution.
    """
    out = []
    for pid, grp in props.groupby("sgo_player_id", sort=False):
        b = _player_baseline(pid, grp)
        out.append({
            "sgo_player_id": b.sgo_player_id, "name": b.name, "team": b.team,
            "sgo_proj": b.sgo_proj, "sgo_markets": ";".join(b.markets),
            "sgo_imputed": ";".join(b.imputed),
        })
    return pd.DataFrame(out, columns=[
        "sgo_player_id", "name", "team", "sgo_proj", "sgo_markets", "sgo_imputed",
    ])


# ---------------------------------------------------------------------------
# Applying to a player pool
# ---------------------------------------------------------------------------

@dataclass
class EnrichReport:
    teams_updated: list[str]
    teams_unmatched: list[str]
    players_matched: int
    players_unmatched: list[str]


def apply_sgo_baseline(
    pool: pd.DataFrame,
    lines: pd.DataFrame | None,
    baseline: pd.DataFrame | None,
    weight: float = 0.5,
    update_lines: bool = True,
) -> tuple[pd.DataFrame, EnrichReport]:
    """
    Merge SGO data into a raw player-pool DataFrame (before `load_player_pool`).

    - Game lines: overwrite `game_total` / `spread` / `team_total` for
      matched teams (unless `update_lines=False`; then only fill blanks).
    - Projections: `proj = (1 - weight) * proj + weight * sgo_proj` for
      players with an SGO baseline; a blank `proj` is filled with
      `sgo_proj` outright. `ceiling` is scaled by the same ratio (or
      estimated from `proj` if blank). DST rows are left untouched.

    Adds audit columns: `proj_input`, `sgo_proj`, `sgo_markets`,
    `sgo_imputed`, `proj_source`.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    df = pool.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("proj", "ceiling", "team_total", "game_total", "spread"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    team_key = df["team"].map(normalize_abbr)

    teams_updated, teams_unmatched = [], []
    if lines is not None and len(lines):
        by_team = lines.drop_duplicates("team", keep="first").set_index("team")
        for team in sorted(team_key.unique()):
            mask = (team_key == team).to_numpy()
            if team not in by_team.index:
                teams_unmatched.append(team)
                continue
            row = by_team.loc[team]
            for col in ("game_total", "spread", "team_total"):
                if update_lines:
                    df.loc[mask, col] = row[col]
                else:
                    df.loc[mask & df[col].isna().to_numpy(), col] = row[col]
            teams_updated.append(team)

    df["proj_input"] = df["proj"]
    df["sgo_proj"] = np.nan
    df["sgo_markets"] = ""
    df["sgo_imputed"] = ""
    df["proj_source"] = np.where(df["proj"].notna(), "input", "")

    matched, unmatched = 0, []
    if baseline is not None and len(baseline):
        bl = baseline.copy()
        bl["_key"] = bl["name"].map(normalize_name)
        bl["_team"] = bl["team"].map(lambda t: normalize_abbr(t) if isinstance(t, str) else None)
        for i, r in df.iterrows():
            pos = r["position"]
            if pos not in CORE_MARKETS:
                continue
            cand = bl[bl["_key"] == normalize_name(r["name"])]
            if len(cand) > 1:
                same_team = cand[cand["_team"] == team_key[i]]
                cand = same_team if len(same_team) else cand
            if cand.empty:
                unmatched.append(r["name"])
                continue
            b = cand.iloc[0]
            if not CORE_MARKETS[pos] & set(b["sgo_markets"].split(";")):
                unmatched.append(r["name"])
                continue
            matched += 1
            df.at[i, "sgo_proj"] = b["sgo_proj"]
            df.at[i, "sgo_markets"] = b["sgo_markets"]
            df.at[i, "sgo_imputed"] = b["sgo_imputed"]

            old = r["proj"]
            if pd.isna(old):
                new, src = b["sgo_proj"], "sgo"
            else:
                new, src = (1 - weight) * old + weight * b["sgo_proj"], "blend"
            df.at[i, "proj"] = round(new, 2)
            df.at[i, "proj_source"] = src
            if pd.notna(r["ceiling"]) and pd.notna(old) and old > 0:
                df.at[i, "ceiling"] = round(r["ceiling"] * new / old, 2)

    missing_ceiling = df["ceiling"].isna() & df["proj"].notna()
    df.loc[missing_ceiling, "ceiling"] = (
        df.loc[missing_ceiling, "proj"] * df.loc[missing_ceiling, "position"].map(CEILING_MULT).fillna(1.8)
    ).round(2)

    return df, EnrichReport(teams_updated, teams_unmatched, matched, unmatched)
