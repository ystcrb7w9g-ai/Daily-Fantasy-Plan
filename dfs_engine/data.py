"""
Player pool loading and validation.

Expected CSV columns (Classic):
    name, team, opp, position, salary, proj, own, ceiling,
    team_total, game_total, spread

Expected CSV columns (Showdown) — same, plus:
    (no roster-slot column; Captain/FLEX decision happens in optimize.py)

`own` is expected as a decimal fraction (0.184) OR a percentage (18.4) —
loader auto-detects and normalizes to a decimal fraction in [0, 1].

Rows with no (or non-positive) `proj` are dropped with a warning, so a
full DraftKings player list can be loaded directly once projected. Missing
`own` is treated as 0%; missing Vegas-line columns are an error. Extra
columns (e.g. `dk_id`, `game_time`) are kept.
"""
from __future__ import annotations

import warnings

import pandas as pd

REQUIRED_COLUMNS = [
    "name", "team", "opp", "position", "salary", "proj", "own", "ceiling",
    "team_total", "game_total", "spread",
]

VALID_POSITIONS = {"QB", "RB", "WR", "TE", "DST"}


def load_player_pool(path: str) -> pd.DataFrame:
    """Load and validate a player pool CSV, returning a normalized DataFrame."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Player pool missing required columns: {missing}")

    df["position"] = df["position"].str.upper().str.strip()
    bad_pos = set(df["position"]) - VALID_POSITIONS
    if bad_pos:
        raise ValueError(f"Unknown position codes found: {bad_pos}")

    # A full DraftKings player list includes backups nobody projects;
    # players without a positive projection can't be rostered sensibly.
    for col in ("proj", "own", "ceiling", "team_total", "game_total", "spread"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    unprojected = df["proj"].isna() | (df["proj"] <= 0)
    if unprojected.any():
        warnings.warn(f"Dropping {int(unprojected.sum())} players with no projection.")
        df = df[~unprojected].copy()

    missing_lines = df[df[["team_total", "game_total", "spread"]].isna().any(axis=1)]
    if len(missing_lines):
        teams = sorted(missing_lines["team"].unique())
        raise ValueError(f"Missing team_total/game_total/spread for teams: {teams} "
                         "(run sgo-enrich or fill them in).")

    # Normalize ownership to a decimal fraction regardless of input scale.
    if df["own"].max() > 1.5:
        df["own"] = df["own"] / 100.0
    if df["own"].isna().any():
        warnings.warn(f"{int(df['own'].isna().sum())} players have no ownership; treating as 0%.")
        df["own"] = df["own"].fillna(0.0)

    df["salary"] = df["salary"].astype(int)
    for col in ("proj", "ceiling", "team_total", "game_total", "spread", "own"):
        df[col] = df[col].astype(float)

    df = df.reset_index(drop=True)
    df["player_id"] = df.index
    return df


def validate_showdown_pool(df: pd.DataFrame) -> None:
    """Showdown slates must contain exactly two teams."""
    teams = df["team"].unique()
    if len(teams) != 2:
        raise ValueError(f"Showdown pool must have exactly 2 teams, found: {teams}")
