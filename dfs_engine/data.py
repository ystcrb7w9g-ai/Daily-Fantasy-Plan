"""
Player pool loading and validation.

Expected CSV columns (Classic):
    name, team, opp, position, salary, proj, own, ceiling,
    team_total, game_total, spread

Expected CSV columns (Showdown) — same, plus:
    (no roster-slot column; Captain/FLEX decision happens in optimize.py)

`own` is expected as a decimal fraction (0.184) OR a percentage (18.4) —
loader auto-detects and normalizes to a decimal fraction in [0, 1].
"""
from __future__ import annotations

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

    # Normalize ownership to a decimal fraction regardless of input scale.
    if df["own"].max() > 1.5:
        df["own"] = df["own"] / 100.0

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
