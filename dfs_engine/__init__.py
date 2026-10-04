"""
dfs_engine — DraftKings NFL DFS lineup optimization engine.

A correlated Monte Carlo + MILP optimal-lineup pipeline for large-field
GPP tournaments (Millionaire Maker style), supporting both Classic and
Showdown (single-game) contest formats.

Modules:
    data        -- load/validate player pool CSVs
    simulate    -- correlated game-script Monte Carlo simulation
    optimize    -- MILP lineup solver (Classic + Showdown)
    outcomes    -- per-player median/ceiling/boom/bust/Optimal% report
    portfolio   -- build N lineups with exposure caps + uniqueness
    diagnostics -- Optimal% leverage diagnostic (SaberSim-style)
    sportsgameodds -- SportsGameOdds lines + prop-implied baseline projections
"""

__version__ = "0.1.0"
