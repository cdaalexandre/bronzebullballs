"""Sector concentration cap for ranked candidate lists (Clenow Cap. 7 + Carver Cap. 11).

Clenow: "if you're really concerned [about extreme sector concentration] you
might want to add a sector cap." Carver Cap. 11: within the same sector/country,
asset correlations cluster around 0.75 by rule of thumb. Multiple picks in one
sector inflate the unmeasured risk that the diversification multiplier does
not capture when sectors are ignored.

Two public functions, both order-preserving:

    - :func:`apply_sector_cap_tickers` -- takes a list of tickers already
      sorted by score and returns the top ``n_top`` respecting the cap. Used
      by the walk-forward selector.
    - :func:`apply_sector_cap_rows` -- takes a list of feature rows (dicts
      with nested ``fund.sector``) and partitions them into kept/excluded.
      Used by the screening pipeline.

Literal port from ``corpus_master_v8.py`` lines 115, 931-981.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: Default maximum number of picks per GICS sector. Clenow Cap. 7.
SECTOR_CAP_MAX_PER_SECTOR: int = 3


def apply_sector_cap_tickers(
    ranked_tickers: Sequence[str],
    ticker_to_sector: Mapping[str, str | None],
    max_per_sector: int = SECTOR_CAP_MAX_PER_SECTOR,
    n_top: int = 25,
) -> list[str]:
    """Apply a sector cap to an already-ranked ticker list.

    Iterates ``ranked_tickers`` in order. For each ticker, looks up its sector;
    keeps it if the running count of that sector is below ``max_per_sector``,
    else skips. Stops once ``n_top`` tickers have been kept.

    Missing or ``None`` sectors are treated as ``"Unknown"`` and counted
    against the same bucket -- the cap thus also constrains the "Unknown"
    tail.

    Args:
        ranked_tickers: tickers sorted by score (highest first).
        ticker_to_sector: mapping from ticker to sector name. Missing keys
            are treated as ``"Unknown"``.
        max_per_sector: cap. Default :data:`SECTOR_CAP_MAX_PER_SECTOR`.
        n_top: maximum size of the returned list.

    Returns:
        List of at most ``n_top`` tickers respecting the cap.
    """
    sector_count: dict[str, int] = {}
    capped: list[str] = []
    for tk in ranked_tickers:
        if len(capped) >= n_top:
            break
        sec = ticker_to_sector.get(tk) or "Unknown"
        n = sector_count.get(sec, 0)
        if n < max_per_sector:
            capped.append(tk)
            sector_count[sec] = n + 1
    return capped


def apply_sector_cap_rows(
    rows: Sequence[Mapping[str, Any]],
    max_per_sector: int = SECTOR_CAP_MAX_PER_SECTOR,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, int]]:
    """Apply a sector cap to a list of screening feature rows.

    Each row is expected to have a nested key ``fund.sector`` (i.e.
    ``row["fund"]["sector"]``). Rows whose sector is missing or ``None`` are
    bucketed under ``"Unknown"``.

    Args:
        rows: feature rows from the screening pipeline, already sorted by
            score (highest first).
        max_per_sector: cap. Default :data:`SECTOR_CAP_MAX_PER_SECTOR`.

    Returns:
        Tuple ``(capped, excluded, sector_count)``:

            - ``capped``: rows kept (subject to the cap).
            - ``excluded``: rows skipped because their sector was already full.
            - ``sector_count``: final per-sector counts in ``capped``.
    """
    sector_count: dict[str, int] = {}
    capped: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    for r in rows:
        fund = r.get("fund") or {}
        sec_raw = fund.get("sector") if isinstance(fund, Mapping) else None
        sec = sec_raw or "Unknown"
        n = sector_count.get(sec, 0)
        if n < max_per_sector:
            capped.append(r)
            sector_count[sec] = n + 1
        else:
            excluded.append(r)
    return capped, excluded, sector_count
