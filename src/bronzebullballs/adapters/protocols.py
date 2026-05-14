"""Structural interfaces (typing.Protocol) for external data sources.

Defined here so that the domain and service layers can depend on these
abstractions instead of concrete adapter implementations. Concrete adapters
(market_data, earnings_calendar, universe) duck-type-match these protocols;
no inheritance is required.

Reference: Fluent Python Cap. 13 (structural subtyping); Architecture
Patterns with Python Cap. 13 (Protocols + ABCs for ports).
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class DataFetcher(Protocol):
    """Source of OHLCV history and fundamental data for a list of tickers.

    Implementations: `adapters.market_data.YahooQueryFetcher` in production;
    fake/synthetic fetchers in tests (see tests/conftest.py FASE D).
    """

    def fetch_history(
        self,
        tickers: list[str],
        period: str = "2y",
    ) -> dict[str, pd.DataFrame]:
        """Return per-ticker OHLCV DataFrames.

        Args:
            tickers: Symbols to fetch (Yahoo format, e.g. "BRK-B" not "BRK.B").
            period: Yahoo-style period string ("2y", "10y", etc.).

        Returns:
            Mapping `{ticker: DataFrame}` with columns at least
            ['open', 'high', 'low', 'close', 'volume'] indexed by date.
            Missing tickers are simply absent from the dict.
        """
        ...

    def fetch_fundamentals(
        self,
        tickers: list[str],
    ) -> dict[str, dict[str, float | str | None]]:
        """Return per-ticker fundamental fields.

        Args:
            tickers: Symbols to fetch.

        Returns:
            Mapping `{ticker: {field: value}}`. Expected fields include
            'sector', 'beta', 'analyst_upside', 'beat_rate', 'target_price'.
            Missing values are None; missing tickers map to empty dict.
        """
        ...


class EarningsCalendar(Protocol):
    """Source of upcoming earnings dates per ticker.

    Implementations: `adapters.earnings_calendar.YFinanceCalendar` in
    production; fakes in tests.
    """

    def days_until_earnings(self, ticker: str) -> int | None:
        """Return business days until the next reported earnings date.

        Args:
            ticker: Symbol to check.

        Returns:
            Non-negative integer if a future earnings date is known,
            otherwise None (no coverage or stale data).
        """
        ...


class UniverseSource(Protocol):
    """Source of the S&P 500 constituent list.

    Implementations: `adapters.universe.SP500CascadingSource` (GitHub CSV ->
    Datahub -> Wikipedia -> hardcoded fallback) in production; fakes in tests.
    """

    def fetch_tickers(self) -> list[str]:
        """Return current S&P 500 constituents in Yahoo ticker format.

        Returns:
            Sorted list of unique tickers. Always non-empty (falls back to a
            hardcoded mega-cap list if all online sources fail).
        """
        ...
