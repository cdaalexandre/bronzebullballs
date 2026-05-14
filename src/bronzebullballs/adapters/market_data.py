"""yahooquery-backed implementation of `DataFetcher`.

Skeleton only -- raises NotImplementedError. Real logic will be ported from
`corpus_master_v8.py` (`fetch_universe`, `fetch_fundamentals`) in a later
phase, under Category 4 (architectural refactor) protocol.
"""

from __future__ import annotations

import pandas as pd

from bronzebullballs.log import get_logger

logger = get_logger(__name__)


class YahooQueryFetcher:
    """Concrete `DataFetcher` using `yahooquery.Ticker`.

    See `adapters.protocols.DataFetcher` for interface contract.
    """

    def fetch_history(
        self,
        tickers: list[str],
        period: str = "2y",
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError("port from corpus_master_v8.fetch_universe")

    def fetch_fundamentals(
        self,
        tickers: list[str],
    ) -> dict[str, dict[str, float | str | None]]:
        raise NotImplementedError(
            "port from corpus_master_v8.fetch_fundamentals"
        )