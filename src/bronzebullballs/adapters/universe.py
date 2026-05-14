"""Cascading source of S&P 500 tickers (GitHub CSV -> Datahub -> Wikipedia).

Skeleton only. Real logic will be ported from `corpus_master_v8.py`
(`fetch_sp500_tickers` and its `_try_*` helpers) under Category 4.
"""

from __future__ import annotations

from bronzebullballs.log import get_logger

logger = get_logger(__name__)


class SP500CascadingSource:
    """Concrete `UniverseSource` cascading through 3 online sources + fallback.

    See `adapters.protocols.UniverseSource` for interface contract.
    """

    def fetch_tickers(self) -> list[str]:
        raise NotImplementedError("port from corpus_master_v8.fetch_sp500_tickers")