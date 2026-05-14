"""yfinance-backed implementation of `EarningsCalendar`.

Skeleton only. Real logic will be ported from `corpus_master_v8.py` (the
yfinance fallback used for top-N earnings dates) under Category 4.
"""

from __future__ import annotations

from bronzebullballs.log import get_logger

logger = get_logger(__name__)


class YFinanceCalendar:
    """Concrete `EarningsCalendar` using `yfinance.Ticker.calendar`.

    See `adapters.protocols.EarningsCalendar` for interface contract.
    """

    def days_until_earnings(self, ticker: str) -> int | None:
        raise NotImplementedError(
            "port from corpus_master_v8 yfinance earnings block"
        )