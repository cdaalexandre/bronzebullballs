"""Shared pytest fixtures for bronzebullballs tests.

Fixtures are factories, not constants (Custom Instructions section 8.2). A
constant fixture shared across test modules is the easiest way to get
"distant test broke for no reason" bugs; factories let each test ask for
exactly the shape it needs.

Fixtures provided here:
    - ``rng``: ``random.Random(42)`` seeded RNG for non-numpy code paths.
    - ``synthetic_ohlcv``: callable returning a deterministic OHLC DataFrame.

Future fixtures (added when first needed, not pre-emptively):
    - ``fake_data_fetcher``: Fake satisfying ``adapters.protocols.DataFetcher``.
"""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> random.Random:
    """``random.Random`` with fixed seed 42. Use for tests that do NOT touch numpy."""
    return random.Random(42)


@pytest.fixture
def synthetic_ohlcv() -> Callable[..., pd.DataFrame]:
    """Factory of deterministic OHLC DataFrames for indicator tests.

    Returns a callable. Default produces 500 business days of geometric
    Brownian motion starting at 100.0 with ``drift=0.0003`` and ``vol=0.015``,
    which feels like a single equity at ~7% annualized return / ~24%
    annualized vol. High/Low are static +/- 0.5% bands around Close -- enough
    for ATR tests, not a realistic intraday range.

    Why ``np.random.default_rng`` instead of ``random.Random``? Walk-forward
    (section 6 of Custom Instructions) uses ``np.random.default_rng(42)`` as
    its determinism contract, so the indicator-test RNG matches that
    contract. Bit-exact reproducibility across Python minor versions is more
    robust with the numpy generator than with stdlib ``random``.
    """

    def _factory(
        n_days: int = 500,
        seed: int = 42,
        drift: float = 0.0003,
        vol: float = 0.015,
        start_price: float = 100.0,
    ) -> pd.DataFrame:
        generator = np.random.default_rng(seed)
        # n_days-1 log-returns so the DataFrame has exactly n_days rows.
        log_returns = generator.normal(loc=drift, scale=vol, size=n_days - 1)
        log_returns = np.insert(log_returns, 0, 0.0)
        prices = start_price * np.exp(np.cumsum(log_returns))
        dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        close = pd.Series(prices, index=dates, name="close")
        high = close * 1.005
        low = close * 0.995
        return pd.DataFrame({"close": close, "high": high, "low": low})

    return _factory
