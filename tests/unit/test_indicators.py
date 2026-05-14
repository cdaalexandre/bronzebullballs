"""Unit tests for ``bronzebullballs.domain.indicators``.

These cover the three Wilder/Clenow primitives ported literally from
``corpus_master_v8.py`` in FASE E. Synthetic OHLC fixtures (``conftest.py``)
keep tests deterministic and free of I/O. Coverage gate per Custom
Instructions section 8.4: >95% on ``domain/indicators.py``.

Tests are grouped by function into classes; each class has a happy-path
check, a boundary check (insufficient data), and at least one behavioural
check (sign, scaling, or NaN placement) that would catch a real refactor
bug, not just a typo.

Fixture parameters are not type-annotated. Pytest resolves them at runtime
and the correct hint would be ``Callable[..., pd.DataFrame]``, which adds an
import for zero static-checking benefit since mypy's CI gate runs against
``src/`` only.
"""

from __future__ import annotations

import math

import pandas as pd

from bronzebullballs.domain.indicators import adjusted_slope, atr, rsi

# ----------------------------------------------------------------------------
# rsi
# ----------------------------------------------------------------------------


class TestRsi:
    def test_returns_series_bounded_zero_to_one_hundred(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=1)
        out = rsi(df["close"], n=14)
        assert isinstance(out, pd.Series)
        valid = out.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_first_n_rows_are_nan_then_value_appears(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=1)
        n = 14
        out = rsi(df["close"], n=n)
        # `diff` yields 1 NaN at index 0; rolling needs `n` valid deltas after
        # that, so the first non-NaN RSI lands at index `n` exactly.
        assert out.iloc[:n].isna().all()
        assert not pd.isna(out.iloc[n])

    def test_insufficient_data_yields_all_nan(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0])
        out = rsi(close, n=14)
        assert out.isna().all()


# ----------------------------------------------------------------------------
# atr
# ----------------------------------------------------------------------------


class TestAtr:
    def test_returns_series_of_strictly_positive_values(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=2)
        out = atr(df["close"], df["high"], df["low"], n=20)
        assert isinstance(out, pd.Series)
        valid = out.dropna()
        assert (valid > 0).all()

    def test_first_n_minus_one_rows_are_nan(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=100, seed=2)
        n = 20
        out = atr(df["close"], df["high"], df["low"], n=n)
        # Unlike RSI, ATR's first TR is well-defined at index 0 (high-low),
        # so the rolling(n).mean() emits its first value at index n-1.
        assert out.iloc[: n - 1].isna().all()
        assert not pd.isna(out.iloc[n - 1])

    def test_wider_bands_yield_larger_atr(self, synthetic_ohlcv) -> None:
        """Sanity on the TR formula: bigger high/low spread => bigger ATR."""
        df = synthetic_ohlcv(n_days=100, seed=3)
        narrow = atr(df["close"], df["close"] * 1.001, df["close"] * 0.999, n=20)
        wide = atr(df["close"], df["close"] * 1.05, df["close"] * 0.95, n=20)
        assert wide.dropna().mean() > narrow.dropna().mean()


# ----------------------------------------------------------------------------
# adjusted_slope
# ----------------------------------------------------------------------------


class TestAdjustedSlope:
    def test_happy_path_returns_finite_float(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=500, seed=4)
        out = adjusted_slope(df["close"], period=90)
        assert isinstance(out, float)
        assert math.isfinite(out)

    def test_insufficient_data_returns_nan(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=50, seed=4)
        out = adjusted_slope(df["close"], period=90)
        assert math.isnan(out)

    def test_positive_drift_yields_positive_slope(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=500, seed=5, drift=0.001, vol=0.005)
        out = adjusted_slope(df["close"], period=90)
        assert out > 0

    def test_negative_drift_yields_negative_slope(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=500, seed=5, drift=-0.001, vol=0.005)
        out = adjusted_slope(df["close"], period=90)
        assert out < 0

    def test_determinism_bit_exact(self, synthetic_ohlcv) -> None:
        """Same seed => same float, bit-exact. Custom Instructions section 6 gate."""
        a = adjusted_slope(synthetic_ohlcv(n_days=500, seed=42)["close"], period=90)
        b = adjusted_slope(synthetic_ohlcv(n_days=500, seed=42)["close"], period=90)
        assert a == b
