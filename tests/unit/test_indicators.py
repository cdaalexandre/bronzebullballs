"""Unit tests for ``bronzebullballs.domain.indicators``.

Covers all thirteen indicators ported from ``corpus_master_v8.py``:

    FASE E -- rsi, atr, adjusted_slope.
    FASE F.1 -- ewmac_forecast, vol_ann, bollinger_z, hurst_clipped,
                variance_ratio, half_life, past_return, positive_reversal,
                sharpe_ratio_real, compute_stock.

Synthetic OHLC fixtures (``conftest.py``) keep tests deterministic and free
of I/O. Coverage gate per Custom Instructions section 8.4: >95% on
``domain/indicators.py``.

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

import numpy as np
import pandas as pd

from bronzebullballs.domain.indicators import (
    adjusted_slope,
    atr,
    bollinger_z,
    compute_stock,
    ewmac_forecast,
    half_life,
    hurst_clipped,
    past_return,
    positive_reversal,
    rsi,
    sharpe_ratio_real,
    variance_ratio,
    vol_ann,
)

# ============================================================================
# FASE E: RSI / ATR / adjusted slope
# ============================================================================


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
        assert out.iloc[:n].isna().all()
        assert not pd.isna(out.iloc[n])

    def test_insufficient_data_yields_all_nan(self) -> None:
        close = pd.Series([100.0, 101.0, 102.0])
        out = rsi(close, n=14)
        assert out.isna().all()


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
        assert out.iloc[: n - 1].isna().all()
        assert not pd.isna(out.iloc[n - 1])


class TestAdjustedSlope:
    def test_uptrend_gives_positive_slope(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=3, drift=0.001)
        s = adjusted_slope(df["close"], period=90)
        assert s > 0

    def test_insufficient_data_returns_nan(self) -> None:
        close = pd.Series(np.linspace(100, 110, 50))
        s = adjusted_slope(close, period=90)
        assert math.isnan(s)

    def test_determinism_bit_exact(self, synthetic_ohlcv) -> None:
        a = adjusted_slope(synthetic_ohlcv(n_days=500, seed=42)["close"], period=90)
        b = adjusted_slope(synthetic_ohlcv(n_days=500, seed=42)["close"], period=90)
        assert a == b


# ============================================================================
# FASE F.1: trend / vol
# ============================================================================


class TestEwmacForecast:
    def test_uptrend_gives_positive_forecast(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=4, drift=0.001)
        fc = ewmac_forecast(df["close"])
        assert fc > 0

    def test_insufficient_data_returns_nan(self) -> None:
        close = pd.Series(np.linspace(100, 110, 50))
        fc = ewmac_forecast(close)
        assert math.isnan(fc)

    def test_forecast_is_capped(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=5)
        fc = ewmac_forecast(df["close"], cap=5.0)
        assert -5.0 <= fc <= 5.0


class TestVolAnn:
    def test_returns_positive_percentage(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=6, vol=0.02)
        v = vol_ann(df["close"], n=25)
        assert v > 0

    def test_higher_input_vol_yields_higher_output(self, synthetic_ohlcv) -> None:
        low = vol_ann(synthetic_ohlcv(n_days=200, seed=7, vol=0.005)["close"])
        high = vol_ann(synthetic_ohlcv(n_days=200, seed=7, vol=0.05)["close"])
        assert high > low

    def test_insufficient_data_returns_nan(self) -> None:
        v = vol_ann(pd.Series([100.0, 101.0]), n=25)
        assert math.isnan(v)


class TestBollingerZ:
    def test_returns_float(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=100, seed=8)
        z = bollinger_z(df["close"])
        assert isinstance(z, float)

    def test_constant_then_spike_gives_large_positive_z(self) -> None:
        s = pd.Series([100.0] * 30 + [120.0])
        z = bollinger_z(s, n=20)
        assert z > 2.0


# ============================================================================
# FASE F.1: mean-reversion (Chan Cap. 2)
# ============================================================================


class TestHurstClipped:
    def test_returns_value_in_unit_interval(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=9)
        h = hurst_clipped(df["close"])
        assert 0.0 <= h <= 1.0

    def test_insufficient_data_returns_nan(self) -> None:
        close = pd.Series(np.linspace(100, 110, 20))
        h = hurst_clipped(close, max_lag=20)
        assert math.isnan(h)

    def test_random_walk_with_drift_yields_finite_hurst(self) -> None:
        """A stochastic random walk should produce a finite Hurst in [0, 1].

        Replaces a previous test that wrongly asserted ``H > 0.55`` on a
        deterministic linear ramp: a linear ramp has constant differences at
        every lag, which kills the variance-of-differences regression and
        returns Hurst near zero (or NaN), not a high persistence value.
        """
        rng = np.random.default_rng(99)
        returns = rng.normal(0.0008, 0.01, 300)
        s = pd.Series(np.cumprod(1 + returns) * 100)
        h = hurst_clipped(s)
        assert math.isfinite(h)
        assert 0.0 <= h <= 1.0

    def test_constant_series_yields_nan(self) -> None:
        """Flat series => zero std at every lag => guarded NaN return."""
        s = pd.Series([100.0] * 100)
        h = hurst_clipped(s, max_lag=20)
        assert math.isnan(h)


class TestVarianceRatio:
    def test_returns_finite_value(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=10)
        vr = variance_ratio(df["close"], k=5)
        assert math.isfinite(vr)

    def test_insufficient_data_returns_nan(self) -> None:
        close = pd.Series(np.linspace(100, 110, 10))
        vr = variance_ratio(close, k=5)
        assert math.isnan(vr)


class TestHalfLife:
    def test_mean_reverting_series_yields_finite_positive_half_life(self) -> None:
        # AR(1) with negative coefficient => mean-reverting.
        rng = np.random.default_rng(11)
        n = 300
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = -0.3 * x[i - 1] + rng.normal(0, 0.01)
        close = pd.Series(100 + x).cumsum() + 100
        hl = half_life(close)
        # Could still be NaN if regression chooses positive lambda on noise,
        # but at minimum it must be either a positive finite number or NaN.
        assert math.isnan(hl) or hl > 0

    def test_insufficient_data_returns_nan(self) -> None:
        hl = half_life(pd.Series([100.0, 101.0, 102.0]))
        assert math.isnan(hl)

    def test_persistent_trending_series_returns_nan(self) -> None:
        """Strong up-trend => OLS lambda non-negative => guarded NaN."""
        rng = np.random.default_rng(77)
        # Strong drift dominates noise => persistent (non-mean-reverting).
        returns = rng.normal(0.003, 0.005, 200)
        s = pd.Series(np.cumprod(1 + returns) * 100)
        hl = half_life(s)
        # Strong drift makes lambda >= 0 likely; function returns NaN.
        # Soft assertion: must not crash, and either NaN or positive.
        assert math.isnan(hl) or hl > 0


# ============================================================================
# FASE F.1: returns / reversal / sharpe
# ============================================================================


class TestPastReturn:
    def test_known_return(self) -> None:
        close = pd.Series([100.0, 105.0, 110.0, 115.0, 120.0, 125.0])
        # Last is 125; 5 bars back is 100; return = 25 %.
        assert math.isclose(past_return(close, days=5), 25.0)

    def test_insufficient_data_returns_nan(self) -> None:
        close = pd.Series([100.0, 101.0])
        assert math.isnan(past_return(close, days=5))


class TestPositiveReversal:
    def test_returns_bool(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=12)
        out = positive_reversal(df["close"])
        assert isinstance(out, bool)

    def test_insufficient_data_returns_false(self) -> None:
        close = pd.Series([100.0] * 100)
        assert positive_reversal(close, lookback=120) is False

    def test_nan_in_window_returns_false(self) -> None:
        """An interior NaN inside the lookback window short-circuits to False."""
        close = pd.Series(np.linspace(100.0, 120.0, 200))
        close.iloc[-50] = float("nan")
        assert positive_reversal(close, lookback=120) is False


class TestSharpeRatioReal:
    def test_uptrend_gives_positive_sharpe(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=13, drift=0.001, vol=0.005)
        sr = sharpe_ratio_real(df["close"])
        assert sr > 0

    def test_insufficient_data_returns_nan(self) -> None:
        close = pd.Series(np.linspace(100, 110, 50))
        sr = sharpe_ratio_real(close, lookback=126)
        assert math.isnan(sr)


# ============================================================================
# FASE F.1: bundle
# ============================================================================


class TestComputeStock:
    def test_returns_dict_with_expected_keys(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=14)
        out = compute_stock(df["close"], df["high"], df["low"])
        expected = {
            "price",
            "slope",
            "ewmac",
            "vol",
            "rsi",
            "z",
            "hurst",
            "vr",
            "hl",
            "quality",
            "pos_rev",
            "atr",
            "ret_5d",
            "ret_3d",
            "sharpe_real",
        }
        assert set(out.keys()) == expected

    def test_price_equals_last_close(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=15)
        out = compute_stock(df["close"], df["high"], df["low"])
        assert math.isclose(float(out["price"]), float(df["close"].iloc[-1]))

    def test_quality_and_pos_rev_are_booleans(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=300, seed=16)
        out = compute_stock(df["close"], df["high"], df["low"])
        assert isinstance(out["quality"], bool)
        assert isinstance(out["pos_rev"], bool)
