"""Unit tests for ``bronzebullballs.domain.regime``."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bronzebullballs.domain.regime import quality_filter, regime_on


class TestQualityFilter:
    def test_uptrending_smooth_series_passes(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=1, drift=0.001, vol=0.005)
        assert quality_filter(df["close"]) is True

    def test_insufficient_data_fails(self) -> None:
        close = pd.Series(np.linspace(100, 110, 50))
        assert quality_filter(close, ma_p=100) is False

    def test_large_gap_fails(self, synthetic_ohlcv) -> None:
        df = synthetic_ohlcv(n_days=200, seed=2)
        s = df["close"].copy()
        # Insert a 20 % overnight gap within the lookback window.
        s.iloc[-10] = s.iloc[-11] * 1.20
        s.iloc[-9:] = s.iloc[-9:] * 1.20
        assert quality_filter(s, gap_thresh=0.15) is False

    def test_below_ma_fails(self) -> None:
        # Series falling steadily after a long flat plateau: last close is
        # below MA-100, regardless of gap status.
        s = pd.Series([100.0] * 100 + list(np.linspace(99.5, 90.0, 50)))
        assert quality_filter(s) is False


class TestRegimeOn:
    def test_spy_above_ma_returns_true(self) -> None:
        s = pd.Series(np.linspace(300.0, 400.0, 300))
        assert regime_on(s, ma_p=200) is True

    def test_spy_below_ma_returns_false(self) -> None:
        s = pd.Series(np.linspace(400.0, 300.0, 300))
        assert regime_on(s, ma_p=200) is False

    def test_insufficient_data_returns_false(self) -> None:
        s = pd.Series(np.linspace(300.0, 400.0, 100))
        assert regime_on(s, ma_p=200) is False
