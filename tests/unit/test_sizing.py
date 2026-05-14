"""Unit tests for ``bronzebullballs.domain.sizing``."""

from __future__ import annotations

import math

from bronzebullballs.domain.sizing import (
    DEFAULT_RISK_FACTOR,
    carver_position_size,
    clenow_risk_parity_size,
)


class TestClenowRiskParitySize:
    def test_basic_sizing(self) -> None:
        # capital * risk_factor = 100_000 * 0.001 = 100 daily $ at risk.
        # atr = 2 $/share -> 50 shares.
        n, notional = clenow_risk_parity_size(price=50.0, atr_dollar=2.0, capital=100_000)
        assert n == 50
        assert math.isclose(notional, 50 * 50.0)

    def test_higher_atr_yields_fewer_shares(self) -> None:
        n_calm, _ = clenow_risk_parity_size(price=50.0, atr_dollar=1.0, capital=100_000)
        n_vola, _ = clenow_risk_parity_size(price=50.0, atr_dollar=5.0, capital=100_000)
        assert n_calm > n_vola

    def test_zero_atr_returns_no_position(self) -> None:
        n, notional = clenow_risk_parity_size(price=50.0, atr_dollar=0.0, capital=100_000)
        assert (n, notional) == (0, 0.0)

    def test_zero_price_returns_no_position(self) -> None:
        n, notional = clenow_risk_parity_size(price=0.0, atr_dollar=1.0, capital=100_000)
        assert (n, notional) == (0, 0.0)

    def test_minimum_one_share_floor(self) -> None:
        # Tiny capital, huge ATR -> integer would round to 0, but we floor at 1.
        n, _ = clenow_risk_parity_size(price=10.0, atr_dollar=1000.0, capital=1000.0)
        assert n == 1


class TestCarverPositionSize:
    def test_basic_sizing_under_cap(self) -> None:
        n, pct, notional, daily_risk = carver_position_size(
            price=100.0, atr_pct=0.02, capital=100_000
        )
        assert n > 0
        assert 0.0 < pct <= 0.35
        assert math.isclose(notional, n * 100.0)
        assert math.isclose(daily_risk, n * 100.0 * 0.02)

    def test_cap_kicks_in_for_low_vol_stock(self) -> None:
        # Very low ATR % -> raw_pct would exceed the cap.
        _, pct, _, _ = carver_position_size(
            price=100.0, atr_pct=0.001, capital=100_000, max_pct_single=0.20
        )
        assert math.isclose(pct, 0.20)

    def test_zero_atr_returns_no_position(self) -> None:
        out = carver_position_size(price=100.0, atr_pct=0.0, capital=100_000)
        assert out == (0, 0.0, 0.0, 0.0)

    def test_higher_vol_target_yields_more_shares(self) -> None:
        # Use a high-vol stock so cap doesn't bind.
        a = carver_position_size(price=100.0, atr_pct=0.05, capital=100_000, vol_target_annual=0.10)
        b = carver_position_size(price=100.0, atr_pct=0.05, capital=100_000, vol_target_annual=0.30)
        assert b[0] > a[0]


class TestConstants:
    def test_default_risk_factor_is_ten_bps(self) -> None:
        # Clenow Cap. 11: 10 bps = 0.001.
        assert DEFAULT_RISK_FACTOR == 0.001
