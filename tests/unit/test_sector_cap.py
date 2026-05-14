"""Unit tests for ``bronzebullballs.domain.sector_cap``."""

from __future__ import annotations

from bronzebullballs.domain.sector_cap import (
    SECTOR_CAP_MAX_PER_SECTOR,
    apply_sector_cap_rows,
    apply_sector_cap_tickers,
)


class TestApplySectorCapTickers:
    def test_no_overflow_passes_through_unchanged(self) -> None:
        ranked = ["A", "B", "C", "D"]
        secs = {"A": "Tech", "B": "Health", "C": "Energy", "D": "Tech"}
        out = apply_sector_cap_tickers(ranked, secs, max_per_sector=3, n_top=25)
        assert out == ["A", "B", "C", "D"]

    def test_cap_kicks_in_skipping_excess(self) -> None:
        ranked = ["A", "B", "C", "D", "E"]
        secs = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Tech", "E": "Health"}
        out = apply_sector_cap_tickers(ranked, secs, max_per_sector=2, n_top=25)
        # D (4th Tech) skipped, E (Health) included.
        assert out == ["A", "B", "E"]

    def test_n_top_truncates_after_cap(self) -> None:
        ranked = ["A", "B", "C", "D"]
        secs = {"A": "Tech", "B": "Health", "C": "Energy", "D": "Finance"}
        out = apply_sector_cap_tickers(ranked, secs, max_per_sector=3, n_top=2)
        assert out == ["A", "B"]

    def test_missing_sector_treated_as_unknown(self) -> None:
        ranked = ["A", "B", "C", "D"]
        secs = {"A": None, "B": "Tech"}  # C, D absent => Unknown
        out = apply_sector_cap_tickers(ranked, secs, max_per_sector=2, n_top=25)
        # A and C are Unknown (max=2), D is Unknown (skipped), B is Tech.
        assert out == ["A", "B", "C"]


class TestApplySectorCapRows:
    def test_keeps_within_cap_partitions_excess(self) -> None:
        rows = [
            {"tk": "A", "fund": {"sector": "Tech"}},
            {"tk": "B", "fund": {"sector": "Tech"}},
            {"tk": "C", "fund": {"sector": "Tech"}},
            {"tk": "D", "fund": {"sector": "Tech"}},  # excess
            {"tk": "E", "fund": {"sector": "Health"}},
        ]
        capped, excluded, counts = apply_sector_cap_rows(rows, max_per_sector=3)
        assert [r["tk"] for r in capped] == ["A", "B", "C", "E"]
        assert [r["tk"] for r in excluded] == ["D"]
        assert counts == {"Tech": 3, "Health": 1}

    def test_default_cap_is_three(self) -> None:
        assert SECTOR_CAP_MAX_PER_SECTOR == 3
        rows = [{"tk": f"T{i}", "fund": {"sector": "Tech"}} for i in range(5)]
        capped, excluded, _ = apply_sector_cap_rows(rows)
        assert len(capped) == 3
        assert len(excluded) == 2

    def test_missing_sector_bucketed_as_unknown(self) -> None:
        rows = [
            {"tk": "A", "fund": {}},
            {"tk": "B", "fund": {"sector": None}},
            {"tk": "C"},  # no `fund` key at all
        ]
        capped, excluded, counts = apply_sector_cap_rows(rows, max_per_sector=2)
        # First two Unknowns kept, third skipped.
        assert [r["tk"] for r in capped] == ["A", "B"]
        assert [r["tk"] for r in excluded] == ["C"]
        assert counts == {"Unknown": 2}
