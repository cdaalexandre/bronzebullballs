"""End-to-end test for the `--version` flag of the bronzebullballs CLI.

This is the bootstrap acceptance test (Custom Instructions section 11.1,
FASE D): if it passes, the package skeleton, pyproject.toml `[project.scripts]`
wiring, and editable install are all correct.
"""

from __future__ import annotations

import pytest

from bronzebullballs.entrypoints.cli import main


def test_version_flag_exits_zero_and_prints_version(capsys: pytest.CaptureFixture[str]) -> None:
    """`bronzebullballs --version` must exit 0 and print 'bronzebullballs X.Y.Z'."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    # argparse `action="version"` raises SystemExit(0)
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert out.startswith("bronzebullballs ")
    # Format is 'bronzebullballs <version>\n' -- version must be non-empty.
    version_str = out.removeprefix("bronzebullballs ").strip()
    assert len(version_str) > 0
    assert version_str != "unknown", (
        "Package version reads 'unknown' -- did you forget `pip install -e .`?"
    )