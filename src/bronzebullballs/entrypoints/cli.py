"""Command-line entrypoint for bronzebullballs.

Parses argv, configures logging, and dispatches to service-layer pipelines.
Today only `--version` is fully wired; subcommands (`validate`, `screen`,
`all`) print a not-implemented notice and exit with code 2 -- they wire up
once `service_layer.*_pipeline` exist (FASE E+).
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from typing import cast

from bronzebullballs.log import get_logger, setup_logging

logger = get_logger(__name__)


def _get_version() -> str:
    """Return installed package version, or 'unknown' if not installed."""
    try:
        return version("bronzebullballs")
    except PackageNotFoundError:
        return "unknown"


def _cmd_validate(args: argparse.Namespace) -> int:
    print("validate: not yet implemented (FASE E)")
    return 2


def _cmd_screen(args: argparse.Namespace) -> int:
    print("screen: not yet implemented (FASE E)")
    return 2


def _cmd_all(args: argparse.Namespace) -> int:
    print("all: not yet implemented (FASE E)")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bronzebullballs",
        description=(
            "Walk-forward validation (Aronson MCPM) + S&P 500 screening "
            "(Clenow adjusted slope, Carver vol-target)."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"bronzebullballs {_get_version()}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Increase log verbosity to DEBUG.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console INFO; show WARNING and above only.",
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p_validate = sub.add_parser(
        "validate",
        help="PHASE 1: walk-forward validation over the full S&P 500.",
    )
    p_validate.set_defaults(func=_cmd_validate)

    p_screen = sub.add_parser(
        "screen",
        help="PHASE 2: today's picks (top 25 + 3 profiles + 3-stock portfolio).",
    )
    p_screen.set_defaults(func=_cmd_screen)

    p_all = sub.add_parser(
        "all",
        help="Run validate then screen (default when no subcommand given).",
    )
    p_all.set_defaults(func=_cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns exit code (0 = success).

    Args:
        argv: Optional argument list (for testing). If None, uses sys.argv[1:].

    Returns:
        Process exit code: 0 on success, 2 on not-yet-implemented subcommand.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    setup_logging(
        level="DEBUG" if args.debug else "INFO",
        quiet=args.quiet,
    )

    if not hasattr(args, "func"):
        # No subcommand: print help and exit non-zero (mimics CLI conventions).
        parser.print_help()
        return 0

    return cast(int, args.func(args))
