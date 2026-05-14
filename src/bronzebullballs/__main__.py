"""Allow `python -m bronzebullballs` as an alias for the installed script."""

from bronzebullballs.entrypoints.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
