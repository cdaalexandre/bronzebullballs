"""Report layer - terminal output formatting (ANSI colors, tables, macro state).

Consumes data already computed by the service layer and renders it for the
human reader. Never calculates anything; never calls adapters; never imports
from `domain` directly (receives pre-computed values).
"""
