"""Entrypoints - CLI parsing and dispatch (no business logic).

`cli.main` is wired into pyproject.toml under `[project.scripts]`. This layer
only parses argv, sets up logging, builds adapters, and hands off to the
service layer.
"""