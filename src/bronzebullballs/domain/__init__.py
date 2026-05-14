"""Domain layer - pure business logic (indicators, scores, sizing, walk-forward).

No I/O, no network, no clock access. All inputs are parameters; all outputs are
deterministic given fixed seeds. Modules here are the only place where the
financial mathematics of bronzebullballs lives.
"""
