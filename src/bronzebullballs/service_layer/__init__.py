"""Service layer - orchestrates adapters + domain into use-case pipelines.

`validation_pipeline` runs PHASE 1 (walk-forward). `screening_pipeline` runs
PHASE 2 (today's picks). Both receive adapters via injection and call domain
functions for all calculations.
"""
