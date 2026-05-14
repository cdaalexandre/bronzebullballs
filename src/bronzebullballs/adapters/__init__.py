"""Adapters layer - external integrations (yahooquery, yfinance, universe sources).

Concrete implementations of protocols defined in `adapters.protocols`. Depends on
third-party SDKs and network. Domain and service_layer never import from here
directly; they receive instances via dependency injection from entrypoints.
"""