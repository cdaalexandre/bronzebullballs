# bronzebullballs

CLI Python que combina validacao walk-forward (Monte Carlo Permutation Method de Aronson) e screening diario do S&P 500 para gerar picks long-only com sizing por volatility-targeting (Carver) e gatilhos de venda. Cada execucao valida estatisticamente que o ranking por adjusted-slope (Clenow) prediz retornos antes de emitir os picks de hoje.

> **Status:** em construcao. FASE A (package skeleton) e FASE B (config + git) concluidas. Logica de dominio ainda nao migrada do monolito de referencia `corpus_master_v8.py`.

## Stack

Python 3.11+, `yahooquery`, `yfinance`, `pandas`, `numpy`, `scipy`, `statsmodels`, `colorama`. Build com `hatchling`. Lint/format com `ruff`, type-check com `mypy`, testes com `pytest`.

## CLI (alvo)

```text
bronzebullballs validate     # PHASE 1: walk-forward historico (~5-10 min)
bronzebullballs screen       # PHASE 2: picks de hoje (~1-2 min)
bronzebullballs all          # default: validate + screen (~7-12 min)
```

## Layout

```text
src/bronzebullballs/
  adapters/        integracoes externas (yahooquery, yfinance, universe)
  domain/          logica pura (indicators, scores, sizing, walk-forward)
  service_layer/   pipelines (validation, screening)
  entrypoints/     CLI dispatch
  report/          formatacao ANSI para terminal
  log.py
tests/
  unit/            dominio puro com fixtures sinteticas
  integration/     adapters + service layer + CLI end-to-end
```

Arquitetura hexagonal: o dominio nao importa adapters. Protocolos em `adapters/protocols.py` definem as interfaces; adapters concretos sao injetados pela entrypoint.

## Fundamentacao teorica

- **Clenow** -- *Stocks on the Move* / *Trading Evolved* (adjusted slope, sector cap, ATR risk-parity)
- **Carver** -- *Systematic Trading* / *Leveraged Trading* (EWMAC, vol-targeting, diversification multiplier)
- **Chan** -- *Algorithmic Trading* (Hurst, variance ratio, half-life, walk-forward)
- **Aronson** -- *Evidence-Based Technical Analysis* (Monte Carlo Permutation Method)
- **Brown** -- *Technical Analysis for the Trading Professional* (positive reversal)

## Desenvolvimento local

```powershell
# Setup (FASE C pendente: codigo de entrypoint nao existe ainda)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Testes
pytest

# Lint + format
ruff check .
ruff format .

# Type check
mypy src
```

## Licenca

A definir.
