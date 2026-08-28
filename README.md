# Undertow

*Friction-aware reinforcement learning for concentrated liquidity provision.*

<!--
Scaffolding placeholder. README will be filled in as the data pipeline
(`undertow.data`) and AMM simulator (`undertow.sim`) take shape.
-->

## Layout

```
undertow/
├── data/   # on-chain Uniswap V3 ingestion & processing pipeline
└── sim/    # AMM / concentrated-liquidity simulator, RL environment
```

## Setup

```bash
uv sync
```

## Status

- [x] Data pipeline scaffolding (T00)
- [ ] AMM simulator scaffolding

## Running network-flavoured tests

Most tests run entirely against fixtures and never touch the network. The handful
marked `@pytest.mark.network` hit live endpoints and are **skipped by default**.
To run them, pass the `--run-network` flag:

```bash
uv run pytest            # fixtures only; network-marked tests skipped
uv run pytest --run-network   # also runs the @pytest.mark.network tests
```

This flag is wired up in `tests/conftest.py` (`pytest_addoption` +
`pytest_collection_modifyitems`); it is not part of pytest's `addopts`, so it
never collides with a later `-m` selector. Later pipeline tasks use it to run
their live fetcher checks.