# QTS Refactor Checklist

Use this file for broad cleanup or review passes where multiple files may violate the repo guide.

## Architecture

- No imports from `omega`, `qsconnect`, `qsresearch`, or `qsautomate`
- Vendor libraries isolated to adapter modules
- No inline `if asset_type == ...` business logic
- Asset type derived via `AssetType.from_symbol(symbol)`
- Concrete dependencies injected, not instantiated by consumers
- Strategy public seam is `BaseStrategy.generate_signals(data)`
- Engine public seam is `BaseEngine.run(...)`
- Flow and config resolution stay registry-backed and YAML-driven

## Plugins And Registries

- New broker, engine, strategy, feature, or source extends the correct `Base*` class
- New pluggable component is registered with a lowercase key
- YAML references the registry key, not a concrete import path
- Feature plugins append columns only and preserve the incoming frame

## Boundaries And Types

- Layer imports respect the `core -> data -> research/execution -> orchestration` boundaries
- Public functions and methods are fully annotated
- `pl.DataFrame` is used for tabular data
- Monetary values use `Decimal`
- `date` is used for calendar bars and `datetime` only for tick-level timestamps

## Runtime And Secrets

- Verification commands use `QTradeSystematic/.venv`
- Credentials are loaded from env vars at adapter `connect()` time
- No secrets are embedded in code or YAML

## Tests

- Unit tests mock only at the ABC boundary
- Integration tests use real in-memory storage instead of mocks
- Paper tests target paper or testnet accounts only
- Changed seams have matching tests or an explicit gap called out
