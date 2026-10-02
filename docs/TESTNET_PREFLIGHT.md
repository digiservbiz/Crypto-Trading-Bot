# Testnet Preflight

This is a read-only gate before any controlled order scenario.

## Required conditions

1. Exchange sandbox/testnet mode is explicitly enabled.
2. Credentials, if required, are supplied only through environment/secret storage.
3. Withdrawals are disabled on the exchange API key.
4. The configured market mode is explicitly spot or futures.
5. Configured symbols are validated with exchange load_markets().
6. Every scenario symbol is active and compatible with the selected market mode.
7. No order is submitted during preflight.

## Evidence

Record commit SHA, exchange sandbox, market mode, configured symbols, eligible symbols, timestamp, and operator. Never record API keys, secrets, or account identifiers.

A successful preflight does not prove order execution, reconciliation, restart recovery, or duplicate protection. Those require separate executed scenarios.
