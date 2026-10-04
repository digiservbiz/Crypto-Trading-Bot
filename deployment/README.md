# VPS Deployment

## Safety rule

This repository is deployment-ready for controlled dry-run/testnet preparation, not funded live trading. Keep DRY_RUN=true until testnet validation and the final release gate pass.

## Prepare

1. Install Docker Engine and the Compose plugin on the VPS.
2. Clone the hardening branch.
3. Copy `.env.example` to `.env`.
4. Put testnet credentials in `.env` only when ready; never commit `.env`.
5. Keep exchange API withdrawals disabled.
6. Keep dashboard and metrics bound to localhost; use an authenticated reverse proxy or VPN for remote access.
7. Create persistent directories: `mkdir -p data/state data/memory models`.
8. Build: `docker compose build`.
9. Start dry-run: `docker compose up -d bot dashboard prometheus`.
10. Check: `docker compose ps` and `docker compose logs --tail=200 bot`.
11. Confirm `data/state` persists across container recreation and the kill-switch sentinel is writable.
12. Run the repository test suite before enabling testnet order flow.

## Testnet transition

Only after dry-run is stable: set sandbox/testnet mode in `config.yaml`, provide testnet credentials through `.env`, keep withdrawals disabled, verify symbol and market mode, execute `docs/TESTNET_RUNBOOK.md`, and record evidence.

## Controlled stop

Preferred stop: `docker compose stop bot`

Emergency entry block: `touch data/state/bot.stop`

Then inspect: `docker compose logs --tail=200 bot`

Do not remove the stop sentinel until the operator confirms the incident is understood and the release gate permits resumption.
