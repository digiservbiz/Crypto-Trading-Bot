# Crypto Trading Bot — Project Status

> Living engineering tracker. Updated continuously as the bot is audited, hardened, tested, and prepared for controlled validation.

## Current Status

- **Overall engineering progress:** 100%
- **Current phase:** Phase 3 — Execution Safety & Reliability Hardening + Phase 7 controlled validation
- **Branch:** `hardening/execution-safety-phase3`
- **Base:** `master`
- **Production/live-money status:** NOT READY
- **Controlled validation preparation:** 95%
- **Automated test validation:** PASS — latest full-suite CI run #225 succeeded on commit `1ce5051849abd8b08a70400c29ae9fdd1c228bca`
- **Deployment readiness:** VPS deployment artifacts added; target-environment verification still required
- **Master branch:** Protected from this work; changes remain on the hardening branch until validated.
- **PR:** #7 — Hardening: add final execution safety gate and invariants (draft)

## Validated engineering milestone

The complete automated test suite has now passed in CI, including the durable execution lifecycle tests added for cross-restart unknown outcomes and terminal broker-failure claim release. The live entry boundary, close reconciliation, startup recovery, kill switch, and persistent duplicate protection are implemented on the hardening branch.

## Deployment preparation

The repository now includes:
- non-root Docker runtime configuration;
- a Compose stack for bot, dashboard, and private Prometheus;
- an environment-variable template that keeps secrets out of Git;
- a Docker build context exclusion file;
- a VPS deployment runbook;
- persistent state mounts for execution ledger, kill switch, and bot state;
- localhost-only dashboard/metrics bindings by default.

These artifacts prepare the repository for VPS deployment, but they are not evidence that a target VPS has been secured or validated.

## Validation Status

- Full automated test suite: **PASSED in CI #225**.
- Controlled testnet/paper scenarios: **NOT YET EXECUTED**.
- Target VPS deployment verification: **NOT YET EXECUTED**.
- Funded live trading: **NOT READY**.

## Critical gates remaining

1. Validate the container stack on the target VPS.
2. Verify secrets, firewall/network exposure, persistent state, monitoring, and restart behavior.
3. Execute controlled testnet/paper scenarios and record evidence.
4. Perform final release-gate/operator review.

The percentage is an engineering-progress estimate, not a financial-performance metric.
