# Crypto Trading Bot — Project Status

> Living engineering tracker. Updated continuously as the bot is audited, hardened, tested, and prepared for controlled validation.

## Current Status

- **Overall engineering progress:** 100%
- **Current phase:** Phase 3 — Execution Safety & Reliability Hardening + Phase 7 controlled validation preparation
- **Branch:** `hardening/execution-safety-phase3`
- **Base:** `master`
- **Production/live-money status:** NOT READY
- **Controlled validation preparation:** 90%
- **Master branch:** Protected from this work; changes remain on the hardening branch until validated.
- **PR:** #7 — Hardening: add final execution safety gate and invariants (draft)

## 90% milestone

Startup recovery is now wired into the real bot startup path. Live mode fails closed when authoritative exchange positions cannot be recovered, and recovered positions rebuild conservative local state before trading resumes. The exchange wrapper now exposes authoritative position recovery as well as order recovery.

## Validation Status

- New tests have been authored.
- Local execution of the test suite has **not** yet completed because the engineering runtime could not resolve GitHub network access for repository cloning.
- Do not treat the new tests as passed until executed in a working environment/CI.

## Critical gates remaining

1. Preserve durable duplicate protection through ambiguous outcomes and restarts.
2. Execute the full automated test suite in CI/working environment.
3. Execute controlled testnet/paper scenarios and record evidence.
4. Verify deployment, secrets, network, and monitoring controls.

The percentage is an engineering-progress estimate, not a financial-performance metric.
