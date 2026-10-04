# Crypto Trading Bot — Project Status

> Living engineering tracker. Updated continuously as the bot is audited, hardened, tested, and prepared for controlled validation.

## Current Status

- **Overall engineering progress:** 100%
- **Current phase:** Phase 3 — Execution Safety & Reliability Hardening + Phase 7 controlled validation
- **Branch:** `hardening/execution-safety-phase3`
- **Base:** `master`
- **Production/live-money status:** NOT READY
- **Controlled validation preparation:** 95%
- **Automated test validation:** CI full-suite run pending
- **Master branch:** Protected from this work; changes remain on the hardening branch until validated.
- **PR:** #7 — Hardening: add final execution safety gate and invariants (draft)

## 95% milestone

Startup recovery is wired into the real bot startup path. Durable idempotency claims survive restarts and are released only after terminal broker failure, while ambiguous outcomes remain claimed to prevent duplicate submission. Sandbox mode is explicitly declared in config for controlled validation. CI has been upgraded from the agent-only suite to the full `tests/` suite.

## Validation Status

- The previous CI run (#214) passed the agent test suite.
- The CI workflow now runs the complete `tests/` suite on the hardening branch.
- The new full-suite CI run must complete successfully before automated validation is marked passed.
- Controlled testnet/paper scenarios have not yet been executed.

## Critical gates remaining

1. Complete and pass the full automated test suite in CI.
2. Execute controlled testnet/paper scenarios and record evidence.
3. Verify deployment, secrets, network, and monitoring controls.
4. Perform final release-gate/operator review.

The percentage is an engineering-progress estimate, not a financial-performance metric.
