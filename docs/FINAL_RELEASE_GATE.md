# Final Release Gate

## Current engineering gate

The hardening branch contains the safety primitives and controlled-release
documentation needed to proceed toward validation.

## Blocking gates before 100%

These are evidence gates, not documentation tasks:

1. Final safety gate is called by the actual entry execution path.
2. Approved position size cannot be mutated after risk approval.
3. Submitted orders are reconciled against authoritative exchange state.
4. Local positions are rebuilt from authoritative exchange state on startup.
5. Duplicate execution is prevented across process restarts with durable storage.
6. Spot/futures semantics are explicitly configured and tested.
7. Close orders are reconciled before local positions are cleared.
8. Full automated test suite executes successfully in a clean environment.
9. Testnet/paper end-to-end scenarios are executed and evidence recorded.
10. Deployment/secrets/network/monitoring controls are verified in the target
    environment.
11. Final operator review confirms the release evidence.

## Release interpretation

Until every blocking gate above has observed evidence, the project must not be
described as ready for funded live trading.

The branch may contain implementation primitives before their integration is
verified. Presence of a module or test does not equal successful production
integration.
