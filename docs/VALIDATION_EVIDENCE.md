# Validation Evidence Template

Use one row per controlled scenario. Do not record API keys, secrets, private
credentials, or account identifiers.

| Scenario | Commit SHA | Environment | Expected result | Observed result | Evidence | Operator | Status |
|---|---|---|---|---|---|---|---|
| Approved trade passes final safety gate | | | | | | | |
| Rejected trade is blocked | | | | | | | |
| Post-approval size mutation is blocked | | | | | | | |
| Stale approval is blocked | | | | | | | |
| Full fill reconciles correctly | | | | | | | |
| Partial fill remains unresolved | | | | | | | |
| Cancel/reject does not open a position | | | | | | | |
| Restart recovers exchange positions | | | | | | | |
| Failed recovery blocks trading | | | | | | | |
| Kill switch blocks new entries | | | | | | | |
| Duplicate execution key is rejected | | | | | | | |
| Broker timeout leaves order unresolved | | | | | | | |
| Testnet end-to-end cycle | | | | | | | |

## Release rule

Only mark a scenario PASS after it has been executed in the stated
environment and the evidence is reproducible.
