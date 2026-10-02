# Controlled Deployment Checklist

This checklist is a release gate for the Crypto Trading Bot. Completing code milestones does not authorize funded trading.

## Before any exchange credentials are supplied

- Keep `DRY_RUN=true`.
- Use exchange sandbox/testnet credentials first.
- Keep API withdrawal permissions disabled.
- Restrict API keys to required trading permissions and IPs where supported.
- Put secrets in the deployment secret store/environment, never in Git.
- Do not expose Streamlit, Prometheus, or Grafana directly to the public Internet.

## Before testnet/paper validation

- Verify exchange mode (spot vs derivatives) and symbol semantics.
- Verify buy/sell behavior against the intended long/short model.
- Verify reconciliation for filled, partial, cancelled, rejected, and unknown outcomes.
- Verify restart recovery against authoritative exchange positions.
- Verify the kill switch blocks new entries.
- Verify duplicate execution attempts are detected by the execution ledger.
- Exercise timeout and exchange-error scenarios.

## Before funded deployment

All of the following must be evidenced in a working CI/test environment:

- Full automated test suite passes.
- Execution safety gate is integrated at the final order boundary.
- No post-approval component can increase approved position size.
- Order lifecycle reconciliation is integrated.
- Position recovery runs before trading resumes.
- Idempotency/duplicate-order protection is integrated and persistent.
- Emergency stop behavior is validated.
- Deployment secrets and network exposure are hardened.
- Monitoring and alerting are verified.
- A human operator has reviewed the release.

**Until every funded-deployment gate above is evidenced, the bot remains NOT READY for funded live trading.**
