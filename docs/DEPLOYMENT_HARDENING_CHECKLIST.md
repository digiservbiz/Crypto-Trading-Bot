# Deployment Hardening Checklist

This checklist is intentionally operational. It does not claim that the repository
is production-ready until the controls are actually evidenced.

## Container/runtime

- Run the bot as a non-root user.
- Keep runtime state and credentials outside the image.
- Use explicit image versions rather than floating latest tags.
- Limit container memory/CPU and restart behavior deliberately.
- Add health checks for bot freshness and service availability.
- Keep Streamlit, Prometheus, and Grafana behind an authenticated/restricted network boundary.
- Do not publish exchange credentials in compose files, images, logs, or repository files.

## Secrets

- Supply exchange keys through the deployment secret store/environment.
- Disable withdrawals on trading API keys.
- Grant only required permissions.
- Restrict source IPs where supported.
- Rotate keys after testnet validation and after any suspected exposure.
- Confirm no secret-like values are present in tracked files before release.

## Dependencies

- Review direct dependencies and transitive vulnerabilities.
- Pin production dependency versions after compatibility testing.
- Avoid unreviewed runtime git dependencies.
- Rebuild from a clean environment and record the resulting lock/version set.

## Network and monitoring

- Restrict inbound ports with firewall/security-group rules.
- Require authentication for dashboards.
- Monitor bot heartbeat, exchange errors, unresolved orders, kill-switch state,
  position reconciliation failures, and repeated execution attempts.
- Preserve logs needed to reconstruct an order lifecycle.

## Release evidence

For each control record: commit SHA, environment, date/time, operator, evidence,
and pass/fail result. A checklist item is not considered passed merely because
the configuration exists in the repository.
