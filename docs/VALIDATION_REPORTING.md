# Validation Reporting

The validation evidence recorder writes append-only JSONL records. This report
layer reads those records and summarizes their current status.

## Release rule

A validation set remains blocked if any recorded scenario is FAIL, BLOCKED,
or NOT_RUN. Only executed scenarios with observed evidence should be marked
PASS.

The report is intentionally read-only and does not interact with the exchange.

## Example

After controlled validation, point the reporting helper at:
data/state/validation-evidence.jsonl

The resulting summary can be used by an operator or release process to identify
which scenarios still require execution.

No API keys, exchange secrets, or account identifiers should be stored in the
evidence file.
