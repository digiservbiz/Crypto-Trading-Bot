# Skill: trader-cloud-backtest

## Overview
Cloud-dispatched heavy backtests for compute-intensive strategy validation. Offloads walk-forward validation and parameter sweeps to cloud compute for large datasets, extended periods, or grid search over large parameter spaces.

## Trigger
```
trader backtest <strategy> --cloud [options]
```

## Description
The `trader-cloud-backtest` skill extends the base `trader-backtest` skill by dispatching computation to cloud infrastructure. This is the recommended approach for:

- Backtests spanning > 2 years of data
- Parameter sweeps over large grids (> 100 combinations)
- Multi-symbol simultaneous backtests
- Production-grade validation with increased Monte Carlo runs (10,000+)

## When to Use Cloud Backtests

| Scenario | Local | Cloud |
|---|---|---|
| Quick validation (< 1 year) | YES | - |
| Production release validation | - | YES |
| Parameter sweep > 50 combos | - | YES |
| Multi-symbol portfolio backtest | - | YES |
| Daily strategy revalidation | YES | - |

## Cloud Configuration

Add to `config.yaml`:

```yaml
cloud_backtest:
  enabled: true
  provider: aws  # aws | gcp | azure
  region: us-east-1
  instance_type: c5.2xlarge
  max_parallel_jobs: 4
  result_bucket: my-backtest-results
  timeout_minutes: 60
```

## Dispatch Protocol

1. Serialize strategy config and data reference
2. Package as job manifest (JSON)
3. Dispatch to cloud queue (SQS / Pub/Sub / Service Bus)
4. Poll for completion (async)
5. Download results when complete
6. Validate and store artifact locally

## Job Manifest Format

```json
{
  "job_id": "backtest-2024-01-01-btc-momentum",
  "strategy_config": {
    "strategy_id": "momentum-btc-v3",
    "strategy_type": "momentum",
    "symbol": "BTC/USDT"
  },
  "data_reference": {
    "source": "s3://my-data/BTC_USDT_5m.parquet",
    "period": "2021-01-01/2024-01-01"
  },
  "backtest_params": {
    "train_months": 6,
    "test_months": 1,
    "monte_carlo_runs": 10000,
    "enable_parameter_sweep": true,
    "parameter_grid": {
      "lookback": [10, 15, 20, 25, 30],
      "z_threshold": [1.0, 1.5, 2.0, 2.5]
    }
  }
}
```

## Output

Same `SignedBacktestArtifact` format as local backtest, with additional fields:

```json
{
  "...": "standard artifact fields...",
  "cloud_job_id": "backtest-2024-01-01-btc-momentum",
  "compute_time_seconds": 847,
  "instance_type": "c5.2xlarge",
  "parameter_sweep_results": [
    {"lookback": 20, "z_threshold": 1.5, "sharpe": 1.94, "passed": true},
    ...
  ]
}
```

## Usage Examples

```bash
# Cloud dispatch with default settings
trader backtest my-strategy --symbol BTC/USDT --cloud

# Extended Monte Carlo
trader backtest my-strategy --symbol BTC/USDT --cloud --mc-runs 10000

# Parameter sweep
trader backtest my-strategy --cloud --param-sweep

# Long period backtest
trader backtest my-strategy --symbol BTC/USDT --cloud --period 2020-01-01/2024-01-01

# Check job status
trader backtest --status job-id-here

# Download completed job
trader backtest --download job-id-here
```

## Cost Estimation

Use `--estimate` to preview cloud cost before dispatching:

```bash
trader backtest my-strategy --cloud --estimate
# Estimated cost: $2.40 (c5.2xlarge × 2 hours)
```

## Memory Namespace
`trading-backtests` — same namespace as local backtests. Cloud artifacts are stored with the same structure.
