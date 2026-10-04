"""Testnet/paper execution runbook.

Use this sequence to validate the bot without funded credentials. No step
assumes that an order submission equals a fill.
"""

# Controlled validation sequence

1. Set DRY_RUN=true and verify the bot starts without exchange credentials.
2. Configure sandbox/testnet credentials with withdrawals disabled.
3. Confirm the intended market mode is spot or futures; do not mix semantics.
4. Validate one-symbol market data and risk calculations.
5. Exercise approved and rejected risk decisions through the final safety gate.
6. Simulate filled, partial, cancelled, rejected, and unknown order outcomes.
7. Restart the process and verify positions are recovered from the exchange.
8. Activate the kill switch and verify new entries are blocked.
9. Submit the same execution key twice and verify the second claim is rejected.
10. Exercise broker timeout/error handling and leave ambiguous orders unresolved
    until authoritative state is known.
11. Record evidence for every gate before considering progression toward funded use.

## Release evidence

Record timestamp, commit SHA, environment, exchange sandbox, symbol, market
mode, scenario, expected result, observed result, and operator sign-off.
