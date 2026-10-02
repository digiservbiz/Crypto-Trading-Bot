from scripts.execution_ledger import ExecutionLedger


def test_claim_is_idempotent():
    ledger = ExecutionLedger()
    assert ledger.claim("abc") is True
    assert ledger.claim("abc") is False
    assert ledger.contains("abc") is True


def test_release_allows_reclaim():
    ledger = ExecutionLedger()
    assert ledger.claim("abc") is True
    ledger.release("abc")
    assert ledger.claim("abc") is True


def test_empty_key_rejected():
    ledger = ExecutionLedger()
    try:
        ledger.claim("")
    except ValueError:
        return
    assert False
