from scripts.persistent_execution_ledger import PersistentExecutionLedger


def test_persistent_ledger_survives_new_instance(tmp_path):
    path = str(tmp_path / "ledger.sqlite3")
    first = PersistentExecutionLedger(path)
    assert first.claim("abc", 100.0) is True

    second = PersistentExecutionLedger(path)
    assert second.contains("abc") is True
    assert second.claim("abc", 101.0) is False


def test_persistent_ledger_rejects_empty_key(tmp_path):
    ledger = PersistentExecutionLedger(str(tmp_path / "ledger.sqlite3"))
    try:
        ledger.claim("", 100.0)
    except ValueError:
        pass
    else:
        raise AssertionError("empty execution key must be rejected")
