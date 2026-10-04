"""Durable execution ledger backed by SQLite.

This module provides crash-safe claim persistence for execution keys. It is
independent from strategy logic and can be used by a broker boundary after the
final safety gate has approved an immutable intent.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


class PersistentExecutionLedger:
    def __init__(self, path: str = "data/state/execution-ledger.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS execution_keys "
                "(execution_key TEXT PRIMARY KEY, claimed_at REAL NOT NULL)"
            )

    def claim(self, execution_key: str, claimed_at: float) -> bool:
        key = str(execution_key or "").strip()
        if not key:
            raise ValueError("execution_key is required")
        try:
            with sqlite3.connect(self.path) as db:
                db.execute(
                    "INSERT INTO execution_keys(execution_key, claimed_at) VALUES (?, ?)",
                    (key, float(claimed_at)),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def release(self, execution_key: str) -> None:
        """Explicitly release a claim only for a terminal broker failure."""
        key = str(execution_key or "").strip()
        if not key:
            return
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM execution_keys WHERE execution_key = ?", (key,))

    def contains(self, execution_key: str) -> bool:
        key = str(execution_key or "").strip()
        if not key:
            return False
        with sqlite3.connect(self.path) as db:
            row = db.execute(
                "SELECT 1 FROM execution_keys WHERE execution_key = ? LIMIT 1",
                (key,),
            ).fetchone()
        return row is not None
