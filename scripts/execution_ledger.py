"""Persistent in-process execution ledger primitives.

The ledger records execution keys before/after broker work so callers can
detect duplicate requests without changing strategy decisions. It is deliberately
small; durable storage integration belongs to the broker boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ExecutionLedger:
    _keys: set[str] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock)

    def claim(self, execution_key: str) -> bool:
        """Atomically claim a key. False means it was already claimed."""
        key = str(execution_key or "").strip()
        if not key:
            raise ValueError("execution_key is required")
        with self._lock:
            if key in self._keys:
                return False
            self._keys.add(key)
            return True

    def contains(self, execution_key: str) -> bool:
        with self._lock:
            return str(execution_key or "").strip() in self._keys

    def release(self, execution_key: str) -> None:
        """Release only when the caller has positively established no order exists."""
        with self._lock:
            self._keys.discard(str(execution_key or "").strip())
