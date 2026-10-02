"""Idempotency primitives for trade execution.

A deterministic execution key lets the broker boundary recognize repeated
requests for the same approved signal. Persistence/integration belongs at the
execution layer; this module only defines the stable key contract.
"""

from __future__ import annotations

import hashlib
import re


def normalize_signal_id(signal_id: str) -> str:
    value = str(signal_id or "").strip()
    if not value:
        raise ValueError("signal_id is required")
    if len(value) > 256:
        raise ValueError("signal_id is too long")
    return value


def build_execution_key(signal_id: str, symbol: str, side: str) -> str:
    """Build a stable, bounded idempotency key from an approved signal."""
    sid = normalize_signal_id(signal_id)
    sym = str(symbol or "").strip().upper()
    action = str(side or "").strip().lower()

    if not re.fullmatch(r"[A-Z0-9._:/-]+", sym):
        raise ValueError("invalid symbol")
    if action not in {"buy", "sell"}:
        raise ValueError("invalid side")

    raw = f"{sid}|{sym}|{action}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
