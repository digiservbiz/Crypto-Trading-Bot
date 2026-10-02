"""Fail-closed emergency trading controls.

The kill switch is intentionally independent of strategy decisions. A caller can
set a durable sentinel and require an explicit clear before new entries resume.
"""

from __future__ import annotations

from pathlib import Path


class KillSwitch:
    def __init__(self, path: str = "data/state/bot.stop") -> None:
        self.path = Path(path)

    def is_active(self) -> bool:
        return self.path.exists()

    def activate(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("KILL_SWITCH_ACTIVE\n", encoding="utf-8")

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def require_clear(self) -> None:
        if self.is_active():
            raise RuntimeError("Trading kill switch is active")
