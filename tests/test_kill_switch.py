from pathlib import Path

import pytest

from scripts.kill_switch import KillSwitch


def test_kill_switch_is_fail_closed(tmp_path: Path):
    switch = KillSwitch(str(tmp_path / "bot.stop"))
    assert not switch.is_active()

    switch.activate()
    assert switch.is_active()
    with pytest.raises(RuntimeError):
        switch.require_clear()

    switch.clear()
    assert not switch.is_active()
    switch.require_clear()
