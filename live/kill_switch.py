"""Kill switch: halts all new order submission instantly, persisted to disk
so the halt survives process restarts. Does not require per-trade approval
to engage -- calling halt() stops everything immediately.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from settings import state_dir


class KillSwitch:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else state_dir() / "kill_switch.json"

    def _write(self, halted: bool, reason: str = "") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "halted": halted,
                "reason": reason,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }, f, indent=2)

    def halt(self, reason: str = "manual kill switch engaged") -> None:
        self._write(True, reason)

    def resume(self) -> None:
        self._write(False, "")

    def is_halted(self) -> bool:
        if not self.path.exists():
            return False
        with open(self.path, "r") as f:
            data = json.load(f)
        return bool(data.get("halted", False))

    def status(self) -> dict:
        if not self.path.exists():
            return {"halted": False, "reason": "", "updated_at": None}
        with open(self.path, "r") as f:
            return json.load(f)


_default_kill_switch: KillSwitch | None = None


def get_kill_switch() -> KillSwitch:
    global _default_kill_switch
    if _default_kill_switch is None:
        _default_kill_switch = KillSwitch()
    return _default_kill_switch
