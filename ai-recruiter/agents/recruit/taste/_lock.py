"""Process-wide lock guarding all taste-module SQLite operations.

The recruit agent shares a single `sqlite3.Connection` across requests
(opened with `check_same_thread=False`). Even with that flag, the
underlying SQLite library serialises at the connection level and
returns `SQLITE_MISUSE` if two threads issue overlapping calls — Python
surfaces that as `sqlite3.InterfaceError: bad parameter or other API
misuse`.

We hit this in practice because:
  • The taste/score endpoint fires two `asyncio.to_thread` calls in a
    row (feature compute, then score), each on a different worker
    thread, both touching `agent.conn`.
  • The capture hook schedules a background refit (`asyncio.to_thread`)
    that races against the next request.

Holding `DB_LOCK` around every DB-touching public entry point in the
taste module serialises these accesses. RLock allows a function that
already holds the lock to call other locked helpers without deadlock.
"""
from __future__ import annotations

import threading

DB_LOCK: threading.RLock = threading.RLock()
