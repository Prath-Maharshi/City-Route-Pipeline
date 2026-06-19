"""
tasks.py — In-process async task registry + per-session travel-time state.
"""
from __future__ import annotations
import threading
import time
import uuid


# ── Task registry ─────────────────────────────────────────────────────────────

_tasks: dict     = {}
_tasks_lock      = threading.Lock()
TASK_TTL_S       = 3600

def make_task() -> str:
    tid = str(uuid.uuid4())
    with _tasks_lock:
        _tasks[tid] = {"status": "running", "result": None,
                       "error": None, "ts": time.time()}
    return tid

def finish_task(tid: str, result: dict) -> None:
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid].update(status="done", result=result)

def fail_task(tid: str, error: str) -> None:
    with _tasks_lock:
        if tid in _tasks:
            _tasks[tid].update(status="error", error=error)

def get_task(tid: str) -> dict | None:
    with _tasks_lock:
        return dict(_tasks[tid]) if tid in _tasks else None

def evict_old_tasks() -> int:
    now = time.time()
    with _tasks_lock:
        stale = [k for k, t in _tasks.items()
                 if t["status"] != "running" and now - t["ts"] > TASK_TTL_S]
        for k in stale:
            del _tasks[k]
    return len(stale)


# ── Session travel-time state ─────────────────────────────────────────────────
# Structure: { session_id: { hour: { edge_idx: tt_val } } }

_session_tt: dict[str, dict]  = {}
_session_ts: dict[str, float] = {}
_tt_lock    = threading.Lock()
SESSION_TTL_S = 7200

def get_session_tt(sid: str) -> dict:
    with _tt_lock:
        return _session_tt.setdefault(sid, {})

def read_session_tt(sid: str, hour: int) -> dict:
    with _tt_lock:
        return dict(_session_tt.get(sid, {}).get(hour, {}))

def update_session_tt(sid: str, hour: int, result: dict) -> None:
    raw = result.get("tt_updated_raw")
    if not raw:
        return
    with _tt_lock:
        sess = _session_tt.setdefault(sid, {})
        hmap = sess.setdefault(hour, {})
        for idx, val in enumerate(raw):
            hmap[idx] = val
        _session_ts[sid] = time.time()

def clear_session(sid: str) -> None:
    with _tt_lock:
        _session_tt.pop(sid, None)
        _session_ts.pop(sid, None)

def evict_old_sessions() -> int:
    now = time.time()
    with _tt_lock:
        stale = [s for s, ts in _session_ts.items() if now - ts > SESSION_TTL_S]
        for s in stale:
            _session_tt.pop(s, None)
            _session_ts.pop(s, None)
    return len(stale)


# ── Periodic cleanup ──────────────────────────────────────────────────────────
import logging
log = logging.getLogger("tasks")

def _cleanup_loop():
    while True:
        time.sleep(600)
        ns = evict_old_sessions()
        nt = evict_old_tasks()
        if ns or nt:
            log.info("Cleanup: evicted %d sessions, %d tasks", ns, nt)

threading.Thread(target=_cleanup_loop, daemon=True).start()
