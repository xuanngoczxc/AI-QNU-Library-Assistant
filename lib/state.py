"""
lib/state.py — Shared mutable runtime state (singletons, counters, caches).
Import this module to read/write app-wide state.
"""

import time
from collections import Counter, deque, OrderedDict
from .config import LLM_CIRCUIT_THRESHOLD, LLM_CIRCUIT_COOLDOWN, MAX_HISTORY

# ── Uptime ────────────────────────────────────────────────
APP_START_TIME = time.time()

# ── Admin tokens ──────────────────────────────────────────
ADMIN_TOKENS: set = set()

# ── LLM call stats ───────────────────────────────────────
llm_stats = {"calls": 0, "failures": 0, "retries": 0, "last_error": None}

# ── Circuit breaker ───────────────────────────────────────
llm_circuit = {
    "failures": 0,
    "state": "closed",       # "closed" | "open" | "half_open"
    "opened_at": 0.0,
    "failure_threshold": LLM_CIRCUIT_THRESHOLD,
    "cooldown_seconds": LLM_CIRCUIT_COOLDOWN,
}

# ── Session history (key = session_id, max 2000 sessions with LRU eviction) ────
MAX_SESSIONS = 2000
session_history: OrderedDict[str, list[dict]] = OrderedDict()

# ── PDF session history (key = "session_id:pdf_id") ──────
pdf_session_history: OrderedDict[str, list[dict]] = OrderedDict()


def get_session_history(session_id: str) -> list[dict]:
    """Retrieve session history and mark as recently accessed (LRU)."""
    if not session_id:
        return []
    if session_id in session_history:
        session_history.move_to_end(session_id)
        return session_history[session_id]
    return []


def save_session_history(session_id: str, history: list[dict]) -> None:
    """Save session history with bounded LRU eviction to prevent memory leaks."""
    if not session_id:
        return
    session_history[session_id] = history[-MAX_HISTORY:]
    session_history.move_to_end(session_id)
    # Evict oldest sessions when capacity is exceeded
    while len(session_history) > MAX_SESSIONS:
        session_history.popitem(last=False)


# ── Lightweight in-memory analytics ───────────────────────
query_analytics = {
    "total": 0,
    "top_queries": Counter(),
    "no_result_queries": Counter(),
    "labels": {},
    "recent": deque(maxlen=100),
    "last_query_at": None,
}
