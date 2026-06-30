"""
lib/state.py — Shared mutable runtime state (singletons, counters, caches).
Import this module to read/write app-wide state.
"""

import time
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

# ── Session history (key = session_id) ────────────────────
session_history: dict[str, list[dict]] = {}

# ── PDF session history (key = "session_id:pdf_id") ──────
pdf_session_history: dict[str, list[dict]] = {}
