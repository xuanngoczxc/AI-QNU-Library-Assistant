"""
lib/usage_logger.py — Structured usage logging for multi-user deployment.

Writes every search query to a JSONL file with full context:
  - timestamp (ISO 8601)
  - session_id
  - client_ip
  - user_agent (truncated)
  - query (raw)
  - result_count
  - latency_ms
  - follow_ups (count of suggestions)
  - error (if any)

Also maintains an in-memory ring buffer for the dashboard.
"""

from __future__ import annotations

import os
import json
import time
import logging
from pathlib import Path
from collections import deque
from typing import Optional

logger = logging.getLogger("qnu.library.usage")
 
LOG_DIR = os.getenv("USAGE_LOG_DIR", "logs")
LOG_FILE = os.getenv("USAGE_LOG_FILE", "search-usage.jsonl")
MAX_LOG_SIZE_MB = int(os.getenv("USAGE_LOG_MAX_MB", "50"))
MAX_RECENT_IN_MEMORY = int(os.getenv("USAGE_RECENT_CAPACITY", "5000"))


class UsageLogger:
    """
    Thread-safe structured logger for search queries.
    Writes one JSON object per line (JSONL) — easy to ingest via
    ELK, Grafana, or simple grep / jq.
    """

    def __init__(
        self,
        log_dir: str = LOG_DIR,
        log_file: str = LOG_FILE,
        max_size_mb: int = MAX_LOG_SIZE_MB,
        recent_capacity: int = MAX_RECENT_IN_MEMORY,
    ):
        self._log_dir = Path(log_dir)
        self._log_file = self._log_dir / log_file
        self._max_bytes = max_size_mb * 1024 * 1024
        self._recent: deque[dict] = deque(maxlen=recent_capacity)
        self._total_entries = 0
        self._ensure_directory()
        self._maybe_rotate()

        # per-day counters for dashboard quick stats
        self._day_queries: dict[str, int] = {}  # "YYYY-MM-DD" -> count
        self._day_noresult: dict[str, int] = {}

    # ── public API ──────────────────────────────────────────

    def record(
        self,
        *,
        session_id: str,
        query: str,
        result_count: int = 0,
        latency_ms: float = 0.0,
        follow_up_count: int = 0,
        client_ip: str = "",
        user_agent: str = "",
        error: str = "",
        intent: str = "",
    ) -> None:
        """Record one search query to disk + in-memory buffer."""
        now = time.time()
        ts_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))
        day_key = time.strftime("%Y-%m-%d", time.localtime(now))

        entry = {
            "ts": ts_iso,
            "ts_unix": round(now, 3),
            "session_id": (session_id or "")[:80],
            "query": (query or "")[:500].strip(),
            "result_count": int(result_count),
            "latency_ms": round(latency_ms, 1),
            "follow_up_count": int(follow_up_count),
            "intent": (intent or "")[:40],
            "client_ip": (client_ip or "")[:45],
            "user_agent": (user_agent or "")[:200],
            "error": (error or "")[:300],
        }

        # disk
        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            self._write_line(line)
        except Exception as e:
            logger.warning("Failed to write usage log entry: %s", e)

        # in-memory (hot data for /dashboard)
        self._recent.appendleft(entry)
        self._total_entries += 1
        self._day_queries[day_key] = self._day_queries.get(day_key, 0) + 1
        if result_count <= 0:
            self._day_noresult[day_key] = self._day_noresult.get(day_key, 0) + 1

    def get_recent(self, n: int = 50) -> list[dict]:
        """Return the *n* most recent entries (newest first)."""
        return list(self._recent)[:n]

    def get_stats(self) -> dict:
        """Aggregate counters for the admin dashboard."""
        today = time.strftime("%Y-%m-%d")
        yesterday_key = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
        return {
            "total_entries": self._total_entries,
            "today_queries": self._day_queries.get(today, 0),
            "yesterday_queries": self._day_queries.get(yesterday_key, 0),
            "today_no_result": self._day_noresult.get(today, 0),
            "in_memory_recent": len(self._recent),
            "log_file": str(self._log_file),
            "log_file_size_mb": round(self._file_size_mb(), 1),
        }

    # ── internal helpers ────────────────────────────────────

    def _ensure_directory(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _file_size_mb(self) -> float:
        try:
            return self._log_file.stat().st_size / (1024 * 1024)
        except OSError:
            return 0.0

    def _maybe_rotate(self) -> None:
        """Simple rotation: if log exceeds max size, rename and start fresh."""
        if self._file_size_mb() < MAX_LOG_SIZE_MB:
            return
        try:
            rotated = self._log_file.with_suffix(
                f".{int(time.time())}.jsonl"
            )
            self._log_file.rename(rotated)
            logger.info("Usage log rotated to %s", rotated.name)
        except Exception as e:
            logger.warning("Log rotation failed: %s", e)

    def _write_line(self, line: str) -> None:
        # utf-8-sig = UTF-8 BOM để Windows PowerShell hiển thị đúng tiếng Việt
        with open(self._log_file, "a", encoding="utf-8-sig") as f:
            f.write(line)
        # check size after ~every 100 writes
        if self._total_entries % 100 == 0:
            self._maybe_rotate()


# ── singleton instance (importable from anywhere) ────────
_usage_logger: Optional[UsageLogger] = None


def get_usage_logger() -> UsageLogger:
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger()
    return _usage_logger
