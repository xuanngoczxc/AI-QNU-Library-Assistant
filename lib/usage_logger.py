"""
lib/usage_logger.py — In-memory usage logging for analytics dashboard.

Maintains an in-memory ring buffer.
No disk writes — data survives only within the current server session.
"""

from __future__ import annotations

import os
import time
import logging
from collections import deque, Counter
from typing import Optional

logger = logging.getLogger("qnu.library.usage")
 
MAX_RECENT_IN_MEMORY = int(os.getenv("USAGE_RECENT_CAPACITY", "5000"))


class UsageLogger:
    """
    In-memory structured logger for search queries.
    Data is kept in a ring buffer — no disk I/O.
    On server restart, analytics start fresh.
    """

    def __init__(
        self,
        recent_capacity: int = MAX_RECENT_IN_MEMORY,
    ):
        self._recent: deque[dict] = deque(maxlen=recent_capacity)
        self._total_entries = 0

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
        }

    # ── JSONL file reader / aggregator ─────────────────────

    def compute_analytics(self, date_from: str = "", date_to: str = "") -> dict:
        """
        Full analytics aggregation from the in-memory buffer
        supplemented by per-day counters which survive deque eviction.
        Optionally filter by date range (inclusive, format YYYY-MM-DD).
        """
        entries = list(self._recent)

        # ── seed per‑day totals from the always‑accurate counters ──
        day_counts: Counter = Counter()
        day_noresult: Counter = Counter()
        for d, c in self._day_queries.items():
            day_counts[d] += c
        for d, c in self._day_noresult.items():
            day_noresult[d] += c

        # ── filter entries by date range ──
        if date_from:
            entries = [e for e in entries if (e.get("ts") or "")[:10] >= date_from]
            day_counts = Counter({d: c for d, c in day_counts.items() if d >= date_from})
            day_noresult = Counter({d: c for d, c in day_noresult.items() if d >= date_from})
        if date_to:
            entries = [e for e in entries if (e.get("ts") or "")[:10] <= date_to]
            day_counts = Counter({d: c for d, c in day_counts.items() if d <= date_to})
            day_noresult = Counter({d: c for d, c in day_noresult.items() if d <= date_to})

        if not entries and not day_counts:
            return {
                "total": 0,
                "daily_queries": [],
                "intent_distribution": [],
                "top_queries": [],
                "top_no_result_queries": [],
                "latency": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "p99_ms": 0},
                "hourly_heatmap": [],
                "recent": [],
                "daily_no_result": [],
            }

        hour_counts: list[int] = [0] * 24
        intent_counts: Counter = Counter()
        query_counts: Counter = Counter()
        noresult_query_counts: Counter = Counter()
        latencies: list[float] = []
        monthly_counts: Counter = Counter()
        weekday_counts: Counter = Counter()

        for e in entries:
            day = e.get("ts", "")[:10]

            rc = int(e.get("result_count", 0))
            if rc <= 0:
                q = (e.get("query") or "").strip()
                if q:
                    noresult_query_counts[q] += 1

            lat = float(e.get("latency_ms", 0) or 0)
            if lat > 0:
                latencies.append(lat)

            intent = (e.get("intent") or "unknown").strip()
            if intent:
                intent_counts[intent] += 1

            q = (e.get("query") or "").strip()
            if q:
                query_counts[q] += 1

            try:
                ts = float(e.get("ts_unix", 0))
                if ts > 0:
                    lt = time.localtime(ts)
                    hour_counts[lt.tm_hour] += 1
                    weekday_counts[lt.tm_wday] += 1
                    if day:
                        monthly_counts[day[:7]] += 1
            except (ValueError, TypeError):
                pass

        latencies.sort()
        n_lat = len(latencies)
        latency_stats = {
            "avg_ms": round(sum(latencies) / n_lat, 1) if n_lat else 0,
            "min_ms": round(latencies[0], 1) if n_lat else 0,
            "max_ms": round(latencies[-1], 1) if n_lat else 0,
            "p50_ms": round(latencies[n_lat // 2], 1) if n_lat else 0,
            "p95_ms": round(latencies[int(n_lat * 0.95)], 1) if n_lat else 0,
            "p99_ms": round(latencies[int(n_lat * 0.99)], 1) if n_lat else 0,
        }

        days_sorted = sorted(day_counts.keys())
        daily_queries = [
            {"date": d, "queries": day_counts[d], "no_result": day_noresult.get(d, 0)}
            for d in days_sorted
        ]

        total_intents = sum(intent_counts.values()) or 1
        intent_distribution = [
            {"intent": k, "count": v, "pct": round(v / total_intents * 100, 1)}
            for k, v in intent_counts.most_common()
        ]

        top_queries = [
            {"query": k, "count": v}
            for k, v in query_counts.most_common(20)
        ]

        top_no_result = [
            {"query": k, "count": v}
            for k, v in noresult_query_counts.most_common(10)
        ]

        monthly_data = [
            {"month": k, "queries": v}
            for k, v in sorted(monthly_counts.items())
        ]

        return {
            "total": sum(day_counts.values()),
            "daily_queries": daily_queries,
            "intent_distribution": intent_distribution,
            "top_queries": top_queries,
            "top_no_result_queries": top_no_result,
            "latency": latency_stats,
            "hourly_heatmap": [{"hour": i, "count": hour_counts[i]} for i in range(24)],
            "daily_no_result": [{"date": d, "count": day_noresult.get(d, 0)} for d in days_sorted],
            "monthly": monthly_data,
            "unique_days": len(days_sorted),
            "weekday_distribution": [
                {"day": {0: "CN", 1: "T2", 2: "T3", 3: "T4", 4: "T5", 5: "T6", 6: "T7"}.get(k, ""), "count": v}
                for k, v in sorted(weekday_counts.items())
            ],
        }

    # ── internal helpers ────────────────────────────────────

_usage_logger: Optional[UsageLogger] = None


def get_usage_logger() -> UsageLogger:
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger()
    return _usage_logger
