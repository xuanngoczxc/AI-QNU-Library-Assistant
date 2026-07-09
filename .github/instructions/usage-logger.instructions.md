---
description: "Hướng dẫn cho UsageLogger — module ghi log JSONL và analytics. Use when: làm việc với lib/usage_logger.py, thêm metric mới, sửa logic analytics."
applyTo: "lib/usage_logger.py"
---

# UsageLogger Instructions

## Core Design

- **JSONL format**: one JSON object per line, append-only
- **Rotation**: auto when file > `USAGE_LOG_MAX_MB` (default 50MB)
- **Fallback mode**: `get_recent()` falls back to `read_all_entries()` when in-memory buffer empty (after restart)
- **Thread-safe**: file write uses Python's `open()` which is atomic on most OS

## Entry Schema

```json
{
  "ts": "2026-07-09T08:56:21",
  "ts_unix": 1783328181.367,
  "session_id": "sess_xxx",
  "query": "tìm sách về AI",
  "result_count": 42,
  "latency_ms": 1234.5,
  "follow_up_count": 3,
  "intent": "catalog_search",
  "client_ip": "",
  "user_agent": "Mozilla/...",
  "error": ""
}
```

## Key Methods

| Method | Purpose |
|--------|---------|
| `record()` | Write one entry to disk + in-memory buffer |
| `get_recent(n)` | Last N entries (memory first, file fallback) |
| `get_stats()` | Quick in-memory counters |
| `read_all_entries()` | Full file read, newest first |
| `compute_analytics()` | Full aggregation (daily, hourly, latency, intents, top queries) |

## compute_analytics() Output Structure

```python
{
  "total": int,
  "daily_queries": [{"date": str, "queries": int, "no_result": int}],
  "intent_distribution": [{"intent": str, "count": int, "pct": float}],
  "top_queries": [{"query": str, "count": int}],
  "top_no_result_queries": [{"query": str, "count": int}],
  "latency": {"avg_ms": float, "min_ms": float, "max_ms": float, "p50_ms": float, "p95_ms": float, "p99_ms": float},
  "hourly_heatmap": [{"hour": int, "count": int}],
  "monthly": [{"month": str, "queries": int}],
  "unique_days": int,
  "weekday_distribution": [{"day": str, "count": int}],
}
```

## Adding New Metrics

To add a new analytics dimension:
1. Add extraction logic in the `for e in entries:` loop inside `compute_analytics()`
2. Add the result to the return dict
3. Update the dashboard JS to consume the new field

## Log Rotation

- Rotates when file > `USAGE_LOG_MAX_MB`
- Renames to `search-usage.{timestamp}.jsonl`
- Does NOT re-read rotated files (old logs ignored by analytics)