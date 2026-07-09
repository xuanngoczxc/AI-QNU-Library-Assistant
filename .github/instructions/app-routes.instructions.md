---
description: "Hướng dẫn cho các route liên quan đến dashboard/admin trong app.py. Use when: thêm route mới cho admin, sửa API usage endpoints."
applyTo: "app.py"
---

# Admin/Usage Routes in app.py

## Route Map

| Route | Method | Purpose |
|-------|--------|---------|
| `/admin` | GET | Serves `admin-dashboard.html` |
| `/api/usage/analytics` | GET | Full analytics aggregation from JSONL |
| `/api/usage/recent?n=20` | GET | Recent usage entries |
| `/api/usage/stats` | GET | Quick in-memory stats |
| `/api/admin/login` | POST | Admin authentication |
| `/api/admin/dashboard` | POST | Protected dashboard payload (documents, cache, files) |

## Adding Usage Routes

When adding a new usage analytics route:
1. Import `get_usage_logger` from `lib.usage_logger`
2. Route goes under `# Usage / Analytics endpoints` section
3. Wrap in try/except, return `JSONResponse(status_code=500)` on error
4. Prefix with `/api/usage/`

## Dashboard HTML Serving

- `GET /admin` serves `admin-dashboard.html` from project root
- Sets `Cache-Control: no-store` to force fresh data on every load
- File is not templated — data fetched client-side via `/api/usage/*` endpoints