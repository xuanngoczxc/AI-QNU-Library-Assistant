# QNU Library Assistant — Eval Suite

Smoke-test harness for verifying `/chat` behaviour, `/stats` observability, and the core search / RAG paths.

## Quick start

```bash
# 1. Start backend (in another terminal)
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# 2. Run all cases
python eval/run_eval.py

# 3. Run a single case
python eval/run_eval.py --only content_001_ai_noi_dung

# 4. Verbose mode (print full responses)
python eval/run_eval.py --verbose

# 5. Against a different host
python eval/run_eval.py --base http://10.0.0.5:8000
```

## Files

| File | Purpose |
|---|---|
| `sample_queries.json` | Test corpus: 15 cases across 8 categories |
| `run_eval.py` | Test runner: POST /chat, verify responses, cross-check `/stats.llm.calls` |

## Categories

| Category | Trigger | LLM? | Purpose |
|---|---|---|---|
| `list_request` | "sách về X", "tìm tài liệu X" | No | Validates BM25 + topic cache retrieval |
| `content_request` | "nội dung", "tóm tắt", "chương 2" | **Yes** | Validates LLM pipeline + circuit breaker |
| `author_request` | "sách của Nguyễn Văn A" | No | Validates author metadata search |
| `publisher_request` | "sách của NXB X" | No | Validates publisher metadata search |
| `major_filter` | `major` field set | No | Validates `Data/_khoa_nganh_tree.json` filter |
| `chitchat` | "xin chào", "cảm ơn" | No | Validates smalltalk short-circuit (no search, no LLM) |
| `existence_check` | "có sách về X không?" | No | Validates existence phrasing |
| `edge_case` | Very short / very long queries | No | Validates graceful handling |

## Verifications per case

| Check | Source |
|---|---|
| HTTP 200 from `/chat` | urllib response |
| `min_sources` threshold | `expect.min_sources` |
| `expect_answer_contains` | substring match in `.answer` |
| `expect_answer_not_contains` | substring must NOT appear |
| `expect_llm_call` | `GET /stats` before/after, expect `llm.calls` to increment |
| `expect_summary_min_len` | applied when query asks for `tóm tắt` |

## Adding a new case

Append to `sample_queries.json`:

```json
{
  "id": "my_new_case",
  "category": "list_request",
  "request": {"query": "your query here", "session_id": "eval-mycase"},
  "expect": {
    "min_sources": 1,
    "expect_llm_call": false,
    "expect_answer_contains": ["Mình tìm thấy"]
  }
}
```

## Why `expect_llm_call` matters

The LLM counter (`/stats.llm.calls`) is the canonical proof that the `rag_chain → call_llm` path was reached. If a content query shows `calls=0` after the test, that means:

1. The query was mis-routed to `catalog_mode` (uses `build_catalog_answer` — no LLM)
2. Or `detect_content_request()` returned False for the query
3. Or there's an exception silently caught upstream

The `content_request` category is the regression test for path #2.

## Known acceptable "failures"

| Case | Why it's expected |
|---|---|
| `major_filter_001_cntt` | No documents have `major="cong nghe thong tin"` in the canonical catalog — this validates the filter works, not a failure |
| `edge_001_short` (query="abc") | Returns 0 sources; the test only checks no exception |

## Exit codes

- `0` — all cases passed
- `1` — at least one case failed
- `2` — config error (file missing, id not found)
