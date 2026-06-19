# Thư viện QNU RAG API

> Trợ lý ảo tra cứu tài liệu thư viện cho Trường Đại học Quy Nhơn (QNU) — hỗ trợ tìm kiếm catalog, đọc nội dung PDF, OCR file scan, và trả lời theo ngữ cảnh qua RAG + LLM.

`FastAPI` · `Python 3.11` · `OpenRouter` · `BM25` · `Sentence-Transformers (lazy)`

---

## Tính năng chính

- **Tra cứu catalog** theo chủ đề, ngành, tác giả, nhà xuất bản, mã DDC.
- **Tìm kiếm hybrid** BM25 (deterministic) + topic cache (fast lookup) + semantic (lazy load).
- **Trả lời nội dung** PDF/luận văn qua RAG kết hối LLM (`openai/gpt-oss-120b:free` mặc định).
- **OCR fallback** cho PDF scan/ảnh (qua `pdf2image` + `pytesseract`).
- **Voice input** (qua `faster-whisper`, optional).
- **Admin upload PDF** + auto reindex vào BM25.
- **Search analytics** + LLM circuit breaker + structured logging.

---

## Cấu trúc dự án

```
app.py                  # FastAPI app chính (chat, search, admin, voice, OCR)
search_engine.py        # BM25 + hybrid search
semantic_search.py      # Sentence-Transformers (lazy load)
pdf_manager.py          # PDF metadata + structure + content
load_pdf.py             # PDF loader → LangChain Documents
text_utils.py           # Vietnamese normalization
voice_input.py          # Whisper transcription (optional)
eval/                   # Test suite (15 cases, stdlib runner)
  run_eval.py
  sample_queries.json
  README.md
Data/                   # Catalog CSV (input — KHÔNG chỉnh sửa)
pdfs/                   # PDF storage (input — KHÔNG chỉnh sửa)
api/index.py            # Vercel serverless entrypoint
chat(1).html            # Frontend chatbot UI
vercel.json             # Vercel deploy config
Dockerfile              # Docker build
requirements.txt        # Python dependencies
```

---

## Cài đặt

### 1. Yêu cầu

- Python **3.11+**
- Windows / macOS / Linux
- API key OpenRouter: <https://openrouter.ai/keys>

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

Dependencies chính:

| Package | Mục đích |
|---|---|
| `fastapi` · `uvicorn` | API server |
| `pandas` · `openpyxl` | Đọc catalog CSV/XLSX |
| `pypdf` · `markitdown[pdf]` | Trích xuất text PDF |
| `python-multipart` | Form/file upload |
| `python-dotenv` · `requests` | Env + HTTP |
| `numpy` · `pydantic` | Tiện ích + schema |
| `pytesseract` · `pdf2image` | OCR (optional) |
| `sentence-transformers` · `torch` | Semantic search (optional, heavy) |
| `faster-whisper` | Voice input (optional) |

### 3. Cấu hình môi trường

Tạo file `.env` ở thư mục gốc:

```env
# === Bắt buộc ===
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxx

# === Tùy chọn ===
OPENROUTER_MODEL=openai/gpt-oss-120b:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_APP_TITLE=Thư viện QNU RAG API
OPENROUTER_HTTP_REFERER=https://qnu-library.example

# === Logging & circuit breaker ===
LOG_LEVEL=INFO
LLM_CIRCUIT_THRESHOLD=5
LLM_CIRCUIT_COOLDOWN=60

# === Admin ===
ADMIN_PASSWORD=admin123
```

### 4. Chuẩn bị dữ liệu

Đặt catalog vào `Data/*.csv` (định dạng: title, author, year, subject, major, publisher, abstract, link, source, doc_type, format, location, notes, ddc, course) và PDF vào `pdfs/`. **Lần đầu chạy** hệ thống sẽ tự build các cache.

### 5. Chạy server

```bash
# Cách 1: uvicorn trực tiếp
python -m uvicorn app:app --host 127.0.0.1 --port 8000

# Cách 2: Windows + PowerShell (nếu gặp lỗi encoding charmap)
$env:PYTHONIOENCODING="utf-8"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000> để truy cập UI chatbot, hoặc <http://127.0.0.1:8000/docs> để xem Swagger.

---

## Kiến trúc

### Dual-path `/chat`

`/chat` xử lý theo **2 luồng** dựa trên intent detection:

```
user query
    │
    ├── detect_list_request() / detect_existence_request() / detect_chitchat()
    │       → catalog_mode = True
    │       → build_catalog_answer(results, query)   ← KHÔNG gọi LLM
    │
    └── detect_content_request() (tóm tắt, nội dung, chương, file này...)
            → catalog_mode = False
            → build_general_chain().run(context, question)   ← GỌI LLM
```

**Lợi ích**: ~80% câu hỏi (tra cứu, liệt kê) skip LLM → nhanh, rẻ, ổn định. Chỉ những câu hỏi cần tổng hợp nội dung mới gọi LLM.

### Hybrid search

```
query
    │
    ├── Topic cache (sub-100ms)
    │     └─ exact / prefix / multi-keyword match
    │
    ├── BM25 (1-10ms)
    │     └─ rank by TF-IDF-like scoring
    │
    └── Semantic (lazy, chỉ khi BM25 yếu)
          └─ Sentence-Transformers cosine similarity
```

Kết hợp: `metadata_search_by_query()` + `bm25_engine.search()` + `get_semantic_engine()` → `rerank_results_for_query()` (boost author/publisher/AI-domain, filter nhiễu).

### LLM circuit breaker

- Mỗi lần LLM fail → `_record_llm_failure()`.
- Sau `LLM_CIRCUIT_THRESHOLD` (mặc định 5) lần fail liên tiếp → circuit **mở** trong `LLM_CIRCUIT_COOLDOWN` giây (mặc định 60s).
- Trong thời gian mở, request trả lỗi ngay → giảm tải LLM endpoint.
- Xem trạng thái real-time: `GET /stats`.

### Logging

- `StreamHandler` (stdout, UTF-8) + `FileHandler("app.log", encoding="utf-8", mode="a")`.
- `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` ép UTF-8 trên Windows.

---

## API Endpoints

### Tìm kiếm & chat

| Method | Path | Mô tả |
|---|---|---|
| `GET`  | `/` | Trang chủ / UI chatbot |
| `POST` | `/chat` | Hỏi đáp RAG (catalog + content) |
| `POST` | `/chat/with-document` | Hỏi đáp với PDF đính kèm |
| `GET`  | `/documents` | Danh sách tài liệu |
| `GET`  | `/search` | Search trả về JSON |
| `GET`  | `/suggest-related` | Gợi ý truy vấn liên quan |
| `GET`  | `/spell-check` | Sửa lỗi chính tả truy vấn |
| `GET`  | `/suggest-terms` | Gợi ý từ khóa |

### Trạng thái & analytics

| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/status` | Trạng thái hệ thống (index, PDF count, cache) |
| `GET` | `/stats` | LLM call stats + circuit breaker state |

### API mở rộng (cho frontend)

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/semantic-search` | Pure semantic search |
| `GET`  | `/api/faculties` | Danh sách khoa |
| `GET`  | `/api/faculties/search` | Tìm khoa |
| `GET`  | `/api/pdfs` | Danh sách PDF |
| `GET`  | `/api/pdf/{pdf_id}/is-scanned` | Kiểm tra PDF có phải scan |
| `GET`  | `/api/pdf/{pdf_id}/content-with-ocr` | Lấy text + OCR fallback |
| `GET`  | `/api/pdfs/{pdf_id}/structure` | Cấu trúc PDF (chương, mục) |
| `GET`  | `/api/pdfs/{pdf_id}/content` | Toàn bộ text PDF |
| `GET`  | `/api/pdfs/{pdf_id}/summary` | Tóm tắt PDF |

### Voice

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/voice/transcribe` | Audio → text (Whisper) |
| `GET`  | `/api/voice/status` | Trạng thái voice engine |

### Admin (cần auth)

| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/api/admin/login` | Đăng nhập (set cookie session) |
| `POST` | `/api/admin/logout` | Đăng xuất |
| `POST` | `/api/admin/upload` | Upload PDF mới (multipart) |
| `DELETE` | `/api/admin/pdfs/{pdf_id}` | Xóa PDF |

---

## Test suite (eval)

Bộ 15 test cases stdlib, không cần pytest:

```bash
python eval/run_eval.py
```

Kết quả mong đợi: `=== Summary: 15 passed, 0 failed, 0 skipped ===`.

Xem chi tiết tại [eval/README.md](eval/README.md).

---

## Triển khai

### Vercel (serverless)

```bash
vercel --prod
```

`vercel.json` đã có sẵn. Xem hướng dẫn trong [DEPLOY_VERCEL.txt](DEPLOY_VERCEL.txt).

### Docker

```bash
docker build -t qnu-library .
docker run -p 8000:8000 --env-file .env qnu-library
```

---

## Xử lý sự cố thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `UnicodeEncodeError: 'charmap' codec` (Windows) | PowerShell cp1252 | Set `$env:PYTHONIOENCODING="utf-8"` trước khi chạy |
| `ModuleNotFoundError: sentence_transformers` | Chưa cài optional dep | `pip install sentence-transformers torch` (hoặc bỏ qua, semantic search tự fallback BM25) |
| `/chat` trả về "Tài liệu không cung cấp thông tin này" | PDF scan không có text layer | Cần OCR trước hoặc upload bản PDF có text |
| `LLM circuit open` log liên tục | OpenRouter key invalid / rate limit | Kiểm tra key, đợi cooldown hoặc tăng `LLM_CIRCUIT_THRESHOLD` |
| Frontend mở nhưng `/api/*` 404 | Chưa start backend | `python -m uvicorn app:app --host 127.0.0.1 --port 8000` |

---

## Biến môi trường đầy đủ

| Tên | Mặc định | Mô tả |
|---|---|---|
| `OPENROUTER_API_KEY` | — | **Bắt buộc**. OpenRouter API key |
| `OPENROUTER_MODEL` | `openai/gpt-oss-120b:free` | Model LLM |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | API base URL |
| `OPENROUTER_APP_TITLE` | `Thư viện QNU RAG API` | App title gửi OpenRouter |
| `OPENROUTER_HTTP_REFERER` | — | HTTP referer cho OpenRouter |
| `LOG_LEVEL` | `INFO` | Mức log (`DEBUG` / `INFO` / `WARNING`) |
| `LLM_CIRCUIT_THRESHOLD` | `5` | Số lần fail liên tiếp để mở circuit |
| `LLM_CIRCUIT_COOLDOWN` | `60` | Thời gian cooldown (giây) |
| `ADMIN_PASSWORD` | `admin123` | Mật khẩu admin panel |
| `PYTHONIOENCODING` | — | Đặt `utf-8` trên Windows để tránh charmap lỗi |

---

## Đóng góp

1. Fork repo.
2. Tạo branch: `git checkout -b feature/<ten-tinh-nang>`.
3. Commit: `git commit -m "Add <tính năng>"`.
4. Push: `git push origin feature/<ten-tinh-nang>`.
5. Tạo Pull Request.

Trước khi PR, chạy `python eval/run_eval.py` để chắc chắn 15/15 pass.

---

## License

MIT License. Xem `LICENSE` (nếu có).

---

## Liên hệ

- Trường Đại học Quy Nhơn — Thư viện
- Tác giả: Đỗ Vũ Nhật Linh & cộng sự
