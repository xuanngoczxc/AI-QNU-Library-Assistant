"""
app.py — FastAPI backend kết nối RAG (FAISS + OpenRouter) với chat.html
Chạy: uvicorn app:app --reload --port 8000
"""

import os
import re
import sys
import time
import logging
import unicodedata
import secrets
import hashlib
import pickle
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Structured logging (thay thế print() rải rác) ─────────
# Cấu hình 1 lần: ghi ra cả console (UTF-8) và file app.log
# StreamHandler dùng UTF-8 để tránh lỗi charmap cp1252 trên Windows
_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format=_LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger("qnu.library")

# Đảm bảo stdout là UTF-8 (Windows console hay lỗi với emoji/ký tự đặc biệt)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# Runtime state (process-uptime, LLM call counter, circuit breaker)
_APP_START_TIME = time.time()
_llm_stats = {"calls": 0, "failures": 0, "retries": 0, "last_error": None}
_llm_circuit = {
    "failures": 0,            # consecutive failures
    "state": "closed",        # "closed" | "open" | "half_open"
    "opened_at": 0.0,         # timestamp when circuit opened
    "failure_threshold": int(os.getenv("LLM_CIRCUIT_THRESHOLD", "5")),
    "cooldown_seconds": int(os.getenv("LLM_CIRCUIT_COOLDOWN", "60")),
}

# Import BM25 search engine (lightweight, no embeddings needed)
from search_engine import DocumentIndexer, BM25SearchEngine, HybridSearchEngine
from text_utils import normalize_text, extract_keywords, suggest_synonyms, spell_correct, suggest_terms_for_query
from intent_classifier import (
    classify_intent, detect_greeting, extract_entities,
)
from load_pdf import load_all_pdfs
from pdf_manager import PDFManager

# Giai đoạn 3 — Semantic + Voice (lazy load để không block startup)
_semantic_engine = None
def get_semantic_engine():
    global _semantic_engine
    if _semantic_engine is None:
        try:
            from semantic_search import SemanticSearchEngine
            _semantic_engine = SemanticSearchEngine(documents)
            logger.info("Semantic search engine loaded (%d docs)", len(documents))
        except Exception as e:
            logger.warning("Semantic search disabled: %s", e)
    return _semantic_engine

load_dotenv()

# Import OpenRouter client
import requests

# ── Config ───────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "poolside/laguna-xs.2:free")
# Fallback chain khi model chính fail (400 invalid model ID, 404, 503...)
# Tự động thử model tiếp theo; ưu tiên các model free phổ biến đang active
OPENROUTER_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OPENROUTER_FALLBACK_MODELS",
        "meta-llama/llama-3.3-70b-instruct:free,"
        "qwen/qwen-2.5-72b-instruct:free,"
        "mistralai/mistral-small-3.2-24b-instruct:free,"
        "google/gemini-2.0-flash-exp:free,"
        "openai/gpt-oss-20b:free"
    ).split(",")
    if m.strip()
]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "QNU Library Assistant")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000")

# ── Admin Config ─────────────────────────────────────────
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123").strip()
ADMIN_TOKENS: set = set()
VERCEL_READ_ONLY_MESSAGE = (
    "Bản demo trên Vercel chạy ở chế độ read-only: không thể upload hoặc xóa PDF "
    "vì Vercel Functions chỉ có filesystem tạm thời. Hãy deploy lên server riêng "
    "hoặc dùng storage ngoài nếu cần chức năng này."
)

def is_vercel_runtime() -> bool:
    return os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))

def check_backend() -> None:
    if not OPENROUTER_API_KEY:
        if is_vercel_runtime():
            logger.warning("OPENROUTER_API_KEY not set at build time — will check at runtime")
            return
        raise RuntimeError("OPENROUTER_API_KEY is required")
    logger.info("OpenRouter ready | Model: %s", OPENROUTER_MODEL)


# Không chạy check_backend() ngay khi build (Vercel build không có env),
# nhưng Vercel runtime sẽ chạy lại module khi function được gọi, lúc đó env có sẵn.
if not is_vercel_runtime():
    check_backend()

# ── Khởi tạo app ─────────────────────────────────────────
app = FastAPI(title="Thư viện QNU RAG API", version="2.0.0")

# Middleware: Permissions-Policy cho phép microphone hoạt động trên HTTPS
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(self)"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Load BM25 Full-Text Search Index...")
documents = DocumentIndexer.load_from_data("Data")

pdf_docs = []
try:
    loaded = load_all_pdfs()
    if loaded:
        logger.info("Adding %d PDF pages to BM25 documents...", len(loaded))
        seen_hashes = {d.get('hash') for d in documents if d.get('hash')}
        for doc in loaded:
            meta = doc.metadata or {}
            h = meta.get('hash')
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)

            # Trích xuất tên tác giả từ tên file PDF khi metadata thiếu
            # VD: "DO VU NHAT LINH - KHDL.pdf" → author = "Đỗ Vũ Nhật Linh"
            pdf_source = meta.get('source', 'pdf')
            pdf_author = meta.get('author', 'Unknown')
            if not pdf_author or pdf_author == 'Unknown' or 'aspose' in pdf_author.lower() or 'epubtopdfconverter' in pdf_author.lower() or 'pdf converter' in pdf_author.lower():
                # Cố gắng trích xuất tên từ tên file
                file_stem = Path(pdf_source).stem if pdf_source else ""
                # Pattern: "TEN TAC GIA - TEN LUAN VAN" hoặc "TEN TAC GIA-TEN..."
                m = re.match(r"^([A-Z][A-Z\s\.]+?)\s*[-–]\s*", file_stem)
                if m:
                    raw_name = m.group(1).strip()
                    # Map chữ cái đầu → tên đầy đủ bằng cách giữ nguyên
                    # "DO VU NHAT LINH" → vẫn dùng uppercase
                    pdf_author = raw_name

            # Subject: ch? dùng keywords t? filename title, KHÔNG dùng section heading (nhi?u)
            # Section heading là dòng UPPERCASE ??u tiên c?a PDF (nh? "B? GIÁO D?C...")
            # Không ph?i subject th?c s? → b? qua
            pdf_subject = ""
            kw = meta.get('keywords', '')
            if kw and len(kw) >= 5 and kw.lower() != 'pdf':
                pdf_subject = kw

            pdf_item = {
                'title': meta.get('title', meta.get('source', 'PDF')).strip(),
                'author': pdf_author,
                'year': meta.get('year', 'N/A'),
                'subject': pdf_subject,
                'link': '',
                'source': meta.get('source', 'pdf'),
                'page': meta.get('page'),
                'text': doc.page_content or '',
                'csv_file': 'pdf:' + meta.get('source', 'pdf') ,
                'hash': h
            }
            pdf_docs.append(pdf_item)
        documents.extend(pdf_docs)
    else:
        logger.info("No PDF pages found to add to BM25 index.")
except Exception as e:
    logger.warning("Error loading PDFs for BM25: %s", e)

bm25_engine = BM25SearchEngine(documents)
retriever = HybridSearchEngine(bm25_engine)

logger.info("Search index ready: %d documents", len(documents))
logger.info("Memory: ~50MB (no ML model loaded)")
use_pdf = len(pdf_docs) > 0

# ── File Change Detection ──────────────────────────────
# Detect khi có file mới/xóa trong pdfs/ hoặc Data/ folder (so với lúc startup)
_startup_pdf_count = sum(1 for _ in Path("pdfs").glob("*.pdf")) if Path("pdfs").exists() else 0
_startup_data_mtime = 0.0  # mtime lớn nhất của tất cả file trong Data/
for _f in Path("Data").glob("*"):
    if _f.is_file() and _f.name != "_documents_cache.pkl":
        _startup_data_mtime = max(_startup_data_mtime, _f.stat().st_mtime)

_last_check_time = 0.0
_check_interval = 10  # chỉ check mỗi 10 giây, không check mỗi request

def check_files_changed():
    """Check xem có file PDF hoặc CSV/XLSX mới/xóa/sửa so với lúc startup không."""
    global _last_check_time, _startup_pdf_count, _startup_data_mtime
    now = time.time()
    if now - _last_check_time < _check_interval:
        return {"pdf": False, "data": False}
    _last_check_time = now

    result = {"pdf": False, "data": False}

    # Check PDF folder
    current_pdf_count = sum(1 for _ in Path("pdfs").glob("*.pdf")) if Path("pdfs").exists() else 0
    if current_pdf_count != _startup_pdf_count:
        _startup_pdf_count = current_pdf_count
        result["pdf"] = True

    # Check Data folder (XLSX, CSV)
    current_data_mtime = 0.0
    if Path("Data").exists():
        for _f in Path("Data").glob("*"):
            if _f.is_file() and _f.name != "_documents_cache.pkl":
                current_data_mtime = max(current_data_mtime, _f.stat().st_mtime)
    if current_data_mtime > _startup_data_mtime:
        _startup_data_mtime = current_data_mtime
        result["data"] = True

    return result

# Giai đoạn 3 — Semantic engine (lazy load, không block startup)
_semantic_engine = None
def get_semantic_engine():
    """Lazy load semantic engine — chỉ load model khi thực sự cần."""
    global _semantic_engine
    if _semantic_engine is not None:
        return _semantic_engine
    try:
        from semantic_search import SemanticSearchEngine
        _semantic_engine = SemanticSearchEngine(documents)
        if _semantic_engine.embeddings is not None:
            logger.info("Semantic engine READY (hybrid BM25 + embeddings)")
        return _semantic_engine
    except Exception as e:
        logger.warning("Semantic engine unavailable: %s", e)
        return None

# ── Topic Index Cache ────────────────────────────────────
# Build a fast lookup: normalized keyword → list of doc indices
# This avoids BM25 search for common subject/major queries.
TOPIC_CACHE: dict[str, list[int]] = {}  # keyword → [doc_index, ...]
ALL_DOC_KEYWORDS: dict[int, set[str]] = {}  # doc_index → {keywords}

def _normalize_topic_token(text: str) -> str:
    """Normalize a single topic token for cache key."""
    t = normalize_text(text).strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _build_topic_cache():
    """Scan all documents and build keyword→docs index from subject/major fields."""
    global TOPIC_CACHE, ALL_DOC_KEYWORDS
    TOPIC_CACHE = {}
    ALL_DOC_KEYWORDS = {}

    subjects_with_docs = 0
    for idx, doc in enumerate(documents):
        keywords = set()
        for field in ("subject", "major", "title"):
            raw = doc.get(field, "")
            if isinstance(raw, (list, tuple)):
                raw = " ".join(str(r) for r in raw)
            raw = str(raw) if raw else ""
            if not raw or raw in ("N/A", "Unknown", "Không xác định"):
                continue
            # Split by separators: ; , / |
            parts = re.split(r"[;/|,]", raw)
            for part in parts:
                normalized = _normalize_topic_token(part)
                if normalized and len(normalized) >= 2:
                    keywords.add(normalized)

        ALL_DOC_KEYWORDS[idx] = keywords

        for kw in keywords:
            if kw not in TOPIC_CACHE:
                TOPIC_CACHE[kw] = []
            TOPIC_CACHE[kw].append(idx)

    # Also build prefix index for partial matching
    # e.g. "kinh te" matches "kinh te vi mo", "kinh te doanh nghiep"
    prefix_cache: dict[str, list[int]] = {}
    for kw, indices in TOPIC_CACHE.items():
        tokens = kw.split()
        for i in range(1, len(tokens) + 1):
            prefix = " ".join(tokens[:i])
            if prefix not in prefix_cache:
                prefix_cache[prefix] = []
            prefix_cache[prefix].extend(indices)

    # Merge prefix cache into main cache
    for prefix, indices in prefix_cache.items():
        if prefix not in TOPIC_CACHE:
            TOPIC_CACHE[prefix] = indices
        # Also add with wildcard key for matching
        TOPIC_CACHE[f"__prefix__{prefix}"] = list(set(indices))

    subjects_with_docs = sum(1 for kw in TOPIC_CACHE if not kw.startswith("__prefix__"))
    logger.info("Topic cache built: %d keywords, %d unique subjects", len(TOPIC_CACHE), subjects_with_docs)

def _search_topic_cache(query: str) -> list[dict] | None:
    """Search topic cache for matching documents. Returns None if no cache hit."""
    q = _normalize_topic_token(query)
    if not q:
        return None

    # Exact match
    if q in TOPIC_CACHE:
        doc_indices = TOPIC_CACHE[q]
        return [{"doc": documents[i], "score": 100.0, "index": i} for i in doc_indices[:120]]

    # Prefix match: "kinh te" matches "kinh te vi mo"
    prefix_key = f"__prefix__{q}"
    if prefix_key in TOPIC_CACHE:
        doc_indices = TOPIC_CACHE[prefix_key]
        return [{"doc": documents[i], "score": 100.0, "index": i} for i in doc_indices[:120]]

    # Multi-keyword: split query into individual keywords and find docs matching ALL
    words = q.split()
    if len(words) >= 2:
        # Find docs that have at least one keyword from each word
        candidate_sets = []
        for w in words:
            w_indices = set()
            for kw, indices in TOPIC_CACHE.items():
                if kw.startswith("__prefix__"):
                    continue
                if w in kw:
                    w_indices.update(indices)
            if not w_indices:
                return None  # At least one word has no matches
            candidate_sets.append(w_indices)

        # Intersection: docs matching ALL words
        common = candidate_sets[0]
        for s in candidate_sets[1:]:
            common = common & s

        if common:
            return [{"doc": documents[i], "score": 80.0, "index": i} for i in list(common)]

    return None

_build_topic_cache()

# Initialize PDF Manager
pdf_manager = PDFManager(pdf_dir="pdfs")
logger.info("PDF Manager initialized")

# ── Hàm tái index PDF vào BM25 ──────────────────────────
def reindex_pdfs(retriever):
    """Load lại tất cả PDF và cập nhật BM25 index (không ảnh hưởng dữ liệu CSV)"""
    from load_pdf import load_all_pdfs
    loaded = load_all_pdfs()
    if not loaded:
        logger.info("No PDFs to reindex")
        return 0

    existing_sources = {d.get('source') for d in retriever.bm25.documents if d.get('csv_file','').startswith('pdf:')}
    new_count = 0
    seen_hashes = {d.get('hash') for d in retriever.bm25.documents if d.get('hash')}

    for doc in loaded:
        meta = doc.metadata or {}
        h = meta.get('hash')
        if h and h in seen_hashes:
            continue
        seen_hashes.add(h)

        source_name = meta.get('source', 'pdf')
        if source_name in existing_sources:
            continue
        existing_sources.add(source_name)

        # Trích xuất tên tác giả từ tên file PDF (giống logic trong module-level load)
        pdf_author = meta.get('author', 'Unknown')
        if not pdf_author or pdf_author == 'Unknown' or 'aspose' in pdf_author.lower() or 'epubtopdfconverter' in pdf_author.lower() or 'pdf converter' in pdf_author.lower():
            file_stem = Path(source_name).stem if source_name else ""
            m = re.match(r"^([A-Z][A-Z\s\.]+?)\s*[-–]\s*", file_stem)
            if m:
                pdf_author = m.group(1).strip()

        # Subject: chỉ dùng keywords từ filename, không dùng section heading (nhiễu)
        pdf_subject = meta.get('keywords', '')
        if not pdf_subject or len(pdf_subject) < 5 or pdf_subject.lower() == 'pdf':
            pdf_subject = ''

        pdf_item = {
            'title': meta.get('title', meta.get('source', 'PDF')).strip(),
            'author': pdf_author,
            'year': meta.get('year', 'N/A'),
            'subject': pdf_subject,
            'link': '',
            'source': source_name,
            'page': meta.get('page'),
            'text': doc.page_content or '',
            'csv_file': 'pdf:' + source_name,
            'hash': h
        }
        retriever.bm25.documents.append(pdf_item)
        new_count += 1

    if new_count > 0:
        retriever.bm25._build_index()
        _build_topic_cache()
        _search_cache.invalidate()  # Xóa cache kết quả cũ
        logger.info("Reindexed %d new PDF pages into BM25 + topic cache rebuilt", new_count)
    else:
        logger.info("No new PDF pages to reindex")
    return new_count

PDF_EMPTY_TEXT_MARKERS = {
    "[Không có nội dung]",
    "[pypdf not available - cannot extract text]",
}

def has_extractable_pdf_text(text: str) -> bool:
    """Return True only when PDF parser produced meaningful text."""
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False
    if stripped in PDF_EMPTY_TEXT_MARKERS or stripped.startswith("[Lỗi:"):
        return False

    compact = re.sub(r"\s+", "", stripped)
    return len(compact) >= 50

def is_general_pdf_content_request(query: str) -> bool:
    """Detect broad questions that should use a document preview/full text."""
    q = normalize_text(query or "")
    broad_phrases = [
        "noi dung",
        "tom tat",
        "file nay",
        "pdf nay",
        "tai lieu nay",
        "van ban nay",
        "noi ve gi",
        "cho biet",
        "gioi thieu",
        "bo cuc",
        "muc luc",
        "cau truc",
        "danh muc",
        "layout",
        "structure",
        "table of contents",
    ]
    return any(phrase in q for phrase in broad_phrases)

def build_empty_pdf_answer(pdf_title: str) -> str:
    return (
        f"File PDF **{pdf_title}** không có lớp văn bản để hệ thống trích xuất nội dung. "
        "File này có thể là bản scan/ảnh hoặc bị khóa text, nên chatbot không thể đọc để trả lời theo nội dung bên trong.\n\n"
        "Cách xử lý: tạo bản PDF đã OCR hoặc upload file PDF có text layer, sau đó hỏi lại nội dung của file."
    )

def remove_pdf_from_index(file_name: str) -> int:
    """Remove a deleted PDF from the in-memory BM25 index."""
    if not file_name or not hasattr(retriever, "bm25"):
        return 0

    before = len(retriever.bm25.documents)
    retriever.bm25.documents = [
        doc for doc in retriever.bm25.documents
        if not (
            doc.get("source") == file_name
            or doc.get("csv_file") == f"pdf:{file_name}"
        )
    ]
    removed = before - len(retriever.bm25.documents)
    if removed:
        retriever.bm25._build_index()
        _build_topic_cache()
        _search_cache.invalidate()  # Xóa cache kết quả cũ
    return removed

def find_pdf_info_for_source(source: str = "", title: str = "") -> Optional[dict]:
    """Find PDF metadata for a search source returned by BM25."""
    source_norm = (source or "").strip().lower()
    title_norm = (title or "").strip().lower()
    if not source_norm and not title_norm:
        return None

    try:
        for pdf in pdf_manager.list_all_pdfs():
            candidates = [
                pdf.get("file_name", ""),
                pdf.get("file_stem", ""),
                pdf.get("title", ""),
                pdf.get("display_name", ""),
                pdf.get("original_title", ""),
            ]
            normalized_candidates = {str(c).strip().lower() for c in candidates if c}
            if source_norm and source_norm in normalized_candidates:
                return pdf
            if title_norm and title_norm in normalized_candidates:
                return pdf
    except Exception as e:
        logger.warning("Error matching PDF source: %s", e)

    return None

def answer_says_no_information(answer: str) -> bool:
    """Detect neu LLM tra ve "khong co thong tin" — mo rong de phat hien nhieu bien the."""
    if not answer:
        return True
    normalized = normalize_text(answer or "")
    no_info_phrases = [
        "tai lieu khong cung cap",
        "khong cung cap thong tin",
        "khong co thong tin",
        "khong du thong tin",
        "khong tim thay thong tin",
        "khong tim thay tai lieu",
        "khong co tai lieu",
        "khong co trong tai lieu",
        "khong duoc de cap trong tai lieu",
        "khong thay trong tai lieu",
        "khong duoc tim thay",
        "khong co du lieu",
        "khong the tra loi",
        "minh khong the",
        "xin loi toi khong",
        "xin loi minh khong",
        "khong co cau tra loi",
        "khong co phan hoi",
        "khong co thong tin cung cap",
        "toi khong tim thay",
        "minh khong tim thay",
        "chua tim thay",
        "hien tai minh chua tim thay",
    ]
    return any(phrase in normalized for phrase in no_info_phrases)

def build_related_pdf_answer(pdf_infos: list[dict], query: str) -> str:
    """Build a clear answer when search found PDFs but text content is unavailable."""
    if not pdf_infos:
        return "Tài liệu không cung cấp thông tin này."

    lines = ["Mình tìm thấy file PDF liên quan trong thư viện:"]
    for pdf in pdf_infos[:3]:
        label = pdf.get("display_name") or pdf.get("title") or pdf.get("file_stem") or pdf.get("file_name") or "PDF"
        year = pdf.get("year")
        pages = pdf.get("pages")
        details = []
        if year and year != "N/A":
            details.append(f"năm {year}")
        if pages:
            details.append(f"{pages} trang")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- **{label}**{suffix}")

    lines.append("")
    lines.append(
        "Bạn có thể bấm **Đính kèm PDF này** trong phần Tài liệu tham khảo để chọn file và hỏi tiếp theo file đó."
    )
    lines.append(
        "Lưu ý: nếu PDF là bản scan/ảnh hoặc không có lớp text, hệ thống chỉ nhận diện được file liên quan chứ chưa đọc được nội dung chi tiết cho tới khi có bản OCR/text layer."
    )
    return "\n".join(lines)

def detect_reference_request(query: str) -> bool:
    q = normalize_text(query or "")
    phrases = [
        "tai lieu tham khao",
        "danh muc tai lieu",
        "danh sach tai lieu",
        "nguon tham khao",
        "tham khao",
        "references",
        "bibliography",
        "citation",
        "trich dan",
    ]
    return any(phrase in q for phrase in phrases)

def is_reference_heading_line(line: str) -> bool:
    normalized = normalize_text(line or "").strip(" .:-")
    headings = {
        "tai lieu tham khao",
        "danh muc tai lieu tham khao",
        "references",
        "reference",
        "bibliography",
    }
    return normalized in headings

def extract_reference_section_from_pdf(pdf_info: dict, max_chars: int = 40000) -> tuple[str, Optional[int]]:
    """Extract the reference section by scanning for its heading in the selected PDF."""
    try:
        total_pages = int(pdf_info.get("pages") or 0)
    except Exception:
        total_pages = 0

    if total_pages <= 0:
        return "", None

    for page_idx in range(total_pages):
        page_text = pdf_manager.get_chapter_text(pdf_info["file_path"], page_idx, page_idx + 1)
        if not has_extractable_pdf_text(page_text):
            continue

        lines = page_text.replace("\xa0", " ").splitlines()
        heading_idx = None
        for idx, line in enumerate(lines):
            if is_reference_heading_line(line):
                heading_idx = idx
                break

        if heading_idx is None:
            continue

        preview_lines = lines[heading_idx:]
        if page_idx + 1 < total_pages:
            next_text = pdf_manager.get_chapter_text(pdf_info["file_path"], page_idx + 1, page_idx + 2)
            if has_extractable_pdf_text(next_text):
                preview_lines.extend(next_text.replace("\xa0", " ").splitlines()[:20])
        preview = "\n".join(preview_lines)
        if not re.search(r"(?m)^\s*(?:\[\s*1\s*\]|1[\.\)])", preview):
            continue

        sections = [f"[Trang PDF {page_idx + 1}]\n" + "\n".join(lines[heading_idx:]).strip()]
        for next_idx in range(page_idx + 1, total_pages):
            if len("\n\n".join(sections)) >= max_chars:
                break
            next_text = pdf_manager.get_chapter_text(pdf_info["file_path"], next_idx, next_idx + 1)
            if has_extractable_pdf_text(next_text):
                sections.append(f"[Trang PDF {next_idx + 1}]\n" + next_text.replace("\xa0", " ").strip())

        return "\n\n".join(sections)[:max_chars], page_idx + 1

    return "", None

def parse_reference_items(section_text: str, max_items: int = 20) -> list[str]:
    items = []
    current = ""

    for raw_line in (section_text or "").replace("\xa0", " ").splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if line.startswith("[Trang PDF"):
            continue
        if is_reference_heading_line(line):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue

        if re.match(r"^\[\d+\]", line):
            if current:
                items.append(current.strip())
                if len(items) >= max_items:
                    return items
            current = line
        elif current:
            current += " " + line

    if current and len(items) < max_items:
        items.append(current.strip())

    return items

def build_reference_answer(pdf_info: dict, section_text: str, start_page: Optional[int]) -> str:
    title = pdf_info.get("display_name") or pdf_info.get("title") or pdf_info.get("file_stem") or "PDF"
    items = parse_reference_items(section_text)
    page_text = f" bắt đầu ở trang PDF {start_page}" if start_page else ""

    lines = [f"Có. Trong PDF **{title}** có mục **Tài liệu tham khảo**{page_text}."]
    if items:
        lines.append("")
        lines.append("Các tài liệu tham khảo trích xuất được:")
        for item in items:
            lines.append(f"- {item}")
        if len(items) >= 20:
            lines.append("- ...")
    else:
        preview = section_text[:2500].strip()
        if preview:
            lines.append("")
            lines.append(preview)

    return "\n".join(lines)

def is_chapter_heading_line(line: str, chapter_num: int) -> bool:
    normalized = normalize_text(line or "")
    return bool(re.match(rf"^chuong\s+{chapter_num}\b", normalized))

def chapter_heading_has_body(pdf_info: dict, page_idx: int, lines_after_heading: list[str], chapter_num: int) -> bool:
    """Avoid matching table-of-contents chapter lines as real chapter starts."""
    preview_lines = list(lines_after_heading)
    try:
        total_pages = int(pdf_info.get("pages") or 0)
    except Exception:
        total_pages = 0

    if page_idx + 1 < total_pages:
        next_text = pdf_manager.get_chapter_text(pdf_info["file_path"], page_idx + 1, page_idx + 2)
        if has_extractable_pdf_text(next_text):
            preview_lines.extend(next_text.replace("\xa0", " ").splitlines()[:25])

    preview = "\n".join(preview_lines)
    return bool(re.search(rf"(?m)^\s*{chapter_num}\s*\.\s*1\b", preview))

def extract_chapter_section_from_pdf(
    pdf_info: dict,
    chapter_num: int,
    max_chars: int = 12000
) -> tuple[str, Optional[int], Optional[int]]:
    """Extract a chapter by scanning real chapter headings in the PDF text."""
    try:
        total_pages = int(pdf_info.get("pages") or 0)
    except Exception:
        total_pages = 0

    if total_pages <= 0 or chapter_num <= 0:
        return "", None, None

    start_page = None
    sections = []
    next_chapter = chapter_num + 1

    for page_idx in range(total_pages):
        page_text = pdf_manager.get_chapter_text(pdf_info["file_path"], page_idx, page_idx + 1)
        if not has_extractable_pdf_text(page_text):
            continue

        lines = page_text.replace("\xa0", " ").splitlines()

        if start_page is None:
            heading_idx = next((idx for idx, line in enumerate(lines) if is_chapter_heading_line(line, chapter_num)), None)
            if heading_idx is None:
                continue
            if re.search(r"\.{5,}", "\n".join(lines[heading_idx:heading_idx + 8])):
                continue
            if not chapter_heading_has_body(pdf_info, page_idx, lines[heading_idx:], chapter_num):
                continue
            start_page = page_idx + 1
            lines = lines[heading_idx:]
        else:
            stop_idx = next((idx for idx, line in enumerate(lines) if is_chapter_heading_line(line, next_chapter)), None)
            if stop_idx is not None:
                if stop_idx > 0:
                    sections.append(f"[Trang PDF {page_idx + 1}]\n" + "\n".join(lines[:stop_idx]).strip())
                return "\n\n".join(sections)[:max_chars], start_page, page_idx + 1

        if lines:
            sections.append(f"[Trang PDF {page_idx + 1}]\n" + "\n".join(lines).strip())

        if len("\n\n".join(sections)) >= max_chars:
            return "\n\n".join(sections)[:max_chars], start_page, page_idx + 1

    if sections:
        return "\n\n".join(sections)[:max_chars], start_page, total_pages

    return "", None, None

# ── Schema ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    document_id: str = None
    session_id: str = "default"
    major: str = None  # Optional filter by major/ngành (uses 526$a field)

# In-memory session tracking (lưu 3 query gần nhất)
session_history: dict[str, list[dict]] = {}
MAX_HISTORY = 3

# ── Search Result Cache (LRU, TTL-based) ─────────────────
import time
from collections import OrderedDict

class SearchCache:
    """LRU cache with TTL for search results to avoid redundant computation."""
    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self._cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> list[dict] | None:
        if key in self._cache:
            ts, results = self._cache[key]
            if time.time() - ts < self._ttl:
                self._cache.move_to_end(key)
                self._hits += 1
                return results
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def set(self, key: str, results: list[dict]):
        if key in self._cache:
            del self._cache[key]
        elif len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)  # evict oldest
        self._cache[key] = (time.time(), results)

    def invalidate(self):
        """Xóa toàn bộ cache — gọi khi data thay đổi."""
        n = len(self._cache)
        self._cache.clear()
        if n > 0:
            logger.info("Search cache invalidated (%d entries cleared)", n)

    def stats(self) -> str:
        total = self._hits + self._misses
        rate = (self._hits / total * 100) if total else 0
        return f"cache={len(self._cache)} hits={self._hits} misses={self._misses} rate={rate:.0f}%"

_search_cache = SearchCache(max_size=256, ttl_seconds=300)


# Giai đoạn 3 — Hybrid search (BM25 + semantic) với fallback thông minh
def hybrid_search_with_semantic_fallback(
    query: str,
    bm25_top_k: int = 20,
    final_top_k: int = 10,
    alpha: float = 0.5,
    min_bm25_score: float = 0.5,
) -> tuple[list[dict], str]:
    """
    Tìm kiếm hybrid:
    - Luôn chạy BM25 trước (nhanh, deterministic)
    - Nếu BM25 score thấp (< min_bm25_score) → bật semantic để bổ sung
    - Nếu semantic có sẵn và BM25 trả về ít → kết hợp

    Returns: (results, mode)
        mode: "bm25" | "hybrid" | "semantic"
    """
    bm25_results = bm25_engine.search(query, top_k=bm25_top_k)

    # Nếu BM25 trả về tốt → dùng luôn
    if bm25_results and len(bm25_results) >= 5:
        top_score = bm25_results[0].get("score", 0) if bm25_results else 0
        if top_score >= min_bm25_score:
            return bm25_results[:final_top_k], "bm25"

    # Thử semantic bổ sung
    sem_engine = get_semantic_engine()
    if not sem_engine or sem_engine.embeddings is None:
        return bm25_results[:final_top_k] if bm25_results else [], "bm25"

    # Hybrid: combine BM25 + semantic
    sem_results = sem_engine.search(query, top_k=bm25_top_k)
    if not sem_results:
        return bm25_results[:final_top_k] if bm25_results else [], "bm25"

    # Merge scores
    bm25_dict = {id(r["doc"]): r for r in bm25_results}
    combined = []

    for sr in sem_results:
        doc = sr["doc"]
        sem_score = sr["score"]
        bm25_score = 0
        if id(doc) in bm25_dict:
            bm25_score = bm25_dict[id(doc)].get("score", 0)

        # Normalize BM25
        max_bm25 = max((r.get("score", 0) for r in bm25_results), default=1) or 1
        bm25_norm = bm25_score / max_bm25

        final = alpha * sem_score + (1 - alpha) * bm25_norm
        combined.append({
            "doc": doc,
            "score": final,
            "bm25_score": bm25_score,
            "semantic_score": sem_score,
        })

    # Sort và lấy top_k
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:final_top_k], "hybrid"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    total_sources: int = 0
    total_found: int = 0  # Tổng số tài liệu tìm thấy (cho phân trang)
    summary: str = ""  # Tóm tắt nếu user yêu cầu
    current_document: Optional[dict] = None
    follow_up_suggestions: list[dict] = []  # Gợi ý câu hỏi tiếp theo [{label, query}]

# ── Helpers ───────────────────────────────────────────────
def format_docs_with_metadata(search_results):
    """Format BM25 search results with metadata"""
    formatted = []
    for i, result in enumerate(search_results, 1):
        doc = result['doc']
        title = doc.get('title', 'Tài liệu không có tên')
        author = doc.get('author', '')
        year = doc.get('year', '')
        subject = doc.get('subject', '')
        link = doc.get('link', '')
        source = doc.get('source', '')
        doc_type = doc.get('doc_type', '')
        doc_format = doc.get('format', 'Số')
        major = doc.get('major', '')
        publisher = doc.get('publisher', '')
        abstract = doc.get('abstract', '')
        location = doc.get('location', '')
        notes = doc.get('notes', '')
        ddc = doc.get('ddc', '')
        text = doc.get('text', '')
        score = result.get('score', 0)
        
        # Catalog records use metadata text; PDF page records must include page text
        # so content questions can be answered from the document itself.
        def valid(v): return v and v not in ('N/A', 'Unknown', 'Không xác định')
        is_pdf_page = str(doc.get('csv_file', '')).startswith('pdf:') or doc.get('page') is not None
        if is_pdf_page and text:
            page_label = f" - Trang {doc.get('page')}" if doc.get('page') else ""
            content = f"[{title}{page_label}]\n{text[:6000]}"
        else:
            clean = title
            if valid(author): clean += f" - {author}"
            if valid(publisher): clean += f" ({publisher})"
            if valid(abstract): clean += f"\n\nTóm tắt: {abstract[:800]}"
            content = clean[:3000]
        
        formatted.append({
            "text": content,
            "metadata": {
                "title": title,
                "author": author,
                "year": year,
                "subject": subject,
                "link": link,
                "source": source,
                "doc_type": doc_type,
                "format": doc_format,
                "major": major,
                "publisher": publisher,
                "abstract": abstract[:300] if abstract else '',
                "location": location[:200] if location else '',
                "notes": notes,
                "ddc": ddc,
            },
            "score": score
        })
    
    # Kết hợp tất cả
    combined_text = "\n\n---\n\n".join([f["text"] for f in formatted])
    return combined_text, formatted

def format_docs(search_results):
    """Simple format: combine text from search results (clean, not raw concatenation)"""
    texts = []
    for result in search_results:
        doc = result['doc']
        t = doc.get('title', '')
        a = doc.get('author', '')
        p = doc.get('publisher', '')
        ab = doc.get('abstract', '')
        parts = [t]
        if a and a not in ('Unknown', 'N/A', 'Không xác định'): parts.append(f"Tác giả: {a}")
        if p and p not in ('Unknown', 'N/A', 'Không xác định'): parts.append(f"Nhà xuất bản: {p}")
        if ab: parts.append(f"Tóm tắt: {ab[:300]}")
        texts.append(" | ".join(parts))
    return "\n\n".join(texts)

def detect_summary_request(query: str) -> bool:
    """Detect nếu user yêu cầu tóm tắt"""
    keywords = ["tóm tắt", "tom tat", "summary", "tóm", "tom", "résumé", "概括", "summarize"]
    q = query.lower()
    return any(kw in q for kw in keywords)


# Stop words tiếng Việt đầy đủ
VIETNAMESE_STOP_WORDS = {
    # Đại từ nhân xưng
    "toi", "minh", "em", "anh", "chi", "co", "chu", "bac", "ong", "ba",
    "ai", "tao", "may", "no", "ho", "chung", "ban", "cau", "cac", "nguoi",
    # Từ hỏi, liên từ, giới từ
    "co", "khong", "ve", "la", "cua", "va", "con", "thi", "nao",
    "sao", "gi", "dau", "bao", "nhieu", "the", "nay", "do", "ay",
    "the nao", "nhu the nao", "ma", "neu", "vi", "nen", "hoac",
    "hay", "song", "tu", "o", "tai", "voi", "trong", "ngoai",
    "tren", "duoi", "ben", "giua", "khi", "luc", "sau", "truoc",
    # Tình thái từ, trợ từ
    "qua", "len", "xuong", "di", "lai", "vao", "ra", "luon",
    "cung", "da", "dang", "se", "sap", "vua", "moi", "con",
    "lam", "rat", "nhe", "nha", "a", "u", "vay", "the",
    # Từ chức năng thường gặp trong câu hỏi thư viện
    "sach", "tai", "lieu", "giao", "trinh", "bai", "bao",
    "tim", "xem", "hoi", "thay", "cho", "gui", "xin",
    "ban", "doc", "tap", "quyen", "cuon",
    "thu", "vien", "qnu", "truong", "dai", "hoc",
    "muon", "de", "den", "lien", "quan", "linh", "vuc",
    "goi", "y", "phu", "hop", "chu", "de", "nganh", "linh-vuc",
    # Số từ
    "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi",
    "tram", "nghin", "trieu", "ty",
}

# These normalized tokens are ambiguous after accent removal. Keep them so
# compound subjects such as "co khi", "giao duc", "hoc may", "quan tri",
# and "nhan tao" are not destroyed by stop-word filtering.
for compound_term in ("co", "chu", "giao", "hoc", "khi", "may", "quan", "tao"):
    VIETNAMESE_STOP_WORDS.discard(compound_term)

QUERY_INTENT_PATTERNS = [
    r"\b(?:toi|minh|em|anh|chi|ban)\s+muon\b",
    r"\b(?:cho toi|cho minh|cho em)\b",
    r"\b(?:hay|vui long)?\s*goi y\b",
    r"\btim kiem\b",
    r"\b(?:tim|tra cuu|liet ke)\b",
    r"\b(?:co nhung|co the|co)\s+(?:cac\s+)?(?:sach|tai lieu|giao trinh)\s+(?:nao\s+)?(?:ve|lien quan den)?\b",
    r"\b(?:cac\s+)?(?:sach|tai lieu|giao trinh)\s+(?:lien quan den|ve)?\b",
    r"\b(?:lam\s+)?de tai\s+(?:lien quan den|ve)?\b",
    r"\blien quan den\b",
    r"\blien quan\b",
    r"\blinh vuc\b",
    r"\bchu de\b",
    r"\bnganh\b",
    r"\btham khao\b",
    r"\bphu hop\b",
    r"\b(?:sach|tai lieu|giao trinh|bai bao|bai)\s+cua\b",
    r"\bcua\s+tac gia\b",
    r"\bcua\s+nha xuat ban\b",
    r"\bcua\s+nxb\b",
]

TOPIC_MARKER_PATTERNS = [
    r"\b(?:lien quan den|ve|linh vuc|chu de|nganh)\s+(.+)$",
]

TOPIC_TRAILING_PATTERNS = [
    r"\bgoi y\b.*$",
    r"\bcac\s+(?:linh\s+vuc|chu\s+de|nganh)\s+do.*$",
    r"\bco\s+(?:sach|tai lieu|giao trinh)\s+nao.*$",
    r"\b(?:sach|tai lieu|giao trinh)\s+(?:nao\s+)?(?:lien quan|phu hop).*$",
    r"\b(?:lien quan|phu hop)\s*(?:khong|ko|k)?\s*$",
    r"\b(?:khong|ko|k)\s*$",
]

DOMAIN_PHRASE_EXPANSIONS = {
    "y khoa": ["y khoa", "y hoc", "noi khoa", "ngoai khoa", "lam sang", "duoc", "benh", "sinh ly", "giai phau"],
    "y hoc": ["y hoc", "y khoa", "noi khoa", "duoc", "benh", "suc khoe", "sinh ly", "giai phau"],
    "y te": ["y te", "y hoc", "y khoa", "suc khoe", "benh vien", "duoc"],
    "thu y": ["thu y", "veterinary", "chan nuoi", "dong vat", "benh dong vat"],
}

CATALOG_TOPIC_INDEX: dict[str, dict] = {}
CATALOG_TOPIC_READY = False
CATALOG_TOPIC_INDEX_VERSION = 2

CATALOG_FIELD_WEIGHTS = {
    "title": 5.0,
    "subject": 10.0,
    "major": 9.0,
    "course": 8.0,
    "doc_type": 4.0,
    "publisher": 2.0,
    "author": 7.0,
}

GENERIC_CATALOG_PHRASES = {
    "sach",
    "tai lieu",
    "tai lieu tham khao",
    "giao trinh",
    "bai giang",
    "de an",
    "luan van",
    "luan an",
    "tham khao",
    "nxb",
    "nha xuat ban",
    "dai hoc",
    "truong dai hoc",
    "truong dai hoc quy nhon",
    "khoa hoc",
    "nghien cuu",
    "phuong phap",
    "phuong phap nghien cuu",
    "nghien cuu khoa hoc",
    "co file",
    "khong co file",
    "loi file",
    "dang xu ly",
}


def normalize_topic_text(text: str) -> str:
    """Normalize search/catalog text and treat punctuation as word separators."""
    normalized = normalize_text(text)
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def topic_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_topic_text(text))


def is_valid_catalog_phrase(phrase: str) -> bool:
    phrase = normalize_topic_text(phrase)
    if not phrase or phrase in GENERIC_CATALOG_PHRASES:
        return False
    tokens = phrase.split()
    if not tokens or len(tokens) > 7:
        return False
    if all(token.isdigit() for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        return len(token) >= 3 and token not in VIETNAMESE_STOP_WORDS

    return any(re.search(r"[a-zA-Z]", token) for token in tokens)


def split_catalog_segments(value: str, field_name: str) -> list[str]:
    if not value or value in ("Unknown", "N/A", "nan"):
        return []

    if field_name in {"subject", "major", "course", "doc_type"}:
        parts = re.split(r"[;#|,\n]+", str(value))
    elif field_name == "title":
        parts = re.split(r"[:;#|\n]+", str(value))
    elif field_name in {"author", "publisher"}:
        # Split by comma, semicolon, newline for multi-author/publisher values
        parts = re.split(r"[,;|\n]+", str(value))
    else:
        parts = [str(value)]

    return [part.strip() for part in parts if part and part.strip()]


def generate_catalog_phrases(value: str, field_name: str) -> set[str]:
    phrases = set()
    for segment in split_catalog_segments(value, field_name):
        normalized_segment = normalize_topic_text(segment)
        if is_valid_catalog_phrase(normalized_segment):
            phrases.add(normalized_segment)

        tokens = normalized_segment.split()
        if field_name in {"title", "subject", "major", "course", "author", "publisher"} and len(tokens) >= 2:
            max_n = min(5, len(tokens))
            for n in range(2, max_n + 1):
                for start in range(0, len(tokens) - n + 1):
                    phrase = " ".join(tokens[start:start + n])
                    if is_valid_catalog_phrase(phrase):
                        phrases.add(phrase)
        elif field_name in {"subject", "major", "course", "author", "publisher"}:
            for token in tokens:
                if is_valid_catalog_phrase(token):
                    phrases.add(token)

    return phrases


def add_catalog_phrase(phrase: str, field_name: str, doc_idx: int) -> None:
    item = CATALOG_TOPIC_INDEX.setdefault(
        phrase,
        {"score": 0.0, "fields": set(), "docs": set()}
    )
    item["score"] += CATALOG_FIELD_WEIGHTS.get(field_name, 1.0)
    item["fields"].add(field_name)
    item["docs"].add(doc_idx)


def get_catalog_source_mtime() -> float:
    data_path = Path("Data")
    source_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.xlsx"))
    if not source_files:
        return 0.0
    return max(path.stat().st_mtime for path in source_files)


def load_catalog_topic_cache(cache_file: Path, source_mtime: float) -> bool:
    global CATALOG_TOPIC_READY
    if not cache_file.exists():
        return False

    try:
        with open(cache_file, "rb") as f:
            payload = pickle.load(f)
        if payload.get("version") != CATALOG_TOPIC_INDEX_VERSION:
            return False
        if not is_vercel_runtime() and payload.get("source_mtime", 0.0) < source_mtime:
            return False
        index = payload.get("index")
        if not isinstance(index, dict):
            return False
        CATALOG_TOPIC_INDEX.clear()
        CATALOG_TOPIC_INDEX.update(index)
        CATALOG_TOPIC_READY = True
        logger.info("Loaded %d catalog topic phrases from cache", len(CATALOG_TOPIC_INDEX))
        return True
    except Exception as e:
        logger.warning("Catalog topic cache load failed: %s", e)
        return False


def save_catalog_topic_cache(cache_file: Path, source_mtime: float) -> None:
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "version": CATALOG_TOPIC_INDEX_VERSION,
                    "source_mtime": source_mtime,
                    "index": CATALOG_TOPIC_INDEX,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception as e:
        logger.warning("Catalog topic cache write failed: %s", e)


def build_catalog_topic_index() -> None:
    global CATALOG_TOPIC_READY
    if CATALOG_TOPIC_READY:
        return

    cache_file = Path("Data") / "_topic_index.pkl"
    source_mtime = get_catalog_source_mtime()
    if load_catalog_topic_cache(cache_file, source_mtime):
        return

    CATALOG_TOPIC_INDEX.clear()
    catalog_fields = ("subject", "major", "course", "doc_type", "title", "publisher", "author")
    source_docs = retriever.bm25.documents if hasattr(retriever, "bm25") else documents
    for doc_idx, doc in enumerate(source_docs):
        if str(doc.get("csv_file", "")).startswith("pdf:"):
            continue
        for field_name in catalog_fields:
            for phrase in generate_catalog_phrases(doc.get(field_name, ""), field_name):
                add_catalog_phrase(phrase, field_name, doc_idx)

    CATALOG_TOPIC_READY = True
    logger.info("Catalog topic index ready: %d phrases", len(CATALOG_TOPIC_INDEX))
    save_catalog_topic_cache(cache_file, source_mtime)


def find_catalog_topic_matches(query_text: str, limit: int = 8) -> list[str]:
    build_catalog_topic_index()
    normalized_query = normalize_topic_text(query_text)
    tokens = normalized_query.split()
    if not tokens:
        return []

    candidates = set()
    if is_valid_catalog_phrase(normalized_query):
        candidates.add(normalized_query)
    max_n = min(7, len(tokens))
    for n in range(1, max_n + 1):
        for start in range(0, len(tokens) - n + 1):
            phrase = " ".join(tokens[start:start + n])
            if phrase in CATALOG_TOPIC_INDEX:
                candidates.add(phrase)

    def rank_key(phrase: str):
        item = CATALOG_TOPIC_INDEX.get(phrase, {})
        exact_bonus = 1_000_000.0 if phrase == normalized_query else 0.0
        length_bonus = len(phrase.split()) * 500.0
        field_bonus = 80.0 if {"subject", "major", "course", "author"} & set(item.get("fields", set())) else 0.0
        return exact_bonus + length_bonus + field_bonus + item.get("score", 0.0)

    ranked = sorted(candidates, key=rank_key, reverse=True)
    return ranked[:limit]


def strip_search_intent_phrases(text: str) -> str:
    """Remove request wording and keep the actual topic terms."""
    q = normalize_topic_text(text)
    for pattern in QUERY_INTENT_PATTERNS:
        q = re.sub(pattern, " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()


def clean_topic_tail(topic: str) -> str:
    topic = topic or ""
    for pattern in TOPIC_TRAILING_PATTERNS:
        topic = re.sub(pattern, " ", topic)
    topic = strip_search_intent_phrases(topic)
    return re.sub(r"\s+", " ", topic).strip()


def _extract_major_from_query(query: str) -> str | None:
    """Try to extract a major/ngành from a natural-language query.

    Looks for patterns like ``ngành X``, ``lĩnh vực X``, ``chủ đề X``,
    ``về X`` (when X looks like a noun phrase). The returned value is the
    raw (un-normalized) form so the LLM sees a human-readable major.
    Returns ``None`` when no major can be confidently detected.
    """
    if not query:
        return None
    q_norm = normalize_topic_text(query)
    captured: str | None = None
    for pattern in TOPIC_MARKER_PATTERNS:
        m = re.search(pattern, q_norm)
        if m:
            captured = m.group(1)
            break
    if not captured:
        return None
    cleaned = clean_topic_tail(captured)
    if not cleaned:
        return None
    # Reject if everything is stop-words
    tokens = [t for t in cleaned.split() if t not in VIETNAMESE_STOP_WORDS]
    if not tokens:
        return None
    # Recover the original (un-normalized) phrase so the LLM sees a
    # human-readable major. Map each normalized token back to the
    # original form in the user query when possible.
    norm_to_orig: dict[str, str] = {}
    for src in re.findall(r"[A-Za-zÀ-ỹ]+", query):
        norm_to_orig.setdefault(normalize_topic_text(src), src)
    pieces = [norm_to_orig.get(t, t) for t in cleaned.split()]
    return " ".join(pieces)


def extract_primary_topic_text(text: str) -> str:
    """Prefer the actual subject after markers like 'về', 'lĩnh vực', 'ngành'."""
    q = normalize_topic_text(text)
    best = ""
    for pattern in TOPIC_MARKER_PATTERNS:
        matches = list(re.finditer(pattern, q))
        if matches:
            best = matches[-1].group(1)
            break

    if best:
        cleaned = clean_topic_tail(best)
        if cleaned:
            return cleaned

    return strip_search_intent_phrases(text)

AI_QUERY_PATTERNS = [
    r"\btri tue nhan tao\b",
    r"\bartificial intelligence\b",
    r"\bmachine learning\b",
    r"\bhoc may\b",
    r"\bdeep learning\b",
    r"\b(?:linh vuc|chu de|de tai|cong nghe|ung dung|nganh|tai lieu|sach|giao trinh|nghien cuu|lien quan den)\s+(?:\w+\s+){0,3}ai\b",
    r"\bai\s+(?:trong|cho|ve|ung dung|giao duc|day hoc|may tinh|du lieu|ngon ngu|network|xam nhap)\b",
]

AI_EXPANSION_TERMS = [
    "AI",
    "tri tue nhan tao",
    "artificial intelligence",
    "ung dung AI",
    "hoc may",
    "machine learning",
    "deep learning",
    "khoa hoc may tinh",
]

AI_DOCUMENT_PHRASES = [
    "tri tue nhan tao",
    "artificial intelligence",
    "machine learning",
    "hoc may",
    "deep learning",
    "ung dung ai",
    "cong nghe ai",
    "ky thuat ai",
    "khoa hoc may tinh",
]


def detect_ai_domain(query: str) -> bool:
    """Detect when AI is meant as the technology field, not the Vietnamese word 'ai'."""
    if not query:
        return False

    if re.search(r"\bAI\b", query):
        return True

    q = normalize_topic_text(query)
    if q == "ai":
        return True
    return any(re.search(pattern, q) for pattern in AI_QUERY_PATTERNS)


def expand_domain_terms(query: str) -> list[str]:
    """Add domain synonyms/acronyms so catalog questions retrieve by subject, not intent words."""
    terms = []
    if detect_ai_domain(query):
        terms.extend(AI_EXPANSION_TERMS)

    topic = extract_primary_topic_text(query)
    for phrase, expansions in DOMAIN_PHRASE_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", topic):
            terms.extend(expansions)

    return list(dict.fromkeys(terms))


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    """Whole-phrase match on already-normalized text."""
    if not text or not phrase:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))


def score_ai_document(doc: dict) -> float:
    """Prefer documents that explicitly describe the AI field."""
    fields = [
        doc.get("title", ""),
        doc.get("subject", ""),
        doc.get("major", ""),
        doc.get("course", ""),
        doc.get("abstract", ""),
        doc.get("notes", ""),
        doc.get("doc_type", ""),
    ]
    raw_text = " ".join(str(field) for field in fields if field)
    normalized = normalize_text(raw_text)

    score = 0.0
    if re.search(r"\bAI\b", raw_text):
        score += 8.0
    if "trí tuệ nhân tạo" in raw_text.lower():
        score += 8.0
    for phrase in AI_DOCUMENT_PHRASES:
        if contains_normalized_phrase(normalized, phrase):
            score += 5.0
    return score


def get_doc_topic_text(doc: dict) -> str:
    """Normalized catalog fields used for topic matching."""
    cached = doc.get("_topic_search_text")
    if cached:
        return cached

    fields = get_doc_topic_fields(doc)
    text = " ".join(field for field in fields.values() if field)
    doc["_topic_search_text"] = text
    return text


def get_doc_topic_fields(doc: dict) -> dict[str, str]:
    cached = doc.get("_topic_search_fields")
    if cached:
        return cached

    fields = [
        ("title", doc.get("title", "")),
        ("author", doc.get("author", "")),
        ("subject", doc.get("subject", "")),
        ("major", doc.get("major", "")),
        ("course", doc.get("course", "")),
        ("publisher", doc.get("publisher", "")),
        ("abstract", doc.get("abstract", "")),
        ("notes", doc.get("notes", "")),
        ("doc_type", doc.get("doc_type", "")),
        # Nội dung PDF/text snippet → giúp rerank match trên nội dung thực tế
        ("text_preview", (doc.get("text") or "")[:500]),
    ]
    normalized_fields = {
        name: normalize_topic_text(str(value))
        for name, value in fields
        if value
    }
    doc["_topic_search_fields"] = normalized_fields
    return normalized_fields


def get_query_topic_terms(query: str) -> tuple[str, list[str], list[str]]:
    topic = extract_primary_topic_text(query)
    catalog_matches = find_catalog_topic_matches(topic)
    if not catalog_matches:
        catalog_matches = find_catalog_topic_matches(strip_search_intent_phrases(query))
    expanded_terms = expand_domain_terms(query)
    phrase_candidates = catalog_matches + [topic] + expanded_terms
    phrases = []
    for phrase in phrase_candidates:
        normalized_phrase = normalize_topic_text(phrase)
        if normalized_phrase and (len(normalized_phrase.split()) > 1 or len(normalized_phrase) > 2):
            phrases.append(normalized_phrase)

    tokens = [
        token for token in re.findall(r"\w+", normalize_topic_text(" ".join(phrase_candidates)))
        if token not in VIETNAMESE_STOP_WORDS and len(token) >= 2
    ]
    return topic, list(dict.fromkeys(tokens)), list(dict.fromkeys(phrases))


def metadata_search_by_query(query: str, top_k: int = 80) -> list[dict]:
    """Catalog-field search for list/recommendation questions."""
    if not hasattr(retriever, "bm25"):
        return []

    topic, topic_tokens, topic_phrases = get_query_topic_terms(query)
    if not topic_tokens and not topic_phrases:
        return []

    primary_phrase = normalize_topic_text(topic)
    scored = []
    for idx, doc in enumerate(retriever.bm25.documents):
        doc_fields = get_doc_topic_fields(doc)
        doc_text = " ".join(field for field in doc_fields.values() if field)
        score = 0.0
        exact_phrase_hit = False

        for phrase in topic_phrases:
            if not phrase:
                continue
            field_weights = {
                "title": 42.0,
                "author": 35.0,
                "subject": 36.0,
                "major": 34.0,
                "course": 28.0,
                "publisher": 16.0,
                "abstract": 12.0,
                "notes": 10.0,
                "doc_type": 8.0,
            }
            for field_name, field_text in doc_fields.items():
                if contains_normalized_phrase(field_text, phrase):
                    exact_phrase_hit = True
                    base = field_weights.get(field_name, 8.0)
                    score += base if phrase == primary_phrase else base * 0.55

        match_count = sum(
            1 for token in topic_tokens
            if re.search(rf"\b{re.escape(token)}\b", doc_text)
        )
        if match_count:
            coverage = match_count / max(len(topic_tokens), 1)
            score += (match_count * 2.0) + (coverage * 6.0)

            # For multi-word subjects, avoid returning broad method/science books
            # when only one weak token matched.
            if len(topic_tokens) >= 2 and coverage < 0.45 and not exact_phrase_hit:
                continue

        if score > 0:
            scored.append({"doc": doc, "score": score + 100000.0, "index": idx})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def merge_search_results(*groups: list[dict]) -> list[dict]:
    merged = {}
    for group in groups:
        for result in group or []:
            doc = result.get("doc", {})
            key = (
                str(doc.get("title", "")).lower().strip(),
                str(doc.get("author", "")).lower().strip(),
                str(doc.get("year", "")).lower().strip(),
            )
            if key not in merged or result.get("score", 0.0) > merged[key].get("score", 0.0):
                merged[key] = result

    return sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)


def collapse_catalog_pdf_pages(search_results: list[dict]) -> list[dict]:
    """Represent matching PDF pages as one document in catalog responses."""
    regular_results = []
    grouped_pdfs = {}

    for result in search_results:
        doc = result.get("doc", {})
        is_pdf_page = str(doc.get("csv_file", "")).startswith("pdf:") or doc.get("page") is not None
        if not is_pdf_page:
            regular_results.append(result)
            continue

        pdf_info = find_pdf_info_for_source(doc.get("source", ""), doc.get("title", ""))
        source_name = (
            (pdf_info or {}).get("file_name")
            or doc.get("source")
            or str(doc.get("csv_file", "")).removeprefix("pdf:")
        )
        group_key = str((pdf_info or {}).get("id") or source_name or doc.get("title", "")).lower()
        existing = grouped_pdfs.get(group_key)
        if existing and existing.get("score", 0.0) >= result.get("score", 0.0):
            continue

        display_title = (
            (pdf_info or {}).get("display_name")
            or (pdf_info or {}).get("title")
            or (pdf_info or {}).get("file_stem")
            or doc.get("title")
            or source_name
            or "Tài liệu PDF"
        )
        author = str(doc.get("author", "") or "")
        if any(marker in normalize_text(author) for marker in ("aspose", "pdf converter", "epubtopdfconverter")):
            author = ""

        grouped_doc = {
            **doc,
            "title": display_title,
            "author": author,
            "subject": "",
            "source": source_name,
            "csv_file": f"pdf:{source_name}" if source_name else doc.get("csv_file", ""),
            "doc_type": doc.get("doc_type") or "PDF",
            "format": "Số",
            "text": "",
        }
        grouped_doc.pop("page", None)
        grouped_pdfs[group_key] = {**result, "doc": grouped_doc}

    collapsed = regular_results + list(grouped_pdfs.values())
    # Boost PDF results so they appear first (same score, PDF wins)
    for item in collapsed:
        doc = item.get("doc", {})
        is_pdf = str(doc.get("csv_file", "")).startswith("pdf:") or doc.get("doc_type") == "PDF"
        if is_pdf:
            item["score"] = item.get("score", 0.0) + 10000.0
    return sorted(collapsed, key=lambda item: item.get("score", 0.0), reverse=True)


# ── Author / Publisher Query Detection ──────────────────────

def detect_author_query(query: str) -> bool:
    """Detect nếu câu hỏi đang tìm tài liệu theo tác giả."""
    if not query:
        return False
    q = normalize_text(query)
    # Explicit markers — always detect
    author_markers = [
        r"\btac gia\b",
        r"\btac gia\s+(?:la|co ten|ten)\b",
        r"\bcua\s+tac gia\b",
        r"\bsach\s+cua\s+tac gia\b",
        r"\btai lieu\s+cua\s+tac gia\b",
        r"\bgiao trinh\s+cua\s+tac gia\b",
    ]
    if any(re.search(pattern, q) for pattern in author_markers):
        return True
    # Pattern: "sách của [tên]" — không cần "tác giả"
    # Ví dụ: "sách của đỗ vũ nhật linh", "tài liệu của nguyễn văn a"
    author_ref_patterns = [
        r"\bsach\s+cua\s+(?P<name>[a-z]+\s+[a-z]+(?:\s+[a-z]+)*)",
        r"\btai lieu\s+cua\s+(?P<name>[a-z]+\s+[a-z]+(?:\s+[a-z]+)*)",
        r"\bgiao trinh\s+cua\s+(?P<name>[a-z]+\s+[a-z]+(?:\s+[a-z]+)*)",
        r"\bbai(?:\s+bao)?\s+cua\s+(?P<name>[a-z]+\s+[a-z]+(?:\s+[a-z]+)*)",
        r"\b(?:cua|tim)\s+(?P<name>[a-z]+\s+[a-z]+(?:\s+[a-z]+)*)(?:\s+khong|\s+ko|\s+ne|\s+nhé|\s+nha|\s+di|\?)?$",
    ]
    for pattern in author_ref_patterns:
        m = re.search(pattern, q)
        if m:
            name = m.group("name").strip()
            # Check if name looks like a Vietnamese name (surname + at least 1 more word)
            VIETNAMESE_SURNAMES = {
                "nguyen", "tran", "le", "pham", "hoang", "huynh", "vo", "dang",
                "bui", "do", "ngo", "duong", "ly", "doan", "dinh", "trinh",
                "nhat", "quach", "mau", "lai", "son", "cu", "tieu",
                "cao", "mac", "ha", "kieu", "tang", "dong", "bac", "kha",
            }
            name_tokens = name.split()
            if len(name_tokens) >= 2 and name_tokens[0] in VIETNAMESE_SURNAMES:
                return True
    # Heuristic: detect name-like queries WITHOUT "tác giả" keyword
    # Only match if query has common Vietnamese surname as first word
    VIETNAMESE_SURNAMES = {
        "nguyen", "tran", "le", "pham", "hoang", "huynh", "vo", "dang",
        "bui", "do", "ngo", "duong", "ly", "doan", "dinh", "trinh",
        "nhat", "quach", "mau", "lai", "son", "cu", "tieu",
        "cao", "mac", "ha", "kieu", "tang", "dong", "bac", "kha",
    }
    words = q.split()
    if 2 <= len(words) <= 4 and words[0] in VIETNAMESE_SURNAMES:
        # Check no academic/topic words in remaining tokens
        topic_words = {
            "te", "vi", "mo", "hoc", "phap", "luat", "su", "hoa",
            "van", "nghe", "thuat", "cong", "nghiep", "xay", "dung",
            "nong", "kinh", "tai", "chinh", "ngan", "giao", "duc",
            "tam", "ly", "xa", "hoi", "tri", "tue", "nhan", "tao",
            "may", "tinh", "phan", "mem", "dien", "tu", "vien",
            "thong", "bao", "chi", "truyen", "sinh", "y", "duoc",
            "ky", "thuat", "nganh", "chu", "de", "linh", "vuc",
        }
        if not any(w in topic_words for w in words[1:]):
            return True
    return False


def detect_publisher_query(query: str) -> bool:
    """Detect nếu câu hỏi đang tìm tài liệu theo nhà xuất bản."""
    if not query:
        return False
    q = normalize_text(query)
    publisher_markers = [
        r"\bnha xuat ban\b",
        r"\bnxb\b",
        r"\bnha xuat ban\s+(?:la|co ten|ten)\b",
        r"\bcua\s+nha xuat ban\b",
        r"\bsach\s+cua\s+nxb\b",
    ]
    return any(re.search(pattern, q) for pattern in publisher_markers)


def extract_author_name(query: str) -> str:
    """Trích xuất tên tác giả từ câu hỏi như 'sách của tác giả Nguyễn Văn A'."""
    if not query:
        return ""
    # Normalize to handle both có dấu và không dấu
    q = normalize_text(query).strip()

    VIETNAMESE_SURNAMES = {
        "nguyen", "tran", "le", "pham", "hoang", "huynh", "vo", "dang",
        "bui", "do", "ngo", "duong", "ly", "doan", "dinh", "trinh",
        "nhat", "quach", "mau", "lai", "son", "cu", "tieu",
        "cao", "mac", "ha", "kieu", "tang", "dong", "bac", "kha",
    }
    topic_words = {
        "te", "vi", "mo", "hoc", "phap", "luat", "su", "hoa",
        "van", "nghe", "thuat", "cong", "nghiep", "xay", "dung",
        "nong", "kinh", "tai", "chinh", "ngan", "giao", "duc",
        "tam", "ly", "xa", "hoi", "tri", "tue", "nhan", "tao",
        "may", "tinh", "phan", "mem", "dien", "tu", "vien",
        "thong", "bao", "chi", "truyen", "sinh", "y", "duoc",
        "ky", "thuat", "nganh", "chu", "de", "linh", "vuc",
    }

    # Pattern 1: Có từ "tác giả"
    patterns = [
        r"(?:sach|cua|tai lieu|giao trinh)\s+tac gia\s+(.+)",
        r"tac gia\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            name = match.group(1).strip().rstrip(",. ")
            name = re.sub(r"\s+(?:khong|ko|lam on|hay|vui long|toi|minh|xin|cho|gui)\b.*$", "", name)
            name = name.strip().rstrip(",. ")
            if len(re.findall(r'\w+', name)) >= 2 and len(name) >= 5:
                return name

    # Pattern 2: "sách của [tên]" — không cần "tác giả"
    name_ref_patterns = [
        r"(?:sach|tai lieu|giao trinh|bai bao|bai)\s+cua\s+(.+?)(?:\s+khong|\s+ko|\s+ne|\s+nha|\?)?$",
        r"cua\s+(.+?)(?:\s+khong|\s+ko|\?)?$",
    ]
    for pattern in name_ref_patterns:
        match = re.search(pattern, q)
        if match:
            name = match.group(1).strip().rstrip(",. ")
            # Remove trailing noise words
            name = re.sub(r"\s+(?:khong|ko|lam on|hay|vui long|toi|minh|xin|cho|gui)\b.*$", "", name)
            name = name.strip().rstrip(",. ?")
            name_tokens = name.split()
            # Name must look Vietnamese: surname + at least 1 more word
            if len(name_tokens) >= 2 and name_tokens[0] in VIETNAMESE_SURNAMES:
                if not any(w in topic_words for w in name_tokens[1:]):
                    return name

    # Heuristic: if query matches name pattern (surname first), treat as author name
    words = q.split()
    if 2 <= len(words) <= 4 and words[0] in VIETNAMESE_SURNAMES:
        if not any(w in topic_words for w in words[1:]):
            return q

    # Pattern 3: Western/Latin name query "L.G. Alexander" hoặc "Alexander, L.G."
    # Match names that are 2-4 tokens and contain a comma OR are mostly uppercase initials
    if 2 <= len(words) <= 5:
        # Check if it looks like a person name: comma OR uppercase initials OR known foreign surname
        if ',' in q:
            parts = [p.strip() for p in q.split(',') if p.strip()]
            if 1 <= len(parts) <= 3:
                # Reject if any part contains topic words or query noise
                all_clean = True
                for p in parts:
                    p_tokens = p.split()
                    if any(t in topic_words for t in p_tokens):
                        all_clean = False
                        break
                if all_clean and all(len(p) >= 2 for p in parts):
                    return q
        else:
            # Check if all tokens are capitalized (Western name style) or contain initials
            # Reject if any token is a topic word
            if all(w[0].isupper() or '.' in w for w in words if w):
                if not any(w in topic_words for w in words):
                    # Reject if very long (likely a sentence, not a name)
                    if len(q) <= 60:
                        return q
    return ""


def extract_publisher_name(query: str) -> str:
    """Trích xuất tên nhà xuất bản từ câu hỏi."""
    if not query:
        return ""
    q = normalize_text(query).strip()
    patterns = [
        r"(?:nha xuat ban|nxb)\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            name = match.group(1).strip().rstrip(",. ")
            name = re.sub(r"\s+(?:khong|ko|lam on|hay|vui long|toi|minh|xin|cho|gui)\b.*$", "", name)
            name = name.strip().rstrip(",. ")
            if len(name) >= 3:
                return name
    return ""


def normalize_author_name(name: str) -> str:
    """Normalize author name for comparison."""
    if not name:
        return ""
    # Remove content after semicolons FIRST (before normalize_text strips them)
    name = re.sub(r"\s*;.*$", "", name)
    normalized = normalize_text(name)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s*\([^)]*\)\s*", " ", normalized)
    # Also handle content after commas that don't match the query
    # e.g. "Trần Thị Kim Hoa" should not match "Nguyễn Thị Kim Hoa"
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def author_name_matches(doc_author: str, query_author: str) -> bool:
    """Check if doc_author matches query_author.
    Yêu cầu chặt chẽ: họ (token đầu) và tên (token cuối) phải khớp,
    và query name phải xuất hiện như một phần trong doc_author.
    """
    if not doc_author or not query_author:
        return False
    doc_norm = normalize_author_name(doc_author)
    query_norm = normalize_author_name(query_author)
    if not doc_norm or not query_norm:
        return False

    query_tokens = query_norm.split()
    doc_tokens = doc_norm.split()
    if not query_tokens or not doc_tokens:
        return False

    # 1. Exact match
    if doc_norm == query_norm:
        return True

    # 2. Query name appears as a whole contiguous phrase in doc name
    # (e.g. "nguyen thi kim hoa" in "pgs.ts. nguyen thi kim hoa")
    if query_norm in doc_norm:
        return True

    # 3. Strict token matching: first token (họ) AND last token (tên) must match
    #    plus at least 2 other tokens must match
    first_matches = doc_tokens[0] == query_tokens[0]
    last_matches = doc_tokens[-1] == query_tokens[-1]

    if first_matches and last_matches:
        # Count how many query tokens appear in doc tokens
        doc_set = set(doc_tokens)
        overlap = sum(1 for t in query_tokens if t in doc_set)
        # Require that at least all query tokens match (full name match)
        # or all but one (e.g. missing middle name)
        if overlap >= len(query_tokens) - 1 and overlap >= 3:
            return True

    # 4. Western/initial style name matching (e.g. "L.G. Alexander" vs "Alexander, L.G.")
    #    Convert both to normalized initials+surname form
    def _to_initials_surname(name):
        """'L.G. Alexander' or 'Alexander, L.G.' → 'l.g.alexander' (alphabetical concat)"""
        n = re.sub(r'[,\.]+', ' ', name).lower()
        n = re.sub(r'\s+', ' ', n).strip()
        # Tokenize: each part may be 'l', 'g', 'alexander' etc.
        toks = n.split()
        # Last token = surname; others = initials
        if len(toks) >= 2:
            surname = toks[-1]
            initials = ''.join(t[0] for t in toks[:-1] if t)
            return initials + surname
        return n.replace(' ', '')

    doc_key = _to_initials_surname(doc_norm)
    query_key = _to_initials_surname(query_norm)
    if doc_key and query_key and len(doc_key) >= 3 and len(query_key) >= 3:
        if doc_key == query_key:
            return True
        # If one is a substring of the other (e.g. longer with extra initials)
        if query_key in doc_key or doc_key in query_key:
            if abs(len(doc_key) - len(query_key)) <= 4:
                return True

    # 5. Surname-only match: query is just a last name like "Alexander"
    if len(query_tokens) == 1 and len(query_tokens[0]) >= 3:
        if query_tokens[0] in doc_norm:
            return True

    return False


def rerank_results_for_query(query: str, search_results: list[dict], prefer_strict: bool = False) -> list[dict]:
    """Boost/filter results for recognized domains such as AI and author/publisher."""
    if not search_results:
        return search_results

    # ── Author/Publisher strict filtering ──────────────────
    author_name = extract_author_name(query) if detect_author_query(query) else ""
    publisher_name = extract_publisher_name(query) if detect_publisher_query(query) else ""

    if author_name:
        # Strict filter: chỉ giữ tài liệu có tác giả matching
        author_results = []
        non_author_results = []
        for result in search_results:
            doc_author = result.get("doc", {}).get("author", "")
            if doc_author and author_name_matches(doc_author, author_name):
                boosted = {**result, "score": result.get("score", 0.0) + 500.0}
                author_results.append(boosted)
            else:
                non_author_results.append(result)

        if author_results:
            author_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            non_author_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            if prefer_strict:
                return author_results
            return author_results + non_author_results[:3]
        # No author match at all — KHÔNG fall through, trả về rỗng
        # (trước đây fall through → hiển thị toàn bộ kết quả không liên quan)
        if prefer_strict:
            logger.info("author strict miss for %r → return empty (0 results)", author_name)
            return []

    if publisher_name:
        publisher_norm = normalize_text(publisher_name)
        pub_results = []
        non_pub_results = []
        for result in search_results:
            doc_pub = result.get("doc", {}).get("publisher", "")
            doc_source = str(result.get("doc", {}).get("source", ""))
            doc_text = result.get("doc", {}).get("text", "")
            combined = normalize_text(f"{doc_pub} {doc_source} {doc_text}")
            if publisher_norm in combined:
                boosted = {**result, "score": result.get("score", 0.0) + 300.0}
                pub_results.append(boosted)
            else:
                non_pub_results.append(result)

        if pub_results:
            pub_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            if prefer_strict:
                return pub_results
            non_pub_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            return pub_results + non_pub_results[:3]
        # No publisher match at all — KHÔNG fall through
        if prefer_strict:
            logger.info("publisher strict miss for %r → return empty (0 results)", publisher_name)
            return []

    if detect_ai_domain(query):
        matched = []
        unmatched = []
        for result in search_results:
            relevance = score_ai_document(result.get("doc", {}))
            boosted = {**result, "score": result.get("score", 0.0) + relevance}
            if relevance > 0:
                matched.append(boosted)
            else:
                unmatched.append(boosted)

        if matched:
            matched.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            if prefer_strict:
                return matched
            unmatched.sort(key=lambda item: item.get("score", 0.0), reverse=True)
            return matched + unmatched

    # ── PDF boost: đưa tài liệu PDF có nội dung liên quan lên đầu ──
    # (collapse_catalog_pdf_pages boost +10000, nhưng đây là lớp bảo vệ thứ 2)
    for result in search_results:
        doc = result.get("doc", {})
        is_pdf = str(doc.get("csv_file", "")).startswith("pdf:") or doc.get("doc_type") == "PDF"
        if is_pdf and doc.get("text") and len(doc["text"].strip()) > 100:
            result["score"] = result.get("score", 0.0) + 500.0

    _, topic_tokens, topic_phrases = get_query_topic_terms(query)
    if not topic_tokens and not topic_phrases:
        return search_results

    unique_tokens = list(dict.fromkeys(topic_tokens))
    matched = []
    unmatched = []

    # Adaptive threshold: require higher coverage when query has many tokens
    # Phrase candidates include very long substrings (e.g. the full query text), so
    # the "phrase hit" check would otherwise let thousands of docs bypass strict coverage.
    # The MIN phrase length protects against that loophole.
    min_phrase_length_for_bypass = 4
    if len(unique_tokens) >= 6:
        strict_coverage_threshold = 0.80
    elif len(unique_tokens) >= 3:
        strict_coverage_threshold = 0.65
    else:
        strict_coverage_threshold = 0.50
    # Adaptive relaxed threshold: shorter queries → stricter, longer → more lenient
    # VD: 5 tokens → 0.40, 10 tokens → 0.25, 20 tokens → 0.15
    relaxed_coverage_threshold = max(0.15, min(0.40, 3.0 / max(len(unique_tokens), 1)))
    is_single_token = len(unique_tokens) == 1

    for result in search_results:
        doc = result.get("doc", {})
        normalized_doc = get_doc_topic_text(doc)
        exact_phrase_hit = any(
            contains_normalized_phrase(normalized_doc, phrase)
            for phrase in topic_phrases
            if len(phrase.split()) >= min_phrase_length_for_bypass
        )
        match_count = sum(1 for token in unique_tokens if re.search(rf"\b{re.escape(token)}\b", normalized_doc))
        if not match_count and not exact_phrase_hit:
            unmatched.append(result)
            continue

        coverage = match_count / max(len(unique_tokens), 1)

        # Single-token query: require token in subject/major OR exact phrase hit
        if is_single_token:
            if not exact_phrase_hit:
                doc_subject = normalize_text(doc.get("subject", ""))
                doc_major = normalize_text(doc.get("major", ""))
                token = unique_tokens[0]
                token_in_subject = token in doc_subject if doc_subject else False
                token_in_major = token in doc_major if doc_major else False
                # Also allow if token appears in title (strong signal)
                doc_title = normalize_text(doc.get("title", ""))
                token_in_title = token in doc_title if doc_title else False
                if not token_in_subject and not token_in_major and not token_in_title:
                    unmatched.append(result)
                    continue

        # Strict mode (list/author queries): require strong coverage
        if prefer_strict and len(unique_tokens) >= 2 and coverage < strict_coverage_threshold and not exact_phrase_hit:
            unmatched.append(result)
            continue

        # Relaxed mode: require minimum token overlap to avoid noise
        if not prefer_strict and not exact_phrase_hit and coverage < relaxed_coverage_threshold:
            unmatched.append(result)
            continue

        # Boost subject/major field matches heavily
        doc_subject = normalize_text(doc.get("subject", ""))
        doc_major = normalize_text(doc.get("major", ""))
        doc_title = normalize_text(doc.get("title", ""))
        subject_boost = 0.0
        for token in unique_tokens:
            if doc_subject and token in doc_subject:
                subject_boost += 25.0
            if doc_major and token in doc_major:
                subject_boost += 20.0
            if doc_title and token in doc_title:
                subject_boost += 10.0

        phrase_boost = 20.0 if exact_phrase_hit else 0.0
        coverage_bonus = coverage * 8.0
        relevance = (match_count * 3.0) + coverage_bonus + phrase_boost + subject_boost
        matched.append({**result, "score": result.get("score", 0.0) + relevance})

    if matched:
        matched.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        if prefer_strict:
            return matched
        unmatched.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return matched + unmatched

    if prefer_strict:
        return []

    # Không có kết quả nào pass rerank → trả rỗng (đúng là không có tài liệu phù hợp)
    logger.info("rerank: 0 matched → return empty for query=%r", query)
    return []


def extract_keywords(text: str) -> str:
    """Trích xuất từ khóa tìm kiếm bằng cách loại bỏ stop words và từ ngắn."""
    q = extract_primary_topic_text(text)
    catalog_matches = find_catalog_topic_matches(q)
    if not catalog_matches:
        catalog_matches = find_catalog_topic_matches(strip_search_intent_phrases(text))
    selected_topics = catalog_matches[:3] if catalog_matches else [q]
    expanded_terms = expand_domain_terms(text)
    if catalog_matches:
        combined = selected_topics + [normalize_topic_text(term) for term in expanded_terms]
        return " ".join(term for term in dict.fromkeys(combined) if term)

    topic_text = " ".join(selected_topics)
    words = topic_text.split()
    protected_short_tokens = {
        token
        for term in expanded_terms + selected_topics + [q]
        for token in normalize_topic_text(term).split()
        if len(token) == 1
    }
    # Bỏ stop words nhưng giữ lại từ thực và token ngắn trong cụm ngành được bảo toàn.
    filtered = [
        w for w in words
        if w not in VIETNAMESE_STOP_WORDS and (len(w) >= 2 or w in protected_short_tokens)
    ]
    combined = filtered + [normalize_topic_text(term) for term in expanded_terms]
    unique_terms = [term for term in dict.fromkeys(combined) if term]
    return " ".join(unique_terms)

def extract_content_search_query(text: str) -> str:
    """Keep rich terms for content/PDF questions instead of collapsing to catalog topics."""
    q = normalize_topic_text(text)
    for pattern in [
        r"\b(?:ban|ban co biet|toi|minh|em|anh|chi)\b",
        r"\b(?:hay|vui long)\b",
        r"\b(?:cho toi|cho minh|cho em)\b",
        r"\b(?:khong|ko|k)\b",
    ]:
        q = re.sub(pattern, " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q or normalize_text(text)

# detect_greeting, classify_intent, detect_author_query, detect_publisher_query
# → moved to intent_classifier.py (score-based entity extraction)

def detect_live_info_request(query: str) -> bool:
    return classify_intent(query) == "live_info"

def detect_exhaustion_query(query: str) -> bool:
    """Detect if user is asking whether there are more results (exhaustion check)."""
    q = normalize_text(query or "")
    phrases = [
        "het chua", "con khong", "con tai lieu nao", "con them khong",
        "con nhieu khong", "het tai lieu", "het sach", "het ket qua",
        "con bao nhieu", "con mot nua", "con gi khong",
    ]
    return any(phrase in q for phrase in phrases)

def detect_content_request(query: str) -> bool:
    """Only answer document content when the user explicitly asks for content."""
    if not query:
        return False

    q = normalize_text(query)
    if detect_summary_request(query) or detect_chapter_reference(query):
        return True

    content_phrases = [
        "noi dung", "chi tiet", "phan tich", "giai thich", "trinh bay",
        "so sanh", "sosanh",
        "cho biet", "hay cho biet", "neu ro", "neu cac", "neu nhung",
        "tom tat", "doc file", "doc tai lieu", "trich", "trich xuat",
        "muc luc", "chuong", "phan nao", "noi ve gi",
        "file nay", "pdf nay", "tai lieu nay", "tai lieu do",
        "tai lieu so", "cuon nay", "sach nay",
        "phuong phap nghien cuu", "ket qua nghien cuu", "ket luan",
        "giai phap trong", "dinh huong trong",
        "khai niem", "dinh nghia", "vi du", "giai thich ca tu",
        "phan mem", "he thong", "ung dung",
        "bo cuc", "cau truc", "danh muc", "layout", "structure",
    ]
    if any(phrase in q for phrase in content_phrases):
        return True

    command_patterns = [
        r"\b(?:hay|vui long)?\s*(?:neu|phan tich|trinh bay|giai thich|tom tat|doc)\b",
        r"\b(?:cho toi|cho minh|cho em)\s+(?:biet|xem|tom tat)\b",
    ]
    return any(re.search(pattern, q) for pattern in command_patterns)

def is_follow_up(query: str) -> bool:
    """Kiểm tra nếu query là follow-up không có chủ đề mới."""
    if detect_ai_domain(query):
        return False
    keywords = extract_keywords(query)
    return len(keywords.split()) == 0

def resolve_search_query(query: str, history: list, catalog_mode: bool = False) -> str:
    """Xác định search query từ câu hỏi + lịch sử."""
    keywords = extract_keywords(query) if catalog_mode else extract_content_search_query(query)
    norm = normalize_text(query)

    if is_follow_up(query) and history:
        # Follow-up: dùng chủ đề từ history + query gốc để giữ ngữ cảnh
        last = history[-1]
        topic = normalize_text(last.get("topic", ""))
        last_q = normalize_text(last.get("search_query", ""))
        if topic and len(topic) >= 3:
            return f"{topic} {norm}".strip()
        if last_q:
            return f"{last_q} {norm}".strip()
    
    # Có từ khóa rõ ràng
    if keywords:
        return keywords
    
    return norm

def detect_list_request(query: str) -> bool:
    """Detect câu hỏi cần liệt kê/tìm sách theo chủ đề."""
    if detect_summary_request(query):
        return False

    q = normalize_text(query)
    keywords = [
        "dua tren tai lieu", "tham khao", "cac tai lieu sau", "ban co the tham khao",
        "goi y cac tai lieu", "co the tham khao", "co the tim", "co the xem",
        "co nhung sach nao", "co tai lieu nao", "sach nao", "tai lieu nao",
        "liet ke", "tim sach", "tim tai lieu", "goi y sach", "goi y tai lieu",
        "goi y de tai", "de tai ve", "de tai lien quan",
        "sach lien quan", "tai lieu lien quan", "ve chu de", "ve linh vuc"
    ]
    return any(kw in q for kw in keywords)

def detect_existence_request(query: str) -> bool:
    """Detect câu hỏi kiểm tra sự tồn tại yes/no (Có tài liệu về X không?).

    Trả về True cho các câu hỏi dạng yes/no hỏi về sự tồn tại của tài liệu.
    Nếu `detect_list_request` đã True thì trả về False để tránh xử lý trùng.
    """
    if not query:
        return False
    if detect_list_request(query):
        return False
    q = normalize_text(query)
    if not q:
        return False

    # Nếu query chứa động từ so sánh/phân tích → không phải existence check đơn thuần
    # VD: "có cuốn nào so sánh U-Net và V-Net không?" muốn tìm sách, không phải yes/no
    content_action_words = [
        "so sanh", "sosanh", "phan tich", "giai thich", "trinh bay",
        "tom tat", "chi tiet", "mo ta",
    ]
    if any(w in q for w in content_action_words):
        return False

    patterns = [
        r"\bco\s+(?:tai\s+lieu|sach|cuon|quyen|an\s+pham|tai\s+lieus|giao\s+trinh)\b[^?]*?\b(?:khong|ko|hem|khong\s+a|nhi|vay)\b\s*\??",
        r"\bco\s+(?:tai\s+lieu|sach|cuon|quyen)\s+(?:nao|nào)\b[^?]*?\b(?:khong|ko|khong\s+a|nhi|vay)\b\s*\??",
        r"\b(?:trong\s+kho|trong\s+thu\s+vien|trong\s+thu\s+vien\s+cu[aà])\s+co\b[^?]*?\b(?:khong|ko|khong\s+a|nhi|vay)\b\s*\??",
    ]
    return any(re.search(p, q) for p in patterns)

def _strip_vi_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt, trả về dạng ASCII không dấu.

    Dùng để so sánh các từ yes/no có/không khi người dùng gõ có dấu.
    """
    mapping = {
        # Lowercase
        "ă": "a", "â": "a", "đ": "d", "ê": "e", "ô": "o", "ơ": "o", "ư": "u",
        "ắ": "a", "ấ": "a", "ạ": "a", "ả": "a", "ã": "a", "ằ": "a", "ầ": "a",
        "ậ": "a", "ẳ": "a", "ẵ": "a",
        "ẹ": "e", "ẻ": "e", "ẽ": "e", "ế": "e", "ề": "e", "ệ": "e",
        "ọ": "o", "ỏ": "o", "õ": "o", "ố": "o", "ồ": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
        "ớ": "o", "ờ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ụ": "u", "ủ": "u", "ũ": "u", "ứ": "u", "ừ": "u", "ự": "u",
        "ỳ": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
        "í": "i", "ì": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ó": "o", "ò": "o", "é": "e", "è": "e", "á": "a", "à": "a",
        "ú": "u", "ù": "u", "ý": "y",
        # Uppercase
        "Ă": "A", "Â": "A", "Đ": "D", "Ê": "E", "Ô": "O", "Ơ": "O", "Ư": "U",
        "Ắ": "A", "Ấ": "A", "Ạ": "A", "Ả": "A", "Ã": "A", "Ằ": "A", "Ầ": "A",
        "Ậ": "A", "Ẳ": "A", "Ẵ": "A",
        "Ẹ": "E", "Ẻ": "E", "Ẽ": "E", "Ế": "E", "Ề": "E", "Ệ": "E",
        "Ọ": "O", "Ỏ": "O", "Õ": "O", "Ố": "O", "Ồ": "O", "Ộ": "O", "Ổ": "O", "Ỗ": "O",
        "Ớ": "O", "Ờ": "O", "Ợ": "O", "Ở": "O", "Ỡ": "O",
        "Ụ": "U", "Ủ": "U", "Ũ": "U", "Ứ": "U", "Ừ": "U", "Ự": "U",
        "Ỳ": "Y", "Ỵ": "Y", "Ỷ": "Y", "Ỹ": "Y",
        "Í": "I", "Ì": "I", "Ị": "I", "Ỉ": "I", "Ĩ": "I",
        "Ó": "O", "Ò": "O", "É": "E", "È": "E", "Á": "A", "À": "A",
        "Ú": "U", "Ù": "U", "Ý": "Y",
    }
    return "".join(mapping.get(c, c) for c in text)


_YESNO_WORD_SET = {
    # Single word
    "khong", "ko", "k", "kh", "hem", "nhi", "vay", "nhe", "a",
    "khong a", "khong vay", "khong nhi", "khong the", "khong va",
    "khong nhe", "khong phai", "khong co", "k vay", "k nhi",
    "gi", "gi a", "chu", "dau", "roi", "the", "ha",
}


def _is_yesno_token(token: str) -> bool:
    """Kiểm tra một token đã bỏ dấu có phải từ yes/no không."""
    return _strip_vi_diacritics(token).strip().lower() in _YESNO_WORD_SET


def extract_existence_topic(query: str) -> str:
    """Trích xuất chủ đề từ câu hỏi kiểm tra sự tồn tại.

    VD: 'Có tài liệu về văn học Việt Nam không?' → 'văn học Việt Nam'
        'Có sách về CNTT không?' → 'CNTT'
    """
    if not query:
        return ""

def _last_existence_topic(history) -> str:
    """Tìm chủ đề cuối cùng mà bot đã xác nhận 'có tài liệu về X' trong lịch sử."""
    if not history:
        return ""
    last_topic = ""
    for msg in history:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "") or ""
        m = re.search(r"hi[ệe]n\s+c[oó]\s+t[aà]i\s+li[ệe]u\s+v[ềe]\s+\*\*([^*]+)\*\*", content, re.IGNORECASE)
        if not m:
            m = re.search(r"t[ìi]m\s+th[aấ]y[^\n]*?v[ềe]\s+\*\*([^*]+)\*\*", content, re.IGNORECASE)
        if m:
            last_topic = m.group(1).strip()
    return last_topic

    cleaned = query
    # Bỏ cụm mở đầu yes/no kiểu "có tài liệu về", "có sách về", ...
    cleaned = re.sub(
        r"^\s*(?:trong\s+(?:kho|thu\s+vien|thu\s+vien\s+cu[aà])\s+)?"
        r"(?:ban\s+)?co\s+(?:tai\s+lieu|sach|cuon|quyen|an\s+pham|giao\s+trinh)\s+"
        r"(?:nao\s+|nào\s+)?(?:ve|về)\s+",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:trong\s+(?:kho|thu\s+vien|thu\s+vien\s+cu[aà])\s+)?"
        r"(?:ban\s+)?co\s+cuon\s+nao\s+ve\s+",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*(?:trong\s+(?:kho|thu\s+vien|thu\s+vien\s+cu[aà])\s+)?"
        r"(?:ban\s+)?co\s+"
        r"(?:tai\s+lieu|sach|cuon|quyen|an\s+pham|giao\s+trinh)\s+",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Bỏ dấu hỏi
    cleaned = re.sub(r"\?+\s*$", " ", cleaned)

    # Bỏ các từ yes/no ở cuối câu (dùng _is_yesno_token để handle dấu TV)
    for _ in range(3):
        cleaned = cleaned.strip()
        if not cleaned:
            break
        m = re.search(r"\s+(\S+)$", cleaned)
        if m:
            if _is_yesno_token(m.group(1)):
                cleaned = cleaned[:m.start()].rstrip()
            else:
                break
        else:
            # Cả câu chỉ có 1 từ
            if _is_yesno_token(cleaned):
                cleaned = ""
            break

    # Bỏ "về" / "lĩnh vực" đứng đầu nếu còn
    cleaned = re.sub(
        r"^\s*(?:v[eề]|lĩnh\s*vực|chủ\s*đề|ngành)\s+",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if not cleaned:
        return ""

    # Nếu vẫn còn marker "về X" → lấy phần sau "về"
    match = re.search(
        r"\b(?:v[eề]|lĩnh\s*vực|chủ\s*đề|ngành)\s+(.+)$",
        cleaned,
        flags=re.IGNORECASE,
    )
    if match:
        cleaned = match.group(1).strip()
        # Lại bỏ yes/no ở cuối
        for _ in range(3):
            cleaned = cleaned.strip()
            if not cleaned:
                break
            m = re.search(r"\s+(\S+)$", cleaned)
            if m:
                if _is_yesno_token(m.group(1)):
                    cleaned = cleaned[:m.start()].rstrip()
                else:
                    break
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_display_value(value, max_length: int = 220) -> str:
    """Clean raw metadata before rendering it in answers/cards."""
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        text = ", ".join(str(item).strip() for item in value if str(item).strip())
    else:
        text = str(value).strip()

    if not text:
        return ""

    if text.lower() in {"n/a", "na", "unknown", "không xác định", "nan", "none", "null"}:
        return ""

    if text.startswith("[") and text.endswith("]"):
        items = [
            part.strip().strip("'\"")
            for part in text.strip("[]").split(",")
        ]
        text = ", ".join(item for item in items if item)

    text = re.sub(r"\s+", " ", text).strip(" ;,")
    if not text:
        return ""

    if normalize_text(text) in {"n a", "na", "unknown", "khong xac dinh", "nan", "none", "null"}:
        return ""

    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text


def build_catalog_answer(search_results, query: str, total_found: int = 0) -> str:
    """Build câu trả lời ngắn gọn: chỉ thông báo có kết quả, hiển thị chi tiết ở phần sources."""
    if not search_results:
        return "Không tìm thấy tài liệu phù hợp trong dữ liệu hiện có."

    total = total_found if total_found > 0 else len(search_results)
    return f"Mình tìm thấy **{total} tài liệu** liên quan đến câu hỏi của bạn. Xem chi tiết bên dưới."


def sanitize_answer_text(answer: str) -> str:
    """Remove source-footers and stray assistant tags from model output."""
    if not answer:
        return ""

    text = answer.replace("</assistant>", "").replace("<assistant>", "")

    if "<think>" in text:
        if "</think>" in text:
            text = text.split("</think>", 1)[1]
        else:
            text = text.split("<think>", 1)[0]

    for marker in ["Nguồn tài liệu:", "Nguồn:", "Tài liệu tham khảo:"]:
        if marker in text:
            text = text.split(marker, 1)[0]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if stripped.lower().startswith(("okay,", "the user", "i will", "here is", "sure,", "certainly,")):
            continue
        if re.match(r"^[-*•\s]*\[Tài liệu\s*\d+\]", stripped, re.IGNORECASE):
            continue
        if re.match(r"^[-*•\s]*(Tài liệu\s*\d+|Nguồn tài liệu)\b", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def call_llm(messages: list, temperature: float = 0.3) -> str:
    """
    Gọi OpenRouter và trả về response text, có retry khi rate limited

    Tích hợp:
    - Logging mỗi attempt + timing
    - Circuit breaker: mở sau N lỗi liên tiếp, cooldown T giây
    - Track stats: _llm_stats (calls/failures/retries)
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")

    # Circuit breaker check
    cb = _llm_circuit
    if cb["state"] == "open":
        if time.time() - cb["opened_at"] < cb["cooldown_seconds"]:
            logger.warning("LLM circuit OPEN, refusing call (cooldown %ss left)", int(cb["cooldown_seconds"] - (time.time() - cb["opened_at"])))
            raise HTTPException(503, "LLM service tạm thời không khả dụng, vui lòng thử lại sau ít phút.")
        cb["state"] = "half_open"
        logger.info("LLM circuit → half_open (retry attempt)")

    _llm_stats["calls"] += 1
    request_start = time.time()
    last_error: Optional[str] = None

    # Build model chain: primary + fallbacks
    model_chain = [OPENROUTER_MODEL] + list(OPENROUTER_FALLBACK_MODELS)
    max_retries = 3

    for model_idx, model_name in enumerate(model_chain):
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
                        "X-Title": OPENROUTER_APP_TITLE,
                    },
                    json={
                        "model": model_name,
                        "messages": messages,
                        "temperature": temperature,
                    },
                    timeout=300,
                )

                if response.status_code == 429 and attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    _llm_stats["retries"] += 1
                    logger.warning("Rate limited on %s (429), retrying in %ss (attempt %d/%d)", model_name, wait, attempt + 1, max_retries)
                    time.sleep(wait)
                    continue

                if response.status_code == 400 or response.status_code == 404:
                    # Invalid model → skip to next fallback immediately
                    err_detail = f"Model '{model_name}' invalid [{response.status_code}]: {response.text[:200]}"
                    logger.warning("LLM %s → fallback to next model: %s", model_name, err_detail)
                    last_error = err_detail
                    break  # break retry loop, try next model

                if response.status_code != 200:
                    last_error = f"OpenRouter error [{response.status_code}]: {response.text[:200]}"
                    _record_llm_failure(last_error)
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        _llm_stats["retries"] += 1
                        logger.warning("LLM error, retrying in %ss (attempt %d/%d): %s", wait, attempt + 1, max_retries, last_error)
                        time.sleep(wait)
                        continue
                    break  # hết retry cho model này, thử fallback

                # Thành công
                payload = response.json()
                content = payload["choices"][0]["message"].get("content", "")

                elapsed = time.time() - request_start
                tag = "primary" if model_idx == 0 else f"fallback#{model_idx}"
                logger.info("LLM call OK in %.2fs (%s, model=%s, len=%d)", elapsed, tag, model_name, len(content))
                if cb["state"] in ("open", "half_open"):
                    logger.info("LLM circuit → closed (recovered)")
                cb["state"] = "closed"
                cb["failures"] = 0
                return content

            except HTTPException:
                raise
            except Exception as e:
                last_error = str(e)[:200]
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    _llm_stats["retries"] += 1
                    logger.warning("LLM error on %s, retrying in %ss (attempt %d/%d): %s", model_name, wait, attempt + 1, max_retries, last_error)
                    time.sleep(wait)
                    continue
                break  # hết retry cho model này, thử fallback

    # All models failed
    _record_llm_failure(last_error or "All models failed")
    logger.error("LLM call FAILED after %d models in %.2fs: %s", len(model_chain), time.time() - request_start, last_error)
    raise HTTPException(503, f"LLM API error: {last_error}")


def _record_llm_failure(error_msg: str) -> None:
    """Track LLM failure, mở circuit breaker nếu đạt ngưỡng."""
    _llm_stats["failures"] += 1
    _llm_stats["last_error"] = error_msg
    _llm_circuit["failures"] += 1
    if (
        _llm_circuit["state"] in ("closed", "half_open")
        and _llm_circuit["failures"] >= _llm_circuit["failure_threshold"]
    ):
        _llm_circuit["state"] = "open"
        _llm_circuit["opened_at"] = time.time()
        logger.error(
            "LLM circuit → OPEN (failures=%d >= threshold=%d, cooldown=%ss)",
            _llm_circuit["failures"],
            _llm_circuit["failure_threshold"],
            _llm_circuit["cooldown_seconds"],
        )

def build_summary_prompt():
    """Build prompt cho tóm tắt nâng cao với cấu trúc"""
    
    prompt_text = """Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn (QNU Library Assistant).

⚠️ TUYỆT ĐỐI KHÔNG được thêm thông tin ngoài nội dung tài liệu bên dưới.
Chỉ tóm tắt ĐÚNG những gì có trong tài liệu, không suy luận, không thêm chi tiết.

Hãy tóm tắt tài liệu theo cấu trúc sau — CHỈ viết khi nội dung tương ứng CÓ trong tài liệu (bỏ qua mục không có):

**MỤC TIÊU:**
Nêu rõ mục tiêu chính của tài liệu là gì? Nếu tài liệu không nêu, bỏ qua mục này.

**PHƯƠNG PHÁP:**
Tài liệu sử dụng phương pháp, công cụ, cách tiếp cận nào? Nếu không có, bỏ qua.

**KẾT QUẢ CHÍNH:**
Những phát hiện, kết luận, kết quả chính là gì? (3-5 dòng). Nếu không có, bỏ qua.

**TỪ KHÓA:**
Liệt kê 5-7 từ khóa chính từ nội dung tài liệu.

**TÓM TẮT NGẮN:**
Tóm tắt lại toàn bộ nội dung thành 2-3 câu dựa trên tài liệu.

NỘI DUNG TÀI LIỆU:
{content}

TRẢ LỜI (Tiếng Việt, tuân thủ cấu trúc trên, chỉ nội dung có thật trong tài liệu):
"""
    
    def summary_chain(content: str) -> str:
        messages = [
            {"role": "system", "content": "Bạn chỉ được trả về phần trả lời cuối cùng bằng tiếng Việt. Không hiển thị suy luận, không dùng thẻ <think>, không thêm chú thích meta. Không thêm thông tin không có trong tài liệu."},
            {"role": "user", "content": prompt_text.format(content=content)}
        ]
        return sanitize_answer_text(call_llm(messages, temperature=0.1))
    
    return summary_chain

def detect_chapter_reference(query: str) -> Optional[int]:
    """Detect số chương được nhắc đến trong câu hỏi."""
    if not query:
        return None

    patterns = [
        r"chương\s*(\d+)",
        r"chuong\s*(\d+)",
        r"chapter\s*(\d+)",
        r"chương\s*(?:số|thứ|thu)?\s*(\d+)",
        r"chuong\s*(?:so|thu)?\s*(\d+)",
    ]

    q = normalize_text(query)
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None

def detect_method_question(query: str) -> bool:
    """Detect nếu user hỏi về phương pháp nghiên cứu"""
    if not query:
        return False

    keywords = [
        "phương pháp", "phuong phap", "phương pháp nghiên cứu",
        "method", "methodology", "cách thức", "công cụ",
        "quy trình", "quy trinh", "cách tiếp cận", "approach",
        "kỹ thuật", "ky thuat", "technique"
    ]
    q = normalize_text(query)
    return any(kw in q for kw in keywords)

def detect_result_question(query: str) -> bool:
    """Detect nếu user hỏi về kết quả/kết luận"""
    if not query:
        return False

    keywords = [
        "kết quả", "ket qua", "kết luận", "ket luan",
        "result", "finding", "conclusion", "phát hiện",
        "phat hien", "outcome", "giải pháp", "giai phap"
    ]
    q = normalize_text(query)
    return any(kw in q for kw in keywords)

def detect_structure_query(query: str) -> bool:
    """Detect nếu user hỏi về bố cục, cấu trúc, mục lục của tài liệu."""
    if not query:
        return False
    q = normalize_text(query)
    keywords = [
        "bo cuc", "bố cục",
        "cau truc", "cấu trúc",
        "muc luc", "mục lục",
        "danh muc", "danh mục",
        "layout", "structure", "table of contents",
    ]
    return any(kw in q for kw in keywords)

def build_structure_prompt() -> str:
    """Build prompt cho câu hỏi về bố cục/cấu trúc/mục lục của tài liệu."""
    return """
Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn.

YÊU CẦU: Trình bày BỐ CỤC / CẤU TRÚC / MỤC LỤC của tài liệu dựa trên nội dung được cung cấp.

HƯỚNG DẪN TRẢ LỜI:
1. Xác định tài liệu (nhan đề, tác giả, năm xuất bản).
2. Trình bày TOÀN BỘ bố cục/cấu trúc tài liệu theo dạng mục lục phân cấp:
   - Chương lớn (I, II, III... hoặc 1, 2, 3...)
   - Phần nhỏ bên trong mỗi chương (1.1, 1.2...)
   - Các mục con (a, b, c... hoặc i, ii, iii...)
3. Nếu có thể xác định trang PDF tương ứng, ghi chú [Trang X].
4. Luôn trình bày dưới dạng danh sách phân cấp rõ ràng, KHÔNG viết thành đoạn văn.
5. Nếu tài liệu không có cấu trúc rõ ràng, hãy liệt kê các phần lớn tìm thấy.
6. 📌 CẤU TRÚC PHÂN CẤP: PHẢI giữ nguyên cấu trúc gốc từ tài liệu. KHÔNG đánh số lại.
7. Dùng ## cho chương lớn, ### cho phần, #### cho mục nhỏ.

TÀI LIỆU:
{context}

CÂU HỎI: {question}

TRẢ LỜI (Tiếng Việt, dạng mục lục phân cấp, KHÔNG viết thành đoạn văn):
"""

def build_pdf_context_for_query(pdf_info: dict, query: str, chapter: Optional[int] = None) -> tuple[str, list[dict], Optional[int]]:
    """Lấy context tốt nhất từ chính PDF được chọn."""
    pdf_source = pdf_info.get('file_name', '')
    pdf_title = pdf_info.get('title') or pdf_info.get('display_name') or pdf_source or "PDF"
    chapter_num = chapter or detect_chapter_reference(query)

    if chapter_num:
        chapter_text, start_page, end_page = extract_chapter_section_from_pdf(pdf_info, chapter_num)
        if chapter_text:
            header = (
                f"Tài liệu được chọn: {pdf_title}\n"
                f"Tên file: {pdf_source}\n"
                f"Mục: Chương {chapter_num}"
                + (f" | Trang PDF: {start_page}-{end_page}" if start_page and end_page else "")
                + "\n"
            )
            return f"{header}\n{chapter_text}", [], start_page

    if detect_reference_request(query):
        reference_text, start_page = extract_reference_section_from_pdf(pdf_info)
        if reference_text:
            header = (
                f"Tài liệu được chọn: {pdf_title}\n"
                f"Tên file: {pdf_source}\n"
                f"Mục: Tài liệu tham khảo"
                + (f" | Trang PDF bắt đầu: {start_page}" if start_page else "")
                + "\n"
            )
            return f"{header}\n{reference_text}", [], start_page

    # Với câu hỏi về bố cục/cấu trúc/mục lục → lấy trang đầu (TOC thường ở trang 1-10)
    if is_general_pdf_content_request(query):
        q_norm = normalize_text(query)
        is_structure_query = any(
            kw in q_norm for kw in (
                "bo cuc", "cau truc", "muc luc", "danh muc", "layout",
                "structure", "table of contents", "noi dung file",
                "noi dung tai lieu", "noi dung pdf", "gioi thieu",
            )
        )
        if is_structure_query:
            # Lấy 8 trang đầu — TOC thường nằm ở đây (giảm context cho free model)
            toc_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 8)
            if has_extractable_pdf_text(toc_text):
                header = (
                    f"Tài liệu được chọn: {pdf_title}\n"
                    f"Tên file: {pdf_source}\n"
                    f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
                    f"Mục trích xuất: 8 trang đầu (mục lục / bố cục)\n"
                )
                return f"{header}\nNội dung trích xuất:\n{toc_text[:8000]}", [], None

        # Câu hỏi chung khác → lấy mẫu đại diện (giảm context cho free model)
        full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)
        if has_extractable_pdf_text(full_text):
            header = (
                f"Tài liệu được chọn: {pdf_title}\n"
                f"Tên file: {pdf_source}\n"
                f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
            )
            # Lấy phần đầu + phần giữa + phần cuối để đại diện toàn bộ tài liệu
            total_len = len(full_text)
            if total_len <= 12000:
                # Ngắn → lấy toàn bộ
                sampled = full_text
            else:
                chunk = 4000  # 4K chars mỗi phần
                head = full_text[:chunk]
                mid_start = max(0, (total_len - chunk) // 2)
                mid = full_text[mid_start:mid_start + chunk]
                tail = full_text[-chunk:]
                sampled = f"{head}\n\n[...Phần giữa trang {total_len // 2000}...]\n\n{mid}\n\n[...Phần cuối trang {total_len // 2000}...]\n\n{tail}"
            return f"{header}\nNội dung trích xuất:\n{sampled}", [], None

    # Nếu không có chương cụ thể, search trong chính file PDF này để lấy các trang liên quan nhất
    selected_results = []
    for result in retriever.search_by_query(query, top_k=20):
        doc = result.get('doc', {})
        if doc.get('source') == pdf_source and has_extractable_pdf_text(doc.get('text', '')):
            selected_results.append(result)

    if selected_results:
        # Format với đầy đủ nội dung text để LLM có thể trả lời
        texts = []
        for r in selected_results[:6]:
            doc = r['doc']
            title = doc.get('title', '')
            text = doc.get('text', '')
            page = doc.get('page', '')
            header = f"[{title}" + (f" - Trang {page}]" if page else "]")
            texts.append(f"{header}\n{text[:3000]}")
        context = "\n\n---\n\n".join(texts)
        return context, [], None

    # Fallback cuối cùng: lấy toàn bộ text PDF (cắt bớt ở mức hợp lý)
    full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)
    if not has_extractable_pdf_text(full_text):
        return "", [], None

    header = (
        f"Tài liệu được chọn: {pdf_title}\n"
        f"Tên file: {pdf_source}\n"
        f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
    )
    return f"{header}\nNội dung trích xuất:\n{full_text[:20000]}", [], None

def build_chapter_prompt(chapter_num: int) -> str:
    """Build prompt chuyên biệt cho câu hỏi về một chương cụ thể"""
    return f"""
Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn.

YÊU CẦU: Trả lời câu hỏi về CHƯƠNG {chapter_num} dựa trên nội dung tài liệu được cung cấp.

HƯỚNG DẪN TRẢ LỜI:
1. Xác định nội dung chính của Chương {chapter_num}.
2. Nêu rõ: Nhan đề sách, tác giả, năm xuất bản.
3. Tóm tắt các nội dung chính trong chương (mục tiêu, các phần nhỏ).
4. Nếu có kết quả/phát hiện quan trọng, liệt kê rõ.
5. Nếu thông tin không đủ, hãy nói "Chương này không được đề cập chi tiết trong tài liệu hiện có".
6. Không viết thành một đoạn dài; luôn chia ý bằng tiêu đề ngắn và gạch đầu dòng.
7. 📌 CẤU TRÚC PHÂN CẤP: PHẢI giữ nguyên cấu trúc gốc từ tài liệu. Nếu chương có "I, 1, 1.1, a, b, c..." thì tái hiện đúng thứ bậc đó. KHÔNG đánh số lại thành 1, 2, 3... liên tục.
8. Dùng ## cho mục lớn, ### cho mục vừa, #### cho mục nhỏ. KHÔNG dùng `-` trước heading.

TÀI LIỆU:
{{context}}

CÂU HỎI: {{question}}

TRẢ LỜI (Tiếng Việt, chính xác, trọng tâm vào chương {chapter_num}):
"""


def build_method_prompt() -> str:
    """Build prompt cho câu hỏi về phương pháp nghiên cứu"""
    return """
Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn.

YÊU CẦU: Phân tích và trả lời về PHƯƠNG PHÁP NGHIÊN CỨU dựa trên tài liệu.

HƯỚNG DẪN TRẢ LỜI:
1. Xác định tài liệu (nhan đề, tác giả, năm xuất bản).
2. Nêu rõ phương pháp nghiên cứu được sử dụng (định tính, định lượng, hỗn hợp...).
3. Mô tả công cụ, quy trình, dữ liệu sử dụng.
4. Giải thích tại sao phương pháp này phù hợp.
5. Nếu tài liệu không cung cấp, hãy nói "Tài liệu không đề cập rõ phương pháp nghiên cứu".
6. Không viết thành một đoạn dài; luôn chia ý bằng tiêu đề ngắn và gạch đầu dòng.
7. 📌 CẤU TRÚC PHÂN CẤP: PHẢI giữ nguyên cấu trúc gốc từ tài liệu. Nếu phương pháp được trình bày theo "I, 1, 1.1, a, b, c..." thì tái hiện đúng thứ bậc đó. KHÔNG đánh số lại thành 1, 2, 3... liên tục.
8. Dùng ## cho mục lớn, ### cho mục vừa, #### cho mục nhỏ. KHÔNG dùng `-` trước heading.

TÀI LIỆU:
{context}

CÂU HỎI: {question}

TRẢ LỜI (Tiếng Việt, chi tiết, tập trung vào phương pháp):
"""


def build_general_chain():
    """Build general RAG chain with prompt selection based on question type"""
    
    prompt_text = """Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn (QNU Library Assistant).

⚠️ QUY TẮC VÀNG — TUYỆT ĐỐI TUÂN THỦ:
0. **KHÔNG ĐƯỢC DÙNG KIẾN THỨC BÊN NGOÀI.** Bạn CHỈ trả lời dựa trên TÀI LIỆU được cung cấp trong mục [TÀI LIỆU] bên dưới.
   - Nếu tài liệu KHÔNG chứa câu trả lời, hãy nói: *"Tài liệu không cung cấp thông tin này."*
   - KHÔNG suy luận, KHÔNG phỏng đoán, KHÔNG bịa đặt dữ liệu.
   - KHÔNG dùng kiến thức phổ thông, sách giáo khoa, hay bất kỳ nguồn nào khác ngoài tài liệu đã cho.

CÁC HƯỚNG DẪN KHÁC — CHỈ ÁP DỤNG KHI TÀI LIỆU CÓ CÂU TRẢ LỜI:
1. CHỈ dùng những tài liệu có liên quan TRỰC TIẾP đến câu hỏi của user. Nếu tài liệu không liên quan trực tiếp, hãy bỏ qua hoàn toàn.
2. Trả lời dựa CHÍNH XÁC trên nội dung tài liệu được cung cấp.
3. Nếu user hỏi về CHƯƠNG/PHẦN CỤ THỂ, tìm nội dung đó và trả lời chi tiết.
4. Nếu user hỏi về PHƯƠNG PHÁP NGHIÊN CỨU, hãy nêu rõ: công cụ, quy trình, dữ liệu.
5. Nếu user hỏi về KẾT QUẢ/KẾT LUẬN, liệt kê rõ ràng các phát hiện quan trọng.
6. Kèm link tài liệu bản số nếu có sẵn.
7. KHÔNG trả lời về những chủ đề không liên quan đến câu hỏi dù tài liệu có nhắc đến.
8. KHÔNG viết thành một đoạn văn dài. Luôn trình bày thành từng ý rõ ràng.

📌 CẤU TRÚC PHÂN CẤP — BẮT BUỘC TUÂN THỦ:
- Khi tài liệu gốc có cấu trúc phân cấp (I, II, III... → 1, 2, 3... → 1.1, 1.2... → a, b, c...), BẮT BUỘC tái hiện đúng cấu trúc đó trong câu trả lời.
- Ví dụ: Nếu tài liệu viết "I. Giới thiệu → 1. Bối cảnh → 1.1 Tại sao nghiên cứu → a. Thực trạng → b. Mục tiêu" thì phải giữ nguyên thứ bậc này.
- KHÔNG đánh số lại thành 1, 2, 3... liên tục nếu tài liệu gốc có cấu trúc phân cấp khác.
- Nếu tài liệu không có cấu trúc phân cấp rõ ràng, dùng markdown headings (##, ###) hoặc bullet points.

📌 QUY TẮC FORMAT (Markdown):
- Dùng ## cho mục lớn (A, B, C... hoặc I, II, III...), ### cho mục vừa (1, 2, 3...), #### cho mục nhỏ (1.1, 1.2...).
- Dùng **inbold** cho tên riêng, thuật ngữ quan trọng.
- KHÔNG dùng dấu `-` trước heading (mục lớn). Chỉ dùng `-` cho list item thực sự (nội dung chi tiết).
- Ví dụ ĐÚNG:
  ## A. Hệ thống
  - Cấu hình hệ thống (đơn vị, module điểm)
  - Từ điển dữ liệu: danh mục điểm thành phần

  Ví dụ SAI:
  - **A. Hệ thống** ← KHÔNG dùng `-` trước heading

📌 ĐỘ CHI TIẾT — BẮT BUỘC:
- Khi tài liệu có cấu trúc phân cấp nhiều tầng (A → I → 1 → 1.1 → a), PHẢI trình bày ĐẦY ĐỦ tất cả các tầng con, KHÔNG được tóm tắt bỏ qua.
- Chỉ tóm tắt khi user yêu cầu "tóm tắt". Nếu user hỏi nội dung → trả lời CHI TIẾT theo đúng cấu trúc tài liệu.
- Nếu có 5 mục con → liệt kê cả 5. Nếu có 3 cấp → hiển thị cả 3 cấp.

[TÀI LIỆU]:
{context}

[CÂU HỎI]: {question}

[TRẢ LỜI] (Tiếng Việt, chi tiết theo cấu trúc tài liệu, KHÔNG tóm tắt bỏ qua mục con; tuân thủ tuyệt đối Quy tắc vàng số 0):"""
    
    def rag_chain(context: str, question: str) -> str:
        # Detect question type
        chapter_num = detect_chapter_reference(question)
        is_method = detect_method_question(question)
        is_result = detect_result_question(question)
        is_structure = detect_structure_query(question)

        # Choose appropriate prompt
        if chapter_num:
            prompt = build_chapter_prompt(chapter_num)
        elif is_method:
            prompt = build_method_prompt()
        elif is_structure:
            prompt = build_structure_prompt()
        else:
            prompt = prompt_text

        messages = [
            {"role": "system", "content": "Bạn chỉ được trả về phần trả lời cuối cùng bằng tiếng Việt. Không hiển thị suy luận, không dùng thẻ <think>, không viết lời dẫn như 'Okay, the user...'. Trả lời dựa trên nội dung tài liệu được cung cấp bên dưới. Nếu câu hỏi thực sự không liên quan đến tài liệu, hãy nói ngắn gọn rằng tài liệu không chứa thông tin này."},
            {"role": "user", "content": prompt.format(context=context, question=question)}
        ]
        return sanitize_answer_text(call_llm(messages, temperature=0.1))
    
    return rag_chain, None

# ── Endpoint /suggest-related ────────────────────────────
class SuggestRequest(BaseModel):
    title: str = ""
    author: str = ""
    subject: str = ""
    top_k: int = 8
    exclude_titles: list[str] = []

@app.post("/suggest-related")
async def suggest_related(req: SuggestRequest):
    """Gợi ý tài liệu liên quan (cùng tác giả, cùng chủ đề, cùng ngành)"""
    try:
        if not any([req.title, req.author, req.subject]):
            return {"suggestions": [], "total": 0, "error": "Need at least one filter"}

        # Build expanded query
        query_parts = []
        if req.title:
            query_parts.append(req.title)
        if req.author:
            query_parts.append(req.author)
        if req.subject:
            query_parts.append(req.subject)
        query = " ".join(query_parts)

        results = retriever.search_by_query(query, top_k=req.top_k * 5)

        # Filter out already shown titles
        exclude_set = {t.lower().strip() for t in req.exclude_titles if t}

        # Build search query from specific fields for precise matching
        # Prefer subject/major/author for high-precision results
        search_query = " ".join(query_parts)
        results = retriever.search_by_query(search_query, top_k=req.top_k * 5)

        # Filter out the original document and already shown
        filtered = []
        seen_titles = set()
        has_author_filter = bool(req.author)
        has_subject_filter = bool(req.subject)

        for r in results:
            doc = r['doc']
            title_lower = doc['title'].lower().strip()

            # Skip same title
            if req.title and title_lower == req.title.lower().strip():
                continue
            if title_lower in exclude_set:
                continue
            if title_lower in seen_titles:
                continue
            seen_titles.add(title_lower)

            # Determine reason — only keep docs with actual matches
            reasons = []
            if has_author_filter and req.author.lower() in doc.get('author', '').lower():
                reasons.append("Cùng tác giả")
            if has_subject_filter:
                doc_subject = doc.get('subject', '').lower()
                doc_major = doc.get('major', '').lower()
                search_subject = req.subject.lower()
                if search_subject in doc_subject:
                    reasons.append("Cùng chủ đề")
                if search_subject in doc_major:
                    reasons.append("Cùng ngành học")

            # Only include if it has a real reason (skip generic "Liên quan đến truy vấn")
            if not reasons:
                continue

            filtered.append({
                "title": doc['title'],
                "author": doc.get('author', 'Unknown'),
                "year": doc.get('year', 'N/A'),
                "subject": doc.get('subject', 'N/A'),
                "major": doc.get('major', ''),
                "publisher": doc.get('publisher', ''),
                "doc_type": doc.get('doc_type', ''),
                "ddc": doc.get('ddc', ''),
                "location": doc.get('location', ''),
                "abstract": (doc.get('abstract', '') or '')[:500],
                "link": doc.get('link', ''),
                "reason": "; ".join(reasons),
                "score": round(r['score'], 2),
            })

        return {"suggestions": filtered[:req.top_k], "total": len(filtered)}

    except Exception as e:
        return {"error": str(e), "suggestions": [], "total": 0}

# ── Endpoint /spell-check ────────────────────────────────
class SpellCheckRequest(BaseModel):
    text: str

@app.post("/spell-check")
async def spell_check_endpoint(req: SpellCheckRequest):
    """Kiểm tra và sửa lỗi chính tả"""
    try:
        original = req.text
        corrected = spell_correct(original)
        term_suggestions = suggest_terms_for_query(original)

        return {
            "original": original,
            "corrected": corrected,
            "is_correct": original.lower().strip() == corrected.lower().strip(),
            "spelling_corrections": term_suggestions.get('spelling_corrections', []),
            "synonym_expansions": term_suggestions.get('synonym_expansions', []),
            "en_terms": term_suggestions.get('en_terms', []),
        }
    except Exception as e:
        return {"error": str(e), "original": req.text, "corrected": req.text}


# ── Endpoint /suggest-terms ──────────────────────────────
class TermRequest(BaseModel):
    term: str
    language: str = "vi"

@app.post("/suggest-terms")
async def suggest_terms(req: TermRequest):
    """Gợi ý từ đồng nghĩa, thuật ngữ Anh-Việt, thuật ngữ chuyên ngành"""
    try:
        term = req.term.lower()

        # 1. Local synonym lookup
        local_suggestions = suggest_synonyms(term)

        # 2. Spelling correction
        correction = spell_correct(term)

        # 3. Tìm tài liệu chứa từ này để lấy context
        results = retriever.search_by_query(term, top_k=5)
        context_docs = [{
            "title": r['doc'].get('title', ''),
            "subject": r['doc'].get('subject', ''),
            "major": r['doc'].get('major', ''),
        } for r in results]

        # 4. Use LLM to expand suggestions with domain context
        context = " ".join([r['doc'].get('text', '')[:600] for r in results])

        prompt_text = f"""Dựa trên đoạn văn bản sau, hãy gợi ý cho thuật ngữ "{term}":

1. Từ đồng nghĩa trong tiếng Việt
2. Thuật ngữ tiếng Anh tương ứng (nếu là tiếng Việt)
3. Thuật ngữ chuyên ngành liên quan
4. Các khái niệm mở rộng (nếu có)

ĐOẠN VĂN BẢN:
{context}

TRẢ LỜI (dạng danh sách, mỗi mục cách nhau bằng dấu xuống dòng, ghi rõ loại):"""

        llm_suggestions = []
        try:
            messages = [{"role": "user", "content": prompt_text}]
            llm_response = call_llm(messages, temperature=0.1)
            llm_suggestions = [s.strip() for s in llm_response.split('\n') if s.strip()]
        except Exception:
            llm_suggestions = []

        return {
            "term": req.term,
            "synonyms": local_suggestions.get('synonyms', []),
            "en_terms": local_suggestions.get('en_terms', []),
            "spelling_correction": correction if correction != term else "",
            "llm_suggestions": llm_suggestions[:10],
            "context_docs": context_docs[:5],
        }

    except Exception as e:
        return {"error": str(e), "synonyms": [], "en_terms": [], "llm_suggestions": []}

# ── Endpoint /documents (get all) ──────────────────────────
@app.get("/documents")
async def list_all_documents(page: int = 1, limit: int = 20):
    """Liệt kê tất cả tài liệu (có phân trang)"""
    try:
        all_docs = retriever.bm25.documents if hasattr(retriever, 'bm25') else []
        start = (page - 1) * limit
        end = start + limit
        page_docs = all_docs[start:end]

        docs = []
        seen = set()
        for doc in page_docs:
            title = doc.get('title', '')
            key = title.lower()[:80]
            if not key or key in seen:
                continue
            seen.add(key)
            docs.append({
                "title": doc.get('title', ''),
                "author": doc.get('author', 'Unknown'),
                "year": doc.get('year', 'N/A'),
                "subject": doc.get('subject', 'N/A'),
                "major": doc.get('major', ''),
                "publisher": doc.get('publisher', ''),
                "doc_type": doc.get('doc_type', 'N/A'),
                "link": doc.get('link', ''),
                "location": doc.get('location', ''),
                "abstract": doc.get('abstract', ''),
                "ddc": doc.get('ddc', ''),
                "notes": doc.get('notes', ''),
            })

        return {"results": docs, "total": len(all_docs), "page": page, "limit": limit}
    except Exception as e:
        return {"error": str(e), "results": [], "total": 0}

# ── Endpoint /search (with filters) ──────────────────────
class SearchRequest(BaseModel):
    query: str = ""
    author: str = ""
    year: str = ""
    subject: str = ""
    major: str = ""
    doc_type: str = ""
    top_k: int = 10

@app.post("/search")
async def search_documents(req: SearchRequest):
    """Tìm kiếm tài liệu với bộ lọc (tác giả, năm, chủ đề, ngành, loại tài liệu)"""
    try:
        if not any([req.query, req.author, req.subject, req.major]):
            return {"results": [], "total": 0}

        # Build query from all fields
        query_parts = []
        if req.query:
            query_parts.append(req.query)
        if req.author:
            query_parts.append(req.author)
        if req.subject:
            query_parts.append(req.subject)
        if req.major:
            query_parts.append(req.major)
        query = " ".join(query_parts)

        filters = {}
        if req.author:
            filters['author'] = req.author
        if req.subject:
            filters['subject'] = req.subject
        if req.year:
            filters['year'] = req.year
        if req.major:
            filters['major'] = req.major

        results = retriever.search_with_filters(query, filters=filters, top_k=req.top_k)

        docs = []
        seen = set()
        for r in results:
            doc = r['doc']
            key = doc['title'].lower()[:80]
            if key in seen:
                continue
            seen.add(key)

            docs.append({
                "title": doc.get('title', ''),
                "author": doc.get('author', 'Unknown'),
                "year": doc.get('year', 'N/A'),
                "subject": doc.get('subject', 'N/A'),
                "major": doc.get('major', ''),
                "publisher": doc.get('publisher', ''),
                "place": doc.get('place', ''),
                "edition": doc.get('edition', ''),
                "bilingual_title": doc.get('bilingual_title', ''),
                "doc_type": doc.get('doc_type', 'N/A'),
                "link": doc.get('link', ''),
                "location": doc.get('location', ''),
                "course": doc.get('course', ''),
                "keywords": doc.get('keywords', []) or [],
                "abstract": doc.get('abstract', ''),
                "ddc": doc.get('ddc', ''),
                "notes": doc.get('notes', ''),
                "score": round(r['score'], 2),
            })

        return {"results": docs, "total": len(docs)}

    except Exception as e:
        return {"error": str(e), "results": [], "total": 0}


# ── Follow-up suggestions ────────────────────────────────

def extract_answer_headings(answer: str) -> list[str]:
    """Trích xuất tên các heading/section từ nội dung answer markdown.
    VD: '## Chương 1: Tổng quan' → ['Chương 1: Tổng quan']
    """
    headings = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        m = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if m:
            heading = m.group(1).strip()
            if len(heading) >= 4 and not re.match(r"^(trang|page|\[)", heading, re.I):
                headings.append(heading)
    return headings[:6]


def extract_answer_sections(answer: str) -> list[str]:
    """Trích xuất tên các phần/chapter từ answer (markdown headings + bold + numbered).
    """
    sections = []
    seen = set()

    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue

        name = None

        # Markdown heading: ##, ###, ####
        m = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if m:
            name = m.group(1).strip()

        # Bold title: **some text**
        if not name:
            m = re.match(r"^\*\*(.+?)\*\*$", stripped)
            if m and len(m.group(1)) >= 5:
                name = m.group(1).strip()

        # Numbered heading: "1. Title", "1.1 Title", "I. Title"
        if not name:
            m = re.match(r"^(?:\d+(?:\.\d+)*|[IVX]+)\.\s+(.{5,60})$", stripped)
            if m:
                name = m.group(1).strip()

        if name and 5 <= len(name) <= 80:
            name_key = name.lower().strip()
            if name_key not in seen:
                seen.add(name_key)
                sections.append(name)

        if len(sections) >= 5:
            break

    return sections


def build_contextual_suggestions(
    answer: str,
    query: str,
    sources: list[dict] = None,
    doc_title: str = "",
    catalog_mode: bool = False,
    has_pdf_sources: bool = False,
) -> list[dict]:
    """Build follow-up suggestions dựa trên nội dung thực tế của answer.
    - Catalog mode: gợi ý tìm sách cùng tác giả/chủ đề/ngành
    - Content mode: gợi ý phân tích section, phương pháp, so sánh...
    """
    suggestions = []
    answer_lower = normalize_text(answer or "")
    query_lower = normalize_text(query or "")
    sources = sources or []

    # ══════════════════════════════════════════════════════════
    # CATALOG MODE — gợi ý tìm sách/thuộc tính
    # ══════════════════════════════════════════════════════════
    if catalog_mode:
        _junk_authors = {
            "aspose", "aspose.pdf", "aspose.pdf.epubtopdfconverter",
            "unknown", "n/a", "khong xac dinh", "pdf converter",
            "epubtopdfconverter", "microsoft", "adobe", "apache",
        }
        _junk_subjects = {"aspose", "n/a", "unknown", "khong xac dinh", "pdf", "tai lieu so"}

        # Extract metadata from first results
        first_author = ""
        first_subject = ""
        first_major = ""
        first_title = ""
        for s in sources[:5]:
            a = (s.get("author") or "").strip()
            if a and len(a) >= 3 and a.lower() not in _junk_authors and not any(kw in a.lower() for kw in ("aspose", "converter")):
                first_author = a
            subj = (s.get("subject") or "").strip()
            if subj and subj.lower() not in _junk_subjects and len(subj) >= 2:
                first_subject = subj
            major = (s.get("major") or "").strip()
            if major and len(major) >= 2:
                first_major = major
            t = (s.get("title") or "").strip()
            if t and len(t) > 5:
                first_title = t.split(" - ")[0].strip()[:80]

        # Ưu tiên: ngành → chủ đề → tác giả → gợi ý chung
        if first_major:
            suggestions.append({
                "label": f"Tìm sách theo ngành: {first_major}",
                "query": f"tìm sách thuộc ngành {first_major}"
            })
        if first_subject:
            suggestions.append({
                "label": f"Sách cùng chủ đề: {first_subject}",
                "query": f"tìm sách về {first_subject}"
            })
        if first_author:
            suggestions.append({
                "label": f"Sách cùng tác giả: {first_author}",
                "query": f"sách của tác giả {first_author}"
            })

        # Fallback nếu không đủ
        if len(suggestions) < 3 and first_title:
            suggestions.append({
                "label": f"Tìm sách liên quan",
                "query": f"tìm sách liên quan đến {first_title}"
            })
        if len(suggestions) < 3:
            suggestions.append({"label": "Duyệt danh mục sách", "query": "liệt kê sách mới nhất"})
        if len(suggestions) < 3:
            suggestions.append({"label": "Gợi ý sách phổ biến", "query": "gợi ý tài liệu phổ biến"})

        # Dedupe & limit
        seen = set()
        unique = []
        for s in suggestions:
            q = s["query"].strip().lower()
            if q not in seen and q:
                seen.add(q)
                unique.append(s)
            if len(unique) >= 3:
                break
        return unique

    # ══════════════════════════════════════════════════════════
    # CONTENT MODE — gợi ý phân tích nội dung tài liệu
    # ══════════════════════════════════════════════════════════
    sections = extract_answer_sections(answer)
    headings = extract_answer_headings(answer)

    # ── Priority 1: Gợi ý phân tích section cụ thể từ answer ──
    for section in sections[:3]:
        section_clean = re.sub(r"^#{1,4}\s*", "", section).strip()
        section_clean = re.sub(r"\*\*|\*", "", section_clean).strip()
        if len(section_clean) >= 5:
            suggestions.append({
                "label": f"Phân tích: {section_clean[:50]}",
                "query": f"Phân tích chi tiết nội dung phần '{section_clean}'"
            })

    # ── Priority 2: Query + answer pattern context ──
    if len(suggestions) < 3:
        if re.search(r"phuong phap|methodology|cach tiep can|cong cu|ky thuat|thuat toan", answer_lower):
            method_names = re.findall(r"(?:\*\*|(?:^##?\s+))([^*\n]{5,50})", answer)
            if method_names:
                m = method_names[0].strip().strip("#*").strip()
                suggestions.append({
                    "label": f"Giải thích: {m[:40]}",
                    "query": f"Giải thích chi tiết hơn về {m}"
                })
            else:
                suggestions.append({"label": "Nói rõ hơn về phương pháp", "query": "Giải thích chi tiết hơn về phương pháp được sử dụng"})

        if re.search(r"ket qua|finding|conclusion|ket luan|danh gia", answer_lower):
            suggestions.append({"label": "Phân tích kết quả", "query": "Phân tích chi tiết hơn về kết quả và đánh giá"})

        if re.search(r"lo ich|uu diem|diem manh|advantage|benefit", answer_lower):
            suggestions.append({"label": "Điểm yếu và hạn chế", "query": "Liệt kê các điểm yếu và hạn chế"})

    # ── Priority 4: Query-based ──
    if len(suggestions) < 3:
        if re.search(r"la gi|dinh nghia|khai niem", query_lower):
            suggestions.append({"label": "Ví dụ minh họa cụ thể", "query": "Cho ví dụ minh họa cụ thể"})
        if re.search(r"so sanh|khac biet", query_lower):
            suggestions.append({"label": "So sánh chi tiết hơn", "query": "So sánh chi tiết hơn giữa các yếu tố"})
        if re.search(r"bo cuc|cau truc|muc luc", query_lower):
            suggestions.append({"label": "Tóm tắt nội dung chính", "query": "Tóm tắt nội dung chính của tài liệu"})

    # ── Priority 4: Smart fallback từ sources metadata ──
    if len(suggestions) < 3 and sources:
        # Lấy subject thực tế từ sources
        for s in sources[:3]:
            subj = (s.get("subject") or "").strip()
            if subj and len(subj) >= 3:
                suggestions.append({
                    "label": f"Kiến thức liên quan: {subj[:40]}",
                    "query": f"Kiến thức liên quan đến {subj}"
                })
                break
    if len(suggestions) < 3 and sources:
        for s in sources[:3]:
            author = (s.get("author") or "").strip()
            if author and len(author) >= 3:
                suggestions.append({
                    "label": f"Sách cùng tác giả: {author}",
                    "query": f"sách của tác giả {author}"
                })
                break
    if len(suggestions) < 3 and answer and len(answer) > 200:
        suggestions.append({"label": "Tóm tắt nội dung", "query": "Tóm tắt lại nội dung thành 3-5 ý chính"})

    # Dedupe & limit
    seen = set()
    unique = []
    for s in suggestions:
        q = s["query"].strip().lower()
        if q not in seen and q:
            seen.add(q)
            unique.append(s)
        if len(unique) >= 3:
            break
    return unique


def build_follow_up_suggestions(
    query: str,
    sources: list[dict],
    answer: str,
    catalog_mode: bool,
    has_pdf_sources: bool,
    session_id: str,
) -> list[dict]:
    """Generate contextual follow-up suggestions. Each item: {label, query}.
    Delegates to build_contextual_suggestions with correct mode + has_pdf flag.
    """
    return build_contextual_suggestions(
        answer=answer,
        query=query,
        sources=sources,
        doc_title=sources[0].get("title", "") if sources else "",
        catalog_mode=catalog_mode,
        has_pdf_sources=has_pdf_sources,
    )


# ── Endpoint /chat ────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # 🔥 OPTIMIZED: Normalize query for better matching
        normalized_query = normalize_text(req.query)
        
        # Lấy lịch sử session
        # Phan loai intent truoc khi xu ly
        intent = classify_intent(req.query)
        logger.info("Intent: %s for query: %s", intent, req.query)

        # Chitchat / unknown → tra ve cung, khong goi LLM
        if intent == "chitchat":
            return ChatResponse(
                answer="Xin chào! Mình là trợ lý Thư viện Trường Đại học Quy Nhơn. Mình có thể giúp gì cho bạn về tra cứu sách, tài liệu số, hoặc nội dung các tài liệu trong thư viện ạ? 📚",
                sources=[],
                total_sources=0,
                total_found=0,
                summary="",
                current_document=None,
                follow_up_suggestions=[],
            )
        if intent == "live_info":
            return ChatResponse(
                answer="Tài liệu không cung cấp thông tin này.",
                sources=[],
                total_sources=0,
                total_found=0,
                summary="",
                current_document=None,
                follow_up_suggestions=[],
            )
        # unknown → nếu có PDF, thử search (có thể user hỏi nội dung file mới upload)
        # nếu không có PDF, trả lời trực tiếp "không có thông tin"
        if intent == "unknown":
            try:
                _has_pdfs = len(pdf_manager.list_all_pdfs()) > 0
            except Exception:
                _has_pdfs = False
            if not _has_pdfs:
                logger.info("Unknown intent %r → không có PDF, trả lời 'không có thông tin'", req.query)
                return ChatResponse(
                    answer="Tài liệu không cung cấp thông tin này.",
                    sources=[],
                    total_sources=0,
                    total_found=0,
                    summary="",
                    current_document=None,
                    follow_up_suggestions=[],
                )
            logger.info("Unknown intent %r → có PDF, thử search...", req.query)

        # clarify -> Hoi lai user vi cau hoi chua ro y dinh
        if intent == "clarify":
            logger.info("Clarify intent %r -> hoi lai user", req.query)
            query_normalized = normalize_text(req.query).strip()
            if "the loai" in query_normalized:
                answer = ("Bạn đang muốn: **TÌM SÁCH** về thể loại này, "
                         "hay hỏi về **NỘI DUNG** thể loại văn học?\n\n"
                         "Vui lòng chọn:")
            elif "tac gia" in query_normalized:
                answer = ("Bạn đang muốn: **TÌM SÁCH** của tác giả này, "
                         "hay hỏi về **THÔNG TIN** tác giả?\n\n"
                         "Vui lòng chọn:")
            elif "nha xuat ban" in query_normalized:
                answer = ("Bạn đang muốn: **TÌM SÁCH** của nhà xuất bản này, "
                         "hay hỏi về **THÔNG TIN** nhà xuất bản?\n\n"
                         "Vui lòng chọn:")
            else:
                answer = ("Bạn đang muốn **TÌM SÁCH** hay **HỎI VỀ NỘI DUNG** liên quan?\n\n"
                         "Vui lòng chọn:")
            return ChatResponse(
                answer=answer,
                sources=[],
                total_sources=0,
                total_found=0,
                summary="",
                current_document=None,
                follow_up_suggestions=[
                    {"label": "Tìm sách", "query": "tim sach ve " + query_normalized},
                    {"label": "Hỏi về nội dung", "query": "noi dung ve " + query_normalized}
                ],
            )

        hist = session_history.get(req.session_id, [])
        request_content = detect_content_request(req.query)
        is_list_request = detect_list_request(req.query)
        catalog_mode = is_list_request or (not request_content and intent != "unknown")

        # Handle greetings & chitchat — respond naturally, no search
        if detect_greeting(req.query):
            greeting_answer = (
                "Xin chào bạn!\n\n"
                "Mình là trợ lý Thư viện Đại học Quy Nhơn. Mình có thể giúp bạn:\n\n"
                "- **Tìm sách, tài liệu** theo tên, tác giả hoặc chủ đề\n"
                "- **Gợi ý tài liệu** liên quan đến lĩnh vực bạn quan tâm\n"
                "- **Tóm tắt nội dung** tài liệu có sẵn trong kho\n\n"
                "Bạn đang muốn tìm tài liệu gì? Hãy thử:\n"
                "- *\"Tìm sách về trí tuệ nhân tạo\"*\n"
                "- *\"Có tài liệu về văn học Việt Nam không?\"*\n"
                "- *\"Sách của tác giả Nguyễn Văn A\"*"
            )
            return ChatResponse(
                answer=greeting_answer,
                sources=[],
                total_sources=0,
                summary="",
                current_document=None,
                follow_up_suggestions=[],
            )

        # Handle exhaustion queries ("hết chưa?", "còn tài liệu nữa không?")
        if detect_exhaustion_query(req.query):
            # Phải dùng CHỦ ĐỀ từ lịch sử session, KHÔNG dùng query gốc
            # (vì query gốc chỉ là câu hỏi "hỏi thêm", không chứa chủ đề)
            entry_count = len(hist)
            if entry_count > 0:
                last_topic = hist[-1].get("topic", "")
                last_search = hist[-1].get("search_query", "")
                query_for_count = last_topic or last_search or normalized_query
            else:
                query_for_count = normalized_query

            # Nếu vẫn không có chủ đề rõ ràng từ history, thử trích xuất từ query
            if query_for_count == normalized_query:
                # Loại bỏ các từ hỏi chung chung để lấy từ khóa còn lại
                stripped = re.sub(r"\b(vay|con|khong|nua|thi|da|het|bao nhieu|them|nào|không|còn|nữa|thì|đã|hết|bao nhiêu|thêm)\b", " ", normalized_query, flags=re.IGNORECASE)
                stripped = re.sub(r"\s+", " ", stripped).strip()
                if stripped:
                    query_for_count = stripped

            # Count total matching docs (không giới hạn top_k)
            total_count = retriever.count_matching(query_for_count)

            # Trả lời tự nhiên dựa trên context
            if entry_count == 0:
                answer_text = (
                    "Bạn chưa hỏi chủ đề nào trước đó. "
                    "Hãy thử hỏi cụ thể hơn, ví dụ: **tìm sách về trí tuệ nhân tạo**, "
                    "**có tài liệu về giáo dục không?**"
                )
            elif total_count <= 5:
                answer_text = (
                    f"Chỉ có **{total_count} tài liệu** liên quan đến chủ đề này "
                    f"trong kho lưu trữ. Đó là tất cả kết quả rồi!"
                )
            else:
                # Đếm số đã hiển thị từ history
                shown_titles = set()
                for h in hist:
                    for title in h.get("shown_titles", []):
                        if title:
                            shown_titles.add(title.lower().strip())
                shown_count = len(shown_titles)

                answer_text = (
                    f"Còn khoảng **{total_count} tài liệu** liên quan đến chủ đề này. "
                    f"Bạn đã xem **{shown_count} tài liệu** trước đó.\n\n"
                    f"Hãy thử hỏi cụ thể hơn để lọc kết quả, ví dụ:\n"
                    f"- **tên sách** hoặc **tác giả** cụ thể\n"
                    f"- **năm xuất bản** bạn muốn tìm\n"
                    f"- **chủ đề chi tiết hơn**"
                )

            return ChatResponse(
                answer=answer_text,
                sources=[],
                total_sources=0,
                total_found=total_count,
                summary="",
                current_document=None,
                follow_up_suggestions=[],
            )


        # ── Xử lý yêu cầu __LIST_TOPIC__:<topic> (từ follow-up của existence check) ──
        list_topic_marker = "__LIST_TOPIC__:"
        list_topic_query = None
        if req.query.startswith(list_topic_marker):
            list_topic_query = req.query[len(list_topic_marker):].strip()
            logger.info("List-topic follow-up: '%s'", list_topic_query)

        # ── Handle existence questions ("Có tài liệu về X không?") ──
        # ── Handle user declining to list ("không", "cảm ơn", ...) sau khi bot đã hỏi "có muốn liệt kê không?" ──
        _decline_pat = re.compile(
            r"^\s*(không|khong|thôi|thoi|thôi nhé|thoi nha|cảm ơn|cam on|cám ơn|"
            r"cam_on|thank|thanks|no|nah|ok|okie|okela|đủ rồi|du roi|"
            r"kệ|ke|thôi bỏ qua|thoi bo qua)\s*[.!]?\s*$",
            re.IGNORECASE,
        )
        if hist and _decline_pat.match(req.query.strip()):
            last_topic = _last_existence_topic(hist)
            if last_topic:
                answer_text = (
                    f"Được, mình sẽ không liệt kê **{last_topic}** nữa. "
                    f"Bạn muốn mình hỗ trợ gì khác không?"
                )
                hist.append({"role": "assistant", "content": answer_text})
                return ChatResponse(
                    answer=answer_text,
                    sources=[],
                    total_sources=0,
                    total_found=0,
                    summary="",
                    current_document=None,
                    follow_up_suggestions=[
                        {"label": "Gợi ý chủ đề khác", "query": "Gợi ý tài liệu phổ biến"},
                        {"label": "Tìm sách theo tác giả", "query": "sách của Nguyễn Nhật Ánh"},
                        {"label": "Xem giới thiệu thư viện", "query": "Giới thiệu thư viện QNU"},
                    ],
                )

        if detect_existence_request(req.query):
            topic = extract_existence_topic(req.query)
            topic_label = topic or "chủ đề này"
            if topic:
                count = retriever.count_matching(topic)
                if count <= 0:
                    answer_text = (
                        f"Hiện tại mình chưa tìm thấy tài liệu nào về **{topic_label}** "
                        f"trong thư viện. Bạn có thể thử hỏi chủ đề khác hoặc diễn đạt "
                        f"khác đi một chút nhé."
                    )
                    return ChatResponse(
                        answer=answer_text,
                        sources=[],
                        total_sources=0,
                        total_found=0,
                        summary="",
                        current_document=None,
                        follow_up_suggestions=[
                            {"label": "Gợi ý chủ đề khác", "query": "Gợi ý tài liệu phổ biến"},
                            {"label": "Tìm chủ đề tương tự", "query": f"tài liệu liên quan đến {topic_label}"},
                        ],
                    )

                # Có tài liệu → hỏi người dùng có muốn liệt kê không (không hiển thị con số)
                answer_text = (
                    f"Có, thư viện hiện có tài liệu về **{topic_label}**. "
                    f"Bạn có muốn mình liệt kê chi tiết ra không? "
                    f"(Quá trình tìm và hiển thị có thể tốn một chút thời gian, bạn vui lòng chờ nhé.)"
                )
                return ChatResponse(
                    answer=answer_text,
                    sources=[],
                    total_sources=0,
                    total_found=count,
                    summary="",
                    current_document=None,
                    follow_up_suggestions=[
                        {
                            "label": "Có, liệt kê chi tiết ra giúp mình",
                            "query": f"{list_topic_marker}{topic}",
                        },
                        {
                            "label": "Không, cảm ơn bạn",
                            "query": "Cảm ơn bạn",
                        },
                    ],
                )

        # ── Follow-up từ existence: list topic thay vì trả lời bằng chính marker ──
        if list_topic_query:
            req_query_for_search = f"tìm sách về {list_topic_query}"
            req.query = req_query_for_search
            normalized_query = normalize_text(req_query_for_search)
            is_list_request = True
            catalog_mode = True
            logger.info("List-topic converted to: '%s'", req_query_for_search)

        # ── Auto-detect major/ngành from natural-language queries ──
        # When the user asks things like "Tìm sách thuộc ngành Công nghệ thông tin"
        # the frontend does not send a `major` field, so the major filter at the
        # end of the search flow is never triggered. Detect patterns like
        # "ngành X", "lĩnh vực X", "chủ đề X" using the existing marker regex
        # and inject the canonical major so the major filter can run.
        if not req.major and not req.document_id:
            try:
                _detected_major = _extract_major_from_query(req.query)
                if _detected_major:
                    req.major = _detected_major
                    logger.info(
                        "Auto-detected major from query: '%s' (query='%s')",
                        _detected_major, req.query,
                    )
            except Exception as _e:
                logger.debug("Major auto-detect failed: %s", _e)

        # Auto-reindex nếu folder pdfs/ hoặc Data/ thay đổi
        file_changes = check_files_changed()
        if file_changes["pdf"]:
            logger.info("PDF folder changed → auto reindexing...")
            try:
                reindex_pdfs(retriever)
            except Exception as _e:
                logger.warning("Auto reindex PDF failed: %s", _e)
        if file_changes["data"]:
            logger.info("Data folder changed (XLSX/CSV) → reloading...")
            try:
                # Xóa documents cache để startup lần sau cũng load lại
                _cache_pkl = Path("Data/_documents_cache.pkl")
                if _cache_pkl.exists():
                    _cache_pkl.unlink()
                # Load lại documents từ Data/ và rebuild BM25 index tại chỗ
                new_documents = DocumentIndexer.load_from_data("Data")
                retriever.bm25.documents = new_documents
                retriever.bm25._build_index()
                # Reset semantic engine để load lại với data mới
                global _semantic_engine
                _semantic_engine = None
                reindex_pdfs(retriever)
                _build_topic_cache()
                _search_cache.invalidate()
                logger.info("Data reloaded: %d documents", len(new_documents))
            except Exception as _e:
                logger.warning("Auto reload Data failed: %s", _e)

        search_query = resolve_search_query(
            req.query,
            hist,
            catalog_mode=is_list_request
        )
        if search_query != normalized_query:
            logger.info("Context-aware search: '%s' → '%s'", normalized_query, search_query)

        if req.major:
            major_clean = (req.major or "").strip()
            if major_clean and major_clean.lower() not in search_query.lower():
                search_query = f"{search_query} {major_clean}".strip()
                logger.info("Major filter injected into query: '%s'", search_query)

        # ── Search Result Cache: check first for catalog queries ──
        # Thu hep: chi strict khi user that su hoi tac gia/nha xuat ban, khong vi list
        _prefer_strict = detect_author_query(req.query) or detect_publisher_query(req.query)
        cache_key = f"catalog:{search_query}:{_prefer_strict}:{req.major or ''}" if catalog_mode else None
        use_cache = catalog_mode and not req.document_id and not req.query.startswith("__SUGGEST")
        search_results = _search_cache.get(cache_key) if use_cache else None

        if search_results is not None:
            logger.info("Search cache HIT (%d docs, key='%s')", len(search_results), search_query)
        else:
            # Retrieve documents: try topic cache first, then merge with BM25
            retrieval_k = 500 if catalog_mode else 30

            cache_results = []
            # Bat lai topic cache cho ca author/publisher (truoc day bi bo qua)
            if catalog_mode:
                cache_results = _search_topic_cache(search_query)
                if cache_results:
                    logger.info("Topic cache HIT: %d docs for '%s'", len(cache_results), search_query)

            # Always run BM25 to catch text/abstract/notes matches too
            bm25_results = retriever.search_by_query(search_query, top_k=retrieval_k)
            # Catalog mode: loại bỏ PDF pages (chỉ giữ tài liệu từ catalog DB)
            # PDF pages thường match từ khóa chung nhưng không liên quan đến chủ đề catalog
            if catalog_mode and bm25_results:
                filtered = []
                for r in bm25_results:
                    doc = r.get("doc", {})
                    csv_file = str(doc.get("csv_file", ""))
                    is_pdf = csv_file.startswith("pdf:")
                    is_pdf_page = doc.get("page") is not None
                    # Giữ tài liệu từ catalog DB, loại PDF pages
                    if not is_pdf and not is_pdf_page:
                        filtered.append(r)
                bm25_results = filtered
                logger.info("Catalog mode: filtered %d PDF pages, keeping %d catalog docs", (len(filtered) - len(bm25_results)) * -1, len(filtered))
            catalog_query_token_count = len(topic_tokens(strip_search_intent_phrases(req.query)))
            use_metadata_search = catalog_mode and (
                is_list_request or catalog_query_token_count <= 7
            )
            if use_metadata_search:
                metadata_results = metadata_search_by_query(req.query, top_k=retrieval_k)
                bm25_results = merge_search_results(metadata_results, bm25_results)

            # Merge: cache results first (subject/major match), then BM25 (text match)
            if cache_results:
                seen = set()
                merged = []
                for r in cache_results:
                    key = (r["doc"].get("title", "").lower(), r["doc"].get("author", "").lower())
                    if key not in seen:
                        seen.add(key)
                        merged.append(r)
                for r in bm25_results:
                    key = (r["doc"].get("title", "").lower(), r["doc"].get("author", "").lower())
                    if key not in seen:
                        seen.add(key)
                        merged.append(r)
                search_results = merged
                logger.info("Merged: %d cache + %d bm25 = %d total", len(cache_results), len(bm25_results), len(search_results))
            else:
                search_results = bm25_results

            is_author_pub_query = detect_author_query(req.query) or detect_publisher_query(req.query)
            bm25_top_score = bm25_results[0].get("score", 0) if bm25_results else 0
            if (
                not is_author_pub_query
                and not catalog_mode  # catalog mode đã có cache riêng
                and (len(bm25_results) < 3 or bm25_top_score < 1.0)
            ):
                try:
                    sem_results, sem_mode = hybrid_search_with_semantic_fallback(
                        query=search_query,
                        bm25_top_k=20,
                        final_top_k=15,
                        alpha=0.5,
                    )
                    if sem_results and sem_mode == "hybrid":
                        # Merge với BM25 (ưu tiên hybrid scores)
                        seen_keys = set()
                        final = []
                        for r in sem_results:
                            key = (r["doc"].get("title", "").lower(), r["doc"].get("author", "").lower())
                            if key not in seen_keys:
                                seen_keys.add(key)
                                final.append(r)
                        for r in bm25_results:
                            key = (r["doc"].get("title", "").lower(), r["doc"].get("author", "").lower())
                            if key not in seen_keys:
                                seen_keys.add(key)
                                final.append(r)
                        search_results = final
                        logger.info("Semantic boost: %d hybrid + %d bm25", len(sem_results), len(bm25_results))
                except Exception as e:
                    logger.warning("Semantic search fallback failed: %s", e)

            search_results = rerank_results_for_query(
                req.query,
                search_results,
                prefer_strict=_prefer_strict
            )
            if catalog_mode:
                search_results = collapse_catalog_pdf_pages(search_results)

            # Store in cache for next time
            if use_cache and search_results:
                _search_cache.set(cache_key, search_results)
                logger.info("Search cached: %d docs (%s)", len(search_results), _search_cache.stats())

        if req.major and not req.document_id:
            major_norm = normalize_text(req.major).strip()
            if major_norm:
                _tree = _load_faculties_tree()
                _canonical_nganh: set[str] = set()
                for f in (_tree.get("faculties") or []):
                    for ng in (f.get("nganh") or []):
                        ng_norm = _normalize_topic_token(ng.get("nganh", ""))
                        if ng_norm and ng_norm == major_norm:
                            _canonical_nganh.add(ng_norm)
                # Nếu user major không khớp ngành nào trong cây khoa-ngành → BỎ filter
                # (trước đây tự add vào canonical làm false-positive filter ra 0 docs)
                if not _canonical_nganh:
                    logger.info("Major '%s' không khớp ngành nào trong cây khoa-ngành → bỏ qua filter", req.major)
                else:
                    _ACCEPT_PREFIXES = (
                        "thac si ", "thac si ngành ", "ngành ",
                        "bo mon ", "cong nghe ", "cong nghe ky thuat ",
                    )

                    def _doc_matches_major(doc_major_norm: str) -> bool:
                        if not doc_major_norm:
                            return False
                        if doc_major_norm in _canonical_nganh:
                            return True
                        for cn in _canonical_nganh:
                            if cn in doc_major_norm:
                                return True
                            if doc_major_norm.startswith(cn + " "):
                                return True
                            if doc_major_norm.startswith(cn + ","):
                                return True
                        for cn in _canonical_nganh:
                            for pfx in _ACCEPT_PREFIXES:
                                if doc_major_norm == (pfx.rstrip() + " " + cn).strip():
                                    return True
                        return False

                    _kept = []
                    for r in search_results:
                        doc = r.get("doc", r)
                        doc_major = normalize_text(str(doc.get("major", "")))
                        if _doc_matches_major(doc_major):
                            _kept.append(r)
                    search_results = _kept
                    logger.info("Major filter '%s': kept %d docs (canonical=%s)", req.major, len(search_results), sorted(_canonical_nganh))

        # Nếu không đính kèm file PDF cụ thể, vẫn tìm kiếm trên tất cả dữ liệu (CSV + PDF)
        # để trả về thông tin liên quan từ mọi nguồn
        total_found = len(search_results)  # Luu con so THUC truoc khi cap
        if not req.document_id:
            if catalog_mode:
                # KHONG cap 500 nua - de UI hien thi du ket qua
                # Chi giu toan bo ket qua cho sources (frontend tu phan trang)
                pass
            else:
                search_results = search_results[:12]  # Giữ nguyên cho non-catalog (Q&A)
        
        # Nếu user chỉ định document_id, filter để lấy tài liệu đó
        if req.document_id:
            # Try exact match first
            search_results = [r for r in search_results if r['doc'].get('title', '').lower() == req.document_id.lower()]
            if not search_results:
                # Try partial match with broader search
                search_results = [
                    r for r in retriever.search_by_query(normalized_query, top_k=20)
                    if req.document_id.lower() in r['doc'].get('title', '').lower()
                ][:6]
            if not search_results:
                raise HTTPException(404, f"Không tìm thấy tài liệu: {req.document_id}")
        
        # Format với metadata chi tiết
        context, doc_metadata_list = format_docs_with_metadata(search_results)
        related_pdf_infos = []
        seen_related_pdf_ids = set()
        for meta in doc_metadata_list:
            pdf_info = find_pdf_info_for_source(meta['metadata'].get('source', ''), meta['metadata'].get('title', ''))
            if not pdf_info:
                continue
            pdf_id = pdf_info.get("id") or pdf_info.get("file_name")
            if pdf_id in seen_related_pdf_ids:
                continue
            seen_related_pdf_ids.add(pdf_id)
            related_pdf_infos.append(pdf_info)

        # ── Tự động inject PDF theo tên file ──
        # CHỈ áp dụng cho non-catalog mode (content Q&A).
        # Catalog mode tìm sách theo ngành/chủ đề → KHÔNG nên ép PDF vào.
        try:
            _injected = False
            if not catalog_mode:
                q_lower = normalize_text(req.query)
                all_pdfs = pdf_manager.list_all_pdfs()
                q_words = [w for w in q_lower.split() if len(w) > 3]
                for pdf in all_pdfs:
                    pdf_id = pdf.get("id") or pdf.get("file_name")
                    if pdf_id in seen_related_pdf_ids:
                        continue
                    pdf_name = normalize_text(pdf.get("title", "") + " " + pdf.get("file_stem", "") + " " + pdf.get("file_name", ""))
                    if not (q_lower and pdf_name and q_words):
                        continue
                    # Yêu cầu ít nhất 2 từ匹配 (>3 ký tự) để tránh false-positive
                    matched_words = [w for w in q_words if w in pdf_name]
                    if len(matched_words) < 2:
                        continue
                    seen_related_pdf_ids.add(pdf_id)
                    related_pdf_infos.append(pdf)
                    pdf_text = pdf_manager.get_chapter_text(pdf['file_path'], 0, 999999)
                    inject_doc = {
                        'title': pdf.get('title', pdf.get('display_name', pdf.get('file_name', 'PDF'))),
                        'author': pdf.get('author', ''),
                        'year': pdf.get('year', 'N/A'),
                        'subject': pdf.get('subject', 'PDF'),
                        'link': '',
                        'source': pdf.get('file_name', ''),
                        'page': None,
                        'text': pdf_text if has_extractable_pdf_text(pdf_text) else '',
                        'csv_file': 'pdf:' + pdf.get('file_name', ''),
                        'hash': '',
                        'doc_type': 'PDF',
                        'format': 'Số',
                    }
                    search_results.insert(0, {'doc': inject_doc, 'score': 99999.0})
                    _injected = True
                    if len(related_pdf_infos) >= 3:
                        break
            # Nếu có inject PDF, rebuild context để LLM thấy nội dung
            if _injected:
                context, doc_metadata_list = format_docs_with_metadata(search_results)
                total_found = len(search_results)
        except Exception as e:
            logger.debug("PDF name search error: %s", e)
        
        # DEBUG: log metadata at debug level (chỉ hiện khi LOG_LEVEL=DEBUG)
        logger.debug("Query: %s", req.query)
        logger.debug("Normalized: %s", normalized_query)
        if req.document_id:
            logger.debug("Document-specific mode: %s", req.document_id)
        logger.debug("Found %d documents:", len(search_results))
        for i, meta in enumerate(doc_metadata_list):
            logger.debug("  [%d] %s | Score: %.2f", i, meta['metadata']['title'], meta['score'])

        logger.debug("CONTEXT LENGTH: %d chars", len(context))
        logger.debug("LLM Backend: OpenRouter | Model: %s", OPENROUTER_MODEL)
        
        # Check if user asks for summary
        request_summary = detect_summary_request(req.query)
        
        # Generate main answer with context
        _relevance_gate = False  # Default: không chặn
        if catalog_mode:
            answer = build_catalog_answer(search_results, req.query, total_found=total_found)
        else:
            # Relevance gate: nếu top score quá thấp, skip LLM và trả "không tìm thấy"
            _top_score = search_results[0].get('score', 0) if search_results else 0
            if not search_results or _top_score < 1.0:
                logger.info("Low relevance gate: top_score=%.2f, query='%s' → skip LLM", _top_score, req.query)
                answer = (
                    f"Hiện tại mình chưa tìm thấy tài liệu nào liên quan đến **\"{req.query}\"** "
                    f"trong thư viện.\n\n"
                    f"Bạn có thể:\n"
                    f"• Thử từ khóa khác hoặc diễn đạt lại\n"
                    f"• Hỏi theo **tên sách**, **tác giả**, hoặc **chủ đề** cụ thể hơn\n"
                    f"• Duyệt danh mục sách theo ngành học ở thanh bên trái"
                )
                _relevance_gate = True  # Ngăn PDF fallback override
                search_results = []    # Xóa kết quả rác, không hiển thị sources
            else:
                _relevance_gate = False
                chain_func, _ = build_general_chain()
                answer = chain_func(context=context, question=req.query)

        pdf_fallback_used = False
        if not _relevance_gate and related_pdf_infos and answer_says_no_information(answer):
            # N?u search_results có sách catalog ?i?m cao → dùng catalog answer thay vì ch? hi?n PDF
            _top_score_cat = search_results[0].get('score', 0) if search_results else 0
            if search_results and _top_score_cat >= 1.0:
                answer = build_catalog_answer(search_results, req.query, total_found=total_found)
                # V?n ?ính kèm PDF info ?? user bi?t có th? m? file
                if related_pdf_infos:
                    pdf_names = [p.get("title") or p.get("file_name", "PDF") for p in related_pdf_infos[:3]]
                    answer += "\n\n" + build_related_pdf_answer(related_pdf_infos, req.query)
            elif not catalog_mode and not search_results and len(related_pdf_infos) > 0:
                try:
                    pdf_text = pdf_manager.get_chapter_text(related_pdf_infos[0]['file_path'], 0, 999999)
                    if has_extractable_pdf_text(pdf_text):
                        pdf_title = related_pdf_infos[0].get('title', 'PDF')
                        new_context = f"Tài li?u ???c ch?n: {pdf_title}\nN?i dung trích xu?t:\n{pdf_text[:30000]}"
                        chain_func, _ = build_general_chain()
                        new_answer = chain_func(context=new_context, question=req.query)
                        if not answer_says_no_information(new_answer):
                            answer = new_answer
                            context = new_context
                            pdf_fallback_used = True
                        else:
                            answer = build_related_pdf_answer(related_pdf_infos, req.query)
                    else:
                        answer = build_related_pdf_answer(related_pdf_infos, req.query)
                except Exception as e:
                    logger.debug("PDF content fallback error: %s", e)
                    answer = build_related_pdf_answer(related_pdf_infos, req.query)
            else:
                answer = build_related_pdf_answer(related_pdf_infos, req.query)

        # Chi suppress sources neu LLM that su khong co thong tin VA search_results rong
        # (truoc day: chi can LLM noi "khong co" la ẩn hết → user không thấy 306 docs đã tìm được)
        # Thêm: nếu top score quá thấp → cũng suppress (kết quả rác)
        _top_score = search_results[0].get('score', 0) if search_results else 0
        _results_are_noisy = _top_score < 1.0 and not catalog_mode
        suppress_sources = (
            not pdf_fallback_used
            and answer_says_no_information(answer)
            and (not search_results or _results_are_noisy)
        )
        if suppress_sources and search_results:
            logger.info("Suppress noisy sources: top_score=%.2f, query='%s'", _top_score, req.query)
            search_results = []
        
        # Generate summary if requested
        summary = ""
        if request_summary and search_results and not suppress_sources:
            try:
                doc_text = format_docs(search_results[:4])  # Tóm tắt top 4 docs
                summary_chain = build_summary_prompt()
                summary = sanitize_answer_text(summary_chain(content=doc_text))
            except Exception as e:
                logger.warning("Lỗi tóm tắt: %s", e)
                summary = ""

        # Extract sources với metadata
        seen, sources = set(), []
        current_doc = None
        for meta in ([] if suppress_sources else doc_metadata_list):
            title = meta['metadata']['title']
            author = meta['metadata'].get('author', '')
            year = meta['metadata'].get('year', '')
            # Dedupe by (title, author, year) — same key as build_catalog_answer
            source_id = (str(title).lower().strip(), str(author).lower().strip(), str(year).lower().strip())
            pdf_info = find_pdf_info_for_source(meta['metadata'].get('source', ''), title)
            
            if source_id not in seen:
                seen.add(source_id)
                
                source_item = {
                    "title":       clean_display_value(meta['metadata']['title'], 300) or meta['metadata']['title'],
                    "author":      clean_display_value(meta['metadata']['author'], 180),
                    "year":        clean_display_value(meta['metadata']['year'], 40),
                    "subject":     clean_display_value(meta['metadata']['subject'], 180),
                    "link":        clean_display_value(meta['metadata']['link'], 500),
                    "source":      clean_display_value(meta['metadata']['source'], 260),
                    "doc_type":    clean_display_value(meta['metadata'].get('doc_type', 'N/A'), 80),
                    "format":      clean_display_value(meta['metadata'].get('format', 'Số'), 80),
                    "major":       clean_display_value(meta['metadata'].get('major', ''), 140),
                    "publisher":   clean_display_value(meta['metadata'].get('publisher', ''), 160),
                    "place":       clean_display_value(meta['metadata'].get('place', ''), 160),
                    "edition":     clean_display_value(meta['metadata'].get('edition', ''), 60),
                    "bilingual_title": clean_display_value(meta['metadata'].get('bilingual_title', ''), 300),
                    "course":      clean_display_value(meta['metadata'].get('course', ''), 140),
                    "keywords":    meta['metadata'].get('keywords', []) or [],
                    "location":    clean_display_value(meta['metadata'].get('location', ''), 800),
                    "abstract":    clean_display_value(meta['metadata'].get('abstract', ''), 360),
                    "ddc":         clean_display_value(meta['metadata'].get('ddc', ''), 80),
                    "notes":       clean_display_value(meta['metadata'].get('notes', ''), 180),
                    "description": meta['text'][:200] if meta.get('text') else "",
                    "is_pdf":      bool(pdf_info),
                    "pdf_id":      pdf_info.get('id') if pdf_info else "",
                    "file_name":   pdf_info.get('file_name') if pdf_info else "",
                    "display_name": pdf_info.get('display_name') if pdf_info else "",
                }
                sources.append(source_item)
                
                # Đánh dấu tài liệu hiện tại
                if req.document_id and title.lower() == req.document_id.lower():
                    current_doc = source_item

        # Nếu dùng PDF fallback, thêm PDF vào sources
        if pdf_fallback_used and related_pdf_infos:
            for pdf in related_pdf_infos:
                fb_id = (pdf.get("title", "").lower(), pdf.get("author", "").lower(), str(pdf.get("year", "")))
                if fb_id not in seen:
                    seen.add(fb_id)
                    sources.append({
                        "title": clean_display_value(pdf.get("title", "PDF"), 300),
                        "author": clean_display_value(pdf.get("author", ""), 180),
                        "year": clean_display_value(str(pdf.get("year", "")), 40),
                        "subject": clean_display_value(pdf.get("subject", "PDF"), 180),
                        "link": "",
                        "source": clean_display_value(pdf.get("file_name", ""), 260),
                        "doc_type": "PDF",
                        "format": "",
                        "major": "",
                        "publisher": "",
                        "place": "",
                        "edition": "",
                        "bilingual_title": "",
                        "course": "",
                        "keywords": [],
                        "location": "",
                        "abstract": "",
                        "ddc": "",
                        "notes": "",
                        "description": "",
                        "is_pdf": True,
                        "pdf_id": pdf.get("id", ""),
                        "file_name": pdf.get("file_name", ""),
                        "display_name": pdf.get("display_name", ""),
                    })
        
        # Lưu lịch sử
        topic = ""
        if sources:
            topic = sources[0].get("subject", "") or sources[0].get("title", "")
        entry = {
            "query": req.query,
            "search_query": search_query,
            "topic": topic,
            "source_count": len(sources),
            "shown_titles": [s.get("title", "") for s in sources[:20]],
        }
        hist = session_history.get(req.session_id, [])
        hist.append(entry)
        session_history[req.session_id] = hist[-MAX_HISTORY:]

        total_found = len(sources)

        # Generate follow-up suggestions
        has_pdf_sources = any(s.get("is_pdf") for s in sources)
        follow_ups = build_follow_up_suggestions(
            query=req.query,
            sources=sources,
            answer=answer,
            catalog_mode=catalog_mode,
            has_pdf_sources=has_pdf_sources,
            session_id=req.session_id,
        )

        return ChatResponse(
            answer=answer, 
            sources=sources,
            total_sources=len(sources),
            total_found=total_found,
            summary=summary,
            current_document=current_doc,
            follow_up_suggestions=follow_ups,
        )

    except Exception as e:
        err = str(e)
        if "connect" in err.lower() or "refused" in err.lower():
            raise HTTPException(503, "OpenRouter API không khả dụng.")
        raise HTTPException(500, err)

# ── Health check ──────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok", 
        "message": "Thư viện QNU RAG API đang chạy",
        "llm_backend": "OpenRouter",
        "model": OPENROUTER_MODEL,
        "ui_url": "/chat",
        "read_only": is_vercel_runtime(),
    }

@app.get("/chat", response_class=FileResponse)
def serve_chat_ui():
    """Serve chat.html with khoa→ngành tree inlined (no client fetch needed)."""
    from fastapi.responses import Response
    html_file = "chat.html"
    if not os.path.exists(html_file):
        return {"error": "chat.html not found"}
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
    # Inline the faculties tree so the sidebar renders synchronously
    # (avoids the "Đang tải cây khoa..." loading state).
    try:
        tree = _load_faculties_tree()
        tree_json = _json.dumps(tree, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        tree_json = '{"faculties":[],"total_faculties":0}'
    inline = (
        '<script id="__FACULTIES_TREE_DATA" type="application/json">'
        + tree_json
        + '</script>'
    )
    marker = "<!-- __FACULTIES_TREE_INJECT__ -->"
    if marker in content:
        content = content.replace(marker, inline)
    else:
        # Fallback: inject right before the main <script> block
        content = content.replace(
            '<script>\nconst BACKEND',
            inline + '\n<script>\nconst BACKEND',
            1,
        )
    return Response(content=content, media_type="text/html", headers={
        "Cache-Control": "no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })

@app.get("/static/SHL-logo.png", response_class=FileResponse, include_in_schema=False)
def serve_logo():
    """Serve only the public logo without exposing project files."""
    return FileResponse("SHL-logo.png", media_type="image/png")

@app.get("/status")
def status():
    """Chi tiết status của API"""
    return {
        "status": "ok",
        "llm_backend": "OpenRouter",
        "model": OPENROUTER_MODEL,
        "openrouter_url": OPENROUTER_BASE_URL,
        "read_only": is_vercel_runtime(),
    }


@app.get("/health")
def health():
    """Lightweight liveness probe — chỉ kiểm tra process còn sống và tài nguyên tối thiểu."""
    try:
        doc_count = len(documents) if 'documents' in globals() else 0
        pdf_count = len(pdf_docs) if 'pdf_docs' in globals() else 0
        bm25_ready = bm25_engine is not None and bool(documents)
        return {
            "status": "ok",
            "documents": doc_count,
            "pdf_pages": pdf_count,
            "indexes_ready": bm25_ready,
            "model": OPENROUTER_MODEL,
            "uptime_seconds": round(time.time() - _APP_START_TIME, 1) if '_APP_START_TIME' in globals() else None,
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "error": str(e)})


@app.get("/stats")
def stats():
    """Thống kê chỉ số runtime: index, cache, majors, calls — không truy cập PDF gốc."""
    try:
        # Document stats
        total = len(documents) if 'documents' in globals() else 0
        pdf_pages = sum(1 for d in documents if str(d.get('csv_file', '')).startswith('pdf:')) if total else 0
        csv_rows = total - pdf_pages

        # Major distribution (top 20, normalized)
        major_counter: dict[str, int] = {}
        empty_major = 0
        for d in documents:
            m = (d.get('major') or '').strip()
            if not m:
                empty_major += 1
                continue
            key = normalize_text(m) if m else ''
            if key:
                major_counter[key] = major_counter.get(key, 0) + 1
        top_majors = sorted(major_counter.items(), key=lambda x: x[1], reverse=True)[:20]

        # Source / type distribution
        source_counter: dict[str, int] = {}
        for d in documents:
            s = (d.get('source') or 'unknown').strip() or 'unknown'
            source_counter[s] = source_counter.get(s, 0) + 1

        # Doc type distribution
        type_counter: dict[str, int] = {}
        for d in documents:
            t = (d.get('doc_type') or '').strip() or 'unknown'
            type_counter[t] = type_counter.get(t, 0) + 1

        # Cache stats
        cache_info = _search_cache.stats() if '_search_cache' in globals() else "n/a"

        # LLM call stats
        llm_info = {
            "calls": _llm_stats["calls"],
            "failures": _llm_stats["failures"],
            "retries": _llm_stats["retries"],
            "last_error": _llm_stats["last_error"],
            "circuit_state": _llm_circuit["state"],
            "consecutive_failures": _llm_circuit["failures"],
        }

        # Topic cache
        topic_cache_size = len(TOPIC_CACHE) if 'TOPIC_CACHE' in globals() else 0

        # Sessions
        chat_sessions = len(session_history) if 'session_history' in globals() else 0
        pdf_sessions = len(pdf_session_history) if 'pdf_session_history' in globals() else 0

        # Semantic engine availability
        sem_engine = get_semantic_engine()
        sem_available = sem_engine is not None and getattr(sem_engine, 'embeddings', None) is not None

        return {
            "status": "ok",
            "documents": {
                "total": total,
                "csv_rows": csv_rows,
                "pdf_pages": pdf_pages,
                "empty_major": empty_major,
                "empty_major_pct": round(empty_major / total * 100, 1) if total else 0,
            },
            "top_majors": [{"name": n, "count": c} for n, c in top_majors],
            "by_source": source_counter,
            "by_type": type_counter,
            "indexes": {
                "bm25_ready": bm25_engine is not None and bool(documents),
                "semantic_available": sem_available,
                "topic_cache_size": topic_cache_size,
            },
            "cache": cache_info,
            "llm": llm_info,
            "sessions": {
                "chat": chat_sessions,
                "pdf": pdf_sessions,
            },
            "model": OPENROUTER_MODEL,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

class VoiceTranscribeRequest(BaseModel):
    language: Optional[str] = "vi"


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    alpha: Optional[float] = 0.5  # 0=BM25, 1=semantic, 0.5=hybrid


class SemanticSearchResponse(BaseModel):
    query: str
    mode: str  # "semantic" | "hybrid" | "bm25_only"
    results: list
    total: int
    semantic_available: bool


@app.post("/api/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(req: SemanticSearchRequest):
    """
    Giai đoạn 3: Tìm kiếm theo ngữ nghĩa (multilingual).
    Kết hợp BM25 + sentence-transformers để hiểu nghĩa câu hỏi.
    """
    engine = get_semantic_engine()
    if not engine or engine.embeddings is None:
        # Fallback về BM25
        bm25_results = BM25SearchEngine(documents).search(req.query, top_k=req.top_k)
        return SemanticSearchResponse(
            query=req.query,
            mode="bm25_only",
            results=bm25_results,
            total=len(bm25_results),
            semantic_available=False,
        )

    # Hybrid: BM25 + semantic
    bm25_engine = BM25SearchEngine(documents)
    bm25_raw = bm25_engine.search(req.query, top_k=req.top_k * 2)
    bm25_results = [{"doc": r, "score": r.get("score", 0)} for r in bm25_raw]

    hybrid = engine.hybrid_search(req.query, bm25_results, alpha=req.alpha, top_k=req.top_k)
    return SemanticSearchResponse(
        query=req.query,
        mode="hybrid",
        results=[
            {
                "doc": r["doc"],
                "score": r["score"],
                "bm25_score": r.get("bm25_score", 0),
                "semantic_score": r.get("semantic_score", 0),
            }
            for r in hybrid
        ],
        total=len(hybrid),
        semantic_available=True,
    )

import json as _json
_FACULTIES_CACHE: dict | None = None
_FACULTIES_CACHE_TS: float = 0.0
_FACULTIES_CACHE_TTL = 60.0  # seconds

def _load_faculties_tree() -> dict:
    """Load khoa→ngành tree from Data/_khoa_nganh_tree.json (cached in memory)."""
    global _FACULTIES_CACHE, _FACULTIES_CACHE_TS
    now = time.time()
    if _FACULTIES_CACHE is not None and (now - _FACULTIES_CACHE_TS) < _FACULTIES_CACHE_TTL:
        return _FACULTIES_CACHE
    tree_path = Path("Data") / "_khoa_nganh_tree.json"
    try:
        with open(tree_path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        if not isinstance(data, list):
            data = []
    except FileNotFoundError:
        data = []
    except Exception as e:
        logger.warning("Failed to load faculties tree: %s", e)
        data = []
    _FACULTIES_CACHE = {"faculties": data, "total_faculties": len(data)}
    _FACULTIES_CACHE_TS = now
    return _FACULTIES_CACHE


@app.get("/api/faculties")
async def get_faculties():
    """Return the 12-khoa → ngành tree for the sidebar quick-search UI."""
    return _load_faculties_tree()


class FacultySearchRequest(BaseModel):
    nganh: str
    khoa: str = None
    top_k: Optional[int] = 20


@app.post("/api/faculties/search", response_model=SemanticSearchResponse)
async def search_by_faculty(req: FacultySearchRequest):
    """
    Search the catalog filtered by a specific ngành (and optionally khoa).
    Hits the topic cache (already indexes the major field), then falls back
    to BM25 + major filter, guaranteeing only docs in the chosen ngành
    are returned.
    """
    nganh = (req.nganh or "").strip()
    khoa = (req.khoa or "").strip() or None
    if not nganh:
        raise HTTPException(status_code=400, detail="Thiếu 'nganh'.")

    nganh_norm = normalize_text(nganh)

    # Build canonical ngành set from the tree so we strictly scope the
    # filter to the chosen ngành + its data-side variations
    # (Thạc sĩ X, Ngành X, Bộ môn X, Công nghệ X ...).
    _tree = _load_faculties_tree()
    _canonical_nganh: set[str] = set()
    for f in (_tree.get("faculties") or []):
        for ng in (f.get("nganh") or []):
            ng_norm = _normalize_topic_token(ng.get("nganh", ""))
            if ng_norm and ng_norm == nganh_norm:
                _canonical_nganh.add(ng_norm)
    # Nếu user nganh không khớp ngành nào trong cây khoa-ngành → BỎ filter
    # (trước đây tự add vào canonical làm false-positive filter ra 0 docs)
    skip_major_filter = (not _canonical_nganh)
    if skip_major_filter:
        logger.info("search_by_faculty: nganh '%s' không khớp ngành nào trong cây khoa-ngành → bỏ qua filter", nganh)

    _ACCEPT_PREFIXES = (
        "thac si ", "thac si ngành ", "ngành ",
        "bo mon ", "cong nghe ", "cong nghe ky thuat ",
    )

    def _doc_matches_faculty(doc_major_norm: str) -> bool:
        if not doc_major_norm:
            return False
        if doc_major_norm in _canonical_nganh:
            return True
        for cn in _canonical_nganh:
            if cn in doc_major_norm:
                return True
            if doc_major_norm.startswith(cn + " "):
                return True
            if doc_major_norm.startswith(cn + ","):
                return True
        for cn in _canonical_nganh:
            for pfx in _ACCEPT_PREFIXES:
                if doc_major_norm == (pfx.rstrip() + " " + cn).strip():
                    return True
        return False

    # 1) Topic cache first (already indexes major field, super fast)
    topic_hits = _search_topic_cache(nganh) or []

    # 2) If topic cache missed, fall back to BM25
    if not topic_hits:
        bm25_raw = bm25_engine.search(nganh, top_k=req.top_k * 3)
        topic_hits = [{"doc": r.get("doc", r), "score": r.get("score", 0.0)} for r in bm25_raw]

    # 3) Filter strictly by major using the canonical ngành set
    if skip_major_filter:
        # Không filter — giữ nguyên topic_hits để user thấy kết quả thay vì 0
        filtered = topic_hits
        logger.info("search_by_faculty: skip filter, giữ nguyên %d docs", len(filtered))
    else:
        filtered = []
        for hit in topic_hits:
            doc = hit.get("doc", hit)
            doc_major = normalize_text(str(doc.get("major", "")))
            if _doc_matches_faculty(doc_major):
                filtered.append(hit)

    filtered = filtered[: req.top_k]

    return SemanticSearchResponse(
        query=nganh,
        mode="faculty",
        results=filtered,
        total=len(filtered),
        semantic_available=False,
    )


@app.post("/api/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...), language: str = Form("vi")):
    """
    Giai đoạn 3: Voice input — chuyển giọng nói thành text.
    Hỗ trợ: webm, wav, mp3, m4a, ogg (từ MediaRecorder API của browser).
    """
    try:
        from voice_input import save_uploaded_audio, transcribe_audio, cleanup_audio, is_available
    except ImportError:
        raise HTTPException(503, "Voice input chưa được cài. Chạy: pip install faster-whisper")

    if not is_available():
        raise HTTPException(503, "Whisper model chưa load xong hoặc chưa cài")

    suffix = Path(audio.filename or "voice.webm").suffix or ".webm"
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "File audio rỗng")

    tmp_path = save_uploaded_audio(audio_bytes, suffix=suffix)
    try:
        result = transcribe_audio(tmp_path, language=language)
        return result
    finally:
        cleanup_audio(tmp_path)


@app.get("/api/voice/status")
async def voice_status():
    """Check voice input có sẵn sàng không."""
    try:
        from voice_input import is_available
        return {"available": is_available(), "lang": os.getenv("WHISPER_LANG", "vi")}
    except ImportError:
        return {"available": False, "error": "voice_input module not found"}


@app.get("/api/pdf/{pdf_id}/content-with-ocr")
async def get_pdf_content_with_ocr(
    pdf_id: str,
    start: int = 0,
    end: int = 1,
    lang: str = "vie+eng",
):
    """
    Giai đoạn 3: Đọc PDF với OCR fallback tự động.
    Nếu text extraction rỗng → dùng Tesseract OCR.
    """
    pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p["id"] == pdf_id), None)
    if not pdf_info:
        raise HTTPException(404, f"PDF không tồn tại: {pdf_id}")

    result = pdf_manager.get_chapter_text_with_ocr_fallback(
        pdf_info["file_path"], start, end, lang=lang
    )
    return {
        "pdf_id": pdf_id,
        "title": pdf_info.get("title", ""),
        "start_page": start,
        "end_page": end,
        **result,
    }


@app.get("/api/pdf/{pdf_id}/is-scanned")
async def check_pdf_scanned(pdf_id: str):
    """Check PDF có phải scan không (text rỗng)."""
    pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p["id"] == pdf_id), None)
    if not pdf_info:
        raise HTTPException(404, f"PDF không tồn tại: {pdf_id}")
    is_scanned = pdf_manager.is_scanned_pdf(pdf_info["file_path"])
    return {"pdf_id": pdf_id, "is_scanned": is_scanned}


@app.get("/api/pdfs")
async def list_pdfs():
    """Liệt kê tất cả PDF documents"""
    try:
        pdfs = pdf_manager.list_all_pdfs()
        return {
            "pdfs": pdfs,
            "total_pdfs": len(pdfs),
            "status": "ok"
        }
    except Exception as e:
        logger.warning("Error listing PDFs: %s", e)
        return {
            "error": str(e),
            "pdfs": [],
            "total_pdfs": 0
        }

@app.post("/api/admin/login")
async def admin_login(data: dict):
    """Đăng nhập admin, trả về token"""
    password = data.get("password", "")
    if password != ADMIN_PASSWORD:
        raise HTTPException(401, "Mật khẩu không đúng")
    token = secrets.token_hex(32)
    ADMIN_TOKENS.add(token)
    return {"token": token, "status": "ok"}

@app.post("/api/admin/logout")
async def admin_logout(data: dict):
    """Đăng xuất admin"""
    token = data.get("token", "")
    ADMIN_TOKENS.discard(token)
    return {"status": "ok"}

@app.post("/api/admin/upload")
async def admin_upload(token: str = Form(...), file: UploadFile = File(...)):
    """Upload PDF (yêu cầu token admin)"""
    if is_vercel_runtime():
        raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    if token not in ADMIN_TOKENS:
        raise HTTPException(401, "Token không hợp lệ")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Chỉ chấp nhận file PDF")

    # Giới hạn dung lượng 50MB
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(400, "File vượt quá 50MB")

    # Kiểm tra trùng
    pdf_dir = Path("pdfs")
    pdf_dir.mkdir(exist_ok=True)
    dest = pdf_dir / file.filename
    if dest.exists():
        raise HTTPException(400, f"File '{file.filename}' đã tồn tại")

    with open(dest, "wb") as f:
        f.write(contents)

    # Xoá cache để lần sau load lại
    cache_file = Path("pdf_cache.json")
    if cache_file.exists():
        cache_file.unlink()
    # Xoá PDF extraction cache (load_pdf incremental cache)
    pdf_extract_cache = Path("Data/_pdf_extract_cache.pkl")
    if pdf_extract_cache.exists():
        pdf_extract_cache.unlink()
    _search_cache.invalidate()  # Xóa kết quả search cũ
    pdf_manager.invalidate_cache()  # Xóa metadata cache

    # Tái index BM25 để có thể search ngay
    try:
        reindex_pdfs(retriever)
    except Exception as e:
        logger.warning("Reindex error (non-fatal): %s", e)

    return {
        "status": "ok",
        "file_name": file.filename,
        "file_size_mb": round(len(contents) / (1024 * 1024), 2),
        "message": f"Đã upload {file.filename} thành công!"
    }

@app.delete("/api/admin/pdfs/{pdf_id}")
async def admin_delete_pdf(pdf_id: str, data: dict = Body(...)):
    """Xóa PDF khỏi thư mục pdfs/ (yêu cầu token admin)."""
    if is_vercel_runtime():
        raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    token = data.get("token", "")
    if token not in ADMIN_TOKENS:
        raise HTTPException(401, "Token không hợp lệ")

    pdfs = pdf_manager.list_all_pdfs()
    pdf_info = next((p for p in pdfs if p["id"] == pdf_id), None)
    if not pdf_info:
        raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")

    target = Path(pdf_info["file_path"]).resolve()
    pdf_dir = Path("pdfs").resolve()
    if target.suffix.lower() != ".pdf" or target.parent != pdf_dir:
        raise HTTPException(400, "Đường dẫn PDF không hợp lệ")

    file_name = pdf_info["file_name"]
    try:
        target.unlink()
    except FileNotFoundError:
        raise HTTPException(404, f"File không tồn tại: {file_name}")
    except Exception as e:
        raise HTTPException(500, f"Không thể xóa file: {e}")

    cache_file = Path("pdf_cache.json")
    if cache_file.exists():
        cache_file.unlink()
    # Xóa PDF extraction cache entry cho file bị xóa
    pdf_extract_cache = Path("Data/_pdf_extract_cache.pkl")
    if pdf_extract_cache.exists():
        try:
            import pickle as _pickle
            with open(pdf_extract_cache, "rb") as f:
                _pcache = _pickle.load(f)
            if file_name in _pcache:
                del _pcache[file_name]
                with open(pdf_extract_cache, "wb") as f:
                    _pickle.dump(_pcache, f, protocol=_pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    pdf_manager.invalidate_cache()
    _search_cache.invalidate()  # Xóa kết quả search cũ

    removed_index_docs = remove_pdf_from_index(file_name)

    return {
        "status": "ok",
        "pdf_id": pdf_id,
        "file_name": file_name,
        "removed_index_docs": removed_index_docs,
        "message": f"Đã xóa {file_name}"
    }

@app.get("/api/pdfs/{pdf_id}/structure")
async def get_pdf_structure(pdf_id: str):
    """Lấy cấu trúc chương/phần của PDF"""
    try:
        pdfs = pdf_manager.list_all_pdfs()
        
        # Tìm PDF theo ID
        pdf_info = None
        for pdf in pdfs:
            if pdf['id'] == pdf_id:
                pdf_info = pdf
                break
        
        if not pdf_info:
            raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")
        
        # Trích xuất chapters
        chapters = pdf_manager.extract_chapters(pdf_info['file_path'])
        
        return {
            "pdf_id": pdf_id,
            "title": pdf_info['title'],
            "author": pdf_info['author'],
            "chapters": chapters,
            "total_chapters": len(chapters)
        }
    
    except Exception as e:
        logger.warning("Error getting PDF structure: %s", e)
        raise HTTPException(500, str(e))

@app.get("/api/pdfs/{pdf_id}/content")
async def get_pdf_content(pdf_id: str):
    """Lấy toàn bộ nội dung text của PDF"""
    try:
        pdfs = pdf_manager.list_all_pdfs()
        pdf_info = next((p for p in pdfs if p['id'] == pdf_id), None)

        if not pdf_info:
            raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")

        full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)

        if not full_text or len(full_text.strip()) < 50:
            full_text = "PDF này không chứa nội dung text có thể trích xuất"

        return {
            "pdf_id": pdf_id,
            "title": pdf_info['title'],
            "total_pages": pdf_info.get('pages', 0),
            "content": full_text,
            "content_length": len(full_text)
        }

    except Exception as e:
        logger.warning("Error getting PDF content: %s", e)
        raise HTTPException(500, str(e))

@app.get("/api/pdfs/{pdf_id}/summary")
async def get_pdf_summary(pdf_id: str):
    """Lấy tóm tắt PDF"""
    try:
        pdfs = pdf_manager.list_all_pdfs()
        pdf_info = next((p for p in pdfs if p['id'] == pdf_id), None)
        
        if not pdf_info:
            raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")
        
        # Trích xuất trang đầu tiên
        text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 5)
        
        # Tạo tóm tắt nhanh (không xử lý toàn bộ)
        if not text or len(text.strip()) < 50:
            summary_text = "PDF này không chứa nội dung text có thể trích xuất"
        else:
            # Tạo summary đơn giản
            summary_text = text[:1000]
        
        return {
            "pdf_id": pdf_id,
            "title": pdf_info['title'],
            "author": pdf_info['author'],
            "year": pdf_info['year'],
            "summary": {
                "objectives": "Xem nội dung PDF để biết mục tiêu",
                "methods": "Xem nội dung PDF để biết phương pháp",
                "results": "Xem nội dung PDF để biết kết quả",
                "keywords": ["pdf", "tài liệu"],
                "preview": summary_text[:500]
            }
        }
    
    except Exception as e:
        logger.warning("Error getting PDF summary: %s", e)
        raise HTTPException(500, str(e))

class DocumentChatRequest(BaseModel):
    query: str
    pdf_id: str
    chapter: Optional[int] = None
    search_type: str = "full_pdf"  # "chapter", "full_pdf", "question"
    session_id: str = "default"

pdf_session_history: dict[str, list[dict]] = {}

@app.post("/chat/with-document")
async def chat_with_document(req: DocumentChatRequest):
    """Hỏi đáp dựa trên nội dung của một PDF cụ thể"""
    try:
        pdfs = pdf_manager.list_all_pdfs()
        pdf_info = next((p for p in pdfs if p['id'] == req.pdf_id), None)
        
        if not pdf_info:
            raise HTTPException(404, f"PDF không tìm thấy: {req.pdf_id}")

        history_key = f"{req.session_id}:{req.pdf_id}"
        pdf_hist = pdf_session_history.get(history_key, [])
        effective_query = req.query
        if is_follow_up(req.query) and pdf_hist:
            previous_query = pdf_hist[-1].get("effective_query") or pdf_hist[-1].get("query") or ""
            effective_query = f"{previous_query} {req.query}".strip()

        if detect_reference_request(effective_query):
            reference_text, start_page = extract_reference_section_from_pdf(pdf_info)
            if reference_text:
                answer = build_reference_answer(pdf_info, reference_text, start_page)
                pdf_hist.append({"query": req.query, "effective_query": effective_query, "topic": "reference"})
                pdf_session_history[history_key] = pdf_hist[-MAX_HISTORY:]
                return {
                    "answer": answer,
                    "pdf_id": req.pdf_id,
                    "pdf_title": pdf_info['title'],
                    "chapter": None,
                    "relevant_pages": [start_page] if start_page else [],
                    "source": {
                        "type": "pdf",
                        "pdf_id": req.pdf_id,
                        "pdf_title": pdf_info['title'],
                        "search_type": "references",
                        "text_extractable": True
                    },
                    "follow_up_suggestions": [
                        {"label": "Tóm tắt nội dung", "query": "Tóm tắt nội dung chính của tài liệu này"},
                        {"label": "Hỏi chương khác", "query": "Hỏi nội dung về một chương cụ thể"},
                        {"label": f"Kiến thức liên quan đến {pdf_info.get('subject') or 'lĩnh vực này'}", "query": f"Kiến thức liên quan đến {pdf_info.get('subject') or 'lĩnh vực này'}"},
                    ],
                }
        
        # Trích xuất context tốt nhất theo đúng PDF được chọn
        context, _, _ = build_pdf_context_for_query(
            pdf_info,
            effective_query,
            None
        )

        if not context.strip():
            pdf_title = pdf_info.get('title') or pdf_info.get('display_name') or pdf_info.get('file_name') or req.pdf_id
            return {
                "answer": build_empty_pdf_answer(pdf_title),
                "pdf_id": req.pdf_id,
                "pdf_title": pdf_title,
                "chapter": None,
                "relevant_pages": [],
                "source": {
                    "type": "pdf",
                    "pdf_id": req.pdf_id,
                    "pdf_title": pdf_title,
                    "search_type": "full_pdf",
                    "text_extractable": False
                }
            }
        
        # Dùng RAG chain
        chain_func, _ = build_general_chain()
        answer = chain_func(context=context, question=effective_query)

        # Không ghi đè câu trả lời "không có thông tin" — đó là câu trả lời đúng
        # khi tài liệu thực sự không chứa thông tin về câu hỏi.
        # Chỉ hiển thị gợi ý nếu LLM trả lời hợp lệ.
        
        # Trích xuất các trang liên quan
        relevant_pages = []
        source_text = context[:4000]
        if source_text:
            relevant_pages = [1]
        
        pdf_hist.append({"query": req.query, "effective_query": effective_query, "topic": "general"})
        pdf_session_history[history_key] = pdf_hist[-MAX_HISTORY:]

        # Generate context-aware follow-up suggestions from actual answer content
        final_sugs = build_contextual_suggestions(
            answer=answer,
            query=req.query,
            doc_title=pdf_info.get('title', ''),
        )

        return {
            "answer": answer,
            "pdf_id": req.pdf_id,
            "pdf_title": pdf_info['title'],
            "chapter": None,
            "relevant_pages": relevant_pages[:5],
            "source": {
                "type": "pdf",
                "pdf_id": req.pdf_id,
                "pdf_title": pdf_info['title'],
                "search_type": "full_pdf"
            },
            "follow_up_suggestions": final_sugs,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Error in PDF chat: %s", e)
        raise HTTPException(500, str(e))
