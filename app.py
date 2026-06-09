"""
app.py — FastAPI backend kết nối RAG (FAISS + OpenRouter) với chat.html
Chạy: uvicorn app:app --reload --port 8000
"""

import os
import re
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

# Import BM25 search engine (lightweight, no embeddings needed)
from search_engine import DocumentIndexer, BM25SearchEngine, HybridSearchEngine
from text_utils import normalize_text, extract_keywords, suggest_synonyms, spell_correct, suggest_terms_for_query
from load_pdf import load_all_pdfs
from pdf_manager import PDFManager

load_dotenv()

# Import OpenRouter client
import requests

# ── Config ───────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "poolside/laguna-xs.2:free")
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
            print("⚠️  OPENROUTER_API_KEY not set at build time — will check at runtime")
            return
        raise RuntimeError("OPENROUTER_API_KEY is required")
    print(f"✓ OpenRouter ready | Model: {OPENROUTER_MODEL}")


# Không chạy check_backend() ngay khi build (Vercel build không có env),
# nhưng Vercel runtime sẽ chạy lại module khi function được gọi, lúc đó env có sẵn.
if not is_vercel_runtime():
    check_backend()

# ── Khởi tạo app ─────────────────────────────────────────
app = FastAPI(title="Thư viện QNU RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Load BM25 Full-Text Search Index...")
documents = DocumentIndexer.load_from_data("Data")

pdf_docs = []
try:
    loaded = load_all_pdfs()
    if loaded:
        print(f"Adding {len(loaded)} PDF pages to BM25 documents...")
        seen_hashes = {d.get('hash') for d in documents if d.get('hash')}
        for doc in loaded:
            meta = doc.metadata or {}
            h = meta.get('hash')
            if h and h in seen_hashes:
                continue
            seen_hashes.add(h)

            pdf_item = {
                'title': meta.get('title', meta.get('source', 'PDF')).strip(),
                'author': meta.get('author', 'Unknown'),
                'year': meta.get('year', 'N/A'),
                'subject': meta.get('section') or meta.get('keywords') or 'PDF',
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
        print("No PDF pages found to add to BM25 index.")
except Exception as e:
    print(f"⚠️  Error loading PDFs for BM25: {e}")

bm25_engine = BM25SearchEngine(documents)
retriever = HybridSearchEngine(bm25_engine)

print(f"✅ Search index ready: {len(documents)} documents")
print(f"✅ Memory: ~50MB (no ML model loaded)")
use_pdf = len(pdf_docs) > 0

# Initialize PDF Manager
pdf_manager = PDFManager(pdf_dir="pdfs")
print(f"📚 PDF Manager initialized")

# ── Hàm tái index PDF vào BM25 ──────────────────────────
def reindex_pdfs(retriever):
    """Load lại tất cả PDF và cập nhật BM25 index (không ảnh hưởng dữ liệu CSV)"""
    from load_pdf import load_all_pdfs
    loaded = load_all_pdfs()
    if not loaded:
        print("No PDFs to reindex")
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

        pdf_item = {
            'title': meta.get('title', meta.get('source', 'PDF')).strip(),
            'author': meta.get('author', 'Unknown'),
            'year': meta.get('year', 'N/A'),
            'subject': meta.get('section') or meta.get('keywords') or 'PDF',
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
        print(f"✅ Reindexed {new_count} new PDF pages into BM25")
    else:
        print("No new PDF pages to reindex")
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
        print(f"⚠️ Error matching PDF source: {e}")

    return None

def answer_says_no_information(answer: str) -> bool:
    normalized = normalize_text(answer or "")
    no_info_phrases = [
        "tai lieu khong cung cap",
        "khong cung cap thong tin",
        "khong co thong tin",
        "khong du thong tin",
        "khong tim thay thong tin",
        "khong tim thay tai lieu",
        "khong co tai lieu",
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

# In-memory session tracking (lưu 3 query gần nhất)
session_history: dict[str, list[dict]] = {}
MAX_HISTORY = 3

class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    total_sources: int = 0
    summary: str = ""  # Tóm tắt nếu user yêu cầu
    current_document: Optional[dict] = None 

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
            content = f"[{title}{page_label}]\n{text[:3500]}"
        else:
            clean = title
            if valid(author): clean += f" - {author}"
            if valid(publisher): clean += f" ({publisher})"
            if valid(abstract): clean += f"\n\nTóm tắt: {abstract[:500]}"
            content = clean[:2000]
        
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
        print(f"📦 Loaded {len(CATALOG_TOPIC_INDEX)} catalog topic phrases from cache")
        return True
    except Exception as e:
        print(f"⚠️ Catalog topic cache load failed: {e}")
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
        print(f"⚠️ Catalog topic cache write failed: {e}")


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
    print(f"✅ Catalog topic index ready: {len(CATALOG_TOPIC_INDEX)} phrases")
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
    return sorted(collapsed, key=lambda item: item.get("score", 0.0), reverse=True)


# ── Author / Publisher Query Detection ──────────────────────

def detect_author_query(query: str) -> bool:
    """Detect nếu câu hỏi đang tìm tài liệu theo tác giả."""
    if not query:
        return False
    q = normalize_text(query)
    author_markers = [
        r"\btac gia\b",
        r"\btac gia\s+(?:la|co ten|ten)\b",
        r"\bcua\s+tac gia\b",
        r"\bsach\s+cua\s+tac gia\b",
        r"\btai lieu\s+cua\s+tac gia\b",
        r"\bgiao trinh\s+cua\s+tac gia\b",
    ]
    return any(re.search(pattern, q) for pattern in author_markers)


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
    patterns = [
        r"(?:sach|cua|tai lieu|giao trinh)\s+tac gia\s+(.+)",
        r"tac gia\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            name = match.group(1).strip().rstrip(",. ")
            # Remove trailing queries
            name = re.sub(r"\s+(?:khong|ko|lam on|hay|vui long|toi|minh|xin|cho|gui)\b.*$", "", name)
            name = name.strip().rstrip(",. ")
            # Must contain at least 2 word-like chunks
            if len(re.findall(r'\w+', name)) >= 2 and len(name) >= 5:
                return name
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
            # In strict mode (list request), chỉ trả về author-matched results
            if prefer_strict:
                return author_results
            # Otherwise, author results first, then rest
            return author_results + non_author_results[:3]
        # If no author match at all, fall through to normal search

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

    _, topic_tokens, topic_phrases = get_query_topic_terms(query)
    if not topic_tokens and not topic_phrases:
        return search_results

    unique_tokens = list(dict.fromkeys(topic_tokens))
    matched = []
    unmatched = []

    for result in search_results:
        doc = result.get("doc", {})
        normalized_doc = get_doc_topic_text(doc)
        exact_phrase_hit = any(contains_normalized_phrase(normalized_doc, phrase) for phrase in topic_phrases)
        match_count = sum(1 for token in unique_tokens if re.search(rf"\b{re.escape(token)}\b", normalized_doc))
        if not match_count and not exact_phrase_hit:
            unmatched.append(result)
            continue

        coverage = match_count / max(len(unique_tokens), 1)
        if prefer_strict and len(unique_tokens) >= 2 and coverage < 0.45 and not exact_phrase_hit:
            unmatched.append(result)
            continue

        phrase_boost = 10.0 if exact_phrase_hit else 0.0
        relevance = (match_count * 2.0) + (coverage * 4.0) + phrase_boost
        matched.append({**result, "score": result.get("score", 0.0) + relevance})

    if matched:
        matched.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        if prefer_strict:
            return matched
        unmatched.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return matched + unmatched

    if prefer_strict:
        return []

    return search_results


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

def detect_live_info_request(query: str) -> bool:
    """Detect out-of-scope current/live-info questions, not catalog lookups."""
    q = normalize_text(query or "")
    live_markers = [
        "hom nay", "ngay mai", "bay gio", "hien tai", "luc nay",
        "thoi tiet the nao", "thoi tiet hom nay", "du bao thoi tiet",
        "tin tuc", "moi nhat", "gia vang", "ty gia", "lich thi dau",
    ]
    return any(marker in q for marker in live_markers)

def detect_content_request(query: str) -> bool:
    """Only answer document content when the user explicitly asks for content."""
    if not query:
        return False

    q = normalize_text(query)
    if detect_summary_request(query) or detect_chapter_reference(query):
        return True

    content_phrases = [
        "noi dung", "chi tiet", "phan tich", "giai thich", "trinh bay",
        "cho biet", "hay cho biet", "neu ro", "neu cac", "neu nhung",
        "tom tat", "doc file", "doc tai lieu", "trich", "trich xuat",
        "muc luc", "chuong", "phan nao", "noi ve gi",
        "file nay", "pdf nay", "tai lieu nay", "tai lieu do",
        "tai lieu so", "cuon nay", "sach nay",
        "phuong phap nghien cuu", "ket qua nghien cuu", "ket luan",
        "giai phap trong", "dinh huong trong",
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


def build_catalog_answer(search_results, query: str) -> str:
    """Build câu trả lời ngắn, ổn định cho truy vấn liệt kê tài liệu."""
    if not search_results:
        return "Không tìm thấy tài liệu phù hợp trong dữ liệu hiện có."

    unique_items = []
    seen = set()

    for result in search_results:
        doc = result["doc"]
        title = clean_display_value(doc.get("title", "Không rõ tên"), 260) or "Không rõ tên"
        author = clean_display_value(doc.get("author", ""), 180)
        subject = clean_display_value(doc.get("subject", ""), 180)
        year = clean_display_value(doc.get("year", ""), 40)
        major = clean_display_value(doc.get("major", ""), 140)
        doc_type = clean_display_value(doc.get("doc_type", ""), 80)
        is_pdf = str(doc.get("csv_file", "")).startswith("pdf:")

        dedupe_key = (title.lower(), str(author).lower(), str(year).lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        unique_items.append({
            "title": title,
            "author": author,
            "subject": "" if is_pdf else subject,
            "year": year,
            "major": major,
            "doc_type": "PDF liên quan" if is_pdf else doc_type,
        })

    lines = [f"Tìm thấy {min(len(unique_items), 6)} tài liệu phù hợp nhất:"]
    for i, item in enumerate(unique_items[:6], 1):
        meta_lines = []
        if item["author"]:
            meta_lines.append(f"Tác giả: {item['author']}")
        if item["year"]:
            meta_lines.append(f"Năm: {item['year']}")
        if item["doc_type"]:
            meta_lines.append(f"Loại: {item['doc_type']}")
        if item["major"]:
            meta_lines.append(f"Ngành: {item['major']}")
        if item["subject"]:
            meta_lines.append(f"Chủ đề: {item['subject']}")

        lines.append("")
        lines.append(f"**{i}. {item['title']}**")
        if meta_lines:
            lines.append("\n".join(meta_lines))

    lines.append("")
    lines.append("Bạn có thể yêu cầu tóm tắt hoặc xem nội dung của một tài liệu cụ thể khi cần.")
    return "\n".join(lines)


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
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(503, "OPENROUTER_API_KEY is not configured")

    import time
    max_retries = 3
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
                    "model": OPENROUTER_MODEL,
                    "messages": messages,
                    "temperature": temperature,
                },
                timeout=300,
            )

            if response.status_code == 429 and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"⏳ Rate limited, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                raise Exception(f"OpenRouter error: {response.text}")

            payload = response.json()
            return payload["choices"][0]["message"].get("content", "")
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"⏳ LLM error, retrying in {wait}s: {e}")
                time.sleep(wait)
                continue
            print(f"LLM error: {e}")
            raise HTTPException(503, f"LLM API error: {e}")

def build_summary_prompt():
    """Build prompt cho tóm tắt nâng cao với cấu trúc"""
    
    prompt_text = """
Hãy tóm tắt tài liệu theo cấu trúc sau:

**MỤC TIÊU:**
Nêu rõ mục tiêu chính của tài liệu là gì?

**PHƯƠNG PHÁP:**
Tài liệu sử dụng phương pháp, công cụ, cách tiếp cận nào?

**KẾT QUẢ CHÍNH:**
Những phát hiện, kết luận, kết quả chính là gì? (3-5 dòng)

**TỪ KHÓA:**
Liệt kê 5-7 từ khóa chính

**TÓM TẮT NGẮN:**
Tóm tắt lại toàn bộ nội dung thành 2-3 câu

NỘI DUNG TÀI LIỆU:
{content}

TRẢ LỜI (Tiếng Việt, tuân thủ cấu trúc trên):
"""
    
    def summary_chain(content: str) -> str:
        messages = [
            {"role": "system", "content": "Bạn chỉ được trả về phần trả lời cuối cùng bằng tiếng Việt. Không hiển thị suy luận, không dùng thẻ <think>, không thêm chú thích meta."},
            {"role": "user", "content": prompt_text.format(content=content)}
        ]
        return call_llm(messages, temperature=0.1)
    
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

    # Với câu hỏi chung như "nội dung file này", lấy text trực tiếp từ PDF đã chọn.
    # BM25 theo từ khóa quá chung thường không trả đúng trang.
    if is_general_pdf_content_request(query):
        full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)
        if has_extractable_pdf_text(full_text):
            header = (
                f"Tài liệu được chọn: {pdf_title}\n"
                f"Tên file: {pdf_source}\n"
                f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
            )
            return f"{header}\nNội dung trích xuất:\n{full_text[:30000]}", [], None

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

TÀI LIỆU:
{context}

CÂU HỎI: {question}

TRẢ LỜI (Tiếng Việt, chi tiết, tập trung vào phương pháp):
"""


def build_general_chain():
    """Build general RAG chain with prompt selection based on question type"""
    
    prompt_text = """Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn (QNU Library Assistant).

HƯỚNG DẪN TRẢ LỜI:
1. Trả lời dựa CHÍNH XÁC trên tài liệu được cung cấp.
2. Kèm theo: Nhan đề sách, tác giả, năm xuất bản, chủ đề, vị trí trong tài liệu (nếu có).
3. Nếu user hỏi về CHƯƠNG/PHẦN CỤ THỂ, tìm nội dung đó và trả lời chi tiết.
4. Nếu user hỏi về PHƯƠNG PHÁP NGHIÊN CỨU, hãy nêu rõ: công cụ, quy trình, dữ liệu.
5. Nếu user hỏi về KẾT QUẢ/KẾT LUẬN, liệt kê rõ ràng các phát hiện quan trọng.
6. Kèm link tài liệu bản số nếu có sẵn.
7. Không bịa dữ liệu nếu không có trong tài liệu - hãy nói "Tài liệu không cung cấp thông tin này".
8. Không viết thành một đoạn văn dài. Luôn trình bày thành từng ý rõ ràng bằng Markdown.
9. Nếu user hỏi "nội dung là gì", "tóm tắt", hoặc hỏi tổng quan tài liệu, dùng đúng bố cục:
   **Tài liệu**
   - Nhan đề, tác giả/năm nếu có.

   **Nội dung chính**
   - 3-6 ý chính, mỗi ý một dòng.

   **Bố cục/Phạm vi**
   - Các chương/phần hoặc phạm vi nghiên cứu nếu tài liệu có nêu.

   **Kết luận**
   - 1-2 ý kết luận ngắn.

TÀI LIỆU:
{context}

CÂU HỎI: {question}

TRẢ LỜI (Tiếng Việt, ngắn gọn, chỉ nội dung trả lời chính; không chèn mục nguồn tài liệu):"""
    
    def rag_chain(context: str, question: str) -> str:
        # Detect question type
        chapter_num = detect_chapter_reference(question)
        is_method = detect_method_question(question)
        is_result = detect_result_question(question)

        # Choose appropriate prompt
        if chapter_num:
            prompt = build_chapter_prompt(chapter_num)
        elif is_method:
            prompt = build_method_prompt()
        else:
            prompt = prompt_text

        messages = [
            {"role": "system", "content": "Bạn chỉ được trả về phần trả lời cuối cùng bằng tiếng Việt. Không hiển thị suy luận, không dùng thẻ <think>, không viết lời dẫn như 'Okay, the user...'."},
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

        results = retriever.search_by_query(query, top_k=req.top_k * 3)

        # Filter out the original document
        filtered = []
        seen_titles = set()
        for r in results:
            doc = r['doc']
            title_lower = doc['title'].lower().strip()

            # Skip same title
            if req.title and title_lower == req.title.lower().strip():
                continue
            if title_lower in seen_titles:
                continue
            seen_titles.add(title_lower)

            # Determine reason
            reasons = []
            if req.author and req.author.lower() in doc.get('author', '').lower():
                reasons.append("Cùng tác giả")
            if req.subject and doc.get('subject', '') and req.subject.lower() in doc.get('subject', '').lower():
                reasons.append("Cùng chủ đề")
            if doc.get('major', '') and req.subject and req.subject.lower() in doc.get('major', '').lower():
                reasons.append("Cùng ngành học")
            if not reasons:
                reasons.append("Liên quan đến truy vấn")

            filtered.append({
                "title": doc['title'],
                "author": doc.get('author', 'Unknown'),
                "year": doc.get('year', 'N/A'),
                "subject": doc.get('subject', 'N/A'),
                "major": doc.get('major', ''),
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
                "doc_type": doc.get('doc_type', 'N/A'),
                "link": doc.get('link', ''),
                "location": doc.get('location', ''),
                "abstract": doc.get('abstract', ''),
                "ddc": doc.get('ddc', ''),
                "notes": doc.get('notes', ''),
                "score": round(r['score'], 2),
            })

        return {"results": docs, "total": len(docs)}

    except Exception as e:
        return {"error": str(e), "results": [], "total": 0}


# ── Endpoint /chat ────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        # 🔥 OPTIMIZED: Normalize query for better matching
        normalized_query = normalize_text(req.query)
        
        # Lấy lịch sử session
        hist = session_history.get(req.session_id, [])
        request_content = detect_content_request(req.query)
        is_list_request = detect_list_request(req.query)
        catalog_mode = is_list_request or not request_content

        if detect_live_info_request(req.query) and catalog_mode and not is_list_request:
            return ChatResponse(
                answer="Tài liệu không cung cấp thông tin này.",
                sources=[],
                total_sources=0,
                summary="",
                current_document=None
            )
        
        # Câu hỏi liệt kê theo lĩnh vực dùng bộ rút gọn chủ đề. Câu hỏi mặc định
        # vẫn trả danh mục, nhưng giữ truy vấn đầy đủ để khớp đúng tên đề tài dài.
        search_query = resolve_search_query(
            req.query,
            hist,
            catalog_mode=is_list_request
        )
        if search_query != normalized_query:
            print(f"🔁 Context-aware search: '{normalized_query}' → '{search_query}'")

        # Retrieve documents using BM25 (super fast, no embedding model)
        retrieval_k = 120 if catalog_mode else 30
        search_results = retriever.search_by_query(search_query, top_k=retrieval_k)
        catalog_query_token_count = len(topic_tokens(strip_search_intent_phrases(req.query)))
        use_metadata_search = catalog_mode and (
            is_list_request or catalog_query_token_count <= 7
        )
        if use_metadata_search:
            metadata_results = metadata_search_by_query(req.query, top_k=retrieval_k)
            search_results = merge_search_results(metadata_results, search_results)
        search_results = rerank_results_for_query(
            req.query,
            search_results,
            prefer_strict=is_list_request or detect_author_query(req.query) or detect_publisher_query(req.query)
        )
        if catalog_mode:
            search_results = collapse_catalog_pdf_pages(search_results)

        # Nếu không đính kèm file PDF cụ thể, vẫn tìm kiếm trên tất cả dữ liệu (CSV + PDF)
        # để trả về thông tin liên quan từ mọi nguồn
        if not req.document_id:
            search_results = search_results[:12]
        
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
        
        # DEBUG: In ra metadata
        print(f"\n🔍 DEBUG - Query: {req.query}")
        print(f"🔍 DEBUG - Normalized: {normalized_query}")
        if req.document_id:
            print(f"Document-specific mode: {req.document_id}")
        print(f"📄 Found {len(search_results)} documents:")
        for i, meta in enumerate(doc_metadata_list):
            print(f"  [{i}] {meta['metadata']['title']} | Score: {meta['score']:.2f}")
        
        print(f"\n📝 CONTEXT LENGTH: {len(context)} chars")
        print(f"🤖 LLM Backend: OpenRouter | Model: {OPENROUTER_MODEL}")
        
        # Check if user asks for summary
        request_summary = detect_summary_request(req.query)
        
        # Generate main answer with context
        if catalog_mode:
            answer = build_catalog_answer(search_results, req.query)
        else:
            chain_func, _ = build_general_chain()
            answer = chain_func(context=context, question=req.query)

        if related_pdf_infos and answer_says_no_information(answer):
            answer = build_related_pdf_answer(related_pdf_infos, req.query)

        suppress_sources = answer_says_no_information(answer)
        
        # Generate summary if requested
        summary = ""
        if request_summary and search_results and not suppress_sources:
            try:
                doc_text = format_docs(search_results[:4])  # Tóm tắt top 4 docs
                summary_chain = build_summary_prompt()
                summary = sanitize_answer_text(summary_chain(content=doc_text))
            except Exception as e:
                print(f"⚠️  Lỗi tóm tắt: {e}")
                summary = ""

        # Extract sources với metadata
        seen, sources = set(), []
        current_doc = None
        for meta in ([] if suppress_sources else doc_metadata_list):
            title = meta['metadata']['title']
            source_id = title
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
                    "location":    clean_display_value(meta['metadata'].get('location', ''), 220),
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
        
        # Lưu lịch sử
        topic = ""
        if sources:
            topic = sources[0].get("subject", "") or sources[0].get("title", "")
        entry = {"query": req.query, "search_query": search_query, "topic": topic}
        hist = session_history.get(req.session_id, [])
        hist.append(entry)
        session_history[req.session_id] = hist[-MAX_HISTORY:]
        
        return ChatResponse(
            answer=answer, 
            sources=sources,
            total_sources=len(sources),
            summary=summary,
            current_document=current_doc
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
    """Serve chat.html"""
    html_file = "chat(1).html"
    if os.path.exists(html_file):
        from fastapi.responses import Response
        with open(html_file, "rb") as f:
            content = f.read()
        return Response(content=content, media_type="text/html", headers={
            "Cache-Control": "no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        })
    else:
        return {"error": "chat.html not found"}

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

# ── PDF Management Endpoints ──────────────────────────────

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
        print(f"⚠️ Error listing PDFs: {e}")
        return {
            "error": str(e),
            "pdfs": [],
            "total_pdfs": 0
        }

# ── Admin Login / Upload ─────────────────────────────────

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

    # Tái index BM25 để có thể search ngay
    try:
        reindex_pdfs(retriever)
    except Exception as e:
        print(f"⚠️ Reindex error (non-fatal): {e}")

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
    pdf_manager.pdf_cache = {}

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
        print(f"⚠️ Error getting PDF structure: {e}")
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
        print(f"⚠️ Error getting PDF content: {e}")
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
        print(f"⚠️ Error getting PDF summary: {e}")
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
                    }
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
        
        # Trích xuất các trang liên quan
        relevant_pages = []
        source_text = context[:4000]
        if source_text:
            relevant_pages = [1]
        
        pdf_hist.append({"query": req.query, "effective_query": effective_query, "topic": "general"})
        pdf_session_history[history_key] = pdf_hist[-MAX_HISTORY:]

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
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️ Error in PDF chat: {e}")
        raise HTTPException(500, str(e))
