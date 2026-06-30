"""
app.py — FastAPI backend for QNU Library RAG (refactored).
All utility functions moved to lib/ modules; this file keeps FastAPI app,
middleware, SearchCache, inline helpers, and all route handlers.
Chay: uvicorn app:app --reload --port 8000
"""

import os
import re
import sys
import time
import json as _json
import pickle
import secrets
from pathlib import Path
from typing import Optional
from collections import OrderedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

# ── Lib imports ────────────────────────────────────────────
from lib.config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_BASE_URL, OPENROUTER_APP_TITLE, OPENROUTER_HTTP_REFERER,
    ADMIN_PASSWORD, VERCEL_READ_ONLY_MESSAGE, is_vercel_runtime, MAX_HISTORY,
)
from lib.state import (
    APP_START_TIME, ADMIN_TOKENS, llm_stats, llm_circuit,
    session_history, pdf_session_history,
)
from lib.logging_setup import logger
from lib.models import (
    ChatRequest, ChatResponse, SearchRequest, SuggestRequest,
    SpellCheckRequest, TermRequest, DocumentChatRequest,
    SemanticSearchRequest, SemanticSearchResponse,
    FacultySearchRequest, VoiceTranscribeRequest,
)
from lib.llm_client import call_llm, check_backend, record_llm_failure
from lib.llm_prompts import (
    sanitize_answer_text, build_general_chain, build_summary_prompt,
    detect_chapter_reference, detect_method_question, detect_structure_query,
    detect_result_question, build_chapter_prompt, build_method_prompt,
    build_structure_prompt,
)
from lib.pdf_helpers import (
    has_extractable_pdf_text, is_general_pdf_content_request,
    build_empty_pdf_answer, detect_reference_request,
    is_reference_heading_line, answer_says_no_information,
    build_related_pdf_answer, build_reference_answer, parse_reference_items,
)
from lib.catalog import (
    normalize_topic_text, topic_tokens, build_catalog_topic_index,
    find_catalog_topic_matches, metadata_search_by_query,
    build_topic_cache, search_topic_cache,
    extract_primary_topic_text,
    extract_content_search_query,
    detect_ai_domain, expand_domain_terms, score_ai_document,
    get_doc_topic_fields, get_doc_topic_text, get_query_topic_terms,
    strip_search_intent_phrases, clean_topic_tail, contains_normalized_phrase,
    VIETNAMESE_STOP_WORDS, _normalize_topic_token,
)
from lib.reranking import (
    detect_author_query, detect_publisher_query, extract_author_name,
    extract_publisher_name, normalize_author_name, author_name_matches,
    merge_search_results as catalog_merge_results,
    collapse_catalog_pdf_pages as catalog_collapse_pdf_pages,
    rerank_results_for_query, clean_display_value,
)
from lib.followup import (
    extract_answer_headings, extract_answer_sections,
    build_contextual_suggestions, build_follow_up_suggestions,
)

# ── External imports ───────────────────────────────────────
from lib.search_engine import DocumentIndexer, BM25SearchEngine, HybridSearchEngine
from lib.text_utils import (
    normalize_text, suggest_synonyms, spell_correct, suggest_terms_for_query,
)
from lib.intent_classifier import classify_intent, detect_greeting, extract_entities
from lib.load_pdf import load_all_pdfs
from lib.pdf_manager import PDFManager

load_dotenv()

# ── Check backend ──────────────────────────────────────────
if not is_vercel_runtime():
    check_backend()

# ── Initialize FastAPI app ─────────────────────────────────
app = FastAPI(title="Th\u01b0 vi\u1ec7n QNU RAG API", version="2.0.0")


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

# ── Load search index ──────────────────────────────────────
logger.info("Load BM25 Full-Text Search Index...")
documents = DocumentIndexer.load_from_data("Data")

pdf_docs = []
use_pdf = False
logger.info("PDF index DISABLED — only CSV/XLSX documents loaded.")
bm25_engine = BM25SearchEngine(documents)
retriever = HybridSearchEngine(bm25_engine)

logger.info("Search index ready: %d documents", len(documents))
logger.info("Memory: ~50MB (no ML model loaded)")

# ── Semantic engine (lazy load) ────────────────────────────
_semantic_engine = None


def get_semantic_engine():
    global _semantic_engine
    if _semantic_engine is not None:
        return _semantic_engine
    try:
        from lib.semantic_search import SemanticSearchEngine
        _semantic_engine = SemanticSearchEngine(documents)
        if _semantic_engine.embeddings is not None:
            logger.info("Semantic engine READY (hybrid BM25 + embeddings)")
        return _semantic_engine
    except Exception as e:
        logger.warning("Semantic engine unavailable: %s", e)
        return None


# ── Topic Cache wrappers ───────────────────────────────────
def _build_topic_cache():
    build_topic_cache(retriever.bm25.documents)


def _search_topic_cache_fn(query):
    return search_topic_cache(query, retriever.bm25.documents)


_build_topic_cache()

# ── PDF Manager ────────────────────────────────────────────
pdf_manager = PDFManager(pdf_dir="pdfs")
logger.info("PDF Manager initialized")

# ── Search Result Cache (LRU, TTL-based) ─────────────────
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
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), results)

    def invalidate(self):
        n = len(self._cache)
        self._cache.clear()
        if n > 0:
            logger.info("Search cache invalidated (%d entries cleared)", n)

    def stats(self) -> str:
        total = self._hits + self._misses
        rate = (self._hits / total * 100) if total else 0
        return f"cache={len(self._cache)} hits={self._hits} misses={self._misses} rate={rate:.0f}%"


_search_cache = SearchCache(max_size=256, ttl_seconds=300)

# ── File Change Detection ──────────────────────────────────
_startup_pdf_count = sum(1 for _ in Path("pdfs").glob("*.pdf")) if Path("pdfs").exists() else 0
_startup_data_mtime = 0.0
for _f in Path("Data").glob("*"):
    if _f.is_file() and _f.name != "_documents_cache.pkl":
        _startup_data_mtime = max(_startup_data_mtime, _f.stat().st_mtime)

_last_check_time = 0.0
_check_interval = 10


def check_files_changed():
    global _last_check_time, _startup_pdf_count, _startup_data_mtime
    now = time.time()
    if now - _last_check_time < _check_interval:
        return {"pdf": False, "data": False}
    _last_check_time = now
    result = {"pdf": False, "data": False}
    current_pdf_count = sum(1 for _ in Path("pdfs").glob("*.pdf")) if Path("pdfs").exists() else 0
    if current_pdf_count != _startup_pdf_count:
        _startup_pdf_count = current_pdf_count
        result["pdf"] = True
    current_data_mtime = 0.0
    if Path("Data").exists():
        for _f in Path("Data").glob("*"):
            if _f.is_file() and _f.name != "_documents_cache.pkl":
                current_data_mtime = max(current_data_mtime, _f.stat().st_mtime)
    if current_data_mtime > _startup_data_mtime:
        _startup_data_mtime = current_data_mtime
        result["data"] = True
    return result


# ── Reindex PDFs ───────────────────────────────────────────
def reindex_pdfs(retriever):
    from lib.load_pdf import load_all_pdfs as _load_all_pdfs
    loaded = _load_all_pdfs()
    if not loaded:
        logger.info("No PDFs to reindex")
        return 0
    existing_sources = {d.get('source') for d in retriever.bm25.documents if d.get('csv_file', '').startswith('pdf:')}
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
        pdf_author = meta.get('author', 'Unknown')
        if not pdf_author or pdf_author == 'Unknown' or 'aspose' in pdf_author.lower() or 'epubtopdfconverter' in pdf_author.lower() or 'pdf converter' in pdf_author.lower():
            file_stem = Path(source_name).stem if source_name else ""
            m = re.match(r"^([A-Z][A-Z\s\.]+?)\s*[-–]\s*", file_stem)
            if m:
                pdf_author = m.group(1).strip()
        pdf_subject = meta.get('keywords', '')
        if isinstance(pdf_subject, list):
            pdf_subject = ' '.join(str(k) for k in pdf_subject if k)
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
        _search_cache.invalidate()
        logger.info("Reindexed %d new PDF pages into BM25 + topic cache rebuilt", new_count)
    else:
        logger.info("No new PDF pages to reindex")
    return new_count


def remove_pdf_from_index(file_name: str) -> int:
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
        _search_cache.invalidate()
    return removed


def find_pdf_info_for_source(source: str = "", title: str = "") -> Optional[dict]:
    source_norm = (source or "").strip().lower()
    title_norm = (title or "").strip().lower()
    if not source_norm and not title_norm:
        return None
    try:
        for pdf in pdf_manager.list_all_pdfs():
            candidates = [
                pdf.get("file_name", ""), pdf.get("file_stem", ""),
                pdf.get("title", ""), pdf.get("display_name", ""),
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


# ── Format helpers ─────────────────────────────────────────
def format_docs_with_metadata(search_results):
    """Format BM25 search results with metadata."""
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
                "title": title, "author": author, "year": year,
                "subject": subject, "link": link, "source": source,
                "doc_type": doc_type, "format": doc_format, "major": major,
                "publisher": publisher,
                "abstract": abstract[:300] if abstract else '',
                "location": location[:200] if location else '',
                "notes": notes, "ddc": ddc,
            },
            "score": score
        })

    combined_text = "\n\n---\n\n".join([f["text"] for f in formatted])
    return combined_text, formatted


def format_docs(search_results):
    """Simple format: combine text from search results."""
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


# ═══════════════════════════════════════════════════════════════
# PDF extraction functions (depend on pdf_manager)
# ═══════════════════════════════════════════════════════════════

def is_chapter_heading_line(line: str, chapter_num: int) -> bool:
    normalized = normalize_text(line or "")
    return bool(re.match(rf"^chuong\s+{chapter_num}\b", normalized))


def chapter_heading_has_body(pdf_info: dict, page_idx: int, lines_after_heading: list[str], chapter_num: int) -> bool:
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
    pdf_info: dict, chapter_num: int, max_chars: int = 12000
) -> tuple[str, Optional[int], Optional[int]]:
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


def extract_reference_section_from_pdf(pdf_info: dict, max_chars: int = 40000) -> tuple[str, Optional[int]]:
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


# ═══════════════════════════════════════════════════════════════
# build_pdf_context_for_query (needs pdf_manager + retriever)
# ═══════════════════════════════════════════════════════════════

def build_pdf_context_for_query(pdf_info: dict, query: str, chapter: Optional[int] = None) -> tuple[str, list[dict], Optional[int]]:
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
            toc_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 8)
            if has_extractable_pdf_text(toc_text):
                header = (
                    f"Tài liệu được chọn: {pdf_title}\n"
                    f"Tên file: {pdf_source}\n"
                    f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
                    f"Mục trích xuất: 8 trang đầu (mục lục / bố cục)\n"
                )
                return f"{header}\nNội dung trích xuất:\n{toc_text[:8000]}", [], None

        full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)
        if has_extractable_pdf_text(full_text):
            header = (
                f"Tài liệu được chọn: {pdf_title}\n"
                f"Tên file: {pdf_source}\n"
                f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
            )
            total_len = len(full_text)
            if total_len <= 12000:
                sampled = full_text
            else:
                chunk = 4000
                head = full_text[:chunk]
                mid_start = max(0, (total_len - chunk) // 2)
                mid = full_text[mid_start:mid_start + chunk]
                tail = full_text[-chunk:]
                sampled = f"{head}\n\n[...Phần giữa trang {total_len // 2000}...]\n\n{mid}\n\n[...Phần cuối trang {total_len // 2000}...]\n\n{tail}"
            return f"{header}\nNội dung trích xuất:\n{sampled}", [], None

    selected_results = []
    for result in retriever.search_by_query(query, top_k=20):
        doc = result.get('doc', {})
        if doc.get('source') == pdf_source and has_extractable_pdf_text(doc.get('text', '')):
            selected_results.append(result)

    if selected_results:
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

    full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)
    if not has_extractable_pdf_text(full_text):
        return "", [], None

    header = (
        f"Tài liệu được chọn: {pdf_title}\n"
        f"Tên file: {pdf_source}\n"
        f"Số trang: {pdf_info.get('pages', 'N/A')}\n"
    )
    return f"{header}\nNội dung trích xuất:\n{full_text[:20000]}", [], None


# ═══════════════════════════════════════════════════════════════
# Detection functions & query helpers
# ═══════════════════════════════════════════════════════════════

def detect_summary_request(query: str) -> bool:
    keywords = ["tóm tắt", "tom tat", "summary", "tóm", "tom", "résumé", "概括", "summarize"]
    q = query.lower()
    return any(kw in q for kw in keywords)


def detect_list_request(query: str) -> bool:
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
    if not query:
        return False
    if detect_list_request(query):
        return False
    q = normalize_text(query)
    if not q:
        return False
    if detect_author_query(query) or detect_publisher_query(query):
        return False
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
    mapping = {
        "ă": "a", "â": "a", "đ": "d", "ê": "e", "ô": "o", "ơ": "o", "ư": "u",
        "ắ": "a", "ấ": "a", "ạ": "a", "ả": "a", "ã": "a", "ằ": "a", "Ầ": "a",
        "ậ": "a", "ẳ": "a", "ẵ": "a",
        "ẹ": "e", "ẻ": "e", "ẽ": "e", "ế": "e", "ề": "e", "ệ": "e",
        "ọ": "o", "ỏ": "o", "õ": "o", "ố": "o", "ồ": "o", "ộ": "o", "ổ": "o", "ỗ": "o",
        "ớ": "o", "ờ": "o", "ợ": "o", "ở": "o", "ỡ": "o",
        "ụ": "u", "ủ": "u", "ũ": "u", "ứ": "u", "ừ": "u", "ự": "u",
        "ỳ": "y", "ỵ": "y", "ỷ": "y", "ỹ": "y",
        "í": "i", "ì": "i", "ị": "i", "ỉ": "i", "ĩ": "i",
        "ó": "o", "ò": "o", "é": "e", "è": "e", "á": "a", "à": "a",
        "ú": "u", "ù": "u", "ý": "y",
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
    "khong", "ko", "k", "kh", "hem", "nhi", "vay", "nhe", "a",
    "khong a", "khong vay", "khong nhi", "khong the", "khong va",
    "khong nhe", "khong phai", "khong co", "k vay", "k nhi",
    "gi", "gi a", "chu", "dau", "roi", "the", "ha",
}


def _is_yesno_token(token: str) -> bool:
    return _strip_vi_diacritics(token).strip().lower() in _YESNO_WORD_SET


def extract_existence_topic(query: str) -> str:
    if not query:
        return ""
    cleaned = normalize_text(query)
    preamble = [
        r"\b(?:vay\s+)?ban\s+(?:co\s+)?the\s*",
        r"\b(?:vay\s+)?ban\s+",
        r"\bcho\s+(?:minh|toi|em|anh|chi)\s+(?:hoi|xem|biet)\s*",
        r"\bminh\s+(?:muon|can)\s+",
        r"\btoi\s+(?:muon|can)\s+",
        r"\bem\s+(?:muon|can)\s+",
        r"\banh\s+(?:muon|can)\s+",
        r"\bchi\s+(?:muon|can)\s+",
        r"\bcac\s+ban\s+cho\s+minh\s+hoi\s*",
        r"\b(?:vay|the|a)\s+",
    ]
    for p in preamble:
        cleaned = re.sub(r"^" + p, " ", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"^\s*co\s+(?:tai\s+lieu|sach|cuon|quyen|an\s+pham|giao\s+trinh)\s+"
        r"(?:cua\s+tac\s+gia\s+)?"
        r"(?:nao\s+|nào\s+)?(?:ve|về)\s+",
        " ", cleaned, flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^\s*co\s+cuon\s+nao\s+ve\s+", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^\s*co\s+"
        r"(?:tai\s+lieu|sach|cuon|quyen|an\s+pham|giao\s+trinh)\s+"
        r"(?:cua\s+tac\s+gia\s+)?",
        " ", cleaned, flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\?+\s*$", " ", cleaned)
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
            if _is_yesno_token(cleaned):
                cleaned = ""
            break
    cleaned = re.sub(
        r"^\s*(?:v[eề]|lĩnh\s*vực|chủ\s*đề|ngành)\s+",
        " ", cleaned, flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    match = re.search(
        r"\b(?:v[eề]|lĩnh\s*vực|chủ\s*đề|ngành)\s+(.+)$",
        cleaned, flags=re.IGNORECASE,
    )
    if match:
        cleaned = match.group(1).strip()
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


def _last_existence_topic(history) -> str:
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


def detect_content_request(query: str) -> bool:
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


def detect_exhaustion_query(query: str) -> bool:
    q = normalize_text(query or "")
    phrases = [
        "het chua", "con khong", "con tai lieu nao", "con them khong",
        "con nhieu khong", "het tai lieu", "het sach", "het ket qua",
        "con bao nhieu", "con mot nua", "con gi khong",
        "con nua khong", "con nua ko", "con nua k",
    ]
    return any(phrase in q for phrase in phrases)


def is_follow_up(query: str) -> bool:
    if detect_ai_domain(query):
        return False
    keywords = extract_primary_topic_text(query)
    return len(keywords.split()) == 0


def resolve_search_query(query: str, history: list, catalog_mode: bool = False) -> str:
    if catalog_mode:
        keywords = extract_keywords(query) if hasattr(extract_keywords, '__call__') else extract_content_search_query(query)
    else:
        keywords = extract_content_search_query(query)
    norm = normalize_text(query)
    if is_follow_up(query) and history:
        last = history[-1]
        topic = normalize_text(last.get("topic", ""))
        last_q = normalize_text(last.get("search_query", ""))
        if topic and len(topic) >= 3:
            return f"{topic} {norm}".strip()
        if last_q:
            return f"{last_q} {norm}".strip()
    if keywords:
        return keywords
    return norm


def build_catalog_answer(search_results, query: str, total_found: int = 0) -> str:
    if not search_results:
        return "Không tìm thấy tài liệu phù hợp trong dữ liệu hiện có."
    total = total_found if total_found > 0 else len(search_results)
    return f"Mình tìm thấy **{total} tài liệu** liên quan đến câu hỏi của bạn. Xem chi tiết bên dưới."


def extract_keywords(text: str) -> str:
    """Wrapper for catalog extract_keywords."""
    from lib.catalog import extract_keywords as _cat_extract_keywords
    return _cat_extract_keywords(text)


# ═══════════════════════════════════════════════════════════════
# Hybrid search (BM25 + semantic)
# ═══════════════════════════════════════════════════════════════

def hybrid_search_with_semantic_fallback(
    query: str,
    bm25_top_k: int = 20,
    final_top_k: int = 10,
    alpha: float = 0.5,
    min_bm25_score: float = 0.5,
) -> tuple[list[dict], str]:
    bm25_results = bm25_engine.search(query, top_k=bm25_top_k)
    if bm25_results and len(bm25_results) >= 5:
        top_score = bm25_results[0].get("score", 0) if bm25_results else 0
        if top_score >= min_bm25_score:
            return bm25_results[:final_top_k], "bm25"
    sem_engine = get_semantic_engine()
    if not sem_engine or sem_engine.embeddings is None:
        return bm25_results[:final_top_k] if bm25_results else [], "bm25"
    sem_results = sem_engine.search(query, top_k=bm25_top_k)
    if not sem_results:
        return bm25_results[:final_top_k] if bm25_results else [], "bm25"
    bm25_dict = {id(r["doc"]): r for r in bm25_results}
    combined = []
    for sr in sem_results:
        doc = sr["doc"]
        sem_score = sr["score"]
        bm25_score = 0
        if id(doc) in bm25_dict:
            bm25_score = bm25_dict[id(doc)].get("score", 0)
        max_bm25 = max((r.get("score", 0) for r in bm25_results), default=1) or 1
        bm25_norm = bm25_score / max_bm25
        final = alpha * sem_score + (1 - alpha) * bm25_norm
        combined.append({
            "doc": doc, "score": final,
            "bm25_score": bm25_score, "semantic_score": sem_score,
        })
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:final_top_k], "hybrid"


# ═══════════════════════════════════════════════════════════════
# Faculties tree cache
# ═══════════════════════════════════════════════════════════════
_FACULTIES_CACHE: dict | None = None
_FACULTIES_CACHE_TS: float = 0.0
_FACULTIES_CACHE_TTL = 60.0


def _load_faculties_tree() -> dict:
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


# ═══════════════════════════════════════════════════════════════
# ROUTE: /suggest-related
# ═══════════════════════════════════════════════════════════════

@app.post("/suggest-related")
async def suggest_related(req: SuggestRequest):
    try:
        if not any([req.title, req.author, req.subject]):
            return {"suggestions": [], "total": 0, "error": "Need at least one filter"}
        query_parts = []
        if req.title: query_parts.append(req.title)
        if req.author: query_parts.append(req.author)
        if req.subject: query_parts.append(req.subject)
        query = " ".join(query_parts)
        search_query = " ".join(query_parts)
        results = retriever.search_by_query(search_query, top_k=req.top_k * 5)
        exclude_set = {t.lower().strip() for t in req.exclude_titles if t}
        filtered = []
        seen_titles = set()
        has_author_filter = bool(req.author)
        has_subject_filter = bool(req.subject)
        for r in results:
            doc = r['doc']
            title_lower = doc['title'].lower().strip()
            if req.title and title_lower == req.title.lower().strip():
                continue
            if title_lower in exclude_set:
                continue
            if title_lower in seen_titles:
                continue
            seen_titles.add(title_lower)
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


# ═══════════════════════════════════════════════════════════════
# ROUTE: /spell-check
# ═══════════════════════════════════════════════════════════════

@app.post("/spell-check")
async def spell_check_endpoint(req: SpellCheckRequest):
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


# ═══════════════════════════════════════════════════════════════
# ROUTE: /suggest-terms
# ═══════════════════════════════════════════════════════════════

@app.post("/suggest-terms")
async def suggest_terms(req: TermRequest):
    try:
        term = req.term.lower()
        local_suggestions = suggest_synonyms(term)
        correction = spell_correct(term)
        results = retriever.search_by_query(term, top_k=5)
        context_docs = [{
            "title": r['doc'].get('title', ''),
            "subject": r['doc'].get('subject', ''),
            "major": r['doc'].get('major', ''),
        } for r in results]
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


# ═══════════════════════════════════════════════════════════════
# ROUTE: /documents
# ═══════════════════════════════════════════════════════════════

@app.get("/documents")
async def list_all_documents(page: int = 1, limit: int = 20):
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


# ═══════════════════════════════════════════════════════════════
# ROUTE: /search
# ═══════════════════════════════════════════════════════════════

@app.post("/search")
async def search_documents(req: SearchRequest):
    try:
        if not any([req.query, req.author, req.subject, req.major]):
            return {"results": [], "total": 0}
        query_parts = []
        if req.query: query_parts.append(req.query)
        if req.author: query_parts.append(req.author)
        if req.subject: query_parts.append(req.subject)
        if req.major: query_parts.append(req.major)
        query = " ".join(query_parts)
        filters = {}
        if req.author: filters['author'] = req.author
        if req.subject: filters['subject'] = req.subject
        if req.year: filters['year'] = req.year
        if req.major: filters['major'] = req.major
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


# ═══════════════════════════════════════════════════════════════
# ROUTE: /chat (main Q&A endpoint)
# ═══════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        normalized_query = normalize_text(req.query)
        intent = classify_intent(req.query)
        logger.info("Intent: %s for query: %s", intent, req.query)

        if intent == "chitchat":
            return ChatResponse(
                answer="Xin chào! Mình là trợ lý Thư viện Trường Đại học Quy Nhơn. Mình có thể giúp gì cho bạn về tra cứu sách, tài liệu số, hoặc nội dung các tài liệu trong thư viện ạ? 📚",
                sources=[], total_sources=0, total_found=0, summary="", current_document=None, follow_up_suggestions=[],
            )
        if intent == "live_info":
            return ChatResponse(
                answer="Mình là trợ lý Thư viện, mình chỉ hỗ trợ tra cứu sách và tài liệu thôi ạ! 😊\n\n"
                       "Bạn có thể thử:\n"
                       "• **Tìm sách** theo tên, tác giả hoặc chủ đề\n"
                       "• **Hỏi về nội dung** tài liệu trong thư viện\n\n"
                       "Ví dụ: *\"Tìm sách về trí tuệ nhân tạo\"*",
                sources=[], total_sources=0, total_found=0, summary="", current_document=None,
                follow_up_suggestions=[
                    {"label": "Tìm sách về công nghệ thông tin", "query": "tìm sách về công nghệ thông tin"},
                    {"label": "Tìm sách về văn học", "query": "tìm sách về văn học"},
                    {"label": "Xem giới thiệu thư viện", "query": "Giới thiệu thư viện QNU"},
                ],
            )

        # ── Xử lý câu trả lời xác nhận ──────────
        AFFIRMATION_WORDS = {
            "co", "ok", "oke", "okie", "duoc", "can", "muon",
            "u", "um", "ừ", "ừm", "được", "cần", "muốn",
            "liet ke", "liệt kê", "xem", "cho xem", "cho minh xem",
        }
        q_normalized_affirm = normalize_text(req.query or "").strip()
        hist = session_history.get(req.session_id, [])

        if q_normalized_affirm in AFFIRMATION_WORDS and hist:
            last_entry = hist[-1]
            prev_topic = last_entry.get("topic", "")
            prev_search = last_entry.get("search_query", "")
            reconstructed_query = prev_topic or prev_search or ""
            if reconstructed_query:
                req.query = reconstructed_query
                normalized_query = normalize_text(req.query)
                logger.info("Affirmation detected → auto-search: %r", req.query)

        if detect_exhaustion_query(req.query):
            entry_count = len(hist)
            if entry_count > 0:
                last_topic = hist[-1].get("topic", "")
                last_search = hist[-1].get("search_query", "")
                query_for_count = last_topic or last_search or normalized_query
            else:
                query_for_count = normalized_query
            if query_for_count == normalized_query:
                stripped = re.sub(r"\b(vay|con|khong|nua|thi|da|het|bao nhieu|them)\b", " ", normalized_query, flags=re.IGNORECASE)
                stripped = re.sub(r"\s+", " ", stripped).strip()
                if stripped:
                    query_for_count = stripped
            total_count = retriever.count_matching(query_for_count)
            if entry_count == 0:
                answer_text = ("Bạn chưa hỏi chủ đề nào trước đó. Hãy thử hỏi cụ thể hơn, ví dụ: **tìm sách về trí tuệ nhân tạo**, **có tài liệu về giáo dục không?**")
            elif total_count <= 0:
                answer_text = f"Không còn tài liệu nào về chủ đề **{last_topic or 'này'}** nữa. Đó là tất cả kết quả trong thư viện rồi! Bạn muốn tìm chủ đề khác không?"
            elif total_count <= 5:
                answer_text = f"Chỉ còn **{total_count} tài liệu** liên quan đến chủ đề này trong kho lưu trữ. Bạn có muốn xem tiếp không? Hãy nói **liệt kê thêm** để mình hiển thị tiếp."
            else:
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
            return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=total_count, summary="", current_document=None, follow_up_suggestions=[])

        if intent == "unknown":
            try:
                _has_pdfs = len(pdf_manager.list_all_pdfs()) > 0
            except Exception:
                _has_pdfs = False
            if not _has_pdfs:
                return ChatResponse(
                    answer="Mình chưa hiểu ý bạn lắm! 🤔\n\n"
                           "Mình là trợ lý Thư viện, mình có thể giúp bạn:\n"
                           "• **Tìm sách, tài liệu** trong thư viện\n"
                           "• **Gợi ý tài liệu** theo chủ đề\n"
                           "• **Tóm tắt nội dung** tài liệu\n\n"
                           "Bạn hãy thử hỏi cụ thể hơn nhé!",
                    sources=[], total_sources=0, total_found=0, summary="", current_document=None,
                    follow_up_suggestions=[
                        {"label": "Tìm sách về trí tuệ nhân tạo", "query": "tìm sách về trí tuệ nhân tạo"},
                        {"label": "Có tài liệu về văn học không?", "query": "có tài liệu về văn học không"},
                        {"label": "Xem giới thiệu thư viện", "query": "Giới thiệu thư viện QNU"},
                    ],
                )

        if intent == "clarify":
            query_normalized = normalize_text(req.query).strip()
            if "the loai" in query_normalized:
                answer = ("Mình muốn giúp bạn tốt hơn! 🤗\n\n"
                         "Bạn đang muốn: **TÌM SÁCH** về thể loại này, hay hỏi về **NỘI DUNG** thể loại văn học?\n\n"
                         "Vui lòng chọn:")
            elif "tac gia" in query_normalized:
                answer = ("Mình muốn giúp bạn tốt hơn! 🤗\n\n"
                         "Bạn đang muốn: **TÌM SÁCH** của tác giả này, hay hỏi về **THÔNG TIN** tác giả?\n\n"
                         "Vui lòng chọn:")
            elif "nha xuat ban" in query_normalized:
                answer = ("Mình muốn giúp bạn tốt hơn! 🤗\n\n"
                         "Bạn đang muốn: **TÌM SÁCH** của nhà xuất bản này, hay hỏi về **THÔNG TIN** nhà xuất bản?\n\n"
                         "Vui lòng chọn:")
            else:
                answer = ("Mình muốn giúp bạn tốt hơn! 🤗\n\n"
                         "Bạn đang muốn **TÌM SÁCH** hay **HỎI VỀ NỘI DUNG** liên quan?\n\n"
                         "Vui lòng chọn bên dưới nhé:")
            return ChatResponse(answer=answer, sources=[], total_sources=0, total_found=0, summary="", current_document=None,
                follow_up_suggestions=[{"label": "Tìm sách", "query": "tim sach ve " + query_normalized}, {"label": "Hỏi về nội dung", "query": "noi dung ve " + query_normalized}])

        request_content = detect_content_request(req.query)
        is_list_request = detect_list_request(req.query)
        catalog_mode = is_list_request or (not request_content and intent != "unknown")

        if detect_greeting(req.query):
            greeting_answer = (
                "Xin chào bạn!\n\nMình là trợ lý Thư viện Đại học Quy Nhơn. Mình có thể giúp bạn:\n\n"
                "- **Tìm sách, tài liệu** theo tên, tác giả hoặc chủ đề\n"
                "- **Gợi ý tài liệu** liên quan đến lĩnh vực bạn quan tâm\n"
                "- **Tóm tắt nội dung** tài liệu có sẵn trong kho\n\n"
                "Bạn đang muốn tìm tài liệu gì? Hãy thử:\n"
                "- *\"Tìm sách về trí tuệ nhân tạo\"*\n"
                "- *\"Có tài liệu về văn học Việt Nam không?\"*\n"
                "- *\"Sách của tác giả Nguyễn Văn A\"*"
            )
            return ChatResponse(answer=greeting_answer, sources=[], total_sources=0, summary="", current_document=None, follow_up_suggestions=[])

        list_topic_marker = "__LIST_TOPIC__:"
        list_topic_query = None
        if req.query.startswith(list_topic_marker):
            list_topic_query = req.query[len(list_topic_marker):].strip()

        _decline_pat = re.compile(
            r"^\s*(không|khong|thôi|thoi|thôi nhé|thoi nha|cảm ơn|cam on|cám ơn|cam_on|thank|thanks|no|nah|ok|okie|okela|đủ rồi|du roi|kệ|ke|thôi bỏ qua|thoi bo qua)\s*[.!]?\s*$",
            re.IGNORECASE,
        )
        if hist and _decline_pat.match(req.query.strip()):
            last_topic = _last_existence_topic(hist)
            if last_topic:
                answer_text = f"Được, mình sẽ không liệt kê **{last_topic}** nữa. Bạn muốn mình hỗ trợ gì khác không?"
                hist.append({"role": "assistant", "content": answer_text})
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=0, summary="", current_document=None,
                    follow_up_suggestions=[{"label": "Gợi ý chủ đề khác", "query": "Gợi ý tài liệu phổ biến"}, {"label": "Tìm sách theo tác giả", "query": "sách của Nguyễn Nhật Ánh"}, {"label": "Xem giới thiệu thư viện", "query": "Giới thiệu thư viện QNU"}])

        if detect_existence_request(req.query):
            topic = extract_existence_topic(req.query)
            topic_label = topic or "chủ đề này"
            if topic:
                count = retriever.count_matching(topic)
                if count <= 0:
                    answer_text = f"Hiện tại mình chưa tìm thấy tài liệu nào về **{topic_label}** trong thư viện. Bạn có thể thử hỏi chủ đề khác hoặc diễn đạt khác đi một chút nhé."
                    return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=0, summary="", current_document=None,
                        follow_up_suggestions=[{"label": "Gợi ý chủ đề khác", "query": "Gợi ý tài liệu phổ biến"}, {"label": "Tìm chủ đề tương tự", "query": f"tài liệu liên quan đến {topic_label}"}])
                hist.append({"query": req.query, "search_query": topic, "topic": topic_label, "source_count": count, "shown_titles": []})
                session_history[req.session_id] = hist[-MAX_HISTORY:]
                answer_text = f"Có, thư viện hiện có tài liệu về **{topic_label}**. Bạn có muốn mình liệt kê chi tiết ra không?"
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=count, summary="", current_document=None,
                    follow_up_suggestions=[{"label": "Có, liệt kê chi tiết ra giúp mình", "query": f"{list_topic_marker}{topic}"}, {"label": "Không, cảm ơn bạn", "query": "Cảm ơn bạn"}])

        if list_topic_query:
            req_query_for_search = f"tìm sách về {list_topic_query}"
            req.query = req_query_for_search
            normalized_query = normalize_text(req_query_for_search)
            is_list_request = True
            catalog_mode = True

        if not req.major and not req.document_id:
            try:
                from lib.catalog import _extract_major_from_query
                _detected_major = _extract_major_from_query(req.query)
                if _detected_major:
                    req.major = _detected_major
            except Exception:
                pass

        file_changes = check_files_changed()
        if file_changes["pdf"]:
            try:
                reindex_pdfs(retriever)
            except Exception:
                pass
        if file_changes["data"]:
            try:
                _cache_pkl = Path("Data/_documents_cache.pkl")
                if _cache_pkl.exists():
                    _cache_pkl.unlink()
                new_documents = DocumentIndexer.load_from_data("Data")
                retriever.bm25.documents = new_documents
                retriever.bm25._build_index()
                global _semantic_engine
                _semantic_engine = None
                reindex_pdfs(retriever)
                _build_topic_cache()
                _search_cache.invalidate()
            except Exception:
                pass

        search_query = resolve_search_query(req.query, hist, catalog_mode=is_list_request)

        if req.major:
            major_clean = (req.major or "").strip()
            if major_clean and major_clean.lower() not in search_query.lower():
                search_query = f"{search_query} {major_clean}".strip()

        _prefer_strict = detect_author_query(req.query) or detect_publisher_query(req.query)
        cache_key = f"catalog:{search_query}:{_prefer_strict}:{req.major or ''}" if catalog_mode else None
        use_cache = catalog_mode and not req.document_id and not req.query.startswith("__SUGGEST")
        search_results = _search_cache.get(cache_key) if use_cache else None

        if search_results is not None:
            logger.info("Search cache HIT (%d docs, key='%s')", len(search_results), search_query)
        else:
            retrieval_k = 500 if catalog_mode else 30
            cache_results = []
            if catalog_mode:
                cache_results = _search_topic_cache_fn(search_query)
            bm25_results = retriever.search_by_query(search_query, top_k=retrieval_k)
            if catalog_mode and bm25_results and not _prefer_strict:
                bm25_results = [r for r in bm25_results if not str(r.get("doc", {}).get("csv_file", "")).startswith("pdf:") and r.get("doc", {}).get("page") is None]
            catalog_query_token_count = len(topic_tokens(strip_search_intent_phrases(req.query)))
            if catalog_mode and (is_list_request or catalog_query_token_count <= 7):
                metadata_results = metadata_search_by_query(req.query, bm25_documents=retriever.bm25.documents, top_k=retrieval_k)
                bm25_results = catalog_merge_results(metadata_results, bm25_results)
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
            else:
                search_results = bm25_results

            is_author_pub_query = detect_author_query(req.query) or detect_publisher_query(req.query)
            bm25_top_score = bm25_results[0].get("score", 0) if bm25_results else 0
            if not is_author_pub_query and not catalog_mode and (len(bm25_results) < 3 or bm25_top_score < 1.0):
                try:
                    sem_results, sem_mode = hybrid_search_with_semantic_fallback(query=search_query, bm25_top_k=20, final_top_k=15, alpha=0.5)
                    if sem_results and sem_mode == "hybrid":
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
                except Exception:
                    pass

            search_results = rerank_results_for_query(req.query, search_results, prefer_strict=_prefer_strict)
            if catalog_mode:
                search_results = catalog_collapse_pdf_pages(search_results, pdf_lookup=find_pdf_info_for_source)
            if use_cache and search_results:
                _search_cache.set(cache_key, search_results)

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
                if _canonical_nganh:
                    def _doc_matches_major(doc_major_norm: str) -> bool:
                        if not doc_major_norm: return False
                        if doc_major_norm in _canonical_nganh: return True
                        for cn in _canonical_nganh:
                            if cn in doc_major_norm: return True
                            if doc_major_norm.startswith(cn + " ") or doc_major_norm.startswith(cn + ","): return True
                        return False
                    search_results = [r for r in search_results if _doc_matches_major(normalize_text(str(r.get("doc", r).get("major", ""))))]

        total_found = len(search_results)
        if not req.document_id and not catalog_mode:
            search_results = search_results[:12]

        if req.document_id:
            search_results = [r for r in search_results if r['doc'].get('title', '').lower() == req.document_id.lower()]
            if not search_results:
                search_results = [r for r in retriever.search_by_query(normalized_query, top_k=20) if req.document_id.lower() in r['doc'].get('title', '').lower()][:6]
            if not search_results:
                raise HTTPException(404, f"Không tìm thấy tài liệu: {req.document_id}")

        context, doc_metadata_list = format_docs_with_metadata(search_results)
        related_pdf_infos = []
        seen_related_pdf_ids = set()
        for meta in doc_metadata_list:
            pdf_info = find_pdf_info_for_source(meta['metadata'].get('source', ''), meta['metadata'].get('title', ''))
            if not pdf_info: continue
            pdf_id = pdf_info.get("id") or pdf_info.get("file_name")
            if pdf_id in seen_related_pdf_ids: continue
            seen_related_pdf_ids.add(pdf_id)
            related_pdf_infos.append(pdf_info)

        # Auto-inject PDF by name
        try:
            _injected = False
            if not catalog_mode:
                q_lower = normalize_text(req.query)
                all_pdfs = pdf_manager.list_all_pdfs()
                q_words = [w for w in q_lower.split() if len(w) > 3]
                for pdf in all_pdfs:
                    pdf_id = pdf.get("id") or pdf.get("file_name")
                    if pdf_id in seen_related_pdf_ids: continue
                    pdf_name = normalize_text(pdf.get("title", "") + " " + pdf.get("file_stem", "") + " " + pdf.get("file_name", ""))
                    if not (q_lower and pdf_name and q_words): continue
                    matched_words = [w for w in q_words if w in pdf_name]
                    if len(matched_words) < 2: continue
                    seen_related_pdf_ids.add(pdf_id)
                    related_pdf_infos.append(pdf)
                    pdf_text = pdf_manager.get_chapter_text(pdf['file_path'], 0, 999999)
                    inject_doc = {'title': pdf.get('title', pdf.get('display_name', pdf.get('file_name', 'PDF'))), 'author': pdf.get('author', ''), 'year': pdf.get('year', 'N/A'), 'subject': pdf.get('subject', 'PDF'), 'link': '', 'source': pdf.get('file_name', ''), 'page': None, 'text': pdf_text if has_extractable_pdf_text(pdf_text) else '', 'csv_file': 'pdf:' + pdf.get('file_name', ''), 'hash': '', 'doc_type': 'PDF', 'format': 'Số'}
                    search_results.insert(0, {'doc': inject_doc, 'score': 99999.0})
                    _injected = True
                    if len(related_pdf_infos) >= 3: break
            if _injected:
                context, doc_metadata_list = format_docs_with_metadata(search_results)
                total_found = len(search_results)
        except Exception:
            pass

        request_summary = detect_summary_request(req.query)
        _relevance_gate = False
        if catalog_mode:
            answer = build_catalog_answer(search_results, req.query, total_found=total_found)
        else:
            _top_score = search_results[0].get('score', 0) if search_results else 0
            if not search_results or _top_score < 1.0:
                answer = f"Mình chưa tìm thấy tài liệu nào liên quan đến **\"{req.query}\"** trong thư viện. 🤔\n\n" \
                         f"Bạn có thể thử:\n" \
                         f"• **Từ khóa khác** hoặc diễn đạt lại câu hỏi\n" \
                         f"• **Tìm theo tên sách**, **tác giả**, hoặc **chủ đề** cụ thể hơn\n" \
                         f"• **Duyệt danh mục** sách theo ngành học\n\n" \
                         f"Ví dụ: *\"Tìm sách về trí tuệ nhân tạo\"*"
                _relevance_gate = True
                search_results = []
            else:
                chain_func, _ = build_general_chain()
                answer = chain_func(context=context, question=req.query)

        pdf_fallback_used = False
        if not _relevance_gate and related_pdf_infos and answer_says_no_information(answer):
            _top_score_cat = search_results[0].get('score', 0) if search_results else 0
            if search_results and _top_score_cat >= 1.0:
                answer = build_catalog_answer(search_results, req.query, total_found=total_found)
                if related_pdf_infos:
                    answer += "\n\n" + build_related_pdf_answer(related_pdf_infos, req.query)
            elif not catalog_mode and not search_results and len(related_pdf_infos) > 0:
                try:
                    pdf_text = pdf_manager.get_chapter_text(related_pdf_infos[0]['file_path'], 0, 999999)
                    if has_extractable_pdf_text(pdf_text):
                        new_context = f"Tài liệu được chọn: {related_pdf_infos[0].get('title', 'PDF')}\nNội dung trích xuất:\n{pdf_text[:30000]}"
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
                except Exception:
                    answer = build_related_pdf_answer(related_pdf_infos, req.query)
            else:
                answer = build_related_pdf_answer(related_pdf_infos, req.query)

        _top_score = search_results[0].get('score', 0) if search_results else 0
        suppress_sources = not pdf_fallback_used and answer_says_no_information(answer) and (not search_results or (_top_score < 1.0 and not catalog_mode))
        if suppress_sources and search_results:
            search_results = []

        summary = ""
        if request_summary and search_results and not suppress_sources:
            try:
                doc_text = format_docs(search_results[:4])
                summary_chain = build_summary_prompt()
                summary = sanitize_answer_text(summary_chain(content=doc_text))
            except Exception:
                summary = ""

        seen, sources = set(), []
        current_doc = None
        for meta in ([] if suppress_sources else doc_metadata_list):
            title = meta['metadata']['title']
            author = meta['metadata'].get('author', '')
            year = meta['metadata'].get('year', '')
            source_id = (str(title).lower().strip(), str(author).lower().strip(), str(year).lower().strip())
            pdf_info = find_pdf_info_for_source(meta['metadata'].get('source', ''), title)
            if source_id not in seen:
                seen.add(source_id)
                source_item = {
                    "title": clean_display_value(meta['metadata']['title'], 300) or meta['metadata']['title'],
                    "author": clean_display_value(meta['metadata']['author'], 180),
                    "year": clean_display_value(meta['metadata']['year'], 40),
                    "subject": clean_display_value(meta['metadata']['subject'], 180),
                    "link": clean_display_value(meta['metadata']['link'], 500),
                    "source": clean_display_value(meta['metadata']['source'], 260),
                    "doc_type": clean_display_value(meta['metadata'].get('doc_type', 'N/A'), 80),
                    "format": clean_display_value(meta['metadata'].get('format', 'Số'), 80),
                    "major": clean_display_value(meta['metadata'].get('major', ''), 140),
                    "publisher": clean_display_value(meta['metadata'].get('publisher', ''), 160),
                    "place": clean_display_value(meta['metadata'].get('place', ''), 160),
                    "edition": clean_display_value(meta['metadata'].get('edition', ''), 60),
                    "bilingual_title": clean_display_value(meta['metadata'].get('bilingual_title', ''), 300),
                    "course": clean_display_value(meta['metadata'].get('course', ''), 140),
                    "keywords": meta['metadata'].get('keywords', []) or [],
                    "location": clean_display_value(meta['metadata'].get('location', ''), 800),
                    "abstract": clean_display_value(meta['metadata'].get('abstract', ''), 360),
                    "ddc": clean_display_value(meta['metadata'].get('ddc', ''), 80),
                    "notes": clean_display_value(meta['metadata'].get('notes', ''), 180),
                    "description": meta['text'][:200] if meta.get('text') else "",
                    "is_pdf": bool(pdf_info),
                    "pdf_id": pdf_info.get('id') if pdf_info else "",
                    "file_name": pdf_info.get('file_name') if pdf_info else "",
                    "display_name": pdf_info.get('display_name') if pdf_info else "",
                }
                sources.append(source_item)
                if req.document_id and title.lower() == req.document_id.lower():
                    current_doc = source_item

        if pdf_fallback_used and related_pdf_infos:
            for pdf in related_pdf_infos:
                fb_id = (pdf.get("title", "").lower(), pdf.get("author", "").lower(), str(pdf.get("year", "")))
                if fb_id not in seen:
                    seen.add(fb_id)
                    sources.append({"title": clean_display_value(pdf.get("title", "PDF"), 300), "author": clean_display_value(pdf.get("author", ""), 180), "year": clean_display_value(str(pdf.get("year", "")), 40), "subject": clean_display_value(pdf.get("subject", "PDF"), 180), "link": "", "source": clean_display_value(pdf.get("file_name", ""), 260), "doc_type": "PDF", "format": "", "major": "", "publisher": "", "place": "", "edition": "", "bilingual_title": "", "course": "", "keywords": [], "location": "", "abstract": "", "ddc": "", "notes": "", "description": "", "is_pdf": True, "pdf_id": pdf.get("id", ""), "file_name": pdf.get("file_name", ""), "display_name": pdf.get("display_name", "")})

        topic = ""
        if sources:
            topic = sources[0].get("subject", "") or sources[0].get("title", "")
        entry = {"query": req.query, "search_query": search_query, "topic": topic, "source_count": len(sources), "shown_titles": [s.get("title", "") for s in sources[:20]]}
        hist = session_history.get(req.session_id, [])
        hist.append(entry)
        session_history[req.session_id] = hist[-MAX_HISTORY:]

        total_found = len(sources)
        has_pdf_sources = any(s.get("is_pdf") for s in sources)
        follow_ups = build_follow_up_suggestions(query=req.query, sources=sources, answer=answer, catalog_mode=catalog_mode, has_pdf_sources=has_pdf_sources, session_id=req.session_id)

        return ChatResponse(answer=answer, sources=sources, total_sources=len(sources), total_found=total_found, summary=summary, current_document=current_doc, follow_up_suggestions=follow_ups)

    except Exception as e:
        err = str(e)
        if "connect" in err.lower() or "refused" in err.lower():
            raise HTTPException(503, "OpenRouter API không khả dụng.")
        raise HTTPException(500, err)


# ═══════════════════════════════════════════════════════════════
# ROUTES: /, /chat (HTML), /status, /health, /stats
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status": "ok", "message": "Thư viện QNU RAG API đang chạy", "llm_backend": "OpenRouter", "model": OPENROUTER_MODEL, "ui_url": "/chat", "read_only": is_vercel_runtime()}

@app.get("/chat", response_class=FileResponse)
def serve_chat_ui():
    html_file = "chat.html"
    if not os.path.exists(html_file):
        return {"error": "chat.html not found"}
    with open(html_file, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        tree = _load_faculties_tree()
        tree_json = _json.dumps(tree, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        tree_json = '{"faculties":[],"total_faculties":0}'
    inline = '<script id="__FACULTIES_TREE_DATA" type="application/json">' + tree_json + '</script>'
    marker = "<!-- __FACULTIES_TREE_INJECT__ -->"
    if marker in content:
        content = content.replace(marker, inline)
    else:
        content = content.replace('<script>\nconst BACKEND', inline + '\n<script>\nconst BACKEND', 1)
    return Response(content=content, media_type="text/html", headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})

@app.get("/static/SHL-logo.png", response_class=FileResponse, include_in_schema=False)
def serve_logo():
    return FileResponse("SHL-logo.png", media_type="image/png")

@app.get("/status")
def status():
    return {"status": "ok", "llm_backend": "OpenRouter", "model": OPENROUTER_MODEL, "openrouter_url": OPENROUTER_BASE_URL, "read_only": is_vercel_runtime()}

@app.get("/health")
def health():
    try:
        doc_count = len(documents)
        bm25_ready = bm25_engine is not None and bool(documents)
        return {"status": "ok", "documents": doc_count, "pdf_pages": 0, "indexes_ready": bm25_ready, "model": OPENROUTER_MODEL, "uptime_seconds": round(time.time() - APP_START_TIME, 1)}
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "degraded", "error": str(e)})

@app.get("/stats")
def stats():
    try:
        total = len(documents)
        pdf_pages = sum(1 for d in documents if str(d.get('csv_file', '')).startswith('pdf:')) if total else 0
        csv_rows = total - pdf_pages
        major_counter: dict[str, int] = {}
        empty_major = 0
        for d in documents:
            m = (d.get('major') or '').strip()
            if not m: empty_major += 1; continue
            key = normalize_text(m)
            if key: major_counter[key] = major_counter.get(key, 0) + 1
        top_majors = sorted(major_counter.items(), key=lambda x: x[1], reverse=True)[:20]
        source_counter: dict[str, int] = {}
        for d in documents:
            s = (d.get('source') or 'unknown').strip() or 'unknown'
            source_counter[s] = source_counter.get(s, 0) + 1
        type_counter: dict[str, int] = {}
        for d in documents:
            t = (d.get('doc_type') or '').strip() or 'unknown'
            type_counter[t] = type_counter.get(t, 0) + 1
        cache_info = _search_cache.stats()
        llm_info = {"calls": llm_stats["calls"], "failures": llm_stats["failures"], "retries": llm_stats["retries"], "last_error": llm_stats["last_error"], "circuit_state": llm_circuit["state"], "consecutive_failures": llm_circuit["failures"]}
        topic_cache_size = len(TOPIC_CACHE)
        sem_engine = get_semantic_engine()
        sem_available = sem_engine is not None and getattr(sem_engine, 'embeddings', None) is not None
        return {"status": "ok", "documents": {"total": total, "csv_rows": csv_rows, "pdf_pages": pdf_pages, "empty_major": empty_major, "empty_major_pct": round(empty_major / total * 100, 1) if total else 0}, "top_majors": [{"name": n, "count": c} for n, c in top_majors], "by_source": source_counter, "by_type": type_counter, "indexes": {"bm25_ready": bm25_engine is not None and bool(documents), "semantic_available": sem_available, "topic_cache_size": topic_cache_size}, "cache": cache_info, "llm": llm_info, "sessions": {"chat": len(session_history), "pdf": len(pdf_session_history)}, "model": OPENROUTER_MODEL}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


# ═══════════════════════════════════════════════════════════════
# ROUTES: Semantic search, Faculties, Voice
# ═══════════════════════════════════════════════════════════════

@app.post("/api/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search_endpoint(req: SemanticSearchRequest):
    engine = get_semantic_engine()
    if not engine or engine.embeddings is None:
        bm25_results = BM25SearchEngine(documents).search(req.query, top_k=req.top_k)
        return SemanticSearchResponse(query=req.query, mode="bm25_only", results=bm25_results, total=len(bm25_results), semantic_available=False)
    bm25_eng = BM25SearchEngine(documents)
    bm25_raw = bm25_eng.search(req.query, top_k=req.top_k * 2)
    bm25_results = [{"doc": r, "score": r.get("score", 0)} for r in bm25_raw]
    hybrid = engine.hybrid_search(req.query, bm25_results, alpha=req.alpha, top_k=req.top_k)
    return SemanticSearchResponse(query=req.query, mode="hybrid", results=[{"doc": r["doc"], "score": r["score"], "bm25_score": r.get("bm25_score", 0), "semantic_score": r.get("semantic_score", 0)} for r in hybrid], total=len(hybrid), semantic_available=True)

@app.get("/api/faculties")
async def get_faculties():
    return _load_faculties_tree()

@app.post("/api/faculties/search", response_model=SemanticSearchResponse)
async def search_by_faculty(req: FacultySearchRequest):
    nganh = (req.nganh or "").strip()
    if not nganh:
        raise HTTPException(status_code=400, detail="Thiếu 'nganh'.")
    nganh_norm = normalize_text(nganh)
    _tree = _load_faculties_tree()
    _canonical_nganh: set[str] = set()
    for f in (_tree.get("faculties") or []):
        for ng in (f.get("nganh") or []):
            ng_norm = _normalize_topic_token(ng.get("nganh", ""))
            if ng_norm and ng_norm == nganh_norm:
                _canonical_nganh.add(ng_norm)
    skip_major_filter = not _canonical_nganh

    def _doc_matches_faculty(doc_major_norm: str) -> bool:
        if not doc_major_norm: return False
        if doc_major_norm in _canonical_nganh: return True
        for cn in _canonical_nganh:
            if cn in doc_major_norm: return True
            if doc_major_norm.startswith(cn + " ") or doc_major_norm.startswith(cn + ","): return True
        return False

    topic_hits = _search_topic_cache_fn(nganh) or []
    if not topic_hits:
        bm25_raw = bm25_engine.search(nganh, top_k=req.top_k * 3)
        topic_hits = [{"doc": r.get("doc", r), "score": r.get("score", 0.0)} for r in bm25_raw]
    if not skip_major_filter:
        topic_hits = [hit for hit in topic_hits if _doc_matches_faculty(normalize_text(str(hit.get("doc", hit).get("major", ""))))]
    return SemanticSearchResponse(query=nganh, mode="faculty", results=topic_hits[:req.top_k], total=min(len(topic_hits), req.top_k), semantic_available=False)

@app.post("/api/voice/transcribe")
async def transcribe_voice(audio: UploadFile = File(...), language: str = Form("vi")):
    try:
        from lib.voice_input import save_uploaded_audio, transcribe_audio, cleanup_audio, is_available
    except ImportError:
        raise HTTPException(503, "Voice input chưa được cài.")
    if not is_available():
        raise HTTPException(503, "Whisper model chưa load xong.")
    suffix = Path(audio.filename or "voice.webm").suffix or ".webm"
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(400, "File audio rỗng")
    tmp_path = save_uploaded_audio(audio_bytes, suffix=suffix)
    try:
        return transcribe_audio(tmp_path, language=language)
    finally:
        cleanup_audio(tmp_path)

@app.get("/api/voice/status")
async def voice_status():
    try:
        from lib.voice_input import is_available
        return {"available": is_available(), "lang": os.getenv("WHISPER_LANG", "vi")}
    except ImportError:
        return {"available": False, "error": "voice_input module not found"}


# ═══════════════════════════════════════════════════════════════
# ROUTES: PDF management
# ═══════════════════════════════════════════════════════════════

@app.get("/api/pdf/{pdf_id}/content-with-ocr")
async def get_pdf_content_with_ocr(pdf_id: str, start: int = 0, end: int = 1, lang: str = "vie+eng"):
    pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p["id"] == pdf_id), None)
    if not pdf_info: raise HTTPException(404, f"PDF không tồn tại: {pdf_id}")
    result = pdf_manager.get_chapter_text_with_ocr_fallback(pdf_info["file_path"], start, end, lang=lang)
    return {"pdf_id": pdf_id, "title": pdf_info.get("title", ""), "start_page": start, "end_page": end, **result}

@app.get("/api/pdf/{pdf_id}/is-scanned")
async def check_pdf_scanned(pdf_id: str):
    pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p["id"] == pdf_id), None)
    if not pdf_info: raise HTTPException(404, f"PDF không tồn tại: {pdf_id}")
    return {"pdf_id": pdf_id, "is_scanned": pdf_manager.is_scanned_pdf(pdf_info["file_path"])}

@app.get("/api/pdfs")
async def list_pdfs():
    try:
        pdfs = pdf_manager.list_all_pdfs()
        return {"pdfs": pdfs, "total_pdfs": len(pdfs), "status": "ok"}
    except Exception as e:
        return {"error": str(e), "pdfs": [], "total_pdfs": 0}

@app.get("/api/pdfs/{pdf_id}/structure")
async def get_pdf_structure(pdf_id: str):
    try:
        pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p['id'] == pdf_id), None)
        if not pdf_info: raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")
        chapters = pdf_manager.extract_chapters(pdf_info['file_path'])
        return {"pdf_id": pdf_id, "title": pdf_info['title'], "author": pdf_info['author'], "chapters": chapters, "total_chapters": len(chapters)}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/pdfs/{pdf_id}/content")
async def get_pdf_content(pdf_id: str):
    try:
        pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p['id'] == pdf_id), None)
        if not pdf_info: raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")
        full_text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 999999)
        if not full_text or len(full_text.strip()) < 50:
            full_text = "PDF này không chứa nội dung text có thể trích xuất"
        return {"pdf_id": pdf_id, "title": pdf_info['title'], "total_pages": pdf_info.get('pages', 0), "content": full_text, "content_length": len(full_text)}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/api/pdfs/{pdf_id}/summary")
async def get_pdf_summary(pdf_id: str):
    try:
        pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p['id'] == pdf_id), None)
        if not pdf_info: raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")
        text = pdf_manager.get_chapter_text(pdf_info['file_path'], 0, 5)
        summary_text = text[:1000] if text and len(text.strip()) >= 50 else "PDF này không chứa nội dung text có thể trích xuất"
        return {"pdf_id": pdf_id, "title": pdf_info['title'], "author": pdf_info['author'], "year": pdf_info['year'], "summary": {"objectives": "Xem nội dung PDF để biết mục tiêu", "methods": "Xem nội dung PDF để biết phương pháp", "results": "Xem nội dung PDF để biết kết quả", "keywords": ["pdf", "tài liệu"], "preview": summary_text[:500]}}
    except HTTPException: raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/api/admin/login")
async def admin_login(data: dict):
    password = data.get("password", "")
    if password != ADMIN_PASSWORD: raise HTTPException(401, "Mật khẩu không đúng")
    token = secrets.token_hex(32)
    ADMIN_TOKENS.add(token)
    return {"token": token, "status": "ok"}

@app.post("/api/admin/logout")
async def admin_logout(data: dict):
    ADMIN_TOKENS.discard(data.get("token", ""))
    return {"status": "ok"}

@app.post("/api/admin/upload")
async def admin_upload(token: str = Form(...), file: UploadFile = File(...)):
    if is_vercel_runtime(): raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    if token not in ADMIN_TOKENS: raise HTTPException(401, "Token không hợp lệ")
    if not file.filename.lower().endswith(".pdf"): raise HTTPException(400, "Chỉ chấp nhận file PDF")
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024: raise HTTPException(400, "File vượt quá 50MB")
    pdf_dir = Path("pdfs"); pdf_dir.mkdir(exist_ok=True)
    dest = pdf_dir / file.filename
    if dest.exists(): raise HTTPException(400, f"File '{file.filename}' đã tồn tại")
    with open(dest, "wb") as f: f.write(contents)
    for p in [Path("pdf_cache.json"), Path("Data/_pdf_extract_cache.pkl")]:
        if p.exists(): p.unlink()
    _search_cache.invalidate()
    pdf_manager.invalidate_cache()
    try: reindex_pdfs(retriever)
    except Exception: pass
    return {"status": "ok", "file_name": file.filename, "file_size_mb": round(len(contents) / (1024 * 1024), 2), "message": f"Đã upload {file.filename} thành công!"}

@app.post("/api/admin/upload-data")
async def admin_upload_data(token: str = Form(...), file: UploadFile = File(...)):
    """Upload file Excel dữ liệu catalog. Tự động reload nếu dữ liệu khác, skip nếu giống."""
    if is_vercel_runtime(): raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    if token not in ADMIN_TOKENS: raise HTTPException(401, "Token không hợp lệ")
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(400, "Chỉ chấp nhận file Excel (.xlsx / .xls)")
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(400, "File vượt quá 50MB")

    # Tính hash nội dung file
    import hashlib
    new_hash = hashlib.md5(contents).hexdigest()
    data_dir = Path("Data")

    # Tìm file dữ liệu hiện tại
    existing_file = None
    for f in data_dir.glob("qnu-allITEM-*-chuan-hoa.xlsx"):
        existing_file = f
        break

    file_name_saved = file.filename
    saved_path = data_dir / file_name_saved

    # Nếu file đã tồn tại & cùng hash → skip
    if existing_file and existing_file.exists():
        old_hash = hashlib.md5(existing_file.read_bytes()).hexdigest()
        if new_hash == old_hash:
            return {
                "status": "skipped",
                "message": "Dữ liệu không thay đổi, bỏ qua reload.",
                "file_name": file_name_saved,
            }

    # Ghi file mới
    with open(saved_path, "wb") as f:
        f.write(contents)

    # Xoá cache cũ
    _cache_pkl = data_dir / "_documents_cache.pkl"
    if _cache_pkl.exists():
        _cache_pkl.unlink()

    # Reload dữ liệu
    try:
        new_documents = DocumentIndexer.load_from_data("Data")
        retriever.bm25.documents = new_documents
        retriever.bm25._build_index()
        global _semantic_engine
        _semantic_engine = None
        _build_topic_cache()
        _search_cache.invalidate()
        # Cập nhật mtime để check_files_changed không reload lại
        global _startup_data_mtime
        _startup_data_mtime = max(_startup_data_mtime, saved_path.stat().st_mtime)
        for _f in data_dir.glob("*"):
            if _f.is_file() and _f.name != "_documents_cache.pkl":
                _startup_data_mtime = max(_startup_data_mtime, _f.stat().st_mtime)
    except Exception as e:
        logger.error("Reload dữ liệu thất bại: %s", e)
        raise HTTPException(500, f"Reload dữ liệu thất bại: {str(e)}")

    return {
        "status": "ok",
        "file_name": file_name_saved,
        "file_size_mb": round(len(contents) / (1024 * 1024), 2),
        "message": f"Đã cập nhật dữ liệu từ {file_name_saved} thành công!",
    }

@app.delete("/api/admin/pdfs/{pdf_id}")
async def admin_delete_pdf(pdf_id: str, data: dict = Body(...)):
    if is_vercel_runtime(): raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    if data.get("token", "") not in ADMIN_TOKENS: raise HTTPException(401, "Token không hợp lệ")
    pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p["id"] == pdf_id), None)
    if not pdf_info: raise HTTPException(404, f"PDF không tìm thấy: {pdf_id}")
    target = Path(pdf_info["file_path"]).resolve()
    if target.suffix.lower() != ".pdf" or target.parent != Path("pdfs").resolve(): raise HTTPException(400, "Đường dẫn PDF không hợp lệ")
    file_name = pdf_info["file_name"]
    try: target.unlink()
    except FileNotFoundError: raise HTTPException(404, f"File không tồn tại: {file_name}")
    except Exception as e: raise HTTPException(500, f"Không thể xóa file: {e}")
    for p in [Path("pdf_cache.json"), Path("Data/_pdf_extract_cache.pkl")]:
        if p.exists():
            if p.name == "_pdf_extract_cache.pkl":
                try:
                    with open(p, "rb") as f: _pcache = pickle.load(f)
                    if file_name in _pcache:
                        del _pcache[file_name]
                        with open(p, "wb") as f: pickle.dump(_pcache, f, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception: pass
            else: p.unlink()
    pdf_manager.invalidate_cache()
    _search_cache.invalidate()
    removed = remove_pdf_from_index(file_name)
    return {"status": "ok", "pdf_id": pdf_id, "file_name": file_name, "removed_index_docs": removed, "message": f"Đã xóa {file_name}"}

@app.post("/chat/with-document")
async def chat_with_document(req: DocumentChatRequest):
    try:
        pdf_info = next((p for p in pdf_manager.list_all_pdfs() if p['id'] == req.pdf_id), None)
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
                return {"answer": answer, "pdf_id": req.pdf_id, "pdf_title": pdf_info['title'], "chapter": None, "relevant_pages": [start_page] if start_page else [], "source": {"type": "pdf", "pdf_id": req.pdf_id, "pdf_title": pdf_info['title'], "search_type": "references", "text_extractable": True}, "follow_up_suggestions": [{"label": "Tóm tắt nội dung", "query": "Tóm tắt nội dung chính"}, {"label": "Hỏi chương khác", "query": "Hỏi nội dung về một chương cụ thể"}]}

        context, _, _ = build_pdf_context_for_query(pdf_info, effective_query, None)
        if not context.strip():
            pdf_title = pdf_info.get('title') or pdf_info.get('display_name') or pdf_info.get('file_name') or req.pdf_id
            return {"answer": build_empty_pdf_answer(pdf_title), "pdf_id": req.pdf_id, "pdf_title": pdf_title, "chapter": None, "relevant_pages": [], "source": {"type": "pdf", "pdf_id": req.pdf_id, "pdf_title": pdf_title, "search_type": "full_pdf", "text_extractable": False}}

        chain_func, _ = build_general_chain()
        answer = chain_func(context=context, question=effective_query)
        pdf_hist.append({"query": req.query, "effective_query": effective_query, "topic": "general"})
        pdf_session_history[history_key] = pdf_hist[-MAX_HISTORY:]
        final_sugs = build_contextual_suggestions(answer=answer, query=req.query, doc_title=pdf_info.get('title', ''))
        return {"answer": answer, "pdf_id": req.pdf_id, "pdf_title": pdf_info['title'], "chapter": None, "relevant_pages": [1], "source": {"type": "pdf", "pdf_id": req.pdf_id, "pdf_title": pdf_info['title'], "search_type": "full_pdf"}, "follow_up_suggestions": final_sugs}
    except HTTPException: raise
    except Exception as e:
        logger.warning("Error in PDF chat: %s", e)
        raise HTTPException(500, str(e))
