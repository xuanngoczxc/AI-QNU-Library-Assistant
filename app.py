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
import asyncio
import json as _json
import pickle
import secrets
import random
from pathlib import Path
from typing import Optional
from collections import OrderedDict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

from lib.config import (
    OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODELS,
    OPENROUTER_BASE_URL, OPENROUTER_APP_TITLE, OPENROUTER_HTTP_REFERER,
    ADMIN_PASSWORD, VERCEL_READ_ONLY_MESSAGE, is_vercel_runtime, MAX_HISTORY,
)
from lib.state import (
    APP_START_TIME, ADMIN_TOKENS, llm_stats, llm_circuit,
    session_history, query_analytics,
)
from lib.logging_setup import logger
from lib.models import (
    ChatRequest, ChatResponse, SearchRequest, SuggestRequest,
    SpellCheckRequest, TermRequest, DocumentChatRequest,
    SemanticSearchRequest, SemanticSearchResponse,
    FacultySearchRequest, VoiceTranscribeRequest,
    FacetRequest, FacetFilterRequest, FacetRefineRequest,
)
from lib.llm_client import call_llm, check_backend, record_llm_failure
from lib.llm_prompts import (
    sanitize_answer_text, build_general_chain,
    detect_chapter_reference, detect_method_question, detect_structure_query,
    detect_result_question, build_chapter_prompt, build_method_prompt,
    build_structure_prompt,
)
# from lib.pdf_helpers import (
#     has_extractable_pdf_text, is_general_pdf_content_request,
#     build_empty_pdf_answer, detect_reference_request,
#     is_reference_heading_line, answer_says_no_information,
#     build_related_pdf_answer, build_reference_answer, parse_reference_items,
# )
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
    detect_study_major_query, detect_title_contains_keyword,
    detect_language_preference, detect_year_constraint,
    is_beginner_friendly_query, is_exam_prep_query,
    detect_teaching_subject, detect_course_name,
    apply_fuzzy_corrections, filter_by_language, filter_by_year,
    detect_generic_request,
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
    random_no_result_suggestions,
)
from lib.facets import extract_facets, filter_by_facet, refine_results
import lib.catalog as catalog_state

from lib.search_engine import DocumentIndexer, BM25SearchEngine, HybridSearchEngine
from lib.text_utils import (
    normalize_text, suggest_synonyms, spell_correct, suggest_terms_for_query,
)
from lib.intent_classifier import classify_intent, detect_greeting, extract_entities
from lib.usage_logger import get_usage_logger
# from lib.load_pdf import load_all_pdfs
# from lib.pdf_manager import PDFManager

load_dotenv()

if not is_vercel_runtime():
    check_backend()

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

logger.info("Load BM25 Full-Text Search Index...")
documents = DocumentIndexer.load_from_data("Data")

# pdf_docs = []
# use_pdf = False
# logger.info("PDF index DISABLED — only CSV/XLSX documents loaded.")
bm25_engine = BM25SearchEngine(documents)
retriever = HybridSearchEngine(bm25_engine)

logger.info("Search index ready: %d documents", len(documents))
logger.info("Memory: ~50MB (no ML model loaded)")

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

def _build_topic_cache():
    build_topic_cache(retriever.bm25.documents)


def _search_topic_cache_fn(query):
    return search_topic_cache(query, retriever.bm25.documents)


_build_topic_cache()

# pdf_manager = PDFManager(pdf_dir="pdfs")
# logger.info("PDF Manager initialized")
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

# PDF change detection disabled
_startup_data_mtime = 0.0
for _f in Path("Data").glob("*"):
    if _f.is_file() and _f.name != "_documents_cache.pkl":
        _startup_data_mtime = max(_startup_data_mtime, _f.stat().st_mtime)

_last_check_time = 0.0
_check_interval = 10

def check_files_changed():
    """PDF change detection disabled; only check data mtime."""
    global _last_check_time, _startup_data_mtime
    now = time.time()
    if now - _last_check_time < _check_interval:
        return {"pdf": False, "data": False}
    _last_check_time = now
    result = {"pdf": False, "data": False}
    # PDF check disabled
    current_data_mtime = 0.0
    if Path("Data").exists():
        for _f in Path("Data").glob("*"):
            if _f.is_file() and _f.name != "_documents_cache.pkl":
                current_data_mtime = max(current_data_mtime, _f.stat().st_mtime)
    if current_data_mtime > _startup_data_mtime:
        _startup_data_mtime = current_data_mtime
        result["data"] = True
    return result

# PDF reindex disabled
def reindex_pdfs(retriever):
    return 0


# PDF remove disabled
def remove_pdf_from_index(file_name: str) -> int:
    return 0

# PDF source lookup disabled
def find_pdf_info_for_source(source: str = "", title: str = "") -> Optional[dict]:
    return None

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
                "location": location[:2000] if location else '',
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


def _restore_topic_diacritics(normalized_topic: str, original_query: str) -> str:
    """Khôi phục dấu tiếng Việt cho topic từ câu hỏi gốc.
    
    VD: normalized_topic="van hoc", original_query="có tài liệu về văn học không"
    → trả về "Văn học"
    """
    if not normalized_topic or not original_query:
        return normalized_topic
    norm_query = normalize_text(original_query)
    idx = norm_query.find(normalized_topic)
    if idx < 0:
        return normalized_topic
    # Đếm số từ trong prefix để map vị trí
    norm_prefix = norm_query[:idx].strip()
    prefix_words = len(norm_prefix.split()) if norm_prefix else 0
    norm_words = normalized_topic.split()
    orig_words = original_query.lower().split()
    if prefix_words + len(norm_words) <= len(orig_words):
        result = ' '.join(orig_words[prefix_words:prefix_words + len(norm_words)])
        result = result.strip('"\'')
        return result[0].upper() + result[1:] if result else normalized_topic
    return normalized_topic


_GENERIC_TOPIC = frozenset([
    "lien quan", "gi", "nao", "sach", "tai lieu", "giao trinh",
    "cuon", "quyen", "an pham",
])

def _is_generic_topic(txt: str) -> bool:
    """Check if extracted text is just a generic stop-word, not a real topic."""
    if not txt:
        return True
    t = txt.strip()
    if t in _GENERIC_TOPIC:
        return True
    first = t.split()[0]
    return first in _GENERIC_TOPIC


def _strip_preamble_trailing(txt: str) -> str:
    """Remove trailing yes/no tokens then trailing preamble words (ban, ban co, etc)."""
    if not txt:
        return ""
    for _ in range(3):
        txt = txt.strip()
        if not txt:
            break
        m = re.search(r"\s+(\S+)$", txt)
        if m:
            if _is_yesno_token(m.group(1)):
                txt = txt[: m.start()].rstrip()
            else:
                break
        else:
            if _is_yesno_token(txt):
                txt = ""
            break
    for _ in range(3):
        txt = txt.strip()
        if not txt:
            break
        m = re.search(
            r"\s+(?:ban(?:\s+co(?:\s+(?:khong|ko|hem|nhi|vay|the|a))?)?|thi|the|vay|a)\s*$",
            txt, flags=re.IGNORECASE,
        )
        if m:
            txt = txt[: m.start()].strip()
        else:
            break
    return txt


def _extract_topic_fragment(txt: str) -> str:
    """Try to pull a topic out of a text fragment.
    Handles the "về X", "lĩnh vực X", "chủ đề X", "ngành X" pattern.
    """
    if not txt:
        return ""
    for _ in range(3):
        txt = txt.strip()
        if not txt:
            break
        m = re.search(r"\s+(\S+)$", txt)
        if m:
            if _is_yesno_token(m.group(1)):
                txt = txt[: m.start()].rstrip()
            else:
                break
        else:
            if _is_yesno_token(txt):
                txt = ""
            break
    txt = re.sub(
        r"^\s*(?:v[eề]|lĩnh\s*vực|chủ\s*đề|ngành)\s+",
        " ", txt, flags=re.IGNORECASE,
    ).strip()
    txt = re.sub(r"\?+\s*$", " ", txt).strip()
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return ""
    m = re.search(
        r".*\b(?:v[eề]|lĩnh\s*vực|chủ\s*đề|ngành)\s+(.+)$",
        txt, flags=re.IGNORECASE,
    )
    if m:
        txt = m.group(1).strip()
        for _ in range(3):
            txt = txt.strip()
            if not txt:
                break
            m2 = re.search(r"\s+(\S+)$", txt)
            if m2:
                if _is_yesno_token(m2.group(1)):
                    txt = txt[: m2.start()].rstrip()
                else:
                    break
            else:
                if _is_yesno_token(txt):
                    txt = ""
                break
        txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def extract_existence_topic(query: str) -> str:
    """Extract the topic/subject from an existence-check query.
    Strategy:
    1. Normalise and strip leading preamble tokens (bạn, cho mình hỏi, …).
    2. Locate "có (sách|tài liệu|…)" **anywhere** — split into before/after.
    3. Try extracting from the *after* fragment first.
    4. If the result is generic/garbage, fall back to the *before* fragment
       (after stripping trailing preamble tokens).
    """
    if not query:
        return ""
    cleaned = normalize_text(query)
    # ---- 1. Strip leading preamble ----
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

    # ---- 2. Split at "có (sách|tài liệu|…)" anywhere in the string ----
    before = ""
    after = ""
    m = re.search(
        r"\bco\s+(?:tai\s+lieu|sach|cuon|quyen|an\s+pham|giao\s+trinh)\b",
        cleaned, flags=re.IGNORECASE,
    )
    if m:
        before = cleaned[: m.start()].strip()
        after = cleaned[m.end() :].strip()
    else:
        before = cleaned

    # ---- 3. Try extracting from 'after' first ----
    topic = _extract_topic_fragment(after)
    if topic and not _is_generic_topic(topic):
        return topic

    # ---- 4. Fallback to 'before' (clean trailing preamble first) ----
    if before:
        before_clean = _strip_preamble_trailing(before)
        topic2 = _extract_topic_fragment(before_clean)
        if topic2:
            return topic2

    return ""


def _last_existence_topic(history) -> str:
    if not history:
        return ""
    # Ưu tiên: tìm entry có key "topic" (do handler existence lưu)
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("topic") and msg.get("search_query"):
            return msg["topic"]
    # Fallback: regex từ assistant messages
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


def _analytics_key(query: str) -> str:
    key = normalize_text(query or "").strip()
    key = re.sub(r"\s+", " ", key)
    return key[:160] or "unknown"


def _record_query_analytics(query: str, result_count: int = 0) -> None:
    key = _analytics_key(query)
    label = (query or "").strip()[:180] or key
    no_result = result_count <= 0
    query_analytics["total"] += 1
    query_analytics["top_queries"][key] += 1
    query_analytics["labels"][key] = label
    if no_result:
        query_analytics["no_result_queries"][key] += 1
    query_analytics["recent"].appendleft({
        "query": label,
        "result_count": int(result_count),
        "no_result": no_result,
        "ts": time.time(),
    })
    query_analytics["last_query_at"] = time.time()


def _format_ts(ts: float | None) -> str:
    if not ts:
        return ""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _latest_data_update_ts() -> float:
    latest = 0.0
    # PDF check disabled; only check Data/
    root = Path("Data")
    if root.exists():
        for item in root.glob("*"):
            if item.is_file() and not item.name.endswith((".pkl", ".cache")):
                latest = max(latest, item.stat().st_mtime)
    return latest


def _counter_items(counter, limit: int = 8) -> list[dict]:
    labels = query_analytics.get("labels", {})
    return [
        {"query": labels.get(key, key), "count": count}
        for key, count in counter.most_common(limit)
    ]


def _dashboard_payload() -> dict:
    total_docs = len(documents)
    latest_ts = _latest_data_update_ts()

    # Danh sách file XLSX trong thư mục Data/
    data_dir = Path("Data")
    files = []
    for f in sorted(data_dir.glob("*.xlsx")):
        if f.name.startswith("~$"):
            continue
        files.append({
            "name": f.name,
            "size": f.stat().st_size,
            "mtime": _format_ts(f.stat().st_mtime),
        })

    return {
        "status": "ok",
        "documents": {
            "total": total_docs,
        },
        "updated": {
            "last_data_update": _format_ts(latest_ts),
        },
        "files": files,
        "indexes": {
            "bm25_ready": bm25_engine is not None and bool(documents),
            "topic_cache_size": len(catalog_state.TOPIC_CACHE),
        },
        "cache": _search_cache.stats(),
    }

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
            if has_author_filter and normalize_text(req.author) in normalize_text(doc.get('author', '')):
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

        # Nếu BM25 không tìm thấy kết quả nhưng có author/subject filter → fallback quét documents
        if not filtered and (has_author_filter or has_subject_filter):
            # Fallback cho author
            if has_author_filter:
                author_norm = normalize_text(req.author).strip()
                for doc in retriever.bm25.documents:
                    _doc_auth_fields = [
                        normalize_text(doc.get('author', '') or '').strip(),
                    ]
                    _co = doc.get('co_authors', '') or ''
                    if isinstance(_co, list):
                        _doc_auth_fields.extend(normalize_text(str(v)).strip() for v in _co)
                    else:
                        _doc_auth_fields.append(normalize_text(_co).strip())
                    _au = doc.get('authors', '') or ''
                    if isinstance(_au, list):
                        _doc_auth_fields.extend(normalize_text(str(v)).strip() for v in _au)
                    else:
                        _doc_auth_fields.append(normalize_text(_au).strip())
                    _matched = any(
                        f and (author_norm in f or f in author_norm)
                        for f in _doc_auth_fields
                    )
                    if not _matched:
                        continue
                    title_lower = doc['title'].lower().strip()
                    if req.title and title_lower == req.title.lower().strip():
                        continue
                    if title_lower in exclude_set:
                        continue
                    if title_lower in seen_titles:
                        continue
                    seen_titles.add(title_lower)
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
                        "reason": "Cùng tác giả",
                        "score": 0,
                    })
                    if len(filtered) >= req.top_k * 5:
                        break
            # Fallback cho subject (chủ đề)
            if not filtered and has_subject_filter:
                subject_lower = req.subject.lower().strip()
                for doc in retriever.bm25.documents:
                    doc_subject = (doc.get('subject', '') or '').lower().strip()
                    doc_major = (doc.get('major', '') or '').lower().strip()
                    doc_title = (doc.get('title', '') or '').lower().strip()
                    if subject_lower not in doc_subject and subject_lower not in doc_major and subject_lower not in doc_title:
                        continue
                    title_lower = doc['title'].lower().strip()
                    if req.title and title_lower == req.title.lower().strip():
                        continue
                    if title_lower in exclude_set:
                        continue
                    if title_lower in seen_titles:
                        continue
                    seen_titles.add(title_lower)
                    reasons = []
                    if subject_lower in doc_subject or subject_lower in doc_major:
                        reasons.append("Cùng chủ đề")
                    elif subject_lower in doc_title:
                        reasons.append("Cùng chủ đề")
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
                        "score": 0,
                    })
                    if len(filtered) >= req.top_k * 5:
                        break

        return {"suggestions": filtered[:req.top_k], "total": len(filtered)}
    except Exception as e:
        return {"error": str(e), "suggestions": [], "total": 0}

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

@app.post("/api/facets/extract")
async def facet_extract(req: FacetRequest):
    """
    Nhóm kết quả tìm kiếm theo các tiêu chí (chủ đề, loại, tác giả, năm...).
    Frontend dùng để hiển thị các nhóm cho người dùng chọn.
    """
    try:
        search_query = req.query or ""
        if not search_query.strip():
            return {"facets": {}, "total": 0, "error": "Query is empty"}

        results = retriever.search_by_query(search_query, top_k=req.top_k)
        facets = extract_facets(results, top_k=10)
        return {"facets": facets, "total": len(results)}
    except Exception as e:
        logger.error("Facet extract error: %s", e)
        return {"facets": {}, "total": 0, "error": str(e)}

@app.post("/api/facets/filter")
async def facet_filter(req: FacetFilterRequest):
    """
    Lọc kết quả theo một nhóm facet cụ thể (VD: chọn "Sách" trong doc_type).
    """
    try:
        search_query = req.query or ""
        if not search_query.strip():
            return {"results": [], "total": 0, "error": "Query is empty"}

        results = retriever.search_by_query(search_query, top_k=req.top_k)
        filtered = filter_by_facet(results, req.group_name, req.group_value)

        docs = []
        seen = set()
        for r in filtered:
            doc = r["doc"]
            key = doc["title"].lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "title": doc.get("title", ""),
                "author": doc.get("author", "Unknown"),
                "year": doc.get("year", "N/A"),
                "subject": doc.get("subject", "N/A"),
                "major": doc.get("major", ""),
                "publisher": doc.get("publisher", ""),
                "doc_type": doc.get("doc_type", "N/A"),
                "link": doc.get("link", ""),
                "location": doc.get("location", ""),
                "abstract": doc.get("abstract", ""),
                "ddc": doc.get("ddc", ""),
                "notes": doc.get("notes", ""),
                "score": round(r.get("score", 0), 2),
                "source": doc.get("source", ""),
                "csv_file": doc.get("csv_file", ""),
            })

        # Trả về cả facets cập nhật để frontend có thể refresh
        updated_facets = extract_facets(filtered, top_k=10)

        return {"results": docs, "total": len(docs),
                "group_name": req.group_name, "group_value": req.group_value,
                "facets": updated_facets}
    except Exception as e:
        logger.error("Facet filter error: %s", e)
        return {"results": [], "total": 0, "error": str(e)}

@app.post("/api/facets/refine")
async def facet_refine(req: FacetRefineRequest):
    """
    Áp dụng nhiều bộ lọc cùng lúc (AND).
    VD: filters={"author": "Nguyễn Văn A", "year": "2020 - 2025"}
    """
    try:
        search_query = req.query or ""
        if not search_query.strip():
            return {"results": [], "total": 0, "error": "Query is empty"}

        results = retriever.search_by_query(search_query, top_k=req.top_k)
        filtered = refine_results(results, req.filters)

        docs = []
        seen = set()
        for r in filtered:
            doc = r["doc"]
            key = doc["title"].lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            docs.append({
                "title": doc.get("title", ""),
                "author": doc.get("author", "Unknown"),
                "year": doc.get("year", "N/A"),
                "subject": doc.get("subject", "N/A"),
                "major": doc.get("major", ""),
                "publisher": doc.get("publisher", ""),
                "doc_type": doc.get("doc_type", "N/A"),
                "link": doc.get("link", ""),
                "location": doc.get("location", ""),
                "abstract": doc.get("abstract", ""),
                "ddc": doc.get("ddc", ""),
                "notes": doc.get("notes", ""),
                "score": round(r.get("score", 0), 2),
                "source": doc.get("source", ""),
                "csv_file": doc.get("csv_file", ""),
            })

        updated_facets = extract_facets(filtered, top_k=10)

        return {"results": docs, "total": len(docs),
                "facets": updated_facets, "filters": req.filters}
    except Exception as e:
        logger.error("Facet refine error: %s", e)
        return {"results": [], "total": 0, "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, fast_request: Request = None):
    _chat_start = time.time()
    # Resolve client IP from header or request
    _client_ip = (req.client_ip or "").strip()
    if not _client_ip and fast_request:
        forwarded = fast_request.headers.get("x-forwarded-for", "")
        if forwarded:
            _client_ip = forwarded.split(",")[0].strip()
        if not _client_ip:
            _client_ip = getattr(fast_request.client, "host", "") or ""
    _user_agent = (req.user_agent or "").strip()
    if not _user_agent and fast_request:
        _user_agent = fast_request.headers.get("user-agent", "") or ""
    try:
        normalized_query = normalize_text(req.query)
        intent = classify_intent(req.query)
        logger.info("Intent: %s for query: %s", intent, req.query)

        # ── Xử lý câu trả lời khẳng định/phủ định (trước intent) ──
        list_topic_marker = "__LIST_TOPIC__:"
        list_topic_query = None
        if req.query.startswith(list_topic_marker):
            list_topic_query = req.query[len(list_topic_marker):].strip()

        hist = session_history.get(req.session_id, [])

        # Phủ định: "không", "thôi"...
        _affirm_yes = {"co", "ok", "oke", "okie", "duoc", "can", "muon",
                       "u", "um", "ừ", "ừm", "được", "cần", "muốn",
                       "liet ke", "liệt kê", "xem", "cho xem", "cho minh xem"}
        q_affirm = normalize_text(req.query or "").strip()
        if q_affirm in _affirm_yes and hist:
            last_entry = hist[-1]
            prev_topic = last_entry.get("topic", "")
            if prev_topic:
                list_topic_query = prev_topic

        _decline_pat = re.compile(
            r"^\s*(không|khong|thôi|thoi|thôi nhé|thoi nha|cảm ơn|cam on|cám ơn|cam_on|thank|thanks|no|nah|đủ rồi|du roi|thôi bỏ qua|thoi bo qua)\s*[.!]?\s*$",
            re.IGNORECASE,
        )
        if not list_topic_query and hist and _decline_pat.match(req.query.strip()):
            last_topic_entry = None
            for entry in reversed(hist):
                if isinstance(entry, dict) and entry.get("topic"):
                    last_topic_entry = entry
                    break
            if last_topic_entry:
                decline_topic = last_topic_entry.get("topic", "")
                answer_text = f"Được, mình sẽ không liệt kê **{decline_topic}** nữa. Bạn muốn mình hỗ trợ gì khác không?"
                hist.append({"role": "assistant", "content": answer_text})
                session_history[req.session_id] = hist[-MAX_HISTORY:]
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=0, current_document=None,
                    follow_up_suggestions=[{"label": "Gợi ý chủ đề khác", "query": "Gợi ý tài liệu phổ biến"}, {"label": "Tìm sách theo tác giả", "query": "sách của Nguyễn Nhật Ánh"}])

        if list_topic_query:
            req_query_for_search = f"tìm sách về {list_topic_query}"
            req.query = req_query_for_search
            normalized_query = normalize_text(req_query_for_search)
            intent = classify_intent(req_query_for_search)
            logger.info("List-topic redirect → query=%r intent=%s", req.query, intent)

        # ── Fuzzy correction: "Pythn" → "Python" (chạy sớm, trước mọi xử lý) ──
        _corrected = apply_fuzzy_corrections(req.query)
        if _corrected and _corrected != normalize_text(req.query):
            req.query = _corrected

        # ── Enhanced detection (chạy trước intent checks) ──
        _study_major_now = detect_study_major_query(req.query)
        _lang_filter_now = detect_language_preference(req.query)
        _year_filter_now, _year_dir_now = detect_year_constraint(req.query)
        _title_kw_now = detect_title_contains_keyword(req.query)
        _generic_now = bool(re.search(
            r"(?:goi y|vai cuon|mot vai|it cuon)\s*(?:sach)?",
            normalize_text(req.query)
        )) and not req.query.startswith(list_topic_marker) and not bool(re.search(r'\b(python|java|javascript|ai|ml|machine|deep|data|web|lap trinh)\b', normalize_text(req.query)))
        _skip_existence = bool(_study_major_now) or bool(_year_filter_now)

        # Nếu intent "unknown" mà có enhanced signals → treat as catalog
        if intent == "unknown" and (_study_major_now or _lang_filter_now or _year_filter_now):
            intent = "catalog"

        if intent == "chitchat":
            return ChatResponse(
                answer="Xin chào! Tôi là trợ lý tra cứu của **Thư viện QNU**.\n\nBạn có thể hỏi tôi về:\n• Tìm sách theo tên, tác giả, chủ đề\n• Tra cứu tài liệu số\n• Gợi ý sách theo ngành học\n\nBạn cần tìm tài liệu gì?",
                sources=[], total_sources=0, total_found=0, current_document=None, follow_up_suggestions=[],
            )
        if intent == "live_info":
            return ChatResponse(
                answer="Mình là trợ lý Thư viện, mình chỉ hỗ trợ tra cứu sách và tài liệu thôi ạ!\n\n"
                       "Bạn có thể thử:\n"
                       "• **Tìm sách** theo tên, tác giả hoặc chủ đề\n"
                       "• **Hỏi về nội dung** tài liệu trong thư viện\n\n"
                       "Ví dụ: *\"Tìm sách về trí tuệ nhân tạo\"*",
                sources=[], total_sources=0, total_found=0, current_document=None,
                follow_up_suggestions=[
                    {"label": "Tìm sách về công nghệ thông tin", "query": "tìm sách về công nghệ thông tin"},
                    {"label": "Tìm sách về văn học", "query": "tìm sách về văn học"},
                ],
            )

        # ── Generic suggestion: chạy trước exhaustion/unknown/clarify ──
        if _generic_now:
            import random as _rnd
            _pool = [r for r in retriever.search_by_query("sach", top_k=200)
                     if r.get("doc", r).get("title", "") and not r.get("doc", r).get("is_pdf", False)]
            if _pool:
                _selected = _rnd.sample(_pool, min(8, len(_pool)))
                _lines = "\n".join(
                    f"• **{r.get('doc', r).get('title','')}**" + (f" – {r.get('doc', r).get('author','')}" if r.get('doc', r).get('author') and r['doc']['author'] not in ('Unknown','N/A') else "")
                    for r in _selected
                )
                answer_text = f"Mình gợi ý một số cuốn sách trong thư viện:\n\n{_lines}"
                return ChatResponse(answer=answer_text, sources=[r.get('doc', r) for r in _selected], total_sources=len(_selected),
                    total_found=len(_selected), current_document=None,
                    follow_up_suggestions=_rnd.sample([
                        {"label": "Tìm sách về trí tuệ nhân tạo", "query": "tìm sách về trí tuệ nhân tạo"},
                        {"label": "Sách ngành Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
                        {"label": "Sách về Kinh tế", "query": "tìm sách về Kinh tế"},
                        {"label": "Tìm sách của Nguyễn Nhật Ánh", "query": "tìm sách của Nguyễn Nhật Ánh"},
                    ], min(3, 4)))

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
            return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=total_count, current_document=None, follow_up_suggestions=[])

        if intent == "unknown":
            # PDF check disabled
            return ChatResponse(
                answer=
                           "Mình là trợ lý Thư viện, mình có thể giúp bạn:\n"
                           "• **Tìm sách, tài liệu** trong thư viện\n"
                           "• **Gợi ý tài liệu** theo chủ đề\n"
                           "Bạn hãy thử hỏi cụ thể hơn nhé!",
                    sources=[], total_sources=0, total_found=0, current_document=None,
                    follow_up_suggestions=[
                        {"label": "Tìm sách về trí tuệ nhân tạo", "query": "tìm sách về trí tuệ nhân tạo"},
                        {"label": "Có tài liệu về văn học không?", "query": "có tài liệu về văn học không"},
                    ],
                )

        if intent == "clarify":
            query_normalized = normalize_text(req.query).strip()
            if "the loai" in query_normalized:
                answer = ("Mình chưa rõ ý bạn lắm!\n\n"
                         "Bạn đang muốn: **TÌM SÁCH** về thể loại này, hay hỏi về **NỘI DUNG** thể loại văn học?\n\n"
                         "Vui lòng chọn:")
                suggestions = [
                    {"label": "Tìm sách thể loại này", "query": "tim sach the loai " + query_normalized.replace("the loai","")},
                    {"label": "Nội dung thể loại văn học", "query": "noi dung the loai van hoc"},
                    {"label": "Sách nổi bật trong thể loại", "query": "sach noi bat the loai " + query_normalized.replace("the loai","")},
                ]
            elif "tac gia" in query_normalized:
                answer = ("Mình chưa rõ ý bạn lắm!\n\n"
                         "Bạn đang muốn: **TÌM SÁCH** của tác giả này, hay hỏi về **THÔNG TIN** tác giả?\n\n"
                         "Vui lòng chọn:")
                suggestions = [
                    {"label": "Tìm sách của tác giả này", "query": "tim sach tac gia " + query_normalized.replace("tac gia","")},
                    {"label": "Thông tin về tác giả", "query": "thong tin tac gia " + query_normalized.replace("tac gia","")},
                    {"label": "Tác phẩm nổi bật", "query": "tac pham noi bat cua " + query_normalized.replace("tac gia","")},
                ]
            elif "nha xuat ban" in query_normalized:
                answer = ("Mình chưa rõ ý bạn lắm!\n\n"
                         "Bạn đang muốn: **TÌM SÁCH** của nhà xuất bản này, hay hỏi về **THÔNG TIN** nhà xuất bản?\n\n"
                         "Vui lòng chọn:")
                suggestions = [
                    {"label": "Tìm sách NXB này", "query": "tim sach nha xuat ban " + query_normalized.replace("nha xuat ban","")},
                    {"label": "Danh sách NXB", "query": "danh sach nha xuat ban"},
                    {"label": "Sách mới của NXB", "query": "sach moi nha xuat ban " + query_normalized.replace("nha xuat ban","")},
                ]
            else:
                _clarify_pool = [
                    {"label": "Tìm sách về trí tuệ nhân tạo", "query": "tìm sách về trí tuệ nhân tạo"},
                    {"label": "Sách ngành Tâm lý học", "query": "tìm sách ngành tâm lý học"},
                    {"label": "Tìm sách của Nguyễn Nhật Ánh", "query": "tìm sách của Nguyễn Nhật Ánh"},
                    {"label": "Sách ngành Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
                    {"label": "Sách về Kinh tế học", "query": "tìm sách về Kinh tế học"},
                    {"label": "Tìm sách của Stuart J. Russell", "query": "tìm sách của Stuart J. Russell"},
                    {"label": "Sách ngành Giáo dục", "query": "tìm sách ngành Giáo dục"},
                    {"label": "Sách về Xây dựng", "query": "tìm sách về Xây dựng"},
                    {"label": "Tìm sách của Lê Xuân Vinh", "query": "tìm sách của Lê Xuân Vinh"},
                    {"label": "Sách ngành Toán ứng dụng", "query": "tìm sách ngành Toán ứng dụng"},
                    {"label": "Sách về Sinh học", "query": "tìm sách về Sinh học"},
                    {"label": "Tìm sách ngành Quản lý kinh tế", "query": "tìm sách ngành Quản lý kinh tế"},
                ]
                suggestions = random.sample(_clarify_pool, min(4, len(_clarify_pool)))
                answer = ("Mình chưa hiểu rõ câu hỏi của bạn.\n\n"
                         "Mình là trợ lý Thư viện, bạn có thể hỏi mình về:\n"
                         "• **Tìm sách** theo tên, tác giả, chủ đề\n"
                         "• **Tra cứu tài liệu** theo ngành học\n"
                         "• **Gợi ý sách** phù hợp với nhu cầu\n\n"
                         "Thử chọn một gợi ý bên dưới nhé:")
            return ChatResponse(answer=answer, sources=[], total_sources=0, total_found=0, current_document=None,
                follow_up_suggestions=suggestions)

        request_content = detect_content_request(req.query)
        is_list_request = detect_list_request(req.query)
        catalog_mode = is_list_request or (not request_content and intent != "unknown")

        if detect_greeting(req.query):
            greeting_answer = (
                "Xin chào! Tôi là trợ lý tra cứu của **Thư viện QNU**.\n\n"
                "Bạn có thể hỏi tôi về:\n"
                "• Tìm sách theo tên, tác giả, chủ đề\n"
                "• Tra cứu tài liệu số\n"
                "• Gợi ý sách theo ngành học\n\n"
                "Bạn cần tìm tài liệu gì?"
            )
            return ChatResponse(answer=greeting_answer, sources=[], total_sources=0, current_document=None, follow_up_suggestions=[])

        _skip_existence = bool(_study_major_now) or bool(_year_filter_now)

        # ── Title-contains-keyword: "có chữ Python trong tên không?" ──
        if _title_kw_now:
            _tk_count = retriever.count_title_match(_title_kw_now)
            hist.append({"query": req.query, "search_query": _title_kw_now, "topic": _title_kw_now, "source_count": _tk_count, "shown_titles": []})
            session_history[req.session_id] = hist[-MAX_HISTORY:]
            if _tk_count <= 0:
                answer_text = f"Hiện tại mình chưa tìm thấy tài liệu nào có chứa từ **{_title_kw_now}** trong thư viện."
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=0, current_document=None,
                    follow_up_suggestions=random_no_result_suggestions(3))
            answer_text = f"Có, thư viện có **{_tk_count} tài liệu** có chứa từ **{_title_kw_now}** trong tên. Bạn có muốn xem danh sách không?"
            return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=_tk_count, current_document=None,
                follow_up_suggestions=[
                    {"label": "Xem danh sách", "query": f"{list_topic_marker}{_title_kw_now}"},
                    {"label": "Không, cảm ơn", "query": "Cảm ơn bạn"},
                ])

        # ── Existence check: "có X không?" (trừ khi có enhanced context) ──
        if not _skip_existence and detect_existence_request(req.query):
            topic = extract_existence_topic(req.query)
            topic_label = topic or "chủ đề này"
            if topic:
                topic_label = _restore_topic_diacritics(topic, req.query)
            if topic:
                count = retriever.count_title_match(topic)
                if count <= 0:
                    answer_text = f"Hiện tại mình chưa tìm thấy tài liệu nào về **{topic_label}** trong thư viện. Bạn có thể thử hỏi chủ đề khác hoặc diễn đạt khác đi một chút nhé."
                    return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=0, current_document=None,
                        follow_up_suggestions=random_no_result_suggestions(3))
                hist.append({"query": req.query, "search_query": topic, "topic": topic_label, "source_count": count, "shown_titles": []})
                session_history[req.session_id] = hist[-MAX_HISTORY:]
                answer_text = f"Có, thư viện hiện có tài liệu về **{topic_label}**. Bạn có muốn mình liệt kê chi tiết ra không?"
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=count, current_document=None,
                    follow_up_suggestions=[
                        {"label": "Liệt kê chi tiết", "query": f"{list_topic_marker}{topic}"},
                        {"label": "Không, cảm ơn bạn", "query": "Cảm ơn bạn"},
                    ])

        if list_topic_query:
            req_query_for_search = f"tìm sách về {list_topic_query}"
            req.query = req_query_for_search
            normalized_query = normalize_text(req_query_for_search)
            is_list_request = True
            catalog_mode = True

        if not req.major and not req.document_id:
            try:
                from lib.catalog import _extract_major_from_query                # "học CNTT" / "học Công nghệ thông tin"
                _study_major = detect_study_major_query(req.query)
                if _study_major:
                    req.major = _study_major
                else:
                    _detected_major = _extract_major_from_query(req.query)
                    if _detected_major:
                        req.major = _detected_major
            except Exception:
                pass

        file_changes = check_files_changed()
        # PDF reindex disabled
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
                _build_topic_cache()
                _search_cache.invalidate()
            except Exception:
                pass

        search_query = resolve_search_query(req.query, hist, catalog_mode=catalog_mode)

        if req.major:
            major_clean = (req.major or "").strip()
            # So sánh normalize (không dấu) để tránh append trùng
            _major_norm = normalize_text(major_clean)
            _search_norm = normalize_text(search_query)
            if major_clean and _major_norm and _major_norm not in _search_norm:
                search_query = f"{search_query} {major_clean}".strip()

        # ── Enhanced query detection (teaching subject, course, beginner, exam) ──
        _teaching_subj = detect_teaching_subject(req.query)
        _course_name = detect_course_name(req.query)
        _is_beginner = is_beginner_friendly_query(req.query)
        _is_exam_prep = is_exam_prep_query(req.query)
        if _teaching_subj and _teaching_subj not in search_query:
            search_query = f"{search_query} {_teaching_subj}".strip()
        if _course_name and _course_name not in search_query:
            search_query = f"{search_query} {_course_name}".strip()
        if _is_beginner:
            _beginner_terms = ["co ban", "nhap mon", "can ban"]
            for bt in _beginner_terms:
                if bt not in search_query:
                    search_query = f"{search_query} {bt}".strip()
                    break

        # Nếu có year filter, loại bỏ năm khỏi search_query (year filter xử lý sau search)
        if _year_filter_now and search_query:
            _stripped = re.sub(r'\b(19\d{2}|20\d{2})\b', '', search_query).strip()
            _stripped = re.sub(r'\s+', ' ', _stripped)
            if _stripped:
                search_query = _stripped

        _prefer_strict = detect_author_query(req.query) or detect_publisher_query(req.query)
        cache_key = f"catalog:{search_query}:{_prefer_strict}:{req.major or ''}" if catalog_mode else None
        use_cache = catalog_mode and not req.document_id and not req.query.startswith("__SUGGEST")
        search_results = _search_cache.get(cache_key) if use_cache else None
        retrieval_k = len(retriever.bm25.documents) if catalog_mode else 30

        if search_results is not None:
            logger.info("Search cache HIT (%d docs, key='%s')", len(search_results), search_query)
        else:
            cache_results = []
            if catalog_mode and not is_list_request:
                cache_results = _search_topic_cache_fn(search_query)
            bm25_results = retriever.search_by_query(search_query, top_k=retrieval_k)
            if catalog_mode and bm25_results and not _prefer_strict:
                bm25_results = [r for r in bm25_results if not str(r.get("doc", {}).get("csv_file", "")).startswith("pdf:") and r.get("doc", {}).get("page") is None]
            catalog_query_token_count = len(topic_tokens(strip_search_intent_phrases(req.query)))
            if catalog_mode and (is_list_request or catalog_query_token_count <= 5):
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
            # PDF page collapsing disabled
            if use_cache and search_results:
                _search_cache.set(cache_key, search_results)

        # Kiểm tra: nếu là author query mà không có kết quả nào khớp tác giả → ép fallback
        if search_results and _prefer_strict and catalog_mode:
            _author_name_check = extract_author_name(req.query) if detect_author_query(req.query) else ""
            if _author_name_check:
                _found_author_match = any(
                    author_name_matches(r.get("doc", {}).get("author", ""), _author_name_check)
                    for r in search_results
                )
                if not _found_author_match:
                    logger.info("Author query '%s' → no match in BM25 results, forcing fallback", _author_name_check)
                    search_results = []

        # ── Fallback search: nếu BM25/cache 0 kết quả, thử query đơn giản hơn ──
        if not search_results and catalog_mode:
            _fallback_queries = []
            _fb_seen = set()
            def _add_fb(q):
                if q and q not in _fb_seen and len(q) >= 2:
                    _fb_seen.add(q)
                    _fallback_queries.append(q)
            # Hàm strip year
            def _strip_year(t):
                return re.sub(r'\s+', ' ', re.sub(r'\b(19\d{2}|20\d{2})\b', '', t)).strip()
            # 0. fallback dành riêng cho author query: quét trực tiếp author field
            if _prefer_strict:
                _author_name = extract_author_name(req.query) if detect_author_query(req.query) else ""
                _pub_name = extract_publisher_name(req.query) if detect_publisher_query(req.query) else ""
                _fb_author_key = normalize_text(_author_name or _pub_name or "")
                if _fb_author_key:
                    _fb_scan = []
                    _seen_t2 = set()
                    for _doc in retriever.bm25.documents:
                        # Kiểm tra tất cả các field chứa tên tác giả
                        _doc_fields_to_check = [
                            _doc.get('author', '') or '',
                            _doc.get('co_authors', '') or '',
                            _doc.get('authors', '') or '',
                        ]
                        if isinstance(_doc_fields_to_check[1], list):
                            _doc_fields_to_check[1] = ' '.join(str(v) for v in _doc_fields_to_check[1])
                        if isinstance(_doc_fields_to_check[2], list):
                            _doc_fields_to_check[2] = ' '.join(str(v) for v in _doc_fields_to_check[2])
                        _matched = False
                        for _field_val in _doc_fields_to_check:
                            _field_norm = normalize_text(_field_val)
                            if not _field_norm:
                                continue
                            _check = _fb_author_key
                            if _check in _field_norm or _field_norm in _check:
                                _matched = True
                                break
                        if not _matched and _pub_name:
                            _doc_pub = normalize_text(_doc.get('publisher', '') or '')
                            _check_pub = normalize_text(_pub_name)
                            if _check_pub in _doc_pub:
                                _matched = True
                        if not _matched:
                            continue
                        _t2 = (_doc.get('title', '') or '').lower().strip()
                        if _t2 and _t2 not in _seen_t2:
                            _seen_t2.add(_t2)
                            _fb_scan.append({'doc': _doc, 'score': 10.0, 'index': 0})
                            if len(_fb_scan) >= retrieval_k:
                                break
                    if _fb_scan:
                        search_results = _fb_scan
                        logger.info("Fallback author scan '%s' → %d results", _fb_author_key[:40], len(search_results))
                        _prefer_strict = False
            # Nếu không phải author query hoặc author scan vẫn 0, thử fallback thường
            if not search_results and not _prefer_strict:
                # 1. Teaching subject / course name
                if _teaching_subj:
                    _add_fb(_teaching_subj)
                elif _course_name:
                    _add_fb(_course_name)
                # 2. normalize_text gốc
                _simple = _strip_year(normalize_text(req.query))
                if len(_simple) >= 3:
                    _add_fb(_simple)
                # 3. extract_primary_topic_text
                from lib.catalog import extract_primary_topic_text
                _core = _strip_year(normalize_text(extract_primary_topic_text(req.query)))
                _add_fb(_core)
                # 4. Content words: lọc stop words, noise, year numbers
                _content_noise = {
                    'co','ve','tu','tro','di','khong','toi','can','de','ban','nao',
                    'phu','hop','bang','xuat','sau','tim','sach','cua','cac','cho',
                    'khi','la','nhu','lan','qua','con','hay','lam','moi','mot',
                    'vao','theo','duoc','neu','hoac','ne','thi','vay','nua','da',
                    'het','them','xuat','nam','gioi','thieu','thu','vien','nguoi',
                    'cung','rat','dang','phai','tu','la','co','cach','nay',
                }
                _cw_parts = [w for w in _strip_year(normalize_text(req.query)).split()
                             if w not in VIETNAMESE_STOP_WORDS and w not in _content_noise and len(w) > 2]
                _cw = ' '.join(_cw_parts)
                _add_fb(_cw)
                # 5. Domain expansion terms
                from lib.catalog import expand_domain_terms
                for _dt in expand_domain_terms(req.query):
                    _add_fb(_strip_year(normalize_text(_dt)))
                # 6. Tech keyword extraction
                _tech_terms = re.findall(
                    r'\b(python|java|javascript|typescript|sql|html|css|react|angular|'
                    r'vue|node|mongodb|docker|kubernetes|flutter|swift|kotlin|rust|go\b(?!\w)|'
                    r'machine\s*learning|deep\s*learning|ai|data\s*science|artificial\s*intelligence|'
                    r'big\s*data|cloud\s*computing|blockchain|iot|devops|agile|scrum)\b',
                    normalize_text(req.query), re.IGNORECASE
                )
                if _tech_terms:
                    _add_fb(' '.join(_tech_terms))
                for _fb in _fallback_queries:
                    if not _fb or len(_fb) < 2:
                        continue
                    _fb_results = retriever.search_by_query(_fb, top_k=retrieval_k)
                    if _fb_results:
                        search_results = _fb_results
                        logger.info("Fallback search '%s' → %d results", _fb[:60], len(search_results))
                        break

        if is_list_request and list_topic_query and search_results:
            _topic_norm = normalize_text(list_topic_query).strip()
            if _topic_norm:
                search_results = [
                    r for r in search_results
                    if _topic_norm in normalize_text(r.get("doc", {}).get("title", "") or "")
                    or _topic_norm in normalize_text(r.get("doc", {}).get("subject", "") or "")
                ]

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

        # ── Language filter: "tiếng Anh" / "Song ngữ" ──
        if _lang_filter_now and search_results:
            search_results = filter_by_language(search_results, _lang_filter_now)

        # ── Year filter: "xuất bản từ 2023", "sau 2022" ──
        if _year_filter_now and search_results:
            search_results = filter_by_year(search_results, _year_filter_now, _year_dir_now or "exact")

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

        # PDF auto-inject and fallback disabled
        related_pdf_infos = []

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

        # PDF fallback disabled
        pdf_fallback_used = False

        _top_score = search_results[0].get('score', 0) if search_results else 0
        # answer_says_no_information check disabled (was from pdf_helpers)
        suppress_sources = not pdf_fallback_used and (not search_results or (_top_score < 1.0 and not catalog_mode))
        if suppress_sources and search_results:
            search_results = []

        seen, sources = set(), []
        current_doc = None
        for meta in ([] if suppress_sources else doc_metadata_list):
            title = meta['metadata']['title']
            author = meta['metadata'].get('author', '')
            year = meta['metadata'].get('year', '')
            source_id = (str(title).lower().strip(), str(author).lower().strip(), str(year).lower().strip())
            # PDF info lookup disabled
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
                    "location": clean_display_value(meta['metadata'].get('location', ''), 3000),
                    "abstract": clean_display_value(meta['metadata'].get('abstract', ''), 360),
                    "ddc": clean_display_value(meta['metadata'].get('ddc', ''), 80),
                    "notes": clean_display_value(meta['metadata'].get('notes', ''), 180),
                    "description": meta['text'][:200] if meta.get('text') else "",
                    "is_pdf": False,
                    "pdf_id": "",
                    "file_name": "",
                    "display_name": "",
                }
                sources.append(source_item)
                if req.document_id and title.lower() == req.document_id.lower():
                    current_doc = source_item

        # PDF fallback in sources disabled
        topic = ""
        if sources:
            topic = sources[0].get("subject", "") or sources[0].get("title", "")
        entry = {"query": req.query, "search_query": search_query, "topic": topic, "source_count": len(sources), "shown_titles": [s.get("title", "") for s in sources[:20]]}
        hist = session_history.get(req.session_id, [])
        hist.append(entry)
        session_history[req.session_id] = hist[-MAX_HISTORY:]

        total_found = len(sources)
        # has_pdf_sources disabled
        follow_ups = build_follow_up_suggestions(query=req.query, sources=sources, answer=answer, catalog_mode=catalog_mode, has_pdf_sources=False, session_id=req.session_id)

        _record_query_analytics(req.query, total_found)

        # ── Usage logging ──
        try:
            get_usage_logger().record(
                session_id=req.session_id,
                query=req.query,
                result_count=total_found,
                latency_ms=(time.time() - _chat_start) * 1000,
                follow_up_count=len(follow_ups),
                client_ip=_client_ip,
                user_agent=_user_agent,
                intent=intent,
            )
        except Exception:
            pass

        return ChatResponse(answer=answer, sources=sources, total_sources=len(sources), total_found=total_found, current_document=current_doc, follow_up_suggestions=follow_ups)

    except Exception as e:
        err = str(e)
        # Log error usage
        try:
            get_usage_logger().record(
                session_id=req.session_id,
                query=req.query,
                result_count=0,
                latency_ms=(time.time() - _chat_start) * 1000,
                error=err[:300],
                client_ip=_client_ip,
                user_agent=_user_agent,
                intent="error",
            )
        except Exception:
            pass
        if "connect" in err.lower() or "refused" in err.lower():
            raise HTTPException(503, "OpenRouter API không khả dụng.")
        raise HTTPException(500, err)


def _stream_chunks(text: str, size: int = 96):
    words = (text or "").split(" ")
    chunk = ""
    for word in words:
        candidate = (chunk + " " + word).strip()
        if len(candidate) >= size and chunk:
            yield chunk
            chunk = word
        else:
            chunk = candidate
    if chunk:
        yield chunk


def _sse_event(payload: dict) -> str:
    return "data: " + _json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n\n"


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    async def events():
        try:
            analytics_before = query_analytics["total"]
            response = await chat(req)
            payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()
            if query_analytics["total"] == analytics_before:
                _record_query_analytics(req.query, payload.get("total_found", 0))
            answer = payload.get("answer", "")
            for chunk in _stream_chunks(answer):
                yield _sse_event({"type": "chunk", "text": chunk + " "})
                await asyncio.sleep(0.015)
            yield _sse_event({"type": "sources", "sources": payload.get("sources", [])})
            yield _sse_event({"type": "total_found", "total_found": payload.get("total_found", 0)})
            yield _sse_event({"type": "follow_ups", "follow_up_suggestions": payload.get("follow_up_suggestions", [])})
            yield _sse_event({"type": "done"})
        except HTTPException as e:
            yield _sse_event({"type": "error", "message": e.detail})
        except Exception as e:
            yield _sse_event({"type": "error", "message": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

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

# ── Serve embed/widget files ──────────────────────────────
@app.get("/embed/{filename}", include_in_schema=False)
def serve_embed(filename: str):
    """Serve files from embed/ directory (qnu-chat-widget.js, demo.html, etc.)"""
    import os as _os
    safe_path = _os.path.normpath(_os.path.join("embed", filename))
    if not safe_path.startswith("embed"):
        raise HTTPException(403, "Forbidden")
    full = _os.path.join(_os.path.dirname(__file__), safe_path)
    if not _os.path.exists(full):
        raise HTTPException(404, "File not found")
    ext = _os.path.splitext(filename)[1].lower()
    media_map = {".js": "application/javascript", ".html": "text/html", ".css": "text/css", ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml"}
    return FileResponse(full, media_type=media_map.get(ext, "application/octet-stream"))

@app.get("/status")
def status():
    return {"status": "ok", "llm_backend": "OpenRouter", "model": OPENROUTER_MODEL, "openrouter_url": OPENROUTER_BASE_URL, "read_only": is_vercel_runtime()}

@app.get("/admin", response_class=FileResponse)
def serve_admin_dashboard():
    html_file = "admin-dashboard.html"
    if not os.path.exists(html_file):
        return {"error": "admin-dashboard.html not found"}
    return FileResponse(html_file, media_type="text/html", headers={
        "Cache-Control": "no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })

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
        topic_cache_size = len(catalog_state.TOPIC_CACHE)
        sem_available = _semantic_engine is not None and getattr(_semantic_engine, 'embeddings', None) is not None
        return {"status": "ok", "documents": {"total": total, "csv_rows": csv_rows, "pdf_pages": pdf_pages, "empty_major": empty_major, "empty_major_pct": round(empty_major / total * 100, 1) if total else 0}, "top_majors": [{"name": n, "count": c} for n, c in top_majors], "by_source": source_counter, "by_type": type_counter, "indexes": {"bm25_ready": bm25_engine is not None and bool(documents), "semantic_available": sem_available, "topic_cache_size": topic_cache_size}, "cache": cache_info, "llm": llm_info, "sessions": {"chat": len(session_history), "pdf": 0}, "model": OPENROUTER_MODEL}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})

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

# ═══════════════════════════════════════════════════════════════
# Usage / Analytics endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/api/usage/analytics")
def usage_analytics(date_from: str = "", date_to: str = ""):
    """Public analytics page. Returns aggregated stats from usage logs.
    Optional query params: date_from, date_to (YYYY-MM-DD)."""
    try:
        return get_usage_logger().compute_analytics(date_from=date_from, date_to=date_to)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/usage/recent")
def usage_recent(n: int = 20):
    """Return recent usage entries."""
    try:
        return {"entries": get_usage_logger().get_recent(n)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/api/usage/stats")
def usage_stats():
    """Quick usage stats (from in-memory counters)."""
    try:
        return get_usage_logger().get_stats()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

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
# PDF management DISABLED — all PDF/OCR routes removed
# ═══════════════════════════════════════════════════════════════

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

@app.post("/api/admin/dashboard")
async def admin_dashboard(data: dict = Body(...)):
    if data.get("token", "") not in ADMIN_TOKENS:
        raise HTTPException(401, "Token không hợp lệ")
    return _dashboard_payload()

@app.post("/api/admin/upload")
async def admin_upload(token: str = Form(...), file: UploadFile = File(...)):
    if is_vercel_runtime(): raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    if token not in ADMIN_TOKENS: raise HTTPException(401, "Token không hợp lệ")
    # PDF upload disabled — only Excel uploads supported via /api/admin/upload-data
    raise HTTPException(400, "Upload PDF đã bị tắt. Chỉ hỗ trợ upload dữ liệu Excel qua /api/admin/upload-data")

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

    # Kiểm tra hash với TẤT CẢ file xlsx hiện có (bỏ qua file cache)
    # Nếu nội dung trùng 1 trong các file cũ → skip
    from datetime import datetime
    import time as _time
    any_match = False
    for old_f in data_dir.glob("*.xlsx"):
        if old_f.name.startswith("~$"):  # skip Excel temp files
            continue
        try:
            old_hash = hashlib.md5(old_f.read_bytes()).hexdigest()
            if new_hash == old_hash:
                any_match = True
                break
        except Exception:
            continue
    if any_match:
        return {
            "status": "skipped",
            "message": "Dữ liệu không thay đổi, bỏ qua reload.",
            "file_name": "(unchanged)",
        }

    # Lưu file mới với tên riêng (KHÔNG xóa file cũ)
    # load_from_data() sẽ load tất cả .xlsx và merge + dedup tự động
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    upload_name = f"upload-{ts}.xlsx"
    saved_path = data_dir / upload_name
    with open(saved_path, "wb") as f:
        f.write(contents)
    logger.info("Saved uploaded file: %s (%.1f MB)", upload_name, len(contents) / 1024 / 1024)

    # ── Cleanup: chỉ xóa các file upload-*.xlsx CŨ (lần upload trước) ──
    # File gốc qnu-allITEM-*.xlsx KHÔNG BAO GIỜ bị xóa
    # load_from_data() sẽ load tất cả + dedup tự động
    for old_f in list(data_dir.glob("upload-*.xlsx")):
        if old_f.name == upload_name:
            continue
        old_f.unlink()
        logger.info("Removed old upload file: %s", old_f.name)

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
        "file_name": upload_name,
        "file_size_mb": round(len(contents) / (1024 * 1024), 2),
        "message": f"Đã cập nhật dữ liệu từ {upload_name} thành công!",
    }

# PDF delete disabled
@app.delete("/api/admin/pdfs/{pdf_id}")
async def admin_delete_pdf(pdf_id: str, data: dict = Body(...)):
    raise HTTPException(404, "PDF management đã bị tắt.")

# ── Xóa file dữ liệu ────────────────────────────────────────
@app.post("/api/admin/delete-file")
async def admin_delete_file(data: dict = Body(...)):
    """Xóa file Excel trong thư mục Data/ và reload lại dữ liệu."""
    if is_vercel_runtime(): raise HTTPException(405, VERCEL_READ_ONLY_MESSAGE)
    if data.get("token", "") not in ADMIN_TOKENS:
        raise HTTPException(401, "Token không hợp lệ")
    file_name = data.get("file_name", "")
    if not file_name:
        raise HTTPException(400, "Thiếu tên file")

    data_dir = Path("Data")
    target = data_dir / file_name

    # An toàn: chỉ cho xóa file .xlsx, không cho xóa file khác
    if target.suffix.lower() not in (".xlsx", ".xls"):
        raise HTTPException(400, "Chỉ hỗ trợ xóa file Excel")
    # Không cho xóa file đang dùng nếu chỉ còn 1 file
    xlsx_files = list(data_dir.glob("*.xlsx"))
    if len(xlsx_files) <= 1:
        raise HTTPException(400, "Không thể xóa file cuối cùng. Cần ít nhất 1 file dữ liệu.")
    if not target.exists():
        raise HTTPException(404, f"File {file_name} không tồn tại")

    target.unlink()
    logger.info("Deleted data file: %s", file_name)

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
    except Exception as e:
        logger.error("Reload dữ liệu thất bại: %s", e)
        raise HTTPException(500, f"Reload dữ liệu thất bại: {str(e)}")

    return {
        "status": "ok",
        "message": f"Đã xóa {file_name} và reload dữ liệu thành công!",
    }

# PDF chat disabled
@app.post("/chat/with-document")
async def chat_with_document(req: DocumentChatRequest):
    raise HTTPException(404, "Chat với PDF đã bị tắt.")
