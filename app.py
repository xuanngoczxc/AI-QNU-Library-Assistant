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
    session_history, query_analytics, get_session_history, save_session_history,
)
from lib.logging_setup import logger
from lib.models import (
    ChatRequest, ChatResponse, SearchRequest, SuggestRequest,
    SpellCheckRequest, TermRequest, DocumentChatRequest,
    SemanticSearchRequest, SemanticSearchResponse,
    FacultySearchRequest, VoiceTranscribeRequest,
    FacetRequest, FacetFilterRequest, FacetRefineRequest,
)
from lib.llm_client import (
    call_llm, call_llm_async, stream_llm_async, check_backend, record_llm_failure,
)
from lib.llm_prompts import (
    sanitize_answer_text, build_general_chain, build_general_chain_async,
    prepare_rag_messages,
    detect_chapter_reference, detect_method_question, detect_structure_query,
    detect_result_question, build_chapter_prompt, build_method_prompt,
    build_structure_prompt,
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
    detect_study_major_query, _extract_major_from_query, detect_title_contains_keyword,
    detect_language_preference, detect_year_constraint,
    is_beginner_friendly_query, is_exam_prep_query,
    detect_teaching_subject, detect_course_name,
    apply_fuzzy_corrections, filter_by_language, filter_by_year,
    detect_generic_request, doc_matches_year, is_pure_year_query,
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
    clean_author_for_suggestion, clean_suggestion_topic, truncate_words,
)
from lib.facets import extract_facets, filter_by_facet, refine_results
import lib.catalog as catalog_state

from lib.search_engine import DocumentIndexer, BM25SearchEngine, HybridSearchEngine
from lib.text_utils import (
    normalize_text, suggest_synonyms, spell_correct, suggest_terms_for_query,
    extract_author_candidate, clean_author_candidate, VIETNAMESE_SURNAMES,
    is_generic_book_listing_query, GENERIC_TITLES,
)
from lib.intent_classifier import classify_intent, detect_greeting, extract_entities
from lib.usage_logger import get_usage_logger
from lib.major_resolver import (
    resolve_canonical_major_from_query,
    get_canonical_major_aliases,
    doc_matches_canonical_major,
)
from lib.doc_type_resolver import (
    detect_doc_type_from_query,
    clean_query_doc_type_terms,
    doc_matches_target_doc_type,
    infer_doc_type_from_doc,
)
from lib.conversation_resolver import (
    detect_ordinal_reference,
    detect_anaphoric_reference,
    detect_attribute_question,
    build_book_attribute_answer,
    build_book_follow_ups,
    detect_exhaustion_query as detect_exhaustion_query_resolved,
    is_follow_up_filter,
    is_follow_up_refinement,
    is_singular_book_request,
    get_inherited_context,
    detect_recency_preference,
    detect_oldest_preference,
    strip_recency_phrases,
    strip_temporal_sorting_phrases,
    clean_conversational_query,
    parse_natural_query,
    sort_results_by_year,
    extract_clarification_content,
)
from lib.author_resolver import AuthorIndex, handle_author_inquiry
# from lib.load_pdf import load_all_pdfs
# from lib.pdf_manager import PDFManager

load_dotenv()

if not is_vercel_runtime():
    check_backend()

from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Th\u01b0 vi\u1ec7n QNU RAG API", version="2.0.0")

# Mount React assets if frontend/dist exists
_frontend_dist = Path("frontend/dist")
if (_frontend_dist / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="react_assets")


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "microphone=(self), camera=(self)"
    return response


cors_origins_env = os.getenv("CORS_ORIGINS", "").strip()
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] if cors_origins_env else [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
if "*" in allowed_origins or not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
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
author_index = AuthorIndex(documents)
logger.info("Author index ready: %d unique authors", len(author_index.author_lexicon))
for _d in documents:
    if '_title_norm' not in _d:
        _d['_title_norm'] = normalize_text(_d.get('title', '')).strip().lower()

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
            clean = f"Tiêu đề: {title}"
            if valid(author): clean += f"\nTác giả: {author}"
            if valid(publisher): clean += f"\nNhà xuất bản: {publisher}"
            if valid(year): clean += f"\nNăm xuất bản: {year}"
            if valid(location): clean += f"\nVị trí / Phòng đọc: {location}"
            if valid(ddc): clean += f"\nMã phân loại DDC: {ddc}"
            if valid(major): clean += f"\nNgành đào tạo: {major}"
            if valid(abstract): clean += f"\nTóm tắt: {abstract[:1200]}"
            else: clean += "\nTóm tắt: (Không có tóm tắt)"
            content = clean[:3000]

        meta_dict = dict(doc)
        meta_dict.pop('text', None)
        meta_dict.pop('flat_text', None)
        formatted.append({
            "text": content,
            "metadata": meta_dict,
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
    # Nếu là câu hỏi tìm sách / giáo trình / danh mục hoặc chuyên ngành, không coi là content request trừ khi yêu cầu đọc/tóm tắt
    if re.search(r"\b(?:giao trinh|tim sach|sach ve|cac sach|danh sach|co sach|nganh)\b", q):
        if not re.search(r"\b(?:tom tat|doc file|chi tiet ve|noi dung cua|chuong|trich xuat)\b", q):
            return False
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
    if any(re.search(r"\b" + re.escape(phrase) + r"\b", q) for phrase in content_phrases):
        return True
    command_patterns = [
        r"\b(?:hay|vui long)?\s*(?:neu|phan tich|trinh bay|giai thich|tom tat|doc)\b",
        r"\b(?:cho toi|cho minh|cho em)\s+(?:biet|xem|tom tat)\b",
    ]
    return any(re.search(pattern, q) for pattern in command_patterns)


def detect_exhaustion_query(query: str) -> bool:
    return detect_exhaustion_query_resolved(query)


def is_follow_up(query: str) -> bool:
    if detect_ai_domain(query):
        return False
    if is_follow_up_refinement(query) or is_follow_up_filter(query):
        return True
    keywords = extract_primary_topic_text(query)
    return len(keywords.split()) == 0

def resolve_search_query(query: str, history: list, catalog_mode: bool = False) -> str:
    # If this is a follow-up refinement (e.g., 'vậy lựa ra quyển có năm xuất bản gần nhất'), inherit topic cleanly
    if is_follow_up_refinement(query) and history:
        inherited = get_inherited_context(history)
        inherited_q = inherited.get("search_query") or inherited.get("topic")
        if inherited_q:
            return inherited_q

    if catalog_mode:
        keywords = extract_keywords(query) if hasattr(extract_keywords, '__call__') else extract_content_search_query(query)
    else:
        keywords = extract_content_search_query(query)
    norm = normalize_text(query)
    if is_follow_up(query) and history:
        last = history[-1]
        topic = normalize_text(last.get("topic", ""))
        last_q = normalize_text(last.get("search_query", ""))
        if last_q:
            return last_q
        if topic and len(topic) >= 3:
            return topic
    if keywords:
        return keywords
    return norm


def _format_catalog_type_breakdown(search_results: list) -> str:
    """Đếm và định dạng cơ cấu loại tài liệu một cách tự nhiên."""
    if not search_results:
        return ""
    counts = {}
    for r in search_results:
        d = r.get("doc", r)
        dt = (d.get("doc_type") or "").strip() or "Sách"
        counts[dt] = counts.get(dt, 0) + 1

    if not counts:
        return ""

    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    parts = [f"**{c} {dt}**" for dt, c in sorted_items]

    if len(parts) == 1:
        return f"toàn bộ là {parts[0]}"
    if len(parts) == 2:
        return f"gồm {parts[0]} và {parts[1]}"
    return f"gồm {', '.join(parts[:-1])} và {parts[-1]}"


def _format_top_book_highlight(search_results: list) -> str:
    """Trích xuất 1 cuốn sách tiêu biểu để giới thiệu tự nhiên trong câu trả lời."""
    if not search_results:
        return ""
    top_doc = search_results[0].get("doc", search_results[0])
    title = (top_doc.get("title") or "").strip().rstrip('.')
    if not title or len(title) < 3:
        return ""
    author = (top_doc.get("author") or "").strip()
    year = str(top_doc.get("year") or "").strip()

    auth_str = f" của tác giả **{author}**" if author and author not in ("N/A", "Unknown", "None", "") else ""
    year_str = f" (năm {year})" if year and year.isdigit() else ""

    return f" Tiêu biểu có cuốn **\"{title}\"**{year_str}{auth_str}."


def build_catalog_answer(
    search_results,
    query: str,
    total_found: int = 0,
    doc_type: str = None,
    major: str = None,
    is_recency: bool = False,
    is_oldest: bool = False,
    topic_display: str = None,
    is_singular: bool = False,
) -> str:
    if not search_results:
        if doc_type:
            if major:
                return f"Hiện tại Thư viện QNU chưa tìm thấy tài liệu loại **{doc_type}** thuộc ngành **{major}**."
            return f"Hiện tại Thư viện QNU chưa tìm thấy tài liệu loại **{doc_type}** phù hợp trong dữ liệu."
        return "Không tìm thấy tài liệu phù hợp trong dữ liệu hiện có."

    displayed = len(search_results)
    total = total_found if total_found > 0 else displayed
    top_doc = search_results[0].get("doc", search_results[0])
    top_title = (top_doc.get("title") or "").strip().rstrip('.')
    top_year = top_doc.get("year", "")
    top_author = (top_doc.get("author") or "").strip()
    type_breakdown = _format_catalog_type_breakdown(search_results)
    type_str = f" ({type_breakdown})" if type_breakdown else ""
    highlight_str = _format_top_book_highlight(search_results)

    if is_recency:
        t_str = f" về **{topic_display}**" if topic_display else ""
        if is_singular and top_title:
            year_info = f" (năm {top_year})" if top_year and str(top_year).strip() not in ('N/A', 'Unknown', 'None', '') else ""
            author_info = f" của tác giả **{top_author}**" if top_author and str(top_author).strip() not in ('N/A', 'Unknown', 'None', '') else ""
            return f"Tài liệu có năm xuất bản gần đây nhất{t_str} là cuốn **\"{top_title}\"**{year_info}{author_info}.\n\nDưới đây là danh sách **{total} tài liệu** liên quan được sắp xếp theo năm mới nhất{type_str}:"
        if displayed < total:
            return f"Thư viện QNU có **{total} tài liệu**{t_str} được xuất bản gần đây nhất{type_str} (dưới đây hiển thị **{displayed} tài liệu** sắp xếp theo năm mới nhất). Xem chi tiết bên dưới:"
        return f"Thư viện QNU có **{total} tài liệu**{t_str} được xuất bản gần đây nhất{type_str} (sắp xếp theo năm mới nhất). Xem chi tiết bên dưới:"

    if is_oldest:
        t_str = f" về **{topic_display}**" if topic_display else ""
        if is_singular and top_title:
            year_info = f" (năm {top_year})" if top_year and str(top_year).strip() not in ('N/A', 'Unknown', 'None', '') else ""
            author_info = f" của tác giả **{top_author}**" if top_author and str(top_author).strip() not in ('N/A', 'Unknown', 'None', '') else ""
            return f"Tài liệu có năm xuất bản lâu đời nhất{t_str} là cuốn **\"{top_title}\"**{year_info}{author_info}.\n\nDưới đây là danh sách **{total} tài liệu** liên quan được sắp xếp từ năm cũ nhất{type_str}:"
        if displayed < total:
            return f"Thư viện QNU có **{total} tài liệu**{t_str} được xuất bản lâu đời nhất{type_str} (dưới đây hiển thị **{displayed} tài liệu** sắp xếp từ năm cũ nhất đến mới nhất). Xem chi tiết bên dưới:"
        return f"Thư viện QNU có **{total} tài liệu**{t_str} được xuất bản lâu đời nhất{type_str} (sắp xếp từ năm cũ nhất đến mới nhất). Xem chi tiết bên dưới:"

    if doc_type == "Kiểm định":
        if major:
            return f"Thư viện QNU hiện có **{total} tài liệu** phục vụ cho **Kiểm định chất lượng** ngành **{major}** (Báo cáo tự đánh giá & Danh mục minh chứng CTĐT). Mời bạn xem chi tiết các hồ sơ bên dưới:"
        return f"Thư viện QNU hiện có **{total} tài liệu** phục vụ cho **Kiểm định chất lượng** (gồm Báo cáo tự đánh giá & Danh mục minh chứng kiểm định chương trình đào tạo của các ngành). Mời bạn xem chi tiết các hồ sơ bên dưới:"

    if doc_type:
        dt_label = doc_type.lower()
        t_part = f" về **{topic_display}**" if topic_display else ""
        m_part = f" thuộc ngành **{major}**" if major else ""
        return f"Thư viện QNU tìm thấy **{total} {dt_label}**{t_part}{m_part}.{highlight_str}\n\nMời bạn xem chi tiết danh sách tài liệu bên dưới để tra cứu vị trí phòng đọc và mã xếp giá nhé:"

    # ── Natural, Conversational Phrasing for General Catalog Search ──
    if topic_display:
        if topic_display.lower().startswith("xuất bản") or topic_display.lower().startswith("năm"):
            t_phrase = f" **{topic_display}**"
        else:
            t_phrase = f" về chủ đề **{topic_display}**"
    else:
        t_phrase = " liên quan"
    m_phrase = f" thuộc ngành **{major}**" if major else ""

    variant = (abs(hash(query)) % 3) if query else 0
    if variant == 0:
        return (
            f"Dạ, mình đã tìm thấy **{total} tài liệu**{t_phrase}{m_phrase} trong kho lưu trữ của Thư viện QNU{type_str}.{highlight_str}\n\n"
            f"Mình đã phân loại chi tiết theo từng nhóm ở danh mục bên dưới, bạn hãy bấm vào từng mục để xem thông tin chi tiết và vị trí phòng đọc nhé! 👇"
        )
    elif variant == 1:
        return (
            f"Thư viện QNU hiện có **{total} tài liệu**{t_phrase}{m_phrase}{type_str}.{highlight_str}\n\n"
            f"Danh sách chi tiết đã được sắp xếp theo từng thể loại ở bên dưới để bạn tiện tra cứu:"
        )
    else:
        return (
            f"Chào bạn, hệ thống tìm thấy **{total} tài liệu** phù hợp{t_phrase}{m_phrase}{type_str}.{highlight_str}\n\n"
            f"Bạn có thể mở từng nhóm tài liệu bên dưới để xem thêm thông tin và chọn cuốn phù hợp nhất nhé!"
        )


def detect_document_types_overview_query(query: str) -> bool:
    """Phát hiện các câu hỏi tổng quan về cơ cấu / phân loại tài liệu (có những loại sách nào, thống kê các loại sách...)."""
    if not query:
        return False
    norm = normalize_text(query).strip()
    norm_clean = re.sub(
        r'^(o\s+day|o\s+thu\s+vien|cho\s+minh\s+hoi|cho\s+hoi|ban\s+oi|xin\s+hoi|ad\s+cho\s+hoi|thu\s+vien\s+qnu|thu\s+vien)\s*',
        '',
        norm
    ).strip()
    norm_clean = re.sub(
        r'\s*(trong\s+thu\s+vien|o\s+thu\s+vien|o\s+day|qnu|a|ha|nhi|vay|ban)\s*$',
        '',
        norm_clean
    ).strip()

    specific_subjects = [
        'toan', 'ly', 'hoa', 'sinh', 'tin hoc', 'cntt', 'kinh te', 'luat',
        'van', 'su', 'dia', 'tieng anh', 'trung quoc', 'python', 'java', 'ai', 'tri tue nhan tao'
    ]
    for subj in specific_subjects:
        if re.search(r'\b' + subj + r'\b', norm_clean):
            return False

    patterns = [
        r'^(co\s+)?(nhung|cac)?\s*loai\s+(sach|tai\s+lieu|hoc\s+lieu)\s+(nao|gi)',
        r'^(co\s+)?(cac|nhung)?\s*(the\s+loai|nhom)\s+(sach|tai\s+lieu)\s+(nao|gi)',
        r'^(co\s+)?(nhung|cac)\s+loai\s+(nao|gi)',
        r'^(co\s+)?bao\s+nhieu\s+loai\s+(sach|tai\s+lieu)',
        r'^(co\s+)?may\s+loai\s+(sach|tai\s+lieu)',
        r'^thong\s+ke\s+(cac\s+)?(loai\s+)?(sach|tai\s+lieu)',
        r'^(danh\s+muc|phan\s+loai|co\s+cau)\s+(cac\s+)?(loai\s+)?(sach|tai\s+lieu)',
        r'^(co\s+)?nhung\s+tai\s+lieu\s+(nao|gi)$',
        r'^(co\s+)?nhung\s+sach\s+(nao|gi)$',
        r'^cac\s+loai\s+(sach|tai\s+lieu)(\s+co)?$',
        r'^(co\s+)?the\s+loai\s+sach\s+(nao|gi)',
    ]
    for p in patterns:
        if re.search(p, norm_clean):
            return True

    if re.search(
        r'(co\s+nhung\s+loai\s+sach|co\s+cac\s+loai\s+sach|co\s+nhung\s+loai\s+tai\s+lieu|thu\s+vien\s+co\s+nhung\s+loai|cac\s+loai\s+sach\s+trong\s+thu\s+vien|thong\s+ke\s+cac\s+loai\s+sach|thong\s+ke\s+loai\s+sach|co\s+nhung\s+loai\s+sach\s+nao)',
        norm
    ):
        return True

    return False


DOCUMENT_OVERVIEW_SUGGESTIONS = [
    {"label": "Tìm sách về chủ đề: Trí tuệ nhân tạo", "query": "tìm sách về Trí tuệ nhân tạo"},
    {"label": "Tìm sách theo ngành: Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
    {"label": "Tìm sách theo ngành: Quản trị kinh doanh", "query": "tìm sách ngành Quản trị kinh doanh"},
    {"label": "Tìm sách của tác giả: Nguyễn Nhật Ánh", "query": "sách của tác giả Nguyễn Nhật Ánh"},
]


def build_document_types_overview_answer(documents) -> str:
    """Tạo báo cáo thống kê tổng quan cơ cấu tài liệu trong Thư viện QNU."""
    total_docs = len(documents) if documents else 37559
    dt_counts = {}
    if documents:
        for d in documents:
            dt = (d.get("doc_type") or "Khác").strip()
            dt_counts[dt] = dt_counts.get(dt, 0) + 1

    giao_trinh = dt_counts.get("Giáo trình", 483)
    tham_khao = dt_counts.get("Tài liệu tham khảo", 872)
    kiem_dinh = dt_counts.get("Kiểm định", 54)
    chinh = dt_counts.get("Tài liệu chính", 15)
    khac = total_docs - (giao_trinh + tham_khao + kiem_dinh + chinh)

    answer = (
        f"📊 **Tổng quan cơ cấu tài liệu Thư viện Trường Đại học Quy Nhơn**\n\n"
        f"Hiện tại, hệ thống Thư viện QNU đang quản lý và phục vụ **{total_docs:,} đầu tài liệu** (sách, giáo trình, học liệu). "
        f"Các tài liệu được phân loại theo mục đích và mức độ quan tâm như sau:\n\n"
        f"1. 📙 **Giáo trình ({giao_trinh:,} đầu sách)**:\n"
        f"   • Phục vụ trực tiếp cho các học phần giảng dạy và học tập theo khung chương trình đào tạo của trường.\n\n"
        f"2. 📘 **Tài liệu tham khảo ({tham_khao:,} đầu sách)**:\n"
        f"   • Nguồn tư liệu mở rộng, hỗ trợ sinh viên và giảng viên làm tiểu luận, nghiên cứu chuyên sâu, đồ án tốt nghiệp.\n\n"
        f"3. 📗 **Tài liệu kiểm định ({kiem_dinh:,} bộ tài liệu)**:\n"
        f"   • Các tài liệu phục vụ công tác kiểm định chất lượng chương trình đào tạo của nhà trường.\n\n"
        f"4. 📕 **Tài liệu chính & Bài giảng ({chinh:,} tài liệu)**:\n"
        f"   • Tài liệu giảng dạy trọng tâm của giảng viên nhà trường.\n\n"
        f"5. 📚 **Sách chuyên ngành & học liệu thư viện khác ({khac:,} đầu sách)**:\n"
        f"   • Bao quát hơn 50 ngành đào tạo (Sư phạm, Ngôn ngữ, Kinh tế - Luật, Công nghệ thông tin, Kỹ thuật - Công nghệ, Khoa học tự nhiên & Xã hội...).\n\n"
        f"💡 *Bạn có thể tra cứu theo tên sách, tác giả, hoặc chọn các nút gợi ý bên dưới để xem sách cụ thể.*"
    )
    return answer


def search_by_identifier(query: str, documents: list) -> list[dict]:
    """
    Tra cứu chính xác tài liệu theo định danh kỹ thuật:
    - URL tài liệu số (https://lib.qnu.edu.vn/...id=79775)
    - ID tài liệu số (id=79775 hoặc số id trong URL)
    - Mã ItemId (ItemId 25384)
    - Số ĐKCB / Mã xếp giá (VVG01191.0010, VVD...)
    - Mã ISBN (020$a)
    """
    if not query:
        return []
    q = query.strip()

    # 1. Trích xuất URL
    url_match = re.search(r'(https?://[^\s]+)', q)
    target_url = url_match.group(1).rstrip('.,;!?') if url_match else None

    # 2. Trích xuất ID trong URL tài liệu số (id=XXXXX hoặc TraCuuTaiLieuSo...id=XXXXX)
    url_id_match = re.search(r'(?:id=|[&?]id=)(\d+)', q, re.IGNORECASE)
    target_url_id = url_id_match.group(1) if url_id_match else None

    # 3. ItemId / Mã tài liệu
    item_id_match = re.search(r'(?:itemid|mã tài liệu|ma tai lieu|mã sách|ma sach)\s*[:=]?\s*(\d+)', q, re.IGNORECASE)
    target_item_id = item_id_match.group(1) if item_id_match else None

    # 4. Mã ĐKCB (Mã xếp giá, vd: VVG01191.0010, VVD.011789...)
    dkcb_match = re.search(r'\b([A-Z0-9]{2,10}(?:\.[0-9]{3,6})+)\b', q)
    target_dkcb = dkcb_match.group(1) if dkcb_match else None

    # 5. Mã ISBN
    isbn_match = re.search(r'(?:isbn\s*[:=]?\s*)?([0-9Xx\-]{10,17})', q, re.IGNORECASE)
    clean_digits = re.sub(r'[^0-9Xx]', '', isbn_match.group(1)) if isbn_match else ''
    target_isbn = clean_digits if len(clean_digits) in (10, 13) else None

    if not (target_url or target_url_id or target_item_id or target_dkcb or target_isbn):
        return []

    results = []
    for idx, d in enumerate(documents):
        link = d.get('link', '') or ''
        doc_id = str(d.get('doc_id', '') or '')
        isbn = re.sub(r'[^0-9Xx]', '', str(d.get('isbn', '') or ''))

        # Khớp URL đầy đủ
        if target_url and target_url.lower() in link.lower():
            results.append({'doc': d, 'score': 100.0, 'index': idx, 'match_type': 'url'})
            continue

        # Khớp ID tài liệu số trong link
        if target_url_id and (f'id={target_url_id}' in link or f'={target_url_id}' in link):
            results.append({'doc': d, 'score': 95.0, 'index': idx, 'match_type': 'url_id'})
            continue

        # Khớp ItemId
        if target_item_id and target_item_id == doc_id:
            results.append({'doc': d, 'score': 90.0, 'index': idx, 'match_type': 'item_id'})
            continue

        # Khớp ISBN
        if target_isbn and target_isbn == isbn:
            results.append({'doc': d, 'score': 85.0, 'index': idx, 'match_type': 'isbn'})
            continue

        # Khớp ĐKCB
        if target_dkcb:
            locs = d.get('locations', [])
            matched_dkcb = any(
                target_dkcb.lower() in acc.lower()
                for l in locs
                for acc in l.get('accession_numbers', [])
            )
            if matched_dkcb:
                results.append({'doc': d, 'score': 80.0, 'index': idx, 'match_type': 'dkcb'})
                continue

    return results


def build_identifier_answer(doc: dict, match_type: str) -> str:
    title = doc.get("title", "")
    author = doc.get("author", "")
    year = doc.get("year", "")
    publisher = doc.get("publisher", "")
    doc_type = doc.get("doc_type", "Sách")
    link = doc.get("link", "")
    major = doc.get("major", "")
    location = doc.get("location", "")
    abstract = doc.get("abstract", "")

    type_desc = {
        "url": "đường dẫn tài liệu số",
        "url_id": "mã tài liệu số",
        "item_id": "mã ItemId",
        "dkcb": "mã ĐKCB (xếp giá)",
        "isbn": "mã ISBN",
    }.get(match_type, "thông tin định danh")

    answer = f"Mình đã tìm thấy chính xác tài liệu theo **{type_desc}** trong cơ sở dữ liệu Thư viện QNU:\n\n" \
             f"📖 **{title}**\n"
    if author and author not in ('Unknown', 'N/A'):
        answer += f"- **Tác giả / Trách nhiệm**: {author}\n"
    if publisher and publisher not in ('Unknown', 'N/A'):
        pub_line = publisher + (f" ({year})" if year and year != 'N/A' else "")
        answer += f"- **Nhà xuất bản**: {pub_line}\n"
    if major:
        answer += f"- **Chuyên ngành**: {major}\n"
    if doc_type:
        answer += f"- **Loại tài liệu**: {doc_type}\n"
    if link:
        first_link = link.split('|')[0].strip()
        answer += f"- **Xem tài liệu số**: [{first_link}]({first_link})\n"
    if location:
        answer += f"- **Vị trí trong kho**: {location}\n"
    if abstract:
        answer += f"\n📝 **Tóm tắt nội dung**:\n{abstract[:400]}...\n"

    answer += f"\n*Bạn có thể mở thẻ tài liệu bên dưới để xem toàn bộ thông số nghiệp vụ.*"
    return answer


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
    tree_path = Path("Data") / "_khoa_nganh_tree.json"
    file_mtime = tree_path.stat().st_mtime if tree_path.exists() else 0.0
    if _FACULTIES_CACHE is not None and (now - _FACULTIES_CACHE_TS) < _FACULTIES_CACHE_TTL and file_mtime <= _FACULTIES_CACHE_TS:
        return _FACULTIES_CACHE
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

def infer_doc_type_from_meta(m: dict) -> str:
    return infer_doc_type_from_doc(m)

def build_source_item(meta_item: dict) -> dict:
    """Tạo đối tượng BookSource hoàn chỉnh gửi về client với đầy đủ các trường dữ liệu MARC21."""
    m = meta_item.get("metadata", meta_item)

    # Locations list
    locations_raw = m.get("locations", [])
    locations_list = locations_raw if isinstance(locations_raw, list) else []

    # Co-authors
    co_authors_raw = m.get("co_authors", [])
    if isinstance(co_authors_raw, list):
        co_authors_list = co_authors_raw
    elif isinstance(co_authors_raw, str) and co_authors_raw.strip():
        co_authors_list = [c.strip() for c in re.split(r'[,;|]', co_authors_raw) if c.strip()]
    else:
        co_authors_list = []

    # Keywords
    keywords_raw = m.get("keywords", [])
    if isinstance(keywords_raw, list):
        keywords_list = keywords_raw
    elif isinstance(keywords_raw, str) and keywords_raw.strip():
        keywords_list = [k.strip() for k in re.split(r'[,;|]', keywords_raw) if k.strip()]
    else:
        keywords_list = []

    return {
        "doc_id": str(m.get("doc_id", "") or "").strip(),
        "title": clean_display_value(m.get("title", ""), 300) or m.get("title", ""),
        "subtitle": clean_display_value(m.get("subtitle", ""), 300),
        "bilingual_title": clean_display_value(m.get("bilingual_title", ""), 300),
        "part_number": clean_display_value(m.get("part_number", ""), 100),
        "part_name": clean_display_value(m.get("part_name", ""), 200),
        "author": clean_display_value(m.get("author", ""), 180),
        "author_100": clean_display_value(m.get("author_100", ""), 180),
        "responsibility": clean_display_value(m.get("responsibility", ""), 300),
        "co_authors": co_authors_list,
        "role": clean_display_value(m.get("role", ""), 100),
        "year": clean_display_value(m.get("year", ""), 40),
        "publisher": clean_display_value(m.get("publisher", ""), 160),
        "place": clean_display_value(m.get("place", ""), 160),
        "edition": clean_display_value(m.get("edition", ""), 60),
        "major": clean_display_value(m.get("major", ""), 140),
        "doc_type": clean_display_value(infer_doc_type_from_meta(m), 80),
        "format": clean_display_value(m.get("format", "Sách"), 80),
        "subject": clean_display_value(m.get("subject", ""), 180),
        "keywords": keywords_list,
        "ddc": clean_display_value(m.get("ddc", ""), 80),
        "ddc_cutter": clean_display_value(m.get("ddc_cutter", ""), 80),
        "udc": clean_display_value(m.get("udc", ""), 80),
        "other_class": clean_display_value(m.get("other_class", ""), 80),
        "isbn": clean_display_value(m.get("isbn", ""), 80),
        "price": clean_display_value(m.get("price", ""), 80),
        "language": clean_display_value(m.get("language", ""), 80),
        "physical": clean_display_value(m.get("physical", ""), 120),
        "illustration": clean_display_value(m.get("illustration", ""), 120),
        "size": clean_display_value(m.get("size", ""), 80),
        "series": clean_display_value(m.get("series", ""), 200),
        "notes": clean_display_value(m.get("notes", ""), 300),
        "abstract": clean_display_value(m.get("abstract", ""), 3000),
        "copyright": clean_display_value(m.get("copyright", ""), 300),
        "location": clean_display_value(m.get("location", ""), 3000),
        "locations": locations_list,
        "copies": int(m.get("copies", 1)),
        "link": clean_display_value(m.get("link", ""), 500),
        "source": clean_display_value(m.get("source", ""), 260),
        "course": clean_display_value(m.get("course", ""), 140),
        "is_pdf": False,
        "pdf_id": "",
        "file_name": "",
        "display_name": "",
    }


def _build_extractive_fallback_answer(query: str, search_results: list) -> str:
    if not search_results:
        return f"Hiện chưa tìm thấy tài liệu phù hợp trong thư viện để trả lời câu hỏi: **{query}**."
    lines = [f"Dưới đây là thông tin trích xuất từ tài liệu thư viện liên quan đến câu hỏi **\"{query}\"**:\n"]
    has_abstract = False
    for i, r in enumerate(search_results[:3], 1):
        doc = r.get("doc", r)
        title = doc.get("title", "")
        author = doc.get("author", "")
        abstract = (doc.get("abstract", "") or "").strip()
        author_str = f" (*Tác giả: {author}*)" if author and author not in ("Unknown", "N/A") else ""
        if abstract:
            has_abstract = True
            lines.append(f"**{i}. {title}**{author_str}:\n> {abstract}\n")
        else:
            lines.append(f"**{i}. {title}**{author_str}\n")
    if not has_abstract:
        lines.append("\n*(Ghi chú: Bạn có thể xem chi tiết tài liệu trong các thẻ tham khảo đính kèm)*")
    return "\n".join(lines).strip()


ATTR_CONFIG = {
    "year": {
        "trailing": [
            r"(?:(?:duoc\s+)?(?:xuat\s+ban|phat\s+hanh|in|ra\s+doi|tai\s+ban)\s+(?:vao\s+)?(?:nam\s+)?(?:nao|bao\s+nhieu|may|\d{4}))",
            r"(?:(?:vao\s+)?nam\s+(?:nao|bao\s+nhieu|may))",
            r"(?:xuat\s+ban\s+(?:nam\s+)?(?:nao|bao\s+nhieu|may))",
            r"(?:(?<!viet\s)nam\s+xuat\s+ban\s+(?:la\s+)?(?:bao\s+nhieu|nao|may|\?))",
            r"(?:nam\s+phat\s+hanh\s+(?:la\s+)?(?:bao\s+nhieu|nao|may|\?))",
            r"(?:ra\s+nam\s+nao)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?<!viet\s)(?:nam\s+xuat\s+ban|nam\s+phat\s+hanh)(?:\s+(?:cua|la))?)\s*",
        ]
    },
    "location": {
        "trailing": [
            r"(?:(?:hien\s+tai\s+)?(?:con\s+o|(?<!viet\s)nam\s+o|o\s+tai|o)\s+(?:phong\s+(?:doc\s+)?nao|kho\s+nao|dau|tang\s+may))",
            r"(?:(?:muon|tim|lay)\s+(?:sach\s+)?(?:o\s+phong\s+nao|o\s+kho\s+nao|o\s+dau|o\s+tang\s+may))",
            r"(?:(?:hien\s+tai\s+)?(?:con\s+o|o)\s+phong\s+.*)",
            r"(?:con\s+o\s+phong\s+giao\s+trinh\s+khong|con\s+o\s+phong\s+.*?(?:khong|ko))",
            r"(?:len\s+phong\s+nao|len\s+tang\s+may|(?<!viet\s)nam\s+o\s+dau|de\s+o\s+dau|o\s+dau)",
            r"(?:o\s+phong\s+(?:doc\s+)?nao)",
            r"(?:o\s+kho\s+nao|kho\s+nao)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:phong\s+nao\s+co|o\s+phong\s+nao\s+co|vi\s+tri\s+(?:cua)?|dia\s+diem\s+(?:cua)?))\s*",
        ]
    },
    "author": {
        "trailing": [
            r"(?:(?:do\s+)?ai\s+(?:la\s+tac\s+gia|viet|sang\s+tac|bien\s+soan|chu\s+bien|dich))",
            r"(?:tac\s+gia\s+(?:la\s+ai|ten\s+gi|nguoi\s+nao))",
            r"(?:chu\s+bien\s+la\s+ai|ai\s+chu\s+bien)",
            r"(?:dich\s+gia\s+la\s+ai|nguoi\s+dich|ai\s+dich)",
            r"(?:tac\s+gia)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:ai\s+la\s+tac\s+gia(?:\s+cua)?|tac\s+gia(?:\s+cua)?|ai\s+viet|ai\s+bien\s+soan|ai\s+chu\s+bien))\s*",
        ]
    },
    "publisher": {
        "trailing": [
            r"(?:(?:do\s+|cua\s+)?(?:nha\s+xuat\s+ban|nxb)\s+(?:nao|gi)(?:\s+(?:in|phat\s+hanh|xuat\s+ban))?)",
            r"(?:thuoc\s+nha\s+xuat\s+ban\s+nao|do\s+ai\s+xuat\s+ban|noi\s+xuat\s+ban)",
            r"(?:nha\s+xuat\s+ban\s+(?:nao|gi))",
            r"(?:cua\s+nha\s+xuat\s+ban\s+nao)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:nha\s+xuat\s+ban|nxb)(?:\s+(?:cua|la))?)\s*",
        ]
    },
    "pages": {
        "trailing": [
            r"(?:co\s+bao\s+nhieu\s+trang|bao\s+nhieu\s+trang|may\s+trang|day\s+bao\s+nhieu|so\s+trang|kho\s+sach)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:so\s+trang(?:\s+cua)?|do\s+day(?:\s+cua)?))\s*",
        ]
    },
    "abstract": {
        "trailing": [
            r"(?:co\s+tom\s+tat\s+(?:khong|ko)|tom\s+tat\s+noi\s+dung|noi\s+dung\s+.*(?:noi\s+ve\s+gi|la\s+gi)|noi\s+ve\s+(?:cai\s+)?gi|gioi\s+thieu\s+ve|so\s+luoc\s+ve)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:tom\s+tat(?:\s+noi\s+dung)?(?:\s+cua)?|noi\s+dung\s+chinh(?:\s+cua)?))\s*",
        ]
    },
    "borrowable": {
        "trailing": [
            r"(?:(?:co\s+)?(?:duoc\s+)?muon\s+ve(?:\s+nha)?(?:\s+duoc)?\s*(?:khong|ko))",
            r"(?:co\s+cho\s+muon\s+ve(?:\s+nha)?|co\s+duoc\s+muon\s+ve)",
        ],
        "leading": []
    },
    "copies": {
        "trailing": [
            r"(?:co\s+bao\s+nhieu\s+ban|co\s+may\s+cuon|may\s+ban|so\s+luong\s+ban)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:so\s+luong\s+ban(?:\s+cua)?))\s*",
        ]
    },
    "major": {
        "trailing": [
            r"(?:danh\s+cho\s+chuyen\s+nganh\s+nao|chuyen\s+nganh\s+nao|nganh\s+nao\s+hoc|nganh\s+dao\s+tao\s+nao|thuoc\s+nganh\s+nao|phuc\s+vu\s+nganh\s+nao|mon\s+hoc\s+nao)",
        ],
        "leading": []
    },
    "ddc": {
        "trailing": [
            r"(?:ma\s+ddc|so\s+phan\s+loai|ky\s+hieu\s+phan\s+loai|ma\s+phan\s+loai|ma\s+xep\s+gia|call\s+number)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:ma\s+ddc|so\s+phan\s+loai)(?:\s+cua)?)\s*",
        ]
    },
    "isbn": {
        "trailing": [
            r"(?:ma\s+isbn|chi\s+so\s+isbn|isbn|so\s+dang\s+ky\s+ca\s+biet|dkcb|ma\s+tai\s+lieu|ma\s+vach)",
        ],
        "leading": [
            r"^(?:(?:cho\s+(?:minh|em|toi)\s+hoi\s+)?(?:ma\s+isbn|isbn|so\s+dang\s+ky\s+ca\s+biet)(?:\s+cua)?)\s*",
        ]
    },
    "language": {
        "trailing": [
            r"(?:viet\s+bang\s+tieng\s+gi|ngon\s+ngu\s+gi|tieng\s+gi|bang\s+tieng\s+gi|tieng\s+viet\s+hay\s+tieng\s+anh)",
        ],
        "leading": []
    },
    "edition": {
        "trailing": [
            r"(?:tai\s+ban\s+lan\s+may|lan\s+xuat\s+ban\s+thu\s+may|tai\s+ban\s+lan\s+thu\s*\d*|phien\s+ban\s+nao)",
        ],
        "leading": []
    },
    "overview": {
        "trailing": [],
        "leading": [
            r"^(?:cho\s+minh\s+xin\s+thong\s+tin\s+cuon|thong\s+tin\s+chi\s+tiet\s+cuon|thong\s+tin\s+ve\s+cuon)\s*",
        ]
    }
}

ATTR_PRECEDENCE = [
    "overview",
    "borrowable",
    "copies",
    "pages",
    "publisher",
    "year",
    "location",
    "author",
    "major",
    "abstract",
    "ddc",
    "isbn",
    "language",
    "edition"
]

def extract_book_title_and_attr(query: str):
    if not query:
        return None, None

    # Nếu câu hỏi có tính chất tìm kiếm theo độ mới / gần đây / cũ nhất hoặc danh sách chung -> không phải hỏi thuộc tính sách đơn lẻ
    if detect_recency_preference(query) or detect_oldest_preference(query) or is_generic_book_listing_query(query):
        return None, None

    q_norm = normalize_text(query).strip()

    # 1. Quoted title check
    m_quote = re.search(r'["\']([^"\']+)["\']', query)
    quoted_title = m_quote.group(1).strip() if m_quote else None

    # 2. Strip leading polite greetings
    leading_greetings = [
        r"^(?:da\s+)?(?:em\s+)?(?:chao\s+(?:anh\s+chi\s+)?(?:ad|bot|ban|admin|thu\s+vien|moi\s+nguoi)?\s*(?:a|nhe|nha)?\s*[,.-]?\s*)+",
        r"^(?:xin\s+chao|chao\s+ban|chao\s+bot|chao\s+ad)\s*[,.-]?\s*",
        r"^(?:lam\s+on\s+)?(?:cho\s+(?:em|minh|toi)\s+hoi|ad\s+cho\s+(?:em|minh)\s+hoi|bot\s+cho\s+(?:em|minh)\s+hoi)\s*",
        r"^(?:em|minh|toi)\s+(?:muon\s+hoi|can\s+hoi|muon\s+biet|can\s+biet|muon\s+tim|can\s+tim)\s*",
    ]
    cand = q_norm
    for lg in leading_greetings:
        cand = re.sub(lg, "", cand).strip()

    # 3. Detect attribute
    matched_attr = None

    # Check leading attribute patterns first
    for attr in ATTR_PRECEDENCE:
        cfg = ATTR_CONFIG[attr]
        for pat in cfg["leading"]:
            m = re.search(pat, cand)
            if m:
                matched_attr = attr
                cand = cand[m.end():].strip()
                break
        if matched_attr:
            break

    # If not leading, check trailing attribute patterns
    if not matched_attr:
        for attr in ATTR_PRECEDENCE:
            cfg = ATTR_CONFIG[attr]
            for pat in cfg["trailing"]:
                m = re.search(r"\b" + pat + r"\b", cand)
                if m:
                    matched_attr = attr
                    # Strip from start of trailing pattern
                    matches = list(re.finditer(r"(?:\s+|,)\b" + pat + r"\b.*$", cand))
                    if matches:
                        cand = cand[:matches[-1].start()].strip()
                    break
            if matched_attr:
                break

    if not matched_attr:
        return None, None

    if quoted_title:
        return quoted_title, matched_attr

    # 4. Clean candidate title
    cand = re.sub(r"[?.,!;:\'\"()]", " ", cand).strip()
    cand = re.sub(r"\s+", " ", cand).strip()

    # Strip book prefix words at beginning
    cand = re.sub(r"^(?:cuon\s+sach|quyen\s+sach|cuon|quyen|sach|giao\s+trinh|tai\s+lieu|an\s+pham|luan\s+van)\s+", "", cand).strip()
    cand = re.sub(r"^(?:ve|cua)\s+", "", cand).strip()
    cand = re.sub(r"^(?:cuon\s+sach|quyen\s+sach|cuon|quyen|sach|giao\s+trinh|tai\s+lieu)\s+", "", cand).strip()

    # Strip trailing particles, question words, and fillers
    trailing_junk = [
        r"\s+\b(?:la\s+gi|la\s+bao\s+nhieu|la\s+ai|la\s+sach\s+gi|la\s+nam\s+nao)\s*$",
        r"\s+\b(?:vay\s+bot|nhi\s+bot|ha\s+bot|ad\s+oi|bot\s+oi|ban\s+oi|thu\s+vien\s+oi)\s*$",
        r"\s+\b(?:vay\s+ad|vay\s+ban|ha\s+ad|ha\s+ban|nhe\s+bot|nha\s+bot)\s*$",
        r"\s+\b(?:vay\s+a|the\s+a|khong\s+a|ko\s+a|dung\s+khong|phai\s+khong)\s*$",
        r"\s+\b(?:vay|nhi|ha|the|a|nhe|nha|khong|ko|bot|ad|ban)\s*$",
    ]
    for _ in range(3):
        for tj in trailing_junk:
            cand = re.sub(tj, "", cand).strip()

    cand = re.sub(r"\s+", " ", cand).strip()
    cand_sub = re.sub(r"^(?:nhung|cac|co\s+(?:nhung|cac)?)\s+", "", cand).strip()
    if cand in GENERIC_TITLES or cand_sub in GENERIC_TITLES or len(cand) <= 1 or len(cand_sub) <= 1:
        return None, None

    return cand, matched_attr

def restore_candidate_diacritics(candidate_norm: str, original_query: str) -> str:
    if not candidate_norm or not original_query:
        return candidate_norm
    norm_q = normalize_text(original_query)
    words_orig = original_query.split()
    words_norm = norm_q.split()
    cand_words = candidate_norm.split()
    for i in range(len(words_norm) - len(cand_words) + 1):
        if words_norm[i:i+len(cand_words)] == cand_words:
            slice_orig = " ".join(words_orig[i:i+len(cand_words)])
            return slice_orig.strip(' ?.,!;:"\'')
    return candidate_norm.title()

def handle_author_location_inquiry(query: str, r=None):
    if not query:
        return None
    r = r or retriever
    return handle_author_inquiry(
        query,
        retriever=r,
        author_index=author_index,
        build_source_item_fn=build_source_item
    )

def find_matching_books(candidate: str, hits, all_documents):
    if not candidate:
        return []
    cand_norm = candidate.strip().lower()
    cand_words = set(cand_norm.split())

    # 1. Exact match on title across all docs (ignore trailing dots)
    exact_matches = [d for d in all_documents if (d.get('_title_norm') or normalize_text(d.get('title', ''))).strip().rstrip('.') == cand_norm]
    if exact_matches:
        def get_year_val(d):
            try:
                return int(d.get('year', 0))
            except Exception:
                return 0
        exact_matches.sort(key=get_year_val, reverse=True)
        return exact_matches

    # 2. Word-boundary phrase match in title (e.g. "giai tich 1" in "Giao trinh giai tich 1")
    escaped = re.escape(cand_norm)
    phrase_matches = [d for d in all_documents if re.search(r"\b" + escaped + r"\b", (d.get('_title_norm') or normalize_text(d.get('title', ''))))]
    if phrase_matches:
        phrase_matches.sort(key=lambda d: len(d.get('title', '')))
        return phrase_matches

    # 3. Check BM25 hits where all candidate words are present
    bm25_matches = []
    for h in hits:
        t_norm = h["doc"].get('_title_norm') or normalize_text(h["doc"].get("title", "")).strip().lower()
        t_words = set(t_norm.split())
        if cand_words.issubset(t_words):
            bm25_matches.append(h["doc"])
    if bm25_matches:
        return bm25_matches

    # 4. High score BM25 hit with >= 75% overlap
    if hits and hits[0].get("score", 0) >= 3.0:
        top_doc = hits[0]["doc"]
        t_norm = top_doc.get('_title_norm') or normalize_text(top_doc.get("title", "")).strip().lower()
        t_words = set(t_norm.split())
        overlap = len(cand_words & t_words) / max(len(cand_words), 1)
        if overlap >= 0.75:
            return [top_doc]

    return []

def find_best_book_match(candidate: str, hits, all_documents):
    matches = find_matching_books(candidate, hits, all_documents)
    if matches:
        return {"doc": matches[0], "score": 100.0}
    return None

def handle_book_attribute_inquiry(query: str, retriever_instance=None):
    if not query:
        return None
    r = retriever_instance or retriever
    if not r:
        return None

    # 0. Check author location inquiry first ("sách của tác giả X ở phòng nào")
    author_loc_res = handle_author_location_inquiry(query, r)
    if author_loc_res is not None:
        return author_loc_res

    # 1. Extract candidate title and attribute
    candidate, matched_attr = extract_book_title_and_attr(query)
    if not matched_attr or not candidate or len(candidate) < 2 or candidate in ("tac gia", "sach", "phong", "doc"):
        return None

    # 2. Search matching documents
    hits = r.bm25.search(candidate, top_k=30)
    matching_docs = find_matching_books(candidate, hits, r.bm25.documents)

    # 3. If NOT found: Return polite not-found message with ZERO sources
    if not matching_docs:
        cand_display = restore_candidate_diacritics(candidate, query)
        ans = (
            f"Hiện tại Thư viện QNU không tìm thấy thông tin hoặc tài liệu về cuốn sách **\"{cand_display}\"** trong hệ thống lưu trữ. 🤔\n\n"
            f"Bạn vui lòng kiểm tra lại chính xác tên sách, hoặc thử tra cứu theo tên tác giả / chuyên ngành liên quan nhé!"
        )
        return {
            "answer": ans,
            "sources": [],
            "total_sources": 0,
            "total_found": 0,
            "matched_attr": matched_attr,
            "found": False,
            "candidate": cand_display
        }

    # Best / primary doc (most recent / highest match)
    doc = matching_docs[0]
    title = doc.get("title", "").rstrip('.')
    author = doc.get("author", "")
    year = doc.get("year", "")
    loc = doc.get("location", "")
    pub = doc.get("publisher", "")
    place = doc.get("place", "")
    ddc = doc.get("ddc", "")
    cutter = doc.get("ddc_cutter", "")
    abstract = doc.get("abstract", "")
    major = doc.get("major", "")
    isbn = doc.get("isbn", "")
    phys = doc.get("physical", "")
    size = doc.get("size", "")
    resp = doc.get("responsibility", "")
    lang = doc.get("language", "")
    edit = doc.get("edition", "")

    def valid(val):
        return val and str(val).strip() not in ("Unknown", "N/A", "None", "Không xác định", "")

    author_str = f" (*tác giả: {author}*)" if valid(author) else ""

    # Group years if multiple editions
    years_map = {}
    for d in matching_docs:
        y = str(d.get('year', '')).strip()
        if valid(y):
            if y not in years_map:
                years_map[y] = []
            years_map[y].append(d)

    # 4. Build answer based on attribute
    if matched_attr == "year":
        if len(years_map) > 1:
            lines = [f"Tài liệu **{title}** trong Thư viện QNU hiện có các ấn bản/tập được xuất bản vào các năm sau:\n"]
            sorted_years = sorted(years_map.keys(), key=lambda y: int(y) if y.isdigit() else 0, reverse=True)
            for y in sorted_years:
                docs = years_map[y]
                pubs = list(dict.fromkeys(d.get('publisher') for d in docs if valid(d.get('publisher'))))
                auths = list(dict.fromkeys(d.get('author') for d in docs if valid(d.get('author'))))
                info_parts = []
                if pubs: info_parts.append(f"NXB {', '.join(pubs[:2])}")
                if auths: info_parts.append(f"tác giả: {', '.join(auths[:2])}")
                detail_str = f" ({'; '.join(info_parts)})" if info_parts else ""
                lines.append(f"• **Năm {y}**: {len(docs)} bản/tập lưu{detail_str}")
            lines.append("\n*(Bạn có thể xem chi tiết từng ấn bản và vị trí phòng đọc trong các thẻ tài liệu đính kèm bên dưới)*")
            ans = "\n".join(lines)
        elif valid(year):
            pub_info = f" do **{pub}** phát hành" if valid(pub) else ""
            ans = f"Cuốn sách **{title}**{author_str} được xuất bản vào năm **{year}**{pub_info}."
        else:
            ans = f"Cuốn sách **{title}**{author_str} chưa có thông tin năm xuất bản cụ thể trong hồ sơ lưu trữ."

    elif matched_attr == "location":
        all_rooms = []
        for d in matching_docs:
            l_val = d.get('location', '')
            if valid(l_val):
                for rm in re.split(r'[;\n]+', l_val):
                    rm_clean = re.sub(r':.*$', '', rm).strip()
                    if rm_clean and rm_clean not in all_rooms:
                        all_rooms.append(rm_clean)
        if all_rooms:
            ans = f"Để tra cứu hoặc mượn cuốn sách **{title}**{author_str}, bạn có thể đến các phòng sau trong thư viện:\n\n"
            for rm in all_rooms:
                ans += f"• **{rm}**\n"
            ans += "\n*(Lưu ý: Bạn đọc có thể làm thủ tục mượn về nhà tại Phòng Mượn/Phòng Giáo trình, hoặc đọc tại chỗ tại Phòng Đọc/Phòng Ngoại văn)*"
        else:
            ans = f"Cuốn sách **{title}**{author_str} hiện chưa cập nhật vị trí phòng đọc cụ thể trong hệ thống thư mục."

    elif matched_attr == "author":
        parts = []
        if valid(author): parts.append(f"**{author}**")
        if valid(resp) and resp != author: parts.append(f"({resp})")
        if parts:
            ans = f"Tác giả của cuốn sách **{title}** là {' '.join(parts)}" + (f" (xuất bản năm {year})." if valid(year) else ".")
        else:
            ans = f"Cuốn sách **{title}** hiện chưa có thông tin tác giả cụ thể trong hệ thống."

    elif matched_attr == "publisher":
        pubs = list(dict.fromkeys(d.get('publisher') for d in matching_docs if valid(d.get('publisher'))))
        if len(pubs) > 1:
            ans = f"Tài liệu **{title}** tại Thư viện QNU được phát hành bởi các Nhà xuất bản sau:\n\n"
            for p in pubs:
                ans += f"• **{p}**\n"
        elif valid(pub):
            place_info = f" tại **{place}**" if valid(place) else ""
            year_info = f" vào năm **{year}**" if valid(year) else ""
            ans = f"Cuốn sách **{title}**{author_str} do Nhà xuất bản **{pub}** phát hành{place_info}{year_info}."
        else:
            ans = f"Cuốn sách **{title}**{author_str} chưa cập nhật thông tin nhà xuất bản trong hệ thống."

    elif matched_attr == "ddc":
        if valid(ddc):
            cutter_info = f", số định danh xếp giá là **{cutter}**" if valid(cutter) else ""
            ans = f"Cuốn sách **{title}**{author_str} có mã phân loại DDC là **{ddc}**{cutter_info}."
        else:
            ans = f"Cuốn sách **{title}**{author_str} chưa có mã phân loại DDC trong hệ thống."

    elif matched_attr == "isbn":
        if valid(isbn):
            ans = f"Mã số ISBN của cuốn sách **{title}**{author_str} là **{isbn}**."
        else:
            ans = f"Cuốn sách **{title}**{author_str} chưa có thông tin mã ISBN trong hồ sơ biên mục."

    elif matched_attr == "pages":
        if valid(phys):
            size_info = f", khổ sách **{size}**" if valid(size) else ""
            ans = f"Cuốn sách **{title}**{author_str} có độ dài / mô tả vật lý: **{phys}**{size_info}."
        else:
            ans = f"Cuốn sách **{title}**{author_str} chưa cập nhật thông tin số trang trong hệ thống."

    elif matched_attr == "abstract":
        if valid(abstract):
            ans = f"Dưới đây là tóm tắt nội dung cuốn sách **{title}**{author_str}:\n\n> {abstract.strip()}"
        else:
            ans = (
                f"Cuốn sách **{title}**{author_str} (xuất bản năm {year or 'N/A'}) hiện chưa có tóm tắt nội dung số hóa trong hệ thống thư viện.\n\n"
                f"Bạn có thể đến trực tiếp tại **{loc or 'Thư viện'}** để tra cứu và đọc toàn văn tài liệu."
            )

    elif matched_attr == "major":
        if valid(major):
            ans = f"Cuốn sách / giáo trình **{title}**{author_str} phục vụ cho chuyên ngành đào tạo: **{major}**."
        else:
            ans = f"Tài liệu **{title}**{author_str} hiện chưa phân loại cụ thể theo ngành đào tạo trong hệ thống."

    elif matched_attr == "borrowable":
        all_locs = " ; ".join(d.get('location', '') for d in matching_docs if valid(d.get('location')))
        is_home = any(rm in all_locs for rm in ("Phòng Mượn", "Phòng Giáo trình", "Giáo trình", "Mượn"))
        if is_home:
            ans = (
                f"Cuốn sách **{title}**{author_str} hiện có bản lưu tại các phòng: **{all_locs[:100]}...**\n\n"
                f"✅ **Bạn hoàn toàn có thể làm thủ tục mượn về nhà** (tại Phòng Mượn / Phòng Giáo trình) theo quy định mượn trả tài liệu của Thư viện QNU."
            )
        elif valid(all_locs):
            ans = (
                f"Cuốn sách **{title}**{author_str} hiện được lưu giữ tại: **{all_locs}**.\n\n"
                f"⚠️ Tài liệu này **chỉ phục vụ đọc tại chỗ** tại phòng đọc thư viện, không áp dụng mượn về nhà."
            )
        else:
            ans = f"Cuốn sách **{title}**{author_str} hiện chưa cập nhật quy chế mượn cụ thể do chưa có vị trí kho."

    elif matched_attr == "copies":
        norm_t = normalize_text(title).strip()
        copy_count = sum(1 for d in r.bm25.documents if (d.get('_title_norm') or normalize_text(d.get('title', ''))).strip().rstrip('.') == norm_t)
        copy_count = max(copy_count, len(matching_docs))
        loc_str = f" tại **{loc}**" if valid(loc) else ""
        ans = f"Hiện tại trong hệ thống Thư viện QNU có khoảng **{copy_count} bản/tập lưu** của cuốn sách **{title}**{author_str}{loc_str}."

    elif matched_attr == "language":
        lang_map = {"vie": "Tiếng Việt", "eng": "Tiếng Anh", "fra": "Tiếng Pháp", "rus": "Tiếng Nga", "chi": "Tiếng Trung"}
        lang_name = lang_map.get(lang.lower(), lang) if valid(lang) else ("Tiếng Anh" if any(w in normalize_text(title) for w in ["english", "skills", "grammar", "physics", "practice"]) else "Tiếng Việt")
        ans = f"Cuốn sách **{title}**{author_str} được viết bằng ngôn ngữ: **{lang_name}**."

    elif matched_attr == "edition":
        if valid(edit):
            ans = f"Cuốn sách **{title}**{author_str} là bản: **{edit}** (xuất bản năm {year or 'N/A'})."
        else:
            ans = f"Cuốn sách **{title}**{author_str} không ghi nhận thông tin tái bản cụ thể (xuất bản năm {year or 'N/A'})."

    elif matched_attr == "overview":
        ans = (
            f"📖 **Thông tin tài liệu: {title}**\n\n"
            f"• **Tác giả:** {author or 'Chưa rõ'}\n"
            f"• **Năm xuất bản:** {year or 'Chưa rõ'}\n"
            f"• **Nhà xuất bản:** {pub or 'Chưa rõ'}\n"
            f"• **Vị trí / Phòng đọc:** {loc or 'Chưa cập nhật'}\n"
            f"• **Mã phân loại DDC:** {ddc or 'Chưa có'}\n"
            f"• **Ngành đào tạo:** {major or 'Đại cương / Đa ngành'}\n"
        )
        if valid(abstract):
            ans += f"\n> **Tóm tắt:** {abstract[:300]}..."
    else:
        ans = f"Thông tin về cuốn sách **{title}**{author_str}: Vị trí: {loc or 'N/A'}, Xuất bản: {year or 'N/A'}."

    srcs = [build_source_item({"metadata": d}) for d in matching_docs[:min(len(matching_docs), 20)]]
    return {
        "answer": ans,
        "sources": srcs,
        "total_sources": len(srcs),
        "total_found": len(matching_docs),
        "matched_attr": matched_attr,
        "found": True,
        "doc": doc,
        "candidate": candidate
    }


def handle_library_policy_inquiry(query: str):
    if not query:
        return None
    q = normalize_text(query).strip()

    # 1. Giờ mở cửa / Lịch phục vụ / Thời gian làm việc
    hours_patterns = [
        r"\b(?:gio\s+mo\s+cua|mo\s+cua\s+luc\s+may\s+gio|may\s+gio\s+mo\s+cua|may\s+gio\s+dong\s+cua|thoi\s+gian\s+mo\s+cua|thoi\s+gian\s+phuc\s+vu|lich\s+phuc\s+vu|lich\s+mo\s+cua|gio\s+lam\s+viec|mo\s+cua\s+ngay\s+nao|co\s+mo\s+cua\s+thu\s+7|co\s+mo\s+cua\s+chu\s+nhat|cuoi\s+tuan\s+co\s+mo\s+cua)\b",
        r"\b(?:khi\s+nao\s+thu\s+vien\s+mo\s+cua|thu\s+vien\s+mo\s+cua\s+khi\s+nao|thu\s+vien\s+mo\s+cua\s+luc\s+nao|thu\s+vien\s+may\s+gio\s+mo|thu\s+vien\s+mo\s+cua\s+luc\s+may\s+gio)\b",
    ]
    if any(re.search(p, q) for p in hours_patterns):
        ans = (
            "🕒 **Thời gian mở cửa & Phục vụ của Thư viện QNU:**\n\n"
            "Thư viện Trường Đại học Quy Nhơn phục vụ bạn đọc từ **Thứ Hai đến Thứ Sáu** hàng tuần:\n"
            "• **Buổi sáng:** **07:00 – 11:30**\n"
            "• **Buổi chiều:** **13:30 – 17:00**\n\n"
            "📌 **Lưu ý:**\n"
            "• **Thứ Bảy, Chủ Nhật và các ngày Lễ, Tết:** Thư viện tạm nghỉ phục vụ trực tiếp tại quầy.\n"
            "• Hệ thống **Tra cứu danh mục & Tài nguyên số trực tuyến** luôn hoạt động 24/7 để phục vụ bạn đọc tự học và nghiên cứu từ xa."
        )
        return {
            "answer": ans,
            "sources": [],
            "total_sources": 0,
            "total_found": 0,
            "follow_up_suggestions": [
                {"label": "Quy định mượn sách về nhà", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                {"label": "Thủ tục làm thẻ thư viện", "query": "thủ tục làm thẻ thư viện thế nào?"},
                {"label": "Tìm giáo trình CNTT", "query": "giáo trình ngành công nghệ thông tin"}
            ]
        }

    # 2. Quy định mượn / trả / gia hạn sách / số lượng sách được mượn
    borrow_patterns = [
        r"\b(?:muon\s+toi\s+da\s+may\s+cuon|muon\s+duoc\s+may\s+cuon|duoc\s+muon\s+bao\s+nhieu\s+cuon|so\s+luong\s+sach\s+duoc\s+muon|quy\s+dinh\s+muon\s+tra|thoi\s+han\s+muon\s+sach|muon\s+sach\s+duoc\s+bao\s+lau|muon\s+sach\s+bao\s+nhieu\s+ngay|gia\s+han\s+sach|tra\s+sach\s+qua\s+han|phi\s+phat\s+tra\s+muon)\b",
        r"\b(?:sinh\s+vien\s+duoc\s+muon\s+.*(?:sach|ve\s+nha)|muon\s+ve\s+nha\s+toi\s+da)\b",
    ]
    if any(re.search(p, q) for p in borrow_patterns) and not re.search(r"\b(?:cuon\s+|quyen\s+|sach\s+|giao\s+trinh\s+)[a-z0-9]", q):
        ans = (
            "📖 **Quy định Mượn - Trả sách tại Thư viện QNU:**\n\n"
            "1. **Số lượng tài liệu được mượn về nhà:**\n"
            "   • **Sinh viên / Học viên:** Được mượn tối đa **3 – 5 cuốn sách/giáo trình** cùng một lúc.\n"
            "   • **Giảng viên / Cán bộ nghiên cứu:** Được mượn tài liệu theo định mức phục vụ công tác giảng dạy & NCKH.\n\n"
            "2. **Thời hạn mượn:**\n"
            "   • **Giáo trình / Sách tham khảo:** Mượn từ **14 đến 30 ngày** (tùy học phần môn học).\n"
            "   • **Gia hạn:** Bạn có thể xin gia hạn thêm 01 lần (từ 7 - 14 ngày) nếu tài liệu chưa có bạn đọc khác đăng ký mượn trước.\n\n"
            "3. **Tài liệu đọc tại chỗ (không mang về nhà):**\n"
            "   • Luận văn thạc sĩ, luận án tiến sĩ, khóa luận tốt nghiệp.\n"
            "   • Báo, tạp chí chuyên ngành, từ điển tra cứu nhanh và sách độc bản lưu trữ tại kho."
        )
        return {
            "answer": ans,
            "sources": [],
            "total_sources": 0,
            "total_found": 0,
            "follow_up_suggestions": [
                {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa lúc mấy giờ?"},
                {"label": "Thủ tục làm thẻ thư viện", "query": "thủ tục làm thẻ thư viện thế nào?"},
                {"label": "Tìm sách theo chuyên ngành", "query": "giáo trình ngành công nghệ thông tin"}
            ]
        }

    # 3. Làm thẻ thư viện / Tài khoản bạn đọc
    card_patterns = [
        r"\b(?:lam\s+the\s+thu\s+vien|thu\s+tuc\s+lam\s+the|dang\s+ky\s+the\s+thu\s+vien|mat\s+the\s+thu\s+vien|the\s+sinh\s+vien\s+co\s+dung\s+lam\s+the\s+thu\s+vien|kich\s+hoat\s+the|tai\s+khoan\s+thu\s+vien)\b"
    ]
    if any(re.search(p, q) for p in card_patterns):
        ans = (
            "💳 **Thủ tục Thẻ Thư viện & Tài khoản Bạn đọc QNU:**\n\n"
            "• **Sử dụng Thẻ sinh viên:** Thư viện QNU áp dụng tích hợp trực tiếp mã vạch trên **Thẻ sinh viên** làm thẻ thư viện. Bạn chỉ cần mang theo thẻ sinh viên còn hiệu lực để quét mã khi vào cửa và mượn tài liệu.\n"
            "• **Kích hoạt / Quên mật khẩu tài khoản:** Bạn có thể liên hệ trực tiếp tại Quầy thông tin (Tầng 1 Thư viện) hoặc dùng email sinh viên (@qnu.edu.vn) để được hỗ trợ cấp lại.\n"
            "• **Trường hợp mất thẻ:** Bạn đọc vui lòng báo ngay cho cán bộ thư viện để tạm khóa quyền mượn sách nhằm bảo vệ tài khoản cá nhân."
        )
        return {
            "answer": ans,
            "sources": [],
            "total_sources": 0,
            "total_found": 0,
            "follow_up_suggestions": [
                {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa lúc mấy giờ?"},
                {"label": "Quy định mượn sách về nhà", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                {"label": "Gợi ý tài liệu phổ biến", "query": "tìm sách về trí tuệ nhân tạo"}
            ]
        }

    # 4. Nội quy phòng đọc / Không gian thư viện
    rules_patterns = [
        r"\b(?:noi\s+quy\s+thu\s+vien|noi\s+quy\s+phong\s+doc|co\s+duoc\s+mang\s+do\s+an|mang\s+ba\s+lo\s+vao|co\s+wifi\s+khong|khong\s+gian\s+tu\s+hoc)\b"
    ]
    if any(re.search(p, q) for p in rules_patterns):
        ans = (
            "🏛️ **Nội quy Chung tại Thư viện QNU:**\n\n"
            "1. **Tác phong & Giữ gìn trật tự:**\n"
            "   • Trang phục lịch sự, văn minh theo quy định của Nhà trường.\n"
            "   • Đi nhẹ, nói khẽ, chuyển điện thoại sang chế độ rung khi vào các phòng đọc và phòng tự học.\n"
            "2. **Tư trang & Hành lý:**\n"
            "   • Bạn đọc vui lòng gửi cặp sách, ba lô tại Tủ đồ cá nhân trước khi vào phòng đọc.\n"
            "   • Không mang đồ ăn, nước uống có màu, kẹo cao su hoặc các chất dễ gây cháy nổ vào thư viện.\n"
            "3. **Bảo quản tài liệu & Trang thiết bị:**\n"
            "   • Không viết, vẽ, gạch chân hoặc làm rách tài liệu.\n"
            "   • Sau khi đọc xong, vui lòng để sách vào xe đẩy trả sách, không tự ý xếp sai vị trí giá sách."
        )
        return {
            "answer": ans,
            "sources": [],
            "total_sources": 0,
            "total_found": 0,
            "follow_up_suggestions": [
                {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa lúc mấy giờ?"},
                {"label": "Quy định mượn sách", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                {"label": "Tìm kiếm tài liệu", "query": "tìm sách về trí tuệ nhân tạo"}
            ]
        }

    return None


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

        hist = get_session_history(req.session_id)

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
                save_session_history(req.session_id, hist)
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

        inherited_ctx = get_inherited_context(hist)

        # ── Conversational clarification / reformulation ("ý là...", "ý mình là...", "không phải, ý là...") ──
        is_clarification, clarified_content = extract_clarification_content(req.query)
        if is_clarification:
            logger.info("Clarification detected: %r -> %r", req.query, clarified_content)
            is_rec = detect_recency_preference(clarified_content)
            is_old = detect_oldest_preference(clarified_content)
            yr, yr_dir = detect_year_constraint(clarified_content)
            dt = detect_doc_type_from_query(clarified_content)
            maj = resolve_canonical_major_from_query(clarified_content)
            inherited_topic = inherited_ctx.get("search_query") or inherited_ctx.get("topic")
            if (is_rec or is_old or yr or dt or maj) and inherited_topic:
                req.query = f"{inherited_topic} {clarified_content}"
            else:
                req.query = clarified_content
            normalized_query = normalize_text(req.query)
            intent = classify_intent(req.query)

        # ── Recency / Oldest sort preference detection ──
        _recency_sort_now = detect_recency_preference(req.query)
        _oldest_sort_now = detect_oldest_preference(req.query)
        _is_refine_now = is_follow_up_refinement(req.query)
        _is_singular_now = is_singular_book_request(req.query)

        # ── Enhanced detection (chạy trước intent checks) ──
        _detected_doc_type = getattr(req, "doc_type", None) or detect_doc_type_from_query(req.query)
        _canonical_major_match = resolve_canonical_major_from_query(req.query)
        if _canonical_major_match:
            _canonical_major_name, _matched_alias = _canonical_major_match
            req.major = _canonical_major_name
            _study_major_now = _canonical_major_name
        else:
            _study_major_raw = detect_study_major_query(req.query)
            if _study_major_raw:
                _study_match = resolve_canonical_major_from_query(_study_major_raw)
                if _study_match:
                    req.major = _study_match[0]
                    _study_major_now = _study_match[0]
                else:
                    _study_major_now = None
            else:
                _study_major_now = None
        _lang_filter_now = detect_language_preference(req.query) if not _canonical_major_match else None
        _year_filter_now, _year_dir_now = detect_year_constraint(req.query)
        _is_pure_yr, _pure_yr_val, _pure_yr_dir = is_pure_year_query(req.query)
        if _is_pure_yr:
            _year_filter_now = _pure_yr_val
            _year_dir_now = _pure_yr_dir
        _title_kw_now = detect_title_contains_keyword(req.query)

        # Kế thừa doc_type từ ngữ cảnh trước nếu câu hiện tại là câu lọc tiếp nối (theo ngành, năm, hoặc filter)
        if not _detected_doc_type and inherited_ctx.get("doc_type"):
            if _canonical_major_match or _study_major_now or _year_filter_now or is_follow_up_filter(req.query):
                _detected_doc_type = inherited_ctx["doc_type"]
                logger.info("Inherited doc_type '%s' from previous turn for query %r", _detected_doc_type, req.query)
        _generic_now = bool(re.search(
            r"(?:goi y|vai cuon|mot vai|it cuon)\s*(?:sach)?",
            normalize_text(req.query)
        )) and not req.query.startswith(list_topic_marker) and not bool(re.search(r'\b(python|java|javascript|ai|ml|machine|deep|data|web|lap trinh)\b', normalize_text(req.query)))
        _skip_existence = bool(_study_major_now) or bool(_year_filter_now) or bool(_detected_doc_type) or _recency_sort_now or _oldest_sort_now or _is_refine_now

        # Nếu intent "unknown" mà có enhanced signals hoặc chủ đề kỹ thuật hoặc yêu cầu độ mới hoặc câu tinh chỉnh tiếp nối → treat as catalog
        _is_tech_topic = bool(re.search(r'\b(python|java|javascript|c\+\+|sql|html|css|ai|ml|machine|deep|data|web|lap trinh|kinh te|tri tue nhan tao|lich su|quan tri|kinh doanh)\b', normalize_text(req.query)))
        if intent == "unknown" and (_is_pure_yr or _study_major_now or _lang_filter_now or _year_filter_now or _detected_doc_type or _is_tech_topic or _recency_sort_now or _oldest_sort_now or _is_refine_now or is_follow_up_filter(req.query)):
            intent = "catalog"

        # ── Direct Identifier Search (URL, Link ID, ItemId, ĐKCB, ISBN) ──
        id_results = search_by_identifier(req.query, retriever.bm25.documents)
        if id_results:
            matched_doc = id_results[0]['doc']
            match_type = id_results[0].get('match_type', 'identifier')
            answer_text = build_identifier_answer(matched_doc, match_type)
            source_item = build_source_item({'metadata': matched_doc})
            hist = get_session_history(req.session_id)
            hist.append({
                "query": req.query,
                "answer": answer_text,
                "search_query": matched_doc.get("title", ""),
                "topic": matched_doc.get("title", ""),
                "source_count": 1,
                "shown_titles": [matched_doc.get("title", "")],
            })
            save_session_history(req.session_id, hist)
            _record_query_analytics(req.query, 1)

            clean_a = clean_author_for_suggestion(matched_doc.get("author"))
            id_follow_ups = []
            if clean_a:
                id_follow_ups.append({"label": f"Tìm sách của tác giả: {clean_a}", "query": f"sách của tác giả {clean_a}"})
            if matched_doc.get("major"):
                clean_m = clean_suggestion_topic(matched_doc.get("major"))
                if clean_m:
                    id_follow_ups.append({"label": f"Tìm sách theo ngành: {clean_m}", "query": f"tìm sách ngành {clean_m}"})
            elif matched_doc.get("subject"):
                clean_s = clean_suggestion_topic(matched_doc.get("subject"))
                if clean_s:
                    id_follow_ups.append({"label": f"Tìm sách về chủ đề: {clean_s}", "query": f"tìm sách về {clean_s}"})
            id_follow_ups.append({"label": "Xem tài liệu tham khảo", "query": "Tìm tài liệu tham khảo"})
            if len(id_follow_ups) < 3:
                id_follow_ups.extend(random_no_result_suggestions(3 - len(id_follow_ups)))

            return ChatResponse(
                answer=answer_text,
                sources=[source_item],
                total_sources=1,
                total_found=1,
                current_document=source_item,
                follow_up_suggestions=id_follow_ups,
            )

        # ── Multi-turn Conversational Book Resolution (ordinal & anaphoric) ──
        _target_book = None
        _target_idx = None
        _cand_sources = inherited_ctx.get("last_sources") or inherited_ctx.get("list_sources") or []

        if _cand_sources:
            # Check ordinal reference ("cuốn đầu tiên", "cuốn thứ 2", "cuốn cuối cùng")
            _target_idx, _target_book = detect_ordinal_reference(req.query, _cand_sources)
            if _target_book is None and _target_idx is not None and inherited_ctx.get("list_sources"):
                _target_idx, _target_book = detect_ordinal_reference(req.query, inherited_ctx["list_sources"])

            # Check anaphoric reference ("cuốn này", "sách đó", "nó")
            if _target_book is None:
                _target_book = detect_anaphoric_reference(req.query, _cand_sources)

        if _target_book is not None:
            _attr = detect_attribute_question(req.query)
            if not _attr and (_target_idx is not None or any(p in normalize_text(req.query) for p in ("chi tiet", "cho xem", "xem cuon", "thong tin", "cuon"))):
                _attr = "detail"

            if _attr:
                _ans_text = build_book_attribute_answer(_target_book, _attr)
                _src_item = build_source_item({"metadata": _target_book.get("metadata", _target_book)})
                _follow_ups = build_book_follow_ups(_target_book, _attr)

                hist.append({
                    "query": req.query,
                    "answer": _ans_text,
                    "search_query": _target_book.get("title", ""),
                    "topic": _target_book.get("title", ""),
                    "doc_type": _target_book.get("doc_type") or inherited_ctx.get("doc_type"),
                    "major": _target_book.get("major") or inherited_ctx.get("major"),
                    "author": _target_book.get("author") or inherited_ctx.get("author"),
                    "source_count": 1,
                    "shown_titles": [_target_book.get("title", "")],
                    "sources": [_src_item],
                })
                save_session_history(req.session_id, hist)
                _record_query_analytics(req.query, 1)

                return ChatResponse(
                    answer=_ans_text,
                    sources=[_src_item],
                    total_sources=1,
                    total_found=1,
                    current_document=_src_item,
                    follow_up_suggestions=_follow_ups,
                )

        # ── Comprehensive Book Attribute Inquiry (Phòng đọc, Tác giả, Năm XB, NXB, Mã DDC, Tóm tắt, Số trang, Chuyên ngành, Mượn về...) ──
        book_inquiry_res = handle_book_attribute_inquiry(req.query, retriever)
        if book_inquiry_res is not None:
            ans = book_inquiry_res["answer"]
            srcs = book_inquiry_res["sources"]
            total_f = book_inquiry_res["total_found"]
            curr_doc = srcs[0] if srcs else None

            _inquiry_author = book_inquiry_res.get("author_display") or (book_inquiry_res.get("doc", {}).get("author") if book_inquiry_res.get("doc") else "")
            _inquiry_topic = f"Tác giả {_inquiry_author}" if book_inquiry_res.get("author_display") else (book_inquiry_res.get("doc", {}).get("title", "") if book_inquiry_res.get("doc") else "")
            hist = get_session_history(req.session_id)
            hist.append({
                "query": req.query,
                "answer": ans,
                "search_query": book_inquiry_res.get("doc", {}).get("title", req.query),
                "topic": _inquiry_topic,
                "author": _inquiry_author,
                "source_count": total_f,
                "shown_titles": [s.get("title", "") for s in srcs[:20]] if srcs else ([book_inquiry_res["doc"].get("title", "")] if book_inquiry_res.get("doc") else []),
                "sources": srcs[:10] if srcs else ([book_inquiry_res["doc"]] if book_inquiry_res.get("doc") else []),
            })
            save_session_history(req.session_id, hist)
            _record_query_analytics(req.query, len(srcs))

            if book_inquiry_res["found"] and book_inquiry_res.get("doc"):
                doc_found = book_inquiry_res["doc"]
                t_name = (doc_found.get("title") or "").strip()
                a_name = (doc_found.get("author") or "").strip()
                s_name = (doc_found.get("subject") or "").strip()
                m_name = (doc_found.get("major") or "").strip()
                matched_attr = book_inquiry_res.get("matched_attr", "")

                clean_a = clean_author_for_suggestion(a_name)

                follow_ups = []
                # Nếu câu hỏi vừa hỏi chưa phải là tóm tắt thì gợi ý tóm tắt
                if matched_attr != "abstract":
                    follow_ups.append({
                        "label": "Xem tóm tắt nội dung sách",
                        "query": f"tóm tắt nội dung cuốn {t_name}" if t_name else "tóm tắt nội dung sách"
                    })
                # Nếu câu hỏi vừa hỏi chưa phải là vị trí phòng đọc thì gợi ý phòng đọc
                if matched_attr != "location":
                    follow_ups.append({
                        "label": "Vị trí phòng đọc của sách",
                        "query": f"cuốn {t_name} ở phòng nào" if t_name else "sách này để ở phòng nào"
                    })
                # Gợi ý sách cùng tác giả (rõ ràng, không cắt cụt chữ)
                if clean_a:
                    follow_ups.append({
                        "label": f"Tìm sách của tác giả: {clean_a}",
                        "query": f"sách của tác giả {clean_a}"
                    })
                # Gợi ý theo ngành nếu có
                if m_name and m_name not in ("Unknown", "N/A") and len(follow_ups) < 4:
                    clean_m = clean_suggestion_topic(m_name)
                    if clean_m:
                        follow_ups.append({
                            "label": f"Tìm sách theo ngành: {clean_m}",
                            "query": f"tìm sách ngành {clean_m}"
                        })
                # Gợi ý theo chủ đề nếu có
                elif s_name and s_name.lower() not in ("unknown", "n/a", "aspose", "tai lieu so") and len(follow_ups) < 4:
                    clean_s = clean_suggestion_topic(s_name, max_words=5)
                    if clean_s:
                        follow_ups.append({
                            "label": f"Tìm sách về chủ đề: {clean_s}",
                            "query": f"tìm sách về {clean_s}"
                        })

                # Đảm bảo đủ ít nhất 3 gợi ý phong phú
                if len(follow_ups) < 3:
                    follow_ups.extend(random_no_result_suggestions(3 - len(follow_ups)))
            else:
                if book_inquiry_res.get("matched_attr") == "author_location" or book_inquiry_res.get("author_display"):
                    follow_ups = random_no_result_suggestions(3, focus="author")
                else:
                    follow_ups = random_no_result_suggestions(3)

            return ChatResponse(
                answer=ans,
                sources=srcs,
                total_sources=len(srcs),
                total_found=total_f,
                current_document=curr_doc,
                follow_up_suggestions=follow_ups,
            )

        # ── Library Policy, Opening Hours, Card & Borrowing Inquiries ──
        policy_res = handle_library_policy_inquiry(req.query)
        if policy_res is not None:
            ans = policy_res["answer"]
            f_ups = policy_res.get("follow_up_suggestions", [])
            hist = get_session_history(req.session_id)
            hist.append({
                "query": req.query,
                "answer": ans,
                "search_query": req.query,
                "topic": "Nội quy & Dịch vụ Thư viện",
                "source_count": 0,
                "shown_titles": [],
            })
            save_session_history(req.session_id, hist)
            _record_query_analytics(req.query, 0)
            return ChatResponse(
                answer=ans,
                sources=[],
                total_sources=0,
                total_found=0,
                current_document=None,
                follow_up_suggestions=f_ups,
            )

        # ── Document Types Overview / Statistical queries ──
        if detect_document_types_overview_query(req.query):
            overview_answer = build_document_types_overview_answer(retriever.bm25.documents)
            hist = get_session_history(req.session_id)
            hist.append({
                "query": req.query,
                "answer": overview_answer,
                "search_query": "thống kê loại sách",
                "topic": "Tổng quan loại tài liệu",
                "source_count": 0,
                "shown_titles": [],
            })
            save_session_history(req.session_id, hist)
            _record_query_analytics(req.query, 0)
            return ChatResponse(
                answer=overview_answer,
                sources=[],
                total_sources=0,
                total_found=0,
                current_document=None,
                follow_up_suggestions=DOCUMENT_OVERVIEW_SUGGESTIONS,
            )

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
                        {"label": "Tìm sách về chủ đề: Trí tuệ nhân tạo", "query": "tìm sách về trí tuệ nhân tạo"},
                        {"label": "Tìm sách theo ngành: Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
                        {"label": "Tìm sách về chủ đề: Kinh tế vi mô", "query": "tìm sách Kinh tế vi mô"},
                        {"label": "Tìm sách của tác giả: Nguyễn Nhật Ánh", "query": "sách của tác giả Nguyễn Nhật Ánh"},
                    ], min(3, 4)))

        if detect_exhaustion_query(req.query):
            entry_count = len(hist)
            if entry_count == 0:
                answer_text = "Bạn chưa hỏi chủ đề hay tác giả nào trước đó. Hãy thử hỏi cụ thể hơn, ví dụ: **tìm sách về trí tuệ nhân tạo**, **sách của Nguyễn Nhật Ánh**."
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=0, current_document=None, follow_up_suggestions=[])

            last_entry = hist[-1]
            last_author = last_entry.get("author") or inherited_ctx.get("author")
            last_topic = last_entry.get("topic") or inherited_ctx.get("topic")
            last_sources = last_entry.get("sources") or inherited_ctx.get("last_sources") or []
            last_source_count = last_entry.get("source_count", len(last_sources))

            # If user was inquiring about an author
            if last_author:
                if last_source_count <= len(last_sources) or len(last_sources) <= 5:
                    count_display = len(last_sources) or last_source_count
                    answer_text = (
                        f"Thư viện QNU hiện tại chỉ có **{count_display} tài liệu** "
                        f"của tác giả **{last_author}** (đã hiển thị ở trên). "
                        f"Không còn tài liệu nào khác của tác giả này trong hệ thống lưu trữ.\n\n"
                        f"Bạn có thể tra cứu theo tên tác giả khác hoặc theo chủ đề/chuyên ngành nhé!"
                    )
                    follow_ups = [
                        {"label": "Gợi ý tác giả nổi tiếng", "query": "sách của Tô Hoài"},
                        {"label": "Tìm sách văn học phổ biến", "query": "tìm sách văn học Việt Nam"},
                    ]
                    return ChatResponse(
                        answer=answer_text, sources=[], total_sources=0, total_found=count_display,
                        current_document=None, follow_up_suggestions=follow_ups
                    )

            # Check shown titles across session
            shown_titles = set()
            for h in hist:
                for title in h.get("shown_titles", []):
                    if title:
                        shown_titles.add(title.lower().strip())
            shown_count = len(shown_titles)

            last_search = last_entry.get("search_query", "")
            query_for_count = last_topic or last_search or normalized_query
            total_count = retriever.count_matching(query_for_count)

            if total_count <= 0 or (last_source_count > 0 and last_source_count <= len(last_sources)):
                topic_desc = f"về **{last_topic}**" if last_topic else "này"
                answer_text = f"Hiện tại thư viện đã hiển thị tất cả **{len(last_sources) or last_source_count} tài liệu** liên quan đến chủ đề {topic_desc} rồi. Không còn tài liệu nào khác trong hệ thống lưu trữ."
                return ChatResponse(
                    answer=answer_text, sources=[], total_sources=0, total_found=len(last_sources) or last_source_count,
                    current_document=None, follow_up_suggestions=[
                        {"label": "Tìm sách theo chuyên ngành", "query": "tìm sách ngành Công nghệ thông tin"},
                        {"label": "Gợi ý tài liệu phổ biến", "query": "Gợi ý tài liệu phổ biến"}
                    ]
                )
            elif total_count <= 5:
                answer_text = f"Chỉ còn **{total_count} tài liệu** liên quan đến chủ đề này trong kho lưu trữ. Bạn có muốn xem tiếp không? Hãy nói **liệt kê thêm** để mình hiển thị tiếp."
            else:
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
                        {"label": "Tìm sách về chủ đề: Trí tuệ nhân tạo", "query": "tìm sách về Trí tuệ nhân tạo"},
                        {"label": "Tìm sách theo ngành: Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
                        {"label": "Tìm sách của tác giả: Nguyễn Nhật Ánh", "query": "sách của tác giả Nguyễn Nhật Ánh"},
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
                    {"label": "Tìm sách về chủ đề: Trí tuệ nhân tạo", "query": "tìm sách về Trí tuệ nhân tạo"},
                    {"label": "Tìm sách theo ngành: Tâm lý học", "query": "tìm sách ngành Tâm lý học"},
                    {"label": "Tìm sách của tác giả: Nguyễn Nhật Ánh", "query": "sách của tác giả Nguyễn Nhật Ánh"},
                    {"label": "Tìm sách theo ngành: Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
                    {"label": "Tìm sách về chủ đề: Kinh tế vi mô", "query": "tìm sách Kinh tế vi mô"},
                    {"label": "Tìm sách của tác giả: Stuart J. Russell", "query": "sách của tác giả Stuart J. Russell"},
                    {"label": "Tìm sách theo ngành: Giáo dục học", "query": "tìm sách ngành Giáo dục học"},
                    {"label": "Tìm sách theo ngành: Kỹ thuật xây dựng", "query": "tìm sách ngành Kỹ thuật xây dựng"},
                    {"label": "Tìm sách của tác giả: Lê Xuân Vinh", "query": "sách của tác giả Lê Xuân Vinh"},
                    {"label": "Tìm sách theo ngành: Sư phạm Toán học", "query": "tìm sách ngành Sư phạm Toán học"},
                    {"label": "Tìm sách về chủ đề: Lịch sử Việt Nam", "query": "tìm sách Lịch sử Việt Nam"},
                    {"label": "Tìm sách theo ngành: Quản trị kinh doanh", "query": "tìm sách ngành Quản trị kinh doanh"},
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
        catalog_mode = is_list_request or bool(_is_pure_yr) or (not request_content and intent != "unknown") or bool(_study_major_now and not request_content) or bool(_detected_doc_type and not request_content)

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

        _skip_existence = bool(_study_major_now) or bool(_year_filter_now) or bool(_detected_doc_type)

        # ── Title-contains-keyword: "có chữ Python trong tên không?" ──
        if _title_kw_now:
            _tk_count = retriever.count_title_match(_title_kw_now)
            hist.append({"query": req.query, "search_query": _title_kw_now, "topic": _title_kw_now, "source_count": _tk_count, "shown_titles": []})
            save_session_history(req.session_id, hist)
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
                save_session_history(req.session_id, hist)
                answer_text = f"Có, thư viện hiện có tài liệu về **{topic_label}**. Bạn có muốn mình liệt kê chi tiết ra không?"
                return ChatResponse(answer=answer_text, sources=[], total_sources=0, total_found=count, current_document=None,
                    follow_up_suggestions=[
                        {"label": "Liệt kê chi tiết", "query": f"{list_topic_marker}{topic_label}"},
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
                _canonical_major_match = resolve_canonical_major_from_query(req.query)
                if _canonical_major_match:
                    req.major = _canonical_major_match[0]
                else:
                    _study_major = detect_study_major_query(req.query)
                    if _study_major:
                        _study_match = resolve_canonical_major_from_query(_study_major)
                        if _study_match:
                            req.major = _study_match[0]
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
                for _d in new_documents:
                    _d['_title_norm'] = normalize_text(_d.get('title', '')).strip().lower()
                retriever.bm25.documents = new_documents
                retriever.bm25._build_index()
                global _semantic_engine
                _semantic_engine = None
                _build_topic_cache()
                _search_cache.invalidate()
            except Exception:
                pass

        _query_for_search = clean_conversational_query(req.query)
        if _is_refine_now and hist:
            inherited_search = inherited_ctx.get("search_query") or inherited_ctx.get("topic")
            if inherited_search:
                search_query = inherited_search
            else:
                search_query = resolve_search_query(_query_for_search or req.query, hist, catalog_mode=catalog_mode)
        else:
            search_query = resolve_search_query(_query_for_search or req.query, hist, catalog_mode=catalog_mode)

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

        # Nếu có detected_doc_type và không phải Kiểm định, loại bỏ từ khóa loại tài liệu khỏi search_query
        if _detected_doc_type and _detected_doc_type != "Kiểm định":
            _clean_dt = clean_query_doc_type_terms(search_query, _detected_doc_type)
            if _clean_dt and len(_clean_dt.strip()) >= 2:
                search_query = _clean_dt.strip()

        # Nếu có yêu cầu sắp xếp theo mới nhất / cũ nhất, loại bỏ các cụm từ đó khỏi search_query
        if (_recency_sort_now or _oldest_sort_now) and search_query:
            _clean_rec = strip_temporal_sorting_phrases(search_query)
            if _clean_rec and len(_clean_rec.strip()) >= 2:
                search_query = _clean_rec.strip()

        _query_for_author = strip_temporal_sorting_phrases(req.query)
        _extracted_author = extract_author_name(_query_for_author) if detect_author_query(_query_for_author) else ""
        _extracted_pub = extract_publisher_name(_query_for_author) if detect_publisher_query(_query_for_author) else ""
        _prefer_strict = bool(_extracted_author or _extracted_pub)
        cache_key = f"catalog:{search_query}:{_prefer_strict}:{req.major or ''}:{_detected_doc_type or ''}" if catalog_mode else None
        use_cache = catalog_mode and not req.document_id and not req.query.startswith("__SUGGEST")
        search_results = _search_cache.get(cache_key) if use_cache else None
        retrieval_k = len(retriever.bm25.documents) if catalog_mode else 30

        if search_results is not None:
            logger.info("Search cache HIT (%d docs, key='%s')", len(search_results), search_query)
        elif _is_pure_yr and _pure_yr_val:
            year_matches = [
                {"doc": d, "score": 100.0}
                for d in retriever.bm25.documents
                if doc_matches_year(d, _pure_yr_val, _pure_yr_dir or "exact")
                and (not _detected_doc_type or doc_matches_target_doc_type(d, _detected_doc_type))
            ]
            search_results = year_matches
            catalog_mode = True
            if use_cache and search_results:
                _search_cache.set(cache_key, search_results)
        elif _detected_doc_type == "Kiểm định":
            _canonical_nganh_kd = get_canonical_major_aliases(req.major) if req.major else set()
            kd_results = []
            for d in retriever.bm25.documents:
                if doc_matches_target_doc_type(d, "Kiểm định"):
                    is_official_kd = (str(d.get("doc_type") or "").strip().lower() == "kiểm định")
                    base_score = 500.0 if is_official_kd else 150.0
                    if _canonical_nganh_kd:
                        if doc_matches_canonical_major(str(d.get("major", "")), _canonical_nganh_kd):
                            kd_results.append({"doc": d, "score": base_score + 200.0})
                    else:
                        kd_results.append({"doc": d, "score": base_score})
            if kd_results:
                search_results = rerank_results_for_query(
                    req.query, kd_results, prefer_strict=False, target_doc_type="Kiểm định"
                )
                if use_cache and search_results:
                    _search_cache.set(cache_key, search_results)
            else:
                search_results = []
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

            _rerank_q = search_query if search_query else req.query
            search_results = rerank_results_for_query(_rerank_q, search_results, prefer_strict=_prefer_strict, target_doc_type=_detected_doc_type)
            # PDF page collapsing disabled
            if use_cache and search_results:
                _search_cache.set(cache_key, search_results)

        # Kiểm tra: nếu là author query mà không có kết quả nào khớp tác giả → ép fallback
        if search_results and _prefer_strict and catalog_mode:
            _author_name_check = extract_author_name(strip_temporal_sorting_phrases(req.query)) if detect_author_query(req.query) else ""
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
            _canonical_nganh = get_canonical_major_aliases(req.major)
            if not _canonical_nganh:
                major_norm = _normalize_topic_token(req.major)
                if major_norm:
                    _tree = _load_faculties_tree()
                    for f in (_tree.get("faculties") or []):
                        for ng in (f.get("nganh") or []):
                            ng_norm = _normalize_topic_token(ng.get("nganh", ""))
                            if ng_norm and (ng_norm == major_norm or major_norm in ng_norm or ng_norm in major_norm):
                                _canonical_nganh.add(ng_norm)

            if _canonical_nganh:
                def _doc_matches_major(doc_major_raw: str) -> bool:
                    return doc_matches_canonical_major(doc_major_raw, _canonical_nganh)

                filtered_results = [r for r in search_results if _doc_matches_major(str(r.get("doc", r).get("major", "")))]

                # Kiểm tra xem câu hỏi có chứa từ khóa chuyên môn hẹp ngoài tên ngành hay không
                _is_general_major_query = False
                if _canonical_major_match:
                    _matched_alias = _canonical_major_match[1]
                    _q_clean = normalize_text(req.query).strip()
                    _q_sub = re.sub(r'\b' + re.escape(_matched_alias) + r'\b', '', _q_clean)
                    _q_sub = re.sub(r'\b(tim|sach|tai lieu|giao trinh|thuoc|cua|nganh|chuyen nganh|cho|sinh vien|minh|toi|em|ban|dang|hoc|co|khong|a|nhi|vay|nhe|can|gi|nao|xin|cho minh)\b', '', _q_sub).strip()
                    if _detected_doc_type:
                        _q_sub = clean_query_doc_type_terms(_q_sub, _detected_doc_type)
                    _q_sub = re.sub(r'\b(phuc vu|chat luong|tham khao|luan van|bai tap|bai giang)\b', '', _q_sub).strip()
                    _is_general_major_query = (len(_q_sub) < 3)
                elif req.query.startswith(list_topic_marker):
                    _is_general_major_query = True

                all_major_docs = [
                    {"doc": d, "score": 100.0}
                    for d in retriever.bm25.documents
                    if _doc_matches_major(str(d.get("major", "")))
                ]

                if _detected_doc_type:
                    all_major_dt = [
                        d for d in all_major_docs
                        if doc_matches_target_doc_type(d.get("doc", d), _detected_doc_type)
                    ]
                    if all_major_dt:
                        all_major_docs = all_major_dt

                if _is_general_major_query or not filtered_results:
                    search_results = all_major_docs
                else:
                    if _detected_doc_type:
                        dt_filtered = [
                            r for r in filtered_results
                            if doc_matches_target_doc_type(r.get("doc", r), _detected_doc_type)
                        ]
                        if dt_filtered:
                            search_results = dt_filtered
                        else:
                            search_results = filtered_results
                    else:
                        search_results = filtered_results

        # ── Language filter: "tiếng Anh" / "Song ngữ" ──
        if _lang_filter_now and search_results:
            search_results = filter_by_language(search_results, _lang_filter_now)

        # ── Year filter: "xuất bản từ 2023", "sau 2022" ──
        if _year_filter_now and search_results:
            search_results = filter_by_year(search_results, _year_filter_now, _year_dir_now or "exact")

        # ── Recency / Oldest sort: "năm gần đây nhất", "mới nhất", "cũ nhất" ──
        if _recency_sort_now and search_results:
            search_results = sort_results_by_year(search_results, descending=True)
        elif _oldest_sort_now and search_results:
            search_results = sort_results_by_year(search_results, descending=False)

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
            _refine_topic = clean_conversational_query(hist[-1].get("query", "")) if (_is_refine_now and hist) else ""
            _clean_cur_topic = clean_conversational_query(req.query)
            if _is_pure_yr and _pure_yr_val:
                if _pure_yr_dir == "after":
                    _disp_topic = f"xuất bản từ năm {_pure_yr_val}"
                elif _pure_yr_dir == "before":
                    _disp_topic = f"xuất bản trước năm {_pure_yr_val}"
                else:
                    _disp_topic = f"xuất bản năm {_pure_yr_val}"
            else:
                _disp_topic = _refine_topic or (_clean_cur_topic if _clean_cur_topic and len(_clean_cur_topic) >= 3 else "") or ("Trí tuệ nhân tạo" if detect_ai_domain(req.query) else (_restore_topic_diacritics(search_query, req.query) or clean_suggestion_topic(search_query, max_words=5) or search_query))
            answer = build_catalog_answer(
                search_results,
                req.query,
                total_found=total_found,
                doc_type=_detected_doc_type,
                major=req.major,
                is_recency=_recency_sort_now,
                is_oldest=_oldest_sort_now,
                topic_display=_disp_topic,
                is_singular=_is_singular_now,
            )
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
                chain_func_async, _ = build_general_chain_async()
                try:
                    answer = await chain_func_async(context=context, question=req.query, history=hist)
                except Exception as _llm_err:
                    logger.warning("LLM call failed (%s); using extractive fallback", _llm_err)
                    answer = _build_extractive_fallback_answer(req.query, search_results)

        # PDF fallback disabled
        pdf_fallback_used = False

        _top_score = search_results[0].get('score', 0) if search_results else 0
        _no_info_phrases = (
            "tai lieu khong cung cap thong tin",
            "khong tim thay tai lieu",
            "chua tim thay tai lieu",
            "khong co thong tin nay",
            "khong co thong tin trong tai lieu",
            "khong tim thay",
        )
        answer_says_no_info = any(p in normalize_text(answer) for p in _no_info_phrases)
        suppress_sources = not pdf_fallback_used and (
            not search_results
            or (_top_score < 1.0 and not catalog_mode)
            or answer_says_no_info
        )
        if suppress_sources and search_results:
            search_results = []

        seen, sources = set(), []
        current_doc = None
        for meta in ([] if suppress_sources else doc_metadata_list):
            title = meta['metadata']['title']
            author = meta['metadata'].get('author', '')
            year = meta['metadata'].get('year', '')
            source_id = meta['metadata'].get('doc_id') or (str(title).lower().strip(), str(author).lower().strip(), str(year).lower().strip(), id(meta['metadata']))
            # PDF info lookup disabled
            if source_id not in seen:
                seen.add(source_id)
                source_item = build_source_item(meta)
                sources.append(source_item)
                if req.document_id and title.lower() == req.document_id.lower():
                    current_doc = source_item

        # PDF fallback in sources disabled
        topic = ""
        if sources:
            topic = sources[0].get("subject", "") or sources[0].get("title", "")
        entry = {
            "query": req.query,
            "answer": answer,
            "search_query": search_query,
            "topic": topic,
            "doc_type": _detected_doc_type,
            "major": getattr(req, "major", None) or _study_major_now,
            "author": _extracted_author or (sources[0].get("author") if sources else ""),
            "source_count": len(sources),
            "shown_titles": [s.get("title", "") for s in sources[:20]],
            "sources": sources[:10],
        }
        hist.append(entry)
        save_session_history(req.session_id, hist)

        total_found = total_found if total_found > 0 else len(sources)
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
async def chat_stream(req: ChatRequest, fast_request: Request = None):
    async def events():
        _stream_start = time.time()
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
            intent = classify_intent(req.query)
            request_content = detect_content_request(req.query)
            is_list_request = detect_list_request(req.query)
            _rec_pref = detect_recency_preference(req.query)
            _old_pref = detect_oldest_preference(req.query)
            _yr_pref, _ = detect_year_constraint(req.query)
            _dt_pref = detect_doc_type_from_query(req.query)
            _maj_pref = resolve_canonical_major_from_query(req.query)
            _refine_pref = is_follow_up_refinement(req.query) or is_follow_up_filter(req.query)
            _has_catalog_signal = bool(_rec_pref or _old_pref or _yr_pref or _dt_pref or _maj_pref or _refine_pref)
            if intent == "unknown" and _has_catalog_signal:
                intent = "catalog"

            catalog_mode = is_list_request or (not request_content and intent != "unknown") or _has_catalog_signal

            # In catalog, book attribute inquiries, library policy, or chitchat (without LLM), chat() is instant (<20ms)
            _book_attr_hit = handle_book_attribute_inquiry(req.query, retriever)
            if _book_attr_hit is not None or handle_library_policy_inquiry(req.query) is not None or catalog_mode or intent in ("chitchat", "live_info", "clarify") or detect_document_types_overview_query(req.query) or bool(search_by_identifier(req.query, retriever.bm25.documents)):
                response = await chat(req, fast_request=fast_request)
                payload = response.model_dump() if hasattr(response, "model_dump") else response.dict()

                yield _sse_event({"type": "sources", "sources": payload.get("sources", [])})
                yield _sse_event({"type": "total_found", "total_found": payload.get("total_found", 0)})

                answer = payload.get("answer", "")
                for chunk in _stream_chunks(answer, size=40):
                    yield _sse_event({"type": "chunk", "text": chunk + " "})
                    await asyncio.sleep(0.005)

                yield _sse_event({"type": "follow_ups", "follow_up_suggestions": payload.get("follow_up_suggestions", [])})
                yield _sse_event({"type": "done"})
                return

            # ── TRUE LLM STREAMING for Content QA ──
            hist = get_session_history(req.session_id)
            search_query = resolve_search_query(req.query, hist, catalog_mode=False)
            bm25_results = retriever.search_by_query(search_query, top_k=30)
            context, doc_metadata_list = format_docs_with_metadata(bm25_results)

            _top_score = bm25_results[0].get('score', 0) if bm25_results else 0
            if not bm25_results or _top_score < 1.0:
                answer = (
                    f"Mình chưa tìm thấy tài liệu nào liên quan đến **\"{req.query}\"** trong thư viện. 🤔\n\n"
                    f"Bạn có thể thử:\n"
                    f"• **Từ khóa khác** hoặc diễn đạt lại câu hỏi\n"
                    f"• **Tìm theo tên sách**, **tác giả**, hoặc **chủ đề** cụ thể hơn\n\n"
                    f"Ví dụ: *\"Tìm sách về trí tuệ nhân tạo\"*"
                )
                yield _sse_event({"type": "sources", "sources": []})
                yield _sse_event({"type": "total_found", "total_found": 0})
                for chunk in _stream_chunks(answer, size=40):
                    yield _sse_event({"type": "chunk", "text": chunk + " "})
                    await asyncio.sleep(0.005)
                yield _sse_event({"type": "follow_ups", "follow_up_suggestions": random_no_result_suggestions(3)})
                yield _sse_event({"type": "done"})
                return

            # Prepare sources
            seen, sources = set(), []
            for meta in doc_metadata_list:
                title = meta['metadata']['title']
                author = meta['metadata'].get('author', '')
                year = meta['metadata'].get('year', '')
                source_id = meta['metadata'].get('doc_id') or (str(title).lower().strip(), str(author).lower().strip(), str(year).lower().strip(), id(meta['metadata']))
                if source_id not in seen:
                    seen.add(source_id)
                    sources.append(build_source_item(meta))

            # Stream LLM tokens live from OpenRouter
            messages = prepare_rag_messages(context=context, question=req.query, history=hist)
            full_answer = ""
            try:
                async for token in stream_llm_async(messages, temperature=0.1):
                    full_answer += token
                    yield _sse_event({"type": "chunk", "text": token})
            except Exception as _stream_err:
                logger.warning("Stream LLM error (%s); streaming extractive fallback", _stream_err)
                full_answer = _build_extractive_fallback_answer(req.query, bm25_results)
                for chunk in _stream_chunks(full_answer, size=40):
                    yield _sse_event({"type": "chunk", "text": chunk + " "})
                    await asyncio.sleep(0.005)

            cleaned_answer = sanitize_answer_text(full_answer)
            _no_info_phrases = (
                "tai lieu khong cung cap thong tin",
                "khong tim thay tai lieu",
                "chua tim thay tai lieu",
                "khong co thong tin nay",
                "khong co thong tin trong tai lieu",
                "khong tim thay",
                "chua tim thay",
                "khong luu tru",
                "chua co tai lieu",
            )
            if any(p in normalize_text(cleaned_answer) for p in _no_info_phrases):
                sources = []

            # Send sources only after streaming if valid information was found
            yield _sse_event({"type": "sources", "sources": sources})
            yield _sse_event({"type": "total_found", "total_found": len(sources)})

            # Record session history
            topic = sources[0].get("subject", "") or sources[0].get("title", "") if sources else ""
            hist.append({
                "query": req.query,
                "answer": cleaned_answer,
                "search_query": search_query,
                "topic": topic,
                "source_count": len(sources),
                "shown_titles": [s.get("title", "") for s in sources[:20]],
                "sources": sources[:10],
            })
            save_session_history(req.session_id, hist)
            _record_query_analytics(req.query, len(sources))

            follow_ups = build_follow_up_suggestions(
                query=req.query, sources=sources, answer=cleaned_answer,
                catalog_mode=False, has_pdf_sources=False, session_id=req.session_id
            )
            yield _sse_event({"type": "follow_ups", "follow_up_suggestions": follow_ups})

            try:
                get_usage_logger().record(
                    session_id=req.session_id,
                    query=req.query,
                    result_count=len(sources),
                    latency_ms=(time.time() - _stream_start) * 1000,
                    follow_up_count=len(follow_ups),
                    client_ip=_client_ip,
                    user_agent=_user_agent,
                    intent=intent,
                )
            except Exception:
                pass

            yield _sse_event({"type": "done"})

        except HTTPException as e:
            yield _sse_event({"type": "error", "message": e.detail})
        except Exception as e:
            yield _sse_event({"type": "error", "message": str(e)})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/")
def root():
    react_index = Path("frontend/dist/index.html")
    if react_index.exists():
        return FileResponse(str(react_index), media_type="text/html", headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})
    return {"status": "ok", "message": "Thư viện QNU RAG API đang chạy", "llm_backend": "OpenRouter", "model": OPENROUTER_MODEL, "ui_url": "/chat", "read_only": is_vercel_runtime()}

@app.get("/chat", response_class=FileResponse)
def serve_chat_ui():
    react_index = Path("frontend/dist/index.html")
    if react_index.exists():
        return FileResponse(str(react_index), media_type="text/html", headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"})
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

@app.get("/SHL-logo.png", include_in_schema=False)
@app.get("/static/SHL-logo.png", include_in_schema=False)
def serve_logo():
    react_logo = Path("frontend/dist/SHL-logo.png")
    if react_logo.exists():
        return FileResponse(str(react_logo), media_type="image/png")
    if os.path.exists("SHL-logo.png"):
        return FileResponse("SHL-logo.png", media_type="image/png")
    raise HTTPException(404, "Logo not found")

@app.get("/favicon.svg", include_in_schema=False)
def serve_dist_favicon_svg():
    fav = Path("frontend/dist/favicon.svg")
    if fav.exists():
        return FileResponse(str(fav), media_type="image/svg+xml")
    raise HTTPException(404, "Favicon not found")

@app.get("/icons.svg", include_in_schema=False)
def serve_dist_icons_svg():
    ico = Path("frontend/dist/icons.svg")
    if ico.exists():
        return FileResponse(str(ico), media_type="image/svg+xml")
    raise HTTPException(404, "Icons not found")

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
    react_index = Path("frontend/dist/index.html")
    if react_index.exists():
        return FileResponse(str(react_index), media_type="text/html", headers={
            "Cache-Control": "no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        })
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
    _canonical_match = resolve_canonical_major_from_query(nganh)
    canonical_name = _canonical_match[0] if _canonical_match else nganh
    _canonical_nganh = get_canonical_major_aliases(canonical_name)
    skip_major_filter = not _canonical_nganh

    def _doc_matches_faculty(doc_major_raw: str) -> bool:
        return doc_matches_canonical_major(doc_major_raw, _canonical_nganh)

    topic_hits = _search_topic_cache_fn(nganh) or []
    if not topic_hits:
        bm25_raw = bm25_engine.search(nganh, top_k=req.top_k * 3)
        topic_hits = [{"doc": r.get("doc", r), "score": r.get("score", 0.0)} for r in bm25_raw]
    if not skip_major_filter:
        topic_hits = [hit for hit in topic_hits if _doc_matches_faculty(str(hit.get("doc", hit).get("major", "")))]
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
    for f in list(data_dir.glob("qnu-allITEM-*-chuan-hoa-new.xlsx")) + list(data_dir.glob("qnu-allITEM-*-chuan-hoa.xlsx")):
        if not f.name.endswith(".backup_raw.xlsx"):
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
