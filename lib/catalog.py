"""
lib/catalog.py — Topic cache, catalog phrase index, and catalog search logic.
Extracted from app.py to reduce the main file size.

All functions take their dependencies as parameters — no global state coupling.
"""

import re
import pickle
import logging
from pathlib import Path
from collections import Counter
from typing import Optional

from .text_utils import normalize_text

logger = logging.getLogger("qnu.library")
VIETNAMESE_STOP_WORDS = {
    "toi", "minh", "em", "anh", "chi", "co", "chu", "bac", "ong", "ba",
    "ai", "tao", "may", "no", "ho", "chung", "ban", "cau", "cac", "nguoi",
    "co", "khong", "ve", "la", "cua", "va", "con", "thi", "nao",
    "sao", "gi", "dau", "bao", "nhieu", "the", "nay", "do", "ay",
    "the nao", "ma", "neu", "vi", "nen", "hoac",
    "hay", "song", "tu", "o", "tai", "voi", "trong", "ngoai",
    "tren", "duoi", "ben", "giua", "khi", "luc", "sau", "truoc",
    "qua", "len", "xuong", "di", "lai", "vao", "ra", "luon",
    "cung", "da", "dang", "se", "sap", "vua", "moi", "con",
    "lam", "rat", "nhe", "nha", "a", "u", "vay", "the",
    "sach", "tai", "lieu", "giao", "trinh", "bai", "bao",
    "tim", "xem", "hoi", "thay", "cho", "gui", "xin",
    "ban", "doc", "tap", "quyen", "cuon",
    "thu", "vien", "qnu", "truong", "dai", "hoc",
    "muon", "de", "den", "lien", "quan", "linh", "vuc",
    "goi", "y", "phu", "hop", "chu", "de", "nganh", "linh-vuc",
    "mot", "hai", "ba", "bon", "nam", "sau", "bay", "tam", "chin", "muoi",
    "tram", "nghin", "trieu", "ty",
}

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

CATALOG_FIELD_WEIGHTS = {
    "title": 5.0, "subject": 10.0, "major": 9.0, "course": 8.0,
    "doc_type": 4.0, "publisher": 2.0, "author": 7.0,
}

GENERIC_CATALOG_PHRASES = {
    "sach", "tai lieu", "tai lieu tham khao", "giao trinh", "bai giang",
    "de an", "luan van", "luan an", "tham khao", "nxb", "nha xuat ban",
    "dai hoc", "truong dai hoc", "truong dai hoc quy nhon",
    "khoa hoc", "nghien cuu", "phuong phap", "phuong phap nghien cuu",
    "nghien cuu khoa hoc", "co file", "khong co file", "loi file", "dang xu ly",
}

CATALOG_TOPIC_INDEX: dict[str, dict] = {}
CATALOG_TOPIC_READY = False
CATALOG_TOPIC_INDEX_VERSION = 2

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
    "AI", "tri tue nhan tao", "artificial intelligence", "ung dung AI",
    "hoc may", "machine learning", "deep learning", "khoa hoc may tinh",
]

AI_DOCUMENT_PHRASES = [
    "tri tue nhan tao", "artificial intelligence", "machine learning",
    "hoc may", "deep learning", "ung dung ai", "cong nghe ai",
    "ky thuat ai", "khoa hoc may tinh",
]

def normalize_topic_text(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _normalize_topic_token(text: str) -> str:
    t = normalize_text(text).strip()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


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


def strip_search_intent_phrases(text: str) -> str:
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


def contains_normalized_phrase(text: str, phrase: str) -> bool:
    if not text or not phrase:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text))

def detect_ai_domain(query: str) -> bool:
    if not query:
        return False
    if re.search(r"\bAI\b", query):
        return True
    q = normalize_topic_text(query)
    if q == "ai":
        return True
    return any(re.search(pattern, q) for pattern in AI_QUERY_PATTERNS)


def expand_domain_terms(query: str) -> list[str]:
    terms = []
    if detect_ai_domain(query):
        terms.extend(AI_EXPANSION_TERMS)
    topic = extract_primary_topic_text(query)
    for phrase, expansions in DOMAIN_PHRASE_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(phrase)}\b", topic):
            terms.extend(expansions)
    return list(dict.fromkeys(terms))

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
    tokens = [t for t in cleaned.split() if t not in VIETNAMESE_STOP_WORDS]
    if not tokens:
        return None
    norm_to_orig: dict[str, str] = {}
    for src in re.findall(r"[A-Za-zÀ-ỹ]+", query):
        norm_to_orig.setdefault(normalize_topic_text(src), src)
    pieces = [norm_to_orig.get(t, t) for t in cleaned.split()]
    return " ".join(pieces)

def extract_primary_topic_text(text: str) -> str:
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

def extract_keywords(text: str) -> str:
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
    filtered = [
        w for w in words
        if w not in VIETNAMESE_STOP_WORDS and (len(w) >= 2 or w in protected_short_tokens)
    ]
    combined = filtered + [normalize_topic_text(term) for term in expanded_terms]
    return " ".join(term for term in dict.fromkeys(combined) if term)


def extract_content_search_query(text: str) -> str:
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


def score_ai_document(doc: dict) -> float:
    fields = [doc.get("title", ""), doc.get("subject", ""), doc.get("major", ""),
              doc.get("course", ""), doc.get("abstract", ""), doc.get("notes", ""),
              doc.get("doc_type", "")]
    raw_text = " ".join(str(f) for f in fields if f)
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


def get_doc_topic_fields(doc: dict) -> dict[str, str]:
    cached = doc.get("_topic_search_fields")
    if cached:
        return cached
    fields = [
        ("title", doc.get("title", "")), ("author", doc.get("author", "")),
        ("subject", doc.get("subject", "")), ("major", doc.get("major", "")),
        ("course", doc.get("course", "")), ("publisher", doc.get("publisher", "")),
        ("abstract", doc.get("abstract", "")), ("notes", doc.get("notes", "")),
        ("doc_type", doc.get("doc_type", "")),
        ("text_preview", (doc.get("text") or "")[:500]),
    ]
    normalized_fields = {name: normalize_topic_text(str(v)) for name, v in fields if v}
    doc["_topic_search_fields"] = normalized_fields
    return normalized_fields


def get_doc_topic_text(doc: dict) -> str:
    cached = doc.get("_topic_search_text")
    if cached:
        return cached
    fields = get_doc_topic_fields(doc)
    text = " ".join(f for f in fields.values() if f)
    doc["_topic_search_text"] = text
    return text


def get_query_topic_terms(query: str) -> tuple[str, list[str], list[str]]:
    topic = extract_primary_topic_text(query)
    catalog_matches = find_catalog_topic_matches(topic)
    if not catalog_matches:
        catalog_matches = find_catalog_topic_matches(strip_search_intent_phrases(query))
    expanded_terms = expand_domain_terms(query)
    phrase_candidates = catalog_matches + [topic] + expanded_terms
    phrases = []
    for phrase in phrase_candidates:
        np = normalize_topic_text(phrase)
        if np and (len(np.split()) > 1 or len(np) > 2):
            phrases.append(np)
    tokens = [
        t for t in re.findall(r"\w+", normalize_topic_text(" ".join(phrase_candidates)))
        if t not in VIETNAMESE_STOP_WORDS and len(t) >= 2
    ]
    return topic, list(dict.fromkeys(tokens)), list(dict.fromkeys(phrases))

def split_catalog_segments(value: str, field_name: str) -> list[str]:
    if not value or value in ("Unknown", "N/A", "nan"):
        return []
    if field_name in {"subject", "major", "course", "doc_type"}:
        parts = re.split(r"[;#|,\n]+", str(value))
    elif field_name == "title":
        parts = re.split(r"[:;#|\n]+", str(value))
    elif field_name in {"author", "publisher"}:
        parts = re.split(r"[,;|\n]+", str(value))
    else:
        parts = [str(value)]
    return [p.strip() for p in parts if p and p.strip()]

def generate_catalog_phrases(value: str, field_name: str) -> set[str]:
    phrases = set()
    for segment in split_catalog_segments(value, field_name):
        ns = normalize_topic_text(segment)
        if is_valid_catalog_phrase(ns):
            phrases.add(ns)
        tokens = ns.split()
        if field_name in {"title", "subject", "major", "course", "author", "publisher"} and len(tokens) >= 2:
            for n in range(2, min(5, len(tokens)) + 1):
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
    item = CATALOG_TOPIC_INDEX.setdefault(phrase, {"score": 0.0, "fields": set(), "docs": set()})
    item["score"] += CATALOG_FIELD_WEIGHTS.get(field_name, 1.0)
    item["fields"].add(field_name)
    item["docs"].add(doc_idx)

def get_catalog_source_mtime() -> float:
    data_path = Path("Data")
    source_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.xlsx"))
    if not source_files:
        return 0.0
    return max(p.stat().st_mtime for p in source_files)

def load_catalog_topic_cache(cache_file: Path, source_mtime: float) -> bool:
    global CATALOG_TOPIC_READY
    from .config import is_vercel_runtime
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
                {"version": CATALOG_TOPIC_INDEX_VERSION, "source_mtime": source_mtime, "index": CATALOG_TOPIC_INDEX},
                f, protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception as e:
        logger.warning("Catalog topic cache write failed: %s", e)

def build_catalog_topic_index(bm25_documents: list[dict] | None = None) -> None:
    global CATALOG_TOPIC_READY
    if CATALOG_TOPIC_READY:
        return
    cache_file = Path("Data") / "_topic_index.pkl"
    source_mtime = get_catalog_source_mtime()
    if load_catalog_topic_cache(cache_file, source_mtime):
        return
    CATALOG_TOPIC_INDEX.clear()
    catalog_fields = ("subject", "major", "course", "doc_type", "title", "publisher", "author")
    docs = bm25_documents or []
    for doc_idx, doc in enumerate(docs):
        if str(doc.get("csv_file", "")).startswith("pdf:"):
            continue
        for field_name in catalog_fields:
            for phrase in generate_catalog_phrases(doc.get(field_name, ""), field_name):
                add_catalog_phrase(phrase, field_name, doc_idx)
    CATALOG_TOPIC_READY = True
    logger.info("Catalog topic index ready: %d phrases", len(CATALOG_TOPIC_INDEX))
    save_catalog_topic_cache(cache_file, source_mtime)

def find_catalog_topic_matches(query_text: str, limit: int = 8) -> list[str]:
    build_catalog_topic_index([])  # Will use existing index if ready
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

def metadata_search_by_query(query: str, bm25_documents: list[dict], top_k: int = 80) -> list[dict]:
    topic, topic_tokens_list, topic_phrases = get_query_topic_terms(query)
    if not topic_tokens_list and not topic_phrases:
        return []
    primary_phrase = normalize_topic_text(topic)
    scored = []
    field_weights = {
        "title": 42.0, "author": 35.0, "subject": 36.0, "major": 34.0,
        "course": 28.0, "publisher": 16.0, "abstract": 12.0, "notes": 10.0, "doc_type": 8.0,
    }
    for idx, doc in enumerate(bm25_documents):
        doc_fields = get_doc_topic_fields(doc)
        doc_text = " ".join(f for f in doc_fields.values() if f)
        score = 0.0
        exact_phrase_hit = False
        for phrase in topic_phrases:
            if not phrase:
                continue
            for field_name, field_text in doc_fields.items():
                if contains_normalized_phrase(field_text, phrase):
                    exact_phrase_hit = True
                    base = field_weights.get(field_name, 8.0)
                    score += base if phrase == primary_phrase else base * 0.55
        match_count = sum(1 for t in topic_tokens_list if re.search(rf"\b{re.escape(t)}\b", doc_text))
        if match_count:
            coverage = match_count / max(len(topic_tokens_list), 1)
            score += (match_count * 2.0) + (coverage * 6.0)
            if len(topic_tokens_list) >= 2 and coverage < 0.45 and not exact_phrase_hit:
                continue
        if score > 0:
            scored.append({"doc": doc, "score": score + 100000.0, "index": idx})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]

TOPIC_CACHE: dict[str, list[int]] = {}
ALL_DOC_KEYWORDS: dict[int, set[str]] = {}


def build_topic_cache(bm25_documents: list[dict]) -> None:
    global TOPIC_CACHE, ALL_DOC_KEYWORDS
    TOPIC_CACHE = {}
    ALL_DOC_KEYWORDS = {}
    for idx, doc in enumerate(bm25_documents):
        keywords = set()
        for field in ("subject", "major", "title"):
            raw = doc.get(field, "")
            if isinstance(raw, (list, tuple)):
                raw = " ".join(str(r) for r in raw)
            raw = str(raw) if raw else ""
            if not raw or raw in ("N/A", "Unknown", "Không xác định"):
                continue
            for part in re.split(r"[;/|,]", raw):
                normalized = _normalize_topic_token(part)
                if normalized and len(normalized) >= 2:
                    keywords.add(normalized)
        ALL_DOC_KEYWORDS[idx] = keywords
        for kw in keywords:
            if kw not in TOPIC_CACHE:
                TOPIC_CACHE[kw] = []
            TOPIC_CACHE[kw].append(idx)

    prefix_cache: dict[str, list[int]] = {}
    for kw, indices in TOPIC_CACHE.items():
        tokens = kw.split()
        for i in range(1, len(tokens) + 1):
            prefix = " ".join(tokens[:i])
            if prefix not in prefix_cache:
                prefix_cache[prefix] = []
            prefix_cache[prefix].extend(indices)
    for prefix, indices in prefix_cache.items():
        if prefix not in TOPIC_CACHE:
            TOPIC_CACHE[prefix] = indices
        TOPIC_CACHE[f"__prefix__{prefix}"] = list(set(indices))

    subjects_with_docs = sum(1 for kw in TOPIC_CACHE if not kw.startswith("__prefix__"))
    logger.info("Topic cache built: %d keywords, %d unique subjects", len(TOPIC_CACHE), subjects_with_docs)

def search_topic_cache(query: str, bm25_documents: list[dict]) -> list[dict] | None:
    q = _normalize_topic_token(query)
    if not q:
        return None
    if q in TOPIC_CACHE:
        return [{"doc": bm25_documents[i], "score": 100.0, "index": i} for i in TOPIC_CACHE[q][:120]]
    prefix_key = f"__prefix__{q}"
    if prefix_key in TOPIC_CACHE:
        return [{"doc": bm25_documents[i], "score": 100.0, "index": i} for i in TOPIC_CACHE[prefix_key][:120]]
    words = q.split()
    if len(words) >= 2:
        candidate_sets = []
        for w in words:
            w_indices = set()
            for kw, indices in TOPIC_CACHE.items():
                if kw.startswith("__prefix__"):
                    continue
                if w in kw:
                    w_indices.update(indices)
            if not w_indices:
                return None
            candidate_sets.append(w_indices)
        common = candidate_sets[0]
        for s in candidate_sets[1:]:
            common = common & s
        if common:
            return [{"doc": bm25_documents[i], "score": 80.0, "index": i} for i in list(common)]
    return None
