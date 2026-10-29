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

from .text_utils import normalize_text, check_diacritic_consistency

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

CONTENT_TERMS_TO_KEEP = (
    "co", "chu", "giao", "hoc", "khi", "may", "quan", "tao", "nam",
    "ban", "lam", "nha", "nuoc", "sinh", "tin", "toan", "phap", "kho",
    "van", "tri", "thanh", "thay", "nguoi", "da", "song", "cau", "tap",
    "quyen", "cuon", "tai", "lieu", "trinh", "bai", "bao", "dong", "gia",
    "hoa", "loai", "luat", "moi", "ngay", "phu", "phong", "qua", "ra",
    "so", "su", "tien", "doc", "thu", "vien", "muon", "de", "den", "nganh",
    "linh", "vuc", "hop"
)
for compound_term in CONTENT_TERMS_TO_KEEP:
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
    r"\b(?:ve\s+)?(?:chu\s+de|linh\s+vuc|chuyen\s+nganh|nganh)\s+([^,.;?]+)",
    r"\b(?:lien quan den|ve|linh vuc|chu de|nganh)\s+(.+)$",
]

TOPIC_TRAILING_PATTERNS = [
    r"\b(?:ban\s+)?(?:hay\s+)?(?:goi\s+y|de\s+xuat|gioi\s+thieu|cho\s+(?:toi|minh|em)\s+xin)\b.*$",
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
    "phat trien ban than": ["ky nang song", "dac nhan tam", "nghe thuat ung xu", "thanh cong", "tam ly ung xu", "ky nang mem"],
    "hoan thien ban than": ["ky nang song", "dac nhan tam", "nghe thuat ung xu", "thanh cong"],
    "ky nang song": ["ky nang song", "giao duc ky nang", "ky nang mem", "ung xu"],
    "lam giau": ["kinh doanh", "quan tri kinh doanh", "tai chinh", "khoi nghiep"],
    "tai chinh ca nhan": ["tai chinh", "kinh te", "dau tu", "ngan hang"],
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
    "cac loai", "the loai", "cac the loai", "cac dang", "nhung loai",
    "cac loai hinh", "loai hinh", "cac van de", "nhung van de",
}

CATALOG_TOPIC_INDEX: dict[str, dict] = {}
CATALOG_TOPIC_READY = False
CATALOG_TOPIC_INDEX_VERSION = 3

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
    topic = re.sub(r"^(?:cac|nhung|mot so|tat ca cac|tat ca|nhom)\s+", " ", topic)
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
    # Filter out generic phrases
    cleaned_matches = []
    for cm in catalog_matches:
        if cm in GENERIC_CATALOG_PHRASES:
            continue
        cleaned_matches.append(cm)
    selected_topics = cleaned_matches[:2] if cleaned_matches else ([q] if q else [])
    expanded_terms = expand_domain_terms(text)

    # Preserve the full topic phrase intact when q was extracted
    if q and q in selected_topics:
        combined = [q] + [normalize_topic_text(term) for term in expanded_terms]
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
        r"\b(?:ban oi|ban co the|ban co biet|toi|minh|em|anh|chi)\b",
        r"\b(?:hay|vui long)\b",
        r"\b(?:cho toi|cho minh|cho em)\b",
        r"\b(?:khong|ko|k)\b",
        r"\b(?:sach|tai lieu|giao trinh|bai bao|bai)\s+(?:ve|noi ve|cua)\b",
        r"\b(?:ve|noi ve)\s+(?:cac|nhung|mot so)\b",
        r"\b(?:ve|noi ve)\b",
        r"^(?:cac|nhung|mot so)\s+",
        r"\bcua\s+tac gia\b",
        r"\btim kiem\b",
        r"\b(?:tim|tra cuu|liet ke)\b",
        r"\blien quan den\b",
        r"\blien quan\b",
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
    ]
    # Include extra_fields (unmapped columns from generic XLSX loader)
    for k, v in doc.get("extra_fields", {}).items():
        fields.append((f"extra_{k}", v))
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
    build_catalog_topic_index([])
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
    has_query_accents = any(c in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúũụưừứửữựỳýỷỹỵđ" for c in query.lower())
    accented_tokens = []
    if has_query_accents and len(topic_tokens_list) in (2, 3):
        accented_tokens = [
            t.lower() for t in re.findall(r'[a-zA-Zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúũụưừứửữựỳýỷỹỵđ]+', query)
            if len(t) >= 2 and normalize_text(t) not in VIETNAMESE_STOP_WORDS
        ]

    for idx, doc in enumerate(bm25_documents):
        doc_fields = get_doc_topic_fields(doc)
        doc_text = " ".join(f for f in doc_fields.values() if f)
        score = 0.0
        exact_phrase_hit = False
        for phrase in topic_phrases:
            if not phrase:
                continue
            for field_name, field_text in doc_fields.items():
                if " " in phrase and contains_normalized_phrase(field_text, phrase):
                    exact_phrase_hit = True
                    base = field_weights.get(field_name, 8.0)
                    score += base if phrase == primary_phrase else base * 0.55
                elif " " not in phrase and contains_normalized_phrase(field_text, phrase):
                    base = field_weights.get(field_name, 8.0)
                    score += base * 0.3
        # Count how many query tokens appear together in the SAME key field
        # (only metadata fields — exclude full-text fields to avoid false matches
        # from diacritic-stripped tokens scattered across unrelated content)
        _token_match_fields = {"title", "author", "subject", "major", "course", "publisher"}
        field_match_counts = []
        for field_name, field_text in doc_fields.items():
            if field_name not in _token_match_fields:
                continue
            field_count = sum(
                1 for t in topic_tokens_list
                if re.search(rf"\b{re.escape(t)}\b", field_text)
            )
            if field_count:
                field_match_counts.append(field_count)
        match_count = max(field_match_counts) if field_match_counts else 0
        if match_count:
            coverage = match_count / max(len(topic_tokens_list), 1)
            score += (match_count * 2.0) + (coverage * 6.0)
            if not exact_phrase_hit:
                if len(topic_tokens_list) <= 3 and coverage < 1.0:
                    continue
                if len(topic_tokens_list) >= 4 and coverage < 0.45:
                    continue

        if accented_tokens:
            raw_check = (str(doc.get("title", "")) + " " + str(doc.get("subject", ""))).lower()
            if not check_diacritic_consistency(raw_check, accented_tokens):
                continue

        if exact_phrase_hit:
            scored.append({"doc": doc, "score": score + 1000.0, "index": idx})
        elif score > 0:
            scored.append({"doc": doc, "score": score + 50.0, "index": idx})
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
    matched_indices = None
    base_score = 100.0
    if q in TOPIC_CACHE:
        exact = set(TOPIC_CACHE[q])
        prefix_key = f"__prefix__{q}"
        if prefix_key in TOPIC_CACHE:
            exact.update(TOPIC_CACHE[prefix_key])
        matched_indices = list(exact)
    elif f"__prefix__{q}" in TOPIC_CACHE:
        matched_indices = list(set(TOPIC_CACHE[f"__prefix__{q}"]))
    else:
        words = [w for w in q.split() if w not in VIETNAMESE_STOP_WORDS and len(w) >= 2]
        if len(words) >= 2:
            candidate_sets = []
            for w in words:
                w_indices = set()
                w_pattern = re.compile(rf"\b{re.escape(w)}\b")
                for kw, indices in TOPIC_CACHE.items():
                    if kw.startswith("__prefix__"):
                        continue
                    if w_pattern.search(kw):
                        w_indices.update(indices)
                if not w_indices:
                    return None
                candidate_sets.append(w_indices)
            common = candidate_sets[0]
            for s in candidate_sets[1:]:
                common = common & s
            if common:
                matched_indices = list(common)
                base_score = 80.0

    if not matched_indices:
        return None

    # Filter with diacritic consistency if query has Vietnamese accents
    has_query_accents = any(c in "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúũụưừứửữựỳýỷỹỵđ" for c in query.lower())
    if has_query_accents:
        accented_tokens = [
            t.lower() for t in re.findall(r'[a-zA-Zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúũụưừứửữựỳýỷỹỵđ]+', query)
            if len(t) >= 2 and normalize_text(t) not in VIETNAMESE_STOP_WORDS
        ]
        if accented_tokens:
            filtered = []
            for i in matched_indices:
                doc = bm25_documents[i]
                raw_check = (str(doc.get("title", "")) + " " + str(doc.get("subject", ""))).lower()
                if check_diacritic_consistency(raw_check, accented_tokens):
                    filtered.append(i)
            matched_indices = filtered

    if matched_indices:
        return [{"doc": bm25_documents[i], "score": base_score, "index": i} for i in matched_indices]
    return None

# ═══════════════════════════════════════════════════════════════════
# Enhanced query understanding helpers — standalone, zero side-effects
# ═══════════════════════════════════════════════════════════════════

MAJOR_ABBREVIATIONS = {
    "cntt": "công nghệ thông tin",
    "httt": "hệ thống thông tin",
    "khmt": "khoa học máy tính",
    "ktmt": "kỹ thuật máy tính",
    "mmt": "mạng máy tính",
    "attt": "an toàn thông tin",
    "cnpm": "công nghệ phần mềm",
    "qtkd": "quản trị kinh doanh",
    "kt": "kế toán",
    "tcnh": "tài chính ngân hàng",
}

def detect_study_major_query(query: str) -> str | None:
    """Trích ngành học từ 'tôi học X', 'em học X' ...
    Trả về tên ngành có dấu tiếng Việt (vd 'Công nghệ thông tin') hoặc None.
    KHÔNG ảnh hưởng _extract_major_from_query — dùng pattern 'học' thay vì 'ngành'.
    """
    if not query:
        return None
    q_norm = normalize_topic_text(query)
    patterns = [
        r"(?:toi|em|anh|chi|minh|ban)\s+(?:dang\s+)?hoc\s+(?:nganh\s+)?(.+)$",
        r"(?:toi|em|anh|chi|minh|ban)\s+hoc\s+(.+)$",
        r"^hoc\s+(?:nganh\s+)?(.+)$",
    ]
    captured = None
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            captured = m.group(1).strip()
            break
    if not captured:
        return None
    for pattern in TOPIC_TRAILING_PATTERNS:
        captured = re.sub(pattern, " ", captured)
    captured = re.sub(r"\s+", " ", captured).strip()
    if not captured:
        return None
    # Check abbreviation (e.g. "cntt")
    abbrev = captured.replace(" ", "").lower()
    if abbrev in MAJOR_ABBREVIATIONS:
        return MAJOR_ABBREVIATIONS[abbrev].title()
    # Map back to original diacritics
    norm_to_orig: dict[str, str] = {}
    for src in re.findall(r"[A-Za-zÀ-ỹ]+", query):
        norm_to_orig.setdefault(normalize_topic_text(src), src)
    tokens = captured.split()
    if len(tokens) >= 2 or (len(tokens) == 1 and len(tokens[0]) >= 3):
        pieces = [norm_to_orig.get(t, t) for t in tokens]
        return " ".join(pieces)
    return None


_TITLE_CONTAINS_PATTERNS = [
    # Quoted string patterns: có chữ "cánh diều", có tên "...", tên chứa "..."
    r'(?:co\s+chu|chua\s+tu|chua\s+chu|co\s+tu|trong\s+ten\s+co\s+chu|trong\s+ten\s+co)\s+["\']([^"\']+)["\']',
    r'["\']([^"\']+)["\']\s+trong\s+ten',
    # Unquoted multi-word patterns: có chữ cánh diều trong tên sách
    r'(?:co\s+chu|chua\s+tu|chua\s+chu|co\s+tu|co\s+cum\s+tu)\s+(.+?)\s+trong\s+(?:ten|tieu\s+de)',
    r'trong\s+(?:ten|tieu\s+de)\s+(?:sach\s+)?(?:co\s+chu|co\s+tu|chua\s+tu|chua\s+chu|co)\s+(.+)$',
    r'co\s+(?:sach|tai lieu|giao trinh)\s+(?:nao\s+)?co\s+chu\s+(.+)$',
    r'ten\s+(?:sach\s+)?(?:co\s+chu|chua\s+chu|chua\s+tu|co\s+tu|co)\s+(.+)$',
]

def detect_title_contains_keyword(query: str) -> str | None:
    """Phát hiện 'có chữ X trong tên', 'có từ "cánh diều" trong tên sách'..."""
    if not query:
        return None

    # 1. Quoted string check: nếu người dùng bỏ trong ngoặc kép "..." hoặc '...'
    quoted_match = re.search(r'["\']([^"\']{2,})["\']', query)
    if quoted_match:
        quoted_val = quoted_match.group(1).strip()
        q_norm = normalize_topic_text(query)
        if any(w in q_norm for w in ("trong ten", "tieu de", "co chu", "co tu", "chua tu", "chua chu", "ten sach", "trong sach")):
            return quoted_val

    # 2. Pattern check
    q_norm = normalize_topic_text(query)
    for p in _TITLE_CONTAINS_PATTERNS:
        m = re.search(p, q_norm)
        if m:
            raw_kw = m.group(1).strip().strip('"\'')
            raw_kw = re.sub(r'\s+(?:khong|ko|k|vay|a|nhe|ha|vay\s+ad|khong\s+ad)$', '', raw_kw).strip()
            if len(raw_kw) >= 2:
                kw_norm = normalize_topic_text(raw_kw)
                orig_words = query.split()
                for i in range(len(orig_words)):
                    for j in range(i + 1, len(orig_words) + 1):
                        sub = " ".join(orig_words[i:j]).strip("\"'.,?!;:")
                        if normalize_topic_text(sub) == kw_norm:
                            return sub
                return raw_kw
    return None


_LANGUAGE_NAMES = {
    "anh": "anh", "english": "anh", "en": "anh",
    "viet": "việt", "việt": "việt", "viet nam": "việt",
    "việt nam": "việt", "vietnamese": "việt", "vi": "việt",
    "phap": "pháp", "pháp": "pháp", "french": "pháp", "fr": "pháp",
    "trung": "trung", "trung quoc": "trung", "hoa": "trung",
    "nhat": "nhật", "nhật": "nhật", "japanese": "nhật", "jp": "nhật",
}

def detect_language_preference(query: str) -> str | None:
    """Phát hiện 'tiếng Anh', 'English', 'Song ngữ' → trả về mã ('anh', ...)."""
    if not query:
        return None
    q_lower = query.lower().strip()
    q_norm = normalize_topic_text(query)
    patterns = [
        r"tieng\s+(\S+)",
        r"(?:sach|tai lieu|giao trinh)\s+tieng\s+(\S+)",
    ]
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            lang_word = m.group(1).strip()
            if lang_word in _LANGUAGE_NAMES:
                return _LANGUAGE_NAMES[lang_word]
    if re.search(r'\benglish\b', q_lower):
        return 'anh'
    if re.search(r'\bvietnamese\b', q_lower):
        return 'việt'
    return None


def detect_year_constraint(query: str) -> tuple[int | None, str | None]:
    """Phát hiện 'từ năm 2023', 'xuất bản sau 2022' ...
    Trả về (year, direction) với direction: 'after' | 'before' | 'exact'.
    """
    if not query:
        return None, None
    q_norm = normalize_topic_text(query)
    q_lower = query.lower()
    patterns = [
        (r"(?:tu|từ)\s+(?:nam|năm)?\s*(\d{4})", "after"),
        (r"(?:sau|sau nam|sau năm)\s+(\d{4})", "after"),
        (r"(?:truoc|trước|truoc nam|trước năm)\s+(\d{4})", "before"),
        (r"(?:nam|năm)\s+(\d{4})\s+(?:tro di|trở đi|tro len|trở lên)", "after"),
        (r"(?:nam|năm)\s+(\d{4})\s+(?:tro ve|trở về|ve truoc|về trước)", "before"),
        (r"(?:xuat ban|xuất bản)\s+(?:nam|năm)?\s*(\d{4})", "exact"),
        (r"(?:xuat ban|xuất bản)\s+(?:tu|từ|sau)\s+(\d{4})", "after"),
    ]
    for pattern, direction in patterns:
        m = re.search(pattern, q_norm)
        if m:
            try:
                year = int(m.group(1))
                if 1900 <= year <= 2030:
                    return year, direction
            except ValueError:
                pass
    years_in_query = re.findall(r'\b(19\d{2}|20\d{2})\b', q_lower)
    if len(years_in_query) == 1:
        year = int(years_in_query[0])
        if 1900 <= year <= 2030:
            if any(w in q_norm for w in ["tu", "sau", "tro di", "tro len"]):
                return year, "after"
            elif any(w in q_norm for w in ["truoc", "ve truoc", "tro ve"]):
                return year, "before"
            else:
                return year, "exact"
    return None, None


def doc_matches_year(doc: dict, year: int, direction: str = 'exact') -> bool:
    """Kiểm tra document có khớp với điều kiện năm (exact / after / before) không."""
    if not doc or not year:
        return False
    doc_year = doc.get('year', 0) if isinstance(doc, dict) else 0
    if not doc_year or doc_year in ('N/A', 'Unknown', 'None', ''):
        return False
    try:
        dy = int(doc_year)
    except (ValueError, TypeError):
        return False
    if direction == 'exact':
        return dy == year
    elif direction == 'after':
        return dy >= year
    elif direction == 'before':
        return dy <= year
    return False


def is_pure_year_query(query: str) -> tuple[bool, int | None, str | None]:
    """
    Phát hiện câu hỏi liệt kê sách thuần túy theo năm (không chứa chủ đề, ngành, hay tác giả cụ thể).
    Ví dụ:
    - 'những sách nào xuất bản năm 1991' -> (True, 1991, 'exact')
    - 'sách xuất bản năm 1991' -> (True, 1991, 'exact')
    - 'những sách từ năm 2020' -> (True, 2020, 'after')
    - 'sách công nghệ thông tin năm 1991' -> (False, None, None)  (vì có chủ đề CNTT)
    """
    if not query:
        return False, None, None
    year, direction = detect_year_constraint(query)
    if not year:
        return False, None, None

    q_norm = normalize_topic_text(query)
    q_sub = re.sub(r'\b\d{4}\b', ' ', q_norm)
    noise = [
        "nhung", "cac", "co", "sach", "tai lieu", "giao trinh", "an pham",
        "cuon", "quyen", "nao", "xuat ban", "xuat", "ban", "nam",
        "tu", "sau", "truoc", "tro di", "tro ve", "tro len", "ve truoc",
        "thoi", "ky", "giai doan", "trong", "vao", "khoang", "thoi diem",
        "danh sach", "liet ke", "tim", "cho", "em", "minh", "toi", "hoi",
        "xin", "biet", "thu vien", "co nhung", "tat ca", "nhung cuon", "cac cuon",
        "nhung quyen", "cac quyen", "nhung sach", "cac sach"
    ]
    for w in sorted(noise, key=len, reverse=True):
        q_sub = re.sub(r'\b' + re.escape(w) + r'\b', ' ', q_sub)

    rem = q_sub.strip()
    rem = re.sub(r'\s+', ' ', rem)
    if len(rem) <= 2:
        return True, year, direction
    return False, None, None



_BEGINNER_PHRASES = frozenset([
    "moi bat dau", "nguoi moi", "cho nguoi moi",
    "moi hoc", "bat dau",
    "co ban", "nhap mon",
    "beginner", "starting", "fundamental", "introduction",
    "tu hoc",
])

def is_beginner_friendly_query(query: str) -> bool:
    """Phát hiện 'mới bắt đầu', 'người mới', 'cơ bản' ..."""
    if not query:
        return False
    q_norm = normalize_topic_text(query)
    return any(phrase in q_norm for phrase in _BEGINNER_PHRASES)


_EXAM_PHRASES = frozenset([
    "on tap", "on thi", "thi", "kiem tra",
    "de cuong", "bai tap", "luyen thi",
    "exam", "test", "revision", "practice",
])

def is_exam_prep_query(query: str) -> bool:
    """Phát hiện 'thi', 'ôn tập', 'kiểm tra' ..."""
    if not query:
        return False
    q_norm = normalize_topic_text(query)
    return any(phrase in q_norm for phrase in _EXAM_PHRASES)


def detect_teaching_subject(query: str) -> str | None:
    """Phát hiện 'giảng dạy môn X' → trả về X."""
    if not query:
        return None
    q_norm = normalize_topic_text(query)
    patterns = [
        r"(?:giang\s+day|day)\s+(?:mon|môn)\s+(.+)$",
        r"(?:giang\s+day|day)\s+(?:ve|lien quan den)\s+(.+)$",
    ]
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            subject = m.group(1).strip()
            subject = re.sub(r"\s+", " ", subject).strip()
            if subject and len(subject) >= 3:
                return subject
    return None


def detect_course_name(query: str) -> str | None:
    """Phát hiện 'môn X', 'học phần X' (không phải 'tiếng Anh' = language)."""
    if not query:
        return None
    q_norm = normalize_topic_text(query)
    patterns = [
        r"(?:mon|môn)\s+(.+)$",
        r"(?:hoc phan)\s+(.+)$",
    ]
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            name = m.group(1).strip()
            name = re.sub(r"\s+", " ", name).strip()
            if name and len(name) >= 3 and name not in _LANGUAGE_NAMES:
                return name
    return None


# Common typos for technical terms
_TYPO_CORRECTIONS = {
    "pythn": "python", "pithon": "python", "pyhton": "python",
    "jave": "java", "jav": "java",
    "javascrip": "javascript", "javasript": "javascript",
    "reac": "react", "reacj": "react",
    "nod": "node", "nodjs": "nodejs",
    "djang": "django", "flas": "flask",
    "tenxorflow": "tensorflow", "tensorflo": "tensorflow",
    "panda": "pandas", "numpty": "numpy", "numpt": "numpy",
    "sklear": "scikit-learn", "sklearn": "scikit-learn",
    "matplolib": "matplotlib", "matplo": "matplotlib",
    "machinelearning": "machine learning",
    "deeplearning": "deep learning",
}

def apply_fuzzy_corrections(query: str) -> str:
    """Sửa lỗi chính tả phổ biến. Trả về query đã sửa, hoặc '' nếu không cần."""
    if not query:
        return ""
    q_lower = query.lower().strip()
    tokens = q_lower.split()
    changed = False
    for i, t in enumerate(tokens):
        t_clean = t.strip(".,!?;:\"'()[]{}")
        if t_clean in _TYPO_CORRECTIONS:
            tokens[i] = tokens[i].replace(t_clean, _TYPO_CORRECTIONS[t_clean])
            changed = True
    if changed:
        return " ".join(tokens)
    return ""


def filter_by_language(results: list[dict], lang: str) -> list[dict]:
    """Lọc kết quả theo ngôn ngữ dựa trên location, title, subject.
    lang: 'anh' | 'việt' | ...
    results: có thể là flat hoặc {"doc":{}} wrapper.
    """
    if not results or not lang:
        return results
    lang = lang.lower().strip()
    filtered = []
    for r in results:
        doc = r.get("doc", r) if isinstance(r, dict) else r
        if not isinstance(doc, dict):
            continue
        title = doc.get('title', '') or ''
        location = doc.get('location', '') or ''
        bilingual = doc.get('bilingual_title', '') or ''
        norm_loc = normalize_text(location)
        norm_bilingual = normalize_text(bilingual)
        if lang == 'anh':
            if 'ngoai van' in norm_loc:
                filtered.append(r)
            elif norm_bilingual and 'song ngu' in norm_bilingual:
                filtered.append(r)
            elif title:
                has_vn = bool(re.search(r'[\u00C0-\u024F]', title))
                if not has_vn and len(title.strip()) > 0:
                    filtered.append(r)
        elif lang == 'việt':
            has_vn = bool(re.search(r'[\u00C0-\u024F]', title))
            if has_vn:
                filtered.append(r)
        else:
            lang_norm = normalize_text(lang)
            doc_lang = (doc.get('language', '') or '').lower()
            if (lang_norm in norm_loc or lang_norm in norm_bilingual or
                (lang == 'trung' and doc_lang in ('chi', 'zh', 'zho')) or
                (lang == 'pháp' and doc_lang in ('fre', 'fr', 'fra'))):
                filtered.append(r)
    return filtered


def filter_by_year(results: list[dict], year: int, direction: str = 'exact') -> list[dict]:
    """Lọc kết quả theo năm. results: có thể là flat hoặc {"doc":{}} wrapper."""
    if not results or not year:
        return results
    filtered = []
    for r in results:
        doc = r.get("doc", r) if isinstance(r, dict) else r
        doc_year = doc.get('year', 0) if isinstance(doc, dict) else 0
        if not doc_year or doc_year in ('N/A', 'Unknown', ''):
            continue
        try:
            dy = int(doc_year)
        except (ValueError, TypeError):
            continue
        if direction == 'exact' and dy == year:
            filtered.append(r)
        elif direction == 'after' and dy >= year:
            filtered.append(r)
        elif direction == 'before' and dy <= year:
            filtered.append(r)
    return filtered


def detect_generic_request(query: str) -> bool:
    """
    Phát hiện câu hỏi gợi ý sách chung chung (generic suggestion) KHÔNG có chủ đề/tác giả/ngành cụ thể.
    Ví dụ: 'gợi ý vài cuốn sách', 'gợi ý sách hay', 'cho mình vài cuốn sách để đọc', 'gợi ý sách'...
    Nếu câu có chủ đề (ví dụ 'gợi ý sách nông nghiệp', 'gợi ý sách về AI', 'đề tài nghiên cứu về...'):
    -> Trả về False để Search Engine tra cứu đúng chủ đề.
    """
    if not query:
        return False
    q_norm = normalize_topic_text(query)
    has_generic_marker = bool(re.search(
        r"\b(?:goi\s+y|de\s+xuat|gioi\s+thieu|vai\s+cuon|mot\s+vai|it\s+cuon|sach\s+hay|doc\s+sach|cho\s+(?:minh|em|toi)\s+(?:xin\s+)?sach)\b",
        q_norm
    ))
    if not has_generic_marker:
        return False

    stripped = q_norm
    stopwords_patterns = [
        r"\b(?:xin\s+chao|chao\s+ban|chao\s+ad|chao\s+bot|ad\s+oi|bot\s+oi|ban\s+oi|em\s+oi)\b",
        r"\b(?:ban|ad|bot|toi|em|minh|anh|chi|ai|gi|nao)\b",
        r"\b(?:co\s+the|giup|cho|xin|hay|lam\s+on|vui\s+long|voi|nhe|nha|a|nhi|khong|chua)\b",
        r"\b(?:goi\s+y|de\s+xuat|gioi\s+thieu|tim|lua|chon|lay|xem|doc|kiem|muon|can|thich|biet|hoi)\b",
        r"\b(?:sach|tai\s+lieu|cuon|quyen|tap|ban|bai\s+giang|giao\s+trinh|dau\s+sach)\b",
        r"\b(?:vai|mot\s+vai|it|nhung|cac|mot\s+so|tat\s+ca|nhieu|1|2|3|mot|vai\s+ba)\b",
        r"\b(?:hay|tot|thu\s+vi|noi\s+tieng|bo\s+ich|pho\s+bien|moi|hot|tieu\s+bieu|de\s+doc|y\s+nghia)\b",
        r"\b(?:trong\s+thu\s+vien|o\s+thu\s+vien|thu\s+vien|o\s+day|hien\s+co|de\s+doc|de\s+tham\s+khao)\b",
        r"\b(?:ve|thuoc|lien\s+quan|de)\b",
    ]
    for p in stopwords_patterns:
        stripped = re.sub(p, " ", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()

    return len(stripped) < 3