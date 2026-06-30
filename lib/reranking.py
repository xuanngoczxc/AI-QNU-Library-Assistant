"""
lib/reranking.py — Search result reranking, author/publisher matching, PDF collapse.
"""

import re
from typing import Optional
from .text_utils import normalize_text
from .catalog import (
    get_query_topic_terms, get_doc_topic_text, contains_normalized_phrase,
    VIETNAMESE_STOP_WORDS,
    detect_ai_domain, score_ai_document,
)


def detect_author_query(query: str) -> bool:
    if not query:
        return False
    q = normalize_text(query)
    author_markers = [
        r"\btac gia\b", r"\btac gia\s+(?:la|co ten|ten)\b",
        r"\bcua\s+tac gia\b", r"\bsach\s+cua\s+tac gia\b",
        r"\btai lieu\s+cua\s+tac gia\b", r"\bgiao trinh\s+cua\s+tac gia\b",
    ]
    if any(re.search(p, q) for p in author_markers):
        return True
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
    words = q.split()
    if 2 <= len(words) <= 4 and words[0] in VIETNAMESE_SURNAMES:
        if not any(w in topic_words for w in words[1:]):
            return True
    return False


def detect_publisher_query(query: str) -> bool:
    if not query:
        return False
    q = normalize_text(query)
    publisher_markers = [
        r"\bnha xuat ban\b", r"\bnxb\b",
        r"\bnha xuat ban\s+(?:la|co ten|ten)\b",
        r"\bcua\s+nha xuat ban\b", r"\bsach\s+cua\s+nxb\b",
    ]
    return any(re.search(p, q) for p in publisher_markers)


def extract_author_name(query: str) -> str:
    if not query:
        return ""
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

    name_ref_patterns = [
        r"(?:sach|tai lieu|giao trinh|bai bao|bai)\s+cua\s+(.+?)(?:\s+khong|\s+ko|\s+ne|\s+nha|\?)?$",
        r"cua\s+(.+?)(?:\s+khong|\s+ko|\?)?$",
    ]
    for pattern in name_ref_patterns:
        match = re.search(pattern, q)
        if match:
            name = match.group(1).strip().rstrip(",. ")
            name = re.sub(r"\s+(?:khong|ko|lam on|hay|vui long|toi|minh|xin|cho|gui)\b.*$", "", name)
            name = name.strip().rstrip(",. ?")
            name_tokens = name.split()
            if len(name_tokens) >= 2 and name_tokens[0] in VIETNAMESE_SURNAMES:
                if not any(w in topic_words for w in name_tokens[1:]):
                    return name
    words = q.split()
    if 2 <= len(words) <= 4 and words[0] in VIETNAMESE_SURNAMES:
        if not any(w in topic_words for w in words[1:]):
            return q
    if 2 <= len(words) <= 5:
        if ',' in q:
            parts = [p.strip() for p in q.split(',') if p.strip()]
            if 1 <= len(parts) <= 3:
                if all(len(p) >= 2 for p in parts):
                    all_clean = all(not any(t in topic_words for t in p.split()) for p in parts)
                    if all_clean:
                        return q
        else:
            if all(w[0].isupper() or '.' in w for w in words if w):
                if not any(w in topic_words for w in words):
                    if len(q) <= 60:
                        return q
    return ""


def extract_publisher_name(query: str) -> str:
    if not query:
        return ""
    q = normalize_text(query).strip()
    patterns = [r"(?:nha xuat ban|nxb)\s+(.+)"]
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
    if not name:
        return ""
    normalized = normalize_text(name)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s*\([^)]*\)\s*", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def author_name_matches(doc_author: str, query_author: str) -> bool:
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
    if doc_norm == query_norm or query_norm in doc_norm:
        return True
    first_matches = doc_tokens[0] == query_tokens[0]
    last_matches = doc_tokens[-1] == query_tokens[-1]
    if first_matches and last_matches:
        doc_set = set(doc_tokens)
        overlap = sum(1 for t in query_tokens if t in doc_set)
        if overlap == len(query_tokens) and overlap >= 3:
            return True

    def _to_initials_surname(n):
        n = re.sub(r'[,\.]+', ' ', n).lower()
        n = re.sub(r'\s+', ' ', n).strip()
        toks = n.split()
        if len(toks) >= 2:
            surname = toks[-1]
            initials = ''.join(t[0] for t in toks[:-1] if t)
            return initials + surname
        return n.replace(' ', '')

    doc_key = _to_initials_surname(doc_norm)
    query_key = _to_initials_surname(query_norm)
    if doc_key and query_key and len(doc_key) >= 3 and len(query_key) >= 3:
        if doc_key == query_key or query_key in doc_key or doc_key in query_key:
            if abs(len(doc_key) - len(query_key)) <= 4:
                return True
    if len(query_tokens) == 1 and len(query_tokens[0]) >= 3:
        if query_tokens[0] in doc_norm:
            return True
    return False


def merge_search_results(*groups: list[dict]) -> list[dict]:
    merged = {}
    for group in groups:
        for result in group or []:
            doc = result.get("doc", {})
            key = (str(doc.get("title", "")).lower().strip(),
                   str(doc.get("author", "")).lower().strip(),
                   str(doc.get("year", "")).lower().strip())
            if key not in merged or result.get("score", 0.0) > merged[key].get("score", 0.0):
                merged[key] = result
    return sorted(merged.values(), key=lambda item: item.get("score", 0.0), reverse=True)


def collapse_catalog_pdf_pages(search_results: list[dict], pdf_lookup=None) -> list[dict]:
    """Represent matching PDF pages as one document in catalog responses."""
    regular_results = []
    grouped_pdfs = {}
    for result in search_results:
        doc = result.get("doc", {})
        is_pdf_page = str(doc.get("csv_file", "")).startswith("pdf:") or doc.get("page") is not None
        if not is_pdf_page:
            regular_results.append(result)
            continue
        pdf_info = pdf_lookup(doc.get("source", ""), doc.get("title", "")) if pdf_lookup else None
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
            **doc, "title": display_title, "author": author, "subject": "",
            "source": source_name, "csv_file": f"pdf:{source_name}" if source_name else doc.get("csv_file", ""),
            "doc_type": doc.get("doc_type") or "PDF", "format": "Số", "text": "",
        }
        grouped_doc.pop("page", None)
        grouped_pdfs[group_key] = {**result, "doc": grouped_doc}

    collapsed = regular_results + list(grouped_pdfs.values())
    for item in collapsed:
        doc = item.get("doc", {})
        is_pdf = str(doc.get("csv_file", "")).startswith("pdf:") or doc.get("doc_type") == "PDF"
        if is_pdf:
            item["score"] = item.get("score", 0.0) + 10000.0
    return sorted(collapsed, key=lambda item: item.get("score", 0.0), reverse=True)


def rerank_results_for_query(query: str, search_results: list[dict], prefer_strict: bool = False) -> list[dict]:
    if not search_results:
        return search_results

    author_name = extract_author_name(query) if detect_author_query(query) else ""
    publisher_name = extract_publisher_name(query) if detect_publisher_query(query) else ""

    if author_name:
        author_results = []
        non_author_results = []
        for result in search_results:
            if author_name_matches(result.get("doc", {}).get("author", ""), author_name):
                author_results.append({**result, "score": result.get("score", 0.0) + 500.0})
            else:
                non_author_results.append(result)
        if author_results:
            author_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            non_author_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return author_results if prefer_strict else author_results + non_author_results[:3]

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
                pub_results.append({**result, "score": result.get("score", 0.0) + 300.0})
            else:
                non_pub_results.append(result)
        if pub_results:
            pub_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            if prefer_strict:
                return pub_results
            non_pub_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return pub_results + non_pub_results[:3]
        if prefer_strict:
            return []

    if detect_ai_domain(query):
        matched = []
        unmatched = []
        for result in search_results:
            relevance = score_ai_document(result.get("doc", {}))
            boosted = {**result, "score": result.get("score", 0.0) + relevance}
            (matched if relevance > 0 else unmatched).append(boosted)
        if matched:
            matched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return matched if prefer_strict else matched + unmatched

    for result in search_results:
        doc = result.get("doc", {})
        is_pdf = str(doc.get("csv_file", "")).startswith("pdf:") or doc.get("doc_type") == "PDF"
        if is_pdf and doc.get("text") and len(doc["text"].strip()) > 100:
            result["score"] = result.get("score", 0.0) + 500.0

    _, topic_tokens_list, topic_phrases = get_query_topic_terms(query)
    if not topic_tokens_list and not topic_phrases:
        return search_results

    unique_tokens = list(dict.fromkeys(topic_tokens_list))
    matched = []
    unmatched = []
    min_phrase_length = 4
    if len(unique_tokens) >= 6:
        strict_threshold = 0.80
    elif len(unique_tokens) >= 3:
        strict_threshold = 0.65
    else:
        strict_threshold = 0.50
    relaxed_threshold = max(0.15, min(0.40, 3.0 / max(len(unique_tokens), 1)))
    is_single_token = len(unique_tokens) == 1

    for result in search_results:
        doc = result.get("doc", {})
        normalized_doc = get_doc_topic_text(doc)
        exact_phrase_hit = any(
            contains_normalized_phrase(normalized_doc, phrase)
            for phrase in topic_phrases
            if len(phrase.split()) >= min_phrase_length
        )
        match_count = sum(1 for t in unique_tokens if re.search(rf"\b{re.escape(t)}\b", normalized_doc))
        if not match_count and not exact_phrase_hit:
            unmatched.append(result)
            continue
        coverage = match_count / max(len(unique_tokens), 1)
        if is_single_token:
            if not exact_phrase_hit:
                doc_subject = normalize_text(doc.get("subject", ""))
                doc_major = normalize_text(doc.get("major", ""))
                doc_title = normalize_text(doc.get("title", ""))
                token = unique_tokens[0]
                if token not in doc_subject and token not in doc_major and token not in doc_title:
                    unmatched.append(result)
                    continue
        if prefer_strict and len(unique_tokens) >= 2 and coverage < strict_threshold and not exact_phrase_hit:
            unmatched.append(result)
            continue
        if not prefer_strict and not exact_phrase_hit and coverage < relaxed_threshold:
            unmatched.append(result)
            continue
        doc_subject = normalize_text(doc.get("subject", ""))
        doc_major = normalize_text(doc.get("major", ""))
        doc_title = normalize_text(doc.get("title", ""))
        subject_boost = 0.0
        for t in unique_tokens:
            if doc_subject and t in doc_subject:
                subject_boost += 25.0
            if doc_major and t in doc_major:
                subject_boost += 20.0
            if doc_title and t in doc_title:
                subject_boost += 10.0
        phrase_boost = 20.0 if exact_phrase_hit else 0.0
        relevance = (match_count * 3.0) + (coverage * 8.0) + phrase_boost + subject_boost
        matched.append({**result, "score": result.get("score", 0.0) + relevance})

    if matched:
        matched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        if prefer_strict:
            return matched
        unmatched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return matched + unmatched
    return [] if prefer_strict else search_results


def clean_display_value(value, max_length: int = 220) -> str:
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
        items = [part.strip().strip("'\"") for part in text.strip("[]").split(",")]
        text = ", ".join(item for item in items if item)
    text = re.sub(r"\s+", " ", text).strip(" ;,")
    if not text:
        return ""
    if normalize_text(text) in {"n a", "na", "unknown", "khong xac dinh", "nan", "none", "null"}:
        return ""
    if len(text) > max_length:
        return text[:max_length].rstrip() + "..."
    return text
