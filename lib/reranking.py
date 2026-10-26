"""
lib/reranking.py — Search result reranking, author/publisher matching, PDF collapse.
"""

import re
from typing import Optional
from .text_utils import normalize_text, clean_author_candidate
from .catalog import (
    get_query_topic_terms, get_doc_topic_text, contains_normalized_phrase,
    VIETNAMESE_STOP_WORDS,
    detect_ai_domain, score_ai_document,
)


def detect_author_query(query: str) -> bool:
    if not query:
        return False
    q = normalize_text(query)

    # ── Pattern 1: Explicit "tác giả" markers ──
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
        "cao", "mac", "ha", "ho", "kieu", "tang", "dong", "bac", "kha",
        "thach", "dieu", "chau", "nghiem", "diep", "thai", "to",
        "lam", "luu", "phung", "vu", "ma", "ton", "dai",
        "phan", "truong", "vuong", "chu", "ta", "luong",
    }
    topic_words = {
        "te", "vi", "mo", "hoc", "phap", "luat", "su", "hoa",
        "van", "nghe", "thuat", "cong", "nghiep", "xay", "dung",
        "nong", "kinh", "tai", "chinh", "ngan", "giao", "duc",
        "tam", "ly", "xa", "hoi", "tri", "tue", "nhan", "tao",
        "may", "tinh", "phan", "mem", "dien", "tu", "vien",
        "thong", "bao", "truyen", "sinh", "y", "duoc",
        "ky", "thuat", "nganh", "chu", "de", "linh", "vuc",
        "lap", "trinh", "python", "java", "web", "ai", "data", "sql",
        "co", "ban", "nang", "cao", "ung", "dung", "he", "thong",
    }

    words = q.split()

    # Nếu câu có từ "về" (về chủ đề...) thì không phải là tìm kiếm theo tác giả
    if "ve" in words:
        # Trừ khi có từ khóa "tác giả" rõ ràng
        if not any(re.search(p, q) for p in author_markers):
            return False

    # ── Pattern 2: "sách của <họ tên>", "tài liệu của <họ tên>" ──
    cua_patterns = [
        r"\bsach\s+cua\s+(.+)",
        r"\btai lieu\s+cua\s+(.+)",
        r"\bgiao trinh\s+cua\s+(.+)",
        r"\bbai\s+cua\s+(.+)",
    ]
    for pattern in cua_patterns:
        m = re.search(pattern, q)
        if m:
            name_part = m.group(1).strip()
            name_tokens = name_part.split()
            if len(name_tokens) >= 2 and name_tokens[0] in VIETNAMESE_SURNAMES:
                return True

    # ── Pattern 3: "của <họ> ..." ở bất kỳ vị trí nào trong câu ──
    for i, w in enumerate(words):
        if w == "cua" and i + 1 < len(words):
            remaining = words[i+1:]
            if len(remaining) >= 2 and remaining[0] in VIETNAMESE_SURNAMES:
                if not any(t in topic_words for t in remaining[1:]):
                    return True
            elif len(remaining) == 1 and remaining[0] in VIETNAMESE_SURNAMES:
                return True

    # ── Pattern 4: Câu ngắn bắt đầu bằng họ (relaxed từ 2-6 từ) ──
    if 2 <= len(words) <= 6 and words[0] in VIETNAMESE_SURNAMES:
        # Kiểm tra từ cuối cùng (tên chính) không phải từ chủ đề
        # Bỏ qua tên đệm (vd "văn", "thị", "xuân") vì là thành phần tên người
        if words[-1] not in topic_words and not any(w in topic_words for w in words[1:]):
            return True

    # ── Pattern 5: Phát hiện tác giả ở cuối câu dạng "tìm sách <họ> ..." ──
    # Một họ Việt Nam xuất hiện ở cuối, các từ trước là hành động/mục tiêu (KHÔNG có từ "về" / topic marker)
    if len(words) >= 3 and not any(w in {"ve", "chu", "de", "nganh"} for w in words):
        for idx, w in enumerate(words):
            if w in VIETNAMESE_SURNAMES and idx >= 1:
                preceding = words[:idx]
                name_part = words[idx:]
                # Từ ngay trước họ phải là catalog noun ("sach", "lieu", "cuon", "quyen")
                if preceding[-1] in {"sach", "lieu", "cuon", "quyen"}:
                    if len(name_part) >= 2 and len(name_part) <= 4:
                        if not any(t in topic_words for t in name_part[1:]):
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
        "cao", "mac", "ha", "ho", "kieu", "tang", "dong", "bac", "kha",
        "thach", "dieu", "chau", "nghiem", "diep", "thai", "to",
        "lam", "luu", "phung", "vu", "ma", "ton", "dai",
        "phan", "truong", "vuong", "chu", "ta", "luong",
    }
    topic_words = {
        "te", "vi", "mo", "hoc", "phap", "luat", "su", "hoa",
        "van", "nghe", "thuat", "cong", "nghiep", "xay", "dung",
        "nong", "kinh", "tai", "chinh", "ngan", "giao", "duc",
        "tam", "ly", "xa", "hoi", "tri", "tue", "nhan", "tao",
        "may", "tinh", "phan", "mem", "dien", "tu", "vien",
        "thong", "bao", "truyen", "sinh", "y", "duoc",
        "ky", "thuat", "nganh", "chu", "de", "linh", "vuc",
        "lap", "trinh", "python", "java", "web", "ai", "data", "sql",
        "co", "ban", "nang", "cao", "ung", "dung", "he", "thong",
    }
    patterns = [
        r"(?:sach|cua|tai lieu|giao trinh)\s+tac gia\s+(.+)",
        r"tac gia\s+(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            name = clean_author_candidate(match.group(1))
            if len(re.findall(r'\w+', name)) >= 2 and len(name) >= 5:
                return name

    # Nếu câu có từ "về" (về chủ đề...) thì không phải là tìm theo tác giả
    words = q.split()
    if "ve" in words:
        return ""

    name_ref_patterns = [
        r"(?:sach|tai lieu|giao trinh|bai bao|bai)\s+cua\s+(.+)",
        r"cua\s+(.+)",
    ]
    for pattern in name_ref_patterns:
        match = re.search(pattern, q)
        if match:
            name = clean_author_candidate(match.group(1))
            name_tokens = name.split()
            if len(name_tokens) >= 2 and name_tokens[0] in VIETNAMESE_SURNAMES:
                if not any(t in topic_words for t in name_tokens[1:]):
                    return name

    # ── Pattern 5: "sách <họ> ...", "tìm sách <họ> ..." (không có "của") ──
    # Xoá các từ chủ ngữ (toi/minh/em...) và action/target ở đầu để lộ tên tác giả
    q_stripped = re.sub(
        r"^(?:toi|minh|em|anh|chi|ban|tao|may|no|ho|chung|cac ban|may)\s+",
        "", q
    ).strip()
    q_stripped = re.sub(
        r"^(?:tim|tra|co|muon|xem|liet ke|hay|vui long|xin)\s+(?:sach|tai lieu|giao trinh|bai bao|bai|cuon|quyen)?\s*",
        "", q_stripped
    ).strip()
    # Nếu còn dư "sach", "tai lieu", "giao trinh" ở đầu sau lần strip
    q_stripped = re.sub(
        r"^(?:sach|tai lieu|giao trinh|bai bao|bai|cuon|quyen)\s+",
        "", q_stripped
    ).strip()
    # Nếu còn dư "cua" ở đầu
    q_stripped = re.sub(r"^cua\s+", "", q_stripped).strip()
    if q_stripped and q_stripped != q:
        name_tokens = q_stripped.split()
        if 2 <= len(name_tokens) <= 5 and name_tokens[0] in VIETNAMESE_SURNAMES:
            if not any(t in topic_words for t in name_tokens[1:]):
                return q_stripped

    if 2 <= len(words) <= 6 and words[0] in VIETNAMESE_SURNAMES:
        if not any(w in topic_words for w in words[1:]):
            return q
    if 2 <= len(words) <= 6:
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
    # Xoá dấu phẩy (MARC format "Last, First" → "Last First")
    normalized = re.sub(r'[,\.;:]+', ' ', normalized)
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

    # ── 1. Exact match ──
    if doc_norm == query_norm:
        return True

    # ── 2. So sánh tập hợp token (không phụ thuộc thứ tự) ──
    #    Xử lý đúng cho cả MARC "Lâm, Hồ Văn" ↔ query "Hồ Văn Lâm"
    doc_set = set(doc_tokens)
    query_set = set(query_tokens)
    if not doc_set or not query_set:
        return False

    # ── 2a. Query có 2 token: yêu cầu EXACT set match ──
    #    Xử lý cả "luong hien" ↔ "hien luong" (MARC reversed 2-token)
    if len(query_tokens) == 2:
        if query_set == doc_set:
            return True

    # ── 2b. Subset check (query_set phải là subset của doc_set) ──
    #    Không chấp nhận chiều ngược lại: doc_set.issubset(query_set) cho phép
    #    doc có ít token hơn query → false positive ("Nguyễn Hữu Nguyên" vs "Nguyễn Hữu Trọn")
    if query_set.issubset(doc_set):        # Với 2-token query và doc >2 tokens: yêu cầu query tokens khớp
        # ĐẦU và CUỐI của doc_tokens (surname + given name position).
        # Tránh false positive: "Lương Hiền" vs "Trương Hiền Lượng"
        #   doc_tokens=["truong","hien","luong"] → doc[0]="truong", doc[-1]="luong"
        #   query_tokens={"luong","hien"} → chỉ "luong" khớp doc[-1], thiếu "truong" → reject
        if len(query_tokens) == 2 and len(doc_tokens) > 2:
            doc_first = doc_tokens[0]
            doc_last = doc_tokens[-1]
            if (query_tokens[0] == doc_first and query_tokens[1] == doc_last
                    and len(doc_tokens) - len(query_tokens) <= 1):
                return True
            return False
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
            doc_key = doc.get("doc_id") or id(doc)
            if doc_key not in merged or result.get("score", 0.0) > merged[doc_key].get("score", 0.0):
                merged[doc_key] = result
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


def rerank_results_for_query(query: str, search_results: list[dict], prefer_strict: bool = False, target_doc_type: Optional[str] = None) -> list[dict]:
    if not search_results:
        return search_results

    if target_doc_type:
        from .doc_type_resolver import doc_matches_target_doc_type
        for result in search_results:
            doc = result.get("doc", {})
            if doc_matches_target_doc_type(doc, target_doc_type):
                result["score"] = result.get("score", 0.0) + 450.0

    author_name = extract_author_name(query) if detect_author_query(query) else ""
    publisher_name = extract_publisher_name(query) if detect_publisher_query(query) else ""

    if author_name:
        author_results = []
        non_author_results = []
        for result in search_results:
            doc = result.get("doc", {})
            primary_author = doc.get("author", "")
            if prefer_strict:
                matches = author_name_matches(primary_author, author_name)
            else:
                all_author_fields = [primary_author]
                co_authors = doc.get("co_authors") or []
                if isinstance(co_authors, list):
                    all_author_fields.extend(co_authors)
                authors_list = doc.get("authors") or []
                if isinstance(authors_list, list):
                    all_author_fields.extend(authors_list)
                matches = any(author_name_matches(a, author_name) for a in all_author_fields if a)
            if matches:
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
    min_phrase_length = 2
    if len(unique_tokens) >= 6:
        strict_threshold = 0.80
        relaxed_threshold = 0.40
    elif len(unique_tokens) >= 3:
        strict_threshold = 0.65
        relaxed_threshold = max(0.15, min(0.40, 3.0 / len(unique_tokens)))
    elif len(unique_tokens) == 2:
        strict_threshold = 0.90
        relaxed_threshold = 0.90
    else:
        strict_threshold = 0.50
        relaxed_threshold = 0.50
    is_single_token = len(unique_tokens) == 1

    # Diacritic awareness
    from .catalog import VIETNAMESE_STOP_WORDS
    q_accented_tokens = [
        t.lower() for t in re.findall(r'[a-zA-Zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúũụưừứửữựỳýỷỹỵđ]+', query)
        if len(t) >= 2 and normalize_text(t) not in VIETNAMESE_STOP_WORDS
    ]
    has_accents = any(t != normalize_text(t) for t in q_accented_tokens)
    accented_phrase = " ".join(q_accented_tokens) if len(q_accented_tokens) >= 2 else ""

    for result in search_results:
        doc = result.get("doc", {})
        normalized_doc = get_doc_topic_text(doc)
        exact_phrase_hit = any(
            contains_normalized_phrase(normalized_doc, phrase)
            for phrase in topic_phrases
            if len(phrase.split()) >= min_phrase_length
        )
        match_count = sum(1 for t in unique_tokens if re.search(rf"\b{re.escape(t)}\b", normalized_doc))
        is_target_doc_match = bool(target_doc_type and doc_matches_target_doc_type(doc, target_doc_type))
        if not match_count and not exact_phrase_hit and not is_target_doc_match:
            unmatched.append(result)
            continue
        coverage = match_count / max(len(unique_tokens), 1)
        if is_single_token and not is_target_doc_match:
            if not exact_phrase_hit:
                doc_subject = normalize_text(doc.get("subject", ""))
                doc_major = normalize_text(doc.get("major", ""))
                doc_title = normalize_text(doc.get("title", ""))
                token = unique_tokens[0]
                if token not in doc_subject and token not in doc_major and token not in doc_title:
                    unmatched.append(result)
                    continue
        if not is_target_doc_match:
            if prefer_strict and len(unique_tokens) >= 2 and coverage < strict_threshold and not exact_phrase_hit:
                unmatched.append(result)
                continue
            if not prefer_strict and not exact_phrase_hit and coverage < relaxed_threshold:
                unmatched.append(result)
                continue
        doc_subject = normalize_text(doc.get("subject", ""))
        doc_major = normalize_text(doc.get("major", ""))
        doc_title = normalize_text(doc.get("title", ""))
        raw_title = str(doc.get("title", "") or "").lower()
        raw_subject = str(doc.get("subject", "") or "").lower()
        raw_abstract = str(doc.get("abstract", "") or "").lower()
        raw_content = f"{raw_title} {raw_subject} {raw_abstract}"

        subject_boost = 0.0
        for t in unique_tokens:
            if doc_subject and t in doc_subject:
                subject_boost += 25.0
            if doc_major and t in doc_major:
                subject_boost += 20.0
            if doc_title and t in doc_title:
                subject_boost += 10.0

        phrase_boost = 0.0
        for phrase in topic_phrases:
            if len(phrase.split()) >= min_phrase_length:
                if contains_normalized_phrase(doc_title, phrase):
                    phrase_boost += 250.0
                    break
                elif contains_normalized_phrase(doc_subject, phrase):
                    phrase_boost += 200.0
                    break
                elif exact_phrase_hit:
                    phrase_boost += 100.0
                    break

        exact_word_boost = 0.0
        if accented_phrase and re.search(rf"(?<!\w){re.escape(accented_phrase)}(?!\w)", raw_title):
            exact_word_boost += 350.0
        elif accented_phrase and re.search(rf"(?<!\w){re.escape(accented_phrase)}(?!\w)", raw_content):
            exact_word_boost += 150.0

        for at in q_accented_tokens:
            if re.search(rf"\b{re.escape(at)}\b", raw_title):
                exact_word_boost += 60.0
            elif re.search(rf"\b{re.escape(at)}\b", raw_subject):
                exact_word_boost += 30.0
            elif re.search(rf"\b{re.escape(at)}\b", raw_abstract):
                exact_word_boost += 15.0

        relevance = (match_count * 3.0) + (coverage * 8.0) + phrase_boost + subject_boost + exact_word_boost
        matched.append({**result, "score": result.get("score", 0.0) + relevance})

    if matched:
        matched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        if prefer_strict:
            return matched
        if len(matched) < 5:
            unmatched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return matched + unmatched[:max(0, 5 - len(matched))]
        return matched
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
