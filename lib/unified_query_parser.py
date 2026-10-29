"""
lib/unified_query_parser.py — Unified Query Parser & Intent Dispatcher.
Single source of truth for parsing natural language queries into structured intent,
entities (author, topic, title), and constraints (facets, sorting, refinement).
"""

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .text_utils import normalize_text
from .reranking import (
    detect_author_query,
    extract_author_name,
    detect_publisher_query,
    extract_publisher_name,
)
from .doc_type_resolver import (
    detect_doc_type_from_query,
    clean_query_doc_type_terms,
)
from .major_resolver import (
    resolve_canonical_major_from_query,
)
from .conversation_resolver import (
    detect_recency_preference,
    detect_oldest_preference,
    strip_temporal_sorting_phrases,
    clean_conversational_query,
    is_follow_up_refinement,
    get_inherited_context,
)
from .catalog import (
    detect_year_constraint,
    is_pure_year_query,
    extract_primary_topic_text,
    strip_search_intent_phrases,
)


@dataclass
class ParsedCatalogQuery:
    raw_query: str
    clean_topic: str
    search_query: str
    is_author_query: bool = False
    author_name: str = ""
    is_publisher_query: bool = False
    publisher_name: str = ""
    is_pure_year: bool = False
    pure_year_val: Optional[int] = None
    pure_year_dir: Optional[str] = None
    year_filter_val: Optional[int] = None
    year_filter_dir: Optional[str] = None
    doc_type: Optional[str] = None
    major: Optional[str] = None
    recency_sort: bool = False
    oldest_sort: bool = False
    is_refinement: bool = False
    is_singular: bool = False
    teaching_subject: Optional[str] = None
    course_name: Optional[str] = None


def parse_catalog_query(
    query: str,
    history: Optional[List[Dict[str, Any]]] = None,
    explicit_major: Optional[str] = None,
    teaching_subject: Optional[str] = None,
    course_name: Optional[str] = None,
) -> ParsedCatalogQuery:
    """
    Phân tích câu truy vấn tự nhiên thành một cấu trúc ParsedCatalogQuery duy nhất,
    đầy đủ và nhất quán. Loại bỏ sự phân mảnh giữa nhiều hàm bóc tách độc lập.
    """
    q_raw = (query or "").strip()
    history = history or []

    # 1. Phát hiện các yêu cầu sắp xếp thời gian (Mới nhất / Cũ nhất)
    is_recency = detect_recency_preference(q_raw)
    is_oldest = detect_oldest_preference(q_raw)

    # 2. Phát hiện ràng buộc Năm xuất bản
    year_val, year_dir = detect_year_constraint(q_raw)
    is_pure_year, pure_year_val, pure_year_dir = is_pure_year_query(q_raw)

    # 3. Phát hiện loại tài liệu (Doc Type)
    doc_type = detect_doc_type_from_query(q_raw)

    # 4. Phát hiện ngành đào tạo (Major)
    canonical_major = explicit_major
    if not canonical_major:
        maj_match = resolve_canonical_major_from_query(q_raw)
        if maj_match:
            canonical_major = maj_match[0]

    # 5. Phát hiện Tác giả / Nhà xuất bản
    q_no_temporal = strip_temporal_sorting_phrases(q_raw)
    is_author = detect_author_query(q_no_temporal)
    author_name = extract_author_name(q_no_temporal) if is_author else ""

    is_publisher = detect_publisher_query(q_no_temporal)
    publisher_name = extract_publisher_name(q_no_temporal) if is_publisher else ""

    # 6. Phát hiện ngữ cảnh kế thừa (Multi-turn Refinement)
    is_refine = is_follow_up_refinement(q_raw)

    # 7. Bóc tách chủ đề cốt lõi sạch (Clean Topic)
    if is_author:
        clean_topic = ""
        search_q = author_name
    elif is_publisher:
        clean_topic = ""
        search_q = publisher_name
    elif is_pure_year and pure_year_val:
        if pure_year_dir == "after":
            clean_topic = f"xuất bản từ năm {pure_year_val}"
        elif pure_year_dir == "before":
            clean_topic = f"xuất bản trước năm {pure_year_val}"
        else:
            clean_topic = f"xuất bản năm {pure_year_val}"
        search_q = str(pure_year_val)
    else:
        clean_topic = clean_conversational_query(q_raw)
        # Khử lặp các tiền tố "chủ đề", "về chủ đề"
        clean_topic = re.sub(r'^(?:về\s+)?chủ\s+đề\s+', '', clean_topic, flags=re.IGNORECASE).strip()
        # Loại bỏ các từ chỉ số lượng / mạo từ đầu topic: "các", "những", "một số"
        clean_topic = re.sub(r'^(?:các|những|một\s+số)\s+', '', clean_topic, flags=re.IGNORECASE).strip()

        # Nếu câu hỏi chứa ngành và phần còn lại chỉ là ngành đó, không coi ngành là clean_topic riêng
        if canonical_major and clean_topic:
            maj_norm = normalize_text(canonical_major)
            topic_norm = normalize_text(clean_topic)
            rem = re.sub(r'\b(?:thuoc\s+|cua\s+)?(?:nganh|chuyen\s+nganh|khoa)?\s*' + re.escape(maj_norm) + r'\b', '', topic_norm, flags=re.IGNORECASE)
            rem = re.sub(r'\b(?:thuoc\s+|cua\s+)?(?:nganh|chuyen\s+nganh|khoa)\b', '', rem, flags=re.IGNORECASE)
            rem = re.sub(r'\b(?:xuat\s+ban\s+)?(?:nam\s+)?\d{4}\b', '', rem, flags=re.IGNORECASE)
            rem = re.sub(r'\b(?:xuat\s+ban|phat\s+hanh|in)\b', '', rem, flags=re.IGNORECASE).strip()
            if len(rem) < 3:
                clean_topic = ""

        # Nếu là câu tinh chỉnh tiếp nối và không có topic mới -> kế thừa từ lịch sử
        if is_refine and history:
            inherited = get_inherited_context(history)
            inherited_topic = inherited.get("topic") or inherited.get("search_query") or ""
            if inherited_topic:
                clean_topic = inherited_topic

        # 8. Xác định Search Query chính cho Search Engine
        search_q = clean_topic or extract_primary_topic_text(q_raw) or q_raw

        # Loại bỏ doc_type terms khỏi search_q nếu doc_type không phải Kiểm định
        if doc_type and doc_type != "Kiểm định":
            c_dt = clean_query_doc_type_terms(search_q, doc_type)
            if c_dt and len(c_dt.strip()) >= 2:
                search_q = c_dt.strip()

        # Loại bỏ năm và cụm từ thời gian khỏi search_q
        if year_val and search_q:
            search_q = re.sub(r'\b(19\d{2}|20\d{2})\b', ' ', search_q)
            search_q = re.sub(r'\b(?:có\s+)?(?:xuất\s+bản|phát\s+hành|in|xuat\s+ban|phat\s+hanh)?\s*(?:từ|sau|trước|vào|tu|sau|truoc|vao)?\s*(?:năm|nam)?\b', ' ', search_q, flags=re.IGNORECASE)
            search_q = re.sub(r'\s+', ' ', search_q).strip()

        if (is_recency or is_oldest) and search_q:
            c_rec = strip_temporal_sorting_phrases(search_q)
            if c_rec and len(c_rec.strip()) >= 2:
                search_q = c_rec.strip()

        if not search_q and canonical_major:
            search_q = canonical_major

    # Phát hiện số ít (quyển / cuốn cụ thể)
    is_singular = bool(re.search(r'\b(?:quyển|cuốn)\b', q_raw, re.IGNORECASE) and not re.search(r'\b(?:những|các|danh sách)\b', q_raw, re.IGNORECASE))

    return ParsedCatalogQuery(
        raw_query=q_raw,
        clean_topic=clean_topic,
        search_query=search_q,
        is_author_query=is_author,
        author_name=author_name,
        is_publisher_query=is_publisher,
        publisher_name=publisher_name,
        is_pure_year=is_pure_year,
        pure_year_val=pure_year_val,
        pure_year_dir=pure_year_dir,
        year_filter_val=year_val,
        year_filter_dir=year_dir,
        doc_type=doc_type,
        major=canonical_major,
        recency_sort=is_recency,
        oldest_sort=is_oldest,
        is_refinement=is_refine,
        is_singular=is_singular,
        teaching_subject=teaching_subject,
        course_name=course_name,
    )
