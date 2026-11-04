"""
lib/semantic_router.py — Semantic Intent Router & Dialogue State Updater.
Solves complex multi-turn conversational scenarios without manual one-off regex patching:
- Select Top-K from working set (e.g. 'Chọn 3 tài liệu phù hợp nhất')
- Table of Contents / Chapter boundary inquiry (e.g. 'Cuốn này có bao nhiêu chương?', 'Chương 3 nói về gì?')
- Multi-attribute inquiry on current book (e.g. 'Cho biết tên tác giả và năm xuất bản')
- Multiple doc-type filtering (e.g. 'Chỉ lấy sách hoặc giáo trình')
- Constraint clearing / resetting (e.g. 'Không giới hạn năm xuất bản nữa')
- Topic expansion / merging (e.g. 'Bổ sung chủ đề quản trị nhân lực')
- Narrowing down topics (e.g. 'Thu hẹp vào trí tuệ nhân tạo')
- Academic pedagogical recommendation (e.g. 'Cho biết tài liệu nào phù hợp với sinh viên năm cuối')
- Cumulative multi-turn synthesis and grouping
"""

import re
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from .text_utils import normalize_text
from .dialogue_state import ConversationState
from .doc_type_resolver import detect_multiple_doc_types_from_query
from .logging_setup import logger


class RouterActionType(str, Enum):
    CATALOG_SEARCH = "catalog_search"         # Tìm kiếm mới trong CSDL
    SELECT_TOP_K = "select_top_k"             # Chọn K cuốn từ kết quả vừa tìm
    FILTER_UPDATE = "filter_update"           # Cập nhật điều kiện lọc (năm, ngôn ngữ, loại tài liệu)
    CLEAR_FILTER = "clear_filter"             # Xóa/hủy điều kiện lọc (năm, ngôn ngữ, ngành...)
    ADD_TOPIC = "add_topic"                   # Bổ sung/gộp chủ đề mới vào chủ đề cũ
    BOOK_ATTRIBUTE = "book_attribute"         # Hỏi thuộc tính của cuốn sách vừa tìm
    BOOK_TOC_INQUIRY = "book_toc_inquiry"     # Hỏi số chương / nội dung chương (giải thích giới hạn)
    ACADEMIC_ADVICE = "academic_advice"       # Tư vấn học thuật định tính (SV năm cuối, tự học...)
    SYNTHESIS = "synthesis"                   # Phân loại, tổng hợp tài liệu theo nhóm
    POLICY = "policy"                         # Giờ mở cửa, mượn trả, làm thẻ
    CHITCHAT = "chitchat"                     # Chào hỏi, cảm ơn


@dataclass
class RoutedAction:
    action_type: RouterActionType
    target_k: int = 3
    attributes: List[str] = field(default_factory=list)
    doc_types: List[str] = field(default_factory=list)
    topic: Optional[str] = None
    major: Optional[str] = None
    author: Optional[str] = None
    language: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    clear_targets: List[str] = field(default_factory=list)
    raw_query: str = ""
    clean_query: str = ""
    advice_prompt: str = ""


def route_conversational_turn(query: str, state: ConversationState) -> RoutedAction:
    """
    Phân loại ý định và trích xuất tham số từ câu hỏi người dùng dựa trên ngữ cảnh phiên hội thoại (DST).
    Sử dụng so khớp trên chuỗi chuẩn hóa unaccented để bao quát cả gõ có dấu và không dấu.
    """
    if not query:
        return RoutedAction(action_type=RouterActionType.CHITCHAT, raw_query=query)

    q = query.strip()
    q_norm = normalize_text(q).strip().lower()

    # ── 1. SELECT_TOP_K: "Chọn 3 tài liệu phù hợp nhất", "lấy 3 cuốn hay nhất"... ──
    # Chỉ kích hoạt khi đã có danh sách tài liệu trước đó (working_set)
    m_top_k = re.search(
        r"^(?:vay\s+|hay\s+)?(?:chon|lay|loc\s+ra|dua\s+ra|cho\s+(?:toi|minh|em)|goi\s+y)\s*(\d+|vai|mot\s+so|may)?\s*"
        r"(?:cuon|quyen|sach|tai\s+lieu|ket\s+qua)?\s*(?:phu\s+hop|tot|hay|tieu\s+bieu|noi\s+bat)?(?:\s+nhat)?\s*[?.]*$",
        q_norm
    )
    if m_top_k and state.working_set:
        k_str = m_top_k.group(1)
        k_val = 3
        if k_str and k_str.isdigit():
            k_val = max(1, min(10, int(k_str)))
        logger.info("SemanticRouter -> SELECT_TOP_K (k=%d) from working_set len=%d", k_val, len(state.working_set))
        return RoutedAction(
            action_type=RouterActionType.SELECT_TOP_K,
            target_k=k_val,
            raw_query=q,
        )

    # ── 2. CLEAR_FILTER: "Không giới hạn năm xuất bản nữa", "bỏ lọc năm"... ──
    clear_year = bool(re.search(
        r"\b(?:khong|bo|huy)\s+(?:can\s+)?(?:gioi\s+han|loc|rang\s+buoc)?\s*(?:ve\s+)?nam(?:\s+xuat\s+ban)?(?:\s+nua)?\b|"
        r"\b(?:nam\s+nao\s+cung\s+duoc|tat\s+ca\s+cac\s+nam|moi\s+nam)\b",
        q_norm
    ))
    clear_major = bool(re.search(r"\b(?:khong|bo)\s+(?:gioi\s+han|loc)?\s*nganh\b", q_norm))
    clear_doc_type = bool(re.search(r"\b(?:khong|bo)\s+(?:gioi\s+han|loc)?\s*loai(?:\s+sach|\s+tai\s+lieu)?\b", q_norm))

    if clear_year or clear_major or clear_doc_type:
        targets = []
        if clear_year:
            targets.append("year")
        if clear_major:
            targets.append("major")
        if clear_doc_type:
            targets.append("doc_type")
        logger.info("SemanticRouter -> CLEAR_FILTER targets=%s", targets)
        return RoutedAction(
            action_type=RouterActionType.CLEAR_FILTER,
            clear_targets=targets,
            raw_query=q,
        )

    # ── 3. BOOK_TOC_INQUIRY: "Cuốn sách này có bao nhiêu chương?", "Chương 3 nói về gì?" ──
    is_toc_query = bool(re.search(
        r"\b(?:bao\s+nhieu\s+chuong|co\s+may\s+chuong|so\s+chuong|muc\s+luc|noi\s+dung\s+cac\s+chuong)\b|"
        r"\b(?:chuong\s+\d+|chuong\s+[ivx]+|phan\s+\d+)\s*(?:noi\s+ve|viet\s+ve|ban\s+ve|co\s+gi|la\s+gi|la\s+van\s+de\s+gi)\b|"
        r"\bnoi\s+dung\s+chuong\s+\d+\b",
        q_norm
    ))
    if is_toc_query:
        logger.info("SemanticRouter -> BOOK_TOC_INQUIRY for %r", q)
        return RoutedAction(
            action_type=RouterActionType.BOOK_TOC_INQUIRY,
            raw_query=q,
        )

    # ── 4. BOOK_ATTRIBUTE: "Cho biết tên tác giả và năm xuất bản" ──
    # Tránh để author_resolver bắt nhầm "Và Năm Xuất Bản" làm tên tác giả!
    is_multi_attr = bool(re.search(
        r"\b(?:cho\s+biet|cho\s+xin|thong\s+tin\s+ve)?\s*(?:ten\s+)?tac\s+gia\s+(?:va|cung|voi)\s+(?:nam\s+xuat\s+ban|nam\s+xb|nam)\b|"
        r"\b(?:ai\s+viet|ai\s+la\s+tac\s+gia)\s+(?:va|vao)\s+(?:nam\s+nao|xuat\s+ban\s+nam\s+nao)\b",
        q_norm
    ))
    if is_multi_attr:
        logger.info("SemanticRouter -> BOOK_ATTRIBUTE (author, year) for %r", q)
        return RoutedAction(
            action_type=RouterActionType.BOOK_ATTRIBUTE,
            attributes=["author", "year"],
            raw_query=q,
        )

    # ── 5. ACADEMIC_ADVICE: "Cho biết tài liệu nào phù hợp với sinh viên năm cuối" ──
    is_advice = bool(re.search(
        r"\b(?:phu\s+hop|thich\s+hop|danh\s+cho|nen\s+doc|huong\s+den)\s+(?:voi\s+|cho\s+)?(?:sinh\s+vien\s+)?(?:nam\s+cuoi|tot\s+nghiep|khoa\s+luan|nghien\s+cuu\s+sau\s+dai\s+hoc)\b|"
        r"\b(?:cuon\s+nao\s+phu\s+hop\s+(?:nhat\s+)?(?:cho\s+|de\s+)?(?:sinh\s+vien\s+nam\s+cuoi|tu\s+hoc|nguoi\s+moi))\b|"
        r"\b(?:tai\s+lieu\s+nao\s+phu\s+hop\s+voi\s+sinh\s+vien\s+nam\s+cuoi)\b",
        q_norm
    ))
    if is_advice and state.working_set:
        logger.info("SemanticRouter -> ACADEMIC_ADVICE for %r", q)
        return RoutedAction(
            action_type=RouterActionType.ACADEMIC_ADVICE,
            advice_prompt=q,
            raw_query=q,
        )

    # ── 6. NARROWING DOWN TOPIC: "Thu hẹp vào trí tuệ nhân tạo" ──
    m_narrow = re.search(r"^(?:thu\s+hep\s+(?:vao|ve|sang)|chuyen\s+sau\s+ve|tap\s+trung\s+vao)\s+(.+)$", q_norm)
    if m_narrow:
        start_idx = m_narrow.start(1)
        narrowed_topic = q[start_idx:].strip().rstrip(".?!")
        logger.info("SemanticRouter -> NARROW_TOPIC %r", narrowed_topic)
        return RoutedAction(
            action_type=RouterActionType.CATALOG_SEARCH,
            topic=narrowed_topic,
            raw_query=q,
        )

    # ── 7. ADD_TOPIC: "Bổ sung chủ đề quản trị nhân lực", "tìm thêm chủ đề X" ──
    m_add_topic = re.search(
        r"^(?:bo\s+sung|them|tim\s+them)(?:\s+ve|\s+chu\s+de|\s+chuyen\s+de)?\s+(.+)$",
        q_norm
    )
    if m_add_topic and state.primary_topic and not any(k in q_norm for k in ["tieng anh", "tieng viet", "tai lieu tieng"]):
        start_idx = m_add_topic.start(1)
        new_topic = q[start_idx:].strip().rstrip(".?!")
        new_topic = re.sub(r"^(?:chủ\s+đề|về\s+chủ\s+đề|về)\s+", "", new_topic, flags=re.IGNORECASE).strip()
        if new_topic and len(new_topic) >= 2:
            logger.info("SemanticRouter -> ADD_TOPIC new=%r (existing=%r)", new_topic, state.primary_topic)
            return RoutedAction(
                action_type=RouterActionType.ADD_TOPIC,
                topic=new_topic,
                raw_query=q,
            )

    # Detect multi doc types early for subsequent filters
    multi_types = detect_multiple_doc_types_from_query(q)

    # ── 8. FILTER_UPDATE: Ràng buộc năm tiếp nối ("Chỉ lấy sách xuất bản từ năm 2020 trở đi") ──
    m_yr_filter = re.search(
        r"(?:tu|sau|tu\s+sau|vao|trong)?\s*nam\s*(18\d{2}|19\d{2}|20\d{2})\s*(?:tro\s+lai\s+day|tro\s+ve\s+day|tro\s+len|tro\s+di|den\s+nay)?",
        q_norm
    )
    if m_yr_filter and state.primary_topic:
        yr = int(m_yr_filter.group(1))
        logger.info("SemanticRouter -> FILTER_UPDATE year_min=%d (doc_types=%s) for topic=%r", yr, multi_types, state.primary_topic)
        return RoutedAction(
            action_type=RouterActionType.FILTER_UPDATE,
            year_min=yr,
            doc_types=multi_types,
            raw_query=q,
        )

    # ── 9. FILTER_UPDATE: Ngôn ngữ tiếp nối ("Chuyển sang tìm tài liệu tiếng Việt", "Ưu tiên tài liệu tiếng Việt") ──
    m_lang_filter = re.search(
        r"^(?:chuyen\s+sang\s+(?:tim\s+)?|uu\s+tien\s+|chi\s+lay\s+|tim\s+them\s+)?(?:sach\s+|tai\s+lieu\s+)?"
        r"(tieng\s+viet|bang\s+tieng\s+viet|tieng\s+anh|bang\s+tieng\s+anh|song\s+ngu)\s*[?.]*$",
        q_norm
    )
    if m_lang_filter and state.primary_topic:
        lang_str = m_lang_filter.group(1)
        lang_code = "vi" if "viet" in lang_str else ("en" if "anh" in lang_str else "bilingual")
        logger.info("SemanticRouter -> FILTER_UPDATE language=%s for topic=%r", lang_code, state.primary_topic)
        return RoutedAction(
            action_type=RouterActionType.FILTER_UPDATE,
            language=lang_code,
            raw_query=q,
        )

    # ── 10. FILTER_UPDATE: Loại tài liệu phức hợp ("sách hoặc giáo trình") ──
    is_pure_type_filter = bool(re.search(r"^(?:chi\s+lay|chi\s+tim|lay|loc|chon|uu\s+tien)\s+", q_norm)) and len(multi_types) >= 1
    if is_pure_type_filter and state.primary_topic:
        logger.info("SemanticRouter -> FILTER_UPDATE doc_types=%s for topic=%r", multi_types, state.primary_topic)
        return RoutedAction(
            action_type=RouterActionType.FILTER_UPDATE,
            doc_types=multi_types,
            raw_query=q,
        )

    # ── 11. Fallback: CATALOG_SEARCH hoặc các intent thông thường ──
    return RoutedAction(
        action_type=RouterActionType.CATALOG_SEARCH,
        raw_query=q,
    )
