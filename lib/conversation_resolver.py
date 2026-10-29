"""
lib/conversation_resolver.py — Multi-turn conversation resolution:
- Ordinal references ("cuốn đầu tiên", "cuốn thứ 2", "cuốn cuối cùng"...)
- Anaphoric / pronoun references ("cuốn này", "sách đó", "tác giả này"...)
- Book attribute Q&A (author, year, publisher, location, call_number, summary, borrow)
- Conversational context inheritance (doc_type, major, author, search_query)
- Recency / Newest / Oldest sorting ("mới nhất", "năm gần đây nhất", "có năm xuất bản gần nhất"...)
- Clarification / Reformulation detection ("ý là...", "ý mình là...", "không phải, ý là...")
"""

import re
from typing import Optional
from .text_utils import normalize_text


ORDINAL_MAP = [
    # Book 1
    (r"\b(?:cuon\s+dau|quyen\s+dau|dau\s+tien|thu\s+nhat|thu\s+1|so\s+1|(?:cuon|quyen|sach|tai lieu)\s+1)\b", 0),
    # Book 2
    (r"\b(?:thu\s+hai|thu\s+2|so\s+2|(?:cuon|quyen|sach|tai lieu)\s+2)\b", 1),
    # Book 3
    (r"\b(?:thu\s+ba|thu\s+3|so\s+3|(?:cuon|quyen|sach|tai lieu)\s+3)\b", 2),
    # Book 4
    (r"\b(?:thu\s+tu|thu\s+4|so\s+4|(?:cuon|quyen|sach|tai lieu)\s+4)\b", 3),
    # Book 5
    (r"\b(?:thu\s+nam|thu\s+5|so\s+5|(?:cuon|quyen|sach|tai lieu)\s+5)\b", 4),
    # Last book
    (r"\b(?:cuon\s+cuoi|quyen\s+cuoi|sach\s+cuoi|cuoi\s+cung)\b", -1),
]

ANAPHORIC_PATTERNS = [
    r"\b(?:cuon|quyen|sach|tai lieu|an ban|tap)\s+(?:nay|do|ay|kia|tren)\b",
    r"\b(?:tac gia|nam xb|nxb|vi tri|tom tat)\s+(?:cuon|quyen|sach|tai lieu)?\s*(?:nay|do|ay)\b",
    r"^(?:no|sach do|cuon do|quyen do)\b",
    r"\b(?:ve cuon nay|ve sach nay|ve tai lieu nay)\b",
]

AUTHOR_QUESTION_PATTERNS = [
    r"\b(?:ai viet|ai la tac gia|tac gia la ai|tac gia nao|do ai viet|do ai sang tac|do ai bien soan|nguoi viet|tac gia cua)\b",
]

YEAR_QUESTION_PATTERNS = [
    r"\b(?:xuat ban nam nao|nam xuat ban nao|nam nao|xuat ban khi nao|ra doi nam nao|nam phat hanh nao|phat hanh nam nao|san xuat nam nao|in nam nao)\b",
]

PLACE_QUESTION_PATTERNS = [
    r"\b(?:noi xuat ban|xuat ban o dau|xuat ban tai dau|in o dau|dia diem xuat ban)\b",
]

PUBLISHER_QUESTION_PATTERNS = [
    r"\b(?:nha xuat ban nao|nxb nao|do ai xuat ban|nxb gi|don vi xuat ban|nha in)\b",
]

LOCATION_QUESTION_PATTERNS = [
    r"\b(?:o dau|phong nao|kho nao|vi tri|tim o dau|lay o dau|ma xep gia|ky hieu xep gia|ma so|ke nao|gia nao|tang may)\b",
]

SUMMARY_QUESTION_PATTERNS = [
    r"\b(?:noi ve gi|noi ve cai gi|tom tat|gioi thieu|chu de gi|noi dung chinh|noi dung gi|viet ve gi|noi dung ra sao)\b",
]

BORROW_QUESTION_PATTERNS = [
    r"\b(?:co muon duoc khong|muon ve duoc khong|cho muon khong|muon nhu the nao|muon ve nha|doc tai cho)\b",
]

DETAIL_QUESTION_PATTERNS = [
    r"\b(?:chi tiet|xem chi tiet|cho xem|thong tin cuon|xem cuon|thong tin chi tiet|thong tin tai lieu)\b",
]

RECENCY_PATTERNS = [
    r"\b(?:co\s+)?nam\s+(?:xuat\s+ban\s+)?(?:gan\s+nhat|gan\s+day\s+nhat|moi\s+nhat)\b",
    r"\b(?:nam\s+)?(?:gan\s+day\s+nhat|gan\s+nhat|moi\s+nhat|moi\s+xuat\s+ban|vua\s+xuat\s+ban|xuat\s+ban\s+gan\s+day)\b",
    r"\b(?:tai\s+ban|an\s+ban|phien\s+ban)\s+(?:moi\s+nhat|gan\s+nhat)\b",
    r"\b(?:moi\s+nhat\s+hien\s+nay|moi\s+nhat)\b",
    r"\b(?:moi\s+ra|moi\s+nhap|vua\s+ve)\b",
]

OLDEST_PATTERNS = [
    r"\b(?:co\s+)?nam\s+(?:xuat\s+ban\s+)?(?:cu\s+nhat|som\s+nhat|lau\s+doi\s+nhat)\b",
    r"\b(?:cu\s+nhat|som\s+nhat|lau\s+nhat|lau\s+doi\s+nhat)\b",
]

CLARIFICATION_PREFIXES = [
    r"^(?:khong\s+phai|khong|ko\s+phai|ko)\s*[,.-]?\s*(?:y\s+(?:la|toi\s+la|minh\s+la|em\s+la)|y\s+noi\s+la|y\s+o\s+day\s+la)\s*",
    r"^(?:y\s+(?:la|toi\s+la|minh\s+la|em\s+la)|y\s+noi\s+la|y\s+o\s+day\s+la|nghia\s+la|tuc\s+la)\s*",
    r"^(?:y\s+(?:toi|minh|em)\s+muon\s+(?:tim|noi|hoi)\s+la)\s*",
    r"^(?:minh|toi|em)\s+(?:muon\s+tim|can\s+tim|muon\s+hoi)\s+la\s*",
]


def detect_recency_preference(query: str) -> bool:
    """
    Phát hiện yêu cầu tìm sách mới nhất / gần đây nhất / có năm xuất bản gần nhất:
    'sách về quản trị kinh doanh có năm xuất bản gần nhất', 'mới nhất', 'gần đây nhất'...
    """
    if not query:
        return False
    q = normalize_text(query).strip().lower()
    for pat in RECENCY_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def detect_oldest_preference(query: str) -> bool:
    """Phát hiện yêu cầu tìm sách cũ nhất / sớm nhất."""
    if not query:
        return False
    q = normalize_text(query).strip().lower()
    for pat in OLDEST_PATTERNS:
        if re.search(pat, q):
            return True
    return False


def strip_temporal_sorting_phrases(query: str) -> str:
    """
    Loại bỏ các cụm từ chỉ thời gian, độ mới/cũ/sớm/lâu đời (cả có dấu và không dấu)
    để giữ lại chủ đề cốt lõi: 'mới nhất', 'gần đây nhất', 'lâu đời nhất', 'cũ nhất', 'sớm nhất'...
    """
    if not query:
        return ""
    q = query
    patterns = [
        # Recency (mới nhất / gần đây)
        r"(?:có\s+)?năm\s+(?:xuất\s+bản\s+)?(?:gần\s+nhất|gần\s+đây\s+nhất|mới\s+nhất)",
        r"(?:năm\s+)?(?:gần\s+đây\s+nhất|gần\s+nhất|mới\s+nhất|mới\s+xuất\s+bản|vừa\s+xuất\s+bản|xuất\s+bản\s+gần\s+đây(?:\s+nhất)?)",
        r"(?:tái\s+bản|ấn\s+bản|phiên\s+bản)\s+(?:mới\s+nhất|gần\s+nhất)",
        r"(?:mới\s+nhất\s+hiện\s+nay|mới\s+nhất|gần\s+đây(?:\s+nhất)?)",
        # Oldest (cũ nhất / lâu đời nhất / sớm nhất)
        r"(?:có\s+)?năm\s+(?:xuất\s+bản\s+)?(?:cũ\s+nhất|sớm\s+nhất|lâu\s+đời\s+nhất|lâu\s+nhất)",
        r"(?:cũ\s+nhất|sớm\s+nhất|lâu\s+đời\s+nhất|lâu\s+nhất|đời\s+đầu|xa\s+xưa)",
        # Unaccented variants
        r"\b(?:co\s+)?nam\s+(?:xuat\s+ban\s+)?(?:gan\s+nhat|gan\s+day(?:\s+nhat)?|moi\s+nhat|cu\s+nhat|som\s+nhat|lau\s+doi\s+nhat|lau\s+nhat)\b",
        r"\b(?:nam\s+)?(?:gan\s+day\s+nhat|gan\s+nhat|moi\s+nhat|moi\s+xuat\s+ban|vua\s+xuat\s+ban|xuat\s+ban\s+gan\s+day(?:\s+nhat)?|cu\s+nhat|som\s+nhat|lau\s+nhat|lau\s+doi\s+nhat)\b",
        r"\b(?:tai\s+ban|an\s+ban|phien\s+ban)\s+(?:moi\s+nhat|gan\s+nhat)\b",
        r"\b(?:moi\s+nhat\s+hien\s+nay|moi\s+nhat|gan\s+day(?:\s+nhat)?|doi\s+dau)\b",
    ]
    for pat in patterns:
        q = re.sub(pat, " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def strip_recency_phrases(query: str) -> str:
    """Loại bỏ các cụm từ chỉ tính thời sự/mới nhất/cũ nhất (backward-compatible alias)."""
    return strip_temporal_sorting_phrases(query)


def clean_conversational_query(query: str) -> str:
    """
    Bóc tách các lớp vỏ ngôn ngữ tự nhiên:
    - Tiền tố chào hỏi / lịch sự: 'dạ em chào anh chị ad ạ', 'cho mình hỏi', 'ad ơi'...
    - Khung câu hỏi bao bọc: 'có sách nào liên quan đến ... không', 'tìm giúp mình'...
    - Câu hỏi tinh chỉnh/lựa chọn tiếp nối: 'vậy lựa ra quyển...', 'chọn cuốn mới nhất'...
    - Hậu tố vị trí / thắc mắc phụ: 'hiện tại còn ở phòng... không', 'không vậy ạ'...
    - Cụm từ thời gian siêu cấp: 'có năm xuất bản lâu đời/mới nhất'...
    Trả về chuỗi từ khóa chủ đề cốt lõi sạch nhất cho Search Engine.
    Nếu câu là câu tinh chỉnh tiếp nối thuần túy (không có chủ đề mới), trả về chuỗi rỗng để kế thừa turn trước.
    """
    if not query:
        return ""
    q = query.strip()

    # 1. Bóc tiền tố giao tiếp & khung câu hỏi & hành vi lựa chọn tiếp nối & bối cảnh nghiên cứu
    prefixes = [
        r"^(?:dạ\s+)?(?:em\s+|mình\s+|tôi\s+)?(?:xin\s+)?chào\s+(?:các\s+)?(?:anh\s+|chị\s+|em\s+|thầy\s+|cô\s+|ad\s+|bot\s+|thư viện\s+|mọi người\s+|bạn\s+|ạ\s*|,\s*|[.-]\s*)+",
        r"^(?:dạ\s+)?(?:chào\s+(?:bạn|ad|bot|thư viện|thầy cô|anh chị)?\s*[,.-]?\s*)",
        r"^(?:ad|bot|bạn)\s+ơi\s*[,.-]?\s*",
        r"^(?:dạ\s+)?(?:làm\s+ơn\s+)?(?:cho\s+(?:em|mình|tôi)\s+hỏi(?:\s+với|\s+ạ)?\s*[,.-]?\s*)",
        r"^(?:cho\s+(?:em|mình|tôi)\s+(?:xin|biết)\s*)",
        r"^(?:bạn\s+có\s+thể\s+)?(?:tìm|tra\s+cứu|cho\s+mình\s+xin|giúp\s+(?:mình|em|tôi)\s+tìm)\s*",
        # Tiền tố nhu cầu / bối cảnh nghiên cứu / học tập: "tôi đang cần tìm hiểu về...", "mình đang làm đề tài...", "em đang nghiên cứu..."
        r"^(?:dạ\s+)?(?:tôi|em|mình)\s+(?:đang\s+)?(?:cần|muốn|chuẩn\s+bị|phải)?\s*(?:tìm\s+hiểu|nghiên\s+cứu|làm|viết|thực\s+hiện|hoàn\s+thành|tham\s+khảo|đọc|xem|học)(?:\s+thêm)?(?:\s+về)?\s*",
        # Khung đề tài / chuyên đề / luận văn: "đề tài nghiên cứu khoa học về chủ đề...", "đề tài về...", "chuyên đề...", "đồ án tốt nghiệp..."
        r"^(?:đề\s+tài\s+(?:nghiên\s+cứu\s+(?:khoa\s+học\s+)?)?|chuyên\s+đề|luận\s+văn|khóa\s+luận|bài\s+tập\s+lớn|đồ\s+án(?:\s+tốt\s+nghiệp)?)\s*(?:về\s+chủ\s+đề|thuộc\s+chủ\s+đề|chủ\s+đề|về)?\s*",
        r"^(?:vậy\s+(?:thì\s+)?|thế\s+(?:thì\s+)?|vậy\s+còn\s+|thế\s+còn\s+)?(?:lựa\s+(?:ra\s+)?|chọn\s+(?:ra\s+)?|lọc\s+(?:ra\s+)?|lấy\s+(?:ra\s+)?|chỉ\s+(?:lấy|xem|chọn)\s+|xem\s+|cho\s+(?:xem|mình\s+xem|tôi\s+xem)\s+|tìm\s+(?:ra\s+)?)(?:những\s+|các\s+|một\s+|1\s+|vài\s+|mấy\s+)?(?:quyển|cuốn|sách|tài\s+liệu|tập|bản)?\s*",
        r"^(?:vậy\s+(?:thì\s+)?|thế\s+(?:thì\s+)?|vậy\s+còn\s+|thế\s+còn\s+)\s*",
        r"^(?:có\s+)?(?:các\s+|những\s+|một\s+số\s+)?(?:sách|tài\s+liệu|giáo\s+trình|cuốn|quyển)\s+(?:nào\s+)?(?:liên\s+quan\s+đến|về\s+chủ\s+đề|thuộc\s+chủ\s+đề|chủ\s+đề|về|nói\s+về)?\s*",
        r"^(?:cuốn|quyển|sách|tài\s+liệu|giáo\s+trình)\s+(?:về\s+chủ\s+đề|thuộc\s+chủ\s+đề|chủ\s+đề|về|nói\s+về|nào\s+)?\s*",
        r"^(?:về\s+chủ\s+đề|thuộc\s+chủ\s+đề|chủ\s+đề|về|nói\s+về)\s+",
    ]

    # 2. Hậu tố câu hỏi, yêu cầu gợi ý, mục đích bổ trợ & trạng thái
    suffixes = [
        # Hậu tố yêu cầu gợi ý / xin sách / tài liệu: ", bạn hãy gợi ý cho tôi những sách liên quan...", ", ad gợi ý giúp em vài tài liệu tham khảo với ạ"
        r"[,.]?\s*(?:ad|bot|bạn)?\s*(?:ơi\s*[,.]?)?\s*(?:hãy\s+)?(?:gợi\s+ý|đề\s+xuất|giới\s+thiệu|tìm|cho\s+(?:tôi|mình|em)\s+(?:xin|biết))(?:\s+(?:cho|giúp)\s+(?:tôi|mình|em))?\s+(?:những\s+|các\s+|một\s+số\s+|vài\s+|ít\s+)?(?:sách|tài\s+liệu|giáo\s+trình|tập|cuốn|quyển)?\s*(?:tham\s+khảo|liên\s+quan|phù\s+hợp)?\s*(?:để\s+(?:giúp\s+)?(?:tôi|mình|em)?\s*(?:có\s+thể\s+)?(?:hoàn\s+thiện|hoàn\s+thành|làm|nghiên\s+cứu|viết|học|tham\s+khảo|phục\s+vụ).*)?\s*(?:với\s+ạ|ạ|nhé|nha)?\s*[?.]*$",
        # Hậu tố mục đích bổ trợ: "để giúp tôi có thể hoàn thiện đề tài này", "để phục vụ nghiên cứu"...
        r"\s*(?:để\s+(?:giúp\s+)?(?:tôi|mình|em)?\s*(?:có\s+thể\s+)?(?:hoàn\s+thiện|hoàn\s+thành|làm|nghiên\s+cứu|viết|học|tham\s+khảo|phục\s+vụ).*)?\s*[?.]*$",
        r"\s*(?:hiện\s+tại\s+)?(?:còn|có)\s+(?:ở\s+)?(?:phòng|kho|kệ|giá|thư viện)\s+.*$",
        r"\s*(?:có\s+)?năm\s+(?:xuất\s+bản\s+)?(?:cũ|sớm|lâu\s+đời|lâu|gần|gần\s+đây|mới)\s+nhất\s*(?:không)?\s*[?.]*$",
        r"\s*\b(?:cũ|sớm|lâu\s+đời|lâu|gần\s+đây|mới\s+nhất|mới)\s+nhất\s*(?:không)?\s*[?.]*$",
        r"\s*(?:không\s+vậy\s+ạ|không\s+ạ|không\s+nhỉ|được\s+không\s*(?:ạ)?|với\s+ạ|không|ạ|nhé|nha)\s*[?.]*$",
    ]

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            m = re.sub(p, "", q, count=1, flags=re.IGNORECASE).strip()
            if m and m != q:
                q = m
                changed = True
        for s in suffixes:
            m = re.sub(s, "", q, count=1, flags=re.IGNORECASE).strip()
            if m and m != q:
                q = m
                changed = True

    # 3. Xóa các cụm từ chỉ thời gian / siêu cấp
    q = strip_temporal_sorting_phrases(q)

    # Bóc các cấu trúc lọc năm hoặc ngành anaphoric nếu chỉ là câu lọc tiếp nối
    q_stripped = re.sub(r'\b(?:có\s+)?(?:xuất\s+bản|phát\s+hành|in)?\s*(?:từ|sau|trước|vào)?\s*(?:năm\s+)?(19\d{2}|20\d{2})\b', ' ', q, flags=re.IGNORECASE)
    q_stripped = re.sub(r'\b(?:thuộc\s+|của\s+)?(?:ngành|chuyên\s+ngành|khoa)\s+(?:này|đó|trên)\b', ' ', q_stripped, flags=re.IGNORECASE)
    q_stripped = re.sub(r'\b(?:xuất\s+bản|phát\s+hành|in)\b', ' ', q_stripped, flags=re.IGNORECASE)
    q_stripped = re.sub(r'\s+', ' ', q_stripped).strip()
    if len(q_stripped) < 3 and is_follow_up_refinement(query):
        return ""

    if not q:
        if is_follow_up_refinement(query):
            return ""
        return query.strip()
    return q


def parse_natural_query(query: str) -> dict:
    """
    Unified Natural Language Query Parser:
    Trích xuất cấu trúc ngữ nghĩa từ câu hỏi tự nhiên:
    - clean_query: chủ đề sạch đã gọt vỏ
    - recency_sort: có yêu cầu mới nhất
    - oldest_sort: có yêu cầu cũ nhất / lâu đời nhất
    """
    if not query:
        return {"raw_query": "", "clean_query": "", "recency_sort": False, "oldest_sort": False}
    return {
        "raw_query": query,
        "clean_query": clean_conversational_query(query),
        "recency_sort": detect_recency_preference(query),
        "oldest_sort": detect_oldest_preference(query),
    }


def sort_results_by_year(results: list[dict], descending: bool = True, top_k_relevant: int = 50) -> list[dict]:
    """
    Sắp xếp kết quả theo năm xuất bản với cơ chế RELEVANCE GATING:
    - Nếu danh sách có nhiều kết quả, CHỈ lấy nhóm ứng viên có độ liên quan cao nhất (Top K candidates)
      để sắp xếp theo năm, tránh việc tài liệu rác (score ~0.001) xuất bản năm 1970 chen lên đầu.
    - Sắp xếp từ mới đến cũ (descending=True) hoặc từ cũ đến mới (descending=False).
    - Các tài liệu có năm hợp lệ (1900-2030) luôn đứng trên các tài liệu không có năm.
    """
    if not results:
        return results

    # Relevance Gating: chỉ sắp xếp trong nhóm ứng viên có độ liên quan cao
    if len(results) > top_k_relevant:
        candidates = results[:top_k_relevant]
        remaining = results[top_k_relevant:]
    else:
        candidates = results
        remaining = []

    def extract_year(item: dict) -> int:
        doc = item.get("doc", item) if isinstance(item, dict) else item
        y_val = doc.get("year", 0) if isinstance(doc, dict) else 0
        if not y_val or str(y_val).lower().strip() in ("n/a", "unknown", "none", "", "nan"):
            return 0 if descending else 9999
        try:
            val = int(str(y_val).strip())
            return val if 1900 <= val <= 2030 else (0 if descending else 9999)
        except Exception:
            return 0 if descending else 9999

    sorted_candidates = sorted(candidates, key=extract_year, reverse=descending)
    return sorted_candidates + remaining


def extract_clarification_content(query: str) -> tuple[bool, str]:
    """
    Phát hiện và trích xuất nội dung làm rõ khi người dùng nói 'ý là...', 'ý mình là...', 'không phải, ý là...'
    Trả về (True, cleaned_content) hoặc (False, query gốc).
    """
    if not query:
        return False, query
    q = query.strip()
    q_norm = normalize_text(q).strip().lower()

    for pat in CLARIFICATION_PREFIXES:
        m = re.search(pat, q_norm)
        if m:
            cleaned = q
            for prefix_term in [
                "không phải, ý là", "không phải ý là", "ko phải ý là", "không, ý là", "ko, ý là",
                "ý mình muốn tìm là", "ý tôi muốn tìm là", "ý em muốn tìm là",
                "ý mình là", "ý tôi là", "ý em là", "ý là", "ý nói là", "nghĩa là", "tức là"
            ]:
                if cleaned.lower().startswith(prefix_term):
                    cleaned = cleaned[len(prefix_term):].strip()
                    break
            cleaned = re.sub(r"^[,.:;-]\s*", "", cleaned).strip()
            if cleaned:
                return True, cleaned
            return True, q[m.end():].strip()

    return False, query


def detect_ordinal_reference(query: str, sources: list[dict]) -> tuple[int | None, dict | None]:
    """
    Phát hiện thứ tự sách được người dùng nhắc đến ('cuốn đầu tiên', 'cuốn thứ 2'...).
    Trả về (index, book_dict) nếu tìm thấy, hoặc (None, None).
    Nếu index phát hiện được nhưng vượt quá len(sources), trả về (index, None).
    """
    if not query:
        return None, None

    q = normalize_text(query).strip().lower()

    for pattern, idx in ORDINAL_MAP:
        if re.search(pattern, q):
            if not sources:
                return idx, None
            target_idx = idx if idx >= 0 else len(sources) - 1
            if 0 <= target_idx < len(sources):
                return target_idx, sources[target_idx]
            return target_idx, None

    return None, None


def detect_anaphoric_reference(query: str, sources: list[dict]) -> dict | None:
    """
    Phát hiện đại từ chỉ định thay thế cho cuốn sách vừa nhắc tới ('cuốn này', 'sách đó'...).
    Nếu có và danh sách sources trước đó có sách, trả về cuốn đầu tiên.
    """
    if not query or not sources:
        return None

    q = normalize_text(query).strip().lower()

    for pattern in ANAPHORIC_PATTERNS:
        if re.search(pattern, q):
            return sources[0]

    return None


def detect_attribute_question(query: str) -> str | None:
    """
    Nhận diện trường thông tin mà người dùng đang hỏi về một cuốn sách:
    - 'author': Ai viết / tác giả
    - 'year': Năm xuất bản (chỉ khi hỏi năm xuất bản của cuốn sách, không phải recency sorting)
    - 'publisher': Nhà xuất bản
    - 'location': Phòng / vị trí / mã xếp giá
    - 'summary': Tóm tắt / nội dung
    - 'borrow': Quy định mượn về / đọc tại chỗ
    - 'detail': Xem chi tiết cuốn sách
    """
    if not query:
        return None

    # Nếu câu hỏi là tìm kiếm theo độ mới nhất / gần nhất thì không phải hỏi thuộc tính sách đơn lẻ
    if detect_recency_preference(query) or detect_oldest_preference(query):
        return None

    q = normalize_text(query).strip().lower()

    for pattern in AUTHOR_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "author"

    for pattern in YEAR_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "year"

    for pattern in PLACE_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "place"

    for pattern in PUBLISHER_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "publisher"

    for pattern in LOCATION_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "location"

    for pattern in SUMMARY_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "summary"

    for pattern in BORROW_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "borrow"

    for pattern in DETAIL_QUESTION_PATTERNS:
        if re.search(pattern, q):
            return "detail"

    return None


def build_book_attribute_answer(book: dict, attr: str) -> str:
    """
    Tạo câu trả lời trực tiếp, rõ ràng cho câu hỏi thuộc tính về một cuốn sách cụ thể.
    """
    meta = book.get("metadata", book)
    title = str(meta.get("title") or "Tài liệu").strip().rstrip('.')
    author = str(meta.get("author") or "").strip()
    year = str(meta.get("year") or "").strip()
    publisher = str(meta.get("publisher") or "").strip()
    place = str(meta.get("place") or "").strip()
    loc = str(meta.get("location") or "").strip()
    call_num = str(meta.get("call_number") or meta.get("call_no") or meta.get("ddc") or "").strip()
    abstract = str(meta.get("abstract") or meta.get("description") or "").strip()
    subject = str(meta.get("subject") or "").strip()
    doc_type = str(meta.get("doc_type") or "").strip()
    major = str(meta.get("major") or "").strip()

    def valid(v: str) -> bool:
        return bool(v and v.lower() not in {"unknown", "n/a", "không xác định", "nan", "none"})

    if attr == "author":
        if valid(author):
            extra = f" (năm xuất bản: {year})" if valid(year) else ""
            return f"Tác giả của cuốn sách **{title}** là **{author}**{extra}."
        return f"Hiện tại dữ liệu thư viện chưa ghi nhận rõ tên tác giả cho cuốn sách **{title}**."

    if attr == "year":
        if valid(year):
            pub_info = f" bởi **Nhà xuất bản {publisher}**" if valid(publisher) else ""
            return f"Cuốn sách **{title}** được xuất bản năm **{year}**{pub_info}."
        return f"Hiện tại thư viện chưa có thông tin chính xác về năm xuất bản của cuốn sách **{title}**."

    if attr == "place":
        if valid(place  ):
            pub_info = f" (NXB {publisher})" if valid(publisher) else ""
            year_info = f", năm {year}" if valid(year) else ""
            return f"Cuốn sách **{title}** được xuất bản tại **{place}**{pub_info}{year_info}."
        return f"Hiện tại thư viện chưa có thông tin nơi xuất bản của cuốn sách **{title}**."

    if attr == "publisher":
        if valid(publisher):
            year_info = f" vào năm {year}" if valid(year) else ""
            return f"Cuốn sách **{title}** do **Nhà xuất bản {publisher}** phát hành{year_info}."
        return f"Hiện tại thư viện chưa có thông tin nhà xuất bản của cuốn sách **{title}**."

    if attr == "location":
        parts = []
        if valid(loc):
            parts.append(f"vị trí tại **{loc}**")
        if valid(call_num):
            parts.append(f"mã xếp giá **{call_num}**")
        if parts:
            return f"Cuốn sách **{title}** hiện có {' với '.join(parts)} tại Thư viện QNU."
        return f"Bạn có thể liên hệ quầy thủ thư Thư viện QNU để được hỗ trợ mượn cuốn **{title}**."

    if attr == "summary":
        if valid(abstract) and len(abstract) >= 15:
            return f"**Nội dung / Tóm tắt cuốn sách \"{title}\":**\n\n{abstract}"
        if valid(subject):
            return f"Cuốn sách **{title}** thuộc chủ đề: **{subject}**."
        return f"Hiện tại chưa có phần tóm tắt chi tiết cho cuốn sách **{title}** trong hệ thống."

    if attr == "borrow":
        loc_str = f" tại **{loc}**" if valid(loc) else ""
        call_str = f" (Mã xếp giá: **{call_num}**)" if valid(call_num) else ""
        return (
            f"Tài liệu **{title}** hiện có lưu trữ{loc_str}{call_str} tại Thư viện QNU. "
            f"Bạn có thể mang thẻ sinh viên / giảng viên đến quầy thủ thư để làm thủ tục mượn về nhà "
            f"hoặc đọc tại chỗ theo quy định của thư viện."
        )

    if attr == "detail":
        lines = [f"**Thông tin chi tiết tài liệu: {title}**"]
        if valid(author):
            lines.append(f"- **Tác giả:** {author}")
        if valid(year):
            lines.append(f"- **Năm xuất bản:** {year}")
        if valid(place):
            lines.append(f"- **Nơi xuất bản:** {place}")
        if valid(publisher):
            lines.append(f"- **Nhà xuất bản:** {publisher}")
        if valid(loc):
            lines.append(f"- **Vị trí lưu trữ:** {loc}")
        if valid(call_num):
            lines.append(f"- **Mã xếp giá / DDC:** {call_num}")
        if valid(doc_type):
            lines.append(f"- **Loại tài liệu:** {doc_type}")
        if valid(major):
            lines.append(f"- **Chuyên ngành:** {major}")
        if valid(subject):
            lines.append(f"- **Chủ đề:** {subject}")
        if valid(abstract) and len(abstract) >= 15:
            lines.append(f"\n**Tóm tắt nội dung:**\n{abstract}")
        return "\n".join(lines)

    return f"Thông tin về cuốn sách **{title}**: Tác giả {author}, năm xuất bản {year}."


def build_book_follow_ups(book: dict, current_attr: str) -> list[dict]:
    """Tạo gợi ý tương tác tiếp theo cho cuốn sách đang được hỏi."""
    meta = book.get("metadata", book)
    title = str(meta.get("title") or "").strip().rstrip('.')
    author = str(meta.get("author") or "").strip()
    follow_ups = []

    if current_attr != "summary":
        follow_ups.append({
            "label": "Xem tóm tắt nội dung sách",
            "query": f"tóm tắt nội dung cuốn {title}" if title else "tóm tắt nội dung sách"
        })

    if current_attr != "location":
        follow_ups.append({
            "label": "Vị trí phòng đọc của sách",
            "query": f"cuốn {title} ở phòng nào" if title else "sách này để ở phòng nào"
        })

    if current_attr != "year":
        follow_ups.append({
            "label": "Năm xuất bản của sách",
            "query": f"sách đó xuất bản năm nào"
        })

    if author and author.lower() not in {"unknown", "n/a", "không xác định"}:
        clean_a = re.sub(r'[\(\[].*?[\)\]]', '', author).split(';')[0].split(',')[0].strip()
        if clean_a:
            follow_ups.append({
                "label": f"Tìm sách của tác giả: {clean_a}",
                "query": f"sách của tác giả {clean_a}"
            })

    return follow_ups[:3]


def detect_exhaustion_query(query: str) -> bool:
    """
    Phát hiện các câu hỏi muốn xem thêm / còn nữa không / liệt kê tiếp:
    'còn sách nào khác không?', 'có cuốn nào nữa không?', 'liệt kê tiếp', 'xem thêm'...
    """
    if not query:
        return False
    q = normalize_text(query).strip().lower()
    phrases = [
        "het chua", "con khong", "con tai lieu nao", "con them khong",
        "con nhieu khong", "het tai lieu", "het sach", "het ket qua",
        "con bao nhieu", "con mot nua", "con gi khong",
        "con nua khong", "con nua ko", "con nua k",
        "con sach nao khac", "con sach nao nua", "co sach nao khac",
        "co cuon nao khac", "co cuon nao nua", "sach khac nua", "cuon khac nua",
        "con tai lieu nao khac", "co tai lieu nao khac",
        "xem them", "xem tiep", "hien thi them", "hien thi tiep",
        "liet ke them", "liet ke tiep", "cho xem them", "cho xem tiep",
        "con cuon nao", "con sach nao", "khong con nua a", "het roi a",
    ]
    return any(p in q for p in phrases)


def is_follow_up_filter(query: str) -> bool:
    """
    Phát hiện câu hỏi lọc tiếp nối theo ngành, năm, ngôn ngữ:
    'Có của ngành Công nghệ thông tin không?', 'Thế còn ngành Luật?', 'Có năm 2021 không?'...
    """
    if not query:
        return False
    q = normalize_text(query).strip().lower()
    prefixes = (
        "co cua", "con cua", "the con", "con nganh", "co nganh",
        "cua nganh", "thuoc nganh", "cua khoa", "thuoc khoa",
        "co nam", "cua nam", "xuat ban nam",
        "co tieng", "bang tieng",
    )
    if any(q.startswith(p) for p in prefixes):
        return True
    suffixes = (
        "thi sao", "co khong", "con khong", "khong",
    )
    if ("nganh " in q or "khoa " in q or "nam " in q) and any(q.endswith(s) for s in suffixes):
        return True
    return False


def is_follow_up_refinement(query: str) -> bool:
    """
    Phát hiện câu hỏi hội thoại tiếp nối tinh chỉnh/lọc/chọn từ kết quả trước:
    - 'vậy lựa ra quyển có năm xuất bản gần nhất'
    - 'chọn cuốn mới nhất'
    - 'vậy cuốn nào mới nhất'
    - 'thế cuốn cũ nhất'
    - 'lọc ra những cuốn từ năm 2020'
    - 'lấy quyển xuất bản gần đây nhất'
    - 'trong số các cuốn trên thì cuốn nào mới nhất'
    - 'chỉ lấy giáo trình thôi'
    - 'có cuốn nào của ngành CNTT không'
    """
    if not query:
        return False
    q = normalize_text(query).strip().lower()

    # Has a refinement signal (temporal, action verb, connective, filter, in-set reference)
    has_recency = detect_recency_preference(query)
    has_oldest = detect_oldest_preference(query)
    has_year = bool(re.search(r'\b(19\d{2}|20\d{2})\b', q))
    has_refine_verb = bool(re.search(r'\b(?:lua\s+(?:ra\s+)?|chon\s+(?:ra\s+)?|loc\s+(?:ra\s+)?|lay\s+(?:ra\s+)?|chi\s+(?:lay|xem|chon)|xem\s+|cho\s+xem|tim\s+(?:ra\s+)?)\b', q))
    has_connective = bool(re.search(r'^(?:vay\s+|the\s+|vay\s+thi\s+|the\s+thi\s+|the\s+con\s+|vay\s+con\s+)', q))
    has_filter = is_follow_up_filter(query)
    has_inset = bool(re.search(r'\b(?:trong\s+(?:so\s+do|so\s+nay|do|cac\s+sach\s+tren|danh\s+sach\s+tren))\b', q))

    if not (has_recency or has_oldest or has_year or has_refine_verb or has_connective or has_filter or has_inset):
        return False

    # Check whether an independent new subject topic is present
    s = strip_temporal_sorting_phrases(q)
    s = re.sub(r'^(?:vay\s+(?:thi\s+)?|the\s+(?:thi\s+)?|vay\s+con\s+|the\s+con\s+|cho\s+(?:minh|toi|em)\s+|hay\s+|nho\s+ban\s+)?\s*', ' ', s)
    s = re.sub(r'\b(?:lua\s+(?:ra\s+)?|chon\s+(?:ra\s+)?|loc\s+(?:ra\s+)?|lay\s+(?:ra\s+)?|chi\s+(?:lay|xem|chon)|xem\s+(?:them\s+)?|tim\s+(?:ra\s+)?|cho\s+(?:xem|biet))\b', ' ', s)
    s = re.sub(r'\b(?:trong\s+(?:so\s+do|so\s+nay|do|cac\s+sach\s+tren|danh\s+sach\s+tren))\b', ' ', s)
    s = re.sub(r'\b(?:co\s+)?(?:xuat\s+ban|phat\s+hanh|in)?\s*(?:tu\s+|truoc\s+|sau\s+|trong\s+|vao\s+)?(?:nam\s+)?(19\d{2}|20\d{2})\b', ' ', s)
    s = re.sub(r'\b(?:xuat\s+ban|phat\s+hanh|xuat|in)\b', ' ', s)
    s = re.sub(r'\b(?:nhung\s+|cac\s+|mot\s+|1\s+|vai\s+|may\s+)?(?:quyen|cuon|sach|tai\s+lieu|tap|an\s+ban|(?<!xuat\s)ban|ket\s+qua)(?:\s+(?:nao|nay|do|kia|ay|o\s+tren))?\b', ' ', s)
    s = re.sub(r'\b(?:loai\s+)?(?:sach|giao\s+trinh|luan\s+van|tai\s+lieu\s+tham\s+khao|kiem\s+dinh)\b', ' ', s)
    s = re.sub(r'\b(?:thuoc\s+|cua\s+)?(?:nganh|chuyen\s+nganh|khoa)(?:\s+(?:nay|do|kia|ay|tren))?\b', ' ', s)
    s = re.sub(r'\b(?:bang\s+|tieng\s+)?(?:tieng\s+anh|tieng\s+viet|song\s+ngu|english|vietnamese)\b', ' ', s)
    s = re.sub(r'\b(?:nao|gi|o\s+dau|khong|ko|nhi|ha|the|vay|a|nhe|voi|nhung|co|ra|cua|duoc|thi\s+sao|nay|do|kia|ay)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()

    # If remaining text has less than 3 characters or is empty, no new standalone topic was provided
    return len(s) < 3


def is_singular_book_request(query: str) -> bool:
    """
    Kiểm tra câu hỏi có hướng tới 1 tài liệu đơn lẻ cụ thể không:
    'lựa ra quyển...', 'chọn cuốn...', 'cuốn nào mới nhất', 'quyển nào gần nhất'...
    """
    if not query:
        return False
    q = normalize_text(query).strip().lower()
    patterns = [
        r"\b(?:lua\s+ra|chon\s+ra|lay\s+ra|lay|chon|lua)\s+(?:1\s+|mot\s+)?(?:quyen|cuon)\b",
        r"\b(?:quyen|cuon)\s+nao\b",
        r"\b(?:1\s+|mot\s+)(?:quyen|cuon|sach|tai lieu)\b",
        r"\b(?:quyen|cuon)\s+(?:co\s+)?(?:nam\s+)?(?:xuat\s+ban\s+)?(?:gan\s+nhat|moi\s+nhat|cu\s+nhat)\b",
    ]
    return any(re.search(pat, q) for pat in patterns)


def get_inherited_context(history: list) -> dict:
    """
    Trích xuất ngữ cảnh kế thừa từ lịch sử hội thoại gần nhất:
    - doc_type (ví dụ 'Kiểm định')
    - major (ví dụ 'Công nghệ thông tin')
    - author (ví dụ 'Nguyễn Nhật Ánh')
    - topic
    - search_query
    - last_sources (danh sách các sách vừa hiển thị ở turn gần nhất)
    - list_sources (danh sách các sách ở turn hiển thị danh sách > 1 cuốn)
    """
    ctx = {
        "doc_type": None,
        "major": None,
        "author": None,
        "topic": None,
        "search_query": None,
        "last_sources": [],
        "list_sources": [],
    }
    if not history:
        return ctx

    for entry in reversed(history):
        if not isinstance(entry, dict):
            continue
        if not ctx["doc_type"] and entry.get("doc_type"):
            ctx["doc_type"] = entry.get("doc_type")
        if not ctx["major"] and entry.get("major"):
            ctx["major"] = entry.get("major")
        if not ctx["author"] and entry.get("author"):
            ctx["author"] = entry.get("author")
        if not ctx["topic"] and entry.get("topic"):
            ctx["topic"] = entry.get("topic")
        if not ctx["search_query"] and entry.get("search_query"):
            ctx["search_query"] = entry.get("search_query")

        srcs = entry.get("sources") or []
        if not ctx["last_sources"] and srcs:
            ctx["last_sources"] = srcs
        if not ctx["list_sources"] and len(srcs) > 1:
            ctx["list_sources"] = srcs

    return ctx
