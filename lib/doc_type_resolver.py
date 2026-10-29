"""
lib/doc_type_resolver.py — Document Type (Loại tài liệu) detection, normalization, and filtering.

Supported document types in QNU Library:
- 'Kiểm định' (Tài liệu / Báo cáo / Minh chứng phục vụ kiểm định chất lượng CTĐT)
- 'Giáo trình' (Giáo trình môn học, textbook)
- 'Tài liệu tham khảo' (Sách và tài liệu tham khảo)
- 'Luận văn' (Luận văn thạc sĩ, luận án tiến sĩ, khóa luận tốt nghiệp)
- 'Bài giảng' (Tài liệu bài giảng giảng viên)
- 'Sách bài tập' (Sách bài tập, bài tập thực hành)
- 'Tài liệu chính' (Tài liệu giảng dạy chính)
- 'Kỷ yếu' (Kỷ yếu hội thảo, hội nghị)
"""

import re
from .text_utils import normalize_text


def detect_doc_type_from_query(query: str) -> str | None:
    """
    Phát hiện loại tài liệu (doc_type) từ câu hỏi ngôn ngữ tự nhiên của bạn đọc.
    """
    if not query:
        return None

    q = normalize_text(query).strip().lower()

    # ── 1. Kiểm định chất lượng / Tài liệu kiểm định ──
    # Loại trừ trường hợp kiểm định giả thuyết thống kê / kinh tế lượng trong toán học
    is_statistical_test = bool(re.search(
        r"\b(?:kiem\s+dinh\s+(?:gia\s+thuyet|thong\s+ke|mo\s+hinh|chi|t\b|f\b|phi\s+tham\s+so|wald|durbin))\b",
        q
    ))
    if not is_statistical_test:
        # Nhận diện rõ ràng mục đích kiểm định chất lượng / kiểm định chương trình / trường
        if re.search(
            r"\b(?:phuc\s+vu\s+(?:cho\s+)?kiem\s+dinh(?:\s+chat\s+luong)?|"
            r"kiem\s+dinh\s+chat\s+luong|"
            r"tai\s+lieu\s+kiem\s+dinh|"
            r"sach\s+kiem\s+dinh|"
            r"bao\s+cao\s+kiem\s+dinh|"
            r"minh\s+chung\s+kiem\s+dinh|"
            r"ho\s+so\s+kiem\s+dinh|"
            r"tu\s+danh\s+gia\s+chat\s+luong|"
            r"kiem\s+dinh\s+chuong\s+trinh(?:\s+dao\s+tao)?|"
            r"kiem\s+dinh\s+nganh|"
            r"kiem\s+dinh\s+truong|"
            r"danh\s+gia\s+ngoai|"
            r"danh\s+gia\s+noi\s+bo|"
            r"chat\s+luong\s+chuong\s+trinh\s+dao\s+tao)\b",
            q
        ):
            return "Kiểm định"

        # Nếu có từ "kiểm định" đi kèm với từ chỉ tài liệu / tổ chức
        if re.search(r"\bkiem\s+dinh\b", q) and any(w in q for w in (
            "tai lieu", "sach", "nganh", "chuong trinh", "truong", "khoa",
            "bao cao", "minh chung", "ho so", "phuc vu", "chat luong", "ctdt"
        )):
            return "Kiểm định"

    # ── 2. Luận văn / Luận án / Khóa luận ──
    if re.search(r"\b(?:luan\s+van|luan\s+an|khoa\s+luan|do\s+an\s+tot\s+nghiep|thesis|dissertation)\b", q):
        return "Luận văn"

    # ── 3. Giáo trình ──
    if re.search(r"\b(?:giao\s+trinh|sach\s+giao\s+trinh|tai\s+lieu\s+giao\s+trinh|textbook)\b", q):
        return "Giáo trình"

    # ── 4. Tài liệu tham khảo ──
    if re.search(r"\b(?:tai\s+lieu\s+tham\s+khao|sach\s+tham\s+khao|tai\s+lieu\s+doc\s+them)\b", q):
        return "Tài liệu tham khảo"

    # ── 5. Bài giảng ──
    if re.search(r"\b(?:bai\s+giang|lecture\s+notes)\b", q):
        return "Bài giảng"

    # ── 6. Sách bài tập ──
    if re.search(r"\b(?:sach\s+bai\s+tap|bai\s+tap\s+va\s+loi\s+giai)\b", q) or (
        re.search(r"\bbai\s+tap\b", q) and not re.search(r"\b(?:giao\s+trinh|ly\s+thuyet)\b", q)
    ):
        return "Sách bài tập"

    # ── 7. Tài liệu chính ──
    if re.search(r"\b(?:tai\s+lieu\s+chinh|giao\s+trinh\s+chinh)\b", q):
        return "Tài liệu chính"

    # ── 8. Kỷ yếu ──
    if re.search(r"\b(?:ky\s+yeu|ky\s+yeu\s+hoi\s+thao)\b", q):
        return "Kỷ yếu"

    return None


def clean_query_doc_type_terms(query: str, doc_type: str) -> str:
    """
    Loại bỏ các cụm từ chỉ loại tài liệu khỏi query để tìm kiếm theo chủ đề/ngành không bị nhiễu.
    """
    if not query or not doc_type:
        return query

    q = query
    patterns_to_remove = {
        "Kiểm định": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*(?:để\s+)?(?:phục\s+vụ\s+(?:cho\s+)?)?(?:công\s+tác\s+)?(?:kiểm\s+định(?:\s+chất\s+lượng)?|tự\s+đánh\s+giá(?:\s+chất\s+lượng)?|báo\s+cáo\s+kiểm\s+định|hồ\s+sơ\s+kiểm\s+định|minh\s+chứng(?:\s+kiểm\s+định)?)\b",
            r"\bphục\s+vụ\s+(?:cho\s+)?kiểm\s+định(?:\s+chất\s+lượng)?\b",
            r"\bkiểm\s+định\s+chất\s+lượng\b",
            r"\btài\s+liệu\s+kiểm\s+định\b",
            r"\bsách\s+kiểm\s+định\b",
        ],
        "Giáo trình": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*(?:giáo\s+trình|sách\s+giáo\s+trình|textbook)\b",
        ],
        "Tài liệu tham khảo": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*(?:tài\s+liệu\s+tham\s+khảo|sách\s+tham\s+khảo|tham\s+khảo)\b",
        ],
        "Luận văn": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*(?:luận\s+văn|luận\s+án|khóa\s+luận|đồ\s+án\s+tốt\s+nghiep)\b",
        ],
        "Bài giảng": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*bài\s+giảng\b",
        ],
        "Sách bài tập": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*(?:sách\s+bài\s+tập|bài\s+tập)\b",
        ],
        "Tài liệu chính": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*tài\s+liệu\s+chính\b",
        ],
        "Kỷ yếu": [
            r"\b(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*kỷ\s+yếu\b",
        ],
    }

    p_list = patterns_to_remove.get(doc_type, [])
    for p in p_list:
        q = re.sub(p, " ", q, flags=re.IGNORECASE)

    # Dọn dẹp các từ nối sót lại ở đầu câu như 'tìm sách', 'để phục vụ cho', 'cho'
    q = re.sub(r"^(?:tìm\s+)?(?:sách|tài\s+liệu)?\s*(?:để\s+)?(?:phục\s+vụ\s+(?:cho\s+)?)?", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def infer_doc_type_from_doc(d: dict) -> str:
    """
    Xác định loại tài liệu chuẩn từ document dictionary.
    """
    dt = str(d.get("doc_type") or "").strip()
    if dt and dt not in ("N/A", "Unknown", "nan", "Khác", "None", ""):
        low = dt.lower()
        if "giáo trình" in low and "tham khảo" in low:
            return "Giáo trình / Tài liệu tham khảo"
        if low in ("tài liệu tham khảo", "tài liệu kham khảo", "tham khảo", "sách tham khảo"):
            return "Tài liệu tham khảo"
        if low in ("giáo trình", "giáo trình chính"):
            return "Giáo trình"
        if low in ("kiểm định", "kiểm định aun"):
            return "Kiểm định"
        return re.sub(r"\s*/\s*", " / ", dt)
    t = str(d.get("title") or "").lower()
    loc = str(d.get("location") or "").lower()
    if "luận văn" in loc or "luan van" in loc or "lv" in loc or t.startswith("luận văn"):
        return "Luận văn"
    if "giáo trình" in t or "giao trinh" in t or "giáo trình" in loc or "giao trinh" in loc:
        return "Giáo trình"
    if "bài tập" in t or "bai tap" in t:
        return "Sách bài tập"
    if "kỷ yếu" in t or "ky yeu" in t:
        return "Kỷ yếu"
    return "Sách"


def doc_matches_target_doc_type(doc: dict, target_doc_type: str) -> bool:
    """
    Kiểm tra một document có khớp với loại tài liệu mong muốn hay không.
    """
    if not target_doc_type:
        return True

    doc_dt = infer_doc_type_from_doc(doc).lower()
    tgt = target_doc_type.lower()

    if tgt == doc_dt:
        return True

    # Đồng nghĩa hoặc bao hàm
    if tgt == "giáo trình":
        return doc_dt in ("giáo trình", "giáo trình/tài liệu tham khảo", "giáo trình / tài liệu tham khảo", "tài liệu chính")
    if tgt == "tài liệu tham khảo":
        return doc_dt in ("tài liệu tham khảo", "giáo trình/tài liệu tham khảo", "giáo trình / tài liệu tham khảo")
    if tgt == "kiểm định":
        return doc_dt == "kiểm định" or "tự đánh giá" in str(doc.get("title", "")).lower()

    return False
