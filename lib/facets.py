"""
lib/facets.py — Phân nhóm (faceted grouping) và lọc kết quả tìm kiếm.

Flow:
  1. Tìm kiếm → lấy danh sách kết quả thô
  2. Nhóm theo các tiêu chí (chủ đề, loại tài liệu, tác giả, năm)
  3. Frontend hiển thị các nhóm, user chọn nhóm
  4. Lọc lại kết quả dựa trên nhóm + bộ lọc phụ (năm, tác giả, loại tài liệu)
"""

import re
from collections import Counter, defaultdict
from typing import Optional

from .text_utils import normalize_text


# ─── Cấu hình ────────────────────────────────────────────────
FACET_GROUPS = {
    "subject": {"label": "Chủ đề",   "field": "subject",   "max": 10},
    "doc_type": {"label": "Loại tài liệu", "field": "doc_type",  "max": 8},
    "author":  {"label": "Tác giả",   "field": "author",    "max": 10},
    "year":    {"label": "Năm xuất bản", "field": "year",      "max": 8},
    "major":   {"label": "Chuyên ngành", "field": "major",    "max": 10},
    "publisher": {"label": "Nhà xuất bản", "field": "publisher", "max": 8},
}

# Khoảng năm để gộp nhóm (year range buckets)
YEAR_BUCKETS = [
    (0, 1990, "Trước 1990"),
    (1990, 2000, "1990 - 2000"),
    (2000, 2005, "2000 - 2005"),
    (2005, 2010, "2005 - 2010"),
    (2010, 2015, "2010 - 2015"),
    (2015, 2020, "2015 - 2020"),
    (2020, 2025, "2020 - 2025"),
    (2025, 9999, "2025 - nay"),
]

# Giá trị "rỗng" cần loại bỏ
_EMPTY_VALUES = {"", "n/a", "na", "unknown", "nan", "none", "không xác định",
                 "chưa xác định", "khong xac dinh", "chua xac dinh", "pdf"}


def _is_empty(val) -> bool:
    if not val:
        return True
    v = str(val).strip().lower()
    return v in _EMPTY_VALUES or len(v) == 0


def _clean(val) -> str:
    """Làm sạch giá trị facet."""
    if _is_empty(val):
        return ""
    v = str(val).strip()
    # Loại bỏ MARC prefix
    v = re.sub(r'^\$[a-z]\s*', '', v).strip()
    return v


def _year_bucket(year_str: str) -> Optional[str]:
    """Xác định khoảng năm cho một năm cụ thể."""
    if _is_empty(year_str):
        return None
    digits = re.sub(r'\D', '', str(year_str))
    if not digits:
        return None
    try:
        y = int(digits)
    except ValueError:
        return None
    for lo, hi, label in YEAR_BUCKETS:
        if lo <= y < hi:
            return label
    return None


def extract_facets(
    search_results: list[dict],
    top_k: int = 5,
) -> dict[str, list[dict]]:
    """
    Từ danh sách kết quả tìm kiếm, trích xuất các nhóm facet.

    Args:
        search_results: list of {doc, score, ...}
        top_k: số lượng mỗi nhóm tối đa

    Returns:
        dict[group_name, list[{"value": str, "count": int}]]
        VD: {"subject": [{"value": "Khoa học máy tính", "count": 12}, ...],
             "doc_type": [{"value": "Sách", "count": 20}, ...],
             "author": [...], "year": [...], "major": [...], "publisher": [...]}
    """
    facets: dict[str, Counter] = {
        "subject": Counter(),
        "doc_type": Counter(),
        "author": Counter(),
        "year": Counter(),
        "major": Counter(),
        "publisher": Counter(),
    }

    for result in search_results:
        doc = result.get("doc", {})
        if not doc:
            continue

        # 1. Chủ đề (subject) — tách theo dấu phẩy/chấm phẩy
        subject_raw = _clean(doc.get("subject", ""))
        for part in re.split(r'[,;|]\s*', subject_raw):
            p = _clean(part)
            if p and len(p) >= 3:
                facets["subject"][p] += 1

        # 2. Loại tài liệu
        dt = _clean(doc.get("doc_type", ""))
        if dt:
            dt_low = dt.lower()
            if "giáo trình" in dt_low and "tham khảo" in dt_low:
                dt = "Giáo trình / Tài liệu tham khảo"
            else:
                dt = re.sub(r"\s*/\s*", " / ", dt)
        if dt and len(dt) >= 2:
            facets["doc_type"][dt] += 1

        # 3. Tác giả — tách theo dấu phẩy/chấm phẩy
        author_raw = _clean(doc.get("author", ""))
        for part in re.split(r'[,;|]\s*', author_raw):
            a = _clean(part)
            if a and len(a) >= 3:
                facets["author"][a] += 1

        # 4. Năm xuất bản — gộp theo khoảng
        yb = _year_bucket(doc.get("year", ""))
        if yb:
            facets["year"][yb] += 1

        # 5. Chuyên ngành
        major_raw = _clean(doc.get("major", ""))
        for part in re.split(r'[,;|]\s*', major_raw):
            m = _clean(part)
            if m and len(m) >= 3:
                facets["major"][m] += 1

        # 6. Nhà xuất bản
        pub_raw = _clean(doc.get("publisher", ""))
        for part in re.split(r'[,;|]\s*', pub_raw):
            p = _clean(part)
            if p and len(p) >= 3:
                facets["publisher"][p] += 1

    result: dict[str, list[dict]] = {}
    for group_name, counter in facets.items():
        items = [{"value": val, "count": cnt}
                 for val, cnt in counter.most_common(top_k)
                 if not _is_empty(val)]
        if items:
            result[group_name] = items

    return result


def filter_by_facet(
    search_results: list[dict],
    group_name: str,
    group_value: str,
) -> list[dict]:
    """
    Lọc kết quả chỉ giữ lại các document thuộc nhóm facet cụ thể.

    Args:
        search_results: list of {doc, score, ...}
        group_name: tên nhóm (subject, doc_type, author, year, major, publisher)
        group_value: giá trị cần lọc (VD: "Khoa học máy tính", "Trước 1990")

    Returns:
        list of results đã lọc
    """
    group_value_lower = group_value.strip().lower()

    def _matches(doc: dict) -> bool:
        if group_name in ("subject", "major", "publisher", "author"):
            raw = _clean(doc.get(group_name, ""))
            # Kiểm tra giá trị có xuất hiện trong field không (tách theo dấu phẩy)
            parts = [p.strip().lower() for p in re.split(r'[,;|]\s*', raw) if p.strip()]
            return group_value_lower in parts
        elif group_name == "doc_type":
            dt = _clean(doc.get("doc_type", "")).lower()
            return group_value_lower == dt
        elif group_name == "year":
            yb = _year_bucket(doc.get("year", ""))
            return yb and yb.lower() == group_value_lower
        return False

    return [r for r in search_results if _matches(r.get("doc", {}))]


def refine_results(
    search_results: list[dict],
    filters: dict[str, Optional[str]],
) -> list[dict]:
    """
    Áp dụng nhiều bộ lọc cùng lúc (AND).

    Args:
        search_results: list of {doc, score, ...}
        filters: dict với key là tên field, value là giá trị cần lọc
                 VD: {"author": "Nguyễn Văn A", "year": "2020 - 2025"}

    Returns:
        list of results đã lọc
    """
    results = search_results
    for group_name, group_value in filters.items():
        if group_value:
            results = filter_by_facet(results, group_name, group_value)
    return results
