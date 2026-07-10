"""
lib/followup.py — Follow-up suggestion builders and answer section extractors.
"""

import re
import random as _random
from .text_utils import normalize_text

# ── Suggestion pools for randomized fallback suggestions ──
_SUGGESTION_POOL_TOPICS = [
    "trí tuệ nhân tạo", "học máy", "khoa học dữ liệu", "an ninh mạng",
    "văn học Việt Nam", "văn học nước ngoài", "ngôn ngữ Anh", "ngôn ngữ Trung",
    "kinh tế học", "quản trị kinh doanh", "kế toán", "tài chính ngân hàng",
    "tâm lý học", "giáo dục học", "xã hội học", "luật học",
    "công nghệ thông tin", "kỹ thuật điện", "xây dựng", "công nghệ thực phẩm",
    "toán học", "vật lý", "hóa học", "sinh học",
    "lịch sử", "địa lý", "triết học", "chính trị",
    "quản lý nhà nước", "công tác xã hội", "du lịch", "văn hóa",
]

_SUGGESTION_POOL_AUTHORS = [
    "Nguyễn Nhật Ánh", "Hồ Chí Minh", "Nam Cao", "Tô Hoài",
    "Vũ Trọng Phụng", "Nguyễn Du", "Xuân Diệu", "Huy Cận",
    "Stuart J. Russell", "Andrew Ng", "Ian Goodfellow",
    "Lê Văn Hùng", "Trần Đình Hượu",
    "Nguyễn Văn Hiệu", "Phan Văn Các", "Hoàng Phê",
    "Trần Văn Giàu", "Nguyễn Khắc Viện", "Cao Xuân Hạo",
]

_SUGGESTION_POOL_MAJORS = [
    "Công nghệ thông tin", "Kỹ thuật điện", "Kỹ thuật xây dựng",
    "Quản trị kinh doanh", "Kế toán", "Tài chính ngân hàng",
    "Ngôn ngữ Anh", "Ngôn ngữ Trung Quốc", "Luật",
    "Tâm lý học", "Giáo dục học", "Công tác xã hội",
    "Toán học", "Vật lý", "Hóa học", "Sinh học",
    "Văn học", "Lịch sử", "Triết học",
    "Kỹ thuật ô tô", "Công nghệ thực phẩm", "Quản lý nhà nước",
    "Sư phạm Toán", "Sư phạm Văn", "Sư phạm Anh",
]


def random_no_result_suggestions(count: int = 3) -> list[dict]:
    """Generate randomized suggestions from topic/author/major pools."""
    types_order = _random.sample(["topic", "author", "major"], min(3, count))
    suggestions = []
    for t in types_order[:count]:
        if t == "topic":
            topic = _random.choice(_SUGGESTION_POOL_TOPICS)
            suggestions.append({"label": f"Tìm sách về {topic}", "query": f"tìm sách về {topic}"})
        elif t == "author":
            author = _random.choice(_SUGGESTION_POOL_AUTHORS)
            suggestions.append({"label": f"Sách của {author}", "query": f"sách của {author}"})
        elif t == "major":
            major = _random.choice(_SUGGESTION_POOL_MAJORS)
            suggestions.append({"label": f"Sách ngành {major}", "query": f"tìm sách ngành {major}"})
    return suggestions


def extract_answer_headings(answer: str) -> list[str]:
    headings = []
    for line in (answer or "").splitlines():
        stripped = line.strip()
        m = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if m:
            heading = m.group(1).strip()
            if len(heading) >= 4 and not re.match(r"^(trang|page|\[)", heading, re.I):
                headings.append(heading)
    return headings[:6]


def extract_answer_sections(answer: str) -> list[str]:
    sections = []
    seen = set()
    for line in (answer or "").splitlines():
        stripped = line.strip()
        if not stripped or len(stripped) < 5:
            continue
        name = None
        m = re.match(r"^#{1,4}\s+(.+)$", stripped)
        if m:
            name = m.group(1).strip()
        if not name:
            m = re.match(r"^\*\*(.+?)\*\*$", stripped)
            if m and len(m.group(1)) >= 5:
                name = m.group(1).strip()
        if not name:
            m = re.match(r"^(?:\d+(?:\.\d+)*|[IVX]+)\.\s+(.{5,60})$", stripped)
            if m:
                name = m.group(1).strip()
        if name and 5 <= len(name) <= 80:
            name_key = name.lower().strip()
            if name_key not in seen:
                seen.add(name_key)
                sections.append(name)
        if len(sections) >= 5:
            break
    return sections


def build_contextual_suggestions(
    answer: str,
    query: str,
    sources: list[dict] = None,
    doc_title: str = "",
    catalog_mode: bool = False,
    has_pdf_sources: bool = False,
) -> list[dict]:
    suggestions = []
    answer_lower = normalize_text(answer or "")
    query_lower = normalize_text(query or "")
    sources = sources or []

    if catalog_mode:
        _junk_authors = {
            "aspose", "aspose.pdf", "aspose.pdf.epubtopdfconverter",
            "unknown", "n/a", "khong xac dinh", "pdf converter",
            "epubtopdfconverter", "microsoft", "adobe", "apache",
        }
        _junk_subjects = {"aspose", "n/a", "unknown", "khong xac dinh", "pdf", "tai lieu so"}
        first_author = first_subject = first_major = first_title = ""
        for s in sources[:5]:
            a = (s.get("author") or "").strip()
            if a and len(a) >= 3 and a.lower() not in _junk_authors and not any(kw in a.lower() for kw in ("aspose", "converter")):
                first_author = a
            subj = (s.get("subject") or "").strip()
            if subj and subj.lower() not in _junk_subjects and len(subj) >= 2:
                first_subject = subj
            major = (s.get("major") or "").strip()
            if major and len(major) >= 2:
                first_major = major
            t = (s.get("title") or "").strip()
            if t and len(t) > 5:
                first_title = t.split(" - ")[0].strip()[:80]
        if first_major:
            suggestions.append({"label": f"Tìm sách theo ngành: {first_major}", "query": f"tìm sách thuộc ngành {first_major}"})
        if first_subject:
            suggestions.append({"label": f"Sách cùng chủ đề: {first_subject}", "query": f"tìm sách về {first_subject}"})
        if first_author:
            suggestions.append({"label": f"Sách cùng tác giả: {first_author}", "query": f"sách của tác giả {first_author}"})
        if len(suggestions) < 3:
            # Fallback: random suggestions thay vì gợi ý chung chung
            suggestions.extend(random_no_result_suggestions(3 - len(suggestions)))
    else:
        sections = extract_answer_sections(answer)
        for section in sections[:3]:
            section_clean = re.sub(r"^#{1,4}\s*", "", section).strip()
            section_clean = re.sub(r"\*\*|\*", "", section_clean).strip()
            if len(section_clean) >= 5:
                suggestions.append({"label": f"Phân tích: {section_clean[:50]}", "query": f"Phân tích chi tiết nội dung phần '{section_clean}'"})
        if len(suggestions) < 3:
            if re.search(r"phuong phap|methodology|cach tiep can|cong cu|ky thuat|thuat toan", answer_lower):
                method_names = re.findall(r"(?:\*\*|(?:^##?\s+))([^*\n]{5,50})", answer)
                if method_names:
                    m = method_names[0].strip().strip("#*").strip()
                    suggestions.append({"label": f"Giải thích: {m[:40]}", "query": f"Giải thích chi tiết hơn về {m}"})
            if re.search(r"ket qua|finding|conclusion|ket luan|danh gia", answer_lower):
                suggestions.append({"label": "Phân tích kết quả", "query": "Phân tích chi tiết hơn về kết quả và đánh giá"})
        if len(suggestions) < 3:
            if re.search(r"la gi|dinh nghia|khai niem", query_lower):
                suggestions.append({"label": "Ví dụ minh họa cụ thể", "query": "Cho ví dụ minh họa cụ thể"})
            if re.search(r"so sanh|khac biet", query_lower):
                suggestions.append({"label": "So sánh chi tiết hơn", "query": "So sánh chi tiết hơn giữa các yếu tố"})
        if len(suggestions) < 3 and sources:
            for s in sources[:3]:
                subj = (s.get("subject") or "").strip()
                if subj and len(subj) >= 3:
                    suggestions.append({"label": f"Kiến thức liên quan: {subj[:40]}", "query": f"Kiến thức liên quan đến {subj}"})
                    break
        if len(suggestions) < 3 and sources:
            for s in sources[:3]:
                author = (s.get("author") or "").strip()
                if author and len(author) >= 3:
                    suggestions.append({"label": f"Sách cùng tác giả: {author}", "query": f"sách của tác giả {author}"})
                    break
        if len(suggestions) < 3 and answer and len(answer) > 200:
            suggestions.append({"label": "Tóm tắt nội dung", "query": "Tóm tắt lại nội dung thành 3-5 ý chính"})

    seen = set()
    unique = []
    for s in suggestions:
        q = s["query"].strip().lower()
        if q not in seen and q:
            seen.add(q)
            unique.append(s)
        if len(unique) >= 3:
            break
    return unique


def build_follow_up_suggestions(
    query: str, sources: list[dict], answer: str,
    catalog_mode: bool, has_pdf_sources: bool, session_id: str,
) -> list[dict]:
    return build_contextual_suggestions(
        answer=answer, query=query, sources=sources,
        doc_title=sources[0].get("title", "") if sources else "",
        catalog_mode=catalog_mode, has_pdf_sources=has_pdf_sources,
    )
