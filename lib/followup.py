"""
lib/followup.py — Follow-up suggestion builders and answer section extractors.
"""

import re
import random as _random
from .text_utils import normalize_text

# ── Helper functions for suggestion cleaning and whole-word truncation ──
def truncate_words(text: str, max_chars: int = 40) -> str:
    """Cắt ngắn văn bản theo ranh giới từ nguyên vẹn, không cắt cụt giữa chừng từ hoặc để dấu phẩy/chấm lơ lửng."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    truncated = re.sub(r"[\s,;:\-\(\[\{/]+$", "", truncated).strip()
    return (truncated + "...") if truncated else text[:max_chars]


def clean_author_for_suggestion(raw_author: str, max_words: int = 5) -> str:
    """Lấy tên tác giả chính đầu tiên, loại bỏ vai trò (ch.b.), (chủ biên) và dấu câu thừa."""
    if not raw_author or raw_author.strip() in ("Unknown", "N/A", "Không xác định"):
        return ""
    author = raw_author.strip()
    parts = re.split(r"[,;]\s*", author)
    first_part = parts[0] if parts else author
    clean = re.sub(r"\s*\((?:ch\.b\.|chu\s+bien|b\.s\.|dich|bien\s+soan|cb|dong\s+tac\s+gia|h\.d\.)[^\)]*\)", "", first_part, flags=re.I)
    clean = re.sub(r"\b(?:ch\.b\.|chu\s+bien|chb|b\.s\.|bien\s+soan)\b", "", clean, flags=re.I)
    clean = re.sub(r"\$[a-zA-Z0-9]", "", clean)
    clean = re.sub(r"^[\s,;:\-\(\[\{/]+|[\s,;:\-\)\]\}/]+$", "", clean).strip()
    if not clean:
        clean = re.sub(r"[\s,;:\-\(\)\[\]/]+$", "", first_part).strip()
    words = clean.split()
    if len(words) > max_words:
        clean = " ".join(words[:max_words])
    return clean


def clean_suggestion_topic(raw_topic: str, max_words: int = 6) -> str:
    """Dọn dẹp chuỗi chủ đề hoặc ngành học để làm nhãn gợi ý."""
    if not raw_topic or raw_topic.strip() in ("Unknown", "N/A", "Không xác định"):
        return ""
    topic = raw_topic.strip()
    topic = re.sub(r"\$[a-zA-Z0-9]", " ", topic)
    topic = re.sub(r"\s+", " ", topic)
    topic = re.sub(r"^[\s,;:\-\(\[\{/]+|[\s,;:\-\)\]\}/]+$", "", topic).strip()
    words = topic.split()
    if len(words) > max_words:
        topic = " ".join(words[:max_words])
    return topic


# ── Curated Pools for Topics, Majors, and Authors ──
SPECIFIC_TOPIC_SUGGESTIONS = [
    {"label": "Tìm sách về chủ đề: Trí tuệ nhân tạo", "query": "tìm sách về Trí tuệ nhân tạo"},
    {"label": "Tìm sách về chủ đề: Lập trình Python", "query": "tìm sách lập trình Python"},
    {"label": "Tìm sách về chủ đề: Kinh tế vi mô", "query": "tìm sách Kinh tế vi mô"},
    {"label": "Tìm sách về chủ đề: Phương pháp NCKH", "query": "tìm sách Phương pháp nghiên cứu khoa học"},
    {"label": "Tìm sách về chủ đề: Tâm lý học", "query": "tìm sách Tâm lý học"},
    {"label": "Tìm sách về chủ đề: Lịch sử Việt Nam", "query": "tìm sách Lịch sử Việt Nam"},
    {"label": "Tìm sách về chủ đề: Quản trị học", "query": "tìm sách Quản trị học"},
    {"label": "Tìm sách về chủ đề: Văn học hiện đại", "query": "tìm sách Văn học hiện đại"},
    {"label": "Tìm sách về chủ đề: Khoa học dữ liệu", "query": "tìm sách Khoa học dữ liệu"},
    {"label": "Tìm sách về chủ đề: Marketing", "query": "tìm sách Marketing"},
    {"label": "Tìm sách về chủ đề: Kỹ thuật lập trình", "query": "tìm sách Kỹ thuật lập trình"},
    {"label": "Tìm sách về chủ đề: Giáo dục học", "query": "tìm sách Giáo dục học"},
]

SPECIFIC_MAJOR_SUGGESTIONS = [
    {"label": "Tìm sách theo ngành: Công nghệ thông tin", "query": "tìm sách ngành Công nghệ thông tin"},
    {"label": "Tìm sách theo ngành: Quản trị kinh doanh", "query": "tìm sách ngành Quản trị kinh doanh"},
    {"label": "Tìm sách theo ngành: Ngôn ngữ Trung Quốc", "query": "tìm sách ngành Ngôn ngữ Trung Quốc"},
    {"label": "Tìm sách theo ngành: Sư phạm Toán học", "query": "tìm sách ngành Sư phạm Toán học"},
    {"label": "Tìm sách theo ngành: Kỹ thuật xây dựng", "query": "tìm sách ngành Kỹ thuật xây dựng"},
    {"label": "Tìm sách theo ngành: Kế toán", "query": "tìm sách ngành Kế toán"},
    {"label": "Tìm sách theo ngành: Luật", "query": "tìm sách ngành Luật"},
    {"label": "Tìm sách theo ngành: Tài chính - Ngân hàng", "query": "tìm sách ngành Tài chính - Ngân hàng"},
    {"label": "Tìm sách theo ngành: Sư phạm Tin học", "query": "tìm sách ngành Sư phạm Tin học"},
    {"label": "Tìm sách theo ngành: Ngôn ngữ Anh", "query": "tìm sách ngành Ngôn ngữ Anh"},
    {"label": "Tìm sách theo ngành: Đông phương học", "query": "tìm sách ngành Đông phương học"},
]

SPECIFIC_AUTHOR_SUGGESTIONS = [
    {"label": "Tìm sách của tác giả: Nguyễn Nhật Ánh", "query": "sách của tác giả Nguyễn Nhật Ánh"},
    {"label": "Tìm sách của tác giả: Tô Hoài", "query": "sách của tác giả Tô Hoài"},
    {"label": "Tìm sách của tác giả: Nam Cao", "query": "sách của tác giả Nam Cao"},
    {"label": "Tìm sách của tác giả: Xuân Diệu", "query": "sách của tác giả Xuân Diệu"},
    {"label": "Tìm sách của tác giả: Hàn Mặc Tử", "query": "sách của tác giả Hàn Mặc Tử"},
    {"label": "Tìm sách của tác giả: Triều Nguyên", "query": "sách của tác giả Triều Nguyên"},
    {"label": "Tìm sách của tác giả: GS.TS. Nguyễn Ngọc Yến Thi", "query": "sách của tác giả Nguyễn Ngọc Yến Thi"},
    {"label": "Tìm sách của tác giả: Hữu Mai", "query": "sách của tác giả Hữu Mai"},
    {"label": "Tìm sách của tác giả: Lê Xuân Vinh", "query": "sách của tác giả Lê Xuân Vinh"},
    {"label": "Tìm sách của tác giả: Hoàng Phê", "query": "sách của tác giả Hoàng Phê"},
]


def random_no_result_suggestions(count: int = 3, focus: str = None) -> list[dict]:
    """Gợi ý cụ thể theo Chủ đề, Ngành học và Tác giả nổi bật trong Thư viện QNU."""
    selected: list[dict] = []

    if focus == "author":
        authors_shuffled = _random.sample(SPECIFIC_AUTHOR_SUGGESTIONS, min(2, len(SPECIFIC_AUTHOR_SUGGESTIONS)))
        selected.extend(authors_shuffled)
        major = _random.choice(SPECIFIC_MAJOR_SUGGESTIONS)
        selected.append(major)
        topic = _random.choice(SPECIFIC_TOPIC_SUGGESTIONS)
        selected.append(topic)
    elif focus == "major":
        majors_shuffled = _random.sample(SPECIFIC_MAJOR_SUGGESTIONS, min(2, len(SPECIFIC_MAJOR_SUGGESTIONS)))
        selected.extend(majors_shuffled)
        topic = _random.choice(SPECIFIC_TOPIC_SUGGESTIONS)
        selected.append(topic)
        author = _random.choice(SPECIFIC_AUTHOR_SUGGESTIONS)
        selected.append(author)
    else:
        topic = _random.choice(SPECIFIC_TOPIC_SUGGESTIONS)
        major = _random.choice(SPECIFIC_MAJOR_SUGGESTIONS)
        author = _random.choice(SPECIFIC_AUTHOR_SUGGESTIONS)
        selected = [topic, major, author]
        _random.shuffle(selected)

    seen_queries = set()
    unique: list[dict] = []
    for item in selected:
        q = item["query"].lower().strip()
        if q not in seen_queries:
            seen_queries.add(q)
            unique.append(item)
        if len(unique) >= count:
            break

    return unique


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
        from .doc_type_resolver import detect_doc_type_from_query
        is_accreditation = (
            detect_doc_type_from_query(query) == "Kiểm định" or
            (sources and any(s.get("doc_type") == "Kiểm định" for s in sources[:3]))
        )
        if is_accreditation:
            acc_majors = [
                ("Công nghệ thông tin", "tài liệu kiểm định ngành Công nghệ thông tin"),
                ("Sư phạm Toán học", "tài liệu kiểm định ngành Sư phạm Toán học"),
                ("Kế toán", "tài liệu kiểm định ngành Kế toán"),
                ("Ngôn ngữ Anh", "tài liệu kiểm định ngành Ngôn ngữ Anh"),
            ]
            acc_sugg = []
            for m_name, m_query in acc_majors:
                if normalize_text(m_name) not in query_lower:
                    acc_sugg.append({
                        "label": f"Tìm tài liệu kiểm định: {m_name}",
                        "query": m_query
                    })
                if len(acc_sugg) == 3:
                    break
            return acc_sugg

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
                clean_a = clean_author_for_suggestion(a)
                if clean_a:
                    first_author = clean_a
            subj = (s.get("subject") or "").strip()
            if subj and subj.lower() not in _junk_subjects and len(subj) >= 2:
                clean_s = clean_suggestion_topic(subj)
                if clean_s:
                    first_subject = clean_s
            major = (s.get("major") or "").strip()
            if major and len(major) >= 2:
                clean_m = clean_suggestion_topic(major)
                if clean_m:
                    first_major = clean_m
            t = (s.get("title") or "").strip()
            if t and len(t) > 5 and not first_title:
                first_title = t.split(" - ")[0].strip()

        if first_major:
            suggestions.append({"label": f"Tìm sách theo ngành: {first_major}", "query": f"tìm sách ngành {first_major}"})
        if first_subject:
            suggestions.append({"label": f"Tìm sách về chủ đề: {first_subject}", "query": f"tìm sách về {first_subject}"})
        if first_author:
            suggestions.append({"label": f"Tìm sách của tác giả: {first_author}", "query": f"sách của tác giả {first_author}"})
        if len(suggestions) < 3:
            suggestions.extend(random_no_result_suggestions(3 - len(suggestions)))
    else:
        sections = extract_answer_sections(answer)
        for section in sections[:3]:
            section_clean = re.sub(r"^#{1,4}\s*", "", section).strip()
            section_clean = re.sub(r"\*\*|\*", "", section_clean).strip()
            if len(section_clean) >= 5:
                clean_sec = truncate_words(section_clean, 40)
                suggestions.append({"label": f"Phân tích: {clean_sec}", "query": f"Phân tích chi tiết nội dung phần '{clean_sec}'"})
        if len(suggestions) < 3:
            if re.search(r"phuong phap|methodology|cach tiep can|cong cu|ky thuat|thuat toan", answer_lower):
                method_names = re.findall(r"(?:\*\*|(?:^##?\s+))([^*\n]{5,50})", answer)
                if method_names:
                    m = method_names[0].strip().strip("#*").strip()
                    clean_m = truncate_words(m, 35)
                    suggestions.append({"label": f"Giải thích: {clean_m}", "query": f"Giải thích chi tiết hơn về {clean_m}"})
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
                    clean_subj = clean_suggestion_topic(subj)
                    if clean_subj:
                        suggestions.append({"label": f"Tìm sách về chủ đề: {clean_subj}", "query": f"tìm sách về {clean_subj}"})
                        break
        if len(suggestions) < 3 and sources:
            for s in sources[:3]:
                author = (s.get("author") or "").strip()
                clean_a = clean_author_for_suggestion(author)
                if clean_a:
                    suggestions.append({"label": f"Tìm sách của tác giả: {clean_a}", "query": f"sách của tác giả {clean_a}"})
                    break
        if len(suggestions) < 3 and answer and len(answer) > 200:
            suggestions.append({"label": "Tóm tắt nội dung chính", "query": "Tóm tắt lại nội dung thành 3-5 ý chính"})

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
