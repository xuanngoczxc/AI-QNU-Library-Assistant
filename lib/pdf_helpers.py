"""
lib/pdf_helpers.py — PDF utility functions extracted from app.py.
"""

import re
from typing import Optional
from .text_utils import normalize_text

# Markers for PDFs with no extractable text
PDF_EMPTY_TEXT_MARKERS = {
    "[Không có nội dung]",
    "[pypdf not available - cannot extract text]",
}


def has_extractable_pdf_text(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in PDF_EMPTY_TEXT_MARKERS or stripped.startswith("[Lỗi:"):
        return False
    compact = re.sub(r"\s+", "", stripped)
    return len(compact) >= 50


def is_general_pdf_content_request(query: str) -> bool:
    q = normalize_text(query or "")
    broad_phrases = [
        "noi dung", "tom tat", "file nay", "pdf nay", "tai lieu nay",
        "van ban nay", "noi ve gi", "cho biet", "gioi thieu", "bo cuc",
        "muc luc", "cau truc", "danh muc", "layout", "structure",
        "table of contents",
    ]
    return any(phrase in q for phrase in broad_phrases)


def build_empty_pdf_answer(pdf_title: str) -> str:
    return (
        f"File PDF **{pdf_title}** không có lớp văn bản để hệ thống trích xuất nội dung. "
        "File này có thể là bản scan/ảnh hoặc bị khóa text, nên chatbot không thể đọc để trả lời theo nội dung bên trong.\n\n"
        "Cách xử lý: tạo bản PDF đã OCR hoặc upload file PDF có text layer, sau đó hỏi lại nội dung của file."
    )


def detect_reference_request(query: str) -> bool:
    q = normalize_text(query or "")
    phrases = [
        "tai lieu tham khao", "danh muc tai lieu", "danh sach tai lieu",
        "nguon tham khao", "tham khao", "references", "bibliography",
        "citation", "trich dan",
    ]
    return any(phrase in q for phrase in phrases)


def is_reference_heading_line(line: str) -> bool:
    normalized = normalize_text(line or "").strip(" .:-")
    headings = {
        "tai lieu tham khao", "danh muc tai lieu tham khao",
        "references", "reference", "bibliography",
    }
    return normalized in headings


def answer_says_no_information(answer: str) -> bool:
    if not answer:
        return True
    normalized = normalize_text(answer or "")
    no_info_phrases = [
        "tai lieu khong cung cap", "khong cung cap thong tin",
        "khong co thong tin", "khong du thong tin", "khong tim thay thong tin",
        "khong tim thay tai lieu", "khong co tai lieu", "khong co trong tai lieu",
        "khong duoc de cap trong tai lieu", "khong thay trong tai lieu",
        "khong duoc tim thay", "khong co du lieu", "khong the tra loi",
        "minh khong the", "xin loi toi khong", "xin loi minh khong",
        "khong co cau tra loi", "khong co phan hoi", "khong co thong tin cung cap",
        "toi khong tim thay", "minh khong tim thay", "chua tim thay",
        "hien tai minh chua tim thay",
    ]
    return any(phrase in normalized for phrase in no_info_phrases)


def build_related_pdf_answer(pdf_infos: list[dict], query: str) -> str:
    if not pdf_infos:
        return "Tài liệu không cung cấp thông tin này."
    lines = ["Mình tìm thấy file PDF liên quan trong thư viện:"]
    for pdf in pdf_infos[:3]:
        label = pdf.get("display_name") or pdf.get("title") or pdf.get("file_stem") or pdf.get("file_name") or "PDF"
        year = pdf.get("year")
        pages = pdf.get("pages")
        details = []
        if year and year != "N/A":
            details.append(f"năm {year}")
        if pages:
            details.append(f"{pages} trang")
        suffix = f" ({', '.join(details)})" if details else ""
        lines.append(f"- **{label}**{suffix}")
    lines.append("")
    lines.append("Bạn có thể bấm **Đính kèm PDF này** trong phần Tài liệu tham khảo để chọn file và hỏi tiếp theo file đó.")
    lines.append("Lưu ý: nếu PDF là bản scan/ảnh hoặc không có lớp text, hệ thống chỉ nhận diện được file liên quan chứ chưa đọc được nội dung chi tiết.")
    return "\n".join(lines)


def parse_reference_items(section_text: str, max_items: int = 20) -> list[str]:
    items = []
    current = ""
    for raw_line in (section_text or "").replace("\xa0", " ").splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        if line.startswith("[Trang PDF"):
            continue
        if is_reference_heading_line(line):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if re.match(r"^\[\d+\]", line):
            if current:
                items.append(current.strip())
                if len(items) >= max_items:
                    return items
            current = line
        elif current:
            current += " " + line
    if current and len(items) < max_items:
        items.append(current.strip())
    return items


def build_reference_answer(pdf_info: dict, section_text: str, start_page: Optional[int]) -> str:
    title = pdf_info.get("display_name") or pdf_info.get("title") or pdf_info.get("file_stem") or "PDF"
    items = parse_reference_items(section_text)
    page_text = f" bắt đầu ở trang PDF {start_page}" if start_page else ""
    lines = [f"Có. Trong PDF **{title}** có mục **Tài liệu tham khảo**{page_text}."]
    if items:
        lines.append("")
        lines.append("Các tài liệu tham khảo trích xuất được:")
        for item in items:
            lines.append(f"- {item}")
        if len(items) >= 20:
            lines.append("- ...")
    else:
        preview = section_text[:2500].strip()
        if preview:
            lines.append("")
            lines.append(preview)
    return "\n".join(lines)
