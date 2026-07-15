"""
lib/llm_prompts.py — Prompt builders for various question types.
Separated from llm_client.py to keep it focused.
"""

import re
from typing import Optional
from .text_utils import normalize_text
from .llm_client import call_llm


def sanitize_answer_text(answer: str) -> str:
    """Remove source footers and stray assistant tags from model output."""
    if not answer:
        return ""
    text = answer.replace("</assistant>", "").replace("<assistant>", "")
    if "<think>" in text:
        text = text.split("</think>", 1)[1] if "</think>" in text else text.split("<think>", 1)[0]
    for marker in ["Nguồn tài liệu:", "Nguồn:", "Tài liệu tham khảo:"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if stripped.lower().startswith(("okay,", "the user", "i will", "here is", "sure,", "certainly,")):
            continue
        if re.match(r"^[-*•\s]*\[Tài liệu\s*\d+\]", stripped, re.IGNORECASE):
            continue
        if re.match(r"^[-*•\s]*(Tài liệu\s*\d+|Nguồn tài liệu)\b", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def detect_chapter_reference(query: str) -> Optional[int]:
    if not query:
        return None
    patterns = [
        r"chương\s*(\d+)", r"chuong\s*(\d+)", r"chapter\s*(\d+)",
        r"chương\s*(?:số|thứ|thu)?\s*(\d+)",
        r"chuong\s*(?:so|thu)?\s*(\d+)",
    ]
    q = normalize_text(query)
    for pattern in patterns:
        match = re.search(pattern, q, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def detect_method_question(query: str) -> bool:
    if not query:
        return False
    keywords = [
        "phương pháp", "phuong phap", "method", "methodology",
        "cách thức", "công cụ", "quy trình", "cach tiep can",
        "kỹ thuật", "ky thuat", "technique",
    ]
    q = normalize_text(query)
    return any(kw in q for kw in keywords)


def detect_structure_query(query: str) -> bool:
    if not query:
        return False
    q = normalize_text(query)
    keywords = [
        "bo cuc", "bố cục", "cau truc", "cấu trúc",
        "muc luc", "mục lục", "danh muc", "danh mục",
        "layout", "structure", "table of contents",
    ]
    return any(kw in q for kw in keywords)

def detect_result_question(query: str) -> bool:
    """Detect nếu user hỏi về kết quả/kết luận"""
    if not query:
        return False
    keywords = [
        "kết quả", "ket qua", "kết luận", "ket luan",
        "result", "finding", "conclusion", "phát hiện",
        "phat hien", "outcome", "giải pháp", "giai phap",
    ]
    q = normalize_text(query)
    return any(kw in q for kw in keywords)



def build_chapter_prompt(chapter_num: int) -> str:
    return f"""
Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn.

YÊU CẦU: Trả lời câu hỏi về CHƯƠNG {chapter_num} dựa trên nội dung tài liệu được cung cấp.

HƯỚNG DẪN TRẢ LỜI:
1. Xác định nội dung chính của Chương {chapter_num}.
2. Nêu rõ: Nhan đề sách, tác giả, năm xuất bản.
3. Tóm tắt các nội dung chính trong chương (mục tiêu, các phần nhỏ).
4. Nếu có kết quả/phát hiện quan trọng, liệt kê rõ.
5. Nếu thông tin không đủ, hãy nói "Chương này không được đề cập chi tiết trong tài liệu hiện có".
6. KHÔNG đánh số lại thành 1, 2, 3... liên tục.
7. Dùng ## cho mục lớn, ### cho mục vừa, #### cho mục nhỏ. KHÔNG dùng `-` trước heading.

TÀI LIỆU:
{{context}}

CÂU HỎI: {{question}}

TRẢ LỜI (Tiếng Việt, chính xác, trọng tâm vào chương {chapter_num}):
"""


def build_method_prompt() -> str:
    return """
Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn.

YÊU CẦU: Phân tích và trả lời về PHƯƠNG PHÁP NGHIÊN CỨU dựa trên tài liệu.

HƯỚNG DẪN TRẢ LỜI:
1. Xác định tài liệu (nhan đề, tác giả, năm xuất bản).
2. Nêu rõ phương pháp nghiên cứu được sử dụng (định tính, định lượng, hỗn hợp...).
3. Mô tả công cụ, quy trình, dữ liệu sử dụng.
4. Giải thích tại sao phương pháp này phù hợp.
5. Nếu tài liệu không cung cấp, hãy nói "Tài liệu không đề cập rõ phương pháp nghiên cứu".
6. Dùng ## cho mục lớn, ### cho mục vừa, #### cho mục nhỏ.

TÀI LIỆU:
{context}

CÂU HỎI: {question}

TRẢ LỜI (Tiếng Việt, chi tiết, tập trung vào phương pháp):
"""


def build_structure_prompt() -> str:
    return """
Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn.

YÊU CẦU: Trình bày BỐ CỤC / CẤU TRÚC / MỤC LỤC của tài liệu dựa trên nội dung được cung cấp.

HƯỚNG DẪN TRẢ LỜI:
1. Xác định tài liệu (nhan đề, tác giả, năm xuất bản).
2. Trình bày TOÀN BỘ bố cục/cấu trúc tài liệu theo dạng mục lục phân cấp.
3. Dùng ## cho chương lớn, ### cho phần, #### cho mục nhỏ.

TÀI LIỆU:
{context}

CÂU HỎI: {question}

TRẢ LỜI (Tiếng Việt, dạng mục lục phân cấp, KHÔNG viết thành đoạn văn):
"""


GENERAL_PROMPT = """Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn (QNU Library Assistant).

QUY TẮC VÀNG — TUYỆT ĐỐI TUÂN THỦ:
0. **KHÔNG ĐƯỢC DÙNG KIẾN THỨC BÊN NGOÀI.** Bạn CHỈ trả lời dựa trên TÀI LIỆU được cung cấp trong mục [TÀI LIỆU] bên dưới.
   - Nếu tài liệu KHÔNG chứa câu trả lời, hãy nói: *"Tài liệu không cung cấp thông tin này."*
   - KHÔNG suy luận, KHÔNG phỏng đoán, KHÔNG bịa đặt dữ liệu.
   - **ĐẶC BIỆT:** Nếu tài liệu KHÔNG có mục "Tóm tắt:" hoặc ghi rõ "(Không có tóm tắt)" thì BẠN KHÔNG ĐƯỢC TỰ VIẾT tóm tắt cho cuốn sách đó. Chỉ nói thư viện không có tóm tắt cho cuốn sách này.

CÁC HƯỚNG DẪN KHÁC — CHỈ ÁP DỤNG KHI TÀI LIỆU CÓ CÂU TRẢ LỜI:
1. CHỈ dùng những tài liệu có liên quan TRỰC TIẾP đến câu hỏi của user.
2. Trả lời dựa CHÍNH XÁC trên nội dung tài liệu được cung cấp.
3. KHÔNG viết thành một đoạn văn dài. Luôn trình bày thành từng ý rõ ràng.

CẤU TRÚC PHÂN CẤP — BẮT BUỘC TUÂN THỦ:
- Khi tài liệu gốc có cấu trúc phân cấp, BẮT BUỘC tái hiện đúng cấu trúc đó trong câu trả lời.

[TÀI LIỆU]:
{context}

[CÂU HỎI]: {question}

[TRẢ LỜI] (Tiếng Việt, chi tiết theo cấu trúc tài liệu; tuân thủ tuyệt đối Quy tắc vàng số 0):"""


def build_general_chain():
    """Build general RAG chain with prompt selection based on question type."""
    def rag_chain(context: str, question: str) -> str:
        chapter_num = detect_chapter_reference(question)
        is_method = detect_method_question(question)
        is_structure = detect_structure_query(question)

        if chapter_num:
            prompt = build_chapter_prompt(chapter_num)
        elif is_method:
            prompt = build_method_prompt()
        elif is_structure:
            prompt = build_structure_prompt()
        else:
            prompt = GENERAL_PROMPT

        messages = [
            {"role": "system", "content": "Bạn chỉ được trả về phần trả lời cuối cùng bằng tiếng Việt. Không hiển thị suy luận, không dùng thẻ <think>. Trả lời dựa trên nội dung tài liệu được cung cấp bên dưới. Nếu câu hỏi thực sự không liên quan đến tài liệu, hãy nói ngắn gọn rằng tài liệu không chứa thông tin này."},
            {"role": "user", "content": prompt.format(context=context, question=question)}
        ]
        return sanitize_answer_text(call_llm(messages, temperature=0.1))

    return rag_chain, None


def build_summary_prompt():
    """Build summary prompt chain."""
    prompt_text = """Bạn là trợ lý AI của Thư viện Trường Đại học Quy Nhơn (QNU Library Assistant).

TUYỆT ĐỐI KHÔNG được thêm thông tin ngoài nội dung tài liệu bên dưới.
Chỉ tóm tắt ĐÚNG những gì có trong tài liệu.

Hãy tóm tắt tài liệu theo cấu trúc sau — CHỈ viết khi nội dung tương ứng CÓ trong tài liệu:

**MỤC TIÊU:**
**PHƯƠNG PHÁP:**
**KẾT QUẢ CHÍNH:**
**TỪ KHÓA:**
**TÓM TẮT NGẮN:**

NỘI DUNG TÀI LIỆU:
{content}

TRẢ LỜI (Tiếng Việt, tuân thủ cấu trúc trên, chỉ nội dung có thật trong tài liệu):"""

    def summary_chain(content: str) -> str:
        messages = [
            {"role": "system", "content": "Bạn chỉ được trả về phần trả lời cuối cùng bằng tiếng Việt. Không thêm thông tin không có trong tài liệu."},
            {"role": "user", "content": prompt_text.format(content=content)}
        ]
        return sanitize_answer_text(call_llm(messages, temperature=0.1))

    return summary_chain
