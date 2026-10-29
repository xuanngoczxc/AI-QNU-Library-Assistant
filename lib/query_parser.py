"""
lib/query_parser.py — Intelligent LLM-based query parsing & intent/entity extraction.
Used as an intelligent Safety-Net Fallback when regex-based parsing is uncertain or yields 0 matches.
"""

import re
import json
from typing import Optional, Tuple
from .llm_client import call_llm_async, call_llm
from .logging_setup import logger

PARSE_SYSTEM_PROMPT = """Bạn là trợ lý phân tích ngữ cảnh truy vấn cho Thư viện QNU (Query Parser).
Nhiệm vụ: Phân tích câu hỏi của bạn đọc để trích xuất tên tài liệu/sách và thuộc tính cần tra cứu.

Các mã thuộc tính hợp lệ (attribute):
- 'place': nơi xuất bản, xuất bản ở đâu, in tại đâu, địa điểm phát hành
- 'publisher': nhà xuất bản, nxb, đơn vị phát hành, ai xuất bản
- 'year': năm xuất bản, xuất bản năm nào, ra đời năm nào
- 'location': vị trí phòng đọc, phòng mượn, kho, tầng mấy, tìm ở đâu
- 'author': tác giả, ai viết, ai biên soạn, chủ biên, người dịch
- 'abstract': tóm tắt, nội dung chính, nói về cái gì, giới thiệu
- 'pages': số trang, độ dày, bao nhiêu trang
- 'ddc': mã phân loại DDC, mã xếp giá, ký hiệu xếp giá
- 'isbn': mã ISBN, số ĐKCB, mã tài liệu
- 'borrowable': mượn về nhà được không, cho mượn không
- 'copies': số lượng bản lưu, có bao nhiêu cuốn, mấy bản
- 'major': chuyên ngành đào tạo, ngành nào học, môn học nào
- 'language': ngôn ngữ, viết bằng tiếng gì, tiếng anh hay tiếng việt
- 'edition': tái bản lần mấy, ấn bản thứ mấy

Quy tắc xuất kết quả:
1. 'book_title': Trích xuất CHÍNH XÁC tên sách / tài liệu, loại bỏ hoàn toàn các từ hỏi, từ nối ('của', 'về', 'ở', 'tại'), đại từ ('sách', 'cuốn sách', 'quyển', 'giáo trình', 'tài liệu'). Giữ nguyên chữ hoa/thường ban đầu. Nếu câu hỏi không nhắc đến một cuốn sách cụ thể, trả về null.
2. 'attribute': Một trong các mã thuộc tính ở trên. Nếu người dùng chỉ tìm sách thông thường mà không hỏi thuộc tính cụ thể, trả về null.
3. CHỈ TRẢ VỀ DUY NHẤT một chuỗi JSON hợp lệ, không kèm văn bản giải thích.

Ví dụ:
User: "Nơi xuất bản của Practice and Progress"
Output: {"book_title": "Practice and Progress", "attribute": "place"}

User: "Cuốn Đắc nhân tâm xuất bản năm nào thế"
Output: {"book_title": "Đắc nhân tâm", "attribute": "year"}

User: "Sách Giải tích 1 do ai viết vậy ad"
Output: {"book_title": "Giải tích 1", "attribute": "author"}

User: "Tìm sách về trí tuệ nhân tạo"
Output: {"book_title": null, "attribute": null}
"""


def parse_attribute_inquiry_with_llm(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Phân tích câu hỏi người dùng bằng LLM để bóc tách (book_title, attribute).
    Trả về: (book_title, attribute) hoặc (None, None) nếu không nhận diện được.
    """
    if not query or len(query.strip()) < 3:
        return None, None

    messages = [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": query.strip()}
    ]

    try:
        raw_resp = call_llm(messages, temperature=0.0)
        if not raw_resp:
            return None, None

        # Clean markdown code blocks if any (```json ... ```)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        # Extract JSON object
        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            title = data.get("book_title")
            attr = data.get("attribute")

            if title and isinstance(title, str):
                title = title.strip()
                if len(title) >= 2:
                    return title, attr if attr and isinstance(attr, str) else None

            return None, None
    except Exception as e:
        logger.warning("LLM query parser fallback error on %r: %s", query, e)
        return None, None

    return None, None


async def parse_attribute_inquiry_with_llm_async(query: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Phiên bản async không block event loop của parse_attribute_inquiry_with_llm.
    """
    if not query or len(query.strip()) < 3:
        return None, None

    messages = [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": query.strip()}
    ]

    try:
        raw_resp = await call_llm_async(messages, temperature=0.0)
        if not raw_resp:
            return None, None

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw_resp.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned.strip())

        json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            title = data.get("book_title")
            attr = data.get("attribute")

            if title and isinstance(title, str):
                title = title.strip()
                if len(title) >= 2:
                    return title, attr if attr and isinstance(attr, str) else None

            return None, None
    except Exception as e:
        logger.warning("LLM query parser async fallback error on %r: %s", query, e)
        return None, None

    return None, None
