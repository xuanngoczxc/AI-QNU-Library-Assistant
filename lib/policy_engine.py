#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lib/policy_engine.py — Tra cứu Quy định, Quy chế & Nội quy Thư viện QNU.

Dữ liệu nguồn: Data/Dư thảo Quy định Sử dụng Thư viện final (4.11).docx
Chuyển đổi sang Markdown bằng thư viện MarkItDown (Microsoft).
Bao gồm:
  - 6 Chương, 16 Điều khoản quy chế chính thức của Trường Đại học Quy Nhơn.
  - Phân tích và trả lời các nội dung: Mượn - Trả tài liệu in, Khai thác tài nguyên số LMS,
    Sử dụng phòng tự học, Đền bù mất sách, Xử lý quá hạn, Làm thẻ thư viện, Giờ mở cửa, Trụ sở...
  - Hỗ trợ tùy chọn 2 chiều (Disambiguation): Luôn cung cấp nút tra cứu tài liệu trong kho
    để người dùng linh hoạt chuyển đổi giữa việc đọc quy định và tìm tài liệu học tập.
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from lib.text_utils import normalize_text

logger = logging.getLogger("qnu.library.policy")

DOCX_POLICY_PATH = Path("Data/Dư thảo Quy định Sử dụng Thư viện final (4.11).docx")
MD_POLICY_PATH = Path("Data/quy_dinh_su_dung_thu_vien.md")


def ensure_policy_markdown_converted() -> bool:
    """
    Sử dụng MarkItDown để chuyển đổi docx sang md nếu file md chưa có
    hoặc file docx mới hơn.
    """
    if not DOCX_POLICY_PATH.exists():
        logger.warning("Không tìm thấy file Word quy định tại %s", DOCX_POLICY_PATH)
        return False

    needs_conversion = False
    if not MD_POLICY_PATH.exists():
        needs_conversion = True
    else:
        try:
            docx_mtime = DOCX_POLICY_PATH.stat().st_mtime
            md_mtime = MD_POLICY_PATH.stat().st_mtime
            if docx_mtime > md_mtime:
                needs_conversion = True
        except Exception:
            needs_conversion = True

    if needs_conversion:
        try:
            from markitdown import MarkItDown
            logger.info("Đang áp dụng MarkItDown chuyển đổi %s sang Markdown...", DOCX_POLICY_PATH.name)
            md_converter = MarkItDown()
            result = md_converter.convert(str(DOCX_POLICY_PATH))
            with open(MD_POLICY_PATH, "w", encoding="utf-8") as f:
                f.write(result.text_content)
            logger.info("Chuyển đổi thành công! Đã lưu: %s (%d ký tự)", MD_POLICY_PATH.name, len(result.text_content))
            return True
        except Exception as e:
            logger.error("Lỗi khi chuyển đổi file quy định bằng MarkItDown: %s", e)
            return False

    return True


class PolicyEngine:
    """Quản lý và tra cứu các điều khoản quy định thư viện QNU."""

    def __init__(self):
        self.articles: List[Dict[str, Any]] = []
        self._initialized = False
        self.load_policy_articles()

    def load_policy_articles(self):
        """Đọc và lập chỉ mục các điều khoản từ file Markdown đã chuyển đổi."""
        ensure_policy_markdown_converted()

        if not MD_POLICY_PATH.exists():
            logger.warning("Chưa có file markdown quy định tại %s", MD_POLICY_PATH)
            return

        try:
            with open(MD_POLICY_PATH, "r", encoding="utf-8") as f:
                content = f.read()

            pattern = r'(\*\*Điều\s+(\d+)\.\s+([^\*\n]+)\*\*[\s\S]*?)(?=\*\*Điều\s+\d+\.|\Z)'
            matches = re.findall(pattern, content)

            self.articles = []
            for full_text, d_num_str, d_title in matches:
                d_num = int(d_num_str)
                clean_title = d_title.strip()
                self.articles.append({
                    "id": f"dieu_{d_num}",
                    "number": d_num,
                    "title": clean_title,
                    "full_text": full_text.strip(),
                    "keywords": self._extract_keywords(clean_title + " " + full_text)
                })

            self._initialized = True
            logger.info("PolicyEngine đã nạp thành công %d Điều khoản quy chế Thư viện QNU.", len(self.articles))
        except Exception as e:
            logger.error("Lỗi khi lập chỉ mục điều khoản quy định: %s", e)

    def _extract_keywords(self, text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_\u00C0-\u1EF9]{2,}\b', text.lower())
        stopwords = {"của", "và", "các", "có", "được", "cho", "trong", "khi", "tại", "với", "người", "trường", "theo"}
        return [w for w in words if w not in stopwords]

    def get_article(self, num: int) -> Optional[Dict[str, Any]]:
        for art in self.articles:
            if art["number"] == num:
                return art
        return None

    def search_policy(self, query: str, normalized_query: str) -> Optional[Dict[str, Any]]:
        """
        Tra cứu quy định thư viện dựa trên nội dung câu hỏi.
        Chỉ kích hoạt khi câu hỏi thực sự liên quan đến nội quy, quy chế, thủ tục thư viện.
        Bảo đảm tuyệt đối không can thiệp vào tính năng tra cứu tài liệu/sách trong kho.
        """
        if not query:
            return None

        q = normalized_query.strip()
        q_raw = query.strip()
        clean_kw = re.sub(r"^(?:ý\s+tôi\s+là|ý\s+là|cho\s+(?:em|mình|tôi)\s+hỏi|bạn\s+ơi|bot\s+ơi|hỏi\s+về|vậy\s+nếu\s+tôi|vậy\s+nếu|nếu\s+tôi|nếu)\s*", "", q_raw, flags=re.IGNORECASE).strip().rstrip("?.!")
        if not clean_kw:
            clean_kw = q_raw

        # Chuẩn hóa lỗi gõ phím tiếng Việt / Telex phổ biến (ciện -> viện, qui -> quy, ctdl, etc.)
        q = re.sub(r"\b(?:thu\s+)?cien\b", "thu vien", q)
        q = re.sub(r"\bcien\b", "vien", q)
        q = re.sub(r"\bqui\s+dinh\b", "quy dinh", q)
        q = re.sub(r"\bqui\s+che\b", "quy che", q)
        q = re.sub(r"\bnoi\s+qui\b", "noi quy", q)

        # ── 0. BẢO VỆ TRA CỨU SÁCH: Nếu truy vấn có chủ đích tìm sách/tài liệu trong kho thì bỏ qua ──
        if re.search(r"^(?:nhan\s+de\s*:|nhande\s*:|nhan\s+de\s+chinh|tac\s+gia\s*:|ma\s+tai\s+lieu\s*:)", q):
            return None

        # Không chặn các câu hỏi cá nhân về mượn trả (sách của tôi, sách tôi đang mượn...)
        is_personal_loan = bool(re.search(r"\b(?:sach\s+cua\s+(?:toi|minh|em|anh|chi|ban|ta)|sach\s+(?:toi|minh|em|anh|chi|ban|ta)\s+(?:da\s+|dang\s+)?muon)\b", q))

        if not is_personal_loan:
            # Phát hiện người dùng có ý định tìm kiếm sách/tài liệu/giáo trình cụ thể
            explicit_book_search = [
                r"\b(?:tim\s+sach|tim\s+tai\s+lieu|tim\s+giao\s+trinh|tra\s+cuu\s+sach|tra\s+cuu\s+tai\s+lieu)\b",
                r"\b(?:co\s+sach\s+nao|co\s+tai\s+lieu\s+nao|co\s+giao\s+trinh\s+nao)\b",
                r"\b(?:sach\s+lien\s+quan|tai\s+lieu\s+lien\s+quan)\b",
                r"\b(?:sach|tai\s+lieu|giao\s+trinh|luan\s+van)\s+ve\s+(?!nha\b)[a-z0-9]",
            ]
            # Loại trừ câu hỏi về quy định định lượng (mượn bao nhiêu cuốn về nhà, mượn được bao lâu)
            is_rule_quota_q = bool(re.search(r"\b(?:bao\s+nhieu\s+cuon|may\s+cuon|thoi\s+han|bao\s+lau)\b", q))
            if any(re.search(p, q) for p in explicit_book_search) and not is_rule_quota_q:
                return None

        if re.search(r"\b(?:tac\s+gia\s+(?:alexander|west|nguyen|tran|le|pham|morris|hoang))\b", q):
            return None
        if re.search(r"\b(?:cu\s+lao\s+xanh)\b", q):
            pass

        # ── 1. ĐỊA CHỈ & CƠ SỞ HOẠT ĐỘNG (Xác minh trụ sở, từ chối Cù Lao Xanh / cơ sở ảo) ──
        location_patterns = [
            r"\b(?:dia\s+chi\s+thu\s+vien|thu\s+vien\s+(?:qnu\s+)?(?:nam\s+)?o\s+dau|vi\s+tri\s+thu\s+vien|tru\s+so\s+thu\s+vien|co\s+so\s+thu\s+vien)\b",
            r"\b(?:thu\s+vien\s+(?:co\s+)?(?:o|tai)\s+cu\s+lao\s+xanh|cu\s+lao\s+xanh|co\s+so\s+2|co\s+so\s+khac)\b",
        ]
        if any(re.search(p, q) for p in location_patterns):
            ans = (
                "📍 **Địa chỉ & Trụ sở Thư viện Trường Đại học Quy Nhơn**\n\n"
                "- **Địa chỉ trụ sở duy nhất:**\n"
                "  Tòa nhà Thư viện (Trung tâm Số và Học liệu), Trường Đại học Quy Nhơn.\n"
                "  Số **170 An Dương Vương, Phường Nguyễn Văn Cừ, TP. Quy Nhơn, Tỉnh Bình Định**.\n\n"
                "- **Phạm vi phục vụ:**\n"
                "  Thư viện QNU hiện chỉ phục vụ tập trung tại khuôn viên chính (170 An Dương Vương).\n"
                "  Nhà trường **hoàn toàn KHÔNG có chi nhánh hay cơ sở tại Cù Lao Xanh** hay bất kỳ địa điểm nào khác ngoài trường.\n\n"
                "- **Hình thức phục vụ:**\n"
                "  Bạn đọc có thể đến trực tiếp thư viện tại trường vào các ngày làm việc từ Thứ Hai đến Thứ Sáu để đọc và mượn tài liệu."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa lúc mấy giờ?"},
                    {"label": "Quy định mượn sách", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                    {"label": "Tìm kiếm tài liệu", "query": "tìm sách về trí tuệ nhân tạo"}
                ]
            }

        # ── 2. GIỜ MỞ CỬA & THỜI GIAN LÀM VIỆC CHÍNH THỨC ──
        hours_patterns = [
            r"\b(?:khung\s+gio\s+nao|khung\s+gio|gio\s+mo\s+cua|mo\s+cua\s+luc\s+may\s+gio|may\s+gio\s+mo\s+cua|may\s+gio\s+dong\s+cua|thoi\s+gian\s+mo\s+cua|thoi\s+gian\s+phuc\s+vu|lich\s+phuc\s+vu|lich\s+mo\s+cua|gio\s+lam\s+viec|mo\s+cua\s+ngay\s+nao|co\s+mo\s+cua\s+thu\s+7|co\s+mo\s+cua\s+chu\s+nhat|cuoi\s+tuan\s+co\s+mo\s+cua)\b",
            r"\b(?:khi\s+nao\s+thu\s+vien\s+mo\s+cua|thu\s+vien\s+mo\s+cua\s+khi\s+nao|thu\s+vien\s+mo\s+cua\s+luc\s+nao|thu\s+vien\s+may\s+gio\s+mo|thu\s+vien\s+mo\s+cua\s+luc\s+may\s+gio)\b",
        ]
        if any(re.search(p, q) for p in hours_patterns):
            ans = (
                "🕒 **Thời gian mở cửa & Phục vụ của Thư viện QNU**\n\n"
                "Thư viện Trường Đại học Quy Nhơn phục vụ bạn đọc từ **Thứ Hai đến Thứ Sáu** hàng tuần:\n\n"
                "- **Buổi sáng:** Từ **07:00 đến 11:30**.\n"
                "- **Buổi chiều:** Từ **13:30 đến 17:00**.\n\n"
                "📌 **Lưu ý:**\n\n"
                "- **Thứ Bảy, Chủ Nhật và các ngày Lễ, Tết:** Thư viện tạm nghỉ phục vụ trực tiếp tại quầy.\n"
                "- **Hệ thống trực tuyến 24/7:** Cổng thông tin tra cứu danh mục và Tài nguyên số LMS luôn hoạt động liên tục phục vụ bạn đọc tự học từ xa."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Quy định mượn sách về nhà", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                    {"label": "Thủ tục làm thẻ thư viện", "query": "thủ tục làm thẻ thư viện thế nào?"},
                    {"label": "Tìm giáo trình CNTT", "query": "giáo trình ngành công nghệ thông tin"}
                ]
            }

        # ── 2.4. TRA CỨU SÁCH CÁ NHÂN ĐANG MƯỢN / TÀI KHOẢN MƯỢN TRẢ (PERSONAL LOANS) ──
        personal_loan_patterns = [
            r"^(?:sach\s+)?cua\s+(?:toi|minh|em|anh|chi|ban|ta)$",
            r"\b(?:sach\s+cua\s+(?:toi|minh|em|anh|chi|ban|ta)|sach\s+(?:toi|minh|em|anh|chi|ban|ta)\s+(?:da\s+|dang\s+)?muon)\b",
            r"\b(?:kiem\s+tra|xem|tra\s+cuu)\s+(?:sach\s+)?(?:da\s+|dang\s+)?(?:muon|cua\s+toi|cua\s+em|cua\s+minh)\b",
            r"\b(?:danh\s+sach\s+)?sach\s+(?:toi|em|minh)\s+(?:da\s+|dang\s+)?muon\b",
            r"\b(?:tai\s+khoan\s+muon\s+sach|lich\s+su\s+muon\s+sach|sach\s+chua\s+tra\s+cua\s+toi)\b",
        ]
        if any(re.search(p, q) for p in personal_loan_patterns):
            ans = (
                "🔒 **Hướng dẫn Tra cứu Thông tin Sách đang mượn cá nhân**\n\n"
                "Hiện tại, hệ thống trợ lý ảo chỉ hỗ trợ tra cứu danh mục tài liệu chung trong kho thư viện và tư vấn quy chế.\n"
                "Hệ thống **chưa thể truy cập vào tài khoản mượn trả cá nhân** của bạn đọc để đảm bảo quyền riêng tư và an toàn dữ liệu.\n\n"
                "Để kiểm tra danh sách sách đang mượn, hạn trả hoặc gia hạn sách, bạn vui lòng thực hiện:\n\n"
                "- **1. Tra cứu trực tuyến:**\n"
                "  Truy cập Cổng thông tin Thư viện QNU tại [thuvien.qnu.edu.vn](http://thuvien.qnu.edu.vn).\n"
                "  Đăng nhập bằng **Mã số sinh viên (MSSV)** / Mã bạn đọc cùng mật khẩu được cấp để xem chi tiết danh sách tài liệu đang mượn.\n\n"
                "- **2. Hỗ trợ trực tiếp:**\n"
                "  Đến **Bộ phận Mượn trả - Tầng 1 Trung tâm Số và Học liệu QNU** (170 An Dương Vương).\n"
                "  Cán bộ thư viện sẽ kiểm tra trực tiếp thông tin mượn trả trên hệ thống lưu thông của Nhà trường.\n\n"
                "💡 *Nếu bạn muốn tìm sách hoặc tài liệu học tập trong kho thư viện, vui lòng nhập tên sách, tác giả hoặc ngành học cụ thể nhé!*"
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Quy định mượn trả sách (Điều 3)", "query": "quy định mượn trả tài liệu in"},
                    {"label": "Quy định xử lý quá hạn (Điều 13)", "query": "trả sách quá hạn bị xử lý thế nào?"},
                    {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa vào những khung giờ nào?"}
                ]
            }

        # ── 2.5. TỔNG QUAN QUY ĐỊNH, QUY CHẾ THƯ VIỆN QNU ──
        general_policy_patterns = [
            r"\b(?:cac\s+)?quy\s+dinh\s+(?:ve\s+viec\s+su\s+dung\s+thu\s+vien|su\s+dung\s+thu\s+vien|tai\s+thu\s+vien|cua\s+thu\s+vien|o\s+thu\s+vien|thu\s+vien\s+qnu|thu\s+vien)\b",
            r"\b(?:cac\s+)?quy\s+che\s+(?:su\s+dung\s+thu\s+vien|tai\s+thu\s+vien|cua\s+thu\s+vien|o\s+thu\s+vien|thu\s+vien\s+qnu|thu\s+vien)\b",
            r"\b(?:noi\s+quy\s+va\s+quy\s+dinh|toan\s+bo\s+quy\s+dinh|tong\s+quan\s+quy\s+dinh|quy\s+dinh\s+chung)\b",
            r"\bthu\s+vien\s+(?:co\s+)?(?:nhung\s+)?quy\s+dinh\s+gi\b",
            r"\b(?:cac\s+quy\s+dinh|cac\s+quy\s+che)\b",
            r"\b(?:quy\s+dinh|quy\s+che)\s+tai\s+thu\s+vien\b",
            r"\b(?:tim\s+hieu|cho\s+(?:em|minh|toi)?\s*biet)?\s*(?:ve\s+)?(?:cac\s+)?quy\s+dinh\s+(?:tai\s+)?thu\s+vien\b",
        ]
        if any(re.search(p, q) for p in general_policy_patterns) and not re.search(r"\b(?:cuon\s+|quyen\s+|tac\s+gia|nhan\s+de)\b", q):
            ans = (
                "📋 **Tổng quan Quy định về việc Sử dụng Thư viện Trường Đại học Quy Nhơn**\n\n"
                "Quy định ban hành nhằm chuẩn hóa công tác phục vụ, khai thác tài nguyên và không gian học tập tại Thư viện QNU (gồm **6 Chương và 16 Điều khoản**).\n\n"
                "Dưới đây là các chính sách cốt lõi bạn đọc cần lưu ý:\n\n"
                "**1. 📖 Mượn - Trả tài liệu in (Theo Điều 3):**\n\n"
                "- **Phòng Đọc mở:** Được mượn tối đa **05 tài liệu/lần** để đọc tại chỗ trong ngày.\n"
                "- Tuyệt đối không mang tài liệu ra khỏi Phòng Đọc mở và phải hoàn trả cho thủ thư trước khi kết thúc ca phục vụ trong ngày.\n"
                "- **Phòng Giáo trình (mượn về nhà):** Được mượn đồng thời tối đa **05 tài liệu** với thời hạn tối đa **01 học kỳ**.\n"
                "- Thời hạn hoàn trả giáo trình: Chậm nhất **01 tuần sau kỳ thi kết thúc học phần** tương ứng.\n"
                "- **Mượn đọc trong khuôn viên Trường:** Mỗi lần được mượn tối đa **02 tài liệu** và phải xuất trình cho thủ thư ghi nhận lên hệ thống.\n"
                "- **Các tài liệu khác mượn về:** Thời hạn tối đa phải hoàn trả là **90 ngày**.\n\n"
                "**2. 🏛️ Không gian & Phòng Tự học (Theo Điều 8 và Điều 11):**\n\n"
                "- Phục vụ nhu cầu học tập, nghiên cứu cá nhân và thảo luận nhóm theo đúng chức năng của từng phòng.\n"
                "- Bạn đọc được phép mang tài liệu đã mượn của Thư viện hoặc tài liệu cá nhân vào Phòng Tự học để học tập.\n"
                "- Chỉ sử dụng phòng trong thời gian phục vụ của Thư viện và phải sử dụng đúng mục đích.\n"
                "- Phải giữ gìn trật tự, vệ sinh chung, bảo vệ tài sản và trang thiết bị của Nhà trường.\n"
                "- Vui lòng gửi cặp sách, ba lô tại Tủ đồ cá nhân trước khi vào các phòng chức năng.\n\n"
                "**3. 💻 Tài nguyên số & Học liệu LMS (Theo Điều 4, 5 và 7):**\n\n"
                "- Khai thác giáo trình số, sách điện tử, tạp chí khoa học, luận án, luận văn và bài giảng số LMS bằng tài khoản sinh viên được cấp.\n"
                "- Được tra cứu, đọc trực tuyến tại Thư viện hoặc truy cập từ xa qua Cổng thông tin Thư viện.\n"
                "- Người dùng có trách nhiệm bảo mật thông tin tài khoản truy cập và không chia sẻ cho người khác.\n"
                "- Nghiêm cấm tự ý sao chép, dùng công cụ tự động để tải hàng loạt hoặc phát tán học liệu trái phép ra ngoài hệ thống.\n\n"
                "**4. ⚖️ Xử lý vi phạm, Quá hạn & Đền bù mất sách (Theo Điều 13):**\n\n"
                "- **Quá hạn từ 01 đến 07 ngày:** Thư viện gửi thông báo nhắc nhở tự động qua hệ thống.\n"
                "- **Quá hạn từ 08 đến 30 ngày:** Tạm ngừng quyền mượn tài liệu; sau khi trả sách, tiếp tục bị tạm dừng quyền mượn bằng số ngày quá hạn.\n"
                "- **Quá hạn trên 30 ngày:** Tạm khóa tài khoản thư viện và thông báo nghĩa vụ chưa hoàn thành.\n"
                "- **Quá hạn trên 90 ngày:** Người sử dụng bắt buộc phải thực hiện đền sách theo quy định bồi thường làm mất tài liệu.\n"
                "- **Làm mất tài liệu:** Có trách nhiệm mua tài liệu mới (cùng tên, tác giả, NXB, năm xuất bản hoặc tái bản mới nhất) để thay thế.\n"
                "- Trường hợp không mua được trên thị trường: Tài liệu xuất bản ≤ 10 năm bồi thường bằng **02 lần giá bìa**; tài liệu xuất bản > 10 năm hoặc tài liệu nội sinh tính **Số trang × 500 đồng** (tối thiểu 50.000 đồng/cuốn).\n"
                "- **Làm hư hỏng nhẹ:** Người sử dụng chịu toàn bộ chi phí sửa chữa, phục hồi tài liệu.\n"
                "- Thời hạn hoàn thành nghĩa vụ bồi hoàn: Trong vòng **07 ngày làm việc** kể từ ngày nhận thông báo.\n"
                "- **Chế tài tốt nghiệp:** Nợ sách chưa hoàn trả sẽ bị khóa tài khoản và **không được Thư viện xác nhận hoàn tất thủ tục khi xét tốt nghiệp, rút hồ sơ, thôi học hoặc chuyển công tác**.\n\n"
                "**5. 🕒 Thời gian phục vụ & Trụ sở Thư viện:**\n\n"
                "- Thời gian mở cửa: Từ **Thứ Hai đến Thứ Sáu** hàng tuần.\n"
                "- Buổi sáng: Từ **07:00 đến 11:30**.\n"
                "- Buổi chiều: Từ **13:30 đến 17:00**.\n"
                "- Thứ Bảy, Chủ Nhật và các ngày Lễ, Tết: Tạm nghỉ phục vụ trực tiếp tại quầy.\n"
                "- Cổng thông tin & Tài nguyên số trực tuyến: Hoạt động liên tục **24/7** để phục vụ tra cứu từ xa.\n"
                "- Địa chỉ trụ sở duy nhất: Tòa nhà Thư viện (Trung tâm Số và Học liệu), Số **170 An Dương Vương, TP. Quy Nhơn, Tỉnh Bình Định**.\n"
                "- Thư viện QNU không có chi nhánh hay cơ sở phụ ngoài trường (kể cả Cù Lao Xanh)."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Quy định mượn trả sách (Điều 3)", "query": "quy định mượn trả tài liệu in"},
                    {"label": "Bồi thường mất sách (Điều 13)", "query": "làm mất sách thư viện thì đền thế nào?"},
                    {"label": "Phòng tự học (Điều 8)", "query": "quy định sử dụng phòng tự học"},
                    {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa vào những khung giờ nào?"}
                ]
            }

        # ── 3. BỒI THƯỜNG / LÀM MẤT SÁCH / LÀM HỎNG TÀI LIỆU (ĐIỀU 13 KHOẢN 2) ──
        lost_damaged_patterns = [
            r"\b(?:mat\s+sach|lam\s+mat\s+sach|den\s+sach|boi\s+thuong\s+sach|lam\s+hong\s+sach|rach\s+sach|hong\s+tai\s+lieu|mat\s+tai\s+lieu|lam\s+mat\s+giao\s+trinh|gia\s+bia|den\s+bao\s+nhieu|den\s+tien|boi\s+thuong\s+tai\s+lieu)\b",
            r"\b(?:den\s+bu|boi\s+hoan)\b.*?\b(?:sach|tai\s+lieu)\b",
        ]
        if any(re.search(p, q) for p in lost_damaged_patterns):
            ans = (
                "⚖️ **Quy định Bồi thường khi Làm Mất hoặc Hư Hỏng Tài liệu (Theo Điều 13 - Quy định Thư viện QNU)**\n\n"
                "Căn cứ Khoản 2 Điều 13 Quy định sử dụng Thư viện Trường Đại học Quy Nhơn:\n\n"
                "**1. Trường hợp làm mất tài liệu:**\n\n"
                "- Người sử dụng có trách nhiệm mua tài liệu mới cùng tên, tác giả, nhà xuất bản và năm xuất bản (hoặc tái bản mới nhất) để thay thế.\n"
                "- Trường hợp không thể mua tài liệu thay thế trên thị trường:\n"
                "  + Đối với tài liệu xuất bản trong vòng 10 năm: Bồi thường bằng **02 lần giá bìa** của tài liệu.\n"
                "  + Đối với tài liệu xuất bản trên 10 năm, tài liệu không có giá bìa, tài liệu biếu tặng hoặc tài liệu nội sinh: Mức bồi thường được tính bằng **Số trang × 500 đồng** (mức áp dụng tối thiểu là 50.000 đồng/tài liệu).\n"
                "  + Đối với tài liệu quý hiếm, bản duy nhất: Giám đốc Trung tâm Số và Học liệu quyết định mức bồi thường cụ thể đối với từng trường hợp.\n\n"
                "**2. Trường hợp làm hư hỏng tài liệu:**\n\n"
                "- Nếu hư hỏng nhẹ và có thể khắc phục: Người sử dụng chịu toàn bộ chi phí sửa chữa, phục hồi tài liệu.\n"
                "- Nếu hư hỏng nặng không thể phục hồi: Xử lý bồi thường tương đương trường hợp làm mất tài liệu.\n\n"
                "**3. Thời hạn và chế tài bắt buộc:**\n\n"
                "- Bạn đọc có trách nhiệm hoàn thành nghĩa vụ bồi hoàn (trả sách mới hoặc nộp tiền bồi thường) trong thời hạn **07 ngày làm việc** kể từ ngày nhận thông báo.\n"
                "- Nếu chưa hoàn thành nghĩa vụ bồi thường, bạn đọc sẽ bị **khóa tài khoản thư viện**.\n"
                "- Đồng thời **không được Thư viện xác nhận hoàn tất thủ tục khi xét tốt nghiệp, rút hồ sơ, thôi học hoặc chuyển công tác**."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "🔍 Tìm tài liệu có từ khóa này trong kho", "query": f"tìm sách về {clean_kw}"},
                    {"label": "Quy định phạt trả sách quá hạn", "query": "trả sách quá hạn bị xử lý thế nào?"},
                    {"label": "Quy định mượn trả sách", "query": "quy định mượn trả tài liệu in"},
                    {"label": "Giờ mở cửa thư viện", "query": "thời gian mở cửa thư viện"}
                ]
            }

        # ── 4. XỬ LÝ TÀI LIỆU QUÁ HẠN / TRẢ TRỄ HẠN (ĐIỀU 13 KHOẢN 1) ──
        overdue_patterns = [
            r"\b(?:qua\s+han|tra\s+muon|tre\s+han|phat\s+qua\s+han|tra\s+sach\s+tre|nop\s+sach\s+muon|tre\s+bao\s+nhieu\s+ngay|bi\s+khoa\s+tai\s+khoan|khoa\s+the|phat\s+tien\s+qua\s+han)\b",
            r"\b(?:tai\s+lieu\s+qua\s+han|xu\s+ly\s+qua\s+han|qua\s+han\s+bao\s+lau)\b",
            r"\b(?:muon\s+sach|muon|giu\s+sach|khong\s+tra)\s+(?:qua|tre|qua\s+han|tren)\s+(?:\d+|bao\s+nhieu|bao\s+lau)\s*(?:ngay|thang|tuan)?\b",
            r"\b(?:qua|tren)\s+(?:90|30|7|\d+)\s+ngay\b",
            r"\b(?:muon\s+sach|muon|giu\s+sach)\s+(?:qua\s+han|tre\s+han|qua\s+lau)\b",
            r"\b(?:khong\s+tra\s+sach|chua\s+tra\s+sach|quen\s+tra\s+sach|khong\s+chiu\s+tra\s+sach)\b",
            r"\b(?:qua\s+han|tre\s+han|tra\s+muon|tra\s+tre)\s+(?:thi\s+sao|nhu\s+the\s+nao|bi\s+gi|co\s+sao\s+khong)\b",
            r"\bmuon\s+(?:sach\s+)?(?:qua\s+)?90\s+ngay\b",
        ]
        if any(re.search(p, q) for p in overdue_patterns) and not re.search(r"\b(?:tac\s+gia|nhan\s+de)\b", q):
            # Nếu người dùng hỏi cụ thể về mốc 90 ngày
            if re.search(r"\b(?:90\s+ngay|chin\s+muoi\s+ngay)\b", q):
                ans = (
                    "⚖️ **Quy định Xử lý khi Mượn sách Quá hạn trên 90 ngày (Theo Điều 13 - Quy định Thư viện QNU)**\n\n"
                    "Căn cứ Điểm 5 Khoản 1 Điều 13 Quy định sử dụng Thư viện Trường Đại học Quy Nhơn:\n\n"
                    "👉 **Mượn sách quá hạn trên 90 ngày: Bạn đọc BẮT BUỘC PHẢI THỰC HIỆN ĐỀN SÁCH** theo quy định bồi thường tài liệu làm mất (Khoản 2 Điều 13).\n\n"
                    "**1. Hình thức bồi thường đền sách:**\n\n"
                    "- Bạn đọc có trách nhiệm mua tài liệu mới (cùng tên, tác giả, nhà xuất bản và năm xuất bản hoặc tái bản mới nhất) để thay thế.\n"
                    "- Trường hợp không thể mua thay thế trên thị trường:\n"
                    "  + Tài liệu xuất bản trong vòng 10 năm: Bồi thường bằng **02 lần giá bìa** của tài liệu.\n"
                    "  + Tài liệu xuất bản trên 10 năm, tài liệu không giá bìa, biếu tặng hoặc nội sinh: Bồi thường theo công thức **Số trang × 500 đồng** (tối thiểu 50.000 đồng/tài liệu).\n"
                    "  + Tài liệu quý hiếm, bản duy nhất: Giám đốc Trung tâm Số và Học liệu quyết định mức bồi thường cụ thể.\n\n"
                    "**2. Chế tài xử lý đi kèm:**\n\n"
                    "- Tài khoản thư viện sẽ bị **khóa hoàn toàn**.\n"
                    "- **Không được xác nhận hoàn tất thủ tục** khi xét tốt nghiệp, rút hồ sơ, thôi học hoặc chuyển công tác cho đến khi hoàn thành nghĩa vụ đền sách.\n\n"
                    "**3. Các mốc thời hạn mượn & xử lý quá hạn cần lưu ý:**\n\n"
                    "- Quá hạn từ 01 đến 07 ngày: Thư viện gửi thông báo nhắc nhở tự động qua phần mềm.\n"
                    "- Quá hạn từ 08 đến 30 ngày: Tạm ngừng quyền mượn tài liệu; sau khi trả tài liệu, tiếp tục bị tạm dừng quyền mượn bằng đúng số ngày quá hạn.\n"
                    "- Quá hạn trên 30 ngày: Tạm khóa tài khoản thư viện và thông báo nghĩa vụ chưa hoàn thành.\n"
                    "- Quá hạn trên 90 ngày: Bắt buộc đền sách.\n"
                    "- Thời hạn mượn giáo trình về nhà: Tối đa 01 học kỳ (hoàn trả chậm nhất 01 tuần sau kỳ thi kết thúc học phần).\n"
                    "- Thời hạn mượn các tài liệu khác: Tối đa 90 ngày."
                )
                return {
                    "answer": ans,
                    "sources": [],
                    "total_sources": 0,
                    "total_found": 0,
                    "follow_up_suggestions": [
                        {"label": "🔍 Tìm tài liệu có từ khóa này trong kho", "query": f"tìm sách về {clean_kw}"},
                        {"label": "Bồi thường mất sách (Điều 13)", "query": "làm mất sách thư viện thì đền thế nào?"},
                        {"label": "Quy định mượn trả sách (Điều 3)", "query": "quy định mượn trả tài liệu in"},
                        {"label": "Thủ tục xét tốt nghiệp", "query": "chưa trả sách có được xét tốt nghiệp không?"}
                    ]
                }

            ans = (
                "⚠️ **Quy định Xử lý Tài liệu Quá hạn (Theo Khoản 1 Điều 13 - Quy định Thư viện QNU)**\n\n"
                "Hệ thống Thư viện QNU áp dụng các mức xử lý quá hạn như sau:\n\n"
                "- **Quá hạn từ 01 đến 07 ngày:**\n"
                "  Thư viện gửi thông báo nhắc nhở tự động qua hệ thống phần mềm của người sử dụng.\n\n"
                "- **Quá hạn từ 08 đến 30 ngày:**\n"
                "  Tạm ngừng quyền mượn tài liệu của người sử dụng.\n"
                "  Sau khi hoàn trả tài liệu quá hạn, bạn đọc tiếp tục bị tạm dừng quyền mượn trong thời gian tương ứng với số ngày quá hạn.\n\n"
                "- **Quá hạn trên 30 ngày:**\n"
                "  Tạm khóa tài khoản thư viện.\n"
                "  Đồng thời thông báo nghĩa vụ chưa hoàn thành cho người sử dụng.\n\n"
                "- **Quá hạn trên 90 ngày:**\n"
                "  Người sử dụng phải thực hiện đền sách theo quy định bồi thường tài liệu làm mất (mua sách mới thay thế hoặc bồi thường theo giá bìa/số trang)."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "🔍 Tìm tài liệu có từ khóa này trong kho", "query": f"tìm sách về {clean_kw}"},
                    {"label": "Quy định bồi thường mất sách", "query": "làm mất sách thư viện thì đền thế nào?"},
                    {"label": "Thời hạn mượn giáo trình", "query": "sinh viên được mượn giáo trình bao lâu?"},
                    {"label": "Thủ tục khi xét tốt nghiệp", "query": "chưa trả sách thư viện có được xét tốt nghiệp không?"}
                ]
            }

        # ── 5. THỦ TỤC TỐT NGHIỆP / RÚT HỒ SƠ / CHUYỂN CÔNG TÁC (ĐIỀU 13 KHOẢN 3) ──
        graduation_patterns = [
            r"\b(?:xet\s+tot\s+nghiep|tot\s+nghiep|rut\s+ho\s+so|thoi\s+hoc|chuyen\s+cong\s+tac|ra\s+truong|xac\s+nhan\s+thu\s+vien|thu\s+tuc\s+tot\s+nghiep)\b",
            r"\b(?:no\s+sach|chua\s+tra\s+sach)\b.*?\b(?:tot\s+nghiep|rut\s+ho\s+so|ra\s+truong)\b"
        ]
        if any(re.search(p, q) for p in graduation_patterns):
            ans = (
                "🎓 **Xác nhận Thư viện khi Xét Tốt nghiệp, Rút hồ sơ hoặc Chuyển công tác (Theo Điều 13 Khoản 3)**\n\n"
                "Theo Khoản 3 Điều 13 Quy định sử dụng Thư viện Trường Đại học Quy Nhơn:\n\n"
                "- **Quy định bắt buộc:**\n"
                "  Trường hợp người sử dụng chưa hoàn thành nghĩa vụ trả sách hoặc bồi thường thiệt hại, bạn đọc sẽ bị **khóa tài khoản thư viện**.\n"
                "  Đồng thời **KHÔNG ĐƯỢC Thư viện xác nhận hoàn tất thủ tục khi xét tốt nghiệp, rút hồ sơ, thôi học hoặc chuyển công tác**.\n\n"
                "- **Khuyến nghị cho sinh viên:**\n"
                "  Trước khi làm thủ tục tốt nghiệp hoặc ra trường, bạn đọc cần kiểm tra kỹ tài khoản thư viện trực tuyến tại [thuvien.qnu.edu.vn](http://thuvien.qnu.edu.vn).\n"
                "  Hoàn trả toàn bộ sách, giáo trình đã mượn cho Thư viện.\n"
                "  Thanh toán các nghĩa vụ bồi hoàn phát sinh (nếu có) tại quầy thủ thư Tầng 1 Thư viện để được ký xác nhận hoàn tất thủ tục kịp thời."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Xử lý khi làm mất sách", "query": "làm mất sách thư viện thì đền thế nào?"},
                    {"label": "Thời gian làm việc của thư viện", "query": "thư viện mở cửa vào những khung giờ nào?"},
                    {"label": "Quy định mượn trả sách", "query": "quy định mượn trả tài liệu in"}
                ]
            }

        # ── 6. QUY ĐỊNH MƯỢN - TRẢ TÀI LIỆU IN (ĐIỀU 3) ──
        borrow_patterns = [
            r"\b(?:muon\s+sach\s+ve\s+nha|muon\s+giao\s+trinh|muon\s+tai\s+lieu\s+in|phong\s+doc\s+mo|muon\s+ve\s+nha|muon\s+duoc\s+may\s+cuon|muon\s+bao\s+nhieu\s+cuon|thoi\s+gian\s+muon|thoi\s+han\s+muon|muon\s+duoc\s+bao\s+lau)\b",
            r"\b(?:quy\s+dinh\s+(?:ve\s+viec\s+)?(?:muon\s+tra|tra\s+sach|muon\s+sach|gia\s+han))\b",
            r"\b(?:thoi\s+han|khi\s+nao\s+phai)\s+(?:tra\s+sach|hoan\s+tra)\b",
            r"\b(?:tra\s+sach|muon\s+sach)\s+(?:cua\s+thu\s+vien|tai\s+thu\s+vien)\b",
            r"\bmuon\s+(?:duoc\s+)?(?:sach\s+|giao\s+trinh\s+|tai\s+lieu\s+)?(?:trong\s+)?bao\s+lau\b",
        ]
        if any(re.search(p, q) for p in borrow_patterns) and not re.search(r"\b(?:cuon\s+|quyen\s+|giao\s+trinh\s+)(?!sach\b|tai\s+lieu\b|gi\b|nao\b|bao\s+nhieu\b|bao\s+lau\b|may\b)[a-z0-9]", q):
            ans = (
                "📖 **Quy định Mượn - Trả Tài liệu in tại Thư viện QNU (Theo Điều 3)**\n\n"
                "Việc mượn và trả tài liệu in được thực hiện và ghi nhận trên hệ thống quản lý của thư viện như sau:\n\n"
                "**1. Mượn đọc tại các Phòng Đọc mở (Khoản 1a & Khoản 2b):**\n\n"
                "- Mỗi lần được mượn sử dụng đồng thời tối đa **05 (năm) tài liệu**.\n"
                "- Phải xuất trình tài liệu cho thủ thư để ghi nhận mượn lên hệ thống trước khi sử dụng.\n"
                "- Chỉ đọc tại chỗ trong thời gian phục vụ, **tuyệt đối không được mang ra khỏi Phòng Đọc mở**.\n"
                "- Phải hoàn trả cho thủ thư trong ngày và trước khi kết thúc ca phục vụ.\n\n"
                "**2. Mượn đọc trong khuôn viên Trường (Khoản 1b):**\n\n"
                "- Mỗi lần được mượn tối đa **02 (hai) tài liệu**.\n"
                "- Phải xuất trình tài liệu cho thủ thư để ghi nhận mượn lên hệ thống trước khi sử dụng.\n\n"
                "**3. Mượn giáo trình về nhà (Phòng Giáo trình - Khoản 1c & Khoản 2c):**\n\n"
                "- Được mượn đồng thời tối đa **05 (năm) tài liệu** để sử dụng ngoài khuôn viên Trường.\n"
                "- **Thời gian mượn:** Tối đa **01 (một) học kỳ**.\n"
                "- **Thời hạn hoàn trả:** Chậm nhất **01 tuần sau kỳ thi kết thúc học phần** tương ứng.\n\n"
                "**4. Các tài liệu khác mượn về (Khoản 2d):**\n\n"
                "- Thời hạn tối đa phải hoàn trả là **90 ngày**.\n"
                "- Trong trường hợp cần thiết, Nhà trường có quyền thu hồi tài liệu trước thời hạn."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "🔍 Tìm tài liệu có từ khóa này trong kho", "query": f"tìm sách về {clean_kw}"},
                    {"label": "Quy định xử lý quá hạn (Điều 13)", "query": "trả sách quá hạn bị xử lý thế nào?"},
                    {"label": "Bồi thường khi mất sách (Điều 13)", "query": "làm mất sách thư viện thì đền thế nào?"},
                    {"label": "Sử dụng phòng tự học (Điều 8)", "query": "quy định sử dụng phòng tự học"}
                ]
            }

        # ── 7. SỬ DỤNG PHÒNG TỰ HỌC & THẢO LUẬN NHÓM (ĐIỀU 8) ──
        study_room_patterns = [
            r"\b(?:phong\s+tu\s+hoc|thao\s+luan\s+nhom|hoc\s+nhom|mang\s+tai\s+lieu\s+vao\s+phong\s+tu\s+hoc|mang\s+sach\s+rieng|hoc\s+ca\s+nhan)\b"
        ]
        if any(re.search(p, q) for p in study_room_patterns):
            ans = (
                "🏛️ **Quy định Sử dụng Phòng Tự học (Theo Điều 8 - Quy định Thư viện QNU)**\n\n"
                "Căn cứ Điều 8 Quy định sử dụng Thư viện Trường Đại học Quy Nhơn:\n\n"
                "- **1. Mục đích sử dụng:**\n"
                "  Phòng Tự học được sử dụng để học tập, nghiên cứu cá nhân và thảo luận nhóm theo đúng chức năng của từng phòng.\n\n"
                "- **2. Quy định mang tài liệu vào phòng:**\n"
                "  Người sử dụng thư viện **hoàn toàn ĐƯỢC PHÉP mang tài liệu đã mượn của Thư viện hoặc tài liệu cá nhân vào Phòng Tự học** để phục vụ học tập, nghiên cứu.\n\n"
                "- **3. Thời gian phục vụ:**\n"
                "  Chỉ được sử dụng Phòng Tự học trong thời gian phục vụ của Thư viện và phải sử dụng đúng mục đích, chức năng của từng phòng.\n\n"
                "- **4. Trách nhiệm của bạn đọc:**\n"
                "  Có trách nhiệm giữ gìn trật tự, vệ sinh chung, bảo vệ tài sản và trang thiết bị trong Phòng Tự học.\n"
                "  Không gây tiếng ồn làm ảnh hưởng đến việc học tập, nghiên cứu của người sử dụng khác.\n"
                "  Gửi cặp sách, ba lô tại Tủ đồ cá nhân trước khi vào phòng tự học."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa vào những khung giờ nào?"},
                    {"label": "Nội quy chung thư viện", "query": "nội quy phòng đọc thư viện"},
                    {"label": "Quy định mượn sách", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"}
                ]
            }

        # ── 8. HỌC LIỆU SỐ LMS & TÀI NGUYÊN GIÁO DỤC MỞ OER (ĐIỀU 4, 5, 7) ──
        digital_patterns = [
            r"\b(?:tai\s+nguyen\s+so|hoc\s+lieu\s+so|he\s+thong\s+lms|tai\s+nguyen\s+giao\s+duc\s+mo|oer|tai\s+lieu\s+so|tai\s+ve\s+may|doc\s+truc\s+tuyen)\b"
        ]
        if any(re.search(p, q) for p in digital_patterns) and not re.search(r"\b(?:sach\s+in|sach\s+giay|ke\s+sach)\b", q):
            ans = (
                "💻 **Khai thác Tài nguyên số & Học liệu LMS (Theo Điều 4, 5, 7 - Quy định Thư viện QNU)**\n\n"
                "**1. Quản lý và khai thác tài nguyên thông tin số (Điều 4):**\n\n"
                "- Tài nguyên số bao gồm: Giáo trình, sách, tạp chí điện tử, luận án, luận văn, khóa luận, công trình NCKH và cơ sở dữ liệu số.\n"
                "- Viên chức, người lao động và người học được khai thác tài nguyên số theo quyền truy cập được cấp.\n"
                "- Truy cập thông qua Cổng thông tin thư viện tại Thư viện, trong mạng nội bộ hoặc từ xa qua mạng Internet.\n"
                "- Người dùng có trách nhiệm bảo mật tài khoản, không cho mượn hoặc chia sẻ thông tin đăng nhập.\n"
                "- Sử dụng tài nguyên đúng mục đích học tập, giảng dạy, nghiên cứu và trích dẫn nguồn đầy đủ khi sử dụng.\n"
                "- Tuyệt đối không dùng công cụ tự động để thu thập, tải hàng loạt tài liệu khi chưa được phép.\n\n"
                "**2. Sử dụng học liệu số trên hệ thống LMS (Điều 5):**\n\n"
                "- Học liệu số phục vụ đào tạo gồm: Đề cương chi tiết học phần, hướng dẫn học tập, video giảng dạy, bài giảng điện tử và xuất bản phẩm bổ trợ.\n"
                "- Giảng viên và người học được cấp tài khoản hợp lệ để khai thác học liệu số theo đúng phân quyền.\n"
                "- Nghiêm cấm tự ý sao chép, tải hàng loạt hoặc phát tán học liệu ra ngoài hệ thống khi chưa được phép.\n\n"
                "**3. Tài nguyên giáo dục mở - OER (Điều 7):**\n\n"
                "- Bao gồm: Giáo trình mở, bài giảng mở, khóa học trực tuyến MOOCs và ấn phẩm học thuật truy cập mở (Open Access).\n"
                "- Bạn đọc được tự do tra cứu, tiếp cận và khai thác OER phục vụ giảng dạy, học tập và nghiên cứu.\n"
                "- Phải tuân thủ nghiêm túc các điều kiện của từng loại giấy phép mở và thực hiện ghi nhận công lao tác giả (ghi công tác giả/chủ thể quyền)."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Làm thẻ & tài khoản", "query": "thủ tục làm thẻ thư viện thế nào?"},
                    {"label": "Quy định mượn sách in", "query": "quy định mượn trả tài liệu in"},
                    {"label": "Tìm kiếm tài liệu số", "query": "tìm sách về trí tuệ nhân tạo"}
                ]
            }

        # ── 9. THỦ TỤC LÀM THẺ & CẤP TÀI KHOẢN THƯ VIỆN (ĐIỀU 9, 10) ──
        card_patterns = [
            r"\b(?:lam\s+the\s+thu\s+vien|thu\s+tuc\s+lam\s+the|cap\s+the|kich\s+hoat\s+the|tai\s+khoan\s+thu\s+vien|lam\s+the\s+sinh\s+vien|the\s+ban\s+doc)\b"
        ]
        if any(re.search(p, q) for p in card_patterns):
            ans = (
                "💳 **Thủ tục Cấp Thẻ & Tài khoản Thư viện QNU (Theo Điều 10 - Quy định Thư viện QNU)**\n\n"
                "- **1. Đối với sinh viên, học viên và cán bộ của Trường:**\n"
                "  Sử dụng Thẻ sinh viên tích hợp đồng thời làm Thẻ Thư viện.\n"
                "  Tài khoản điện tử đăng nhập hệ thống thư viện và LMS được cấp tự động theo Mã số sinh viên (MSSV) ngay khi nhập học.\n"
                "  Được sử dụng miễn phí tài nguyên thông tin tại Thư viện và mượn tài liệu in theo thời hạn quy định.\n\n"
                "- **2. Đối với bạn đọc ngoài trường:**\n"
                "  Đến trực tiếp quầy thủ thư tại Tầng 1 Thư viện QNU (170 An Dương Vương, TP. Quy Nhơn).\n"
                "  Xuất trình Căn cước công dân (CCCD) hoặc Hộ chiếu còn hiệu lực.\n"
                "  Làm thủ tục đăng ký và cấp thẻ bạn đọc ngoài trường theo quy định của Nhà trường."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Quy định mượn sách", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                    {"label": "Phòng tự học", "query": "quy định sử dụng phòng tự học"},
                    {"label": "Giờ mở cửa", "query": "thư viện mở cửa vào những khung giờ nào?"}
                ]
            }

        # ── 10. QUYỀN HẠN & TRÁCH NHIỆM BẠN ĐỌC (ĐIỀU 10, 11) ──
        rights_patterns = [
            r"\b(?:quyen\s+han|trach\s+nhiem\s+cua\s+sinh\s+vien|nghia\s+vu\s+ban\s+doc|trach\s+nhiem\s+ban\s+doc|quyen\s+loi\s+khi\s+su\s+dung)\b"
        ]
        if any(re.search(p, q) for p in rights_patterns):
            ans = (
                "📜 **Quyền và Trách nhiệm của Bạn đọc (Theo Điều 10, 11 - Quy định Thư viện QNU)**\n\n"
                "**1. Quyền của người sử dụng thư viện (Điều 10):**\n\n"
                "- Được tiếp cận và sử dụng tài nguyên thông tin, tiện ích và dịch vụ thư viện theo quy định.\n"
                "- Được sử dụng miễn phí tài nguyên thông tin tại Thư viện và mượn tài liệu in theo thời hạn quy định.\n"
                "- Được cấp tài khoản điện tử để khai thác tài nguyên thông tin số và học liệu trực tuyến.\n"
                "- Được hướng dẫn, hỗ trợ kỹ năng tìm kiếm, lựa chọn, khai thác và đánh giá thông tin.\n"
                "- Được sử dụng không gian học tập, phòng tự học cá nhân và thảo luận nhóm.\n"
                "- Được phản ánh, kiến nghị về chất lượng dịch vụ và yêu cầu hỗ trợ khi gặp khó khăn, sự cố.\n\n"
                "**2. Trách nhiệm của người sử dụng thư viện (Điều 11):**\n\n"
                "- Chấp hành nghiêm chỉnh Quy định, nội quy của Thư viện và của Trường Đại học Quy Nhơn.\n"
                "- Xuất trình thẻ sinh viên hoặc giấy tờ hợp lệ khi sử dụng Thư viện và khi mượn tài liệu.\n"
                "- Sử dụng tài nguyên thông tin, tài khoản điện tử, trang thiết bị đúng mục đích và quy định.\n"
                "- Bảo mật tài khoản truy cập; không tải xuống hoặc chia sẻ trái phép tài liệu có bản quyền.\n"
                "- Giữ gìn trật tự, vệ sinh chung, có trang phục lịch sự và thái độ ứng xử văn minh.\n"
                "- Bồi thường thiệt hại và thanh toán chi phí phát sinh nếu làm mất, hư hỏng tài liệu của Thư viện.\n"
                "- Sinh viên vi phạm quy định sẽ bị lập danh sách phối hợp xử lý và trừ điểm rèn luyện."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa lúc mấy giờ?"},
                    {"label": "Quy định mượn sách về nhà", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                    {"label": "Gợi ý tài liệu phổ biến", "query": "tìm sách về trí tuệ nhân tạo"}
                ]
            }

        # ── 11. NỘI QUY CHUNG / TÁC PHONG BẢO QUẢN (ĐIỀU 11 & NỘI QUY) ──
        rules_patterns = [
            r"\b(?:noi\s+quy\s+thu\s+vien|noi\s+quy\s+phong\s+doc|co\s+duoc\s+mang\s+do\s+an|mang\s+ba\s+lo\s+vao|co\s+wifi\s+khong|quy\s+dinh\s+trang\s+phuc)\b"
        ]
        if any(re.search(p, q) for p in rules_patterns):
            ans = (
                "🏛️ **Nội quy Chung tại Thư viện QNU (Theo Điều 11 - Quy định Thư viện QNU)**\n\n"
                "Người sử dụng thư viện có trách nhiệm chấp hành nghiêm túc các quy định sau:\n\n"
                "**1. Tác phong & Trang phục:**\n\n"
                "- Trang phục lịch sự, văn minh, phù hợp với môi trường sư phạm theo quy định của Nhà trường.\n"
                "- Xuất trình thẻ sinh viên hoặc giấy tờ hợp lệ khi vào Thư viện và khi mượn tài liệu.\n"
                "- Đi nhẹ, nói khẽ, giữ gìn trật tự chung trong các phòng đọc và phòng tự học.\n"
                "- Chuyển điện thoại sang chế độ rung, không đàm thoại to trong khu vực đọc sách.\n\n"
                "**2. Tư trang & Hành lý cá nhân:**\n\n"
                "- Bạn đọc vui lòng gửi cặp sách, ba lô tại Tủ đồ cá nhân trước khi vào phòng đọc.\n"
                "- Tự bảo quản tài sản có giá trị (tiền bạc, điện thoại, máy tính cá nhân...).\n"
                "- Tuyệt đối không mang đồ ăn, nước uống có màu, kẹo cao su hoặc các chất dễ gây cháy nổ vào thư viện.\n\n"
                "**3. Bảo quản tài liệu & Trang thiết bị:**\n\n"
                "- Kiểm tra tình trạng tài liệu trước khi nhận mượn; báo ngay cho thủ thư nếu tài liệu bị rách, bẩn hoặc thiếu trang.\n"
                "- Không viết, vẽ, gạch chân, cắt xé hoặc làm hư hỏng tài liệu.\n"
                "- Sau khi đọc xong tại Phòng Đọc mở, để sách vào xe đẩy trả sách, không tự ý xếp sai vị trí trên giá sách.\n"
                "- Có trách nhiệm bồi thường thiệt hại nếu làm mất hoặc làm hư hỏng tài liệu, tài sản của Thư viện."
            )
            return {
                "answer": ans,
                "sources": [],
                "total_sources": 0,
                "total_found": 0,
                "follow_up_suggestions": [
                    {"label": "Giờ mở cửa thư viện", "query": "thư viện mở cửa lúc mấy giờ?"},
                    {"label": "Quy định mượn sách", "query": "sinh viên được mượn tối đa mấy cuốn sách về nhà?"},
                    {"label": "Tìm kiếm tài liệu", "query": "tìm sách về trí tuệ nhân tạo"}
                ]
            }

        # ── 12. TRA CỨU ĐIỀU KHOẢN CỤ THỂ (Ví dụ: "Điều 3 quy định gì", "Điều 13 là gì") ──
        art_match = re.search(r"\bdieu\s+(\d{1,2})\b", q)
        if art_match:
            art_num = int(art_match.group(1))
            art = self.get_article(art_num)
            if art:
                clean_txt = art["full_text"]
                ans = f"📜 **Trích dẫn {art['title']} (Điều {art['number']} - Quy định Thư viện QNU):**\n\n{clean_txt}"
                return {
                    "answer": ans,
                    "sources": [],
                    "total_sources": 0,
                    "total_found": 0,
                    "follow_up_suggestions": [
                        {"label": "Quy định mượn sách (Điều 3)", "query": "quy định mượn trả tài liệu in"},
                        {"label": "Xử lý vi phạm (Điều 13)", "query": "quy định xử lý vi phạm thư viện"},
                        {"label": "Phòng tự học (Điều 8)", "query": "quy định phòng tự học"}
                    ]
                }

        # ── 13. DỰ PHÒNG THÔNG MINH (Smart Article Matcher): Khi câu hỏi liên quan đến quy định/thủ tục thư viện ──
        policy_signals = [
            r"\b(?:quy\s+dinh|quy\s+che|noi\s+quy|thu\s+tuc|thu\s+vien|muon\s+sach|tra\s+sach|boi\s+thuong|xu\s+ly|vi\s+pham|the\s+thu\s+vien|hoc\s+lieu|lms|tai\s+nguyen|thoi\s+han|den\s+sach)\b"
        ]
        if any(re.search(p, q) for p in policy_signals) and not re.search(r"^(?:tim\s+sach|tim\s+tai\s+lieu|co\s+sach\s+nao\s+ve)\b", q):
            best_art = None
            best_score = 0
            q_words = set(re.findall(r"\w+", q))
            stopwords = {"toi", "ban", "cho", "hoi", "ve", "viec", "tai", "cua", "o", "la", "gi", "sao", "the", "nao", "co", "khong", "thi", "neu", "vay", "hay", "minh", "em"}
            meaningful_words = q_words - stopwords

            for art in self.articles:
                art_text_norm = normalize_text(art["full_text"])
                score = sum(1 for w in meaningful_words if w in art_text_norm)
                art_title_norm = normalize_text(art["title"])
                score += sum(2 for w in meaningful_words if w in art_title_norm)
                if score > best_score:
                    best_score = score
                    best_art = art

            if best_art and best_score >= 2:
                ans = f"📋 **Quy định Thư viện QNU (Theo {best_art['title']} - Điều {best_art['number']}):**\n\n{best_art['full_text']}"
                return {
                    "answer": ans,
                    "sources": [],
                    "total_sources": 0,
                    "total_found": 0,
                    "follow_up_suggestions": [
                        {"label": "🔍 Tìm tài liệu có từ khóa này trong kho", "query": f"tìm sách về {clean_kw}"},
                        {"label": "Quy định mượn trả sách (Điều 3)", "query": "quy định mượn trả tài liệu in"},
                        {"label": "Xử lý vi phạm (Điều 13)", "query": "quy định xử lý vi phạm thư viện"},
                        {"label": "Giờ mở cửa thư viện", "query": "thời gian mở cửa thư viện"}
                    ]
                }

        return None


# Khởi tạo singleton
policy_engine = PolicyEngine()
