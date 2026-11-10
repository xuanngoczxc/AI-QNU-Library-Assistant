# Báo Cáo Đánh Giá Chất Lượng Chatbot Thư Viện QNU

- **Hệ thống đánh giá:** QNU Library Assistant Standard Benchmark (QNU-LAB)
- **Phiên bản:** 2.0.0
- **Thời gian chạy:** 2026-11-10 14:30:09
- **Tổng điểm đạt được:** **112.0 / 115.0 (97.4%)**
- **Xếp loại chất lượng:** **XUẤT SẮC (EXCELLENT)**

---

## 1. Chỉ số Hiệu năng & Độ chính xác cốt lõi (KPI Metrics)

| Chỉ số đo lường | Giá trị đạt được | Tiêu chuẩn chất lượng | Trạng thái |
| :--- | :---: | :---: | :---: |
| **Retrieval Hit Rate@K** | **66.7%** | >= 85.0% | ⚠️ Chưa đạt |
| **SLA Latency Compliance** | **62.5%** | >= 90.0% | ⚠️ Chưa đạt |
| **Negative Constraint Adherence** | **100.0%** | 100.0% | ✅ Hoàn hảo |
| **Adversarial Safety Rejection** | **100.0%** | 100.0% | ✅ Hoàn hảo |

---

## 2. Chi tiết theo từng Nhóm Nghiệp vụ

| Nhóm nghiệp vụ (Category) | Số ca test | Số ca Đạt | Điểm số đạt được | Tỷ lệ chính xác |
| :--- | :---: | :---: | :---: | :---: |
| `TITLE_SEARCH` | 4 | 3 | 19.5 / 20.0 | **97.5%** |
| `AUTHOR_SEARCH` | 3 | 3 | 15.0 / 15.0 | **100.0%** |
| `MULTI_CONSTRAINT` | 1 | 1 | 10.0 / 10.0 | **100.0%** |
| `MAJOR_FILTER` | 2 | 0 | 7.5 / 10.0 | **75.0%** |
| `TEEN_CODE_SLANG` | 1 | 1 | 10.0 / 10.0 | **100.0%** |
| `LIBRARY_POLICY` | 4 | 4 | 20.0 / 20.0 | **100.0%** |
| `ADVERSARIAL_GATE` | 1 | 1 | 10.0 / 10.0 | **100.0%** |
| `FAST_PATH_SUGGEST` | 1 | 1 | 5.0 / 5.0 | **100.0%** |
| `MULTI_TURN_DST` | 2 | 2 | 10.0 / 10.0 | **100.0%** |
| `CONTENT_TOC_QA` | 1 | 1 | 5.0 / 5.0 | **100.0%** |

---

## 3. Danh sách Chi tiết Từng Ca Kiểm thử (Case-by-Case Breakdown)

| Mã ca test | Tên kiểm thử | Nhóm | Trọng số | Điểm đạt | Trạng thái |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `TC_PREF_001` | Bóc tách tiền tố trường Excel 'nhan đề : ' | `TITLE_SEARCH` | 5.0đ | **4.5đ** | ❌ FAIL |
| `TC_PREF_002` | Bóc tách tiền tố 'Nhan đề chính ' có dấu | `TITLE_SEARCH` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_PREF_003` | Bảo toàn từ 'cho' bên trong nhan đề sách thực tế | `TITLE_SEARCH` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_PREF_004` | Bóc tách tiền tố dính liền dấu hai chấm 'nhan đề:' | `TITLE_SEARCH` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_AUTH_001` | Tác giả nước ngoài có tên viết tắt và dấu chấm (Alexander, L.G.) | `AUTHOR_SEARCH` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_AUTH_002` | Tác giả nước ngoài họ trước tên sau (West, Morris L.) | `AUTHOR_SEARCH` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_CONS_001` | Tác giả + Nhan đề + Lọc năm xuất bản (Trần Thiên Thành sau năm 2020) | `MULTI_CONSTRAINT` | 10.0đ | **10.0đ** | ✅ PASS |
| `TC_MAJ_001` | Khử nhiễu Tháp Chăm Bánh Ít (Không bị kích hoạt nhầm ngành CNTT) | `MAJOR_FILTER` | 5.0đ | **4.5đ** | ❌ FAIL |
| `TC_MAJ_002` | Khử nhiễu Đại Việt Sử Ký Toàn Thư (Không bị gán nhầm ngành Kế toán) | `MAJOR_FILTER` | 5.0đ | **3.0đ** | ❌ FAIL |
| `TC_TEEN_001` | Chuẩn hóa tiếng lóng teen-code sinh viên (ctdl>, thầy nghĩa, zới, huhu) | `TEEN_CODE_SLANG` | 10.0đ | **10.0đ** | ✅ PASS |
| `TC_POL_001` | Xác minh địa chỉ Thư viện QNU (Từ chối chi nhánh ảo Cù Lao Xanh) | `LIBRARY_POLICY` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_POL_002` | Hỏi thời gian làm việc & giờ mở cửa chính thức | `LIBRARY_POLICY` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_ADV_001` | Relevance Gating: Chặn mã định danh rác đối kháng không tồn tại | `ADVERSARIAL_GATE` | 10.0đ | **10.0đ** | ✅ PASS |
| `TC_FAST_001` | Phản hồi tức thì nút 'Gợi ý tài liệu phổ biến' | `FAST_PATH_SUGGEST` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_DST_001` | Hội thoại nhiều lượt: Từ chủ đề rộng thu hẹp và lọc top K | `MULTI_TURN_DST` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_DST_002` | Hội thoại nhiều lượt: Thay đổi điều kiện năm xuất bản | `MULTI_TURN_DST` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_TOC_001` | Hỏi số chương / Mục lục sách (Minh bạch giới hạn CSDL thư mục) | `CONTENT_TOC_QA` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_AUTH_003` | Tác giả chỉ nhập tên gọi 1 từ (sách của hiển) -> Làm rõ danh sách tác giả cùng tên | `AUTHOR_SEARCH` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_PERS_001` | Tra cứu sách cá nhân đang mượn (sách của tôi) -> Hướng dẫn truy cập tài khoản OPAC | `LIBRARY_POLICY` | 5.0đ | **5.0đ** | ✅ PASS |
| `TC_POL_003` | Mượn sách quá hạn trên 90 ngày -> Áp dụng chế tài đền sách bắt buộc (Điều 13) | `LIBRARY_POLICY` | 5.0đ | **5.0đ** | ✅ PASS |
