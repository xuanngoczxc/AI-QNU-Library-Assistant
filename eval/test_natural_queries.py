"""
eval/test_natural_queries.py — Automated test suite for natural language conversational queries.
Tests:
  1. Oldest sorting query with conversational wrapper ("có sách nào liên quan đến tin học có năm xuất bản lâu đời nhất không")
  2. Recency sorting query ("sách về quản trị kinh doanh có năm xuất bản gần nhất")
  3. Polite greeting & availability query ("dạ em chào anh chị ad ạ, cho em hỏi cuốn Giải tích 1 hiện tại còn ở phòng giáo trình không vậy ạ?")
  4. Conversational query with "ad ơi" ("ad ơi cho mình xin tài liệu về trí tuệ nhân tạo mới nhất với ạ")
  5. Year constrained query ("sách kinh tế xuất bản từ năm 2022")
  6. Author recency query ("có cuốn nào của Nguyễn Nhật Ánh mới nhất không")
"""

import sys
import codecs
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app import app

# Ensure utf-8 output on Windows
if sys.platform == "win32":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)

client = TestClient(app)

TEST_CASES = [
    {
        "id": "NL_01_OLDEST_TIN_HOC",
        "query": "có sách nào liên quan đến tin học có năm xuất bản lâu đời nhất không",
        "description": "Câu hỏi tìm sách tin học lâu đời nhất kèm lớp vỏ tự nhiên",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            # Top 1 source title must be relevant to tin học / máy tính / thông tin học
            lambda res: any(
                kw in (res.json()["sources"][0].get("title", "") + " " + res.json()["sources"][0].get("subject", "")).lower()
                for kw in ["tin học", "máy tính", "thông tin học", "kỹ thuật vi xử lý", "basic", "cơ sở dữ liệu"]
            ),
            # Must NOT be unrelated books like "Sợi hóa học" or "Lầu năm góc"
            lambda res: "sợi hóa học" not in res.json()["sources"][0].get("title", "").lower(),
            lambda res: "lầu năm góc" not in res.json()["sources"][0].get("title", "").lower(),
            # Year must be <= 1995
            lambda res: int(str(res.json()["sources"][0].get("year", 9999))) <= 1995,
            # Answer text mentions lâu đời nhất
            lambda res: "lâu đời nhất" in res.json().get("answer", "").lower(),
        ],
    },
    {
        "id": "NL_02_RECENCY_QTKD",
        "query": "sách về quản trị kinh doanh có năm xuất bản gần nhất",
        "description": "Tìm sách QTKD mới nhất / gần đây nhất",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            # Top 1 year should be recent (>= 2020)
            lambda res: int(str(res.json()["sources"][0].get("year", 0))) >= 2020,
            lambda res: "gần đây nhất" in res.json().get("answer", "").lower(),
        ],
    },
    {
        "id": "NL_03_GREETING_WRAPPER_GIAI_TICH",
        "query": "dạ em chào anh chị ad ạ, cho em hỏi cuốn Giải tích 1 hiện tại còn ở phòng giáo trình không vậy ạ?",
        "description": "Chào hỏi lịch sự và hỏi sách Giải tích 1",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            lambda res: any("giải tích" in s.get("title", "").lower() for s in res.json()["sources"][:3]),
        ],
    },
    {
        "id": "NL_04_AD_OI_AI_RECENCY",
        "query": "ad ơi cho mình xin tài liệu về trí tuệ nhân tạo mới nhất với ạ",
        "description": "Xưng hô ad ơi và xin tài liệu AI mới nhất",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            lambda res: any(
                kw in (res.json()["sources"][0].get("title", "") + " " + res.json()["sources"][0].get("subject", "")).lower()
                for kw in ["trí tuệ nhân tạo", "ai", "machine learning", "học máy", "khoa học máy tính", "deep learning"]
            ),
        ],
    },
    {
        "id": "NL_05_YEAR_CONSTRAINT_KINH_TE",
        "query": "sách kinh tế xuất bản từ năm 2022",
        "description": "Lọc năm >= 2022 cho ngành kinh tế",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            lambda res: all(
                int(str(s.get("year", 0))) >= 2022
                for s in res.json().get("sources", [])[:5]
                if str(s.get("year", 0)).isdigit()
            ),
        ],
    },
    {
        "id": "NL_06_AUTHOR_RECENCY_HO_CHI_MINH",
        "query": "có cuốn nào của Hồ Chí Minh mới nhất không",
        "description": "Tìm sách của tác giả Hồ Chí Minh mới nhất",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            lambda res: any("hồ chí minh" in s.get("author", "").lower() for s in res.json().get("sources", [])[:3]),
        ],
    },
    {
        "id": "NL_07_MULTI_TURN_CONSECUTIVE_REFINEMENT",
        "query": "vậy lựa ra quyển có năm xuất bản gần nhất",
        "turn1_query": "những sách nào nói về đời sống con người việt nam",
        "description": "Hội thoại nhiều lượt: Turn 1 tìm sách đời sống người VN -> Turn 2 lựa quyển năm XB gần nhất",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            lambda res: "chưa tìm thấy tài liệu" not in res.json().get("answer", "").lower(),
            lambda res: int(str(res.json()["sources"][0].get("year", 0))) >= 2021,
            lambda res: "đời sống con người việt nam" in res.json().get("answer", "").lower(),
        ],
    },
    {
        "id": "NL_08_AUTHOR_CONVERSATIONAL_TAIL",
        "query": "vậy sách của tác giả nguyễn thành đạt có không?",
        "description": "Hỏi sách của tác giả kèm đuôi câu hỏi tự nhiên 'vậy... có không?'",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            # Tên tác giả phải sạch, tuyệt đối không bị dính đuôi câu hỏi "có không"
            lambda res: "nguyễn thành đạt có không" not in res.json().get("answer", "").lower(),
            lambda res: "nguyễn thành đạt" in res.json().get("answer", "").lower(),
            # Phản hồi phải khẳng định có tài liệu
            lambda res: "dạ có" in res.json().get("answer", "").lower() or "thư viện qnu hiện có" in res.json().get("answer", "").lower(),
            # Các sách trả về phải đúng tác giả
            lambda res: any("nguyễn thành đạt" in (s.get("author", "") + " " + s.get("responsibility", "")).lower() for s in res.json().get("sources", [])[:3]),
        ],
    },
    {
        "id": "NL_09_AUTHOR_MULTI_ATTRIBUTE_YEAR_FILTER",
        "query": "sách của tác giả hồ chí minh xuất bản năm 2025",
        "description": "Truy vấn đa thuộc tính: Tác giả + Năm xuất bản (sách tác giả Hồ Chí Minh năm 2025)",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            # Tên tác giả phải sạch, không bị dính "xuất bản năm 2025"
            lambda res: "hồ chí minh xuất bản năm 2025" not in res.json().get("answer", "").lower(),
            lambda res: "hồ chí minh" in res.json().get("answer", "").lower(),
            # Bot giải thích rõ ràng không có ấn phẩm năm 2025 và nêu khoảng năm hiện có
            lambda res: "2025" in res.json().get("answer", "").lower(),
            lambda res: "chưa có tài liệu nào xuất bản vào năm" in res.json().get("answer", "").lower(),
            # Gợi ý sách tái bản gần nhất của tác giả
            lambda res: any("hồ chí minh" in (s.get("author", "") + " " + s.get("responsibility", "")).lower() for s in res.json().get("sources", [])[:3]),
        ],
    },
    {
        "id": "NL_10_PURE_YEAR_LISTING_1991",
        "query": "nhung sách nào xuất bản năm 1991",
        "description": "Truy vấn danh mục theo năm xuất bản: không nhầm 'nhung sách nào' thành tên sách",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) >= 500,
            # Tuyệt đối không nhầm thành tên sách 'nhung sách nào'
            lambda res: "không tìm thấy thông tin hoặc tài liệu về cuốn sách \"nhung sách nào\"" not in res.json().get("answer", "").lower(),
            lambda res: "nhung sách nào" not in res.json().get("answer", "").lower(),
            # Phản hồi phải nhận diện đúng 1991 và trả về danh mục
            lambda res: "1991" in res.json().get("answer", ""),
            lambda res: all(int(s.get("year", 0)) == 1991 for s in res.json().get("sources", [])[:10]),
        ],
    },
    {
        "id": "NL_11_NONEXISTENT_TOPIC_SIEU_NHAN",
        "query": "sách chủ đề siêu nhân",
        "description": "Tìm chủ đề không có trong thư viện: báo không có, không ghép từ sai/lặp 'chủ đề chủ đề'",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) == 0,
            lambda res: "về chủ đề chủ đề" not in res.json().get("answer", "").lower(),
            lambda res: "chưa tìm thấy tài liệu nào về chủ đề" in res.json().get("answer", "").lower(),
            lambda res: "siêu nhân" in res.json().get("answer", "").lower(),
        ],
    },
    {
        "id": "NL_12_NONEXISTENT_AUTHOR_NGUYEN_LU",
        "query": "sách của nguyễn lữ",
        "description": "Tìm tác giả không có trong thư viện: báo không có, không trả về tác giả khác",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) == 0,
            lambda res: "chưa tìm thấy tài liệu nào của tác giả" in res.json().get("answer", "").lower(),
            lambda res: "nguyễn lữ" in res.json().get("answer", "").lower(),
        ],
    },
    {
        "id": "NL_13_FLOWER_BOOKS_DIACRITICS",
        "query": "sách về các loài hoa",
        "description": "Tìm sách về các loài hoa: Top 1 phải là 'Sự tích các loài hoa', không nhầm sang luận văn sinh hóa/tiêu hóa/Phú Hòa",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            # Top 1 source must be 'Sự tích các loài hoa'
            lambda res: "sự tích các loài hoa" in res.json()["sources"][0].get("title", "").lower(),
            # No unrelated dissertations with diacritic collisions (tiến hóa, tiêu hóa, Phú Hòa, Copepoda)
            lambda res: not any(
                bad in s.get("title", "").lower()
                for s in res.json().get("sources", [])
                for bad in ["copepoda", "vi nhựa", "tiêu hóa", "phú hòa", "tiến hóa"]
            ),
            # Bot answer must highlight 'Sự tích các loài hoa'
            lambda res: "sự tích các loài hoa" in res.json().get("answer", "").lower(),
        ],
    },
    {
        "id": "NL_14_RESEARCH_TOPIC_AGRICULTURE",
        "query": "Tôi đang cần tìm hiểu về đề tài nghiên cứu khoa học về chủ đề nông nghiệp, bạn hãy gợi ý cho tôi những sách liên quan để giúp tôi có thể hoàn thiện đề tài này.",
        "description": "Câu hỏi tự nhiên phức hợp: nghiên cứu khoa học về chủ đề nông nghiệp kèm đề nghị gợi ý sách",
        "checks": [
            lambda res: res.status_code == 200,
            lambda res: len(res.json().get("sources", [])) > 0,
            # Must extract agriculture topic and return agriculture books
            lambda res: any("nông nghiệp" in (s.get("title", "") + " " + s.get("subject", "")).lower() for s in res.json().get("sources", [])[:5]),
            # Must not return generic unrelated books
            lambda res: not any(bad in s.get("title", "").lower() for s in res.json().get("sources", [])[:10] for bad in ["tiếng hoa mỗi ngày", "sách ngoài trời", "cung cấp điện xí nghiệp"]),
            # Answer must highlight nông nghiệp
            lambda res: "nông nghiệp" in res.json().get("answer", "").lower(),
        ],
    },
]


def run_tests():
    print("=" * 65)
    print("🚀 BẮT ĐẦU KIỂM THỬ TỰ ĐỘNG CÁC TRUY VẤN NGÔN NGỮ TỰ NHIÊN")
    print("=" * 65)
    passed = 0
    failed = 0

    for tc in TEST_CASES:
        print(f"\n▶ [{tc['id']}] {tc['description']}")
        print(f"  Câu hỏi: \"{tc['query']}\"")
        try:
            sid = f"test_{tc['id']}"
            if tc.get("turn1_query"):
                print(f"  Turn 1: \"{tc['turn1_query']}\"")
                client.post("/chat", json={"query": tc["turn1_query"], "session_id": sid})
                print(f"  Turn 2: \"{tc['query']}\"")
            resp = client.post("/chat", json={"query": tc["query"], "session_id": sid})
            data = resp.json()
            sources = data.get("sources", [])
            answer = data.get("answer", "")
            
            all_ok = True
            for i, check_fn in enumerate(tc["checks"]):
                if not check_fn(resp):
                    all_ok = False
                    print(f"  ❌ Kiểm tra #{i+1} THẤT BẠI!")
                    break

            if all_ok:
                passed += 1
                top_title = sources[0].get("title") if sources else "N/A"
                top_year = sources[0].get("year") if sources else "N/A"
                top_author = sources[0].get("author") if sources else "N/A"
                print(f"  ✅ ĐẠT! Top 1: [{top_year}] \"{top_title}\" — {top_author}")
                print(f"     Bot trả lời: {answer[:100]}...")
            else:
                failed += 1
                top_title = sources[0].get("title") if sources else "N/A"
                print(f"  ❌ KHÔNG ĐẠT! Top 1 nhận được: \"{top_title}\"")
                print(f"     Toàn bộ sources: {[s.get('title') for s in sources[:3]]}")
                print(f"     Bot trả lời: {answer[:120]}...")
        except Exception as e:
            failed += 1
            print(f"  ❌ LỖI NGOẠI LỆ: {e}")

    print("\n" + "=" * 65)
    print(f"📊 KẾT QUẢ: {passed}/{len(TEST_CASES)} ca kiểm thử ĐẠT ({passed/len(TEST_CASES)*100:.1f}%)")
    print("=" * 65)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
