"""
major_resolver.py — Chuẩn hóa và nhận diện ngành học từ ngôn ngữ tự nhiên.
Ánh xạ các cách gọi tự nhiên, viết tắt, khẩu ngữ của người dùng về đúng 59 ngành chuẩn thuộc 12 Khoa của Trường ĐH Quy Nhơn.
"""

import re
from lib.text_utils import normalize_text

# Bảng ánh xạ 59 ngành chuẩn và tất cả các biến thể tên gọi / khẩu ngữ / viết tắt
CANONICAL_MAJORS_CATALOG = {
    # ── Khoa Ngoại ngữ ──
    "Ngôn ngữ Trung Quốc": [
        "ngôn ngữ trung quốc", "ngành ngôn ngữ trung quốc", "ngôn ngữ trung",
        "tiếng trung", "tiếng trung quốc", "trung quốc học", "khoa tiếng trung",
        "trung quốc", "hán ngữ"
    ],
    "Ngôn ngữ Anh": [
        "ngôn ngữ anh", "thạc sĩ ngôn ngữ anh", "ngành ngôn ngữ anh",
        "sư phạm tiếng anh", "sư phạm tiếng anh", "tiếng anh",
        "lý luận và phương pháp dạy học bộ môn tiếng anh",
        "thạc sĩ lý luận và phương pháp dạy học bộ môn tiếng anh",
        "khoa tiếng anh", "anh ngữ"
    ],
    "Ngôn ngữ học": [
        "ngôn ngữ học", "thạc sĩ ngành ngôn ngữ học", "thạc sĩ ngôn ngữ học"
    ],

    # ── Khoa Sư phạm ──
    "Sư phạm Tin học": [
        "sư phạm tin học", "ngành sư phạm tin học", "sư pham tin học",
        "sư phạm tin", "sp tin", "tin học sư phạm"
    ],
    "Sư phạm Lịch sử": [
        "sư phạm lịch sử", "ngành sư phạm lịch sử", "sư phạm sử", "sp sử", "sử sư phạm"
    ],
    "Sư phạm Vật lý": [
        "sư phạm vật lý", "ngành sư phạm vật lý", "sư phạm lý", "sp lý", "sp vật lý", "vật lý sư phạm"
    ],
    "Sư phạm Sinh học": [
        "sư phạm sinh học", "ngành sư phạm sinh học", "sư phạm sinh", "sp sinh", "sinh học sư phạm"
    ],
    "Sư phạm Địa lý": [
        "sư phạm địa lý", "ngành sư phạm địa lý", "sư phạm địa", "sp địa", "địa lý sư phạm"
    ],
    "Giáo dục Chính trị": [
        "giáo dục chính trị", "ngành giáo dục chính trị", "chính trị sư phạm"
    ],
    "Sư phạm Toán học": [
        "sư phạm toán học", "sư phạm toán", "ngành sư phạm toán học",
        "sp toán", "toán sư phạm"
    ],
    "Sư phạm Lịch sử - Địa lý": [
        "sư phạm lịch sử - địa lý", "sư phạm lịch sử - địa lí",
        "sư phạm lịch sử địa lý", "sp sử địa"
    ],
    "Sư phạm Ngữ văn": [
        "sư phạm ngữ văn", "ngành sư phạm ngữ văn", "sư phạm văn", "sp văn", "ngữ văn sư phạm"
    ],
    "Sư phạm Hóa học": [
        "sư phạm hóa học", "ngành sư phạm hóa học", "sư phạm hóa", "sp hóa", "hóa học sư phạm"
    ],
    "Sư phạm Khoa học tự nhiên": [
        "sư phạm khoa học tự nhiên", "sư phạm khoa học tự nhiênc", "sp khtn"
    ],

    # ── Khoa Kỹ thuật và Công nghệ ──
    "Công nghệ thực phẩm": [
        "công nghệ thực phẩm", "ngành công nghệ thực phẩm", "cntp"
    ],
    "Kỹ thuật điện": [
        "kỹ thuật điện", "thạc sĩ kỹ thuật điện", "thạc sĩ ngành kỹ thuật điện",
        "ngành kỹ thuật điện", "lý thuyết mạch", "điện dân dụng", "điện công nghiệp"
    ],
    "Kỹ thuật xây dựng": [
        "kỹ thuật xây dựng", "ngành xây dựng", "xây dựng", "kỹ thuật công trình xây dựng"
    ],
    "Kỹ thuật Điện tử - Viễn thông": [
        "kỹ thuật điện tử - viễn thông", "kỹ thuật điện tử viễn thông",
        "điện tử - viễn thông", "điện tử viễn thông", "thạc sĩ kỹ thuật viễn thông",
        "ngành kỹ thuật điện tử - viễn thông", "đtvt", "viễn thông"
    ],
    "Công nghệ kỹ thuật ô tô": [
        "công nghệ kỹ thuật ô tô", "bộ môn kỹ thuật ô tô", "công nghệ ô tô", "kỹ thuật ô tô", "ô tô"
    ],
    "Kỹ thuật điều khiển và Tự động hóa": [
        "kỹ thuật điều khiển và tự động hóa", "kỹ thuât điều khiển và tự động hóa",
        "tự động hóa", "điều khiển tự động hóa", "tự động hoá"
    ],
    "Công nghệ kỹ thuật hóa học": [
        "công nghệ kỹ thuật hóa học", "hóa kỹ thuật"
    ],

    # ── Khoa Tài chính - Ngân hàng và Quản trị kinh doanh ──
    "Quản trị kinh doanh": [
        "quản trị kinh doanh", "quản trị kinhh doanh", "qtkd",
        "thạc sĩ quản trị kinh doanh", "thạc sĩ ngành quản trị kinh doanh",
        "đề án thạc sỹ ngành quản trị kinh doanh", "kinh doanh"
    ],
    "Tài chính - Ngân hàng": [
        "tài chính - ngân hàng", "tài chính-ngân hàng", "tài chính ngân hàng",
        "tài chính, ngân hàng", "thạc sĩ ngành tài chính ngân hàng",
        "ngành tài chính ngân hàng", "tcnh", "ngân hàng"
    ],
    "Quản trị khách sạn": [
        "quản trị khách sạn", "ngành quản trị khách sạn", "qtks", "khách sạn"
    ],
    "Quản trị dịch vụ du lịch và lữ hành": [
        "quản trị dịch vụ du lịch và lữ hành", "quản trị du lịch và lữ hành",
        "du lịch và lữ hành", "du lịch", "quản trị du lịch"
    ],
    "Logistics và Quản lý chuỗi cung ứng": [
        "logistics và quản lý chuỗi cung ứng", "logistics", "chuỗi cung ứng",
        "quản lý chuỗi cung ứng"
    ],

    # ── Khoa Kinh tế và Kế toán ──
    "Kế toán": [
        "kế toán", "kế toán, kiểm toán", "kế toán-kiểm toán",
        "kế toán clc", "kế toán định hướng acca", "kế toán acca",
        "thạc sĩ kế toán", "thạc sĩ ngành kế toán", "ngành kế toán", "kế toánt",
        "kế toán, kế toán clc, kiểm toán"
    ],
    "Quản lý kinh tế": [
        "quản lý kinh tế", "quản lí kinh tế", "thạc sĩ quản lý kinh tế",
        "thạc sĩ ngành quản lý kinh tế", "thạc sĩ quản lí kinh tế"
    ],
    "Kinh tế": [
        "kinh tế", "ngành kinh tế", "kinh tếbtài liệu tham khảo", "kinh tế học"
    ],
    "Kiểm toán": [
        "kiểm toán", "kiểm toán, kế toán", "ngành kiểm toán"
    ],

    # ── Khoa Khoa học Xã hội và Nhân văn ──
    "Quản lý giáo dục": [
        "quản lý giáo dục", "quản lí giáo dục", "thạc sĩ quản lý giáo dục",
        "thạc sĩ ngành quản lý giáo dục", "tâm lý học giáo dục - quản lý giáo dục"
    ],
    "Đông phương học": [
        "đông phương học", "ngành đông phương học"
    ],
    "Văn học": [
        "văn học", "văn học việt nam", "thạc sĩ văn học việt nam"
    ],
    "Việt Nam học": [
        "việt nam học", "ngành việt nam học"
    ],
    "Công tác xã hội": [
        "công tác xã hội", "ngành công tác xã hội", "ctxh"
    ],
    "Tâm lý học": [
        "tâm lý học", "tâm lý học giáo dục", "ngành tâm lý", "tâm lý"
    ],
    "Lịch sử": [
        "lịch sử", "lịch sử việt nam"
    ],

    # ── Khoa Toán và Thống kê ──
    "Toán ứng dụng": [
        "toán ứng dụng", "ngành toán ứng dụng"
    ],
    "Toán học": [
        "toán học", "thạc sĩ toán học"
    ],
    "Toán giải tích": [
        "toán giải tích", "thạc sĩ toán giải tích", "giải tích"
    ],
    "Phương pháp toán sơ cấp": [
        "phương pháp toán sơ cấp", "thạc sĩ phương pháp toán sơ cấp"
    ],
    "Đại số và Lý thuyết số": [
        "đại số và lý thuyết số", "đại số và lí thuyết số", "đại số"
    ],

    # ── Khoa Lý luận chính trị - Luật và Quản lý Nhà nước ──
    "Luật": [
        "luật", "ngành luật", "luật học"
    ],
    "Chính trị học": [
        "chính trị học", "thạc sĩ chính trị học"
    ],
    "Quản lý nhà nước": [
        "quản lý nhà nước", "ngành quản lý nhà nước", "qlnn"
    ],
    "Triết học": [
        "triết học", "ngành triết học"
    ],

    # ── Khoa Khoa học Tự nhiên ──
    "Quản lý Tài nguyên và Môi trường": [
        "quản lý tài nguyên và môi trường", "quản lý tài nguyên & môi trường",
        "quản lý tài nguyên môi trường", "ngành quản lý tài nguyên và môi trường",
        "tài nguyên môi trường", "môi trường"
    ],
    "Hóa học": [
        "hóa học", "hoá lý thuyết và hoá lý", "thạc sĩ hóa lí thuyết và hóa lí"
    ],
    "Sinh học": [
        "sinh học", "sinh học thực nghiệm", "thạc sĩ sinh học thực nghiệm"
    ],
    "Vật lý": [
        "vật lý", "vật lý chất rắn", "thạc sĩ vật lí chất rắn",
        "thạc sĩ ngành vật lý chất rắn", "ngành vật lý chất rắn", "ngành vật lí chất rắn"
    ],
    "Quản lý đất đai": [
        "quản lý đất đai", "ngành quản lý đất đai", "đất đai"
    ],
    "Nông học": [
        "nông học", "ngành nông học"
    ],
    "Khoa học vật liệu": [
        "khoa học vật liệu"
    ],

    # ── Khoa Công nghệ thông tin ──
    "Khoa học dữ liệu": [
        "khoa học dữ liệu", "khoa học dữ liệu ứng dụng",
        "thạc sĩ khoa học dữ liệu", "thạc sĩ quản trị dữ liệu",
        "data science", "khdl"
    ],
    "Khoa học máy tính": [
        "thạc sĩ khoa học máy tính", "khoa học máy tính", "khmt", "computer science"
    ],
    "Công nghệ thông tin": [
        "công nghệ thông tin", "ngành công nghệ thông tin", "công nghề thông tin",
        "cntt", "it", "ngành it", "ngành cntt"
    ],

    # ── Khoa Giáo dục Tiểu học và Mầm non ──
    "Giáo dục Tiểu học": [
        "giáo dục tiểu học", "ngành giáo dục tiểu học",
        "lý luận và phương pháp dạy học giáo dục tiểu học",
        "thạc sĩ lý luận và phương pháp dạy học giáo dục tiểu học",
        "giáo dục tiểu học và mầm non", "giáo dục tiểu học và màm non",
        "tiểu học", "sư phạm tiểu học"
    ],
    "Giáo dục Mầm non": [
        "giáo dục mầm non", "ngành giáo dục mầm non",
        "mầm non", "sư phạm mầm non"
    ],

    # ── Khoa Giáo dục Thể chất ──
    "Giáo dục Thể chất": [
        "giáo dục thể chất", "ngành giáo dục thể chất",
        "lý luận và phương pháp dạy học bộ môn giáo dục thể chất",
        "thể chất", "thể dục", "sư phạm thể dục"
    ],
}

# Tiền xử lý tập normalized aliases cho từng ngành (được sắp xếp theo độ dài giảm dần)
_NORMALIZED_ALIASES_MAP: dict[str, list[str]] = {}
for major_name, aliases in CANONICAL_MAJORS_CATALOG.items():
    s = set(aliases)
    s.add(major_name)
    norm_list = sorted(list(set(normalize_text(a).strip() for a in s if a.strip())), key=len, reverse=True)
    _NORMALIZED_ALIASES_MAP[major_name] = norm_list


def get_canonical_major_aliases(canonical_major: str) -> set[str]:
    """Trả về tập hợp các chuỗi normalized aliases của một ngành chuẩn."""
    if canonical_major in _NORMALIZED_ALIASES_MAP:
        return set(_NORMALIZED_ALIASES_MAP[canonical_major])
    norm = normalize_text(canonical_major).strip()
    return {norm} if norm else set()


def resolve_canonical_major_from_query(query: str) -> tuple[str, str] | None:
    """
    Nhận diện xem câu hỏi tự nhiên của người dùng có đang tra cứu một ngành học hay không.
    Trả về: (canonical_major_name, matched_alias) hoặc None.
    
    Ví dụ:
    - "tìm sách ngành ngôn ngữ trung" -> ("Ngôn ngữ Trung Quốc", "ngon ngu trung")
    - "sách ngành sư phạm toán" -> ("Sư phạm Toán học", "su pham toan")
    - "mình đang học quản trị kinh doanh, có tài liệu gì không?" -> ("Quản trị kinh doanh", "quan tri kinh doanh")
    - "em học CNTT cần tìm tài liệu" -> ("Công nghệ thông tin", "cntt")
    - "tôi muốn tìm sách của ngành việt nam học" -> ("Việt Nam học", "nganh viet nam hoc")
    """
    if not query:
        return None
        
    q_norm = normalize_text(query).strip()
    
    # 1. Bóc tách tiền tố tự nhiên
    cleaned = q_norm
    common_prefixes = [
        "tim sach cua nganh", "tim sach thuoc nganh", "tim sach nganh", "tim sach",
        "sach cua nganh", "sach thuoc nganh", "sach nganh", "sach cho sinh vien nganh", "sach cho sinh vien",
        "tai lieu cua nganh", "tai lieu thuoc nganh", "tai lieu nganh",
        "giao trinh cua nganh", "giao trinh thuoc nganh", "giao trinh nganh",
        "minh dang hoc nganh", "minh dang hoc", "toi dang hoc nganh", "toi dang hoc",
        "em dang hoc nganh", "em dang hoc", "minh hoc nganh", "minh hoc",
        "toi hoc nganh", "toi hoc", "em hoc nganh", "em hoc", "ban hoc nganh",
        "cho minh xin tai lieu nganh", "cho minh xin sach nganh", "cho minh xin tai lieu", "cho minh xin sach",
        "co sach nganh", "co tai lieu nganh", "co giao trinh nganh",
        "gioi thieu sach nganh", "gioi thieu tai lieu nganh",
    ]
    for prefix in common_prefixes:
        if cleaned.startswith(prefix + " "):
            cleaned = cleaned[len(prefix):].strip()
            break

    # 2. Bóc tách hậu tố khẩu ngữ
    cleaned = re.sub(
        r"\s*(co\s+tai\s+lieu\s+gi\s+khong|co\s+sach\s+gi\s+khong|co\s+tai\s+lieu\s+nao\s+khong|co\s+sach\s+nao\s+khong|co\s+khong|khong\s+a|khong\s+nhi|khong|can\s+tim\s+tai\s+lieu|can\s+tim\s+sach|a|nhi|vay|nhe)\s*$",
        "",
        cleaned
    ).strip()

    # 3. Ưu tiên khớp chính xác hoặc có cụm từ alias
    best_match = None
    best_len = 0
    for major_name, aliases in _NORMALIZED_ALIASES_MAP.items():
        for alias in aliases:
            # Từ ngắn <= 3 ký tự (vd "it", "cs") cần ranh giới từ chính xác
            pat = r"\b" + re.escape(alias) + r"\b"
            if re.search(pat, cleaned) or alias == cleaned:
                if len(alias) > best_len:
                    best_match = (major_name, alias)
                    best_len = len(alias)
            elif len(alias) >= 4 and re.search(pat, q_norm):
                if len(alias) > best_len:
                    best_match = (major_name, alias)
                    best_len = len(alias)

    return best_match


def doc_matches_canonical_major(doc_major_raw: str, canonical_aliases: set[str]) -> bool:
    """
    Kiểm tra xem một dòng tài liệu trong CSDL có thuộc ngành chuẩn này không.
    So sánh giá trị trường major của tài liệu với tất cả các alias hợp lệ của ngành.
    """
    if not doc_major_raw:
        return False
    doc_tok = normalize_text(doc_major_raw).strip()
    if not doc_tok:
        return False
    return doc_tok in canonical_aliases

