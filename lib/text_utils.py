"""
text_utils.py — Text processing utilities cho search accuracy
- normalize_text: Lowercase + remove diacritics (cho embedding/search)
- clean_text: Remove extra spaces, special chars (cho display)
- extract_keywords: Extract keywords từ text
- generate_document_hash: Tạo hash để detect duplicates
- suggest_synonyms: Gợi ý từ đồng nghĩa / thuật ngữ
- spell_correct: Sửa lỗi chính tả tiếng Việt cơ bản
"""

import unicodedata
import hashlib
import re
from typing import Optional


# ── Vietnamese spelling dictionary ────────────────────────
VIETNAMESE_SPELLING_MAP = {
    # Common misspellings -> correct
    'chat luong': 'chất lượng',
    'chuong trinh': 'chương trình',
    'cong nghe': 'công nghệ',
    'giao duc': 'giáo dục',
    'quan ly': 'quản lý',
    'nghien cuu': 'nghiên cứu',
    'thong tin': 'thông tin',
    'tai lieu': 'tài liệu',
    'thu vien': 'thư viện',
    'sinh vien': 'sinh viên',
    'giang vien': 'giảng viên',
    'mon hoc': 'môn học',
    'de tai': 'đề tài',
    'bao cao': 'báo cáo',
    'luan van': 'luận văn',
    'khoa hoc': 'khoa học',
    'ky thuat': 'kỹ thuật',
    'kinh te': 'kinh tế',
    'van hoa': 'văn hóa',
    'xa hoi': 'xã hội',
    'truong hoc': 'trường học',
    'phat trien': 'phát triển',
    'dieu tra': 'điều tra',
    'thuc hanh': 'thực hành',
    'ly thuyet': 'lý thuyết',
    # Missing diacritics
    'ky nang': 'kỹ năng',
    'dao tao': 'đào tạo',
    'co so': 'cơ sở',
    'doanh nghiep': 'doanh nghiệp',
    'chinh sach': 'chính sách',
}

# ── Synonym dictionary (Vietnamese) ───────────────────────
SYNONYM_MAP = {
    'nghiên cứu': ['khảo cứu', 'tìm hiểu', 'phân tích', 'khảo sát', 'research'],
    'phương pháp': ['cách thức', 'kỹ thuật', 'phương thức', 'method', 'methodology'],
    'phân tích': ['đánh giá', 'xem xét', 'khảo sát', 'analysis'],
    'đánh giá': ['nhận xét', 'thẩm định', 'appraisal', 'evaluation'],
    'mô hình': ['khuôn mẫu', 'kiểu mẫu', 'model', 'framework'],
    'hệ thống': ['system', 'network', 'cơ cấu'],
    'dữ liệu': ['data', 'thông tin', 'số liệu', 'database'],
    'giáo trình': ['sách giáo khoa', 'textbook', 'course book'],
    'luận văn': ['luận án', 'thesis', 'dissertation'],
    'thực nghiệm': ['thí nghiệm', 'experiment', 'empirical'],
    'định lượng': ['quantitative', 'số lượng'],
    'định tính': ['qualitative', 'chất lượng'],
    'tổng quan': ['literature review', 'overview', 'survey'],
    'kết quả': ['result', 'outcome', 'phát hiện', 'finding'],
    'chương trình': ['program', 'curriculum', 'giáo trình'],
    'quản lý': ['management', 'administration', 'điều hành'],
    'kinh tế': ['economy', 'economic', 'tài chính'],
}


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    replacements = {
        'ă': 'a', 'â': 'a',
        'ê': 'e',
        'ô': 'o', 'ơ': 'o',
        'ư': 'u',
        'đ': 'd',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)

    return text.strip()


def clean_text(text: str) -> str:
    """
    Clean text: remove extra spaces, normalize line breaks
    Dùng cho: display, output
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    
    # Remove trailing spaces from lines
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """
    Extract keywords từ text (bỏ qua stopwords)
    Dùng cho: metadata enrichment
    """
    vietnamese_stopwords = {
        "và", "hoặc", "nhưng", "mà", "để", "từ", "tới", "bởi",
        "là", "có", "được", "cái", "chiếc", "những",
        "của", "đó", "này", "đó", "kia", "nào",
        "không", "chưa", "hay", "mới", "lại", "cũng",
        "phải", "đã", "sẽ", "có", "được",
        "a", "an", "the", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "as", "by", "is", "was"
    }
    
    # Normalize text
    normalized = normalize_text(text)
    
    # Split by spaces and filter
    words = normalized.split()
    keywords = [
        word for word in words 
        if len(word) > 2 and word not in vietnamese_stopwords
    ]
    
    # Return top N unique keywords
    return list(dict.fromkeys(keywords))[:max_keywords]


def generate_document_hash(content: str, title: str = "") -> str:
    """
    Generate hash để detect duplicate documents
    Dùng cho: deduplication
    """
    # Normalize content
    normalized = normalize_text(content + " " + title)
    
    # Take first 500 chars
    normalized = normalized[:500]
    
    # Generate MD5 hash
    return hashlib.md5(normalized.encode()).hexdigest()


def create_searchable_content(original: str, normalized: str = "") -> dict:
    """
    Create metadata object cho document
    Returns:
        {
            "original": original text,
            "normalized": normalized text (cho embedding),
            "keywords": extracted keywords,
            "hash": document hash (cho dedup)
        }
    """
    if not normalized:
        normalized = normalize_text(original)
    
    return {
        "original": original.strip(),
        "normalized": normalized,
        "keywords": extract_keywords(original),
        "hash": generate_document_hash(original)
    }


def is_duplicate(doc1_hash: str, doc2_hash: str) -> bool:
    """Check if two documents are duplicates based on hash"""
    return doc1_hash == doc2_hash


def batch_normalize(texts: list[str]) -> list[str]:
    """Normalize multiple texts efficiently"""
    return [normalize_text(text) for text in texts]


def suggest_synonyms(word: str) -> dict:
    """
    Gợi ý từ đồng nghĩa, thuật ngữ Anh-Việt, thuật ngữ chuyên ngành

    Args:
        word: từ cần gợi ý

    Returns:
        dict với các keys: synonyms (list), en_terms (list), related (list)
    """
    word_lower = word.strip().lower()
    result = {
        'synonyms': [],
        'en_terms': [],
        'related': [],
        'spelling': '',
    }

    # 1. Check synonym map
    if word_lower in SYNONYM_MAP:
        for item in SYNONYM_MAP[word_lower]:
            if re.match(r'^[a-zA-Z]', item):
                result['en_terms'].append(item)
            else:
                result['synonyms'].append(item)

    # 2. Check reverse (if word is English, find Vietnamese)
    for vi_word, terms in SYNONYM_MAP.items():
        if word_lower in [t.lower() for t in terms]:
            result['synonyms'].append(vi_word)
            en_terms = [t for t in terms if re.match(r'^[a-zA-Z]', t) and t.lower() != word_lower]
            result['en_terms'].extend(en_terms)

    # 3. Spelling check
    if word_lower in VIETNAMESE_SPELLING_MAP:
        result['spelling'] = VIETNAMESE_SPELLING_MAP[word_lower]

    # Deduplicate
    result['synonyms'] = list(dict.fromkeys(result['synonyms']))
    result['en_terms'] = list(dict.fromkeys(result['en_terms']))

    return result


def spell_correct(text: str) -> str:
    """
    Sửa lỗi chính tả tiếng Việt cơ bản

    Args:
        text: câu/văn bản cần sửa

    Returns:
        text đã sửa lỗi chính tả
    """
    text_lower = text.lower().strip()

    # Check full phrase matches first
    for wrong, correct in VIETNAMESE_SPELLING_MAP.items():
        # Match as whole phrase (not partial)
        pattern = r'\b' + re.escape(wrong) + r'\b'
        if re.search(pattern, text_lower):
            # Replace preserving original case somewhat
            text_lower = re.sub(pattern, correct, text_lower)

    # Token-level check
    words = text_lower.split()
    corrected_words = []
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if cleaned in VIETNAMESE_SPELLING_MAP:
            punct = re.sub(r'\w', '', word)
            corrected_words.append(VIETNAMESE_SPELLING_MAP[cleaned] + punct)
        else:
            corrected_words.append(word)

    return ' '.join(corrected_words)


def add_to_spelling_map(correct: str, wrong: str):
    """Add a custom spelling correction pair"""
    VIETNAMESE_SPELLING_MAP[wrong.lower().strip()] = correct


def suggest_terms_for_query(query: str) -> dict:
    """
    Gợi ý từ đồng nghĩa, thuật ngữ chuyên ngành cho câu truy vấn

    Args:
        query: câu truy vấn của người dùng

    Returns:
        dict gợi ý
    """
    words = query.split()
    suggestions = {
        'synonym_expansions': [],
        'spelling_corrections': [],
        'en_terms': [],
    }

    # Check each word
    for word in words:
        cleaned = re.sub(r'[^\w]', '', word)
        if not cleaned:
            continue

        result = suggest_synonyms(cleaned)

        if result['synonyms']:
            suggestions['synonym_expansions'].append({
                'word': cleaned,
                'synonyms': result['synonyms']
            })

        if result['en_terms']:
            suggestions['en_terms'].append({
                'word': cleaned,
                'en_terms': result['en_terms']
            })

        if result['spelling']:
            suggestions['spelling_corrections'].append({
                'original': cleaned,
                'corrected': result['spelling']
            })

VIETNAMESE_SURNAMES = {
    "nguyen", "tran", "le", "pham", "hoang", "huynh", "phan", "vu", "vo", "dang",
    "bui", "do", "ho", "ngo", "duong", "ly", "dinh", "doan", "lam", "trinh",
    "mai", "dao", "cao", "ha", "luu", "luong", "thai", "chau", "ta", "phung",
    "to", "vuong", "van", "tong", "truong", "dieu", "quach", "nham", "lieu", "ton",
    "khuu", "chu", "luc", "trieu", "la", "giang"
}


def clean_author_candidate(cand: str) -> str:
    """
    Cleans raw author candidate string by removing leading author prefixes,
    trailing question words, location phrases, availability checks,
    and conversational filler particles in Vietnamese.
    """
    if not cand:
        return ""
    c = cand.strip().strip(" ?.,!;:'\"")
    # Strip leading markers like "tac gia", "cua", etc.
    c = re.sub(r"^(?:tac\s+gia|dong\s+tac\s+gia|chu\s+bien|dich\s+gia|cua\s+tac\s+gia|cua)\s+", "", c, flags=re.I).strip()

    patterns = [
        # Location queries (ở đâu, ở phòng nào, phòng đọc nào...)
        r"\s+\b(?:nam\s+o|o\s+tai|o|muon\s+o|tim\s+o|doc\s+o|lay\s+o|xem\s+o)\s+(?:dau|phong\s+(?:doc\s+)?nao|kho\s+nao|tang\s+(?:may|\d+)|ke\s+nao|gia\s+nao|cho\s+nao)\b.*$",
        r"\s+\b(?:o\s+dau|nam\s+o\s+dau|o\s+phong\s+nao|o\s+kho\s+nao|o\s+tang\s+nao|o\s+tang\s+may|o\s+dau\s+vay|o\s+dau\s+a)\b.*$",
        r"\s+\b(?:phong\s+(?:doc\s+)?nao|kho\s+nao|tang\s+(?:may|\d+)|ke\s+nao|gia\s+nao)\b.*$",
        # Quantity / Listing (có những sách nào, có sách gì...)
        r"\s+\b(?:co\s+(?:nhung\s+|cac\s+)?(?:sach|tai\s+lieu|giao\s+trinh|an\s+pham|cuon|quyen)\s+nao)\b.*$",
        r"\s+\b(?:co\s+(?:sach|tai\s+lieu|giao\s+trinh|cuon|quyen)\s+(?:gi|nao))\b.*$",
        r"\s+\b(?:co\s+(?:tai\s+lieu|sach|giao\s+trinh)\s+gi\s+khong)\b.*$",
        r"\s+\b(?:co\s+(?:nhung\s+|cac\s+)?(?:gi|gi\s+khong))\b.*$",
        r"\s+\b(?:nhung\s+sach\s+nao|cac\s+sach\s+nao|sach\s+nao|tai\s+lieu\s+nao|cuon\s+nao|giao\s+trinh\s+nao)\b.*$",
        # Borrowing / Availability
        r"\s+\b(?:co\s+(?:duoc\s+)?(?:muon|muon\s+ve|doc)\s*(?:duoc\s+)?(?:khong|ko|k)?)\b.*$",
        r"\s+\b(?:co\s+(?:ban\s+)?(?:dien\s+tu|ebook|pdf)\s*(?:khong|ko|k)?)\b.*$",
        # Yes/No questioning (có không, có ko, có k, có chưa, còn không...)
        r"\s+\b(?:co\s+khong|co\s+ko|co\s+k|co\s+chua|con\s+khong|con\s+ko|con\s+k)\b.*$",
        r"\s+\b(?:dung\s+khong|phai\s+khong|dung\s+ko|phai\s+ko)\b.*$",
        r"\s+\b(?:co\s+(?:chua|khong|ko|k))\b.*$",
        # Trailing polite / question particles
        r"\s+\b(?:khong\s+a|ko\s+a|k\s+a|khong\s+vay|ko\s+vay|k\s+vay|khong\s+nhi|ko\s+nhi|khong\s+ha|ko\s+ha)\b.*$",
        r"\s+\b(?:khong|ko|k|chua|ha|nhi|the|vay|a|nhe|nha|voi|oi|ne)\b\s*$",
        r"\s+\b(?:co|con)\b\s*$",
        r"\s+\b(?:bot|ad|admin|ban|thu\s+vien)\b\s*$",
    ]
    for _ in range(3):
        c = c.strip().strip(" ?.,!;:'\"")
        for pat in patterns:
            c = re.sub(pat, "", c, flags=re.I).strip()
    return re.sub(r"\s+", " ", c).strip()


def extract_author_candidate(query: str) -> str:
    """
    Extracts and cleans normalized author candidate from a user query.
    Handles various phrasings:
      - 'sách của tác giả X có không?'
      - 'sách tác giả X ở phòng nào'
      - 'tác giả X có sách gì'
      - 'sách của X có không' (if X starts with a Vietnamese surname)
    """
    if not query:
        return ""
    q_norm = normalize_text(query).strip()
    # Strip leading polite prefixes
    q_norm = re.sub(r"^(?:da\s+)?(?:cho\s+(?:minh|em|toi)\s+(?:hoi|xin|biet)\s+)?(?:thu\s+vien\s+co\s+)?(?:co\s+)?", "", q_norm)
    q_norm = re.sub(r"^(?:vay|the)\s+", "", q_norm)

    patterns = [
        r"\b(?:sach|tai\s+lieu|giao\s+trinh|an\s+pham|luan\s+van)\s+(?:cua\s+(?:tac\s+gia\s+)?|ve\s+tac\s+gia\s+|tac\s+gia\s+)(.+)$",
        r"\b(?:cua\s+)?tac\s+gia\s+(.+)$",
        r"\b(?:sach|tai\s+lieu|giao\s+trinh)\s+cua\s+(.+)$",
    ]
    raw = None
    for p in patterns:
        m = re.search(p, q_norm)
        if m:
            raw = m.group(1).strip()
            break

    if not raw:
        return ""

    cleaned = clean_author_candidate(raw)
    return cleaned


GENERIC_BOOK_LISTING_PATTERNS = [
    r"^(?:da\s+)?(?:cho\s+(?:em|minh|toi)\s+(?:hoi|xin|biet)\s+)?(?:thu\s+vien\s+co\s+)?(?:nhung|cac|co\s+(?:nhung|cac)?)\s+(?:sach|tai\s+lieu|giao\s+trinh|an\s+pham|cuon|quyen)(?:\s+nao)?\b",
    r"^(?:da\s+)?(?:sach|tai\s+lieu|giao\s+trinh|cuon|quyen)\s+nao\b",
    r"^(?:liet\s+ke|danh\s+sach|tim\s+(?:cac|nhung)?)\s*(?:sach|tai\s+lieu|giao\s+trinh)\b",
    r"^(?:sach|tai\s+lieu|giao\s+trinh)\s+(?:xuat\s+ban|nam\s+\d{4}|tu\s+nam|sau\s+nam)\b",
]

GENERIC_TITLES = {
    "nhung sach nao", "sach nao", "cac sach nao", "nhung tai lieu nao", "tai lieu nao",
    "nhung giao trinh nao", "giao trinh nao", "nhung cuon nao", "cac cuon nao", "cuon nao",
    "nhung quyen nao", "cac quyen nao", "quyen nao", "nhung sach", "cac sach", "nhung tai lieu",
    "cac tai lieu", "sach", "tai lieu", "giao trinh", "cuon", "quyen", "an pham", "luan van",
    "co nhung sach nao", "co sach nao", "co tai lieu nao", "co nhung tai lieu nao", "nhung cuon sach",
    "cac cuon sach", "nhung", "cac", "nao", "co", "khong"
}


def is_generic_book_listing_query(query: str) -> bool:
    """
    Returns True if the query is asking to list or search books in general
    (e.g., 'những sách nào xuất bản năm 1991', 'sách nào xuất bản năm 2020')
    rather than asking about the attributes of a single specific book.
    """
    if not query:
        return False
    qn = normalize_text(query).strip()
    for pat in GENERIC_BOOK_LISTING_PATTERNS:
        if re.search(pat, qn):
            return True
    return False


if __name__ == "__main__":
    # Test
    test_text = "Kỹ Năng Lãnh Đạo Hiệu Quả"
    print(f"Original:  {test_text}")
    print(f"Normalized: {normalize_text(test_text)}")
    print(f"Keywords:  {extract_keywords(test_text)}")
    print(f"Hash:      {generate_document_hash(test_text)}")
    print(f"Synonyms of 'nghiên cứu': {suggest_synonyms('nghiên cứu')}")
    print(f"Spell check 'chat luong': {spell_correct('chat luong')}")
