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

    return suggestions


if __name__ == "__main__":
    # Test
    test_text = "Kỹ Năng Lãnh Đạo Hiệu Quả"
    print(f"Original:  {test_text}")
    print(f"Normalized: {normalize_text(test_text)}")
    print(f"Keywords:  {extract_keywords(test_text)}")
    print(f"Hash:      {generate_document_hash(test_text)}")
    print(f"Synonyms of 'nghiên cứu': {suggest_synonyms('nghiên cứu')}")
    print(f"Spell check 'chat luong': {spell_correct('chat luong')}")
