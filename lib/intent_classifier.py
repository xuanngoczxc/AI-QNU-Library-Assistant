"""
intent_classifier.py — Score-based intent classification + entity extraction
Thay thế keyword matching thuần túy bằng entity-based scoring.

Architecture:
  1. extract_entities(query) → identified entities (author, topic, chapter, action...)
  2. classify_intent(query)  → score-based intent selection

Entities detected:
  - action: tim/tra cuu/liet ke/tom tat/phan tich/giai thich/hoi/...
  - target: sach/tai lieu/giao trinh/luan van/pdf/...
  - author_name: extracted Vietnamese/Western name
  - publisher: NXB name
  - subject: topic/field keywords
  - chapter: chapter number
  - has_pdf_ref: references "file nay", "pdf nay"
  - has_author_keyword: "tac gia", "cua tac gia", "sach cua"
  - word_count: number of tokens
"""

import re
from typing import Optional
from .text_utils import extract_author_candidate, clean_author_candidate

VIETNAMESE_STOP_WORDS = {
    "cua", "va", "hoac", "nhung", "ma", "de", "tu", "toi", "boi",
    "o", "tai", "voi", "trong", "ngoai", "tren", "duoi", "ben", "giua",
    "khi", "luc", "sau", "truoc", "neu", "vi", "nen", "hay", "song",
    "toi", "minh", "em", "anh", "chi", "co", "ban", "no", "ho", "chung",
    "co", "khong", "sao", "gi", "dau", "bao", "nhieu", "nao", "the nao",
    "tim", "xem", "hoi", "cho", "gui", "xin", "muon", "can", "dang",
    "mot", "cai", "cuon", "quyen", "tap",
    "ok", "okay", "da", "roi", "nhe", "nha", "a", "e", "u",
}

VIETNAMESE_SURNAMES = {
    "nguyen", "tran", "le", "pham", "hoang", "huynh", "vo", "dang",
    "bui", "do", "ngo", "duong", "ly", "doan", "dinh", "trinh",
    "nhat", "quach", "mau", "lai", "son", "cu", "tieu",
    "cao", "mac", "ha", "ho", "kieu", "tang", "dong", "bac", "kha",
    "thach", "dieu", "chau", "nghiem", "diep", "thai", "to",
    "lam", "luu", "phung", "vu", "ma", "ton", "dai",
    "phan", "truong", "vuong", "chu", "ta", "luong",
}

ACTION_CATALOG = {
    "tim", "tra cuu", "liet ke", "search", "browse", "xem",
    "co sach nao", "co tai lieu nao", "co cuon nao",
    "goi y", "de xuat", "gioi thieu",
}
ACTION_CONTENT = {
    "tom tat", "phan tich", "giai thich", "trinh bay", "mo ta",
    "noi dung", "chi tiet", "doc", "trich", "doc file",
    "noi ve gi", "noi ve cai gi", "viet ve gi", "viet ve cai gi", "ban ve gi",
    "khai niem", "dinh nghia", "vi du", "giai thich ca tu",
    "bo cuc", "cau truc", "muc luc", "layout",
    "phuong phap", "ket qua", "ket luan", "danh gia",
    "gioi thieu ve", "ly do", "muc tieu",
    "so sanh", "sosanh",
}
ACTION_CHITCHAT = {
    "chao", "hello", "hi", "hey", "halo",
    "cam on", "thank", "thanks",
    "tam biet", "bye", "goodbye",
    "ban la ai", "ban ten gi", "ban co khoe khong",
}
ACTION_LIVE_INFO = {
    "hom nay", "ngay mai", "bay gio", "hien tai", "luc nay",
    "thoi tiet", "du bao", "tin tuc", "gia vang", "ty gia",
}

TARGET_BOOK = {
    "sach", "cuon sach", "giao trinh", "tai lieu", "luan van", "luan an",
    "de tai", "bai bao", "tap chi", "an pham",
}
TARGET_CONTENT = {
    "noi dung", "chuong", "phan", "muc", "trang",
    "file nay", "pdf nay", "tai lieu nay", "tai lieu do",
    "cuon nay", "sach nay", "van ban nay",
}

SUBJECT_KEYWORDS = {
    "nganh", "chu de", "linh vuc", "ky thuat", "phuong phap",
    "kinh te", "van hoc", "su pham", "cong nghe", "sinh hoc",
    "y hoc", "luat", "quan ly", "giao duc",
}

SUBJECT_NAMES = {
    "toan", "ly", "hoa", "sinh", "van", "su", "dia",
    "tin", "anh", "phap", "nga", "trung", "nhat", "han",
    "gdcd", "the duc", "quoc phong",
    "toan hoc", "vat ly", "hoa hoc", "sinh hoc",
    "van hoc", "lich su", "dia ly",
    "cong nghe", "tin hoc", "thong tin",
    "kinh te", "chinh tri", "triet hoc", "tam ly",
    "giao duc", "su pham", "luat", "y hoc", "duoc",
    "khoa hoc", "ky thuat", "moi truong",
    "thu vien", "thong tin hoc",
}

PUBLISHER_KEYWORDS = {"nha xuat ban", "nxb", "dai hoc", "khoa hoc"}
class QueryEntities:
    """Container cho các entities trích xuất từ query."""
    def __init__(self):
        self.raw: str = ""
        self.normalized: str = ""
        self.word_count: int = 0
        self.words: list[str] = []

        # Actions
        self.actions: set[str] = set()
        self.has_catalog_action: bool = False
        self.has_content_action: bool = False
        self.has_chitchat_action: bool = False
        self.has_live_action: bool = False

        # Targets
        self.has_book_target: bool = False
        self.has_content_target: bool = False

        # Named entities
        self.has_author_kw: bool = False  # "tac gia", "cua tac gia"
        self.has_publisher_kw: bool = False  # "nxb", "nha xuat ban"
        self.author_name: str = ""
        self.publisher_name: str = ""

        # Content references
        self.has_pdf_ref: bool = False
        self.chapter_num: Optional[int] = None
        self.has_summary_kw: bool = False

        # Subject
        self.has_subject_kw: bool = False
        self.subject_tokens: list[str] = []
        self.is_subject_name: bool = False  # 1-2 từ là tên môn học (vd: "toán", "vật lý")

        # Ambiguity
        self.is_short: bool = False  # <= 3 words
        self.is_very_short: bool = False  # <= 1 word

    def __repr__(self):
        flags = []
        if self.has_catalog_action: flags.append("CAT_ACT")
        if self.has_content_action: flags.append("CON_ACT")
        if self.has_chitchat_action: flags.append("CHIT_ACT")
        if self.has_book_target: flags.append("BOOK")
        if self.has_content_target: flags.append("CONT_TGT")
        if self.has_author_kw: flags.append(f"AUTHOR={self.author_name}")
        if self.has_publisher_kw: flags.append("PUB")
        if self.chapter_num: flags.append(f"CH{self.chapter_num}")
        if self.has_pdf_ref: flags.append("PDF_REF")
        if self.has_subject_kw: flags.append("SUBJECT")
        return f"Entities({self.word_count}w, {'|'.join(flags)})"


def extract_entities(query: str) -> QueryEntities:
    """Trích xuất entities từ query bằng NLP-inspired heuristics."""
    import unicodedata

    ent = QueryEntities()
    ent.raw = query or ""

    # Normalize
    text = query.lower().strip()
    # Remove diacritics
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    replacements = {'ă': 'a', 'â': 'a', 'ê': 'e', 'ô': 'o', 'ơ': 'o', 'ư': 'u', 'đ': 'd'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    ent.normalized = text
    ent.words = text.split()
    ent.word_count = len(ent.words)
    ent.is_short = ent.word_count <= 3
    ent.is_very_short = ent.word_count <= 1

    # ── 1. Detect actions ──
    for kw in ACTION_CATALOG:
        if kw in text:
            ent.has_catalog_action = True
            ent.actions.add("catalog")

    for kw in ACTION_CONTENT:
        if kw in text:
            ent.has_content_action = True
            ent.actions.add("content")

    for kw in ACTION_CHITCHAT:
        if kw == text or text.startswith(kw) or text.endswith(kw):
            ent.has_chitchat_action = True
            ent.actions.add("chitchat")

    for kw in ACTION_LIVE_INFO:
        if kw in text:
            ent.has_live_action = True
            ent.actions.add("live_info")

    # ── 2. Detect targets ──
    for kw in TARGET_BOOK:
        if kw in text:
            ent.has_book_target = True
            break

    for kw in TARGET_CONTENT:
        if kw in text:
            ent.has_content_target = True
            break

    # ── 3. Detect author ──
    is_asking_author_name = bool(re.search(r"\b(?:ai\s+la\s+tac\s+gia|tac\s+gia\s+.*(?:la\s+ai|ten\s+gi|nguoi\s+nao)|ai\s+viet|ai\s+bien\s+soan)\b", text))
    if not is_asking_author_name and re.search(r"(?:tac gia|cua tac gia|sach cua|tai lieu cua|giao trinh cua)", text):
        ent.has_author_kw = True
        # Extract author name
        cand = extract_author_candidate(text)
        if cand:
            name_tokens = cand.split()
            # Filter out stop words from name
            name_tokens = [t for t in name_tokens if t not in VIETNAMESE_STOP_WORDS and len(t) >= 2]
            if len(name_tokens) >= 2:
                ent.author_name = " ".join(name_tokens)

    # Heuristic: surname first without "tac gia"
    if not ent.author_name and not ent.has_author_kw:
        words = ent.words
        if 2 <= len(words) <= 5 and words[0] in VIETNAMESE_SURNAMES:
            topic_words = {"hoc", "phap", "luat", "su", "hoa", "van", "nghe", "thuat",
                           "cong", "nghiep", "xay", "dung", "nong", "kinh", "tai", "chinh",
                           "giao", "duc", "tam", "ly", "xa", "hoi", "tri", "nhan", "tao",
                           "may", "tinh", "phan", "mem", "dien", "tu", "thong", "bao",
                           "sinh", "y", "duoc", "ky", "thuat", "nganh", "chu", "de", "linh", "vuc"}
            if not any(t in topic_words for t in words[1:]):
                ent.has_author_kw = True
                ent.author_name = " ".join(words)

    # Heuristic: "sách <họ>..." hoặc "tìm sách <họ>..." (không có "của")
    if not ent.author_name and not ent.has_author_kw:
        words = ent.words
        if len(words) >= 3:
            for idx, w in enumerate(words):
                if w in VIETNAMESE_SURNAMES and idx >= 1:
                    # Các từ trước họ: phải chứa "sach", "tai lieu" hoặc action word
                    preceding = words[:idx]
                    has_book_action = any(p in {"tim", "tra", "co", "muon", "xem", "liet ke"} for p in preceding)
                    has_book_target = any(p in {"sach", "tai lieu", "giao trinh", "bai", "cuon", "quyen"} for p in preceding)
                    if has_book_action or has_book_target:
                        name_part = words[idx:]
                        if 2 <= len(name_part) <= 5:
                            topic_words = {"hoc", "phap", "luat", "su", "hoa", "van", "nghe", "thuat",
                                           "cong", "nghiep", "xay", "dung", "nong", "kinh", "tai", "chinh",
                                           "giao", "duc", "tam", "ly", "xa", "hoi", "tri", "nhan", "tao",
                                           "may", "tinh", "phan", "mem", "dien", "tu", "thong", "bao",
                                           "sinh", "y", "duoc", "ky", "thuat", "nganh", "chu", "de", "linh", "vuc"}
                            if not any(t in topic_words for t in name_part[1:]):
                                ent.has_author_kw = True
                                ent.author_name = " ".join(name_part)
                                break

    # ── 4. Detect publisher ──
    if re.search(r"(?:nha xuat ban|nxb)", text):
        ent.has_publisher_kw = True
        m = re.search(r"(?:nha\s*xuat\s*ban|nxb)\s+(.+?)(?:\s+(?:khong|ko|ne|nha)\s*$|$)", text)
        if m:
            pub = m.group(1).strip().rstrip(",. ")
            pub_tokens = [t for t in pub.split() if t not in VIETNAMESE_STOP_WORDS and len(t) >= 2]
            if pub_tokens:
                ent.publisher_name = " ".join(pub_tokens)

    # ── 5. Detect chapter ──
    cm = re.search(r"(?:chuong|chapter)\s*(\d+)", text)
    if cm:
        try:
            ent.chapter_num = int(cm.group(1))
        except ValueError:
            pass

    # ── 6. Detect PDF reference ──
    if re.search(r"(?:file\s*nay|pdf\s*nay|tai\s*lieu\s*nay|cuon\s*nay|sach\s*nay)", text):
        ent.has_pdf_ref = True

    # ── 7. Detect summary ──
    if re.search(r"(?:tom tat|summary|tóm tắt)", text):
        ent.has_summary_kw = True

    # ── 8. Detect subject/topic ──
    for kw in SUBJECT_KEYWORDS:
        if kw in text:
            ent.has_subject_kw = True
            # Extract subject tokens (words after subject indicator)
            m = re.search(rf"(?:nganh|chu de|linh vuc)\s+(.+?)(?:\s+(?:khong|ko|ne)\s*$|$)", text)
            if m:
                subj = m.group(1).strip()
                subj_tokens = [t for t in subj.split() if t not in VIETNAMESE_STOP_WORDS and len(t) >= 2]
                ent.subject_tokens = subj_tokens
            break

    # ── 9. Detect existence check ("co...khong?") ──
    ent.is_existence_check = bool(re.search(r"\bco\b.*\bkhong\b", text))

    # ── 10. Detect subject name (1 từ là tên môn học) ──
    ent.is_subject_name = False
    if ent.word_count == 1 and ent.words[0] in SUBJECT_NAMES:
        ent.is_subject_name = True
        ent.has_subject_kw = True
    elif ent.word_count == 2 and " ".join(ent.words) in SUBJECT_NAMES:
        ent.is_subject_name = True
        ent.has_subject_kw = True

    return ent


def classify_intent(query: str) -> str:
    """Score-based intent classification.
    Returns one of: chitchat, live_info, content_qa, catalog_search, clarify, unknown
    """
    if not query or not query.strip():
        return "unknown"

    ent = extract_entities(query)

    # ── Score each intent ──
    scores = {
        "chitchat": 0.0,
        "live_info": 0.0,
        "content_qa": 0.0,
        "catalog_search": 0.0,
        "clarify": 0.0,
        "unknown": 0.0,
    }

    # ── Chitchat scoring ──
    if "chitchat" in ent.actions:
        scores["chitchat"] += 10
    if ent.is_very_short and ent.has_chitchat_action:
        scores["chitchat"] += 5
    # Short query with no catalog/content signals → possible chitchat
    if ent.is_very_short and not ent.has_book_target and not ent.has_content_target:
        if ent.word_count == 1 and ent.words[0] in {"ok", "okay", "vl", "hay", "gioi", "tot", "xau"}:
            scores["chitchat"] += 3

    # ── Live info scoring ──
    if "live_info" in ent.actions:
        scores["live_info"] += 10

    # ── Content QA scoring ──
    if "content" in ent.actions:
        scores["content_qa"] += 8
    if ent.has_content_target:
        scores["content_qa"] += 6
    if ent.chapter_num:
        scores["content_qa"] += 5
    if ent.has_pdf_ref:
        scores["content_qa"] += 7
    if ent.has_summary_kw:
        scores["content_qa"] += 4
    # "noi dung" without book target → likely content
    if "noi dung" in ent.normalized and not ent.has_book_target:
        scores["content_qa"] += 3

    # ── Catalog search scoring ──
    if "catalog" in ent.actions:
        scores["catalog_search"] += 8
    if ent.has_book_target:
        scores["catalog_search"] += 6
    if ent.has_author_kw and ent.author_name:
        scores["catalog_search"] += 7
    if ent.has_publisher_kw:
        scores["catalog_search"] += 6
    if ent.has_subject_kw:
        scores["catalog_search"] += 4
    if ent.is_existence_check:
        scores["catalog_search"] += 3
    # "co" + book keyword pattern
    if re.search(r"\bco\b.*\b(?:sach|tai lieu|cuon)\b", ent.normalized):
        scores["catalog_search"] += 5
    # Question word + book/topic pattern
    if re.search(r"\b(?:co|nao|gi)\b", ent.normalized) and ent.has_book_target:
        scores["catalog_search"] += 4

    # ── Clarify scoring ──
    # Only when query is VERY short + ambiguous
    if ent.is_short and not ent.has_catalog_action and not ent.has_content_action:
        if ent.word_count <= 2 and not ent.is_subject_name:
            # Single word like "toan", "ly", "hoa" → NẾU là tên môn học thì chuyển catalog
            scores["clarify"] += 2
        elif ent.word_count == 3 and ent.has_book_target:
            # "cuon nao hay" → borderline
            scores["clarify"] += 2

    # ── Subject name boost: "toán" → catalog_search ──
    if ent.is_subject_name:
        scores["catalog_search"] += 6
        scores["clarify"] -= 3  # giảm clarify để catalog thắng

    # ── Penalty: reduce ambiguous intents when strong signals exist ──
    if ent.has_book_target or ent.has_author_kw:
        scores["clarify"] -= 5
        scores["unknown"] -= 5
    if ent.has_content_target or ent.chapter_num:
        scores["catalog_search"] -= 2
    # Short query with ONLY author name → catalog
    if ent.has_author_kw and ent.author_name and not ent.has_content_action:
        scores["catalog_search"] += 3

    # ── Select winner ──
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    # Minimum threshold — if nothing scored, return unknown
    if best_score <= 0:
        return "unknown"

    # If top 2 are very close and one is clarify → prefer the other
    sorted_intents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_intents) >= 2:
        top1, top2 = sorted_intents[0], sorted_intents[1]
        if top2[0] == "clarify" and top1[1] - top2[1] < 3:
            return top1[0]

    return best_intent


def detect_greeting(query: str) -> bool:
    """Detect greeting/chitchat queries."""
    if not query:
        return False
    ent = extract_entities(query)
    return ent.has_chitchat_action and ent.word_count <= 4


def detect_author_query(query: str) -> bool:
    """Detect if query is searching for books by author."""
    ent = extract_entities(query)
    return ent.has_author_kw and bool(ent.author_name)


def detect_publisher_query(query: str) -> bool:
    """Detect if query is searching by publisher."""
    ent = extract_entities(query)
    return ent.has_publisher_kw and bool(ent.publisher_name)
