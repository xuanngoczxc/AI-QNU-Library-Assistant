"""
lib/author_resolver.py — Systematic multi-attribute resolution for author queries.
Technique:
  1. Lexicon-backed Maximal Matching for Named Entity Recognition (NER)
  2. Multi-slot query decomposition (Author + Year + DocType + Location + Sort)
  3. Context-aware constraint filtering & intelligent conversational explanations
"""

import re
from typing import Optional, Dict, List, Tuple, Any
from collections import OrderedDict

from .text_utils import normalize_text, VIETNAMESE_SURNAMES
from .catalog import detect_year_constraint, detect_language_preference
from .doc_type_resolver import detect_doc_type_from_query, doc_matches_target_doc_type
from .conversation_resolver import detect_recency_preference, detect_oldest_preference


class AuthorIndex:
    """
    Inverted Index and Lexicon for all authors in the QNU Library database.
    Enables O(1) author lookup and Maximal Matching NER extraction.
    """
    def __init__(self, documents: List[dict]):
        self.author_to_docs: Dict[str, List[dict]] = {}
        self.author_lexicon: Dict[str, int] = {}
        self.author_display_map: Dict[str, str] = {}
        self._build_index(documents)

    def _build_index(self, documents: List[dict]):
        seen_pairs = set()
        for d in documents:
            title_key = d.get('title', '').strip().lower()
            combined_people = re.split(r'[;,\n/]+', d.get('author', '')) + re.split(r'[;,\n/]+', d.get('responsibility', ''))
            for sa in combined_people:
                sa_clean = re.sub(r'\(.*?\)|\[.*?\]', '', sa).strip()
                sa_clean = re.sub(r'\b(?:h\.d\.|ch\.b\.|b\.s\.|h\.d|chb|dich|bien\s+soan|chu\s+bien)\b', '', sa_clean, flags=re.I).strip()
                norm = normalize_text(sa_clean).strip()
                toks = norm.split()
                if 2 <= len(toks) <= 5:
                    self.author_lexicon[norm] = self.author_lexicon.get(norm, 0) + 1
                    if norm not in self.author_display_map and any(c.isupper() for c in sa_clean):
                        self.author_display_map[norm] = sa_clean.strip()
                    
                    pair_key = (norm, title_key)
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        if norm not in self.author_to_docs:
                            self.author_to_docs[norm] = []
                        self.author_to_docs[norm].append(d)


_SLOT_BOUNDARY_PATTERN = re.compile(
    r"\s+\b("
    r"(?:xuat\s+ban\s+(?:vao\s+)?nam|nam\s+xb|nam\s+xuat\s+ban|xuat\s+ban\s+sau\s+nam|xuat\s+ban\s+truoc\s+nam|tai\s+ban|nam)\s+\d{4}"
    r"|(?:tu\s+nam|den\s+nam|sau\s+nam|truoc\s+nam)\s+\d{4}"
    r"|nam\s+\d{4}"
    r"|moi\s+nhat|gan\s+day\s+nhat|cu\s+nhat|lau\s+doi\s+nhat"
    r"|o\s+dau|o\s+phong\s+(?:doc\s+)?nao|o\s+kho\s+nao|phong\s+nao|kho\s+nao"
    r"|co\s+(?:nhung\s+|cac\s+)?(?:sach|tai\s+lieu|giao\s+trinh|cuon|quyen)\s+nao"
    r"|co\s+(?:sach|tai\s+lieu|cuon)\s+(?:gi|nao)"
    r"|co\s+(?:duoc\s+)?(?:muon|muon\s+ve|doc)\s*(?:khong|ko)?"
    r"|co\s+khong|co\s+ko|co\s+k|co\s+chua|con\s+khong|con\s+ko"
    r"|dung\s+khong|phai\s+khong"
    r"|khong\s+a|ko\s+a|k\s+a|khong|ko|k|ha|nhi|the|vay|a|nhe|nha|voi|oi"
    r"|nganh\s+[a-z\s]+"
    r"|tieng\s+(?:anh|viet|nga|trung|phap)"
    r")\b.*$",
    flags=re.I
)


def extract_author_and_remaining(query: str, author_index: Optional[AuthorIndex] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts (author_norm, author_display, remaining_query) using
    Dictionary-backed Maximal Matching with grammar fallback.
    """
    if not query:
        return None, None, None

    q_norm = normalize_text(query).strip()
    m = re.search(r'\b(?:sach|tai\s+lieu|giao\s+trinh|an\s+pham)?\s*(?:cua\s+(?:tac\s+gia\s+)?|ve\s+tac\s+gia\s+|tac\s+gia\s+)', q_norm)
    if not m:
        m = re.search(r'\bcua\s+', q_norm)

    matched_author = None
    remaining = ""

    if m:
        tail = q_norm[m.end():].strip()
        words = tail.split()
        if not words:
            return None, None, None

        # 1. Lexicon maximal matching (5, 4, 3, 2 grams)
        if author_index and author_index.author_lexicon:
            for k in range(min(5, len(words)), 1, -1):
                cand = " ".join(words[:k])
                if cand in author_index.author_lexicon:
                    matched_author = cand
                    remaining = " ".join(words[k:])
                    break

        # 2. Grammar fallback with slot boundary stripping
        if not matched_author:
            cleaned_tail = _SLOT_BOUNDARY_PATTERN.sub('', tail).strip()
            c_words = cleaned_tail.split()
            if 2 <= len(c_words) <= 5:
                cand = " ".join(c_words)
                matched_author = cand
                remaining = tail[len(cand):].strip()
    else:
        # Pattern: "<Tên tác giả> có sách / tài liệu / giáo trình..."
        m_start = re.match(r'^([a-z\s]{3,35}?)\s+co\s+(?:nhung\s+|cac\s+)?(?:sach|tai\s+lieu|giao\s+trinh|cuon)\b', q_norm)
        if m_start:
            cand_head = m_start.group(1).strip()
            head_words = cand_head.split()
            if 2 <= len(head_words) <= 5 and (head_words[0] in VIETNAMESE_SURNAMES or (author_index and cand_head in author_index.author_lexicon)):
                matched_author = cand_head
                remaining = q_norm[len(cand_head):].strip()

    if not matched_author or len(matched_author) < 2:
        return None, None, None

    # Resolve display name
    author_display = None
    if author_index and matched_author in author_index.author_display_map:
        author_display = author_index.author_display_map[matched_author]

    if not author_display:
        words_orig = query.split()
        words_norm = q_norm.split()
        cand_words = matched_author.split()
        for i in range(len(words_norm) - len(cand_words) + 1):
            if words_norm[i:i+len(cand_words)] == cand_words:
                author_display = " ".join(words_orig[i:i+len(cand_words)]).strip(' ?.,!;:"\'').title()
                break

    if not author_display:
        author_display = matched_author.title()

    return matched_author, author_display, remaining


def handle_author_inquiry(query: str, retriever=None, author_index: Optional[AuthorIndex] = None, build_source_item_fn=None) -> Optional[Dict[str, Any]]:
    """
    Comprehensive multi-attribute resolver for author inquiries.
    Handles:
      - Author + Year (e.g. 'sách của tác giả Hồ Chí Minh xuất bản năm 2025')
      - Author + Doc Type (e.g. 'giáo trình của tác giả Hoàng Phê')
      - Author + Location (e.g. 'sách của tác giả Tô Hoài ở đâu / ở phòng nào')
      - Author + Recency / Oldest (e.g. 'sách của Nguyễn Nhật Ánh mới nhất')
      - Author existence (e.g. 'vậy sách của tác giả Nguyễn Thành Đạt có không?')
      - Not-found author explanations
    """
    if not query:
        return None

    q_norm = normalize_text(query).strip()

    # Step 1: Extract author entity and remaining slots
    author_norm, author_display, remaining = extract_author_and_remaining(query, author_index)
    if not author_norm:
        return None

    # Step 2: Validate not a department / major / subject false positive
    category_pattern = r"\b(?:nganh|chuyen\s+nganh|khoa|bo\s+mon|mon\s+hoc|mon|nxb|nha\s+xuat\s+ban|truong|lop|chu\s+de|linh\s+vuc|hoc\s+phan)\b"
    if re.search(category_pattern, author_norm):
        return None

    has_explicit_author = bool(re.search(r"\b(?:tac\s+gia|dong\s+tac\s+gia|chu\s+bien|dich\s+gia|nha\s+van|nha\s+tho)\b", q_norm))
    norm_words = author_norm.split()
    if norm_words and norm_words[0] in VIETNAMESE_SURNAMES:
        has_explicit_author = True

    if not has_explicit_author and author_index and (author_norm not in author_index.author_lexicon):
        return None

    # Step 3: Fetch base documents for this author
    all_author_docs = []
    if author_index is not None:
        if author_norm in author_index.author_to_docs:
            all_author_docs = list(author_index.author_to_docs[author_norm])
        else:
            # Check if author_norm matches any full author name in the index with word boundary
            seen_titles = set()
            pattern = r'(?<!\w)' + re.escape(author_norm) + r'(?!\w)'
            for key, kdocs in author_index.author_to_docs.items():
                if re.search(pattern, key):
                    for d in kdocs:
                        t_k = d.get('title', '').strip().lower()
                        if t_k not in seen_titles:
                            seen_titles.add(t_k)
                            all_author_docs.append(d)
    elif retriever and hasattr(retriever, 'bm25'):
        # Fallback search ONLY if index not yet initialized
        seen = set()
        pattern = r'(?<!\w)' + re.escape(author_norm) + r'(?!\w)'
        for d in retriever.bm25.documents:
            combined_people = re.split(r'[;,\n/]+', d.get('author', '')) + re.split(r'[;,\n/]+', d.get('responsibility', ''))
            matched = False
            for sa in combined_people:
                sa_clean = re.sub(r'\(.*?\)|\[.*?\]', '', sa).strip()
                sa_clean = re.sub(r'\b(?:h\.d\.|ch\.b\.|b\.s\.|h\.d|chb|dich|bien\s+soan|chu\s+bien)\b', '', sa_clean, flags=re.I).strip()
                norm = normalize_text(sa_clean).strip()
                if re.search(pattern, norm):
                    matched = True
                    break
            if matched:
                t_k = d.get('title', '').strip().lower()
                if t_k not in seen:
                    seen.add(t_k)
                    all_author_docs.append(d)

    # Step 4: Handle Author NOT FOUND in Library
    if not all_author_docs:
        if not has_explicit_author:
            return None
        ans = (
            f"Hiện tại Thư viện QNU chưa tìm thấy tài liệu nào của tác giả **\"{author_display}\"** trong hệ thống lưu trữ. 🤔\n\n"
            f"Bạn vui lòng kiểm tra lại chính xác tên tác giả hoặc tìm kiếm theo tên giáo trình / tài liệu cụ thể nhé!"
        )
        return {
            "answer": ans,
            "sources": [],
            "total_sources": 0,
            "total_found": 0,
            "matched_attr": "author_location",
            "found": False,
            "author_display": author_display,
            "candidate": author_display
        }

    # Step 5: Multi-attribute Slot Extraction on Query
    target_year, year_dir = detect_year_constraint(query)
    target_doc_type = detect_doc_type_from_query(query)
    is_recency = detect_recency_preference(query)
    is_oldest = detect_oldest_preference(query)
    is_loc_q = bool(re.search(r"\b(?:nam\s+o|o\s+tai|muon\s+o|tim\s+o|thi\s+qua|thi\s+o|phong\s+doc\s+nao|phong\s+nao|kho\s+nao|o\s+dau|dau)\b", q_norm))

    filtered_docs = list(all_author_docs)

    # Filter by Doc Type if requested
    if target_doc_type:
        filtered_docs = [d for d in filtered_docs if doc_matches_target_doc_type(d, target_doc_type)]

    # Filter by Year if requested
    year_matched = True
    if target_year:
        docs_matching_year = []
        for d in filtered_docs:
            try:
                y = int(d.get('year', 0))
                if year_dir == 'exact' and y == target_year:
                    docs_matching_year.append(d)
                elif year_dir == 'after' and y >= target_year:
                    docs_matching_year.append(d)
                elif year_dir == 'before' and y <= target_year:
                    docs_matching_year.append(d)
            except Exception:
                pass

        if docs_matching_year:
            filtered_docs = docs_matching_year
        else:
            year_matched = False

    # Sort documents
    def _doc_year_val(d):
        try:
            val = str(d.get('year', '')).strip()
            return int(val) if val.isdigit() else 0
        except Exception:
            return 0

    if is_recency:
        filtered_docs.sort(key=_doc_year_val, reverse=True)
    elif is_oldest:
        filtered_docs.sort(key=_doc_year_val)
    else:
        # Default: newest first
        filtered_docs.sort(key=lambda d: (-_doc_year_val(d), len(d.get('location', '')) == 0))

    # Step 6: Formulate Intelligent Conversational Responses
    if target_year and not year_matched:
        # User specified a year, but author has no books for that year (e.g. Hồ Chí Minh năm 2025)
        valid_years = sorted([_doc_year_val(d) for d in all_author_docs if _doc_year_val(d) > 1900])
        if valid_years:
            if valid_years[0] == valid_years[-1]:
                year_range_info = f"được xuất bản vào năm **{valid_years[0]}**"
            else:
                year_range_info = f"được xuất bản từ năm **{valid_years[0]}** đến **{valid_years[-1]}**"
        else:
            year_range_info = "được xuất bản ở các năm trước đó"

        ans = (
            f"Tác giả **{author_display}** hiện có **{len(all_author_docs)} tài liệu** trong Thư viện QNU, "
            f"tuy nhiên chưa có tài liệu nào xuất bản vào năm **{target_year}** "
            f"(các tài liệu hiện lưu trữ {year_range_info}).\n\n"
            f"Dưới đây là một số tài liệu tái bản gần đây nhất của tác giả để bạn tham khảo:"
        )
        # Present recent editions by this author
        recent_docs = sorted(all_author_docs, key=_doc_year_val, reverse=True)[:10]
        final_docs = recent_docs
    elif target_doc_type and not filtered_docs:
        # User specified a doc type, but author doesn't have that type
        ans = (
            f"Tác giả **{author_display}** hiện có **{len(all_author_docs)} tài liệu** trong Thư viện QNU, "
            f"tuy nhiên không có tài liệu nào thuộc loại **{target_doc_type}**.\n\n"
            f"Dưới đây là các tài liệu hiện có của tác giả để bạn tham khảo:"
        )
        final_docs = all_author_docs[:10]
    elif is_loc_q:
        rooms_found = OrderedDict()
        for d in filtered_docs:
            loc = d.get('location', '')
            for part in re.split(r'[;,\n]', loc):
                part = part.strip()
                m_rm = re.match(r'^(Phòng [^:]+|Kho [^:]+|Luận văn)', part)
                if m_rm:
                    rooms_found[m_rm.group(1).strip()] = True

        if rooms_found:
            rooms_str = ", ".join(f"**{r}**" for r in rooms_found.keys())
            ans = (
                f"Các tài liệu của tác giả **{author_display}** ({len(filtered_docs)} tài liệu) hiện đang được lưu trữ tại phòng đọc / kho: {rooms_str}.\n\n"
                f"Bạn có thể xem chi tiết vị trí và tình trạng từng cuốn ở danh sách bên dưới:"
            )
        else:
            ans = (
                f"Dạ có nhé! Thư viện QNU hiện có **{len(filtered_docs)} tài liệu** của tác giả **{author_display}**.\n\n"
                f"Bạn có thể xem chi tiết ở danh sách bên dưới:"
            )
        final_docs = filtered_docs
    elif target_year:
        if year_dir == 'exact':
            y_phrase = f"xuất bản vào năm **{target_year}**"
        elif year_dir == 'after':
            y_phrase = f"xuất bản từ năm **{target_year}** trở lại đây"
        else:
            y_phrase = f"xuất bản trước năm **{target_year}**"
        ans = (
            f"Dạ có nhé! Thư viện QNU hiện có **{len(filtered_docs)} tài liệu** của tác giả **{author_display}** {y_phrase}.\n\n"
            f"Bạn có thể xem chi tiết ở danh sách bên dưới:"
        )
        final_docs = filtered_docs
    elif target_doc_type:
        ans = (
            f"Dạ có nhé! Thư viện QNU hiện có **{len(filtered_docs)} {target_doc_type}** của tác giả **{author_display}**.\n\n"
            f"Bạn có thể xem chi tiết ở danh sách bên dưới:"
        )
        final_docs = filtered_docs
    elif is_recency:
        top_y = _doc_year_val(filtered_docs[0]) if filtered_docs else 0
        top_t = filtered_docs[0].get('title', '') if filtered_docs else ''
        ans = (
            f"Tài liệu xuất bản gần đây nhất của tác giả **{author_display}** trong Thư viện QNU là cuốn **\"{top_t}\"** (năm {top_y}).\n\n"
            f"Dưới đây là danh sách các tài liệu được sắp xếp theo năm mới nhất:"
        )
        final_docs = filtered_docs
    elif is_oldest:
        top_y = _doc_year_val(filtered_docs[0]) if filtered_docs else 0
        top_t = filtered_docs[0].get('title', '') if filtered_docs else ''
        ans = (
            f"Tài liệu xuất bản lâu đời nhất của tác giả **{author_display}** trong Thư viện QNU là cuốn **\"{top_t}\"** (năm {top_y}).\n\n"
            f"Dưới đây là danh sách các tài liệu được sắp xếp từ năm cũ nhất:"
        )
        final_docs = filtered_docs
    else:
        if len(filtered_docs) == 1:
            ans = (
                f"Dạ có nhé! Thư viện QNU hiện có **1 tài liệu** của tác giả **{author_display}**.\n\n"
                f"Bạn có thể xem chi tiết thông tin và tình trạng tài liệu bên dưới:"
            )
        else:
            ans = (
                f"Dạ có nhé! Thư viện QNU hiện có **{len(filtered_docs)} tài liệu** của tác giả **{author_display}**.\n\n"
                f"Bạn có thể xem chi tiết ở danh sách bên dưới:"
            )
        final_docs = filtered_docs

    if build_source_item_fn:
        srcs = [build_source_item_fn({"metadata": d}) for d in final_docs[:min(len(final_docs), 60)]]
    else:
        srcs = [{"title": d.get("title", ""), "author": d.get("author", ""), "year": d.get("year", ""), "metadata": d} for d in final_docs[:60]]

    return {
        "answer": ans,
        "sources": srcs,
        "total_sources": len(srcs),
        "total_found": len(filtered_docs),
        "matched_attr": "author_location",
        "found": True,
        "author_display": author_display,
        "doc": final_docs[0] if final_docs else None,
        "candidate": author_display
    }
