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
                while sa_clean and sa_clean[-1] in " ?,!;:'\"\r\n":
                    sa_clean = sa_clean[:-1].strip()
                norm = normalize_text(sa_clean).strip()
                toks = norm.split()
                if 2 <= len(toks) <= 5:
                    forms = [norm]
                    if len(toks) == 2:
                        forms.append(f"{toks[1]} {toks[0]}")
                    elif len(toks) == 3:
                        forms.append(f"{toks[2]} {toks[0]} {toks[1]}")
                        forms.append(f"{toks[2]} {toks[1]} {toks[0]}")
                        forms.append(f"{toks[1]} {toks[2]} {toks[0]}")

                    for f_norm in set(forms):
                        self.author_lexicon[f_norm] = self.author_lexicon.get(f_norm, 0) + 1
                        if f_norm not in self.author_display_map and any(c.isupper() for c in sa_clean):
                            self.author_display_map[f_norm] = sa_clean.strip()
                        
                        pair_key = (f_norm, title_key)
                        if pair_key not in seen_pairs:
                            seen_pairs.add(pair_key)
                            if f_norm not in self.author_to_docs:
                                self.author_to_docs[f_norm] = []
                            self.author_to_docs[f_norm].append(d)


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


def extract_author_and_remaining(query: str, author_index: Optional[AuthorIndex] = None) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Extracts (author_norm, author_display, remaining_query) using
    Dictionary-backed Maximal Matching with grammar fallback.
    """
    if not query:
        return None, None, None, None

    q_norm = normalize_text(query).strip()
    # Không nhận diện "mở cửa", "đóng cửa" là từ nối "của"
    if re.search(r'\b(?:mo|dong|khoa|canh)\s+cua\b', q_norm) and not re.search(r'\b(?:tac\s+gia|dong\s+tac\s+gia|chu\s+bien)\b', q_norm):
        return None, None, None, None

    m = re.search(r'\b(?:sach|tai\s+lieu|giao\s+trinh|an\s+pham)?\s*(?:(?<!\bmo\s)(?<!\bdong\s)(?<!\bkhoa\s)(?<!\bcanh\s)cua\s+(?:tac\s+(?:gia|ga)\s+|tg\s+)?|ve\s+tac\s+(?:gia|ga)\s+|tac\s+(?:gia|ga)\s+|tg\s+)', q_norm)
    if not m:
        m = re.search(r'(?<!\bmo\s)(?<!\bdong\s)(?<!\bkhoa\s)(?<!\bcanh\s)\bcua\s+', q_norm)

    matched_author = None
    remaining = ""
    title_cand = ""

    if m:
        head = q_norm[:m.start()].strip()
        head = re.sub(r"^(?:sach|tai\s+lieu|giao\s+trinh|cuon\s+sach|cuon|quyen|tim|an\s+pham)\s+", "", head).strip()
        if len(head.split()) >= 2 and head not in ("nhung sach", "cac sach", "tai lieu", "giao trinh"):
            title_cand = head
        tail = q_norm[m.end():].strip()
        words = tail.split()
        if not words:
            return None, None, None, None

        # 1. Lexicon maximal matching (5, 4, 3, 2 grams)
        if author_index and author_index.author_lexicon:
            for k in range(min(5, len(words)), 1, -1):
                cand = " ".join(words[:k])
                if cand in author_index.author_lexicon:
                    matched_author = cand
                    remaining = " ".join(words[k:])
                    # Kiểm tra xem có từ thừa (như XYZ) đứng trước mốc thời gian/hành vi không
                    slot_b = re.search(r'\b(?:moi\s+nhat|gan\s+day\s+nhat|gan\s+nhat|cu\s+nhat|lau\s+doi\s+nhat|som\s+nhat|(?:xuat\s+ban\s+)?(?:vao\s+)?nam\s+\d{4}|(?:tu|sau|truoc|den)\s+nam\s+\d{4}|xuat\s+ban|nam\s+\d{4}|\d{4}|phat\s+hanh|tai\s+ban|co\s+khong|khong|o\s+dau|phong|kho)\b', remaining)
                    extra_words = remaining[:slot_b.start()].strip() if slot_b else ""
                    if extra_words and len(extra_words.split()) <= 2 and not re.search(r'\b(?:moi|gan|cu|lau|som|nhat)\b', extra_words):
                        matched_author = f"{matched_author} {extra_words}"
                        remaining = remaining[len(extra_words):].strip()
                    break

        # 1.5. Fuzzy author lexicon fallback (khi người dùng gõ sai 1-2 ký tự tên tác giả)
        if not matched_author and author_index and author_index.author_lexicon:
            import difflib
            for k in range(min(4, len(words)), 1, -1):
                cand = " ".join(words[:k])
                if len(cand) >= 6:
                    close = difflib.get_close_matches(cand, author_index.author_lexicon.keys(), n=1, cutoff=0.88)
                    if close:
                        matched_author = close[0]
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
        return None, None, None, None

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
                slice_orig = " ".join(words_orig[i:i+len(cand_words)])
                while slice_orig and slice_orig[-1] in " ?,!;:\'\"":
                    slice_orig = slice_orig[:-1].strip()
                author_display = slice_orig
                break

    if not author_display:
        author_display = matched_author.title()

    return matched_author, author_display, remaining, title_cand


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
    author_norm, author_display, remaining, title_cand = extract_author_and_remaining(query, author_index)
    if not author_norm:
        return None

    # Step 2: Validate not a department / major / subject false positive
    category_pattern = r"\b(?:nganh|chuyen\s+nganh|khoa|bo\s+mon|mon\s+hoc|mon|nxb|nha\s+xuat\s+ban|truong|lop|chu\s+de|linh\s+vuc|hoc\s+phan)\b"
    if re.search(category_pattern, author_norm):
        return None

    question_pattern = r"\b(?:luc\s+may\s+gio|may\s+gio|bao\s+nhieu|o\s+dau|khi\s+nao|the\s+nao|sao|thu\s+may|ngay\s+nao)\b"
    if re.search(question_pattern, author_norm):
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

    # Step 3.5: Kiểm tra tính nhất quán dấu tiếng Việt (VD: Nguyễn Nhật Ánh vs Nguyễn Nhật Anh)
    if all_author_docs:
        m_a = re.search(r'\b(?:tác\s+(?:giả|gả|gia)\s+|của\s+(?:tác\s+(?:giả|gả|gia)\s+)?)([A-Za-zÀ-ỹ\s]+?)(?:\s+(?:xuất\s+bản|năm|ở|thuộc|không|chưa)|\?|$)', query, re.IGNORECASE)
        if m_a:
            u_author = m_a.group(1).strip()
            u_words = u_author.lower().split()
            r_words = author_display.lower().split()
            if len(u_words) == len(r_words):
                for uw, rw in zip(u_words, r_words):
                    if any(ord(c) > 127 for c in uw) and uw != rw:
                        all_author_docs = []
                        author_display = u_author.title()
                        break

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
    title_specific_doc = None

    # Filter by Title Candidate if user asked for a specific book title of this author
    if title_cand:
        t_words = set(title_cand.split())
        matched_by_title = []
        for d in all_author_docs:
            d_t_norm = normalize_text(d.get('title', '')).strip()
            d_words = set(d_t_norm.split())
            if title_cand in d_t_norm or len(t_words & d_words) / max(len(t_words), 1) >= 0.70:
                matched_by_title.append(d)
        if matched_by_title:
            title_specific_doc = matched_by_title[0]

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
    if title_specific_doc and target_year:
        t_y = 0
        try:
            t_y = int(str(title_specific_doc.get('year', 0)).strip())
        except Exception:
            pass
        t_title = title_specific_doc.get('title', 'Tài liệu')
        t_loc = title_specific_doc.get('location', '')
        t_loc_str = f" tại **{t_loc}**" if t_loc else ""
        y_phrase = f"sau năm {target_year}" if year_dir == "after" else (f"trước năm {target_year}" if year_dir == "before" else f"năm {target_year}")
        if (year_dir == "after" and t_y <= target_year) or (year_dir == "before" and t_y >= target_year) or (year_dir == "exact" and t_y != target_year):
            ans = (
                f"Thư viện QNU hiện có cuốn **'{t_title}'** của tác giả **{author_display}** "
                f"được xuất bản vào năm **{t_y}**{t_loc_str}, tuy nhiên **không có ấn bản nào xuất bản {y_phrase}**.\n\n"
                f"Bạn có thể tham khảo ấn bản năm {t_y} của sách hoặc các tài liệu gần đây của tác giả dưới đây nhé:"
            )
            # Put the specific doc first, then recent docs
            res_sources = [title_specific_doc] + [d for d in filtered_docs if d.get('title') != t_title][:9]
            return {
                "answer": ans,
                "sources": res_sources,
                "total_sources": len(res_sources),
                "total_found": len(res_sources),
                "matched_attr": "author_location",
                "found": True,
                "author_display": author_display,
                "candidate": author_display,
                "doc": title_specific_doc
            }

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
        ans = (
            f"Dạ có nhé! Thư viện QNU hiện có **{len(all_author_docs)} tài liệu** của tác giả **{author_display}**"
            f" (tuy nhiên chưa có tài liệu phân loại riêng là **{target_doc_type}**).\n\n"
            f"Bạn có thể xem chi tiết tài liệu hiện có của tác giả bên dưới:"
        )
        final_docs = all_author_docs[:10]
        filtered_docs = final_docs
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
        "total_found": len(final_docs),
        "matched_attr": "author_location",
        "found": True,
        "author_display": author_display,
        "doc": final_docs[0] if final_docs else None,
        "candidate": author_display
    }
