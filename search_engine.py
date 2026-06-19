"""
search_engine.py — BM25 + Regex Hybrid Full-Text Search Engine
Lightweight alternative to FAISS embedding (no ML model needed)

Hỗ trợ load từ:
  - CSV (clean_BaoCao_DsTaiLieuSo.csv)
  - XLSX (tailieu_templates_AI.xlsx)
"""

import re
import os
import pickle
import pandas as pd
from pathlib import Path
from collections import Counter
from text_utils import normalize_text

DATA_DIR = Path("Data")

class BM25SearchEngine:
    """BM25 ranking algorithm for efficient full-text search"""

    def __init__(self, documents):
        self.documents = documents
        self.k1 = 1.5
        self.b = 0.75
        self.idf_cache = {}
        self._build_index()

    def _build_index(self):
        self.tokenized_docs = []
        self.doc_lengths = []
        self.vocabulary = Counter()

        for doc in self.documents:
            # Ưu tiên flat_text cho BM25 (không có markdown noise)
            text = normalize_text(doc.get('flat_text', '') or doc.get('text', ''))
            text = text + " " + normalize_text(doc.get('title', ''))
            tokens = self._tokenize(text)
            self.tokenized_docs.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.vocabulary.update(tokens)

        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1
        self.num_docs = len(self.documents)
        print(f"✅ BM25 Index built: {self.num_docs} docs, {len(self.vocabulary)} unique terms")

    def _tokenize(self, text):
        tokens = re.findall(r'\w+', text.lower(), re.UNICODE)
        return [t for t in tokens if len(t) > 1]

    def _get_idf(self, term):
        if term in self.idf_cache:
            return self.idf_cache[term]
        doc_freq = sum(1 for tokens in self.tokenized_docs if term in tokens)
        idf = max(0.1, self.num_docs - doc_freq + 0.5) / (doc_freq + 0.5)
        self.idf_cache[term] = idf
        return idf

    def _bm25_score(self, tokens, doc_idx):
        score = 0.0
        doc_tokens = self.tokenized_docs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        for token in tokens:
            if token in doc_tokens:
                term_freq = doc_tokens.count(token)
                idf = self._get_idf(token)
                numerator = idf * term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                score += numerator / denominator

        return score

    def search(self, query, top_k=6, use_regex=True):
        normalized_query = normalize_text(query)
        tokens = self._tokenize(normalized_query)

        if not tokens:
            return []

        scores = []
        # Dynamic minimum score: more tokens → higher bar to filter noise
        min_score = max(0.3, len(tokens) * 0.15) if len(tokens) >= 2 else 0.1

        for doc_idx in range(len(self.documents)):
            bm25_score = self._bm25_score(tokens, doc_idx)

            regex_boost = 0.0
            if use_regex and len(query) > 3:
                doc_text = normalize_text(self.documents[doc_idx].get('text', ''))
                pattern = re.escape(normalized_query)
                if re.search(pattern, doc_text, re.IGNORECASE):
                    regex_boost = 5.0

            # Token coverage: fraction of query tokens found in this doc
            doc_tokens_set = set(self.tokenized_docs[doc_idx])
            matched_count = sum(1 for t in tokens if t in doc_tokens_set)
            coverage = matched_count / len(tokens) if tokens else 0.0

            # Penalize docs with very low token coverage
            if coverage < 0.25 and len(tokens) >= 2:
                continue

            total_score = bm25_score + regex_boost
            if total_score >= min_score:
                scores.append((doc_idx, total_score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, score in scores[:top_k]:
            results.append({
                'doc': self.documents[doc_idx],
                'score': score,
                'index': doc_idx
            })

        return results

    def count_matching(self, query, use_regex=True):
        """Count ALL documents matching query without top_k limit."""
        normalized_query = normalize_text(query)
        tokens = self._tokenize(normalized_query)

        if not tokens:
            return 0

        min_score = max(0.3, len(tokens) * 0.15) if len(tokens) >= 2 else 0.1
        count = 0

        for doc_idx in range(len(self.documents)):
            bm25_score = self._bm25_score(tokens, doc_idx)

            regex_boost = 0.0
            if use_regex and len(query) > 3:
                doc_text = normalize_text(self.documents[doc_idx].get('text', ''))
                pattern = re.escape(normalized_query)
                if re.search(pattern, doc_text, re.IGNORECASE):
                    regex_boost = 5.0

            doc_tokens_set = set(self.tokenized_docs[doc_idx])
            matched_count = sum(1 for t in tokens if t in doc_tokens_set)
            coverage = matched_count / len(tokens) if tokens else 0.0

            if coverage < 0.25 and len(tokens) >= 2:
                continue

            total_score = bm25_score + regex_boost
            if total_score >= min_score:
                count += 1

        return count


class DocumentIndexer:
    """Build and manage document index from CSV and XLSX files"""

    @staticmethod
    def load_from_data(data_dir=None):
        """
        Load documents từ tất cả file dữ liệu trong thư mục Data/

        Args:
            data_dir: đường dẫn đến thư mục Data (mặc định: "Data")

        Returns:
            list of document dicts
        """
        if data_dir is None:
            data_dir = DATA_DIR
        data_path = Path(data_dir)

        # ── Cache check ──────────────────────────────────────
        cache_file = data_path / "_documents_cache.pkl"
        source_files = list(data_path.glob("*.csv")) + list(data_path.glob("*.xlsx"))
        use_cache = False
        if cache_file.exists():
            cache_mtime = cache_file.stat().st_mtime
            running_on_vercel = os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))
            if running_on_vercel or all(f.stat().st_mtime < cache_mtime for f in source_files):
                try:
                    with open(cache_file, "rb") as f:
                        documents = pickle.load(f)
                    print(f"📦 Loaded {len(documents)} documents from cache ({cache_file.name})")
                    return documents
                except Exception as e:
                    print(f"⚠️ Cache load failed: {e}")

        # ── Full load ────────────────────────────────────────
        documents = []

        # 1. Load CSV (clean_BaoCao_DsTaiLieuSo.csv) — DISABLED: chỉ dùng 1 file qnu-allITEM
        # csv_file = data_path / "clean_BaoCao_DsTaiLieuSo.csv"
        # if csv_file.exists():
        #     try:
        #         csv_docs = DocumentIndexer._load_csv(str(csv_file))
        #         documents.extend(csv_docs)
        #         print(f"✅ Loaded {len(csv_docs)} documents from {csv_file.name}")
        #     except Exception as e:
        #         print(f"⚠️ Error loading {csv_file.name}: {e}")
        # else:
        #     print(f"⚠️  File not found: {csv_file}")

        # 2. Load XLSX — chỉ dùng 1 file `qnu-allITEM-*-chuan-hoa.xlsx`
        new_xlsx = None
        for candidate in sorted(data_path.glob("qnu-allITEM-*-chuan-hoa.xlsx")):
            new_xlsx = candidate
            break

        legacy_xlsx = data_path / "tailieu_templates_AI.xlsx"

        xlsx_docs = []
        xlsx_source_name = None

        if new_xlsx is not None:
            try:
                xlsx_docs = DocumentIndexer._load_xlsx_marc(str(new_xlsx))
                documents.extend(xlsx_docs)
                xlsx_source_name = new_xlsx.name
                print(f"✅ Loaded {len(xlsx_docs)} documents from {new_xlsx.name} (MARC structure, markdown text)")
            except Exception as e:
                print(f"⚠️ Error loading {new_xlsx.name}: {e}")
        else:
            print(f"⚠️  No qnu-allITEM-*-chuan-hoa.xlsx found in {data_path}")

        # 3. Cross-reference: gán link từ CSV vào XLSX dựa trên tên sách (DISABLED khi không dùng CSV)
        # csv_link_map = {}
        # for doc in documents:
        #     if 'csv' in doc.get('source', '') and doc.get('link'):
        #         norm = normalize_text(doc['title'])
        #         if norm and len(norm) > 5:
        #             csv_link_map[norm] = doc['link']

        link_count = 0
        for doc in documents:
            if 'xlsx' in doc.get('source', '') and not doc.get('link'):
                norm = normalize_text(doc['title'])
                if not norm or len(norm) <= 5:
                    continue
                # if norm in csv_link_map:
                #     doc['link'] = csv_link_map[norm]
                #     link_count += 1
                # else:
                #     for csv_norm, csv_link in csv_link_map.items():
                #         if len(csv_norm) > 10 and (csv_norm in norm or norm in csv_norm):
                #             doc['link'] = csv_link
                #             link_count += 1
                #             break
                # Fallback: nếu XLSX row đã có 856$u thì doc['link'] đã được set
                pass

        if link_count:
            print(f"🔗 Cross-referenced {link_count} links from CSV to XLSX documents")

        # 4. Save cache for next startup
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"💾 Cached {len(documents)} documents to {cache_file.name}")
        except Exception as e:
            print(f"⚠️ Cache write failed: {e}")

        print(f"📦 Total documents loaded: {len(documents)}")
        return documents

    @staticmethod
    def _load_csv(filepath):
        """Load documents from CSV file"""
        documents = []

        df = pd.read_csv(filepath, encoding='utf-8', delimiter=';', skiprows=1)
        df = df.rename(columns={
            'Tiêu đề': 'title',
            'Tác giả': 'author',
            'Năm xuất bản': 'year',
            'Nhà xuất bản': 'publisher',
            'Chủ đề': 'subject',
            'Chuyên ngành': 'major',
            'Học phần': 'course',
            'Loại tài liệu': 'doc_type',
            'Kiểu tài liệu': 'format',
            'Ngôn ngữ': 'language',
            'Link URL': 'link',
            'Ghi chú': 'notes',
            'Tác giả phụ': 'co_author',
        })

        for idx, row in df.iterrows():
            title = str(row.get('title', 'Untitled')).strip() if pd.notna(row.get('title')) else 'Untitled'
            author = str(row.get('author', 'Unknown')).strip() if pd.notna(row.get('author')) else 'Unknown'
            year = str(row.get('year', 'N/A')).strip() if pd.notna(row.get('year')) else 'N/A'
            subject = str(row.get('subject', 'N/A')).strip() if pd.notna(row.get('subject')) else 'N/A'
            link = str(row.get('link', '')).strip() if pd.notna(row.get('link')) else ''
            doc_type = str(row.get('doc_type', 'N/A')).strip() if pd.notna(row.get('doc_type')) else 'N/A'
            doc_format = str(row.get('format', 'Số')).strip() if pd.notna(row.get('format')) else 'Số'
            notes = str(row.get('notes', '')).strip() if pd.notna(row.get('notes')) else ''
            publisher = str(row.get('publisher', '')).strip() if pd.notna(row.get('publisher')) else ''
            major = str(row.get('major', '')).strip() if pd.notna(row.get('major')) else ''
            course = str(row.get('course', '')).strip() if pd.notna(row.get('course')) else ''
            language = str(row.get('language', '')).strip() if pd.notna(row.get('language')) else ''

            # Build searchable text
            text_parts = [title, author, subject, doc_type, publisher, major, course, notes]
            text = ' '.join(p for p in text_parts if p)

            document = {
                'title': title,
                'author': author,
                'year': year,
                'subject': subject,
                'link': link,
                'doc_type': doc_type,
                'format': doc_format,
                'publisher': publisher,
                'major': major,
                'course': course,
                'language': language,
                'notes': notes,
                'source': filepath,
                'csv_file': str(filepath),
                'text': text,
            }
            documents.append(document)

        return documents

    @staticmethod
    def _clean_marc_value(val):
        """Remove MARC prefix markers like $a, $b, $c from values"""
        if not val or not isinstance(val, str):
            return val
        # Remove MARC subfield codes: $a, $b, $c, etc.
        cleaned = re.sub(r'\$[a-z]\s*', '', val)
        # Remove hanging punctuation at start/end
        cleaned = cleaned.strip().strip(',').strip(';').strip('/').strip(':').strip()
        return cleaned

    @staticmethod
    def _load_xlsx(filepath):
        """Load documents from XLSX file (tailieu_templates_AI.xlsx) với cấu trúc MARC"""
        documents = []

        df = pd.read_excel(filepath, dtype=str)

        # Map MARC-like columns to readable fields
        column_map = {
            'Chỉ số phân loai DDC(082$a)': 'ddc',
            'Nhan đề chính(245$a)': 'title',
            'Nhan đề khác(245$b)': 'subtitle',
            'Trách nhiệm(245$c)': 'author',
            'Tên nhà xuất bản (260$b)': 'publisher',
            'Năm xuất bản (260$c)': 'year',
            'Tóm tắt (520$a)': 'abstract',
            'Chuyên ngành đào tạo(526$a)': 'major',
            'Loại tài liệu(526$b)': 'doc_type',
        }

        df_renamed = df.rename(columns=column_map)
        cleaner = DocumentIndexer._clean_marc_value

        for idx, row in df_renamed.iterrows():
            title = cleaner(str(row.get('title', ''))) if pd.notna(row.get('title')) else ''
            subtitle = cleaner(str(row.get('subtitle', ''))) if pd.notna(row.get('subtitle')) else ''
            author = cleaner(str(row.get('author', 'Unknown'))) if pd.notna(row.get('author')) else 'Unknown'
            year = cleaner(str(row.get('year', 'N/A'))) if pd.notna(row.get('year')) else 'N/A'
            year = re.sub(r'\D', '', year)  # Extract digits only
            publisher = cleaner(str(row.get('publisher', ''))) if pd.notna(row.get('publisher')) else ''
            abstract = cleaner(str(row.get('abstract', ''))) if pd.notna(row.get('abstract')) else ''
            major = cleaner(str(row.get('major', ''))) if pd.notna(row.get('major')) else ''
            doc_type = cleaner(str(row.get('doc_type', 'Sách'))) if pd.notna(row.get('doc_type')) else 'Sách'
            ddc = cleaner(str(row.get('ddc', ''))) if pd.notna(row.get('ddc')) else ''

            if not title and not author:
                continue

            # Build full title: avoid double punctuation
            if subtitle:
                subtitle_clean = subtitle.lstrip(': ').lstrip(':').strip()
                if title.endswith(':') or title.endswith(':'):
                    full_title = f"{title} {subtitle_clean}"
                else:
                    full_title = f"{title}: {subtitle_clean}"
            else:
                full_title = title

            # Collect location information from location columns
            location_parts = []
            location_cols = [c for c in df.columns if 'phòng' in c.lower() or 'mượn' in c.lower() or 'đọc' in c.lower() or 'thiếu' in c.lower() or 'thừa' in c.lower()]
            for col in location_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    clean_val = str(val).strip().lstrip('|').strip()
                    if clean_val not in ('nan', ''):
                        short_col = col.replace('(526$b)', '').replace('(082$a)', '').strip()
                        # Remove duplicate room name prefix from value
                        if clean_val.lower().startswith(short_col.lower()):
                            clean_val = clean_val[len(short_col):].lstrip(': ').lstrip('|').strip()
                        location_parts.append(f"{short_col}: {clean_val[:60]}")

            location_str = ' ; '.join(location_parts) if location_parts else ''

            # Build searchable text
            text_parts = [full_title, author, publisher, major, abstract, doc_type]
            text = ' '.join(p for p in text_parts if p and p not in ('Unknown', 'N/A'))
            if abstract:
                text = f"{full_title} {author} {abstract}"

            document = {
                'title': full_title,
                'author': author,
                'year': year if year else 'N/A',
                'subject': major if major else publisher,
                'link': '',
                'doc_type': doc_type,
                'format': 'Sách',
                'publisher': publisher,
                'major': major,
                'ddc': ddc,
                'abstract': abstract,
                'location': location_str,
                'source': str(filepath),
                'csv_file': str(filepath),
                'text': text,
            }
            documents.append(document)

        return documents

    @staticmethod
    def _parse_marc_subfields(value):
        """Split a MARC value like '$aFoo :$bBar /$cBaz' into {'a': 'Foo :', 'b': 'Bar /', 'c': 'Baz'}."""
        if not isinstance(value, str) or not value.strip():
            return {}
        parts = re.split(r'(\$[a-z])', value)
        result = {}
        current_key = None
        for p in parts:
            if not p:
                continue
            if re.match(r'^\$[a-z]$', p):
                current_key = p[1]
                result.setdefault(current_key, '')
            elif current_key:
                result[current_key] = (result[current_key] + ' ' + p).strip()
        for k in result:
            result[k] = re.sub(r'\s+', ' ', result[k]).strip(' ,;/:').strip()
        return result

    @staticmethod
    def _doc_to_markdown(title, bilingual, subtitle, edition, author, co_authors,
                         place, publisher, year, major, doc_type, course, abstract,
                         keywords, locations):
        """Convert parsed MARC fields into a single markdown block (used as searchable text).

        245 (title), 246 (bilingual), 250 (edition) are grouped on the same heading line.
        260 (publication), 291 (course), 526 (major + doc_type), 650 (keywords), 700 (author)
        are kept as separate fields.
        """
        # ── Line 1: tiêu đề | tên song ngữ (nếu có) | lần xuất bản (nếu có) ──
        heading_parts = []
        if title:
            heading_parts.append(f"## {title}")
        if bilingual:
            heading_parts.append(f"*Song ngữ: {bilingual}*")
        if edition:
            heading_parts.append(f"*Lần XB: {edition}*")
        if heading_parts:
            lines = [" | ".join(heading_parts), ""]
        else:
            lines = [""]
        if author:
            lines.append(f"**Tác giả:** {author}")
        if co_authors:
            lines.append(f"**Đồng tác giả:** {', '.join(co_authors)}")
        if publisher:
            lines.append(f"**Nhà xuất bản:** {publisher}")
        if place:
            lines.append(f"**Nơi xuất bản:** {place}")
        if year:
            lines.append(f"**Năm xuất bản:** {year}")
        if doc_type:
            lines.append(f"**Loại tài liệu:** {doc_type}")
        if major:
            lines.append(f"**Chuyên ngành:** {major}")
        if course:
            lines.append(f"**Học phần:** {course}")
        if keywords:
            lines.append(f"**Từ khóa:** {', '.join(keywords)}")
        if locations:
            lines.append("")
            lines.append("### Vị trí & Mã kho")
            for loc in locations:
                lines.append(f"- {loc}")
        if abstract:
            lines.append("")
            lines.append("### Tóm tắt")
            lines.append(abstract)
        return "\n".join(lines).strip()

    @staticmethod
    def _load_xlsx_marc(filepath):
        """
        Load documents from new XLSX file `qnu-allITEM-*-chuan-hoa.xlsx` (MARC structure).
        Mỗi record → dict có title, author, year, publisher, place, edition, bilingual_title,
        subtitle, major, doc_type, course, keywords, locations, link, text (markdown).
        """
        documents = []
        df = pd.read_excel(filepath, header=0, dtype=str)

        tag_cols = {str(c).strip(): c for c in df.columns if str(c).strip().isdigit()}
        location_cols = [c for c in df.columns
                         if not str(c).strip().isdigit() and str(c) != 'ItemId']

        def _clean_str(v):
            if pd.isna(v):
                return ''
            return re.sub(r'\s+', ' ', str(v)).strip()

        def _split_list(v, sep=','):
            if not v:
                return []
            return [s.strip() for s in re.split(r'[,;|]', v) if s.strip()]

        for idx, row in df.iterrows():
            try:
                f100 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('100'))))
                f245 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('245'))))
                f246 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('246'))))
                f250 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('250'))))
                f260 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('260'))))
                f291 = _clean_str(row.get(tag_cols.get('291')))
                f520 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('520'))))
                f526 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('526'))))
                f650 = _clean_str(row.get(tag_cols.get('650')))
                f700 = _clean_str(row.get(tag_cols.get('700')))
                f856 = DocumentIndexer._parse_marc_subfields(_clean_str(row.get(tag_cols.get('856'))))

                title = f245.get('a', '').strip(' :/')
                subtitle = f245.get('b', '').strip(' :/')
                bilingual = f246.get('a', '').strip(' :/')
                edition = f250.get('a', '').strip(' :/')
                author_245c = f245.get('c', '').strip(' :/')
                place = f260.get('a', '').strip(' :/')
                publisher = f260.get('b', '').strip(' ,/')
                year = re.sub(r'\D', '', f260.get('c', ''))
                major = f526.get('a', '').strip(' :/')
                doc_type = f526.get('b', '').strip(' :/')
                course = re.sub(r'^\$[a-z]\s*', '', f291).strip() if f291 else ''
                abstract = f520.get('a', '').strip(' :/')
                link = f856.get('u', '').strip()

                # Authors: 100$a + 700$a + 245$c. Strip MARC prefix and role qualifiers.
                # Each source may contain a single author OR multiple separated by ";" or "/" or newlines
                f700_parsed = DocumentIndexer._parse_marc_subfields(f700)
                author_candidates = []
                for raw in [f100.get('a', ''), f700_parsed.get('a', ''), author_245c]:
                    if not raw:
                        continue
                    # Authors are typically separated by ";" or "/" (but keep comma inside "Last, First")
                    parts = re.split(r'[;/]|\sand\s', raw)
                    for a in parts:
                        a_clean = re.sub(r'^\$[a-z]\s*', '', a).strip(' :,').strip()
                        # Strip role qualifiers in parens like (b.s.), (ch.b.), (bìa sách)
                        a_clean = re.sub(r'\s*\([^)]{1,20}\)\s*$', '', a_clean).strip()
                        # Strip trailing ", Jr." or "Jr"
                        a_clean = re.sub(r',?\s*Jr\.?\s*$', '', a_clean, flags=re.IGNORECASE).strip()
                        if a_clean and len(a_clean) > 1 and a_clean not in author_candidates:
                            author_candidates.append(a_clean)
                # Prefer full "Last, First" form; avoid "First" only or "Last" only
                author = ''
                co_authors = []
                for cand in author_candidates:
                    if ',' in cand and len(cand.split(',')) == 2:
                        # Standard "Last, First" form
                        if not author:
                            author = cand
                        else:
                            co_authors.append(cand)
                    elif not author:
                        # First non-empty becomes primary if no comma form
                        author = cand
                    else:
                        if cand not in co_authors:
                            co_authors.append(cand)

                keywords = []
                for kw in _split_list(f650, sep=';'):
                    kw_clean = re.sub(r'^\$[a-z]\s*', '', kw).strip()
                    if kw_clean and kw_clean.lower() != 'nan' and kw_clean not in keywords:
                        keywords.append(kw_clean)

                locations = []
                for col in location_cols:
                    val = _clean_str(row.get(col))
                    if val and val.lower() != 'nan':
                        acc_numbers = _split_list(val)
                        if acc_numbers:
                            locations.append({
                                'room': str(col).strip(),
                                'accession_numbers': acc_numbers,
                                'count': len(acc_numbers),
                            })

                if not title and not author:
                    continue

                loc_strings = []
                for loc in locations:
                    accs = ', '.join(loc['accession_numbers'][:5])
                    if len(loc['accession_numbers']) > 5:
                        accs += f" ... (+{len(loc['accession_numbers'])-5})"
                    loc_strings.append(f"**{loc['room']}**: {accs} (tổng: {loc['count']})")

                text = DocumentIndexer._doc_to_markdown(
                    title=title,
                    bilingual=bilingual,
                    subtitle=subtitle,
                    edition=edition,
                    author=author,
                    co_authors=co_authors,
                    place=place,
                    publisher=publisher,
                    year=year,
                    major=major,
                    doc_type=doc_type,
                    course=course,
                    abstract=abstract,
                    keywords=keywords,
                    locations=loc_strings,
                )

                flat_parts = [title, subtitle, author, ' '.join(co_authors), publisher, place, year,
                              edition, major, doc_type, course, ' '.join(keywords)]
                flat_text = ' '.join(p for p in flat_parts if p)

                location_display = ' ; '.join(
                    f"{loc['room']}: {', '.join(loc['accession_numbers'][:3])}"
                    f"{' ...' if loc['count'] > 3 else ''} (SL: {loc['count']})"
                    for loc in locations
                )

                document = {
                    'title': title,
                    'author': author,
                    'co_authors': co_authors,
                    'authors': [author] + co_authors if author else co_authors,
                    'year': year if year else 'N/A',
                    'subject': (keywords[0] if keywords else ''),
                    'link': link,
                    'doc_type': doc_type if doc_type else 'Sách',
                    'format': 'Sách',
                    'publisher': publisher,
                    'place': place,
                    'edition': edition,
                    'major': major,
                    'bilingual_title': bilingual,
                    'subtitle': subtitle,
                    'course': course,
                    'keywords': keywords,
                    'abstract': abstract,
                    'locations': locations,
                    'location': location_display,
                    'source': str(filepath),
                    'csv_file': str(filepath),
                    'text': text,
                    'flat_text': flat_text,
                }
                documents.append(document)
            except Exception as e:
                if idx < 5:
                    print(f"⚠️ Error parsing row {idx}: {e}")
                continue

        return documents


class HybridSearchEngine:
    """Wrapper combining BM25 + regex + filtering"""

    def __init__(self, bm25_engine):
        self.bm25 = bm25_engine

    def search_by_query(self, query, top_k=6):
        return self.bm25.search(query, top_k=top_k, use_regex=True)

    def count_matching(self, query):
        """Count ALL documents matching query (no top_k limit)."""
        return self.bm25.count_matching(query, use_regex=True)

    def search_by_author(self, author, query, top_k=5):
        results = self.bm25.search(query, top_k=top_k*2)
        author_results = [r for r in results if author.lower() in r['doc'].get('author', '').lower()]
        return author_results[:top_k]

    def search_by_subject(self, subject, query, top_k=5):
        results = self.bm25.search(query, top_k=top_k*2)
        subject_results = [r for r in results if subject.lower() in r['doc'].get('subject', '').lower()]
        return subject_results[:top_k]

    def get_related_documents(self, title, author=None, top_k=5):
        query = f"{title} {author if author else ''}"
        results = self.bm25.search(query, top_k=top_k*3)
        related = [r for r in results if r['doc']['title'].lower() != title.lower()]
        return related[:top_k]

    def search_with_filters(self, query, filters=None, top_k=6):
        results = self.bm25.search(query, top_k=top_k*2)

        if filters:
            try:
                import json as _json
                _tree_path = Path("Data") / "_khoa_nganh_tree.json"
                with open(_tree_path, "r", encoding="utf-8") as _f:
                    _tree = _json.load(_f) or []
            except Exception:
                _tree = []
            _all_canonicals: set[str] = set()
            for _fac in (_tree if isinstance(_tree, list) else []):
                for _ng in (_fac.get("nganh") or []):
                    _cn = normalize_text(_ng.get("nganh", ""))
                    if _cn:
                        _all_canonicals.add(_cn)
            _ACCEPT_PREFIXES = (
                "thac si ", "thac si ngành ", "ngành ",
                "bo mon ", "cong nghe ", "cong nghe ky thuat ",
            )

            def _major_matches_filter(fmajor: str, doc_major: str) -> bool:
                if not doc_major:
                    return False
                if fmajor == doc_major:
                    return True
                f_tokens = set(fmajor.split())
                candidates = {fmajor}
                for cn in _all_canonicals:
                    if cn == fmajor:
                        candidates.add(cn)
                for c in candidates:
                    if c and c in doc_major:
                        return True
                    if c and doc_major.startswith(c + " "):
                        return True
                    if c and doc_major.startswith(c + ","):
                        return True
                # Prefix variations
                for c in candidates:
                    for pfx in _ACCEPT_PREFIXES:
                        if doc_major == (pfx.rstrip() + " " + c).strip():
                            return True
                return False

            for result in results[:]:
                doc = result['doc']

                if 'author' in filters and filters['author'].lower() not in doc.get('author', '').lower():
                    results.remove(result)
                    continue

                if 'subject' in filters and filters['subject'].lower() not in doc.get('subject', '').lower():
                    results.remove(result)
                    continue

                if 'year' in filters:
                    doc_year = str(doc.get('year', ''))
                    if str(filters['year']) not in doc_year:
                        results.remove(result)
                        continue

                if 'major' in filters and filters['major']:
                    fmajor = normalize_text(filters['major']).strip()
                    doc_major = normalize_text(str(doc.get('major', '')))
                    if not _major_matches_filter(fmajor, doc_major):
                        results.remove(result)
                        continue

        return results[:top_k]
