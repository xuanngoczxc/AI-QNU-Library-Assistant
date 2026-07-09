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
from .text_utils import normalize_text

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

    def _tokens_in_proximity(self, doc_tokens, query_tokens, max_gap=6):
        """Kiểm tra các query_tokens có xuất hiện đúng thứ tự và gần nhau.
        
        Args:
            doc_tokens: list token của document
            query_tokens: list token cần tìm (đã tokenize)
            max_gap: khoảng cách tối đa giữa token đầu và token cuối
        
        Returns:
            True nếu tất cả query_tokens xuất hiện đúng thứ tự trong max_gap
        """
        # Tìm vị trí đầu tiên của query_tokens[0]
        positions = []
        start = 0
        for qt in query_tokens:
            for i in range(start, len(doc_tokens)):
                if doc_tokens[i] == qt:
                    positions.append(i)
                    start = i + 1
                    break
            else:
                return False
        
        # Kiểm tra gap giữa vị trí đầu và cuối
        span = positions[-1] - positions[0]
        return span <= max_gap

    def search(self, query, top_k=6, use_regex=True):
        normalized_query = normalize_text(query)
        tokens = self._tokenize(normalized_query)

        if not tokens:
            return []

        # Nếu có token 1 ký tự bị drop (vd "b" trong "lê văn b")
        # → yêu cầu exact match với word boundary
        raw_tokens = re.findall(r'\w+', normalized_query, re.UNICODE)
        dropped = [t for t in raw_tokens if len(t) <= 1]
        if dropped:
            pattern = re.compile(r'\b' + re.escape(normalized_query) + r'\b')
            if not any(pattern.search(normalize_text(d.get('text', ''))) for d in self.documents):
                return []

        # Dynamic minimum score: stricter for short queries
        min_score = max(1.0, len(tokens) * 0.4) if len(tokens) >= 3 else 1.0
        min_coverage = 1.0 if len(tokens) <= 3 else 0.6

        scores = []
        for doc_idx in range(len(self.documents)):
            # Token coverage: every token must appear for short queries
            doc_tokens_set = set(self.tokenized_docs[doc_idx])
            matched_count = sum(1 for t in tokens if t in doc_tokens_set)
            coverage = matched_count / len(tokens) if tokens else 0.0

            if coverage < min_coverage:
                continue

            # Các token phải xuất hiện đúng thứ tự và gần nhau (cách tối đa 3 từ)
            if len(tokens) >= 2:
                if not self._tokens_in_proximity(self.tokenized_docs[doc_idx], tokens, max_gap=3):
                    continue

            bm25_score = self._bm25_score(tokens, doc_idx)

            regex_boost = 0.0
            if use_regex and len(query) > 3:
                doc_text = normalize_text(self.documents[doc_idx].get('text', ''))
                if re.search(re.escape(normalized_query), doc_text, re.IGNORECASE):
                    regex_boost = 5.0

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

        # Nếu có token bị drop (1 ký tự) → yêu cầu exact match word boundary
        raw_tokens = re.findall(r'\w+', normalized_query, re.UNICODE)
        dropped = [t for t in raw_tokens if len(t) <= 1]
        if dropped:
            pattern = re.compile(r'\b' + re.escape(normalized_query) + r'\b')
            if not any(pattern.search(normalize_text(d.get('text', ''))) for d in self.documents):
                return 0

        # Stricter for short queries: raise min score
        min_score = max(0.8, len(tokens) * 0.35) if len(tokens) >= 2 else 0.8

        # Require ALL tokens appear for short queries
        min_coverage = 1.0 if len(tokens) <= 3 else 0.6

        count = 0
        for doc_idx in range(len(self.documents)):
            doc_tokens_set = set(self.tokenized_docs[doc_idx])
            matched_count = sum(1 for t in tokens if t in doc_tokens_set)
            coverage = matched_count / len(tokens) if tokens else 0.0

            if coverage < min_coverage:
                continue

            # Các token phải xuất hiện đúng thứ tự và gần nhau (cách tối đa 3 từ)
            if len(tokens) >= 2:
                if not self._tokens_in_proximity(self.tokenized_docs[doc_idx], tokens, max_gap=3):
                    continue

            bm25_score = self._bm25_score(tokens, doc_idx)
            if bm25_score >= min_score:
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

        # 2. Load XLSX files — ưu tiên upload-* nếu có (bản đầy đủ hơn)
        #    File gốc qnu-allITEM-* vẫn giữ trong thư mục để backup
        upload_files = sorted(data_path.glob("upload-*.xlsx"))
        main_files = sorted(data_path.glob("qnu-allITEM-*-chuan-hoa.xlsx"))
        if upload_files:
            xlsx_files = upload_files  # chỉ load file upload
            skip_main = True
        else:
            xlsx_files = main_files
            skip_main = False
        xlsx_loaded = 0
        for xlsx_f in xlsx_files:
            try:
                fmt = DocumentIndexer._detect_xlsx_format(str(xlsx_f))
                if fmt == "marc":
                    xlsx_docs = DocumentIndexer._load_xlsx_marc(str(xlsx_f))
                    print(f"✅ Loaded {len(xlsx_docs)} documents from {xlsx_f.name} (MARC structure)")
                else:
                    xlsx_docs = DocumentIndexer._load_xlsx_generic(str(xlsx_f))
                    print(f"✅ Loaded {len(xlsx_docs)} documents from {xlsx_f.name} (generic columns)")
                documents.extend(xlsx_docs)
                xlsx_loaded += 1
            except Exception as e:
                print(f"⚠️ Error loading {xlsx_f.name}: {e}")
        if skip_main and main_files:
            print(f"   📌 File gốc {main_files[0].name} được giữ làm backup (bỏ qua load)")
        if xlsx_loaded == 0:
            print(f"⚠️  No XLSX files found in {data_path}")

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

        # ── Dedup: chỉ chạy khi có NHIỀU file xlsx ──────────
        # 1 file = không thể trùng nội bộ hợp lệ (cùng title nhưng khác edition/author)
        # Nhiều file = có thể trùng do upload file chồng lên nhau
        if xlsx_loaded > 1:
            documents = DocumentIndexer._dedup_documents(documents)
        else:
            print(f"📦 Single file loaded, skipping dedup")

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
    def _normalize_author(name):
        """Chuẩn hóa author: đảo 'Last, First' → 'first last' để so khớp."""
        if not name:
            return ''
        name = name.strip().lower()
        name = re.sub(r'\s*\([^)]{0,30}\)\s*', '', name).strip()
        name = re.sub(r',?\s*jr\.?\s*$', '', name, flags=re.IGNORECASE).strip()
        if ',' in name:
            parts = [p.strip() for p in name.split(',', 1)]
            if len(parts) == 2 and parts[1]:
                name = f"{parts[1]} {parts[0]}"
        return re.sub(r'\s+', ' ', name).strip()

    @staticmethod
    def _dedup_documents(documents):
        """Remove duplicate documents (2-pass với chuẩn hóa author).

        Pass 1: exact match title+author+year (author đã chuẩn hóa).
        Pass 2: title+year match, áp dụng khi 1 trong 2 author bị thiếu
                hoặc author khác nhau do cách parse MARC (Last, First vs First Last).
        """
        if not documents:
            return documents

        total_before = len(documents)

        # ── Pass 1: title+author+year (author chuẩn hóa) ─────
        seen = {}
        dedup_count = 0
        for i, doc in enumerate(documents):
            title_norm = normalize_text(doc.get('title', ''))
            author_norm = DocumentIndexer._normalize_author(doc.get('author', ''))
            year = str(doc.get('year', '')).strip()
            if not title_norm:
                continue
            key = f"{title_norm}|{author_norm}|{year}"
            if key in seen:
                old_doc = documents[seen[key]]
                new_score = sum(1 for v in doc.values() if v and str(v).strip() and str(v) != 'N/A')
                old_score = sum(1 for v in old_doc.values() if v and str(v).strip() and str(v) != 'N/A')
                if new_score >= old_score:
                    documents[seen[key]] = None
                    seen[key] = i
                else:
                    documents[i] = None
                dedup_count += 1
            else:
                seen[key] = i

        # ── Pass 2: title+year (fallback an toàn) ────────────
        # Chỉ áp dụng khi 1 trong 2 bị thiếu author
        surviving = [d for d in documents if d is not None]
        seen_titledoc = {}
        for i, doc in enumerate(surviving):
            title_norm = normalize_text(doc.get('title', ''))
            year = str(doc.get('year', '')).strip()
            if not title_norm:
                continue
            key = f"{title_norm}|{year}"
            if key in seen_titledoc:
                old_doc = surviving[seen_titledoc[key]]
                old_author = DocumentIndexer._normalize_author(old_doc.get('author', ''))
                new_author = DocumentIndexer._normalize_author(doc.get('author', ''))
                old_missing = not old_author or old_author in ('unknown', 'n/a', '')
                new_missing = not new_author or new_author in ('unknown', 'n/a', '')
                if old_missing or new_missing:
                    new_score = sum(1 for v in doc.values() if v and str(v).strip() and str(v) != 'N/A')
                    old_score = sum(1 for v in old_doc.values() if v and str(v).strip() and str(v) != 'N/A')
                    if new_score >= old_score:
                        surviving[seen_titledoc[key]] = None
                        seen_titledoc[key] = i
                    else:
                        surviving[i] = None
            else:
                seen_titledoc[key] = i

        surviving = [d for d in surviving if d is not None]
        total_removed = total_before - len(surviving)
        if total_removed:
            print(f"🔄 Dedup: removed {total_removed} duplicates → {len(surviving)} unique docs")
        return surviving

    @staticmethod
    def _load_csv(filepath):
        # Load documents from CSV file (legacy, kept for reference)
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
        # Remove MARC prefix markers (e.g. $a, $2, $b, $c) from values.
        # Replaces subfield markers with ', ' for clear separation.
        if not val or not isinstance(val, str):
            return val
        cleaned = re.sub(r'\$[a-zA-Z0-9]\s*', ', ', val)
        cleaned = re.sub(r'^\s*,\s*', '', cleaned)
        cleaned = re.sub(r',\s*,', ',', cleaned)
        cleaned = cleaned.strip().strip(',').strip(';').strip('/').strip(':').strip()
        return cleaned

    @staticmethod
    def _load_xlsx(filepath):
        # Load documents from XLSX file (tailieu_templates_AI.xlsx) with MARC structure
        documents = []

        df = pd.read_excel(filepath, dtype=str)

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

            if subtitle:
                subtitle_clean = subtitle.lstrip(': ').lstrip(':').strip()
                if title.endswith(':') or title.endswith(':'):
                    full_title = f"{title} {subtitle_clean}"
                else:
                    full_title = f"{title}: {subtitle_clean}"
            else:
                full_title = title

            location_parts = []
            location_cols = [c for c in df.columns if 'phòng' in c.lower() or 'mượn' in c.lower() or 'đọc' in c.lower() or 'thiếu' in c.lower() or 'thừa' in c.lower()]
            for col in location_cols:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    clean_val = str(val).strip().lstrip('|').strip()
                    if clean_val not in ('nan', ''):
                        short_col = col.replace('(526$b)', '').replace('(082$a)', '').strip()
                        if clean_val.lower().startswith(short_col.lower()):
                            clean_val = clean_val[len(short_col):].lstrip(': ').lstrip('|').strip()
                        location_parts.append(f"{short_col}: {clean_val[:500]}")

            location_str = ' ; '.join(location_parts) if location_parts else ''

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
    def _detect_xlsx_format(filepath):
        """Auto-detect whether an XLSX file uses MARC tag columns (100, 245, 260...)
        or generic Vietnamese/English column names (Tiêu đề, Tác giả, Title...).

        Returns: 'marc' or 'generic'
        """
        try:
            df = pd.read_excel(filepath, header=0, nrows=3, dtype=str)
        except Exception:
            return "generic"

        cols = [str(c).strip() for c in df.columns]
        # MARC format: has numeric tag columns like 100, 245, 260, 520, 650, 856
        marc_tags = {"100", "245", "246", "250", "260", "261", "291", "520", "526", "650", "700", "856"}
        numeric_cols = {c for c in cols if c.isdigit()}
        if numeric_cols & marc_tags:
            return "marc"
        return "generic"

    @staticmethod
    def _load_xlsx_generic(filepath):
        # Load documents from a generic XLSX file with common Vietnamese/English columns.
        #
        # Supported column name mappings (case-insensitive, partial match):
        #   title:     Tieu de, Title, Ten tai lieu
        #   author:    Tac gia, Author, Tac gia chinh
        #   year:      Nam, Year, Nam xuat ban
        #   publisher: Nha xuat ban, Publisher, NXB
        #   subject:   Chu de, Subject, Tu khoa, Keywords
        #   major:     Chuyen nganh, Major, Nganh
        #   doc_type:  Loai tai lieu, Doc type, The loai
        #   link:      Link, URL, Duong dan
        #   abstract:  Tom tat, Abstract, Mo ta
        #   course:    Hoc phan, Course
        #   doc_id:    Ma, ID, Ma tai lieu
        documents = []
        try:
            df = pd.read_excel(filepath, header=0, dtype=str)
        except Exception as e:
            print(f"⚠️ Cannot read {filepath}: {e}")
            return documents

        if df.empty:
            return documents

        # ── Build column mapping (fuzzy match) ──────────────────
        col_map = {}  # canonical_name → actual column name
        _CANONICAL = {
            "title":     ["tiêu đề", "tieu de", "title", "tên tài liệu", "ten tai lieu", "tên", "ten", "sách", "sach"],
            "author":    ["tác giả", "tac gia", "author", "tác giả chính", "tac gia chinh", "tác giả phụ"],
            "year":      ["năm", "nam", "year", "năm xuất bản", "nam xuat ban", "năm xb"],
            "publisher": ["nhà xuất bản", "nha xuat ban", "publisher", "nxb", "nhà xb"],
            "subject":   ["chủ đề", "chu de", "subject", "từ khóa", "tu khoa", "keywords"],
            "major":     ["chuyên ngành", "chuyen nganh", "major", "ngành", "nganh"],
            "doc_type":  ["loại tài liệu", "loai tai lieu", "doc type", "thể loại", "the loai", "kiểu"],
            "link":      ["link", "url", "đường dẫn", "duong dan", "đường link", "liên kết"],
            "abstract":  ["tóm tắt", "tom tat", "abstract", "mô tả", "mo ta", "nội dung"],
            "course":    ["học phần", "hoc phan", "course"],
            "doc_id":    ["mã", "ma", "id", "mã tài liệu", "ma tai lieu", "item id", "itemid"],
            "language":  ["ngôn ngữ", "ngon ngu", "language"],
            "notes":     ["ghi chú", "ghi chu", "notes", "chú thích"]
        }

        cols_lower = {str(c).strip().lower(): c for c in df.columns}
        for canonical, keywords in _CANONICAL.items():
            for kw in keywords:
                for col_lower, col_orig in cols_lower.items():
                    if kw in col_lower or col_lower in kw:
                        col_map[canonical] = col_orig
                        break
                if canonical in col_map:
                    break

        # Must have at least title or author
        if "title" not in col_map and "author" not in col_map:
            # Try first 2 non-empty columns as title fallback
            for c in df.columns:
                sample = df[c].dropna().head(5).astype(str).tolist()
                avg_len = sum(len(s) for s in sample) / max(len(sample), 1)
                if avg_len > 5 and "title" not in col_map:
                    col_map["title"] = c
                    break

        if "title" not in col_map and "author" not in col_map:
            print(f"⚠️ Cannot identify title/author columns in {Path(filepath).name}")
            return documents

        def _cell(row, key, default=""):
            if key not in col_map:
                return default
            val = row.get(col_map[key])
            if pd.isna(val):
                return default
            return re.sub(r"\s+", " ", str(val)).strip()

        def _cell_raw(row, col):
            val = row.get(col)
            if pd.isna(val):
                return ''
            return re.sub(r"\s+", " ", str(val)).strip()

        mapped_names = {k: col_map[k] for k in col_map}
        print(f"   📋 Column mapping: {mapped_names}")

        for idx, row in df.iterrows():
            try:
                title = _cell(row, "title")
                author = _cell(row, "author")
                if not title and not author:
                    continue

                year = _cell(row, "year")
                year = re.sub(r"\D", "", year) if year else ""
                publisher = _cell(row, "publisher")
                subject = _cell(row, "subject")
                major = _cell(row, "major")
                doc_type = _cell(row, "doc_type") or "Sách"
                link = _cell(row, "link")
                abstract = _cell(row, "abstract")
                course = _cell(row, "course")
                language = _cell(row, "language")
                notes = _cell(row, "notes")
                doc_id = _cell(row, "doc_id")

                # Split multi-value fields
                keywords = [k.strip() for k in re.split(r"[,;|]", subject) if k.strip()] if subject else []
                authors_list = [a.strip() for a in re.split(r"[/;]", author) if a.strip()] if author else []

                flat_parts = [title, author, publisher, major, year, doc_type, course, subject, abstract, language, notes, doc_id]
                flat_text = " ".join(p for p in flat_parts if p)

                text_parts = [title, author, publisher, major, abstract, doc_type, course, subject, language, notes, doc_id]
                text = " ".join(p for p in text_parts if p and p not in ("Unknown", "N/A"))

                # Include ALL unmapped columns as extra_ fields for searchability
                mapped_cols = set(col_map.values())
                extra_fields = {}
                for c in df.columns:
                    if c not in mapped_cols:
                        val = _cell_raw(row, c)
                        if val and val.lower() != 'nan':
                            extra_fields[str(c).strip()] = val
                            # Also add to flat_text so BM25 can find it
                            flat_text += " " + val

                document = {
                    "title": title,
                    "author": author,
                    "co_authors": authors_list[1:] if len(authors_list) > 1 else [],
                    "authors": authors_list if authors_list else [author],
                    "year": year if year else "N/A",
                    "subject": subject,
                    "link": link,
                    "doc_type": doc_type,
                    "format": "Sách",
                    "publisher": publisher,
                    "major": major,
                    "course": course,
                    "keywords": keywords,
                    "abstract": abstract,
                    "language": language,
                    "notes": notes,
                    "doc_id": doc_id,
                    "extra_fields": extra_fields,
                    "source": str(filepath),
                    "csv_file": str(filepath),
                    "text": text,
                    "flat_text": flat_text,
                }
                documents.append(document)
            except Exception as e:
                if idx < 5:
                    print(f"⚠️ Error parsing row {idx}: {e}")
                continue

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
        documents = []
        df = pd.read_excel(filepath, header=0, dtype=str)

        tag_cols = {str(c).strip(): c for c in df.columns if str(c).strip().isdigit()}
        # Chỉ nhận các cột phòng/thư viện thực tế — tránh cột lạ (Ghi chú, Ngày nhập, ...)
        _LOCATION_KEYWORDS = ("phòng", "phong", "mượn", "muon", "đọc", "doc",
                              "kho", "số", "so", "giá", "gia", "tủ", "tu")
        location_cols = [
            c for c in df.columns
            if not str(c).strip().isdigit()
            and str(c) != 'ItemId'
            and any(kw in str(c).strip().lower() for kw in _LOCATION_KEYWORDS)
        ]

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
                              edition, major, doc_type, course, abstract, ' '.join(keywords)]
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

    def count_title_match(self, query: str) -> int:
        """
        Count documents where query appears in title, subject, or keywords ONLY.
        Unlike count_matching(), this does NOT search full-text (abstract, notes, locations, etc.),
        preventing false positives when a generic term coincidentally appears in unrelated fields.
        Use this for existence checks like "có sách abc không?".
        """
        if not query or not query.strip():
            return 0
        q = normalize_text(query).strip()
        if not q or len(q) <= 1:
            return 0
        tokens = set(re.findall(r'\w+', q, re.UNICODE))
        count = 0
        for doc in self.bm25.documents:
            # Check title
            title = normalize_text(doc.get('title', '') or '')
            # Check subject (first keyword)
            subject = normalize_text(doc.get('subject', '') or '')
            # Check all keywords
            keywords = doc.get('keywords', []) or []
            keywords_norm = [normalize_text(kw) for kw in keywords if kw]
            # Check major field
            major = normalize_text(doc.get('major', '') or '')

            combined = f"{title} {subject} {' '.join(keywords_norm)} {major}"
            if q in combined:
                count += 1
                continue
            # Also check if all query tokens appear in title (for multi-word queries)
            if len(tokens) > 1:
                title_tokens = set(re.findall(r'\w+', title, re.UNICODE))
                if tokens.issubset(title_tokens):
                    count += 1
                    continue
        return count

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
