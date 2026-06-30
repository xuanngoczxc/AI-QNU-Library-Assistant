"""
pdf_manager.py — Quản lý PDF: metadata, chapters, preview, OCR

Chức năng chính:
  1. Liệt kê tất cả PDF trong thư mục pdfs/ (list_all_pdfs)
  2. Trích xuất metadata PDF: title, author, year, pages, subject (_extract_metadata)
  3. Trích xuất cấu trúc chương/phần từ outline hoặc heuristic (extract_chapters)
  4. Đọc nội dung text PDF: MarkItDown → pypdf → OCR (get_chapter_text, get_chapter_text_with_ocr_fallback)
  5. Kiểm tra PDF scan/ảnh (is_scanned_pdf)
  6. OCR fallback bằng Tesseract (_ocr_page)

Thư viện sử dụng:
  - MarkItDown (Microsoft): đọc PDF → Markdown chất lượng cao
  - pypdf: đọc metadata, outline, text theo trang (fallback)
  - pytesseract + pdf2image: OCR cho PDF scan/ảnh (tùy chọn)
"""

import os
import re
import json
import threading
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    from markitdown import MarkItDown
    _markitdown = MarkItDown()
except ImportError:
    _markitdown = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import pytesseract
    from pdf2image import convert_from_path
    _ocr_available = True
except ImportError:
    _ocr_available = False
    pytesseract = None
    convert_from_path = None

_TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
if _TESSERACT_CMD and pytesseract:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

class PDFManager:
    """
    Quản lý tất cả thao tác với file PDF:
      - Liệt kê PDF (list_all_pdfs): quét thư mục pdfs/, trích xuất metadata
      - Trích xuất chương (extract_chapters): từ outline hoặc chia đều
      - Đọc nội dung (get_chapter_text): ưu tiên MarkItDown, fallback pypdf
      - OCR (get_chapter_text_with_ocr_fallback): cho PDF scan/ảnh
      - Cache: metadata cache, MarkItDown text cache, OCR cache
    """
    
    def __init__(self, pdf_dir: str = "pdfs", cache_file: str = "pdf_cache.json"):
        self.pdf_dir = Path(pdf_dir)          
        self.cache_file = cache_file          
        self.pdf_cache = self._load_cache()
        
        self._md_text_cache: dict[str, str] = {}

        self._ocr_cache: dict[str, dict[int, str]] = {}
        self._ocr_cache_lock = threading.Lock()
   
        self.pdf_dir.mkdir(exist_ok=True)
    
    def invalidate_cache(self):
        """Xóa cache metadata — gọi khi có file upload/xóa."""
        self._list_cache = None
        self._md_text_cache.clear()
    
    def list_all_pdfs(self) -> List[Dict]:
        """
        Liệt kê tất cả PDF trong thư mục pdfs/ với metadata đầy đủ.
        Quy trình:
          1. Quét tất cả file .pdf trong thư mục
          2. Với mỗi file: trích xuất metadata (title, author, year, pages, subject)
          3. Resolve display name (ưu tiên tên file nếu metadata title rác)
          4. Cache kết quả TTL 5 giây để tránh load lại mỗi request
        Trả về list dict: [{id, title, author, year, pages, file_path, file_size_mb, ...}]
        """
        self._md_text_cache.clear()
        now = datetime.now().timestamp()
        if hasattr(self, '_list_cache') and self._list_cache is not None:
            cache_time, cached_pdfs = self._list_cache
            if now - cache_time < 5:
                return cached_pdfs
        self._list_cache = None
        
        pdfs = []
        
        if not self.pdf_dir.exists():
            return []
        
        for i, pdf_file in enumerate(sorted(self.pdf_dir.glob("*.pdf"))):
            try:
                metadata = self._extract_metadata(pdf_file)
                file_stem = pdf_file.stem.strip()
                file_name = pdf_file.name
                display_name = self._resolve_display_name(metadata, file_stem)
                pdfs.append({
                    "id": f"pdf_{i:03d}",
                    "title": display_name,
                    "display_name": display_name,
                    "original_title": metadata.get('title', file_stem),
                    "author": metadata.get('author', 'Unknown'),
                    "year": metadata.get('year', 'N/A'),
                    "pages": metadata.get('pages', 0),
                    "subject": metadata.get('subject', ''),
                    "file_path": str(pdf_file),
                    "file_name": file_name,
                    "file_stem": file_stem,
                    "file_size_mb": round(pdf_file.stat().st_size / (1024 * 1024), 2),
                    "added_date": datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat(),
                    "thumbnail_url": f"/api/pdfs/pdf_{i:03d}/thumbnail"
                })
            except Exception as e:
                print(f"Lỗi xử lý PDF {pdf_file.name}: {e}")
                continue
        
        # Lưu cache
        self._list_cache = (datetime.now().timestamp(), pdfs)
        return pdfs
    
    def extract_chapters(self, pdf_path: str) -> List[Dict]:
        """
        Trích xuất cấu trúc chương/phần từ PDF.
        Chiến lược:
          1. Thử lấy outline (mục lục) từ PDF bằng pypdf
          2. Validate outline: phải có ≥2 mục hợp lệ (loại bỏ entry rác)
          3. Nếu outline hợp lệ → chia PDF thành các chương theo số trang
          4. Nếu outline không có hoặc quá nông → fallback heuristic (chia đều)
        Trả về list: [{chapter, title, pages, start_page, end_page}]
        """
        if not PdfReader:
            print("pypdf not available, returning simple chapters")
            return self._generate_simple_chapters(pdf_path)
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                chapters = []
                
                outline_items = []
                if reader.outline and len(reader.outline) > 0:
                    outline_items = self._flatten_outline(reader.outline)

                valid_outline_titles = [
                    self._sanitize_outline_title(item.get("title", ""))
                    for item in outline_items
                    if self._sanitize_outline_title(item.get("title", ""))
                ]

                if len(valid_outline_titles) >= 2:
                    total_pages = len(reader.pages)
                    pages_per_chapter = max(8, total_pages // max(2, len(valid_outline_titles)))
                    for i, title in enumerate(valid_outline_titles):
                        start_page = min(i * pages_per_chapter, max(total_pages - 1, 0))
                        end_page = min(start_page + pages_per_chapter, total_pages)
                        if end_page <= start_page:
                            end_page = min(start_page + 1, total_pages)

                        chapters.append({
                            "chapter": i + 1,
                            "title": title[:80],
                            "pages": f"{start_page + 1}-{end_page}",
                            "start_page": start_page,
                            "end_page": end_page
                        })

                if len(chapters) < 2:
                    chapters = self._generate_simple_chapters(pdf_path)
                
                if not chapters:
                    chapters = self._generate_simple_chapters(pdf_path)
                
                return chapters
        
        except Exception as e:
            print(f"Lỗi trích xuất chapters: {e}")
            return self._generate_simple_chapters(pdf_path)

    def _flatten_outline(self, outline) -> List[Dict]:
        """
        Chuyển outline lồng nhau (nested) thành danh sách phẳng.
        Outline của PDF có thể lồng nhau (chapter → section → subsection)
        → flatten để xử lý dễ dàng hơn.
        """
        flattened = []

        def walk(items):
            for item in items:
                if isinstance(item, list):
                    walk(item)
                    continue

                title = ""
                page_number = None

                try:
                    if hasattr(item, "title"):
                        title = str(item.title)
                    elif isinstance(item, dict):
                        title = str(item.get("/Title") or item.get("title") or "")
                    else:
                        title = str(item)

                    if not title or self._looks_like_bad_outline_title(title):
                        continue

                    if hasattr(item, "page"):
                        page_number = self._safe_page_number(item.page)
                    elif isinstance(item, dict) and item.get("/Page") is not None:
                        page_number = self._safe_page_number(item.get("/Page"))
                except Exception:
                    continue

                flattened.append({"title": title, "page_number": page_number})

        walk(outline)
        return flattened

    def _safe_page_number(self, page_ref) -> Optional[int]:
        """Resolve page reference to zero-based page number if possible."""
        try:
            if not PdfReader:
                return None
            return None
        except Exception:
            return None

    def _looks_like_bad_outline_title(self, title: str) -> bool:
        normalized = str(title).strip().lower()
        if not normalized:
            return True
        bad_markers = [
            "indirectobject",
            "{'title':",
            '[{\'title\':',
            "page': indirectobject",
            "page\": indirectobject",
            "<class '",
        ]
        return any(marker in normalized for marker in bad_markers)

    def _sanitize_outline_title(self, title: str) -> str:
        """Chuẩn hóa title outline; loại bỏ entry rác."""
        if title is None:
            return ""

        cleaned = str(title).strip()
        if self._looks_like_bad_outline_title(cleaned):
            return ""

        cleaned = cleaned.replace("\n", " ").replace("\r", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned
    
    def _generate_simple_chapters(self, pdf_path: str) -> List[Dict]:
        """
        Tạo chapters đơn giản dựa trên số trang (heuristic fallback).
        Khi PDF không có outline → chia đều:
          - PDF ≤12 trang: mỗi phần ~4 trang
          - PDF >12 trang: mỗi phần ~10 trang
        Trả về list: [{chapter, title, pages, start_page, end_page}]
        """
        if not PdfReader:
            return [
                {
                    "chapter": i + 1,
                    "title": f"Phần {i + 1}",
                    "pages": f"1-20",
                    "start_page": 0,
                    "end_page": 20
                }
                for i in range(5)
            ]
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                total_pages = len(reader.pages)
                if total_pages <= 12:
                    pages_per_chapter = max(4, total_pages // 3)
                else:
                    pages_per_chapter = max(10, total_pages // 8)
                
                chapters = []
                for i in range(0, total_pages, pages_per_chapter):
                    chapter_num = (i // pages_per_chapter) + 1
                    chapter = {
                        "chapter": chapter_num,
                        "title": f"Phần {chapter_num}",
                        "pages": f"{i+1}-{min(i+pages_per_chapter, total_pages)}",
                        "start_page": i,
                        "end_page": min(i + pages_per_chapter, total_pages)
                    }
                    chapters.append(chapter)
                
                return chapters
        except:
            return []
    
    def _load_markitdown_text(self, pdf_path: str) -> str | None:
        """
        Đọc toàn bộ PDF bằng MarkItDown, cache kết quả.
        MarkItDown (Microsoft) chuyển PDF → Markdown chất lượng cao,
        giữ nguyên cấu trúc bảng biểu, heading, danh sách.
        Kết quả được cache trong _md_text_cache để tránh đọc lại.
        """
        if pdf_path in self._md_text_cache:
            return self._md_text_cache[pdf_path]
        
        if not _markitdown:
            return None
        
        try:
            result = _markitdown.convert(pdf_path)
            text = (result.text_content or "").strip()
            if text:
                self._md_text_cache[pdf_path] = text
            return text or None
        except Exception as e:
            print(f"  ⚠️ MarkItDown error for {Path(pdf_path).name}: {e}")
            return None
    
    def get_chapter_text(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """
        Trích xuất nội dung text từ PDF (theo khoảng trang).
        Chiến lược:
          1. Nếu đọc toàn bộ hoặc khoảng lớn (≥3 trang) → dùng MarkItDown
          2. Nếu đọc 1-2 trang cụ thể → ưu tiên pypdf (chính xác theo trang)
          3. Fallback: MarkItDown full text nếu pypdf thất bại
          4. Fallback cuối: pypdf nếu MarkItDown không khả dụng
        """
        is_full_read = (end_page - start_page) >= 3 or end_page >= 9999
        
        md_text = self._load_markitdown_text(pdf_path)
        if md_text:
            if is_full_read:
                return md_text
            else:
                if PdfReader:
                    try:
                        with open(pdf_path, 'rb') as f:
                            reader = PdfReader(f)
                            text = ""
                            for page_num in range(max(0, start_page), min(end_page, len(reader.pages))):
                                try:
                                    page = reader.pages[page_num]
                                    text += page.extract_text() + "\n"
                                except:
                                    text += f"[Trang {page_num + 1}: Không thể trích xuất]\n"
                            if text.strip():
                                return text
                    except:
                        pass
                return md_text
        
        # Fallback: pypdf
        if not PdfReader:
            return "[MarkItDown và pypdf đều không khả dụng]"
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                text = ""
                for page_num in range(max(0, start_page), min(end_page, len(reader.pages))):
                    try:
                        page = reader.pages[page_num]
                        text += page.extract_text() + "\n"
                    except:
                        text += f"[Trang {page_num + 1}: Không thể trích xuất]\n"
                return text if text.strip() else "[Không có nội dung]"
        except Exception as e:
            print(f"Lỗi trích xuất text từ PDF: {e}")
            return f"[Lỗi: {str(e)}]"
    
    def _extract_metadata(self, pdf_path: Path) -> Dict:
        """
        Trích xuất metadata từ file PDF.
        Lấy từ pypdf: /Title, /Author, /Subject, /CreationDate, số trang.
        Xử lý垃圾 metadata:
          - Author rác (aspose, pdf converter, unknown) → parse từ tên file
          - Subject rác (aspose, pdf, n/a) → bỏ trống
          - Year: parse từ /CreationDate (format D:YYYYMMDDHHmmSS)
        """
        _junk_authors = {
            'aspose', 'aspose.pdf', 'aspose.pdf.epubtopdfconverter',
            'pdf converter', 'epubtopdfconverter', 'unknown',
            'microsoft', 'adobe', 'itext', 'itextsharp',
        }
        _junk_subjects = {'aspose', 'pdf', 'n/a', 'unknown', ''}

        def _clean_author(val):
            v = (val or '').strip()
            if not v or v.lower() in _junk_authors:
                return 'Unknown'
            if any(kw in v.lower() for kw in ('aspose', 'converter', 'epubtopdf', 'pdftool')):
                return 'Unknown'
            return v

        def _clean_subject(val):
            v = (val or '').strip()
            if not v or v.lower() in _junk_subjects:
                return ''
            return v

        if not PdfReader:
            return {
                'title': pdf_path.stem,
                'author': 'Unknown',
                'pages': 0,
                'subject': '',
                'year': 'N/A'
            }
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                info = reader.metadata
                
                raw_author = info.get('/Author', 'Unknown') if info else 'Unknown'
                raw_subject = info.get('/Subject', '') if info else ''
                
                # Nếu metadata author là tool name, thử lấy từ tên file
                cleaned_author = _clean_author(raw_author)
                if cleaned_author == 'Unknown':
                    file_stem = pdf_path.stem
                    m = re.match(r'^([A-Z][A-Z\s\.]+?)\s*[-–]\s*', file_stem)
                    if m:
                        cleaned_author = m.group(1).strip()
                
                return {
                    'title': info.get('/Title', pdf_path.stem) if info else pdf_path.stem,
                    'author': cleaned_author,
                    'pages': len(reader.pages),
                    'subject': _clean_subject(raw_subject),
                    'year': self._extract_year_from_metadata(info) if info else 'N/A'
                }
        except Exception as e:
            print(f"Lỗi trích xuất metadata: {e}")
            return {
                'title': pdf_path.stem,
                'author': 'Unknown',
                'pages': 0,
                'subject': '',
                'year': 'N/A'
            }

    def _resolve_display_name(self, metadata: Dict, file_stem: str) -> str:
        """
        Ưu tiên tên file thay vì title metadata khi metadata là tên chung/generic.
        Nhiều PDF có title rác (aspose, document, untitled) → dùng tên file.
        """
        title = str(metadata.get('title', '') or '').strip()
        normalized = title.lower()
        generic_titles = {
            'aspose',
            'document',
            'untitled',
            'unknown',
            'pdf'
        }

        if not title or normalized in generic_titles:
            return file_stem

        if len(title) <= 3:
            return file_stem

        return title
    
    def _extract_year_from_metadata(self, info) -> str:
        """Cố gắng trích xuất năm từ creation date"""
        if not info:
            return 'N/A'
        
        try:
            creation_date = info.get('/CreationDate', '')
            if creation_date:
                # Format: D:YYYYMMDDHHmmSS
                year_str = str(creation_date)
                # Tìm 4 chữ số liên tiếp
                import re
                match = re.search(r'(\d{4})', year_str)
                if match:
                    year = match.group(1)
                    if 1900 <= int(year) <= 2100:
                        return year
        except:
            pass
        
        return 'N/A'
    
    def _load_cache(self) -> Dict:
        """Tải cached PDF metadata"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Lưu PDF metadata cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.pdf_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Lỗi lưu cache: {e}")

    def is_scanned_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """
        Heuristic phát hiện PDF scan/ảnh.
        Kiểm tra vài trang đầu: nếu text trích xuất được < 50 chars/trang
        và ≥2/3 trang rỗng → coi là PDF scan.
        """
        if not PdfReader:
            return False

        try:
            with open(pdf_path, 'rb') as f:
                reader = PdfReader(f)
                total = min(sample_pages, len(reader.pages))
                if total == 0:
                    return False
                low_text_pages = 0
                for i in range(total):
                    try:
                        t = (reader.pages[i].extract_text() or "").strip()
                        if len(t) < 50:
                            low_text_pages += 1
                    except Exception:
                        low_text_pages += 1
                # ≥ 2/3 trang đầu rỗng → coi như scan
                return low_text_pages >= max(1, int(total * 0.66))
        except Exception:
            return False

    def _ocr_page(self, pdf_path: str, page_num: int, lang: str = "vie+eng") -> str:
        """
        OCR 1 trang PDF cụ thể bằng Tesseract.
        Quy trình: chuyển trang PDF → ảnh (pdf2image, dpi=200) → OCR (pytesseract).
        Có cache: mỗi trang chỉ OCR 1 lần, kết quả lưu trong _ocr_cache.
        Ngôn ngữ mặc định: vie+eng (tiếng Việt + tiếng Anh).
        """
        if not _ocr_available:
            return ""

        cache_key = (pdf_path, page_num)
        with self._ocr_cache_lock:
            if pdf_path in self._ocr_cache and page_num in self._ocr_cache[pdf_path]:
                return self._ocr_cache[pdf_path][page_num]

        try:
            images = convert_from_path(
                pdf_path,
                dpi=200,
                first_page=page_num + 1,
                last_page=page_num + 1,
            )
            if not images:
                return ""
            text = pytesseract.image_to_string(images[0], lang=lang) or ""
            text = text.strip()
        except Exception as e:
            print(f"OCR error page {page_num + 1} of {Path(pdf_path).name}: {e}")
            text = ""

        with self._ocr_cache_lock:
            self._ocr_cache.setdefault(pdf_path, {})[page_num] = text
        return text

    def get_chapter_text_with_ocr_fallback(
        self, pdf_path: str, start_page: int, end_page: int, lang: str = "vie+eng"
    ) -> Dict:
        """
        Đọc PDF với 3 cấp fallback:
          1. MarkItDown → text chất lượng cao (Markdown)
          2. pypdf → text theo trang
          3. Tesseract OCR → cho PDF scan/ảnh
        Trả về dict:
          - text: nội dung trích xuất
          - method: "markitdown" | "pypdf" | "ocr" | "none"
          - scanned: True nếu PDF là bản scan
        """
        md_text = self._load_markitdown_text(pdf_path) or ""
        if md_text and len(md_text.strip()) >= 50:
            return {"text": md_text, "method": "markitdown", "scanned": False}

        pypdf_text = ""
        if PdfReader:
            try:
                with open(pdf_path, "rb") as f:
                    reader = PdfReader(f)
                    for p in range(max(0, start_page), min(end_page, len(reader.pages))):
                        try:
                            pypdf_text += (reader.pages[p].extract_text() or "") + "\n"
                        except Exception:
                            pypdf_text += f"[Trang {p + 1}: lỗi trích xuất]\n"
            except Exception:
                pass

        if pypdf_text.strip() and len(pypdf_text.strip()) >= 50:
            return {"text": pypdf_text, "method": "pypdf", "scanned": False}

        if not _ocr_available:
            return {
                "text": pypdf_text or "[PDF scan — text rỗng. Cài pytesseract + pdf2image + Tesseract OCR để đọc.]",
                "method": "none",
                "scanned": True,
            }

        scanned = self.is_scanned_pdf(pdf_path)
        if not scanned:
            scanned = True

        ocr_texts = []
        for p in range(max(0, start_page), min(end_page, self.get_page_count(pdf_path) or end_page)):
            ocr_texts.append(self._ocr_page(pdf_path, p, lang=lang))
        ocr_text = "\n\n".join([t for t in ocr_texts if t])

        return {
            "text": ocr_text or "[OCR không trích xuất được text]",
            "method": "ocr",
            "scanned": scanned,
        }

    def get_page_count(self, pdf_path: str) -> int:
        """Số trang PDF, fallback 0 nếu lỗi."""
        if not PdfReader:
            return 0
        try:
            with open(pdf_path, "rb") as f:
                return len(PdfReader(f).pages)
        except Exception:
            return 0
