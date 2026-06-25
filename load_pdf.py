"""
load_pdf.py — Load tất cả file PDF từ thư mục pdfs/
Sử dụng MarkItDown (Microsoft) để chuyển PDF → Markdown chất lượng cao,
giữ nguyên cấu trúc bảng biểu, heading, danh sách — phù hợp cho LLM.
Fallback: dùng pypdf nếu MarkItDown không hoạt động.
Tối ưu: Extract metadata (page, title, section), normalize text
"""

import os
import re
import pickle
import hashlib
from pathlib import Path
from typing import Optional
from text_utils import normalize_text, generate_document_hash, extract_keywords

# ── MarkItDown (Microsoft) — giữ cấu trúc Markdown chất lượng cao ──
try:
    from markitdown import MarkItDown
    _markitdown = MarkItDown()
except ImportError:
    _markitdown = None

# ── pypdf (fallback) ─────────────────────────────────────
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

PDF_FOLDER = "pdfs"
_PDF_CACHE_FILE = "Data/_pdf_extract_cache.pkl"


class Document:
    """Minimal document class — thay thế langchain_core.documents.Document."""
    def __init__(self, page_content: str, metadata: Optional[dict] = None):
        self.page_content = page_content or ""
        self.metadata = metadata or {}


def _read_pdf_with_markitdown(pdf_path: str) -> list[Document]:
    """
    Đọc PDF bằng MarkItDown.
    Output là Markdown với page markers: <!-- page N -->
    Trả về list[Document] (mỗi trang một Document), nội dung là Markdown.
    """
    docs: list[Document] = []
    if not _markitdown:
        return docs

    result = _markitdown.convert(pdf_path)
    markdown_text = result.text_content or ""

    if not markdown_text.strip():
        return docs

    # Tách theo page markers: <!-- page N --> hoặc <!-- Page N -->
    pages = re.split(r"<!--\s*[Pp]age\s*\d+\s*-->", markdown_text)
    pages = [p.strip() for p in pages if p.strip()]

    if not pages:
        # Nếu không có page marker, xem như 1 trang duy nhất
        pages = [markdown_text.strip()]

    for i, page_content in enumerate(pages):
        docs.append(Document(
            page_content=page_content,
            metadata={"page": i + 1}
        ))

    return docs


def _read_pdf_with_pypdf(pdf_path: str) -> list[Document]:
    """Đọc PDF bằng pypdf (fallback), trả về list[Document] (mỗi trang một Document)."""
    docs: list[Document] = []
    if not PdfReader:
        print("   ⚠️  pypdf not available — skipping PDF text extraction")
        return docs

    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"page": page_num}))

    return docs


def extract_pdf_title(pdf_path: str, pages: list[Document]) -> str:
    """
    Extract title t? first page c?a PDF.
    ?u tiên dùng filename, ch? dùng first-line n?u không ph?i boilerplate.
    """
    # ?u tiên: dùng filename (luôn ch?nh x?c nh?t)
    filename = Path(pdf_path).stem
    # Gi? l?i tên file g?c thay vì thay - b?ng space (-
    # là d?u phân c?ch tên tác gi? - tên lu?n v?n)
    pdf_title = filename.strip()
    if pdf_title and len(pdf_title) >= 5:
        return pdf_title

    # Fallback: th? l?y t? first page
    try:
        if pages:
            first_page_text = pages[0].page_content.strip()
            first_line = first_page_text.split("\n")[0].strip()
            # Lo?i b? boilerplate Vi?t Nam ph? bi?n
            boilerplate = [
                "bo giao duc va dao tao",
                "bo giao duc",
                "truong dai hoc quy nhon",
                "dai hoc quy nhon",
                "cong hoa xa hoi chu nghia viet nam",
                "cong hoa",
                "bo	ai hoc",
            ]
            first_norm = "".join(c for c in first_line.lower() if c.isalnum() or c == ' ').strip()
            first_norm = " ".join(first_norm.split())
            if first_line and len(first_line) < 80 and not any(bp in first_norm for bp in boilerplate):
                return first_line
    except Exception:
        pass

    return filename.strip()


def extract_section_heading(page_text: str) -> str:
    """
    Extract section heading từ page text
    Logic: Tìm dòng đầu tiên với < 80 chars (probable heading)
    """
    lines = page_text.split("\n")
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if 5 < len(line) < 80 and line.isupper():  # UPPERCASE heading
            return line
    return ""


def _load_pdf_cache() -> dict:
    """Load PDF extraction cache: {filename: (mtime, docs_data)}"""
    try:
        if Path(_PDF_CACHE_FILE).exists():
            with open(_PDF_CACHE_FILE, "rb") as f:
                return pickle.load(f)
    except Exception:
        pass
    return {}

def _save_pdf_cache(cache: dict):
    """Save PDF extraction cache"""
    try:
        with open(_PDF_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(f"PDF cache save failed: {e}")

def _docs_to_cacheable(docs: list[Document]) -> list[dict]:
    """Convert Document objects to storable dicts"""
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

def _cacheable_to_docs(data: list[dict]) -> list[Document]:
    """Convert cached dicts back to Document objects"""
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]

def load_all_pdfs() -> list[Document]:
    """
    Load tất cả PDF trong thư mục pdfs/.
    Incremental: chỉ extract PDF mới hoặc thay đổi (so với cache).
    PDF đã extract trước đó → load từ cache instantly.
    """
    folder = Path(PDF_FOLDER)
    if not folder.exists():
        print(f"Thư mục '{PDF_FOLDER}' không tồn tại. Tạo thư mục và thêm file PDF vào.")
        folder.mkdir(exist_ok=True)
        return []

    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"Không có file PDF nào trong thư mục '{PDF_FOLDER}'.")
        return []

    cache = _load_pdf_cache()

    all_docs = []
    total_pages = 0
    cache_hits = 0
    cache_misses = 0
    skipped = 0

    for pdf_path in pdf_files:
        fname = pdf_path.name
        current_mtime = pdf_path.stat().st_mtime

        # Check cache: same filename + same mtime → skip extraction
        if fname in cache:
            cached_mtime, cached_docs_data = cache[fname]
            if cached_mtime == current_mtime:
                docs = _cacheable_to_docs(cached_docs_data)
                all_docs.extend(docs)
                total_pages += len(docs)
                cache_hits += 1
                continue

        try:
            print(f"Đang extract: {fname} (mới/thay đổi)")

            docs = _read_pdf_with_markitdown(str(pdf_path))
            if not docs:
                docs = _read_pdf_with_pypdf(str(pdf_path))

            if not docs:
                print(f"Không đọc được nội dung từ {fname}")
                skipped += 1
                continue

            # Extract PDF-level metadata
            pdf_title = extract_pdf_title(str(pdf_path), docs)
            pdf_keywords = extract_keywords(pdf_title)

            # Thêm metadata tên file + page number + section
            for page_idx, doc in enumerate(docs, 1):
                doc.metadata["source"] = fname
                doc.metadata["file_type"] = "pdf"
                doc.metadata["page"] = page_idx
                doc.metadata["title"] = pdf_title
                doc.metadata["keywords"] = pdf_keywords

                # Extract section heading
                section = extract_section_heading(doc.page_content)
                if section:
                    doc.metadata["section"] = section

                # Normalize text for embedding
                doc.metadata["normalized"] = normalize_text(doc.page_content)

                # Hash for deduplication
                doc.metadata["hash"] = generate_document_hash(doc.page_content, pdf_title)

            # Save to cache
            cache[fname] = (current_mtime, _docs_to_cacheable(docs))

            all_docs.extend(docs)
            total_pages += len(docs)
            cache_misses += 1
            print(f"{len(docs)} trang (đã cache)")
        except Exception as e:
            print(f"Lỗi load {fname}: {e}")
            skipped += 1

    # Cleanup: remove cache entries for deleted PDFs
    deleted = [k for k in cache if k not in {f.name for f in pdf_files}]
    for k in deleted:
        del cache[k]

    # Save updated cache
    if cache_misses > 0:
        _save_pdf_cache(cache)

    print(f"\nTổng cộng: {total_pages} trang từ {len(pdf_files)} PDF")
    print(f"Cache hit: {cache_hits} | Extract mới: {cache_misses} | Bỏ qua: {skipped}")
    return all_docs


if __name__ == "__main__":
    docs = load_all_pdfs()
    for i, d in enumerate(docs[:3]):
        print(f"\n--- Trang {i+1} ---")
        print(f"Nguồn: {d.metadata.get('source')} | Trang: {d.metadata.get('page')}")
        print(d.page_content[:200] + "...")
