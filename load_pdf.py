"""
load_pdf.py — Load tất cả file PDF từ thư mục pdfs/
Sử dụng MarkItDown (Microsoft) để chuyển PDF → Markdown chất lượng cao,
giữ nguyên cấu trúc bảng biểu, heading, danh sách — phù hợp cho LLM.
Fallback: dùng pypdf nếu MarkItDown không hoạt động.
Tối ưu: Extract metadata (page, title, section), normalize text
"""

import os
import re
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
    Extract title từ first page của PDF
    Logic: First line nếu < 100 chars, hoặc filename
    """
    try:
        if pages:
            first_page_text = pages[0].page_content.strip()
            first_line = first_page_text.split("\n")[0].strip()
            if 5 < len(first_line) < 100:
                return first_line
    except Exception:
        pass

    # Fallback: dùng filename
    return Path(pdf_path).stem.replace("_", " ").replace("-", " ")


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


def load_all_pdfs() -> list[Document]:
    """Load tất cả PDF trong thư mục pdfs/, trả về danh sách Document"""
    folder = Path(PDF_FOLDER)
    if not folder.exists():
        print(f"⚠️  Thư mục '{PDF_FOLDER}' không tồn tại. Tạo thư mục và thêm file PDF vào.")
        folder.mkdir(exist_ok=True)
        return []

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"⚠️  Không có file PDF nào trong thư mục '{PDF_FOLDER}'.")
        return []

    all_docs = []
    total_pages = 0

    for pdf_path in pdf_files:
        try:
            print(f"📄 Đang load: {pdf_path.name}")

            # Ưu tiên MarkItDown → fallback pypdf
            docs = _read_pdf_with_markitdown(str(pdf_path))
            if not docs:
                docs = _read_pdf_with_pypdf(str(pdf_path))

            if not docs:
                print(f"   ⚠️  Không đọc được nội dung từ {pdf_path.name}")
                continue

            # Extract PDF-level metadata
            pdf_title = extract_pdf_title(str(pdf_path), docs)
            pdf_keywords = extract_keywords(pdf_title)

            # Thêm metadata tên file + page number + section
            for page_idx, doc in enumerate(docs, 1):
                doc.metadata["source"] = pdf_path.name
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

            all_docs.extend(docs)
            total_pages += len(docs)
            print(f"   ✅ {len(docs)} trang")
        except Exception as e:
            print(f"   ❌ Lỗi load {pdf_path.name}: {e}")

    print(f"\n📦 Tổng cộng: {total_pages} trang từ {len(pdf_files)} file PDF")
    return all_docs


if __name__ == "__main__":
    docs = load_all_pdfs()
    for i, d in enumerate(docs[:3]):
        print(f"\n--- Trang {i+1} ---")
        print(f"Nguồn: {d.metadata.get('source')} | Trang: {d.metadata.get('page')}")
        print(d.page_content[:200] + "...")
