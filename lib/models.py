"""
lib/models.py — Pydantic request/response schemas.
"""

from typing import Optional
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    document_id: str = None
    session_id: str = "default"
    major: str = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    total_sources: int = 0
    total_found: int = 0
    summary: str = ""
    current_document: Optional[dict] = None
    follow_up_suggestions: list[dict] = []


class SearchRequest(BaseModel):
    query: str = ""
    author: str = ""
    year: str = ""
    subject: str = ""
    major: str = ""
    doc_type: str = ""
    top_k: int = 10


class SuggestRequest(BaseModel):
    title: str = ""
    author: str = ""
    subject: str = ""
    top_k: int = 8
    exclude_titles: list[str] = []


class SpellCheckRequest(BaseModel):
    text: str


class TermRequest(BaseModel):
    term: str
    language: str = "vi"


class DocumentChatRequest(BaseModel):
    query: str
    pdf_id: str
    chapter: Optional[int] = None
    search_type: str = "full_pdf"
    session_id: str = "default"

class FacetRequest(BaseModel):
    query: str
    top_k: int = 50

class FacetFilterRequest(BaseModel):
    query: str
    group_name: str
    group_value: str
    top_k: int = 50

class FacetRefineRequest(BaseModel):
    query: str
    filters: dict[str, Optional[str]] = {}
    top_k: int = 50


class SemanticSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    alpha: Optional[float] = 0.5


class SemanticSearchResponse(BaseModel):
    query: str
    mode: str
    results: list
    total: int
    semantic_available: bool


class FacultySearchRequest(BaseModel):
    nganh: str
    khoa: str = None
    top_k: Optional[int] = 20


class VoiceTranscribeRequest(BaseModel):
    language: Optional[str] = "vi"
