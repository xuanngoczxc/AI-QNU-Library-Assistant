"""
lib/dialogue_state.py — Formal Dialogue State Tracking (DST) for multi-turn sessions.
Manages persistent conversational context, active search filters, cumulative working sets,
and selected book focus across multiple conversation turns.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from collections import OrderedDict
import time

MAX_DST_SESSIONS = 2000


@dataclass
class ConversationState:
    session_id: str
    topics: List[str] = field(default_factory=list)               # Các chủ đề đang tìm kiếm/tích lũy
    primary_topic: str = ""                                       # Chủ đề chính hiện tại
    doc_types: List[str] = field(default_factory=list)            # Danh sách loại tài liệu (hỗ trợ OR: ["Sách", "Giáo trình"])
    major: Optional[str] = None                                   # Ngành học hiện tại (VD: "Quản trị kinh doanh")
    author: Optional[str] = None                                  # Tác giả đang tra cứu
    language: Optional[str] = None                                # Ngôn ngữ ("vi", "en", "song ngữ")
    year_min: Optional[int] = None                                # Năm xuất bản từ (>=)
    year_max: Optional[int] = None                                # Năm xuất bản đến (<=)
    working_set: List[Dict[str, Any]] = field(default_factory=list) # Danh sách tài liệu của kết quả tìm kiếm gần nhất
    all_session_docs: List[Dict[str, Any]] = field(default_factory=list) # Toàn bộ tài liệu tích lũy trong phiên
    selected_doc: Optional[Dict[str, Any]] = None                 # Cuốn sách cụ thể đang được xem xét / hỏi thuộc tính
    last_intent: Optional[str] = None                             # Intent của lượt trước
    last_query: str = ""                                          # Câu hỏi lượt trước
    last_search_query: str = ""                                   # Search query gửi vào engine ở lượt trước
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        return cls(
            session_id=data.get("session_id", ""),
            topics=data.get("topics") or [],
            primary_topic=data.get("primary_topic", ""),
            doc_types=data.get("doc_types") or [],
            major=data.get("major"),
            author=data.get("author"),
            language=data.get("language"),
            year_min=data.get("year_min"),
            year_max=data.get("year_max"),
            working_set=data.get("working_set") or [],
            all_session_docs=data.get("all_session_docs") or [],
            selected_doc=data.get("selected_doc"),
            last_intent=data.get("last_intent"),
            last_query=data.get("last_query", ""),
            last_search_query=data.get("last_search_query", ""),
            updated_at=data.get("updated_at", time.time()),
        )

    def set_working_set(self, docs: List[Dict[str, Any]]):
        """Cập nhật danh sách tài liệu hiện hành và tích lũy vào toàn bộ phiên."""
        self.working_set = docs[:50]  # Giữ tối đa 50 tài liệu gần nhất
        if docs:
            self.selected_doc = docs[0]
            existing_ids = {
                d.get("doc", d).get("id") or d.get("doc", d).get("title")
                for d in self.all_session_docs
            }
            for d in docs[:20]:
                doc_id = d.get("doc", d).get("id") or d.get("doc", d).get("title")
                if doc_id not in existing_ids:
                    self.all_session_docs.append(d)
                    existing_ids.add(doc_id)
            if len(self.all_session_docs) > 100:
                self.all_session_docs = self.all_session_docs[-100:]

    def clear_filter(self, filter_name: str):
        """Xóa một bộ lọc cụ thể (năm, ngôn ngữ, loại tài liệu, ngành)."""
        fn = filter_name.lower().strip()
        if "year" in fn or "năm" in fn:
            self.year_min = None
            self.year_max = None
        elif "lang" in fn or "tiếng" in fn or "ngôn ngữ" in fn:
            self.language = None
        elif "doc_type" in fn or "loại" in fn:
            self.doc_types = []
        elif "major" in fn or "ngành" in fn:
            self.major = None
        elif "author" in fn or "tác giả" in fn:
            self.author = None
        elif "all" in fn or "tất cả" in fn:
            self.year_min = None
            self.year_max = None
            self.language = None
            self.doc_types = []
            self.major = None
            self.author = None


# Global in-memory storage for dialogue states (LRU)
_dialogue_state_store: OrderedDict[str, ConversationState] = OrderedDict()


def get_dialogue_state(session_id: str) -> ConversationState:
    """Lấy ConversationState của session, nếu chưa có thì khởi tạo mới."""
    if not session_id:
        return ConversationState(session_id="")
    if session_id in _dialogue_state_store:
        _dialogue_state_store.move_to_end(session_id)
        return _dialogue_state_store[session_id]
    state = ConversationState(session_id=session_id)
    _dialogue_state_store[session_id] = state
    while len(_dialogue_state_store) > MAX_DST_SESSIONS:
        _dialogue_state_store.popitem(last=False)
    return state


def save_dialogue_state(session_id: str, state: ConversationState) -> None:
    """Lưu trữ ConversationState của session."""
    if not session_id:
        return
    state.updated_at = time.time()
    _dialogue_state_store[session_id] = state
    _dialogue_state_store.move_to_end(session_id)
    while len(_dialogue_state_store) > MAX_DST_SESSIONS:
        _dialogue_state_store.popitem(last=False)

