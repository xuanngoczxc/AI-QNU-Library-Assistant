"""
semantic_search.py — Giai đoạn 3: Tìm kiếm theo ngữ nghĩa
Dùng sentence-transformers (multilingual) kết hợp BM25 để tăng độ chính xác.

Tại sao cần?
- BM25 chỉ khớp từ khoá → "AI trong giáo dục" không tìm được "trí tuệ nhân tạo ứng dụng vào dạy học"
- Semantic search hiểu nghĩa → trả về kết quả liên quan dù khác từ

Cài: pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
"""

import os
import time
import pickle
import hashlib
import threading
from pathlib import Path
from typing import List, Dict, Optional
from text_utils import normalize_text

# Lazy import để tránh load model khi không dùng
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",  # ~120MB, hỗ trợ 50+ ngôn ngữ
)

# Cache file
_EMBEDDINGS_CACHE = Path("_semantic_embeddings.pkl")
_CACHE_VERSION = "v2_f16"  # float16 để giảm ~50% dung lượng cache


def _get_model():
    """Lazy load sentence-transformers model (chỉ load lần đầu)."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from sentence_transformers import SentenceTransformer
            print(f"  ⏳ Loading semantic model: {_MODEL_NAME}...")
            t0 = time.time()
            _MODEL = SentenceTransformer(_MODEL_NAME, device="cpu")
            print(f"  ✅ Model loaded in {time.time() - t0:.1f}s")
            return _MODEL
        except ImportError:
            print("  ⚠️ sentence-transformers not installed. Run:")
            print("     pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu")
            return None
        except Exception as e:
            print(f"  ⚠️ Failed to load semantic model: {e}")
            return None


def is_available() -> bool:
    """Check semantic search có khả dụng không."""
    return _get_model() is not None


def _doc_key(doc: Dict) -> str:
    """Tạo key cho doc để dùng trong cache."""
    text = (doc.get("text", "") or "") + "|" + (doc.get("title", "") or "")
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


class SemanticSearchEngine:
    """
    Semantic search kết hợp BM25.
    - Encode documents thành vectors 1 lần (cached)
    - Encode query → tìm top-k gần nhất bằng cosine similarity
    """

    def __init__(self, documents: List[Dict]):
        self.documents = documents
        self.embeddings = None
        self._cache_meta = None

        # Thử load cache
        if self._load_cache():
            print(f"  ✓ Semantic embeddings loaded from cache: {len(self.documents)} docs")
            return

        # Nếu không có cache, build lại
        self._build_embeddings()

    def _build_embeddings(self):
        """Encode tất cả documents thành vectors."""
        model = _get_model()
        if not model:
            return

        texts = []
        for doc in self.documents:
            # Kết hợp title + text để embedding capture đầy đủ ngữ nghĩa
            title = (doc.get("title", "") or "").strip()
            text = (doc.get("text", "") or "").strip()
            combined = f"{title}. {text}" if title else text
            # Giới hạn 512 ký tự để tiết kiệm (model có max_seq_length ~128-512)
            combined = combined[:512]
            texts.append(normalize_text(combined))

        if not texts:
            return

        print(f"  ⏳ Encoding {len(texts)} documents...")
        t0 = time.time()
        try:
            self.embeddings = model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,  # để dùng dot product thay cosine
            )
            print(f"  ✅ Encoded {len(texts)} docs in {time.time() - t0:.1f}s")
            self._save_cache()
        except Exception as e:
            print(f"  ⚠️ Encoding failed: {e}")
            self.embeddings = None

    def _save_cache(self):
        """Lưu embeddings ra disk để lần sau load nhanh."""
        if self.embeddings is None:
            return
        try:
            # Lưu float16 để tiết kiệm ~50% dung lượng cache.
            # Search sẽ cast về float32 trước khi dot-product.
            cache_data = {
                "version": _CACHE_VERSION,
                "model": _MODEL_NAME,
                "doc_keys": [_doc_key(d) for d in self.documents],
                "embeddings": self.embeddings.astype("float16"),
            }
            with open(_EMBEDDINGS_CACHE, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  💾 Saved semantic embeddings to {_EMBEDDINGS_CACHE}")
        except Exception as e:
            print(f"  ⚠️ Failed to save embeddings cache: {e}")

    def _load_cache(self) -> bool:
        """Load embeddings từ cache nếu hợp lệ."""
        if not _EMBEDDINGS_CACHE.exists():
            return False
        try:
            with open(_EMBEDDINGS_CACHE, "rb") as f:
                cache = pickle.load(f)

            if cache.get("version") != _CACHE_VERSION:
                return False
            if cache.get("model") != _MODEL_NAME:
                return False

            # Check docs có khớp không
            current_keys = [_doc_key(d) for d in self.documents]
            if cache.get("doc_keys") != current_keys:
                return False

            # Cache lưu float16 (~50% dung lượng). Cast về float32 để dot-product chính xác.
            emb = cache["embeddings"]
            if hasattr(emb, "dtype") and emb.dtype != "float32":
                emb = emb.astype("float32")
            self.embeddings = emb
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to load embeddings cache: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Semantic search: trả về top_k documents gần nhất với query.
        Mỗi result: {doc, score} với score ∈ [0, 1]
        """
        if self.embeddings is None or not self.documents:
            return []

        model = _get_model()
        if not model:
            return []

        try:
            # Cache lưu float16 (~50% dung lượng). Encode query cũng float16 rồi cast về float32 khi dot.
            query_emb = model.encode(
                [normalize_text(query)[:512]],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )[0].astype("float32")

            # Dot product (vì embeddings đã normalize → tương đương cosine)
            scores = self.embeddings @ query_emb

            # Top-k indices
            import numpy as np
            top_indices = np.argsort(-scores)[:top_k]

            results = []
            for idx in top_indices:
                if idx >= len(self.documents):
                    continue
                results.append({
                    "doc": self.documents[idx],
                    "score": float(scores[idx]),
                })
            return results
        except Exception as e:
            print(f"  ⚠️ Semantic search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        bm25_results: List[Dict],
        alpha: float = 0.5,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Kết hợp BM25 + Semantic.
        - alpha = 0.0: chỉ dùng BM25
        - alpha = 1.0: chỉ dùng semantic
        - alpha = 0.5: trung bình 2 nguồn

        Score kết hợp = alpha * semantic_score + (1 - alpha) * normalized_bm25_score
        """
        sem_results = self.search(query, top_k=len(self.documents))

        if not sem_results:
            return bm25_results[:top_k]

        # Build dict: doc_index → semantic_score
        sem_scores = {i: r["score"] for i, r in enumerate(sem_results)}

        # Normalize BM25 scores
        bm25_scores = [r.get("score", 0) for r in bm25_results]
        max_bm25 = max(bm25_scores) if bm25_scores and max(bm25_scores) > 0 else 1.0

        # Combine
        combined = []
        bm25_dict = {id(r["doc"]): r for r in bm25_results}

        # Add BM25 results
        for r in bm25_results:
            doc = r["doc"]
            doc_id = id(doc)
            bm25_norm = r.get("score", 0) / max_bm25
            sem_score = 0
            # Tìm semantic score tương ứng
            for i, sd in enumerate(self.documents):
                if id(sd) == doc_id and i in sem_scores:
                    sem_score = sem_scores[i]
                    break

            final_score = alpha * sem_score + (1 - alpha) * bm25_norm
            combined.append({
                "doc": doc,
                "score": final_score,
                "bm25_score": r.get("score", 0),
                "semantic_score": sem_score,
            })

        # Add semantic-only results (docs không có trong BM25 top)
        bm25_doc_ids = {id(r["doc"]) for r in bm25_results}
        for i, sem_r in enumerate(sem_results):
            if id(sem_r["doc"]) not in bm25_doc_ids:
                combined.append({
                    "doc": sem_r["doc"],
                    "score": alpha * sem_r["score"],
                    "bm25_score": 0,
                    "semantic_score": sem_r["score"],
                })

        # Sort và lấy top-k
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
