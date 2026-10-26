import React, { useState, useEffect } from 'react';
import type { BookSource } from '../../types';
import { fetchBrowseDocuments, searchDocuments } from '../../services/api';
import { BookCard } from '../Chat/BookCard';
import { X, BookOpen } from 'lucide-react';

interface BrowseModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const BrowseModal: React.FC<BrowseModalProps> = ({ isOpen, onClose }) => {
  const [documents, setDocuments] = useState<BookSource[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');

  const limit = 20;

  useEffect(() => {
    if (!isOpen) return;

    let isMounted = true;
    setLoading(true);
    setError(null);

    if (!searchTerm.trim()) {
      fetchBrowseDocuments(page, limit)
        .then((data) => {
          if (isMounted) {
            setDocuments(data.results || []);
            setTotal(data.total || 0);
            setLoading(false);
          }
        })
        .catch((err) => {
          if (isMounted) {
            setError(err.message);
            setLoading(false);
          }
        });
    } else {
      const timer = setTimeout(() => {
        searchDocuments(searchTerm.trim(), 'browse_search')
          .then((data) => {
            if (isMounted) {
              setDocuments(data.sources || []);
              setTotal(data.sources?.length || 0);
              setLoading(false);
            }
          })
          .catch((err) => {
            if (isMounted) {
              setError(err.message);
              setLoading(false);
            }
          });
      }, 350);

      return () => clearTimeout(timer);
    }

    return () => {
      isMounted = false;
    };
  }, [isOpen, page, searchTerm]);

  if (!isOpen) return null;

  const totalPages = Math.ceil(total / limit) || 1;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BookOpen size={20} color="var(--primary)" />
            <span>Danh mục Tài liệu Thư viện ({total} tài liệu)</span>
          </h3>
          <button type="button" className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-search-bar">
          <input
            type="text"
            className="modal-search-input"
            placeholder="Tìm kiếm nhanh tiêu đề, tác giả, chuyên ngành..."
            value={searchTerm}
            onChange={(e) => {
              setSearchTerm(e.target.value);
              setPage(1);
            }}
          />
        </div>

        <div className="modal-body">
          {loading && <div style={{ padding: '20px', textAlign: 'center' }}>Đang tải danh mục...</div>}
          {error && <div style={{ padding: '20px', color: 'var(--error)', textAlign: 'center' }}>Lỗi: {error}</div>}
          {!loading && !error && documents.length === 0 && (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--on-surface-tertiary)' }}>
              Không tìm thấy tài liệu phù hợp.
            </div>
          )}

          {!loading &&
            documents.map((book, idx) => (
              <BookCard key={idx} book={book} index={(page - 1) * limit + idx} />
            ))}
        </div>

        {!searchTerm && totalPages > 1 && (
          <div className="modal-footer">
            <button
              type="button"
              className="pagination-btn"
              onClick={() => setPage(1)}
              disabled={page <= 1 || loading}
            >
              Đầu
            </button>
            <button
              type="button"
              className="pagination-btn"
              onClick={() => setPage((p) => Math.max(p - 1, 1))}
              disabled={page <= 1 || loading}
            >
              ‹
            </button>
            <span className="pagination-info">
              Trang {page} / {totalPages}
            </span>
            <button
              type="button"
              className="pagination-btn"
              onClick={() => setPage((p) => Math.min(p + 1, totalPages))}
              disabled={page >= totalPages || loading}
            >
              ›
            </button>
            <button
              type="button"
              className="pagination-btn"
              onClick={() => setPage(totalPages)}
              disabled={page >= totalPages || loading}
            >
              Cuối
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
