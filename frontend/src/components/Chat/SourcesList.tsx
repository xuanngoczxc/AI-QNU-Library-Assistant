import React, { useState } from 'react';
import type { BookSource } from '../../types';
import { BookCard } from './BookCard';
import { BookOpen, ChevronLeft, ChevronRight } from 'lucide-react';

interface SourcesListProps {
  sources: BookSource[];
  totalFound?: number;
}

const PAGE_SIZE = 15;

export const SourcesList: React.FC<SourcesListProps> = ({ sources, totalFound }) => {
  const [currentPage, setCurrentPage] = useState(1);

  if (!sources || sources.length === 0) return null;

  const totalItems = sources.length;
  const totalPages = Math.ceil(totalItems / PAGE_SIZE);

  // Lấy danh sách sách của trang hiện tại
  const startIndex = (currentPage - 1) * PAGE_SIZE;
  const endIndex = Math.min(startIndex + PAGE_SIZE, totalItems);
  const currentSources = sources.slice(startIndex, endIndex);

  const goToPage = (page: number) => {
    const p = Math.max(1, Math.min(page, totalPages));
    setCurrentPage(p);
  };

  return (
    <div className="sources-container">
      <div className="sources-header">
        <BookOpen size={16} />
        <span>
          Danh sách tài liệu liên quan ({totalFound || totalItems} kết quả):
        </span>
      </div>

      <div className="sources-grid">
        {currentSources.map((book, idx) => (
          <BookCard key={startIndex + idx} book={book} />
        ))}
      </div>

      {totalPages > 1 && (
        <div className="sources-pagination">
          <button
            type="button"
            className="sources-page-btn nav-btn"
            onClick={() => goToPage(currentPage - 1)}
            disabled={currentPage <= 1}
            title="Trang trước"
          >
            <ChevronLeft size={14} />
            <span>Trước</span>
          </button>

          <div className="sources-page-numbers">
            {Array.from({ length: totalPages }, (_, i) => i + 1).map((p) => {
              // Hiển thị thông minh nếu nhiều trang: trang đầu, trang cuối, và các trang lân cận trang hiện tại
              if (
                totalPages > 7 &&
                p !== 1 &&
                p !== totalPages &&
                Math.abs(p - currentPage) > 1
              ) {
                if (p === 2 || p === totalPages - 1) {
                  return (
                    <span key={p} className="sources-page-dots">
                      ...
                    </span>
                  );
                }
                return null;
              }

              const isActive = p === currentPage;
              return (
                <button
                  type="button"
                  key={p}
                  className={`sources-page-btn num-btn ${isActive ? 'active' : ''}`}
                  onClick={() => goToPage(p)}
                >
                  {p}
                </button>
              );
            })}
          </div>

          <button
            type="button"
            className="sources-page-btn nav-btn"
            onClick={() => goToPage(currentPage + 1)}
            disabled={currentPage >= totalPages}
            title="Trang sau"
          >
            <span>Sau</span>
            <ChevronRight size={14} />
          </button>
        </div>
      )}
    </div>
  );
};
