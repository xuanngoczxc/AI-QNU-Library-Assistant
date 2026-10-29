import React, { useState, useMemo } from 'react';
import type { BookSource } from '../../types';
import { BookCard } from './BookCard';
import { normalizeDocType } from '../../utils/docType';
import {
  BookOpen,
  Book,
  FileCheck,
  GraduationCap,
  Library,
  FileText,
  FlaskConical,
  Newspaper,
  Folder,
  ChevronRight,
  ChevronLeft,
} from 'lucide-react';

interface SourcesListProps {
  sources: BookSource[];
  totalFound?: number;
}

const PAGE_SIZE = 15;

function getDocTypeIcon(dt: string) {
  const low = dt.toLowerCase();
  if (low.includes('sách') || low.includes('book')) return <BookOpen size={16} />;
  if (low.includes('giáo trình')) return <Book size={16} />;
  if (low.includes('kiểm định')) return <FileCheck size={16} />;
  if (low.includes('luận văn')) return <GraduationCap size={16} />;
  if (low.includes('tham khảo') || low.includes('chính')) return <Library size={16} />;
  if (low.includes('pdf') || low.includes('tài liệu')) return <FileText size={16} />;
  if (low.includes('đề tài') || low.includes('nghiên cứu')) return <FlaskConical size={16} />;
  if (low.includes('báo') || low.includes('tạp chí')) return <Newspaper size={16} />;
  return <Folder size={16} />;
}

function getVisiblePages(currentPage: number, totalPages: number): (number | string)[] {
  if (totalPages <= 7) {
    return Array.from({ length: totalPages }, (_, i) => i + 1);
  }
  if (currentPage <= 4) {
    return [1, 2, 3, 4, 5, '...', totalPages];
  }
  if (currentPage >= totalPages - 3) {
    return [1, '...', totalPages - 4, totalPages - 3, totalPages - 2, totalPages - 1, totalPages];
  }
  return [1, '...', currentPage - 1, currentPage, currentPage + 1, '...', totalPages];
}

export const SourcesList: React.FC<SourcesListProps> = ({ sources, totalFound }) => {
  if (!sources || sources.length === 0) return null;

  // 1. Nhóm các quyển sách theo loại tài liệu (doc_type) với suy luận thông minh
  const groupedSources = useMemo(() => {
    const map: Record<string, BookSource[]> = {};
    for (const book of sources) {
      const dt = normalizeDocType(book.doc_type || book.loai_tai_lieu, book);
      if (!map[dt]) map[dt] = [];
      map[dt].push(book);
    }
    // Sắp xếp các nhóm theo số lượng giảm dần
    return Object.entries(map).sort((a, b) => b[1].length - a[1].length);
  }, [sources]);

  // 2. State mở/đóng accordion cho từng loại tài liệu (Mặc định mở nhóm đầu tiên)
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(() => {
    const init: Record<string, boolean> = {};
    groupedSources.forEach(([dtName], idx) => {
      init[dtName] = idx === 0; // Mặc định mở nhóm đầu tiên
    });
    return init;
  });

  // 3. State trang hiện tại của từng nhóm
  const [groupPages, setGroupPages] = useState<Record<string, number>>({});

  const toggleGroup = (dtName: string) => {
    setOpenGroups((prev) => ({
      ...prev,
      [dtName]: !prev[dtName],
    }));
  };

  const setPage = (dtName: string, newPage: number) => {
    setGroupPages((prev) => ({
      ...prev,
      [dtName]: newPage,
    }));
  };

  return (
    <div className="sources-container">
      <div className="sources-header">
        <BookOpen size={16} />
        <span>
          {totalFound && totalFound > sources.length
            ? `Danh sách tài liệu liên quan (hiển thị ${sources.length} / ${totalFound} kết quả):`
            : `Danh sách tài liệu liên quan (${sources.length} kết quả):`}
        </span>
      </div>

      <div className="sources-grid">
        {groupedSources.map(([dtName, items], gIdx) => {
          const isOpen = Boolean(openGroups[dtName]);
          const currentPage = groupPages[dtName] || 1;
          const totalPages = Math.ceil(items.length / PAGE_SIZE);

          const startIdx = (currentPage - 1) * PAGE_SIZE;
          const endIdx = Math.min(startIdx + PAGE_SIZE, items.length);
          const currentItems = items.slice(startIdx, endIdx);

          return (
            <div key={dtName} className="doc-group-section">
              <button
                type="button"
                className={`doc-group-header ${isOpen ? 'open' : ''}`}
                onClick={() => toggleGroup(dtName)}
                aria-expanded={isOpen}
              >
                <ChevronRight size={16} className="doc-group-chevron" />
                <span className="doc-group-icon">{getDocTypeIcon(dtName)}</span>
                <span className="doc-group-title">Loại tài liệu: {dtName}</span>
                <span className="doc-group-badge">{items.length}</span>
              </button>

              {isOpen && (
                <div className="doc-group-body">
                  {currentItems.map((book, idx) => (
                    <BookCard key={gIdx + '-' + (startIdx + idx)} book={book} />
                  ))}

                  {totalPages > 1 && (
                    <div className="sources-pagination">
                      <button
                        type="button"
                        className="sources-page-btn nav-btn"
                        onClick={() => setPage(dtName, currentPage - 1)}
                        disabled={currentPage <= 1}
                      >
                        <ChevronLeft size={13} />
                        <span>Trước</span>
                      </button>

                      <div className="sources-page-numbers">
                        {getVisiblePages(currentPage, totalPages).map((p, idx) => {
                          if (p === '...') {
                            return (
                              <span key={`dots-${idx}`} className="sources-page-dots">
                                ...
                              </span>
                            );
                          }
                          const pageNum = p as number;
                          return (
                            <button
                              type="button"
                              key={pageNum}
                              className={`sources-page-btn num-btn ${pageNum === currentPage ? 'active' : ''}`}
                              onClick={() => setPage(dtName, pageNum)}
                            >
                              {pageNum}
                            </button>
                          );
                        })}
                      </div>

                      <button
                        type="button"
                        className="sources-page-btn nav-btn"
                        onClick={() => setPage(dtName, currentPage + 1)}
                        disabled={currentPage >= totalPages}
                      >
                        <span>Sau</span>
                        <ChevronRight size={13} />
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
