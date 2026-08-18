import React from 'react';
import type { FacultyItem } from '../../types';

import { ChevronDown, X } from 'lucide-react';

interface FacultyTreeProps {
  faculties: FacultyItem[];
  loading: boolean;
  error: string | null;
  selectedMajor: string | null;
  selectedKhoa: string | null;
  openGroups: Record<string, boolean>;
  onToggleGroup: (khoa: string) => void;
  onSelectMajor: (nganh: string, khoa: string) => void;
  onClearMajor: () => void;
}

export const FacultyTree: React.FC<FacultyTreeProps> = ({
  faculties,
  loading,
  error,
  selectedMajor,
  selectedKhoa,
  openGroups,
  onToggleGroup,
  onSelectMajor,
  onClearMajor,
}) => {
  return (
    <div>
      <div className="section-title">Danh mục Khoa &amp; Ngành</div>

      {selectedMajor && (
        <button className="faculty-clear show" onClick={onClearMajor}>
          <X size={12} strokeWidth={2.5} />
          <span>{selectedKhoa ? `${selectedKhoa} – ${selectedMajor}` : selectedMajor} (Xóa)</span>
        </button>
      )}

      <div className="faculty-tree">
        {loading && <div className="faculty-tree-loading">Đang tải danh mục khoa...</div>}
        {error && <div className="faculty-tree-loading" style={{ color: 'var(--error)' }}>Lỗi: {error}</div>}
        {!loading && !error && faculties.length === 0 && (
          <div className="faculty-tree-loading">Chưa có dữ liệu khoa.</div>
        )}

        {!loading && faculties.map((f, idx) => {
          const isOpen = Boolean(openGroups[f.khoa]);
          const kId = `fk_${idx}`;
          const sortedMajors = (f.nganh || []).slice().sort((a, b) => (b.count || 0) - (a.count || 0));

          return (
            <div key={f.khoa} className={`faculty-group ${isOpen ? 'open' : ''}`}>
              <button
                type="button"
                className={`faculty-header ${isOpen ? 'open' : ''}`}
                onClick={() => onToggleGroup(f.khoa)}
                aria-expanded={isOpen}
                aria-controls={kId}
              >
                <ChevronDown size={14} className="faculty-chevron" />
                <span className="faculty-name">{f.khoa}</span>
              </button>

              <div className="faculty-children" id={kId}>
                {sortedMajors.map((ng) => {
                  const isActive = selectedMajor === ng.nganh;
                  return (
                    <button
                      type="button"
                      key={ng.nganh}
                      className={`faculty-row ${isActive ? 'active' : ''}`}
                      onClick={() => onSelectMajor(ng.nganh, f.khoa)}
                    >
                      <span className="name">{ng.nganh}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
