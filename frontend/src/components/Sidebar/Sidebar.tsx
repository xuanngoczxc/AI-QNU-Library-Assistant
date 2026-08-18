import React from 'react';
import { BrandHeader } from './BrandHeader';
import { FacultyTree } from './FacultyTree';
import { AdminLoginBar } from './AdminLoginBar';
import type { FacultyItem } from '../../types';

import { Plus } from 'lucide-react';

interface SidebarProps {
  faculties: FacultyItem[];
  facultyLoading: boolean;
  facultyError: string | null;
  selectedMajor: string | null;
  selectedKhoa: string | null;
  openGroups: Record<string, boolean>;
  onToggleGroup: (khoa: string) => void;
  onSelectMajor: (nganh: string, khoa: string) => void;
  onClearMajor: () => void;
  onNewChat: () => void;
  onShowToast: (msg: string, type?: 'error' | 'success') => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  faculties,
  facultyLoading,
  facultyError,
  selectedMajor,
  selectedKhoa,
  openGroups,
  onToggleGroup,
  onSelectMajor,
  onClearMajor,
  onNewChat,
  onShowToast,
  mobileOpen,
  onCloseMobile,
}) => {
  return (
    <>
      <div
        className={`sidebar-backdrop ${mobileOpen ? 'show' : ''}`}
        onClick={onCloseMobile}
      />
      <aside className={`sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-scroll">
          <div className="sticky-header">
            <BrandHeader />
            <button type="button" className="new-chat-btn" onClick={onNewChat}>
              <Plus size={18} strokeWidth={2.5} />
              <span>Đoạn chat mới</span>
            </button>
          </div>

          <FacultyTree
            faculties={faculties}
            loading={facultyLoading}
            error={facultyError}
            selectedMajor={selectedMajor}
            selectedKhoa={selectedKhoa}
            openGroups={openGroups}
            onToggleGroup={onToggleGroup}
            onSelectMajor={(nganh, khoa) => {
              onSelectMajor(nganh, khoa);
              onCloseMobile();
            }}
            onClearMajor={onClearMajor}
          />
        </div>

        <AdminLoginBar onShowToast={onShowToast} />
      </aside>
    </>
  );
};
