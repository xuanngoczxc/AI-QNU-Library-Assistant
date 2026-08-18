import React, { useRef } from 'react';
import { BarChart3, MessageSquare, Upload, LogOut } from 'lucide-react';

interface AdminSidebarProps {
  onUploadExcel: (file: File) => void;
  onLogout: () => void;
  onGoToChat: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export const AdminSidebar: React.FC<AdminSidebarProps> = ({
  onUploadExcel,
  onLogout,
  onGoToChat,
  mobileOpen,
  onCloseMobile,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onUploadExcel(file);
      e.target.value = '';
    }
  };

  return (
    <>
      <div
        className={`sidebar-backdrop ${mobileOpen ? 'show' : ''}`}
        onClick={onCloseMobile}
      />
      <aside className={`admin-sidebar ${mobileOpen ? 'open' : ''}`}>
        <div className="admin-sidebar-header">
          <div className="admin-brand">
            <img src="/SHL-logo.png" alt="SHL Logo" className="admin-brand-logo" />
            <div>
              <div className="admin-brand-title">Trung tâm Số và Học liệu</div>
              <div className="admin-brand-sub">Thư viện QNU</div>
            </div>
          </div>
          <span className="admin-badge">Quản trị hệ thống</span>
        </div>

        <nav className="admin-nav">
          <button type="button" className="admin-nav-item active">
            <BarChart3 size={17} />
            <span>Tổng quan</span>
          </button>

          <button type="button" className="admin-nav-item" onClick={onGoToChat}>
            <MessageSquare size={17} />
            <span>Giao diện Chat</span>
          </button>

          <button
            type="button"
            className="admin-nav-item"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload size={17} />
            <span>Cập nhật dữ liệu</span>
          </button>

          <button
            type="button"
            className="admin-nav-item logout"
            onClick={onLogout}
            style={{ marginTop: 'auto' }}
          >
            <LogOut size={17} />
            <span>Đăng xuất</span>
          </button>
        </nav>

        <input
          ref={fileInputRef}
          type="file"
          accept=".xlsx,.xls"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
      </aside>
    </>
  );
};
