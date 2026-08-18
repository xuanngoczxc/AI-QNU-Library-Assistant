import React from 'react';
import { Menu, Trash2, Sun, Moon } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';

interface ChatHeaderProps {
  onToggleSidebar: () => void;
  onClearChat: () => void;
  onOpenBrowseModal?: () => void;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  onToggleSidebar,
  onClearChat,
}) => {
  const { theme, toggleTheme } = useTheme();

  return (
    <header className="chat-header">
      <div className="chat-header-main">
        <button
          type="button"
          className="hamburger"
          onClick={onToggleSidebar}
          title="Đóng / Mở Danh mục"
          aria-label="Toggle Menu"
        >
          <Menu size={19} strokeWidth={2} />
        </button>

        <div className="header-info">
          <h2 className="header-main-title">Tra cứu Sách QNU</h2>
          <p className="header-sub-desc">Trung tâm Số và Học liệu · Đại học Quy Nhơn</p>
        </div>
      </div>

      <div className="header-actions">
        <button
          type="button"
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={theme === 'dark' ? 'Chuyển sang Giao diện Sáng' : 'Chuyển sang Giao diện Tối'}
          aria-label="Toggle Light/Dark Theme"
        >
          {theme === 'dark' ? (
            <Sun size={17} strokeWidth={2} color="#FFD271" className="theme-icon sun" />
          ) : (
            <Moon size={17} strokeWidth={1.9} className="theme-icon moon" />
          )}
        </button>

        <button
          type="button"
          className="clear-btn"
          onClick={onClearChat}
          title="Xóa đoạn chat hiện tại"
          aria-label="Xóa đoạn chat"
        >
          <Trash2 size={17} strokeWidth={1.8} />
        </button>
      </div>
    </header>
  );
};
