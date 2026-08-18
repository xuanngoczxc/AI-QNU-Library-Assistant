import React from 'react';
import { Menu, Trash2 } from 'lucide-react';

interface ChatHeaderProps {
  onToggleSidebar: () => void;
  onClearChat: () => void;
  onOpenBrowseModal?: () => void;
}

export const ChatHeader: React.FC<ChatHeaderProps> = ({
  onToggleSidebar,
  onClearChat,
}) => {
  return (
    <header className="chat-header">
      <button
        type="button"
        className="hamburger"
        onClick={onToggleSidebar}
        title="Đóng / Mở Menu"
        aria-label="Toggle Menu"
      >
        <Menu size={20} strokeWidth={2} />
      </button>

      <div className="chat-header-left">
        <div className="eyebrow">Trợ lý ảo Thư viện</div>
        <h2>Tra cứu Sách QNU</h2>
      </div>

      <div className="header-actions">
        <button
          type="button"
          className="clear-btn"
          onClick={onClearChat}
          title="Xóa đoạn chat hiện tại"
          aria-label="Xóa đoạn chat"
        >
          <Trash2 size={16} strokeWidth={1.8} />
        </button>
      </div>
    </header>
  );
};
