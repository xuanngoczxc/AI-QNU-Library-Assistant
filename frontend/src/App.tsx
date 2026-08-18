import React, { useState, useCallback } from 'react';
import { Sidebar } from './components/Sidebar/Sidebar';
import { ChatHeader } from './components/Chat/ChatHeader';
import { MessageList } from './components/Chat/MessageList';
import { Composer } from './components/Chat/Composer';
import { BrowseModal } from './components/Modals/BrowseModal';
import { Toast } from './components/Modals/Toast';
import { useChatStream } from './hooks/useChatStream';
import { useFacultyTree } from './hooks/useFacultyTree';

import './styles/global.css';
import './styles/sidebar.css';
import './styles/chat.css';
import './styles/book-card.css';
import './styles/modals.css';

export const App: React.FC = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [browseModalOpen, setBrowseModalOpen] = useState(false);

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: 'error' | 'success'; visible: boolean }>({
    message: '',
    type: 'success',
    visible: false,
  });

  const showToast = useCallback((msg: string, type: 'error' | 'success' = 'success') => {
    setToast({ message: msg, type, visible: true });
    setTimeout(() => {
      setToast((prev) => ({ ...prev, visible: false }));
    }, 3000);
  }, []);

  // Chat streaming hook
  const {
    messages,
    isStreaming,
    sendMessage,
    stopStreaming,
    clearChat,
  } = useChatStream();

  // Faculty tree hook
  const {
    faculties,
    loading: facultyLoading,
    error: facultyError,
    selectedMajor,
    selectedKhoa,
    openGroups,
    toggleGroup,
    selectMajor,
    clearMajor,
  } = useFacultyTree();

  const handleSelectMajor = useCallback(
    (nganh: string, khoa: string) => {
      selectMajor(nganh, khoa);
      sendMessage(`Tìm sách thuộc ngành ${nganh}`);
    },
    [selectMajor, sendMessage]
  );

  const handleToggleSidebar = () => {
    // Nếu màn hình mobile -> mở drawer, desktop -> toggle collapse
    if (window.innerWidth <= 768) {
      setMobileSidebarOpen((prev) => !prev);
    } else {
      setSidebarCollapsed((prev) => !prev);
    }
  };

  return (
    <div className={`shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <Sidebar
        faculties={faculties}
        facultyLoading={facultyLoading}
        facultyError={facultyError}
        selectedMajor={selectedMajor}
        selectedKhoa={selectedKhoa}
        openGroups={openGroups}
        onToggleGroup={toggleGroup}
        onSelectMajor={handleSelectMajor}
        onClearMajor={clearMajor}
        onNewChat={clearChat}
        onShowToast={showToast}
        mobileOpen={mobileSidebarOpen}
        onCloseMobile={() => setMobileSidebarOpen(false)}
      />

      <main className="chat-panel">
        <ChatHeader
          onToggleSidebar={handleToggleSidebar}
          onClearChat={clearChat}
          onOpenBrowseModal={() => setBrowseModalOpen(true)}
        />

        <MessageList
          messages={messages}
          onSelectTopic={(query) => sendMessage(query)}
          onShowToast={showToast}
        />

        <Composer
          isStreaming={isStreaming}
          onSendMessage={(text) => sendMessage(text)}
          onStopStreaming={stopStreaming}
          onShowToast={showToast}
          selectedMajor={selectedMajor}
        />
      </main>

      <BrowseModal
        isOpen={browseModalOpen}
        onClose={() => setBrowseModalOpen(false)}
      />

      <Toast
        message={toast.message}
        type={toast.type}
        visible={toast.visible}
      />
    </div>
  );
};

export default App;
