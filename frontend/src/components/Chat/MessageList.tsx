import React, { useRef, useEffect, useState } from 'react';
import type { ChatMessage } from '../../types';
import { MessageBubble } from './MessageBubble';
import { WelcomeBanner } from './WelcomeBanner';
import { ScrollDownBtn } from './ScrollDownBtn';

interface MessageListProps {
  messages: ChatMessage[];
  onSelectTopic: (query: string) => void;
  onShowToast: (msg: string, type?: 'error' | 'success') => void;
}

export const MessageList: React.FC<MessageListProps> = ({
  messages,
  onSelectTopic,
  onShowToast,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior,
      });
      setShowScrollBtn(false);
    }
  };

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const diff = scrollHeight - clientHeight - scrollTop;
    setShowScrollBtn(diff > 120);
  };

  // Tự động cuộn xuống khi có tin nhắn mới hoặc streaming
  useEffect(() => {
    if (!showScrollBtn) {
      scrollToBottom('auto');
    }
  }, [messages, showScrollBtn]);

  return (
    <div className="messages-wrap">
      <div className="messages" ref={containerRef} onScroll={handleScroll}>
        <WelcomeBanner onSelectTopic={onSelectTopic} />

        {messages.map((msg) => (
          <MessageBubble
            key={msg.id}
            message={msg}
            onSelectSuggestion={onSelectTopic}
            onShowToast={onShowToast}
          />
        ))}
      </div>

      <ScrollDownBtn show={showScrollBtn} onClick={() => scrollToBottom('smooth')} />
    </div>
  );
};
