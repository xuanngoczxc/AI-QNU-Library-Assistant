import React, { useRef, useEffect, useState, useCallback } from 'react';
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
  const bottomAnchorRef = useRef<HTMLDivElement>(null);
  const [showScrollBtn, setShowScrollBtn] = useState(false);
  const prevMsgLengthRef = useRef(0);
  const userScrolledUpRef = useRef(false);

  const scrollToBottom = useCallback((behavior: ScrollBehavior = 'smooth') => {
    if (bottomAnchorRef.current) {
      bottomAnchorRef.current.scrollIntoView({ behavior });
    } else if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior,
      });
    }
    setShowScrollBtn(false);
    userScrolledUpRef.current = false;
  }, []);

  const handleScroll = () => {
    if (!containerRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const diff = scrollHeight - clientHeight - scrollTop;
    const isUp = diff > 100;
    setShowScrollBtn(isUp);
    userScrolledUpRef.current = isUp;
  };

  // 1. CHỈ KHI TÌM KIẾM MỚI (có tin nhắn mới được gửi đi hoặc nhận về) -> Cuộn xuống đáy
  useEffect(() => {
    if (messages.length > prevMsgLengthRef.current) {
      userScrolledUpRef.current = false;
      scrollToBottom('smooth');
      const timer = setTimeout(() => scrollToBottom('smooth'), 120);
      prevMsgLengthRef.current = messages.length;
      return () => clearTimeout(timer);
    }
    prevMsgLengthRef.current = messages.length;
  }, [messages.length, scrollToBottom]);

  // 2. Khi bot đang STREAMING chữ -> Cuộn theo dòng văn bản nếu người dùng chưa scroll lên
  const lastMsg = messages[messages.length - 1];
  const isStreaming = lastMsg?.isStreaming;
  useEffect(() => {
    if (isStreaming && !userScrolledUpRef.current) {
      scrollToBottom('auto');
    }
  }, [isStreaming, lastMsg?.text, scrollToBottom]);

  // Không dùng ResizeObserver để tránh tự cuộn khi người dùng click mở/xổ accordion thẻ sách & loại tài liệu

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

        {/* Anchor point at bottom of message list */}
        <div ref={bottomAnchorRef} style={{ height: '1px', width: '100%', flexShrink: 0 }} />
      </div>

      <ScrollDownBtn show={showScrollBtn} onClick={() => scrollToBottom('smooth')} />
    </div>
  );
};
