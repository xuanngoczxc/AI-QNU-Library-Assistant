import React, { useState } from 'react';
import type { ChatMessage } from '../../types';
import { renderMarkdown } from '../../services/markdown';
import { TypingIndicator } from './TypingIndicator';
import { SourcesList } from './SourcesList';
import { FollowUpChips } from './FollowUpChips';
import { Copy, Check } from 'lucide-react';

interface MessageBubbleProps {
  message: ChatMessage;
  onSelectSuggestion: (query: string) => void;
  onShowToast: (msg: string, type?: 'error' | 'success') => void;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  onSelectSuggestion,
  onShowToast,
}) => {
  const [copied, setCopied] = useState(false);

  const isUser = message.sender === 'user';
  const showTyping = !isUser && message.isStreaming && !message.text;
  const showCursor = !isUser && message.isStreaming && Boolean(message.text);

  const handleCopy = () => {
    if (!message.text) return;
    navigator.clipboard.writeText(message.text);
    setCopied(true);
    onShowToast('Đã sao chép câu trả lời!', 'success');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`msg ${isUser ? 'user' : 'bot'} ${message.isError ? 'error' : ''}`}>
      {!isUser && (
        <div className="avatar">
          <img src="/SHL-logo.png" alt="SHL" />
        </div>
      )}

      <div className="msg-body">
        <div className="bubble">
          {showTyping ? (
            <TypingIndicator />
          ) : (
            <>
              <div
                dangerouslySetInnerHTML={{
                  __html: renderMarkdown(message.text),
                }}
              />
              {showCursor && <span className="cursor" />}
            </>
          )}

          {/* Render Book Sources list with pagination (suppressed if no info found) */}
          {!Boolean(
            message.text &&
              /(tài liệu không cung cấp thông tin|chưa tìm thấy|không tìm thấy|không có thông tin|không lưu trữ|chưa có tài liệu)/i.test(
                message.text
              )
          ) &&
            message.sources &&
            message.sources.length > 0 && (
              <SourcesList
                sources={message.sources}
                totalFound={message.totalFound}
              />
            )}

          {/* Render Follow-up suggestions */}
          {message.followUps && message.followUps.length > 0 && (
            <FollowUpChips
              followUps={message.followUps}
              onSelectSuggestion={onSelectSuggestion}
            />
          )}
        </div>

        {/* Copy button & timestamp */}
        <div className="msg-actions">
          {!isUser && !message.isStreaming && message.text && (
            <button type="button" className="copy-btn" onClick={handleCopy}>
              {copied ? (
                <>
                  <Check size={12} color="var(--success)" />
                  <span style={{ color: 'var(--success)' }}>Đã copy</span>
                </>
              ) : (
                <>
                  <Copy size={12} />
                  <span>Copy</span>
                </>
              )}
            </button>
          )}
          <div className="msg-time">{message.timestamp}</div>
        </div>
      </div>
    </div>
  );
};
