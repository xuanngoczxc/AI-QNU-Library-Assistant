import React, { useState, useRef, useEffect } from 'react';
import { useVoiceInput } from '../../hooks/useVoiceInput';
import { Mic, Send, Square } from 'lucide-react';

interface ComposerProps {
  isStreaming: boolean;
  onSendMessage: (text: string) => void;
  onStopStreaming: () => void;
  onShowToast: (msg: string, type?: 'error' | 'success') => void;
  selectedMajor: string | null;
}

export const Composer: React.FC<ComposerProps> = ({
  isStreaming,
  onSendMessage,
  onStopStreaming,
  onShowToast,
  selectedMajor,
}) => {
  const [inputText, setInputText] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isComposingRef = useRef(false);

  const { isRecording, toggleRecording } = useVoiceInput(
    (transcribedText) => {
      setInputText(transcribedText);
      onShowToast('Đã nhận diện giọng nói thành công!', 'success');
      setTimeout(() => {
        onSendMessage(transcribedText);
        setInputText('');
      }, 500);
    },
    (errMsg) => {
      onShowToast(errMsg, 'error');
    }
  );

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollH = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(Math.max(scrollH, 24), 140)}px`;
    }
  }, [inputText]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || isStreaming) return;
    onSendMessage(inputText);
    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Không submit khi đang gõ IME (Telex/VNI)
    if (isComposingRef.current || e.nativeEvent.isComposing || e.keyCode === 229) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
      }
      return;
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="composer-wrap">
      <form className="composer-form" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="composer-textarea"
          rows={1}
          placeholder={
            selectedMajor
              ? `Hỏi tài liệu ngành ${selectedMajor}...`
              : 'Nhập câu hỏi tra cứu sách, giáo trình, vị trí...'
          }
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => {
            isComposingRef.current = true;
          }}
          onCompositionEnd={() => {
            isComposingRef.current = false;
          }}
        />

        <div className="composer-actions">
          <button
            type="button"
            className={`voice-btn ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            title={isRecording ? 'Dừng ghi âm' : 'Nhập bằng giọng nói (Voice)'}
            aria-label="Voice input"
          >
            <Mic size={18} />
          </button>

          {isStreaming ? (
            <button
              type="button"
              className="stop-btn"
              onClick={onStopStreaming}
              title="Dừng phản hồi"
              aria-label="Stop response"
            >
              <Square size={16} fill="currentColor" />
            </button>
          ) : (
            <button
              type="submit"
              className="send-btn"
              disabled={!inputText.trim()}
              title="Gửi câu hỏi (Enter)"
              aria-label="Send message"
            >
              <Send size={16} />
            </button>
          )}
        </div>
      </form>
      <div className="composer-hint">
        Nhấn <strong>Enter</strong> để gửi, <strong>Shift + Enter</strong> để xuống dòng
      </div>
    </div>
  );
};
