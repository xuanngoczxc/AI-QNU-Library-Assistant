import { useState, useRef, useCallback } from 'react';
import type { ChatMessage, BookSource, FollowUpSuggestion } from '../types';
import { fetchChatStream } from '../services/api';

function getCurrentTime(): string {
  return new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
}

export function useChatStream() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [statusText, setStatusText] = useState<string>('Sẵn sàng');
  const [statusType, setStatusType] = useState<'ok' | 'loading' | 'error'>('ok');
  const [sessionId] = useState<string>(() => 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7));

  const abortControllerRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(async (queryText: string) => {
    const trimmed = queryText.trim();
    if (!trimmed || isStreaming) return;

    // Display formatted query if list-topic marker
    let displayQuery = trimmed;
    const listTopicPrefix = '__LIST_TOPIC__:';
    if (displayQuery.startsWith(listTopicPrefix)) {
      const topicName = displayQuery.slice(listTopicPrefix.length).trim();
      displayQuery = 'Liệt kê chi tiết về ' + topicName;
    }

    const userMsg: ChatMessage = {
      id: 'user_' + Date.now(),
      sender: 'user',
      text: displayQuery,
      timestamp: getCurrentTime(),
    };

    const botMsgId = 'bot_' + Date.now();
    const botMsgPlaceholder: ChatMessage = {
      id: botMsgId,
      sender: 'bot',
      text: '',
      timestamp: getCurrentTime(),
      isStreaming: true,
      sources: [],
      followUps: [],
    };

    setMessages((prev) => [...prev, userMsg, botMsgPlaceholder]);
    setIsStreaming(true);

    if (trimmed.startsWith('__LIST_TOPIC__:')) {
      setStatusText('Đang tìm tài liệu, có thể tốn một chút thời gian, vui lòng chờ...');
    } else {
      setStatusText('Đang trả lời...');
    }
    setStatusType('loading');

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    let accumulatedText = '';
    let fetchedSources: BookSource[] = [];
    let fetchedTotal = 0;
    let fetchedFollowUps: FollowUpSuggestion[] = [];

    try {
      await fetchChatStream(trimmed, sessionId, abortController.signal, {
        onChunk: (chunk) => {
          accumulatedText += chunk;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMsgId
                ? { ...msg, text: accumulatedText, isStreaming: true }
                : msg
            )
          );
        },
        onSources: (sources) => {
          fetchedSources = sources;
        },
        onTotalFound: (total) => {
          fetchedTotal = total;
        },
        onFollowUps: (followUps) => {
          fetchedFollowUps = followUps;
        },
        onDone: () => {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMsgId
                ? {
                    ...msg,
                    text: accumulatedText,
                    isStreaming: false,
                    sources: fetchedSources,
                    totalFound: fetchedTotal,
                    followUps: fetchedFollowUps,
                  }
                : msg
            )
          );
        },
        onError: (errMsg) => {
          throw new Error(errMsg);
        },
      });

      setStatusText('Sẵn sàng');
      setStatusType('ok');
    } catch (err: any) {
      if (err.name === 'AbortError' || err.message === 'user_stop') {
        setStatusText('Đã dừng');
        setStatusType('ok');
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === botMsgId
              ? { ...msg, isStreaming: false }
              : msg
          )
        );
      } else {
        const errorDetails = err.message || 'Lỗi không xác định khi kết nối với máy chủ.';
        setStatusText('Lỗi');
        setStatusType('error');
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === botMsgId
              ? {
                  ...msg,
                  text: accumulatedText || `⚠️ Đã xảy ra lỗi: ${errorDetails}`,
                  isStreaming: false,
                  isError: true,
                }
              : msg
          )
        );
      }
    } finally {
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  }, [isStreaming, sessionId]);

  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort('user_stop');
      abortControllerRef.current = null;
    }
  }, []);

  const clearChat = useCallback(() => {
    stopStreaming();
    setMessages([]);
    setStatusText('Sẵn sàng');
    setStatusType('ok');
  }, [stopStreaming]);

  return {
    messages,
    isStreaming,
    statusText,
    statusType,
    sendMessage,
    stopStreaming,
    clearChat,
  };
}
