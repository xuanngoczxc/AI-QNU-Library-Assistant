import React from 'react';

export const TypingIndicator: React.FC = () => {
  return (
    <div className="typing-indicator" aria-label="AI đang soạn câu trả lời">
      <span></span>
      <span></span>
      <span></span>
    </div>
  );
};
