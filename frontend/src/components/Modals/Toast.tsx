import React from 'react';

interface ToastProps {
  message: string;
  type: 'error' | 'success';
  visible: boolean;
}

export const Toast: React.FC<ToastProps> = ({ message, type, visible }) => {
  if (!visible || !message) return null;

  return (
    <div className={`toast-msg ${type === 'success' ? 'success' : ''}`}>
      {message}
    </div>
  );
};
