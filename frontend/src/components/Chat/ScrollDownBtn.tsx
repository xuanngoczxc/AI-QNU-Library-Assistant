import React from 'react';
import { ArrowDown } from 'lucide-react';

interface ScrollDownBtnProps {
  show: boolean;
  onClick: () => void;
}

export const ScrollDownBtn: React.FC<ScrollDownBtnProps> = ({ show, onClick }) => {
  if (!show) return null;

  return (
    <button
      type="button"
      className="scroll-down-btn"
      onClick={onClick}
      title="Cuộn xuống dưới cùng"
      aria-label="Cuộn xuống dưới cùng"
    >
      <ArrowDown size={18} strokeWidth={2.2} />
    </button>
  );
};
