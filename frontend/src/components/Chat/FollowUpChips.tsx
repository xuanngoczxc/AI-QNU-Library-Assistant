import React from 'react';
import type { FollowUpSuggestion } from '../../types';

interface FollowUpChipsProps {
  followUps: FollowUpSuggestion[];
  onSelectSuggestion: (query: string) => void;
}

export const FollowUpChips: React.FC<FollowUpChipsProps> = ({
  followUps,
  onSelectSuggestion,
}) => {
  if (!followUps || followUps.length === 0) return null;

  return (
    <div className="follow-up-container">
      <div className="follow-up-title">Gợi ý câu hỏi tiếp theo:</div>
      <div className="follow-up-chips">
        {followUps.map((chip, idx) => (
          <button
            type="button"
            key={idx}
            className="follow-up-chip"
            onClick={() => onSelectSuggestion(chip.query)}
          >
            {chip.label}
          </button>
        ))}
      </div>
    </div>
  );
};
