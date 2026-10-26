import React from 'react';
import type { FollowUpSuggestion } from '../../types';
import {
  BookOpen,
  Book,
  GraduationCap,
  Library,
  FileText,
  Search,
  MapPin,
  User,
  Sparkles,
} from 'lucide-react';

interface FollowUpChipsProps {
  followUps: FollowUpSuggestion[];
  onSelectSuggestion: (query: string) => void;
}

// Hàm gán icon chuẩn từ thư viện lucide-react dựa trên loại sách / nội dung
function getFollowUpIcon(text: string) {
  const low = text.toLowerCase();
  if (low.includes('tác giả')) return <User size={14} className="follow-up-icon" />;
  if (low.includes('ngành') || low.includes('khoa') || low.includes('chuyên ngành'))
    return <Library size={14} className="follow-up-icon" />;
  if (low.includes('chủ đề')) return <BookOpen size={14} className="follow-up-icon" />;
  if (low.includes('giáo trình')) return <Book size={14} className="follow-up-icon" />;
  if (low.includes('luận văn') || low.includes('luận án') || low.includes('tốt nghiệp'))
    return <GraduationCap size={14} className="follow-up-icon" />;
  if (low.includes('phòng') || low.includes('vị trí') || low.includes('ở đâu'))
    return <MapPin size={14} className="follow-up-icon" />;
  if (low.includes('tóm tắt') || low.includes('bài tập'))
    return <FileText size={14} className="follow-up-icon" />;
  if (low.includes('tham khảo') || low.includes('sách'))
    return <BookOpen size={14} className="follow-up-icon" />;
  if (low.includes('phổ biến') || low.includes('gợi ý'))
    return <Sparkles size={14} className="follow-up-icon" />;
  return <Search size={14} className="follow-up-icon" />;
}

// Loại bỏ các emoji có sẵn nếu còn sót lại từ chuỗi label
function cleanChipLabel(str: string): string {
  if (!str) return '';
  return str
    .replace(
      /([\u2700-\u27BF]|[\uE000-\uF8FF]|\uD83C[\uDC00-\uDFFF]|\uD83D[\uDC00-\uDFFF]|[\u2011-\u26FF]|\uD83E[\uDD10-\uDDFF])/g,
      ''
    )
    .trim();
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
        {followUps.map((chip, idx) => {
          const label = cleanChipLabel(chip.label || '');
          if (!label) return null;
          return (
            <button
              type="button"
              key={idx}
              className="follow-up-chip"
              onClick={() => onSelectSuggestion(chip.query)}
            >
              {getFollowUpIcon(label)}
              <span>{label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
};
