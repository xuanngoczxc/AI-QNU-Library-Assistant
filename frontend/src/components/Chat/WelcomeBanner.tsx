import React from 'react';
import {
  BookOpen,
  Bot,
  TrendingUp,
  Code2,
  Landmark,
  GraduationCap,
  Briefcase,
  Brain,
  Cpu,
} from 'lucide-react';

interface WelcomeBannerProps {
  onSelectTopic: (query: string) => void;
}

interface TopicItem {
  name: string;
  icon: React.ReactNode;
}

const QUICK_TOPICS: TopicItem[] = [
  { name: 'Trí tuệ nhân tạo', icon: <Bot size={14} color="#1A6B8A" /> },
  { name: 'Kinh tế vi mô', icon: <TrendingUp size={14} color="#10B981" /> },
  { name: 'Lập trình Python', icon: <Code2 size={14} color="#3B82F6" /> },
  { name: 'Lịch sử Việt Nam', icon: <Landmark size={14} color="#D97706" /> },
  { name: 'Giáo dục mầm non', icon: <GraduationCap size={14} color="#EC4899" /> },
  { name: 'Quản trị kinh doanh', icon: <Briefcase size={14} color="#8B5CF6" /> },
  { name: 'Tâm lý học đại cương', icon: <Brain size={14} color="#06B6D4" /> },
  { name: 'Kỹ thuật điện tử', icon: <Cpu size={14} color="#EA580C" /> },
];

export const WelcomeBanner: React.FC<WelcomeBannerProps> = ({ onSelectTopic }) => {
  return (
    <div className="welcome-banner">
      <div className="welcome-title">
        <BookOpen size={20} className="text-primary" />
        <span>Xin chào! Mình có thể giúp gì cho bạn?</span>
      </div>
      <p className="welcome-desc">
        Hệ thống hỗ trợ tra cứu thông tin sách, giáo trình, vị trí lưu trữ (phòng đọc, phòng mượn), mã xếp giá và gợi ý tài liệu học tập tại Thư viện Trường Đại học Quy Nhơn.
      </p>

      <div className="quick-topics">
        {QUICK_TOPICS.map((topic) => (
          <button
            type="button"
            key={topic.name}
            className="quick-topic-chip"
            onClick={() => onSelectTopic(`Tìm sách về ${topic.name}`)}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center' }}>{topic.icon}</span>
            <span>{topic.name}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
