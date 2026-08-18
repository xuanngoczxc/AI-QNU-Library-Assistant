import React from 'react';
import type { RecentActivityEntry } from '../../types';
import { Search, AlertCircle, Clock } from 'lucide-react';

interface RecentActivityListProps {
  entries: RecentActivityEntry[];
  loading: boolean;
}

function timeAgo(tsStr?: string): string {
  if (!tsStr) return '';
  const sec = Math.floor((Date.now() - new Date(tsStr).getTime()) / 1000);
  if (sec < 60) return 'Vừa xong';
  if (sec < 3600) return `${Math.floor(sec / 60)}ph trước`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h trước`;
  return new Date(tsStr).toLocaleDateString('vi-VN', {
    day: '2-digit',
    month: '2-digit',
  });
}

export const RecentActivityList: React.FC<RecentActivityListProps> = ({ entries, loading }) => {
  const hasData = entries.length > 0;

  return (
    <div className="admin-chart-card admin-card-recent">
      <div className="admin-card-header">
        <div>
          <div className="admin-card-title">Hoạt động gần đây</div>
          <div className="admin-card-subtitle">Các câu hỏi mới nhất từ người dùng</div>
        </div>
      </div>

      {loading && !hasData && (
        <div style={{ padding: '20px', textAlign: 'center', color: 'var(--on-surface-tertiary)' }}>
          Đang tải hoạt động...
        </div>
      )}

      {!loading && hasData && (
        <div className="admin-activity-list">
          {entries.map((e, idx) => {
            const isError = Boolean(e.error);
            return (
              <div key={idx} className="admin-activity-item">
                <div className="admin-activity-q" title={e.query}>
                  {isError ? (
                    <AlertCircle size={14} color="var(--error)" style={{ flexShrink: 0 }} />
                  ) : (
                    <Search size={14} color="var(--primary)" style={{ flexShrink: 0 }} />
                  )}
                  <span>{e.query}</span>
                </div>

                <div className="admin-activity-meta">
                  <span className="admin-activity-rc">{e.result_count ?? 0} KQ</span>
                  <span className="admin-activity-lat">
                    {(((e.latency_ms ?? 0) / 1000).toFixed(1))}s
                  </span>
                  <span className="admin-activity-time">{timeAgo(e.ts)}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {!loading && !hasData && (
        <div className="empty-state">
          <Clock size={28} style={{ opacity: 0.35, marginBottom: '6px' }} />
          <h3>Chưa có hoạt động nào</h3>
        </div>
      )}
    </div>
  );
};
