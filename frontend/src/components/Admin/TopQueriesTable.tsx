import React from 'react';
import type { TopQueryItem } from '../../types';
import { Search } from 'lucide-react';

interface TopQueriesTableProps {
  topQueries?: TopQueryItem[];
}

export const TopQueriesTable: React.FC<TopQueriesTableProps> = ({ topQueries = [] }) => {
  const hasData = topQueries.length > 0;

  return (
    <div className="admin-chart-card admin-card-top">
      <div className="admin-card-header">
        <div>
          <div className="admin-card-title">Truy vấn phổ biến</div>
          <div className="admin-card-subtitle">Nhiều lượt tra cứu nhất</div>
        </div>
      </div>

      {hasData ? (
        <div className="admin-table-wrap">
          <table className="admin-table">
            <thead>
              <tr>
                <th style={{ width: '34px' }}>#</th>
                <th>Truy vấn</th>
                <th style={{ width: '60px', textAlign: 'right' }}>Lượt</th>
              </tr>
            </thead>
            <tbody>
              {topQueries.slice(0, 10).map((q, idx) => (
                <tr key={idx}>
                  <td style={{ color: 'var(--on-surface-tertiary)' }}>{idx + 1}</td>
                  <td>
                    <div className="admin-query-cell" title={q.query}>
                      {q.query}
                    </div>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <span className="admin-count-badge">{q.count}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-state">
          <Search size={28} style={{ opacity: 0.35, marginBottom: '6px' }} />
          <h3>Chưa có dữ liệu</h3>
        </div>
      )}
    </div>
  );
};
