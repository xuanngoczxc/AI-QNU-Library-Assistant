import React from 'react';
import type { AnalyticsResponse } from '../../types';
import { Search, Calendar, AlertCircle, CheckCircle2 } from 'lucide-react';

interface KpiGridProps {
  data: AnalyticsResponse | null;
  loading: boolean;
}

export const KpiGrid: React.FC<KpiGridProps> = ({ data, loading }) => {
  if (loading || !data) {
    return (
      <div className="admin-kpi-grid">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="admin-kpi-card" style={{ opacity: 0.6 }}>
            <div className="admin-kpi-label">Đang tải...</div>
            <div className="admin-kpi-value">--</div>
          </div>
        ))}
      </div>
    );
  }

  const total = data.total || 0;
  const daily = data.daily_queries || [];
  const last = daily[daily.length - 1];
  const prev = daily[daily.length - 2];
  const todayStr = new Date().toISOString().slice(0, 10);
  const todayEntry = daily.find((d) => d.date === todayStr);
  const todayVal = todayEntry?.queries ?? 0;
  const todayNo = todayEntry?.no_result ?? 0;
  const ud = data.unique_days || 0;
  const totalEmpty = (data.daily_no_result || []).reduce((a, b) => a + b.count, 0);
  const successRate = total > 0 ? ((1 - totalEmpty / total) * 100).toFixed(0) : '0';

  return (
    <div className="admin-kpi-grid">
      <div className="admin-kpi-card">
        <div className="admin-kpi-top">
          <span className="admin-kpi-label">Tổng truy vấn</span>
          <div className="admin-kpi-icon" style={{ background: 'var(--primary-container)', color: 'var(--primary)' }}>
            <Search size={16} />
          </div>
        </div>
        <div className="admin-kpi-value" style={{ color: 'var(--primary)' }}>
          {total.toLocaleString('vi-VN')}
        </div>
        <div className="admin-kpi-sub">{ud} ngày hoạt động</div>
      </div>

      <div className="admin-kpi-card">
        <div className="admin-kpi-top">
          <span className="admin-kpi-label">Hôm nay</span>
          <div className="admin-kpi-icon" style={{ background: 'var(--success-container)', color: 'var(--success)' }}>
            <Calendar size={16} />
          </div>
        </div>
        <div className="admin-kpi-value" style={{ color: 'var(--success)' }}>
          {todayVal.toLocaleString('vi-VN')}
        </div>
        <div className="admin-kpi-sub">
          {last?.date === todayStr && (prev?.queries ?? 0) > 0 ? `Hôm qua: ${prev?.queries}` : 'Cập nhật hôm nay'}
        </div>
      </div>

      <div className="admin-kpi-card">
        <div className="admin-kpi-top">
          <span className="admin-kpi-label">Chưa có kết quả</span>
          <div className="admin-kpi-icon" style={{ background: 'var(--warning-container)', color: 'var(--warning)' }}>
            <AlertCircle size={16} />
          </div>
        </div>
        <div className="admin-kpi-value" style={{ color: 'var(--warning)' }}>
          {todayNo}
        </div>
        <div className="admin-kpi-sub">
          {todayVal > 0 ? `${((todayNo / todayVal) * 100).toFixed(1)}% hôm nay` : '0% hôm nay'}
        </div>
      </div>

      <div className="admin-kpi-card">
        <div className="admin-kpi-top">
          <span className="admin-kpi-label">Tỷ lệ có kết quả</span>
          <div className="admin-kpi-icon" style={{ background: 'var(--primary-container)', color: 'var(--primary)' }}>
            <CheckCircle2 size={16} />
          </div>
        </div>
        <div className="admin-kpi-value" style={{ color: Number(successRate) > 80 ? 'var(--success)' : 'var(--warning)' }}>
          {successRate}%
        </div>
        <div className="admin-kpi-sub">{totalEmpty} lượt trống</div>
      </div>
    </div>
  );
};
