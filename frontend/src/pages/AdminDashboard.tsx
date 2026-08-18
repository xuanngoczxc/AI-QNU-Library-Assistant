import React, { useState, useEffect, useCallback } from 'react';
import type {
  AdminDashboardData,
  AnalyticsResponse,
  RecentActivityEntry,
} from '../types';
import {
  fetchAdminDashboard,
  uploadAdminData,
  deleteAdminFile,
  fetchUsageAnalytics,
  fetchRecentActivity,
} from '../services/api';
import { AdminSidebar } from '../components/Admin/AdminSidebar';
import { KpiGrid } from '../components/Admin/KpiGrid';
import { DataInfoBar } from '../components/Admin/DataInfoBar';
import { DailyQueryChart } from '../components/Admin/DailyQueryChart';
import { IntentChart } from '../components/Admin/IntentChart';
import { TopQueriesTable } from '../components/Admin/TopQueriesTable';
import { RecentActivityList } from '../components/Admin/RecentActivityList';
import { CustomDateRangePicker } from '../components/Admin/CustomDateRangePicker';
import { Toast } from '../components/Modals/Toast';
import { BarChart3, RefreshCw, Menu } from 'lucide-react';

import '../styles/admin.css';

interface AdminDashboardProps {
  onGoToChat: () => void;
}

function getDefaultDateRange() {
  const today = new Date();
  const sevenDaysAgo = new Date(today);
  sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);
  const fmt = (d: Date) => d.toISOString().slice(0, 10);
  return { from: fmt(sevenDaysAgo), to: fmt(today) };
}

export const AdminDashboard: React.FC<AdminDashboardProps> = ({ onGoToChat }) => {
  const [token, setToken] = useState<string>(() => localStorage.getItem('qnu_admin_token') || '');
  const [adminData, setAdminData] = useState<AdminDashboardData | null>(null);
  const [analyticsData, setAnalyticsData] = useState<AnalyticsResponse | null>(null);
  const [recentEntries, setRecentEntries] = useState<RecentActivityEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [recentLoading, setRecentLoading] = useState(false);
  const [lastUpdatedText, setLastUpdatedText] = useState<string>('Đang tải...');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  const handleToggleSidebar = () => {
    if (window.innerWidth <= 960) {
      setMobileOpen((prev) => !prev);
    } else {
      setSidebarCollapsed((prev) => !prev);
    }
  };

  // Date range state
  const defaultDates = getDefaultDateRange();
  const [dateFrom, setDateFrom] = useState<string>(defaultDates.from);
  const [dateTo, setDateTo] = useState<string>(defaultDates.to);

  // Toast state
  const [toast, setToast] = useState<{ message: string; type: 'error' | 'success'; visible: boolean }>({
    message: '',
    type: 'success',
    visible: false,
  });

  const showToast = useCallback((msg: string, type: 'error' | 'success' = 'success') => {
    setToast({ message: msg, type, visible: true });
    setTimeout(() => {
      setToast((prev) => ({ ...prev, visible: false }));
    }, 3000);
  }, []);

  // Kiểm tra token khi vào trang
  useEffect(() => {
    if (!token) {
      onGoToChat();
    }
  }, [token, onGoToChat]);

  // Load Admin Document metadata
  const loadAdminMetadata = useCallback(async () => {
    if (!token) return;
    try {
      const data = await fetchAdminDashboard(token);
      setAdminData(data);
    } catch {
      // Bỏ qua lỗi nhẹ
    }
  }, [token]);

  // Load Analytics & Recent queries
  const loadAnalytics = useCallback(async () => {
    setLoading(true);
    try {
      if (dateFrom && dateTo && dateFrom > dateTo) {
        showToast('Ngày bắt đầu phải trước ngày kết thúc!', 'error');
        setLoading(false);
        return;
      }

      const data = await fetchUsageAnalytics(dateFrom, dateTo);
      setAnalyticsData(data);

      let rangeStr = '';
      if (dateFrom && dateTo) rangeStr = ` · ${dateFrom} → ${dateTo}`;
      else if (dateFrom) rangeStr = ` · Từ ${dateFrom}`;
      else if (dateTo) rangeStr = ` · Đến ${dateTo}`;

      setLastUpdatedText(
        `Cập nhật lúc ${new Date().toLocaleTimeString('vi-VN')}${rangeStr} · ${data.total || 0} truy vấn`
      );
    } catch {
      setLastUpdatedText('Lỗi kết nối');
    } finally {
      setLoading(false);
    }
  }, [dateFrom, dateTo, showToast]);

  const loadRecent = useCallback(async () => {
    setRecentLoading(true);
    try {
      const data = await fetchRecentActivity(10);
      setRecentEntries(data.entries || []);
    } catch {
      // ignore
    } finally {
      setRecentLoading(false);
    }
  }, []);

  const refreshAll = useCallback(() => {
    loadAdminMetadata();
    loadAnalytics();
    loadRecent();
  }, [loadAdminMetadata, loadAnalytics, loadRecent]);

  // Khởi chạy ban đầu và đặt interval 30s
  useEffect(() => {
    refreshAll();
    const interval = setInterval(refreshAll, 30000);
    return () => clearInterval(interval);
  }, [refreshAll]);

  // Xử lý upload file Excel
  const handleUploadExcel = async (file: File) => {
    const ext = file.name.toLowerCase();
    if (!ext.endsWith('.xlsx') && !ext.endsWith('.xls')) {
      showToast('Chỉ chấp nhận file Excel (.xlsx, .xls)!', 'error');
      return;
    }

    showToast('Đang tải lên và xử lý dữ liệu...', 'success');
    try {
      const data = await uploadAdminData(token, file);
      if (data.status === 'skipped') {
        showToast('Dữ liệu đã được cập nhật trước đó, bỏ qua.', 'success');
      } else if (data.reindexed !== undefined) {
        showToast(`Hoàn tất! ${data.total || 0} tài liệu, ${data.reindexed || 0} chỉ mục.`, 'success');
      } else {
        showToast('Cập nhật dữ liệu thành công!', 'success');
      }
      refreshAll();
    } catch (err: any) {
      showToast(err.message || 'Tải file thất bại!', 'error');
    }
  };

  // Xử lý xóa file dữ liệu
  const handleDeleteFile = async (fileName: string) => {
    if (!window.confirm(`Xác nhận xóa file "${fileName}" và reload lại toàn bộ dữ liệu?`)) {
      return;
    }

    try {
      await deleteAdminFile(token, fileName);
      showToast('Đã xóa file và reload thành công!', 'success');
      refreshAll();
    } catch (err: any) {
      showToast(err.message || 'Xóa file thất bại!', 'error');
    }
  };

  // Đăng xuất
  const handleLogout = () => {
    localStorage.removeItem('qnu_admin_token');
    setToken('');
    onGoToChat();
  };

  return (
    <div className={`admin-app ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <AdminSidebar
        onUploadExcel={handleUploadExcel}
        onLogout={handleLogout}
        onGoToChat={onGoToChat}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <main className="admin-main">
        <div className="admin-main-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <button
              type="button"
              className="hamburger"
              onClick={handleToggleSidebar}
              title="Đóng / Mở Menu"
              aria-label="Toggle Menu"
            >
              <Menu size={20} />
            </button>
            <div>
              <div className="admin-header-title">
                <BarChart3 size={22} color="var(--primary)" />
                <span>Tổng quan Quản trị</span>
              </div>
              <div className="admin-last-updated">{lastUpdatedText}</div>
            </div>
          </div>

          <div className="admin-toolbar">
            <CustomDateRangePicker
              startDate={dateFrom}
              endDate={dateTo}
              onChange={(start, end) => {
                setDateFrom(start);
                setDateTo(end);
              }}
            />

            <button
              type="button"
              className="admin-btn-apply"
              onClick={loadAnalytics}
              disabled={loading}
            >
              <RefreshCw size={14} className={loading ? 'spin' : ''} />
              <span>Áp dụng</span>
            </button>
          </div>
        </div>

        <KpiGrid data={analyticsData} loading={loading} />

        <DataInfoBar data={adminData} onDeleteFile={handleDeleteFile} />

        <div className="admin-chart-grid">
          <DailyQueryChart dailyQueries={analyticsData?.daily_queries} />

          <IntentChart intentDistribution={analyticsData?.intent_distribution} />

          <TopQueriesTable topQueries={analyticsData?.top_queries} />

          <RecentActivityList entries={recentEntries} loading={recentLoading} />
        </div>
      </main>

      <Toast message={toast.message} type={toast.type} visible={toast.visible} />
    </div>
  );
};
