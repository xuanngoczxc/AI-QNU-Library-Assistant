import React from 'react';
import type { AdminDashboardData } from '../../types';
import { Trash2 } from 'lucide-react';

interface DataInfoBarProps {
  data: AdminDashboardData | null;
  onDeleteFile: (fileName: string) => void;
}

function formatShortDate(val?: string): string {
  if (!val) return '--';
  const parts = String(val).split(' ');
  if (parts.length > 1) return parts[0].slice(0, 10) + ' ' + parts[1].slice(0, 5);
  return String(val).slice(0, 16);
}

export const DataInfoBar: React.FC<DataInfoBarProps> = ({ data, onDeleteFile }) => {
  const docTotal = data?.documents?.total ?? 0;
  const lastUpdate = formatShortDate(data?.updated?.last_data_update);
  const firstFile = data?.files?.[0];

  return (
    <div className="admin-data-info">
      <div className="admin-data-item">
        <span className="admin-data-lbl">Tài liệu:</span>
        <span className="admin-data-val">{docTotal.toLocaleString('vi-VN')}</span>
      </div>

      <div className="admin-data-sep" />

      <div className="admin-data-item">
        <span className="admin-data-lbl">Cập nhật:</span>
        <span className="admin-data-val">{lastUpdate}</span>
      </div>

      <div className="admin-data-sep" />

      <div className="admin-file-info">
        {firstFile ? (
          <>
            <span className="admin-file-name" title={firstFile.name}>
              {firstFile.name}
            </span>
            <span className="admin-file-meta">
              ({(firstFile.size / 1024).toFixed(0)} KB · {formatShortDate(firstFile.mtime)})
            </span>
          </>
        ) : (
          <span className="admin-file-meta">Chưa có file dữ liệu</span>
        )}
      </div>

      {firstFile && (
        <button
          type="button"
          className="admin-file-delete-btn"
          title={`Xóa file "${firstFile.name}" và reload`}
          onClick={() => onDeleteFile(firstFile.name)}
        >
          <Trash2 size={14} />
        </button>
      )}
    </div>
  );
};
