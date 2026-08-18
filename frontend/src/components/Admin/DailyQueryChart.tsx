import React, { useEffect, useRef } from 'react';
import type { DailyQueryItem } from '../../types';
import { Chart, BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend } from 'chart.js';
import { Calendar } from 'lucide-react';

Chart.register(BarController, BarElement, CategoryScale, LinearScale, Tooltip, Legend);

interface DailyQueryChartProps {
  dailyQueries?: DailyQueryItem[];
}

export const DailyQueryChart: React.FC<DailyQueryChartProps> = ({ dailyQueries = [] }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  const hasData = dailyQueries.length > 0 && dailyQueries.some((d) => d.queries > 0);

  useEffect(() => {
    if (!canvasRef.current || !hasData) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const labels = dailyQueries.map((d) => {
      const p = d.date.split('-');
      return `${parseInt(p[2], 10)}/${parseInt(p[1], 10)}`;
    });

    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    const textColor = isDark ? '#C0CAD7' : '#4A4E58';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(222, 226, 234, 0.6)';
    const tooltipBg = isDark ? '#1C222C' : '#FFFFFF';
    const tooltipTitle = isDark ? '#EFF3F8' : '#1C1E24';
    const tooltipBorder = isDark ? '#414E62' : '#DEE2EA';

    chartInstanceRef.current = new Chart(canvasRef.current, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: 'Lượt tìm',
            data: dailyQueries.map((d) => d.queries),
            backgroundColor: isDark ? 'rgba(88, 196, 230, 0.75)' : 'rgba(26, 107, 138, 0.75)',
            borderColor: isDark ? '#58C4E6' : '#1A6B8A',
            borderWidth: 1.5,
            borderRadius: 6,
          },
          {
            label: 'Không có kết quả',
            data: dailyQueries.map((d) => d.no_result || 0),
            backgroundColor: isDark ? 'rgba(255, 180, 171, 0.6)' : 'rgba(186, 26, 26, 0.5)',
            borderColor: isDark ? '#FFB4AB' : '#BA1A1A',
            borderWidth: 1.5,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: {
              color: textColor,
              font: { size: 11, family: 'Inter' },
            },
          },
          tooltip: {
            backgroundColor: tooltipBg,
            titleColor: tooltipTitle,
            bodyColor: textColor,
            borderColor: tooltipBorder,
            borderWidth: 1,
            padding: 10,
            cornerRadius: 8,
          },
        },
        scales: {
          x: {
            ticks: { color: isDark ? '#8391A4' : '#767B87', font: { size: 10, family: 'Inter' } },
            grid: { color: gridColor },
          },
          y: {
            beginAtZero: true,
            ticks: { color: isDark ? '#8391A4' : '#767B87', font: { size: 10, family: 'Inter' } },
            grid: { color: gridColor },
          },
        },
      },
    });

    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.destroy();
        chartInstanceRef.current = null;
      }
    };
  }, [dailyQueries, hasData]);

  return (
    <div className="admin-chart-card admin-card-daily">
      <div className="admin-card-header">
        <div>
          <div className="admin-card-title">Truy vấn theo ngày</div>
          <div className="admin-card-subtitle">Lượt tìm kiếm / ngày</div>
        </div>
      </div>

      {hasData ? (
        <div className="admin-canvas-wrap">
          <canvas ref={canvasRef} />
        </div>
      ) : (
        <div className="empty-state">
          <Calendar size={32} style={{ opacity: 0.35, marginBottom: '8px' }} />
          <h3>Chưa có dữ liệu trong khoảng thời gian này</h3>
        </div>
      )}
    </div>
  );
};
