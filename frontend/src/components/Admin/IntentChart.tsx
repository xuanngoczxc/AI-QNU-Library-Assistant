import React, { useEffect, useRef } from 'react';
import type { IntentDistItem } from '../../types';
import { Chart, DoughnutController, ArcElement, Tooltip, Legend } from 'chart.js';
import { Target } from 'lucide-react';

Chart.register(DoughnutController, ArcElement, Tooltip, Legend);

const INTENT_LABELS: Record<string, string> = {
  catalog_search: 'Tra cứu sách',
  content_qa: 'Hỏi đáp nội dung',
  general_chat: 'Hỏi đáp chung',
  greeting: 'Chào hỏi',
  unknown: 'Khác',
};

const CHART_COLORS = [
  '#1A6B8A', // Teal Primary
  '#1E7A44', // Green Success
  '#F59E0B', // Amber Warning
  '#8B5CF6', // Purple
  '#06B6D4', // Cyan
  '#EC4899', // Pink
  '#EA580C', // Orange
];

interface IntentChartProps {
  intentDistribution?: IntentDistItem[];
}

export const IntentChart: React.FC<IntentChartProps> = ({ intentDistribution = [] }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const chartInstanceRef = useRef<Chart | null>(null);

  const hasData = intentDistribution.length > 0;

  useEffect(() => {
    if (!canvasRef.current || !hasData) return;

    if (chartInstanceRef.current) {
      chartInstanceRef.current.destroy();
      chartInstanceRef.current = null;
    }

    const colors = intentDistribution.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]);

    chartInstanceRef.current = new Chart(canvasRef.current, {
      type: 'doughnut',
      data: {
        labels: intentDistribution.map((d) => INTENT_LABELS[d.intent] || d.intent),
        datasets: [
          {
            data: intentDistribution.map((d) => d.count),
            backgroundColor: colors,
            borderColor: '#FFFFFF',
            borderWidth: 2,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#FFFFFF',
            titleColor: '#1C1E24',
            bodyColor: '#4A4E58',
            borderColor: '#DEE2EA',
            borderWidth: 1,
            padding: 8,
            cornerRadius: 6,
            callbacks: {
              label: (ctx) => {
                const item = intentDistribution[ctx.dataIndex];
                const label = INTENT_LABELS[item.intent] || item.intent;
                return `${label}: ${ctx.parsed} lần (${item.pct}%)`;
              },
            },
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
  }, [intentDistribution, hasData]);

  return (
    <div className="admin-chart-card admin-card-intent">
      <div className="admin-card-header">
        <div>
          <div className="admin-card-title">Mục đích tìm kiếm</div>
          <div className="admin-card-subtitle">Phân loại câu hỏi của người dùng</div>
        </div>
      </div>

      {hasData ? (
        <>
          <div className="admin-canvas-wrap" style={{ height: '180px' }}>
            <canvas ref={canvasRef} />
          </div>
          <div className="admin-intent-legend">
            {intentDistribution.map((d, i) => (
              <div key={d.intent} className="admin-intent-row">
                <div
                  className="admin-intent-dot"
                  style={{ background: CHART_COLORS[i % CHART_COLORS.length] }}
                />
                <span className="admin-intent-name">
                  {INTENT_LABELS[d.intent] || d.intent}
                </span>
                <span className="admin-intent-count">{d.count}</span>
                <span className="admin-intent-pct">{d.pct}%</span>
              </div>
            ))}
          </div>
        </>
      ) : (
        <div className="empty-state">
          <Target size={32} style={{ opacity: 0.35, marginBottom: '8px' }} />
          <h3>Chưa có dữ liệu</h3>
        </div>
      )}
    </div>
  );
};
