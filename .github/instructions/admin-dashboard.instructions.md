---
description: "Hướng dẫn cho dashboard admin: tối ưu UI/UX, hiển thị Chart.js, xử lý analytics từ JSONL. Use when: làm việc với admin-dashboard.html, thêm chart, tối ưu giao diện dark theme."
applyTo: "admin-dashboard.html"
---

# Admin Dashboard Instructions

## Stack & Conventions

- **Chart.js 4** via CDN (`chart.umd.min.js`) — NO chartjs-adapter-luxon unless time axis needed
- **Dark theme** CSS variables (`--bg: #0f1117`, `--surface: #1a1d28`, `--surface-card: #1e2235`)
- **Font**: Inter (Google Fonts), system fallback
- **Icons**: Unicode emoji (📈 📊 🔍) — NO icon library
- **No framework** — vanilla HTML/CSS/JS

## Color Palette

```
--accent: #6c8cff     (blue primary)
--green: #34d399      (success)
--yellow: #fbbf24     (warning)
--red: #f87171        (danger)
--cyan: #22d3ee       (info)
--purple: #a78bfa     (secondary)
```

## KPI Cards

- 6 cards in `.kpi-grid` using `grid-template-columns: repeat(auto-fit, minmax(200px, 1fr))`
- Each card uses `kpi-{color}` class for `.value` color
- Data: `renderKPI(data)` — total, today, no-result, avg latency, intent count, success rate

## Charts

- `renderDailyChart()`: bar chart (queries + no-result overlay), 2 y-axis datasets
- `renderIntentChart()`: doughnut chart with cutout 62%, custom legend below
- `renderLatency()`: static metric cards (avg, P50, P95, P99, min, max) + optional bar chart
- ALWAYS destroy previous Chart instance before creating new one (prevent memory leak)
- Chart tooltip: `backgroundColor: '#1a1d28'`, `borderColor: '#2a2e42'`

## Data Flow

1. `loadAll()` → `GET /api/usage/analytics` → render all KPIs + charts
2. Then `GET /api/usage/recent?n=30` → render recent activity
3. Auto-refresh every 30s via `setInterval(loadAll, 30000)`
4. Button "Làm mới" calls `loadAll()`

## Empty States

Every chart section MUST have an `.empty-state` div shown when data is empty:
- Show only when data array is empty or all zeros
- Hidden class: `display:none`

## Responsive

- `@media(max-width:960px)`: sidebar 200px, charts 1-col
- `@media(max-width:640px)`: hide sidebar, 2-col KPI grid
- Heatmap cells: `aspect-ratio:1` + `min-height:28px`

## Structure Conventions

- All `id=` selectors for JS hooks, class-based for CSS
- Never inline event handlers — use `onclick=` in HTML only for `loadAll()`
- Templates use template literals with `\n` for multi-line
- `.chart-card.full` spans full width with `grid-column: 1 / -1`