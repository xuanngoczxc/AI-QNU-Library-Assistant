---
description: "Thêm một chart mới vào admin dashboard. Use when: cần thêm biểu đồ thống kê mới vào admin-dashboard.html và API tương ứng."
name: "➕ Add Dashboard Chart"
---

# Add New Chart to Dashboard

## Steps

1. **Add endpoint** (if needed) in `app.py` under `# Usage / Analytics endpoints`
   - Use `get_usage_logger().compute_analytics()` as data source
   - Or add extraction logic inside `compute_analytics()` in `lib/usage_logger.py`

2. **Add chart card** in `admin-dashboard.html`:
   - Use `.chart-card` (or `.chart-card.full` for full-width)
   - Add `<canvas id="myNewChart">` with unique id
   - Add `.empty-state` div with id `myNewChartEmpty`

3. **Add render function** in the `<script>` section:
   - Follow pattern: check empty → destroy previous instance → create new Chart()
   - Use `CHART_COLORS` array for color palette
   - Chart options: `responsive: true`, `maintainAspectRatio: false`
   - Tooltip: dark theme (`backgroundColor: '#1a1d28'`)

4. **Call the render function** inside `loadAll()`

5. **Register chart instance** at top of script for cleanup on re-render

## Chart.js Patterns

```javascript
let myChartInstance = null;

function renderMyChart(data) {
  const canvas = document.getElementById('myNewChart');
  const empty = document.getElementById('myNewChartEmpty');
  const values = data.my_field || [];
  
  if (values.length === 0) {
    canvas.style.display = 'none';
    empty.style.display = 'block';
    return;
  }
  canvas.style.display = 'block';
  empty.style.display = 'none';
  
  if (myChartInstance) { myChartInstance.destroy(); }
  
  myChartInstance = new Chart(canvas, {
    type: 'bar', // or 'line', 'doughnut', 'polarArea'
    data: { ... },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#9498b0', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#1a1d28',
          titleColor: '#e4e6f0',
          bodyColor: '#9498b0',
          borderColor: '#2a2e42',
          borderWidth: 1,
          cornerRadius: 8,
        },
      },
      scales: {
        x: { ticks: { color: '#5c6080' }, grid: { color: 'rgba(42,46,66,0.4)' } },
        y: { ticks: { color: '#5c6080' }, grid: { color: 'rgba(42,46,66,0.4)' } },
      },
    },
  });
}
```