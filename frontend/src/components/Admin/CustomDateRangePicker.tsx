import React, { useState, useRef, useEffect } from 'react';
import { Calendar as CalendarIcon, ChevronLeft, ChevronRight, ChevronDown, Check } from 'lucide-react';
import '../../styles/datepicker.css';

interface CustomDateRangePickerProps {
  startDate: string; // YYYY-MM-DD
  endDate: string;   // YYYY-MM-DD
  onChange: (start: string, end: string) => void;
}

const PRESETS = [
  { label: 'Hôm nay', days: 0 },
  { label: '7 ngày qua', days: 7 },
  { label: '30 ngày qua', days: 30 },
  { label: '90 ngày qua', days: 90 },
  { label: 'Tất cả', days: 365 * 3 },
];

const WEEK_DAYS = ['CN', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];

function formatDateVN(dateStr: string): string {
  if (!dateStr) return '';
  const [y, m, d] = dateStr.split('-');
  return `${d}/${m}/${y}`;
}

function toDateString(d: Date): string {
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

export const CustomDateRangePicker: React.FC<CustomDateRangePickerProps> = ({
  startDate,
  endDate,
  onChange,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Month navigation state
  const [currentMonth, setCurrentMonth] = useState<Date>(() => {
    return endDate ? new Date(endDate) : new Date();
  });

  // Temporary selection state when panel is open
  const [tempStart, setTempStart] = useState<string>(startDate);
  const [tempEnd, setTempEnd] = useState<string>(endDate);
  const [hoverDate, setHoverDate] = useState<string | null>(null);

  // Sync state when props change
  useEffect(() => {
    setTempStart(startDate);
    setTempEnd(endDate);
  }, [startDate, endDate]);

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const prevMonth = () => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() - 1, 1));
  };

  const nextMonth = () => {
    setCurrentMonth(new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 1));
  };

  const handleDateClick = (dateStr: string) => {
    if (!tempStart || (tempStart && tempEnd)) {
      setTempStart(dateStr);
      setTempEnd('');
    } else if (tempStart && !tempEnd) {
      if (dateStr < tempStart) {
        setTempEnd(tempStart);
        setTempStart(dateStr);
      } else {
        setTempEnd(dateStr);
      }
    }
  };

  const applyPreset = (days: number) => {
    const today = new Date();
    const start = new Date(today);
    start.setDate(today.getDate() - days);
    const startStr = toDateString(start);
    const endStr = toDateString(today);
    setTempStart(startStr);
    setTempEnd(endStr);
    onChange(startStr, endStr);
    setIsOpen(false);
  };

  const handleApply = () => {
    if (tempStart && tempEnd) {
      onChange(tempStart, tempEnd);
    } else if (tempStart && !tempEnd) {
      onChange(tempStart, tempStart);
    }
    setIsOpen(false);
  };

  // Generate calendar grid for currentMonth
  const year = currentMonth.getFullYear();
  const month = currentMonth.getMonth();

  const firstDayOfWeek = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const daysInPrevMonth = new Date(year, month, 0).getDate();

  const calendarCells: { dateStr: string; dayNum: number; isCurrentMonth: boolean }[] = [];

  // Previous month trailing days
  for (let i = firstDayOfWeek - 1; i >= 0; i--) {
    const d = daysInPrevMonth - i;
    const prevDate = new Date(year, month - 1, d);
    calendarCells.push({
      dateStr: toDateString(prevDate),
      dayNum: d,
      isCurrentMonth: false,
    });
  }

  // Current month days
  for (let d = 1; d <= daysInMonth; d++) {
    const currDate = new Date(year, month, d);
    calendarCells.push({
      dateStr: toDateString(currDate),
      dayNum: d,
      isCurrentMonth: true,
    });
  }

  // Next month leading days to fill 42 cells (6 rows)
  const remaining = 42 - calendarCells.length;
  for (let d = 1; d <= remaining; d++) {
    const nextDate = new Date(year, month + 1, d);
    calendarCells.push({
      dateStr: toDateString(nextDate),
      dayNum: d,
      isCurrentMonth: false,
    });
  }

  const todayStr = toDateString(new Date());

  return (
    <div className="custom-datepicker-container" ref={containerRef}>
      {/* Trigger Button */}
      <button
        type="button"
        className={`custom-datepicker-trigger ${isOpen ? 'active' : ''}`}
        onClick={() => setIsOpen((prev) => !prev)}
      >
        <CalendarIcon size={15} color="var(--primary)" />
        <span className="datepicker-text">
          {startDate && endDate
            ? `${formatDateVN(startDate)} – ${formatDateVN(endDate)}`
            : 'Chọn khoảng thời gian'}
        </span>
        <ChevronDown size={14} className={`datepicker-chevron ${isOpen ? 'rotate' : ''}`} />
      </button>

      {/* Calendar Dropdown Modal */}
      {isOpen && (
        <div className="custom-datepicker-popover">
          <div className="datepicker-layout">
            {/* Presets Sidebar */}
            <div className="datepicker-presets">
              <div className="presets-title">Khoảng nhanh</div>
              {PRESETS.map((p) => (
                <button
                  type="button"
                  key={p.label}
                  className="preset-btn"
                  onClick={() => applyPreset(p.days)}
                >
                  {p.label}
                </button>
              ))}
            </div>

            {/* Main Calendar Area */}
            <div className="datepicker-calendar-main">
              {/* Header Month Nav */}
              <div className="calendar-header">
                <button type="button" className="cal-nav-btn" onClick={prevMonth}>
                  <ChevronLeft size={16} />
                </button>
                <div className="cal-month-title">
                  Tháng {month + 1}, {year}
                </div>
                <button type="button" className="cal-nav-btn" onClick={nextMonth}>
                  <ChevronRight size={16} />
                </button>
              </div>

              {/* Weekday labels */}
              <div className="calendar-weekdays">
                {WEEK_DAYS.map((wd) => (
                  <div key={wd} className="cal-weekday">
                    {wd}
                  </div>
                ))}
              </div>

              {/* Day cells grid */}
              <div className="calendar-grid">
                {calendarCells.map((cell) => {
                  const isStart = cell.dateStr === tempStart;
                  const isEnd = cell.dateStr === tempEnd;
                  const inRange =
                    tempStart &&
                    tempEnd &&
                    cell.dateStr >= tempStart &&
                    cell.dateStr <= tempEnd;
                  const isHoverInRange =
                    tempStart &&
                    !tempEnd &&
                    hoverDate &&
                    ((cell.dateStr >= tempStart && cell.dateStr <= hoverDate) ||
                      (cell.dateStr <= tempStart && cell.dateStr >= hoverDate));
                  const isToday = cell.dateStr === todayStr;

                  return (
                    <button
                      type="button"
                      key={cell.dateStr}
                      className={`cal-day-cell ${!cell.isCurrentMonth ? 'outside' : ''} ${
                        isStart ? 'range-start' : ''
                      } ${isEnd ? 'range-end' : ''} ${inRange ? 'in-range' : ''} ${
                        isHoverInRange ? 'hover-range' : ''
                      } ${isToday ? 'today' : ''}`}
                      onClick={() => handleDateClick(cell.dateStr)}
                      onMouseEnter={() => setHoverDate(cell.dateStr)}
                      onMouseLeave={() => setHoverDate(null)}
                    >
                      <span className="day-num">{cell.dayNum}</span>
                    </button>
                  );
                })}
              </div>

              {/* Footer */}
              <div className="calendar-footer">
                <div className="calendar-selected-range">
                  {tempStart ? formatDateVN(tempStart) : '--/--/----'}
                  {' → '}
                  {tempEnd ? formatDateVN(tempEnd) : tempStart ? formatDateVN(tempStart) : '--/--/----'}
                </div>

                <div className="calendar-actions">
                  <button
                    type="button"
                    className="cal-btn-cancel"
                    onClick={() => setIsOpen(false)}
                  >
                    Hủy
                  </button>
                  <button
                    type="button"
                    className="cal-btn-apply"
                    onClick={handleApply}
                    disabled={!tempStart}
                  >
                    <Check size={14} />
                    <span>Áp dụng</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
