# Hướng Dẫn & Skill Bộ Giao Diện Web Hiện Đại (Modern Web UI Design System)
> **Chuẩn thiết kế:** Material Design 3 (M3) × Apple Human Interface Guidelines (HIG) × Soft UI Glassmorphism  
> **Áp dụng:** Dành cho mọi website, React, Vue, Next.js, Vite, hoặc Vanilla HTML/CSS/JS.

---

## 📑 Mục Lục
1. [Triết Lý & Nguyên Tắc Thiết Kế](#1-triết-lý--nguyên-tắc-thiết-kế)
2. [Hệ Thống Design Tokens (CSS Variables)](#2-hệ-thống-design-tokens-css-variables)
3. [Kiến Trúc Shell Layout (Khung Cố Định & Responsive)](#3-kiến-trúc-shell-layout)
4. [Hệ Thống Chế Độ Sáng / Tối (Light & Dark Mode)](#4-hệ-thống-chế-độ-sáng--tối-light--dark-mode)
5. [Cấu Trúc Các Component Chuẩn Mực](#5-cấu-trúc-các-component-chuẩn-mực)
   - [5.1 Capsule Pill Buttons & Navigation](#51-capsule-pill-buttons--navigation)
   - [5.2 Căn Chỉnh Đối Xứng Tuyệt Đối Ở Đáy (Symmetrical Bottom Alignment)](#52-căn-chỉnh-đối-xứng-tuyệt-đối-ở-đáy)
   - [5.3 Accordion Phân Nhóm Danh Mục (Category / Doc Type Grouping)](#53-accordion-phân-nhóm-danh-mục)
   - [5.4 Smart Auto-Scroll (Cuộn Thông Minh Không Giật Trang)](#54-smart-auto-scroll)
   - [5.5 Custom Date Range Picker Chuẩn M3 & Apple HIG](#55-custom-date-range-picker)
6. [Kinh Nghiệm Tránh Các Lỗi Thường Gặp](#6-kinh-nghiệm-tránh-các-lỗi-thường-gặp)

---

## 1. Triết Lý & Nguyên Tắc Thiết Kế

1. **M3 Tonal Palette × Soft UI Surfaces:** Không dùng các màu cờ đơn điệu (`red`, `blue`). Sử dụng bảng màu Tonal phân tầng rõ rệt (Container, Surface Dim, Surface Container High), giúp giao diện có chiều sâu.
2. **Glassmorphism Tinh Tế:** Sử dụng `backdrop-filter: blur(16px)` kết hợp màu nền trong suốt có biến số (`var(--glass-bg)`), tuyệt đối không hardcode `rgba(255,255,255,...)` để Dark Mode không bị vệt sáng loang lổ.
3. **Capsule Pill Shapes (`--cap: 9999px`):** Bo tròn hoàn toàn các nút hành động, badge đếm, thanh input để tạo cảm giác thân thiện, mềm mại và hiện đại.
4. **Trải Nghiệm Responsive 100dvh:** Trên Desktop là khung Shell bo tròn 24px lơ lửng giữa màn hình; trên Mobile tự động bung full tràn viền `100dvh` và dùng Drawer trượt từ bên trái.

---

## 2. Hệ Thống Design Tokens (CSS Variables)

Tạo file `variables.css` và import vào dự án:

```css
:root {
  /* ── 1. M3 Tonal Palette (Light Mode) ── */
  --primary: #1A6B8A;
  --on-primary: #FFFFFF;
  --primary-container: #C0E8F7;
  --on-primary-container: #001E2C;
  --primary-hover: #155E7A;
  --secondary: #4B5A6A;
  --on-secondary: #FFFFFF;
  --secondary-container: #D5E3F2;

  /* ── 2. Surfaces & Backgrounds ── */
  --surface: #FFFFFF;
  --surface-dim: #E8ECF0;
  --surface-container: #F2F5F9;
  --surface-container-high: #EAEEF3;
  --surface-container-higher: #E0E5EB;
  --surface-container-low: #F6F8FA;
  --bg: #EEF1F5;
  --bg-app: #E4E9F0;

  /* ── 3. Glassmorphism Tokens (Light) ── */
  --glass-bg: rgba(255, 255, 255, 0.75);
  --glass-bg-sticky: rgba(255, 255, 255, 0.9);
  --glass-bg-composer: rgba(255, 255, 255, 0.8);
  --shell-inner-border: rgba(255, 255, 255, 0.5);
  --shell-border: rgba(0, 0, 0, 0.04);

  /* ── 4. Text & Strokes ── */
  --on-surface: #1C1E24;
  --on-surface-variant: #4A4E58;
  --on-surface-tertiary: #767B87;
  --outline: #BCC2CC;
  --outline-variant: #DEE2EA;

  /* ── 5. Status Colors ── */
  --error: #BA1A1A;
  --on-error: #FFFFFF;
  --error-container: #FFDAD6;
  --on-error-container: #410002;
  --success: #1E7A44;
  --success-container: #C7F0D4;
  --warning: #8A6A00;
  --warning-container: #FFE3A0;

  /* ── 6. Shadows — Soft UI Layered ── */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.04), 0 1px 2px rgba(0, 0, 0, 0.03);
  --shadow-md: 0 4px 14px rgba(0, 0, 0, 0.05), 0 2px 4px rgba(0, 0, 0, 0.04);
  --shadow-lg: 0 10px 32px rgba(0, 0, 0, 0.06), 0 4px 10px rgba(0, 0, 0, 0.04);
  --shadow-xl: 0 20px 56px rgba(0, 0, 0, 0.07), 0 8px 20px rgba(0, 0, 0, 0.04);
  --shadow-2xl: 0 32px 80px rgba(0, 0, 0, 0.08), 0 12px 32px rgba(0, 0, 0, 0.05);
  --shadow-glow: 0 0 0 3px rgba(26, 107, 138, 0.15);
  --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0, 0, 0, 0.03);
  --shadow-btn: 0 4px 12px rgba(26, 107, 138, 0.2);
  --shadow-btn-hover: 0 6px 20px rgba(26, 107, 138, 0.25);

  /* ── 7. Shapes & Motion ── */
  --font: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
  --cap: 9999px;
  --round-xs: 8px;
  --round-sm: 12px;
  --round-md: 16px;
  --round-lg: 20px;
  --round-xl: 24px;
  --spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease: cubic-bezier(0.16, 1, 0.3, 1);
  --fast: 180ms var(--spring);
  --slow: 350ms var(--ease);
}

/* ══════════════════ DARK THEME TOKENS ══════════════════ */
[data-theme="dark"] {
  /* M3 Dark Palette — Teal Nocturne */
  --primary: #58C4E6;
  --on-primary: #003546;
  --primary-container: #004D64;
  --on-primary-container: #BAEAFA;
  --primary-hover: #7BD2ED;
  --secondary: #B2C5D6;
  --on-secondary: #1C3140;
  --secondary-container: #334857;

  /* Surfaces — Deep Blue-Grey Soft UI */
  --surface: #141820;
  --surface-dim: #0E1116;
  --surface-container: #1C222C;
  --surface-container-high: #242B38;
  --surface-container-higher: #2E3748;
  --surface-container-low: #10131A;
  --bg: #0D1016;
  --bg-app: #090B0F;

  /* Glassmorphism surfaces — Dark */
  --glass-bg: rgba(20, 24, 32, 0.85);
  --glass-bg-sticky: rgba(20, 24, 32, 0.95);
  --glass-bg-composer: rgba(20, 24, 32, 0.9);
  --shell-inner-border: rgba(255, 255, 255, 0.05);
  --shell-border: rgba(255, 255, 255, 0.08);

  /* Text & stroke */
  --on-surface: #EFF3F8;
  --on-surface-variant: #C0CAD7;
  --on-surface-tertiary: #8391A4;
  --outline: #414E62;
  --outline-variant: #262E3B;

  /* Status */
  --error: #FFB4AB;
  --on-error: #690005;
  --error-container: #93000A;
  --on-error-container: #FFDAD6;
  --success: #7ADBA0;
  --success-container: #005327;
  --warning: #FFD271;
  --warning-container: #5C4300;

  /* Shadows — Dark Elevation */
  --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5), 0 2px 6px rgba(0, 0, 0, 0.35);
  --shadow-lg: 0 10px 32px rgba(0, 0, 0, 0.6), 0 4px 12px rgba(0, 0, 0, 0.4);
  --shadow-xl: 0 20px 56px rgba(0, 0, 0, 0.7), 0 8px 24px rgba(0, 0, 0, 0.5);
  --shadow-2xl: 0 32px 80px rgba(0, 0, 0, 0.85), 0 12px 36px rgba(0, 0, 0, 0.6);
  --shadow-glow: 0 0 0 3px rgba(88, 196, 230, 0.25);
  --shadow-card: 0 2px 8px rgba(0, 0, 0, 0.4), 0 1px 3px rgba(0, 0, 0, 0.3);
  --shadow-btn: 0 4px 14px rgba(88, 196, 230, 0.3);
  --shadow-btn-hover: 0 6px 22px rgba(88, 196, 230, 0.45);
}
```

---

## 3. Kiến Trúc Shell Layout

```css
body {
  margin: 0;
  padding: 0;
  min-height: 100vh;
  background: var(--bg-app);
  display: flex;
  justify-content: center;
  align-items: center;
  font-family: var(--font);
  overflow: hidden;
}

.shell {
  width: 100%;
  max-width: 1440px;
  height: calc(100vh - 28px);
  display: flex;
  border-radius: var(--round-xl);
  overflow: hidden;
  position: relative;
  background: var(--surface);
  box-shadow: var(--shadow-2xl), 0 0 0 1px var(--shell-inner-border) inset, 0 0 0 1px var(--shell-border);
}

@media (max-width: 768px) {
  body {
    padding: 0;
    align-items: stretch;
  }
  .shell {
    height: 100dvh;
    border-radius: 0;
    max-width: 100%;
  }
}
```

---

## 4. Hệ Thống Chế Độ Sáng / Tối (Light & Dark Mode)

### Hook `useTheme.ts`:
```typescript
import { useState, useEffect, useCallback } from 'react';

export type Theme = 'light' | 'dark';

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    const saved = localStorage.getItem('app_theme') as Theme | null;
    if (saved === 'light' || saved === 'dark') return saved;
    if (typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('app_theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'light' ? 'dark' : 'light'));
  }, []);

  return { theme, toggleTheme };
}
```

### Theme Toggle Button Component (Nút tròn 36px kèm animation):
```tsx
<button
  type="button"
  className="theme-toggle-btn"
  onClick={toggleTheme}
  title={theme === 'dark' ? 'Giao diện Sáng' : 'Giao diện Tối'}
>
  {theme === 'dark' ? <Sun size={17} color="#FFD271" className="sun" /> : <Moon size={17} className="moon" />}
</button>
```

```css
.theme-toggle-btn {
  width: 36px;
  height: 36px;
  border-radius: var(--cap);
  border: 1px solid var(--outline-variant);
  background: var(--surface);
  cursor: pointer;
  color: var(--on-surface-variant);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: var(--fast);
  box-shadow: var(--shadow-sm);
}
.theme-toggle-btn:hover {
  background: var(--surface-container-high);
  color: var(--primary);
  transform: scale(1.05);
}
```

---

## 5. Cấu Trúc Các Component Chuẩn Mực

### 5.1 Capsule Pill Buttons & Navigation
```css
.btn-capsule {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 8px 18px;
  border-radius: var(--cap);
  border: none;
  background: var(--primary);
  color: var(--on-primary);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: var(--slow);
  box-shadow: var(--shadow-btn);
}
.btn-capsule:hover {
  background: var(--primary-hover);
  transform: translateY(-1px);
  box-shadow: var(--shadow-btn-hover);
}
```

---

### 5.2 Căn Chỉnh Đối Xứng Tuyệt Đối Ở Đáy (Symmetrical Bottom Alignment)
```css
/* Sidebar Bottom */
.sidebar-sticky {
  flex-shrink: 0;
  margin: auto -16px 0;
  padding: 10px 16px 8px;
  border-top: 1px solid var(--outline-variant);
  background: var(--glass-bg);
  min-height: 76px;
  height: 76px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-sizing: border-box;
}

/* Main Content Composer Bottom */
.composer-wrap {
  padding: 10px 20px 8px;
  background: var(--glass-bg-composer);
  backdrop-filter: blur(16px);
  border-top: 1px solid var(--outline-variant);
  min-height: 76px;
  height: 76px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
```

---

### 5.3 Accordion Phân Nhóm Danh Mục (Category / Doc Type Grouping)
```tsx
// Nhóm dữ liệu theo thuộc tính danh mục
const grouped = useMemo(() => {
  const map: Record<string, Item[]> = {};
  items.forEach(item => {
    const key = item.category || 'Khác';
    if (!map[key]) map[key] = [];
    map[key].push(item);
  });
  return Object.entries(map).sort((a, b) => b[1].length - a[1].length);
}, [items]);
```

```css
.doc-group-section {
  border: 1px solid var(--outline-variant);
  border-radius: var(--round-md);
  overflow: hidden;
  background: var(--surface);
  margin-bottom: 8px;
  box-shadow: var(--shadow-card);
}
.doc-group-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 14px;
  cursor: pointer;
  background: var(--surface-container);
  border: none;
  width: 100%;
}
.doc-group-header.open {
  background: var(--primary-container);
  color: var(--on-primary-container);
  border-bottom: 1px solid var(--outline-variant);
}
```

---

### 5.4 Smart Auto-Scroll (Cuộn Thông Minh Không Giật Trang)
- **Quy tắc:** Chỉ cuộn khi người dùng gửi câu hỏi mới hoặc đang streaming dữ liệu từ server. Tuyệt đối **không cuộn** khi người dùng nhấp mở Accordion hay xem chi tiết.

```typescript
// Khi có tin nhắn mới -> Cuộn xuống đáy
useEffect(() => {
  if (messages.length > prevLength.current) {
    userScrolledUp.current = false;
    bottomAnchorRef.current?.scrollIntoView({ behavior: 'smooth' });
    prevLength.current = messages.length;
  }
}, [messages.length]);

// Khi đang stream văn bản
useEffect(() => {
  if (isStreaming && !userScrolledUp.current) {
    bottomAnchorRef.current?.scrollIntoView({ behavior: 'auto' });
  }
}, [isStreaming, lastMessageText]);
```

---

## 6. Kinh Nghiệm Tránh Các Lỗi Thường Gặp

| Lỗi Thường Gặp | Nguyên Nhân | Cách Khắc Phục Chuẩn Xác |
|---|---|---|
| **Dark Mode loang lổ vệt trắng** | Dùng hardcode `rgba(255,255,255,...)` hoặc `#fff` cho nền kính. | Dùng biến `--glass-bg`, `--glass-bg-composer` và `--surface`. |
| **Đường kẻ ngang ở đáy bị lệch** | Padding và chiều cao của Sidebar đáy và Main Content đáy khác nhau. | Cùng đặt `min-height: 76px`, đồng bộ `padding: 10px ... 8px` và mở rộng `margin: auto -16px 0`. |
| **Bấm mở Accordion bị giật cuộn trang xuống đáy** | Dùng `ResizeObserver` bắt sự kiện resize toàn bộ container. | Bỏ `ResizeObserver`, chỉ kích hoạt cuộn khi `messages.length` tăng hoặc `isStreaming === true`. |
| **Vỡ layout trên mobile** | Chiều cao dùng `100vh` bị thanh địa chỉ trình duyệt che mất. | Sử dụng `100dvh` (Dynamic Viewport Height) cho khung Shell trên mobile. |
