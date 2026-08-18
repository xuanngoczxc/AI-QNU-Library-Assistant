---
name: modern-web-ui-design-system
description: >-
  Áp dụng hệ thống thiết kế giao diện web hiện đại (Material Design 3 × Apple HIG × Soft UI Glassmorphism) 
  với hỗ trợ Light/Dark mode toàn diện, bố cục Shell cố định, capsule components, đồng bộ cao độ đáy, 
  và tương tác mượt mà cho bất kỳ ứng dụng web React / Vue / Vanilla JS nào.
---

# Modern Web UI Design System (M3 × Apple HIG × Soft UI)

Tài liệu này đóng gói toàn bộ quy chuẩn, cấu trúc mã nguồn, design tokens và kinh nghiệm triển khai bộ giao diện người dùng cao cấp chuẩn 2025, có thể áp dụng cho bất kỳ dự án web nào (Dashboard, Chatbot, E-commerce, SaaS Web App).

---

## 1. Triết Lý & Nguyên Tắc Thiết Kế Cốt Lõi

1. **Material Design 3 Tonal Palette × Apple HIG:** Sử dụng bảng màu có phân tầng sắc thái (Tonal surfaces), độ tương phản đạt chuẩn WCAG AAA, typography sắc nét (Inter, SF Pro).
2. **Soft UI Glassmorphism:** Kết hợp nền kính mờ `backdrop-filter: blur(16px)` với bóng đổ đa tầng (`--shadow-2xl`) tạo cảm giác sâu, sang trọng và không bị thô cứng.
3. **Capsule Pill Shapes (`--cap: 9999px`):** Bo tròn toàn phần cho các nút hành động, thanh tìm kiếm, input filter và badge để tạo cảm giác thân thiện, mềm mại.
4. **Shell Layout Cố Định (Fixed Viewport Desktop):** Ứng dụng luôn nằm gọn trong khung Shell bo góc 24px `height: calc(100vh - 28px)` trên máy tính, và full `100dvh` trên thiết bị di động.
5. **Đồng Bộ Đối Xứng Tuyệt Đối (Symmetrical Horizontal Alignment):** Các đường phân cách ngang và chiều cao các khối ở đáy luôn trùng khớp trên cùng một đường trục tọa độ Y.

---

## 2. Hệ Thống Design Tokens (`variables.css`)

Sao chép toàn bộ khối biến CSS dưới đây vào file `variables.css` của dự án:

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

## 3. Kiến Trúc Khung Ứng Dụng (Shell Layout)

```css
/* Layout khung ứng dụng luôn nằm trọn trong Viewport */
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

## 4. Quản Lý Theme Sáng / Tối (`useTheme.ts`)

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

---

## 5. Quy Tắc Đồng Bộ Cao Độ & Đường Kẻ Đáy (Symmetrical Bottom Alignment)

Để tránh hiện tượng đường phân cách ngang bị lệch giữa Sidebar bên trái và Khung nội dung bên phải:
1. Đặt cả 2 khối đáy cùng có `min-height` cố định (ví dụ `76px`).
2. Đường `border-top: 1px solid var(--outline-variant)` của Sidebar mở rộng chạm mép (`margin: auto -16px 0;`).
3. Khối input và nút bấm bên trong 2 khối có cùng chiều cao (ví dụ `34px` hoặc `38px`), căn giữa theo trục Y (`display: flex; align-items: center`).

---

## 6. Quy Tắc Smart Auto-Scroll (Không giật trang)

1. **Chỉ cuộn khi có tương tác tìm kiếm mới** hoặc bot đang streaming văn bản (`isStreaming === true`).
2. **Khi người dùng bấm mở/xổ Accordion** (thẻ chi tiết, nhóm danh mục): **TUYỆT ĐỐI KHÔNG TỰ CUỘN**, để người dùng xem thông tin tĩnh tại vị trí hiện tại.

```typescript
// Chỉ cuộn khi messages.length tăng (câu hỏi mới)
useEffect(() => {
  if (messages.length > prevMsgLengthRef.current) {
    userScrolledUpRef.current = false;
    scrollToBottom('smooth');
    prevMsgLengthRef.current = messages.length;
  }
}, [messages.length]);
```

---

## 7. Phân Nhóm Accordion Theo Danh Mục (Grouping Accordion)

- Luôn nhóm danh sách trả về theo thuộc tính danh mục (`doc_type`, `category`, `status`).
- Mặc định mở nhóm đầu tiên (index 0), các nhóm còn lại đóng để tiết kiệm diện tích.
- Mỗi nhóm có icon trực quan, badge số lượng (`count`) và phân trang riêng nếu nhiều phần tử.
