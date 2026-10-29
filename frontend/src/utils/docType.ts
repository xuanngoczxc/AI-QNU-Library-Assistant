import type { BookSource } from '../types';

/**
 * Chuẩn hóa tên Loại tài liệu (doc_type) để đảm bảo đồng nhất,
 * tránh việc bị tách nhóm do khoảng trắng hoặc định dạng khác nhau (ví dụ: 'Giáo trình/Tài liệu tham khảo' vs 'Giáo trình / Tài liệu tham khảo').
 */
export function normalizeDocType(raw?: string, book?: BookSource): string {
  const trimmed = (raw || '').trim();
  if (trimmed && trimmed !== 'N/A' && trimmed !== 'Unknown' && trimmed !== 'nan' && trimmed !== 'Khác') {
    const low = trimmed.toLowerCase();
    if (low === 'tài liệu tham khảo') return 'Tài liệu tham khảo';
    if (low === 'giáo trình') return 'Giáo trình';
    if (low === 'sách') return 'Sách';
    if (low === 'kiểm định') return 'Kiểm định';
    if (low === 'luận văn') return 'Luận văn';
    if (low === 'bài giảng') return 'Bài giảng';
    if (low === 'sách bài tập') return 'Sách bài tập';
    if (low === 'tài liệu chính') return 'Tài liệu chính';
    if (low === 'kỷ yếu') return 'Kỷ yếu';

    // Xử lý các biến thể Giáo trình / Tài liệu tham khảo (với hoặc không có khoảng trắng quanh dấu gạch chéo)
    if (low.includes('giáo trình') && low.includes('tham khảo')) {
      return 'Giáo trình / Tài liệu tham khảo';
    }

    // Chuẩn hóa khoảng cách quanh dấu gạch chéo cho các loại khác
    return trimmed.replace(/\s*\/\s*/g, ' / ');
  }

  // Suy luận thông minh khi trường doc_type bị trống hoặc ghi 'Khác'
  if (book) {
    const loc = (book.location || '').toLowerCase();
    const t = (book.title || '').toLowerCase();
    if (loc.includes('luận văn') || loc.includes('luan van') || /\blv\d+/i.test(loc) || t.startsWith('luận văn')) {
      return 'Luận văn';
    }
    if (t.includes('tự đánh giá') || t.includes('kiểm định') || t.includes('minh chứng')) {
      return 'Kiểm định';
    }
    if (t.includes('giáo trình') || t.includes('giao trinh') || loc.includes('giáo trình') || loc.includes('giao trinh')) {
      return 'Giáo trình';
    }
    if (t.includes('bài tập') || t.includes('bai tap')) {
      return 'Sách bài tập';
    }
    if (t.includes('kỷ yếu') || t.includes('ky yeu')) {
      return 'Kỷ yếu';
    }
    if (loc.includes('phòng đọc') || loc.includes('phòng mượn') || loc.includes('kho')) {
      return 'Sách';
    }
  }
  return 'Sách';
}
