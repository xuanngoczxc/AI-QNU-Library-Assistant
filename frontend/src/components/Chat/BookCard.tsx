import React, { useState } from 'react';
import type { BookSource } from '../../types';
import { normalizeDocType } from '../../utils/docType';
import {
  BookOpen,
  Book,
  FileCheck,
  GraduationCap,
  ChevronDown,
  ExternalLink,
  MapPin,
  Building2,
  Bookmark,
  FileText,
  Tag,
  Layers,
  Library,
} from 'lucide-react';

interface BookCardProps {
  book: BookSource;
  index?: number;
}

function cleanVal(v: any): string {
  if (v === undefined || v === null) return '';
  let s = String(v).trim();
  const low = s.toLowerCase();
  if (
    !s ||
    low === 'nan' ||
    low === 'n/a' ||
    low === 'unknown' ||
    low === 'không xác định' ||
    low === 'null' ||
    low === 'none'
  ) {
    return '';
  }
  // Loại bỏ các ký tự dấu câu MARC21 thừa ở cuối chuỗi: dấu chấm '.', gạch chéo '/', hai chấm ':', phẩy ','
  return s.replace(/[\s/.,;:]+$/, '').trim();
}

function getDocTypeIcon(dt: string) {
  const low = dt.toLowerCase();
  if (low.includes('giáo trình')) return <Book size={16} />;
  if (low.includes('kiểm định')) return <FileCheck size={16} />;
  if (low.includes('luận văn')) return <GraduationCap size={16} />;
  if (low.includes('tham khảo')) return <Library size={16} />;
  return <BookOpen size={16} />;
}

export const BookCard: React.FC<BookCardProps> = ({ book }) => {
  const [isOpen, setIsOpen] = useState(false);

  // 1. Nhóm định danh & nhan đề
  const title = cleanVal(book.title) || 'Tài liệu không có tên';
  const subtitle = cleanVal(book.subtitle);
  const bilingualTitle = cleanVal(book.bilingual_title);
  const partNumber = cleanVal(book.part_number);
  const partName = cleanVal(book.part_name);
  const partDisplay = [partNumber, partName].filter(Boolean).join(' - ');

  // 2. Nhóm tác giả & xuất bản
  const responsibility = cleanVal(book.responsibility);
  const author100 = cleanVal(book.author_100);
  const author = responsibility || cleanVal(book.author) || author100;
  const coAuthors = Array.isArray(book.co_authors)
    ? book.co_authors.map(cleanVal).filter(Boolean)
    : [];
  const role = cleanVal(book.role);
  const publisher = cleanVal(book.publisher);
  const place = cleanVal(book.place);
  const year = cleanVal(book.year);
  const edition = cleanVal(book.edition);
  const series = cleanVal(book.series);
  const language = cleanVal(book.language);

  // 3. Nhóm phân loại & đào tạo
  const docType = normalizeDocType(cleanVal(book.doc_type) || cleanVal(book.loai_tai_lieu), book);
  const major = cleanVal(book.major);
  const subject = cleanVal(book.subject);
  const ddc = cleanVal(book.ddc);
  const ddcCutter = cleanVal(book.ddc_cutter);
  const ddcDisplay = ddc ? (ddcCutter ? `${ddc} / ${ddcCutter}` : ddc) : ddcCutter;
  const udc = cleanVal(book.udc);
  const otherClass = cleanVal(book.other_class);
  const keywords = Array.isArray(book.keywords)
    ? book.keywords.map(cleanVal).filter(Boolean)
    : [];

  // 4. Nhóm thông số vật lý & tiêu chuẩn
  const isbn = cleanVal(book.isbn);
  const price = cleanVal(book.price);
  const physical = cleanVal(book.physical);
  const illustration = cleanVal(book.illustration);
  const size = cleanVal(book.size);

  // 5. Mô tả & ghi chú
  const abstractText = cleanVal(book.abstract);
  const notes = cleanVal(book.notes);
  const copyright = cleanVal(book.copyright);

  // 6. Vị trí & liên kết
  const location = cleanVal(book.location);
  const locations = Array.isArray(book.locations) ? book.locations : [];
  const link = cleanVal(book.link);


  // Parse location tags fallback
  const locationTags = location
    ? location.split(';').map((loc) => loc.trim()).filter(Boolean)
    : [];

  // Kiểm tra xem từng nhóm có dữ liệu để hiển thị không
  const hasAuthorSection = !!(
    author ||
    coAuthors.length > 0 ||
    publisher ||
    place ||
    year ||
    edition ||
    series ||
    language
  );
  const hasClassificationSection = !!(
    major ||
    docType ||
    ddcDisplay ||
    udc ||
    otherClass ||
    subject ||
    keywords.length > 0
  );
  const hasPhysicalSection = !!(isbn || price || physical || illustration || size);
  const hasDescriptionSection = !!(subtitle || bilingualTitle || abstractText || notes || copyright);
  const hasLocationSection = locations.length > 0 || locationTags.length > 0;

  // Subtitle trên Header tóm tắt: Ghi rõ ràng từng trường có nhãn, phân tách bằng dấu phẩy, không dùng dấu chấm
  const headerMetaItems: { label: string; value: string }[] = [];
  if (author) headerMetaItems.push({ label: 'Tác giả', value: author });
  if (year) headerMetaItems.push({ label: 'Năm', value: year });
  if (publisher) headerMetaItems.push({ label: 'NXB', value: publisher });

  return (
    <div className={`book-card ${isOpen ? 'open' : ''}`}>
      {/* ── CARD HEADER ── */}
      <div className="book-card-header" onClick={() => setIsOpen((prev) => !prev)}>
        <div className="book-card-icon" title={docType}>
          {getDocTypeIcon(docType)}
        </div>

        <div className="book-card-info">
          <div className="book-card-title-row">
            <span className="book-card-title">{title}</span>
            {partDisplay && (
              <span className="book-part-badge" title="Tập / Phần">
                {partDisplay}
              </span>
            )}
          </div>
          {headerMetaItems.length > 0 && (
            <div className="book-card-subtitle">
              {headerMetaItems.map((item, idx) => (
                <span key={item.label} className="book-sub-item">
                  <span className="book-sub-label">{item.label}:</span>{' '}
                  <span className="book-sub-val">{item.value}</span>
                  {idx < headerMetaItems.length - 1 && (
                    <span className="book-sub-comma">, </span>
                  )}
                </span>
              ))}
            </div>
          )}
        </div>

        <div className="book-card-meta-right">
          {book.copies && book.copies > 1 && (
            <span className="book-copies-badge" title={`Thư viện có ${book.copies} bản lưu trữ`}>
              {book.copies} bản
            </span>
          )}
          {docType && (
            <span className="book-type-badge">
              {docType}
            </span>
          )}
          <ChevronDown size={16} className="book-card-chevron" />
        </div>
      </div>

      {/* ── CARD DETAILS BODY ── */}
      {isOpen && (
        <div className="book-details-wrapper">
          {/* Nhóm 1: Tác giả & Xuất bản */}
          {hasAuthorSection && (
            <div className="book-section">
              <div className="book-section-title">
                <Building2 size={13} />
                <span>Thông tin tác giả &amp; Xuất bản</span>
              </div>
              <div className="book-details-grid">
                {author && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Tác giả:</span>{' '}
                    <span className="book-field-val">{author}</span>
                  </div>
                )}
                {!author && author100 && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Tác giả chính:</span>{' '}
                    <span className="book-field-val">{author100}</span>
                  </div>
                )}
                {responsibility && responsibility !== author && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Trách nhiệm:</span>{' '}
                    <span className="book-field-val">{responsibility}</span>
                  </div>
                )}
                {coAuthors.length > 0 && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Đồng tác giả / Dịch giả:</span>{' '}
                    <span className="book-field-val">
                      {coAuthors.join(', ')} {role ? `(${role})` : ''}
                    </span>
                  </div>
                )}
                {publisher && (
                  <div className="book-field">
                    <span className="book-field-label">Nhà xuất bản:</span>{' '}
                    <span className="book-field-val">{publisher}</span>
                  </div>
                )}
                {place && (
                  <div className="book-field">
                    <span className="book-field-label">Nơi xuất bản:</span>{' '}
                    <span className="book-field-val">{place}</span>
                  </div>
                )}
                {year && (
                  <div className="book-field">
                    <span className="book-field-label">Năm xuất bản:</span>{' '}
                    <span className="book-field-val">{year}</span>
                  </div>
                )}
                {edition && (
                  <div className="book-field">
                    <span className="book-field-label">Lần xuất bản:</span>{' '}
                    <span className="book-field-val">{edition}</span>
                  </div>
                )}
                {language && (
                  <div className="book-field">
                    <span className="book-field-label">Ngôn ngữ:</span>{' '}
                    <span className="book-field-val">{language}</span>
                  </div>
                )}
                {series && (
                  <div className="book-field">
                    <span className="book-field-label">Tùng thư / Bộ sách:</span>{' '}
                    <span className="book-field-val">{series}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Nhóm 2: Chuyên ngành & Phân loại nghiệp vụ */}
          {hasClassificationSection && (
            <div className="book-section">
              <div className="book-section-title">
                <Bookmark size={13} />
                <span>Chuyên ngành &amp; Phân loại nghiệp vụ</span>
              </div>
              <div className="book-details-grid">
                {major && (
                  <div className="book-field">
                    <span className="book-field-label">Chuyên ngành đào tạo:</span>{' '}
                    <span className="book-field-val">{major}</span>
                  </div>
                )}
                {docType && (
                  <div className="book-field">
                    <span className="book-field-label">Loại tài liệu:</span>{' '}
                    <span className="book-field-val">{docType}</span>
                  </div>
                )}
                {subject && (
                  <div className="book-field">
                    <span className="book-field-label">Chủ đề:</span>{' '}
                    <span className="book-field-val">{subject}</span>
                  </div>
                )}
                {ddcDisplay && (
                  <div className="book-field">
                    <span className="book-field-label">Chỉ số DDC / Cutter:</span>{' '}
                    <span className="book-field-val">{ddcDisplay}</span>
                  </div>
                )}
                {udc && (
                  <div className="book-field">
                    <span className="book-field-label">Chỉ số UDC:</span>{' '}
                    <span className="book-field-val">{udc}</span>
                  </div>
                )}
                {otherClass && (
                  <div className="book-field">
                    <span className="book-field-label">Phân loại khác:</span>{' '}
                    <span className="book-field-val">{otherClass}</span>
                  </div>
                )}
              </div>

              {keywords.length > 0 && (
                <div className="book-keywords-row">
                  <span className="book-field-label" style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                    <Tag size={12} />
                    <span>Từ khóa:</span>
                  </span>
                  <div className="book-keywords-list">
                    {keywords.map((kw, kIdx) => (
                      <span key={kIdx} className="book-keyword-pill">{kw}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Nhóm 3: Thông số kỹ thuật & Tiêu chuẩn */}
          {hasPhysicalSection && (
            <div className="book-section">
              <div className="book-section-title">
                <Layers size={13} />
                <span>Thông số vật lý &amp; Mã tiêu chuẩn</span>
              </div>
              <div className="book-details-grid">
                {physical && (
                  <div className="book-field">
                    <span className="book-field-label">Khối lượng / Trang:</span>{' '}
                    <span className="book-field-val">{physical}</span>
                  </div>
                )}
                {size && (
                  <div className="book-field">
                    <span className="book-field-label">Khổ sách:</span>{' '}
                    <span className="book-field-val">{size}</span>
                  </div>
                )}
                {isbn && (
                  <div className="book-field">
                    <span className="book-field-label">Mã ISBN:</span>{' '}
                    <span className="book-field-val">{isbn}</span>
                  </div>
                )}
                {price && (
                  <div className="book-field">
                    <span className="book-field-label">Giá tiền:</span>{' '}
                    <span className="book-field-val">{price}</span>
                  </div>
                )}
                {illustration && (
                  <div className="book-field">
                    <span className="book-field-label">Minh họa:</span>{' '}
                    <span className="book-field-val">{illustration}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Nhóm 4: Nhan đề phụ, Tóm tắt & Ghi chú */}
          {hasDescriptionSection && (
            <div className="book-section">
              <div className="book-section-title">
                <FileText size={13} />
                <span>Nội dung &amp; Mô tả chi tiết</span>
              </div>
              <div className="book-details-grid">
                {subtitle && subtitle !== title && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Nhan đề phụ / khác:</span>{' '}
                    <span className="book-field-val">{subtitle}</span>
                  </div>
                )}
                {bilingualTitle && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Nhan đề song song:</span>{' '}
                    <span className="book-field-val"><em>{bilingualTitle}</em></span>
                  </div>
                )}
                {notes && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Ghi chú chung:</span>{' '}
                    <span className="book-field-val">{notes}</span>
                  </div>
                )}
                {copyright && (
                  <div className="book-field full-width">
                    <span className="book-field-label">Bản quyền / Xuất xứ:</span>{' '}
                    <span className="book-field-val">{copyright}</span>
                  </div>
                )}
              </div>
              {abstractText && (
                <div className="book-abstract-box">
                  <div className="book-abstract-header">Tóm tắt nội dung:</div>
                  <div className="book-abstract-content">{abstractText}</div>
                </div>
              )}
            </div>
          )}

          {/* Nhóm 5: Vị trí kho & Mã xếp giá (Số ĐKCB) */}
          {hasLocationSection && (
            <div className="book-section">
              <div className="book-section-title">
                <MapPin size={13} />
                <span>Vị trí &amp; Mã xếp giá trong kho</span>
              </div>

              {locations.length > 0 ? (
                <div className="book-rooms-grid">
                  {locations.map((loc, lIdx) => (
                    <div key={lIdx} className="book-room-card">
                      <div className="book-room-header">
                        <span className="book-room-name">{loc.room}</span>
                        <span className="book-room-badge">SL: {loc.count}</span>
                      </div>
                      <div className="book-room-codes">
                        {loc.accession_numbers &&
                          loc.accession_numbers.slice(0, 12).map((acc, aIdx) => (
                            <span key={aIdx} className="book-accession-code">
                              {acc}
                            </span>
                          ))}
                        {loc.accession_numbers && loc.accession_numbers.length > 12 && (
                          <span className="book-accession-more">
                            +{loc.accession_numbers.length - 12} mã khác
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="book-locations-legacy">
                  {locationTags.map((loc, idx) => (
                    <span key={idx} className="book-location-tag">
                      {loc}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Nhóm 6: Nút liên kết tài liệu trực tuyến */}
          {link && link.startsWith('http') && (
            <div className="book-actions">
              <a
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="book-link-btn"
              >
                <ExternalLink size={13} />
                <span>Đọc tài liệu trực tuyến / Xem mục lục</span>
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
