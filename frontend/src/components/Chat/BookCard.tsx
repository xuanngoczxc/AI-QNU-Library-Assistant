import React, { useState } from 'react';
import type { BookSource } from '../../types';
import { BookOpen, ChevronDown, ExternalLink, MapPin } from 'lucide-react';

interface BookCardProps {
  book: BookSource;
  index?: number;
}

export const BookCard: React.FC<BookCardProps> = ({ book }) => {
  const [isOpen, setIsOpen] = useState(false);

  const title = book.title || 'Tài liệu không có tên';
  const author = book.author && book.author !== 'Unknown' && book.author !== 'N/A' ? book.author : '';
  const year = book.year && book.year !== 'N/A' ? book.year : '';
  const publisher = book.publisher && book.publisher !== 'Unknown' && book.publisher !== 'N/A' ? book.publisher : '';
  const major = book.major && book.major !== 'Unknown' && book.major !== 'N/A' ? book.major : '';
  const docType = book.doc_type && book.doc_type !== 'N/A' ? book.doc_type : 'Sách';
  const ddc = book.ddc && book.ddc !== 'N/A' ? book.ddc : '';
  const subject = book.subject && book.subject !== 'N/A' ? book.subject : '';
  const location = book.location || '';
  const link = book.link || '';

  const subtitleParts = [author, year, publisher].filter(Boolean);
  const subtitle = subtitleParts.join(' • ');

  // Parse location strings into tags
  const locationTags = location
    ? location.split(';').map((loc) => loc.trim()).filter(Boolean)
    : [];

  return (
    <div className={`book-card ${isOpen ? 'open' : ''}`}>
      <div className="book-card-header" onClick={() => setIsOpen((prev) => !prev)}>
        <div className="book-card-icon">
          <BookOpen size={16} />
        </div>

        <div className="book-card-info">
          <div className="book-card-title">{title}</div>
          {subtitle && <div className="book-card-subtitle">{subtitle}</div>}
        </div>

        <ChevronDown size={16} className="book-card-chevron" />
      </div>

      {isOpen && (
        <div className="book-details-wrapper">
          <div className="book-details-grid">
            {author && (
              <div className="book-field">
                <span className="book-field-label">Tác giả:</span> {author}
              </div>
            )}
            {year && (
              <div className="book-field">
                <span className="book-field-label">Năm xuất bản:</span> {year}
              </div>
            )}
            {publisher && (
              <div className="book-field">
                <span className="book-field-label">Nhà xuất bản:</span> {publisher}
              </div>
            )}
            {major && (
              <div className="book-field">
                <span className="book-field-label">Chuyên ngành:</span> {major}
              </div>
            )}
            {docType && (
              <div className="book-field">
                <span className="book-field-label">Loại tài liệu:</span> {docType}
              </div>
            )}
            {ddc && (
              <div className="book-field">
                <span className="book-field-label">Phân loại DDC:</span> {ddc}
              </div>
            )}
            {subject && (
              <div className="book-field">
                <span className="book-field-label">Chủ đề:</span> {subject}
              </div>
            )}
          </div>

          {locationTags.length > 0 && (
            <div className="book-locations-section">
              <div className="book-locations-title" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <MapPin size={13} />
                <span>Vị trí &amp; Mã xếp giá trong kho:</span>
              </div>
              <div>
                {locationTags.map((loc, idx) => (
                  <span key={idx} className="book-location-tag">
                    {loc}
                  </span>
                ))}
              </div>
            </div>
          )}

          {link && link.startsWith('http') && (
            <div className="book-actions">
              <a
                href={link}
                target="_blank"
                rel="noopener noreferrer"
                className="book-link-btn"
              >
                <ExternalLink size={13} />
                <span>Mở liên kết tài liệu</span>
              </a>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
