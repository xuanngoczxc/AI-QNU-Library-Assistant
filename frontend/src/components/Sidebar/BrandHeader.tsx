import React from 'react';

export const BrandHeader: React.FC = () => {
  return (
    <div className="brand">
      <div className="brand-mark">
        <img src="/SHL-logo.png" alt="SHL Logo" className="brand-logo" />
      </div>
      <div className="brand-text">
        <h1>Thư viện QNU</h1>
        <p>Tra cứu Sách &amp; Tài liệu</p>
      </div>
    </div>
  );
};
