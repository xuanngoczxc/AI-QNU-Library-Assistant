# QNU Chat Widget — Floating Portlet

Widget chat nổi dạng **floating action button (FAB)** để nhúng vào bất kỳ trang web nào.

## Cách dùng

Chỉ cần thêm 1 dòng `<script>` vào trang:

```html
<script src="https://YOUR_DOMAIN/embed/qnu-chat-widget.js"
        data-backend="https://YOUR_DOMAIN"
        data-title="Thư viện QNU"
        data-position="right"
        data-primary-color="#2E6B8A">
</script>
```

## Tuỳ chỉnh

| Thuộc tính | Mặc định | Mô tả |
|---|---|---|
| `data-backend` | `http://localhost:8000` | URL backend FastAPI |
| `data-title` | `"Thư viện QNU"` | Tiêu đề hiển thị trên widget |
| `data-position` | `"right"` | Vị trí: `"right"` hoặc `"left"` |
| `data-primary-color` | `"#2E6B8A"` | Màu chủ đạo (hex) |
| `data-welcome` | (mặc định) | Lời chào tuỳ chỉnh (Markdown) |
| `data-agent-id` | `""` | ID tác tử (nếu backend hỗ trợ) |

## File cấu trúc

```
embed/
├── qnu-chat-widget.js   # Script widget chính (tự động inject CSS + HTML)
├── demo.html            # Trang demo để test widget
└── README.md            # Hướng dẫn này
```

## Tính năng

- ✅ Icon nổi ở góc màn hình, click để mở/đóng
- ✅ Gọi API `/chat` của backend FastAPI (hỗ trợ streaming SSE)
- ✅ Hiển thị danh sách tài liệu tham khảo dạng card
- ✅ Gợi ý câu hỏi tiếp theo (follow-up chips)
- ✅ Responsive — chiếm toàn màn hình trên mobile
- ✅ Animation mượt, giao diện tối giản
- ✅ Tuỳ chỉnh màu sắc, vị trí, lời chào
