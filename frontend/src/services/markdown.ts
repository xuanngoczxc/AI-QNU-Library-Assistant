import { marked } from 'marked';
import DOMPurify from 'dompurify';

// Configure marked renderer options
marked.setOptions({
  gfm: true,
  breaks: true,
});

export function renderMarkdown(markdownText: string): string {
  if (!markdownText) return '';
  try {
    const rawHtml = marked.parse(markdownText) as string;
    return DOMPurify.sanitize(rawHtml, {
      ALLOWED_TAGS: [
        'p', 'br', 'strong', 'em', 'b', 'i', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'ul', 'ol', 'li', 'code', 'pre', 'blockquote', 'a', 'span', 'div', 'table',
        'thead', 'tbody', 'tr', 'th', 'td', 'hr',
      ],
      ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'style'],
    });
  } catch (err) {
    console.error('Markdown parse error:', err);
    return markdownText;
  }
}
