"""
Mobile UI smoke test for chat.html.

Usage:
  python eval/ui_mobile_playwright.py --base http://localhost:8000/chat

Requires:
  pip install playwright
  python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import sys


def _fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Mobile overflow/sidebar smoke test")
    parser.add_argument("--base", default="http://localhost:8000/chat")
    parser.add_argument("--width", type=int, default=390)
    parser.add_argument("--height", type=int, default=844)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("SKIP: playwright is not installed. Run: pip install playwright && python -m playwright install chromium")
        return 2

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height}, is_mobile=True)
        page.goto(args.base, wait_until="networkidle", timeout=30000)
        page.wait_for_selector(".chat-panel", timeout=10000)

        def assert_no_overflow(label: str) -> None:
            overflow = page.evaluate(
                """
                () => {
                  const width = document.documentElement.clientWidth;
                  const selectors = [
                    '.chat-header', '.composer', '.bubble', '.book-card',
                    '.follow-up-chip', '.doc-group-header'
                  ];
                  const bad = [];
                  document.querySelectorAll(selectors.join(',')).forEach((el) => {
                    const style = getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') return;
                    const r = el.getBoundingClientRect();
                    if (r.width > 0 && (r.left < -1 || r.right > width + 1)) {
                      bad.push({ selector: el.className || el.tagName, left: r.left, right: r.right, width });
                    }
                  });
                  return bad;
                }
                """
            )
            if overflow:
                _fail(f"{label}: horizontal overflow detected: {overflow[:3]}")

        assert_no_overflow("initial")

        page.click("#hamburgerBtn")
        page.wait_for_selector("#sidebar.open", timeout=5000)
        page.wait_for_selector(".faculty-header", timeout=10000)
        page.click(".faculty-header")
        page.wait_for_selector(".faculty-row", timeout=5000)
        page.click(".faculty-row")
        page.wait_for_function("!document.querySelector('#sidebar').classList.contains('open')", timeout=10000)

        page.evaluate(
            """
            () => {
              const bubble = document.querySelector('.msg.bot .bubble');
              const wrap = document.createElement('div');
              wrap.style.marginTop = '10px';
              wrap.innerHTML = '<div class="follow-up-chips"><button class="follow-up-chip">Tim sach theo nganh: Thac si nganh Tai chinh Ngan hang va quan tri kinh doanh quoc te rat dai</button></div>';
              bubble.appendChild(wrap);
            }
            """
        )
        assert_no_overflow("with long follow-up")

        browser.close()
    print("PASS: mobile UI smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
