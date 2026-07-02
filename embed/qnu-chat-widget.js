(function () {
  'use strict';

  var S = document.currentScript || document.scripts[document.scripts.length - 1];
  var BE = (S.getAttribute('data-backend') || 'http://localhost:8000').replace(/\/+$/, '');
  var posL = (S.getAttribute('data-position') || 'right') === 'left';

  // ── Google Fonts ──
  if (!document.getElementById('qnu-gf')) {
    var gf = document.createElement('link');
    gf.id = 'qnu-gf'; gf.rel = 'stylesheet';
    gf.href = 'https://fonts.googleapis.com/css2?family=Roboto:wght@400;600;700&display=swap';
    document.head.appendChild(gf);
  }

  var _S = 'qnu-wc';
  if (!document.getElementById(_S)) {
    var p = posL ? 'left:20px;right:auto' : 'right:20px;left:auto';
    var st = document.createElement('style');
    st.id = _S;
    var dr = posL ? 'row' : 'row-reverse';
    st.textContent = '' +
      '.qnu-w *,.qnu-w *::before,.qnu-w *::after{box-sizing:border-box;margin:0;padding:0}' +
      '.qnu-f{position:fixed;bottom:20px;' + p + ';z-index:2147483645;cursor:pointer;' +
      'display:flex;align-items:center;gap:10px;' +
      'background:#fff;border:2px solid #E0E0E0;border-radius:28px;' +
      'box-shadow:0 4px 16px rgba(0,0,0,.18),0 2px 6px rgba(0,0,0,.10);' +
      'padding:6px 16px 6px 6px;flex-direction:' + dr + ';' +
      'transition:transform .2s,box-shadow .25s;animation:qf .4s}' +
      '.qnu-f:hover{transform:scale(1.05);box-shadow:0 6px 24px rgba(0,0,0,.25)}' +
      '.qnu-f:active{transform:scale(.97)}' +
      '.qnu-f .qnu-flogo{width:44px;height:44px;border-radius:50%;overflow:hidden;flex-shrink:0}' +
      '.qnu-f .qnu-flogo img{width:100%;height:100%;object-fit:cover;display:block}' +
      '.qnu-f .qnu-flabel{font-family:Roboto,-apple-system,sans-serif;font-size:13px;font-weight:600;color:#1A2A3A;line-height:1.3;white-space:nowrap;letter-spacing:-.01em}' +
      '.qnu-f .qnu-flabel small{display:block;font-size:10px;font-weight:400;color:#6B7B8E;letter-spacing:0}' +
      '@keyframes qf{from{opacity:0;transform:scale(0)}to{opacity:1;transform:scale(1)}}';
    document.head.appendChild(st);
  }

  var fab = document.createElement('button');
  fab.className = 'qnu-f qnu-w';
  fab.setAttribute('aria-label', 'Thu vien QNU - Chatbot tra cuu sach');

  var wrap = document.createElement('span');
  wrap.className = 'qnu-flogo';
  var img = document.createElement('img');
  img.src = BE + '/static/SHL-logo.png';
  img.alt = 'SHL';
  wrap.appendChild(img);
  fab.appendChild(wrap);

  var label = document.createElement('span');
  label.className = 'qnu-flabel';
  label.innerHTML = 'Tra c\u1EE9u S\u00E1ch<small>Th\u01B0 vi\u1EC7n QNU</small>';
  fab.appendChild(label);

  document.body.appendChild(fab);
  fab.addEventListener('click', function () {
    window.location.href = BE + '/chat';
  });
})();
