/* Renders one post: metadata from posts.json, body from the .md file.
   posts.json is the single source of truth for title/date/tags so the
   heading, <title> and OG tags cannot drift from each other. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var OTHER = LANG === 'en' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';
  var ORIGIN = 'https://muhendis.github.io';

  /* Allowlist. Without this the slug parameter is concatenated straight into
     a fetch path (?slug=../../something). */
  var SLUG_RE = /^[a-z0-9]+(?:-[a-z0-9]+)*$/;

  var T = {
    en: { notFound: 'Post not found.', failIndex: 'Could not load the post index.',
          jekyll: 'Content failed to load. (Is .nojekyll present at repo root?)',
          min: ' min read', by: 'By Engin Bozaba', noTrans: 'This post is not available in English.',
          noTransNotice: 'This post has no English version yet.',
          allPosts: 'All English posts',
          markRead: 'Mark as read', isRead: '✓ Read · mark as unread',
          toc: 'Contents', tocNav: 'Table of contents', inArticle: 'In this article',
          pct: function (p) { return p + '% read'; },
          left: function (m) { return '~' + m + ' min left'; },
          top: '↑ Back to top', part: function (n, m) { return 'Part ' + n + ' of ' + m; },
          prev: '← Previous', next: 'Next →', seriesNav: 'Series navigation',
          copyLink: 'Copy link to this section', copied: 'Link copied', close: 'Close' },
    tr: { notFound: 'Yazı bulunamadı.', failIndex: 'Yazı dizini yüklenemedi.',
          jekyll: 'İçerik yüklenemedi. (.nojekyll deposu kökünde var mı?)',
          min: ' dk okuma', by: 'Yazan: Engin Bozaba', noTrans: 'Bu yazı Türkçe olarak mevcut değil.',
          noTransNotice: 'Bu yazının henüz Türkçe çevirisi yok.',
          allPosts: 'Tüm Türkçe yazılar',
          markRead: 'Okundu olarak işaretle', isRead: '✓ Okundu · okunmadı yap',
          toc: 'İçindekiler', tocNav: 'İçindekiler', inArticle: 'Bu yazıda',
          pct: function (p) { return '%' + p + ' okundu'; },
          left: function (m) { return '~' + m + ' dk kaldı'; },
          top: '↑ Başa dön', part: function (n, m) { return 'Bölüm ' + n + ' / ' + m; },
          prev: '← Önceki', next: 'Sonraki →', seriesNav: 'Seri gezintisi',
          copyLink: 'Bu bölümün bağlantısını kopyala', copied: 'Bağlantı kopyalandı', close: 'Kapat' }
  };
  var t = T[LANG];
  var tOther = T[OTHER];

  var statusEl = document.getElementById('post-status');
  var articleEl = document.getElementById('post-article');

  function fail(msg) {
    statusEl.hidden = false;
    statusEl.textContent = msg;
    articleEl.hidden = true;
    document.title = msg + ' — Engin Bozaba';
  }

  /* Math source goes into the DOM as text, not markup: KaTeX reads it back
     from textContent, so anything that looks like a tag must stay inert. */
  function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function fmtDate(iso) {
    return new Date(iso + 'T00:00:00').toLocaleDateString(
      LANG === 'tr' ? 'tr-TR' : 'en-GB',
      { year: 'numeric', month: 'long', day: 'numeric' });
  }

  function setMeta(attr, key, value) {
    var m = document.head.querySelector('meta[' + attr + '="' + key + '"]');
    if (!m) { m = document.createElement('meta'); m.setAttribute(attr, key); document.head.appendChild(m); }
    m.setAttribute('content', value);
  }

  /* post.html is a shared shell, so it carries no static hreflang: that would
     claim every post has a translation. Set it only when one exists. */
  function wireTranslation(translationSlug) {
    var link = document.querySelector('.lang-switch');
    if (!link) return;

    if (translationSlug && SLUG_RE.test(translationSlug)) {
      link.href = '/' + OTHER + '/blog/post.html?slug=' + encodeURIComponent(translationSlug);
      var alt = document.createElement('link');
      alt.rel = 'alternate'; alt.hreflang = OTHER;
      alt.href = ORIGIN + '/' + OTHER + '/blog/post.html?slug=' + encodeURIComponent(translationSlug);
      document.head.appendChild(alt);
      return;
    }

    /* No translation: disable the control rather than silently sending the
       reader to the other language's index, which is not what they clicked. */
    var span = document.createElement('span');
    span.className = 'lang-switch is-unavailable';
    span.textContent = link.textContent;
    span.setAttribute('aria-disabled', 'true');
    span.setAttribute('lang', OTHER);
    span.title = tOther.noTrans;
    link.replaceWith(span);

    var notice = document.getElementById('translation-notice');
    if (notice) {
      notice.hidden = false;
      notice.textContent = tOther.noTransNotice + ' ';
      var a = document.createElement('a');
      a.href = '/' + OTHER + '/blog/';
      a.setAttribute('lang', OTHER);
      a.textContent = tOther.allPosts;
      notice.appendChild(a);
    }
  }

  function injectJsonLd(meta, slug) {
    var s = document.createElement('script');
    s.type = 'application/ld+json';
    s.textContent = JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'BlogPosting',
      headline: meta.title,
      description: meta.summary,
      datePublished: meta.date,
      dateModified: meta.updated || meta.date,
      inLanguage: LANG,
      keywords: (meta.tags || []).join(', '),
      mainEntityOfPage: ORIGIN + '/' + LANG + '/blog/post.html?slug=' + slug,
      author: { '@type': 'Person', name: 'Engin Bozaba', url: ORIGIN + '/' + LANG + '/' }
    });
    document.head.appendChild(s);
  }

  /* Per-browser reading state for the blog index. 'lastOpened' powers
     "continue reading"; 'readPosts' only gains a slug once the reader
     reaches the end of the article (or marks it by hand), so module
     progress reflects reading rather than clicking. Storage can be
     blocked, so all of it is best-effort. */
  function loadRead() {
    try {
      var r = JSON.parse(localStorage.getItem('readPosts') || '[]');
      return Array.isArray(r) ? r : [];
    } catch (e) { return []; }
  }

  function saveRead(read) {
    try { localStorage.setItem('readPosts', JSON.stringify(read)); } catch (e) {}
  }

  function trackReading(slug) {
    try { localStorage.setItem('lastOpened', slug); } catch (e) {}

    var btn = document.getElementById('read-toggle');
    var end = document.getElementById('post-end');

    function isRead() { return loadRead().indexOf(slug) !== -1; }
    function setRead(on) {
      var read = loadRead().filter(function (s) { return s !== slug; });
      if (on) read.push(slug);
      saveRead(read);
      sync();
    }
    function sync() {
      if (!btn) return;
      var on = isRead();
      btn.textContent = on ? t.isRead : t.markRead;
      btn.setAttribute('aria-pressed', String(on));
    }

    if (btn) {
      btn.hidden = false;
      btn.addEventListener('click', function () { setRead(!isRead()); });
      sync();
    }

    /* Reaching the end marks the post read once; un-marking by hand later
       sticks for this page view. */
    if (end && 'IntersectionObserver' in window && !isRead()) {
      var io = new IntersectionObserver(function (entries) {
        if (!entries[0].isIntersecting) return;
        io.disconnect();
        if (!isRead()) setRead(true);
      });
      /* Wait a tick so the observer does not fire before the body renders. */
      setTimeout(function () { io.observe(end); }, 1500);
    }
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  /* This post's place in its module: the ordered, published parts. */
  function seriesInfo(meta, idx) {
    if (!meta.seriesId) return null;
    var s = (idx.series || []).filter(function (x) { return x.id === meta.seriesId; })[0];
    if (!s) return null;
    var bySlug = {};
    (idx.posts || []).forEach(function (p) { if (!p.draft) bySlug[p.slug] = p; });
    var parts = (s.posts || []).map(function (sl) { return bySlug[sl]; }).filter(Boolean);
    for (var i = 0; i < parts.length; i++) {
      if (parts[i].slug === meta.slug) return { series: s, parts: parts, i: i };
    }
    return null;
  }

  function postHref(p) { return 'post.html?slug=' + encodeURIComponent(p.slug); }

  function buildSeriesNav(info) {
    var nav = document.getElementById('series-nav');
    if (!nav || !info || info.parts.length < 2) return;
    nav.setAttribute('aria-label', t.seriesNav);
    [[info.parts[info.i - 1], t.prev, 'prev'], [info.parts[info.i + 1], t.next, 'next']].forEach(function (x) {
      if (!x[0]) { nav.appendChild(el('span', 'series-nav__gap')); return; }
      var a = el('a', 'series-nav__link series-nav__link--' + x[2]);
      a.href = postHref(x[0]);
      a.appendChild(el('span', 'series-nav__dir', x[1]));
      a.appendChild(el('span', 'series-nav__title', x[0].title));
      nav.appendChild(a);
    });
    nav.hidden = false;
  }

  /* Reading navigation: a sticky contents rail on wide screens, a floating
     button + sheet on narrow ones, and a progress bar on both. All of it is
     built from the rendered headings, so it never drifts from the body. */
  function buildReadingNav(meta, idx) {
    var body = document.getElementById('post-body');
    var aside = document.getElementById('post-aside');
    var heads = [].slice.call(body.querySelectorAll('h2, h3')).filter(function (h) { return h.id; });

    /* The hand-written "In this article" list duplicates the rail. */
    [].slice.call(body.querySelectorAll(':scope > p')).forEach(function (p) {
      var next = p.nextElementSibling;
      if (p.textContent.trim() === t.inArticle && next && next.tagName === 'UL') {
        p.classList.add('inline-toc');
        next.classList.add('inline-toc');
      }
    });

    var info = seriesInfo(meta, idx);
    buildSeriesNav(info);
    if (heads.length < 3) return;

    /* Entries keep a clone of the heading so inline math renders in the list. */
    var entries = heads.map(function (h) {
      var label = document.createElement('span');
      [].slice.call(h.childNodes).forEach(function (n) { label.appendChild(n.cloneNode(true)); });
      return { h: h, level: h.tagName === 'H2' ? 2 : 3, label: label, links: [] };
    });
    var parentOf = [];
    var lastH2 = -1;
    entries.forEach(function (e, i) {
      if (e.level === 2) lastH2 = i;
      parentOf[i] = e.level === 2 ? i : lastH2;
    });

    function tocList() {
      var ol = el('ol', 'toc');
      var cur = null;
      entries.forEach(function (e) {
        var li = el('li', 'toc__item toc__item--h' + e.level);
        var a = el('a', 'toc__link');
        a.href = '#' + e.h.id;
        a.appendChild(e.links.length ? e.label.cloneNode(true) : e.label);
        e.links.push(a);
        li.appendChild(a);
        if (e.level === 2 || !cur) {
          ol.appendChild(li);
          cur = li;
        } else {
          var sub = cur.querySelector('.toc__sub');
          if (!sub) { sub = el('ol', 'toc__sub'); cur.appendChild(sub); }
          sub.appendChild(li);
        }
      });
      return ol;
    }

    function progressBlock() {
      var p = el('p', 'toc-progress');
      p.setAttribute('aria-live', 'off');
      return p;
    }

    /* Wide screens: the rail. */
    var asideProgress = progressBlock();
    aside.setAttribute('aria-label', t.tocNav);
    if (info) {
      var card = el('a', 'toc-series');
      card.href = './?module=' + encodeURIComponent(info.series.id);
      card.appendChild(el('span', 'toc-series__title', info.series.title));
      if (info.parts.length > 1) {
        card.appendChild(el('span', 'toc-series__part', t.part(info.i + 1, info.parts.length)));
        var dots = el('span', 'toc-series__dots');
        dots.setAttribute('aria-hidden', 'true');
        info.parts.forEach(function (_, k) {
          dots.appendChild(el('span', 'toc-series__dot' + (k < info.i ? ' is-done' : k === info.i ? ' is-current' : '')));
        });
        card.appendChild(dots);
      }
      aside.appendChild(card);
    }
    aside.appendChild(el('p', 'toc-heading', t.toc));
    var asideScroll = el('div', 'toc-scroll');
    asideScroll.appendChild(tocList());
    aside.appendChild(asideScroll);
    var asideFoot = el('div', 'toc-foot');
    asideFoot.appendChild(asideProgress);
    var topBtn = el('button', 'toc-top', t.top);
    topBtn.type = 'button';
    topBtn.addEventListener('click', function () {
      window.scrollTo({ top: 0 });
      history.replaceState(null, '', location.pathname + location.search);
    });
    asideFoot.appendChild(topBtn);
    aside.appendChild(asideFoot);
    aside.hidden = false;
    document.getElementById('post-article').classList.add('has-aside');

    /* Narrow screens: floating button + modal sheet. */
    var R = 9, C = 2 * Math.PI * R;
    var fab = el('button', 'toc-fab');
    fab.type = 'button';
    fab.setAttribute('aria-haspopup', 'dialog');
    fab.innerHTML = '<svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">' +
      '<circle cx="11" cy="11" r="' + R + '" class="toc-fab__track"/>' +
      '<circle cx="11" cy="11" r="' + R + '" class="toc-fab__ring" stroke-dasharray="' + C + '" stroke-dashoffset="' + C + '"/></svg>';
    fab.appendChild(el('span', 'toc-fab__label', t.toc));
    var ring = fab.querySelector('.toc-fab__ring');

    var sheet = el('dialog', 'toc-sheet');
    sheet.setAttribute('aria-label', t.tocNav);
    var sheetHead = el('div', 'toc-sheet__head');
    sheetHead.appendChild(el('p', 'toc-heading', t.toc));
    var closeBtn = el('button', 'toc-sheet__close', '×');
    closeBtn.type = 'button';
    closeBtn.setAttribute('aria-label', t.close);
    closeBtn.addEventListener('click', function () { sheet.close(); });
    sheetHead.appendChild(closeBtn);
    sheet.appendChild(sheetHead);
    var sheetScroll = el('div', 'toc-scroll');
    sheetScroll.appendChild(tocList());
    sheet.appendChild(sheetScroll);
    var sheetProgress = progressBlock();
    sheet.appendChild(sheetProgress);
    /* Close before the jump so the page is no longer inert when it scrolls. */
    sheet.addEventListener('click', function (ev) {
      if (ev.target === sheet) { sheet.close(); return; }
      if (ev.target.closest && ev.target.closest('a.toc__link')) sheet.close();
    });
    fab.addEventListener('click', function () {
      if (typeof sheet.showModal !== 'function') return;
      sheet.showModal();
      var on = sheet.querySelector('.toc__link[aria-current]');
      if (on) keepVisible(sheetScroll, on);
    });
    document.body.appendChild(fab);
    document.body.appendChild(sheet);

    var bar = el('div', 'read-progress');
    bar.setAttribute('aria-hidden', 'true');
    bar.appendChild(el('span', 'read-progress__fill'));
    document.body.appendChild(bar);
    var fill = bar.firstChild;

    function keepVisible(box, link) {
      var b = box.getBoundingClientRect(), r = link.getBoundingClientRect();
      if (r.top < b.top + 8) box.scrollTop -= (b.top + 8 - r.top);
      else if (r.bottom > b.bottom - 8) box.scrollTop += (r.bottom - b.bottom + 8);
    }

    var active = -2;
    var OFFSET = 110; /* sticky header + breathing room */
    function update() {
      var i = -1;
      for (var k = 0; k < entries.length; k++) {
        if (entries[k].h.getBoundingClientRect().top <= OFFSET) i = k; else break;
      }
      if (i !== active) {
        active = i;
        var open = i === -1 ? -1 : parentOf[i];
        entries.forEach(function (e, k) {
          e.links.forEach(function (a) {
            if (k === i) a.setAttribute('aria-current', 'location');
            else a.removeAttribute('aria-current');
            var li = a.parentNode;
            if (e.level === 2) li.classList.toggle('is-open', k === open);
            a.classList.toggle('is-past', k < i);
          });
        });
        if (i !== -1) keepVisible(asideScroll, entries[i].links[0]);
      }

      var r = body.getBoundingClientRect();
      var span = Math.max(1, r.height - innerHeight);
      var p = Math.min(1, Math.max(0, -r.top / span));
      fill.style.transform = 'scaleX(' + p + ')';
      ring.setAttribute('stroke-dashoffset', String(C * (1 - p)));
      var pct = Math.round(p * 100);
      var txt = t.pct(pct);
      if (meta.readingMinutes && pct < 98) txt += ' · ' + t.left(Math.max(1, Math.ceil(meta.readingMinutes * (1 - p))));
      asideProgress.textContent = txt;
      sheetProgress.textContent = txt;
      fab.classList.toggle('is-visible', r.top < 0);
    }

    var queued = false;
    function onScroll() {
      if (queued) return;
      queued = true;
      requestAnimationFrame(function () { queued = false; update(); });
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();
  }

  /* Section links: hover "#" that also copies the full URL. Added after the
     contents are built so the "#" never leaks into a contents label. */
  function addHeadingAnchors() {
    document.querySelectorAll('#post-body h2[id], #post-body h3[id]').forEach(function (h) {
      var a = el('a', 'heading-anchor', '#');
      a.href = '#' + h.id;
      a.setAttribute('aria-label', t.copyLink);
      a.addEventListener('click', function () {
        var url = location.origin + location.pathname + location.search + '#' + h.id;
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(url).then(function () {
            a.dataset.copied = t.copied;
            setTimeout(function () { delete a.dataset.copied; }, 1600);
          }, function () {});
        }
      });
      h.appendChild(a);
    });
  }

  /* The body renders after load, so the browser's own jump to #section has
     already missed. Redo it, and again once math has reflowed the page,
     unless the reader has started scrolling by then. */
  var readerMoved = false;
  ['wheel', 'touchmove', 'keydown', 'mousedown'].forEach(function (ev) {
    window.addEventListener(ev, function () { readerMoved = true; }, { passive: true, once: true });
  });
  function jumpToHash() {
    if (readerMoved || location.hash.length < 2) return;
    var target;
    try { target = document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (e) { return; }
    if (target) target.scrollIntoView({ behavior: 'instant', block: 'start' });
  }

  function run() {
    var slug = new URLSearchParams(location.search).get('slug') || '';
    if (!SLUG_RE.test(slug)) return fail(t.notFound);

    fetch(BASE + '/posts.json', { cache: 'no-cache' })
      .then(function (r) { if (!r.ok) throw new Error('index'); return r.json(); })
      .catch(function () { throw new Error('index'); })
      .then(function (idx) {
        var meta = (idx.posts || []).filter(function (p) {
          return p.slug === slug && !p.draft;
        })[0];
        if (!meta) throw new Error('missing');

        return fetch(BASE + '/posts/' + slug + '.md', { cache: 'no-cache' })
          .then(function (res) {
            if (!res.ok) throw new Error('missing');
            return res.text();
          })
          .then(function (md) {
            /* Without .nojekyll, Pages can return 200 with an HTML body
               instead of the raw Markdown. Name the cause rather than
               rendering the 404 page into the article. */
            if (/^\s*<(!doctype|html)\b/i.test(md)) throw new Error('jekyll');

            document.getElementById('post-title').textContent = meta.title;

            trackReading(slug);

            var eyebrow = document.getElementById('post-eyebrow');
            if (eyebrow) {
              if (meta.seriesId) {
                var sList = idx.series || [];
                var sObj = null;
                for (var si = 0; si < sList.length; si++) {
                  if (sList[si].id === meta.seriesId) { sObj = sList[si]; break; }
                }
                var sTitle = sObj ? sObj.title : (LANG === 'tr' ? 'Seri' : 'Series');
                eyebrow.textContent = sTitle + (meta.seriesPart ? ' · ' + (LANG === 'tr' ? 'Bölüm ' : 'Part ') + meta.seriesPart : '');
                eyebrow.hidden = false;
              } else if ((meta.tags || []).length) {
                eyebrow.textContent = meta.tags[0].replace(/-/g, ' ');
                eyebrow.hidden = false;
              }
            }
            var bylineEl = document.getElementById('post-byline');
            if (bylineEl) bylineEl.textContent = t.by;

            var timeEl = document.getElementById('post-date');
            timeEl.textContent = fmtDate(meta.date);
            timeEl.setAttribute('datetime', meta.date);

            var readEl = document.getElementById('post-reading');
            if (meta.readingMinutes) readEl.textContent = meta.readingMinutes + t.min;
            else readEl.hidden = true;

            var tagsEl = document.getElementById('post-tags');
            (meta.tags || []).forEach(function (tag) {
              var li = document.createElement('li');
              var span = document.createElement('span');
              span.className = 'tag';
              span.textContent = tag.replace(/-/g, ' ');
              li.appendChild(span);
              tagsEl.appendChild(li);
            });

            marked.setOptions({ gfm: true, breaks: false });

            /* Math must be claimed before the inline lexer runs: otherwise the
               underscores in \mathbf{x}_{scaled} are parsed as emphasis and the
               LaTeX reaches KaTeX already broken. These tokens emit the source
               verbatim inside a marker element for KaTeX to render in place. */
            marked.use({
              extensions: [
                {
                  name: 'mathBlock', level: 'block',
                  start: function (src) { return src.indexOf('$$'); },
                  tokenizer: function (src) {
                    var m = /^\$\$([\s\S]+?)\$\$(?:\n|$)/.exec(src);
                    if (m) return { type: 'mathBlock', raw: m[0], text: m[1].trim() };
                  },
                  renderer: function (token) {
                    return '<div class="math-block">' + escapeHtml(token.text) + '</div>';
                  }
                },
                {
                  name: 'mathInline', level: 'inline',
                  start: function (src) { return src.indexOf('$'); },
                  tokenizer: function (src) {
                    var m = /^\$([^\s$][^$\n]*?)\$(?!\d)/.exec(src);
                    if (m) return { type: 'mathInline', raw: m[0], text: m[1].trim() };
                  },
                  renderer: function (token) {
                    return '<span class="math-inline">' + escapeHtml(token.text) + '</span>';
                  }
                }
              ]
            });

            document.getElementById('post-body').innerHTML = marked.parse(md);

            /* marked v5+ dropped headerIds; in-page ToC links need them. */
            document.querySelectorAll('#post-body h2, #post-body h3').forEach(function (h) {
              h.id = h.textContent.toLowerCase()
                .replace(/[^\p{L}\p{N}\s-]/gu, '')
                .trim()
                .replace(/\s+/g, '-');
            });

            buildReadingNav(meta, idx);
            addHeadingAnchors();

            /* KaTeX: lazy-load stylesheet and renderer only when a post has math. */
            var mathNodes = document.querySelectorAll('#post-body .math-block, #post-body .math-inline, .toc .math-inline');
            if (mathNodes.length) {
              var kcss = document.createElement('link');
              kcss.rel = 'stylesheet';
              kcss.href = '../../assets/js/vendor/katex/katex.min.css';
              document.head.appendChild(kcss);

              var kjs = document.createElement('script');
              kjs.src = '../../assets/js/vendor/katex/katex.min.js';
              kjs.onload = function () {
                mathNodes.forEach(function (node) {
                  var display = node.classList.contains('math-block');
                  try {
                    window.katex.render(node.textContent, node, {
                      displayMode: display,
                      throwOnError: false,
                      output: 'html'
                    });
                  } catch (e) { /* leave the LaTeX source visible as text */ }
                });
                jumpToHash();
              };
              document.head.appendChild(kjs);
            }

            /* Mermaid diagrams: lazy-load the renderer only when a post uses them. */
            var mermaidBlocks = document.querySelectorAll('#post-body code.language-mermaid');
            if (mermaidBlocks.length) {
              mermaidBlocks.forEach(function (code) {
                var holder = document.createElement('pre');
                holder.className = 'mermaid';
                holder.textContent = code.textContent;
                code.parentElement.replaceWith(holder);
              });
              var mjs = document.createElement('script');
              mjs.src = '../../assets/js/vendor/mermaid.min.js';
              mjs.onload = function () {
                var root = document.documentElement;
                var dark = root.getAttribute('data-theme') === 'dark' ||
                  (root.getAttribute('data-theme') !== 'light' &&
                   window.matchMedia('(prefers-color-scheme: dark)').matches);
                try {
                  window.mermaid.initialize({ startOnLoad: false, theme: dark ? 'dark' : 'neutral', fontFamily: 'inherit' });
                  var run = window.mermaid.run({ querySelector: '#post-body pre.mermaid' });
                  if (run && run.catch) run.catch(function () {});
                } catch (e) { /* diagram source stays readable as text */ }
              };
              document.head.appendChild(mjs);
            }

            var url = ORIGIN + '/' + LANG + '/blog/post.html?slug=' + slug;
            document.title = meta.title + ' — Engin Bozaba';
            setMeta('name', 'description', meta.summary || '');
            setMeta('property', 'og:title', meta.title);
            setMeta('property', 'og:description', meta.summary || '');
            setMeta('property', 'og:url', url);
            setMeta('property', 'og:type', 'article');
            var canon = document.querySelector('link[rel="canonical"]');
            if (canon) canon.href = url;

            injectJsonLd(meta, slug);
            wireTranslation(meta.translationSlug);

            statusEl.hidden = true;
            articleEl.hidden = false;
            jumpToHash();
          });
      })
      .catch(function (err) {
        if (err && err.message === 'index') return fail(t.failIndex);
        if (err && err.message === 'jekyll') return fail(t.jekyll);
        fail(t.notFound);
      });
  }

  run();
})();
