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
          allPosts: 'All English posts' },
    tr: { notFound: 'Yazı bulunamadı.', failIndex: 'Yazı dizini yüklenemedi.',
          jekyll: 'İçerik yüklenemedi. (.nojekyll deposu kökünde var mı?)',
          min: ' dk okuma', by: 'Yazan: Engin Bozaba', noTrans: 'Bu yazı Türkçe olarak mevcut değil.',
          noTransNotice: 'Bu yazının henüz Türkçe çevirisi yok.',
          allPosts: 'Tüm Türkçe yazılar' }
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

            var eyebrow = document.getElementById('post-eyebrow');
            if (eyebrow && (meta.tags || []).length) {
              eyebrow.textContent = meta.tags[0];
              eyebrow.hidden = false;
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
              span.textContent = tag;
              li.appendChild(span);
              tagsEl.appendChild(li);
            });

            marked.setOptions({ gfm: true, breaks: false });
            document.getElementById('post-body').innerHTML = marked.parse(md);

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
