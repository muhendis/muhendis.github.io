/* A one-line link to the blog at the end of About, as progressive
   enhancement: the page is about the person, so the blog gets a pointer
   (article count + newest title), not a section. No-JS, fetch failure and
   all-drafts states keep the static page, whose hero already links the blog. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';
  var T = {
    en: { label: 'Blog', count: function (n) { return n + ' in-depth articles on AI systems'; },
          latest: 'Latest: ', go: 'Read the blog →' },
    tr: { label: 'Blog', count: function (n) { return 'Yapay zekâ sistemleri üzerine ' + n + ' derinlemesine yazı'; },
          latest: 'En yeni: ', go: 'Bloga git →' }
  }[LANG];

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  fetch(BASE + '/posts.json', { cache: 'no-cache' })
    .then(function (r) { if (!r.ok) throw new Error('http'); return r.json(); })
    .then(function (data) {
      var posts = (data.posts || []).filter(function (p) { return !p.draft; });
      var box = document.querySelector('#about .container');
      if (!posts.length || !box) return;
      posts.sort(function (a, b) {
        if (a.date !== b.date) return a.date < b.date ? 1 : -1;
        return (b.seriesPart || 0) - (a.seriesPart || 0);
      });
      var p = posts[0];
      var title = p.seriesId ? p.title.replace(/^.*?\(\d+\):\s*/, '') : p.title;

      var a = el('a', 'blog-strip');
      a.href = BASE + '/';
      a.appendChild(el('span', 'blog-strip__label', T.label));
      var text = el('span', 'blog-strip__text');
      text.appendChild(el('span', 'blog-strip__count', T.count(posts.length)));
      text.appendChild(el('span', 'blog-strip__latest', T.latest + title));
      a.appendChild(text);
      a.appendChild(el('span', 'blog-strip__go', T.go));
      box.appendChild(a);
    })
    .catch(function () { /* silent: the hero's blog button remains */ });
})();
