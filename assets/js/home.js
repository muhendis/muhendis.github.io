/* Featured writing on the homepage, as progressive enhancement.
   The static page is the executive hero + About; when posts.json has at least
   one published post, this inserts a featured-story section and a "latest" row
   between them. No-JS, fetch failure and all-drafts states are the static page. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';

  var T = {
    en: { featured: 'Featured', latest: 'Latest insights', by: 'By Engin Bozaba',
          min: ' min read' },
    tr: { featured: 'Öne çıkan', latest: 'Son yazılar', by: 'Yazan: Engin Bozaba',
          min: ' dk okuma' }
  }[LANG];

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function fmtDate(iso) {
    return new Date(iso + 'T00:00:00').toLocaleDateString(
      LANG === 'tr' ? 'tr-TR' : 'en-GB',
      { year: 'numeric', month: 'long', day: 'numeric' });
  }

  function byline(p) {
    var m = el('p', 'byline');
    m.appendChild(el('span', 'byline__name', T.by));
    var time = el('time', null, fmtDate(p.date));
    time.setAttribute('datetime', p.date);
    m.append(' · ', time);
    if (p.readingMinutes) m.append(' · ', el('span', null, p.readingMinutes + T.min));
    return m;
  }

  function postHref(p) {
    return BASE + '/post.html?slug=' + encodeURIComponent(p.slug);
  }

  fetch(BASE + '/posts.json', { cache: 'no-cache' })
    .then(function (r) { if (!r.ok) throw new Error('http'); return r.json(); })
    .then(function (data) {
      var posts = (data.posts || [])
        .filter(function (p) { return !p.draft; })
        .sort(function (a, b) { return a.date < b.date ? 1 : -1; });
      if (!posts.length) return;                 // all drafts: bio hero stands

      var main = document.getElementById('main');
      var anchor = main && document.getElementById('about');
      if (!main || !anchor) return;

      // Featured story
      var feat = el('section', 'section featured');
      var fc = el('div', 'container');
      fc.appendChild(el('p', 'eyebrow', T.featured));
      var art = el('article', 'featured__article');
      var h2 = el('h2', 'featured__title');
      var a = el('a', null, posts[0].title);
      a.href = postHref(posts[0]);
      h2.appendChild(a);
      art.appendChild(h2);
      if (posts[0].summary) art.appendChild(el('p', 'featured__summary', posts[0].summary));
      art.appendChild(byline(posts[0]));
      fc.appendChild(art);
      feat.appendChild(fc);
      main.insertBefore(feat, anchor);

      // Latest row (up to 3 more)
      var rest = posts.slice(1, 4);
      if (rest.length) {
        var sec = el('section', 'section latest');
        var lc = el('div', 'container');
        lc.appendChild(el('h2', 'section-label', T.latest));
        var grid = el('div', 'grid-auto');
        rest.forEach(function (p) {
          var card = el('article', 'card card--post');
          if ((p.tags || []).length) card.appendChild(el('p', 'eyebrow', p.tags[0]));
          var t2 = el('h2', 'card__title');
          var ca = el('a', null, p.title);
          ca.href = postHref(p);
          t2.appendChild(ca);
          card.appendChild(t2);
          if (p.summary) card.appendChild(el('p', 'card__desc', p.summary));
          card.appendChild(byline(p));
          grid.appendChild(card);
        });
        lc.appendChild(grid);
        sec.appendChild(lc);
        main.insertBefore(sec, anchor);
      }

    })
    .catch(function () { /* silent: the bio hero is the fallback */ });
})();
