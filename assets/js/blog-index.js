/* Renders the post list for the current language from posts.json.
   One request total: summaries live in the manifest, not in the .md files. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';

  var T = {
    en: { empty: 'No posts published yet.', fail: 'Could not load posts.',
          min: ' min read', all: 'All', by: 'By Engin Bozaba' },
    tr: { empty: 'Henüz yazı yayımlanmadı.', fail: 'Yazılar yüklenemedi.',
          min: ' dk okuma', all: 'Tümü', by: 'Yazan: Engin Bozaba' }
  }[LANG];

  var listEl = document.getElementById('post-list');
  var statusEl = document.getElementById('list-status');
  var filterEl = document.getElementById('tag-filter');
  var posts = [];
  var activeTag = null;

  function fmtDate(iso) {
    var d = new Date(iso + 'T00:00:00');
    return d.toLocaleDateString(LANG === 'tr' ? 'tr-TR' : 'en-GB',
      { year: 'numeric', month: 'long', day: 'numeric' });
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;   // textContent: never innerHTML for data
    return n;
  }

  function render() {
    var shown = activeTag
      ? posts.filter(function (p) { return (p.tags || []).indexOf(activeTag) !== -1; })
      : posts;

    listEl.replaceChildren();
    if (!shown.length) { statusEl.hidden = false; statusEl.textContent = T.empty; return; }
    statusEl.hidden = true;

    shown.forEach(function (p, i) {
      var li = el('li', i === 0 ? 'post-item post-item--lead' : 'post-item');
      var h2 = el('h2', 'post-item__title');
      var a = el('a', null, p.title);
      a.href = BASE + '/post.html?slug=' + encodeURIComponent(p.slug);
      h2.appendChild(a);

      var meta = el('div', 'post-item__meta byline');
      meta.appendChild(el('span', 'byline__name', T.by));
      var time = el('time', null, fmtDate(p.date));
      time.setAttribute('datetime', p.date);
      meta.appendChild(time);
      if (p.readingMinutes) meta.appendChild(el('span', null, p.readingMinutes + T.min));

      li.append(h2, meta);
      if (p.summary) li.appendChild(el('p', 'post-item__summary', p.summary));

      if ((p.tags || []).length) {
        var ul = el('ul', 'tags');
        p.tags.forEach(function (t) {
          var tli = el('li');
          tli.appendChild(el('span', 'tag', t));
          ul.appendChild(tli);
        });
        li.appendChild(ul);
      }
      listEl.appendChild(li);
    });
  }

  function buildFilter() {
    if (!filterEl) return;
    var all = [];
    posts.forEach(function (p) {
      (p.tags || []).forEach(function (t) { if (all.indexOf(t) === -1) all.push(t); });
    });
    if (!all.length) return;
    all.sort();

    function mkBtn(label, value) {
      var b = el('button', 'tag tag--filter', label);
      b.type = 'button';
      b.setAttribute('aria-pressed', String(activeTag === value));
      b.addEventListener('click', function () {
        activeTag = (activeTag === value) ? null : value;
        filterEl.querySelectorAll('button').forEach(function (x) {
          x.setAttribute('aria-pressed', String(x.dataset.tag === (activeTag || '')));
        });
        render();
      });
      b.dataset.tag = value || '';
      return b;
    }

    filterEl.appendChild(mkBtn(T.all, null));
    all.forEach(function (t) { filterEl.appendChild(mkBtn(t, t)); });
  }

  fetch(BASE + '/posts.json', { cache: 'no-cache' })
    .then(function (r) { if (!r.ok) throw new Error('http ' + r.status); return r.json(); })
    .then(function (data) {
      posts = (data.posts || [])
        .filter(function (p) { return !p.draft; })
        .sort(function (a, b) { return a.date < b.date ? 1 : -1; });  // ISO sorts lexically
      buildFilter();
      render();
    })
    .catch(function () { statusEl.hidden = false; statusEl.textContent = T.fail; });
})();
