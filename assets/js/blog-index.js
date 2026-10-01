/* Blog index dashboard for the current language, built from posts.json.
   Left: search, views (overview / all / unread), modules (= series)
   grouped by learning-path stage, topics. Right: one view at a time,
   driven by the URL (?module=id, ?view=all&tag=x&q=y, ?view=unread) so
   every state is linkable and the back button works.

   Reading state is per browser and written by blog-post.js:
   'readPosts' (finished articles) and 'lastOpened' (most recent article).
   'blogLastVisit' is kept here to surface what is new since last time. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';
  var NEW_DAYS = 14;
  var STANDALONE = '_standalone';
  var SITE_TITLE = document.title;

  var T = {
    en: {
      title: 'Engineering Notes',
      lead: 'In-depth technical writing on AI architecture, large language models and deep learning systems, organised as a learning path.',
      overview: 'Overview',
      all: 'All articles',
      unread: 'Unread',
      allLead: 'Every article, newest first. Filter by topic or search from the sidebar.',
      unreadLead: 'Articles you have not finished yet in this browser, in learning-path order.',
      standalone: 'Standalone',
      standaloneDesc: 'Self-contained deep dives that are not part of a module.',
      other: 'More',
      module: 'Module',
      path: 'Learning path',
      recent: 'Recently published',
      sinceVisit: 'New since your last visit',
      seeAll: 'See all',
      none: 'No posts published yet.',
      fail: 'Could not load posts.',
      empty: 'No articles match these filters.',
      unreadEmpty: 'You have finished every article. New ones will show up here.',
      clear: 'Clear filters',
      min: ' min',
      minRead: ' min read',
      parts: function (n) { return n + (n === 1 ? ' part' : ' parts'); },
      partsWord: function (n) { return n === 1 ? 'part' : 'parts'; },
      articlesWord: function (n) { return n === 1 ? 'article' : 'articles'; },
      count: function (n, total) { return n === total ? n + ' articles' : n + ' of ' + total + ' articles'; },
      metaLine: function (a, m, h) { return a + ' articles · ' + m + ' modules · about ' + h + ' hours of reading'; },
      readLine: function (r) { return r + ' finished'; },
      readOf: function (r, n) { return r + ' of ' + n + ' finished'; },
      latest: 'Latest',
      cont: 'Continue reading',
      startHere: 'Start here',
      upNext: 'Up next',
      isNew: 'New',
      visited: 'Read',
      done: 'Completed',
      partN: function (n) { return 'Part ' + n; },
      startWith: 'Start with Part 1',
      contPart: function (n) { return 'Continue with Part ' + n; },
      reread: 'Read again from Part 1',
      open: 'Open',
      about: 'About this module',
      nextModule: 'Next module',
      remove: 'Remove filter',
      decimal: function (x) { return String(x); }
    },
    tr: {
      title: 'Mühendislik Notları',
      lead: 'Yapay zeka mimarisi, büyük dil modelleri ve derin öğrenme sistemleri üzerine bir öğrenme yolu olarak düzenlenmiş derinlemesine teknik yazılar.',
      overview: 'Genel Bakış',
      all: 'Tüm Yazılar',
      unread: 'Okunmamışlar',
      allLead: 'Bütün yazılar, en yeniden eskiye. Soldaki menüden konuya göre süzün ya da arayın.',
      unreadLead: 'Bu tarayıcıda henüz bitirmediğiniz yazılar, öğrenme yolu sırasıyla.',
      standalone: 'Tekil Yazılar',
      standaloneDesc: 'Bir modüle bağlı olmayan, kendi başına okunabilen derinlemesine yazılar.',
      other: 'Diğer',
      module: 'Modül',
      path: 'Öğrenme yolu',
      recent: 'Son yayımlananlar',
      sinceVisit: 'Son ziyaretinizden beri yeni',
      seeAll: 'Tümünü gör',
      none: 'Henüz yazı yayımlanmadı.',
      fail: 'Yazılar yüklenemedi.',
      empty: 'Bu filtrelerle eşleşen yazı yok.',
      unreadEmpty: 'Bütün yazıları bitirdiniz. Yenileri burada görünecek.',
      clear: 'Filtreleri temizle',
      min: ' dk',
      minRead: ' dk okuma',
      parts: function (n) { return n + ' bölüm'; },
      partsWord: function () { return 'bölüm'; },
      articlesWord: function () { return 'yazı'; },
      count: function (n, total) { return n === total ? n + ' yazı' : total + ' yazıdan ' + n + ' tanesi'; },
      metaLine: function (a, m, h) { return a + ' yazı · ' + m + ' modül · yaklaşık ' + h + ' saatlik okuma'; },
      readLine: function (r) { return r + ' yazı bitirildi'; },
      readOf: function (r, n) { return n + ' yazının ' + r + ' tanesi bitirildi'; },
      latest: 'En yeni',
      cont: 'Kaldığınız yerden',
      startHere: 'Buradan başlayın',
      upNext: 'Sıradaki',
      isNew: 'Yeni',
      visited: 'Okundu',
      done: 'Tamamlandı',
      partN: function (n) { return 'Bölüm ' + n; },
      startWith: '1. bölümle başla',
      contPart: function (n) { return n + '. bölümle devam et'; },
      reread: 'Baştan yeniden oku',
      open: 'Aç',
      about: 'Bu modül hakkında',
      nextModule: 'Sonraki modül',
      remove: 'Filtreyi kaldır',
      decimal: function (x) { return String(x).replace('.', ','); }
    }
  }[LANG];

  var $ = function (id) { return document.getElementById(id); };
  var headEl = $('dash-head');
  var viewEl = $('dash-view');
  var mainEl = $('dash-main');
  var statusEl = $('list-status');
  var searchEl = $('post-search');
  var viewsEl = $('side-views');
  var modulesEl = $('side-modules');
  var tagsEl = $('side-tags');

  var posts = [];       // newest first
  var bySlug = {};
  var stages = [];      // [{ id, title, tagline, modules: [module] }]
  var modules = [];     // path order: [{ id, title, tagline, summary, badge, pinned, num, stage, posts }]
  var moduleMap = {};

  var read = [];
  try { read = JSON.parse(localStorage.getItem('readPosts') || '[]'); } catch (e) { read = []; }
  if (!Array.isArray(read)) read = [];
  var lastOpened = '';
  try { lastOpened = localStorage.getItem('lastOpened') || ''; } catch (e) {}
  var prevVisit = lastVisit();

  var state = { view: 'overview', module: '', tag: '', q: '' };

  /* The previous visit's date, pinned for this tab session so reloading
     does not wipe the "new since" list. */
  function lastVisit() {
    var d = new Date();
    var today = d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
    try {
      var prev = sessionStorage.getItem('blogPrevVisit');
      if (prev === null) {
        prev = localStorage.getItem('blogLastVisit') || '';
        sessionStorage.setItem('blogPrevVisit', prev);
        localStorage.setItem('blogLastVisit', today);
      }
      return prev;
    } catch (e) { return ''; }
  }

  /* ---------- helpers ---------- */

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function svg(d, size) {
    var NS = 'http://www.w3.org/2000/svg';
    var s = document.createElementNS(NS, 'svg');
    s.setAttribute('width', size || 14); s.setAttribute('height', size || 14);
    s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('fill', 'none');
    s.setAttribute('stroke', 'currentColor'); s.setAttribute('stroke-width', '2');
    s.setAttribute('stroke-linecap', 'round'); s.setAttribute('aria-hidden', 'true');
    var p = document.createElementNS(NS, 'path');
    p.setAttribute('d', d);
    s.appendChild(p);
    return s;
  }

  function href(p) { return BASE + '/post.html?slug=' + encodeURIComponent(p.slug); }

  function fmtDate(iso) {
    var d = new Date(iso + 'T00:00:00');
    return d.toLocaleDateString(LANG === 'tr' ? 'tr-TR' : 'en-GB',
      { month: 'short', day: 'numeric', year: 'numeric' });
  }

  function timeEl(iso) {
    var t = el('time', null, fmtDate(iso));
    t.setAttribute('datetime', iso);
    return t;
  }

  function isRead(p) { return read.indexOf(p.slug) !== -1; }
  function returning() { return !!(read.length || lastOpened || prevVisit); }

  /* Returning readers: published after their last visit and not finished.
     First-time readers: the newest date, if it is recent. */
  function isNew(p) {
    if (prevVisit) return p.date > prevVisit && !isRead(p);
    if (!posts.length || p.date !== posts[0].date) return false;
    var age = (Date.now() - new Date(p.date + 'T00:00:00').getTime()) / 864e5;
    return age >= 0 && age <= NEW_DAYS;
  }

  function shortTitle(p) {
    return p.seriesId ? p.title.replace(/^.*?\(\d+\):\s*/, '') : p.title;
  }

  /* One-line description: the curated tagline, else the "What:" clause of
     the 5W1H summary. */
  function blurb(p) {
    if (p.tagline) return p.tagline;
    var s = p.summary || '';
    var m = s.match(/^(?:What|Ne):\s*([\s\S]*?)(?:\s+(?:Why|Neden önemli|Neden):|$)/);
    return m ? m[1] : s;
  }

  function tagLabel(t) { return t.replace(/-/g, ' '); }
  function lower(s) { return String(s).toLocaleLowerCase(LANG === 'tr' ? 'tr-TR' : 'en-US'); }
  function pad(n) { return (n < 10 ? '0' : '') + n; }
  function minutes(list) { return list.reduce(function (t, p) { return t + (p.readingMinutes || 0); }, 0); }
  function readCount(list) { return list.filter(isRead).length; }
  function nextUnread(m) { return m.posts.filter(function (p) { return !isRead(p); })[0]; }
  function moduleOf(p) { return moduleMap[p.seriesId] || moduleMap[STANDALONE]; }
  function modLabel(m) { return m.num ? T.module + ' ' + pad(m.num) : T.standalone; }

  function progressBar(cls, done, total, label) {
    var bar = el('div', cls);
    bar.setAttribute('role', 'progressbar');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', String(total));
    bar.setAttribute('aria-valuenow', String(done));
    if (label) bar.setAttribute('aria-label', label);
    var fill = el('span');
    fill.style.width = (total ? 100 * done / total : 0) + '%';
    bar.appendChild(fill);
    return bar;
  }

  function seriesChip(p) {
    var m = moduleMap[p.seriesId];
    if (!m) return null;
    var chip = el('span', 'series-chip');
    chip.appendChild(el('span', 'series-chip__name', m.title));
    if (p.seriesPart) chip.appendChild(el('span', 'series-chip__part', String(p.seriesPart)));
    return chip;
  }

  /* ---------- URL <-> state ---------- */

  function urlFor(st) {
    var q = new URLSearchParams();
    if (st.view === 'module' && st.module) q.set('module', st.module);
    else if (st.view === 'all' || st.view === 'unread') q.set('view', st.view);
    if (st.view === 'all') {
      if (st.tag) q.set('tag', st.tag);
      if (st.q) q.set('q', st.q);
    }
    var s = q.toString();
    return location.pathname + (s ? '?' + s : '');
  }

  function readUrl() {
    var q = new URLSearchParams(location.search);
    var mod = q.get('module') || q.get('track') || '';
    var view = q.get('view') || '';
    if (view === 'queue') view = 'unread';
    var st = { view: 'overview', module: '', tag: q.get('tag') || '', q: q.get('q') || '' };
    if (mod && moduleMap[mod]) { st.view = 'module'; st.module = mod; }
    else if (view === 'all' || view === 'unread') st.view = view;
    else if (st.tag || st.q) st.view = 'all';
    if (st.view !== 'all') { st.tag = ''; st.q = ''; }
    return st;
  }

  /* Internal links carry data-nav; clicking them swaps the view in place. */
  function navLink(cls, st, text) {
    var a = el('a', cls, text);
    a.href = urlFor(st);
    a.dataset.nav = JSON.stringify(st);
    return a;
  }

  function go(patch, opts) {
    opts = opts || {};
    var st = { view: state.view, module: state.module, tag: state.tag, q: state.q };
    Object.keys(patch).forEach(function (k) { st[k] = patch[k]; });
    if (st.view !== 'module') st.module = '';
    if (st.view !== 'all') { st.tag = ''; st.q = ''; }
    state = st;
    var url = urlFor(state);
    if (url !== location.pathname + location.search) {
      history[opts.replace ? 'replaceState' : 'pushState'](null, '', url);
    }
    render();
    if (opts.focus) {
      window.scrollTo({ top: 0 });
      mainEl.focus({ preventScroll: true });
    }
  }

  document.addEventListener('click', function (e) {
    var a = e.target.closest && e.target.closest('a[data-nav]');
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    go(JSON.parse(a.dataset.nav), { focus: true });
  });

  window.addEventListener('popstate', function () {
    state = readUrl();
    render();
  });

  /* ---------- sidebar ---------- */

  function buildSidebar() {
    viewsEl.querySelectorAll('a[data-view]').forEach(function (a) {
      var st = { view: a.dataset.view, module: '', tag: '', q: '' };
      a.href = urlFor(st);
      a.dataset.nav = JSON.stringify(st);
    });

    modulesEl.replaceChildren();
    stages.forEach(function (s) {
      var head = el('li', 'side-stage', s.title);
      head.setAttribute('aria-hidden', 'true');
      modulesEl.appendChild(head);
      s.modules.forEach(function (m) {
        var a = navLink('side-link side-module', { view: 'module', module: m.id });
        a.dataset.module = m.id;
        var icon = el('span', 'side-link__icon', m.num ? pad(m.num) : '·');
        icon.setAttribute('aria-hidden', 'true');
        var r = readCount(m.posts);
        a.append(icon, el('span', 'side-link__text', m.title),
          el('span', 'side-link__count', r ? r + '/' + m.posts.length : String(m.posts.length)));
        a.appendChild(progressBar('side-module__bar', r, m.posts.length, T.readOf(r, m.posts.length)));
        if (r === m.posts.length) a.classList.add('is-done');
        var li = el('li');
        li.appendChild(a);
        modulesEl.appendChild(li);
      });
    });

    var counts = {};
    posts.forEach(function (p) {
      (p.tags || []).forEach(function (t) { counts[t] = (counts[t] || 0) + 1; });
    });
    var tags = Object.keys(counts).filter(function (t) {
      return counts[t] >= 2 && counts[t] < posts.length;
    }).sort(function (a, b) { return counts[b] - counts[a] || a.localeCompare(b); }).slice(0, 16);

    tagsEl.replaceChildren();
    tags.forEach(function (t) {
      var b = el('button', 'tag tag--filter', tagLabel(t));
      b.type = 'button';
      b.dataset.tag = t;
      b.appendChild(el('span', 'tag__count', String(counts[t])));
      b.addEventListener('click', function () {
        var on = state.view === 'all' && state.tag === t;
        go({ view: 'all', tag: on ? '' : t, q: state.view === 'all' ? state.q : '' },
           { focus: !on && state.view !== 'all' });
      });
      var li = el('li');
      li.appendChild(b);
      tagsEl.appendChild(li);
    });

    $('count-all').textContent = String(posts.length);
  }

  function syncSidebar() {
    viewsEl.querySelectorAll('a[data-view]').forEach(function (a) {
      if (a.dataset.view === state.view) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    modulesEl.querySelectorAll('a[data-module]').forEach(function (a) {
      if (state.view === 'module' && a.dataset.module === state.module) a.setAttribute('aria-current', 'page');
      else a.removeAttribute('aria-current');
    });
    tagsEl.querySelectorAll('button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(state.view === 'all' && b.dataset.tag === state.tag));
    });
    $('count-unread').textContent = String(posts.length - readCount(posts));
    if (document.activeElement !== searchEl && searchEl.value !== state.q) searchEl.value = state.q;
  }

  /* ---------- shared pieces ---------- */

  function setHead(opts) {
    headEl.replaceChildren();
    if (opts.crumb) {
      var c = el('p', 'dash-crumb');
      c.appendChild(navLink(null, { view: 'overview' }, T.overview));
      opts.crumb.forEach(function (x) { c.appendChild(el('span', null, x)); });
      headEl.appendChild(c);
    }
    headEl.appendChild(el('h1', 'dash-title', opts.title));
    if (opts.lead) headEl.appendChild(el('p', 'dash-lead', opts.lead));
    if (opts.meta) headEl.appendChild(el('p', 'dash-meta', opts.meta));
    document.title = opts.docTitle ? opts.docTitle + ' — ' + SITE_TITLE : SITE_TITLE;
  }

  function panel(title, action) {
    var s = el('section', 'panel');
    var h = el('h2', 'panel-title');
    h.appendChild(el('span', null, title));
    if (action) h.appendChild(action);
    s.appendChild(h);
    return s;
  }

  function postRow(p, withBlurb) {
    var a = el('a', 'post-row');
    a.href = href(p);
    if (isRead(p)) a.classList.add('is-read');

    var body = el('span', 'post-row__body');
    var top = el('span', 'post-row__top');
    var chip = seriesChip(p);
    if (chip) top.appendChild(chip);
    if (isNew(p)) top.appendChild(el('span', 'pill pill--new', T.isNew));
    if (top.childNodes.length) body.appendChild(top);
    body.appendChild(el('span', 'post-row__title', shortTitle(p)));
    if (withBlurb) body.appendChild(el('span', 'post-row__teaser', blurb(p)));
    a.appendChild(body);

    var meta = el('span', 'post-row__meta');
    meta.appendChild(timeEl(p.date));
    if (p.readingMinutes) meta.appendChild(el('span', null, p.readingMinutes + T.minRead));
    if (isRead(p)) meta.appendChild(el('span', 'post-row__check', '✓ ' + T.visited));
    a.appendChild(meta);

    var li = el('li');
    li.appendChild(a);
    return li;
  }

  function rows(list, withBlurb) {
    var ol = el('ol', 'post-rows');
    list.forEach(function (p) { ol.appendChild(postRow(p, withBlurb)); });
    return ol;
  }

  function spot(p, label, accent) {
    var card = el('article', 'spot' + (accent ? ' spot--accent' : ''));
    var eb = el('p', 'spot__eyebrow');
    eb.appendChild(el('span', accent ? 'pill pill--accent' : 'pill', label));
    var m = moduleMap[p.seriesId];
    if (m && p.seriesPart) eb.appendChild(el('span', null, m.title + ' · ' + T.partN(p.seriesPart)));
    card.appendChild(eb);
    var h = el('h3', 'spot__title');
    var a = el('a', null, shortTitle(p));
    a.href = href(p);
    h.appendChild(a);
    card.appendChild(h);
    card.appendChild(el('p', 'spot__text', blurb(p)));
    var meta = el('p', 'spot__meta');
    meta.appendChild(timeEl(p.date));
    if (p.readingMinutes) meta.appendChild(el('span', null, p.readingMinutes + T.minRead));
    meta.appendChild(el('span', 'spot__go', '→'));
    card.appendChild(meta);
    return card;
  }

  /* What a returning reader should open next:
     1. the article they last opened but did not finish;
     2. the next part of the module holding their latest finished article;
     3. any module they are partway through;
     4. the first unfinished module on the path. */
  function resumeTarget() {
    var lo = bySlug[lastOpened];
    if (lo && !isRead(lo)) return { post: lo, label: T.cont };
    for (var i = read.length - 1; i >= 0; i--) {
      var p = bySlug[read[i]];
      var m = p && moduleOf(p);
      if (m && m.num) {
        var n = nextUnread(m);
        if (n) return { post: n, label: T.cont };
      }
    }
    var partial = modules.filter(function (x) {
      var r = readCount(x.posts);
      return x.num && r && r < x.posts.length;
    })[0];
    if (partial) return { post: nextUnread(partial), label: T.cont };
    var fresh = modules.filter(function (x) { return x.num && nextUnread(x); })[0];
    return fresh ? { post: nextUnread(fresh), label: T.upNext } : null;
  }

  function startTarget() {
    var first = modules.filter(function (x) { return x.pinned; })[0] || modules[0];
    var p = first && (nextUnread(first) || first.posts[0]);
    return p ? { post: p, label: T.startHere } : null;
  }

  /* ---------- views ---------- */

  function moduleNode(m) {
    var rc = readCount(m.posts);
    var card = el('article', 'module-card' + (m.pinned ? ' is-pinned' : ''));
    if (rc === m.posts.length) card.classList.add('is-done');
    var top = el('div', 'module-card__top');
    top.appendChild(el('span', 'module-card__num', modLabel(m)));
    if (rc === m.posts.length) top.appendChild(el('span', 'pill pill--done', '✓ ' + T.done));
    else if (rc) top.appendChild(el('span', 'pill pill--new', rc + '/' + m.posts.length));
    else if (m.pinned && m.badge) top.appendChild(el('span', 'pill pill--accent', m.badge));
    card.appendChild(top);
    var h = el('h3', 'module-card__title');
    h.appendChild(navLink(null, { view: 'module', module: m.id }, m.title));
    card.appendChild(h);
    if (m.tagline) card.appendChild(el('p', 'module-card__desc', m.tagline));
    var foot = el('div', 'module-card__foot');
    var meta = el('div', 'module-card__meta');
    meta.appendChild(el('span', null, T.parts(m.posts.length) + ' · ' + minutes(m.posts) + T.min));
    foot.appendChild(meta);
    foot.appendChild(progressBar('progress', rc, m.posts.length, T.readOf(rc, m.posts.length)));
    card.appendChild(foot);
    return card;
  }

  function pathMap() {
    var p = panel(T.path);
    var ol = el('ol', 'path');
    stages.forEach(function (s, i) {
      var mods = s.modules.filter(function (m) { return m.num; });
      if (!mods.length) return;
      var all = [];
      mods.forEach(function (m) { all = all.concat(m.posts); });
      var li = el('li', 'path-stage');
      var rc = readCount(all);
      if (rc === all.length) li.classList.add('is-done');
      else if (rc) li.classList.add('is-active');
      var dot = el('span', 'path-stage__dot', rc === all.length ? '✓' : String(i + 1));
      dot.setAttribute('aria-hidden', 'true');
      li.appendChild(dot);
      var body = el('div', 'path-stage__body');
      var head = el('h3', 'path-stage__head');
      head.appendChild(el('span', 'path-stage__title', s.title));
      if (s.tagline) head.appendChild(el('span', 'path-stage__tagline', s.tagline));
      body.appendChild(head);
      var ul = el('ul', 'module-grid' + (mods.length === 1 ? ' module-grid--single' : ''));
      mods.forEach(function (m) {
        var mi = el('li');
        mi.appendChild(moduleNode(m));
        ul.appendChild(mi);
      });
      body.appendChild(ul);
      li.appendChild(body);
      ol.appendChild(li);
    });
    p.appendChild(ol);
    return p;
  }

  function renderOverview() {
    var total = minutes(posts);
    var r = readCount(posts);
    var hours = T.decimal(Math.round(total / 6) / 10);
    var meta = T.metaLine(posts.length, modules.filter(function (m) { return m.num; }).length, hours);
    if (r) meta += ' · ' + T.readLine(r);
    setHead({ title: T.title, lead: T.lead, meta: meta });

    var back = returning();
    var first = back ? resumeTarget() : startTarget();
    var latest = posts[0];
    var row = el('div', 'spot-row');
    if (first) row.appendChild(spot(first.post, first.label, true));
    var second = first && first.post === latest ? posts[1] : latest;
    if (second) row.appendChild(spot(second, T.latest, false));
    viewEl.appendChild(row);

    // Articles already featured in the cards above are not listed twice.
    var fresh = prevVisit ? posts.filter(function (p) {
      return isNew(p) && p !== second && !(first && p === first.post);
    }) : [];
    if (fresh.length) {
      var np = panel(T.sinceVisit + ' (' + fresh.length + ')');
      np.appendChild(rows(fresh, true));
      viewEl.appendChild(np);
    }

    viewEl.appendChild(pathMap());

    if (!fresh.length) {
      var rp = panel(T.recent, navLink(null, { view: 'all' }, T.seeAll + ' →'));
      rp.appendChild(rows(posts.slice(0, 5), true));
      viewEl.appendChild(rp);
    }

    var loose = moduleMap[STANDALONE];
    if (loose) {
      var lp = panel(T.standalone, navLink(null, { view: 'module', module: STANDALONE }, T.open + ' →'));
      lp.appendChild(rows(loose.posts, true));
      viewEl.appendChild(lp);
    }
  }

  function renderModule(m) {
    var rc = readCount(m.posts);
    var next = nextUnread(m);
    var crumb = [];
    if (m.stage) crumb.push(m.stage.title);
    crumb.push(modLabel(m));
    setHead({ crumb: crumb, title: m.title, lead: m.tagline || m.summary, docTitle: m.title });

    if (m.summary && m.tagline) {
      var about = el('details', 'module-about');
      about.appendChild(el('summary', null, T.about));
      about.appendChild(el('p', null, m.summary));
      viewEl.appendChild(about);
    }

    var hero = el('div', 'module-hero');
    var st = el('div', 'module-hero__stats');
    function stat(v, l) {
      var s = el('div', 'module-hero__stat');
      s.append(el('b', null, v), el('span', null, l));
      st.appendChild(s);
    }
    stat(String(m.posts.length), m.num ? T.partsWord(m.posts.length) : T.articlesWord(m.posts.length));
    stat(String(minutes(m.posts)), T.min.trim());
    stat(rc + '/' + m.posts.length, T.visited.toLocaleLowerCase(LANG));
    hero.appendChild(st);

    var prog = el('div', 'module-hero__progress');
    prog.appendChild(progressBar('progress', rc, m.posts.length, T.readOf(rc, m.posts.length)));
    prog.appendChild(el('span', null, rc === m.posts.length ? '✓ ' + T.done : T.readOf(rc, m.posts.length)));
    hero.appendChild(prog);

    if (m.num) {
      var actions = el('div', 'module-hero__actions');
      var cta = el('a', 'btn btn--primary',
        !rc ? T.startWith : next ? T.contPart(m.posts.indexOf(next) + 1) : T.reread);
      cta.href = href(next || m.posts[0]);
      cta.appendChild(svg('M5 12h14M13 6l6 6-6 6'));
      actions.appendChild(cta);
      hero.appendChild(actions);
    }
    viewEl.appendChild(hero);

    var steps = el('ol', 'steps');
    m.posts.forEach(function (p, i) {
      var li = el('li', 'step');
      var isNext = m.num && p === next;
      if (isRead(p)) li.classList.add('is-read');
      if (isNext) li.classList.add('is-next');
      var dot = el('span', 'step__dot', isRead(p) ? '✓' : pad(i + 1));
      dot.setAttribute('aria-hidden', 'true');
      li.appendChild(dot);

      var card = el('article', 'step__card');
      var meta = el('p', 'step__meta');
      if (m.num) meta.appendChild(el('span', null, T.partN(i + 1)));
      if (p.readingMinutes) meta.appendChild(el('span', null, p.readingMinutes + T.minRead));
      meta.appendChild(timeEl(p.date));
      if (isRead(p)) meta.appendChild(el('span', 'pill pill--done', '✓ ' + T.visited));
      else if (isNext && rc) meta.appendChild(el('span', 'pill pill--accent', T.upNext));
      else if (isNew(p)) meta.appendChild(el('span', 'pill pill--new', T.isNew));
      card.appendChild(meta);

      var h = el('h2', 'step__title');
      var a = el('a', null, shortTitle(p));
      a.href = href(p);
      h.appendChild(a);
      card.appendChild(h);
      card.appendChild(el('p', 'step__teaser', blurb(p)));
      li.appendChild(card);
      steps.appendChild(li);
    });
    viewEl.appendChild(steps);

    var after = modules[modules.indexOf(m) + 1];
    if (m.num && after && after.num) {
      var nx = navLink('module-next', { view: 'module', module: after.id });
      nx.appendChild(el('span', 'module-next__label', T.nextModule + ' · ' + modLabel(after)));
      nx.appendChild(el('span', 'module-next__title', after.title + ' →'));
      if (after.tagline) nx.appendChild(el('span', 'module-next__text', after.tagline));
      viewEl.appendChild(nx);
    }
  }

  function matches(p) {
    if (state.tag && (p.tags || []).indexOf(state.tag) === -1) return false;
    if (!state.q) return true;
    var m = moduleMap[p.seriesId];
    var hay = lower([p.title, p.tagline || '', p.summary, (p.tags || []).map(tagLabel).join(' '),
                     m ? m.title : ''].join(' '));
    return lower(state.q).split(/\s+/).every(function (w) { return hay.indexOf(w) !== -1; });
  }

  function emptyBox(text, withClear) {
    var box = el('div', 'empty');
    box.appendChild(el('p', null, text));
    if (withClear) {
      var b = el('button', 'btn btn--ghost', T.clear);
      b.type = 'button';
      b.addEventListener('click', function () { go({ view: 'all', tag: '', q: '' }, { replace: true }); });
      box.appendChild(b);
    }
    return box;
  }

  function renderAll() {
    setHead({ title: T.all, lead: T.allLead, docTitle: T.all });
    var list = posts.filter(matches);

    var bar = el('div', 'list-toolbar');
    var count = el('p', 'list-toolbar__count', T.count(list.length, posts.length));
    count.setAttribute('role', 'status');
    count.setAttribute('aria-live', 'polite');
    bar.appendChild(count);
    function chip(label, patch) {
      var b = el('button', 'filter-chip', label);
      b.type = 'button';
      b.setAttribute('aria-label', T.remove + ': ' + label);
      b.appendChild(svg('M18 6 6 18M6 6l12 12', 12));
      b.addEventListener('click', function () { go(patch, { replace: true }); });
      bar.appendChild(b);
    }
    if (state.tag) chip('#' + tagLabel(state.tag), { tag: '' });
    if (state.q) chip('“' + state.q + '”', { q: '' });

    var sec = el('section');
    sec.append(bar, list.length ? rows(list, true) : emptyBox(T.empty, true));
    viewEl.appendChild(sec);
  }

  function renderUnread() {
    setHead({ title: T.unread, lead: T.unreadLead, docTitle: T.unread });
    var any = false;
    modules.forEach(function (m) {
      var list = m.posts.filter(function (p) { return !isRead(p); });
      if (!list.length) return;
      any = true;
      var p = panel((m.num ? pad(m.num) + ' · ' : '') + m.title,
        navLink(null, { view: 'module', module: m.id }, T.open + ' →'));
      p.appendChild(rows(list, false));
      viewEl.appendChild(p);
    });
    if (!any) viewEl.appendChild(emptyBox(T.unreadEmpty, false));
  }

  function render() {
    viewEl.replaceChildren();
    if (state.view === 'module' && moduleMap[state.module]) renderModule(moduleMap[state.module]);
    else if (state.view === 'all') renderAll();
    else if (state.view === 'unread') renderUnread();
    else { state.view = 'overview'; renderOverview(); }
    syncSidebar();
  }

  /* ---------- search ---------- */

  var searchTimer;
  searchEl.addEventListener('input', function () {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () {
      var q = searchEl.value.trim();
      if (q === state.q) return;
      var entering = state.view !== 'all';
      if (!q && entering) return;
      go({ view: 'all', q: q, tag: entering ? '' : state.tag }, { replace: !entering });
    }, 150);
  });
  searchEl.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && searchEl.value) {
      e.preventDefault();
      searchEl.value = '';
      if (state.view === 'all') go({ q: '' }, { replace: true });
    }
  });
  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    e.preventDefault();
    searchEl.focus();
  });

  /* ---------- data ---------- */

  function build(data) {
    posts = (data.posts || []).filter(function (p) { return !p.draft; });
    posts.sort(function (a, b) {
      if (a.date !== b.date) return a.date < b.date ? 1 : -1;
      return (b.seriesPart || 0) - (a.seriesPart || 0);
    });
    posts.forEach(function (p) { bySlug[p.slug] = p; });

    // Stages in posts.json order; series without a known stage go last.
    var stageMap = {};
    stages = (data.stages || []).map(function (s) {
      var o = { id: s.id, title: s.title, tagline: s.tagline, modules: [] };
      stageMap[s.id] = o;
      return o;
    });
    var otherStage = { id: '_other', title: T.other, tagline: '', modules: [] };

    var known = {};
    (data.series || []).forEach(function (s) {
      known[s.id] = true;
      var list = (s.posts || []).map(function (slug) { return bySlug[slug]; }).filter(Boolean);
      if (!list.length) return;
      var st = stageMap[s.stage] || otherStage;
      st.modules.push({ id: s.id, title: s.title, tagline: s.tagline, summary: s.summary,
                        badge: s.badge, pinned: !!s.pinned, stage: st, posts: list });
    });
    var loose = posts.filter(function (p) { return !p.seriesId || !known[p.seriesId]; });
    if (loose.length) {
      otherStage.modules.push({ id: STANDALONE, title: T.standalone, tagline: T.standaloneDesc,
                                summary: '', stage: null, num: 0, posts: loose });
    }
    if (otherStage.modules.length) stages.push(otherStage);
    stages = stages.filter(function (s) { return s.modules.length; });

    modules = [];
    var n = 0;
    stages.forEach(function (s) {
      s.modules.forEach(function (m) {
        if (m.id !== STANDALONE) m.num = ++n;
        modules.push(m);
      });
    });
    moduleMap = {};
    modules.forEach(function (m) { moduleMap[m.id] = m; });
  }

  fetch(BASE + '/posts.json', { cache: 'no-cache' })
    .then(function (r) {
      if (!r.ok) throw new Error('http ' + r.status);
      return r.json();
    })
    .then(function (data) {
      build(data);
      if (!posts.length) { statusEl.textContent = T.none; return; }
      statusEl.hidden = true;
      buildSidebar();
      state = readUrl();
      history.replaceState(null, '', urlFor(state) + location.hash);
      render();
    })
    .catch(function (err) {
      // Show the pre-rendered list baked into the page instead.
      document.documentElement.classList.remove('js');
      statusEl.hidden = false;
      statusEl.textContent = T.fail;
      if (window.console) console.error(err);
    });
})();
