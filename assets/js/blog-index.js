/* Blog index for the current language, built from posts.json in one request:
   a lead story (newest post), one full-width band per series whose parts are
   all visible as a card grid (built to hold 10+ parts), and a searchable,
   filterable card grid of every article whose state lives in the URL. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';
  var NEW_DAYS = 14;

  var T = {
    en: {
      empty: 'No articles match. Try another topic or clear the search.',
      none: 'No posts published yet.',
      fail: 'Could not load posts.',
      min: ' min',
      minRead: ' min read',
      all: 'All topics',
      latest: 'Latest',
      partOf: function (n, s) { return 'Part ' + n + ' of ' + s; },
      read: 'Read article',
      startSeries: 'Start from Part 1',
      parts: function (n) { return n + (n === 1 ? ' part' : ' parts'); },
      progress: function (r, n) { return r + ' of ' + n + ' read'; },
      cont: 'Continue',
      done: 'Completed',
      isNew: 'New',
      visited: 'Read',
      pinned: 'Pinned',
      stats: function (a, s, m) { return a + ' articles · ' + s + ' series · ' + m + ' min of reading'; },
      count: function (n, total) { return n === total ? n + ' articles' : n + ' of ' + total + ' articles'; },
      clear: 'Clear filters',
      startWith: 'Start with Part 1',
      contPart: function (n) { return 'Continue with Part ' + n; },
      reread: 'Read again from Part 1'
    },
    tr: {
      empty: 'Eşleşen yazı yok. Başka bir konu deneyin ya da aramayı temizleyin.',
      none: 'Henüz yazı yayımlanmadı.',
      fail: 'Yazılar yüklenemedi.',
      min: ' dk',
      minRead: ' dk okuma',
      all: 'Tüm konular',
      latest: 'En yeni',
      partOf: function (n, s) { return s + ' · Bölüm ' + n; },
      read: 'Yazıyı oku',
      startSeries: '1. bölümden başla',
      parts: function (n) { return n + ' bölüm'; },
      progress: function (r, n) { return n + ' bölümün ' + r + ' tanesi okundu'; },
      cont: 'Devam et',
      done: 'Tamamlandı',
      isNew: 'Yeni',
      visited: 'Okundu',
      pinned: 'Sabit',
      stats: function (a, s, m) { return a + ' yazı · ' + s + ' seri · ' + m + ' dk okuma'; },
      count: function (n, total) { return n === total ? n + ' yazı' : total + ' yazıdan ' + n + ' tanesi'; },
      clear: 'Filtreleri temizle',
      startWith: '1. bölümle başla',
      contPart: function (n) { return n + '. bölümle devam et'; },
      reread: 'Baştan yeniden oku'
    }
  }[LANG];

  var $ = function (id) { return document.getElementById(id); };
  var leadEl = $('lead-story');
  var seriesSectionEl = $('series-section');
  var seriesContainerEl = $('series-container');
  var listEl = $('post-list');
  var statusEl = $('list-status');
  var filterEl = $('tag-filter');
  var searchEl = $('post-search');
  var countEl = $('result-count');
  var statsEl = $('blog-stats');

  var posts = [];
  var seriesList = [];
  var seriesMap = {};
  var read = [];
  var state = { tag: '', q: '' };

  try { read = JSON.parse(localStorage.getItem('readPosts') || '[]'); } catch (e) { read = []; }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;   // textContent: never innerHTML for data
    return n;
  }

  function href(p) { return BASE + '/post.html?slug=' + encodeURIComponent(p.slug); }

  function fmtDate(iso, short) {
    var d = new Date(iso + 'T00:00:00');
    return d.toLocaleDateString(LANG === 'tr' ? 'tr-TR' : 'en-GB',
      short ? { month: 'short', day: 'numeric', year: 'numeric' }
            : { year: 'numeric', month: 'long', day: 'numeric' });
  }

  /* Only the most recent publishing day counts as new, and only for two weeks;
     a badge on half the list would carry no signal. */
  function isNew(p) {
    if (p.date !== posts[0].date) return false;
    var age = (Date.now() - new Date(p.date + 'T00:00:00').getTime()) / 864e5;
    return age >= 0 && age <= NEW_DAYS;
  }

  function isRead(p) { return read.indexOf(p.slug) !== -1; }

  /* "A Prompt's Journey (5): Multi-Head Attention — …" -> "Multi-Head Attention — …".
     The series name is shown separately wherever the short title is used. */
  function shortTitle(p) {
    return p.seriesId ? p.title.replace(/^.*?\(\d+\):\s*/, '') : p.title;
  }

  /* 5N1K summaries ("What: … Why: … How: …") are written for meta descriptions
     and run past 1,000 characters. Cards show only the "What" clause. */
  function teaser(p) {
    var s = p.summary || '';
    var m = s.match(/^(?:What|Ne):\s*([\s\S]*?)(?:\s+(?:Why|Neden önemli|Neden):|$)/);
    return m ? m[1] : s;
  }

  function lower(s) { return String(s).toLocaleLowerCase(LANG === 'tr' ? 'tr-TR' : 'en-US'); }

  function seriesChip(p) {
    var s = seriesMap[p.seriesId];
    if (!s) return null;
    var chip = el('span', 'series-chip');
    chip.appendChild(el('span', 'series-chip__name', s.title));
    if (p.seriesPart) chip.appendChild(el('span', 'series-chip__part', String(p.seriesPart)));
    return chip;
  }

  function renderLead() {
    if (!leadEl || !posts.length) return;
    var p = posts[0];
    leadEl.replaceChildren();

    var eyebrow = el('p', 'lead-story__eyebrow');
    eyebrow.appendChild(el('span', 'pill pill--accent', T.latest));
    var s = seriesMap[p.seriesId];
    if (s && p.seriesPart) eyebrow.appendChild(el('span', null, T.partOf(p.seriesPart, s.title)));
    leadEl.appendChild(eyebrow);

    var h = el('h2', 'lead-story__title');
    var a = el('a', null, shortTitle(p));
    a.href = href(p);
    h.appendChild(a);
    leadEl.appendChild(h);

    leadEl.appendChild(el('p', 'lead-story__summary', teaser(p)));

    var meta = el('p', 'lead-story__meta');
    var time = el('time', null, fmtDate(p.date));
    time.setAttribute('datetime', p.date);
    meta.appendChild(time);
    if (p.readingMinutes) meta.appendChild(el('span', null, p.readingMinutes + T.minRead));
    leadEl.appendChild(meta);

    var actions = el('div', 'lead-story__actions');
    var go = el('a', 'btn btn--primary', T.read + ' →');
    go.href = href(p);
    actions.appendChild(go);
    if (s && p.seriesPart > 1) {
      var first = posts.filter(function (x) { return x.slug === s.posts[0]; })[0];
      if (first) {
        var start = el('a', 'btn btn--ghost', T.startSeries);
        start.href = href(first);
        actions.appendChild(start);
      }
    }
    leadEl.appendChild(actions);
    leadEl.hidden = false;
  }

  function renderSeries() {
    seriesContainerEl.replaceChildren();
    var shown = 0;

    seriesList.forEach(function (s) {
      var sPosts = (s.posts || []).map(function (slug) {
        return posts.filter(function (p) { return p.slug === slug; })[0];
      }).filter(Boolean);
      if (!sPosts.length) return;

      var minutes = sPosts.reduce(function (t, p) { return t + (p.readingMinutes || 0); }, 0);
      var readCount = sPosts.filter(isRead).length;
      var next = sPosts.filter(function (p) { return !isRead(p); })[0];

      var band = el('article', sPosts.length >= 4 ? 'series series--long' : 'series');
      if (s.pinned) band.classList.add('series--pinned');
      band.setAttribute('aria-labelledby', 'series-' + s.id);

      var head = el('header', 'series__head');
      var intro = el('div', 'series__intro');
      var meta = el('p', 'series__meta');
      // Only pinned series show their badge; the others just restate the part count.
      if (s.pinned && s.badge) meta.appendChild(el('span', 'pill pill--accent', s.badge));
      meta.appendChild(el('span', null, T.parts(sPosts.length)));
      if (minutes) meta.appendChild(el('span', null, minutes + T.min));
      intro.appendChild(meta);
      var h = el('h3', 'series__title', s.title);
      h.id = 'series-' + s.id;
      intro.appendChild(h);
      if (s.summary) intro.appendChild(el('p', 'series__desc', s.summary));
      head.appendChild(intro);

      var aside = el('div', 'series__aside');
      if (readCount) {
        var prog = el('div', 'series-progress');
        var bar = el('div', 'series-progress__bar');
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('aria-valuemin', '0');
        bar.setAttribute('aria-valuemax', String(sPosts.length));
        bar.setAttribute('aria-valuenow', String(readCount));
        bar.setAttribute('aria-label', T.progress(readCount, sPosts.length));
        var fill = el('span', 'series-progress__fill');
        fill.style.width = (100 * readCount / sPosts.length) + '%';
        bar.appendChild(fill);
        prog.append(bar, el('span', 'series-progress__label',
          next ? T.progress(readCount, sPosts.length) : T.done));
        aside.appendChild(prog);
      }
      var target = next || sPosts[0];
      var cta = el('a', readCount && next ? 'btn btn--primary' : 'btn btn--ghost',
        !readCount ? T.startWith : next ? T.contPart(sPosts.indexOf(next) + 1) : T.reread);
      cta.href = href(target);
      aside.appendChild(cta);
      head.appendChild(aside);
      band.appendChild(head);

      // Every part is visible up front; the grid wraps, so 10 parts is two rows.
      var parts = el('ol', 'series__parts');
      sPosts.forEach(function (p, i) {
        var li = el('li');
        var a = el('a', 'part-card');
        a.href = href(p);
        if (isRead(p)) a.classList.add('is-read');
        if (readCount && p === next) a.classList.add('is-next');

        var n = i + 1;
        var num = el('span', 'part-card__num', (n < 10 ? '0' : '') + n);
        num.setAttribute('aria-hidden', 'true');
        var title = el('span', 'part-card__title', shortTitle(p));
        var foot = el('span', 'part-card__meta');
        if (p.readingMinutes) foot.appendChild(el('span', null, p.readingMinutes + T.min));
        if (readCount && p === next) foot.appendChild(el('span', 'pill pill--accent', T.cont));
        else if (isRead(p)) foot.appendChild(el('span', 'pill', '✓ ' + T.visited));
        else if (isNew(p)) foot.appendChild(el('span', 'pill pill--new', T.isNew));

        a.append(num, title, foot);
        li.appendChild(a);
        parts.appendChild(li);
      });
      band.appendChild(parts);

      seriesContainerEl.appendChild(band);
      shown++;
    });

    seriesSectionEl.hidden = !shown;
  }

  function matches(p) {
    if (state.tag && (p.tags || []).indexOf(state.tag) === -1) return false;
    if (!state.q) return true;
    var s = seriesMap[p.seriesId];
    var hay = lower([p.title, p.summary, (p.tags || []).join(' '), s ? s.title : ''].join(' '));
    return lower(state.q).split(/\s+/).every(function (w) { return hay.indexOf(w) !== -1; });
  }

  function renderList() {
    // Pinned posts lead the grid; the rest keep newest-first order.
    var shown = posts.filter(matches).sort(function (a, b) {
      return (b.pinned ? 1 : 0) - (a.pinned ? 1 : 0);
    });
    listEl.replaceChildren();
    countEl.textContent = T.count(shown.length, posts.length);

    if (!shown.length) {
      statusEl.replaceChildren(el('span', null, T.empty + ' '));
      var clr = el('button', 'link-button', T.clear);
      clr.type = 'button';
      clr.addEventListener('click', function () { setState({ tag: '', q: '' }, true); });
      statusEl.appendChild(clr);
      statusEl.hidden = false;
      return;
    }
    statusEl.hidden = true;

    shown.forEach(function (p) {
      var li = el('li', 'post-card');
      if (isRead(p)) li.classList.add('is-read');
      if (p.pinned) li.classList.add('is-pinned');

      var top = el('div', 'post-card__top');
      if (p.pinned) top.appendChild(el('span', 'pill pill--accent', T.pinned));
      var chip = seriesChip(p);
      if (chip) top.appendChild(chip);
      if (isNew(p)) top.appendChild(el('span', 'pill pill--new', T.isNew));
      if (top.childNodes.length) li.appendChild(top);

      // The title link stretches over the whole card (see .post-card__title a::after).
      var h = el('h3', 'post-card__title');
      var a = el('a', null, shortTitle(p));
      a.href = href(p);
      h.appendChild(a);
      li.appendChild(h);
      li.appendChild(el('p', 'post-card__summary', teaser(p)));

      var meta = el('p', 'post-card__meta');
      var time = el('time', null, fmtDate(p.date, true));
      time.setAttribute('datetime', p.date);
      meta.appendChild(time);
      if (p.readingMinutes) meta.appendChild(el('span', null, p.readingMinutes + T.minRead));
      if (isRead(p)) meta.appendChild(el('span', 'post-card__read', '✓ ' + T.visited));
      li.appendChild(meta);

      listEl.appendChild(li);
    });
  }

  /* Only tags that actually narrow the list: used by 2+ posts, but not by all. */
  function buildFilter() {
    var counts = {};
    posts.forEach(function (p) {
      (p.tags || []).forEach(function (t) { counts[t] = (counts[t] || 0) + 1; });
    });
    var tags = Object.keys(counts).filter(function (t) {
      return counts[t] >= 2 && counts[t] < posts.length;
    }).sort(function (a, b) { return counts[b] - counts[a] || a.localeCompare(b); });

    // A tag arriving via URL keeps its chip even if it would be filtered out.
    if (state.tag && tags.indexOf(state.tag) === -1 && counts[state.tag]) tags.push(state.tag);

    filterEl.replaceChildren();
    [''].concat(tags).forEach(function (t) {
      var li = el('li');
      var b = el('button', 'tag tag--filter', t ? t : T.all);
      if (t) b.appendChild(el('span', 'tag__count', String(counts[t])));
      b.type = 'button';
      b.dataset.tag = t;
      b.addEventListener('click', function () {
        setState({ tag: state.tag === t ? '' : t }, true);
      });
      li.appendChild(b);
      filterEl.appendChild(li);
    });
  }

  function syncControls() {
    filterEl.querySelectorAll('button').forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.dataset.tag === state.tag));
    });
    if (searchEl.value !== state.q) searchEl.value = state.q;
  }

  function setState(patch, pushUrl) {
    Object.keys(patch).forEach(function (k) { state[k] = patch[k]; });
    syncControls();
    renderList();
    if (pushUrl) {
      var u = new URL(location.href);
      ['tag', 'q'].forEach(function (k) {
        if (state[k]) u.searchParams.set(k, state[k]); else u.searchParams.delete(k);
      });
      history.replaceState(null, '', u.pathname + u.search + u.hash);
    }
  }

  var searchTimer;
  searchEl.addEventListener('input', function () {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(function () { setState({ q: searchEl.value.trim() }, true); }, 120);
  });
  searchEl.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && searchEl.value) { e.preventDefault(); setState({ q: '' }, true); }
  });
  // "/" jumps to search, as on most documentation sites.
  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target;
    if (t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName))) return;
    e.preventDefault();
    searchEl.focus();
  });

  fetch(BASE + '/posts.json', { cache: 'no-cache' })
    .then(function (r) {
      if (!r.ok) throw new Error('http ' + r.status);
      return r.json();
    })
    .then(function (data) {
      posts = (data.posts || []).filter(function (p) { return !p.draft; });
      // Newest first; same-day posts in a series put the later part first.
      posts.sort(function (a, b) {
        if (a.date !== b.date) return a.date < b.date ? 1 : -1;
        return (b.seriesPart || 0) - (a.seriesPart || 0);
      });

      if (!posts.length) {
        statusEl.textContent = T.none;
        return;
      }

      var latest = {};
      posts.forEach(function (p) {
        if (p.seriesId && !latest[p.seriesId]) latest[p.seriesId] = p.date;
      });
      // Pinned series (e.g. a "start here" overview) first, then freshest first.
      seriesList = (data.series || []).slice().sort(function (a, b) {
        if (!!a.pinned !== !!b.pinned) return a.pinned ? -1 : 1;
        var ad = latest[a.id] || '', bd = latest[b.id] || '';
        return ad === bd ? 0 : (ad < bd ? 1 : -1);
      });
      seriesMap = {};
      seriesList.forEach(function (s) { seriesMap[s.id] = s; });

      var total = posts.reduce(function (t, p) { return t + (p.readingMinutes || 0); }, 0);
      if (statsEl) statsEl.textContent = T.stats(posts.length, seriesList.length, total);

      var params = new URLSearchParams(location.search);
      state.tag = params.get('tag') || '';
      state.q = params.get('q') || '';

      renderLead();
      renderSeries();
      buildFilter();
      setState({}, false);
    })
    .catch(function () {
      statusEl.hidden = false;
      statusEl.textContent = T.fail;
    });
})();
