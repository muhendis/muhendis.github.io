/* Renders the post list and mini-series showcase for the current language from posts.json.
   One request total: summaries and series structure live in the manifest. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var BASE = '/' + LANG + '/blog';

  var T = {
    en: {
      empty: 'No posts published yet.',
      fail: 'Could not load posts.',
      min: ' min read',
      all: 'All',
      seriesFilter: '✦ Series',
      by: 'By Engin Bozaba',
      seriesBadge: 'Series',
      totalRead: 'total read',
      part: 'Part'
    },
    tr: {
      empty: 'Henüz yazı yayımlanmadı.',
      fail: 'Yazılar yüklenemedi.',
      min: ' dk okuma',
      all: 'Tümü',
      seriesFilter: '✦ Seriler',
      by: 'Yazan: Engin Bozaba',
      seriesBadge: 'Seri',
      totalRead: 'toplam okuma',
      part: 'Bölüm'
    }
  }[LANG];

  var listEl = document.getElementById('post-list');
  var statusEl = document.getElementById('list-status');
  var filterEl = document.getElementById('tag-filter');
  var seriesSectionEl = document.getElementById('series-section');
  var seriesContainerEl = document.getElementById('series-container');

  var posts = [];
  var seriesList = [];
  var seriesMap = {};
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

  function renderSeriesShowcase() {
    if (!seriesContainerEl || !seriesSectionEl) return;
    seriesContainerEl.replaceChildren();

    if (!seriesList.length) {
      seriesSectionEl.hidden = true;
      return;
    }

    seriesList.forEach(function (s) {
      var sPosts = (s.posts || []).map(function (slug) {
        for (var i = 0; i < posts.length; i++) {
          if (posts[i].slug === slug) return posts[i];
        }
        return null;
      }).filter(Boolean);

      if (!sPosts.length) return;

      var totalMinutes = 0;
      sPosts.forEach(function (p) {
        totalMinutes += (p.readingMinutes || 0);
      });

      var card = el('article', 'series-card');

      // Left column: Series identity & summary
      var header = el('div', 'series-card__header');
      var metaBar = el('div', 'series-card__meta-bar');

      var defaultBadgeText = T.seriesBadge + ' · ' + sPosts.length + ' ' + (LANG === 'tr' ? 'Bölüm' : 'Parts');
      var badge = el('span', 'series-badge', s.badge || defaultBadgeText);
      metaBar.appendChild(badge);

      if (totalMinutes > 0) {
        var dur = el('span', 'series-duration', totalMinutes + T.min + ' ' + T.totalRead);
        metaBar.appendChild(dur);
      }
      header.appendChild(metaBar);

      var h3 = el('h3', 'series-card__title', s.title);
      header.appendChild(h3);

      if (s.summary) {
        var desc = el('p', 'series-card__desc', s.summary);
        header.appendChild(desc);
      }
      card.appendChild(header);

      // Right column: Sequential progression track (01 -> 02 -> 03)
      var track = el('ol', 'series-track');
      sPosts.forEach(function (p, idx) {
        var itemWrap = el('li', 'series-track__item-wrap');
        var a = el('a', 'series-track__item');
        a.href = BASE + '/post.html?slug=' + encodeURIComponent(p.slug);

        var stepNum = idx + 1;
        var num = el('span', 'series-track__num', (stepNum < 10 ? '0' : '') + stepNum);
        var body = el('div', 'series-track__body');

        // Clean title for track: strip series name prefix if present
        var cleanTitle = p.title.replace(/^.*?\(\d+\):\s*/, '');
        var itemTitle = el('div', 'series-track__title', cleanTitle);
        body.appendChild(itemTitle);

        var meta = el('div', 'series-track__meta');
        if (p.readingMinutes) {
          meta.appendChild(el('span', null, p.readingMinutes + T.min));
        }
        if ((p.tags || []).length) {
          var tagPills = p.tags.slice(0, 3).join(' · ');
          meta.appendChild(el('span', null, tagPills));
        }
        body.appendChild(meta);

        var arrow = el('span', 'series-track__arrow', '→');
        arrow.setAttribute('aria-hidden', 'true');

        a.append(num, body, arrow);
        itemWrap.appendChild(a);
        track.appendChild(itemWrap);
      });

      card.appendChild(track);
      seriesContainerEl.appendChild(card);
    });

    seriesSectionEl.hidden = false;
  }

  function render() {
    var shown;
    if (activeTag === '__series__') {
      shown = posts.filter(function (p) { return !!p.seriesId; });
      if (seriesSectionEl) seriesSectionEl.hidden = false;
    } else if (activeTag) {
      shown = posts.filter(function (p) { return (p.tags || []).indexOf(activeTag) !== -1; });
      if (seriesSectionEl) seriesSectionEl.hidden = true;
    } else {
      shown = posts;
      if (seriesSectionEl) seriesSectionEl.hidden = false;
    }

    listEl.replaceChildren();
    if (!shown.length) {
      statusEl.hidden = false;
      statusEl.textContent = T.empty;
      return;
    }
    statusEl.hidden = true;

    shown.forEach(function (p) {
      var li = el('li', 'post-item');

      // Series badge
      if (p.seriesId && seriesMap[p.seriesId]) {
        var sObj = seriesMap[p.seriesId];
        var sBadge = el('div', 'post-series-badge');
        sBadge.textContent = '✦ ' + sObj.title + (p.seriesPart ? ' · ' + T.part + ' ' + p.seriesPart : '');
        li.appendChild(sBadge);
      }

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
      (p.tags || []).forEach(function (t) {
        if (all.indexOf(t) === -1) all.push(t);
      });
    });
    all.sort();

    function mkBtn(label, value, isSpecial) {
      var cls = isSpecial ? 'tag tag--filter tag--special' : 'tag tag--filter';
      var b = el('button', cls, label);
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

    filterEl.appendChild(mkBtn(T.all, null, false));

    // If any series exist, add dedicated Series filter button
    if (seriesList.length > 0) {
      filterEl.appendChild(mkBtn(T.seriesFilter, '__series__', true));
    }

    all.forEach(function (t) {
      filterEl.appendChild(mkBtn(t, t, false));
    });
  }

  fetch(BASE + '/posts.json', { cache: 'no-cache' })
    .then(function (r) {
      if (!r.ok) throw new Error('http ' + r.status);
      return r.json();
    })
    .then(function (data) {
      seriesList = data.series || [];
      seriesMap = {};
      seriesList.forEach(function (s) {
        seriesMap[s.id] = s;
      });

      posts = (data.posts || []).filter(function (p) {
        return !p.draft;
      });

      // Compute latest date per series so each series ranks at its freshest update
      var seriesMaxDate = {};
      posts.forEach(function (p) {
        if (p.seriesId) {
          if (!seriesMaxDate[p.seriesId] || p.date > seriesMaxDate[p.seriesId]) {
            seriesMaxDate[p.seriesId] = p.date;
          }
        }
      });

      // Sequential grouping & chronological sort:
      // Series stay together in reading progression (Part 1 -> Part 2 -> Part 3)
      // and sit at their latest publication date relative to standalone articles.
      posts.sort(function (a, b) {
        var aKey = a.seriesId ? seriesMaxDate[a.seriesId] : a.date;
        var bKey = b.seriesId ? seriesMaxDate[b.seriesId] : b.date;

        if (aKey !== bKey) {
          return aKey < bKey ? 1 : -1;
        }

        // Within the same series, preserve sequential reading order
        if (a.seriesId && b.seriesId && a.seriesId === b.seriesId) {
          return (a.seriesPart || 0) - (b.seriesPart || 0);
        }

        return a.date < b.date ? 1 : -1;
      });

      renderSeriesShowcase();
      buildFilter();
      render();
    })
    .catch(function () {
      statusEl.hidden = false;
      statusEl.textContent = T.fail;
    });
})();
