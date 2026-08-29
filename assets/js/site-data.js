/* Live data layer. The HTML ships with baked values as the fallback; this
   refreshes them at page load from two sources:

   1. /assets/data/profile.json — the single file to edit when numbers change
      (Scholar citations, h-index, stat bar). Google Scholar sends no CORS
      headers, so a browser cannot read it directly; update the JSON instead.
   2. api.github.com — repository star counts, which DO allow browser fetches.

   Every update is a textContent write to elements found by data attributes;
   if either fetch fails, the baked values simply remain. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';

  // --- profile.json: stat bar, scholar stats, per-publication citations ---
  fetch('/assets/data/profile.json', { cache: 'no-cache' })
    .then(function (r) { if (!r.ok) throw new Error('http'); return r.json(); })
    .then(function (d) {
      var stats = d.stats || {};
      document.querySelectorAll('[data-stat]').forEach(function (el) {
        var v = stats[el.dataset.stat];
        if (v != null) el.textContent = v;
      });

      var sch = d.scholar || {};
      document.querySelectorAll('[data-scholar]').forEach(function (el) {
        var v = sch[el.dataset.scholar];
        if (v != null) el.textContent = v;
      });

      var cites = d.publicationCitations || {};
      document.querySelectorAll('[data-pub]').forEach(function (li) {
        var n = cites[li.dataset.pub];
        var span = li.querySelector('.pub__cites');
        if (n == null) return;
        if (!n) { if (span) span.hidden = true; return; }
        var word = LANG === 'tr' ? 'atıf' : (n === 1 ? 'citation' : 'citations');
        if (span) { span.hidden = false; span.textContent = n + ' ' + word; }
      });
    })
    .catch(function () { /* baked values stand */ });

  // --- GitHub API: live star counts on project cards ---
  var starEls = document.querySelectorAll('[data-repo]');
  if (starEls.length) {
    // The account holds >100 repos; one page misses some, so fetch up to
    // three pages (covers 300) and stop early when a page comes back short.
    var fetchPage = function (page) {
      return fetch('https://api.github.com/users/muhendis/repos?per_page=100&page=' + page, {
        headers: { Accept: 'application/vnd.github+json' }
      }).then(function (r) { if (!r.ok) throw new Error('http'); return r.json(); });
    };
    fetchPage(1)
      .then(function (first) {
        if (first.length < 100) return first;
        return fetchPage(2).then(function (second) {
          if (second.length < 100) return first.concat(second);
          return fetchPage(3).then(function (third) {
            return first.concat(second, third);
          });
        });
      })
      .then(function (repos) {
        var stars = {};
        repos.forEach(function (repo) { stars[repo.name] = repo.stargazers_count; });
        starEls.forEach(function (el) {
          var n = stars[el.dataset.repo];
          if (n == null) return;
          if (!n) { el.hidden = true; return; }
          el.hidden = false;
          el.textContent = '★ ' + n + ' ' + el.dataset.starWord;
        });
      })
      .catch(function () { /* baked values stand */ });
  }
})();
