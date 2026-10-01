/* Publications page: research-map cards filter the list by theme, and the
   list sorts by citations or year. The static page is complete without it;
   the controls stay hidden until this runs. Citation counts come from the
   data-cites attributes, which site-data.js refreshes from profile.json. */
(function () {
  'use strict';

  var LANG = document.documentElement.lang === 'tr' ? 'tr' : 'en';
  var list = document.getElementById('pub-list');
  if (!list) return;
  var rows = Array.prototype.slice.call(list.querySelectorAll('.pub-row'));
  var cards = document.querySelectorAll('[data-theme-filter]');
  var sortBtns = document.querySelectorAll('[data-sort]');
  var countEl = document.getElementById('pub-count');
  var resetEl = document.getElementById('pub-reset');
  var controls = document.querySelector('.pub-toolbar__controls');

  var state = { theme: '', sort: 'year' };

  function count(n, total) {
    if (LANG === 'tr') return n === total ? total + ' yayın' : total + ' yayından ' + n + ' tanesi';
    return n === total ? total + ' publications' : n + ' of ' + total + ' publications';
  }

  function num(row, key) { return parseInt(row.dataset[key], 10) || 0; }

  function render() {
    var sorted = rows.slice().sort(function (a, b) {
      var primary = state.sort === 'year' ? num(b, 'year') - num(a, 'year') : num(b, 'cites') - num(a, 'cites');
      return primary || (state.sort === 'year' ? num(b, 'cites') - num(a, 'cites') : num(b, 'year') - num(a, 'year'));
    });
    var shown = 0;
    sorted.forEach(function (r) {
      var on = !state.theme || r.dataset.theme === state.theme;
      r.hidden = !on;
      if (on) shown++;
      list.appendChild(r);
    });
    cards.forEach(function (c) { c.setAttribute('aria-pressed', String(c.dataset.themeFilter === state.theme)); });
    sortBtns.forEach(function (b) { b.setAttribute('aria-pressed', String(b.dataset.sort === state.sort)); });
    if (countEl) countEl.textContent = count(shown, rows.length);
    if (resetEl) resetEl.hidden = !state.theme;
  }

  cards.forEach(function (c) {
    c.addEventListener('click', function () {
      state.theme = state.theme === c.dataset.themeFilter ? '' : c.dataset.themeFilter;
      render();
    });
  });
  sortBtns.forEach(function (b) {
    b.addEventListener('click', function () { state.sort = b.dataset.sort; render(); });
  });
  if (resetEl) resetEl.addEventListener('click', function () { state.theme = ''; render(); });

  // site-data.js updates data-cites after profile.json loads; re-sort then.
  document.addEventListener('profile:loaded', render);

  if (controls) controls.hidden = false;
  render();
})();
