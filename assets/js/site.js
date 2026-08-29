/* Shared behaviour: nav disclosure + theme toggle.
   The theme is APPLIED by the inline <head> script to avoid a flash of the
   wrong theme; this file only wires the button. */
(function () {
  'use strict';

  var toggle = document.querySelector('.nav-toggle');
  var nav = document.querySelector('.site-nav');
  if (toggle && nav) {
    toggle.addEventListener('click', function () {
      var open = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(open));
    });
  }

  var themeBtn = document.querySelector('.theme-toggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      var current = document.documentElement.dataset.theme ||
                    (systemDark ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) { /* private mode */ }
      themeBtn.setAttribute('aria-label',
        next === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
    });
  }
})();
