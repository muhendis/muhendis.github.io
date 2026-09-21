/* Verifies math in a post survives the real render pipeline:
   1. marked + the mathBlock/mathInline extensions must not mangle LaTeX
      (a bare marked turns \mathbf{x}_{a} into \mathbf{x}<em>{a}).
   2. KaTeX must render every formula without throwing.
   Usage: node check_math.js <file.md> [more.md ...] */
const fs = require('fs');
const path = require('path');
const ROOT = path.resolve(__dirname, '../../../..');

global.window = {};
const m = require(path.join(ROOT, 'assets/js/vendor/marked.umd.js'));
const marked = global.marked || m;
const katex = require(path.join(ROOT, 'assets/js/vendor/katex/katex.min.js'));
const K = global.katex || katex;

const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

marked.setOptions({ gfm: true, breaks: false });
marked.use({ extensions: [
  { name: 'mathBlock', level: 'block', start: s => s.indexOf('$$'),
    tokenizer(src) { const x = /^\$\$([\s\S]+?)\$\$(?:\n|$)/.exec(src);
      if (x) return { type: 'mathBlock', raw: x[0], text: x[1].trim() }; },
    renderer: t => `<div class="math-block">${esc(t.text)}</div>` },
  { name: 'mathInline', level: 'inline', start: s => s.indexOf('$'),
    tokenizer(src) { const x = /^\$([^\s$][^$\n]*?)\$(?!\d)/.exec(src);
      if (x) return { type: 'mathInline', raw: x[0], text: x[1].trim() }; },
    renderer: t => `<span class="math-inline">${esc(t.text)}</span>` },
]});

let bad = 0;
for (const file of process.argv.slice(2)) {
  const html = marked.parse(fs.readFileSync(file, 'utf8'));
  const nodes = html.match(/<(div|span) class="math-(block|inline)">[\s\S]*?<\/\1>/g) || [];
  const leftover = (html.match(/\$\$/g) || []).length;
  let mangled = 0, failed = 0;

  for (const node of nodes) {
    const src = node.replace(/<\/?(div|span)[^>]*>/g, '')
                    .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
    if (/<em>|<strong>/.test(node)) { mangled++; console.log(`  MANGLED: ${src.slice(0, 70)}`); continue; }
    try { K.renderToString(src, { displayMode: node.startsWith('<div'), throwOnError: true }); }
    catch (e) { failed++; console.log(`  KATEX FAIL: ${src.slice(0, 55)} -> ${e.message.split('\n')[0].slice(0, 60)}`); }
  }
  const status = (mangled || failed || leftover) ? 'FAIL' : 'ok';
  console.log(`${file}: ${nodes.length} formulas, mangled=${mangled}, katex-fail=${failed}, raw $$=${leftover} [${status}]`);
  if (mangled || failed || leftover) bad++;
}
process.exit(bad ? 1 : 0);
