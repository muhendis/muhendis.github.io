const fs = require('fs');

function slugify(text) {
  return text.toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, '')
    .trim()
    .replace(/\s+/g, '-');
}

for (const file of process.argv.slice(2)) {
  const md = fs.readFileSync(file, 'utf8');
  const lines = md.split('\n');
  let inFence = false;
  const ids = [];
  for (const line of lines) {
    if (/^```/.test(line)) { inFence = !inFence; continue; }
    if (inFence) continue;
    const m = line.match(/^(##+)\s+(.*)$/);
    if (m && m[1].length <= 3) {
      // marked renders inline markdown; strip **bold**/`code` markers for textContent
      const text = m[2].replace(/[*`]/g, '');
      ids.push(slugify(text));
    }
  }
  const tocLinks = [...md.matchAll(/\]\(#([^)]+)\)/g)].map(m => m[1]);
  const idSet = new Set(ids);
  let bad = 0;
  for (const link of tocLinks) {
    if (!idSet.has(link)) { console.log(`MISSING in ${file}: #${link}`); bad++; }
  }
  console.log(`${file}: ${tocLinks.length} TOC links, ${ids.length} headings, ${bad} broken`);
}
