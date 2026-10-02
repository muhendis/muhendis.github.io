#!/usr/bin/env python3
"""SVG figürlerini açık/koyu temada önizler, ekran görüntüsü alır ve metin taşmalarını raporlar.

Kullanım:
  # 1) Figür JSON'u: {"en": {"prefill": "<svg ...>", ...}, "tr": {...}}
  python3 .claude/skills/cizim/scripts/preview.py figs.json --out <scratchpad>/cizim
      -> <out>/shot_<lang>_<theme>.png  (Read ile bak)
      -> konsola taşma/çakışma raporu (boşsa "0 sorun")

  # 2) Gerçek makale sayfasında marked'dan sağ çıkıyor mu?
  python3 .claude/skills/cizim/scripts/preview.py --page en/how-llms-work --page tr/llm-nasil-calisir
      -> sayfadaki role=img SVG'lerin viewBox listesi + <p><svg> / kaçışlı &lt;rect hatası sayısı

Kontroller (tarayıcıda getBBox ile):
  - viewBox dışına taşan metin
  - metnin içinde başladığı en küçük dikdörtgenden taşması (kutu/chip sınırı)
  - iki metnin birbirine binmesi
  Döndürülmüş (transform) metinler atlanır; onlara gözle bak.
"""
import argparse, glob, json, os, shutil, socket, subprocess, sys, time, html

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
TOKENS = os.path.join(ROOT, "assets/css/tokens.css")

CHECK_JS = r"""
<script>
window.addEventListener('load', () => {
  const issues = [];
  document.querySelectorAll('svg[role=img]').forEach((svg, si) => {
    const vb = svg.viewBox.baseVal;
    const name = svg.dataset.name || ('svg#' + si);
    const texts = [...svg.querySelectorAll('text')].filter(t => !t.getAttribute('transform'));
    const rects = [...svg.querySelectorAll('rect')].map(r => r.getBBox());
    const bbs = texts.map(t => t.getBBox());
    texts.forEach((t, i) => {
      const b = bbs[i], s = t.textContent;
      if (b.x < vb.x - 1 || b.y < vb.y - 1 || b.x + b.width > vb.x + vb.width + 1 || b.y + b.height > vb.y + vb.height + 1)
        issues.push(`${name}: "${s}" viewBox dışına taşıyor`);
      const ax = +t.getAttribute('x'), ay = +t.getAttribute('y') - 3;
      const anchor = t.getAttribute('text-anchor') || 'start';
      const px = anchor === 'middle' ? ax : (anchor === 'end' ? ax - 2 : ax + 2);
      const host = rects.filter(r => px >= r.x && px <= r.x + r.width && ay >= r.y && ay <= r.y + r.height)
                        .sort((a, c) => a.width * a.height - c.width * c.height)[0];
      if (host && (b.x < host.x - 1 || b.x + b.width > host.x + host.width + 1))
        issues.push(`${name}: "${s}" kutusundan taşıyor (${Math.round(b.width)}px metin, ${Math.round(host.width)}px kutu)`);
      // bir kutuya kısmen giren metin (kutuyu tamamen içinde barındırmayan dikdörtgen kenarı kesiyor)
      for (const r of rects) {
        const ox = Math.min(b.x + b.width, r.x + r.width) - Math.max(b.x, r.x);
        const oy = Math.min(b.y + b.height, r.y + r.height) - Math.max(b.y, r.y);
        const inside = b.x >= r.x - 1 && b.x + b.width <= r.x + r.width + 1 && b.y >= r.y - 2 && b.y + b.height <= r.y + r.height + 2;
        if (ox > 2 && oy > 4 && !inside && r !== host) {
          issues.push(`${name}: "${s}" bir kutunun kenarına biniyor (x=${Math.round(r.x)}, y=${Math.round(r.y)})`); break;
        }
      }
      for (let j = i + 1; j < texts.length; j++) {
        const c = bbs[j];
        const ox = Math.min(b.x + b.width, c.x + c.width) - Math.max(b.x, c.x);
        const oy = Math.min(b.y + b.height, c.y + c.height) - Math.max(b.y, c.y);
        if (ox > 1 && oy > 3) issues.push(`${name}: "${s}" ile "${texts[j].textContent}" çakışıyor`);
      }
    });
  });
  const pre = document.createElement('pre'); pre.id = 'cizim-report';
  pre.textContent = JSON.stringify(issues); document.body.appendChild(pre);
});
</script>
"""


def chrome():
    for c in sorted(glob.glob(os.path.expanduser("~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome")), reverse=True):
        return c
    for n in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        if shutil.which(n):
            return shutil.which(n)
    sys.exit("Chrome/Chromium bulunamadı (playwright cache veya PATH).")


def run(args):
    return subprocess.run([chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                           "--virtual-time-budget=6000", *args], capture_output=True, text=True).stdout


def preview(figs_path, out):
    os.makedirs(out, exist_ok=True)
    figs = json.load(open(figs_path))
    css = open(TOKENS).read()
    total = 0
    for lang, d in figs.items():
        body = "\n".join(s.replace("<svg ", f'<svg data-name="{lang}/{n}" ', 1) for n, s in d.items())
        for theme in ("light", "dark"):
            page = os.path.join(out, f"preview_{lang}_{theme}.html")
            open(page, "w").write(
                f'<!doctype html><html data-theme="{theme}"><head><meta charset="utf-8"><style>{css}'
                f'body{{background:var(--c-bg);width:700px;margin:0;padding:16px}}'
                f'#cizim-report{{display:none}}</style></head><body>{body}{CHECK_JS}</body></html>')
            shot = os.path.join(out, f"shot_{lang}_{theme}.png")
            run([f"--window-size=740,{400 * len(d) + 200}", f"--screenshot={shot}", "file://" + page])
            if theme == "light":  # metin ölçüleri temadan bağımsız; raporu bir kez al
                dom = run(["--dump-dom", "file://" + page])
                start = dom.find('<pre id="cizim-report">')
                raw = dom[start:].split(">", 1)[1].split("</pre>", 1)[0] if start >= 0 else "[]"
                issues = json.loads(html.unescape(raw))
                total += len(issues)
                for i in issues:
                    print("  ⚠", i)
            print(f"{lang}/{theme}: {shot}")
    print(f"{total} sorun" if total else "0 sorun")


def check_pages(pages):
    s = socket.socket(); s.bind(("", 0)); port = s.getsockname()[1]; s.close()
    srv = subprocess.Popen([sys.executable, "-m", "http.server", str(port)], cwd=ROOT,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1)
        for p in pages:
            lang, slug = p.split("/", 1)
            dom = run(["--window-size=390,844", "--dump-dom",
                       f"http://localhost:{port}/{lang}/blog/post.html?slug={slug}"])
            import re
            vbs = re.findall(r'<svg viewBox="([^"]+)" role="img"', dom)
            broken = len(re.findall(r"<p><svg|<p><rect|&lt;rect|&lt;svg", dom))
            print(f"{p}: {len(vbs)} figür {vbs}  bozuk={broken}")
    finally:
        srv.terminate()  # pkill -f KULLANMA: kendi kabuğunu da öldürür


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("figs", nargs="?")
    ap.add_argument("--out", default="cizim_preview")
    ap.add_argument("--page", action="append", default=[])
    a = ap.parse_args()
    if a.figs:
        preview(a.figs, a.out)
    if a.page:
        check_pages(a.page)
    if not a.figs and not a.page:
        ap.print_help()
