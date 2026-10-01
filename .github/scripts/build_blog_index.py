#!/usr/bin/env python3
"""Bakes a plain HTML article list into each blog index page.

The blog index is rendered client-side from posts.json, so crawlers that
do not run JavaScript (and readers without it) would otherwise see an empty
page. This writes the learning path -- stages, modules and their articles --
between the `generated:blog-index` markers in {en,tr}/blog/index.html.
The page script hides it (`.js .static-index`) and draws the dashboard.

Run after every posts.json change:
    python3 .github/scripts/build_blog_index.py          # rewrite
    python3 .github/scripts/build_blog_index.py --check  # exit 1 if stale
validate_site.py runs the check, so a stale list fails CI.
"""
import html, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LANGS = ("en", "tr")
REGION = re.compile(r"(<!-- #region generated:blog-index -->\n).*?(<!-- #endregion generated:blog-index -->)", re.S)

T = {
    "en": {"module": "Module", "minRead": " min read", "standalone": "Standalone", "more": "More",
           "months": "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()},
    "tr": {"module": "Modül", "minRead": " dk okuma", "standalone": "Tekil Yazılar", "more": "Diğer",
           "months": "Oca Şub Mar Nis May Haz Tem Ağu Eyl Eki Kas Ara".split()},
}
SERIES_PREFIX = re.compile(r"^.*?\(\d+\):\s*")


def fmt_date(iso, lang):
    y, m, d = iso.split("-")
    return f"{int(d)} {T[lang]['months'][int(m) - 1]} {y}"


def row(p, lang):
    e = html.escape
    title = SERIES_PREFIX.sub("", p["title"]) if p.get("seriesId") else p["title"]
    blurb = p.get("tagline") or ""
    mins = f"<span>{p['readingMinutes']}{T[lang]['minRead']}</span>" if p.get("readingMinutes") else ""
    return (
        f'<li><a class="post-row" href="/{lang}/blog/post.html?slug={e(p["slug"])}">'
        f'<span class="post-row__body"><span class="post-row__title">{e(title)}</span>'
        + (f'<span class="post-row__teaser">{e(blurb)}</span>' if blurb else "")
        + f'</span><span class="post-row__meta"><time datetime="{e(p["date"])}">{fmt_date(p["date"], lang)}</time>{mins}</span>'
        f"</a></li>"
    )


def render(lang):
    data = json.loads((ROOT / lang / "blog" / "posts.json").read_text(encoding="utf-8"))
    posts = {p["slug"]: p for p in data.get("posts", []) if not p.get("draft")}
    e = html.escape
    stage_ids = [s["id"] for s in data.get("stages", [])]
    groups = {s["id"]: (s, []) for s in data.get("stages", [])}
    other = ({"id": "_other", "title": T[lang]["more"]}, [])
    for s in data.get("series", []):
        (groups.get(s.get("stage")) or other)[1].append(s)

    out, n = ['<div class="static-index">'], 0
    for sid in stage_ids + ["_other"]:
        stage, series = groups[sid] if sid in groups else other
        for s in series:
            items = [posts[x] for x in s.get("posts", []) if x in posts]
            if not items:
                continue
            n += 1
            out.append('<section class="panel">')
            out.append(f'<h2 class="panel-title"><span>{e(stage["title"])} · {T[lang]["module"]} {n:02d} · {e(s["title"])}</span></h2>')
            if s.get("tagline"):
                out.append(f'<p class="dash-lead">{e(s["tagline"])}</p>')
            out.append('<ol class="post-rows">' + "".join(row(p, lang) for p in items) + "</ol>")
            out.append("</section>")
    in_series = {x for s in data.get("series", []) for x in s.get("posts", [])}
    loose = sorted((p for p in posts.values() if p["slug"] not in in_series), key=lambda p: p["date"], reverse=True)
    if loose:
        out.append('<section class="panel">')
        out.append(f'<h2 class="panel-title"><span>{T[lang]["standalone"]}</span></h2>')
        out.append('<ol class="post-rows">' + "".join(row(p, lang) for p in loose) + "</ol>")
        out.append("</section>")
    out.append("</div>")
    return "\n".join(out) + "\n"


def stale_pages(write):
    stale = []
    for lang in LANGS:
        page = ROOT / lang / "blog" / "index.html"
        text = page.read_text(encoding="utf-8")
        if not REGION.search(text):
            stale.append(f"{lang}/blog/index.html has no generated:blog-index region")
            continue
        new = REGION.sub(lambda m: m.group(1) + render(lang) + m.group(2), text)
        if new != text:
            stale.append(f"{lang}/blog/index.html")
            if write:
                page.write_text(new, encoding="utf-8")
    return stale


if __name__ == "__main__":
    check = "--check" in sys.argv
    changed = stale_pages(write=not check)
    if check and changed:
        print("Stale pre-rendered blog index: " + ", ".join(changed)
              + "\nRun: python3 .github/scripts/build_blog_index.py")
        sys.exit(1)
    print("updated: " + ", ".join(changed) if changed else "blog index up to date")
