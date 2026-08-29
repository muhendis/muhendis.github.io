#!/usr/bin/env python3
"""Consistency checks for the no-build static site.

Catches the failure modes that are invisible locally but break production:
a missing .nojekyll, a post that exists but is not indexed, an asymmetric
translation link, or shared header/footer markup that has drifted apart.

Run: python3 .github/scripts/validate_site.py
Exit 0 = clean, 1 = problems found.
"""
import json, pathlib, re, sys, xml.dom.minidom

ROOT = pathlib.Path(__file__).resolve().parents[2]
LANGS = ("en", "tr")
PAGES = ("index.html", "publications.html", "projects.html", "speaking.html")
errors, warnings = [], []

def err(m): errors.append(m)
def warn(m): warnings.append(m)

# 1. .nojekyll -- without it Pages runs Jekyll and every .md fetch 404s.
nj = ROOT / ".nojekyll"
if not nj.exists():
    err(".nojekyll is missing from the repo root. Every .md fetch will 404 in production.")
elif nj.stat().st_size != 0:
    warn(f".nojekyll should be empty (it is {nj.stat().st_size} bytes).")

# 2. Manifests parse, and slugs match files both ways.
manifests = {}
for lang in LANGS:
    mp = ROOT / lang / "blog" / "posts.json"
    if not mp.exists():
        err(f"{lang}/blog/posts.json is missing."); continue
    try:
        manifests[lang] = json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err(f"{lang}/blog/posts.json is not valid JSON: {e}"); continue

    entries = manifests[lang].get("posts", [])
    slugs = set()
    for p in entries:
        slug = p.get("slug", "")
        if not slug:
            err(f"{lang}: a post entry has no slug."); continue
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug):
            err(f"{lang}: slug '{slug}' is not lowercase-hyphen; blog-post.js will reject it.")
        if slug in slugs:
            err(f"{lang}: duplicate slug '{slug}'.")
        slugs.add(slug)
        for field in ("title", "date", "summary"):
            if not p.get(field):
                err(f"{lang}/{slug}: missing '{field}'.")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(p.get("date", ""))):
            err(f"{lang}/{slug}: date must be YYYY-MM-DD (got {p.get('date')!r}).")
        if not (ROOT / lang / "blog" / "posts" / f"{slug}.md").exists():
            err(f"{lang}/{slug}: posts.json references {slug}.md but that file does not exist.")

    # Every .md must be indexed. Leading underscore = template, exempt.
    for md in sorted((ROOT / lang / "blog" / "posts").glob("*.md")):
        if md.name.startswith("_"):
            continue
        if md.stem not in slugs:
            err(f"{lang}: {md.name} exists but has no posts.json entry -- it is unreachable.")
        body = md.read_text(encoding="utf-8").lstrip()
        # Skip a leading HTML comment / blockquote before checking the heading.
        probe = re.sub(r"^(?:<!--.*?-->|>.*?(?:\n|$))+", "", body, flags=re.S).lstrip()
        if probe.startswith("# "):
            err(f"{lang}/{md.name}: starts with an H1. The <h1> comes from posts.json; "
                f"start the body at '##'.")

# 3. translationSlug must be symmetric.
if len(manifests) == 2:
    idx = {l: {p["slug"]: p for p in manifests[l].get("posts", []) if p.get("slug")}
           for l in LANGS}
    for lang in LANGS:
        other = "tr" if lang == "en" else "en"
        for slug, p in idx[lang].items():
            ts = p.get("translationSlug")
            if ts is None:
                continue
            if ts not in idx[other]:
                err(f"{lang}/{slug}: translationSlug '{ts}' has no entry in {other}/blog/posts.json.")
            elif idx[other][ts].get("translationSlug") != slug:
                err(f"Asymmetric translation: {lang}/{slug} -> {other}/{ts}, "
                    f"but {other}/{ts} points back to {idx[other][ts].get('translationSlug')!r}.")

# 4. Shared header/footer regions must be byte-identical within a language.
region_re = re.compile(r"<!-- #region (shared:\w+) v(\d+) -->(.*?)<!-- #endregion \1 -->", re.S)
def normalise(s):
    """Compare structure, ignoring the two attributes that are meant to vary
    per page: the active-nav marker and the language-switch target."""
    s = re.sub(r'\s+aria-current="page"', "", s)
    s = re.sub(r'(class="lang-switch"\s+href=)"[^"]*"', r'\1"*"', s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

for lang in LANGS:
    files = [ROOT / lang / p for p in PAGES]
    files += [ROOT / lang / "blog" / "index.html", ROOT / lang / "blog" / "post.html"]
    seen = {}
    for f in files:
        if not f.exists():
            err(f"{f.relative_to(ROOT)} is missing."); continue
        text = f.read_text(encoding="utf-8")
        found = region_re.findall(text)
        names = {n for n, _, _ in found}
        for required in ("shared:header", "shared:footer"):
            if required not in names:
                err(f"{f.relative_to(ROOT)}: missing '{required}' region markers.")
        for name, ver, block in found:
            key = (name, ver)
            norm = normalise(block)
            if key in seen:
                ref_file, ref = seen[key]
                if ref != norm:
                    err(f"{lang}: {name} v{ver} differs between "
                        f"{ref_file.relative_to(ROOT)} and {f.relative_to(ROOT)}.")
            else:
                seen[key] = (f, norm)

# 5. Sitemap parses and covers every page.
sm = ROOT / "sitemap.xml"
if not sm.exists():
    err("sitemap.xml is missing.")
else:
    try:
        xml.dom.minidom.parse(str(sm))
    except Exception as e:
        err(f"sitemap.xml is not valid XML: {e}")
    text = sm.read_text(encoding="utf-8")
    for lang in LANGS:
        for p in PAGES:
            want = f"/{lang}/" if p == "index.html" else f"/{lang}/{p}"
            if want not in text:
                warn(f"sitemap.xml does not list {want}")
        if f"/{lang}/blog/" not in text:
            warn(f"sitemap.xml does not list /{lang}/blog/")
    for lang, man in manifests.items():
        for p in man.get("posts", []):
            if p.get("draft"):
                continue
            if f"slug={p['slug']}" not in text:
                warn(f"sitemap.xml does not list the published post {lang}/{p['slug']}")

# 6. Vendored dependency keeps its licence.
if (ROOT / "assets/js/vendor/marked.umd.js").exists():
    lic = ROOT / "assets/js/vendor/marked.LICENSE.txt"
    if not lic.exists() or lic.stat().st_size < 500:
        err("assets/js/vendor/marked.LICENSE.txt is missing or truncated; "
            "MIT requires the notice to ship with the code.")

for w in warnings:
    print(f"warning: {w}")
for e in errors:
    print(f"ERROR: {e}")
print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
sys.exit(1 if errors else 0)
