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
    posts_dir = ROOT / lang / "blog" / "posts"
    stage_ids = [s.get("id") for s in manifests[lang].get("stages", [])]
    groups = {sid: [] for sid in stage_ids}
    other_series = []
    for s in manifests[lang].get("series", []):
        (groups[s.get("stage")] if s.get("stage") in groups else other_series).append(s)
    ordered_series = [s for sid in stage_ids for s in groups[sid]] + other_series
    expected_series_paths = {}
    for mod_idx, s in enumerate(ordered_series, 1):
        folder = f"{mod_idx:02d}-{s.get('id', '')}"
        for part_idx, s_slug in enumerate(s.get("posts", []), 1):
            expected_series_paths[s_slug] = f"{folder}/{part_idx:02d}-{s_slug}.md"

    slugs = set()
    indexed_paths = set()
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
        rel = p.get("path") or f"{slug}.md"
        if ".." in rel or not re.fullmatch(r"[a-z0-9_-]+(?:/[a-z0-9_-]+)*\.md", rel):
            err(f"{lang}/{slug}: invalid post path {rel!r}.")
        elif rel in indexed_paths:
            err(f"{lang}/{slug}: duplicate post path {rel!r}.")
        else:
            indexed_paths.add(rel)
            if not (posts_dir / rel).exists():
                err(f"{lang}/{slug}: posts.json references {rel} but that file does not exist.")
            if slug in expected_series_paths and rel != expected_series_paths[slug]:
                warn(f"{lang}/{slug}: series path is {rel!r}, expected {expected_series_paths[slug]!r}.")

    # Every .md must be indexed. Leading underscore filename = template, exempt.
    for md in sorted(posts_dir.rglob("*.md")):
        if md.name.startswith("_"):
            continue
        rel = md.relative_to(posts_dir).as_posix()
        if rel not in indexed_paths:
            err(f"{lang}: {rel} exists but has no posts.json entry -- it is unreachable.")
        body = md.read_text(encoding="utf-8").lstrip()
        # Skip a leading HTML comment / blockquote before checking the heading.
        probe = re.sub(r"^(?:<!--.*?-->|>.*?(?:\n|$))+", "", body, flags=re.S).lstrip()
        if probe.startswith("# "):
            err(f"{lang}/{rel}: starts with an H1. The <h1> comes from posts.json; "
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
    files += [ROOT / lang / "aklimda-kalanlar" / "index.html", ROOT / lang / "aklimda-kalanlar" / "post.html"]
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
        if f"/{lang}/aklimda-kalanlar/" not in text:
            warn(f"sitemap.xml does not list /{lang}/aklimda-kalanlar/")
        aklimda_json = ROOT / lang / "aklimda-kalanlar" / "posts.json"
        if aklimda_json.exists():
            try:
                ak_data = json.loads(aklimda_json.read_text(encoding="utf-8"))
                ak_dir = ROOT / lang / "aklimda-kalanlar" / "posts"
                for ak_p in ak_data.get("posts", []):
                    ak_rel = ak_p.get("path") or f"{ak_p.get('slug', '')}.md"
                    if not (ak_dir / ak_rel).exists():
                        err(f"{lang}/aklimda-kalanlar: {ak_rel} does not exist on disk.")
                    if not ak_p.get("draft") and f"slug={ak_p['slug']}" not in text:
                        warn(f"sitemap.xml does not list the published essay {lang}/aklimda-kalanlar/{ak_p['slug']}")
            except Exception as e:
                err(f"{lang}/aklimda-kalanlar/posts.json error: {e}")
    for lang, man in manifests.items():
        for p in man.get("posts", []):
            if p.get("draft"):
                continue
            if f"slug={p['slug']}" not in text:
                warn(f"sitemap.xml does not list the published post {lang}/{p['slug']}")

# 6. Pre-rendered blog index (no-JS / crawler fallback) matches posts.json,
#    and every series sits on a known learning-path stage.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_blog_index
for page in build_blog_index.stale_pages(write=False):
    err(f"Pre-rendered list is stale: {page}. Run python3 .github/scripts/build_blog_index.py")
for lang, m in manifests.items():
    stage_ids = {s.get("id") for s in m.get("stages", [])}
    for s in m.get("series", []):
        if s.get("stage") not in stage_ids:
            err(f"{lang}: series '{s.get('id')}' has stage {s.get('stage')!r}, not one of {sorted(stage_ids)}.")
        if not s.get("tagline"):
            warn(f"{lang}: series '{s.get('id')}' has no tagline; the dashboard falls back to the long summary.")
    for p in m.get("posts", []):
        if not p.get("draft") and not p.get("tagline"):
            warn(f"{lang}/{p.get('slug')}: no tagline; cards fall back to the summary's first clause.")

# 7. Vendored dependency keeps its licence.
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
