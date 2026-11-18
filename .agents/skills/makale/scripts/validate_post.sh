#!/usr/bin/env bash
set -e

# Tek komutla tam makale doğrulaması:
# Kullanım:
#   ./validate_post.sh <slug_veya_dosya>
# Örnek:
#   ./.agents/skills/makale/scripts/validate_post.sh tokenizasyon-nasil-calisir
#   ./.agents/skills/makale/scripts/validate_post.sh embedding-katmani-derinlemesine

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "==> 1. Genel site doğrulaması (.github/scripts/validate_site.py)..."
python3 "$ROOT_DIR/.github/scripts/validate_site.py"

if [ "$#" -gt 0 ]; then
    TARGET="$1"
    SLUG="${TARGET##*/}"
    SLUG="${SLUG%.md}"
    # Numaralı dosya adı verildiyse (örn. 01-how-llms-work.md) baştaki NN- önekini temizle
    CLEAN_SLUG="${SLUG#[0-9][0-9]-}"

    read -r EN_FILE TR_FILE < <(python3 - "$ROOT_DIR" "$TARGET" "$SLUG" "$CLEAN_SLUG" <<'PY'
import json, pathlib, sys

root = pathlib.Path(sys.argv[1])
target = sys.argv[2]
slug = sys.argv[3]
clean_slug = sys.argv[4]

manifests = {}
for lang in ("en", "tr"):
    mp = root / lang / "blog" / "posts.json"
    manifests[lang] = json.loads(mp.read_text(encoding="utf-8")).get("posts", []) if mp.exists() else []

def resolve(lang, s):
    posts_dir = root / lang / "blog" / "posts"
    for p in manifests[lang]:
        if p.get("slug") == s or p.get("path") == target:
            rel = p.get("path") or f"{p['slug']}.md"
            cand = posts_dir / rel
            if cand.exists():
                return str(cand), p.get("translationSlug", "")
    # Fallback: recursive search by exact or numbered filename
    for cand in sorted(posts_dir.rglob("*.md")):
        if cand.stem in (s, clean_slug) or cand.stem.endswith(f"-{clean_slug}"):
            return str(cand), ""
    return "", ""

en_file, en_trans = resolve("en", clean_slug)
tr_file, tr_trans = resolve("tr", clean_slug)

if not en_file and tr_trans:
    en_file, _ = resolve("en", tr_trans)
if not tr_file and en_trans:
    tr_file, _ = resolve("tr", en_trans)

print(f"{en_file}\t{tr_file}")
PY
)

    FILES_TO_CHECK=()
    [ -n "$EN_FILE" ] && FILES_TO_CHECK+=("$EN_FILE")
    [ -n "$TR_FILE" ] && FILES_TO_CHECK+=("$TR_FILE")

    if [ ${#FILES_TO_CHECK[@]} -gt 0 ]; then
        echo ""
        echo "==> 2. İçindekiler Tablosu (TOC) link kontrolü..."
        node "$SCRIPT_DIR/check_toc.js" "${FILES_TO_CHECK[@]}"

        echo ""
        echo "==> 3. LaTeX / KaTeX matematik render ve marked uyumluluk kontrolü..."
        node "$SCRIPT_DIR/check_math.js" "${FILES_TO_CHECK[@]}"

        if [ -n "$EN_FILE" ] && [ -n "$TR_FILE" ]; then
            echo ""
            echo "==> 4. İki dilli (EN <-> TR) yapısal simetri kontrolü..."
            python3 "$SCRIPT_DIR/check_bilingual_symmetry.py" "$EN_FILE" "$TR_FILE"
        fi
    else
        echo "Uyarı: '$TARGET' ile eşleşen markdown dosyası bulunamadı."
    fi
fi

echo "==> Tüm doğrulamalar başarıyla tamamlandı!"
