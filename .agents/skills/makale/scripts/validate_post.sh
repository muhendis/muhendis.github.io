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

    EN_FILE=""
    TR_FILE=""

    if [ -f "$ROOT_DIR/en/blog/posts/$SLUG.md" ]; then
        EN_FILE="$ROOT_DIR/en/blog/posts/$SLUG.md"
    fi
    if [ -f "$ROOT_DIR/tr/blog/posts/$SLUG.md" ]; then
        TR_FILE="$ROOT_DIR/tr/blog/posts/$SLUG.md"
    fi

    # Eğer tek dil eşleştiyse translationSlug üzerinden diğerini bul
    if [ -n "$TR_FILE" ] && [ -z "$EN_FILE" ]; then
        TRANS=$(python3 -c "import json; data=json.load(open('$ROOT_DIR/tr/blog/posts.json')); p=next((x for x in data.get('posts',[]) if x.get('slug')=='$SLUG'), None); print(p.get('translationSlug','') if p else '')")
        if [ -n "$TRANS" ] && [ -f "$ROOT_DIR/en/blog/posts/$TRANS.md" ]; then
            EN_FILE="$ROOT_DIR/en/blog/posts/$TRANS.md"
        fi
    elif [ -n "$EN_FILE" ] && [ -z "$TR_FILE" ]; then
        TRANS=$(python3 -c "import json; data=json.load(open('$ROOT_DIR/en/blog/posts.json')); p=next((x for x in data.get('posts',[]) if x.get('slug')=='$SLUG'), None); print(p.get('translationSlug','') if p else '')")
        if [ -n "$TRANS" ] && [ -f "$ROOT_DIR/tr/blog/posts/$TRANS.md" ]; then
            TR_FILE="$ROOT_DIR/tr/blog/posts/$TRANS.md"
        fi
    fi

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
