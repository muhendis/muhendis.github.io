#!/usr/bin/env python3
"""
İki dilli (EN + TR) makale simetrisi kontrol aracı.
Kullanım:
    python3 check_bilingual_symmetry.py <slug>
    python3 check_bilingual_symmetry.py en/blog/posts/<en_slug>.md tr/blog/posts/<tr_slug>.md
"""
import sys
import re
from pathlib import Path

def analyze_post(path: Path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()

    # Başlıklar
    h2 = [l for l in lines if re.match(r'^##\s+', l)]
    h3 = [l for l in lines if re.match(r'^###\s+', l)]

    # Fenced code blocks
    code_blocks = re.findall(r'^```[\w-]*', text, flags=re.MULTILINE)

    # Formüller
    math_blocks = re.findall(r'\$\$[\s\S]+?\$\$', text)
    math_inline = re.findall(r'\$([^\s$][^$\n]*?)\$(?!\d)', text)

    # Tablo satırları (header ayırıcıları: | --- |)
    table_headers = re.findall(r'\|(?:\s*:?-+:?\s*\|)+', text)

    # 6 satırda bütün hikaye kontrolü
    six_lines_section = False
    six_lines_count = 0
    glossary_count = 0
    in_glossary = False

    for line in lines:
        if re.search(r'##\s+(The whole story in six lines|Bütün hikâye altı satırda)', line, re.I):
            six_lines_section = True
            in_glossary = False
            continue
        elif re.search(r'##\s+(Glossary|Terimler sözlüğü)', line, re.I):
            six_lines_section = False
            in_glossary = True
            continue
        elif re.match(r'^##\s+', line):
            six_lines_section = False
            in_glossary = False

        if six_lines_section and re.match(r'^\s*-\s+', line):
            six_lines_count += 1
        elif in_glossary and re.match(r'^\s*-\s+\*\*', line):
            glossary_count += 1

    return {
        'h2_count': len(h2),
        'h3_count': len(h3),
        'code_blocks': len(code_blocks),
        'math_blocks': len(math_blocks),
        'math_inline': len(math_inline),
        'tables': len(table_headers),
        'six_lines': six_lines_count,
        'glossary': glossary_count,
        'h2_list': [re.sub(r'^##\s+', '', l).strip() for l in h2]
    }

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python3 check_bilingual_symmetry.py <slug> veya <en_path> <tr_path>")
        sys.exit(1)

    root = Path(__file__).resolve().parents[4]

    if len(sys.argv) == 2:
        slug = sys.argv[1].replace('.md', '')
        # slug translationSlug olabilir; kontrol et
        en_path = root / f"en/blog/posts/{slug}.md"
        tr_path = root / f"tr/blog/posts/{slug}.md"

        if not en_path.exists() or not tr_path.exists():
            # posts.json'dan translationSlug bak
            import json
            for p_file in [root / "tr/blog/posts.json", root / "en/blog/posts.json"]:
                if p_file.exists():
                    data = json.loads(p_file.read_text())
                    for item in data.get('posts', []):
                        if item.get('slug') == slug:
                            trans = item.get('translationSlug')
                            if trans:
                                if not en_path.exists():
                                    en_path = root / f"en/blog/posts/{trans}.md"
                                if not tr_path.exists():
                                    tr_path = root / f"tr/blog/posts/{trans}.md"
    else:
        en_path = Path(sys.argv[1])
        tr_path = Path(sys.argv[2])

    if not en_path.exists():
        print(f"Hata: EN dosyası bulunamadı: {en_path}")
        sys.exit(1)
    if not tr_path.exists():
        print(f"Hata: TR dosyası bulunamadı: {tr_path}")
        sys.exit(1)

    en_res = analyze_post(en_path)
    tr_res = analyze_post(tr_path)

    print(f"\n=======================================================")
    print(f"Bilingual Symmetry Check: {en_path.name} <---> {tr_path.name}")
    print(f"=======================================================")
    print(f"{'Metrik':<25} | {'EN':<10} | {'TR':<10} | {'Durum'}")
    print(f"-------------------------------------------------------")

    metrics = [
        ('H2 Bölüm Sayısı', 'h2_count', True),
        ('H3 Alt Başlık Sayısı', 'h3_count', False),
        ('Kod Blokları (fenced)', 'code_blocks', True),
        ('Blok Formüller ($$)', 'math_blocks', True),
        ('Satır İçi Formüller ($)', 'math_inline', False),
        ('Tablo Sayısı', 'tables', True),
        ('6 Satırda Hikaye (Hedef=6)', 'six_lines', True),
        ('Sözlük Terim Sayısı', 'glossary', True),
    ]

    has_issue = False
    for label, key, strict in metrics:
        en_v = en_res[key]
        tr_v = tr_res[key]
        ok = (en_v == tr_v)
        if key == 'six_lines':
            status = "OK" if (en_v == 6 and tr_v == 6) else "UYARI (6 olmalı)"
            if en_v != 6 or tr_v != 6:
                has_issue = True
        elif ok:
            status = "SIMETRİK"
        else:
            diff = abs(en_v - tr_v)
            if strict:
                status = f"FARK: {diff} (KONTROL ET)"
                has_issue = True
            else:
                status = f"FARK: {diff} (Kabul edilebilir)"

        print(f"{label:<25} | {en_v:<10} | {tr_v:<10} | {status}")

    print(f"-------------------------------------------------------")
    if en_res['six_lines'] != 6:
        print(f"UYARI: EN 'The whole story in six lines' tam 6 madde içermiyor ({en_res['six_lines']} bulundu).")
    if tr_res['six_lines'] != 6:
        print(f"UYARI: TR 'Bütün hikâye altı satırda' tam 6 madde içermiyor ({tr_res['six_lines']} bulundu).")

    if not has_issue:
        print("Tebrikler! EN ve TR sürümleri yüksek simetriye sahip.\n")
    else:
        print("Not: Sayısal veya yapısal farklar varsa, bir dilde eksik veya fazla bölüm olup olmadığını kontrol edin.\n")

if __name__ == '__main__':
    main()
