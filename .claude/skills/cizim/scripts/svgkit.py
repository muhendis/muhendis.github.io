"""Blog içi inline SVG diyagramları için küçük yardımcı kütüphane.

Her şekil tema token'larıyla (var(--c-*)) boyanır; böylece aynı SVG açık ve koyu temada
kendiliğinden doğru renk alır. Fonksiyonlar SVG parçası (str) döndürür; bir figür bu
parçaların "\n".join'idir. Markdown'a gömülen çıktıda BOŞ SATIR OLMAMALIDIR (marked'ın
HTML bloğu boş satırda biter).

Kullanım:
    import sys; sys.path.insert(0, ".claude/skills/cizim/scripts")
    from svgkit import *
    o = [svg_open(560, 300, "aria açıklaması"), "<defs>", marker("fig-arr"), "</defs>"]
    o.append(box(16, 16, 200, 50)); o.append(text(116, 46, "Katman", TXT, "middle"))
    o.append("</svg>"); svg = "\n".join(o)
"""
from html import escape as esc

# --- Renk anlamları -------------------------------------------------------
# Girdi öğeleri (token, satır, istek) için döngüsel palet; aynı öğe figür boyunca
# aynı rengi taşır (ör. token -> KV cache yuvası).
PAL = ["var(--c-accent)", "var(--c-success)", "var(--c-warn)", "var(--c-danger)",
       "var(--c-text-mute)", "var(--c-accent)"]
GEN = "var(--c-accent-2)"        # modelin ürettiği / yeni olan her şey (mor)
ACC = "var(--c-accent)"          # birincil vurgu (teal)
MUTE_C = "var(--c-text-mute)"

# --- Metin stilleri -------------------------------------------------------
MUTE = "fill:var(--c-text-mute);font-size:12px"            # açıklama, alt yazı
TXT = "fill:var(--c-text);font-size:13px"                  # kutu başlığı
SMALL = "fill:var(--c-text-mute);font-size:11px"           # ok etiketi (alt sınır)
BOLD = "fill:var(--c-text);font-size:13px;font-weight:600"
MONO = "fill:var(--c-text);font-size:12px;font-family:var(--font-mono)"


def title_style(color=ACC):
    return f"fill:{color};font-size:15px;font-weight:700;letter-spacing:.06em"


# --- Kutu stilleri --------------------------------------------------------
BOX = "fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"
FRAME = "fill:none;stroke:var(--c-border);stroke-width:1.2"


def inner(color):
    """Bir kutunun içindeki vurgulu alt kutu (ör. attention = accent, MLP = success)."""
    return f"fill:var(--c-surface);stroke:{color};stroke-width:1.2"


# --- Temel parçalar -------------------------------------------------------
def svg_open(w, h, aria):
    """Kök <svg>. aria: figürü görmeyen okura her şeyi anlatan tam cümle(ler)."""
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(aria)}" '
            'style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;'
            'font-family:var(--font-sans)">')


def marker(mid, color=MUTE_C):
    """Ok ucu. mid sayfa genelinde benzersiz olmalı: figür önekiyle adlandır (pf-arr, dc-arr)."""
    return (f'<marker id="{mid}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
            f'markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" '
            f'style="fill:{color}"/></marker>')


def box(x, y, w, h, style=BOX, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" style="{style}"/>'


def text(x, y, s, style=MUTE, anchor="start", extra=""):
    """extra: ek öznitelik, ör. ' transform="rotate(-90 203 210)"'."""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="{style}"{extra}>{esc(s)}</text>'


def chip(x, y, w, h, color, label=None, size=13, dashed=False, opacity=".16"):
    """Renkli etiket/token kutusu: renkli kontur + düşük opaklıkta dolgu + --c-text yazı.
    Bu kombinasyon iki temada da okunur; dolu renk üstüne beyaz yazı KULLANMA."""
    d = ";stroke-dasharray:4 3" if dashed else ""
    s = (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" '
         f'style="fill:{color};fill-opacity:{opacity};stroke:{color};stroke-width:1.3{d}"/>')
    if label is not None:
        s += (f'<text x="{x + w / 2}" y="{y + h / 2 + size * 0.35:.1f}" text-anchor="middle" '
              f'style="fill:var(--c-text);font-size:{size}px;font-family:var(--font-mono)">{esc(label)}</text>')
    return s


def arrow(x1, y1, x2, y2, mid, color=MUTE_C, dash=False, width=1.5):
    d = ";stroke-dasharray:5 4" if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#{mid})" '
            f'style="stroke:{color};stroke-width:{width}{d}"/>')


def path(d, mid, color=MUTE_C, dash=False):
    """Kırık/eğri ok: 'M228 166 H208 V245 H182' veya 'M332 169 C 292 169, 292 65, 352 65'."""
    ds = ";stroke-dasharray:5 4" if dash else ""
    return f'<path d="{d}" marker-end="url(#{mid})" style="fill:none;stroke:{color};stroke-width:1.5{ds}"/>'


def bars_v(x0, base_y, heights, step=24, w=14, hot=None, hot_color=GEN):
    """Dikey çubuklar (logit/olasılık siluet). hot: vurgulanacak indeks."""
    out = []
    for i, h in enumerate(heights):
        c, op = (hot_color, ".85") if i == hot else (MUTE_C, ".35")
        out.append(f'<rect x="{x0 + i * step}" y="{base_y - h}" width="{w}" height="{h}" rx="2" '
                   f'style="fill:{c};fill-opacity:{op}"/>')
    return "\n".join(out)


def bars_h(x0, y0, widths, step=18, h=12, hot=0, hot_color=GEN):
    """Yatay çubuklar (sıralı olasılıklar)."""
    out = []
    for i, w in enumerate(widths):
        c, op = (hot_color, ".85") if i == hot else (MUTE_C, ".35")
        out.append(f'<rect x="{x0}" y="{y0 + i * step}" width="{w}" height="{h}" rx="2" '
                   f'style="fill:{c};fill-opacity:{op}"/>')
    return "\n".join(out)


def text_w(s, size=12, mono=False):
    """Kaba genişlik tahmini (px). Yerleşim planlarken kullan; kesin kontrol preview.py'de."""
    return len(s) * size * (0.60 if mono else 0.55)


def svg_close():
    return "</svg>"


def assert_no_blank_lines(svg):
    assert "\n\n" not in svg, "SVG içinde boş satır var: marked HTML bloğunu orada keser"
    return svg
