"""Örnek: how-llms-work / llm-nasil-calisir 7. bölümdeki üç figür (prefill, decode, karşılaştırma).

Çalıştır:  python3 .claude/skills/cizim/examples/prefill_decode.py figs.json
Sonra:     python3 .claude/skills/cizim/scripts/preview.py figs.json --out <scratchpad>/cizim

Desen: çizim fonksiyonu dil bağımsızdır, tüm metinler EN/TR sözlüklerinden gelir (L[...]).
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))
from svgkit import *


# ---------------------------------------------------------------- prefill
def prefill(L):
    o = [svg_open(560, 345, L["pf_aria"]), "<defs>", marker("pf-arr"), "</defs>"]
    o.append(text(16, 20, L["pf_head"]))
    xs = [16 + i * 64 for i in range(6)]
    for i, (x, t) in enumerate(zip(xs, L["prompt"])):
        o.append(chip(x, 30, 56, 26, PAL[i], t, 12))
        o.append(arrow(x + 28, 58, x + 28, 77, "pf-arr", PAL[i]))
    o.append(text(410, 40, L["pf_note1"]))
    o.append(text(410, 56, L["pf_note2"]))
    for y, title, sub, kv in ((80, L["pf_l1"], L["pf_l1_sub"], L["pf_kv1"]),
                              (168, L["pf_ln"], L["pf_ln_sub"], L["pf_kvn"])):
        o.append(box(16, y, 376, 50))
        o.append(text(204, y + 21, title, TXT, "middle"))
        o.append(text(204, y + 39, sub, MUTE, "middle"))
        o.append(arrow(392, y + 25, 420, y + 25, "pf-arr"))
        o.append(box(422, y, 122, 50))
        o.append(text(432, y + 16, kv))
        for i in range(6):
            o.append(chip(432 + i * 18, y + 24, 14, 18, PAL[i]))
    o.append(text(204, 152, "⋮", "fill:var(--c-text-mute);font-size:16px", "middle"))
    o.append(text(216, 151, L["pf_repeat"]))
    o.append(text(483, 152, "⋮", "fill:var(--c-text-mute);font-size:16px", "middle"))
    o.append(arrow(204, 218, 204, 240, "pf-arr"))
    o.append(box(16, 242, 376, 56))
    o.append(text(28, 258, L["pf_logits"]))
    hs = [6, 9, 5, 12, 7, 4, 10, 26, 8, 5, 11, 6, 4, 7]
    for i, hh in enumerate(hs):
        hot = hh == 26
        c = GEN if hot else "var(--c-text-mute)"
        op = ".85" if hot else ".35"
        o.append(f'<rect x="{40 + i * 24}" y="{292 - hh}" width="14" height="{hh}" rx="2" '
                 f'style="fill:{c};fill-opacity:{op}"/>')
    o.append(arrow(392, 270, 420, 270, "pf-arr"))
    o.append(box(422, 242, 122, 56))
    o.append(text(483, 258, L["pf_first"], MUTE, "middle"))
    o.append(chip(438, 266, 90, 24, GEN, L["out1"]))
    o.append(text(16, 324, L["pf_foot1"]))
    o.append(text(16, 340, L["pf_foot2"]))
    o.append("</svg>")
    return "\n".join(o)


# ---------------------------------------------------------------- decode
def decode(L):
    o = [svg_open(560, 330, L["dc_aria"]), "<defs>", marker("dc-arr"), marker("dc-arr-g", GEN), "</defs>"]
    # KV cache panel
    o.append(box(16, 16, 180, 256))
    o.append(text(26, 36, L["dc_kv"], "fill:var(--c-text);font-size:13px;font-weight:600"))
    o.append(text(26, 52, L["dc_kv_sub"]))
    o.append(text(104, 72, "K", MUTE, "middle"))
    o.append(text(158, 72, "V", MUTE, "middle"))
    rows = list(zip(L["prompt"], PAL)) + [(L["out1"], GEN)]
    for i, (t, c) in enumerate(rows):
        y = 80 + i * 26
        last = i == len(rows) - 1
        o.append(text(26, y + 14, t.strip(), "fill:var(--c-text);font-size:11px;font-family:var(--font-mono)"))
        for bx in (82, 136):
            dash = ";stroke-dasharray:4 3" if last else ""
            o.append(f'<rect x="{bx}" y="{y}" width="44" height="18" rx="4" '
                     f'style="fill:{c};fill-opacity:{".08" if last else ".2"};stroke:{c};stroke-width:1.2{dash}"/>')
    # new token on top
    o.append(text(320, 14, L["dc_new"], MUTE, "middle"))
    o.append(chip(270, 20, 100, 26, GEN, L["out1"]))
    o.append(arrow(320, 46, 320, 66, "dc-arr"))
    # layer box
    o.append(box(216, 68, 188, 204))
    o.append(text(310, 86, L["dc_layer"], TXT, "middle"))
    o.append(box(228, 96, 164, 80, "fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"))
    o.append(text(310, 114, L["dc_attn"], TXT, "middle"))
    for j, s in enumerate(L["dc_attn_lines"]):
        o.append(text(310, 132 + j * 16, s, MUTE, "middle"))
    o.append(arrow(196, 120, 226, 120, "dc-arr"))
    o.append(text(205, 100, L["dc_read"], "fill:var(--c-text-mute);font-size:11px", "middle",
                  ' transform="rotate(-90 205 100)"'))
    o.append(path("M228 166 H208 V245 H182", "dc-arr-g", GEN))
    o.append(text(203, 210, L["dc_append"], f"fill:{GEN};font-size:11px", "middle",
                  ' transform="rotate(-90 203 210)"'))
    o.append(arrow(310, 176, 310, 188, "dc-arr"))
    o.append(box(228, 190, 164, 44, "fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"))
    o.append(text(310, 208, "MLP", TXT, "middle"))
    o.append(text(310, 225, L["dc_mlp_sub"], MUTE, "middle"))
    o.append(text(310, 258, L["dc_full"], MUTE, "middle"))
    # logits + next token
    o.append(arrow(404, 150, 432, 150, "dc-arr"))
    o.append(box(434, 68, 110, 132))
    o.append(text(489, 86, "logits", TXT, "middle"))
    for j, w in enumerate([80, 52, 30, 20, 12]):
        c = GEN if j == 0 else "var(--c-text-mute)"
        op = ".85" if j == 0 else ".35"
        o.append(f'<rect x="444" y="{98 + j * 18}" width="{w}" height="12" rx="2" style="fill:{c};fill-opacity:{op}"/>')
    o.append(arrow(489, 200, 489, 222, "dc-arr"))
    o.append(box(434, 224, 110, 48))
    o.append(text(489, 240, L["dc_next"], MUTE, "middle"))
    o.append(chip(446, 246, 86, 20, GEN, L["out2"], 12))
    # loop back to the input chip
    o.append(path("M544 248 H553 V33 H374", "dc-arr-g", GEN, dash=True))
    o.append(text(550, 26, L["dc_loop"], f"fill:{GEN};font-size:12px", "end"))
    o.append(text(16, 298, L["dc_foot1"]))
    o.append(text(16, 314, L["dc_foot2"]))
    o.append("</svg>")
    return "\n".join(o)


# ---------------------------------------------------------------- compare
def compare(L):
    o = [svg_open(560, 290, L["cmp_aria"]), "<defs>", marker("cmp-arr"), marker("cmp-arr-g", GEN), "</defs>"]
    o.append(box(16, 16, 256, 226, "fill:none;stroke:var(--c-border);stroke-width:1.2"))
    o.append(box(288, 16, 256, 226, "fill:none;stroke:var(--c-border);stroke-width:1.2"))
    # prefill side
    o.append(text(32, 40, "PREFILL", "fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em"))
    for i in range(6):
        x = 33 + i * 38
        o.append(chip(x, 54, 32, 22, PAL[i]))
        o.append(arrow(x + 16, 78, x + 16, 96, "cmp-arr", PAL[i]))
    o.append(box(33, 98, 222, 40))
    o.append(text(144, 123, L["cmp_model_pf"], TXT, "middle"))
    o.append(arrow(144, 138, 144, 156, "cmp-arr"))
    o.append(chip(99, 158, 90, 22, GEN, L["cmp_first"], 12))
    o.append(text(32, 206, L["cmp_pf1"], "fill:var(--c-text);font-size:12px"))
    o.append(text(32, 224, L["cmp_pf2"], "fill:var(--c-text-mute);font-size:11.5px"))
    # decode side
    o.append(text(304, 40, "DECODE", f"fill:{GEN};font-size:15px;font-weight:700;letter-spacing:.06em"))
    o.append(chip(356, 54, 32, 22, GEN))
    o.append(arrow(372, 78, 372, 96, "cmp-arr"))
    o.append(box(304, 98, 136, 40))
    o.append(text(372, 123, L["cmp_model_dc"], TXT, "middle"))
    o.append(box(458, 98, 72, 40))
    o.append(text(494, 123, "KV cache", "fill:var(--c-text);font-size:12px", "middle"))
    o.append(arrow(458, 112, 442, 112, "cmp-arr"))
    o.append(arrow(440, 126, 456, 126, "cmp-arr-g", GEN))
    o.append(arrow(372, 138, 372, 156, "cmp-arr"))
    o.append(chip(332, 158, 80, 22, GEN, L["cmp_one"], 12))
    o.append(path("M332 169 C 292 169, 292 65, 352 65", "cmp-arr-g", GEN, dash=True))
    o.append(text(296, 192, L["cmp_repeat"], f"fill:{GEN};font-size:11px", "start"))
    o.append(text(304, 206, L["cmp_dc1"], "fill:var(--c-text);font-size:12px"))
    o.append(text(304, 224, L["cmp_dc2"], "fill:var(--c-text-mute);font-size:11.5px"))
    o.append(text(16, 266, L["cmp_foot1"]))
    o.append(text(16, 282, L["cmp_foot2"]))
    o.append("</svg>")
    return "\n".join(o)


EN = dict(
    prompt=["Why", " is", " the", " sky", " blue", "?"], out1="Sunlight", out2=" scatters",
    pf_aria=("Prefill. Six prompt tokens, Why is the sky blue question mark, are all known up front and enter "
             "transformer layer 1 together as one matrix. Each of the N layers runs self-attention plus MLP over all "
             "six positions at once and writes the keys and values of all six tokens into that layer's KV cache. "
             "The last position's logits over the vocabulary pick the first output token, Sunlight; that one pass is "
             "the bulk of time to first token."),
    pf_head="Prompt — all 6 tokens are known up front",
    pf_note1="processed together,", pf_note2="in one pass",
    pf_l1="Layer 1: self-attention + MLP", pf_l1_sub="6 positions as one matrix → GEMM",
    pf_ln="Layer N: self-attention + MLP", pf_ln_sub="causal mask: position i sees only 1…i",
    pf_kv1="KV cache · layer 1", pf_kvn="KV cache · layer N", pf_repeat="same for every layer",
    pf_logits="logits over the vocabulary (last position only)", pf_first="first output token",
    pf_foot1="The first token falls out of the prefill pass itself, so this pass is most of TTFT:",
    pf_foot2="a longer prompt means more rows in every matrix multiply, and a longer wait.",
    dc_aria=("Decode. The KV cache holds keys and values for the six prompt tokens. One new token, Sunlight, enters "
             "the layer stack alone. In each layer, self-attention compares the new token's query against all cached "
             "keys and mixes the cached values, then appends the new token's own key and value to the cache; the MLP "
             "runs for the new token only. The logits pick the next token, scatters, which loops back as the next "
             "input until an end-of-sequence token."),
    dc_kv="KV cache", dc_kv_sub="from prefill + past steps", dc_new="one new token per step",
    dc_layer="each of the N layers", dc_attn="self-attention",
    dc_attn_lines=["q of the new token only", "vs. every cached K", "→ mix of cached V"],
    dc_read="read", dc_append="append k, v", dc_mlp_sub="new token only",
    dc_full="full forward pass, 1 position", dc_next="next token", dc_loop="repeat until <eos>",
    dc_foot1="Nothing is skipped for the new token: it still runs every attention and MLP block. What the cache",
    dc_foot2="saves is re-running the old tokens. Each step re-reads all weights + the KV cache for one token.",
    cmp_aria=("Side by side. Prefill: many known tokens enter the model in parallel, the weights are read once and "
              "reused across all of them, the first token comes out at the end; matrix-matrix work, compute-bound. "
              "Decode: one new token per step, the model reads and appends to the KV cache, one token comes out and "
              "loops back; matrix-vector work, memory-bandwidth bound. Batching many requests restores parallelism "
              "in decode."),
    cmp_model_pf="model · weights read once", cmp_first="1st token",
    cmp_pf1="S known tokens share one weight read", cmp_pf2="matrix × matrix (GEMM) → compute-bound",
    cmp_model_dc="model", cmp_one="1 token", cmp_repeat="repeat",
    cmp_dc1="1 new token per weight read", cmp_dc2="matrix × vector (GEMV) → memory-bound",
    cmp_foot1="One request's decode is strictly sequential; parallelism comes back across requests.",
    cmp_foot2="Batching B decode streams turns that 1 token per weight read into B.",
)

TR = dict(
    prompt=["Gök", "yüzü", " neden", " mavi", " görünür", "?"], out1="Güneş", out2=" ışığı",
    pf_aria=("Prefill. Gökyüzü neden mavi görünür soru işareti prompt'unun altı token'ının hepsi baştan bilinir ve "
             "tek bir matris olarak birlikte 1. transformer katmanına girer. N katmanın her biri altı pozisyonun "
             "hepsi için aynı anda self-attention ve MLP çalıştırır ve altı token'ın anahtar ile değerlerini o "
             "katmanın KV cache'ine yazar. Son pozisyonun sözlük üzerindeki logit'leri ilk çıktı token'ı Güneş'i "
             "seçer; bu tek geçiş TTFT'nin büyük kısmıdır."),
    pf_head="Prompt — 6 token'ın hepsi baştan belli",
    pf_note1="hepsi birlikte,", pf_note2="tek geçişte işlenir",
    pf_l1="Katman 1: self-attention + MLP", pf_l1_sub="6 pozisyon tek matris → GEMM",
    pf_ln="Katman N: self-attention + MLP", pf_ln_sub="causal mask: i. pozisyon yalnız 1…i'yi görür",
    pf_kv1="KV · katman 1", pf_kvn="KV · katman N", pf_repeat="her katmanda aynısı",
    pf_logits="sözlük üzerinde logit'ler (yalnız son pozisyon)", pf_first="ilk çıktı token'ı",
    pf_foot1="İlk token prefill geçişinin kendisinden çıkar; bu yüzden TTFT'nin çoğu bu geçiştir:",
    pf_foot2="prompt uzadıkça her matris çarpımında satır sayısı, dolayısıyla bekleme artar.",
    dc_aria=("Decode. KV cache altı prompt token'ının anahtar ve değerlerini tutar. Tek yeni token, Güneş, katman "
             "yığınına tek başına girer. Her katmanda self-attention yeni token'ın sorgusunu önbellekteki tüm "
             "anahtarlarla karşılaştırıp önbellekteki değerleri karıştırır, ardından yeni token'ın kendi anahtar ve "
             "değerini önbelleğe ekler; MLP yalnız yeni token için çalışır. Logit'ler sıradaki token'ı, ışığı, seçer; "
             "o da dizi sonu token'ı gelene kadar bir sonraki girdi olarak döngüye döner."),
    dc_kv="KV cache", dc_kv_sub="prefill + önceki adımlar", dc_new="her adımda tek yeni token",
    dc_layer="N katmanın her biri", dc_attn="self-attention",
    dc_attn_lines=["yalnız yeni token'ın q'su", "önbellekteki tüm K'lerle", "→ önbellekteki V karışımı"],
    dc_read="oku", dc_append="k, v ekle", dc_mlp_sub="yalnız yeni token",
    dc_full="tam ileri geçiş, 1 pozisyon", dc_next="sıradaki token", dc_loop="<eos> gelene kadar tekrar",
    dc_foot1="Yeni token için hiçbir blok atlanmaz: her attention ve MLP yine çalışır. Önbelleğin kurtardığı şey",
    dc_foot2="eski token'ları yeniden işlemektir. Her adım tek token için bütün ağırlıkları + KV cache'i okur.",
    cmp_aria=("Yan yana. Prefill: baştan bilinen çok sayıda token modele paralel girer, ağırlıklar bir kez okunup "
              "hepsi için kullanılır, ilk token sonda çıkar; matris-matris işi, hesaplama-bağımlı. Decode: her adımda "
              "tek yeni token, model KV cache'i okur ve ona ekler, tek token çıkar ve döngüye döner; matris-vektör "
              "işi, bellek-bant-genişliği bağımlı. Çok sayıda isteği gruplamak decode'a paralelliği geri getirir."),
    cmp_model_pf="model · ağırlıklar bir kez", cmp_first="ilk token",
    cmp_pf1="S token tek ağırlık okumasını paylaşır", cmp_pf2="GEMM (matris × matris) → hesap-bağımlı",
    cmp_model_dc="model", cmp_one="1 token", cmp_repeat="tekrar",
    cmp_dc1="her ağırlık okumasına 1 yeni token", cmp_dc2="GEMV (matris × vektör) → bellek-bağımlı",
    cmp_foot1="Tek isteğin decode'u kesinlikle seridir; paralellik istekler arasından geri gelir.",
    cmp_foot2="B decode akışını gruplamak, ağırlık okuması başına 1 token'ı B'ye çıkarır.",
)

if __name__ == "__main__":
    out = {lang: dict(prefill=prefill(L), decode=decode(L), compare=compare(L))
           for lang, L in (("en", EN), ("tr", TR))}
    for d in out.values():
        for svg in d.values():
            assert_no_blank_lines(svg)
    json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
