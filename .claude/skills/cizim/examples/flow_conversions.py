"""Örnek: Mermaid/ASCII → SVG dönüşümleri (Prompt'un Yolculuğu 5–7). node()/link() ile akış, kart, zaman çizelgesi desenleri.
Çalıştır: python3 .claude/skills/cizim/examples/flow_conversions.py figs.json"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))
from svgkit import *

OK, BAD, WARN = "var(--c-success)", "var(--c-danger)", "var(--c-warn)"


def dbl(x1, y1, x2, y2, mid, color=MUTE_C, d=None):
    """double-headed connector"""
    p = d or f"M{x1} {y1} L{x2} {y2}"
    return (f'<path d="{p}" marker-start="url(#{mid})" marker-end="url(#{mid})" '
            f'style="fill:none;stroke:{color};stroke-width:1.5"/>')


def circ(x, y, label, color):
    return (f'<circle cx="{x}" cy="{y}" r="17" style="fill:{color};fill-opacity:.18;stroke:{color};stroke-width:1.5"/>'
            + text(x, y + 4.5, label, "fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)", "middle"))


# ------------------------------------------------------------------ causal
def bidir(L):
    o = [svg_open(560, 196, L["bd_aria"]), "<defs>", marker("bd-arr"), marker("bd-g", GEN), "</defs>"]
    for p, (title, sub) in enumerate(L["bd_panels"]):
        x0 = 16 + p * 272
        o.append(box(x0, 8, 256, 180, FRAME))
        o.append(text(x0 + 14, 30, title, title_style(ACC if p == 0 else GEN)))
        o.append(text(x0 + 14, 47, sub, MUTE))
        xs = [x0 + 48, x0 + 128, x0 + 208]
        y = 112
        if p == 0:
            o.append(dbl(xs[0] + 19, y, xs[1] - 19, y, "bd-arr"))
            o.append(dbl(xs[1] + 19, y, xs[2] - 19, y, "bd-arr"))
            o.append(dbl(0, 0, 0, 0, "bd-arr", d=f"M{xs[0]} {y - 19} C {xs[0] + 30} {y - 52}, {xs[2] - 30} {y - 52}, {xs[2]} {y - 19}"))
        else:
            o.append(arrow(xs[0] + 19, y, xs[1] - 21, y, "bd-g", GEN))
            o.append(arrow(xs[1] + 19, y, xs[2] - 21, y, "bd-g", GEN))
            o.append(path(f"M{xs[0]} {y - 19} C {xs[0] + 30} {y - 52}, {xs[2] - 30} {y - 52}, {xs[2]} {y - 21}", "bd-g", GEN))
        for k, x in enumerate(xs):
            o.append(circ(x, y, f"x{'₁₂₃'[k]}", ACC if p == 0 else GEN))
            o.append(text(x, y + 38, L["bd_sees"][p][k], "fill:var(--c-text-mute);font-size:11px", "middle"))
        o.append(text(x0 + 128, 176, L["bd_foot"][p], "fill:var(--c-text);font-size:11.5px", "middle"))
    o.append(svg_close())
    return "\n".join(o)


def pages(L):
    o = [svg_open(560, 150, L["pg_aria"])]
    o.append(text(16, 18, L["pg_head"], MUTE))
    for k, (tok, frac, l1, l2) in enumerate(L["pg_rows"]):
        y = 30 + k * 40
        o.append(chip(16, y + 4, 82, 24, [ACC, WARN, OK][k], tok, 12))
        bx, bw = 112, 160
        o.append(f'<rect x="{bx}" y="{y + 8}" width="{bw}" height="16" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:.8"/>')
        w = max(4, bw * frac)
        o.append(f'<rect x="{bx}" y="{y + 8}" width="{w}" height="16" rx="3" style="fill:{GEN};fill-opacity:.6"/>')
        if frac < 1:
            o.append(text(bx + w + (bw - w) / 2, y + 20, L["pg_sealed"], "fill:var(--c-text-mute);font-size:10.5px", "middle"))
        o.append(text(288, y + 13, l1, "fill:var(--c-text);font-size:12px;font-weight:600"))
        o.append(text(288, y + 28, l2, MUTE))
    o.append(svg_close())
    return "\n".join(o)


def trainfer(L):
    o = [svg_open(560, 180, L["ti_aria"])]
    for p, (title, sub, lines) in enumerate(L["ti_cards"]):
        x0 = 16 + p * 272
        o.append(box(x0, 8, 256, 164, FRAME))
        o.append(text(x0 + 14, 32, title, title_style(ACC if p == 0 else GEN)))
        o.append(text(x0 + 14, 49, sub, MUTE))
        for k, (ln, kind) in enumerate(lines):
            st = {"b": "fill:var(--c-text);font-size:12px", "s": "fill:var(--c-text-mute);font-size:11.5px",
                  "h": f"fill:{OK};font-size:12px;font-weight:600", "m": "fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)"}[kind]
            o.append(text(x0 + 14, 76 + k * 19, ln, st))
    o.append(svg_close())
    return "\n".join(o)


# ------------------------------------------------------------------ multi-head
def bottleneck(L):
    o = [svg_open(560, 262, L["bn_aria"]), "<defs>", marker("bn-arr"), marker("bn-g", GEN), "</defs>"]
    o.append(text(16, 18, L["bn_l1"], BOLD))
    o.append(node(16, 28, 76, 42, "Z", None, None, mono=True))
    o.append(arrow(92, 49, 110, 49, "bn-arr"))
    o.append(node(112, 28, 200, 42, L["bn_map"], "M = W_Q W_Kᵀ"))
    o.append(arrow(312, 49, 330, 49, "bn-arr"))
    o.append(node(332, 28, 212, 42, L["bn_comp"][0], L["bn_comp"][1], BAD))
    o.append(text(16, 104, L["bn_l2"], f"fill:{GEN};font-size:13px;font-weight:600"))
    o.append(node(16, 160, 76, 42, "Z", None, None, mono=True))
    hy = [114, 146, 178, 210]
    for k, y in enumerate(hy):
        o.append(path(f"M92 181 C 102 181, 102 {y + 13}, 112 {y + 13}", "bn-g", GEN))
        o.append(box(114, y, 200, 26, inner(GEN), 5))
        o.append(text(124, y + 17, L["bn_heads"][k], "fill:var(--c-text);font-size:11.5px"))
        o.append(path(f"M314 {y + 13} C 322 {y + 13}, 322 181, 330 181", "bn-arr"))
    o.append(node(332, 150, 96, 62, "Concat", "+ Wᴼ"))
    o.append(arrow(428, 181, 442, 181, "bn-g", GEN))
    o.append(node(444, 150, 100, 62, L["bn_out"][0], L["bn_out"][1:], GEN))
    o.append(svg_close())
    return "\n".join(o)


def pipeline(L):
    o = [svg_open(560, 208, L["pl_aria"]), "<defs>", marker("pl-arr"), "</defs>"]
    W, G = 96, 12
    xs = [16 + k * (W + G) for k in range(5)]
    row1, row2 = L["pl_nodes"][:5], L["pl_nodes"][5:]
    per = {3, 4, 5, 6}   # global index of per-head steps
    for k, (t, s) in enumerate(row1):
        col = BAD if k == 4 else (ACC if k in per else None)
        o.append(node(xs[k], 26, W, 48, t, s, col))
        if k < 4:
            o.append(arrow(xs[k] + W, 50, xs[k + 1] - 1, 50, "pl-arr"))
    o.append(path(f"M{xs[4] + W / 2} 74 V92 H{xs[0] + W / 2} V108", "pl-arr"))
    for k, (t, s) in enumerate(row2):
        g = 5 + k
        col = GEN if g == 8 else (ACC if g in per else None)
        o.append(node(xs[k], 110, W, 48, t, s, col))
        if k < 3:
            o.append(arrow(xs[k] + W, 134, xs[k + 1] - 1, 134, "pl-arr"))
    o.append(f'<rect x="16" y="170" width="12" height="12" rx="2" style="fill:var(--c-surface);stroke:{ACC};stroke-width:1.2"/>')
    o.append(text(34, 180, L["pl_leg"], MUTE))
    o.append(text(16, 200, L["pl_foot"], MUTE))
    o.append(svg_close())
    return "\n".join(o)


def gpu(L):
    o = [svg_open(560, 250, L["gp_aria"]), "<defs>", marker("gp-arr"), "</defs>"]
    W, H, G = 256, 42, 14
    for k, (t, s) in enumerate(L["gp_nodes"]):
        col_i, row = divmod(k, 4)
        x = 16 + col_i * 272
        y = 10 + row * (H + G)
        c = ACC if k in (1, 7) else (GEN if k == 4 else None)
        o.append(node(x, y, W, H, t, s, c))
        if row < 3:
            o.append(arrow(x + W / 2, y + H, x + W / 2, y + H + G - 1, "gp-arr"))
    o.append(path(f"M{16 + W / 2} {10 + 3 * (H + G) + H} V{10 + 4 * (H + G) - 4} H{280} V{6} H{288 + W / 2} V9", "gp-arr"))
    o.append(text(16, 244, L["gp_foot"], MUTE))
    o.append(svg_close())
    return "\n".join(o)


# ------------------------------------------------------------------ transformer block
def plus(x, y):
    return (f'<circle cx="{x}" cy="{y}" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/>'
            + text(x, y + 5.5, "+", "fill:var(--c-text);font-size:16px;font-weight:700", "middle"))


def blockflow(L):
    o = [svg_open(560, 284, L["bf_aria"]), "<defs>", marker("bf-arr"), marker("bf-g", GEN), "</defs>"]
    rows = [(26, L["bf_sub1"], ACC), (148, L["bf_sub2"], OK)]
    for r, (fy, title, col) in enumerate(rows):
        o.append(box(16, fy, 528, 92, f"fill:none;stroke:{col};stroke-width:1.2;stroke-dasharray:5 4"))
        o.append(text(28 if r == 0 else 90, fy - 6, title, f"fill:{col};font-size:12.5px;font-weight:600"))
    # row 1
    c1 = 26 + 58
    o.append(node(26, c1 - 18, 38, 36, "X", None, None, mono=True))
    o.append(arrow(64, c1, 78, c1, "bf-arr"))
    o.append(node(80, c1 - 18, 44, 36, "Z", None, None, mono=True))
    o.append(arrow(124, c1, 140, c1, "bf-arr"))
    o.append(node(142, c1 - 22, 150, 44, L["bf_mha"], None, ACC))
    o.append(arrow(292, c1, 297, c1, "bf-arr"))
    o.append(plus(312, c1))
    o.append(path(f"M102 {c1 - 18} C 102 {c1 - 40}, 312 {c1 - 40}, 312 {c1 - 15}", "bf-arr", dash=True))
    o.append(text(206, c1 - 47, L["bf_skip"].format("Z"), "fill:var(--c-text-mute);font-size:11px", "middle"))
    o.append(arrow(325, c1, 338, c1, "bf-arr"))
    o.append(node(340, c1 - 18, 100, 36, L["bf_ln"], None))
    o.append(arrow(440, c1, 456, c1, "bf-arr"))
    o.append(node(458, c1 - 18, 72, 36, "H₁", None, None, mono=True))
    # connector to row 2
    c2 = 148 + 58
    o.append(path(f"M494 {c1 + 18} V{c1 + 46} H52 V{c2 - 20}", "bf-arr"))
    o.append(node(26, c2 - 18, 52, 36, "H₁", None, None, mono=True))
    o.append(arrow(78, c2, 92, c2, "bf-arr"))
    o.append(node(94, c2 - 22, 198, 44, L["bf_ffn"], "W₂·σ(W₁·H₁ + b₁) + b₂", OK))
    o.append(arrow(292, c2, 297, c2, "bf-arr"))
    o.append(plus(312, c2))
    o.append(path(f"M60 {c2 - 18} C 60 {c2 - 40}, 312 {c2 - 40}, 312 {c2 - 15}", "bf-arr", dash=True))
    o.append(text(196, c2 - 47, L["bf_skip"].format("H₁"), "fill:var(--c-text-mute);font-size:11px", "middle"))
    o.append(arrow(325, c2, 338, c2, "bf-arr"))
    o.append(node(340, c2 - 18, 100, 36, L["bf_ln"], None))
    o.append(arrow(440, c2, 456, c2, "bf-g", GEN))
    o.append(node(458, c2 - 18, 72, 36, "H₂", None, GEN, mono=True))
    o.append(text(16, 262, L["bf_foot"][0], MUTE))
    o.append(text(16, 278, L["bf_foot"][1], MUTE))
    o.append(svg_close())
    return "\n".join(o)


def skipc(L):
    o = [svg_open(560, 150, L["sk_aria"]), "<defs>", marker("sk-arr"), marker("sk-g", OK), "</defs>"]
    o.append(node(16, 70, 56, 40, "x", None, None, mono=True))
    o.append(path("M72 82 C 100 82, 100 36, 130 36 H 352 C 380 36, 380 74, 380 76", "sk-g", OK, dash=True))
    o.append(text(240, 28, L["sk_skip"], f"fill:{OK};font-size:12px;font-weight:600", "middle"))
    o.append(arrow(72, 98, 168, 98, "sk-arr"))
    o.append(node(170, 76, 150, 44, "F(x)", L["sk_proc"], None, mono=True))
    o.append(arrow(320, 98, 365, 98, "sk-arr"))
    o.append(plus(380, 98))
    o.append(arrow(393, 98, 420, 98, "sk-arr"))
    o.append(node(422, 76, 122, 44, "y = x + F(x)", None, GEN, mono=True))
    o.append(text(16, 142, L["sk_foot"], MUTE))
    o.append(svg_close())
    return "\n".join(o)


def ffnkv(L):
    o = [svg_open(560, 210, L["kv_aria"]), "<defs>", marker("kv-arr"), marker("kv-g", GEN), "</defs>"]
    nodes = [(16, 92, L["kv_n"][0]), (126, 130, L["kv_n"][1]), (274, 130, L["kv_n"][2]), (422, 122, L["kv_n"][3])]
    cols = [None, ACC, WARN, GEN]
    for k, ((x, w, (t, s)), c) in enumerate(zip(nodes, cols)):
        o.append(node(x, 20, w, 56, t, s, c))
        if k < 3:
            nx = nodes[k + 1][0]
            o.append(arrow(x + w, 48, nx - 1, 48, "kv-arr"))
    # firing pattern from the walkthrough (neurons 1, 2, 4 fire)
    for i in range(6):
        on = i in (1, 2, 4)
        cx = 296 + i * 17
        o.append(f'<circle cx="{cx}" cy="94" r="6" style="fill:{WARN if on else "var(--c-surface-2)"};fill-opacity:{".8" if on else "1"};stroke:var(--c-border);stroke-width:1"/>')
    o.append(text(339, 116, L["kv_fire"], "fill:var(--c-text-mute);font-size:11px", "middle"))
    # residual stream band
    o.append(f'<rect x="16" y="138" width="528" height="26" rx="5" style="fill:{ACC};fill-opacity:.12;stroke:{ACC};stroke-width:1"/>')
    o.append(text(280, 155, L["kv_stream"], f"fill:var(--c-text);font-size:12px", "middle"))
    o.append(arrow(62, 138, 62, 78, "kv-arr"))
    o.append(arrow(483, 78, 483, 136, "kv-g", GEN))
    o.append(text(491, 112, L["kv_add"], f"fill:{GEN};font-size:11px"))
    o.append(text(16, 192, L["kv_foot"], MUTE))
    o.append(svg_close())
    return "\n".join(o)


EN = dict(
    bd_aria=("Bidirectional versus causal attention over three tokens. Unmasked, the god's eye view: x1, x2 and x3 are "
             "all linked in both directions, so every token sees every other. Causal, the arrow of time: information "
             "only flows forward; x1 attends to itself, x2 to x1 and itself, x3 to x1, x2 and itself."),
    bd_panels=[("Bidirectional", "unmasked: god's eye view"), ("Causal", "masked: the arrow of time")],
    bd_sees=[["sees all", "sees all", "sees all"], ["sees x₁", "sees x₁, x₂", "sees x₁–x₃"]],
    bd_foot=["every token sees the whole sequence", "each token sees only itself and the past"],
    pg_aria=("A murder mystery with the pages glued. dog is at page 1 and must deduce clues with zero knowledge of the "
             "culprit. cat is at page 50: it remembers page 1, but the future pages stay sealed. chased reads the final "
             "chapter and holds the full context of the narrative."),
    pg_head="each token reads the book only up to its own page",
    pg_rows=[("dog", 0.03, "page 1", "no idea who the culprit is"), ("cat", 0.45, "page 50", "remembers page 1, rest sealed"),
             ("chased", 1.0, "final chapter", "the whole story is in view")],
    pg_sealed="sealed",
    ti_aria=("Causal attention in training and inference. Training, parallel with teacher forcing: the full sequence "
             "goes in at once, the causal mask is applied as a T by T matrix, the masked gradients are zero, and the "
             "work is a compute-bound dense GEMM. Inference splits in two: prefill ingests the whole prompt with the "
             "causal mask; decode generates token by token, a single query against a KV cache of past tokens, and since "
             "no future tokens exist, no mask is needed."),
    ti_cards=[("TRAINING", "parallel · teacher forcing · GEMM",
               [("full sequence passed at once", "b"), ("causal mask as a T × T matrix", "b"),
                ("backward: masked dL/dS = 0", "b"), ("compute-bound dense GEMM", "b")]),
              ("INFERENCE", "prefill vs decode · GEMV",
               [("1. prefill: whole prompt ingested", "b"), ("causal mask across the prompt", "s"),
                ("2. decode: one token at a time", "b"), ("Q [1, d_k] vs KV cache [T, d_k]", "m"),
                ("no future tokens → no mask needed", "h")])],
    bn_aria=("Single head versus multi-head. With one head, the input Z goes through a single bilinear map M = W_Q W_K "
             "transpose and ends in a compromise: blurred, averaged attention. With multiple heads, Z feeds four heads in "
             "parallel, head 1 on the verb or predicate, head 2 on local adjacency and syntax, head 3 on syntactic "
             "agreement, head 4 on long-range coreference; their outputs are concatenated and projected by W O into a "
             "rich, multi-layered representation."),
    bn_l1="single head", bn_map="one bilinear map", bn_comp=("compromise trap", "blurred, averaged attention"),
    bn_l2="multi-head subspaces",
    bn_heads=["head 1: verb / predicate", "head 2: local adjacency, syntax", "head 3: syntactic agreement", "head 4: long-range coreference"],
    bn_out=["rich", "multi-layered", "representation"],
    pl_aria=("Multi-head causal attention in six steps. Input Z is projected to Q, K and V and partitioned into H heads. "
             "For each head, in parallel: 1, scores S_h = Q_h K_h transpose over square root of d_k; 2, masking, S_h plus "
             "M with minus infinity for the future; 3, softmax gives A_h; 4, context, head_h = A_h V_h. Then 5, the heads "
             "are concatenated, and 6, multiplied by W O to give the output O."),
    pl_nodes=[("input Z", None), ("Q, K, V", "projections"), ("split", "into H heads"), ("1. scores", "QₕKₕᵀ / √d_k"),
              ("2. mask", "Sₕ + M (−∞)"), ("3. softmax", "Aₕ"), ("4. context", "Aₕ Vₕ"), ("5. concat", "head₁ … head_H"),
              ("6. Wᴼ", "→ O")],
    pl_leg="per head, in parallel",
    pl_foot="The mask is applied in every head before softmax; heads never see each other until the concat.",
    gp_aria=("How a GPU computes multi-head attention. Input Z of shape B, N, d_model goes through one fused GEMM with "
             "W_qkv of shape d_model by 3 d_model, giving packed QKV of shape B, N, 3, H, d_k. It is split and permuted "
             "to B, H, N, d_k. A batched GEMM computes Q times K transpose, softmax, times V for all heads at once, "
             "giving head outputs of shape B, H, N, d_k. Transpose and view bring it back to B, N, d_model, and a final "
             "GEMM with W_o gives the output of shape B, N, d_model."),
    gp_nodes=[("input Z", "(B, N, d_model)"), ("one fused GEMM", "W_qkv: (d_model, 3·d_model)"),
              ("packed QKV", "(B, N, 3, H, d_k)"), ("split & permute", "(B, H, N, d_k)"),
              ("batched GEMM (bmm)", "Q @ Kᵀ → softmax → @ V"), ("head outputs", "(B, H, N, d_k)"),
              ("transpose(1, 2).view", "(B, N, d_model)"), ("final GEMM W_o", "→ output (B, N, d_model)")],
    gp_foot="Two big GEMMs and one batched attention call: no Python loop over heads anywhere.",
    bf_aria=("Data path through one transformer block, Post-LN layout. Sub-layer 1, communication: token input X becomes "
             "the position-enriched Z, multi-head self-attention produces AttnOut, a skip path carries Z around it, the "
             "two are summed and LayerNorm or RMSNorm gives the intermediate H1. Sub-layer 2, computation: H1 goes "
             "through the feed-forward network W2 sigma of W1 H1 plus b1, plus b2, a skip path carries H1 around it, "
             "the sum is normalized again and gives the final block output H2."),
    bf_sub1="sub-layer 1: communication (attention)", bf_sub2="sub-layer 2: computation (FFN)",
    bf_mha="multi-head attention", bf_ln="LayerNorm", bf_ffn="FFN / MLP", bf_skip="skip path ({})",
    bf_foot=["Post-LN (2017), as drawn: the norm sits after each sum.", "Pre-LN models (Llama, Mistral) move it in front of each sub-layer, off the skip path."],
    sk_aria=("A residual block as two parallel tracks. The input x goes through the processing track F(x) and, at the "
             "same time, along an identity express line that skips it; the two meet at a sum and the output is y = x "
             "plus F(x)."),
    sk_skip="skip / identity track: the express line", sk_proc="processing track",
    sk_foot="However noisy F(x) gets, x always has an untouched path to the output.",
    kv_aria=("The FFN as an associative key-value memory. The token's vector is read from the residual stream; W1 "
             "projection matches it against keys; the activation acts as a threshold filter so only matched keys fire, "
             "in our walkthrough neurons 1, 2 and 4 of six; W2 projection injects the corresponding values, and the "
             "fact is added back into the residual stream."),
    kv_n=[("token", "input"), ("W₁ projection", "key matching"), ("activation", "threshold filter"), ("W₂ projection", "value injection")],
    kv_fire="our walkthrough: 3 of 6 fire", kv_stream="residual stream", kv_add="fact added",
    kv_foot="Keys decide which memories wake up; values decide what they write back.",
)

TR = dict(
    bd_aria=("Üç token üzerinde çift yönlü ve nedensel dikkat. Maskesiz, tanrı gözü: x1, x2 ve x3 birbirine iki yönde "
             "bağlıdır, her token diğer hepsini görür. Nedensel, zamanın oku: bilgi yalnızca ileri akar; x1 kendine, x2 "
             "x1'e ve kendine, x3 x1'e, x2'ye ve kendine bakar."),
    bd_panels=[("Çift yönlü", "maskesiz: tanrı gözü"), ("Nedensel", "maskeli: zamanın oku")],
    bd_sees=[["hepsini görür", "hepsini görür", "hepsini görür"], ["x₁'i görür", "x₁, x₂'yi görür", "x₁–x₃'ü görür"]],
    bd_foot=["her token bütün diziyi görür", "her token yalnızca kendini ve geçmişi görür"],
    pg_aria=("Sayfaları yapıştırılmış bir cinayet romanı. köpek romanın 1. sayfasındadır, katilin kim olduğunu asla "
             "bilemez. kediyi 50. sayfadadır: 1. sayfayı hatırlar, sonrasını göremez. kovaladı son sayfadadır ve bütün "
             "hikâyeye hâkimdir."),
    pg_head="her token kitabı yalnızca kendi sayfasına kadar okur",
    pg_rows=[("köpek", 0.03, "1. sayfa", "katilin kim olduğunu bilemez"), ("kediyi", 0.45, "50. sayfa", "1. sayfayı hatırlar, gerisi kapalı"),
             ("kovaladı", 1.0, "son sayfa", "bütün hikâyeye hâkim")],
    pg_sealed="kapalı",
    ti_aria=("Eğitimde ve çıkarımda nedensel dikkat. Eğitim, teacher forcing ile paralel: bütün dizi GPU'ya tek seferde "
             "verilir, causal maske T'ye T matris olarak uygulanır, maskeli gradyanlar sıfırdır ve iş compute-bound "
             "yoğun bir GEMM'dir. Çıkarım ikiye ayrılır: prefill promptun tamamını causal maskeyle işler; decode token "
             "token üretir, tek bir sorgu geçmiş token'ların KV cache'ine bakar ve gelecek olmadığı için maske gerekmez."),
    ti_cards=[("TRAINING", "paralel · teacher forcing · GEMM",
               [("tüm dizi tek seferde verilir", "b"), ("causal maske T × T matris", "b"),
                ("backward: maskeli dL/dS = 0", "b"), ("compute-bound yoğun GEMM", "b")]),
              ("INFERENCE", "prefill vs decode · GEMV",
               [("1. prefill: promptun tamamı işlenir", "b"), ("causal maske prompta uygulanır", "s"),
                ("2. decode: token token üretim", "b"), ("Q [1, d_k] vs KV cache [T, d_k]", "m"),
                ("gelecek yok → maske gerekmez", "h")])],
    bn_aria=("Tek kafa ile multi-head. Tek kafada girdi Z, tek bir çift doğrusal eşleme M = W_Q W_K devrikten geçer ve "
             "bir uzlaşmada biter: bulanık, ortalama dikkat. Çok kafada Z dört kafayı paralel besler: 1. kafa fiil veya "
             "eylem, 2. kafa yerel komşuluk ve sıralama, 3. kafa sözdizimsel uyum, 4. kafa uzun menzilli zamir gönderimi; "
             "çıktıları birleştirilir ve W O ile zengin, katmanlı bir temsile izdüşürülür."),
    bn_l1="tek kafa", bn_map="tek çift doğrusal eşleme", bn_comp=("uzlaşma çıkmazı", "bulanık, ortalama dikkat"),
    bn_l2="multi-head alt uzayları",
    bn_heads=["1. kafa: fiil / eylem odağı", "2. kafa: yerel komşuluk", "3. kafa: sözdizimsel uyum", "4. kafa: uzun menzilli zamir"],
    bn_out=["zengin", "katmanlı", "nihai temsil"],
    pl_aria=("Altı adımda multi-head causal attention. Girdi Z, Q, K ve V'ye izdüşürülür ve H kafaya ayrılır. Her kafa "
             "için paralel olarak: 1, skorlar S_h = Q_h K_h devrik bölü karekök d_k; 2, maskeleme, S_h artı geleceğe "
             "eksi sonsuz koyan M; 3, softmax A_h'yi verir; 4, bağlam, head_h = A_h V_h. Sonra 5, kafalar birleştirilir "
             "ve 6, W O ile çarpılarak çıktı O elde edilir."),
    pl_nodes=[("girdi Z", None), ("Q, K, V", "izdüşümleri"), ("ayrıştır", "H kafaya"), ("1. skorlar", "QₕKₕᵀ / √d_k"),
              ("2. maske", "Sₕ + M (−∞)"), ("3. softmax", "Aₕ"), ("4. bağlam", "Aₕ Vₕ"), ("5. birleştir", "head₁ … head_H"),
              ("6. Wᴼ", "→ O")],
    pl_leg="kafa başına, paralel",
    pl_foot="Maske her kafada softmax'tan önce uygulanır; kafalar birleştirmeye kadar birbirini görmez.",
    gp_aria=("GPU'nun multi-head attention'ı nasıl hesapladığı. B, N, d_model boyutlu girdi Z, d_model'e 3 d_model "
             "boyutlu W_qkv ile tek bir kaynaşık GEMM'den geçer ve B, N, 3, H, d_k boyutlu paketlenmiş QKV olur. "
             "Ayrıştırılıp B, H, N, d_k'ye transpoze edilir. Yığın GEMM bütün kafalar için Q çarpı K devrik, softmax, "
             "çarpı V'yi tek seferde hesaplar ve B, H, N, d_k boyutlu kafa çıktıları verir. Transpose ve view bunu B, "
             "N, d_model'e geri getirir, W_o ile son GEMM de B, N, d_model boyutlu çıktıyı verir."),
    gp_nodes=[("girdi Z", "(B, N, d_model)"), ("tek kaynaşık GEMM", "W_qkv: (d_model, 3·d_model)"),
              ("paketlenmiş QKV", "(B, N, 3, H, d_k)"), ("ayrıştır ve transpoze", "(B, H, N, d_k)"),
              ("yığın GEMM (bmm)", "Q @ Kᵀ → softmax → @ V"), ("kafa çıktıları", "(B, H, N, d_k)"),
              ("transpose(1, 2).view", "(B, N, d_model)"), ("nihai GEMM W_o", "→ çıktı (B, N, d_model)")],
    gp_foot="İki büyük GEMM ve tek bir yığın attention çağrısı: kafalar üzerinde hiçbir Python döngüsü yok.",
    bf_aria=("Tek bir transformer bloğundan geçen veri yolu, Post-LN düzeni. 1. alt katman, iletişim: token girdisi X "
             "pozisyon bilgisi eklenmiş Z olur, multi-head self-attention AttnOut'u üretir, bir atlama yolu Z'yi onun "
             "etrafından taşır, ikisi toplanır ve LayerNorm ya da RMSNorm ara temsil H1'i verir. 2. alt katman, "
             "hesaplama: H1 ileri beslemeli ağdan, W2 sigma W1 H1 artı b1, artı b2'den geçer, bir atlama yolu H1'i "
             "etrafından taşır, toplam yeniden normalize edilir ve nihai blok çıktısı H2'yi verir."),
    bf_sub1="1. alt katman: iletişim (attention)", bf_sub2="2. alt katman: hesaplama (FFN)",
    bf_mha="multi-head attention", bf_ln="LayerNorm", bf_ffn="FFN / MLP", bf_skip="atlama yolu ({})",
    bf_foot=["Çizildiği gibi Post-LN (2017): norm her toplamadan sonra gelir.", "Pre-LN modeller (Llama, Mistral) normu her alt katmanın önüne, atlama yolunun dışına taşır."],
    sk_aria=("İki paralel hat olarak residual blok. Girdi x hem işlem hattı F(x)'ten geçer hem de onu atlayan bir "
             "özdeşlik ekspres hattından ilerler; iki hat bir toplamada buluşur ve çıktı y = x artı F(x) olur."),
    sk_skip="atlama / özdeşlik hattı: ekspres yol", sk_proc="işlem yolu",
    sk_foot="F(x) ne kadar gürültülü olursa olsun, x'in çıkışa dokunulmamış bir yolu her zaman vardır.",
    kv_aria=("Çağrışımsal anahtar-değer belleği olarak FFN. Token'ın vektörü residual akıştan okunur; W1 çarpımı onu "
             "anahtarlarla eşleştirir; aktivasyon bir eşik filtresi gibi çalışır ve yalnızca eşleşen anahtarlar yanar, "
             "yürüyüşümüzde altı nörondan 1, 2 ve 4; W2 çarpımı karşılık gelen değerleri enjekte eder ve bilgi residual "
             "akışa geri eklenir."),
    kv_n=[("token", "girdi"), ("W₁ çarpımı", "anahtar taraması"), ("aktivasyon", "eşik filtresi"), ("W₂ çarpımı", "değer enjeksiyonu")],
    kv_fire="yürüyüşümüz: 6'dan 3'ü yanar", kv_stream="residual akış", kv_add="bilgi eklenir",
    kv_foot="Anahtarlar hangi anının uyanacağına, değerler geri ne yazılacağına karar verir.",
)

out = {lang: dict(bidir=bidir(L), pages=pages(L), trainfer=trainfer(L), bottleneck=bottleneck(L), pipeline=pipeline(L),
                  gpu=gpu(L), blockflow=blockflow(L), skipc=skipc(L), ffnkv=ffnkv(L)) for lang, L in (("en", EN), ("tr", TR))}
for d in out.values():
    for s in d.values():
        assert_no_blank_lines(s)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
