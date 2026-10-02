"""Batch 5: optimization algorithms (3) + initialization & normalization (7) -> SVG."""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))
from svgkit import *

SUB = "fill:var(--c-text-mute);font-size:11px"
WARN, OK, BAD = "var(--c-warn)", "var(--c-success)", "var(--c-danger)"
MONO12 = "fill:var(--c-text);font-size:12px;font-family:var(--font-mono)"
SANS = "fill:var(--c-text);font-size:12.5px"


def mbox(x, y, w, h, label, col=None, mono=True, dashed=False):
    st = inner(col) if col else BOX
    r = box(x, y, w, h, st, rx=6)
    if dashed:
        r += f'\n<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" style="fill:none;stroke:{col};stroke-width:1.2;stroke-dasharray:4 3"/>'
    return r + "\n" + text(x + w / 2, y + h / 2 + 4.5, label, MONO12 if mono else SANS, "middle")


def ln(d, col="var(--c-text-mute)", w=1.5, dash=False):
    return f'<path d="{d}" style="fill:none;stroke:{col};stroke-width:{w}{";stroke-dasharray:5 4" if dash else ""}"/>'


# ================================================================== optimization
def ravine(L):
    o = [svg_open(560, 330, L["rv_aria"]), "<defs>", marker("rv-arr"), marker("rv-r", BAD), "</defs>"]
    o.append(text(16, 20, L["rv_title"], title_style(BAD)))
    cx, cy = 230, 100
    for rx, ry in ((200, 62), (140, 43), (80, 25), (24, 7.5)):
        o.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>')
    # zig-zag SGD path: large transverse bounces, tiny forward progress
    pts, x, s = [], 60, 1
    for k in range(9):
        amp = 52 - k * 2.5
        pts.append(f"{x:.0f} {cy - s * amp:.0f}")
        x += 13
        s = -s
    o.append(f'<path d="M{" L".join(pts)}" marker-end="url(#rv-r)" style="fill:none;stroke:{BAD};stroke-width:1.6"/>')
    o.append(f'<circle cx="{cx}" cy="{cy}" r="3" style="fill:var(--c-text)"/>')
    o.append(text(cx + 8, cy + 4, L["rv_min"], SUB))
    # direction labels
    o.append(f'<path d="M448 60 V140" marker-start="url(#rv-arr)" marker-end="url(#rv-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>')
    o.append(text(458, 84, L["rv_wall"][0], "fill:var(--c-text);font-size:12px;font-weight:600"))
    for j, s_ in enumerate(L["rv_wall"][1:]):
        o.append(text(458, 99 + j * 14, s_, SUB))
    o.append(f'<path d="M150 178 H310" marker-start="url(#rv-arr)" marker-end="url(#rv-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>')
    o.append(text(230, 196, L["rv_floor"], SUB, "middle"))
    o.append(box(16, 206, 528, 30, inner(BAD), rx=6))
    o.append(text(280, 225.5, L["rv_sgd"], "fill:var(--c-text);font-size:12px", "middle"))
    o.append(text(16, 258, L["rv_sol"], title_style(OK)))
    xs = [16, 196, 376]
    for x, (t, a, b) in zip(xs, L["rv_cards"]):
        o.append(node(x, 268, 168, 56, t, [a, b], OK))
    o.append(svg_close())
    return "\n".join(o)


def adamw(L):
    o = [svg_open(560, 230, L["aw_aria"]), "<defs>", marker("aw-arr"), "</defs>"]
    for p, (title, verdict) in enumerate(L["aw_rows"]):
        y0 = 8 + p * 112
        col = BAD if p == 0 else OK
        o.append(box(16, y0, 528, 100 if p else 96, FRAME))
        o.append(text(30, y0 + 21, title, title_style(col)))
        o.append(text(530, y0 + 21, verdict, f"fill:{col};font-size:12px;font-weight:600", "end"))
    # Adam + L2
    o.append(mbox(30, 40, 150, 34, "g + λ·w", BAD))
    o.append(mbox(206, 40, 150, 34, L["aw_mom"], None, mono=False))
    o.append(mbox(382, 40, 148, 34, L["aw_div"], None, mono=False))
    o.append(arrow(180, 57, 204, 57, "aw-arr"))
    o.append(arrow(356, 57, 380, 57, "aw-arr"))
    o.append(text(30, 92, L["aw_bad"], SUB))
    # AdamW
    o.append(mbox(30, 152, 150, 30, "g", None))
    o.append(mbox(206, 152, 150, 30, L["aw_mom2"], None, mono=False))
    o.append(mbox(382, 152, 148, 30, L["aw_step"], None, mono=False))
    o.append(arrow(180, 167, 204, 167, "aw-arr"))
    o.append(arrow(356, 167, 380, 167, "aw-arr"))
    o.append(mbox(30, 190, 150, 30, "−η·λ·w", OK))
    o.append(mbox(382, 190, 148, 30, L["aw_sub"], OK, mono=False))
    o.append(arrow(180, 205, 380, 205, "aw-arr", OK, dash=True))
    o.append(text(280, 199, L["aw_bypass"], SUB, "middle"))
    o.append(svg_close())
    return "\n".join(o)


def muon(L):
    o = [svg_open(560, 270, L["mu_aria"]), "<defs>", marker("mu-arr"), "</defs>"]
    cols = [None, ACC, GEN, OK, None]
    for k, ((t, f), c) in enumerate(zip(L["mu_steps"], cols)):
        y = 8 + k * 52
        o.append(box(16, y, 528, 40, inner(c) if c else BOX))
        o.append(text(30, y + 25, t, "fill:var(--c-text);font-size:13px"))
        o.append(text(530, y + 25, f, "fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)", "end"))
        if k < 4:
            o.append(arrow(60, y + 40, 60, y + 50, "mu-arr"))
    o.append(svg_close())
    return "\n".join(o)


# ================================================================== initialization & normalization
def chain(L):
    o = [svg_open(560, 110, L["ch_aria"]), "<defs>", marker("ch-arr"), "</defs>"]
    items = [("x₀ ~ N(0,1)", "t", 92), ("W₁", "w", 40), ("x₁", "t", 34), ("W₂", "w", 40), ("…", "d", 22),
             ("W_L", "w", 44), ("x_L", "t", 40)]
    gap = (528 - sum(w for _, _, w in items)) / (len(items) - 1)
    x = 16
    for k, (lab, kind, w) in enumerate(items):
        if kind == "d":
            o.append(text(x + w / 2, 46, lab, "fill:var(--c-text-mute);font-size:16px", "middle"))
        else:
            o.append(mbox(x, 24, w, 30, lab, ACC if kind == "t" else WARN))
        if k < len(items) - 1:
            o.append(arrow(x + w + 2, 39, x + w + gap - 2, 39, "ch-arr"))
        x += w + gap
    o.append(text(16, 80, L["ch_n1"], MUTE))
    o.append(text(16, 98, L["ch_n2"], SUB))
    o.append(svg_close())
    return "\n".join(o)


def sym(L):
    o = [svg_open(560, 190, L["sy_aria"]), "<defs>", marker("sy-arr"), "</defs>"]
    o.append(node(16, 66, 90, 44, L["sy_in"], None, ACC))
    for k in range(3):
        y = 12 + k * 54
        o.append(path(f"M106 88 H124 V{y + 21} H150", "sy-arr"))
        o.append(box(152, y, 240, 42, inner(WARN), rx=6))
        o.append(text(166, y + 18, L["sy_n"].format(k + 1), "fill:var(--c-text);font-size:12.5px;font-weight:600"))
        o.append(text(166, y + 34, f"z{'₁₂₃'[k]}, a{'₁₂₃'[k]}, ∂L/∂z{'₁₂₃'[k]}", "fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)"))
        o.append(text(380, y + 26, "=", "fill:var(--c-text-mute);font-size:14px", "end"))
    o.append(ln("M400 12 Q410 12 410 24 V76 Q410 88 420 88 Q410 88 410 100 V152 Q410 164 400 164"))
    for j, s in enumerate(L["sy_res"]):
        o.append(text(428, 76 + j * 16, s, "fill:var(--c-text);font-size:12px" if j < 2 else SUB))
    o.append(text(16, 182, L["sy_foot"], SUB))
    o.append(svg_close())
    return "\n".join(o)


def resid(L):
    o = [svg_open(560, 184, L["rs_aria"]), "<defs>", marker("rs-arr"), marker("rs-g", GEN), "</defs>"]
    xs = [(16, "x₀", "Var = 1"), (150, "x₁", "Var = 2"), (300, "…", ""), (444, "x_L", "Var = 1 + L")]
    adds = [112, 262, 412]
    for x, lab, v in xs:
        if lab == "…":
            o.append(text(x + 6, 48, "…", "fill:var(--c-text-mute);font-size:16px", "middle"))
            continue
        w = 100 if x == 444 else 76
        o.append(mbox(x, 22, w, 34, lab, ACC))
        o.append(text(x + w / 2, 72, v, "fill:var(--c-text);font-size:11.5px", "middle"))
    for k, ax in enumerate(adds):
        o.append(f'<circle cx="{ax}" cy="39" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>')
        o.append(text(ax, 44, "+", "fill:var(--c-text);font-size:15px", "middle"))
        o.append(mbox(ax - 46, 106, 92, 30, ["F(x₀)", "F(x₁)", "F(x_L−1)"][k], GEN))
        o.append(text(ax, 152, "Var = 1.0" if L is EN else "Var = 1,0", SUB, "middle"))
        o.append(arrow(ax, 106, ax, 52, "rs-g", GEN))
    o.append(arrow(92, 39, 99, 39, "rs-arr"))
    o.append(arrow(123, 39, 148, 39, "rs-arr"))
    o.append(arrow(226, 39, 249, 39, "rs-arr"))
    o.append(arrow(273, 39, 296, 39, "rs-arr"))
    o.append(arrow(314, 39, 399, 39, "rs-arr"))
    o.append(arrow(423, 39, 442, 39, "rs-arr"))
    o.append(text(16, 178, L["rs_note"], SUB))
    o.append(svg_close())
    return "\n".join(o)


def lnrms(L):
    o = [svg_open(560, 210, L["lr_aria"]), "<defs>", marker("lr-arr"), "</defs>"]
    xs = [60 + k * 98 for k in range(5)]
    for p, (title, nodes) in enumerate(L["lr_rows"]):
        y0 = 30 + p * 100
        col = WARN if p == 0 else OK
        o.append(text(16, y0 - 10, title, title_style(col)))
        o.append(mbox(16, y0 + 8, 30, 36, "x", ACC))
        if p == 0:
            for k, (x, (t, s)) in enumerate(zip(xs, nodes)):
                o.append(node(x, y0, 88, 52, t, s, WARN if k < 3 else None))
                o.append(arrow((46 if k == 0 else x - 10), y0 + 26, x - 2, y0 + 26, "lr-arr"))
        else:
            spans = [(xs[0], xs[2] + 88), (xs[3], xs[3] + 88), (xs[4], xs[4] + 88)]
            for k, ((a, b), (t, s)) in enumerate(zip(spans, nodes)):
                o.append(node(a, y0, b - a, 52, t, s, OK if k == 0 else None, mono=(k == 0)))
                o.append(arrow((46 if k == 0 else a - 10), y0 + 26, a - 2, y0 + 26, "lr-arr"))
    o.append(text(16, 202, L["lr_foot"], SUB))
    o.append(svg_close())
    return "\n".join(o)


def prepost(L):
    o = [svg_open(560, 318, L["pp_aria"]), "<defs>", marker("pp-arr"), marker("pp-g", GEN), "</defs>"]
    for p, (title, sub) in enumerate(L["pp_panels"]):
        x0 = 16 + p * 272
        col = WARN if p == 0 else OK
        o.append(box(x0, 8, 256, 302, FRAME))
        o.append(text(x0 + 14, 30, title, title_style(col)))
        o.append(text(x0 + 14, 47, sub, SUB))
        mx = x0 + 70          # main stream (residual highway)
        bx = x0 + 176         # branch column
        o.append(mbox(mx - 34, 60, 68, 28, "x_l", ACC))
        o.append(f'<circle cx="{mx}" cy="214" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>')
        o.append(text(mx, 219, "+", "fill:var(--c-text);font-size:15px", "middle"))
        o.append(ln(f"M{mx} 102 H{bx} V112"))
        o.append(arrow(bx, 112, bx, 114, "pp-arr"))
        if p == 0:
            o.append(arrow(mx, 88, mx, 201, "pp-g", GEN, ))
            o.append(mbox(bx - 54, 116, 108, 32, "SubLayer", None, mono=False))
            o.append(path(f"M{bx} 148 V214 H{mx + 13}", "pp-arr"))
            o.append(mbox(mx - 50, 238, 100, 30, "LayerNorm", WARN, mono=False))
            o.append(arrow(mx, 225, mx, 236, "pp-arr"))
            o.append(mbox(mx - 34, 276, 68, 26, "x_l+1", ACC))
            o.append(arrow(mx, 268, mx, 274, "pp-arr"))
            o.append(text(bx - 48, 262, L["pp_warn"][0], SUB))
            o.append(text(bx - 48, 276, L["pp_warn"][1], SUB))
        else:
            o.append(arrow(mx, 88, mx, 201, "pp-g", GEN))
            o.append(mbox(bx - 54, 116, 108, 32, L["pp_norm"], OK, mono=False))
            o.append(mbox(bx - 54, 162, 108, 32, "SubLayer", None, mono=False))
            o.append(arrow(bx, 148, bx, 160, "pp-arr"))
            o.append(path(f"M{bx} 194 V214 H{mx + 13}", "pp-arr"))
            o.append(mbox(mx - 34, 250, 68, 26, "x_l+1", ACC))
            o.append(arrow(mx, 225, mx, 248, "pp-arr"))
            o.append(text(bx - 48, 262, L["pp_ok"][0], SUB))
            o.append(text(bx - 48, 276, L["pp_ok"][1], SUB))
        o.append(text(mx - 8, 160, L["pp_hw"], f"fill:{GEN};font-size:11px", "end", f'transform="rotate(-90 {mx - 8} 160)"'))
    o.append(svg_close())
    return "\n".join(o)


def hbm_bar(o, y, label):
    o.append(box(16, y, 528, 30, "fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2", rx=6))
    o.append(text(280, y + 19.5, label, "fill:var(--c-text);font-size:12px;font-weight:600", "middle"))


def unfused(L):
    o = [svg_open(560, 190, L["uf_aria"]), "<defs>", marker("uf-arr"), marker("uf-w", WARN), "</defs>"]
    o.append(text(16, 18, L["uf_sm"], SUB))
    xs = [16 + k * 108 for k in range(5)]
    for k, (x, (t, s)) in enumerate(zip(xs, L["uf_k"])):
        o.append(node(x, 26, 96, 52, t, s, GEN if k == 4 else None))
    hbm_bar(o, 130, L["uf_hbm"])
    for k in range(4):
        a, b = xs[k] + 70, xs[k + 1] + 26
        o.append(arrow(a, 78, a, 128, "uf-w", WARN))
        o.append(arrow(b, 130, b, 80, "uf-w", WARN))
        o.append(text(a + 5, 98, L["uf_w"], SUB))
        o.append(text(b - 5, 118, L["uf_r"], SUB, "end"))
    o.append(text(16, 182, L["uf_foot"], "fill:var(--c-text);font-size:12px"))
    o.append(svg_close())
    return "\n".join(o)


def fused(L):
    o = [svg_open(560, 236, L["fu_aria"]), "<defs>", marker("fu-arr"), marker("fu-w", WARN), "</defs>"]
    o.append(box(16, 8, 528, 150, inner(ACC)))
    o.append(text(30, 28, L["fu_sm"], "fill:var(--c-text);font-size:12px;font-weight:600"))
    xs = [32, 210, 388]
    for k, (x, t) in enumerate(zip(xs, L["fu_top"])):
        o.append(mbox(x, 40, 140, 34, t, None, mono=False))
        if k < 2:
            o.append(arrow(x + 140, 57, x + 176, 57, "fu-arr"))
    o.append(path("M458 74 V115 H352", "fu-arr"))
    for k, (x, t) in enumerate(zip((32, 210), L["fu_bot"])):
        o.append(mbox(x, 98, 140, 34, t, OK if k == 0 else None, mono=False))
    o.append(arrow(210, 115, 174, 115, "fu-arr"))
    o.append(text(530, 150, L["fu_reg"], SUB, "end"))
    hbm_bar(o, 198, L["fu_hbm"])
    o.append(arrow(102, 132, 102, 196, "fu-w", WARN))
    o.append(text(112, 176, L["fu_one"], "fill:var(--c-text);font-size:12px"))
    o.append(svg_close())
    return "\n".join(o)


EN = dict(
    rv_aria=("The ill-conditioned ravine, with condition number kappa far above 10 to the 4. Along the wall direction, "
             "lambda max, curvature is steep and plain SGD bounces in violent transverse oscillations. Along the floor "
             "direction, lambda min, curvature is flat and forward progress is near zero. SGD is stuck: with a step size "
             "above 2 over lambda max it diverges; below it, it crawls along the floor. Three algorithmic fixes: momentum "
             "cancels the oscillations, since alternating gradients sum to about zero, and accumulates velocity along the "
             "floor; AdamW normalizes the step size per coordinate with an RMS; Muon orthogonalizes the whole 2D update "
             "matrix with Newton-Schulz iterations."),
    rv_title="The ravine dilemma (κ ≫ 10⁴)",
    rv_min="minimum",
    rv_wall=["wall · λ_max", "steep curvature:", "violent oscillations"],
    rv_floor="floor · λ_min · flat curvature: near-zero forward progress",
    rv_sgd="SGD: η > 2/λ_max → diverges;  η < 2/λ_max → crawls along the floor",
    rv_sol="Algorithmic solutions",
    rv_cards=[("Momentum", "cancels oscillations", "(Σ ±g ≈ 0), builds speed"),
              ("AdamW", "normalizes step size", "per-coordinate RMS"),
              ("Muon", "orthogonalizes the 2D", "matrix (Newton–Schulz)")],
    aw_aria=("Adam with L2 regularization versus AdamW. In Adam with L2, the decay term lambda times w is added to the "
             "gradient, fed into the first and second moments, and divided by the square root of v, so weights with large "
             "gradient history are barely decayed. In AdamW, decoupled weight decay, the plain gradient g goes through "
             "the moments to a standard Adam step, and the weight decay, minus eta times lambda times w, bypasses the "
             "moments and is subtracted from the weights directly."),
    aw_rows=[("Adam + L2 regularization", "decay distorted"), ("AdamW (decoupled weight decay)", "decay intact")],
    aw_mom="1st & 2nd moments", aw_div="divided by √v",
    aw_bad="the decay term is rescaled per coordinate along with the gradient",
    aw_mom2="moments m, v", aw_step="standard step",
    aw_sub="subtract from w", aw_bypass="bypasses the moments",
    mu_aria=("Muon's update in five steps. Start from the raw momentum matrix G of a 2D hidden weight. Divide by its "
             "Frobenius norm: X0 equals G over the norm of G. Run five Newton-Schulz iterations: X k plus 1 equals one "
             "half X k times 3 I minus X k transpose X k. The result is an orthogonal update matrix: X transpose X is "
             "approximately the identity, all singular values equal 1. Update the weights: W becomes W minus eta X, a "
             "uniform spectral step."),
    mu_steps=[("Raw momentum G (2D weight)", "G"), ("Normalize energy", "X₀ = G / ‖G‖_F"),
              ("5× Newton–Schulz", "X_k+1 = ½·X_k(3I − X_kᵀX_k)"), ("Orthogonal update", "XᵀX ≈ I (all σᵢ = 1)"),
              ("Uniform spectral step", "W ← W − η·X")],
    ch_aria=("A deep linear chain at initialization. x0, drawn from a standard normal, passes through W1 to give x1, "
             "through W2, and so on through W L to give x L. Every weight matrix has variance sigma squared, so each "
             "layer multiplies the signal's variance by the same factor, and over L layers that factor compounds."),
    ch_n1="Every W has Var(W) = σ²; each layer rescales the variance by the same factor.",
    ch_n2="Over L layers the factor compounds: it explodes above 1 and vanishes below 1.",
    sy_aria=("Why symmetric initialization fails. The same input x reaches neurons 1, 2 and 3. With identical weights, "
             "their pre-activations z, activations a and gradients dL/dz are identical, so every update keeps them "
             "identical. The network's rank collapses to a single effective neuron."),
    sy_in="input x",
    sy_n="Neuron {}",
    sy_res=["all identical:", "rank collapses to", "one effective neuron"],
    sy_foot="Same weights → same outputs → same gradients → same updates, forever.",
    rs_aria=("Variance growth along the residual stream. x0 has variance 1. Each block adds F of x, whose variance is "
             "1.0, so x1 has variance 2, and after L blocks x L has variance 1 plus L."),
    rs_note="each block adds a variance-1 branch",
    lr_aria=("LayerNorm versus RMSNorm. LayerNorm computes the mean mu in pass 1 over x, centers x minus mu in pass 2, "
             "computes the variance sigma squared in pass 3, then normalizes and applies the affine gamma and beta. "
             "RMSNorm computes the root mean square of x in a single pass, then normalizes and applies gamma only."),
    lr_rows=[("LayerNorm", [("mean μ", "pass 1"), ("x − μ", "pass 2"), ("variance σ²", "pass 3"), ("normalize", None), ("affine", "γ, β")]),
             ("RMSNorm", [("RMS = √(Σx²/D)", "one pass over x"), ("normalize", None), ("affine", "γ only")])],
    lr_foot="Three reads of x become one, and the mean and β disappear.",
    pp_aria=("Post-LN versus Pre-LN. In Post-LN, the original 2017 Transformer, x l goes both into the sublayer and along "
             "the residual path; the two are added, and LayerNorm sits on the main path after the addition, producing x "
             "l plus 1, so every gradient must pass through a normalization. In Pre-LN, the modern standard from 2019 to "
             "2023, the branch first applies LayerNorm or RMSNorm and then the sublayer; the result is added to the "
             "untouched residual path, giving x l plus 1, so the residual highway stays clean."),
    pp_panels=[("Post-LN", "original Transformer, 2017"), ("Pre-LN", "modern standard, 2019–2023")],
    pp_norm="LN / RMSNorm",
    pp_hw="residual",
    pp_warn=["norm sits on the", "highway itself"],
    pp_ok=["highway stays", "untouched"],
    uf_aria=("Unfused normalization. The residual addition writes its result to HBM; the mean reduction reads it back "
             "and writes; the variance reduction reads and writes; standardization plus affine reads and writes; the "
             "QKV GEMM finally reads the normalized tensor. Four HBM round trips for one normalization."),
    uf_sm="separate kernels",
    uf_k=[("residual", "addition"), ("mean", "reduction"), ("variance", "reduction"), ("standardize", "+ affine"), ("QKV GEMM", None)],
    uf_hbm="HBM (off-chip)",
    uf_w="write", uf_r="read",
    uf_foot="4 HBM round trips for a single normalization.",
    fu_aria=("Fused RMSNorm kernel. Inside one GPU streaming multiprocessor, in SRAM and registers: the residual input is "
             "summed in registers, a warp-shuffle reduction computes the RMS, the result is multiplied by gamma, and the "
             "scaled output is ready. One single HBM write sends it directly into the GEMM input buffer."),
    fu_sm="GPU streaming multiprocessor (SM) · SRAM / registers",
    fu_top=["residual in", "in-register sum", "warp-shuffle RMS"],
    fu_bot=["scaled output", "multiply by γ"],
    fu_reg="nothing leaves the chip until the end",
    fu_hbm="HBM · GEMM input buffer",
    fu_one="single HBM write",
)

TR = dict(
    rv_aria=("Kötü koşullu vadi, koşul sayısı kappa 10 üzeri 4'ün çok üstünde. Yamaç yönünde, lambda max, eğrilik diktir "
             "ve düz SGD şiddetli enine salınımlarla sekip durur. Taban yönünde, lambda min, eğrilik düzdür ve ileri "
             "ilerleme neredeyse sıfırdır. SGD sıkışmıştır: adım boyu 2 bölü lambda max'ın üstündeyse patlar, altındaysa "
             "taban boyunca sürünür. Üç algoritmik çözüm: momentum salınımları sönümler, çünkü yön değiştiren gradyanların "
             "toplamı sıfıra yakındır, ve taban boyunca hız biriktirir; AdamW adım boyutunu koordinat bazlı RMS ile "
             "eşitler; Muon tüm 2D güncelleme matrisini Newton-Schulz yinelemeleriyle ortogonalize eder."),
    rv_title="Kötü koşullu vadi çıkmazı (κ ≫ 10⁴)",
    rv_min="minimum",
    rv_wall=["yamaç · λ_max", "dik eğrilik:", "şiddetli salınım"],
    rv_floor="taban · λ_min · düz eğrilik: neredeyse sıfır ilerleme",
    rv_sgd="SGD: η > 2/λ_max → patlar;  η < 2/λ_max → taban boyunca sürünür",
    rv_sol="Algoritmik çözümler",
    rv_cards=[("Momentum", "salınımları sönümler", "(Σ ±g ≈ 0), hız biriktirir"),
              ("AdamW", "adım boyunu eşitler", "koordinat bazlı RMS"),
              ("Muon", "2D matrisi ortogonalize", "eder (Newton–Schulz)")],
    aw_aria=("L2 regülarizasyonlu Adam ile AdamW. L2'li Adam'da sönümleme terimi lambda çarpı w gradyana eklenir, birinci "
             "ve ikinci momentlere girer ve v'nin kareköküne bölünür; büyük gradyan geçmişi olan ağırlıklar neredeyse hiç "
             "sönümlenmez. Ayrıştırılmış sönümlemeli AdamW'de saf gradyan g momentlerden geçip standart Adam adımına "
             "gider; ağırlık sönümleme, eksi eta çarpı lambda çarpı w, momentleri atlar ve ağırlıklardan doğrudan "
             "çıkarılır."),
    aw_rows=[("Adam + L2 regülarizasyonu", "sönümleme BOZULUR"), ("AdamW (ayrıştırılmış sönümleme)", "sönümleme KORUNUR")],
    aw_mom="momentlere girer", aw_div="√v ile bölünür",
    aw_bad="sönümleme terimi gradyanla birlikte koordinat bazında yeniden ölçeklenir",
    aw_mom2="momentler m, v", aw_step="standart adım",
    aw_sub="w'den çıkar", aw_bypass="momentleri atlar",
    mu_aria=("Muon güncellemesi beş adımda. 2D gizli bir ağırlığın ham momentum matrisi G'den başla. Frobenius normuna böl: "
             "X0, G bölü G'nin normu. Beş Newton-Schulz yinelemesi çalıştır: X k artı 1, yarım çarpı X k çarpı 3 I eksi X k "
             "devrik X k. Sonuç ortogonal bir güncelleme matrisidir: X devrik X yaklaşık birim matristir, tüm tekil "
             "değerler 1'dir. Ağırlığı güncelle: W, W eksi eta X olur; eşit bir spektral adım."),
    mu_steps=[("Ham momentum G (2D ağırlık)", "G"), ("Enerjiyi normalle", "X₀ = G / ‖G‖_F"),
              ("5× Newton–Schulz", "X_k+1 = ½·X_k(3I − X_kᵀX_k)"), ("Ortogonal güncelleme", "XᵀX ≈ I (tüm σᵢ = 1)"),
              ("Eşit spektral adım", "W ← W − η·X")],
    ch_aria=("Başlatma anında derin, doğrusal bir zincir. Standart normalden çekilen x0, W1'den geçip x1'i, W2'den geçip "
             "sonrakini verir ve W L'den geçip x L'ye ulaşır. Her ağırlık matrisinin varyansı sigma karedir; her katman "
             "sinyalin varyansını aynı katsayıyla çarpar ve L katmanda bu katsayı katlanarak büyür."),
    ch_n1="Her W için Var(W) = σ²; her katman varyansı aynı katsayıyla ölçekler.",
    ch_n2="L katmanda katsayı katlanır: 1'in üstündeyse patlar, altındaysa söner.",
    sy_aria=("Simetrik başlatma neden çöker. Aynı girdi x 1, 2 ve 3 numaralı nöronlara ulaşır. Ağırlıklar aynıysa z "
             "ön-aktivasyonları, a aktivasyonları ve dL/dz gradyanları da aynıdır; her güncelleme onları aynı tutar. "
             "Ağın rankı tek bir etkin nörona çöker."),
    sy_in="girdi x",
    sy_n="Nöron {}",
    sy_res=["tamamen özdeş:", "rank tek etkin", "nörona çöker"],
    sy_foot="Aynı ağırlık → aynı çıktı → aynı gradyan → aynı güncelleme, sonsuza dek.",
    rs_aria=("Residual akış boyunca varyansın büyümesi. x0'ın varyansı 1'dir. Her blok varyansı 1,0 olan F(x)'i ekler; "
             "x1'in varyansı 2 olur ve L bloktan sonra x L'nin varyansı 1 artı L'dir."),
    rs_note="her blok varyansı 1 olan bir kol ekler",
    lr_aria=("LayerNorm ile RMSNorm. LayerNorm x üzerinde 1. geçişte ortalama mu'yu hesaplar, 2. geçişte x eksi mu ile "
             "merkezler, 3. geçişte varyans sigma kareyi hesaplar, sonra normalize eder ve gamma ile beta afinini "
             "uygular. RMSNorm x'in karelerinin ortalamasının karekökünü tek geçişte hesaplar, sonra normalize eder ve "
             "yalnızca gamma uygular."),
    lr_rows=[("LayerNorm", [("ortalama μ", "1. geçiş"), ("x − μ", "2. geçiş"), ("varyans σ²", "3. geçiş"), ("normalize", None), ("afin", "γ, β")]),
             ("RMSNorm", [("RMS = √(Σx²/D)", "x üzerinde TEK geçiş"), ("normalize", None), ("afin", "yalnız γ")])],
    lr_foot="x üç kez yerine bir kez okunur; ortalama ve β ortadan kalkar.",
    pp_aria=("Post-LN ile Pre-LN. Orijinal 2017 Transformer'ı olan Post-LN'de x l hem alt katmana hem residual yola gider; "
             "ikisi toplanır ve LayerNorm toplamadan sonra ana yolun üstünde durur, x l artı 1'i üretir; her gradyan bir "
             "normalizasyondan geçmek zorundadır. 2019-2023'ün modern standardı Pre-LN'de kol önce LayerNorm ya da "
             "RMSNorm'u, sonra alt katmanı uygular; sonuç el değmemiş residual yola eklenir ve x l artı 1 çıkar; "
             "residual otoyolu temiz kalır."),
    pp_panels=[("Post-LN", "orijinal Transformer, 2017"), ("Pre-LN", "modern standart, 2019–2023")],
    pp_norm="LN / RMSNorm",
    pp_hw="residual",
    pp_warn=["norm otoyolun", "tam üstünde"],
    pp_ok=["otoyol el", "değmeden kalır"],
    uf_aria=("Birleştirilmemiş normalizasyon. Residual toplama sonucunu HBM'e yazar; ortalama indirgemesi onu geri okur "
             "ve yazar; varyans indirgemesi okur ve yazar; standartlaştırma ve afin okur ve yazar; QKV GEMM en sonunda "
             "normalize tensörü okur. Tek bir normalizasyon için dört HBM gidiş-dönüşü."),
    uf_sm="ayrı kernel'ler",
    uf_k=[("residual", "toplama"), ("ortalama", "indirgeme"), ("varyans", "indirgeme"), ("standartlaş.", "+ afin"), ("QKV GEMM", None)],
    uf_hbm="HBM (çip dışı)",
    uf_w="yaz", uf_r="oku",
    uf_foot="Tek bir normalizasyon için 4 HBM gidiş-dönüşü.",
    fu_aria=("Birleştirilmiş RMSNorm kernel'i. Tek bir GPU streaming multiprocessor içinde, SRAM ve yazmaçlarda: residual "
             "girdi yazmaçlarda toplanır, warp-shuffle indirgemesi RMS'i hesaplar, sonuç gamma ile çarpılır ve ölçekli "
             "çıktı hazır olur. Tek bir HBM yazımı onu doğrudan GEMM girdi tamponuna gönderir."),
    fu_sm="GPU streaming multiprocessor (SM) · SRAM / yazmaçlar",
    fu_top=["residual girdi", "yazmaç içi toplam", "warp-shuffle RMS"],
    fu_bot=["ölçekli çıktı", "γ ile çarp"],
    fu_reg="sonuna kadar hiçbir şey çipten çıkmaz",
    fu_hbm="HBM · GEMM girdi tamponu",
    fu_one="TEK HBM yazımı",
)

out = {}
for lang, L in (("en", EN), ("tr", TR)):
    out[lang] = dict(ravine=ravine(L), adamw=adamw(L), muon=muon(L), chain=chain(L), sym=sym(L), resid=resid(L),
                     lnrms=lnrms(L), prepost=prepost(L), unfused=unfused(L), fused=fused(L))
for d in out.values():
    for s in d.values():
        assert_no_blank_lines(s)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
