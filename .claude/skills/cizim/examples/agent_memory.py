"""Batch 3: how-ai-agent-memory-works / ajan-hafizasi-nasil-calisir — 4 Mermaid -> SVG."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../scripts"))
from svgkit import *

SUB = "fill:var(--c-text-mute);font-size:11px"
SUB2 = "fill:var(--c-text-mute);font-size:11.5px"
WARN, OK, BAD = "var(--c-warn)", "var(--c-success)", "var(--c-danger)"


def cyl(x, y, w, h, title, sub=None, col=WARN):
    st = f"fill:{col};fill-opacity:.12;stroke:{col};stroke-width:1.5"
    o = [f'<path d="M{x} {y + 7} V{y + h - 7} A{w / 2} 7 0 0 0 {x + w} {y + h - 7} V{y + 7}" style="{st}"/>',
         f'<ellipse cx="{x + w / 2}" cy="{y + 7}" rx="{w / 2}" ry="7" style="{st}"/>']
    ty = y + 7 + (h - 7) / 2 + (4.5 if sub is None else -2)
    o.append(text(x + w / 2, ty, title, "fill:var(--c-text);font-size:12.5px", "middle"))
    if sub:
        o.append(text(x + w / 2, ty + 15, sub, SUB, "middle"))
    return "\n".join(o)


def line(d, col="var(--c-text-mute)", width=1.5, dash=False):
    return f'<path d="{d}" style="fill:none;stroke:{col};stroke-width:{width}{";stroke-dasharray:5 4" if dash else ""}"/>'


# ------------------------------------------------------------------ runtime loop
def loop(L):
    o = [svg_open(560, 268, L["lp_aria"]), "<defs>", marker("aml-arr"), marker("aml-g", GEN), marker("aml-w", WARN), "</defs>"]
    o.append(text(16, 18, L["lp_title"], "fill:var(--c-text);font-size:12px;font-weight:600"))
    o.append(node(16, 30, 104, 56, L["lp_u"][0], L["lp_u"][1], ACC))
    o.append(node(140, 30, 194, 56, L["lp_rp"][0], L["lp_rp"][1:], ACC))
    o.append(node(354, 30, 190, 56, L["lp_pr"][0], L["lp_pr"][1:]))
    o.append(node(354, 116, 190, 40, L["lp_llm"]))
    o.append(node(354, 192, 190, 40, L["lp_resp"], None, GEN))
    o.append(node(140, 186, 194, 56, L["lp_wp"][0], L["lp_wp"][1:], WARN))
    o.append(cyl(16, 104, 104, 64, L["lp_store"][0], L["lp_store"][1]))
    o.append(arrow(120, 58, 138, 58, "aml-arr"))
    o.append(arrow(334, 58, 352, 58, "aml-arr"))
    o.append(arrow(449, 86, 449, 114, "aml-arr"))
    o.append(arrow(449, 156, 449, 190, "aml-g", GEN))
    o.append(path("M120 136 H237 V88", "aml-arr"))                     # store -> read path
    o.append(arrow(354, 212, 336, 212, "aml-w", WARN))                  # response -> write path
    o.append(path("M354 136 H300 V184", "aml-w", WARN))                 # LLM -> write path
    o.append(path("M140 214 H68 V170", "aml-w", WARN))                  # write path -> store
    o.append(text(16, 260, L["lp_foot"], MUTE))
    o.append(svg_close())
    return "\n".join(o)


# ------------------------------------------------------------------ ADD / UPDATE / DELETE / NOOP
def ops(L):
    o = [svg_open(560, 300, L["op_aria"]), "<defs>", marker("amo-arr"), "</defs>"]
    o.append(text(16, 18, L["op_in"], SUB2))
    o.append(arrow(60, 24, 60, 38, "amo-arr"))
    o.append(node(16, 40, 160, 52, L["op_ex"][0], L["op_ex"][1]))
    o.append(node(200, 40, 160, 52, L["op_ss"][0], L["op_ss"][1:]))
    o.append(node(384, 40, 160, 52, L["op_gate"], None, ACC))
    o.append(arrow(176, 66, 198, 66, "amo-arr"))
    o.append(arrow(360, 66, 382, 66, "amo-arr"))
    xs = [16, 148, 280, 412]
    cols = [OK, ACC, BAD, "var(--c-text-mute)"]
    o.append(line("M464 92 V108 H78 V122"))
    for x in xs[1:]:
        o.append(arrow(x + 62, 108, x + 62, 120, "amo-arr"))
    o.append(arrow(78, 108, 78, 120, "amo-arr"))
    for k, (x, (op, cond, eff)) in enumerate(zip(xs, L["op_ops"])):
        o.append(box(x, 122, 124, 66, inner(cols[k])))
        o.append(text(x + 62, 141, op, f"fill:{cols[k]};font-size:13px;font-weight:700;font-family:var(--font-mono)", "middle"))
        for j, s in enumerate(cond):
            o.append(text(x + 62, 158 + j * 14, s, SUB, "middle"))
        for j, s in enumerate(eff):
            o.append(text(x + 62, 206 + j * 14, s, "fill:var(--c-text);font-size:11px", "middle"))
        if k < 3:
            o.append(arrow(x + 62, 230, x + 62, 248, "amo-arr"))
    o.append(cyl(16, 250, 388, 42, L["op_store"]))
    o.append(f'<circle cx="474" cy="268" r="9" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>')
    o.append(f'<circle cx="474" cy="268" r="4.5" style="fill:var(--c-text-mute)"/>')
    o.append(arrow(474, 230, 474, 256, "amo-arr"))
    o.append(svg_close())
    return "\n".join(o)


# ------------------------------------------------------------------ read path
def readpath(L):
    o = [svg_open(560, 392, L["rp_aria"]), "<defs>", marker("amr-arr"), marker("amr-g", GEN), "</defs>"]
    o.append(node(16, 8, 130, 48, L["rp_q"], None, ACC))
    o.append(node(170, 8, 160, 48, L["rp_gate"], None, WARN))
    o.append(node(410, 8, 134, 48, L["rp_by"][0], L["rp_by"][1]))
    o.append(arrow(146, 32, 168, 32, "amr-arr"))
    o.append(arrow(330, 32, 408, 32, "amr-arr"))
    o.append(text(370, 26, L["rp_no"], SUB, "middle"))
    o.append(arrow(250, 56, 250, 86, "amr-arr"))
    o.append(text(258, 75, L["rp_yes"], SUB))
    o.append(node(150, 88, 200, 44, L["rp_rw"]))
    # fan-out
    o.append(line("M250 132 V146 M100 146 H460"))
    xs = [16, 196, 376]
    for x, (t, s) in zip(xs, L["rp_search"]):
        o.append(arrow(x + 84, 146, x + 84, 160, "amr-arr"))
        o.append(node(x, 162, 168, 48, t, s))
    # fan-in
    o.append(line("M280 210 V224 M460 210 V224 H100"))
    o.append(arrow(100, 210, 100, 240, "amr-arr"))
    row = [16, 196, 376]
    for k, (x, (t, s)) in enumerate(zip(row, L["rp_row"])):
        o.append(node(x, 242, 168, 58, t, s))
        if k < 2:
            o.append(arrow(x + 168, 271, x + 178, 271, "amr-arr"))
    o.append(arrow(460, 300, 460, 326, "amr-arr"))
    o.append(node(376, 328, 168, 58, L["rp_pack"][0], L["rp_pack"][1:]))
    o.append(node(196, 336, 152, 42, L["rp_llm"], None, GEN))
    o.append(arrow(376, 357, 350, 357, "amr-g", GEN))
    o.append(svg_close())
    return "\n".join(o)


# ------------------------------------------------------------------ memory scopes
def scopes(L):
    o = [svg_open(560, 364, L["sc_aria"]), "<defs>", marker("ams-arr"), marker("ams-a", ACC), "</defs>"]

    def layer(y, col, title, sub, store):
        r = [box(16, y, 528, 52, inner(col)),
             text(30, y + 21, title, f"fill:{col};font-size:13px;font-weight:700"),
             text(30, y + 38, sub, SUB2)]
        if store:
            r.append(cyl(330, y + 9, 202, 36, store, None, col))
        return "\n".join(r)

    o.append(layer(8, WARN, *L["sc_org"]))
    o.append(layer(96, ACC, *L["sc_prj"]))
    o.append(arrow(280, 96, 280, 62, "ams-arr", dash=True))
    o.append(text(288, 84, L["sc_ro"], SUB))
    o.append(box(16, 184, 528, 88, inner(GEN)))
    o.append(text(30, 203, L["sc_ag"], f"fill:{GEN};font-size:13px;font-weight:700"))
    for k, (x, a) in enumerate(zip((32, 202, 372), L["sc_agents"])):
        o.append(node(x, 214, 156, 44, a))
    o.append(arrow(450, 184, 450, 150, "ams-arr", dash=True))
    o.append(text(442, 172, L["sc_w"], SUB, "end"))
    o.append(layer(304, OK, *L["sc_usr"]))
    for x in (110, 280):
        o.append(arrow(x, 304, x, 262, "ams-a", ACC, width=3))
    o.append(text(122, 288, L["sc_f"], "fill:var(--c-text);font-size:11px"))
    o.append(svg_close())
    return "\n".join(o)


EN = dict(
    lp_aria=("The runtime execution loop of agent memory. The user input q_t enters the read path, which also reads the "
             "persistent memory store M and runs a need gate, query rewrite, hybrid search, RRF and filtering. The read "
             "path builds a dynamic prompt: system prompt plus relevant memories plus history plus q_t. The stateless LLM "
             "produces the response a_t. The LLM and the response feed the write path, which extracts facts, applies "
             "bi-temporal updates and passes a PII gate, then writes back to the store."),
    lp_title="Runtime execution loop",
    lp_u=("User input", "q_t"),
    lp_rp=("Read path", "need gate · rewrite · hybrid", "search · RRF · filter"),
    lp_pr=("Dynamic prompt", "system + relevant memories", "+ history + q_t"),
    lp_llm="Stateless LLM",
    lp_resp="Response a_t",
    lp_wp=("Write path", "extraction · bi-temporal", "update · PII gate"),
    lp_store=("Memory", "store M"),
    lp_foot="The model stays stateless; memory lives in the store and the two paths around it.",
    op_aria=("How a write decides what to do. A raw turn, q_t and a_t, goes to extraction, which produces a candidate "
             "proposition. Similarity search compares it against the top-k existing records. A decision gate then picks "
             "one of four operations. ADD when no equivalent record is found: insert a new vector and metadata. UPDATE "
             "when it refines or extends an existing record: enrich the existing entity. DELETE when it directly "
             "contradicts an existing record: mark the old one superseded, bi-temporally. NOOP when the information is "
             "already stored identically: no operation. ADD, UPDATE and DELETE write to the store."),
    op_in="raw turn (q_t, a_t)",
    op_ex=("Extraction", "candidate proposition"),
    op_ss=("Similarity search", "vs top-k existing"),
    op_gate="Decision gate",
    op_ops=[("ADD", ["no equivalent", "record found"], ["insert new", "vector & meta"]),
            ("UPDATE", ["refines or extends", "an existing record"], ["enrich existing", "entity"]),
            ("DELETE", ["directly contradicts", "an existing record"], ["mark superseded", "(bi-temporal)"]),
            ("NOOP", ["already stored", "identically"], ["no operation", ""])],
    op_store="Memory store",
    rp_aria=("The read path, step by step. User input q_t meets a need detection gate. If no memory is needed, the "
             "request goes straight to the LLM with zero added latency. If memory is required, the query is rewritten, "
             "with HyDE. Three searches run in parallel: dense vector search by cosine, sparse keyword search with BM25, "
             "and a temporal knowledge graph walk. Reciprocal rank fusion merges their lists. Multi-signal scoring "
             "combines recency, importance and relevance. A scope and bi-temporal validity filter drops what does not "
             "apply. Cache-aware prompt packaging assembles the result, which becomes the LLM context."),
    rp_q="User input q_t",
    rp_gate="1 · Need detection",
    rp_by=("Direct to LLM", "+0 ms latency"),
    rp_no="not needed",
    rp_yes="memory required",
    rp_rw="2 · Query rewrite & HyDE",
    rp_search=[("Dense vector search", "cosine"), ("Sparse keyword search", "BM25"), ("Temporal graph walk", "knowledge graph")],
    rp_row=[("4 · RRF", "reciprocal rank fusion"), ("5 · Multi-signal score", ["recency + importance", "+ relevance"]),
            ("6 · Validity filter", ["scope + bi-temporal"])],
    rp_pack=("7 · Prompt packaging", "cache-aware"),
    rp_llm="LLM context",
    sc_aria=("Memory scopes in a multi-agent system. At the top, the organization scope holds global policies and company "
             "runbooks in the organization store. Below it, the project scope holds architectural decisions and code "
             "standards in the project store; the project reads the organization store as a read-only reference. The "
             "private agent working memory holds scratchpads for a researcher agent, a coder agent and a security "
             "auditor; the agents make explicit scoped writes to the project store. At the bottom, the user profile "
             "scope holds personal preferences and style in a store partitioned by user_id, which the researcher and "
             "coder agents read through a strict authenticated filter."),
    sc_org=("Organization scope", "global policies, company runbooks", "Organization store"),
    sc_prj=("Project scope", "architectural decisions, code standards", "Project store"),
    sc_ro="read-only reference",
    sc_ag="Private agent working memory (scratchpads)",
    sc_agents=["Researcher agent", "Coder agent", "Security auditor"],
    sc_w="explicit scoped writes",
    sc_usr=("User profile scope", "personal preferences, style", "User store (by user_id)"),
    sc_f="strict authenticated filter",
)

TR = dict(
    lp_aria=("Ajan hafızasının çalışma zamanı döngüsü. Kullanıcı mesajı q_t okuma yoluna girer; okuma yolu uzun süreli "
             "hafıza deposu M'yi de okur ve ihtiyaç tespiti, yeniden yazım, hibrit arama, RRF ve filtre adımlarını "
             "çalıştırır. Okuma yolu dinamik bir prompt kurar: sistem prompt'u, ilgili anılar, son turlar ve q_t. "
             "Durumsuz model cevabı, a_t'yi üretir. Model ve cevap yazma yoluna akar; yazma yolu olguları ayıklar, "
             "bi-temporal güncelleme yapar, maskeler, PII kapısından geçirir ve depoya geri yazar."),
    lp_title="Çalışma zamanı döngüsü",
    lp_u=("Kullanıcı", "mesajı q_t"),
    lp_rp=("Okuma yolu", "ihtiyaç · yeniden yazım ·", "hibrit arama · RRF · filtre"),
    lp_pr=("Dinamik prompt", "sistem + ilgili anılar", "+ son turlar + q_t"),
    lp_llm="Durumsuz model (LLM)",
    lp_resp="Cevap a_t",
    lp_wp=("Yazma yolu", "ayıkla · bi-temporal güncelle", "maskele · PII kapısı"),
    lp_store=("Hafıza", "deposu M"),
    lp_foot="Model durumsuz kalır; hafıza depoda ve onu saran iki yolda yaşar.",
    op_aria=("Bir yazmanın ne yapacağına nasıl karar verdiği. Ham tur, q_t ve a_t, ayıklamaya gider ve aday bir atomik "
             "olgu çıkar. Benzerlik araması onu mevcut en yakın anılarla kıyaslar. Karar kapısı dört işlemden birini "
             "seçer. Eşdeğer kayıt yoksa ADD: yeni vektör ve kayıt eklenir. Mevcut bilgiyi zenginleştiriyorsa UPDATE: "
             "bilgi genişletilir. Mevcut bilgiyle çelişiyorsa DELETE: eski kayıt geçersiz kılınır, supersede. Bilgi "
             "zaten aynen mevcutsa NOOP: değişiklik yok. ADD, UPDATE ve DELETE depoya yazar."),
    op_in="ham tur (q_t, a_t)",
    op_ex=("Ayıklama", "aday atomik olgu"),
    op_ss=("Benzerlik araması", "en yakın anılarla", "kıyasla"),
    op_gate="Karar kapısı",
    op_ops=[("ADD", ["eşdeğer", "kayıt yoksa"], ["yeni vektör", "& kayıt"]),
            ("UPDATE", ["mevcut bilgiyi", "zenginleştiriyorsa"], ["bilgiyi", "genişlet"]),
            ("DELETE", ["mevcut bilgiyle", "çelişiyorsa"], ["geçersiz kıl", "(supersede)"]),
            ("NOOP", ["bilgi zaten", "aynen mevcutsa"], ["değişiklik yok", ""])],
    op_store="Hafıza deposu",
    rp_aria=("Okuma yolu adım adım. Kullanıcı girdisi q_t bir ihtiyaç tespiti kapısına gelir. Hafıza gerekmiyorsa istek "
             "doğrudan LLM'e gider, ek gecikme sıfırdır. Hafıza gerekiyorsa sorgu HyDE ile yeniden yazılır. Üç arama "
             "paralel çalışır: kosinüsle yoğun vektör araması, BM25 ile seyrek anahtar kelime araması ve zamansal "
             "graf yürüyüşü. Karşılıklı sıra füzyonu, RRF, listeleri birleştirir. Çok sinyalli puanlama yakınlık, önem "
             "ve ilgiyi birleştirir. Kapsam ve zaman geçerlilik filtresi geçersizleri atar. Önbellek dostu prompt "
             "paketleme sonucu toplar ve sonuç LLM bağlamı olur."),
    rp_q="Girdi q_t",
    rp_gate="1 · İhtiyaç tespiti",
    rp_by=("Doğrudan LLM", "+0 ms ek gecikme"),
    rp_no="gerekmiyor",
    rp_yes="hafıza gerekiyor",
    rp_rw="2 · Yeniden yazım & HyDE",
    rp_search=[("Yoğun vektör arama", "dense cosine"), ("Seyrek arama", "BM25 keyword"), ("Zamansal graf", "graph walk")],
    rp_row=[("4 · RRF", "karşılıklı sıra füzyonu"), ("5 · Çok sinyalli puan", ["yakınlık + önem", "+ ilgi"]),
            ("6 · Geçerlilik filtresi", ["kapsam + zaman"])],
    rp_pack=("7 · Prompt paketleme", "önbellek dostu"),
    rp_llm="LLM bağlamı",
    sc_aria=("Çok ajanlı bir sistemde hafıza kapsamları. En üstte organizasyon kapsamı, global kuralları ve şirket "
             "runbook'larını organizasyon deposunda tutar. Altında proje kapsamı mimari kararları ve ortak kod "
             "standartlarını proje deposunda tutar; proje, organizasyon deposunu salt okunur referans olarak okur. Ajan "
             "özel bellekleri araştırmacı ajan, kodlayıcı ajan ve güvenlik denetçisi için geçici çalışma notlarını "
             "tutar; ajanlar proje deposuna izin gerektiren kısıtlı yazmalar yapar. En altta kullanıcı profili kapsamı "
             "kişisel tercihleri ve iletişim stilini user_id bazlı bir depoda tutar; araştırmacı ve kodlayıcı ajanlar "
             "onu katı bir kimlik filtresiyle okur."),
    sc_org=("Organizasyon kapsamı", "global kurallar, şirket runbook'ları", "Organizasyon deposu"),
    sc_prj=("Proje kapsamı", "mimari kararlar, ortak kod standartları", "Proje deposu"),
    sc_ro="salt okunur referans",
    sc_ag="Ajan özel bellekleri (geçici çalışma notları)",
    sc_agents=["Araştırmacı ajan", "Kodlayıcı ajan", "Güvenlik denetçisi"],
    sc_w="kısıtlı yazma (izin gerekir)",
    sc_usr=("Kullanıcı profili kapsamı", "kişisel tercihler, iletişim stili", "Kullanıcı deposu (user_id)"),
    sc_f="katı kimlik filtresiyle okuma",
)

out = {}
for lang, L in (("en", EN), ("tr", TR)):
    out[lang] = dict(loop=loop(L), ops=ops(L), readpath=readpath(L), scopes=scopes(L))
for d in out.values():
    for s in d.values():
        assert_no_blank_lines(s)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
