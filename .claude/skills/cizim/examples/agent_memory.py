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
    lp_aria=("The runtime execution loop of agent memory: User query q_t enters the read path, which queries the persistent "
             "memory store M through need detection, query rewriting, hybrid retrieval, and bi-temporal filtering. The read path "
             "assembles a dynamic context (system instructions, relevant memories, short-term history, and q_t) for the stateless "
             "foundation LLM. The model response a_t flows into the write path, which extracts atomic propositions, applies "
             "bi-temporal updates, checks PII policies, and consolidates new facts back into the persistent store."),
    lp_title="Runtime Execution Architecture",
    lp_u=("User Input", "turn q_t"),
    lp_rp=("Read Pipeline", "need detection · query rewrite", "hybrid search · RRF · temporal filter"),
    lp_pr=("Dynamic Prompt Assembly", "system prompt + retrieved memories", "+ recent dialog history + q_t"),
    lp_llm="Stateless Foundation LLM",
    lp_resp="Response a_t",
    lp_wp=("Write Pipeline", "fact extraction · bi-temporal log", "PII guardrails · consolidation"),
    lp_store=("Persistent Store", "vector + graph (M)"),
    lp_foot="The model remains strictly stateless; memory lives in the persistent store and the pipelines around it.",
    op_aria=("Memory write decision engine: A raw turn (q_t, a_t) undergoes fact extraction to isolate standalone atomic "
             "propositions. Semantic search compares candidate facts against top-k existing records. The decision gate selects "
             "one of four operations: ADD when no equivalent record exists; UPDATE when new details enrich an existing entity; "
             "SUPERSEDE when the new fact directly contradicts prior knowledge; or NOOP when identical data is already stored."),
    op_in="raw turn (q_t, a_t)",
    op_ex=("Fact Extraction", "atomic proposition"),
    op_ss=("Semantic Search", "compare vs top-k", "existing memories"),
    op_gate="Decision Gate",
    op_ops=[("ADD", ["no equivalent", "record exists"], ["insert vector", "& metadata"]),
            ("UPDATE", ["enriches existing", "knowledge record"], ["expand entity", "attributes"]),
            ("SUPERSEDE", ["contradicts prior", "memory record"], ["mark superseded", "(bi-temporal)"]),
            ("NOOP", ["identical fact", "already stored"], ["no operation", "(drop duplicate)"])],
    op_store="Persistent Memory Store (Vector, Relational & Graph)",
    rp_aria=("Step-by-step memory retrieval pipeline: User input q_t first passes a memory need classifier. If no memory is "
             "needed, the query bypasses retrieval directly to the LLM with +0 ms added latency. When required, the query is "
             "expanded with HyDE. Three retrieval modes run concurrently: dense cosine vector search, sparse BM25 keyword search, "
             "and a temporal knowledge graph walk. Reciprocal rank fusion (RRF) merges candidate lists, followed by multi-signal "
             "scoring (recency, importance, relevance) and bi-temporal validity filtering. Prefix-cache-aware packaging builds the "
             "final prompt context for the LLM."),
    rp_q="User Input q_t",
    rp_gate="1 · Need Classifier",
    rp_by=("Direct to LLM", "+0 ms added latency"),
    rp_no="memory not needed",
    rp_yes="memory required",
    rp_rw="2 · Query Rewrite & HyDE",
    rp_search=[("Dense Vector Search", "cosine similarity"), ("Sparse Text Search", "BM25 keyword"), ("Temporal Graph Walk", "entity graph traversal")],
    rp_row=[("4 · RRF Fusion", "reciprocal rank fusion"), ("5 · Multi-Signal Score", ["recency decay + importance", "+ semantic relevance"]),
            ("6 · Validity Filter", ["scope & bi-temporal window"])],
    rp_pack=("7 · Prompt Packaging", "prefix-cache optimized"),
    rp_llm="LLM Context Window",
    sc_aria=("Hierarchical memory scopes in a multi-agent system: At the top level, organization scope stores enterprise governance "
             "policies and runbooks in the organization store. Project scope manages shared architectural decisions and code standards, "
             "reading organization records via read-only reference. Agent-private working memory provides isolated scratchpads for "
             "specialized agents (Researcher, Coder, Auditor), each performing authorized scoped writes to the project store. At the base, "
             "user profile scope manages individualized preferences in a user_id-partitioned store accessed through strict authenticated filters."),
    sc_org=("Organization Scope (Global)", "governance policies, enterprise runbooks", "Organization Store"),
    sc_prj=("Project Scope (Shared)", "architectural decisions, code standards", "Project Store"),
    sc_ro="read-only reference access",
    sc_ag="Agent-Private Working Memory (Isolated Scratchpads)",
    sc_agents=["Researcher Agent", "Coder Agent", "Security Auditor"],
    sc_w="authorized scoped writes",
    sc_usr=("User Profile Scope (Tenant-Isolated)", "personal preferences, interaction style", "User Store (user_id)"),
    sc_f="strict authenticated filter",
)

TR = dict(
    lp_aria=("Ajan hafızasının çalışma zamanı mimarisi: Kullanıcı mesajı q_t okuma hattına girer; okuma hattı kalıcı hafıza "
             "deposu M&#x27;yi sorgulayarak ihtiyaç tespiti, sorgu yeniden yazımı, hibrit arama, RRF ve bi-temporal filtreleme uygular. "
             "Okuma hattı dinamik bir prompt montajı yapar: sistem talimatı, ilgili anılar, yakın diyalog geçmişi ve q_t birleştirilir. "
             "Durumsuz temel LLM modeli cevabı (a_t) üretir. Model ve cevap yazma hattına akar; yazma hattı atomik önermeleri ayıklar, "
             "bi-temporal güncellemeleri işler, PII süzgecinden geçirir ve kalıcı depoya konsolide eder."),
    lp_title="Çalışma Zamanı Mimarisi",
    lp_u=("Kullanıcı Mesajı", "girdi q_t"),
    lp_rp=("Okuma Hattı (Read Path)", "ihtiyaç tespiti · yeniden yazım", "hibrit arama · RRF · zaman süzgeci"),
    lp_pr=("Dinamik Prompt Montajı", "sistem prompt&#x27;u + ilgili anılar", "+ diyalog geçmişi + q_t"),
    lp_llm="Durumsuz Temel Model (LLM)",
    lp_resp="Üretilen Yanıt a_t",
    lp_wp=("Yazma Hattı (Write Path)", "önerme ayıklama · bi-temporal log", "PII maskeleme · kalıcı kayıt"),
    lp_store=("Hafıza Deposu", "vektör &amp; graf (M)"),
    lp_foot="Model her çağrıda tamamen durumsuz kalır; hafıza, depoda ve onu çevreleyen iki hatta yaşar.",
    op_aria=("Hafıza yazma karar mekanizması: Ham diyalog turu (q_t, a_t) atomik önerme ayıklama adımına gider ve bağlamdan "
             "bağımsız doğrulanabilir iddialar çıkarılır. Anlamsal benzerlik araması adayı mevcut top-k anılarla karşılaştırır. "
             "Karar kapısı dört işlemden birini yürütür: Eşdeğer kayıt yoksa ADD ile yeni vektör ve metaveri eklenir; mevcut bilgiyi "
             "zenginleştiriyorsa UPDATE ile kayıt genişletilir; önceki bilgiyle çelişiyorsa SUPERSEDE ile eski kayıt bi-temporal "
             "olarak hükümsüz kılınır; bilgi zaten aynen mevcutsa NOOP ile mükerrer kayıt düşürülür."),
    op_in="ham diyalog turu (q_t, a_t)",
    op_ex=("Önerme Ayıklama", "bağımsız atomik olgu"),
    op_ss=("Anlamsal Arama", "mevcut top-k anılarla", "kıyaslama"),
    op_gate="Karar Kapısı",
    op_ops=[("ADD", ["eşdeğer kayıt", "yoksa"], ["yeni vektör", "&amp; metaveri ekle"]),
            ("UPDATE", ["mevcut bilgiyi", "derinleştiriyorsa"], ["kaydı genişlet", "ve zenginleştir"]),
            ("SUPERSEDE", ["önceki bilgiyle", "çelişiyorsa"], ["hükümsüz kıl", "(bi-temporal)"]),
            ("NOOP", ["bilgi depoda", "birebir varsa"], ["işlem yapma", "(kopya düşür)"])],
    op_store="Kalıcı Hafıza Deposu (Vektör, İlişkisel &amp; Graf)",
    rp_aria=("Adım adım hafıza okuma ve erişim hattı: Kullanıcı girdisi q_t önce ihtiyaç tespiti kapısına gelir. Hafıza gerekmiyorsa "
             "istek doğrudan LLM&#x27;e aktarılır ve ek gecikme sıfırdır (+0 ms). Hafıza gerekiyorsa sorgu HyDE ile zenginleştirilerek "
             "yeniden yazılır. Üç arama motoru paralel çalışır: kosinüs benzerliğiyle yoğun vektör araması, BM25 ile seyrek anahtar "
             "kelime araması ve zamansal bilgi grafı yürüyüşü. Karşılıklı sıra füzyonu (RRF) listeleri birleştirir. Çok sinyalli puanlama "
             "yakınlık, önem katsayısı ve anlamsal alakayı harmanlar. Kapsam ve bi-temporal geçerlilik filtresi süresi dolmuşları eler. "
             "Prefix-cache dostu prompt paketleme nihai bağlamı hazırlar ve LLM çalışma penceresine iletir."),
    rp_q="Kullanıcı Girdisi q_t",
    rp_gate="1 · İhtiyaç Tespiti",
    rp_by=("Doğrudan LLM", "+0 ms ek gecikme"),
    rp_no="hafıza gerekmiyor",
    rp_yes="hafıza gerekli",
    rp_rw="2 · Yeniden Yazım &amp; HyDE",
    rp_search=[("Yoğun Vektör Arama", "kosinüs benzerliği"), ("Seyrek Metin Arama", "BM25 anahtar kelime"), ("Zamansal Bilgi Grafı", "ilişkisel graf yürüyüşü")],
    rp_row=[("4 · RRF Birleştirme", "karşılıklı sıra füzyonu"), ("5 · Çok Sinyalli Skor", ["zamansal yakınlık + önem", "+ anlamsal alaka"]),
            ("6 · Geçerlilik Süzgeci", ["kapsam &amp; bi-temporal zaman"])],
    rp_pack=("7 · Prompt Paketleme", "prefix-cache dostu"),
    rp_llm="LLM Çalışma Bağlamı",
    sc_aria=("Çok ajanlı bir sistemde hiyerarşik hafıza kapsamları: En üstte organizasyon kapsamı kurumsal ilkeleri, güvenlik politikalarını "
             "ve şirket runbook&#x27;larını global depoda tutar. Proje kapsamı mimari kararları, repo sözleşmelerini ve kodlama standartlarını "
             "paylaşılan proje deposunda yönetir; organizasyon deposuna salt okunur referansla erişir. Ajan özel bellekleri uzman ajanlar "
             "(Araştırmacı, Kodlayıcı, Güvenlik Denetçisi) için izole çalışma scratchpad&#x27;leri sağlar; ajanlar proje deposuna yetkili ve "
             "denetimli yazma yapar. En altta kullanıcı profili kapsamı kişisel tercihleri, etkileşim stilini user_id bazlı kiracı-yalıtımlı "
             "depoda saklar ve ajanlar buraya katı kimlik doğrulamalı güvenli filtreyle erişir."),
    sc_org=("Organizasyon Kapsamı (Global)", "kurumsal ilkeler, şirket runbook&#x27;ları", "Organizasyon Deposu"),
    sc_prj=("Proje Kapsamı (Paylaşılan)", "mimari kararlar, repo sözleşmeleri", "Proje Deposu"),
    sc_ro="salt okunur referans erişimi",
    sc_ag="Ajan Özel Bellekleri (İzole Scratchpad&#x27;ler)",
    sc_agents=["Araştırmacı Ajan", "Kodlayıcı Ajan", "Güvenlik Denetçisi"],
    sc_w="yetkili ve denetimli yazma",
    sc_usr=("Kullanıcı Profili Kapsamı (Kiracı Yalıtımlı)", "bireysel tercihler, etkileşim dili", "Kullanıcı Deposu (user_id)"),
    sc_f="kimlik doğrulamalı güvenli filtre",
)

out = {}
for lang, L in (("en", EN), ("tr", TR)):
    out[lang] = dict(loop=loop(L), ops=ops(L), readpath=readpath(L), scopes=scopes(L))
for d in out.values():
    for s in d.values():
        assert_no_blank_lines(s)
json.dump(out, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
