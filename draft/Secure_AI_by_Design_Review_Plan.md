# Secure AI by Design — Denetim ve Düzeltme Planı

**Tarih:** 2026-11-14 · **Kapsam:** Büyük Resim (3) + 12 bölüm, EN + TR (26 dosya), `posts.json`, `sitemap.xml`

## 0. Nasıl denetlendi

| Yöntem | Sonuç |
|---|---|
| 13 EN makalenin tamamı satır satır okundu (SVG metinleri dahil) | Aşağıdaki bulgular |
| Bütün Python blokları çalıştırıldı, çıktılar makaledeki çıktıyla karşılaştırıldı | 3 uyuşmazlık, 6 çıktısız blok, 1 mantık hatası |
| EN ↔ TR sayı diff'i | TR, EN ile satır hizalı; TR'ye özgü sayı hatası yok, EN'deki her hata TR'de de var |
| `validate_post.sh` (site + TOC + KaTeX + simetri) | 0 hata; tek asimetri: Bölüm 11 TR TOC'u 10, EN 17 bağlantı |
| İç bağlantılar (`post.html?slug=`) | EN'de 4 kırık slug |
| Dış bağlantılar (79 URL, HTTP) | 1 adet 404 (NVIDIA), 1 adet taslak sürüme giden (NIST) |
| Birincil kaynak teyidi (web) | OWASP LLM Top 10 2026 numaraları, vLLM güvenlik dokümanı, NVIDIA H100 CC blogu, Vec2Text özeti, Prompt Guard 2 model kartı, Many-shot Jailbreaking, SLSA v1.2, LiteLLM guardrail modları, OTel GenAI, MCP 2026-07-28 taşıma katmanı, pgvector 0.8.0 |

**Genel teşhis.** Büyük Resim makalesi ile Bölüm 10–12 tutarlı ve ölçülü. Bölüm 1–9 sonradan "zenginleştirilmiş"; bu geçişte (a) bazı teknik iddialar kaynağın söylediğinin ötesine taşmış, (b) Büyük Resim'in "her sayı bir bölümdeki örnekten gelir" dediği örneklerin bir kısmı bölümlerden düşmüş ya da değişmiş, (c) kod blokları çalıştırılmadan çıktı yazılmış, (d) SVG-only kuralına aykırı ASCII kutu çizimleri girmiş.

Öncelik sırası: **A** teknik/olgusal hata → **B** seri içi çelişki → **C** kod/çıktı → **D** blog kuralı ihlali → **E** eksik içerik → **F** metadata.

---

## 1. Seri genelinde (G)

| # | Sorun | Düzeltme |
|---|---|---|
| G1 | **Seri içi sayı çelişkisi (katmanlar).** Büyük Resim §9: `p=(0,10; 0,20; 0,30)`, ortak kör nokta %5 → %0,6 / %5,57 / 9,3×. Bölüm 1 §6: `p3=0,05`, β=%4 → %0,1 / %4,096 / 41×. Büyük Resim "sayılar bölümlerden gelir" diyor. | Bölüm 1 örneği Büyük Resim'in sayılarına çekilir (β=0,05; p3=0,30). Bu aynı anda P1-2'yi de çözer. Özet/tagline/altı satır güncellenir. Büyük Resim'e "Bölüm 1 bu orana β der" notu. |
| G2 | **Güven düzeyi çelişkisi.** Bölüm 5 ölçeği: iç veri τ=1. Büyük Resim §6: politika koleksiyonu τ=1. Bölüm 7: "denetlenmiş politika τ=2"; bellek figürü "τ≥2, fact" ama kod ve Bölüm 8 matrisi (`write_memory` r=1) τ≥1 diyor. | Bölüm 7'de politika τ=1; figür etiketi "τ≥1". |
| G3 | **Zarar oranı / eşik çelişkisi.** Bölüm 12 ve Büyük Resim: "1.200 vakada 0 zarar, üst sınır %0,25 ≤ %0,5". Bölüm 10 raporu: zarar %0,1–0,3 (420/600/180 vakada tam sayı olmayan sayımlar), kapı eşiği τ_harm=%1, hakem κ eşiği 0,75 ama TR κ=0,74 "geçti", figürde κ≥0,70. | Bölüm 10 raporu tam sayılı k/n ile yeniden hesaplanır (zarar 0/1.200), eşik %0,5, κ≥0,75 her yerde, TR κ=0,78. |
| G4 | **Kalan riski kim kabul eder?** Bölüm 1: iş sahibi. Bölüm 10 raporu: "Security Architect Approved". Bölüm 4 formu: "AI Platform Lead (Accountable)" — Bölüm 1 RACI'sinde bu satırın A'sı Güvenlik Mimarı. | İkisi de Bölüm 1 RACI'sine uydurulur. |
| G5 | **"Von Neumann kırılması" kavramsal olarak ters.** Von Neumann mimarisi kod ile veriyi *aynı* bellekte tutan mimaridir; ayrımı sonradan NX/W^X ve parametreli sorgular getirdi. Metin ".text/.data donanımsal olarak ayrıdır" ve "SQL parametreleri donanım sınırı" diyor. Bölüm 2 (aksiyom, sözlük, özet) ve Bölüm 5 girişinde. | "Bant içi talimat sorunu (token'lar için W^X yok)" olarak yeniden yazılır; sözlük ve `posts.json` özeti güncellenir. |
| G6 | **ASCII kutu çizimleri** (blog kuralı: yalnız SVG). Bölüm 2 (2 adet), Bölüm 4 (1), Bölüm 7 (3), Bölüm 9 (1). | Tabloya / numaralı listeye / `json` bloğuna çevrilir (içerik korunur). |
| G7 | **EN SVG'lerinde Türkçe metin.** Bölüm 8: iki `aria-label` Türkçe. Bölüm 9 akış figürü: "başlangıç", "ihlal üretildi", "karakter". Bölüm 10: eksen `%0…%16`, `1.000/3.000`, ve tamamı Türkçe dört satırlık kutu. | EN'e çevrilir; Bölüm 10 kutusundaki κ≥0,70 → 0,75 (EN+TR). |
| G8 | **Kırık iç bağlantılar (EN).** `cikti-katmani-derinlemesine` (B6), `langgraph-llm-applications` (B8), `inside-the-vllm-engine` ve `llm-cost-and-latency-optimization` (B9). | Doğru EN slug'ları: `inside-the-output-layer`, `langgraph-flow-design-llm-applications`, `vllm-from-paper-to-production`, `optimizing-llm-cost-and-latency`. |
| G9 | **Çıktısı gösterilmeyen kod** (skill kuralı: çıktısı gösterilen çalıştırılabilir kod). Bölüm 4 §8.2, 5 §8.2, 6 §8.3, 10 §8.1, 11 §7.2, 12 §7.3. | Kod çalıştırılır, gerçek çıktı eklenir; 10/11/12'ye kısa demo eklenir. |
| G10 | **Seri gezinmesi.** Bölüm 7–12'de "önceki/sonraki bölüm" var; 4, 5, 6'da sonraki bölüm bağlantısı yok. | 4→5, 5→6, 6→7 bağlantıları eklenir. |
| G11 | **Ton.** "99.41% CERTAIN BREACH!", "battle-tested enterprise fortress", "(480x Risk!)" gibi bağırış; skill "kıdemli mühendis, beyaz tahta" tonu istiyor. | Yalnız dokunulan pasajlarda sadeleştirilir; kapsamlı üslup revizyonu bu planın dışında (bkz. §4). |

---

## 2. Bölüm bölüm

### Büyük Resim (3) — `how-ai-security-works` / `yapay-zeka-guvenligi-nasil-calisir`
Durum: tutarlı, kod çıktısı birebir eşleşiyor. Yalnız G1 notu ve Bölüm 9 düzeltmesinin yansıması (önbellek cümlesi yok; değişiklik gerekmiyor).

### Bölüm 1 — Tasarımda başlar
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P1-1 | B | G1 (sayılar) | β=0,05; p=(0,10; 0,20; 0,30) → 0,006 / 0,0557 / 9,3×; Monte Carlo yeniden koşulur. |
| P1-2 | A | 3. katman "deterministik şema/allowlist doğrulayıcı" ama "ortak tokenizer kör noktası"nı paylaşıyor; ardından "tabanı kıran tek şey deterministik kontroller" deniyor — örnek kendiyle çelişiyor. | 3. katman model tabanlı "içerik denetimi" olur; deterministik allowlist tabanı kıran *dördüncü, farklı türden* kontrol olarak anlatılır. |
| P1-3 | A | β-faktör doğrudan IEC 61508'den alınmış gibi; aslında "saldırıların β kadarı ortak kör noktaya düşer" bizim uyarlamamız. | "uyarlama" olduğu açıkça yazılır. |
| P1-4 | A | Altı güven bölgesi "NIST AI 600-1 ve Google SAIF ile hizalı" — bölgeler bu serinin kendi ayrımı. | "kendi ayrımımız; … çerçeveleriyle uyumlu düşünülmüştür". |
| P1-5 | A | "OWASP … Excessive Agency'yi üst sıralara koyar" belirsiz. | LLM03:2026 (teyit edildi). |
| P1-6 | A | Kaynak: SP 800-218A bağlantısı `/ipd` (taslak). | `/final`. |
| P1-7 | B | Yol haritası tablosu: "GCG jailbreaks", "Clopper-Pearson" (seri Wilson kullanıyor), "NIST AI 600-1 mapping, signed deployment manifests" (Bölüm 12'de yok). | Bölümlerin gerçek içeriğine göre düzeltilir. |
| P1-8 | G11 | "battle-tested enterprise fortress" | sadeleştirilir. |

### Bölüm 2 — Tehdit modelleme
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P2-1 | A | G5 (Von Neumann) | yeniden yazım + sözlük. |
| P2-2 | A | "NIST AI 100-2 dört boyutta profillemeyi **zorunlu kılar**" — taksonomi, zorunluluk değil; "erişim" boyutu NIST'te ayrı eksen değil. | "amaç, yetenek, bilgi üzerinden sınıflar; erişimi biz ekliyoruz". |
| P2-3 | A | T2 için `AML.T0056` (System Prompt Extraction) verilmiş; bu T4'ün tekniği. | T2 → `AML.T0057` (LLM Data Leakage). |
| P2-4 | A | "Lethal trifecta … **if and only if**", figürde "exfiltration is inevitable". | "mümkün olur"; figür metni "possible". |
| P2-5 | D | G6: standartlar kutusu + yönlü çizge ASCII. | Tablo + düz metin yol tanımı (figür `p2_trifecta` zaten var). |
| P2-6 | G11 | "99.41% CERTAIN BREACH!", "63.40% PROBABILITY OF BREACH!" | sade. |

### Bölüm 3 — Veri
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P3-1 | A | Vec2Text: "klinik notlardan isim, tanı, ilaç %95+ kesinlikle" — makale yalnız *tam adların* çıkarılabildiğini söylüyor; "gradyan tabanlı arama" değil, yinelemeli düzelt-yeniden göm. | Kaynağa uygun yazılır. |
| P3-2 | A | DP-SGD formülü: gürültü ortalamanın dışında. Abadi vd.: `(1/B)(Σ ḡ + N(0, σ²C²I))`. | Formül düzeltilir. |
| P3-3 | E | DP-SGD için sayısal yürüyüş yok (formül var, örnek yok). | C=1, üç örnekli küçük kırpma + gürültü yürüyüşü eklenir. |
| P3-4 | A | "HMAC takma adını anahtarı bilen **tersine çevirir**" — HMAC tersinmez; anahtar sahibi *yeniden hesaplayıp eşleştirir* (TCKN uzayı küçük, sayılabilir). | Metin + figür etiketi. |
| P3-5 | A | Kripto-imha "dokuz depoya dokunmadan hepsini gürültüye çevirir" — ANN dizinindeki vektörler aranabilmek için açık durur; sağlayıcıdaki prompt'lar, analist dışa aktarımları ve ince ayarlı ağırlıklar KMS'inizin dışında. | "Kripto-imhanın ulaşmadığı dört yer" paragrafı; kontrol listesi 6. madde. |
| P3-6 | A | "Semantik önbellek … vLLM Radix attention tamponları" — RadixAttention SGLang'e ait; önek KV önbelleği semantik önbellek değil. | Düzeltilir. |
| P3-7 | A | "veri **üstel** çoğalır", "vektörler GDPR/CCPA'da düz metinle **aynı hukuki sınıf**" | "çoğalır"; "aynı sınıfta koruyun". |
| P3-8 | C | Kod çıktısı `1.00%`, gerçek çıktı `0.99%`. | Gerçek çıktı. |
| P3-9 | — | Yazım: "Crated datasets". | "Curated". |
| P3-10 | E | EN'de KVKK yok (özet "GDPR and KVKK" diyor). | KVKK m.3 anonimleştirme tanımı bir cümle. |

### Bölüm 4 — Tedarik zinciri
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P4-1 | A | "SLSA Build L3 = hermetik build". v1.x'te hermetiklik şart değil; L3 = sertleştirilmiş platform, çalıştırmalar arası yalıtım, imza anahtarına build adımının erişememesi (teyit edildi). | L3 tanımı düzeltilir; hermetik build ayrı, ek bir pratik olarak anlatılır (öğrenme kontrolü, sözlük, tablo dahil). |
| P4-2 | A | "Pickle Turing-tam" (döngüsü yok; doğru ifade: rastgele çağrılabilir çalıştırır). | Metin + figür altı. |
| P4-3 | C | PVM disassembly çıktısı uydurma (MEMOIZE satırları, ofsetler yanlış). | Gerçek `pickletools.dis` çıktısı. |
| P4-4 | C | `MaliciousPickleInspector` zararsız pickle'ı da REDDEDİYOR (her sınıf örneği `STACK_GLOBAL`+`BUILD` üretir). | Opcode değil *import hedefi* allowlist'i (weights_only mantığı); çıktı eklenir. |
| P4-5 | A | Safetensors başlık sınırı metinde `100×10⁶`, kodda `100·1024·1024`; `s<e` boş tensörü reddeder. | `100_000_000`; `s ≤ e`. |
| P4-6 | A | "CycloneDX v1.6 ML bileşenlerini ekledi" — ML-BOM v1.5 ile geldi. | Düzeltilir. |
| P4-7 | A | `chat_template.json` — şablon `tokenizer_config.json` alanı ya da `chat_template.jinja`. (Bölüm 5 kontrol listesinde de.) | Düzeltilir. |
| P4-8 | A | Tool shadowing tanımı (Invariant Labs): kötü sunucunun açıklaması *başka* sunucunun aracının kullanımını değiştirir; metin ad çakışması olarak anlatıyor, uydurma olasılık formülüyle. Bölüm 8'de de. | İki bölümde tek tanım. |
| P4-9 | A | Slopsquatting: "PyPI'de yok" — çalışma Python + JavaScript. `p=10⁻⁴` "tipik taban" diye sunulmuş. NIST aktörleri "System Integrators". | "PyPI ve npm"; "örnek varsayım"; "AI system producers". |
| P4-10 | B | Regresyon kapısı metinde %25, figürde %30. Düşünce deneyi başlıkta "14 GB dev", gövdede "14 milyar parametre ≈ 28 GB". | %30; "7B ≈ 14 GB (fp16)". |
| P4-11 | A | JCS: `json.dumps(sort_keys)` "RFC 8785'e göre" deniyor; yaklaşık. Kabul formunda uydurma commit öneki `e07e...`. | Docstring'e sınır notu; `<40-hex commit SHA>`. |
| P4-12 | E | Büyük Resim'in alıntıladığı "bölümün kilit cümlesi" (*onayladığım artefakt mı? / onaylamalı mıydım?*) bölümde yok. Jinja SSTI için somut kanıt yok. | Cümle §3'e eklenir; CVE-2024-34359 anılır. |
| P4-13 | G4, G6, G9, G10 | Form imzası, MCP ASCII, çıktı, sonraki bölüm bağlantısı. | — |

### Bölüm 5 — Prompt injection
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P5-1 | A | Many-shot: "**zararsız** sahte diyaloglar", uydurma formül `P ∝ 1−e^{−αk^β}`, "k=128'de %90+". Kaynak: sahte diyaloglarda asistan zararlı soruları yanıtlar; ölçeklenme güç yasası; 256 atışa kadar test. | Kaynağa uygun; formül NLL güç yasası olarak. |
| P5-2 | A | GCG: "NP-hard", "tek-sıcak embedding gradyanı ∈ ℝ^d", "güvenlik ağırlıklarını etkisizleştirir". | "|V|^L aday; kaba kuvvet imkânsız", gradyan tanımı netleştirilir. |
| P5-3 | E | GCG aday seçimi için sayısal yürüyüş yok. | 4 token'lık oyuncak sözlükte `−E·G` yürüyüşü. |
| P5-4 | A | `n50 ≈ 69,3 ⟹ 70 deneme` — tam değer 68,97; 69 deneme yeter (figür ve kod 69 diyor). | 69. |
| P5-5 | A | G5: "klasik von Neumann sınırının silinmesi". | Yeniden yazılır. |
| P5-6 | G9, G10, P4-7 | Çıktı, sonraki bölüm, `chat_template.json`. | — |

### Bölüm 6 — Guardrails
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P6-1 | A | "Neyman-Pearson" adı yanlış: `t* = C_FP/(C_FP+C_FN)` Bayes asgari-risk kuralıdır. Neyman–Pearson, FPR bütçesi sabitken TPR'yi en çoklayan kuraldır ("%1 FPR'de TPR" tam olarak budur). | Ad düzeltilir, ikisinin farkı bir paragrafta anlatılır; sözlük, kod, kontrol listesi. |
| P6-2 | A | "Cross-encoder classifier" — Prompt Guard tek metni sınıflayan encoder'dır; cross-encoder çift puanlar. | "Encoder sınıflandırıcı" (tablo, L2, YAML, kod). |
| P6-3 | A | TCKN için "Luhn sağlaması" (Bölüm 3'te TCKN'nin kendi mod-10 algoritması anlatılıyor). | "sağlama toplamı". |
| P6-4 | E | §4 başlığı "morfoloji ve büyük/küçük harf tuzakları", altı satır "ekler, düşen diakritikler, kod değiştirme" diyor; gövdede yalnız İ/ı var. Prompt Guard 2'nin değerlendirme dillerinde Türkçe olmadığı (teyitli) bilgisi düşmüş. | Dört alt madde + model kartı notu; kod yorumları gerçek çıktıya uydurulur. |
| P6-5 | B | Kademe: figür 3 kademe 61/331 ms; metin 4 kademe 71/341 ms. L2 "s>0,9 emin" derken L3 gri bölge "0,10–0,80". | Metin figürü açıklar (61 → L1 eklenince 71); eşik 0,8. Paralel bileşim cümlesi (Büyük Resim anıyor). |
| P6-6 | A | Ablasyon tablosu ölçüm gibi sunulmuş, ama uydurma; ayrıca "%1 FPR'de TPR" sütunu ile "π=0,01'de kesinlik" sütunu matematiksel olarak tutarsız (TPR %92,1 → kesinlik %48,2 olmalı, tabloda %31). | "Şablon; değerler örnektir" etiketi; kesinlik sütunu TPR'den yeniden hesaplanır. |
| P6-7 | B | Politika motoru "beş karar" tanımlıyor; YAML altı farklı, tanımsız karar adı kullanıyor. Politika sürümü `2026-10-31.v1` (seri: `2026-10-29.3`). | YAML beş karara indirgenir; sürüm hizalanır. |
| P6-8 | C | Simülatör: eşik mantığı kademe mantığıyla çelişiyor (`send_email` eşiği 0,02 iken 0,10 altı "emin zararsız"); `time.sleep` yüzünden çıktı deterministik değil. | Karar, kademe sonrası skora eylem eşiği uygulanarak verilir; gecikme toplanır (31 / 331 ms); çıktı eklenir. |
| P6-9 | A | Latent yönlendirme sınırsız çözüm gibi; aynı yön ağırlık erişimi olan saldırganca silinebilir, probe da olasılıksaldır. | Sınır cümlesi. |
| P6-10 | — | Figür "0,6", metin "0,65"; matriste jailbreak için iki "Birincil"; Prompt Guard 2 yılı 2024 (→2025). | Düzeltilir. |
| P6-11 | G8, G10 | Kırık bağlantı, sonraki bölüm. | — |

### Bölüm 7 — RAG ve bellek
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P7-1 | A | PoisonedRAG: uydurma `max_δ min_q cos(…)` formülü ve "cos ≥ 0,88". Gerçek: `P = S ⊕ I` (S erişim koşulu — kara kutuda hedef sorunun kendisi; I üretim koşulu). | Kaynağa uygun; sözlük. |
| P7-2 | A | Perkolasyon eşiği `p_c ≈ ln|V|/|V|` — bu Erdős–Rényi *kenar* olasılığı; düğüm silmeye uygulanamaz. | Formül kaldırılır, olgu sözle anlatılır. |
| P7-3 | A | "Graf içi bitset filtreleme: Qdrant, Milvus **ve pgvector**" — pgvector dizin taramasından *sonra* filtreler; 0.8.0 yinelemeli tarama ekledi (teyit edildi). | Düzeltilir. |
| P7-4 | A | Epoch geçersizleme: epoch artınca belge *herkes* için kaybolur (yetkisi sürenler dahil); bu söylenmiyor. Büyük Resim'in alıntıladığı "bayat izin, bayat retten kötüdür" cümlesi bölümde yok. | "Kapalı arıza + arka planda metadata yeniden damgalama (yeniden gömme yok)" + cümle. |
| P7-5 | A | Late chunking: "cross-attention" (encoder'da self-attention); "maaş *ayrıntıları* sızar" — sızan konu sinyalidir. | Düzeltilir, iddia daraltılır. |
| P7-6 | A | Vec2Text modelleri "contriever" (→ GTR-base); "Llama-Guard RAG checker" diye ürün yok; `​` "joiner" değil. | Düzeltilir. |
| P7-7 | A | "Deontik kip filtresi" bir anahtar kelime regex'i; "müşteri paketi **asla** almadı" gibi olgusal cümleleri de reddeder. Sınır olarak sunulmuş. | "Ucuz ilk elek; ret değil incelemeye yönlendir; asıl sınır karantina + belleğin yetki taşımaması" notu. |
| P7-8 | A | §5.3 ve R-8: kripto-imha vektörleri de "çözülemez" yapar (bkz. P3-5). Önbellek anahtarı `HMAC(sorgu ∥ yetki)` — semantik önbellek benzerlikle eşleşir; doğru yapı yetki kümesine göre *bölme*. | Düzeltilir. |
| P7-9 | G2, G6 | τ düzeyleri; 3 ASCII kutu. | — |

### Bölüm 8 — Ajanlarda en az yetki
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P8-1 | A | MCP taşıma: "stdio ve SSE". Güncel spesifikasyon (2026-07-28): stdio + **Streamable HTTP**. | Başlık + TOC + metin; RFC 8707 hedef kitle bağlama cümlesi. |
| P8-2 | C | `Macaroon.verify` bilinmeyen operatörde sessizce geçiyor (metindeki `expires < 15m` caveat'i hiç denetlenmez) — açık arıza. | Bilinmeyen operatör → ret. |
| P8-3 | A | Saga "işlemsel geri almayı **garanti eder**". Gönderilmiş e-posta telafi edilemez; telafi de PEP'ten geçen bir araç çağrısıdır. | "Sıra önemli: geri alınamaz adım en sona (pivot)" paragrafı. |
| P8-4 | E | Büyük Resim'in andığı iki yürüyüş bölümde yok: 50 adım × 30.000 token = 1,5 M vs 8 adım = 240.000; %0,5 zaman aşımı × 1.000 iade ≈ 5 mükerrer. | §6.2'ye eklenir. |
| P8-5 | A | "Yetki kesişimi **teoremi**"; onay yorgunluğu "1,5 saniyenin altında" (kaynaksız sayı); onay kuralı `r≥2 veya τ=0` (Bölüm 5 kuralı: `r > τ(c)`). | "ilke"; sayı çıkarılır; kural hizalanır. |
| P8-6 | P4-8, G7, G8 | Shadowing tanımı, aria-label, kırık bağlantı. | — |

### Bölüm 9 — Sunum (en çok düzeltme gereken bölüm)
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P9-1 | A | **"H100, PCIe IDE ile AES-GCM-256 hat şifrelemesi yapar".** NVIDIA'nın kendi anlatımı: SPDM oturumu + *şifreli bounce buffer*; komut tamponları ve CUDA çekirdekleri şifreli ve imzalı; **paket üstü HBM şifrelenmez**; doğrulama NRAS ile. PCIe IDE/TDISP, H100'ün kullandığı mekanizma değil, bağlantı düzeyindeki standart. | §2 kaynağa göre yeniden yazılır; IDE/TDISP "standart, yeni platform kuşakları" olarak konumlanır; quiz, düşünce deneyi, sözlük, altı satır, özet. |
| P9-2 | A | **Önek önbelleği.** "vLLM … Radix Attention" (SGLang'e ait; vLLM blok hash'i kullanır). "B kiracısının attention'ı A'nın KV sayfalarına bakar" ve "önek ağacını bozar" — yanlış: önbellek içerik adresli, isabet ancak *aynı* önekte olur. Gerçek risk **zamanlama yan kanalı** (CVE-2025-46570; 8 token'da ROC AUC 0,99). vLLM'in yerleşik çözümü `cache_salt` (ilk bloğun hash'ine karışır); doküman kiracı başına ayrı örnek ya da gateway'in tuzu kapsamlamasını öneriyor. | §4.2 baştan; başlık + TOC; S-8, quiz, sözlük, altı satır, özet, Python sınıfı. |
| P9-3 | A | Yan kanallar: uydurma `Var(Δt) ∝ H(…)` formülü, kanıtsız MoE iddiası, uzaktan gözlenemeyen "sayfa tahsis örüntüsü". | Belgelenmiş kanallar: token uzunluğu (paket boyutu), önbellek zamanlaması; önlem: dolgu/gruplama. |
| P9-4 | A | Portlar: "`:14579` Ray / KV" — `--kv-port` (KV aktarımı); `:29500` veri paralel ana portu; "RCE'ye izin verir" → doküman: şifresiz, kimlik doğrulamasız. | Düzeltilir. |
| P9-5 | A | "Spekülatif geri alma: istemci tamponu DOM'dan siler" — bekletilen metin istemciye hiç gitmemiştir; gitmiş metin geri alınamaz (bölümün kendi tezi). | Algoritma netleştirilir + pencerenin sınırı (s > W). |
| P9-6 | A | LiteLLM YAML: `hosted_external/gpt-4o` diye sağlayıcı öneki yok; `qwen-72b` (seri 7B model kullanıyor). | `openai/gpt-4o`; `mistral-7b-instruct-v0.3`. |
| P9-7 | — | "SSN" (TCKN olmalı); `v` / `v_gen` karışık; kota tablosu "eşzamanlılık: 60 istek/dk". | Düzeltilir. |
| P9-8 | D | NVIDIA bağlantısı 404. | Çalışan blog adresi. |
| P9-9 | G6, G7, G8 | ASCII, figürde Türkçe, 2 kırık bağlantı. | — |

### Bölüm 10 — Ölçüm
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P10-1 | B | G3 (rapor). Ayrıca "1.200 saldırı, 140 tohum" — bölümün kendi figürü "aynı tohumun varyantları bağımsız örnek sayılmaz" diyor; n=1.200 ile aralık hesaplamak bunu çiğner. Gateway `kong`, politika `OPA-…` (seri: LiteLLM, `2026-10-29.3`). ART bütçesi "10.000 token". | Tohum başına tek varyant; sürümler hizalanır; "2.000 konuşma". |
| P10-2 | A | "Kesinlik arttıkça örneklem **üstel** büyür" — `n = 3/p`, ters orantı. | Düzeltilir. |
| P10-3 | A | "%0 ASR bildiren 12 savunma" — makale: "sıfıra yakın". | Düzeltilir. |
| P10-4 | G9 | Harness'ın demosu/çıktısı yok. | Rapor sayılarıyla demo + çıktı. |
| P10-5 | G7 | Figürlerde Türkçe / κ. | — |

### Bölüm 11 — İzlenebilirlik
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P11-1 | A | Ortam değişkeni `OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT` → `…_CAPTURE_MESSAGE_CONTENT` (varsayılan `NO_CONTENT`). | Düzeltilir. |
| P11-2 | C | §7.2 başlığı ve metni "outbox'a yazar" diyor; kodda outbox yok; demo/çıktı yok. | Küçük outbox + sıra no + boşluk tespiti + demo çıktısı. |
| P11-3 | F | TR TOC'unda H3'ler eksik (10 vs 17). | Eklenir. |

### Bölüm 12 — Kanıt
| # | Tür | Sorun | Düzeltme |
|---|---|---|---|
| P12-1 | C | Kapı kodu `unverified_gap_count > 0` ise reddediyor; yürüyüşte "16 açık madde, hepsi sahipli → geçti". Kod ile yürüyüş çelişiyor. | Alan `unowned_gap_count`; demo + çıktı. |

---

## 3. Metadata (F)

- `posts.json` (EN+TR): Bölüm 1 (41× → 9,3×), Bölüm 2 (Von Neumann), Bölüm 8 ("theorem"), Bölüm 9 (PCIe IDE / Radix) özetleri; değişen her yazıya `updated: 2026-11-14`; kök `updated`.
- `sitemap.xml`: değişen 26 URL için `lastmod`.
- `build_blog_index.py` yeniden koşulur.
- Doğrulama: 13 slug için `validate_post.sh`, bütün Python bloklarının yeniden koşumu, EN SVG'lerinde Türkçe taraması, iç bağlantı taraması.
- **Commit/push yok** — onay bekler.

## 4. Bilerek kapsam dışı bırakılanlar

| Konu | Neden |
|---|---|
| Pedagojik bölüm adlarının birleştirilmesi (4–6 "Pedagogical stops", 7–9 "Pedagogical checkpoints", 1–3 satır içi kutular, 10–12 yok) | Yapısal yeniden düzen; anchor/TOC riski yüksek, olgusal kazanç yok. Ayrı bir tur. |
| Kaynak etiketleri (`[K1]` yalnız 1–6'da) | Blogun başka EN yazıları da kullanıyor; tutarsız ama hatalı değil. |
| SVG altlarındaki "💡 … AXIOM" şeritleri ve genel üslup revizyonu | 34 figürün yeniden üretimi; üreteçler kalıcı değil. Yalnız olgusal hatası olan figür metinleri düzeltildi. |
| "Production-grade" başlıkları (7.3, 8.3, 9.2) | Kod oyuncak simülasyon; başlık TOC'a bağlı. Metinde "simülasyon" olduğu belirtilir, başlık korunur. |
| AISVS bölüm numaraları (C1–C12 atıfları) | Bölüm adlarını birincil kaynaktan teyit edemedim; dokunulmadı. |

## 5. Uygulama durumu (2026-11-15)

**Uygulandı:** G1–G10 ve P1–P12 tablolarındaki bütün maddeler, EN + TR birlikte. 26 makale dosyası, iki `posts.json`, `sitemap.xml` değişti (29 dosya). Commit/push yapılmadı.

| Doğrulama | Sonuç |
|---|---|
| `validate_site.py` | 0 hata, 0 uyarı |
| `validate_post.sh` × 13 (TOC, KaTeX, simetri) | TOC'ta 0 kırık bağlantı, 0 KaTeX hatası; Bölüm 11 TOC'u artık EN/TR eşit |
| Bütün Python blokları (EN + TR) yeniden koşuldu | Hepsi makaledeki çıktıyla birebir; tek fark Bölüm 4 disassembly'sindeki bilerek eklenen ok açıklamaları |
| İç bağlantı taraması | 0 kırık slug |
| ASCII kutu çizimi / dilsiz kod bloğu | 0 |
| EN SVG'lerinde Türkçe metin | 0 |
| Değişen SVG etiketlerinin genişlik tahmini | viewBox içinde (iki etiket kısaltıldı) |

**Planın yazımından sonra netleşen ayrıntılar**

- G11 (ton) yalnız dokunulan pasajlarda uygulandı; Bölüm 4–9'un genel üslubu olduğu gibi duruyor.
- Bölüm 9'da iki başlık değişti (2.2 ve 4.2), Bölüm 8'de iki başlık (2.1 "teorem" → "ilke", 4.2 "SSE" → "Streamable HTTP"); TOC anchor'ları güncellendi ve doğrulandı. Bu başlıklara dışarıdan verilmiş derin bağlantı varsa kırılır.
- Bölüm 4 ve 9'daki figür altlarında kalan iki genel cümle ("Confidential AI … shields model weights and KV memory") metinle çelişmediği için bırakıldı.

**Teyit edemediğim, dokunmadığım noktalar**

- AISVS 1.0 bölüm numaraları (C1–C12 atıfları).
- `iso.org/standard/42001`, `/42005`, `/27090`, `/27701` bağlantıları bot isteğine 403 dönüyor; tarayıcıda açıldıkları doğrulanmadı.
- LiteLLM'in akışta `post_call` davranışı: doküman sayfası bunu açıkça yazmıyor; makaledeki "akışta engelleyemez" ifadesi önceki oturumun teyidine dayanıyor.
- Many-shot güç yasası formülü (`NLL ≈ C·k^(-α) + K`) Anthropic yazısındaki sözlü tarife dayanarak yazıldı; makalenin kendi denklemiyle karşılaştırılmadı.
