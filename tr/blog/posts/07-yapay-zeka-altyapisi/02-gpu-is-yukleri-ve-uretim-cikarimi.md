Bir yapay zekâ modelini eğitirken (training), bir prompt'un tamamını modele ilk kez beslerken (prefill) ve kullanıcıya kelime kelime yanıt üretirken (decode) aynı GPU silikonu kullanılır. Hatta kullanılan Transformer blokları, tensör boyutları ve ağırlık matrisleri de tıpatıp aynıdır. Buna rağmen donanım bu üç evrede birbirinden tamamen farklı fiziksel sınırlarla karşılaşır: Eğitimde milyarlarca FLOP hesaplama gücü isterken, ilk token'ı üretirken devasa matris çarpımlarına, ardından gelen her bir token'da ise çaresizce bellek bant genişliğine kilitlenir.

Geliştirdiğiniz 70 milyar parametreli bir modelin üretim sunucusunda çalıştığını hayal edin. Bir kullanıcı 2.000 token uzunluğunda bir sözleşme metni yüklüyor ve bir özet istiyor. İstek sisteme girdiği ilk anda GPU'nun Tensor Core birimleri %70 verimle çalışarak saniyeler içinde binlerce token'lık prompt'u tek hamlede yutar. Ancak sıra cevabın ilk kelimesinden sonrasını üretmeye geldiğinde, saniyede yüzlerce TFLOP hesaplayabilen o devasa silikon bir anda yavaşlar; ekranda harflerin tek tek belirmesi için her adımda 140 gigabaytlık ağırlık matrisinin tamamını baştan sona bellekten çekmek zorunda kalır.

**Bu yazının temel fikri:** Büyük dil modellerinde "çıkarım maliyeti" tek bir sayı değildir. Sistem iki ayrı fiziksel rejim arasında bölünmüştür: Prompt'u işleyen hesaplama kısıtlı (compute-bound) **prefill** evresi ve token'ları teker teker üreten bellek bant genişliği kısıtlı (memory-bandwidth bound) **decode** evresi. Bir üretim altyapısını ölçeklendirmenin sırrı; KV cache büyümesini, continuous batching mantığını ve bellek parçalanmasını doğru yöneterek donanımı aç bırakmamaktır.

> **Serideki yerimiz:** Bu makale, beş bölümlük *Yapay Zekâ Altyapısı* serisinin ikinci yazısıdır. Birinci bölümde SM mimarisini, bellek piramidini ve Roofline modelini kurmuştuk. Burada o donanım üzerinde koşan iş yüklerinin (training, prefill ve decode) bellek, KV cache ve gecikme dinamiklerini inceliyoruz. Üçüncü bölümde bu iş yüklerini birden çok GPU'ya dağıtacak ağ kumaşlarını ve paralellik türlerini ele alacağız.

> **Bu yazının sonunda**
>
> - Aynı Transformer katmanının eğitimde, prefill'de ve decode'da neden taban tabana zıt darboğazlara girdiğini kanıtlayabileceksiniz.
> - Bir modelin ağırlık, KV cache ve aktivasyon bellek ayak izini kalem kalem hesaplayabileceksiniz.
> - Grouped-Query Attention (GQA) ve Multi-Head Latent Attention (MLA) mimarilerinin KV cache faturasını nasıl düşürdüğünü gösterebileceksiniz.
> - Continuous batching, PagedAttention ve prefix caching mekanizmalarının donanım seviyesindeki çalışma mantığını kavrayacaksınız.
> - Disaggregated serving (prefill ve decode havuzlarını ayırma) stratejisinin ne zaman hayat kurtarıp ne zaman maliyet tuzağına dönüştüğünü savunabileceksiniz.

**Bu yazıda**

- [1. Aynı Transformer neden üç farklı iş yüküdür?](#1-aynı-transformer-neden-üç-farklı-iş-yüküdür)
- [2. Bellek bütçesi nasıl çıkarılır?](#2-bellek-bütçesi-nasıl-çıkarılır)
- [3. Serving motoru GPU’yu nasıl verimli tutar?](#3-serving-motoru-gpuyu-nasıl-verimli-tutar)
- [4. Kapasite ve gecikme nasıl birlikte planlanır?](#4-kapasite-ve-gecikme-nasıl-birlikte-planlanır)
- [5. Görsel ve video iş yükleri neyi değiştirir?](#5-görsel-ve-video-iş-yükleri-neyi-değiştirir)
- [6. 70B serving örneği](#6-70b-serving-örneği)
- [7. Sistem tasarımı için altı temel soru](#7-sistem-tasarımı-için-altı-temel-soru)
- [Okuyucu artık hangi sorulara kendi başına yanıt verebilir?](#okuyucu-artık-hangi-sorulara-kendi-başına-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. Aynı Transformer neden üç farklı iş yüküdür?

Modern bir Decoder-only Transformer mimarisinde her katman temel olarak Query, Key, Value projeksiyonları, çok kafalı dikkat (Multi-Head Attention / GQA), çıkış projeksiyonu ve ardından gelen iki katmanlı Feed-Forward / MLP ağından (Gate, Up, Down projeksiyonları) oluşur.

<svg viewBox="0 0 720 428" role="img" aria-label="Bir Transformer bloğunun yukarıdan aşağı akışı: token ID&#x27;leri; N × d boyutunda X aktivasyonlarına embedding okuması; GEMM olan Q, K, V lineer projeksiyonları; maliyeti dizi uzunluğuyla büyüyen attention, yani softmax(Q·Kᵀ/√d)·V; çıkış projeksiyonu, residual ve normalizasyon; ve yine GEMM olan MLP, W2 · act(W1 · X). Projeksiyonlardan MLP&#x27;ye kadar olan kısım L kez tekrarlanır (70B modelde 80); son norm logits ve sonraki token olasılıklarını üretir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tb-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir Transformer bloğu ne hesaplar?</text>
<rect x="150" y="40" width="360" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="162" y="57" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Token ID&#x27;leri</text>
<text x="162" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">[t1, t2, …, tN]</text>
<line x1="330.0" y1="80" x2="330.0" y2="92" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="92" width="360" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="162" y="109" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Embedding</text>
<text x="162" y="125" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">tablo okuma → X  (N × d)</text>
<rect x="524" y="103" width="93.312" height="18" rx="9" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="570.656" y="115.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">bellek okuma</text>
<line x1="330.0" y1="132" x2="330.0" y2="144" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="144" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="161" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Q, K, V projeksiyonları</text>
<text x="162" y="177" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Q = XW_Q,  K = XW_K,  V = XW_V</text>
<rect x="524" y="155" width="39.104" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="543.552" y="167.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM</text>
<line x1="330.0" y1="184" x2="330.0" y2="196" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="196" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="162" y="213" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Attention</text>
<text x="162" y="229" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">softmax(Q·Kᵀ / √d) · V</text>
<rect x="524" y="207" width="161.072" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="604.5360000000001" y="219.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">dizi uzunluğuyla büyür</text>
<line x1="330.0" y1="236" x2="330.0" y2="248" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="248" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Çıkış proj. + residual + norm</text>
<text x="162" y="281" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">X + Attn·W_O → norm</text>
<rect x="524" y="259" width="120.416" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="584.208" y="271.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM + pointwise</text>
<line x1="330.0" y1="288" x2="330.0" y2="300" marker-end="url(#tb-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="300" width="360" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="162" y="317" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">MLP / Feed-Forward</text>
<text x="162" y="333" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">W2 · act(W1 · X)</text>
<rect x="524" y="311" width="39.104" height="18" rx="9" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="543.552" y="323.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GEMM</text>
<path d="M140 144 H128 V340 H140" style="fill:none;stroke:var(--c-accent);stroke-width:1.6"/>
<text x="118" y="238.0" text-anchor="end" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">× L blok</text>
<text x="118" y="254.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">(70B&#x27;de 80)</text>
<path d="M330.0 340 V358" marker-end="url(#tb-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="150" y="360" width="360" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330.0" y="382.0" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Son norm → logits → sonraki token olasılıkları</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Teal = büyük matris çarpımları (Tensor Core). Norm, softmax ve veri taşıma da gerçek zaman alır.</text>
</svg>

*Şekil 1.1: Bir Transformer bloğunun yukarıdan aşağıya veri akışı. Token ID'leri embedding katmanından geçerek $N$ katman boyunca dikkat ve MLP GEMM işlemlerine girer; son katmanda logits üretilir.*

Aynı katman mimarisi kullanılmasına rağmen, sistemin performansı bu katmanların tek bir token için mi yoksa binlerce token için mi aynı anda çalıştırıldığına bağlı olarak kökten değişir:

| Evre | Girdi Boyutu ($B, S$) | Baskın İşlem Türü | Donanımsal Darboğaz | Kritik Sistem Metriği |
| :--- | :--- | :--- | :--- | :--- |
| **Training (Eğitim)** | Büyük batch, uzun dizi ($B \\ge 64, S \\ge 4.096$) | Forward + Backward GEMM + Optimizer | Compute FLOP/s, HBM kapasitesi (Aktivasyonlar), Ağ | Sürdürülen MFU, Step süresi |
| **Prefill (Prompt İşleme)** | Büyük token dizisi ($B=1..16, S \\ge 2.048$) | Paralel Attention + Yoğun GEMM | Compute FLOP/s, HBM bant genişliği, TMA transferi | Time-To-First-Token (TTFT) |
| **Decode (Token Üretimi)** | Teker teker token ($B=1..64, S=1$) | Vektör-Matris Çarpımı (GEMV) + KV Cache Lookup | HBM Bellek Bant Genişliği (Memory-Bandwidth Bound) | Time-Per-Output-Token (TPOT / ITL) |

<svg viewBox="0 0 720 314" role="img" aria-label="70B modelde 2.000 token&#x27;lık prompt ve 200 token&#x27;lık yanıt. CPU ve servis tarafında istek tokenizer&#x27;dan ve isteği kuyruğa alıp sürekli batch&#x27;e ekleyen scheduler&#x27;dan geçer. GPU&#x27;da prefill 2.000 token&#x27;ı büyük GEMM&#x27;lerle birlikte işler ve KV cache&#x27;i doldurur. Ardından decode her adımda bir token üretir, bunu 200 kez yapar; ağırlıkları ve büyüyen KV cache&#x27;i okuduğu için çoğunlukla memory-bound&#x27;dur. Her yeni token decode döngüsüne geri döner, detokenize edilip akıtılır. Altta kullanıcının zaman çizgisi: kuyruk, tokenize, prefill ve ilk decode TTFT&#x27;yi oluşturur; sonraki token&#x27;lar arasındaki süre TPOT&#x27;tur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="il-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="il-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="il-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tek istek: 2.000 token prompt, 200 token yanıt (70B, BF16 ≈ 140 GB)</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">CPU / servis</text>
<text x="16" y="148" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">GPU</text>
<line x1="16" y1="120" x2="704" y2="120" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<rect x="16" y="60" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="71.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">İstek</text>
<text x="71.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">prompt metni</text>
<rect x="140" y="60" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="195.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Tokenizer</text>
<text x="195.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">metin → ID</text>
<rect x="264" y="60" width="140" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="334.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Scheduler</text>
<text x="334.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kuyruk + sürekli batch</text>
<line x1="126" y1="83" x2="138" y2="83" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="250" y1="83" x2="262" y2="83" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="250" y="132" width="164" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="332.0" y="151.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Prefill</text>
<text x="332.0" y="166.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2.000 token birlikte</text>
<text x="332.0" y="181.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">büyük GEMM, compute ağırlıklı</text>
<line x1="334" y1="106" x2="334" y2="130" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="132" width="92" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="476.0" y="151.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">KV cache</text>
<text x="476.0" y="166.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">token başına</text>
<text x="476.0" y="181.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">büyür</text>
<line x1="414" y1="161" x2="428" y2="161" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="538" y="132" width="166" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="621.0" y="151.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Decode</text>
<text x="621.0" y="166.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım başına 1 token × 200</text>
<text x="621.0" y="181.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">ağırlık + KV → memory-bound</text>
<line x1="522" y1="161" x2="536" y2="161" marker-end="url(#il-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M690 190 V212 H560 V192" marker-end="url(#il-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="625" y="226" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px">sonraki token</text>
<rect x="538" y="60" width="166" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="621.0" y="80.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Detokenize + akış</text>
<text x="621.0" y="95.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">token → metin</text>
<line x1="621" y1="130" x2="621" y2="108" marker-end="url(#il-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="16" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">kullanıcının hissettiği</text>
<rect x="16" y="252" width="130" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="267.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">kuyruk + tokenize</text>
<rect x="146" y="252" width="210" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="251.0" y="267.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">prefill + ilk decode</text>
<rect x="360" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="394.4" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="428.8" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="463.2" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="497.6" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="532" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="566.4" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="600.8" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="635.2" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="669.6" y="252" width="30" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="700.0" y="268" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">…</text>
<path d="M16 286 H356" marker-start="url(#il-arr)" marker-end="url(#il-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="186" y="302" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">TTFT</text>
<path d="M360 286 H390" marker-start="url(#il-arr)" marker-end="url(#il-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="392" y="302" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">TPOT = token&#x27;lar arası süre</text>
</svg>

*Şekil 1.2: Tek bir istekte (2.000 token prompt, 200 token yanıt) CPU ve GPU iş yükü ayrımı. Prefill prompt'u büyük GEMM'lerle toplu işlerken, decode her adımda bir token üreterek 200 kez ardışık döngüye girer.*

### 1.1. Training: Hesaplama neden büyür?

Model eğitimi, çıkarımdan temelde iki büyük farkla ayrılır: Gradyanların hesaplandığı **geri yayılım (backward pass)** ve ağırlıkların güncellendiği **optimizer adımı**.

Standart bir Transformer bloğunda ileri geçişte (forward pass) her bir token ve her bir model parametresi için kabaca $2$ FLOP işlem yapılır (bir çarpma, bir toplama):

$$C_{\text{forward}} \approx 2 N \text{ FLOP/token}$$

Burada $N$ modelin parametre sayısıdır. Geri yayılım evresinde ise zincir kuralı gereği iki ayrı gradyan türetilir:
1. Bir önceki katmana aktarılacak aktivasyon gradyanları ($\nabla_X$): $\approx 2 N \text{ FLOP/token}$
2. Ağırlıkları güncellemek için kullanılacak ağırlık gradyanları ($\nabla_W$): $\approx 2 N \text{ FLOP/token}$

Yani geri yayılım evresi, ileri geçişin tam iki katı hesaplama gerektirir ($4 N \text{ FLOP/token}$). Bir eğitim adımında token başına harcanan toplam teorik hesaplama maliyeti şudur:

$$C_{\text{train}} \approx 2N + 4N = \mathbf{6N \text{ FLOP/token}}$$

Toplam $D$ adet token içeren bir veri setinde model eğitmenin kaba maliyeti:

$$C_{\text{toplam}} \approx 6 N D \text{ FLOP}$$

Llama 3 70B modelini ($N = 70 \times 10^9$) 15 trilyon token ($D = 15 \times 10^{12}$) ile eğitmenin toplam hesaplama faturası:

```text
6 × (70 × 10^9) × (15 × 10^12) = 6,3 × 10^24 FLOP = 6,3 YottaFLOP!
```

Eğitim evresinde batch boyutu binlerce token'dan oluştuğu için GPU matris çarpımlarında (GEMM) çalışır ve Tensor Core'lar compute-bound bölgesinde doyar. Ancak eğitimin asıl bedeli bellek tarafındadır: Geri yayılımda kullanılmak üzere ileri geçişteki tüm ara aktivasyonların bellekte saklanması gerekir. Bellek patlamasını engellemek için **Activation Checkpointing (Gradient Checkpointing)** kullanılır: İleri geçiş aktivasyonları saklanmaz, backward sırasında katman baştan bir kez daha hesaplanır. Bu teknik belleği devasa oranda rahatlatır; ancak toplam FLOP maliyetini $6N$'den $8N$'e (%33 ek hesap) çıkarır.

### 1.2. Prefill: Prompt neden toplu işlenir?

Kullanıcı sisteme bir prompt gönderdiğinde (örneğin 2.000 tokenlik bir makale), bu 2.000 token'ın tamamı zaten elimizdedir. Bir sonraki token'ı tahmin etmek için geçmişi beklememize gerek yoktur; model bu 2.000 token'ı tek bir devasa matris halinde GPU'ya yükler.

Tüm prompt tek seferde işlendiği için:
- Aktivasyon matrisi geniştir ($S = 2.000$).
- $X \in \mathbb{R}^{S \times D}$ ile ağırlık matrisi $W \in \mathbb{R}^{D \times F}$ çarpımı tam bir **GEMM (Genel Matris Çarpımı)** operasyonudur.
- Aritmetik yoğunluk Roofline dirsek noktasının ($\alpha_{\text{crit}} \approx 295$ FLOP/bayt) çok üzerindedir ($I \approx 400 \dots 600$ FLOP/bayt).

Hopper H100 mimarisinde bu evre, **TMA (Tensor Memory Accelerator)** ve **WGMMA (Warpgroup Matrix Multiply-Accumulate)** çift tamponlama (double-buffering) mekanizmasıyla donanım seviyesinde örtüştürülür:

<svg viewBox="0 0 720 264" role="img" aria-label="Dört şeritli zaman çizelgesi: HBM, TMA, shared memory ve WGMMA. K/V blok 1, 2 ve 3 HBM&#x27;de durur. TMA blok 1&#x27;i tampon A&#x27;ya, blok 2&#x27;yi tampon B&#x27;ye, blok 3&#x27;ü yeniden tampon A&#x27;ya yükler. WGMMA her bloğu yüklemeden bir dilim sonra hesaplar; blok 1&#x27;i hesaplarken TMA blok 2&#x27;yi, blok 2&#x27;yi hesaplarken blok 3&#x27;ü yükler. Gölgeli örtüşme pencereleri, k+1. bloğun veri taşımasının k. bloğun hesabının arkasına gizlendiğini gösterir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tm-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tm-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Hopper&#x27;da çift tampon: WGMMA k. bloğu hesaplarken TMA k+1&#x27;i yükler</text>
<text x="16" y="70" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">HBM</text>
<line x1="144" y1="82" x2="704" y2="82" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">TMA</text>
<line x1="144" y1="132" x2="704" y2="132" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">Shared</text>
<line x1="144" y1="182" x2="704" y2="182" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="220" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600;font-family:var(--font-mono)">WGMMA</text>
<line x1="144" y1="232" x2="704" y2="232" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="16" y="254" text-anchor="start" style="fill:var(--c-text);font-size:1px"></text>
<rect x="150" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="214.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V blok 1</text>
<rect x="286" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="350.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V blok 2</text>
<rect x="422" y="50" width="128" height="28" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="486.0" y="68.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">K/V blok 3</text>
<rect x="150" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="214.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">yükle 1</text>
<rect x="154" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="214.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">tampon A</text>
<line x1="214.0" y1="128" x2="214.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="286" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="350.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">hesapla 1</text>
<path d="M272 178 L296 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="286" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="350.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">yükle 2</text>
<rect x="290" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="350.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">tampon B</text>
<line x1="350.0" y1="128" x2="350.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="422" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="486.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">hesapla 2</text>
<path d="M408 178 L432 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="422" y="100" width="128" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="486.0" y="118.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">yükle 3</text>
<rect x="426" y="150" width="120" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="486.0" y="168.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">tampon A</text>
<line x1="486.0" y1="128" x2="486.0" y2="148" marker-end="url(#tm-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="558" y="200" width="128" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="622.0" y="218.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">hesapla 3</text>
<path d="M544 178 L568 198" marker-end="url(#tm-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="286" y="94" width="128" height="140" rx="6" style="fill:var(--c-accent-2);fill-opacity:.06;stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="422" y="94" width="128" height="140" rx="6" style="fill:var(--c-accent-2);fill-opacity:.06;stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:4 3"/>
<text x="418.0" y="252" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">örtüşme: k+1&#x27;in veri taşıması k&#x27;nın hesabının arkasına gizlenir</text>
<text x="704" y="44" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">zaman →</text>
</svg>

*Şekil 1.3: Hopper mimarisinde çift tamponlama. WGMMA $k$. tensör bloğunu Tensor Core'larda hesaplarken, TMA donanım motoru bağımsız olarak $k+1$. bloğu HBM'den shared memory'ye taşır; veri transferi hesaplamanın arkasına gizlenir.*

Prefill evresinin temel performans göstergesi **Time-To-First-Token (TTFT)** metriğidir. Kullanıcı butona bastığı andan ilk kelimenin ekranda belirdiği ana kadar geçen süredir. Prefill ne kadar hızlı biterse TTFT o kadar düşük olur ve model o kadar "canlı" hissedilir.

### 1.3. Decode: Neden bellek sınırlıdır?

İlk token üretildikten sonra oyunun kuralları 180 derece değişir. Büyük dil modelleri **otoregresif (autoregressive)** çalışır: $N+1$. token'ı üretebilmek için $N$. token'ın çıktısına muhtaçsınızdır. 200 tokenlik bir yanıt almak istiyorsanız, modeli art arda tam **200 kez** çalıştırmak zorundasınızdır.

Bu durum donanımda felaket bir verimsizlik yaratır:
- Her adımda modele sadece **tek bir token** ($B=1, S=1$) girer.
- Aktivasyon matrisi artık bir matris değil, tek bir satır vektörüdür ($X \in \mathbb{R}^{1 \times D}$).
- Yapılan işlem GEMM'den **GEMV'ye (Matris-Vektör Çarpımı)** çöker.
- Bölüm 1'de kanıtladığımız gibi, tekil decode evresinde aritmetik yoğunluk **$I \approx 1$ FLOP/bayt** seviyesine düşer!

990 TFLOP/s hesaplayabilen bir H100 GPU, tekil decode sırasında HBM bellek bant genişliğinin (3,35 TB/s) sınırına takılır ve yalnızca **3,35 TFLOP/s** üretebilir (donanım verimi: **%0,34**).

<svg viewBox="0 0 720 326" role="img" aria-label="Aynı 70B modelde iki istek için şematik zaman çizgileri. Kısa “Merhaba!” istemi kısa süre kuyrukta bekler, küçük bir prefill&#x27;le ilk token&#x27;a erken ulaşır, sonra 40 decode adımı çalışır; KV cache&#x27;i küçük ve kısa ömürlüdür. 50 sayfalık sözleşme uzun bir prefill gerektirir, ilk token&#x27;ı daha geç gelir, sonra 1.000 decode adımı boyunca büyük bir KV cache tutar. Renkler mantıksal prefill ve decode iş yüklerini ayırır; bunlar aynı GPU&#x27;yu paylaşabilir ya da ayrı havuzlarda çalışabilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tq-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tq-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tq-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Aynı 70B modelde iki istek (şematik, ölçekli değil)</text>
<text x="16" y="70" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">“Merhaba!”</text>
<text x="16" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kısa prompt · 40 token yanıt</text>
<rect x="196" y="56" width="46" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="219.0" y="74.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">kuyruk</text>
<rect x="246" y="56" width="30" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="280" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="292" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="304" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="316" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="328" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="340" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="352" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="364" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="376" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="388" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="400" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="412" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="424" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="436" y="60" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="452" y="76" text-anchor="start" style="fill:var(--c-text-mute);font-size:13px"></text>
<text x="452" y="76" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">40 decode adımı</text>
<line x1="278" y1="50" x2="278" y2="96" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="278" y="110" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TTFT</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">50 sayfalık sözleşme</text>
<text x="16" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">uzun prompt · 1.000 token yanıt</text>
<rect x="196" y="148" width="46" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="219.0" y="166.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">kuyruk</text>
<rect x="246" y="148" width="170" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="331.0" y="167.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="420" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="432" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="444" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="456" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="468" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="480" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="492" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="504" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="516" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="528" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="540" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="552" y="152" width="8" height="22" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="568" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:13px">…</text>
<text x="580" y="168" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">1.000 decode adımı</text>
<line x1="418" y1="142" x2="418" y2="188" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="418" y="202" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TTFT</text>
<text x="16" y="248" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">tutulan KV cache</text>
<rect x="196" y="238" width="40" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="242" y="251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">“Merhaba!”: küçük, kısa ömürlü</text>
<rect x="196" y="262" width="364" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="566" y="275" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sözleşme: büyük, 1.000</text>
<text x="566" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">adım boyunca tutulur</text>
<text x="16" y="314" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Renkler iki mantıksal iş yükünü (prefill, decode) ayırır; aynı GPU&#x27;da ya da ayrı havuzlarda çalışabilirler.</text>
</svg>

*Şekil 1.4: Kısa istek ("Merhaba!") ile uzun sözleşme analizi isteğinin KV cache tutma davranışları. Decode adımları uzadıkça bellekte tutulan KV cache hacmi ve süresi katlanarak büyür.*

Decode evresinin temel kullanıcı metriği **Time-Per-Output-Token (TPOT)** veya **Inter-Token Latency (ITL)** değeridir. Yani ekranda akan iki kelime arasında geçen süredir. İnsan gözünün rahatça okuyabilmesi için bu sürenin 30–50 milisaniye (saniyede 20–30 token) bandında olması hedeflenir.

### 1.4. Bir token fiziksel olarak hangi yolu izler?

Bir metin isteğinin sunucuya gelişiyle ekranda belirmesi arasındaki uçtan uca veri yolunu izleyelim:

1. **HTTP/gRPC Girişi ve Tokenizasyon:** Kullanıcının gönderdiği ham metin CPU üzerindeki serving sunucusuna (örneğin vLLM API gateway) ulaşır. Tokenizer metni sayısal token ID dizisine çevirir.
2. **Kuyruk ve Dinamik Çizelgeleme (Scheduler):** İstek öncelik ve kaynak durumuna göre kuyruğa alınır. Çizelgeleyici o an GPU'da çalışan sürekli batch'e (continuous batch) bu isteği ekler.
3. **Prefill Evresi (GEMM):** Prompt token'ları GPU'ya kopyalanır. Tüm katmanlar paralel olarak çalıştırılır; prompt'un ürettiği Key ve Value tensörleri HBM'de tahsis edilen **KV Cache** belleğine yazılır.
4. **İlk Token Örnekleme (Sampling):** Son katmanın ürettiği logits vektöründen sıcaklık (temperature), top-p ve top-k filtreleriyle ilk token seçilir. TTFT saati burada durur.
5. **Otoregresif Decode Döngüsü (GEMV):**
   - Yeni token'ın ID'si girdi olarak alınır.
   - Her katmanda model ağırlıkları HBM'den okunur.
   - Yeni token geçmişteki tüm token'ların KV cache değerleriyle attention yapar.
   - Yeni token'ın kendi Key ve Value değerleri cache'e eklenir.
   - Yeni bir token örneklenir. Bu döngü model `<EOS>` (end of sequence) token'ı üretene veya azami uzunluğa ulaşana kadar yüzlerce kez tekrarlanır.
6. **Detokenizasyon ve Streaming:** Üretilen her token anında metne çevrilerek Server-Sent Events (SSE) protokolü üzerinden istemciye akıtılır (streaming response).

---

## 2. Bellek bütçesi nasıl çıkarılır?

Bir AI sunucusu boyutlandırılırken yapılan en büyük hata, GPU belleğine yalnızca model ağırlıklarının konulacağını varsaymaktır. Gerçek bir çıkarım sisteminde VRAM dört ana paydaş arasında bölüşülür:

$$M_{\text{toplam}} = M_{\text{ağırlık}} + M_{\text{KV}} + M_{\text{aktivasyon}} + M_{\text{çalışma\_alanı}}$$

### 2.1. Ağırlık belleği

Model ağırlıklarının kapladığı ham alan, parametre sayısı ile sayısal duyarlığın çarpımıdır:

$$M_{\text{ağırlık}} = N \times \frac{b}{8} \text{ bayt}$$

Burada $N$ parametre sayısı, $b$ ise parametre başına düşen bit genişliğidir:

<svg viewBox="0 0 720 311" role="img" aria-label="Ham ağırlık belleğini gösteren yatay çubuklar. 8B model: BF16&#x27;da 16 GB, INT8&#x27;de 8 GB, INT4&#x27;te 4 GB. 70B: 140, 70 ve 35 GB. 405B: 810, 405 ve 202,5 GB. Kesikli referans çizgileri tek H100&#x27;ü (80 GB) ve 8 GPU&#x27;lu DGX&#x27;i (640 GB) gösterir: 70B model tek H100&#x27;e yalnızca INT4&#x27;te sığar, 405B model BF16&#x27;da tek DGX&#x27;e sığmaz. Ölçek, sıfır noktası, KV cache ve çalışma alanı dahil değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="qb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Hassasiyete göre ağırlık belleği ve GPU belleği</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca ham ağırlık (ölçek, sıfır noktası, KV cache, çalışma alanı hariç)</text>
<text x="16" y="110" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">8B</text>
<text x="142" y="93" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16</text>
<rect x="150" y="82" width="9.6" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="165.6" y="94" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">16 GB</text>
<text x="142" y="112" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT8</text>
<rect x="150" y="101" width="4.8" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<text x="160.8" y="113" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">8 GB</text>
<text x="142" y="131" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT4</text>
<rect x="150" y="120" width="2.4" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.75"/>
<text x="158.4" y="132" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">4 GB</text>
<text x="16" y="181" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">70B</text>
<text x="142" y="164" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16</text>
<rect x="150" y="153" width="84" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="240.0" y="165" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">140 GB</text>
<text x="142" y="183" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT8</text>
<rect x="150" y="172" width="42" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<text x="198.0" y="184" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">70 GB</text>
<text x="142" y="202" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT4</text>
<rect x="150" y="191" width="21" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.75"/>
<text x="177.0" y="203" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">35 GB</text>
<text x="16" y="252" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">405B</text>
<text x="142" y="235" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16</text>
<rect x="150" y="224" width="486" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="642.0" y="236" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">810 GB</text>
<text x="142" y="254" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT8</text>
<rect x="150" y="243" width="243" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<text x="399.0" y="255" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">405 GB</text>
<text x="142" y="273" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">INT4</text>
<rect x="150" y="262" width="121.5" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.75"/>
<text x="277.5" y="274" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">202,5 GB</text>
<line x1="198" y1="74" x2="198" y2="285" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="202.0" y="299" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">1 × H100 = 80 GB</text>
<line x1="534" y1="74" x2="534" y2="285" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="538.0" y="299" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">8 × H100 (DGX) = 640 GB</text>
<text x="150.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="270.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">200</text>
<text x="390.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">400</text>
<text x="510.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">600</text>
<text x="630.0" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">800</text>
</svg>

*Şekil 2.1: Hassasiyet seviyelerine (BF16, INT8, INT4) göre 8B, 70B ve 405B modellerin ham ağırlık belleği gereksinimi ve H100 (80 GB) ile DGX (640 GB) donanım sınırları.*

| Model Ölçeği | BF16 / FP16 (16-bit, 2 B) | FP8 / INT8 (8-bit, 1 B) | INT4 / FP4 (4-bit, 0,5 B) | Asgari GPU İhtiyacı (BF16) |
| :--- | :--- | :--- | :--- | :--- |
| **8B (Llama 3)** | 16 GB | 8 GB | 4 GB | 1x GPU (24 GB / 80 GB) |
| **70B (Llama 3)** | 140 GB | 70 GB | 35 GB | 2x H100 (FP8) veya 8x H100 (TP=8) |
| **405B (Llama 3)** | 810 GB | 405 GB | 202,5 GB | 2x DGX Düğümü (16x H100, 1.280 GB) |

### 2.2. KV cache formülü

Transformer modelinin her katmanında hesaplanan Key ve Value tensörleri, sonraki adımlarda tekrar hesaplanmamak üzere saklanır. Tek bir token için tüm model katmanlarında üretilen KV tensörlerinin kapladığı bellek miktarı şu formülle hesaplanır:

$$M_{\text{KV\_token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times b_{\text{bayt}}$$

Şöyle okuyun:
- $2$: Hem Key hem Value vektörleri için iki ayrı tensör tutulur.
- $n_{\text{layers}}$: Modeldeki toplam Transformer katmanı sayısı.
- $n_{\text{kv\_heads}}$: Katman başına düşen Key/Value dikkat kafası sayısı.
- $d_{\text{head}}$: Her bir dikkat kafasının gizli vektör boyutu ($d_{\text{model}} / n_{\text{heads}}$).
- $b_{\text{bayt}}$: Sayısal hassasiyet (16-bit için 2 bayt, FP8 için 1 bayt).

Toplam $s$ uzunluğundaki bir bağlam (prompt + yanıt) için tek bir isteğin KV cache tüketimi:

$$M_{\text{KV\_istek}} = M_{\text{KV\_token}} \times s$$

Eğer sistemde aynı anda $C$ adet eşzamanlı istek (concurrency) işleniyorsa, toplam KV cache yükü:

$$M_{\text{KV\_toplam}} = C \times M_{\text{KV\_token}} \times s$$

### 2.3. GQA neden önemlidir?

Llama 1 döneminde kullanılan klasik **Multi-Head Attention (MHA)** mimarisinde, her Query kafasına karşılık bir Key ve bir Value kafası bulunurdu ($n_{\text{kv\_heads}} = n_{\text{q\_heads}} = 64$).

Llama 3 70B modelinde MHA kullanılsaydı ne olurdu hesaplayalım:
- $n_{\text{layers}} = 80$, $n_{\text{kv\_heads}} = 64$, $d_{\text{head}} = 128$, $b = 2$ bayt (FP16).
- Token başına KV cache:

$$M_{\text{KV\_token}} = 2 \times 80 \times 64 \times 128 \times 2 = 2.621.440 \text{ bayt} \approx \mathbf{2,62 \text{ MB/token}}$$

8.192 tokenlik tek bir istek için KV cache boyutu:
$$2,62 \text{ MB} \times 8.192 \approx \mathbf{21,47 \text{ GB}}!$$

Tek bir kullanıcının sohbet geçmişi 21 GB VRAM yer! 10 kullanıcı aynı anda bağlandığında 214 GB bellek tükenir; sistem anında CUDA Out-Of-Memory (OOM) hatası verir.

<svg viewBox="0 0 720 244" role="img" aria-label="8.192 token bağlamlı 70B model için yığılmış çubuklar. Token başına KV 2 × 80 × 8 × 128 × 2 byte = 320 KiB; istek başına 2,5 GiB yani 2,68 GB. 10 kullanıcıda 140 GB ağırlık + 26,8 GB KV ≈ 167 GB; çalışma alanı hariç, 4 × H100 = 320 GB&#x27;lık öğretici havuza sığar. 100 kullanıcıda 140 + 268,4 GB ≈ 408 GB; sığmaz." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="kg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Aynı 70B model: 8.192 token bağlamlı 10 ve 100 kullanıcı</text>
<text x="16" y="44" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Token başına KV = 2 × 80 × 8 × 128 × 2 B = 320 KiB → 8K istek başına 2,5 GiB (2,68 GB) · öğretici havuz: 4 × H100 = 320 GB</text>
<text x="16" y="94" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">10 kullanıcı</text>
<rect x="130" y="76" width="161.778" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210.88888888888889" y="95.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">ağırlık 140</text>
<rect x="291.778" y="76" width="30.9689" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="330.74666666666667" y="88" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">167 GB</text>
<text x="330.74666666666667" y="104" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">sığar (+ çalışma alanı?)</text>
<text x="324.74666666666667" y="122" text-anchor="start" style="fill:var(--c-warn);font-size:11px">KV 26,8</text>
<text x="16" y="156" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">100 kullanıcı</text>
<rect x="130" y="138" width="161.778" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210.88888888888889" y="157.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">ağırlık 140</text>
<rect x="291.778" y="138" width="310.151" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="446.85333333333335" y="157.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">KV 268,4</text>
<text x="609.9288888888889" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">408 GB</text>
<text x="609.9288888888889" y="166" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">sığmaz</text>
<line x1="499.778" y1="60" x2="499.778" y2="202" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="499.77777777777777" y="216" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">4 × 80 GB = 320 GB</text>
<text x="130.0" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="245.55555555555554" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<text x="361.1111111111111" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">200</text>
<text x="476.66666666666663" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">300</text>
<text x="592.2222222222222" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">400</text>
<text x="690" y="232" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">GB</text>
</svg>

*Şekil 2.2: 8.192 token bağlamlı 70B modelde MHA, GQA ve MLA mimarilerinin KV cache bellek tüketiminin karşılaştırılması.*

**Grouped-Query Attention (GQA)** bu darboğazı çözer: 8 adet Query kafası tek bir ortak Key ve Value kafasını paylaşır ($n_{\text{kv\_heads}} = 8$, yani 8:1 oranında kafa paylaşımı):
- Token başına KV cache:

$$M_{\text{KV\_token}} = 2 \times 80 \times 8 \times 128 \times 2 = 327.680 \text{ bayt} \approx \mathbf{0,328 \text{ MB/token}}$$

8.192 tokenlik istek başına KV cache:
$$0,328 \text{ MB} \times 8.192 \approx \mathbf{2,68 \text{ GB}}$$

KV cache tam **8 kat küçülür**! Aynı sunucuda 8 kat daha fazla eşzamanlı kullanıcı barındırılabilir.

**DeepSeek-V3** mimarisinde kullanılan **Multi-Head Latent Attention (MLA)** ise bu fikri bir adım öteye taşır: Key ve Value kafaları doğrudan saklanmaz; düşük boyutlu bir gizli uzaya ($d_c = 512$) sıkıştırılarak (low-rank compression) saklanır. Çıkarım esnasında bu gizli vektör çalışma anında projekte edilir. Sonuç: KV cache boyutu GQA'den de kat kat aşağı iner.

### 2.4. Training belleği neden daha büyüktür?

Çıkarımda yalnızca model ağırlıkları ve KV cache saklanırken, model eğitiminde (AdamW optimizer ile) her parametre için muazzam bir durum listesi tutulur:

| Durum Bileşeni | Sayısal Format | Parametre Başına Bellek | 70B Model İçin Toplam |
| :--- | :--- | :--- | :--- |
| **Model Ağırlıkları** | BF16 (16-bit) | 2 bayt | 140 GB |
| **Gradyanlar** | BF16 (16-bit) | 2 bayt | 140 GB |
| **FP32 Master Ağırlıklar** | FP32 (32-bit) | 4 bayt | 280 GB |
| **AdamW Momentum (1. moment)**| FP32 (32-bit) | 4 bayt | 280 GB |
| **AdamW Varyans (2. moment)** | FP32 (32-bit) | 4 bayt | 280 GB |
| **Toplam Statik Eğitim Durumu**| — | **16 bayt/parametre** | **1.120 GB (1,12 TB)** |

70B bir modeli eğitmek için sadece ağırlık ve optimizer durumları **1.120 GB** yer kaplar! Buna ileri geçiş aktivasyonları ($O(B \cdot S \cdot d_{\text{model}} \cdot n_{\text{layers}})$) eklendiğinde toplam eğitim belleği kolaylıkla 2–3 Terabayt seviyesine ulaşır. Bu yüzden eğitim daima çıkarımdan kat kat daha fazla bellek ister.

### 2.5. 70B model için tam bellek defteri

Sekiz adet NVIDIA H100 SXM5 GPU barındıran bir DGX sunucusunda (Toplam VRAM: $8 \times 80 = 640 \text{ GB}$) LLaMA 3.1 70B modelini (BF16, TP=8) yayına aldığımızı varsayalım:

```text
Toplam Kullanılabilir HBM3:                    640,0 GB
- Model Ağırlıkları (BF16, 70B):              -140,0 GB
- CUDA Sürücüsü ve PyTorch Runtime:            -16,0 GB (GPU başına ~2 GB)
- Aktivasyon Çalışma Alanı (Scratchpad/GEMM):  -24,0 GB (GPU başına ~3 GB)
========================================================================
KV Cache İçin Kalan Serbest Bellek Havuzu:     460,0 GB
```

Bu 460 GB'lık devasa KV cache havuzunun bize ne sağladığını hesaplayalım:
GQA kullanan Llama 3 70B'de 4.096 tokenlik tipik bir istek yaklaşık $1,34 \text{ GB}$ KV cache tüketir.
$$\text{Maksimum Eşzamanlı İstek (Concurrency)} = \frac{460 \text{ GB}}{1,34 \text{ GB}} \approx \mathbf{343 \text{ eşzamanlı istek}}$$

Sunucu aynı anda 343 kullanıcıya tek bir bellek taşması yaşamadan kesintisiz yanıt verebilir!

---

## 3. Serving motoru GPU’yu nasıl verimli tutar?

Geleneksel web servislerinde bir thread bir isteği alır, CPU'da çalıştırır ve cevabı döner. Ancak LLM çıkarımında saf model döngüsü çalıştırmak donanım kaynaklarının %80'inden fazlasını çöpe atar. Modern serving motorları (vLLM, TensorRT-LLM, TGI) GPU'yu doyurmak için devrimsel bellek ve zamanlama teknikleri kullanır.

### 3.1. Continuous batching

Klasik (statik) batching yönteminde, örneğin 4 istek bir araya getirilir ve aynı anda modele verilir. Ancak isteklerin uzunlukları birbirinden farklıdır: Biri 10 token'da biterken, diğeri 500 token üretebilir:

**Statik Batching'in İki Büyük Kusuru:**
1. **Padding İsrafı:** Kısa istekler en uzun isteğin boyutuna kadar sahte `<PAD>` token'ları ile doldurulur. GPU bu boş token'lar için milyarlarca gereksiz FLOP hesaplar.
2. **Kuyruk Tıkanması (Head-of-Line Blocking):** Erken biten 10 tokenlik istek GPU'dan ayrılamaz; en uzun istek bitene kadar GPU belleğini işgal etmeye devam eder. Yeni gelen istekler kapıda bekler.

**Orca (OSDI 2022)** makalesiyle literatüre giren **Continuous Batching (Hücresel / İterasyon Düzeyinde Batching)** bu sorunu kökten çözer:
- Batching istek düzeyinde değil, **tekil token iterasyonu** düzeyinde yapılır.
- Her decode adımında, işi biten istek derhal batch'ten çıkarılır ve istemciye yanıtı gönderilir.
- Açılan yuvaya kuyrukta bekleyen yeni bir isteğin prefill adımı veya başka bir decode adımı dinamik olarak enjekte edilir.
- GPU tek bir döngü dahi boş kalmaz; throughput 2 ila 4 kat artar.

### 3.2. PagedAttention

Geleneksel derin öğrenme çerçeveleri tensörleri bellekte **ardışık (contiguous)** bloklar olarak ayırır. Bir isteğin maksimum 2.048 token üretebileceği belirtilmişse, sistem baştan 2.048 tokenlik devasa bir bellek bloğu rezerve eder.

Bu durum iki büyük israfa yol açar:
- **Dahili Parçalanma (Internal Fragmentation):** İstek sadece 50 token üretip bitse bile, kalan 1.998 tokenlik rezerve alan boş kalır ve başka istekler tarafından kullanılamaz.
- **Harici Parçalanma (External Fragmentation):** Bellekte parça parça boşluklar kalır; ancak ardışık olmadıkları için yeni bir istek yerleşemez.

Geleneksel sistemlerde KV cache belleğinin **%60 ila %80'i** bu parçalanma yüzünden kullanılamazdı!

**vLLM (SOSP 2023)** ekibi bu sorunu işletim sistemlerinin 1960'larda çözdüğü gibi çözdü: **Sanal Bellek Sayfalaması (Virtual Memory Paging)** mantığını GPU'ya taşıyarak **PagedAttention** algoritmasını geliştirdiler:
- KV cache ardışık bellekte tutulmaz.
- Bellek, örneğin 16 veya 32 token barındıran sabit boyutlu **fiziksel bloklara (KV blocks)** bölünür.
- İstek uzadıkça serving motoru boşta duran herhangi bir fiziksel bloğu tahsis eder ve mantıksal adres tablosuna (page table) kaydeder.
- Parçalanma israfı %60-80 seviyesinden **%4'ün altına** düşer! Aynı GPU'da barındırılabilecek eşzamanlı istek sayısı neredeyse ikiye katlanır.

### 3.3. Prefix caching

Gerçek hayat senaryolarında kullanıcı istekleri sıfırdan başlamaz. Sistem prompt'ları ("Sen yardımsever bir asistansın..."), RAG bağlamları, few-shot örnekleri veya çok turlu (multi-turn) sohbet geçmişleri istekler arasında sürekli tekrarlanır.

<svg viewBox="0 0 720 258" role="img" aria-label="İki istek ortak bir prefix paylaşır. İstek A&#x27;nın mantıksal KV blokları P0, P1, A2 ve A3; istek B&#x27;ninkiler P0, P1 ve B2&#x27;dir. Blok tablosu her mantıksal bloğu HBM&#x27;deki bir fiziksel yuvaya eşler: P0 satır 0 sütun 3&#x27;e, P1 satır 2 sütun 2&#x27;ye, A2 satır 0 sütun 1&#x27;e, A3 satır 1 sütun 4&#x27;e, B2 satır 1 sütun 0&#x27;a; diğer yuvalar boştur. İki istek de aynı fiziksel P0 ve P1 bloklarına işaret ettiği için ortak prefix bir kez prefill edilir. Blokların bitişik olması gerekmez, bu da parçalanmayı azaltır. Tenant&#x27;lar arası paylaşım çitlenmelidir (ör. vLLM cache_salt); decode işi ve token başına K/V yok olmaz." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="pg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="pg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">PagedAttention + prefix caching: mantıksal dizi ≠ fiziksel yerleşim</text>
<text x="16" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">mantıksal KV blokları (istek başına)</text>
<text x="430" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">fiziksel GPU belleği (HBM)</text>
<text x="16" y="86" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">istek A</text>
<rect x="100" y="70" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="156" y="70" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="182.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="212" y="70" width="52" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="238.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A2</text>
<rect x="268" y="70" width="52" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="87.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A3</text>
<text x="16" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">istek B</text>
<rect x="100" y="134" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="156" y="134" width="52" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="182.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="212" y="134" width="52" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="238.0" y="151.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B2</text>
<rect x="430" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="456.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="486" y="62" width="52" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="512.0" y="81.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A2</text>
<rect x="542" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="568.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="598" y="62" width="52" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="624.0" y="81.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P0</text>
<rect x="654" y="62" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="680.0" y="80.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="430" y="98" width="52" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="456.0" y="117.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">B2</text>
<rect x="486" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="512.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="542" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="568.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="598" y="98" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="624.0" y="116.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="654" y="98" width="52" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="680.0" y="117.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">A3</text>
<rect x="430" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="456.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="486" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="512.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="542" y="134" width="52" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="568.0" y="153.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">P1</text>
<rect x="598" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="624.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="654" y="134" width="52" height="30" rx="4" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="680.0" y="152.78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">boş</text>
<rect x="330" y="64" width="84" height="100" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="372" y="82" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">blok tablosu</text>
<text x="372" y="98" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">P0 → (0,3)</text>
<text x="372" y="111" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">P1 → (2,2)</text>
<text x="372" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">A2 → (0,1)</text>
<text x="372" y="137" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">A3 → (1,4)</text>
<text x="372" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">B2 → (1,0)</text>
<line x1="316" y1="114" x2="328" y2="114" marker-end="url(#pg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="414" y1="114" x2="428" y2="114" marker-end="url(#pg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="182" width="688" height="64" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="202" text-anchor="start" style="fill:var(--c-text);font-size:12px">Ortak prefix (P0, P1): iki istek aynı fiziksel bloklara işaret eder → prefill bir kez yapılır.</text>
<text x="28" y="220" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bloklar bitişik olmak zorunda değil → önden dev rezervasyon yok, daha az parçalanma.</text>
<text x="28" y="238" text-anchor="start" style="fill:var(--c-danger);font-size:12px">Tenant&#x27;lar arası paylaşım çitlenmeli (ör. vLLM cache_salt); decode işi ve token başına K/V yok olmaz.</text>
</svg>

*Şekil 3.1: Prefix caching (Radix Attention). İki farklı istek aynı sistem prompt'unu paylaştığında, ortak KV blokları ikinci kez hesaplanmaz; mantıksal işaretçilerle aynı fiziksel bellekten okunur.*

**Prefix Caching (Radix Attention)** tekniği ile:
- İşlenen prompt bloklarının hash değerleri bir **Radix Ağacı (Radix Tree)** veri yapısında saklanır.
- Yeni bir istek geldiğinde, prompt'un ortak başlangıç kısmı tespit edilir.
- Ortak kısmın KV cache'i HBM'de zaten hazır olduğu için bu token'lar için **prefill hesabı tamamen atlanır**!
- Prefill süresi sıfıra yaklaşır, TTFT onlarca kat hızlanır ve GPU hesaplama gücü tasarruf edilir.

### 3.4. Prefill ve decode’u ayırmak (Disaggregated Serving)

Klasik mimaride aynı GPU havuzu hem prefill hem decode işlemlerini birlikte yürütür. Ancak bu iki evre donanımsal olarak birbirine düşmandır:
- Prefill: Büyük matris çarpımı, yüksek FLOP, GPU'yu anlık %100 meşgul eder (bursty).
- Decode: Küçük vektör çarpımı, düşük FLOP, sürekli bellek bant genişliği ister, uzun sürer.

Aynı GPU'da çalışan uzun bir prefill isteği geldiğinde, o sırada decode yapmakta olan onlarca kullanıcının token üretimi duraklar (inter-token jitter / lag). Kullanıcı metin akışında donma hisseder.

<svg viewBox="0 0 720 324" role="img" aria-label="İki panel. A, birleşik: router yükü, her biri hem prefill hem decode çalıştıran worker&#x27;lara dağıtır; KV transferi yoktur ama evreler birbirini etkiler. B, ayrıştırılmış: KV farkındalıklı gateway istekleri uzun girişlere uygun, hesap ağırlıklı prefill havuzuna gönderir; oluşan KV cache RDMA veya NVLink üzerinden uzun çıktılara uygun, HBM sınırlı ayrı decode havuzuna aktarılır; decode havuzu router&#x27;a boş KV kapasitesini, önbellekteki prefix&#x27;leri ve kuyruk derinliğini bildirir. Ayrıştırma ancak KV transferi ortadan kaldırdığı etkileşimden ucuzsa kazandırır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker><marker id="dg-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Birleşik ve ayrıştırılmış (disaggregated) serving</text>
<rect x="16" y="38" width="688" height="92" rx="6" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">A · birleşik: her worker prefill + decode yapar</text>
<rect x="28" y="70" width="110" height="46" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83.0" y="90.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Router</text>
<text x="83.0" y="105.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yük dağıtımı</text>
<rect x="176" y="68" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="184" y="76" width="66" height="34" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="217.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="256" y="76" width="72" height="34" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="292.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">decode</text>
<rect x="348" y="68" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="356" y="76" width="66" height="34" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="389.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="428" y="76" width="72" height="34" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="464.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">decode</text>
<rect x="520" y="68" width="160" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="528" y="76" width="66" height="34" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="561.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">prefill</text>
<rect x="600" y="76" width="72" height="34" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="636.0" y="97.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">decode</text>
<line x1="138" y1="93" x2="174" y2="93" marker-end="url(#dg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="704" y="58" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px">+ KV transferi yok  − evreler birbirini etkiler</text>
<rect x="16" y="142" width="688" height="150" rx="6" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">B · ayrıştırılmış: ayrı prefill ve decode havuzları</text>
<rect x="28" y="186" width="110" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83.0" y="211.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Gateway / router</text>
<text x="83.0" y="226.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KV farkındalıklı</text>
<rect x="176" y="176" width="170" height="76" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="261" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Prefill havuzu</text>
<text x="261" y="212" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">compute ağırlıklı · uzun ISL</text>
<rect x="190" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="227" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="264" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="301" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="496" y="176" width="170" height="76" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="581" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Decode havuzu</text>
<text x="581" y="212" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">HBM sınırlı · uzun OSL</text>
<rect x="510" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="547" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="584" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="621" y="222" width="30" height="20" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="138" y1="214" x2="174" y2="214" marker-end="url(#dg-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="346" y1="214" x2="494" y2="214" marker-end="url(#dg-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<text x="420" y="206" text-anchor="middle" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">KV transferi</text>
<text x="420" y="232" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">RDMA / NVLink</text>
<path d="M581 252 V272 H83 V244" marker-end="url(#dg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="332" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">geri bildirim: boş KV kapasitesi, önbellekteki prefix&#x27;ler, kuyruk</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca KV transferi, ortadan kaldırdığı etkileşimden ucuzsa kazandırır; ikisini kendi ISL/OSL karışımınızla ölçün.</text>
</svg>

*Şekil 3.2: Disaggregated Serving mimarisi. Prefill havuzu hesaplama ağırlıklı çalışıp TTFT'yi optimize ederken, Decode havuzu bellek bant genişliğini optimize ederek sabit TPOT sunar; aradaki KV cache aktarımı yüksek hızlı RDMA ile sağlanır.*

**Disaggregated Serving (Ayrık Servis Mimarisi - Splitwise / Tetris):**
- Küme ikiye bölünür: **Prefill Düğümleri** ve **Decode Düğümleri**.
- Prefill düğümleri yalnızca prompt'u işler, yüksek FLOP üretir ve KV cache'i doldurur.
- Doldurulan KV cache tensörleri, düğümler arası yüksek hızlı ağ (InfiniBand / RoCE) üzerinden **GPUDirect RDMA** ile decode düğümlerine fırlatılır.
- Decode düğümleri yalnızca token üretir; prefill fırtınalarından etkilenmeden pürüzsüz ve garantili bir TPOT sunar.

> [!WARNING]
> **Ödünleşim Uyarısı:** Disaggregated serving sihirli bir reçete değildir. Prefill'de üretilen gigabaytlarca KV cache'in ağ üzerinden aktarılması ciddi bir ağ bant genişliği faturası çıkarır. Eğer kümenizde 400 Gb/s InfiniBand omurgası yoksa, KV aktarım süresi prefill'den kazandığınız zamanı yiyebilir!

### 3.5. Speculative decoding

Decode evresinin bellek bant genişliği darboğazını aşmanın en zarif yollarından biri **Speculative Decoding (Tahminli Kod Çözme)** tekniğidir:

1. **Taslak Model (Draft Model):** Ana modelin yanında küçük ve çok hızlı bir taslak model (örneğin 70B'nin yanında 8B bir model veya n-gram tablosu) çalıştırılır.
2. Taslak model, düşük bellek maliyetiyle ardışık olarak hızlıca örneğin $K = 4$ adet token tahmin eder ($t_1, t_2, t_3, t_4$).
3. **Büyük Modelin Doğrulaması:** 70B ana model bu 4 token'ı tek tek değil; **tek bir paralel forward GEMM adımıyla** aynı anda doğrular!
4. Eğer ana model taslak token'ları onaylarsa, tek bir decode iterasyonunda **4 token birden** üretilmiş olur!
5. Hızlanma çarpanı onay oranına ($\alpha$) bağlıdır:

$$\text{Hızlanma} \approx \frac{1}{1 - \alpha + \frac{\alpha}{K}}$$

Onay oranı %80 olduğunda, 70B model 2 ila 3 kat daha hızlı token üretmeye başlar; üstelik çıktının matematiksel dağılımı ana modelle birebir aynı kalır!

---

## 4. Kapasite ve gecikme nasıl birlikte planlanır?

Bir üretim sistemi tasarlarken throughput ile gecikme arasındaki gerilimi yönetmek hayati önem taşır.

### 4.1. Dört metriği karıştırmayın

Sistem performansını değerlendirirken şu dört kavramı birbirinden kesin çizgilerle ayırın:
- **TTFT (Time-To-First-Token):** İlk token gecikmesi (prompt uzunluğuna ve kuyruk süresine bağlıdır).
- **TPOT (Time-Per-Output-Token):** İki ardışık token arasındaki süre (model boyutuna ve bellek bant genişliğine bağlıdır).
- **Throughput (Ham Verim):** Sistemin saniyede ürettiği toplam token sayısı ($\text{Tokens/s}$). Büyük batch ile artar.
- **Goodput (Faydalı Verim):** Belirlenmiş SLA gecikme sınırları (örneğin $\text{TTFT} < 500 \text{ ms}$ ve $\text{TPOT} < 40 \text{ ms}$) içinde başarıyla tamamlanan faydalı token sayısı.

<svg viewBox="0 0 720 298" role="img" aria-label="Toplam 1,45 saniyelik temsili TTFT şelalesi: kimlik doğrulama 20 ms, güvenlik 30 ms, opsiyonel veri servisi 150 ms, kuyruk 500 ms, prefill 650 ms, ilk decode 50 ms ve ağ/proxy 50 ms. Kuyruk ve prefill toplamın %79&#x27;udur. Her satır gerçek tracing ile ölçülmelidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">1,45 sn&#x27;lik TTFT nereye gidiyor? (temsili şelale)</text>
<text x="180" y="59" text-anchor="end" style="fill:var(--c-text);font-size:12px">auth</text>
<rect x="190" y="47" width="6.34483" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="202.3448275862069" y="60" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">20 ms</text>
<text x="180" y="87" text-anchor="end" style="fill:var(--c-text);font-size:12px">güvenlik</text>
<rect x="196.345" y="75" width="9.51724" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="211.86206896551724" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">30 ms</text>
<line x1="196.345" y1="65" x2="196.345" y2="75" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="115" text-anchor="end" style="fill:var(--c-text);font-size:12px">veri servisi (opsiyonel)</text>
<rect x="205.862" y="103" width="47.5862" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="259.44827586206895" y="116" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">150 ms</text>
<line x1="205.862" y1="93" x2="205.862" y2="103" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="143" text-anchor="end" style="fill:var(--c-text);font-size:12px">kuyruk</text>
<rect x="253.448" y="131" width="158.621" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.6"/>
<text x="418.0689655172414" y="144" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">500 ms</text>
<line x1="253.448" y1="121" x2="253.448" y2="131" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="171" text-anchor="end" style="fill:var(--c-text);font-size:12px">prefill</text>
<rect x="412.069" y="159" width="206.207" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="624.2758620689656" y="172" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">650 ms</text>
<line x1="412.069" y1="149" x2="412.069" y2="159" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="199" text-anchor="end" style="fill:var(--c-text);font-size:12px">ilk decode</text>
<rect x="618.276" y="187" width="15.8621" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="640.1379310344828" y="200" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">50 ms</text>
<line x1="618.276" y1="177" x2="618.276" y2="187" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<text x="180" y="227" text-anchor="end" style="fill:var(--c-text);font-size:12px">ağ / proxy</text>
<rect x="634.138" y="215" width="15.8621" height="18" rx="2" style="fill:var(--c-text-mute);fill-opacity:.6"/>
<text x="656.0" y="228" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">50 ms</text>
<line x1="634.138" y1="205" x2="634.138" y2="215" style="stroke:var(--c-border);stroke-width:0.8;stroke-dasharray:4 3"/>
<line x1="190" y1="246" x2="650" y2="246" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="190" y1="246" x2="190" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="190.0" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0 sn</text>
<line x1="348.621" y1="246" x2="348.621" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="348.62068965517244" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0,5 sn</text>
<line x1="507.241" y1="246" x2="507.241" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="507.2413793103448" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1,0 sn</text>
<line x1="650" y1="246" x2="650" y2="251" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="650.0" y="264" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1,45 sn</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Burada kuyruk + prefill = %79. Her satır gerçek tracing&#x27;le ölçülmeli; terimler pratikte örtüşebilir.</text>
</svg>

*Şekil 4.1: Temsili bir TTFT şelalesi. İlk token süresinin ağ, kimlik doğrulama, kuyruk beklemesi, tokenizasyon ve prefill GEMM adımlarına ayrışımı.*

### 4.2. Kuyruk neden aniden patlar?

Kuyruk gecikmelerini anlamanın evrensel yasası **Little's Law** ile ifade edilir:

$$L = \lambda W$$

Burada $L$ sistemdeki ortalama istek sayısı, $\lambda$ sisteme gelen istek akış hızı (istek/s), $W$ ise bir isteğin sistemde geçirdiği ortalama süredir.

Kuyruk teorisinde (M/M/1 ve M/G/1 modelleri), sunucu kullanım oranı ($\rho = \lambda / \mu$, burada $\mu$ servis kapasitesidir) arttıkça ortalama kuyruk bekleme süresi çizgisel değil, asimptotik olarak patlar:

$$W_q = \frac{\rho}{\mu (1 - \rho)}$$

```text
Kullanım Oranı (ρ) = %50  --> Bekleme Çarpanı: 1x
Kullanım Oranı (ρ) = %80  --> Bekleme Çarpanı: 4x
Kullanım Oranı (ρ) = %90  --> Bekleme Çarpanı: 9x
Kullanım Oranı (ρ) = %95  --> Bekleme Çarpanı: 19x!
Kullanım Oranı (ρ) = %99  --> Bekleme Çarpanı: 99x (Kuyruk kilitlenir!)
```

Sunucu kullanımını %80'den %95'e çıkardığınızda throughput sadece %15 artar; ancak kuyruk gecikmesi **5 katına fırlar**! Bu yüzden gecikmeye duyarlı kurumsal çıkarım sistemleri asla %90'ın üzerinde planlanmaz; optimal çalışma noktası %65–75 bandıdır.

---

## 5. Görsel ve video iş yükleri neyi değiştirir?

Geleneksel metin modellerinde girdi boyutu kelimelerle sınırlıdır. Ancak **Vision-Language Models (VLM)** ve çok modlu (multimodal) sistemlerde kapasite hesabı tamamen altüst olur.

<svg viewBox="0 0 720 246" role="img" aria-label="VLM istek şelalesi: medya girdisi decode edilir, frame, crop veya tile&#x27;lara örneklenir, patch&#x27;ler üzerinde çalışan ViT vision encoder&#x27;dan geçer, dil modelinin uzayına projekte edilir; ardından dil modeli metin ve görsel token&#x27;ları prefill eder ve yanıtı decode eder. Decode&#x27;dan önceki her şey TTFT&#x27;ye yazılır. Görsel örneği: 1024 × 1024 piksel ve 16 × 16 patch, tek düzlemli öğretici modelde 64 × 64 = 4.096 patch eder; merging ve pooling gerçek LLM token sayısını değiştirir. Video örneği: 10 saniyelik klip 2 fps&#x27;te 20, 30 fps&#x27;te 300 frame&#x27;dir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="vl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="vl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="vl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">VLM istek şelalesi: ilk metin token&#x27;ından önceki iş</text>
<rect x="16" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="70.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Medya girdisi</text>
<text x="70.0" y="76.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">görsel / video</text>
<text x="70.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">decode, EXIF</text>
<rect x="132" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="186.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Örnekleme</text>
<text x="186.0" y="76.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">frame, crop,</text>
<text x="186.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tile</text>
<line x1="124" y1="71" x2="131" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="248" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="302.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Vision encoder</text>
<text x="302.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">patch&#x27;lerde ViT</text>
<line x1="240" y1="71" x2="247" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="418.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Projeksiyon</text>
<text x="418.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">görsel → LM uzayı</text>
<line x1="356" y1="71" x2="363" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="480" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="534.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LM prefill</text>
<text x="534.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">metin + görsel token</text>
<line x1="472" y1="71" x2="479" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="596" y="42" width="108" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="650.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">LM decode</text>
<text x="650.0" y="83.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yanıt token&#x27;ları</text>
<line x1="588" y1="71" x2="595" y2="71" marker-end="url(#vl-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M16 112 H592" marker-start="url(#vl-arr)" marker-end="url(#vl-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="306.0" y="128" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">bunların hepsi TTFT&#x27;ye yazılır</text>
<rect x="16" y="142" width="340" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Görsel: “görsel başına 1 token” değil</text>
<text x="28" y="182" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">1024 × 1024 px ÷ 16 × 16 patch</text>
<text x="28" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">= 64 × 64 = 4.096 patch (tek düzlemli öğretici)</text>
<text x="28" y="220" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">merging/pooling LLM token sayısını değiştirir</text>
<rect x="364" y="142" width="340" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Video: “10 sn klip” yük tanımı değil</text>
<text x="376" y="189" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">2 fps</text>
<rect x="430" y="179" width="12" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="448.0" y="190" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">20 frame</text>
<text x="376" y="215" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">30 fps</text>
<rect x="430" y="205" width="180" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="616.0" y="216" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">300 frame</text>
</svg>

*Şekil 5.1: Çok modlu (VLM) istek şelalesi. Medya verisi decode edilir, patch/crop'lara ayrılır ve Vision Transformer (ViT) tarafından yüzlerce görsel token'a dönüştürülerek dil modeline aktarılır.*

### 5.1. Görsel neden "bir token" değildir?

Bir kullanıcı sisteme tek bir $1.024 \times 1.024$ boyutunda görsel yüklediğinde sistem bunu tek bir girdi olarak görmez:
1. Görsel $14 \times 14$ piksellik küçük parçalara (**patch**) bölünür: $(1.024 / 14) \times (1.024 / 14) \approx 73 \times 73 \approx 5.329 \text{ patch}$.
2. **Vision Transformer (ViT)** bu 5.329 parçayı işler ve bir projeksiyon katmanı üzerinden dil modeline tam **yüzlerce veya binlerce görsel token** olarak aktarır!
3. Kullanıcı tek bir resim göndermiştir; ancak model bunu 3.000 tokenlik devasa bir prompt gibi karşılar. Prefill süresi uzar, KV cache anında şişer.

### 5.2. Video neden ayrı benchmark ister?

Saniyede 30 kare (fps) içeren 10 saniyelik kısa bir video klibi ele alalım. Her saniyeden yalnızca 2 kare örneklesek bile 20 görsel işlenir. Bu durum tek bir istekte **10.000 ila 25.000 tokenlik** devasa bir bağlam yaratır!
VLM servis ederken:
- Vision encoder katmanı GPU belleğinin ciddi bir kısmını kaplar.
- TTFT süreleri saniyeler mertebesine tırmanabilir.
- Çözüm: Görselleri dinamik çözünürlükle ölçeklemek, gereksiz arka plan patch'lerini budamak (token pruning) ve multimodal encoder'ı ayrı bir inference worker'ına taşımaktır.

---

## 6. 70B serving örneği

<svg viewBox="0 0 720 412" role="img" aria-label="Serving için karar ağacı. Trace ve profil sonrası hangi p95/p99 SLO&#x27;nun bozulduğundan başlayın. TTFT geç ise: kuyruk büyüyorsa admission control ve sıcak replika; prefill baskınsa chunked prefill ve prefix caching; hâlâ yavaşsa kendi giriş/çıkış uzunluklarınızla ayrı prefill ve decode havuzlarını A/B test edin ve yalnızca kazanç KV transfer maliyetini aşıyorsa ayrıştırın, değilse birleşik kalın. TPOT yavaşsa sorun decode, HBM ve KV baskısındadır: paged KV ve sürekli batch kullanın ya da quantization veya speculative decoding ile token başına byte&#x27;ı azaltın. Her değişiklikten sonra kaliteyi ve p99&#x27;u yeniden ölçün." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="st-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="st-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="st-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker><marker id="st-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Hangi serving çözümü? Bozulan SLO&#x27;dan başlayın</text>
<rect x="250" y="40" width="220" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="360.0" y="62.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Hangi p95/p99 SLO bozuluyor?</text>
<text x="360.0" y="77.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">önce trace + profil</text>
<path d="M300 90 V108 H140 V122" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M420 90 V108 H560 V122" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="124" width="200" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="140.0" y="146.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">TTFT (ilk token geç)</text>
<text x="140.0" y="161.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kuyruk mu, uzun prefill mi?</text>
<rect x="460" y="124" width="200" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="560.0" y="146.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">TPOT (akış yavaş)</text>
<text x="560.0" y="161.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">decode, HBM, KV baskısı</text>
<path d="M90 174 V192" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M190 174 V192" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="16" y="194" width="140" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">kuyruk büyüyor</text>
<text x="86.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">admission control,</text>
<text x="86.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sıcak replika</text>
<rect x="164" y="194" width="150" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="239.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">prefill baskın</text>
<text x="239.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">chunked prefill,</text>
<text x="239.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">prefix caching</text>
<path d="M239 256 V274" marker-end="url(#st-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="150" y="276" width="178" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="239.0" y="298.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">hâlâ yavaş? ayrı havuz dene</text>
<text x="239.0" y="313.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kendi ISL/OSL&#x27;nizle A/B</text>
<path d="M190 326 V344 H100 V354" marker-end="url(#st-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<path d="M290 326 V344 H380 V354" marker-end="url(#st-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="20" y="356" width="160" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="100.0" y="375.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">kazanç &gt; KV maliyeti</text>
<text x="100.0" y="390.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">→ ayrıştır</text>
<rect x="300" y="356" width="160" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="380.0" y="375.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">kazanç ≤ KV maliyeti</text>
<text x="380.0" y="390.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">→ birleşik kal</text>
<path d="M510 174 V192" marker-end="url(#st-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<path d="M610 174 V192" marker-end="url(#st-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="470" y="194" width="112" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="526.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">batch / KV</text>
<text x="526.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">paged KV,</text>
<text x="526.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">sürekli batch</text>
<rect x="590" y="194" width="114" height="62" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="647.0" y="215.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">byte / token</text>
<text x="647.0" y="230.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">quantization,</text>
<text x="647.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">speculative</text>
<text x="704" y="290" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">her değişiklikte kaliteyi</text>
<text x="704" y="305" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">ve p99&#x27;u yeniden ölç</text>
</svg>

*Şekil 6.1: Üretim çıkarım altyapısı için mühendislik karar ağacı. Gecikme, maliyet ve bağlam gereksinimlerine göre quantization, batching ve mimari seçimleri.*

Sekiz adet NVIDIA H100 SXM5 GPU (640 GB HBM3) üzerinde LLaMA 3.1 70B modelini yayına alan bir referans mimari kuralım:

1. **Paralellik ve Ağırlık:** 8 GPU üzerinde Tensor Parallelism $Y = 8$ olarak kurulur. Ağırlıklar tamamen NVLink crossbar alanında kalır ($140 \text{ GB}$ BF16).
2. **Kalan Bellek ve KV Havuzu:** 640 GB'tan ağırlıklar ve runtime düşüldüğünde geriye **480 GB** net KV cache havuzu kalır.
3. **Maksimum Concurrency:** 4.096 token bağlamında (istek başına $1,34 \text{ GB}$ KV cache) sistem aynı anda **358 eşzamanlı isteği** belleğinde tutabilir.
4. **Teorik TPS Tavanı:** 8 GPU toplamda $8 \times 3.350 = 26.800 \text{ GB/s}$ HBM bant genişliği sunar. Her token üretiminde 140 GB ağırlık okunursa, kuramsal azami token üretim hızı:
   $$\text{Azami TPS} = \frac{26.800 \text{ GB/s}}{140 \text{ GB}} \approx \mathbf{191 \text{ token/s}}$$
   (Bu sayı $B=1$ için tavan değerdir; continuous batching ile $B \ge 16$ yapıldığında ağırlık okuma maliyeti paylaşılarak toplam throughput saniyede 1.500–2.500 token seviyesine çıkar).

---

## 7. Sistem tasarımı için altı temel soru

### 7.1. Prefill compute-bound, decode memory-bound neden?

Prefill aşamasında prompt'taki yüzlerce veya binlerce token tek seferde paralel olarak işlenir. Aktivasyon matrisi geniştir; işlem büyük matris çarpımıdır (GEMM). Aritmetik yoğunluk 400 \dots 600 FLOP/bayt seviyesindedir ve donanımın kritik eşiğini ($\alpha_{\text{crit}} \approx 295$) aştığı için Tensor Core'ları doyurur (Compute-Bound).

Decode aşamasında ise model otoregresif olarak her adımda yalnızca tek bir token üretir ($B=1, S=1$). Yapılan işlem matris-vektör çarpımıdır (GEMV). Tek bir token için modelin 140 GB'lık tüm ağırlıkları HBM'den okunmak zorundadır. Aritmetik yoğunluk 1 FLOP/bayta çöker; işlem hızı hesaplama gücüyle değil, HBM bellek bant genişliğiyle sınırlanır (Memory-Bound).

### 7.2. KV cache neden concurrency ile doğrusal büyür?

Her aktif istek, ürettiği her yeni token'ın Key ve Value vektörlerini geçmişe dikkat edebilmek için bellekte saklamak zorundadır. $C$ adet eşzamanlı kullanıcı olduğunda, her kullanıcının sohbet geçmişi bağımsızdır. Dolayısıyla toplam KV cache ihtiyacı doğrudan $C \times \text{Bağlam Uzunluğu} \times M_{\text{KV\_token}}$ formülüyle hem istek sayısıyla hem de bağlam uzunluğuyla tam doğrusal ($O(C \cdot S)$) olarak büyür.

### 7.3. PagedAttention KV belleğini azaltır mı?

PagedAttention tek bir token'ın kapladığı teorik matematiksel KV boyutunu azaltmaz; ancak **bellek parçalanmasını (fragmentation) ve aşırı rezervasyon israfını ortadan kaldırır**. Geleneksel sistemlerde maksimum uzunluk için ayrılan ancak kullanılmayan bellek israfını %60–80'den %4'ün altına indirir. Böylece fiziksel HBM içinde aynı anda barındırılabilecek eşzamanlı istek sayısını 2 ila 4 kat artırır.

### 7.4. Quantization neden 2× daha az bit ile her zaman 2× hız vermez?

16-bit ağırlıkları 8-bit'e (FP8/INT8) indirmek bellekten taşınan veri hacmini yarıya indirir; bu durum bellek bant genişliği kısıtlı decode evresinde hızı neredeyse 2 kat artırabilir. Ancak:
1. Dequantization ek yükü: Ağırlıklar matris çarpımından önce FP16/FP32'ye geri dönüştürülüyorsa ALU döngüleri harcanır.
2. İş yükü zaten compute-bound ise (büyük batch veya prefill), darboğaz bellek bant genişliği değil silikon işlem gücüdür; bu durumda ağırlık transferini hızlandırmak toplam süreyi doğrudan yarıya indirmez.

### 7.5. Disaggregated serving ne zaman kötüleşir?

Prefill ve decode düğümleri arasındaki ağ bant genişliği yetersiz olduğunda veya prompt uzunlukları çok kısa olduğunda kötüleşir. Kısa prompt'larda prefill hesaplama süresi zaten birkaç milisaniyedir; üretilen KV cache'i ağ üzerinden decode düğümüne aktarmak için harcanan ağ gecikmesi, prefill'in kendi süresinden daha uzun sürebilir. Bu durumda ayrık mimari genel gecikmeyi düşürmek yerine artırır.

### 7.6. Throughput mu latency mi optimize edilmeli?

Bu tamamen iş yükünün kullanım senaryosuna bağlıdır. Kullanıcıyla doğrudan etkileşime giren sohbet (chatbot) sistemlerinde öncelik **Latency (düşük TTFT ve insan okuma hızına uygun TPOT)** tarafındadır; batch boyutları küçük tutulur. Arka planda çalışan toplu doküman özetleme, veri zenginleştirme veya sentetik veri üretiminde ise kullanıcı beklemediği için öncelik **Throughput (azami token/s ve minimum GPU başı maliyet)** tarafındadır; batch boyutları tavan yaptırılır.

---

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

1. 70B bir model için tek bir kullanıcının 8.192 tokenlik KV cache faturasını MHA ve GQA mimarileri altında nasıl hesaplarsınız?
2. Tekil decode adımında 990 TFLOP/s kapasiteli bir H100'ün neden 3,35 TFLOP/s hızında tıkandığını formülle nasıl açıklarsınız?
3. Continuous batching'in statik batching'e kıyasla padding israfını nasıl yok ettiğini adım adım nasıl çizersiniz?
4. PagedAttention algoritmasının işletim sistemi sanal bellek sayfalamasıyla benzerliği nedir ve parçalanmayı nasıl önler?
5. Bir API servisinde sunucu kullanım oranı %80'den %95'e çıktığında kuyruk gecikmesinin neden patladığını Little's Law ile nasıl savunursunuz?

---

## Bütün hikâye altı satırda

- Bir Transformer katmanı eğitimde forward ve backward GEMM ile compute-bound, tekil decode'da ise ağırlık okuma yüzünden memory-bound çalışır.
- Training durumunun saklanması ($16N$ bayt AdamW + aktivasyonlar), çıkarım durumundan ($2N$ bayt) en az 4 ila 8 kat daha fazla VRAM tüketir.
- GQA (Grouped-Query Attention), Key ve Value kafalarını gruplayarak KV cache faturasını MHA'ya kıyasla 8 kat küçültür ve eşzamanlılığı artırır.
- PagedAttention, KV cache'i sabit bloklar halinde sayfalı tahsis ederek bellek parçalanmasını %60–80'den %4'ün altına indirir.
- Disaggregated serving, compute-bound prefill ile memory-bound decode havuzlarını ayırarak prefill dalgalanmalarının token akışını dondurmasını engeller.
- Sistem kapasitesi kullanım oranı %80'i aştığında kuyruk teorisi gereği gecikme katlanarak patlar; kararlı bir goodput için sistem %65–75 dolulukta boyutlandırılmalıdır.

---

## Terimler sözlüğü

- **Prefill:** Modelin prompt token'larının tamamını tek seferde paralel olarak işlediği, Tensor Core ağırlıklı ve compute-bound çalışan başlangıç evresi.
- **Decode:** Modelin bir sonraki token'ı otoregresif olarak teker teker ürettiği, ağırlıkları her adımda HBM'den çeken memory-bandwidth bound çalışan döngü evresi.
- **KV Cache:** Geçmiş token'ların Key ve Value tensörlerini HBM'de saklayarak her adımda dikkat matrisini baştan hesaplama zorunluluğunu ortadan kaldıran bellek tamponu.
- **Continuous Batching:** İsteklerin statik bir grup yerine token iterasyonu düzeyinde dinamik olarak batch'e girip çıkmasını sağlayan hücre tabanlı çizelgeleme yöntemi.
- **PagedAttention:** Sanal bellek sayfalamasını GPU'ya uygulayarak KV cache tensörlerini ardışık olmayan sabit fiziksel bloklarda tutan ve parçalanmayı yok eden algoritma.
- **Grouped-Query Attention (GQA):** Birden fazla Query kafasının tek bir Key ve Value kafasını ortaklaşa kullanmasını sağlayarak KV cache ayak izini küçülten dikkat deseni.
- **Disaggregated Serving:** Prefill işlemlerini compute odaklı, decode işlemlerini bellek bant genişliği odaklı ayrı GPU havuzlarında yürüten ayrık çıkarım mimarisi.
- **Speculative Decoding:** Küçük bir taslak modelin tahmin ettiği token dizisini büyük modelin tek bir paralel forward adımında doğrulamasına dayanan çıkarım hızlandırma tekniği.
- **Goodput:** Sistem gecikmesi belirlenmiş SLA sınırları içinde kalan ve başarıyla teslim edilen faydalı token akış miktarı.

---

## Kaynaklar ve doğrulama bağlantıları

- [K1] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). SOSP 2023. vLLM mimarisi ve PagedAttention tasarımı.
- [K2] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun (2022). [Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022. Continuous (iteration-level) batching formülasyonu.
- [K3] Joshua Ainslie et al. (2023). [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245). EMNLP 2023. Grouped-Query Attention mimarisi ve KV cache tasarrufu.
- [K4] Pratyush Patel et al. (2024). [Splitwise: Efficient Generative LLM Serving Using Phase Splitting](https://arxiv.org/abs/2311.18677). ISCA 2024. Disaggregated serving mimarisi ve prefill/decode ayrımı.
- [K5] Charlie Chen et al. (2023). [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318). Speculative decoding ve kabul olasılığı analitiği.
- [K6] DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). Multi-Head Latent Attention (MLA) mimarisi ve düşük rank'li KV sıkıştırması.
- [K7] Meta AI (2024). [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). Llama 3 8B, 70B ve 405B mimari hiperparametreleri ($n_{\text{layers}}$, $n_{\text{kv\_heads}}$, $d_{\text{head}}$).
- [K8] Bu blogda: [Yapay Zekâ Altyapısı (1): GPU Nasıl Çalışır? — SM'den 16.384 GPU'ya](post.html?slug=gpu-mimarisi-ve-llm-olcekleme).
- [K9] Bu blogda: [Yapay Zekâ Altyapısı (3): GPU'lar Nasıl Birlikte Çalışır? — NVLink, InfiniBand ve Paralellik](post.html?slug=coklu-gpu-aglari-ve-dagitik-egitim).
