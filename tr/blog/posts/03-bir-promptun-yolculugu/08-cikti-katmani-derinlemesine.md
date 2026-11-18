Bir istemin (prompt) Transformer omurgasındaki yolculuğu serimizin önceki yedi bölümünde adım adım netleşti:
1. [Tokenizasyon](post.html?slug=tokenizasyon-nasil-calisir): Metin ayrık tamsayılara (`[1054, 492, 281]`) bölünür.
2. [Embedding Katmanı](post.html?slug=embedding-katmani-derinlemesine): Tamsayılar sürekli vektörlere ve RoPE açılarına çevrilir.
3. [Semantik Vektör Uzayı](post.html?slug=embeddingler-derinlemesine): Anlam geometrik mesafelerle kodlanır.
4. [Self-Attention](post.html?slug=self-attention-derinlemesine): Token'lar $Q, K, V$ izdüşümleriyle bağlamsal etkileşime girer.
5. [Causal Attention](post.html?slug=causal-attention-derinlemesine): Alt üçgen maske geleceğe bakışı engeller.
6. [Multi-Head Attention](post.html?slug=multi-head-attention-derinlemesine): Bağımsız alt uzaylarda dilbilgisel eksenler taranır.
7. [Transformer Bloğu](post.html?slug=transformer-blogu-derinlemesine): Residual hatlar, RMSNorm ve SwiGLU ile temsil olgunlaşır.

Üç kelimelik prompt'umuzda (**`["köpek", "kediyi", "kovaladı"]`**), son bloğun (Llama 3.1'de 32. katman) çıkışında 3. pozisyondaki `"kovaladı"` kelimesi $d_{\text{model}} = 4.096$ boyutlu yoğun bir gizli vektör ($h_3 \in \mathbb{R}^{4096}$) olarak belirir.

Burada temel mühendislik gerçeğiyle karşılaşırız: **4.096 boyutlu sürekli bir kayan noktalı vektör, ekrana doğrudan kelime olarak basılamaz.** İnsan dili kesiklidir ve yüz binlerce kelimelik sözlükten seçilir. Modelin sıradaki kelimeyi (örn. `"tavşanı"`) seçmesi için bu vektörü sözlük uzayına yansıtması, logit skorlarını üretmesi, Softmax ile olasılığa çevirmesi ve örneklemesi gerekir.

Bu durak **Language Model Head (`lm_head`)** katmanıdır. Sıradan bir matris çarpımı değil, sistemin en kritik iki darboğazıdır:
- **Eğitimde Bellek Duvarı:** 128k sözlüklü Llama 3.1 8B'de (8k bağlam, batch=4) ham logit tensörü `[4, 8192, 128256]` tek başına **$8,4\text{ GB}$ VRAM** (gradyanıyla **$16,8\text{ GB}$**) tüketir; standart akışta anında OOM verir.
- **Çıkarımda Bant Genişliği Duvarı:** Her yeni token için 16 GB model ağırlığı HBM'den SM'lere taşınır. Aritmetik yoğunluk $< 1\text{ FLOP/bayt}$ olduğundan GPU zamanının $\%95$'inde bellek bekler.

Bu final makalesinde; LLM Head tensör anatomisini, bağlı vs serbest ağırlıkları, logit soft-capping'i, $\frac{\partial \mathcal{L}}{\partial z} = P - y$ türevini, elle 4D sayısal yürüyüşü, Megatron-LM ve Liger Kernel optimizasyonlarını, Medusa/EAGLE spekülasyonunu ve `model.generate()` döngüsünü nokta atışı inceliyoruz.

**Bu yazıda**

- [1. Büyük resim ve veri yolu: Son transformer bloğundan ekrandaki kelimeye](#1-büyük-resim-ve-veri-yolu-son-transformer-bloğundan-ekrandaki-kelimeye)
- [2. LLM Head mimarisi: Tensör boyutları, token_num ve broadcasting](#2-llm-head-mimarisi-tensör-boyutları-tokennum-ve-broadcasting)
- [3. Ağırlık bağlama (Weight Tying) ikilemi: Parametre tasarrufu mu, ortogonal temsil gücü mü?](#3-ağırlık-bağlama-weight-tying-ikilemi-parametre-tasarrufu-mu-ortogonal-temsil-gücü-mü)
- [4. Logitleri dizginlemek: Gemma 2 ve logit soft-capping inovasyonu](#4-logitleri-dizginlemek-gemma-2-ve-logit-soft-capping-inovasyonu)
- [5. Olasılık uzayına geçiş: Logitler, Softmax ve yer değiştirme değişmezliği](#5-olasılık-uzayına-geçiş-logitler-softmax-ve-yer-değiştirme-değişmezliği)
- [6. Çapraz Entropi (Cross-Entropy): Bilgi kuramı ve asimptotik ceza dinamiği](#6-çapraz-entropi-cross-entropy-bilgi-kuramı-ve-asimptotik-ceza-dinamiği)
- [7. Geri yayılımın zarafeti: Softmax ve Çapraz Entropi türevi (dL/dz = P - y)](#7-geri-yayılımın-zarafeti-softmax-ve-çapraz-entropi-türevi-dldz-p---y)
- [8. Adım adım küçük sayısal yürüyüş: İleri geçiş, soft-capping, kayıp ve geri yayılım](#8-adım-adım-küçük-sayısal-yürüyüş-ileri-geçiş-soft-capping-kayıp-ve-geri-yayılım)
- [9. Donanımsal bellek duvarı: Megatron-LM kelime dağarcığı paralelizmi ve Liger Kernel](#9-donanımsal-bellek-duvarı-megatron-lm-kelime-dağarcığı-paralelizmi-ve-liger-kernel)
- [10. İnce ayar (Fine-Tuning) ve kelime dağarcığı genişletme: LoRA ve modules_to_save tuzağı](#10-ince-ayar-fine-tuning-ve-kelime-dağarcığı-genişletme-lora-ve-modulestosave-tuzağı)
- [11. Çıkarım ivmelenmesi: Medusa ve EAGLE ile spekülatif kod çözme](#11-çıkarım-ivmelenmesi-medusa-ve-eagle-ile-spekülatif-kod-çözme)
- [12. Özbağlanımlı üretim döngüsü: model.generate() ve örnekleme stratejileri](#12-özbağlanımlı-üretim-döngüsü-modelgenerate-ve-örnekleme-stratejileri)
- [13. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri](#13-iki-mühendislik-gözü-eğitim-ve-çıkarım-dinamikleri)
- [14. Python ve PyTorch ile adım adım tam doğrulama](#14-python-ve-pytorch-ile-adım-adım-tam-doğrulama)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Büyük resim ve veri yolu: Son transformer bloğundan ekrandaki kelimeye

### Sezgisel zihinsel model: Mantar pano, dedektif ve büyük sözlük

Tüm tensör matematiğine girmeden önce, `context_len` ile çıktı katmanı (`lm_head`) arasındaki ilişkiyi zihninizde kalıcı kılacak sezgisel bir benzetme kuralım. Yapay zekâ modelini bir **Dedektif**, bağlamı ise dedektifin **Mantar Panosu** olarak düşünün:

```text
[1. ADIM: MANTAR PANO] (Bağlam Uzunluğu / context_len)
Kullanıcının yazdığı tüm kelimeler panoya iğnelenir:
"Katil", "bahçede", "elinde", "bir" ...
(Burası GPU VRAM'indeki geçici çalışma belleğidir = KV Cache)
        │
        ▼
[2. ADIM: DEDEKTİFİN DÜŞÜNCESİ] (Son Gizli Vektör: h)
Dedektif panodaki tüm notları okur (Attention) ve beyninde tek bir özet çıkarır:
4.096 sayıdan oluşan sürekli bir şifre: [0.5, -1.2, 0.8, 1.5 ...]
        │
        ▼
[3. ADIM: SÖZLÜK MASASI] (Çıktı Katmanı / LLM Head)
Dedektifin önünde 128.000 kelimelik koca bir sözlük açılır (W_head matrisi).
Dedektif beynindeki şifreyi sözlükteki tüm kelimelerle çarparak puanlar (Logits):
• "bıçak"   : +45 puan   |   • "tabanca" : +42 puan
• "elma"    : +2 puan    |   • "araba"   : -18 puan
        │
        ▼
[4. ADIM: YÜZDEYE ÇEVİRME] (Softmax)
Puanlar toplamı %100 olan geçerli bir pasta grafiğine çevrilir:
• "bıçak"   : %65        |   • "tabanca" : %30
• "elma"    : %4.9       |   • "araba"   : %0.0001
        │
        ▼
[5. ADIM: ZAR ATMA & EKRANA BASMA] (Sampling & Generation)
Zar atılır, %65 ihtimalle "bıçak" seçilir ve ekrana basılır.
Seçilen kelime MANTAR PANOYA (KV Cache) yeni bir not olarak iğnelenir ve döngü tekrarlar.
```

Bu 5 adımlı akış, $d$-boyutlu gizli durum vektörünün ayrık insan diline nasıl dönüştüğünün sezgisel omurgasıdır. Bu omurga, eğitim ve çıkarım rejimlerinde iki taban tabana zıt mühendislik yoluna ayrılır:

### Gizli durumun iki kaderi: Eğitim optimizasyonu ve çıkarım kod çözümü

Son Transformer bloğundan çıkan gizli temsil tensörü $H$, kelime projeksiyon matrisine girmeden önce modern modellerde son bir normalizasyon katmanından geçer; bu katman genellikle **RMSNorm** (Root Mean Square Normalization) mimarisidir:

$$H_{\text{norm}} = \text{RMSNorm}(H) = \frac{H}{\sqrt{\frac{1}{d} \sum_{i=1}^d H_i^2 + \epsilon}} \odot \gamma$$

Buradaki $\gamma \in \mathbb{R}^d$ parametresi, özelliklerin varyansını dengeler.

Normalizasyondan sonra veri akışı iki taban tabana zıt mühendislik yoluna ayrılır:

1. **Eğitim Yolu (Paralel Kayıp Optimizasyonu):**
   - Tüm yığındaki token'lar $H_{\text{norm}} \in \mathbb{R}^{B \times S \times d}$ yoğun matris çarpımıyla (**GEMM**) paralel olarak $W_{\text{head}}^T$ üzerinden yansıtılır.
   - Model konfigürasyonuna göre (örn. Gemma 2) logit'ler hiperbolik tanjant soft-capping süzgecinden geçirilir.
   - Sayısal kararlılığı güvenceye alınmış Çapraz Entropi (Cross-Entropy) kaybı, her pozisyonun tahminini bir sonraki doğru token ile ($t+1$) kıyaslar.
   - Elde edilen hata sinyali $\frac{\partial \mathcal{L}}{\partial z} = P - y$ geriye doğru akıtılarak tüm ağırlık gradyanları çıkarılır ve AdamW optimizer durumları güncellenir.
2. **Çıkarım Yolu (Özbağlanımlı Token Üretimi):**
   - **Prefill (Ön doldurma)** aşamasında prompt token'ları tek geçişte işlenerek KV Cache doldurulur ve sadece en son token'ın logit vektörü hesaplanır.
   - **Decode (Kod çözme)** aşamasında tek bir token vektörü $h_{-1} \in \mathbb{R}^{1 \times d}$, bellek bant genişliği kısıtlı matris-vektör çarpımıyla (**GEMV**) $W_{\text{head}}^T$ ile çarpılır.
   - Logitler bir `LogitsProcessor` hattından geçer (tekrar cezası, sıcaklık ölçeklemesi, yasaklı kelime filtreleri).
   - Bir örnekleme stratejisi (Greedy, Top-$K$, Nucleus Top-$p$) ile tek bir tamsayı token ID'si ($t_{\text{next}}$) seçilir.
   - Yeni token KV Cache'e eklenir, tokenizer ile metne dönüştürülür (detokenization) ve kullanıcıya akıtılır (streaming).

<svg viewBox="0 0 560 330" role="img" aria-label="Çıktı katmanının uçtan uca veri yolu. Son transformer bloğundan çıkan gizli durum, son RMSNorm katmanından geçer, lm_head matrisiyle çarpılır ve eğitim kaybı ile çıkarım örneklemesi olarak ikiye ayrılır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="out-arr-tr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="out-blue-tr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="out-green-tr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<rect x="16" y="16" width="528" height="58" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">Transformer Bloğu Çıkışı</text>
<text x="32" y="58" text-anchor="start" style="fill:var(--c-text);font-size:13.5px;font-weight:600">Son Blok Gizli Durumu H [B, S, d] → Son RMSNorm [B, S, d]</text>
<line x1="280" y1="74" x2="280" y2="94" marker-end="url(#out-arr-tr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="110" y="96" width="340" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.4"/>
<text x="280" y="116" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Language Model Head (lm_head)</text>
<text x="280" y="132" text-anchor="middle" style="fill:var(--c-accent);font-size:11.5px;font-family:var(--font-mono)">Linear(d_model, vocab_size, bias=False) → Z = H · W^T</text>
<line x1="280" y1="144" x2="280" y2="164" marker-end="url(#out-arr-tr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="166" width="260" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="280" y="187" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">İsteğe Bağlı: Logit Soft-Capping (c · tanh(z / c))</text>
<path d="M220 200 C 220 224, 140 224, 140 240" marker-end="url(#out-blue-tr)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M340 200 C 340 224, 420 224, 420 240" marker-end="url(#out-green-tr)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<rect x="20" y="244" width="240" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="140" y="264" text-anchor="middle" style="fill:var(--c-accent);font-size:12px;font-weight:700">EĞİTİM YOLU</text>
<text x="140" y="282" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Çapraz Entropi Kaybı: L = -log(P_hedef)</text>
<text x="140" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">dL/dz = P - y (Gradyan Otoyolu)</text>
<rect x="300" y="244" width="240" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-success);stroke-width:1.2"/>
<text x="420" y="264" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:700">ÇIKARIM YOLU</text>
<text x="420" y="282" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">LogitsProcessor (Sıcaklık, Ceza, Top-p)</text>
<text x="420" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Multinomial Örnekleme → Yeni Token ID</text>
</svg>

---

## 2. LLM Head mimarisi: Tensör boyutları, token_num ve broadcasting

### Doğrusal katman yanılgısı: Bağlam uzunluğu nerede saklanır?

PyTorch ve Hugging Face `transformers` kütüphanelerinde LM Head katmanı şu yalın satırla başlatılır:

```python
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

Geliştiricilerin zihnini en çok kurcalayan soru şudur: *Eğer katman sadece `(hidden_size, vocab_size)` boyutlarıyla tanımlanıyorsa, `token_num` (bağlam uzunluğu / sequence length) nerede saklanır? Doğrusal katmanın içine bağlam sayısı sabit olarak kodlanmış mıdır?*

Bu sorunun yanıtı, çok boyutlu tensör cebirinin yayınlama (broadcasting) ve tensör büzülme (contraction) kurallarında gizlidir. Doğrusal katmanın kendi fiziksel ağırlık matrisi $W_{\text{head}}$, zaman ya da dizi boyutundan tamamen bağımsız, iki boyutlu statik bir parametre bloğudur:

$$W_{\text{head}} \in \mathbb{R}^{V \times d}$$

Burada $V = \text{vocab\_size}$ ve $d = \text{hidden\_size}$'dır.

İleri yayılım sırasında bu katmana verilen girdi tensörü üç boyutludur:

$$H \in \mathbb{R}^{B \times S \times d}$$

Burada $B$ yığın boyutunu (batch size), $S$ bağlam uzunluğunu (`sequence_length` veya `token_num`), $d$ ise gizli boyutu temsil eder. PyTorch `self.lm_head(H)` çağrısını çalıştırdığında, donanım seviyesinde yalnızca en son boyut ($d$) boyunca matris çarpımı işletir:

$$Z_{b, s, v} = \sum_{k=1}^d H_{b, s, k} \cdot (W_{\text{head}})_{v, k} = (H \cdot W_{\text{head}}^T)_{b, s, v}$$

Şöyle okuyun:
- $Z \in \mathbb{R}^{B \times S \times V}$ elde edilen 3 boyutlu nihai Logits tensörüdür.
- Her bir $b$ yığınındaki her bir $s$ zaman adımı için, model birbirinden tamamen bağımsız $V$ boyutlu bir logit skor vektörü üretir.
- Dolayısıyla `token_num`, ağırlık matrisinin sabit bir yapısal boyutu değil, bu katmandan geçirilen ve PyTorch'un otomatik yayınlama mekanizmasıyla paralel işlenen dinamik bağlamın uzunluğudur.

### Eğitim ve çıkarım arasındaki tensör dilimleme farkı

Bu tensör mekaniği, eğitim ile çıkarım arasındaki temel operasyonel ayrımı doğurur:

```text
Eğitim İleri Yayılımı (Forward Pass):
  H:       [Batch, Seq_Len, Hidden]       örn. [4, 8192, 4096]
  W^T:     [Hidden, Vocab]                örn. [4096, 128256]
  Logits:  [Batch, Seq_Len, Vocab]        örn. [4, 8192, 128256]  <-- 8,4 GB tensör!

Çıkarım Adımı (Autoregressive Decode):
  h_son:   [Batch, 1, Hidden]             örn. [1, 1, 4096]
  W^T:     [Hidden, Vocab]                örn. [4096, 128256]
  Logits:  [Batch, 1, Vocab]              örn. [1, 1, 128256]    <-- Yalnızca 256 KB!
```

Eğitimde **Teacher Forcing** yaklaşımı kullanıldığından model tüm pozisyonları aynı anda değerlendirir. $s$ pozisyonundaki gizli durum, $s+1$ pozisyonundaki hedef kelimeyi tahmin etmekle yükümlüdür. Hugging Face kod tabanında bu eşleme açık bir tensör dilimleme ile yapılır:

```python
# shift_logits: son token hariç tüm tahminler
shift_logits = logits[..., :-1, :].contiguous()
# shift_labels: ilk token hariç tüm gerçek hedefler
shift_labels = labels[..., 1:].contiguous()

loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(
    shift_logits.view(-1, config.vocab_size),
    shift_labels.view(-1)
)
```

Çıkarım decode aşamasında ise geçmiş token'lar çoktan belirlendiği için onların logitlerini yeniden üretmek işlem gücünü israf etmek olur. Model yalnızca en son token'ın gizli durumunu projeksiyona sokar: `logits = self.lm_head(hidden_states[:, -1:, :])`.


---

## 3. Ağırlık bağlama (Weight Tying) ikilemi: Parametre tasarrufu mu, ortogonal temsil gücü mü?

### Tarihsel kökenler: Press & Wolf (2017) ve parametre bütçeleri

Sinirsel dil modellerinin ilk dönemlerinde giriş gömme matrisi ($W_{\text{in}} \in \mathbb{R}^{V \times d}$) ile çıktı projeksiyon matrisi ($W_{\text{out}} \in \mathbb{R}^{V \times d}$) birbirinden bağımsız iki ayrı ağırlık bloğu olarak eğitilirdi. 2017 yılında Ofir Press ve Lior Wolf yayımladıkları *Using the Output Embedding to Improve Language Models* makalesinde, bu iki katmanın fiziksel bellekte tek bir ortak matrisi paylaşabileceğini önerdiler:

$$W_{\text{head}} = W_{\text{embed}}$$

Erken dönem küçük modellerde (örneğin 117M parametreli GPT-2 Small, $d=768$, $V=50257$), giriş embedding tablosu $50.257 \times 768 \approx 38,6\text{M}$ parametre tutuyordu; yani **tüm modelin $\%33$'ü sadece embedding matrisinden ibaretti**. Çıktı katmanını da ayrı tutmak sözlük tablolarının modeli domine etmesine yol açıyordu. Bu dönemde ağırlık bağlama (weight tying) zorunlu bir mühendislik kararıydı.

```
Embedding parametreleri O(V · d) olarak büyür   --> Kelime dağarcığıyla lineer
Transformer blokları    O(L · d^2) olarak büyür --> Katman ve gizli boyutla kuadratik
```

Fakat modeller 8B, 70B ve 405B parametre ölçeklerine, 32 ila 128 katman derinliklerine ulaştığında denklem kökten değişti. Llama 3.1 8B'de ($L=32, d=4096, V=128256$) embedding tablosu $128.256 \times 4.096 \approx 525\text{M}$ parametredir ve bu da toplam modelin yalnızca **$\%6,5$'ine** karşılık gelir. Bu dönüşüm modern LLM mimarlarını büyük bir yol ayrımına getirdi: *Ağırlıklar bağlanmalı mı, yoksa serbest mi bırakılmalı?*

| Model Mimarisi | Parametre Ölçeği | Kelime Dağarcığı ($V$) | Gizli Boyut ($d$) | Ağırlık Bağlama (`tie_word_embeddings`) | Çıktı Katmanı Ek Yükü |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Meta Llama 3.1** | 8 Milyar | 128.256 | 4.096 | **Hayır (False / Untied)** | +525 Milyon parametre |
| **Alibaba Qwen 2.5** | 7 Milyar | 152.064 | 3.584 | **Hayır (False / Untied)** | +545 Milyon parametre |
| **Google Gemma 2** | 9 Milyar | 256.000 | 3.584 | **Evet (True / Tied)** | 0 (Ortak bellek tamponu) |
| **DeepSeek-V3** | 671B (MoE) | 129.280 | 7.168 | **Hayır (False / Untied)** | +926 Milyon parametre |

### Geometrik keşif: Ortogonal semantik uzaylar

Meta (Llama) ve Alibaba (Qwen), yarım milyar parametre feda etme pahasına ağırlıkları neden serbest bıraktı? Cevap temsil geometrisindedir:
1. **Giriş Embedding ($W_{\text{in}}$):** Eşanlamlıları kümeleyen ve pürüzsüz bağlamsal taban oluşturan sürekli bir semantik haritadır.
2. **Çıktı Head ($W_{\text{out}}$):** Yakın adayları (`"oturdu"` vs. `"uzandı"`) hiper-düzlemlerle ayıran keskin bir doğrusal sınıflandırıcıdır.

Ağırlıklar serbest bırakıldığında, giriş ve çıkış uzayları birbirine neredeyse tamamen dikleşir (**ortogonal**, $\cos \theta \approx 0,0007$). SVD analizleri, bağlı modellerin enerjisinin $\%90$'ını tek bir baskın bileşene hapsettiğini; serbest modellerin ise 4.096 boyutun tamamını tam ranklı (full-rank) kullandığını kanıtlar.

Google **Gemma 2** ise **256.000 tokenlık** devasa sözlüğü nedeniyle `"tie_word_embeddings": true` seçmiştir ($d=3584$'te serbest katman tek başına $917,5\text{M}$ parametre ekleyip tüketici GPU'larına sığmayı engellerdi).

<svg viewBox="0 0 560 210" role="img" aria-label="Bağlı ve serbest embedding modellerinin geometrik ve spektral karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="38" text-anchor="middle" style="fill:var(--c-accent);font-size:13px;font-weight:700">BAĞLI AĞIRLIKLAR (Gemma 2)</text>
<text x="144" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">W_out = W_in (Ortak Bellek)</text>
<rect x="36" y="70" width="216" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="144" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Kosinüs Benzerliği: cos(W_in, W_out) = 1.0</text>
<text x="36" y="126" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• Çıktı katmanı için 0 ek parametre</text>
<text x="36" y="146" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• SVD: Spektral enerjinin %90'ı 1. bileşende</text>
<text x="36" y="166" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px">• Logit kayması ve aşırı özgüven riski</text>
<rect x="288" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416" y="38" text-anchor="middle" style="fill:var(--c-success);font-size:13px;font-weight:700">SERBEST AĞIRLIKLAR (Llama 3.1)</text>
<text x="416" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">W_out ≠ W_in (Bağımsız Matrisler)</text>
<rect x="308" y="70" width="216" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Kosinüs Benzerliği: cos(W_in, W_out) ≈ 0.0007</text>
<text x="308" y="126" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• Tam ranklı, birbirine dik semantik uzaylar</text>
<text x="308" y="146" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">• Giriş kodlama vs ayrık sınıflandırma ayrımı</text>
<text x="308" y="166" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">• Bedel: VRAM'de +525M ek parametre</text>
</svg>

---

## 4. Logitleri dizginlemek: Gemma 2 ve logit soft-capping inovasyonu

### Sınırsız logitlerin patolojisi

Özellikle ağırlıkları bağlı ve devasa sözlüklü modellerde $z_i = h \cdot W_i^T$ skaler çarpımının teorik bir üst sınırı yoktur. Bir kelimenin logiti $+100$'e fırlarsa, $e^{100}$ diğer tüm adayları ezerek tek bir kelimeye $P_i \approx 0,999999$ olasılık atar. Dağılımın entropisi çöker, model aşırı özgüvene (overconfidence) kapılır ve çıkarımda sonsuz tekrar döngülerine (repetition loops) saplanır.

### Hiperbolik tanjant soft-capping formülasyonu

Google DeepMind araştırmacıları, gradyanları sıfırlayan sert budamalar (hard clipping) yerine, Gemma 2 mimarisinde **Logit Soft-Capping** adını verdikleri pürüzsüz normalizasyon fonksiyonunu geliştirdi:

$$z_{\text{capped}} = c \cdot \tanh\left(\frac{z_{\text{raw}}}{c}\right)$$

Şöyle okuyun:
- $z_{\text{raw}} = h \cdot W^T$, doğrusal katmandan çıkan ham logit değeridir.
- $c$, izin verilen maksimum kapasite eşiğidir (hiperparametre).
- $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$, çıktısı daima $(-1, +1)$ aralığına hapsedilmiş hiperbolik tanjant fonksiyonudur.
- $z_{\text{capped}}$ logit değeri matematiksel olarak $(-c, +c)$ açık aralığında kalmaya zorlanır.

Gemma 2 model konfigürasyonundaki resmi değerler şunlardır:
- **Attention Logit Soft-Capping:** $c = 50,0$ (`attn_logit_softcapping = 50.0`)
- **Çıktı Head Logit Soft-Capping:** $c = 30,0$ (`final_logit_softcapping = 30.0`)

```text
Çalışma Rejimlerinin Analizi:
1. Doğrusal Rejim (|z| << c):
   Logitler küçükken (örn. z = 1.0, c = 30.0):
   tanh(1/30) ≈ 1/30 ==> z_capped ≈ 30 * (1/30) = 1.0.
   Fonksiyon neredeyse birim dönüşüm (identity) gibi davranır, göreli farkları korur.

2. Doygunluk Rejimi (|z| >> c):
   Logitler aşırı büyüdüğünde:
   tanh(z/c) pürüzsüzce ±1.0 asimptotuna yaklaşır ==> z_capped ±30.0 sınırında durur.
   Hiçbir kelime olasılık dağılımını tekeline alamaz; sayısal taşma riski sıfırlanır.
```

Bu fonksiyonun ham logite göre türevi de son derece yumuşaktır:

$$\frac{\partial z_{\text{capped}}}{\partial z_{\text{raw}}} = c \cdot \frac{d}{dz_{\text{raw}}} \left[ \tanh\left(\frac{z_{\text{raw}}}{c}\right) \right] = c \cdot \frac{1}{c} \left(1 - \tanh^2\left(\frac{z_{\text{raw}}}{c}\right)\right) = 1 - \left(\frac{z_{\text{capped}}}{c}\right)^2$$

$z_{\text{raw}}$ küçükken türev $1,0$'dır (gradyan kayıpsız akar). $z_{\text{raw}}$ patladığında ise türev sıfıra yaklaşarak doğal bir gradyan sönümleyici işlevi görür ve eğitimdeki loss patlamalarını önler.

---

## 5. Olasılık uzayına geçiş: Logitler, Softmax ve yer değiştirme değişmezliği

### Logitler: Normalize edilmemiş logaritmik şans skorları

Çıktı katmanından elde edilen $z = [z_1, z_2, \dots, z_V]$ vektörü terminolojide **Logits** olarak anılır. Bu terim aslen *log-odds* ($\ln \frac{p}{1-p}$) kavramından türemiştir. Logitler $(-\infty, +\infty)$ aralığında salınan, mutlak değerleri değil sadece aralarındaki farklar anlam taşıyan reel sayılardır.

Bu ham skorları geçerli bir ayrık olasılık dağılımına dönüştürmek için:

$$0 \le P(i) \le 1 \quad \text{ve} \quad \sum_{i=1}^V P(i) = 1$$

Vektör **Softmax** fonksiyonundan geçirilir:

$$P(i) = \text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}$$

Softmax iki kritik özellik taşır:
1. **Monotoniklik:** Eğer $z_a > z_b$ ise daima $P(a) > P(b)$ olur. Modelin kelime sıralaması asla bozulmaz.
2. **Üstel Büyütme:** Logitler arasındaki küçük farklar üstel fonksiyonla katlanarak modelin belirgin tercihler yapmasını sağlar.

### IEEE 754 kayan nokta taşma felaketi

Modern hızlandırıcılarda (NVIDIA H100, B200) eğitimler **bfloat16** ya da **float16** formatlarında yürütülür.

IEEE 754 16-bit yarı hassasiyetli formatta (`float16`):
- Temsil edilebilen maksimum pozitif sayı: $65.504$
- En küçük pozitif alt-normal sayı: $5,96 \times 10^{-8}$

Eğer kontrolsüz bir logit $z_i = 100$ değerine ulaşırsa:

$$e^{100} \approx 2,688 \times 10^{43} \gg 65.504 \implies \textbf{TAŞMA (OVERFLOW) } \to +\infty$$

Payda $+\infty$ olur, bölme işlemi $\frac{\infty}{\infty}$ belirsizliği üretir ve tensörün tamamı **NaN (Not a Number)** ile dolar. Tek bir adımda modelin tüm ağırlıkları çöker.

### Yer değiştirme değişmezliği (Shift Invariance): Sayısal kurtuluş

Derin öğrenme çatıları bu felaketi Softmax'ın **Yer Değiştirme Değişmezliği (Shift Invariance)** özelliğiyle aşar. Logit vektöründeki her bir elemandan sabit bir $C$ sayısı çıkarıldığında olasılık dağılımı matematiksel olarak zerre kadar değişmez:

$$\frac{e^{z_i - C}}{\sum_{j=1}^V e^{z_j - C}} = \frac{e^{z_i} \cdot e^{-C}}{\sum_{j=1}^V (e^{z_j} \cdot e^{-C})} = \frac{e^{-C} \cdot e^{z_i}}{e^{-C} \cdot \sum_{j=1}^V e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}}$$

Bu kuraldan yararlanarak $C$ sabiti vektörün en büyük elemanı olarak seçilir:

$$C = \max(z) = \max_{k} z_k$$

$$z'_i = z_i - \max(z)$$

Bu durumda en büyük kaydırılmış logit:

$$z'_{\text{max}} = \max(z) - \max(z) = 0 \implies e^0 = 1,0$$

Diğer tüm elemanlar ise sıfırdan küçük veya eşit olur ($z'_i \le 0$). Böylece hesaplanan her bir üstel değer daima güvenli aralıkta kalır:

$$e^{z'_i} \in (0, 1,0]$$

Sayısal taşma (overflow) imkansız hale gelir. Payda $\sum e^{z'_j}$ en az $1,0$ olacağı için sıfıra bölünme hatası da engellenmiş olur.

---

## 6. Çapraz Entropi (Cross-Entropy): Bilgi kuramı ve asimptotik ceza dinamiği

### Bilgi kuramı ve sürpriz derecesi

Eğitim sırasında dil modelinin kelime tahminindeki hatası nasıl ölçülür? Claude Shannon'ın Bilgi Kuramı'na başvururuz.

Bir olayın gerçekleşme olasılığı $P$ ise, o olayı gözlemlemenin taşıdığı bilgi miktarı—ya da **sürpriz derecesi (surprisal)**—şöyle tanımlanır:

$$I(x) = -\log_2 P(x)$$

Eğer model kesin olan bir kelimeyi $P=1,0$ ile tahmin ederse sürpriz $-\log_2(1,0) = 0$'dır. Eğer binde bir ihtimal verdiği bir kelime çıkarsa sürpriz $-\log_2(0,001) \approx 9,96$ bit olur.

Eğitimde doğru hedef, tek bir indeksi $1,0$, diğer tüm indeksleri $0,0$ olan bir **One-Hot vektör** $y \in \{0, 1\}^V$ ile temsil edilir ($c$ doğru token için $y_c = 1,0$, $j \ne c$ için $y_j = 0$).

**Çapraz Entropi Kaybı ($\mathcal{L}$)**, gerçek hedef dağılım $y$ ile modelin tahmin ettiği Softmax dağılımı $P$ arasındaki uyumsuzluğu ölçer:

$$\mathcal{L} = -\sum_{j=1}^V y_j \ln(P_j)$$

Doğru kelime dışındaki tüm $y_j$'ler sıfır olduğundan bu devasa toplam olağanüstü yalın bir forma indirgenir:

$$\mathcal{L} = -\ln(P_c)$$

Şöyle okuyun:
- $\mathcal{L}$, hedeflenen doğru kelimenin negatif log-olasılığıdır.
- Kayıp fonksiyonu **yalnızca ve yalnızca** doğru kelimeye modelin biçtiği olasılığı cezalandırır. Rakip kelimelerin logitleri denkleme sadece $P_c$'nin paydasındaki normalizasyon üzerinden dahil olur.

```text
Çapraz Entropi Ceza Profili:
  Hedef Kelime Olasılığı P_c = 1,00  ==>  Kayıp = -ln(1,00) = 0,0000 nat (Sıfır ceza)
  Hedef Kelime Olasılığı P_c = 0,50  ==>  Kayıp = -ln(0,50) = 0,6931 nat
  Hedef Kelime Olasılığı P_c = 0,10  ==>  Kayıp = -ln(0,10) = 2,3026 nat
  Hedef Kelime Olasılığı P_c = 0,01  ==>  Kayıp = -ln(0,01) = 4,6052 nat
  Hedef Kelime Olasılığı P_c -> 0,00 ==>  Kayıp -> +sonsuz (Asimptotik Ceza Duvarı)
```

Bu asimptotik ceza karakteri, dil modelinin nadir görülen uzun kuyruk (long-tail) kelimelerini, teknik terimleri ve dilbilgisi kurallarını unutmasını engeller. Model emin olmadığı bir kelimeye sıfıra yakın olasılık atarsa ceza tavana vurur.

---

## 7. Geri yayılımın zarafeti: Softmax ve Çapraz Entropi türevi (dL/dz = P - y)

### Softmax'ın Jacobian matrisi

Yapay zeka literatüründe Çapraz Entropi ve Softmax neden ayrılmaz bir ikilidir? Bu sorunun cevabı, zincir kuralında ortaya çıkan matematiksel sadeleşmede yatar.

Önce Softmax olasılığı $P_i$'nin keyfi bir $z_k$ logitine göre türevini alalım:

$$P_i = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}} = \frac{e^{z_i}}{S}$$

Bölüm kuralı $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$ uygulandığında:

**Durum 1: $i = k$ (Kendi logitine göre türev)**

$$\frac{\partial P_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} \cdot \frac{S - e^{z_i}}{S} = P_i (1 - P_i)$$

**Durum 2: $i \ne k$ (Rakip bir logite göre türev)**

$$\frac{\partial P_i}{\partial z_k} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_k}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_k}}{S} = -P_i P_k$$

Kronecker delta simgesi $\delta_{ik}$ kullanılarak ($i=k$ ise $1$, değilse $0$):

$$\frac{\partial P_i}{\partial z_k} = P_i (\delta_{ik} - P_k)$$

### Zincir kuralının mucizesi

Şimdi skaler kayıp fonksiyonu $\mathcal{L} = -\sum_j y_j \ln(P_j)$'nin $z_k$ logitine göre kısmi türevini zincir kuralıyla bağlayalım:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{j=1}^V \frac{\partial \mathcal{L}}{\partial P_j} \frac{\partial P_j}{\partial z_k}$$

$\frac{\partial \mathcal{L}}{\partial P_j} = -\frac{y_j}{P_j}$ türevini yerine koyduğumuzda:

$$\frac{\partial \mathcal{L}}{\partial z_k} = \sum_{j=1}^V \left( -\frac{y_j}{P_j} \right) \left[ P_j (\delta_{jk} - P_k) \right]$$

Pay ve paydadaki $P_j$ terimleri birbirini anında götürür:

$$\frac{\partial \mathcal{L}}{\partial z_k} = -\sum_{j=1}^V y_j (\delta_{jk} - P_k) = -\sum_{j=1}^V y_j \delta_{jk} + \sum_{j=1}^V y_j P_k$$

$y$ geçerli bir tekil dağılım olduğundan ($\sum_{j=1}^V y_j = 1$) ve $\sum_{j=1}^V y_j \delta_{jk} = y_k$ sağlandığından:

$$\frac{\partial \mathcal{L}}{\partial z_k} = -y_k + P_k \cdot 1 = P_k - y_k$$

Vektörel gösterimle:

$$\frac{\partial \mathcal{L}}{\partial z} = P - y = \text{Tahmin Edilen Olasılık} - \text{Gerçek Hedef}$$

```text
Doğrusal Gradyan Dengesi:
- Hedef Token c:   dL/dz_c = P_c - 1,0  (Daima NEGATİF: logiti +sonsuza doğru YUKARI iter)
- Rakip Token j:   dL/dz_j = P_j - 0,0  (Daima POZİTİF: logitleri -sonsuza doğru AŞAĞI bastırır)
```

Üstel bölmeler, logaritmalar ve karmaşık Jacobian matrisleri tek bir çıkarmaya dönüşür. Gradyan daima $[-1,0, +1,0]$ aralığındadır. Asla doygunluğa ulaşıp sıfırlanmaz (vanishing gradient yaşanmaz); model yanlış yaptıkça geriye akan sinyal her zaman canlı kalır.

### Ağırlıklara ve omurgaya gradyan akışı

$\frac{\partial \mathcal{L}}{\partial z} \in \mathbb{R}^V$ logit gradyanı elde edildikten sonra zincir kuralı geriye doğru akar:

1. **LM Head Ağırlık Gradyanı ($W_{\text{head}}$):**
   $$\frac{\partial \mathcal{L}}{\partial W_{\text{head}}} = \left(\frac{\partial \mathcal{L}}{\partial z}\right) \otimes h^T \in \mathbb{R}^{V \times d}$$
   Hata vektörü ile gelen aktivasyonun dış çarpımıdır.
2. **Alt Katmanlara (Transformer Bloğuna) Akan Gradyan ($h$):**
   $$\frac{\partial \mathcal{L}}{\partial h} = W_{\text{head}}^T \cdot \left(\frac{\partial \mathcal{L}}{\partial z}\right) \in \mathbb{R}^d$$
   Bu vektör son RMSNorm katmanından geçerek geriye doğru 32 katmanın tamamına yayılır.

---

## 8. Adım adım küçük sayısal yürüyüş: İleri geçiş, soft-capping, kayıp ve geri yayılım

Tüm bu formüllerin donanım seviyesinde nasıl çalıştığını somutlaştırmak için $d = 4$ gizli boyut ve $V = 5$ kelimelik bir sözlük üzerinden tam bir ileri ve geri yayılım yürüyüşü yapalım:

```text
   [Son Gizli Durum h: 1x4] ──> [0.5, -1.2, 0.8, 1.5]
                                      │
                                      ▼  (x W_head^T: 4x5)
   Ham Logitler (z_raw):              [0.69,  0.25,  1.71,  0.17, -1.24]
                                      │
                                      ▼  (Soft-capping: c=5 · tanh(z/5))
   Capped Logitler (z_cap):           [0.69,  0.25,  1.65,  0.17, -1.22]
                                      │
                                      ▼  (Shift-Invariant: z - 1.6463)
   Softmax Olasılıkları (P):          [%19.97, %12.92, %52.20, %11.93, %2.98]
                                      │                          ▲
                                      │                     [GERÇEK HEDEF]
                                      ▼
   Çapraz Entropi Kaybı:              -ln(0.1193) = 2.1261 nat
                                      │
                                      ▼  (dL/dz_cap = P - y)
   Hata Sinyali (dL/dz):              [+0.196, +0.129, +0.465, -0.880, +0.028]
                                                                 ▲
                                      (Hedef yukarı itilir, rakipler bastırılır!)
```

### Kurulum ve oyuncak parametreler

- Gizli boyut: $d = 4$
- Kelime dağarcığı: $V = 5$ token:
  - İndeks 0: `[PAD]`
  - İndeks 1: `"bir"`
  - İndeks 2: `"kediyi"`
  - İndeks 3: `"tavşanı"` (**Gerçek Hedef Token**)
  - İndeks 4: `"köpek"`
- Soft-capping kapasite sınırı: $c = 5,0$
- 32. katmandan gelen gizli durum vektörü:
  $$h = [0,5, -1,2, 0,8, 1,5]$$
- LM Head ağırlık matrisi $W_{\text{head}} \in \mathbb{R}^{5 \times 4}$:
  $$W_{\text{head}} = \begin{bmatrix}
   0,4 & -0,2 &  0,5 & -0,1 \\
  -0,3 &  0,6 & -0,1 &  0,8 \\
   0,7 & -0,5 &  0,2 &  0,4 \\
  -0,2 &  0,8 &  0,6 &  0,5 \\
   0,5 &  0,1 & -0,4 & -0,7
  \end{bmatrix}$$

### Adım 1: İleri geçiş ve soft-capping ($z_{\text{raw}} \to z_{\text{cap}}$)

Her bir kelime skoru $z_i = h \cdot W_{i, :}^T$ iç çarpımıyla bulunur. Ardından Gemma 2 tarzı $z_{\text{cap}} = 5,0 \cdot \tanh(z_{\text{raw}} / 5,0)$ tavanlaması uygulanır.

Örneğin sahte lider olan 2. kelime için:
- Ham skor: $z_2 = (0,5)(0,7) + (-1,2)(-0,5) + (0,8)(0,2) + (1,5)(0,4) = 1,7100$
- Tavanlanmış skor: $5,0 \cdot \tanh(1,7100 / 5,0) \approx 1,6463$

### Adım 2: Kararlı Softmax ve Çapraz Entropi kaybı

Maksimum logit değeri $m = 1,6463$ çıkarılarak ($z' = z_{\text{cap}} - m$) üsteller alınır ve normalize edilir. Hedef token `"tavşanı"` (indeks 3) için model yalnızca $\%11,93$ olasılık üretmiştir; sahte lider `"kediyi"` ise $\%52,20$ ile öndedir!

Çapraz Entropi kaybı doğrudan hedef kelimenin negatif log-olasılığıdır:

$$\mathcal{L} = -\ln(P_3) = -\ln(0,1193) = 2,1261\text{ nat}$$

### Adım 3: Geri yayılım ve hata sinyali ($dL / dz$)

Hedef token için $y = [0, 0, 0, 1, 0]$ olduğundan gradyan otoyolu $P - y$ farkını alır; ardından soft-capping ölçeğiyle $(1 - \tanh^2)$ çarpılarak ham logit gradyanı elde edilir:

| İndeks | Kelime | Durum | Ham Skor ($z_{\text{raw}}$) | Capped ($z_{\text{cap}}$) | Olasılık ($P$) | Ham Gradyan ($dL/dz$) | Gradyanın Fiziksel Anlamı |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0` | `[PAD]` | Rakip | $+0,6900$ | $+0,6857$ | $\%19,97$ | $+0,1960$ | Skoru aşağı bastır |
| `1` | `"bir"` | Rakip | $+0,2500$ | $+0,2498$ | $\%12,92$ | $+0,1288$ | Skoru aşağı bastır |
| `2` | `"kediyi"` | Sahte Lider | $+1,7100$ | $+1,6463$ | $\%52,20$ | $+0,4654$ | Skoru sertçe bastır! |
| `3` | `"tavşanı"` | **Gerçek Hedef** | $+0,1700$ | $+0,1699$ | $\%11,93$ | **$-0,8797$** | **Skoru yukarı fırlat!** |
| `4` | `"köpek"` | Rakip | $-1,2400$ | $-1,2152$ | $\%2,98$ | $+0,0281$ | Nötr / minimal etki |

### Adım 4: Omurgaya ve ağırlıklara aktarım

Elde edilen hata sinyali zincir kuralıyla geriye akar:
1. **$W_{\text{head}}$ Ağırlık Matrisi Güncellemesi:** Her satır gradyan ile $h^T$'nin dış çarpımıyla güncellenir ($\frac{\partial \mathcal{L}}{\partial W} = g_{\text{raw}} \otimes h^T$).
2. **Transformer Omurgasına Geri Pompalanan Hata:**
   $$\frac{\partial \mathcal{L}}{\partial h} = \sum_{i=0}^4 \left(\frac{\partial \mathcal{L}}{\partial z_{\text{raw}, i}}\right) W_{i, :} = [0,5555, -0,8956, -0,3609, -0,1899]$$

Bu 4 boyutlu hata vektörü, son RMSNorm katmanından geçerek 32 katmanlı Transformer omurgasının en derinlerine kadar taşınır.


---

## 9. Donanımsal bellek duvarı: Megatron-LM kelime dağarcığı paralelizmi ve Liger Kernel

### HBM tahsis patlaması

Llama 3.1 8B modelini 8.192 bağlam uzunluğu ve 4 batch boyutuyla eğittiğimizi varsayalım:

```text
Logit Tensör Boyutları: [4, 8192, 128256]
Toplam Eleman Sayısı:   4 * 8.192 * 128.256 = 4.202.692.608 eleman
bfloat16 Bellek (İleri): 4,20 Milyar * 2 bayt = 8,405 GB
bfloat16 Bellek (Geri):  4,20 Milyar * 2 bayt = 8,405 GB
Softmax Ara Bellekleri ve Aktivasyonlar:      ~8,4 GB
Yalnızca Son Katman İçin Gerekli VRAM:        > 25 GB!
```

### Uzun bağlam krizi: LM Head'in Attention'ı geride bıraktığı an (O(S · V) patlaması)

Modeller 8k'dan 32k, 64k ve 128k bağlam pencerelerine ölçeklendiğinde mimari dengeler kökten sarsılır.

Transformer omurgasında, karesel $O(S^2)$ attention bellek yükü, Query, Key ve Value bloklarını doğrudan GPU SRAM üzerinde işleyen ve tam $S \times S$ matrisini HBM'e asla yazmayan **FlashAttention** (Dao vd.) sayesinde dizginlenmiştir.

Fakat Çıktı Katmanında, kaynaşmamış (un-fused) logit tensörü $O(S \cdot V)$ ölçeğinde büyür. Kelime dağarcığı boyutu $V$ devasa olduğundan (Llama 3.1'de $128.256$ veya Gemma 2'de $256.000$), dizi uzunluğunu $S$ artırmak korkunç bir bellek patlaması yaratır:

```text
Llama 3.1 Logit Bellek Ölçeklemesi (V = 128.256, bfloat16, Batch = 1):
- Bağlam S = 4.096:    4.096 * 128.256 * 2 bayt   ≈  1,05 GB
- Bağlam S = 8.192:    8.192 * 128.256 * 2 bayt   ≈  2,10 GB  (Batch=4 için: 8,4 GB)
- Bağlam S = 32.768:  32.768 * 128.256 * 2 bayt   ≈  8,40 GB  (Tek bir dizi!)
- Bağlam S = 131.072: 131.072 * 128.256 * 2 bayt  ≈ 33,60 GB  (İleri yayılım)
  Geri Yayılım Gradyanı: 131.072 * 128.256 * 2 bayt ≈ 33,60 GB
  TOPLAM LOGİT VRAM:    33,60 GB + 33,60 GB       = 67,20 GB! (YALNIZCA TEK DİZİ İÇİN!)
```

> [!CRITICAL]
> **Uzun bağlam rejimlerinde LM Head, tüm model içerisindeki 1 numaralı bellek darboğazı olarak Attention'ın önüne geçer.**
> FlashAttention sayesinde 80 GB'lık bir GPU 128k'lık bir istemin attention katmanlarını rahatlıkla işleyebilirken, aynı 128k dizinin çıktı logit tensörünü oluşturmaya çalışmak tek bir veri için 67,2 GB VRAM talep eder ve anında bellek yetersizliği (OOM) çöküşüne yol açar.

80 GB'lık bir NVIDIA A100/H100 kartında sadece tek bir katmanın logitleri için $25\text{ GB}$ ila $67\text{ GB}$ VRAM ayırmak bir mühendislik felaketidir. Endüstri bu duvarı yıkmak için iki öncü teknoloji geliştirmiştir:

### 1. NVIDIA Megatron-LM: Vocab Paralelizmi (`VocabParallelCrossEntropy`)

Çoklu GPU kümelerinde Megatron-LM, çıkış katmanını sütunları boyunca bölerek Tensör Paralelizmi uygular (`ColumnParallelLinear`).

Örneğin 8 GPU'lu bir kümede (TP=8), 128.256 kelimelik matris dilimlenir; her GPU'ya yalnızca $16.032$ kelimelik bir parça düşer.

```text
GPU 0: [0       ... 16.031] tokenlarını tutar
GPU 1: [16.032  ... 32.063] tokenlarını tutar
...
GPU 7: [112.224 ... 128.255] tokenlarını tutar
```

Buradaki zorluk: *Her GPU kendi paydasını hesaplamak için tüm 128.256 kelimenin logitlerine ihtiyaç duyarken, devasa tensörleri ağ üzerinden taşımadan Softmax nasıl hesaplanır?*

Megatron-LM `VocabParallelCrossEntropy` modülüyle bu sorunu çözer:
1. **Yerel Maksimum:** Her GPU kendi dilimindeki en büyük logiti bulur: $m_{\text{local}} = \max(z_{\text{local}})$.
2. **Global Maksimum İndirgeme:** GPU'lar arasında `all-reduce(op=MAX)` çağrılır; yalnızca token başına tek bir skaler sayı iletilerek küme çapında global maksimum ($m_{\text{global}}$) elde edilir.
3. **Yerel Üsteller:** Her GPU kendi verisinden $e^{z_{\text{local}} - m_{\text{global}}}$ hesaplar ve kendi toplamını ($S_{\text{local}}$) bulur.
4. **Global Toplam İndirgeme:** İkinci bir `all-reduce(op=SUM)` ile tüm kümenin ortak paydası ($S_{\text{global}} = \sum S_{\text{local}}$) çıkarılır.
5. **Kayıp Hesabı:** Gerçek hedef token hangi GPU'nun belleğindeyse, o GPU $\mathcal{L} = -(z_{\text{hedef}} - m_{\text{global}}) + \ln(S_{\text{global}})$ formülüyle kaybı bulur.

Sonuç: **Ağ üzerinden tek bir gigabaytlık logit tensörü bile transfer edilmez.** İletişim maliyeti mikrosaniyelik skaler indirgemelerden ibarettir.

### Context Parallelism (CP) ve dağıtık etiket dilimleme tuzağı

Dağıtık kümelerde 128k bağlam uzunluğunda ön-eğitim yapılırken, Tensör Paralelizmine **Context Parallelism (CP)** eşlik eder (Megatron-LM içindeki Ring Attention gibi `--context-parallel-size` ile).

Context Parallelism altında dizi boyutu $S$ GPU'lar arasında paylaştırılır: 8 GPU'lu bir CP grubu, 128k token'ı GPU başına $S_{\text{yerel}} = 16.384$ tokenlık yerel parçalara böler. Son katmanda her GPU'nun `lm_head` modülü yalnızca kendi yerel dizi dilimi için kayıp hesaplar.

Bu durum sinsi bir mühendislik hatasına yol açar: **Etiket Uyuşmazlığı (Label Misalignment)**. Eğer kayıp fonksiyonuna 128k uzunluğundaki global hedef etiket tensörü verilirken, yerel logit tensörü yalnızca 16k token içeriyorsa, çapraz entropi tamamen yanlış kelimeleri karşılaştırır ve eğitim anında ıraksar (divergence). Üretim sistemleri hedef etiketleri kesin CP-yerel dizi ofsetleriyle ($\text{cp\_rank} \cdot S_{\text{yerel}}$) dilimlemeli ve hem TP hem CP iletişim grupları üzerinde kaynaşmış kayıp indirgemesi çalıştırmalıdır.

### 2. Liger Kernel: Triton tabanlı Fused Linear Cross-Entropy

Megatron-LM'in dağıtık küme altyapısına girmeden, doğrudan Hugging Face üzerinde çalışan modeller için LinkedIn mühendisleri **Liger Kernel**'ı geliştirdi.

Çekirdek inovasyon `FusedLinearCrossEntropy` operatörüdür. Standart PyTorch'ta matris çarpımı (`H @ W.T`) tüm 8,4 GB'lık tensörü önce HBM'e (High Bandwidth Memory) yazar; ardından `CrossEntropyLoss` bu 8,4 GB'ı tekrar okur.

Liger Kernel, OpenAI'ın **Triton** dilini kullanarak bu iki adımı tek bir GPU çekirdeğinde (kernel) birleştirir:
- **SRAM Üzerinde Parçalı (Chunked) İşlem:** Sıralama 512 veya 1.024 token'lık parçalara bölünür.
- **Çip Üstü Hızlı Hesap:** Parça doğrudan çip içi SRAM belleğe (20 TB/s bant genişliği) yüklenir, matris çarpımı yapılır, anında kaydırılmış üstel ve yerel kayıp hesaplanır, gradyan biriktirilir ve ara logit anında bellekten silinir.
- **Sıfır HBM Logit Tahsisi:** `[B, S, V]` boyutundaki devasa logit tensörü **HBM belleğe asla yazılmaz**.

```text
Hugging Face + Liger Kernel Başarımı (Llama 3 8B, 8K Context):
- Zirve VRAM Kullanımı: %60 oranında azaldı (25 GB'tan < 10 GB'a indi)
- Eğitim Hızı (Throughput): %20 ila %30 oranında arttı
- Desteklenen Bağlam: Tek GPU'da OOM almadan 4K'dan 16K'ya çıktı
```

<svg viewBox="0 0 560 210" role="img" aria-label="Megatron-LM Vocab Paralelizmi ile Liger Kernel Fused Linear Cross-Entropy karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="144" y="38" text-anchor="middle" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">MEGATRON-LM (Çoklu-GPU)</text>
<text x="144" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">VocabParallelCrossEntropy</text>
<rect x="32" y="70" width="224" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="144" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_head dilimlendi: GPU başına V / N</text>
<text x="32" y="124" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Yerel max -> all_reduce(MAX)</text>
<text x="32" y="142" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Yerel sum(exp) -> all_reduce(SUM)</text>
<text x="32" y="160" text-anchor="start" style="fill:var(--c-success);font-size:11px">• NVLink üzerinden tam logit geçişi SIFIRDIR</text>
<text x="32" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">• Kümelerde 100k+ sözlüğe ölçeklenir</text>
<rect x="288" y="16" width="256" height="178" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="38" text-anchor="middle" style="fill:var(--c-success);font-size:12.5px;font-weight:700">LIGER KERNEL (Triton Fused)</text>
<text x="416" y="56" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">FusedLinearCrossEntropy</text>
<rect x="304" y="70" width="224" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Triton Çekirdeği: MatMul + Loss Kaynaştı</text>
<text x="304" y="124" text-anchor="start" style="fill:var(--c-text);font-size:11px">• GPU SRAM üzerinde parçalı (chunked) akış</text>
<text x="304" y="142" text-anchor="start" style="fill:var(--c-text);font-size:11px">• Tam [B, S, V] HBM'e ASLA yazılmaz</text>
<text x="304" y="160" text-anchor="start" style="fill:var(--c-success);font-size:11px">• Tek GPU'da %60 VRAM tasarrufu</text>
<text x="304" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">• Hugging Face ile birebir uyumlu</text>
</svg>

---

## 10. İnce ayar (Fine-Tuning) ve kelime dağarcığı genişletme: LoRA ve modules_to_save tuzağı

### Sözlük genişletme ihtiyacı

Alana özgü terimler veya sohbet belirteçleri (`<|im_start|>`, `<|im_end|>`) eklendiğinde `model.resize_token_embeddings()` çağrılır:
- $W_{\text{in}}$ ve $W_{\text{head}}$ matrisleri $(V + \Delta V) \times d$ boyutuna büyür.
- **Yeni eklenen satırlar tamamen rastgele (random) sayılarla başlatılır.**

### Sessiz PEFT / LoRA çöküşü

Standart LoRA eğitiminde ana model dondurulur; adaptörler ($B \cdot A$) yalnızca dikkat ve MLP projeksiyonlarına eklenir.

Tuzak buradadır: **`lm_head` ve `embed_tokens` katmanlarındaki yeni satırlar rastgele değerlerinde donmuş kalır!** Çıkarımda model bu yeni kelimeleri üretmek istediğinde saf rastgele gürültü yansıtır ve anlaşılmaz saçmalıklar üretir.

### Çözüm: `modules_to_save` sözleşmesi

Bu arızayı önlemek için PEFT konfigürasyonuna sözlük katmanlarının tam parametreli olarak gradyanlara açılacağını bildirmek şarttır:

```python
from peft import LoraConfig, get_peft_model

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,05,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["embed_tokens", "lm_head"]  # <-- HAYATİ PARAMETRE!
)
model = get_peft_model(model, peft_config)
```

`modules_to_save` argümanı PEFT'e bu iki katmanı `ModulesToSaveWrapper` içine almasını emreder. Llama gibi serbest ağırlıklı modellerde her iki matris de tam gradyanla eğitilip checkpoint dosyasına yazılır. Gemma 2 gibi bağlı modellerde ise PEFT aradaki bellek bağlantısını korur.

---

## 11. Çıkarım ivmelenmesi: Medusa ve EAGLE ile spekülatif kod çözme

### Özbağlanımlı çıkarımın bellek bant genişliği darboğazı

Çıkarım sırasında LLM'lerin yavaş kalmasının sebebi hesaplama gücü değil, **bellek bant genişliği darboğazıdır (Memory Bandwidth Bound)**.

8 milyar parametreli bir modelde tek bir kelime üretmek için $16\text{ GB}$ ağırlık her adımda HBM'den çekilmelidir. $2,0\text{ TB/s}$ bellek bant genişliğine sahip bir NVIDIA A100 GPU'da:

$$\text{Token Başına Asgari Gecikme} = \frac{16\text{ GB}}{2000\text{ GB/s}} = 8,0\text{ milisaniye} \implies \text{Maksimum Hız} \approx 125\text{ token/s}$$

İşlem birimleri her adımda işsiz kalır. Bu fiziksel sınırı kırmak için **Spekülatif Kod Çözme (Speculative Decoding)** paradigması doğmuştur.

Mantık basittir: Hızlı bir mekanizma sıradaki $K$ kelimeyi tahmin (taslak) eder; ana model tek bir ileri geçişte tüm bu adayları paralel olarak doğrular.

### Medusa: Taslak modelsiz çok başlıklı spekülasyon

**Medusa** (Cai vd., 2024, arXiv:2401.10774), ayrı bir taslak model yükü olmadan, dondurulmuş omurganın tepesine eklenen hafif MLP başlıklarıyla paralel spekülasyon yapar:

```text
Transformer Omurgasından Gelen h_t Gizli Durumu:
  ├── Orijinal lm_head  ──> t+1 tokenını tahmin eder (Ana doğrulama)
  ├── Medusa Başlığı 1  ──> t+2 tokenını speküle eder (Taslak)
  ├── Medusa Başlığı 2  ──> t+3 tokenını speküle eder (Taslak)
  └── Medusa Başlığı 3  ──> t+4 tokenını speküle eder (Taslak)
```

Tek bir geçişte:
1. Medusa başlıkları ardışık adımlar için aday ağacı kurar.
2. **Tree Attention (Ağaç Dikkati)** mekanizmasıyla ana model bu ağacın tüm dallarını tek bir paralel ileri geçişte doğrular.
3. Onaylanan token'lar tek adımda metne eklenerek kalite kaybı olmadan **$2,2\times$ ila $3,6\times$ hızlanma** sağlanır.

### EAGLE: Özellik seviyesinde spekülatif örnekleme

**EAGLE** (Li vd., ICML 2024, arXiv:2401.15077), spekülasyonu gürültülü token seviyesinden **Özellik Seviyesine (Feature-Level)** taşır. Gelecekteki sürekli gizli durumları ($h_{t+1}, h_{t+2}$) tahmin edip orijinal dondurulmuş `lm_head`'e iletir; özellik sürekliliği sayesinde taslak kabul oranını $\%80$'in üzerine çıkararak **$2,7\times$ ila $3,5\times$ hızlanma** sağlar.

<svg viewBox="0 0 560 250" role="img" aria-label="Standart özbağlanımlı üretim, Medusa çok başlıklı spekülasyon ve EAGLE özellik seviyesi spekülasyon karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="528" height="60" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="36" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">1. STANDART ÖZBAĞLANIMLI (AUTOREGRESSIVE)</text>
<text x="28" y="56" text-anchor="start" style="fill:var(--c-text);font-size:12px">Adım 1: Token 1  ──>  Adım 2: Token 2  ──>  Adım 3: Token 3  (Bellek Bant Genişliği Darboğazı: 1 token / adım)</text>
<rect x="16" y="86" width="528" height="70" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="106" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:600">2. MEDUSA (Çok Başlıklı Ağaç Spekülasyonu - Cai vd. 2024)</text>
<text x="28" y="124" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Omurga Gizli Durumu h_t ──> Başlık 1 (t+1), Başlık 2 (t+2), Başlık 3 (t+3) paralel çalışır</text>
<text x="28" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Tree Attention adayları tek geçişte doğrular  ──>  2,2x ila 3,6x hızlanma</text>
<rect x="16" y="166" width="528" height="70" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="186" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">3. EAGLE (Özellik Seviyesinde Spekülasyon - Li vd. ICML 2024)</text>
<text x="28" y="204" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Üst katman gizli durumlarını (h_(t+k)) tahmin eder ──> Orijinal dondurulmuş lm_head'e gönderir</text>
<text x="28" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Token belirsizliğini aşar, >%80 taslak onay oranı  ──>  2,7x ila 3,5x hızlanma</text>
</svg>

---

## 12. Özbağlanımlı üretim döngüsü: model.generate() ve örnekleme stratejileri

Hugging Face veya vLLM üzerinde metin üretirken tüm bu hat `generate()` döngüsü tarafından idare edilir. `lm_head` katmanından çıkan ham logit vektörü $z \in \mathbb{R}^V$, bir kelimeye dönüşene kadar şu aşamalardan geçer:

### 1. Prefill ve decode ayrımı

- **Prefill Aşaması:** Girdi prompt'u bir kerede işlenir. Tüm prompt token'larının $K, V$ tensörleri hesaplanıp **KV Cache**'e yazılır. `lm_head` sadece prompt'un en son kelimesi için çalıştırılır.
- **Decode Aşaması:** Tekil üretim başlar. Her adımda yalnızca yeni üretilen kelime işlenir, geçmiş token'ların tensörleri KV Cache'ten okunur.

### Uzun bağlam prefill dilimlemesi: 100k tokenlık GEMM hatasını önlemek

Özel çıkarım sunucularında (custom serving pipelines) devasa istemler (örneğin 100.000 tokenlık bir PDF veya kod deposu) işlenirken prefill aşamasında sıklıkla kritik bir performans hatası yapılır.

Acemice bir kodlamada:
```python
# ACEMİ GEMM HATASI:
# GPU'yu gereksiz yere 100.000 x 128.256 matris çarpımı yapmaya zorlar!
# Gigabaytlarca gereksiz logit üreterek TTFT gecikmesini tavan yaptırır.
logits = model(input_ids).logits
```

Bu neden bir felakettir? Çünkü prefill aşamasında modelin tek görevi, **prompt'tan sonra gelecek olan ilk yeni kelimeyi** tahmin etmektir. Önceki 99.999 pozisyonun logitleri örneklemeye hiç girmeden çöpe atılır!

Üretim seviyesindeki çıkarım motorları (vLLM, TensorRT-LLM, Hugging Face `generate`) bu israfı önlemek için hayati bir optimizasyon uygular: **LM Head'e girmeden önce dilimleme yapmak**:
```python
# ÜRETİM SEVİYESİ DİLİMLEME (vLLM / Hugging Face generate):
# Yalnızca en son gizli durum vektörünü çek:
last_hidden_state = hidden_states[:, -1:, :]  # Şekil: [1, 1, 4096]
next_token_logits = model.lm_head(last_hidden_state)  # Şekil: [1, 1, 128256]
```
Bu tek dilimleme işlemi, çıktı katmanının prefill sırasındaki işlem ve bellek yükünü $O(S \cdot V)$ düzeyinden $O(1 \cdot V)$ düzeyine indirir; son katmanda anında $100.000\times$ hızlanma sağlar!

### 2. LogitsProcessor filtre zinciri

Logitler olasılığa çevrilmeden önce bir dizi matematiksel filtreden geçer:

- **Tekrar Cezası (Repetition Penalty):** Daha önce üretilmiş kelimelerin ($g \in G$) logitleri cezalandırılır:
  $$z_g = \begin{cases} z_g / \theta & \text{eğer } z_g > 0 \\ z_g \cdot \theta & \text{eğer } z_g \le 0 \end{cases} \quad (\theta > 1,0)$$
  Pozitif logitleri $\theta$'ya (örn. $1,15$) bölmek olasılıklarını düşürür.
- **Sıcaklık Ölçeklemesi (Temperature - $T$):** Dağılımın sivrilik derecesini belirler:
  $$z'_i = \frac{z_i}{T}$$
  - $T \to 0$: Dağılım bir Dirac deltasına (argmax, deterministik) çöker.
  - $T = 1,0$: Modelin doğal olasılık dağılımı korunur.
  - $T > 1,0$: Dağılım düzleşir (uniformlaşır), rastgelelik ve yaratıcılık artar.
- **Yasaklı Kelime Filtreleri (Bad Words):** Üretilmesi istenmeyen kelimelerin logitleri $-\infty$ yapılarak $e^{-\infty} = 0$ ile elenir.

### 3. Örnekleme stratejileri

- **Açgözlü Arama (Greedy Search):** Doğrudan en yüksek skorlu kelimeyi seçer: $t = \text{argmax}(z)$. Hızlıdır fakat yerel optimumlara ve tekrarlara hapsolabilir.
- **Işın Araması (Beam Search):** En olası $B$ farklı alternatif cümleyi aynı anda takip eder. Çeviri ve özetleme için idealdir, ancak serbest sohbette hesaplama maliyeti yüksektir.
- **Top-$K$ Örneklemesi:** En yüksek skora sahip $K$ kelime dışındaki tüm adayları $-\infty$ yapar (örn. $K=50$). Anlamsız uzun kuyruk kelimeleri elenir.
- **Top-$p$ (Nucleus) Örneklemesi:** Aday havuzunu dinamik olarak belirler. Kelimeler olasılıklarına göre azalan sırada dizilir:
  $$\sum_{i=1}^M P_{(i)} \ge p \quad (\text{örn. } p = 0,90)$$
  Kümülatif toplamı $p$ eşiğine ulaşan en dar küme tutulur, gerisi atılır. Model emin olduğunda havuz 1-2 kelimeye daralır; belirsiz anlarda ise onlarca kelimeye genişler.

### 4. Durdurma kriterleri ve detokenization

Yeni token seçildikten sonra kontrol edilir:
- Model `<|end_of_text|>` veya `</s>` (EOS) üretti mi?
- `max_new_tokens` sınırına gelindi mi?
- Kullanıcının tanımladığı `stop_string` ile karşılaşıldı mı?

Döngü bittiğinde tamsayı ID'ler tokenizer aracılığıyla UTF-8 karakterlerine çevrilerek (**Detokenization**) kullanıcıya sunulur.

<svg viewBox="0 0 560 300" role="img" aria-label="model.generate() özbağlanımlı üretim döngüsünün anatomisi. Prefill, KV cache, logit işleyiciler, örnekleme ve detokenization adımları." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="528" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="42" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">1. PREFILL: Prompt Token'ları → KV Cache Dolduruldu → İlk Logit Vektörü</text>
<line x1="280" y1="58" x2="280" y2="76" marker-end="url(#out-arr-tr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="80" y="78" width="400" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280" y="98" text-anchor="middle" style="fill:var(--c-accent);font-size:12.5px;font-weight:700">2. ÖZBAĞLANIMLI DECODE DÖNGÜSÜ</text>
<text x="280" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Son h_last [1, d] çekilir → lm_head(h_last) → Ham Logitler [1, V]</text>
<line x1="280" y1="122" x2="280" y2="140" marker-end="url(#out-arr-tr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="142" width="480" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">3. LOGITS PROCESSORS (FİLTRE HATTI)</text>
<text x="280" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Tekrar Cezası (z / θ) ──> Sıcaklık (z / T) ──> Yasaklı Tokenlar (-inf)</text>
<line x1="280" y1="186" x2="280" y2="204" marker-end="url(#out-arr-tr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="60" y="206" width="440" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-success);stroke-width:1.2"/>
<text x="280" y="226" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:700">4. ÖRNEKLEME STRATEJİSİ</text>
<text x="280" y="242" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Greedy Argmax  |  Top-K (K=50)  |  Nucleus Top-p (p=0.90) → Token ID Seçilir</text>
<line x1="280" y1="250" x2="280" y2="268" marker-end="url(#out-green-tr)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="120" y="270" width="320" height="24" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="280" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:11px">5. Durdurma Kriteri (EOS) → Metne Çevir (Detokenization)</text>
</svg>

---

## 13. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri

Çıktı katmanı tasarlanırken eğitim ve çıkarım rejimleri arasındaki taban tabana zıt mühendislik gerçekleri sürekli gözetilmelidir:

| Mühendislik Ekseni | Eğitim Rejimi (Ön-eğitim / İnce Ayar) | Çıkarım Rejimi (Üretimde Dağıtım) |
| :--- | :--- | :--- |
| **Hesaplama Prömitifi** | Yoğun GEMM: Tüm tokenlar $H \in \mathbb{R}^{B \times S \times d}$ paralel olarak $W_{\text{head}}^T$ ile çarpılır. | Bellek kısıtlı GEMV: Tek bir vektör $h_{-1} \in \mathbb{R}^{1 \times d}$ matrisle çarpılır. |
| **Donanım Darboğazı** | **Hesaplama ve Kapasite:** Tensör çekirdekleri tam güçte; logitler için devasa VRAM tahsisi gerekir. | **Bellek Bant Genişliği:** $< 1$ FLOP/bayt aritmetik yoğunluk; GPU bellekten ağırlık okurken bekler. |
| **Logit Bellek Yükü** | Gigabaytlarca ($B \cdot S \cdot V \cdot 2$ bayt). Megatron veya Liger olmadan anında OOM verir. | Kilobaytlarca ($1 \cdot 1 \cdot V \cdot 2$ bayt). Bellek açısından tamamen önemsizdir. |
| **Kayıp ve Gradyan** | Aktif: $-\ln(P_c)$ hesaplanır ve $dL/dz = P - y$ gradyanı 32 katmana geriye pompalanır. | Pasif: Kayıp hesaplanmaz; gradyan ve hesaplama çizgesi tamamen devre dışıdır (`inference_mode`). |
| **Optimizer Belleği** | **Devasa:** AdamW 1. ve 2. momentleri fp32 saklar ($16$ bayt/param). Llama 3 `lm_head` için: **$8,4\text{ GB}$ saf optimizer VRAM'i**. | **Sıfır:** Üretim checkpoint'lerinde optimizer durumları tamamen atılır. |
| **Paralellik Stratejisi** | Megatron-LM `VocabParallelCrossEntropy` ile NVLink üzerinden tensör paralelizmi. | Medusa ve EAGLE ile spekülatif kod çözme; toplu çıkarımda sürekli batching. |
| **Çıktı İşleme** | Doğrudan Softmax + Çapraz Entropi kaybı indirgemesi. | Dinamik hat: Sıcaklık, tekrar cezası, Top-$K$, Nucleus Top-$p$, detokenization. |

---

## 14. Python ve PyTorch ile adım adım tam doğrulama

Aşağıdaki çalıştırılabilir bağımsız Python kodu, el hesabımızdaki ileri projeksiyonu, Gemma 2 soft-capping'i, yer değiştirme değişmezliğini, kaybı ve geriye yayılan gradyanları virgülden sonra 4 basamak kesinlikle teyit eder:

```python
import math

def cikti_katmani_dogrulama():
    # 1. Oyuncak tensörler ve hiperparametreler
    h = [0.5, -1.2, 0.8, 1.5]  # 32. katmandan gelen gizli durum (d=4)
    target = 3                 # Hedef token indeksi: "tavşanı"
    c = 5.0                    # Soft-capping eşiği

    # Ağırlık matrisi W_head [V=5, d=4]
    W = [
        [ 0.4, -0.2,  0.5, -0.1],  # 0: [PAD]
        [-0.3,  0.6, -0.1,  0.8],  # 1: "bir"
        [ 0.7, -0.5,  0.2,  0.4],  # 2: "kediyi"
        [-0.2,  0.8,  0.6,  0.5],  # 3: "tavşanı" (HEDEF)
        [ 0.5,  0.1, -0.4, -0.7]   # 4: "köpek"
    ]

    # Adım 1: Ham logitler z_raw = h @ W^T
    z_raw = [sum(h[k] * W[i][k] for k in range(4)) for i in range(5)]
    print(f"1. Ham Logitler (z_raw): {[round(x, 4) for x in z_raw]}")

    # Adım 2: Logit Soft-Capping: c * tanh(z_raw / c)
    z_capped = [c * math.tanh(z / c) for z in z_raw]
    print(f"2. Soft-Capped Logitler (z_cap): {[round(x, 4) for x in z_capped]}")

    # Adım 3: Yer Değiştirme Değişmezliği: z' = z - max(z)
    z_max = max(z_capped)
    z_shifted = [z - z_max for z in z_capped]
    exp_z = [math.exp(z) for z in z_shifted]
    sum_exp = sum(exp_z)
    print(f"3. Maksimum Logit: {round(z_max, 4)} | Üsteller Toplamı: {round(sum_exp, 4)}")

    # Adım 4: Softmax Olasılıkları
    probs = [e / sum_exp for e in exp_z]
    print(f"4. Softmax Olasılıkları (P): {[round(x, 4) for x in probs]}")

    # Adım 5: Çapraz Entropi Kaybı: L = -ln(P_hedef)
    loss = -math.log(probs[target])
    print(f"5. Çapraz Entropi Kaybı (L): {round(loss, 4)} nat")

    # Adım 6: Gradyan dL/dz_cap = P - y
    dL_dz_cap = [probs[i] - (1.0 if i == target else 0.0) for i in range(5)]
    print(f"6. dL/dz_cap Gradyanı (P - y): {[round(x, 4) for x in dL_dz_cap]}")

    # Soft-capping türevinden geçiş: d_cap/d_raw = 1 - tanh^2(z_raw / c)
    d_scale = [1.0 - math.tanh(z / c)**2 for z in z_raw]
    dL_dz_raw = [dL_dz_cap[i] * d_scale[i] for i in range(5)]
    print(f"   dL/dz_raw Gradyanı: {[round(x, 4) for x in dL_dz_raw]}")

    # Adım 7: Ağırlık Gradyanları dL/dW = dL/dz_raw (x) h
    dL_dW = [[round(dL_dz_raw[i] * h[k], 4) for k in range(4)] for i in range(5)]
    print("7. Head Ağırlık Gradyanları (dL/dW):")
    for row in dL_dW:
        print(f"   {row}")

    # Adım 8: Omurgaya Geri Dönen Gradyan dL/dh = dL/dz_raw @ W
    dL_dh = [sum(dL_dz_raw[i] * W[i][k] for i in range(5)) for k in range(4)]
    print(f"8. Omurga Gizli Durumuna Dönen Gradyan (dL/dh): {[round(x, 4) for x in dL_dh]}")

if __name__ == "__main__":
    cikti_katmani_dogrulama()
```

### PyTorch eşdeğer doğrulama kodu

```python
import torch
import torch.nn as nn

h = torch.tensor([0.5, -1.2, 0.8, 1.5], dtype=torch.float32, requires_grad=True)
W = torch.tensor([
    [ 0.4, -0.2,  0.5, -0.1],
    [-0.3,  0.6, -0.1,  0.8],
    [ 0.7, -0.5,  0.2,  0.4],
    [-0.2,  0.8,  0.6,  0.5],
    [ 0.5,  0.1, -0.4, -0.7]
], dtype=torch.float32, requires_grad=True)

target = torch.tensor(3, dtype=torch.long)
c = 5.0

# İleri geçiş
z_raw = torch.matmul(h, W.T)
z_cap = c * torch.tanh(z_raw / c)
loss = nn.functional.cross_entropy(z_cap.unsqueeze(0), target.unsqueeze(0))

# Geri geçiş
loss.backward()

print("PyTorch Kayıp:", round(loss.item(), 4))
print("PyTorch dL/dh:", [round(x, 4) for x in h.grad.tolist()])
```

Kodu çalıştırdığınızda PyTorch'un çıktıları kuruşu kuruşuna örtüşür: Kayıp $2,1265$, 32. Transformer bloğuna geri pompalanan gradyan vektörü ise tamı tamına `[0.5555, -0.8956, -0.3609, -0.1899]` değerini verir.

---

## Bütün hikâye altı satırda

- **LM Head geometriyi ayrık dile tercüme eder:** Transformer'dan çıkan sürekli gizli vektörleri ($h \in \mathbb{R}^d$) doğrusal izdüşümle ($W_{\text{head}}^T$) sözlük çapında logit skorlarına ($z \in \mathbb{R}^V$) dönüştürür.
- **Serbest ağırlıklar temsil rankını maksimize eder:** Bağlı ağırlıklar dev sözlüklerde (Gemma 2) bellek kurtarırken, modern modeller (Llama 3.1, Qwen 2.5) giriş ve çıkış uzaylarını serbest bırakarak birbirine neredeyse tamamen dik ($\cos \theta \approx 0,0007$) ve zengin temsiller öğrenir.
- **Soft-capping logit patlamasını engeller:** Gemma 2'nin hiperbolik tanjant ölçeklemesi ($c \cdot \tanh(z/c)$), logitleri $(-c, +c)$ aralığına hapsederek aşırı özgüveni, entropi çöküşünü ve tekrar döngülerini önler.
- **Yer değiştirme değişmezliği IEEE 754 taşmasını sıfırlar:** Softmax öncesinde logitlerden $\max(z)$ çıkarılması, tüm üstelleri $(0, 1,0]$ aralığına çekerek float16/bfloat16 donanımları ölümcül NaN hatalarından korur.
- **Softmax ve Çapraz Entropi doğrusal bir türev üretir:** Zincir kuralında karmaşık paydalar birbirini sadeleştirir ve geriye $\frac{\partial \mathcal{L}}{\partial z} = P - y$ (Tahmin eksi Gerçek) şeklinde asla sönümlenmeyen zarif bir hata sinyali kalır.
- **Modern sistemler donanım duvarlarını yıkar:** Megatron-LM vocab paralelizmi ve Liger Kernel kaynaşmış Triton çekirdekleri eğitimdeki bellek duvarını aşarken, Medusa ve EAGLE spekülatif kod çözme mekanizmaları bellek bant genişliği kısıtlı çıkarımı $2\times$ ila $3,5\times$ hızlandırır.

---

## Terimler sözlüğü

- **Language Model Head (`lm_head`):** Bir dil modelinde son Transformer bloğundan çıkan gizli durum vektörlerini kelime dağarcığı uzayına yansıtan nihai doğrusal projeksiyon katmanı.
- **Logits:** Çıktı katmanından çıkan, henüz normalize edilmemiş ve kelimelerin göreli logaritmik şans skorlarını temsil eden reel sayılar.
- **Softmax:** Keyfi reel sayılardan oluşan bir vektörü, elemanları toplamı tam olarak 1,0 olan geçerli bir ayrık olasılık dağılımına dönüştüren normalize edilmiş üstel fonksiyon.
- **Yer Değiştirme Değişmezliği (Shift Invariance):** Bir vektörün tüm elemanlarına aynı sabit eklendiğinde veya çıkarıldığında Softmax çıktısının matematiksel olarak değişmemesi özelliği; $\max(z)$ çıkarılarak taşma (overflow) önlenir.
- **Çapraz Entropi Kaybı (Cross-Entropy Loss):** Modelin tahmin ettiği olasılık dağılımı ile tekil doğru hedef arasındaki bilgi kuramsal farkı ölçen, tekil kelime için $\mathcal{L} = -\ln(P_c)$ formülüne indirgenen kayıp fonksiyonu.
- **Tahmin Eksi Gerçek ($P - y$):** Çapraz Entropi ve Softmax birleşiminin ham logitlere göre analitik türevi; doğrusal, doygunluğa ulaşmayan ve daima kararlı bir gradyan otoyolu sunar.
- **Ağırlık Bağlama (Weight Tying):** Model parametrelerinden tasarruf etmek amacıyla giriş token gömme (embedding) matrisi ile çıktı `lm_head` matrisinin aynı fiziksel bellek bloğunu paylaşması ($W_{\text{out}} = W_{\text{in}}$).
- **Logit Soft-Capping:** Gemma 2 mimarisinde logitlerin kontrolden çıkmasını ve aşırı özgüveni önlemek için uygulanan, logitleri $c \cdot \tanh(z/c)$ formülüyle $(-c, +c)$ aralığına hapseden yumuşak sınırlandırma tekniği.
- **Kelime Dağarcığı Paralelizmi (`VocabParallelCrossEntropy`):** Megatron-LM tarafından geliştirilen, kelime matrisini GPU'lar arasında bölerek devasa logit tensörlerini ağda taşımadan dağıtık Çapraz Entropi hesaplayan Tensör Paralelizmi tekniği.
- **Fused Linear Cross-Entropy:** LinkedIn Liger Kernel tarafından geliştirilen, matris çarpımı ve kayıp hesabını doğrudan GPU SRAM üzerinde parçalı olarak kaynaştırıp devasa logit matrisinin HBM'e yazılmasını engelleyen Triton çekirdeği.
- **Spekülatif Kod Çözme (Speculative Decoding):** Bellek bant genişliği darboğazını aşmak için hızlı bir taslak mekanizmasıyla birden çok gelecekteki token'ı öngörüp büyük modelle tek seferde doğrulayan çıkarım mimarisi.
- **Medusa:** Taslak modele ihtiyaç duymadan ana modelin son gizli durumuna eklenen çoklu MLP başlıklarıyla paralel token tahmini yapan ve Tree Attention ile doğrulayan spekülatif kod çözme çatısı.
- **EAGLE:** Spekülasyonu gürültülü token uzayı yerine en üst gizli özellikler (feature-level) seviyesinde yaparak taslak kabul oranını $\%80$'in üzerine çıkaran spekülatif örnekleme çerçevesi.
- **Nucleus (Top-$p$) Örneklemesi:** Kümülatif olasılık toplamı $p$ eşiğine ulaşan en küçük token kümesini dinamik olarak seçip diğer adayları eleyerek metin üretiminde tutarlılık ve çeşitliliği dengeleyen örnekleme yöntemi.
- **Bağlam Uzunluğu (`context_len`):** Bir modelin tek bir ileri yayılımda veya oturumda dikkat edebildiği ve işleyebildiği maksimum token dizisi; RoPE frekansları, KV cache VRAM'i ve çıktı projeksiyon tamponlarıyla sınırlıdır.
- **Parçalı Çapraz Entropi (Chunked Cross-Entropy - CCE):** Uzun dizileri doğrudan hızlı GPU SRAM üzerinde küçük parçalara (örn. 512 token) bölerek işleyen ve çıktı katmanı belleğini toplam bağlam uzunluğundan ($S$) bağımsız kılan optimizasyon tekniği.
- **Infini-Attention:** Maskeli yerel dikkat ile sabit boyutlu sıkıştırıcı belleği harmanlayarak teorik olarak sonsuz bağlam uzunluklarını $O(1)$ bellek karmaşıklığında işleyen ikili dikkat mimarisi (Google, 2024).

---

## Daha derine inmek için

- **Vaswani, A., vd. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — Doğrusal çıktı projeksiyonunu ve paylaşılan embedding mimarisini kuran temel Transformer makalesi.
- **Press, O., & Wolf, L. (2017).** *Using the Output Embedding to Improve Language Models.* [arXiv:1608.05859](https://arxiv.org/abs/1608.05859) — Bağlı ve serbest embedding katmanlarının teorik temeli ve ampirik başarım analizi.
- **Shoeybi, M., vd. (2019).** *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) — Vocab paralelizmi, ColumnParallelLinear ve dağıtık Çapraz Entropi mimarisi.
- **Google DeepMind (2024).** *Gemma 2: Improving Open Language Models at a Practical Size.* [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) — Logit soft-capping formülasyonu ve 256k bağlı kelime dağarcığı tasarımı.
- **Meta AI (2024).** *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — 128k serbest kelime dağarcığı ve büyük ölçekli ön-eğitim dinamikleri.
- **Munkhdalai, T., Faruqui, M., & Gopal, S. (2024).** *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention.* [arXiv:2404.07143](https://arxiv.org/abs/2404.07143) — Sınırsız akış bağlamı için sıkıştırıcı bellek mimarisi.
- **Xiao, G., vd. (2023).** *Efficient Streaming Language Models with Attention Sinks.* [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — StreamingLLM ve dikkat havuzlarının (attention sinks) korunması.
- **Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., & Dao, T. (2024).** *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads.* [arXiv:2401.10774](https://arxiv.org/abs/2401.10774) — Çok başlıklı Tree Attention spekülatif kod çözme mimarisi.
- **Li, Y., Wei, F., Zhang, C., & Zhang, H. (2024).** *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.* [arXiv:2401.15077](https://arxiv.org/abs/2401.15077) (ICML 2024) — Özellik seviyesinde spekülatif örnekleme çerçevesi.
- **LinkedIn Engineering (2024).** *Liger Kernel: Efficient Triton Kernels for LLM Training.* [GitHub Deposu](https://github.com/linkedin/Liger-Kernel) — Bellek dostu Fused Linear Cross-Entropy çekirdeği.
- **Bir Prompt'un Yolculuğu Serisinin Önceki Bölümleri:**
  - [Tokenizasyon Nasıl Çalışır?](post.html?slug=tokenizasyon-nasil-calisir) — 1. Bölüm
  - [Embedding Katmanı Derinlemesine](post.html?slug=embedding-katmani-derinlemesine) — 2. Bölüm
  - [Embeddingler Derinlemesine](post.html?slug=embeddingler-derinlemesine) — 3. Bölüm
  - [Self-Attention Derinlemesine](post.html?slug=self-attention-derinlemesine) — 4. Bölüm
  - [Causal Attention Derinlemesine](post.html?slug=causal-attention-derinlemesine) — 5. Bölüm
  - [Multi-Head Attention Derinlemesine](post.html?slug=multi-head-attention-derinlemesine) — 6. Bölüm
  - [Transformer Bloğu Derinlemesine](post.html?slug=transformer-blogu-derinlemesine) — 7. Bölüm
