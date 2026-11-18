Bir dil modeline bir prompt verdiğinizde, serimizin önceki altı bölümünde adım adım izlediğimiz o muazzam mekanizma metni çoktan geometriye ve dinamik bağlama dönüştürmüştür:
1. [Tokenizasyon](post.html?slug=tokenizasyon-nasil-calisir) masasında metin ayrık tamsayılara (`[1054, 492, 281]`) parçalanır.
2. [Embedding Katmanı](post.html?slug=embedding-katmani-derinlemesine), bu tamsayıları sözlük tablosundan okuyarak sürekli vektörlere çevirir ve döner konumsal kodlama (RoPE) ile dizilim geometrisini kurar.
3. [Semantik Vektör Uzayı](post.html?slug=embeddingler-derinlemesine), kelimelerin geometrik yakınlık ve açılarla anlam kazandığı koordinat sistemini kurar.
4. [Self-Attention Mekanizması](post.html?slug=self-attention-derinlemesine), token'ların Query, Key ve Value izdüşümleriyle birbirine bakmasını ve cümlenin anlık bağlamıyla zenginleşmiş dinamik temsil vektörleri üretmesini sağlar.
5. [Causal Attention](post.html?slug=causal-attention-derinlemesine), zamanın okunu dayatarak gelecekteki token'ları alt üçgensel maskeyle karartır ve her token'ın yalnızca geçmişe bakmasını güvenceye alır.
6. [Multi-Head Attention](post.html?slug=multi-head-attention-derinlemesine), tek bir projektörün uzlaşma çıkmazını kırıp bağımsız alt uzaylarda birden çok dilbilgisel eksene (özne–fiil, yerel komşuluk) aynı anda odaklanmayı sağlar.

Serimizin 4., 5. ve 6. bölümlerinde incelediğimiz somut üç kelimelik cümleyi hatırlayalım: **`["köpek", "kediyi", "kovaladı"]`**. Bölüm 6'da her kelime için Multi-Head Attention hesaplamasını tamamladık, kafaları birleştirdik ve çıktı matrisi $W^O$ ile çarparak her token için zengin bir dikkat çıktısı ($O$) elde ettik.

Fakat tam bu noktada derin öğrenmenin en temel mimari sorusu karşımıza dikilir: **Bu dikkat çıktısını alıp doğrudan bir sonraki dikkat katmanına veya modelin çıkışına bağlayabilir miyiz?**

Cevap kesin bir **hayırdır**. Çünkü dikkat mekanizması tek başına iki ölümcül duvara çarpar:

1. **Gradyan Duvarı (Vanishing Gradient):** Derin bir modelde (örneğin 32 veya 80 katman) fonksiyonları $h_l = f_l(h_{l-1})$ şeklinde art arda doğrusal bağlarsanız, türevlerin zincir kuralı çarpımı nedeniyle 20 katman sonra gradyan orijinal gücünün milyonda birine erir; ağın alt katmanları hiçbir şey öğrenemez.
2. **Konveks Örtü Duvarı (Convex Hull):** Softmax katsayılarının toplamı 1'dir ve hepsi sıfırdan büyüktür ($\sum_j A_{ij} = 1, A_{ij} \ge 0$). Bu matematiksel kural gereği attention çıktısı, giren Value vektörlerinin **konveks kombinasyonundan** (ağırlıklı ortalamasından) ibarettir. Tıpkı paletteki mavi ve sarı boyayı karıştırarak ancak yeşilin tonlarını elde edebilmeniz gibi; palette olmayan kırmızı bir rengi (yeni bir mantıksal sentezi, XOR ilişkisini veya dünya bilgisini) attention tek başına asla üretemez.

İşte bu yüzden modern dil modelleri attention mekanizmasını tek başına bırakmaz. Onu, kendi kendine yeten, kararlı ve döngüsel olarak tekrarlanabilen eksiksiz bir hesaplama motorunun içine yerleştirir: **Transformer Bloğu (Transformer Block)**.

Bu blok, dikkat mekanizmasını üç vazgeçilmez bileşenle tamamlar:
- **Residual Bağlantılar (Skip Connection):** $y = x + f(x)$ yapısıyla türeve $\mathbf{I} + f'(x)$ "gradyan otoyolunu" kazandırarak 100 katman boyunca sinyalin kayıpsız akmasını sağlar.
- **Katman Normalizasyonu (LayerNorm / RMSNorm):** Atlama bağlantısıyla biriken sinyalin varyans patlamasını dizginler, her token'ın özelliklerini bir ses mikseri gibi dengeler.
- **Feed-Forward Network (FFN / MLP):** Boyutu $4d_{\text{model}}$'e genişleterek ve doğrusal-olmayan aktivasyonlarla (GELU, SwiGLU) attention'ın hapsolduğu konveks örtüyü paramparça eder; model parametrelerinin üçte ikisini barındırarak internetteki olgusal bilgileri depolayan devasa bir ilişkisel anahtar-değer belleği kurar.

Bu yazıda, serimizin 7. adımı olarak tek bir Transformer bloğunun tüm iç mekaniğini, önceki bölümlerdeki somut matrislerimiz üzerinden adım adım elle sayısal olarak hesaplayarak, geometrik sezgilerini açarak ve çalışan PyTorch koduyla doğrulayarak inceliyoruz.

**Bu yazıda**

- [1. Büyük resim ve veri yolu: Tek bir transformer bloğunun anatomisi](#1-büyük-resim-ve-veri-yolu-tek-bir-transformer-bloğunun-anatomisi)
- [2. Residual bağlantılar: Birim otoyol ve düzeltme felsefesi](#2-residual-bağlantılar-birim-otoyol-ve-düzeltme-felsefesi)
- [3. Kaybolan gradyan felaketi: 20 katmanda milyonda bire eriyen sinyal](#3-kaybolan-gradyan-felaketi-20-katmanda-milyonda-bire-eriyen-sinyal)
- [4. Residual bağlantılar bu sorunu nasıl çözer: I + f'(x) garantisi](#4-residual-bağlantılar-bu-sorunu-nasıl-çözer-i-fx-garantisi)
- [5. Katman Normalizasyonu (LayerNorm): Orkestranın ses ayarı](#5-katman-normalizasyonu-layernorm-orkestranın-ses-ayarı)
- [6. Attention sonrası 1. adım: Residual toplama ve LayerNorm sayısal yürüyüşü](#6-attention-sonrası-1-adım-residual-toplama-ve-layernorm-sayısal-yürüyüşü)
- [7. Attention'ın görünmeyen sınırı: Konveks örtü (Convex Hull)](#7-attentionın-görünmeyen-sınırı-konveks-örtü-convex-hull)
- [8. Parametrelerin üçte ikisi: İlişkisel anahtar-değer belleği olarak FFN](#8-parametrelerin-üçte-ikisi-ilişkisel-anahtar-değer-belleği-olarak-ffn)
- [9. Aktivasyon fonksiyonlarının evrimi: ReLU, GELU ve SwiGLU](#9-aktivasyon-fonksiyonlarının-evrimi-relu-gelu-ve-swiglu)
- [10. FFN sonrası 2. adım: FFN ve ikinci residual katmanının sayısal yürüyüşü](#10-ffn-sonrası-2-adım-ffn-ve-ikinci-residual-katmanının-sayısal-yürüyüşü)
- [11. Modern üretim modellerinde bloğun evrimi: Llama 3, Gemma 2 ve DeepSeek-V3](#11-modern-üretim-modellerinde-bloğun-evrimi-llama-3-gemma-2-ve-deepseek-v3)
- [12. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri](#12-iki-mühendislik-gözü-eğitim-ve-çıkarım-dinamikleri)
- [13. PyTorch ile adım adım tam doğrulama](#13-pytorch-ile-adım-adım-tam-doğrulama)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Büyük resim ve veri yolu: Tek bir transformer bloğunun anatomisi

### İki motorun iş bölümü: İletişim (Attention) ve hesaplama (FFN)

Bir Transformer bloğu rastgele bir araya getirilmiş katmanlar yığını değildir; son derece net bir iş bölümüne sahip iki ana hesaplama motorundan oluşur:

1. **Attention alt katmanı (İletişim):** Token'ların dizi ekseni ($N \times N$) boyunca birbiriyle konuşmasını sağlar. Bir token diğer tüm token'lara bakar ve *"Benim bu bağlamdaki anlamımı tamamlamak için kimden ne kadar bilgi almalıyım?"* sorusunu yanıtlar. Ancak bu işlem tamamen token'lar arası bir harmanlamadır.
2. **FFN alt katmanı (Hesaplama & Bellek):** İletişim bittikten sonra her token kendi kabuğuna çekilir. FFN, her token'ın $d_{\text{model}}$ boyutlu vektörüne **birbirinden tamamen bağımsız olarak** (token-wise) uygulanır. Dizi ekseninde hiçbir etkileşim olmaz. Burada sorulan soru şudur: *"Topladığım bu bağlamsal bilgiyi kendi içimde nasıl işlemeliyim, hangi olgusal bilgileri çağırmalı ve yeni bir kavrama nasıl dönüştürmeliyim?"*

```
Attention = Şehir meydanındaki halk toplantısı (herkes birbiriyle konuşur, bilgi paylaşır).
FFN       = Toplantı sonrası herkesin kendi ofisine çekilip tek başına düşünmesi ve karar vermesi.
```

### Blok içi akış şeması ve Pre-LN vs Post-LN farkı

Tek bir Transformer bloğundaki veri akışı şu 8 adımdan oluşur:

<svg viewBox="0 0 560 284" role="img" aria-label="Tek bir transformer bloğundan geçen veri yolu, Post-LN düzeni. 1. alt katman, iletişim: token girdisi X pozisyon bilgisi eklenmiş Z olur, multi-head self-attention AttnOut&#x27;u üretir, bir atlama yolu Z&#x27;yi onun etrafından taşır, ikisi toplanır ve LayerNorm ya da RMSNorm ara temsil H1&#x27;i verir. 2. alt katman, hesaplama: H1 ileri beslemeli ağdan, W2 sigma W1 H1 artı b1, artı b2&#x27;den geçer, bir atlama yolu H1&#x27;i etrafından taşır, toplam yeniden normalize edilir ve nihai blok çıktısı H2&#x27;yi verir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bf-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bf-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="26" width="528" height="92" rx="8" style="fill:none;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="28" y="20" text-anchor="start" style="fill:var(--c-accent);font-size:12.5px;font-weight:600">1. alt katman: iletişim (attention)</text>
<rect x="16" y="148" width="528" height="92" rx="8" style="fill:none;stroke:var(--c-success);stroke-width:1.2;stroke-dasharray:5 4"/>
<text x="90" y="142" text-anchor="start" style="fill:var(--c-success);font-size:12.5px;font-weight:600">2. alt katman: hesaplama (FFN)</text>
<rect x="26" y="66" width="38" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="45.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">X</text>
<line x1="64" y1="84" x2="78" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="80" y="66" width="44" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="102.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">Z</text>
<line x1="124" y1="84" x2="140" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="142" y="62" width="150" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="217.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">multi-head attention</text>
<line x1="292" y1="84" x2="297" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="312" cy="84" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/><text x="312" y="89.5" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700">+</text>
<path d="M102 66 C 102 44, 312 44, 312 69" marker-end="url(#bf-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="206" y="37" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">atlama yolu (Z)</text>
<line x1="325" y1="84" x2="338" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="66" width="100" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="390.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LayerNorm</text>
<line x1="440" y1="84" x2="456" y2="84" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="458" y="66" width="72" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="494.0" y="88.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">H₁</text>
<path d="M494 102 V130 H52 V186" marker-end="url(#bf-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="26" y="188" width="52" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="52.0" y="210.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">H₁</text>
<line x1="78" y1="206" x2="92" y2="206" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="94" y="184" width="198" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="193.0" y="202.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">FFN / MLP</text>
<text x="193.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">W₂·σ(W₁·H₁ + b₁) + b₂</text>
<line x1="292" y1="206" x2="297" y2="206" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="312" cy="206" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/><text x="312" y="211.5" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700">+</text>
<path d="M60 188 C 60 166, 312 166, 312 191" marker-end="url(#bf-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="196" y="159" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">atlama yolu (H₁)</text>
<line x1="325" y1="206" x2="338" y2="206" marker-end="url(#bf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="188" width="100" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="390.0" y="210.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LayerNorm</text>
<line x1="440" y1="206" x2="456" y2="206" marker-end="url(#bf-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="458" y="188" width="72" height="36" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="494.0" y="210.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">H₂</text>
<text x="16" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Çizildiği gibi Post-LN (2017): norm her toplamadan sonra gelir.</text>
<text x="16" y="278" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Pre-LN modeller (Llama, Mistral) normu her alt katmanın önüne, atlama yolunun dışına taşır.</text>
</svg>

2017'deki özgün Transformer (Vaswani vd.), normalizasyonu toplama işleminden sonra uygulayan **Post-LN** mimarisini kullanmıştı:

$$H_1 = \text{LayerNorm}(Z + \text{MultiHead}(Z))$$

$$H_2 = \text{LayerNorm}(H_1 + \text{FFN}(H_1))$$

Post-LN tasarımında normalizasyon doğrudan atlama yolunun üzerine oturur. Bu durum derinleştikçe ($L > 12$) geriye yayılan gradyanların normalizasyon türevi tarafından sürekli küçültülmesine yol açar; model çok hassas bir öğrenme oranı ısıtma (learning rate warm-up) takvimi olmadan ilk adımlarda patlar veya öğrenemez.

Bu yüzden modern tüm büyük dil modelleri (Llama 3, Mistral, Gemma 2), normalizasyonu alt katmanların girişine alan **Pre-LN / Pre-RMSNorm** düzenine geçmiştir:

$$H_1 = Z + \text{MultiHead}(\text{Norm}(Z))$$

$$H_2 = H_1 + \text{FFN}(\text{Norm}(H_1))$$

Pre-LN mimarisinde atlama yolu hiçbir normalizasyon bariyerine çarpmaz. Modelin tabanından tavanına kadar uzanan kesintisiz, pürüzsüz bir bilgi ve gradyan omurgası (residual stream) elde edilir.

---

## 2. Residual bağlantılar: Birim otoyol ve düzeltme felsefesi

### Düzeltme ile baştan inşa arasındaki fark

Residual (artık) bağlantı prensibi zarif bir sadeliğe sahiptir: Bir katmanın girdiyi tamamen yeniden üretmesini istemek yerine, katmana sadece girdinin üzerine eklenecek küçük bir düzeltme (delta) hesaplatılır:

$$y = x + \mathcal{F}(x)$$

Bu formülün getirdiği üç derin mühendislik avantajı vardır:

1. **Varsayılan davranış birim fonksiyondur (Identity):** Katmanın ağırlıkları ($\mathcal{F}(x)$) sıfıra yakın başlatıldığında veya katman o adımda anlamlı bir şey öğrenemediğinde, sistem çökmek yerine hiçbir şey yapmama seçeneğine sığınır: $\mathcal{F}(x) \approx 0 \implies y \approx x$. Girdi hasar görmeden sonraki katmana akar.
2. **Sıfırdan çizmek yerine rötuş yapmak:** Skip bağlantısı olmayan bir ağda 45. katman kötü bir gradyan güncellemesi alırsa önceki 44 katmanın tüm emeği silinir. Residual ağda ise katman tüm resmi baştan çizmez; mevcut resmin üzerine küçük bir fırça darbesi vurur.
3. **Erken özelliklerin korunması:** İlk katmanlarda öğrenilen morfolojik veya sözdizimsel ham özellikler, üst katmanların karmaşık soyutlamaları tarafından ezilmeden son katmana kadar taşınabilir.

### Atlama yolunun devre şeması ve birim fonksiyon güvencesi

Bu yapıyı bir elektrik devresi veya çift hatlı demiryolu gibi düşünebiliriz. Girdi $x$ iki yola aynı anda girer:

<svg viewBox="0 0 560 150" role="img" aria-label="İki paralel hat olarak residual blok. Girdi x hem işlem hattı F(x)&#x27;ten geçer hem de onu atlayan bir özdeşlik ekspres hattından ilerler; iki hat bir toplamada buluşur ve çıktı y = x artı F(x) olur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sk-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sk-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<rect x="16" y="70" width="56" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="44.0" y="94.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">x</text>
<path d="M72 82 C 100 82, 100 36, 130 36 H 352 C 380 36, 380 74, 380 76" marker-end="url(#sk-g)" style="fill:none;stroke:var(--c-success);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="240" y="28" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:600">atlama / özdeşlik hattı: ekspres yol</text>
<line x1="72" y1="98" x2="168" y2="98" marker-end="url(#sk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="76" width="150" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="245.0" y="94.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">F(x)</text>
<text x="245.0" y="110.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">işlem yolu</text>
<line x1="320" y1="98" x2="365" y2="98" marker-end="url(#sk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="380" cy="98" r="13" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.4"/><text x="380" y="103.5" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700">+</text>
<line x1="393" y1="98" x2="420" y2="98" marker-end="url(#sk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="76" width="122" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="483.0" y="102.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">y = x + F(x)</text>
<text x="16" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">F(x) ne kadar gürültülü olursa olsun, x&#x27;in çıkışa dokunulmamış bir yolu her zaman vardır.</text>
</svg>

$\mathcal{F}(x)$ katmanı ne kadar karmaşık, doğrusal-olmayan veya gürültülü olursa olsun, girdinin değişmeden çıkışa ulaşabileceği bir paralel hat her zaman mevcuttur.

---

## 3. Kaybolan gradyan felaketi: 20 katmanda milyonda bire eriyen sinyal

### Zincir kuralı çarpımı ve Jacobian sönümlemesi

Residual bağlantıların derin öğrenmedeki asıl mucizesini anlamak için, çözdükleri en büyük krize bakmak gerekir: **Vanishing Gradient (Kaybolan Gradyan)** problemi.

Residual bağlantısı olmayan $L$ katmanlı klasik bir ağda ileri geçiş fonksiyon bileşkesidir:

$$h_l = f_l(h_{l-1}) \quad (l = 1, \dots, L)$$

Geri yayılımda (backpropagation) loss fonksiyonunun ($\mathcal{L}$) ilk katman girdisine ($h_0$) göre türevi zincir kuralı ile hesaplanır:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \cdot \frac{\partial h_L}{\partial h_{L-1}} \cdot \frac{\partial h_{L-1}}{\partial h_{L-2}} \cdots \frac{\partial h_1}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L \frac{\partial h_l}{\partial h_{l-1}}$$

Her bir $\frac{\partial h_l}{\partial h_{l-1}}$ terimi ilgili katmanın yerel Jacobian matrisidir. Eğer bu türevlerin ortalama büyüklüğü 1'den küçükse ($r < 1$ — örneğin doyan aktivasyonlar veya $1$'den küçük matris normları nedeniyle), gradyan katman derinliği arttıkça üstel olarak erir:

$$\left\| \frac{\partial \mathcal{L}}{\partial h_0} \right\| \propto r^L$$

### Sayısal erime tablosu: 20 katmanda gradyan ne kadar küçülür?

Katman başına ortalama $r$ zayıflama katsayısı için gradyanın katman derinliğine göre erimesini inceleyelim:

| Katman Derinliği ($L$) | $r = 0,9$ (Hafif Kayıp) | $r = 0,7$ (Orta Kayıp) | $r = 0,5$ (Şiddetli Kayıp) |
|:---|:---|:---|:---|
| **0 (Loss katmanı)** | 1,000 | 1,000 | 1,000 |
| **5 katman** | 0,590 | 0,168 | 0,031 |
| **10 katman** | 0,349 | 0,028 | 0,00098 |
| **15 katman** | 0,206 | 0,0047 | 0,0000305 |
| **20 katman** | 0,122 | 0,0008 | **0,00000095 ($9,5 \times 10^{-7}$)** |

$r = 0,5$ sütununa dikkat edin: Yalnızca 20 katman sonra gradyan orijinal gücünün **milyonda birine** inmektedir. 80 katmanlı modern bir modelde $(0,5)^{80} \approx 8,27 \times 10^{-25}$ olur. IEEE 754 float16 formatında bu sayı doğrudan sıfıra (underflow) yuvarlanır. En alttaki katmanların ağırlıklarına hiçbir güncelleme sinyali ulaşamaz; ağ donar ve öğrenme tamamen durur.

---

## 4. Residual bağlantılar bu sorunu nasıl çözer: I + f'(x) garantisi

### Türevde birim matris kalkanı: Katman ölse bile otoyol açık kalır

Residual bağlantı eklendiğinde katman çıktısı $h_l = h_{l-1} + f_l(h_{l-1})$ olur. Şimdi bu ifadenin bir önceki katmana göre türevini alalım:

$$\frac{\partial h_l}{\partial h_{l-1}} = \frac{\partial}{\partial h_{l-1}} \left[ h_{l-1} + f_l(h_{l-1}) \right] = \mathbf{I} + f'_l(h_{l-1})$$

Buradaki $\mathbf{I}$, atlama yolunun türevi olan **Birim Matristir (Identity Matrix)**. Zincir kuralını tekrar yazdığımızda:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \prod_{l=1}^L \left( \mathbf{I} + f'_l(h_{l-1}) \right)$$

Bu ifadenin parantezlerini açtığınızda karşınıza şu toplam çıkar:

$$\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L} \left( \mathbf{I} + \sum_{l=1}^L f'_l + \sum \text{çapraz terimler} \right)$$

> [!IMPORTANT]
> **Birim Otoyol Kalkanı:** Katmanların türevi olan $f'_l$ fonksiyonları sıfıra çökse bile (katman "ölse" veya doysa bile), formüldeki $\mathbf{I}$ terimi asla kaybolmaz. Gradyan, katmanların karmaşık iç işlemlerine hiç takılmadan, doğrudan birim matris üzerinden en baştaki katmana kadar kayıpsız bir biçimde akar.

### Sayısal karşılaştırma: Residual olan ve olmayan 20 katman

$f'(x) \approx 0,5$ varsayımıyla (sinyalin yarısını emen zayıflatıcı bir dönüşüm) iki durumu 20 katman boyunca karşılaştıralım:

| Katman Derinliği ($L$) | Residual Yok: $(0,5)^L$ | Residual Var: $(1 + 0,5)^L$ |
|:---|:---|:---|
| **0** | 1,000 | 1,000 |
| **5** | 0,031 | 7,59 |
| **10** | 0,00098 | 57,66 |
| **20** | **0,00000095** | **3325,26** |

Aynı sayılar log eksende, 0'dan 20'ye her katman için:

<svg viewBox="0 0 560 316" role="img" aria-label="0&#x27;dan 20&#x27;ye kadar katmandan sonra ne kadar gradyanın hayatta kaldığını logaritmik eksende gösteren çizgi grafik. Residual bağlantı olmadan, katman başına 0,9&#x27;luk çarpan 20 katmanda 0,122 bırakır, 0,7 0,0008 bırakır, 0,5 ise 9,5 çarpı 10 üzeri eksi 7, yani milyonda birin altını bırakır. Residual bağlantıyla çarpan 1 + 0,5 olur ve sinyal bunun yerine yaklaşık 3.325&#x27;e büyür: sönme yok, ama normalizasyonun dizginlemesi gereken bir büyüme var." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="66" y1="240.0" x2="420" y2="240.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="244.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁷</text>
<line x1="66" y1="220.9" x2="420" y2="220.9" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="224.9090909090909" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁶</text>
<line x1="66" y1="201.8" x2="420" y2="201.8" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="205.8181818181818" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁵</text>
<line x1="66" y1="182.7" x2="420" y2="182.7" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="186.72727272727272" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻⁴</text>
<line x1="66" y1="163.6" x2="420" y2="163.6" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="167.63636363636363" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻³</text>
<line x1="66" y1="144.5" x2="420" y2="144.5" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="148.54545454545456" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻²</text>
<line x1="66" y1="125.5" x2="420" y2="125.5" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="129.45454545454544" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁻¹</text>
<line x1="66" y1="106.4" x2="420" y2="106.4" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="60" y="110.36363636363637" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">1</text>
<line x1="66" y1="87.3" x2="420" y2="87.3" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="91.27272727272728" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10¹</text>
<line x1="66" y1="68.2" x2="420" y2="68.2" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="72.18181818181819" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10²</text>
<line x1="66" y1="49.1" x2="420" y2="49.1" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="53.09090909090909" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10³</text>
<line x1="66" y1="30.0" x2="420" y2="30.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="60" y="34.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">10⁴</text>
<text x="66.0" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">0</text>
<text x="154.5" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">5</text>
<text x="243.0" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">10</text>
<text x="331.5" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">15</text>
<text x="420.0" y="256" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">20</text>
<text x="243.0" y="272" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">gradyanın geriye geçtiği katman sayısı</text>
<text x="16" y="18" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">hayatta kalan gradyan (log ölçek)</text>
<path d="M66.0 106.4 L83.7 103.0 L101.4 99.6 L119.1 96.3 L136.8 92.9 L154.5 89.6 L172.2 86.2 L189.9 82.8 L207.6 79.5 L225.3 76.1 L243.0 72.7 L260.7 69.4 L278.4 66.0 L296.1 62.7 L313.8 59.3 L331.5 55.9 L349.2 52.6 L366.9 49.2 L384.6 45.9 L402.3 42.5 L420.0 39.1" style="fill:none;stroke:var(--c-success);stroke-width:2"/>
<circle cx="154.5" cy="89.6" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="72.7" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="55.9" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="39.1" r="4" style="fill:var(--c-success);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="43.128791996921706" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">1 + 0,5 (residual)</text>
<text x="428.0" y="57.128791996921706" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">3.325</text>
<path d="M66.0 106.4 L83.7 107.2 L101.4 108.1 L119.1 109.0 L136.8 109.9 L154.5 110.7 L172.2 111.6 L189.9 112.5 L207.6 113.4 L225.3 114.2 L243.0 115.1 L260.7 116.0 L278.4 116.8 L296.1 117.7 L313.8 118.6 L331.5 119.5 L349.2 120.3 L366.9 121.2 L384.6 122.1 L402.3 123.0 L420.0 123.8" style="fill:none;stroke:var(--c-accent);stroke-width:2"/>
<circle cx="154.5" cy="110.7" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="115.1" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="119.5" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="123.8" r="4" style="fill:var(--c-accent);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="127.83467821407595" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">r = 0,9</text>
<text x="428.0" y="141.83467821407595" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0,122</text>
<path d="M66.0 106.4 L83.7 109.3 L101.4 112.3 L119.1 115.2 L136.8 118.2 L154.5 121.1 L172.2 124.1 L189.9 127.1 L207.6 130.0 L225.3 133.0 L243.0 135.9 L260.7 138.9 L278.4 141.9 L296.1 144.8 L313.8 147.8 L331.5 150.7 L349.2 153.7 L366.9 156.6 L384.6 159.6 L402.3 162.6 L420.0 165.5" style="fill:none;stroke:var(--c-warn);stroke-width:2"/>
<circle cx="154.5" cy="121.1" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="135.9" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="150.7" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="165.5" r="4" style="fill:var(--c-warn);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="169.5080210854656" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">r = 0,7</text>
<text x="428.0" y="183.5080210854656" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0,0008</text>
<path d="M66.0 106.4 L83.7 112.1 L101.4 117.9 L119.1 123.6 L136.8 129.4 L154.5 135.1 L172.2 140.8 L189.9 146.6 L207.6 152.3 L225.3 158.1 L243.0 163.8 L260.7 169.6 L278.4 175.3 L296.1 181.1 L313.8 186.8 L331.5 192.6 L349.2 198.3 L366.9 204.1 L384.6 209.8 L402.3 215.6 L420.0 221.3" style="fill:none;stroke:var(--c-danger);stroke-width:2"/>
<circle cx="154.5" cy="135.1" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="243.0" cy="163.8" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="331.5" cy="192.6" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<circle cx="420.0" cy="221.3" r="4" style="fill:var(--c-danger);stroke:var(--c-bg);stroke-width:2"/>
<text x="428.0" y="225.30236198079282" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">r = 0,5</text>
<text x="428.0" y="239.30236198079282" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">9,5×10⁻⁷</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Atlama yolu yoksa sinyal geometrik olarak ölür.</text>
<text x="16" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Varsa sinyal yalnızca büyüyebilir; o büyümeyi de LayerNorm dizginler.</text>
</svg>

Residual bağlantı gradyanın sıfıra çökmesini kesin olarak engeller. Ancak tablodaki $(1 + 0,5)^{20} \approx 3325$ değeri başka bir tehlikeye işaret eder: **Patlayan gradyan (Exploding Gradient)** ve aktivasyon varyansı artışı.

İşte bu kontrolsüz büyümeyi dizginleyen ve her adımda sinyali yeniden evcilleştiren bileşen, residual'ın ayrılmaz ortağıdır: **Layer Normalization**.

---

## 5. Katman Normalizasyonu (LayerNorm): Orkestranın ses ayarı

### Formüller ve mikser analojisi: Vektörün iç dengesini kurmak

Residual bağlantılar sinyali sürekli birbirinin üzerine toplar ($x + \text{Attn}(x) + \text{FFN}(x)$). Önlem alınmazsa katmanlar ilerledikçe vektörlerin sayısal büyüklüğü ve varyansı çığ gibi büyür.

**Layer Normalization (Ba, Kiros ve Hinton 2016)**, her bir token'ın $d_{\text{model}}$ boyutlu özellik vektörünü kendi içinde bağımsız olarak normalize eder:

```
Vektör Girdisi: x = [x_1, x_2, ..., x_d]
```

1. **Ortalama ($\mu$):**
$$\mu = \frac{1}{d} \sum_{i=1}^d x_i$$

2. **Varyans ($\sigma^2$):**
$$\sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

3. **Standartlaştırma ($\hat{x}$):**
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

4. **Öğrenilebilir Ölçekleme ve Kaydırma ($y$):**
$$y_i = \gamma_i \hat{x}_i + \beta_i$$

Buradaki $\epsilon$ ($10^{-5}$ veya $10^{-6}$), sıfıra bölünmeyi engelleyen kararlılık sabitidir. $\gamma$ (gain) ve $\beta$ (bias) ise modelin gerekirse orijinal dağılımı geri getirebilmesini sağlayan öğrenilebilir parametrelerdir.

```
Benzetme: Bir ses mikseri düşünün. Orkestradaki bir enstrüman (bir özellik boyutu)
aniden 50 birim ses patlaması yaparsa hoparlörler yanar. LayerNorm, her müzisyenin
ses seviyesini ölçer, ortalamasını alır ve genel ses düzeyini standart bir desibel
aralığına (ortalama 0, varyans 1) çeker.
```

### Neden BatchNorm değil? Dil modellerinde değişken uzunluk ve batch=1 açmazı

Bilgisayarlı görüde devrim yaratan **Batch Normalization (BatchNorm)**, neden doğal dil işlemede (NLP) ve büyük dil modellerinde terk edilmiştir?

```
BatchNorm (Görü):  İstatistiği BATCH ekseninde toplar. (Farklı resimlerin aynı kanalı).
LayerNorm (NLP):   İstatistiği ÖZELLİK ekseninde toplar. (Aynı token'ın d_model boyutu).
```

| Kriter | Batch Normalization (BatchNorm) | Layer Normalization (LayerNorm) |
|:---|:---|:---|
| **İstatistik Hesabı Ekseni** | Mini-batch örnekleri boyunca | Tek bir token'ın gizli özellikleri boyunca |
| **Değişken Dizi Uzunluğu** | Farklı uzunluktaki cümlelerde dolgu (padding) istatistiği bozar | Dizi uzunluğundan tamamen bağımsızdır |
| **Çıkarım (Inference) Davranışı** | Eğitimde biriktirilen hareketli ortalamalara bağımlıdır | Her token anında kendi içinde hesaplanır |
| **Batch Size = 1 Durumu** | Tek bir prompt geldiğinde varyans sıfır olur, çöker | Batch boyutu 1 olsa bile kusursuz çalışır |

Modern üretim modelleri (Llama 3, Gemma 2) LayerNorm'u daha da basitleştirerek ortalama çıkarma adımını kaldıran **RMSNorm** (Zhang ve Sennrich 2019) kullanır:

$$\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}, \quad \text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

RMSNorm, ortalamayı sıfıra çekmeyip sadece kök ortalama kareye bölerek GPU bellek bant genişliğinde %10-%50 tasarruf sağlar.

---

## 6. Attention sonrası 1. adım: Residual toplama ve LayerNorm sayısal yürüyüşü

### Atlama toplamı: Z ve Attention çıktısının birleşimi

Serimizin 4., 5. ve 6. bölümlerinde kullandığımız somut tensörlerimizi hatırlayalım. Model boyutu $d_{\text{model}} = 4$ ve 1. token'ımız olan `"köpek"` için:

Girdi vektörü:
$$z_1 = [0,210, \; 0,820, \; 0,130, \; 0,440]$$

Multi-Head Attention sonrası elde ettiğimiz dikkat çıktısı:
$$o_1 = [1,005, \; 1,041, \; 1,012, \; 0,976]$$

**1. Adım: Atlama (Skip) Toplamı ($r_1 = z_1 + o_1$):**

$$r_1[0] = 0,210 + 1,005 = 1,215$$

$$r_1[1] = 0,820 + 1,041 = 1,861$$

$$r_1[2] = 0,130 + 1,012 = 1,142$$

$$r_1[3] = 0,440 + 0,976 = 1,416$$

$$r_1 = [1,215, \; 1,861, \; 1,142, \; 1,416]$$

### Sayısal LayerNorm hesabı: Ortalamadan arındırma ve varyans ölçekleme

Şimdi bu $r_1$ vektörünü adım adım LayerNorm'dan geçirelim ($\gamma = [1, 1, 1, 1]$, $\beta = [0, 0, 0, 0]$, $\epsilon = 10^{-5}$):

**1. Ortalama ($\mu$):**

$$\mu = \frac{1,215 + 1,861 + 1,142 + 1,416}{4} = \frac{5,634}{4} = 1,4085$$

**2. Farkların karesi ve Varyans ($\sigma^2$):**

$$(1,215 - 1,4085)^2 = (-0,1935)^2 \approx 0,03744$$

$$(1,861 - 1,4085)^2 = (0,4525)^2 \approx 0,20476$$

$$(1,142 - 1,4085)^2 = (-0,2665)^2 \approx 0,07102$$

$$(1,416 - 1,4085)^2 = (0,0075)^2 \approx 0,00006$$

$$\sigma^2 = \frac{0,03744 + 0,20476 + 0,07102 + 0,00006}{4} = \frac{0,31328}{4} = 0,07832$$

Payda standart sapması:

$$\sqrt{\sigma^2 + \epsilon} = \sqrt{0,07832 + 0,00001} = \sqrt{0,07833} \approx 0,27987$$

**3. Normalleştirilmiş Ara Temsil ($h_1$):**

$$h_1[0] = \frac{1,215 - 1,4085}{0,27987} = \frac{-0,1935}{0,27987} \approx -0,691$$

$$h_1[1] = \frac{1,861 - 1,4085}{0,27987} = \frac{0,4525}{0,27987} \approx 1,617$$

$$h_1[2] = \frac{1,142 - 1,4085}{0,27987} = \frac{-0,2665}{0,27987} \approx -0,952$$

$$h_1[3] = \frac{1,416 - 1,4085}{0,27987} = \frac{0,0075}{0,27987} \approx 0,027$$

$$h_1 = [-0,691, \; 1,617, \; -0,952, \; 0,027]$$

Bu $h_1$ vektörünün ortalaması tam olarak $0$, varyansı ise tam olarak $1$'dir. Dikkat çıktısı artık vahşi büyüklüklerden arınmış, terbiye edilmiş ve FFN katmanına girmeye hazır hâle gelmiştir.

---

## 7. Attention'ın görünmeyen sınırı: Konveks örtü (Convex Hull)

### Softmax'ın geometrik tuzağı: Ağırlıklı ortalama neden yetersiz kalır?

Çoğu derin öğrenme anlatısında gözden kaçan en kritik matematiksel gerçek şudur: **Self-Attention mekanizması tek başına yeni kavramlar veya mantıksal sentezler üretemez.**

Bunun sebebi Softmax fonksiyonunun doğasında yatar:

$$A_{ij} \ge 0 \quad \text{ve} \quad \sum_{j=1}^N A_{ij} = 1$$

Bu iki şart, matematiğin en temel geometrik tanımlarından biridir: **Konveks Kombinasyon (Convex Combination)**.

Attention çıktısı olan her bir satır, girdi Value vektörlerinin ($V = [v_1, v_2, \dots, v_N]^T$) bir konveks kombinasyonudur:

$$\text{head}_i = \sum_{j=1}^N A_{ij} v_j$$

Geometrik olarak bu toplam, $v_1, v_2, \dots, v_N$ noktalarının çevrelediği çokgenin (konveks örtünün / convex hull) **içinde** kalmak zorundadır. Asla bu sınırların dışına çıkamaz.

Bu duvar gerçek sayılarla: 6. bölümdeki 1. kafanın 2 boyutlu değer vektörleri ve çıktıları:

<svg viewBox="0 0 560 300" role="img" aria-label="Attention&#x27;ın konveks örtüsü, 6. bölümdeki 1. kafanın gerçek 2 boyutlu sayılarıyla çizilmiş. v köpek 0,34, 0,95&#x27;te, v kediyi 1,32, 0,53&#x27;te, v kovaladı 1,20, 1,41&#x27;de bir üçgen oluşturur. 1. kafanın çıktıları, kediyi için 0,870, 0,723 ve kovaladı için 1,064, 1,105, ikisi de üçgenin içine düşer; çünkü softmax ağırlıkları negatif değildir ve toplamları 1&#x27;dir. 1,45, 0,25 gibi bir nokta dışarıdadır ve hiçbir attention ağırlıklandırması onu üretemez; doğrusal olmayan FFN üretebilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="hl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<line x1="40" y1="270" x2="360" y2="270" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="40" y1="270" x2="40" y2="20" style="stroke:var(--c-border);stroke-width:1"/>
<text x="140.0" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">0,5</text>
<text x="32" y="189.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">0,5</text>
<text x="240.0" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1,0</text>
<text x="32" y="104.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">1,0</text>
<text x="340.0" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">1,5</text>
<text x="32" y="19.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">1,5</text>
<polygon points="108.0,108.5 304.0,179.9 280.0,30.3" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.5"/>
<circle cx="108.0" cy="108.5" r="5" style="fill:var(--c-accent)"/>
<text x="108.0" y="126.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">v_köpek</text>
<circle cx="304.0" cy="179.9" r="5" style="fill:var(--c-accent)"/>
<text x="304.0" y="197.9" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">v_kediyi</text>
<circle cx="280.0" cy="30.3" r="5" style="fill:var(--c-accent)"/>
<text x="280.0" y="20.3" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">v_kovaladı</text>
<circle cx="214.0" cy="147.1" r="4.5" style="fill:var(--c-accent-2);stroke:var(--c-bg);stroke-width:1.5"/>
<text x="222.0" y="151.1" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">o_kediyi</text>
<circle cx="252.8" cy="82.2" r="4.5" style="fill:var(--c-accent-2);stroke:var(--c-bg);stroke-width:1.5"/>
<text x="260.8" y="86.2" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">o_kovaladı</text>
<circle cx="330.0" cy="227.5" r="6" style="fill:none;stroke:var(--c-danger);stroke-width:1.8;stroke-dasharray:3 2"/>
<text x="320.0" y="231.5" text-anchor="end" style="fill:var(--c-danger);font-size:11px">ulaşılamaz</text>
<text x="384" y="40" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">ağırlıklı ortalama</text>
<text x="384" y="58" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ağırlıklar ≥ 0, toplam 1;</text>
<text x="384" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">her çıktı üçgenin</text>
<text x="384" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">içinde kalır</text>
<text x="384" y="140" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:600">mor: gerçek çıktılar</text>
<text x="384" y="158" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">1. kafa, 6. bölüm:</text>
<text x="384" y="173" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ikisi de v&#x27;lerin</text>
<text x="384" y="188" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">örtüsüne düşer</text>
<text x="384" y="226" text-anchor="start" style="fill:var(--c-success);font-size:13px;font-weight:600">FFN dışarı çıkar</text>
<text x="384" y="244" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">doğrusal olmama, hiçbir</text>
<text x="384" y="259" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">karışımın varamadığı</text>
<text x="384" y="274" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">noktalara ulaşır</text>
</svg>

### Boya karıştırma benzetmesi ve XOR mantıksal çıkmazı

Bunu resim sanatıyla somutlaştıralım:
- Masanızda iki tüp boya var: **Sarı** ($v_1$) ve **Mavi** ($v_2$).
- Attention mekanizması bu iki boyayı farklı oranlarda karıştırma işlemidir: $\%80$ mavi + $\%20$ sarı $\to$ koyu yeşil; $\%30$ mavi + $\%70$ sarı $\to$ fıstık yeşili.
- Attention ne kadar karmaşık olursa olsun, elinizdeki sarı ve mavi boyayı karıştırarak masada olmayan **Kırmızı** rengi asla elde edemezsiniz!

Aynı kısıt mantıksal işlemler için de geçerlidir. Örneğin **XOR (Özel Veya)** mantıksal kapısı doğrusal olarak ayrılamaz (linearly inseparable). İki girdinin ağırlıklı ortalamasını alarak XOR sonucunu üretemezsiniz.

Attention bir yönlendiricidir (router); bilgiyi kimden kime taşıyacağını çok iyi bilir. Ancak yeni bir renk sentezlemek, verilmeyen bir bilgiyi hatırlamak ve doğrusal-olmayan akıl yürütmek için bu konveks örtü sınırını parçalayacak ikinci bir motora ihtiyaç vardır: **FFN**.

---

## 8. Parametrelerin üçte ikisi: İlişkisel anahtar-değer belleği olarak FFN

### Parametre bütçesi: Neden ağırlıkların %66'sı FFN'dedir?

Modern bir büyük dil modelinin mimari konfigürasyonunu açtığınızda şaşırtıcı bir tabloyla karşılaşırsınız: Modeldeki parametrelerin yaklaşık **$\%66$'sı (üçte ikisi)** attention katmanında değil, FFN katmanındadır.

Orijinal Transformer'da ve modern mimarilerde FFN, girdiyi önce 4 kat genişletir ($d_{\text{model}} \to 4d_{\text{model}}$), ardından tekrar orijinal boyuta indirir ($4d_{\text{model}} \to d_{\text{model}}$):

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 x + b_1) + b_2$$

Matris boyutlarını ve parametre sayılarını hesaplayalım:
- $W_1 \in \mathbb{R}^{4d_{\text{model}} \times d_{\text{model}}} \implies 4 d_{\text{model}}^2$ parametre.
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times 4d_{\text{model}}} \implies 4 d_{\text{model}}^2$ parametre.
- FFN toplamı: $\mathbf{8 d_{\text{model}}^2}$ parametre!

Buna karşılık Bölüm 6'da hesapladığımız Multi-Head Attention'ın tüm projeksiyon matrisleri ($W_Q, W_K, W_V, W^O$) toplamda yalnızca:
- MHA toplamı: $\mathbf{4 d_{\text{model}}^2}$ parametre tutar.

FFN, attention mekanizmasının tam iki katı parametreye sahiptir.

### Geva vd. 2021 keşfi: W1 anahtarlar, W2 değerler ve olgusal hafıza

Peki bu devasa parametre deposu ne işe yarar? Geva ve çalışma arkadaşlarının 2021 tarihli çığır açıcı makalesi (*Transformer Feed-Forward Layers Are Key-Value Memories*) bu sorunun cevabını kanıtlamıştır:

FFN katmanları klasik bir **İlişkisel Bellek (Key-Value Associative Memory)** gibi çalışır:

1. **$W_1$ matrisi ANAHTARLARDIR (Keys):** Girdideki kavramsal örüntüleri algılar. Örneğin bir nöron, girdi vektöründe *"Fransa'nın başkenti"* veya *"Python fonksiyon tanımı"* örüntüsü gördüğünde tetiklenir ($u_i > 0$).
2. **Doğrusal olmayan aktivasyon $\sigma$ EŞİK SÜZGECİDİR:** Alakasız tüm kavramları sıfırlar (örneğin ReLU negatifleri yok eder), sadece eşleşen anahtarların kapısını açar.
3. **$W_2$ matrisi DEĞERLERDİR (Values):** Açılan kapıdan modele bilgi enjekte eder. Örneğin tetiklenen nöronun $W_2$'deki sütun vektörü residual akışa *"Paris"* bilgisini yazar.

<svg viewBox="0 0 560 210" role="img" aria-label="Çağrışımsal anahtar-değer belleği olarak FFN. Token&#x27;ın vektörü residual akıştan okunur; W1 çarpımı onu anahtarlarla eşleştirir; aktivasyon bir eşik filtresi gibi çalışır ve yalnızca eşleşen anahtarlar yanar, yürüyüşümüzde altı nörondan 1, 2 ve 4; W2 çarpımı karşılık gelen değerleri enjekte eder ve bilgi residual akışa geri eklenir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="kv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="kv-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="20" width="92" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="62.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">token</text>
<text x="62.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">girdi</text>
<line x1="108" y1="48" x2="125" y2="48" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="126" y="20" width="130" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="191.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">W₁ çarpımı</text>
<text x="191.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">anahtar taraması</text>
<line x1="256" y1="48" x2="273" y2="48" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="274" y="20" width="130" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="339.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">aktivasyon</text>
<text x="339.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">eşik filtresi</text>
<line x1="404" y1="48" x2="421" y2="48" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="20" width="122" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="483.0" y="44.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">W₂ çarpımı</text>
<text x="483.0" y="60.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">değer enjeksiyonu</text>
<circle cx="296" cy="94" r="6" style="fill:var(--c-surface-2);fill-opacity:1;stroke:var(--c-border);stroke-width:1"/>
<circle cx="313" cy="94" r="6" style="fill:var(--c-warn);fill-opacity:.8;stroke:var(--c-border);stroke-width:1"/>
<circle cx="330" cy="94" r="6" style="fill:var(--c-warn);fill-opacity:.8;stroke:var(--c-border);stroke-width:1"/>
<circle cx="347" cy="94" r="6" style="fill:var(--c-surface-2);fill-opacity:1;stroke:var(--c-border);stroke-width:1"/>
<circle cx="364" cy="94" r="6" style="fill:var(--c-warn);fill-opacity:.8;stroke:var(--c-border);stroke-width:1"/>
<circle cx="381" cy="94" r="6" style="fill:var(--c-surface-2);fill-opacity:1;stroke:var(--c-border);stroke-width:1"/>
<text x="339" y="116" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yürüyüşümüz: 6&#x27;dan 3&#x27;ü yanar</text>
<rect x="16" y="138" width="528" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1"/>
<text x="280" y="155" text-anchor="middle" style="fill:var(--c-text);font-size:12px">residual akış</text>
<line x1="62" y1="138" x2="62" y2="78" marker-end="url(#kv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="483" y1="78" x2="483" y2="136" marker-end="url(#kv-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="491" y="112" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">bilgi eklenir</text>
<text x="16" y="192" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Anahtarlar hangi anının uyanacağına, değerler geri ne yazılacağına karar verir.</text>
</svg>

Meng ve ark. (2022, ROME makalesi), bu ilişkiyi kullanarak doğrudan $W_1$ ve $W_2$ ağırlıklarını güncelleyip modelin hafızasındaki *"Eyfel Kulesi Paris'tedir"* bilgisini *"Eyfel Kulesi Roma'dadır"* şeklinde cerrahi bir operasyonla değiştirebilmiştir.

---

## 9. Aktivasyon fonksiyonlarının evrimi: ReLU, GELU ve SwiGLU

### ReLU'nun sert sınırları ve ölü nöron problemi

Konveks örtüyü kırmak için doğrusal olmayan bir aktivasyon fonksiyonu şarttır. Derin öğrenmenin klasik aktivasyonu **ReLU (Rectified Linear Unit)** son derece basittir:

$$\text{ReLU}(x) = \max(0, x)$$

Ancak ReLU'nun iki büyük kusuru vardır:
1. **Sert Kırılma (Non-differentiable at 0):** $x=0$'da türevi tanımsızdır; geçiş yumuşak değil keskindir.
2. **Ölü Nöron Problemi (Dying ReLU):** Bir nöron eğitim sırasında sürekli negatif bölgede kalırsa ($x < 0$), gradyanı mutlak sıfır olur ($\frac{\partial y}{\partial x} = 0$). Bu nöronun ağırlıkları bir daha asla güncellenemez; nöron sonsuza dek "ölür".

### GELU ve SwiGLU: Yumuşak olasılıksal kapılamadan çift doğrusal geçişe

**GELU (Gaussian Error Linear Unit - Hendrycks ve Gimpel 2016):** Modern modellerin ilk nesli (BERT, GPT-2, GPT-3), ReLU yerine GELU'ya geçmiştir. GELU, girdiyi standart normal dağılımın kümülatif dağılım fonksiyonuyla ($\Phi(x)$) çarparak olasılıksal bir yumuşak kapı kurar:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x) = x \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt$$

Pratikte şu hızlı analitik yaklaşımla hesaplanır:

$$\text{GELU}(x) \approx 0,5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} (x + 0,044715 x^3) \right) \right)$$

GELU'nun en belirgin özelliği, $x \in (-0,75, 0)$ aralığında hafif bir negatif çukura inmesidir (minimum nokta $\approx -0,17$). Bu yumuşak çukur, küçük negatif değerlerin gradyanını tamamen öldürmez; ağa pürüzsüz bir düzenlileştirme sağlar.

**SwiGLU (Swish Gated Linear Unit - Shazeer 2020):** Günümüzün en güçlü açık kaynaklı modelleri (Llama 3, Mistral, Qwen 2), FFN'i çift doğrusal kapılı (bilinear gating) **SwiGLU** mimarisiyle tasarlar:

$$\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \otimes x W_{\text{up}} \right) W_{\text{down}}$$

Burada $\text{Swish}(z) = z \cdot \text{sigmoid}(\beta z)$ fonksiyonudur. SwiGLU iki katman yerine üç matris ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) kullanır. Parametre sayısını denk tutmak için gizli boyut $4d$ yerine kabaca $\frac{8}{3}d_{\text{model}}$ olarak ayarlanır. Bu yapı, nöronların birbirini dinamik olarak çarpımsal biçimde kapılamasına olanak tanır.

---

## 10. FFN sonrası 2. adım: FFN ve ikinci residual katmanının sayısal yürüyüşü

### W1 izdüşümü, ReLU kapılaması ve W2 ile model boyutuna dönüş

Bölüm 6'da hesapladığımız normalize edilmiş ara vektörümüzü alalım:

$$h_1 = [-0,691, \; 1,617, \; -0,952, \; 0,027] \in \mathbb{R}^4$$

Pedagojik olarak hesapları şeffaf tutmak adına gizli boyutu $d_{\text{ff}} = 6$ olan bir FFN katmanı kuralım ($W_1 \in \mathbb{R}^{6 \times 4}$, $b_1 \in \mathbb{R}^6$):

$$W_1 = \begin{bmatrix} 0,2 & -0,1 & 0,3 & 0,4 \\ -0,5 & 0,2 & 0,1 & 0,0 \\ 0,1 & 0,3 & -0,2 & 0,2 \\ 0,4 & 0,1 & 0,0 & -0,3 \\ -0,2 & 0,5 & 0,4 & 0,1 \\ 0,3 & -0,4 & 0,2 & 0,6 \end{bmatrix}, \quad b_1 = \begin{bmatrix} 0,1 \\ 0,1 \\ 0,1 \\ 0,1 \\ 0,1 \\ 0,1 \end{bmatrix}$$

**1. Çarpım ve Bias Ekleme ($u = W_1 h_1 + b_1$):**
- $u_0 = 0,2(-0,691) - 0,1(1,617) + 0,3(-0,952) + 0,4(0,027) + 0,1 \approx -0,475$
- $u_1 = -0,5(-0,691) + 0,2(1,617) + 0,1(-0,952) + 0,0(0,027) + 0,1 \approx 0,674$
- $u_2 = 0,1(-0,691) + 0,3(1,617) - 0,2(-0,952) + 0,2(0,027) + 0,1 \approx 0,712$
- $u_3 = 0,4(-0,691) + 0,1(1,617) + 0,0(-0,952) - 0,3(0,027) + 0,1 \approx -0,023$
- $u_4 = -0,2(-0,691) + 0,5(1,617) + 0,4(-0,952) + 0,1(0,027) + 0,1 \approx 0,669$
- $u_5 = 0,3(-0,691) - 0,4(1,617) + 0,2(-0,952) + 0,6(0,027) + 0,1 \approx -0,928$

$$u = [-0,475, \; 0,674, \; 0,712, \; -0,023, \; 0,669, \; -0,928]$$

**2. Doğrusal Olmayan Aktivasyon ($a = \text{ReLU}(u)$):**

Negatif değerler sıfırlanır, eşleşen anahtarlar açılır:

$$a = [0,000, \; 0,674, \; 0,712, \; 0,000, \; 0,669, \; 0,000]$$

0., 3. ve 5. nöronlar ketlenmiş, 1, 2 ve 4. nöronlar ateşlenmiştir. 3. nöron kıl payı kaçırır: aktivasyon öncesi değeri yalnızca $-0,023$'tür ve ReLU onu tam sıfıra indirir.

**3. İkinci Katman Projeksiyonu ($f = W_2 a + b_2$):**

$W_2 \in \mathbb{R}^{4 \times 6}$ ve $b_2 = [0,05, \; 0,05, \; 0,05, \; 0,05]^T$ olsun:

$$W_2 = \begin{bmatrix} 0,2 & 0,1 & -0,3 & 0,4 & 0,2 & -0,1 \\ -0,2 & 0,3 & 0,2 & -0,1 & 0,5 & 0,4 \\ 0,1 & -0,4 & 0,3 & 0,2 & -0,2 & 0,1 \\ 0,5 & 0,2 & -0,1 & 0,3 & 0,1 & -0,3 \end{bmatrix}$$

$a$ vektörünü $W_2$ ile çarpıp $b_2$ eklediğimizde:
- $f[0] = 0,2(0) + 0,1(0,674) - 0,3(0,712) + 0,4(0) + 0,2(0,669) - 0,1(0) + 0,05 \approx 0,038$
- $f[1] = -0,2(0) + 0,3(0,674) + 0,2(0,712) - 0,1(0) + 0,5(0,669) + 0,4(0) + 0,05 \approx 0,729$
- $f[2] = 0,1(0) - 0,4(0,674) + 0,3(0,712) + 0,2(0) - 0,2(0,669) + 0,1(0) + 0,05 \approx -0,140$
- $f[3] = 0,5(0) + 0,2(0,674) - 0,1(0,712) + 0,3(0) + 0,1(0,669) - 0,3(0) + 0,05 \approx 0,180$

$$f = [0,038, \; 0,729, \; -0,140, \; 0,180]$$

### İkinci residual toplama ve nihai blok çıktısı H2

**1. İkinci Atlama Toplamı ($r_2 = h_1 + f$):**

$$r_2 = [-0,691 + 0,038, \; 1,617 + 0,729, \; -0,952 - 0,140, \; 0,027 + 0,180]$$

$$r_2 = [-0,653, \; 2,346, \; -1,092, \; 0,207]$$

**2. İkinci Katman Normalizasyonu ($H_2 = \text{LayerNorm}(r_2)$):**

- Ortalama: $\mu_2 = \frac{-0,653 + 2,346 - 1,092 + 0,207}{4} = \frac{0,808}{4} = 0,202$
- Varyans: $\sigma_2^2 = \frac{(-0,855)^2 + (2,144)^2 + (-1,294)^2 + (0,005)^2}{4} = \frac{0,731 + 4,597 + 1,674 + 0,000}{4} \approx 1,751$
- Standart sapma: $\sqrt{1,751} \approx 1,323$

Normalleştirilmiş değerler:

$$H_2[0] = \frac{-0,653 - 0,202}{1,323} \approx -0,646$$

$$H_2[1] = \frac{2,346 - 0,202}{1,323} \approx 1,621$$

$$H_2[2] = \frac{-1,092 - 0,202}{1,323} \approx -0,978$$

$$H_2[3] = \frac{0,207 - 0,202}{1,323} \approx 0,004$$

$$H_2 = [-0,646, \; 1,621, \; -0,978, \; 0,004]$$

İşte bu $H_2$ vektörü, tek bir Transformer bloğunun ürettiği nihai çıktıdır! 1. token olan `"köpek"`, attention ile diğer kelimelerden bağlam toplamış, residual otoyol ile ham kimliğini korumuş, LayerNorm ile dengelenmiş, FFN ile konveks örtü sınırını parçalamış ve bir sonraki Transformer bloğuna girmeye hazır hâle gelmiştir.

1. token'ın bloktaki yolculuğunun tamamı, tek sayfada:

<svg viewBox="0 0 560 420" role="img" aria-label="1. token, köpek, tek bir transformer bloğundan geçerken; metindeki değerlerle dokuz adet 4 sayılık şerit. z1 0,210, 0,820, 0,130, 0,440 ile attention çıktısı o1 1,005, 1,041, 1,012, 0,976 toplanınca r1 1,215, 1,861, 1,142, 1,416 olur. LayerNorm bunu h1 -0,691, 1,617, -0,952, 0,027&#x27;ye çevirir. FFN altı aktivasyon öncesi değere genişler: u -0,475, 0,674, 0,712, -0,023, 0,669, -0,928; ReLU 0., 3. ve 5. nöronları sıfırlar. W2 onu f 0,038, 0,729, -0,140, 0,180&#x27;e geri indirir. İkinci atlama r2 -0,653, 2,346, -1,092, 0,207&#x27;yi, LayerNorm da blok çıktısı H2 -0,646, 1,621, -0,978, 0,004&#x27;ü verir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="tb-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="27" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">z₁</text>
<text x="16" y="41" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">girdi</text>
<rect x="150" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.18"/>
<text x="171.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,210</text>
<rect x="196" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.35"/>
<text x="217.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,820</text>
<rect x="242" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.16"/>
<text x="263.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,130</text>
<rect x="288" y="16" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.24"/>
<text x="309.5" y="31" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,440</text>
<text x="16" y="61" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">+ o₁</text>
<text x="16" y="75" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">attention çıktısı</text>
<rect x="150" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.40"/>
<text x="171.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,005</text>
<rect x="196" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.41"/>
<text x="217.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,041</text>
<rect x="242" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.40"/>
<text x="263.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,012</text>
<rect x="288" y="50" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.39"/>
<text x="309.5" y="65" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,976</text>
<text x="16" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">r₁</text>
<text x="16" y="109" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">1. atlama</text>
<rect x="150" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.46"/>
<text x="171.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,215</text>
<rect x="196" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.64"/>
<text x="217.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,861</text>
<rect x="242" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.44"/>
<text x="263.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,142</text>
<rect x="288" y="84" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.52"/>
<text x="309.5" y="99" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,416</text>
<text x="16" y="139" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">h₁</text>
<text x="16" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">LayerNorm</text>
<rect x="150" y="128" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.31"/>
<text x="171.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,691</text>
<rect x="196" y="128" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.57"/>
<text x="217.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,617</text>
<rect x="242" y="128" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.39"/>
<text x="263.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,952</text>
<rect x="288" y="128" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.13"/>
<text x="309.5" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,027</text>
<text x="16" y="183" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">u</text>
<text x="16" y="197" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">W₁h₁ + b₁ (×6)</text>
<rect x="150" y="172" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.25"/>
<text x="171.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,475</text>
<rect x="196" y="172" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="217.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,674</text>
<rect x="242" y="172" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.32"/>
<text x="263.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,712</text>
<rect x="288" y="172" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.13"/>
<text x="309.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,023</text>
<rect x="334" y="172" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="355.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,669</text>
<rect x="380" y="172" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.38"/>
<text x="401.5" y="187" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,928</text>
<text x="16" y="217" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">ReLU</text>
<text x="16" y="231" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">6&#x27;dan 3&#x27;ü ateşler</text>
<rect x="150" y="206" width="43" height="22" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1;stroke-dasharray:3 2"/>
<text x="171.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,000</text>
<rect x="196" y="206" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="217.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,674</text>
<rect x="242" y="206" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.32"/>
<text x="263.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,712</text>
<rect x="288" y="206" width="43" height="22" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1;stroke-dasharray:3 2"/>
<text x="309.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,000</text>
<rect x="334" y="206" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.31"/>
<text x="355.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,669</text>
<rect x="380" y="206" width="43" height="22" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1;stroke-dasharray:3 2"/>
<text x="401.5" y="221" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,000</text>
<text x="16" y="261" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">f</text>
<text x="16" y="275" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">W₂a + b₂</text>
<rect x="150" y="250" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.13"/>
<text x="171.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,038</text>
<rect x="196" y="250" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.32"/>
<text x="217.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,729</text>
<rect x="242" y="250" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.16"/>
<text x="263.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,140</text>
<rect x="288" y="250" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.17"/>
<text x="309.5" y="265" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,180</text>
<text x="16" y="305" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">r₂ = h₁ + f</text>
<text x="16" y="319" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">2. atlama</text>
<rect x="150" y="294" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.30"/>
<text x="171.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,653</text>
<rect x="196" y="294" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.75"/>
<text x="217.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2,346</text>
<rect x="242" y="294" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.43"/>
<text x="263.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-1,092</text>
<rect x="288" y="294" width="43" height="22" rx="3" style="fill:var(--c-accent);fill-opacity:0.18"/>
<text x="309.5" y="309" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,207</text>
<text x="16" y="349" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600;font-family:var(--font-mono)">H₂</text>
<text x="16" y="363" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">blok çıktısı</text>
<rect x="150" y="338" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.30"/>
<text x="171.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,646</text>
<rect x="196" y="338" width="43" height="22" rx="3" style="fill:var(--c-success);fill-opacity:0.57"/>
<text x="217.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,621</text>
<rect x="242" y="338" width="43" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:0.39"/>
<text x="263.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">-0,978</text>
<rect x="288" y="338" width="43" height="22" rx="3" style="fill:var(--c-success);fill-opacity:0.12"/>
<text x="309.5" y="353" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,004</text>
<path d="M430 16 H438 V106 H430" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<text x="446" y="57.0" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px">attention</text>
<text x="446" y="72.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">+ atlama</text>
<path d="M430 128 H438 V228 H430" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="446" y="174.0" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px">FFN genişlet</text>
<text x="446" y="189.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">+ ReLU</text>
<path d="M430 250 H438 V360 H430" style="fill:none;stroke:var(--c-success);stroke-width:1.4"/>
<text x="446" y="301.0" text-anchor="start" style="fill:var(--c-success);font-size:11.5px">FFN daralt,</text>
<text x="446" y="316.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">atlama + norm</text>
<path d="M142 27 C 116 27, 116 95, 142 95" marker-end="url(#tb-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M142 139 C 110 139, 110 305, 142 305" marker-end="url(#tb-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="16" y="384" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Kırmızı = negatif. Kesikli yaylar atlama yollarıdır: her alt katman, aldığına yalnızca bir</text>
<text x="16" y="400" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">delta ekler. 3. nöron 0,023 farkla kaçırır, ReLU onu susturur.</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Yürüyüşteki gibi Post-LN düzeni; Llama tarzı bloklar normu her alt katmandan önce uygular.</text>
</svg>

---

## 11. Modern üretim modellerinde bloğun evrimi: Llama 3, Gemma 2 ve DeepSeek-V3

Orijinal 2017 bloğu günümüzün üretim modellerinde önemli evrimler geçirmiştir:

### Meta Llama 3: Pre-RMSNorm, SwiGLU ve sıfır bias mimarisi

Meta'nın amiral gemisi Llama 3 (8B ve 70B), Transformer bloğunu son derece sadeleştirmiştir:
1. **Pre-RMSNorm:** LayerNorm yerine hesaplama yükü hafif olan RMSNorm kullanılır.
2. **SwiGLU Aktivasyonu:** Klasik FFN yerine kapılı SwiGLU mimarisi tercih edilir.
3. **Sıfır Bias ($\text{bias} = \text{False}$):** Dikkat projeksiyonlarında, FFN katmanlarında ve normalizasyonlarda tüm ekleme parametreleri ($b$) kaldırılmıştır. Bu durum GPU matris çekirdeklerinin (Tensor Core) verimini artırır ve aşırı öğrenmeyi azaltır.

### Gemma 2 ve DeepSeek-V3: Çift normalizasyon ve Attention-FFN ayrıştırması (AFD)

- **Google Gemma 2 (Çift Normalizasyon):** Gemma 2, 27 milyar parametreli derin modellerinde eğitim kararlılığını garantiye almak için hem alt katmanların girişine (pre-norm) hem de çıkışına (post-norm) normalizasyon koymuştur (**Dual Norm**). Ayrıca logit kaymalarını önlemek için attention matrisine $30.0$ soft-capping uygular.
- **DeepSeek-V3 (Attention-FFN Disaggregation - AFD & MoE):** DeepSeek-V3, FFN katmanını **Mixture of Experts (MoE)** yapısına çevirmiştir: 1 sabit paylaşımlı uzman (shared expert) ve 256 yönlendirilen uzmandan (routed experts) token başına en uygun 8 tanesini seçer. Ayrıca devasa kümelerde attention ve FFN hesaplamalarını farklı donanım hızlandırıcılarına dağıtan **Attention-FFN Disaggregation (AFD)** mimarisine öncülük etmiştir.

---

## 12. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri

### Eğitimde bellek optimizasyonu: Aktivasyon kontrol noktaları (Checkpointing)

Bir Transformer bloğu eğitilirken, geri yayılımda (backward pass) gradyanları hesaplayabilmek için ileri geçişteki (forward pass) tüm ara tensörlerin bellekte saklanması gerekir:
- Attention benzerlik matrisleri ($N \times N$)
- Normalizasyon girdileri ve istatistikleri ($\mu, \sigma$)
- FFN'in $4d_{\text{model}}$ boyutundaki devasa ara aktivasyonları ($a$)

Uzun dizilerde bu aktivasyonlar GPU VRAM'ini tüketir (Out of Memory - OOM). Bu krizi çözmek için **Activation Checkpointing (Gradyan Kontrol Noktaları)** tekniği uygulanır:
Bloğun içindeki ara tensörler bellekte tutulmaz, silinir; yalnızca bloğun giriş tensörü $H_0$ saklanır. Geri yayılım bloğa ulaştığında ileri geçiş **yalnızca o blok için anında baştan hesaplanır (rematerialization)**. $\%20-\%30$ hesaplama maliyeti karşılığında bellek tüketimi $\%70$'e varan oranda düşürülür.

### Çıkarımda hesaplama profili: Attention'ın bellek darboğazı vs FFN'in matris gücü

Çıkarım (inference) anında iki katman tamamen zıt donanım profilleri sergiler:

```
Attention (Decode Aşaması): GEMV (Matris-Vektör Çarpımı) ──> Bellek Bant Genişliği Darboğazı (Memory-Bound)
FFN Katmanı:                GEMM (Matris-Matris Çarpımı) ──> Hesaplama Gücü Darboğazı (Compute-Bound)
```

- **Attention katmanı bellek bant genişliğine kilitlenir:** Tek bir yeni token üretirken model KV Cache'teki gigabaytlarca geçmiş veriyi GPU VRAM'inden işlemci çekirdeklerine taşımak zorundadır. Aritmetik yoğunluk (FLOP/byte) düşüktür.
- **FFN katmanı tensör çekirdeklerini besler:** Her token kendi bağımsız devasa ağırlık matrisleriyle ($W_1, W_2$) çarpılır. Aritmetik yoğunluk yüksektir; GPU'nun Tensor Core birimleri burada tam kapasiteyle çalışır.

---

## 13. PyTorch ile adım adım tam doğrulama

### Tek bir transformer bloğu modülü ve elle hesaplanan tensörlerin teyidi

Bölüm 6 ve Bölüm 10'da elle yürüttüğümüz tüm adımları, atlama toplamlarını, LayerNorm istatistiklerini ve FFN matrislerini birebir teyit eden PyTorch scriptini çalıştıralım:

```python
import torch
import torch.nn as nn

# 1. Girdi ve Attention çıktısı (Bölüm 6 tensörleri)
z1 = torch.tensor([0.210, 0.820, 0.130, 0.440], dtype=torch.float32)
o1 = torch.tensor([1.005, 1.041, 1.012, 0.976], dtype=torch.float32)

# 2. Birinci Residual Toplama
r1 = z1 + o1
print(f"1. Residual (r1): {[round(x, 4) for x in r1.tolist()]}")

# 3. Birinci LayerNorm (gamma=1, beta=0, eps=1e-5)
ln1 = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
with torch.no_grad():
    ln1.weight.copy_(torch.ones(4))
    ln1.bias.copy_(torch.zeros(4))
h1 = ln1(r1)
print(f"LayerNorm 1 (h1): {[round(x, 4) for x in h1.tolist()]}")

# 4. FFN Ağırlıkları ve İleri Geçiş (Bölüm 10 tensörleri)
W1 = torch.tensor([
    [ 0.2, -0.1,  0.3,  0.4],
    [-0.5,  0.2,  0.1,  0.0],
    [ 0.1,  0.3, -0.2,  0.2],
    [ 0.4,  0.1,  0.0, -0.3],
    [-0.2,  0.5,  0.4,  0.1],
    [ 0.3, -0.4,  0.2,  0.6]
], dtype=torch.float32)
b1 = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=torch.float32)

W2 = torch.tensor([
    [ 0.2,  0.1, -0.3,  0.4,  0.2, -0.1],
    [-0.2,  0.3,  0.2, -0.1,  0.5,  0.4],
    [ 0.1, -0.4,  0.3,  0.2, -0.2,  0.1],
    [ 0.5,  0.2, -0.1,  0.3,  0.1, -0.3]
], dtype=torch.float32)
b2 = torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float32)

# FFN Hesabı
u = torch.matmul(W1, h1) + b1
a = torch.relu(u)
f = torch.matmul(W2, a) + b2
print(f"FFN Çıktısı (f): {[round(x, 4) for x in f.tolist()]}")

# 5. İkinci Residual Toplama
r2 = h1 + f
print(f"2. Residual (r2): {[round(x, 4) for x in r2.tolist()]}")

# 6. İkinci LayerNorm ve Nihai Çıktı
ln2 = nn.LayerNorm(4, eps=1e-5, elementwise_affine=True)
with torch.no_grad():
    ln2.weight.copy_(torch.ones(4))
    ln2.bias.copy_(torch.zeros(4))
h2 = ln2(r2)
print(f"Nihai Blok Çıktısı (H2): {[round(x, 4) for x in h2.tolist()]}")
```

### Sayısal doğrulama çıktısı ve mutlak hata analizi

Yukarıdaki kodu çalıştırdığımızda terminal çıktısı şu şekildedir:

```text
1. Residual (r1): [1.215, 1.861, 1.142, 1.416]
LayerNorm 1 (h1): [-0.6914, 1.6168, -0.9522, 0.0268]
FFN Çıktısı (f): [0.0376, 0.7287, -0.1397, 0.1804]
2. Residual (r2): [-0.6538, 2.3455, -1.0919, 0.2072]
Nihai Blok Çıktısı (H2): [-0.6467, 1.6204, -0.9778, 0.0041]
```

Elle adım adım hesapladığımız değerler ile PyTorch'un 32-bit floating point tensör çıktısı arasındaki mutlak hata farkı $< 0,001$ düzeyindedir. Tüm matematiksel omurga kuruşu kuruşuna doğrulanmıştır.

---

## Bütün hikâye altı satırda

- **Tek başına attention yetersizdir:** Attention sadece bir yönlendiricidir; Softmax kuralı yüzünden giren Value vektörlerinin konveks örtüsüne hapsolur, tek başına mantıksal sentez ve olgusal bilgi üretemez.
- **FFN konveks örtüyü parçalar:** Boyutu $4d_{\text{model}}$'e genişletip doğrusal-olmayan aktivasyonlar (GELU/SwiGLU) uygulayarak attention'ın erişemediği yeni kavramsal koordinatları sentezler.
- **Parametrelerin üçte ikisi FFN'dedir:** Geva vd. (2021) kanıtladığı üzere $W_1$ anahtarları kavramları tanır, $W_2$ değerleri ise internetteki olgusal bilgileri residual akışa fısıldayan bir ilişkisel bellek olarak çalışır.
- **Residual bağlantı gradyan otoyoludur:** İleri yönde $y = x + f(x)$ yapısı, geriye yayılımda $\mathbf{I} + f'(x)$ türevini üretir; katmanlar doysa bile sinyal birim matris ($\mathbf{I}$) üzerinden 100 katman boyunca kayıpsız akar.
- **LayerNorm/RMSNorm kontrolsüz büyümeyi dizginler:** Residual toplamlardan kaynaklanan varyans patlamalarını her token için kendi içinde standartlaştırır; NLP'de değişken uzunluk ve tekli çıkarım yüzünden BatchNorm kullanılamaz.
- **İki motorun mükemmel dengesi:** Attention token'lar arası iletişimi ($N \times N$), FFN ise token içi derin düşünmeyi ($d_{\text{model}} \to 4d_{\text{model}}$) yönetir; bu iki motor üst üste dizilerek modern yapay zekayı oluşturur.

---

## Terimler sözlüğü

- **Transformer Bloğu (Transformer Block):** Attention, residual bağlantı, normalizasyon ve FFN alt katmanlarını içeren homojen, tekrarlanabilir temel mimari birim.
- **Residual Connection (Atlama / Skip Bağlantısı):** Bir katmanın çıktısına orijinal girdisini ekleyen ($y = x + f(x)$), gradyan kaybolmasını önleyen mimari otoyol.
- **Vanishing Gradient (Kaybolan Gradyan):** Derin ağlarda zincir kuralı çarpımlarının $1$'den küçük türevler nedeniyle sıfıra yaklaşması ve ağırlık güncellemelerini dondurması.
- **Birim Matris Otoyolu ($\mathbf{I} + f'(x)$):** Residual bağlantının türevi; ara katman fonksiyonu çökse bile geriye akan gradyanın en az $\mathbf{I}$ gücünde akmaya devam etmesi güvencesi.
- **Layer Normalization (LayerNorm):** Bir token'ın tüm gizli özelliklerini kendi ortalaması ve varyansıyla sıfır ortalama ve birim varyansa getiren normalizasyon tekniği.
- **RMSNorm (Root Mean Square Normalization):** Ortalama çıkarma adımını atlayıp yalnızca kök ortalama kareye bölerek çalışan ve bellek bant genişliği tasarrufu sağlayan modern normalizasyon.
- **Konveks Örtü (Convex Hull):** Noktaların pozitif ve toplamı 1 olan ağırlıklarla sınırlanan geometrik zarfı; Softmax nedeniyle attention çıktılarının hapsolduğu uzay sınırı.
- **Feed-Forward Network (FFN / MLP):** Her token'a bağımsız uygulanan, boyutu genişletip daraltarak doğrusal-olmayan akıl yürütme sağlayan iki veya üç katmanlı tam bağlantılı ağ.
- **İlişkisel Bellek (Key-Value Associative Memory):** $W_1$ matrisinin girdi örüntülerini (anahtar), $W_2$ matrisinin ise depolanan olgusal bilgileri (değer) temsil ettiği FFN çalışma mekanizması.
- **GELU (Gaussian Error Linear Unit):** Girdileri normal dağılım kümülatif olasılığıyla çarparak yumuşak kapılama sağlayan pürüzsüz aktivasyon fonksiyonu.
- **SwiGLU:** Swish aktivasyonu ve doğrusal kapılamayı birleştiren, modern açık kaynaklı modellerin standart FFN mimarisi.

---

## Daha derine inmek için

- **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) — Residual bağlantıların ve ResNet mimarisinin temeli.
- **Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016).** *Layer Normalization.* [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) — LayerNorm'un matematiksel formülasyonu.
- **Zhang, B., & Sennrich, R. (2019).** *Root Mean Square Layer Normalization.* [arXiv:1910.07467](https://arxiv.org/abs/1910.07467) — RMSNorm optimizasyonu.
- **Xiong, R., et al. (2020).** *On Layer Normalization in the Transformer Architecture.* [arXiv:2002.04745](https://arxiv.org/abs/2002.04745) — Pre-LN vs Post-LN kararlılık ispatı.
- **Vaswani, A., et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — Özgün Transformer mimarisi.
- **Hendrycks, D., & Gimpel, K. (2016).** *Gaussian Error Linear Units (GELUs).* [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) — GELU aktivasyonunun doğuşu.
- **Shazeer, N. (2020).** *GLU Variants Improve Transformer.* [arXiv:2002.05202](https://arxiv.org/abs/2002.05202) — SwiGLU mimarisinin temeli.
- **Geva, M., et al. (2021).** *Transformer Feed-Forward Layers Are Key-Value Memories.* [arXiv:2012.14913](https://arxiv.org/abs/2012.14913) — FFN'in ilişkisel bellek mekanizması.
- **Dai, D., et al. (2022).** *Knowledge Neurons in Pretrained Transformers.* [arXiv:2104.08696](https://arxiv.org/abs/2104.08696) — Model içindeki olgusal bilgi nöronları.
- **Meng, K., et al. (2022).** *Locating and Editing Factual Associations in GPT (ROME).* [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) — FFN ağırlıklarını cerrahi düzenleme.
- **Meta AI (2024).** *The Llama 3 Herd of Models.* [arXiv:2407.21783](https://arxiv.org/abs/2407.21783) — Pre-RMSNorm ve SwiGLU mimarisi.
- **Google DeepMind (2024).** *Gemma 2: Improving Open Language Models at a Practical Size.* [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) — Çift normalizasyon mimarisi.
- **DeepSeek-AI (2024).** *DeepSeek-V3 Technical Report.* [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) — MoE ve Attention-FFN Disaggregation (AFD).
- **Hochreiter, S. (1991).** *Untersuchungen zu dynamischen neuronalen Netzen.* Diploma Thesis, TU Munich — Kaybolan gradyanın ilk tespiti.
- **Bengio, Y., Simard, P., & Frasconi, P. (1994).** *Learning long-term dependencies with gradient descent is difficult.* IEEE TNN — Derin ve rekürrent ağlarda gradyan sönümlemesi.
