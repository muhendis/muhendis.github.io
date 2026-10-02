Çok sınıflı bir görüntü sınıflandırma veya bir sonraki token'ı tahmin eden bir dil modelini eğittiğinizi hayal edin. Model mimarisini kurdunuz, çıktı katmanına Softmax olasılık dağıtımını yerleştirdiniz ve amaç fonksiyonu olarak da en sezgisel görünen Ortalama Karesel Hatayı (MSE - Mean Squared Error) seçtiniz. Eğitim sırasında model bir kedi görseliyle karşılaşıyor; ancak kediye %0.05 olasılık verirken "kamyon" sınıfına %99.9 olasılık atıyor.

Mühendislik sezgisi, model böylesine vahim bir hata yaptığında geriye yayılımın (backpropagation) hatayı hızla düzeltecek devasa bir düzeltici gradyan üretmesini bekler. Oysa gerçekte tam tersi yaşanır: eğitim donar. Kayıp eğrisi aşılmaz düz bir platoya saplanır, parametre güncelleme normları sıfıra yaklaşır ve model öğrenmeyi neredeyse tamamen durdurur.

Bu arızanın kaynağı bütünüyle geometridir: Ortalama Karesel Hata doymuş aktivasyon katmanlarıyla birleştiğinde, logit uzayında gradyanı sıfırlanmış devasa düzlükler barındıran konveks-olmayan bir kayıp yüzeyi üretir. Kayıp fonksiyonu, gradyan inişinin (gradient descent) pusulasıdır; eğer pusula modelin en çok yanıldığı yerde eğimin "sıfır" olduğunu söylüyorsa optimizasyon çöker. Bu yazı, derin ağlarda hatanın geometrisini inceliyor; Shannon bilgi teorisi ve KL uzaklığından Çapraz Entropiye (Cross-Entropy) uzanan köprüyü kuruyor; $\hat{y} - y$ gradyan sadeleşmesinin matematiksel zarafetini ispatlıyor; IEEE 754 float16 donanımlarının `NaN` çöküşlerini engellemek için neden Log-Sum-Exp hilesine muhtaç olduğunu açıklıyor; InfoNCE gibi modern kontrastif kayıpları irdeliyor ve elle yapılan sayısal tensör yürüyüşünü doğrulanabilir kodlarla sunuyor.

**Bu yazıda**

- [1. En yalın hâliyle temel sezgi: Sınıflandırma neden MSE altında iflas eder?](#1-en-yalın-hâliyle-temel-sezgi-sınıflandırma-neden-mse-altında-iflas-eder)
- [2. Bilgi teorisi temelleri: Entropi, Çapraz Entropi ve KL Uzaklığı](#2-bilgi-teorisi-temelleri-entropi-çapraz-entropi-ve-kl-uzaklığı)
- [3. Matematiksel şaheser: Softmax ve Cross-Entropy gradyan sadeleşmesi](#3-matematiksel-şaheser-softmax-ve-cross-entropy-gradyan-sadeleşmesi)
- [4. Donanım gerçeği ve Log-Sum-Exp hilesi: float16 taşmasından kurtuluş](#4-donanım-gerçeği-ve-log-sum-exp-hilesi-float16-taşmasından-kurtuluş)
- [5. Modern kayıp mimarileri: Label Smoothing, Focal Loss ve InfoNCE](#5-modern-kayıp-mimarileri-label-smoothing-focal-loss-ve-infonce)
- [6. Adım adım elle sayısal tensör yürüyüşü](#6-adım-adım-elle-sayısal-tensör-yürüyüşü)
- [7. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri](#7-iki-mühendislik-gözü-eğitim-ve-çıkarım-dinamikleri)
- [8. Python ve PyTorch ile doğrulama](#8-python-ve-pytorch-ile-doğrulama)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. En yalın hâliyle temel sezgi: Sınıflandırma neden MSE altında iflas eder?

### Özgüvenli Hataların Plato Geometrisi

Ortalama Karesel Hatanın kategorik hedeflerde neden iflas ettiğini anlamak için, olasılık çıktıları üzerinden türevin cebirsel davranışını inceleyelim.

Modelin ikili bir sınıflandırma problemi için $z$ logitinden $\hat{y} = \sigma(z)$ tahmini ürettiğini varsayalım ($y \in \{0, 1\}$). MSE kaybı:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{2} (\hat{y} - y)^2 = \frac{1}{2} (\sigma(z) - y)^2$$

Zincir kuralıyla aktivasyon öncesi $z$ logitine göre türev aldığımızda:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = (\sigma(z) - y) \cdot \sigma'(z) = (\sigma(z) - y) \cdot \sigma(z)(1 - \sigma(z))$$

Şimdi en vahim hata senaryosunu ele alalım: Gerçek etiket $y = 1$, ancak model feci şekilde yanılıyor ve $z = -10$ logiti üreterek $\hat{y} = \sigma(-10) \approx 0.000045$ olasılığı veriyor.
1. Saf tahmin hatası $(\hat{y} - y) = 0.000045 - 1.0 = -0.999955$ (neredeyse maksimum hata).
2. Ancak sigmoid türev terimi $\sigma'(-10) = 0.000045 \times (1 - 0.000045) \approx \mathbf{0.000045}$.
3. Ağa geriye iletilen nihai gradyan:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = (-0.999955) \times (0.000045) \approx \mathbf{-0.000045}$$

Gradyan neredeyse sıfırdır. Model *kendinden emin* bir şekilde yanıldığı için sigmoid türevi doymuş (saturation) ve ağırlıkları düzeltecek hata sinyalini yok etmiştir. MSE altında kayıp yüzeyi logit uzayında konveks değildir: en büyük hatalar dik vadiler yerine dümdüz, yatay platolara denk gelir.

<svg viewBox="0 0 560 300" role="img" aria-label="Emin bir hata üzerinde MSE ile cross-entropy. Hedef y eşittir 1, model ise logit eksi 10&#x27;da p yaklaşık 0,000045 veriyor. MSE&#x27;de sigmoid türevi p çarpı 1 eksi p sıfıra doyar; gradyan, yani hata çarpı türev, yaklaşık eksi 0,000045 olur ve optimizasyon platoda donup kalır. Cross-entropy&#x27;de logaritma softmax paydasını sadeleştirir, gradyan doğrusaldır, p eksi y, yaklaşık eksi 0,999955; model en güçlü düzeltici itişi alır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="mc-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="mc-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<rect x="16" y="8" width="256" height="284" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="30" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">MSE</text>
<rect x="30" y="44" width="228" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="63.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Vahim hata</text>
<text x="144.0" y="79.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">y = 1, p ≈ 0,000045</text>
<line x1="144" y1="90" x2="144" y2="104" marker-end="url(#mc-r)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="30" y="106" width="228" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="125.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Türev doyar</text>
<text x="144.0" y="141.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">p(1 − p) → 0</text>
<line x1="144" y1="152" x2="144" y2="166" marker-end="url(#mc-r)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="30" y="168" width="228" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="187.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Gradyan = hata × türev</text>
<text x="144.0" y="203.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">≈ −0,000045</text>
<line x1="144" y1="214" x2="144" y2="228" marker-end="url(#mc-r)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="30" y="230" width="228" height="46" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="144.0" y="249.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Platoda donup kalır</text>
<text x="144.0" y="265.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">sinyal kayboldu</text>
<rect x="288" y="8" width="256" height="284" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="302" y="30" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">Cross-entropy</text>
<rect x="302" y="44" width="228" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="63.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Vahim hata</text>
<text x="416.0" y="79.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">y = 1, p ≈ 0,000045</text>
<line x1="416" y1="90" x2="416" y2="104" marker-end="url(#mc-o)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="302" y="106" width="228" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="125.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Logaritma softmax</text>
<text x="416.0" y="141.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">paydasını sadeleştirir</text>
<line x1="416" y1="152" x2="416" y2="166" marker-end="url(#mc-o)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="302" y="168" width="228" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="187.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Doğrusal gradyan = p − y</text>
<text x="416.0" y="203.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">≈ −0,999955</text>
<line x1="416" y1="214" x2="416" y2="228" marker-end="url(#mc-o)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="302" y="230" width="228" height="46" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416.0" y="249.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">En güçlü düzeltici itiş</text>
<text x="416.0" y="265.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">adımı hata sürer</text>
</svg>

### Cross-Entropy Çözümü

Cross-Entropy kaybında $\mathcal{L}_{\text{CE}} = -y \log \hat{y} - (1 - y) \log(1 - \hat{y})$, logaritmik ceza terimi sigmoid türevini tam olarak nötrler:

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = \hat{y} - y$$

Aynı vahim hata noktasında ($y = 1, \hat{y} = 0.000045$) gradyanı değerlendirelim:

$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = 0.000045 - 1.0 = \mathbf{-0.999955}$$

Mikroskobik bir $0.000045$ yerine Cross-Entropy, logit katmanına tam şiddette bir $-0.999955$ gradyanı iletir. Gradyan saf hatayla doğrudan orantılıdır: model doğru tahmin yaptığında gradyan pürüzsüzce sıfırlanır; vahim bir hata yaptığında ise maksimum düzeltici kuvvet uygular.

---

## 2. Bilgi teorisi temelleri: Entropi, Çapraz Entropi ve KL Uzaklığı

Cross-Entropy fonksiyonunun bu üstünlüğü bir tesadüf değildir; Claude Shannon'ın 1948 yılında kurduğu matematiksel iletişim teorisinin doğal bir sonucudur.

### Öz-Bilgi (Surprisal)

$P(x)$ olasılığına sahip bir $x$ olayının gerçekleştiğini öğrendiğimizde ne kadar bilgi elde ederiz?

Kesin olarak beklenen bir olay ($P(x) = 1$) gerçekleştiğinde yeni hiçbir bilgi kazanılmaz. Çok nadir ve beklenmedik bir olay ($P(x) \to 0$) gerçekleştiğinde ise muazzam miktarda bilgi açığa çıkar. Shannon bu sezgiyi bit cinsinden **öz-bilgi (surprisal)** olarak tanımladı:

$$I(x) = -\log_2 P(x)$$

### Shannon Entropisi: Temel Sınır

Ayrık bir $P$ olasılık dağılımının **Shannon Entropisi** $H(P)$, bu dağılımdan gelen bir olayın taşıdığı beklenen bilgi miktarıdır. Bir dağılımı kodlamak için gereken ortalama bit sayısının teorik alt sınırını temsil eder:

$$H(P) = \mathbb{E}_{x \sim P}[I(x)] = -\sum_{i=1}^K P(x_i) \log_2 P(x_i)$$

Tekil bir doğru cevabı olan tek-sıcak (one-hot) hedef dağılımı ($P = [0, \dots, 1, \dots, 0]$) için belirsizlik sıfırdır:

$$H(P) = - (1 \log 1 + 0 \log 0) = \mathbf{0}$$

### Çapraz Entropi: Kusurlu Bir Modelin Maliyeti

Gerçek veri dağılımının $P$ olduğunu, ancak bizim bunu bir $Q$ model dağılımıyla (yapay sinir ağının tahminleri) temsil ettiğimizi düşünelim. $Q$ hipotezine göre optimize edilmiş bir kodlama sistemi tasarlar, ancak bu sistemi gerçekte $P$ dağılımına göre gelen sembolleri iletmek için kullanırsak, sembol başına ortalama kod uzunluğu **Çapraz Entropi (Cross-Entropy)** olur:

$$H(P, Q) = \mathbb{E}_{x \sim P}[-\log Q(x)] = -\sum_{i=1}^K P(x_i) \log Q(x_i)$$

Çapraz Entropi toplam kodlama maliyetini ölçer: hem verinin kendi iç belirsizliğini ($H(P)$) hem de modelimizin hatalı tahminlerinden kaynaklanan ek kodlama yükünü içerir.

### Kullback-Leibler (KL) Uzaklığı

Gerçek $P$ dağılımı yerine $Q$ dağılımını kullanmanın getirdiği fazladan bilgi maliyeti **Kullback-Leibler (KL) Uzaklığı** (göreli entropi) olarak tanımlanır:

$$D_{\text{KL}}(P \parallel Q) = H(P, Q) - H(P) = \sum_{i=1}^K P(x_i) \log \left( \frac{P(x_i)}{Q(x_i)} \right)$$

Gibbs eşitsizliğine göre $D_{\text{KL}}(P \parallel Q) \ge 0$'dır ve yalnızca $Q(x) = P(x)$ olduğunda sıfıra eşit olur.

Terimleri yeniden düzenlediğimizde:

$$H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)$$

Derin öğrenmede gerçek eğitim etiketleri $P$ sabittir; dolayısıyla $H(P)$ değişmez bir sabittir. Bu nedenle:

$$\arg\min_{\theta} H(P, Q_\theta) \equiv \arg\min_{\theta} D_{\text{KL}}(P \parallel Q_\theta) \equiv \arg\max_{\theta} \sum_{i=1}^N \log Q_\theta(y_i \mid x_i)$$

Çapraz Entropiyi minimize etmek, sinir ağının ürettiği dağılım ile gerçeklik arasındaki KL uzaklığını sıfırlamakla ve **En Çok Olabilirlik Kestirimi (Maximum Likelihood Estimation - MLE)** yapmakla birebir aynıdır.

---

## 3. Matematiksel şaheser: Softmax ve Cross-Entropy gradyan sadeleşmesi

Çok sınıflı sınıflandırmada ve dil modellerinde ağ, ham puanlardan oluşan bir **logit** vektörü üretir: $z \in \mathbb{R}^K$.

**Softmax** operatörü bu logitleri geçerli bir kategorik olasılık dağılımına $p \in \Delta^{K-1}$ dönüştürür:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

Tek-sıcak (one-hot) hedef vektörü $y$ için kategorik **Cross-Entropy kaybı**:

$$\mathcal{L} = -\sum_{k=1}^K y_k \log p_k$$

### Adım 1: Softmax Fonksiyonunun Jacobian Matrisi

$\frac{\partial \mathcal{L}}{\partial z_i}$ gradyanını bulmak için önce bir $p_k$ olasılığının herhangi bir $z_i$ logitine göre türevini almalıyız. İki durum ortaya çıkar:

**Durum A: $k = i$ (Köşegen Elemanlar)**
Bölüm kuralı uygulandığında:

$$\frac{\partial p_i}{\partial z_i} = \frac{\frac{d}{dz_i}(e^{z_i}) \cdot \sum e^{z_j} - e^{z_i} \cdot \frac{d}{dz_i}(\sum e^{z_j})}{(\sum e^{z_j})^2} = \frac{e^{z_i} \sum e^{z_j} - e^{z_i} e^{z_i}}{(\sum e^{z_j})^2}$$

Terimleri ayırdığımızda:

$$\frac{\partial p_i}{\partial z_i} = \frac{e^{z_i}}{\sum e^{z_j}} - \left(\frac{e^{z_i}}{\sum e^{z_j}}\right)^2 = p_i - p_i^2 = p_i (1 - p_i)$$

**Durum B: $k \ne i$ (Köşegen Dışı Elemanlar)**
Burada paydaki $e^{z_k}$ ifadesi $z_i$'ye göre sabittir:

$$\frac{\partial p_k}{\partial z_i} = \frac{0 \cdot \sum e^{z_j} - e^{z_k} \cdot e^{z_i}}{(\sum e^{z_j})^2} = -\frac{e^{z_k}}{\sum e^{z_j}} \cdot \frac{e^{z_i}}{\sum e^{z_j}} = -p_k p_i$$

Her iki durumu Kronecker delta $\delta_{ik}$ (eğer $k=i$ ise $1$, değilse $0$) ile tek bir denklemde birleştirelim:

$$\frac{\partial p_k}{\partial z_i} = p_k (\delta_{ik} - p_i)$$

### Adım 2: Cross-Entropy ile Zincir Kuralının İşletilmesi

Şimdi tüm $K$ sınıf üzerinden zincir kuralını işletelim:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^K \frac{\partial \mathcal{L}}{\partial p_k} \frac{\partial p_k}{\partial z_i}$$

Kaybın $p_k$ olasılığına göre türevi:

$$\frac{\partial \mathcal{L}}{\partial p_k} = \frac{\partial}{\partial p_k} \left( -y_k \log p_k \right) = -\frac{y_k}{p_k}$$

Her iki ifadeyi toplamın içine yerleştirelim:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \sum_{k=1}^K \left( -\frac{y_k}{p_k} \right) \left[ p_k (\delta_{ik} - p_i) \right]$$

Paydadaki $p_k$ ile Softmax türevinin çarpanı olan $p_k$ birbirini kusursuz bir şekilde götürür:

$$\frac{\partial \mathcal{L}}{\partial z_i} = -\sum_{k=1}^K y_k (\delta_{ik} - p_i) = -\sum_{k=1}^K y_k \delta_{ik} + p_i \sum_{k=1}^K y_k$$

$\delta_{ik}$ yalnızca $k = i$ iken $1$ olduğundan ilk toplam $-y_i$ değerine çöker. Ayrıca $y$ geçerli bir olasılık dağılımı olduğundan elemanlarının toplamı tam olarak $1$'dir ($\sum_{k=1}^K y_k = 1$):

$$\frac{\partial \mathcal{L}}{\partial z_i} = -y_i + p_i (1) = \mathbf{p_i - y_i}$$

Vektör gösterimiyle:

$$\nabla_z \mathcal{L} = \mathbf{p} - \mathbf{y} = \mathbf{\hat{y}} - \mathbf{y}$$

Bu sadeleşme derin öğrenmenin en büyük yapısal zaferlerinden biridir:
1. Tüm karmaşık üstel ve bölümlü terimler birbirini yok eder.
2. Gradyan, modelin tahmin ettiği olasılık ile gerçek hedef arasındaki saf doğrusal farka (artık hataya) dönüşür.
3. Tüm sınıfların gradyanları toplamı daima sıfırdır ($\sum (p_i - y_i) = 1 - 1 = 0$); bu durum geriye yayılım sırasında toplam olasılık kütlesinin korunmasını garanti eder.

---

## 4. Donanım gerçeği ve Log-Sum-Exp hilesi: float16 taşmasından kurtuluş

Teorik formül $\mathcal{L} = -\log p_k$ matematiksel olarak kusursuz görünse de, bunu doğrudan GPU üzerinde kayan noktalı sayılarla kodlamak kaçınılmaz bir eğitim çöküşüne yol açar.

### Float16 Üstel Taşma Duvarı

Modern yapay zeka modelleri IEEE 754 **Float16 (FP16)** ve **Bfloat16 (BF16)** formatlarında eğitilir.

Standart FP16 formatında:
* 1 işaret biti, 5 üs biti ve 10 kesir biti bulunur.
* Temsil edilebilecek en büyük sonlu sayı:

$$\text{MAKS}_{\text{FP16}} = (2 - 2^{-10}) \times 2^{15} = \mathbf{65504}$$

Şimdi Softmax paydası olan $\sum e^{z_j}$ ifadesini hesapladığımızı düşünelim. Hangi logit değeri bu sınırı aşar?

$$e^{z} > 65504 \iff z > \ln(65504) \approx \mathbf{11.0898}$$

Eğitim sırasında tek bir logit $z_i \ge 11.1$ değerine ulaştığında $e^{z_i}$ anında `+inf` değerine taşar (overflow). Payda `+inf` olduğunda bölme işlemi `+inf / +inf` haline gelir ve sonuç **`NaN` (Not a Number)** olur. Tek bir adımda `NaN` tüm ağ ağırlıklarına bulaşır ve eğitimi yok eder.

Tersine, tüm logitler büyük negatif sayılar olduğunda (örneğin $z = [-100, -105, -110]$), tüm $e^{z_i}$ değerleri sıfıra çöker (underflow). Payda $0.0$ olur ve **sıfıra bölme** hatası patlar.

### Log-Sum-Exp (LSE) Özdeşliği

Sayısal kararlılığı sağlamak için logaritma ve Softmax işlemlerini birleştiririz. $\log p_k$ ifadesini açtığımızda:

$$\log p_k = \log \left( \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}} \right) = z_k - \log \left( \sum_{j=1}^K e^{z_j} \right)$$

Buradaki kritik hesaplama darboğazı şu terimdir:

$$\text{LSE}(z) = \log \left( \sum_{j=1}^K e^{z_j} \right)$$

Logit vektörünün maksimum elemanına $c = \max_j z_j$ diyelim. Toplamın içinden $e^c$ çarpanını dışarı çıkaralım:

$$\sum_{j=1}^K e^{z_j} = \sum_{j=1}^K e^{z_j - c + c} = e^c \sum_{j=1}^K e^{z_j - c}$$

Her iki tarafın doğal logaritmasını aldığımızda:

$$\log \left( \sum_{j=1}^K e^{z_j} \right) = \log \left( e^c \sum_{j=1}^K e^{z_j - c} \right) = c + \log \left( \sum_{j=1}^K e^{z_j - c} \right)$$

İşte bu meşhur **Log-Sum-Exp (LSE) hilesidir**:

$$\text{LSE}(z) = \max(z) + \log \left( \sum_{j=1}^K e^{z_j - \max(z)} \right)$$

```
Ham Logitler:  [ 20.0, 15.0, 10.0 ] 
  --> Kaydırılmamış: e^20.0 = 4.85 x 10^8  (FP16'DA TAŞMA / OVERFLOW!)
  --> Maksimum c = 20.0
  --> Kaydırılmış:   [ 0.0, -5.0, -10.0 ]
  --> Üstel:         [ 1.0,  0.0067, 0.000045 ]
  --> Toplam = 1.00678  (KARARLI, >= 1.0, SIFIRA İNMEZ, TAŞMAZ)
  --> LSE = 20.0 + ln(1.00678) = 20.00676
```

### LSE Neden Asla Çökmez?

1. **Taşma İmkânsızdır (Overflow Immunity):** Maksimum $c = \max(z)$ çıkarıldığı için her yeni üs $(z_j - c) \le 0$ olur. Alınabilecek en büyük üstel değer $e^0 = 1.0$'dır. Hiçbir ara terim $1.0$'ı aşamayacağı için FP16'da taşma yaşanması imkânsızdır.
2. **Sıfıra Bölme İmkânsızdır (Underflow Immunity):** Maksimum eleman için $z_{\text{maks}} - c = 0$ olduğundan, toplamın içinde en az bir adet $e^0 = 1.0$ terimi bulunur. Bu nedenle $\sum e^{z_j - c} \ge 1.0$'dır. Logaritmanın içi daima $1$'e eşit veya büyük olduğundan $\log(0)$ tanımsızlığı tamamen ortadan kalkar.

Bu sebeple PyTorch gibi kütüphanelerde `torch.log(torch.softmax(z))` yazılmasına izin verilmez; bunun yerine `LogSoftmax` ve `NLLLoss` işlemlerini tek bir kaynaşmış CUDA LSE çekirdeğinde birleştiren `nn.CrossEntropyLoss` kullanılır.

---

## 5. Modern kayıp mimarileri: Label Smoothing, Focal Loss ve InfoNCE

Klasik Cross-Entropy sınıflandırmanın omurgası olsa da, modern modeller aşırı özgüven, veri dengesizliği ve sürekli uzayda metrik öğrenimi için özelleşmiş kayıp fonksiyonları kullanır.

### Label Smoothing: Aşırı Özgüveni Frenleme

Standart Cross-Entropy'de tek-sıcak hedef, doğru sınıfa $1.0$, diğer tüm sınıflara $0.0$ atar.

Softmax olasılıkları üstel oranlar ürettiği için:

$$p_k = \frac{e^{z_k}}{\sum e^{z_j}} = 1.0 \iff z_k - z_j \to \infty \quad \forall j \ne k$$

Kaybı tam olarak sıfıra indirmek için ağın doğru sınıf logitini sonsuza ($+\infty$), diğer logitleri ise eksi sonsuza ($-\infty$) sürmesi gerekir. Bu durum **aşırı özgüvene (overconfidence)** yol açar: model ağırlık normlarını şişirir, eğitim verisindeki gürültüyü ezberler ve kalibrasyonunu kaybeder.

2016 yılında Christian Szegedy ve ekibi **Label Smoothing** yöntemini önerdi:

$$y_k^{\text{düzeltilmiş}} = (1 - \epsilon) y_k + \frac{\epsilon}{K}$$

Burada $\epsilon$ genellikle $0.1$ seçilen bir yumuşatma katsayısı, $K$ ise sınıf sayısıdır.
* Doğru sınıfın hedefi $1 - \epsilon + \frac{\epsilon}{K} \approx 0.90 + \frac{0.1}{K}$ olur.
* Yanlış sınıfların hedefi ise $\frac{\epsilon}{K} > 0$ haline gelir.

Hedef olasılıklar kesin $0$ ve $1$ olmaktan çıktığı için ağ logit farklarını sonsuza sürmeye çalışmaz. Label smoothing yapısal bir düzenlileştirici (regularizer) işlevi görerek Transformer ve görüntü modellerinde genelleme gücünü artırır.

### Focal Loss: Zor Örnekleri Öne Çıkarma

Nesne tespiti ve dengesiz veri setlerinde arka plan gibi kolay negatif örnekler gradyanı boğar. 2017 yılında Tsung-Yi Lin ve ekibi **Focal Loss**'u geliştirdi:

$$\text{FL}(p_t) = -(1 - p_t)^\gamma \log(p_t)$$

Burada $p_t$ modelin doğru sınıf için tahmin ettiği olasılık, $\gamma \ge 0$ ise odaklanma parametresidir.
* Model zor bir örnekle karşılaştığında ($p_t \to 0$), $(1 - p_t)^\gamma \to 1$ olur ve standart Cross-Entropy korunur.
* Model örneği zaten rahatça öğrendiğinde ($p_t \ge 0.9$), $(1 - 0.9)^2 = 0.01$ katsayısı kaybı 100 kat bastırır.

Böylece kolay örnekler gradyan güncellemesini domine edemez ve model zor örneklere odaklanır.

### InfoNCE: Kontrastif Temsil Öğreniminin Motoru

Çok modlu mimarilerde (CLIP) ve yoğun vektör arama modellerinde (OpenAI `text-embedding-3`, BGE, NV-Embed) modeller statik sınıfları tahmin etmek yerine sürekli uzayda vektörleri karşılaştırarak öğrenir.

2018 yılında Aaron van den Oord ve ekibi **InfoNCE** (Information Noise-Contrastive Estimation) kaybını tanıttı:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left( \frac{\text{sim}(q, k^+)}{\tau} \right)}{\exp\left( \frac{\text{sim}(q, k^+)}{\tau} \right) + \sum_{i=1}^M \exp\left( \frac{\text{sim}(q, k_i^-)}{\tau} \right)}$$

Burada:
* $q \in \mathbb{R}^d$: Sorgu vektörü (örneğin arama metni veya görsel).
* $k^+ \in \mathbb{R}^d$: Pozitif eşleşen vektör (alakalı doküman veya metin başlığı).
* $k_i^- \in \mathbb{R}^d$: $M$ adet negatif çeldirici vektör.
* $\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$: Kosinüs benzerliği.
* $\tau > 0$: Sıcaklık (temperature) parametresi.

Buradaki mimari keşfe dikkat edin: **InfoNCE, $M+1$ adaylık dinamik bir küme üzerinde pozitif eşleşmenin 0. sınıf olduğu standart bir kategorik Cross-Entropy işlemidir!**

$\tau$ sıcaklığı olasılık dağılımının keskinliğini ayarlar; küçük bir $\tau$ değeri pozitif vektöre çok yakın duran zor negatif çeldiricilere devasa cezalar uygular.

| Kayıp Fonksiyonu | Temel Kullanım Alanı | Matematiksel Mekanizma | Donanım ve Eğitim Faydası |
| :--- | :--- | :--- | :--- |
| **Cross-Entropy** | Sınıflandırma ve Otoregresif LLM | $-\sum y_k \log p_k$ | Doğrusal gradyan $\hat{y}-y$; sıfır doyma |
| **Log-Sum-Exp CE** | Tüm üretim sistemleri | $\max(z) + \log \sum e^{z - \max(z)} - z_y$ | FP16 taşma/çöküşlerine karşı mutlak koruma |
| **Label Smoothing** | Düzenlileştirme ve Kalibrasyon | $(1-\epsilon)y + \frac{\epsilon}{K}$ | Logitlerin sonsuza ıraksamasını engeller |
| **Focal Loss** | Dengesiz veri setleri | $-(1-p_t)^\gamma \log p_t$ | Kolay örnekleri baskılayıp zora odaklanır |
| **InfoNCE** | Vektör embedding ve CLIP | Kosinüs benzerlikleri üzerinde Cross-Entropy | Sürekli uzayda karşılıklı bilgiyi maksimize eder |

---

## 6. Adım adım elle sayısal tensör yürüyüşü

Teorik denklemleri somutlaştırmak için 3 sınıflı bir sınıflandırma problemi ($K = 3$) üzerinden elle tam bir sayısal yürüyüş yapalım.

### Başlangıç Durumu

Modelin ürettiği ham logit vektörü:

$$z = \begin{bmatrix} 2.0 & 1.0 & 0.1 \end{bmatrix}$$

Doğru sınıf etiketi indeks **1** olsun (0-indeksli tek-sıcak hedef):

$$y = \begin{bmatrix} 0.0 & 1.0 & 0.0 \end{bmatrix}$$

Mühendislik sahnesi: Model şu anda yanlış tahmin yapıyor. En yüksek güveni sınıf 0'a ($z_0 = 2.0$) verirken, doğru sınıf 1 daha düşük bir logite ($z_1 = 1.0$) sahiptir.

---

### Adım 1: Sayısal Kararlı Log-Sum-Exp ve Olasılıklar

Maksimum logiti bulalım: $c = \max(z) = 2.0$. Tüm logitlerden $c$ çıkaralım:

$$z - c = \begin{bmatrix} 2.0 - 2.0 & 1.0 - 2.0 & 0.1 - 2.0 \end{bmatrix} = \begin{bmatrix} 0.0 & -1.0 & -1.9 \end{bmatrix}$$

Kaydırılmış üstel değerler:
* $e^{0.0} = \mathbf{1.000000}$
* $e^{-1.0} = \mathbf{0.367879}$
* $e^{-1.9} = \mathbf{0.149569}$

Üstsellerin toplamı:

$$\sum_{j=0}^2 e^{z_j - c} = 1.000000 + 0.367879 + 0.149569 = \mathbf{1.517448}$$

Log-Sum-Exp değeri:

$$\text{LSE}(z) = c + \ln(1.517448) = 2.0 + 0.417031 = \mathbf{2.417031}$$

Softmax olasılıkları $p_i = \frac{e^{z_i - c}}{\sum e^{z_j - c}}$:
* $p_0 = \frac{1.000000}{1.517448} = \mathbf{0.659001}$
* $p_1 = \frac{0.367879}{1.517448} = \mathbf{0.242433}$
* $p_2 = \frac{0.149569}{1.517448} = \mathbf{0.098566}$

Toplam: $0.659001 + 0.242433 + 0.098566 = \mathbf{1.000000}$.

---

### Adım 2: Cross-Entropy Kaybı ve Gradyanı

LSE formülasyonu ile kayıp:

$$\mathcal{L}_{\text{CE}} = \text{LSE}(z) - z_{\text{hedef}} = 2.417031 - 1.0 = \mathbf{1.417031}$$

Logitlere göre gradyan:

$$\nabla_z \mathcal{L}_{\text{CE}} = \mathbf{p} - \mathbf{y}$$

* $\frac{\partial \mathcal{L}}{\partial z_0} = 0.659001 - 0.0 = \mathbf{+0.659001}$ (Hatalı 0. sınıfı aşağı bastırır)
* $\frac{\partial \mathcal{L}}{\partial z_1} = 0.242433 - 1.0 = \mathbf{-0.757567}$ (Doğru 1. sınıfı güçlü şekilde yukarı çeker)
* $\frac{\partial \mathcal{L}}{\partial z_2} = 0.098566 - 0.0 = \mathbf{+0.098566}$ (Hatalı 2. sınıfı aşağı bastırır)

Gradyanlar toplamı: $0.659001 - 0.757567 + 0.098566 = \mathbf{0.000000}$.

---

### Adım 3: Aynı Olasılıklar Üzerinde Ortalama Karesel Hata (MSE)

Aynı Softmax olasılıkları üzerinde MSE kaybını hesaplayalım:

$$\mathcal{L}_{\text{MSE}} = \frac{1}{3} \sum_{k=0}^2 (p_k - y_k)^2$$

* $(p_0 - 0)^2 = (0.659001)^2 = 0.434282$
* $(p_1 - 1)^2 = (0.242433 - 1)^2 = (-0.757567)^2 = 0.573908$
* $(p_2 - 0)^2 = (0.098566)^2 = 0.009715$
* Hata kareleri toplamı $= 0.434282 + 0.573908 + 0.009715 = 1.017905$

$$\mathcal{L}_{\text{MSE}} = \frac{1.017905}{3} = \mathbf{0.339302}$$

MSE'nin doğru sınıf logiti $z_1$'e göre gradyanı:

$$\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z_1} = \frac{2}{3} \sum_{k=0}^2 (p_k - y_k) p_k (\delta_{1k} - p_1) = \mathbf{-0.164516}$$

Doğru sınıf için düzeltici gradyanları karşılaştıralım:
* Cross-Entropy gradyanı ($z_1$): $\mathbf{-0.757567}$
* MSE gradyanı ($z_1$): $\mathbf{-0.164516}$

$$\text{Gradyan Oranı} = \frac{|-0.757567|}{|-0.164516|} \approx \mathbf{4.6048\times}$$

Cross-Entropy, tamamen aynı hata durumunda MSE'ye göre **4.6 kat daha güçlü bir düzeltici gradyan** uygular ve türev doymasına uğramaz.

```text
Doğru Sınıf (İndeks 1) İçin Gradyan Karşılaştırması:
  Cross-Entropy: -0.757567  <-- Doğrudan artık hatayla orantılı, güçlü güncelleme.
  MSE:           -0.164516  <-- Softmax türeviyle zayıflatılmış, cılız güncelleme.
```

---

### Adım 4: InfoNCE Kontrastif Sayısal Yürüyüşü

Bir sorgu $q$, bir pozitif anahtar $k^+$ ve iki negatif çeldirici $k_1^-, k_2^-$ alalım. Benzerlikler ve sıcaklık $\tau = 0.5$:

$$\text{Benzerlikler} = \begin{bmatrix} 0.8 & 0.2 & 0.1 \end{bmatrix}$$

Sıcaklığa bölelim:

$$s = \frac{\text{sim}}{\tau} = \begin{bmatrix} \frac{0.8}{0.5} & \frac{0.2}{0.5} & \frac{0.1}{0.5} \end{bmatrix} = \begin{bmatrix} 1.6 & 0.4 & 0.2 \end{bmatrix}$$

Softmax olasılıkları:
* $c = \max(s) = 1.6 \implies s - c = [0.0, -1.2, -1.4]$
* $e^{0.0} = 1.000000, \quad e^{-1.2} = 0.301194, \quad e^{-1.4} = 0.246597$
* Toplam $= 1.000000 + 0.301194 + 0.246597 = 1.547791$
* $p_{\text{pozitif}} = \frac{1.0}{1.547791} = \mathbf{0.646082}$
* $p_{\text{neg1}} = \frac{0.301194}{1.547791} = \mathbf{0.194596}, \quad p_{\text{neg2}} = \frac{0.246597}{1.547791} = \mathbf{0.159322}$

InfoNCE Kaybı:

$$\mathcal{L}_{\text{InfoNCE}} = -\ln(p_{\text{pozitif}}) = -\ln(0.646082) = \mathbf{0.436829}$$

| Boyut | Cross-Entropy | Ortalama Karesel Hata (MSE) | InfoNCE Kontrastif Kayıp |
| :--- | :--- | :--- | :--- |
| **Girdi Temsili** | Logitler $z = [2.0, 1.0, 0.1]$ | Logitler $z = [2.0, 1.0, 0.1]$ | $\tau=0.5$ ile ölçeklenmiş kosinüs benzerlikleri |
| **Hedef Dağılım** | $y = [0.0, 1.0, 0.0]$ | $y = [0.0, 1.0, 0.0]$ | Aday kümesindeki 0. pozitif indeks |
| **Kayıp Değeri** | $\mathbf{1.417031}$ | $\mathbf{0.339302}$ | $\mathbf{0.436829}$ |
| **Hedefteki Gradyan** | $\mathbf{-0.757567}$ | $\mathbf{-0.164516}$ | $\mathbf{-0.353918}$ ($p_0 - 1$) |
| **Optimizasyon Davranışı** | Dik, doğrusal düzeltme | $4.6\times$ sönümlenmiş; platoya açık | Pozitifi çeker, negatifleri iter |

---

## 7. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri

Kayıp fonksiyonları üretim modellerine uyarlandığında matematiksel türevlerden GPU bellek hiyerarşisi kısıtlarına geçiş yapılır.

| Mühendislik Boyutu | Eğitim Rejimi (Ön-Eğitim / SFT) | Çıkarım Rejimi (Sunum / Üretim) |
| :--- | :--- | :--- |
| **Kayıp Hesabı** | **Temel Amaç:** Kaynaşmış Log-Sum-Exp Cross-Entropy | **Yok:** Kayıp fonksiyonu tamamen devre dışıdır |
| **Bellek Ayak İzi** | Devasa: Sözlük logit tensörü ($B \times S \times V$) | Minimal: Yalnızca anlık üretilen token logiti ($1 \times V$) |
| **Aritmetik Yoğunluk** | Uzun dizilerde hesaplama kısıtlı (compute-bound) | Decode sırasında bellek bant genişliği kısıtlı |
| **Hassasiyet Stratejisi** | LSE toplamlarında Bfloat16 / Float32 birikimi | FP16, BF16 veya FP8/INT4 projeksiyon |
| **Donanım Darboğazı** | Logit tensörünün HBM VRAM kapasitesini tüketmesi | Ağırlıkların HBM'den SRAM'e taşınma hızı |

### Eğitim Gözü: Sözlük Bellek Duvarı ($B \times S \times V$)

Modern otoregresif dil modellerinde çıktı katmanı, modelin gizli durumunu ($d_{\text{model}}$) sözlük boyutu ($V$) kadar logite yansıtır:
* **LLaMA 3:** $V = 128{,}256$
* **Gemma 2 / Gemma 3:** $V = 256{,}000$

Standart bir eğitim mini-batch'ini ele alalım:
* Batch boyutu $B = 4$
* Dizi uzunluğu $S = 4096$
* Sözlük boyutu $V = 128{,}000$
* Sayısal format: 32-bit float (4 bayt)

Ham logit tensörünün bellek boyutu:

$$\text{Bellek} = 4 \times 4096 \times 128{,}000 \times 4 \text{ bayt} \approx \mathbf{8.39\text{ GB}}$$

Yalnızca tek bir eğitim adımında tek bir GPU üzerinde logitleri belleğe açmak 8.3 GB'tan fazla VRAM gerektirir. Bu tensör HBM belleğe yazılır, Softmax için tekrar okunur ve Cross-Entropy için bir kez daha okunursa bellek bant genişliği kilitlenir ve Out-Of-Memory (OOM) arızası kaçınılmaz olur.

### Mühendislik Çözümü: Parçalı ve Kaynaşmış Cross-Entropy (Chunked Loss)

Bu bellek duvarını aşmak için Megatron-LM ve modern LLM motorları **Parçalı Cross-Entropy (Chunked Cross-Entropy)** tekniğini kullanır:
1. $S = 4096$ token'lık dizinin tamamı bir kerede sözlük logitlerine dönüştürülmez; dizi küçük parçalara bölünür (örneğin $C = 512$).
2. Kaynaşmış bir CUDA çekirdeği doğrusal projeksiyonu ($h W_v$) hesaplar, Log-Sum-Exp indirgemesini yapar, kaybı çıkarır ve geri yayılım gradyanını doğrudan çip içi hızlı SRAM üzerinde toplar.
3. 8.4 GB'lık devasa logit tensörü **GPU HBM belleğinde hiçbir zaman tam olarak oluşturulmaz**; böylece kayıp hesaplama belleği %90 oranında düşürülür.

### Çıkarım Gözü: Kaybın Tamamen Devre Dışı Kalması

Çıkarım (metin üretimi) sırasında kayıp fonksiyonu tamamen çöpe atılır. Model asla Cross-Entropy hesaplamaz.

Son katman yalnızca üretilmekte olan anlık token ($N = 1$) için $V$ adet logit üretir:
1. Logitler isteğe bağlı olarak sıcaklık parametresine ($T$) bölünür: $z_i' = z_i / T$.
2. Filtreleme algoritmaları işletilir (Top-$K$, Top-$P$ / Nucleus örnekleme).
3. Yalnızca elenen aday kümesi üzerinde Softmax hesaplanır veya `argmax` ile en yüksek logite sahip token doğrudan seçilerek çıktı verilir.

---

## 8. Python ve PyTorch ile doğrulama

Aşağıdaki bağımsız Python scripti temel matematik kütüphanesini kullanarak tüm kayıp fonksiyonlarını ve türevlerini sıfırdan hesaplamakta, Log-Sum-Exp sayısal kararlılığını doğrulamakta ve yerel PyTorch fonksiyonlarıyla tam sayısal denklik sağlamaktadır:

```python
import math
import torch
import torch.nn.functional as F

# 1. Sıfırdan temel matematiksel fonksiyonlar
def softmax(logits):
    c = max(logits)
    exp_shifted = [math.exp(z - c) for z in logits]
    sum_exp = sum(exp_shifted)
    return [e / sum_exp for e in exp_shifted]

def log_sum_exp(logits):
    c = max(logits)
    sum_exp = sum(math.exp(z - c) for z in logits)
    return c + math.log(sum_exp)

def cross_entropy_loss(logits, target_idx):
    lse = log_sum_exp(logits)
    return lse - logits[target_idx]

def ce_gradient(logits, target_idx):
    p = softmax(logits)
    grad = list(p)
    grad[target_idx] -= 1.0
    return grad

def mse_loss_and_grad(logits, target_idx):
    K = len(logits)
    p = softmax(logits)
    y = [1.0 if i == target_idx else 0.0 for i in range(K)]
    diff = [p[i] - y[i] for i in range(K)]
    mse = sum(d**2 for d in diff) / K

    grad = []
    for i in range(K):
        g_i = 0.0
        for k in range(K):
            jacobian = p[k] * ((1.0 if k == i else 0.0) - p[i])
            g_i += diff[k] * jacobian
        grad.append((2.0 / K) * g_i)
    return mse, grad

# Sayısal doğrulama kurgusu
logits = [2.0, 1.0, 0.1]
target_idx = 1 # Gerçek etiket: sınıf 1 (y = [0, 1, 0])

print("=== 1. OYUNCAK LOGİTLER VE SOFTMAX OLASILIKLARI ===")
c = max(logits)
print(f"Ham Logitler z:             {logits}")
print(f"Maksimum Logit c:           {c:.6f}")
shifted = [z - c for z in logits]
print(f"Kaydırılmış Logitler (z-c): {[round(s, 6) for s in shifted]}")
exp_shifted = [math.exp(s) for s in shifted]
print(f"exp(z-c):                   {[round(e, 6) for e in exp_shifted]}")
sum_exp = sum(exp_shifted)
print(f"sum(exp(z-c)):              {sum_exp:.6f}")
lse = log_sum_exp(logits)
print(f"Log-Sum-Exp LSE(z):         {lse:.6f}")

probs = softmax(logits)
print(f"Softmax Olasılıkları p:     {[round(p, 6) for p in probs]}")

print("\n=== 2. CROSS-ENTROPY KAYBI VE GRADYANI ===")
ce_loss = cross_entropy_loss(logits, target_idx)
print(f"Cross-Entropy Kaybı:        {ce_loss:.6f}")
ce_grad = ce_gradient(logits, target_idx)
print(f"CE Gradyanı (p - y):        {[round(g, 6) for g in ce_grad]}")
print(f"CE Gradyanları Toplamı:     {sum(ce_grad):.10f}")

print("\n=== 3. MSE KAYBI VE GRADYAN KARŞILAŞTIRMASI ===")
mse_val, mse_grad = mse_loss_and_grad(logits, target_idx)
print(f"MSE Kaybı:                  {mse_val:.6f}")
print(f"MSE Gradyanı:               {[round(g, 6) for g in mse_grad]}")
print(f"Gradyan Oranı (CE / MSE doğru sınıf için): {abs(ce_grad[target_idx]) / abs(mse_grad[target_idx]):.4f}x")

print("\n=== 4. INFONCE KONTRASTİF KAYBI ===")
sims = [0.8, 0.2, 0.1]
tau = 0.5
scaled_sims = [s / tau for s in sims]
print(f"Ham Benzerlikler:           {sims}")
print(f"Sıcaklık tau:               {tau}")
print(f"Ölçeklenmiş Logitler:       {[round(s, 6) for s in scaled_sims]}")
infonce_lse = log_sum_exp(scaled_sims)
infonce_loss = infonce_lse - scaled_sims[0]
infonce_probs = softmax(scaled_sims)
print(f"InfoNCE Softmax P:          {[round(p, 6) for p in infonce_probs]}")
print(f"InfoNCE Kaybı:              {infonce_loss:.6f}")

# PyTorch doğrulama testi
z_tensor = torch.tensor([logits], dtype=torch.float32, requires_grad=True)
target_tensor = torch.tensor([target_idx], dtype=torch.long)
torch_ce = F.cross_entropy(z_tensor, target_tensor)
torch_ce.backward()

print(f"\nPyTorch Cross-Entropy Kaybı: {torch_ce.item():.6f}")
print(f"PyTorch Autograd Gradyanı:  {z_tensor.grad.numpy().round(6).tolist()[0]}")
```

Script koşturulduğunda analitik değerlerle birebir örtüşen konsol çıktısı elde edilir:

```text
=== 1. OYUNCAK LOGİTLER VE SOFTMAX OLASILIKLARI ===
Ham Logitler z:             [2.0, 1.0, 0.1]
Maksimum Logit c:           2.000000
Kaydırılmış Logitler (z-c): [0.0, -1.0, -1.9]
exp(z-c):                   [1.0, 0.367879, 0.149569]
sum(exp(z-c)):              1.517448
Log-Sum-Exp LSE(z):         2.417030
Softmax Olasılıkları p:     [0.659001, 0.242433, 0.098566]

=== 2. CROSS-ENTROPY KAYBI VE GRADYANI ===
Cross-Entropy Kaybı:        1.417030
CE Gradyanı (p - y):        [0.659001, -0.757567, 0.098566]
CE Gradyanları Toplamı:     0.0000000000

=== 3. MSE KAYBI VE GRADYAN KARŞILAŞTIRMASI ===
MSE Kaybı:                  0.339302
MSE Gradyanı:               [0.175146, -0.164516, -0.01063]
Gradyan Oranı (CE / MSE doğru sınıf için): 4.6048x

=== 4. INFONCE KONTRASTİF KAYBI ===
Ham Benzerlikler:           [0.8, 0.2, 0.1]
Sıcaklık tau:               0.5
Ölçeklenmiş Logitler:       [1.6, 0.4, 0.2]
InfoNCE Softmax P:          [0.646082, 0.194596, 0.159322]
InfoNCE Kaybı:              0.436829

PyTorch Cross-Entropy Kaybı: 1.417031
PyTorch Autograd Gradyanı:  [0.659001, -0.757567, 0.098566]
```

---

## Bütün hikâye altı satırda

- Kategorik olasılıklar üzerinde Ortalama Karesel Hata (MSE) doymaya uğrayarak vahim hatalarda gradyanı sıfırlar.
- Cross-Entropy bilgi teorisinden türer ve model dağılımını gerçekliğe yaklaştırmanın En Çok Olabilirlik Kestirimi (MLE) olduğunu kanıtlar.
- Softmax ve Cross-Entropy birleştiğinde pay ve paydadaki terimler sadeleşir; gradyan saf $\hat{y} - y$ artık farkına çöker.
- Float16 donanımlarında ham üstel hesaplama $z > 11.09$ iken taşar; Log-Sum-Exp hilesi `NaN` çöküşlerini mutlak olarak engeller.
- Label Smoothing logitlerin sonsuza ıraksamasını önler; InfoNCE ise kategorik Cross-Entropy'yi sürekli vektör uzayına uyarlar.
- Geniş sözlüklü LLM'lerde ($V \ge 128\text{k}$) parçalı kayıp (chunked cross-entropy) gigabaytlarca VRAM tasarrufu sağlar.

---

## Terimler sözlüğü

- **Öz-Bilgi (Surprisal)** — Bir olayın gerçekleşmesiyle açığa çıkan bilgi miktarı ($-\log P(x)$); olasılık düştükçe üstel olarak büyür.
- **Shannon Entropisi** — Bir olasılık dağılımının taşıdığı ortalama belirsizlik ve teorik minimum kodlama uzunluğu ($-\sum P(x) \log P(x)$).
- **Kullback-Leibler (KL) Uzaklığı** — Gerçek $P$ dağılımı yerine $Q$ modelini kullanmanın getirdiği fazladan kodlama maliyeti ve istatistiksel sapma.
- **Logit** — Çıktı katmanının Softmax öncesinde ürettiği kısıtlanmamış, ham gerçel sayı puanı.
- **Log-Sum-Exp (LSE) Hilesi** — $\max(z) + \log \sum e^{z - \max(z)}$ özdeşliğiyle donanımda kayan noktalı sayı taşmalarını önleyen matematiksel teknik.
- **Label Smoothing** — Modelin aşırı özgüvenini ve ağırlık şişmesini önlemek için katı $0/1$ etiketlerini yumuşatan düzenlileştirici.
- **Focal Loss** — Kolay sınıflandırılan örneklerin kaybını katsayıyla bastırıp zor örneklere odaklanan uyarlanabilir kayıp fonksiyonu.
- **InfoNCE Kaybı** — Pozitif ve negatif adaylar arasındaki kosinüs benzerlikleri üzerinde kategorik Cross-Entropy kuran kontrastif amaç fonksiyonu.

---

## Daha derine inmek için

- [Claude Shannon (Bell System Technical Journal 1948): A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — Entropi, öz-bilgi ve bilgi teorisinin kurucu makalesi.
- [Szegedy et al. (CVPR 2016): Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) — Label smoothing düzenlileştirmesini literatüre kazandıran çalışma.
- [Lin et al. (ICCV 2017): Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002) — Dengesiz veri setleri için Focal Loss'u tanıtan seminal makale.
- [van den Oord, Li, & Vinyals (arXiv:1807.03748, 2018): Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) — InfoNCE ve modern kontrastif öğrenmenin matematiksel temeli.
- Bu blogda: [Sinir Ağlarının Yapı Taşları (1): Aktivasyon Fonksiyonları](post.html?slug=aktivasyon-fonksiyonlari-derinlemesine) — Doğrusalsızlığın geometrisi, rank çöküşü ve kayıp fonksiyonuna giren logitlerin nasıl üretildiği.
