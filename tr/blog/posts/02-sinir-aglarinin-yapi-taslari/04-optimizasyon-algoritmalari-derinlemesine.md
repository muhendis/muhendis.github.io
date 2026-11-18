100 katmandan oluşan bir Transformer modelini klasik Stokastik Gradyan İnişi (SGD) ile eğitmeye çalıştığınızı hayal edin. Parametreleri başlattınız, ilk eğitim batch'ini verdiniz ve $\eta = 0.01$ gibi gayet makul görünen bir öğrenme oranı seçtiniz. Yalnızca yirmi adım sonra kayıp değeri aniden patlayarak `NaN` verir. Müdahale edip öğrenme oranını yüz kat düşürür ve $\eta = 0.0001$ yaparsınız. Bu kez patlama durur; ancak eğitim neredeyse tamamen donar: koca bir hafta sonu boyunca GPU'lar çalışmasına rağmen kayıp değeri yerinden oynamaz.

Basit gradyan inişi neden derin ağlarda asla makul bir orta yol bulamaz?

Bu başarısızlığın kökeni, **yüksek boyutlu kayıp yüzeylerinin anizotropik (yöne bağımlı) geometrisinde** yatar. Derin yapay sinir ağları pürüzsüz ve simetrik çanaklarda değil, son derece dik ve dar **kanyonlarda (ravines)** optimize edilir. Kanyonun enine yamaçları boyunca eğrilik vahşice diktir: Hessian matrisinin en büyük özdeğeri $\lambda_{\max}$ devasadır. Küresel minimuma doğru uzanan kanyon tabanı boyunca ise yüzey neredeyse dümdüzdür ($\lambda_{\min} \approx 0$). Bu dengesiz rejimde kanyon tabanında ilerlemeye yetecek büyüklükteki her öğrenme oranı, SGD'nin iki yamaç arasında kontrolsüzce savrulmasına ve patlamasına yol açar.

Optimizasyon algoritmaları, gradyan inişini bu çetin arazide yönlendiren seyrüsefer sistemleridir. Onlarca yıl boyunca eğitim, momentumun fiziksel analojilerine ve Adam/AdamW'nin skaler koordinat bazlı momentlerine dayandı. Bugün ise büyük dil modellerinin (LLM) ön-eğitimi, son on yılın en büyük optimizasyon devrimiyle sarsılıyor: ağırlık matrislerini bağımsız skaler listeleri olarak değil, Newton-Schulz yinelemeleriyle ortogonalize edilen 2 boyutlu doğrusal operatörler olarak ele alan **Muon**. Bu yazı, dengesiz vadi patolojisinin geometrisini inceliyor; AdamW'nin L2 regülarizasyon paradoksunu nasıl çözdüğünü ortaya koyuyor; Muon algoritmasının spektral mekaniğini analiz ediyor ve elle yapılan sayısal tensör yürüyüşünü doğrulanabilir kodlarla sunuyor.

**Bu yazıda**

- [1. En yalın hâliyle temel sezgi: Vadi patolojisi ve gradyanlar neden salınır?](#1-en-yalın-hâliyle-temel-sezgi-vadi-patolojisi-ve-gradyanlar-neden-salınır)
- [2. Fiziksel momentumdan adaptif momentlere: Polyak, Nesterov ve RMSprop](#2-fiziksel-momentumdan-adaptif-momentlere-polyak-nesterov-ve-rmsprop)
- [3. AdamW devrimi: Ayrıştırılmış ağırlık sönümlemesi vs L2 regülarizasyonu](#3-adamw-devrimi-ayrıştırılmış-ağırlık-sönümlemesi-vs-l2-regülarizasyonu)
- [4. 2024-2026 ön-eğitim devrimi: Muon optimizasyon algoritması](#4-2024-2026-ön-eğitim-devrimi-muon-optimizasyon-algoritması)
- [5. Öğrenme oranı dinamikleri: Warmup, Cosine Annealing ve WSD çizelgeleri](#5-öğrenme-oranı-dinamikleri-warmup-cosine-annealing-ve-wsd-çizelgeleri)
- [6. Adım adım elle sayısal tensör yürüyüşü](#6-adım-adım-elle-sayısal-tensör-yürüyüşü)
- [7. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri](#7-iki-mühendislik-gözü-eğitim-ve-çıkarım-dinamikleri)
- [8. Python ve PyTorch ile doğrulama](#8-python-ve-pytorch-ile-doğrulama)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. En yalın hâliyle temel sezgi: Vadi patolojisi ve gradyanlar neden salınır?

### Kötü Koşullu Kanyon (The Ill-Conditioned Ravine)

Basit gradyan inişinin derin ağlarda neden iflas ettiğini anlamak için iki boyutlu basık bir kuadratik yüzeyi ele alalım:

$$f(w_1, w_2) = \frac{1}{2} w_1^2 + \frac{L}{2} w_2^2 \quad (L \gg 1)$$

Herhangi bir $(w_1, w_2)$ noktasındaki gradyan vektörü:

$$\nabla f(w) = \begin{bmatrix} w_1 \\ L w_2 \end{bmatrix}$$

Bu yüzeyin eğriliği Hessian matrisi ile tanımlanır:

$$\mathbf{H} = \begin{bmatrix} 1 & 0 \\ 0 & L \end{bmatrix}$$

Hessian matrisinin **koşul sayısı (condition number)** $\kappa$, en büyük özdeğerin en küçük özdeğere oranıdır:

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{L}{1} = L$$

Derin yapay sinir ağlarında koşul sayısı sıklıkla $\kappa \ge 10^4 - 10^6$ seviyelerine fırlar.

<svg viewBox="0 0 560 330" role="img" aria-label="Kötü koşullu vadi çıkmazı, koşul sayısı kappa 10 üzeri 4&amp;#x27;ün çok üstünde. Yamaç doğrultusunda (lambda max) eğrilik diktir ve düz SGD şiddetli enine salınımlarla sekip durur. Vadi tabanı doğrultusunda (lambda min) eğrilik düzdür ve ileri ilerleme neredeyse sıfırdır. SGD sıkışmıştır: Adım boyu 2 bölü lambda max&amp;#x27;ın üstündeyse patlar, altındaysa tabanda sürünür. Üç algoritmik çözüm: Momentum enine salınımları sönümler (Σ ±g ≈ 0) ve taban boyunca hız biriktirir; AdamW koordinat bazlı RMS ile adım boyunu eşitler; Muon tüm 2D momentum matrisini Newton-Schulz ile ortogonalize eder." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="rv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="rv-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
</defs>
<text x="16" y="20" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">Kötü Koşullu Vadi Çıkmazı (κ ≫ 10⁴)</text>
<ellipse cx="230" cy="100" rx="200" ry="62" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<ellipse cx="230" cy="100" rx="140" ry="43" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<ellipse cx="230" cy="100" rx="80" ry="25" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<ellipse cx="230" cy="100" rx="24" ry="7.5" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<path d="M60 48 L73 150 L86 53 L99 144 L112 58 L125 140 L138 63 L151 134 L164 68" marker-end="url(#rv-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.6"/>
<circle cx="230" cy="100" r="3" style="fill:var(--c-text)"/>
<text x="238" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">minimum</text>
<path d="M448 60 V140" marker-start="url(#rv-arr)" marker-end="url(#rv-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="458" y="84" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">yamaç · λ_max</text>
<text x="458" y="99" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">dik eğrilik:</text>
<text x="458" y="113" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">şiddetli salınımlar</text>
<path d="M150 178 H310" marker-start="url(#rv-arr)" marker-end="url(#rv-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="230" y="196" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">taban · λ_min · düz eğrilik: neredeyse sıfır ilerleme</text>
<rect x="16" y="206" width="528" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280" y="225.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px">SGD: η &gt; 2/λ_max → patlar;  η &lt; 2/λ_max → taban boyunca sürünür</text>
<text x="16" y="258" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">Algoritmik çözümler</text>
<rect x="16" y="268" width="168" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="100.0" y="284.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Momentum</text>
<text x="100.0" y="300.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">salınımları sönümler</text>
<text x="100.0" y="316.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">(Σ ±g ≈ 0), hız biriktirir</text>
<rect x="196" y="268" width="168" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="284.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">AdamW</text>
<text x="280.0" y="300.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">adım boyunu eşitler</text>
<text x="280.0" y="316.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">koordinat bazlı RMS</text>
<rect x="376" y="268" width="168" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="460.0" y="284.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Muon</text>
<text x="460.0" y="300.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">2D matrisi ortogonal</text>
<text x="460.0" y="316.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">yapar (Newton–Schulz)</text>
</svg>

### Klasik SGD'nin Kararlılık Tavanı

Öğrenme oranı $\eta$ olan standart gradyan inişinde parametre güncellemeleri şu şekilde evrilir:

$$w_1^{(t+1)} = (1 - \eta) w_1^{(t)}, \quad w_2^{(t+1)} = (1 - \eta L) w_2^{(t)}$$

Yinelemenin sayısal olarak kararlı kalabilmesi ve yakınsayabilmesi için her eksendeki daralma katsayısı $|1 - \eta \lambda_i| < 1$ şartını sağlamalıdır:

$$\eta < \frac{2}{\lambda_{\max}} = \frac{2}{L}$$

Mühendis $\eta \ge \frac{2}{L}$ seçtiği anda $w_2$ yönündeki güncellemeler üstel olarak büyür: $w_2^{(t)} \to \pm \infty$ (eğitim patlar).

Ancak mühendis yamaç patlamalarını önlemek için güvenli tarafta kalıp $\eta = \frac{1}{L}$ seçtiğinde, kanyon tabanı boyunca $w_1$ koordinatının ilerleyişine bakalım:

$$w_1^{(t+1)} = \left( 1 - \frac{1}{L} \right) w_1^{(t)}$$

$L = 10{,}000$ olduğunda, 1.000 tam eğitim adımından sonra model kanyon tabanında katettiği mesafe:

$$\left( 1 - 10^{-4} \right)^{1000} \approx 0.9048$$

Model minimuma giden yolun yalnızca %9.5'ini katedebilmiştir. Ağ, parametre uzayındaki en dik yönün esiri olarak kanyon tabanında felç olmuştur.

---

## 2. Fiziksel momentumdan adaptif momentlere: Polyak, Nesterov ve RMSprop

Vadi patolojisini aşmak için optimizasyon teorisi üç büyük aşamadan geçti: fiziksel momentum, adaptif koordinat ölçekleme ve moment kestirimi.

### Polyak Momentumu (Ağır Top Yöntemi)

1964 yılında Sovyet matematikçi Boris Polyak, optimizasyonu bir potansiyel kuyusunda yuvarlanan kütleli bir parçacık olarak modellemeyi önerdi:

$$v_t = \beta v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \eta v_t$$

Burada $v_t$ hız vektörü, $\beta \in [0, 1)$ ise sürtünme katsayısıdır (genellikle $\beta = 0.9$).

Hız vektörünün geçmişe doğru açılımı Üstel Ağırlıklı Hareketli Ortalama (EWMA) yapısını ortaya koyar:

$$v_t = \sum_{\tau=0}^{t} \beta^{t-\tau} g_\tau$$

Geometrik olarak bu kurgu muazzam bir güç sağlar:
* **Kanyon yamaçları boyunca ($w_2$):** Gradyanlar ardışık adımlarda zıt işaretler alır ($+L, -L, +L, -L$). Hız toplamında bu zıt vektörler birbirini sönümler: $\sum \pm g \approx 0$.
* **Kanyon tabanı boyunca ($w_1$):** Gradyanlar sürekli aynı yöne bakar. Hızlar birbirinin üzerine eklenerek birikir ve etkin bir hızlanma çarpanı üretir:

$$\text{Etkin Hızlanma} = \frac{1}{1 - \beta} = \frac{1}{1 - 0.9} = \mathbf{10\times}$$

Momentum, yüksek frekanslı yanal salınımları sönümlerken kanyon tabanındaki istikrarlı ilerlemeyi 10 kat hızlandırır.

### Nesterov Hızlandırılmış Gradyan (NAG)

1983 yılında Yurii Nesterov, gradyanı bulunulan noktada değil, **parçacığın mevcut momentumuyla taşınacağı öngörülen noktada** hesaplayarak Polyak'ın yöntemini geliştirdi:

$$g_t = \nabla f(\theta_{t-1} - \eta \beta v_{t-1})$$

$$v_t = \beta v_{t-1} + g_t$$

$$\theta_t = \theta_{t-1} - \eta v_t$$

NAG, topun momentumunun onu götüreceği yerdeki eğimi önceden koklayarak eğimin tekrar dikleşmeye başladığı virajlarda öngörülü bir frenleme (predictive braking) uygular ve savrulmaları önler.

### Adaptif Koordinat Dönemi: AdaGrad ve RMSprop

Momentum kararlı yönleri hızlandırsa da tüm koordinatlara tek bir skaler öğrenme oranı uygular.

2011 yılında John Duchi ve ekibi **AdaGrad**'ı geliştirerek her parametreyi geçmiş gradyan karelerinin toplamının kareköküyle ölçeklendirdi:

$$G_t = G_{t-1} + g_t^2, \quad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon} \odot g_t$$

AdaGrad seyrek özelliklerin daha büyük adımlar atmasını sağladı. Ancak $G_t = \sum_{\tau=1}^t g_\tau^2$ toplamı sürekli büyüdüğü için $G_t \to \infty$ olur ve etkin öğrenme oranı $\frac{\eta}{\sqrt{G_t}} \to 0$ değerine çöker. Derin ağlarda AdaGrad yakınsamadan çok önce donar.

2012 yılında Geoffrey Hinton, Coursera derslerinde monoton toplamı Üstel Hareketli Ortalama (EMA) ile değiştirerek **RMSprop** algoritmasını sundu:

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon} \odot g_t$$

$\beta_2 = 0.99$ seçildiğinde RMSprop yalnızca yakın geçmişin gradyan büyüklüklerini hatırlar ve paydanın sonsuza ıraksayıp eğitimi durdurmasını engeller.

### Adam: Uyarlanabilir Moment Kestirimi

2014 yılında Diederik Kingma ve Jimmy Ba, momentum ile RMSprop'u birleştirerek birinci momenti (ortalama hız) ve ikinci momenti (merkezlenmemiş varyans) aynı anda takip eden **Adam**'ı tanıttı:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\text{Birinci Moment: Momentum})$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\text{İkinci Moment: RMSprop})$$

#### Başlangıç Sapması Düzeltmesinin (Bias Correction) İspatı
Momentler sıfır olarak başlatıldığından ($m_0 = \mathbf{0}, v_0 = \mathbf{0}$), ilk adımlarda değerler sıfıra doğru aşırı yanlıdır:
$t = 1$ anında: $m_1 = (1 - \beta_1) g_1 = (1 - 0.9) g_1 = \mathbf{0.1 g_1}$.

$m_t$ ifadesini geçmişe doğru açalım:

$$m_t = (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i$$

Gradyanların durağan bir dağılımdan geldiğini ve $\mathbb{E}[g_i] = \mathbb{E}[g_t]$ olduğunu varsayarak beklenti alalım:

$$\mathbb{E}[m_t] = \mathbb{E}\left[ (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i \right] = \mathbb{E}[g_t] (1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} = \mathbb{E}[g_t] (1 - \beta_1) \frac{1 - \beta_1^t}{1 - \beta_1} = \mathbb{E}[g_t] \left( 1 - \beta_1^t \right)$$

Her iki tarafı $1 - \beta_1^t$ terimine böldüğümüzde yansız bir kestirici elde edilir:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Nihai Adam güncelleme kuralı:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 3. AdamW devrimi: Ayrıştırılmış ağırlık sönümlemesi vs L2 regülarizasyonu

Adam'ın yayınlanmasını takip eden yıllarda araştırmacılar garip bir ampirik gerçekle karşılaştı: **Klasik momentumlu SGD, ImageNet ve dil modellerinde Adam'dan sürekli daha iyi genelleme (generalization) başarımları elde ediyordu.**

Uzun süre Adam'ın teorik bir zayıflığı olduğu düşünüldü. 2017 yılında Ilya Loshchilov ve Frank Hutter bu gizemi çözdü: **L2 regülarizasyonu ile Ağırlık Sönümlemesinin (Weight Decay) ölümcül bir şekilde birbirine karıştırılması.**

### SGD'de İki Kavramın Eşitliği

Klasik SGD'de kayıp fonksiyonuna bir $L_2$ ağırlık cezası $\frac{\lambda}{2} \|\theta\|_2^2$ eklendiğinde:

$$\nabla_{\theta} \left( \mathcal{L}(\theta) + \frac{\lambda}{2} \|\theta\|_2^2 \right) = g_t + \lambda \theta_{t-1}$$

Güncelleme denklemi:

$$\theta_t = \theta_{t-1} - \eta (g_t + \lambda \theta_{t-1}) = (1 - \eta \lambda) \theta_{t-1} - \eta g_t$$

Görüldüğü üzere standart SGD'de $L_2$ regülarizasyonu matematiksel olarak **Ağırlık Sönümlemesi (Weight Decay)** ile birebir özdeştir (her adımda ağırlıklar $1 - \eta \lambda$ katsayısıyla küçültülür).

### Adam'da L2 Regülarizasyonu Neden Bozulur?

Kütüphaneler $L_2$ regülarizasyonunu Adam'a uyarlarken bu ceza terimini saf gradyana ekleyip optimizere verdiler:

$$\tilde{g}_t = g_t + \lambda \theta_{t-1}$$

Şimdi bu terimin Adam'ın momentlerine nasıl girdiğine bakalım:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) (g_t + \lambda \theta_{t-1})$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (g_t + \lambda \theta_{t-1})^2$$

Açılmış güncelleme denklemi:

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \left( \hat{m}_t + \frac{\lambda \theta_{t-1}}{1 - \beta_1^t} \right)$$

Buradaki düzenlileştirici terime dikkat edin: **Ağırlık cezası $\sqrt{\hat{v}_t}$ ile bölünmektedir!**

Bu durum sistemi tamamen tersine çalıştırır:
1. **Sık Güncellenen / Büyük Gradyanlı Parametreler:** Büyük $\hat{v}_t$ değerine sahiptir. Ağırlık cezası devasa bir paydaya bölündüğü için **sönümleme etkisi neredeyse sıfırlanır**.
2. **Seyrek / Küçük Gradyanlı Parametreler:** $\hat{v}_t$ değerleri sıfıra yakındır. Ağırlık cezası küçük bir sayıya bölündüğü için **aşırı şiddetli bir şekilde sönümlenir**.

Model sık kullanılan özellikleri düzenlileştirmeyi bırakırken, az görülen seyrek özellikleri acımasızca ezer; bu da genelleme gücünü yerle bir eder.

### AdamW Çözümü: Ayrıştırılmış Ağırlık Sönümlemesi

Loshchilov ve Hutter, ağırlık sönümlemesini gradyan momentlerinden tamamen **ayrıştırarak (decoupled)** orijinal formuna döndürdü:

$$\theta_t = \theta_{t-1} - \eta \lambda \theta_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

<svg viewBox="0 0 560 230" role="img" aria-label="L2 regülarizasyonlu Adam ile AdamW karşılaştırması: L2&amp;#x27;li Adam&amp;#x27;da ağırlık sönümleme terimi lambda çarpı w gradyana eklenir, birinci ve ikinci momentlere girer ve v&amp;#x27;nin kareköküne bölünür; seyrek güncellenen ağırlıklar aşırı sönümlenirken sık güncellenenler sönümlenmez. Ayrıştırılmış sönümlemeli AdamW&amp;#x27;de saf gradyan g momentlerden geçip standart Adam adımına gider; ağırlık sönümleme (eksi eta lambda w) momentleri tamamen atlayarak doğrudan ağırlıklardan çıkarılır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="aw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="8" width="528" height="96" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="29" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">Adam + L2 Regülarizasyonu</text>
<text x="530" y="29" text-anchor="end" style="fill:var(--c-danger);font-size:12px;font-weight:600">sönümleme BOZULUR</text>
<rect x="16" y="120" width="528" height="100" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="141" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">AdamW (Ayrıştırılmış Sönümleme)</text>
<text x="530" y="141" text-anchor="end" style="fill:var(--c-success);font-size:12px;font-weight:600">sönümleme KORUNUR</text>
<rect x="30" y="40" width="150" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="105.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g + λ·w</text>
<rect x="206" y="40" width="150" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="281.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">momentlere girer (m, v)</text>
<rect x="382" y="40" width="148" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="456.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">√v ile bölünür</text>
<line x1="180" y1="57" x2="204" y2="57" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="57" x2="380" y2="57" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="30" y="92" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sönümleme gradyana bağlanır; seyrek koordinatlar aşırı sönümlenir</text>
<rect x="30" y="152" width="150" height="30" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="105.0" y="171.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g</text>
<rect x="206" y="152" width="150" height="30" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="281.0" y="171.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">saf gradyan g</text>
<rect x="382" y="152" width="148" height="30" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="456.0" y="171.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">standart adaptif adım</text>
<line x1="180" y1="167" x2="204" y2="167" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="167" x2="380" y2="167" marker-end="url(#aw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="30" y="190" width="150" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="105.0" y="209.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">−η·λ·w</text>
<rect x="382" y="190" width="148" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="456.0" y="209.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px">w&#x27;den çıkar</text>
<line x1="180" y1="205" x2="380" y2="205" marker-end="url(#aw-arr)" style="stroke:var(--c-success);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="280" y="199" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">momentleri atlar</text>
</svg>

AdamW'de her ağırlık, gradyanının büyüklüğünden bağımsız olarak adil bir şekilde $(1 - \eta \lambda)$ oranıyla sönümlenir. Bu sade düzeltme Adam'ın genelleme kabiliyetini geri kazandırdı ve modern Transformer mimarilerinin (LLaMA, GPT-3, Mistral) evrensel standardı oldu.

---

## 4. 2024-2026 ön-eğitim devrimi: Muon optimizasyon algoritması

AdamW'nin başarısına rağmen derin bir mimari eksiklik varlığını sürdürüyordu: **AdamW, 2 boyutlu ağırlık matrislerini bağımsız 1 boyutlu skaler listeleri olarak ele alır.**

Bir Transformer bloğunda gizli katman ağırlıkları doğrusal operatörlerdir: $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Çok boyutlu uzayda koordinat eksenlerini **tekil değer spektrumlarına (singular value spectrum)** göre döndürür ve ölçekler ($W = U \Sigma V^T$). Matrisleri düzleştirip koordinat bazlı normalize etmek, matrisin rankını, tekil vektörlerini ve spektral geometrisini tamamen çöpe atar.

2024 yılı sonunda Keller Jordan ve ekibi tarafından tanıtılan **Muon (MomentUm Orthogonalized by Newton-Schulz)**, AdamW'den bu yana ön-eğitimdeki en büyük optimizasyon atılımını gerçekleştirdi.

### Temel Sezgi: Spektral Norm Altında En Dik İniş

AdamW'nin yaptığı koordinat bazlı normalizasyon ($\sqrt{\hat{v}_t}$'ye bölme), $\ell_\infty$ normu altında en dik iniş yönünü bulmaktır (güncellemeleri bir hiperküpe hapseder).

Muon daha derin bir geometrik soru sorar: **2 boyutlu bir doğrusal haritalama için spektral norm (operatör normu) altında en dik iniş yönü nedir?**

Matematiksel cevap **en yakın ortogonal matristir**:

$$O = \arg\min_{Q^T Q = \mathbf{I}} \|G - Q\|_F = U V^T$$

Burada $G = U \Sigma V^T$, momentum matrisinin Tekil Değer Ayrışımıdır (SVD).

Ham momentum matrisi $G$ yerine onun ortogonal kutup faktörü $U V^T$ konulduğunda, **tüm tekil değerler $1.0$'a eşitlenir**:

$$\Sigma_{\text{güncellenmiş}} = \text{diag}(1, 1, \dots, 1)$$

* Standart gradyanları domine eden baskın tekil yönler sınırlandırılır.
* İnce ve zengin dilsel temsilleri taşıyan arka plandaki kuyruk yönleri ise birim spektral büyüklüğe yükseltilerek güçlendirilir.

### SVD Olmadan Ortogonalizasyon: Newton-Schulz Yinelemeleri

$4096 \times 4096$ boyutunda bir matrisin tam SVD'sini hesaplamak $O(d^3)$ FLOP tutar ve GPU Tensor Çekirdeklerinde paralel yürütülemez.

Muon bu hesaplama darboğazını yalnızca matris çarpımlarına (GEMM) dayanan **Newton-Schulz yinelemeleri** ile çözer:

1. **Matris Enerjisini Ölçekle:**
   $$X_0 = \frac{G}{\|G\|_F}$$
2. **$K$ Adım Newton-Schulz Yürüt ($K \approx 5$):**
   $$X_{k+1} = \frac{1}{2} X_k \left( 3\mathbf{I} - X_k^T X_k \right)$$

Bu yineleme yalnızca matris çarpımları ve toplamalardan oluştuğu için modern GPU Tensor Çekirdekleri 5 adımı bir milisaniyenin altında, tepe donanım verimiyle tamamlar.

<svg viewBox="0 0 560 270" role="img" aria-label="Muon optimizasyon güncellemesi beş adımda: 2D gizli ağırlıkların ham momentum matrisi G&amp;#x27;den başlanır. Frobenius normuna bölünerek enerji normalizasyonu yapılır: X0 = G / ‖G‖_F. Beş Newton-Schulz yinelemesi çalıştırılır: X_k+1 = 1/2 X_k (3I − X_k^T X_k). Sonuç ortogonal bir güncelleme tensörüdür (X^T X ≈ I, tüm tekil değerler 1&amp;#x27;e eşitlenir). Ağırlıklar eşit spektral adımla güncellenir: W = W − eta X." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="mu-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="8" width="528" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="33" text-anchor="start" style="fill:var(--c-text);font-size:13px">Ham 2D momentum matrisi</text>
<text x="530" y="33" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">G</text>
<line x1="60" y1="48" x2="60" y2="58" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="60" width="528" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="30" y="85" text-anchor="start" style="fill:var(--c-text);font-size:13px">Frobenius normalizasyonu</text>
<text x="530" y="85" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">X₀ = G / ‖G‖_F</text>
<line x1="60" y1="100" x2="60" y2="110" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="112" width="528" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="30" y="137" text-anchor="start" style="fill:var(--c-text);font-size:13px">5× Newton–Schulz yinelemesi</text>
<text x="530" y="137" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">X_k+1 = ½·X_k(3I − X_kᵀX_k)</text>
<line x1="60" y1="152" x2="60" y2="162" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="164" width="528" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="30" y="189" text-anchor="start" style="fill:var(--c-text);font-size:13px">Ortogonal güncelleme tensörü</text>
<text x="530" y="189" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">XᵀX ≈ I (tüm σᵢ ≈ 1)</text>
<line x1="60" y1="204" x2="60" y2="214" marker-end="url(#mu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="216" width="528" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="241" text-anchor="start" style="fill:var(--c-text);font-size:13px">Eşit spektral adım</text>
<text x="530" y="241" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-family:var(--font-mono)">W ← W − η·X</text>
</svg>

### Üretimdeki Hibrit Mimari

Büyük modellerde (örneğin Moonshot AI Kimi serisi ve modern açık kaynak ön-eğitim tarifleri) Muon **hibrit bir yapıda** çalışır:

1. **Muon Motoru:** Tüm 2D dahili matris parametrelerini optimize eder (Attention Q, K, V, Projeksiyon ve SwiGLU Gate, Up, Down katmanları).
2. **AdamW Motoru:** Tüm 1D vektörleri ve matris-dışı parametreleri optimize eder (Token Embedding'leri, RMSNorm kazançları, biaslar ve Unembedding başlığı).

### Somut Mühendislik Kazanımları

* **%25–%35 Daha Hızlı Yakınsama:** Muon ile eğitilen modeller AdamW'ye kıyasla yaklaşık %30 daha az eğitim token'ı ile hedef kayıp değerine ulaşır.
* **VRAM Tasarrufu:** AdamW her parametre için FP32 formatında iki durum saklar (1. moment $m$ ve 2. moment $v$). Muon ise 2D matris katmanlarında **yalnızca 1. momenti (momentum tamponu $G$)** tutar; 2. moment tamponunu tamamen çöpe atarak GPU başına gigabaytlarca VRAM tasarrufu sağlar.

| Optimizasyon Algoritması | Parametre Temsili | Güncelleme Normalizasyonu | Ağırlık Sönümleme | Durum Bellek Ayak İzi | Temel Zayıflık |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SGD + Momentum** | Düzleştirilmiş 1D Skaler | Yok (ham gradyan) | Öğrenme oranıyla kenetli | $1\times$ FP32 tampon ($v$) | Şiddetli vadi salınımları ($\kappa \gg 10^4$) |
| **Adam** | Düzleştirilmiş 1D Skaler | Koordinat bazlı $\sqrt{v_t}$ | L2 bölmesiyle bozulmuş | $2\times$ FP32 tampon ($m, v$) | L2 altında zayıf genelleme |
| **AdamW** | Düzleştirilmiş 1D Skaler | Koordinat bazlı $\sqrt{v_t}$ | Ayrıştırılmış ($1 - \eta \lambda$) | $2\times$ FP32 tampon ($m, v$) | 2D matris geometrisini skaler gibi görür |
| **Muon** | **2D Doğrusal Operatör** | **Spektral Ortogonalizasyon (Newton-Schulz)** | Ayrıştırılmış spektral sönümleme | **$1\times$ FP32 tampon** (2D matrisler için) | Yalnızca 2D matrislere uygulanır (AdamW hibriti şart) |

---

## 5. Öğrenme oranı dinamikleri: Warmup, Cosine Annealing ve WSD çizelgeleri

Güncelleme denklemi hikâyenin yarısıdır; milyarlarca adım boyunca öğrenme oranı skalerinin ($\eta(t)$) nasıl bir yörünge izlediği modelin başarısını belirler.

### Warmup Neden Zorunludur?

$t = 0$ anında model ağırlıkları rastgeledir, gradyanlar gürültülüdür ve optimizer'ın momentleri ($m_0, v_0$) henüz istatistiksel geçmiş biriktirmemiştir. Eğer 0. adımda tepe öğrenme oranı $\eta_{\max}$ uygulanırsa, yüksek varyanslı gradyan patlamaları ağırlıkları bozuk vadilere savurur.

**Öğrenme Oranı Isınması (Warmup)**, ilk %1 ila %5'lik adımda $\eta$'yı 0'dan $\eta_{\max}$ değerine doğrusal olarak yükseltir:

$$\eta(t) = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} \quad (t \le T_{\text{warmup}})$$

Warmup, büyük adımlar atılmadan önce AdamW'nin 2. momentinin ve Muon'un momentum tamponlarının kararlı hale gelmesini sağlar.

### Cosine Annealing

Loshchilov ve Hutter (2016) tarafından önerilen **Kosinüs Sönümlemesi**, öğrenme oranını yarım periyotluk kosinüs dalgasıyla kademeli olarak düşürür:

$$\eta(t) = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos\left( \pi \frac{t - T_{\text{warmup}}}{T_{\text{toplam}} - T_{\text{warmup}}} \right) \right)$$

Kosinüs eğrisi, modelin genelleme başarısı yüksek olan düz ve geniş yerel minimumlara yumuşakça yerleşmesini sağlar.

### Modern Cephe: Warmup-Stable-Decay (WSD)

Kosinüs sönümlemesi başarılı olsa da bir üretim kusuruna sahiptir: **sönümleme eğrisi sabit bir toplam adım sayısına ($T_{\text{toplam}}$) kenetlenmiştir**. Bir model 1 trilyon token için kosinüsle eğitildiğinde eğitimi 2 trilyon token'a uzatmak çizelgeyi bozar.

Modern modeller (MiniCPM, DeepSeek) **Warmup-Stable-Decay (WSD)** çizelgesine geçmiştir:
1. **Warmup Fazı:** $\eta_{\max}$'a doğrusal çıkış (ilk %1-%2).
2. **Sabit (Stable) Faz:** Eğitimin büyük bölümünde (%80-%90) $\eta = \eta_{\max}$ sabit tutulur. Model yeni verileri sürekli öğrenir. Eğitim istenildiği an kesilmeden uzatılabilir.
3. **Sönümleme (Decay) Fazı:** Eğitim bütçesinin sonuna yaklaşıldığında agresif bir %10'luk sönümleme ile öğrenme oranı sıfıra çekilerek nihai model performansı kilitlenir.

---

## 6. Adım adım elle sayısal tensör yürüyüşü

Optimizasyon dinamiklerini elle adım adım hesaplanan bir sayısal örnek üzerinde inceleyelim: 2 boyutlu dengesiz bir vadi ve ardından 2x2'lik bir matrisin 5 adımlı Newton-Schulz ortogonalizasyonu.

### Başlangıç Durumu

Parametreler $w = \begin{bmatrix} w_1 & w_2 \end{bmatrix}$ şu yüzeyi minimize etsin:

$$f(w_1, w_2) = \frac{1}{2} w_1^2 + 10.0 w_2^2 \implies \nabla f(w) = \begin{bmatrix} w_1 \\ 20.0 w_2 \end{bmatrix}$$

Koşul sayısı $\kappa = \frac{20}{1} = \mathbf{20}$.
Başlangıç noktası:

$$w^{(0)} = \begin{bmatrix} 2.0 & 1.0 \end{bmatrix}, \quad g^{(0)} = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}$$

---

### Adım 1: Klasik SGD Güncellemesi ($\eta = 0.05$)

$$w^{(1)} = w^{(0)} - \eta g^{(0)} = \begin{bmatrix} 2.0 - 0.05 \times 2.0 \\ 1.0 - 0.05 \times 20.0 \end{bmatrix} = \begin{bmatrix} 2.0 - 0.10 \\ 1.0 - 1.00 \end{bmatrix} = \begin{bmatrix} \mathbf{1.900000} & \mathbf{0.000000} \end{bmatrix}$$

* $w_2$ ekseninde tek adımda tabana inilmiştir.
* Ancak $w_1$ ekseninde yalnızca $2.0$'dan $1.90$'a (%5 ilerleme) gidilebilmiştir.
* Eğer $\eta = 0.11$ seçilseydi, $w_2^{(1)} = 1.0 - 0.11(20) = -1.20$ olur, karşı yamaca savrulup patlardı.

---

### Adım 2: Momentum Güncellemesi ($\beta = 0.9, \eta = 0.05$)

İlk hız $v_1 = g^{(0)} = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}$.

$$w^{(1)} = w^{(0)} - \eta v_1 = \begin{bmatrix} 2.0 - 0.05(2.0) \\ 1.0 - 0.05(20.0) \end{bmatrix} = \begin{bmatrix} \mathbf{1.900000} & \mathbf{0.000000} \end{bmatrix}$$

2. adımda gradyan $g^{(1)} = \begin{bmatrix} 1.90 & 20.0(0.0) \end{bmatrix} = \begin{bmatrix} 1.90 & 0.0 \end{bmatrix}$.

$$v_2 = 0.9 v_1 + g^{(1)} = 0.9 \begin{bmatrix} 2.0 \\ 20.0 \end{bmatrix} + \begin{bmatrix} 1.90 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 1.80 + 1.90 \\ 18.00 + 0.00 \end{bmatrix} = \begin{bmatrix} \mathbf{3.700000} \\ \mathbf{18.000000} \end{bmatrix}$$

$$w^{(2)} = w^{(1)} - \eta v_2 = \begin{bmatrix} 1.90 - 0.05(3.70) \\ 0.00 - 0.05(18.00) \end{bmatrix} = \begin{bmatrix} \mathbf{1.715000} & \mathbf{-0.900000} \end{bmatrix}$$

Kanyon tabanındaki hız $2.0 \to 3.70$'ye çıkarak adımı iki kat hızlandırmıştır.

---

### Adım 3: Adam vs. AdamW Yürüyüşü ($t=1, \eta=0.1, \lambda=0.01, \beta_1=0.9, \beta_2=0.999$)

Gradyan: $g = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}$.

1. 1. moment: $m_1 = (1 - 0.9) g = \begin{bmatrix} 0.2 & 2.0 \end{bmatrix}$.
2. 2. moment: $v_1 = (1 - 0.999) g^2 = 0.001 \begin{bmatrix} 4.0 & 400.0 \end{bmatrix} = \begin{bmatrix} 0.004 & 0.400 \end{bmatrix}$.
3. $t=1$ anında sapma düzeltmesi:
   $$\hat{m}_1 = \frac{m_1}{1 - 0.9} = \begin{bmatrix} 2.0 & 20.0 \end{bmatrix}, \quad \hat{v}_1 = \frac{v_1}{1 - 0.999} = \begin{bmatrix} 4.0 & 400.0 \end{bmatrix}$$
4. Normalleştirilmiş adım:
   $$\text{adım}_1 = \frac{\hat{m}_{1, 1}}{\sqrt{\hat{v}_{1, 1}} + \epsilon} = \frac{2.0}{\sqrt{4.0}} = \mathbf{1.000000}$$
   $$\text{adım}_2 = \frac{\hat{m}_{1, 2}}{\sqrt{\hat{v}_{1, 2}} + \epsilon} = \frac{20.0}{\sqrt{400.0}} = \mathbf{1.000000}$$

Adam gradyanlar arasındaki 20 katlık uçurumu tamamen nötrleyerek her iki yöne de eşit birim adım boyutu atar.

#### Parametre Güncellemeleri:
* **Adam (Ağırlık Sönümleme Yok):**
  $$w^{(1)} = \begin{bmatrix} 2.0 - 0.1(1.0) & 1.0 - 0.1(1.0) \end{bmatrix} = \begin{bmatrix} \mathbf{1.900000} & \mathbf{0.900000} \end{bmatrix}$$
* **AdamW (Ayrıştırılmış Ağırlık Sönümlemesi $\lambda = 0.01$):**
  $$w_1^{(1)} = 2.0 - \eta \lambda (2.0) - \eta(1.0) = 2.0 - 0.002 - 0.1 = \mathbf{1.898000}$$
  $$w_2^{(1)} = 1.0 - \eta \lambda (1.0) - \eta(1.0) = 1.0 - 0.001 - 0.1 = \mathbf{0.899000}$$

---

### Adım 4: Muon Newton-Schulz Matris Ortogonalizasyonu

$2 \times 2$'lik bir momentum matrisini ele alalım:

$$G = \begin{bmatrix} 2.0 & 1.0 \\ 0.5 & 3.0 \end{bmatrix}$$

#### Adım 4.1: Frobenius Normu ile Başlatma
$$\|G\|_F = \sqrt{2.0^2 + 1.0^2 + 0.5^2 + 3.0^2} = \sqrt{4.0 + 1.0 + 0.25 + 9.0} = \sqrt{14.25} \approx \mathbf{3.774917}$$

$$X_0 = \frac{G}{\|G\|_F} = \begin{bmatrix} \mathbf{0.529813} & \mathbf{0.264906} \\ \mathbf{0.132453} & \mathbf{0.794719} \end{bmatrix}$$

#### Adım 4.2: Newton-Schulz Yinelemeleri ($X_{k+1} = \frac{1}{2} X_k (3\mathbf{I} - X_k^T X_k)$)
5 ardışık matris çarpımı adımı:
* **1. Yineleme ($X_1$):**
  $$X_1 = \begin{bmatrix} 0.683180 & 0.239345 \\ 0.081331 & 0.896964 \end{bmatrix}$$
* **2. Yineleme ($X_2$):**
  $$X_2 = \begin{bmatrix} 0.834780 & 0.175106 \\ -0.003304 & 0.949314 \end{bmatrix}$$
* **3. Yineleme ($X_3$):**
  $$X_3 = \begin{bmatrix} 0.948780 & 0.121369 \\ -0.071699 & 0.981894 \end{bmatrix}$$
* **4. Yineleme ($X_4$):**
  $$X_4 = \begin{bmatrix} 0.990978 & 0.101423 \\ -0.097063 & 0.993884 \end{bmatrix}$$
* **5. Yineleme ($X_5$):**
  $$X_5 = \begin{bmatrix} \mathbf{0.995005} & \mathbf{0.099519} \\ \mathbf{-0.099485} & \mathbf{0.995028} \end{bmatrix}$$

#### Adım 4.3: Ortogonallik Doğrulaması ($X_5^T X_5$)
$$X_5^T X_5 = \begin{bmatrix} 0.999933 & 0.000032 \\ 0.000032 & 0.999985 \end{bmatrix} \approx \begin{bmatrix} \mathbf{1.0} & \mathbf{0.0} \\ \mathbf{0.0} & \mathbf{1.0} \end{bmatrix} = \mathbf{I}$$

Yalnızca beş matris çarpımında Newton-Schulz algoritması güncelleme matrisinin tüm tekil değerlerini birime eşitlemiştir.

| Optimizasyon Adımı | İşlem | Koordinat 1 ($w_1$) | Koordinat 2 ($w_2$) | Geometrik Sonuç |
| :--- | :--- | :--- | :--- | :--- |
| **Başlangıç Durumu** | Yok | $2.000000$ | $1.000000$ | Kayıp $= 12.000000$ |
| **Klasik SGD** | $w - \eta g$ | $1.900000$ | $0.000000$ | $w_2$ indi, $w_1$ %5 ilerleyebildi |
| **Momentum ($v_2$)** | $w - \eta (\beta v_1 + g)$ | $1.715000$ | $-0.900000$ | Taban yönündeki hız ikiye katlandı |
| **AdamW ($t=1$)** | Ayrıştırılmış Sönümleme | $1.898000$ | $0.899000$ | Eşitlenmiş adımlar; adil sönümleme |
| **Muon ($X_5^T X_5$)** | Newton-Schulz ($k=5$) | $\text{diag}_1 \approx 0.9999$ | $\text{diag}_2 \approx 0.9999$ | Tüm 2D matris ortogonalize edildi |

```text
Elle Hesaplama Özeti:
  SGD Adımı:              [ 1.900000, 0.000000 ]
  Momentum 2. Adım:       [ 1.715000, -0.900000 ]
  AdamW 1. Adım:          [ 1.898000, 0.899000 ]
  Muon X_5 Frobenius Normu: 1.414167 (2x2 Ortogonal Matris İçin İdeal sqrt(2))
  Nihai Ortogonallik:     X^T X = [[ 0.999933, 0.000032 ], [ 0.000032, 0.999985 ]]
```

---

## 7. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri

Optimizer eğitim sırasında GPU belleğinin tartışmasız en büyük canavarıdır; ancak çıkarım (inference) aşamasında tamamen yok olur.

| Mühendislik Boyutu | Eğitim Rejimi (Ön-Eğitim / SFT) | Çıkarım Rejimi (Sunum / Üretim) |
| :--- | :--- | :--- |
| **Optimizer Ayak İzi** | **VRAM'i Domine Eder:** FP32 durumlarla modelin 6 katı bellek | **Sıfır:** Optimizer tamamen devre dışıdır |
| **Durum Tamponları** | AdamW: FP32 Ana Ağırlıklar, 1. Momentler, 2. Momentler | Yok; yalnızca statik ağırlıklar bellekte tutulur |
| **Öğrenme Oranı Çizelgesi** | Adım sayaçlarıyla dinamik olarak değişir (Warmup, WSD) | Anlamsızdır; ağırlıklar sabittir |
| **Dağıtık Parçalama** | ZeRO-1 / ZeRO-2 / FSDP optimizer durumlarını kümede böler | Yalnızca model paralelliği (TP, PP) uygulanır |
| **Donanım Darboğazı** | Optimizer durumları için HBM kapasitesi; all-gather iletişimi | Ağırlık okuma sırasında HBM bellek bant genişliği |

### Eğitim Gözü: Optimizer Bellek Canavarı

Karma duyarlılıkta (BF16 ağırlıklar ve gradyanlar) eğitilen 8 milyar parametreli bir modeli düşünelim:
* **Model Ağırlıkları (BF16):** $8\text{B} \times 2\text{ bayt} = \mathbf{16\text{ GB}}$
* **Gradyanlar (BF16):** $8\text{B} \times 2\text{ bayt} = \mathbf{16\text{ GB}}$

Şimdi **AdamW**'nin tükettiği belleğe bakalım:
1. **FP32 Ana Ağırlıklar (Master Weights):** Küçük $\eta \Delta w$ güncellemelerinin basamak silinmesine uğramaması için: $8\text{B} \times 4\text{ bayt} = \mathbf{32\text{ GB}}$.
2. **1. Moment Tamponu ($m$ in FP32):** $8\text{B} \times 4\text{ bayt} = \mathbf{32\text{ GB}}$.
3. **2. Moment Tamponu ($v$ in FP32):** $8\text{B} \times 4\text{ bayt} = \mathbf{32\text{ GB}}$.

**Toplam AdamW Belleği: $96\text{ GB}$!**
Optimizer, **modelin kendi ağırlıklarından 6 kat daha fazla bellek tüketir**. Model, gradyanlar ve optimizer toplamda $128\text{ GB}$ VRAM gerektirir; bu da tek bir aktivasyon tensörü dahi belleğe girmeden 80 GB'lık NVIDIA H100 GPU'sunun kapasitesini aşar.

### Muon ve ZeRO Rahatlatması
Bu bellek duvarını kırmak için:
1. **Muon:** Tüm 2D katmanlarda her iki FP32 moment yerine tek bir momentum tamponu tutarak $32\text{ GB}$ bellek tasarrufu sağlar.
2. **ZeRO-1 / FSDP:** $96\text{ GB}$'lık optimizer durumlarını kümedeki $N$ adet GPU'ya paylaştırarak GPU başına düşen payı $96 / N\text{ GB}$ seviyesine indirir.

### Çıkarım Gözü: Sıfır Bellek Yükü
Çıkarım sırasında optimizer, moment tamponları, ana ağırlıklar ve öğrenme oranı çizelgeleri tamamen silinir. Model yalnızca statik bir parametre kontrol noktası olarak yüklenir; böylece GPU'nun 80 GB belleği yüzlerce eşzamanlı kullanıcıya hizmet verecek KV Cache havuzlarına tahsis edilir.

---

## 8. Python ve PyTorch ile doğrulama

Aşağıdaki script SGD, Momentum, Adam, AdamW ve 5 adımlı Muon Newton-Schulz ortogonalizasyonunu temel matematik kütüphanesiyle sıfırdan hesaplamakta ve PyTorch ile tam denklik sağlamaktadır:

```python
import math
import torch

# 1. Kuadratik vadi kurgusu
w0 = [2.0, 1.0]
g0 = [w0[0], 20.0 * w0[1]]

print("=== 1. KAYIP YÜZEYİ VE GRADYANLAR ===")
print(f"w0:                {w0}")
print(f"g0:                {g0}")

# 2. Klasik SGD
eta_sgd = 0.05
w1_sgd = [w0[0] - eta_sgd * g0[0], w0[1] - eta_sgd * g0[1]]
print("\n=== 2. KLASİK SGD ===")
print(f"w1 (SGD eta=0.05): {[round(v, 6) for v in w1_sgd]}")

# 3. Momentum
beta_m = 0.9
v1 = [g0[0], g0[1]]
w1_mom = [w0[0] - eta_sgd * v1[0], w0[1] - eta_sgd * v1[1]]
print("\n=== 3. MOMENTUM ===")
print(f"v1 (Momentum):     {v1}")
print(f"w1 (Momentum):     {[round(v, 6) for v in w1_mom]}")

# 4. Adam vs AdamW
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
eta = 0.1
wd = 0.01

m1 = [(1 - beta1) * g for g in g0]
v1_adam = [(1 - beta2) * (g**2) for g in g0]
m1_hat = [m / (1 - beta1**1) for m in m1]
v1_hat = [v / (1 - beta2**1) for v in v1_adam]
step_adam = [m_h / (math.sqrt(v_h) + eps) for m_h, v_h in zip(m1_hat, v1_hat)]

w1_adam_nowd = [w - eta * s for w, s in zip(w0, step_adam)]
w1_adamw = [w - eta * wd * w - eta * s for w, s in zip(w0, step_adam)]

print("\n=== 4. ADAM VS ADAMW ===")
print(f"m1_hat:            {[round(v, 6) for v in m1_hat]}")
print(f"v1_hat:            {[round(v, 6) for v in v1_hat]}")
print(f"Normalleştirilmiş: {[round(v, 6) for v in step_adam]}")
print(f"Adam (Sönümlemesiz): {[round(v, 6) for v in w1_adam_nowd]}")
print(f"AdamW (Ayrıştırılmış): {[round(v, 6) for v in w1_adamw]}")

# 5. Muon Newton-Schulz Ortogonalizasyonu
G = [[2.0, 1.0], [0.5, 3.0]]

def mat_frob_norm(M):
    return math.sqrt(sum(M[i][j]**2 for i in range(len(M)) for j in range(len(M[0]))))

def mat_mul(A, B):
    n, k, m = len(A), len(A[0]), len(B[0])
    res = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            res[i][j] = sum(A[i][p] * B[p][j] for p in range(k))
    return res

def mat_transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def mat_sub(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def mat_scale(A, s):
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]

frob = mat_frob_norm(G)
X = mat_scale(G, 1.0 / frob)
I = [[1.0, 0.0], [0.0, 1.0]]

print("\n=== 5. MUON NEWTON-SCHULZ ORTOGONALİZASYONU ===")
print(f"Girdi Matrisi G:   {G}")
print(f"Frobenius Normu:   {frob:.6f}")

for step in range(1, 6):
    Xt = mat_transpose(X)
    XtX = mat_mul(Xt, X)
    three_I = mat_scale(I, 3.0)
    bracket = mat_sub(three_I, XtX)
    X = mat_scale(mat_mul(X, bracket), 0.5)
    print(f"X_{step}:              {[[round(c, 6) for c in row] for row in X]}")

XtX_final = mat_mul(mat_transpose(X), X)
print(f"Nihai X^T @ X:     {[[round(c, 6) for c in row] for row in XtX_final]}")

# PyTorch AdamW doğrulaması
t_w = torch.tensor([2.0, 1.0], dtype=torch.float32, requires_grad=True)
opt = torch.optim.AdamW([t_w], lr=0.1, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8)
t_loss = 0.5 * t_w[0]**2 + 10.0 * t_w[1]**2
t_loss.backward()
opt.step()
print(f"\nPyTorch AdamW Güncellemesi: {t_w.detach().numpy().round(6).tolist()}")
```

Script çalıştırıldığında analitik elle hesap ile PyTorch AdamW çıktısı tam sayısal uyum sergiler:

```text
=== 1. KAYIP YÜZEYİ VE GRADYANLAR ===
w0:                [2.0, 1.0]
g0:                [2.0, 20.0]

=== 2. KLASİK SGD ===
w1 (SGD eta=0.05): [1.9, 0.0]

=== 3. MOMENTUM ===
v1 (Momentum):     [2.0, 20.0]
w1 (Momentum):     [1.9, 0.0]

=== 4. ADAM VS ADAMW ===
m1_hat:            [2.0, 20.0]
v1_hat:            [4.0, 400.0]
Normalleştirilmiş: [1.0, 1.0]
Adam (Sönümlemesiz): [1.9, 0.9]
AdamW (Ayrıştırılmış): [1.898, 0.899]

=== 5. MUON NEWTON-SCHULZ ORTOGONALİZASYONU ===
Girdi Matrisi G:   [[2.0, 1.0], [0.5, 3.0]]
Frobenius Normu:   3.774917
X_1:              [[0.68318, 0.239345], [0.081331, 0.896964]]
X_2:              [[0.83478, 0.175106], [-0.003304, 0.949314]]
X_3:              [[0.94878, 0.121369], [-0.071699, 0.981894]]
X_4:              [[0.990978, 0.101423], [-0.097063, 0.993884]]
X_5:              [[0.995005, 0.099519], [-0.099485, 0.995028]]
Nihai X^T @ X:     [[0.999933, 3.2e-05], [3.2e-05, 0.999985]]

PyTorch AdamW Güncellemesi: [1.898, 0.899]
```

---

## Bütün hikâye altı satırda

- Derin kayıp yüzeyleri kötü koşullu kanyonlar ($\kappa \gg 10^4$) oluşturarak klasik SGD'nin yamaçlarda savrulup tabanda donmasına yol açar.
- Polyak momentumu zıt yanal salınımları sönümler ve kanyon tabanında $\frac{1}{1-\beta}$ ($10\times$) hız birikimi sağlar.
- L2 regülarizasyonunun cezayı $\sqrt{\hat{v}_t}$'ye bölmesi sık güncellenen ağırlıkların sönümünü sıfırlarken seyrek özellikleri yok eder.
- AdamW ağırlık sönümlemesini momentlerden ayrıştırarak her parametreyi adilce küçültür ve genelleme gücünü kurtarır.
- Muon optimizasyon algoritması 2D ağırlık matrislerini doğrusal operatör olarak görüp Newton-Schulz yinelemeleriyle tekil değerleri birime eşitler.
- LLM ön-eğitiminde AdamW durumları modelin 6 katı bellek tüketirken, Muon bir tamponu silerek %30 daha hızlı yakınsama sunar.

---

## Terimler sözlüğü

- **Koşul Sayısı ($\kappa$)** — Hessian matrisinin maksimum özdeğerinin minimum özdeğerine oranı ($\lambda_{\max}/\lambda_{\min}$); yüzeyin yön dengesizliğini ölçer.
- **Momentum** — Geçmiş gradyanları $\beta$ sönümleme katsayısıyla biriktiren, salınımları bastıran hız vektörü.
- **Ayrıştırılmış Ağırlık Sönümlemesi (Decoupled Weight Decay)** — Ağırlık cezasını gradyan momentlerine karıştırmadan doğrudan parametreden $\eta \lambda \theta$ düşen kurgu.
- **AdamW** — Birinci momentleri, ikinci momentleri ve ayrıştırılmış ağırlık sönümlemesini birleştiren endüstri standardı optimizer.
- **Muon (MomentUm Orthogonalized by Newton-Schulz)** — Momentum matrislerini spektral norm altında en yakın ortogonal matrise eşleyen 2D matris optimizer'ı.
- **Newton-Schulz Yinelemesi** — Ağır SVD işlemlerine girmeden yalnızca matris çarpımlarıyla matris polar ortogonalizasyonu hesaplayan algoritma.
- **Warmup-Stable-Decay (WSD)** — Eğitimin büyük bölümünde öğrenme oranını sabit tutup sonda agresif düşüren modern LLM çizelgesi.
- **ZeRO (Zero Redundancy Optimizer)** — Optimizer durumlarını, gradyanları ve parametreleri kümedeki GPU'lara dağıtarak bellek duvarını kıran mimari.

---

## Daha derine inmek için

- [Polyak (USSR Computational Mathematics 1964): Some methods of speeding up the convergence of iteration methods](https://www.sciencedirect.com/science/article/pii/0041555364901375) — Optimizasyonda momentumun kurucu teorisi.
- [Kingma & Ba (ICLR 2015): Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Adam optimizasyon algoritmasını tanıtan kurucu makale.
- [Loshchilov & Hutter (ICLR 2019): Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) — AdamW'yi literatüre kazandıran ve L2 regülarizasyonunun çöküşünü ispatlayan çalışma.
- [Keller Jordan (2024): Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — Muon algoritmasının orijinal teknik raporu.
- Bu blogda: [Sinir Ağlarının Yapı Taşları (3): Geriye Yayılım ve Otomatik Türev](post.html?slug=geriye-yayilim-ve-otomatik-turev) — Optimizer'a giren gradyanların hesaplama çizgesinde nasıl üretildiği.
