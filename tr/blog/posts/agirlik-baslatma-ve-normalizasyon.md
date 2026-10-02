Modern derin öğrenme hassas bir fiziksel paradoks üzerinde durur. 100 katmanlı bir ağda veya 70 milyar parametreli bir Transformer modelinde, ileri geçiş aktivasyonlarının sayısal sıfıra ($10^{-45}$) sönümlenmeden ya da kayan nokta taşmasına ($10^{+38}$) fırlamadan yüzlerce ardışık matris çarpımını geçmesi gerekir. Geri yayılım (backward pass) esnasında ise gelen hata gradyanlarının, basamak silinmesi veya tanımsız (`NaN`) değerlere kurban gitmeden ilk embedding katmanına kadar sağ salim ulaşması zorunludur.

Ağırlıkları rastgele standart normal dağılımla $\mathcal{N}(0, 1)$ başlatırsanız, aktivasyonlar derinlik boyunca üstel olarak patlar. Bunları sezgisel olarak $\mathcal{N}(0, 0.01)$ gibi keyfi bir değere bastırırsanız, her katmanda katbekat küçülen sinyal, kayıp fonksiyonuna varamadan yok olur. Ağırlıklar varyansı koruyacak kusursuzlukta başlatılsa bile (Xavier veya Kaiming başlatması), residual bağlantılar ($x_{l+1} = x_l + F(x_l)$) artık akışına sürekli varyans pompalar ve temsil büyüklüğünü katman derinliği $L$ ile doğrusal oranda şişirir.

Bu varyans patlamasını dizginlemek için modern mimariler iki kenetli sisteme güvenir: sıfırıncı adımı stabilize eden **matematiksel ağırlık başlatma** ve milyarlarca token boyunca dinamik bir termostat gibi çalışan **aktivasyon normalizasyonu** (LayerNorm ve RMSNorm). Bu makale; her iki sistemin matematiğini, tensör kalkülüsünü, geri yayılım dinamiklerini ve donanım yürütmesini tüm açıklığıyla inceler.

---

**Bu yazıda:**

- [1. Derin Sinyal Yayılımının Kırılganlığı](#1-derin-sinyal-yayılımının-kırılganlığı)
- [2. Simetri Kırılması: Neden Sıfır Başlatma Öğrenmeyi Felç Eder?](#2-simetri-kırılması-neden-sıfır-başlatma-öğrenmeyi-felç-eder)
- [3. Varyans Koruma: Xavier ve Glorot Başlatma](#3-varyans-koruma-xavier-ve-glorot-başlatma)
- [4. Doğrusal-Olmayan Kırılma: Kaiming ve He Başlatma](#4-doğrusal-olmayan-kırılma-kaiming-ve-he-başlatma)
- [5. Residual Akış Patlaması ve DeepNorm Varyans Ölçeklemesi](#5-residual-akış-patlaması-ve-deepnorm-varyans-ölçeklemesi)
- [6. Normalizasyon Taksonomisi: BatchNorm, LayerNorm ve GroupNorm](#6-normalizasyon-taksonomisi-batchnorm-layernorm-ve-groupnorm)
- [7. LayerNorm Mekaniği: Matematiksel ve Geri Yayılım Türevi](#7-layernorm-mekaniği-matematiksel-ve-geri-yayılım-türevi)
- [8. RMSNorm Atılımı: Hız Uğruna Ortalamayı Çıkarmak](#8-rmsnorm-atılımı-hız-uğruna-ortalamayı-çıkarmak)
- [9. Pre-LN, Post-LN ve Gemma 2 Çift Normalizasyon Mimarisi](#9-pre-ln-post-ln-ve-gemma-2-çift-normalizasyon-mimarisi)
- [10. Adım Adım Elle Tensör Hesabı ve Python Doğrulaması](#10-adım-adım-elle-tensör-hesabı-ve-python-doğrulaması)
- [11. İki Mühendislik Gözü: Eğitim Çekirdekleri ve Çıkarımda Katlama](#11-iki-mühendislik-gözü-eğitim-çekirdekleri-ve-çıkarımda-katlama)

---

## 1. Derin Sinyal Yayılımının Kırılganlığı

Bias terimi içermeyen $L$ adet afin katmandan oluşan derin bir ileri beslemeli ağ düşünelim:

$$
x_l = W_l x_{l-1}
$$

Girdi vektörü $x_0 \in \mathbb{R}^d$, sıfır ortalamaya ve bileşen başına birim varyansa sahip olsun: $\mathbb{E}[x_0] = 0$, $\text{Var}(x_0) = 1$. Tüm ağırlık matrislerinin $W_l \in \mathbb{R}^{d \times d}$ bağımsız ve $\mathcal{N}(0, \sigma_w^2)$ dağılımından örneklendiğini varsayalım.

<svg viewBox="0 0 560 110" role="img" aria-label="Başlatma anında derin, doğrusal bir zincir. Standart normalden çekilen x0, W1&#x27;den geçip x1&#x27;i, W2&#x27;den geçip sonrakini verir ve W L&#x27;den geçip x L&#x27;ye ulaşır. Her ağırlık matrisinin varyansı sigma karedir; her katman sinyalin varyansını aynı katsayıyla çarpar ve L katmanda bu katsayı katlanarak büyür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ch-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="24" width="92" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="62.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x₀ ~ N(0,1)</text>
<line x1="110" y1="39" x2="142.0" y2="39" marker-end="url(#ch-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="144.0" y="24" width="40" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="164.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W₁</text>
<line x1="186.0" y1="39" x2="218.0" y2="39" marker-end="url(#ch-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="220.0" y="24" width="34" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x₁</text>
<line x1="256.0" y1="39" x2="288.0" y2="39" marker-end="url(#ch-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="290.0" y="24" width="40" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="310.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W₂</text>
<line x1="332.0" y1="39" x2="364.0" y2="39" marker-end="url(#ch-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="377.0" y="46" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px">…</text>
<line x1="390.0" y1="39" x2="422.0" y2="39" marker-end="url(#ch-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="424.0" y="24" width="44" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="446.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_L</text>
<line x1="470.0" y1="39" x2="502.0" y2="39" marker-end="url(#ch-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="504.0" y="24" width="40" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="524.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x_L</text>
<text x="16" y="80" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her W için Var(W) = σ²; her katman varyansı aynı katsayıyla ölçekler.</text>
<text x="16" y="98" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">L katmanda katsayı katlanır: 1&#x27;in üstündeyse patlar, altındaysa söner.</text>
</svg>

İlk katmanın herhangi bir $i$ koordinatındaki çıktısı, bağımsız rastgele değişkenlerin doğrusal bir kombinasyonudur:

$$
x_{1, i} = \sum_{j=1}^d W_{1, ij} x_{0, j}
$$

$W_{1, ij}$ ve $x_{0, j}$ sıfır beklentiye sahip bağımsız değişkenler olduğundan:

$$
\mathbb{E}[x_{1, i}] = \sum_{j=1}^d \mathbb{E}[W_{1, ij}] \mathbb{E}[x_{0, j}] = 0
$$

Bu bağımsız değişkenler toplamının varyansı şu şekilde açılır:

$$
\text{Var}(x_{1, i}) = \sum_{j=1}^d \text{Var}(W_{1, ij} x_{0, j}) = \sum_{j=1}^d \left( \mathbb{E}[W_{1, ij}^2] \mathbb{E}[x_{0, j}^2] - (\mathbb{E}[W_{1, ij}] \mathbb{E}[x_{0, j}])^2 \right)
$$

$$
\text{Var}(x_{1, i}) = \sum_{j=1}^d \sigma_w^2 \cdot \text{Var}(x_{0, j}) = d \cdot \sigma_w^2 \cdot \text{Var}(x_0)
$$

Bu doğrusal yinelemeyi $L$ ardışık katman boyunca zincirlediğimizde:

$$
\text{Var}(x_L) = (d \cdot \sigma_w^2)^L \cdot \text{Var}(x_0)
$$

Bu formül, sinyal iletimindeki temel ikilemi gözler önüne serer:

1. **Eğer $d \cdot \sigma_w^2 > 1$ ise:** Varyans, katman derinliği $L$ ile üstel olarak patlar. $d = 1024$ ve $\sigma_w = 0.05$ için büyüme çarpanı $(1024 \times 0.0025) = 2.56$ olur. $L = 50$ katmanda:

$$
\text{Var}(x_{50}) \approx (2.56)^{50} \approx 2.45 \times 10^{20}
$$

32-bit kayan nokta biçiminde (`FP32`, temsil edilebilen tavan değer $\approx 3.4 \times 10^{38}$), aktivasyonlar hızla aşırı değerlere fırlar, ardışık aktivasyon fonksiyonlarını doyurur ve derhal aritmetik taşmaya yol açar.

2. **Eğer $d \cdot \sigma_w^2 < 1$ ise:** Sinyal üstel olarak sönümlenir. $d = 1024$ ve $\sigma_w = 0.02$ için $d \cdot \sigma_w^2 = 0.4096$ olur. $L = 50$ katmanda:

$$
\text{Var}(x_{50}) \approx (0.4096)^{50} \approx 3.78 \times 10^{-20}
$$

Aktivasyonlar fiilen sıfıra çöker. Geri yayılımda ise hata gradyanı $\frac{\partial \mathcal{L}}{\partial W_1}$ ileri aktivasyonlarla doğru orantılı olduğundan gradyan yok olur (vanishing gradient). Ağ henüz ilk optimizasyon adımını tamamlayamadan öğrenmeyi durdurur.

Doğrusal derinlik boyunca sinyalin izometrik korunmasının tek şartı şudur:

$$
d \cdot \sigma_w^2 = 1 \implies \sigma_w = \frac{1}{\sqrt{d}}
$$

---

## 2. Simetri Kırılması: Neden Sıfır Başlatma Öğrenmeyi Felç Eder?

Akla şu soru gelebilir: Neden tüm ağırlıkları sıfır veya aynı küçük $c$ sabit değeriyle başlatmıyoruz?

Dışbükey (convex) optimizasyonda (örneğin doğrusal veya lojistik regresyon), sıfır başlatma tamamen zararsızdır çünkü kayıp yüzeyi tek bir küresel minimuma sahiptir. Derin sinir ağlarında ise sıfır başlatma, felaket getiren bir **permütasyon simetrisini (permutation symmetry)** tetikler.

Eleman-bazlı bir doğrusal-olmayan $\phi$ fonksiyonu üzerinden aktivasyon üreten bir gizli katmanı ele alalım:

$$
a^{(1)} = \phi(W^{(1)} x + b^{(1)})
$$

Eğer tüm $i \in \{1, \dots, m\}$ nöronları için $W^{(1)}_{ij} = 0$ ve $b^{(1)}_i = 0$ ise:

$$
a^{(1)}_i = \phi(0) = \text{sabit} \quad \forall i
$$

Gizli katmandaki her bir nöron birebir aynı skaler çıktıyı üretir. Şimdi bir sonraki katmanın $W^{(2)}$ ağırlıklarının geri yayılım gradyan hesabını inceleyelim:

$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}_{k, i}} = \delta^{(2)}_k \cdot a^{(1)}_i
$$

Tüm $i \in \{1, \dots, m\}$ indeksleri için $a^{(1)}_i$ özdeş olduğundan, $W^{(2)}$ matrisinin her bir sütunu birebir aynı gradyan güncellemesini alır. Benzer şekilde, hatayı ilk katmana geri yaydığımızda:

$$
\delta^{(1)}_i = \left( \sum_{k} \delta^{(2)}_k W^{(2)}_{k, i} \right) \cdot \phi'(z^{(1)}_i)
$$

Eğer tüm başlangıç ağırlıkları $W^{(2)}_{k, i}$ sıfıra (veya aynı sabite) eşitse, her gizli nöron tam olarak aynı geri yayılan hata sinyalini $\delta^{(1)}_i$ alır.

<svg viewBox="0 0 560 190" role="img" aria-label="Simetrik başlatma neden çöker. Aynı girdi x 1, 2 ve 3 numaralı nöronlara ulaşır. Ağırlıklar aynıysa z ön-aktivasyonları, a aktivasyonları ve dL/dz gradyanları da aynıdır; her güncelleme onları aynı tutar. Ağın rankı tek bir etkin nörona çöker." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sy-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="66" width="90" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="61.0" y="92.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">girdi x</text>
<path d="M106 88 H124 V33 H150" marker-end="url(#sy-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="152" y="12" width="240" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="166" y="30" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Nöron 1</text>
<text x="166" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">z₁, a₁, ∂L/∂z₁</text>
<text x="380" y="38" text-anchor="end" style="fill:var(--c-text-mute);font-size:14px">=</text>
<path d="M106 88 H124 V87 H150" marker-end="url(#sy-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="152" y="66" width="240" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="166" y="84" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Nöron 2</text>
<text x="166" y="100" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">z₂, a₂, ∂L/∂z₂</text>
<text x="380" y="92" text-anchor="end" style="fill:var(--c-text-mute);font-size:14px">=</text>
<path d="M106 88 H124 V141 H150" marker-end="url(#sy-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="152" y="120" width="240" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="166" y="138" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Nöron 3</text>
<text x="166" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">z₃, a₃, ∂L/∂z₃</text>
<text x="380" y="146" text-anchor="end" style="fill:var(--c-text-mute);font-size:14px">=</text>
<path d="M400 12 Q410 12 410 24 V76 Q410 88 420 88 Q410 88 410 100 V152 Q410 164 400 164" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="428" y="76" text-anchor="start" style="fill:var(--c-text);font-size:12px">tamamen özdeş:</text>
<text x="428" y="92" text-anchor="start" style="fill:var(--c-text);font-size:12px">rank tek etkin</text>
<text x="428" y="108" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">nörona çöker</text>
<text x="16" y="182" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Aynı ağırlık → aynı çıktı → aynı gradyan → aynı güncelleme, sonsuza dek.</text>
</svg>

İstediğiniz kadar gradyan inişi adımı atın:

$$
W^{(1)}_{t+1} = W^{(1)}_t - \eta \nabla_{W^{(1)}} \mathcal{L}
$$

Katmandaki tüm nöronlar matematiksel olarak birbirinden ayırt edilemez kalmaya devam eder. 4096 gizli nörona sahip devasa bir katman, tek bir nöronun gücüne çöker ve mimarinin tüm temsil kapasitesi yok olur. **Permütasyon simetrisini kırmak için rastgele başlatma zorunludur.**

---

## 3. Varyans Koruma: Xavier ve Glorot Başlatma

2010 yılındaki çığır açan makalelerinde (*Understanding the difficulty of training deep feedforward neural networks*), Xavier Glorot ve Yoshua Bengio, orijin etrafında simetrik doğrusal ve sigmoid benzeri aktivasyon fonksiyonlarında ($\phi'(0) \approx 1$) sinyal yayılımını modellediler.

### İleri Geçiş Şartının Matematiksel Türevi

$n_{in}$ boyutlu bir girdiyi $n_{out}$ boyutlu bir çıktıya bağlayan bir katman olsun:

$$
z_i = \sum_{j=1}^{n_{in}} W_{ij} x_j
$$

$W_{ij}$ ve $x_j$ değişkenlerinin sıfır ortalamalı ve bağımsız olduğunu varsayarsak:

$$
\text{Var}(z_i) = n_{in} \cdot \text{Var}(W) \cdot \text{Var}(x)
$$

Ağ boyunca ileri akan sinyalin patlamasını veya sönümlenmesini önlemek için $\text{Var}(z) = \text{Var}(x)$ eşitliğini zorunlu kılarız:

$$
n_{in} \cdot \text{Var}(W) = 1 \implies \text{Var}(W) = \frac{1}{n_{in}}
$$

### Geri Geçiş Şartının Matematiksel Türevi

Şimdi geri yayılıma bakalım. Kayıp fonksiyonunun $x_j$ aktivasyonuna göre gradyanı:

$$
\frac{\partial \mathcal{L}}{\partial x_j} = \sum_{i=1}^{n_{out}} W_{ij} \frac{\partial \mathcal{L}}{\partial z_i}
$$

Yine sıfır ortalama ve bağımsızlık varsayımları altında:

$$
\text{Var}\left( \frac{\partial \mathcal{L}}{\partial x_j} \right) = n_{out} \cdot \text{Var}(W) \cdot \text{Var}\left( \frac{\partial \mathcal{L}}{\partial z} \right)
$$

Hata sinyalinin geriye doğru yayılırken varyansını koruması için:

$$
n_{out} \cdot \text{Var}(W) = 1 \implies \text{Var}(W) = \frac{1}{n_{out}}
$$

### Glorot Uzlaşısı

$n_{in} = n_{out}$ olmadığı sürece tek bir varyans değeri her iki koşulu aynı anda sağlayamaz. Glorot ve Bengio, iki hedefin harmonik ortalamasını almayı önerdiler:

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

Normal dağılım $\mathcal{N}(0, \sigma^2)$ için:

$$
W \sim \mathcal{N}\left( 0, \frac{2}{n_{in} + n_{out}} \right)
$$

Varyansı $\frac{(2a)^2}{12} = \frac{a^2}{3}$ olan sürekli düzgün dağılım $\mathcal{U}(-a, a)$ için:

$$
\frac{a^2}{3} = \frac{2}{n_{in} + n_{out}} \implies a = \sqrt{\frac{6}{n_{in} + n_{out}}}
$$

$$
W \sim \mathcal{U}\left( -\sqrt{\frac{6}{n_{in} + n_{out}}}, +\sqrt{\frac{6}{n_{in} + n_{out}}} \right)
$$

---

## 4. Doğrusal-Olmayan Kırılma: Kaiming ve He Başlatma

Xavier başlatması, aktivasyon fonksiyonunun orijin civarında birim eğimli ($\phi'(0) \approx 1$) doğrusal bir davranış sergilediği varsayımına dayanır. Derin ağlar **Düzeltilmiş Doğrusal Birime (ReLU)** geçtiğinde:

$$
\phi(z) = \max(0, z)
$$

Xavier varsayımı temelinden çöktü.

### Xavier Neden ReLU Karşısında İflas Eder?

Ön-aktivasyon $z$ değerinin sıfır etrafında simetrik ve $\sigma_z^2$ varyansına sahip olduğunu varsayalım. ReLU tüm negatif değerleri doğrudan sıfırlar:

$$
\mathbb{E}[\phi(z)^2] = \int_{-\infty}^{\infty} (\max(0, z))^2 p(z) \, dz = \int_0^{\infty} z^2 p(z) \, dz = \frac{1}{2} \int_{-\infty}^{\infty} z^2 p(z) \, dz = \frac{1}{2} \text{Var}(z)
$$

Dağılımın yarısı budandığı için, aktivasyonun beklenen kare büyüklüğü her katmanda tam yarı yarıya azalır!

```
Ön-aktivasyon z:     Ortalama = 0, Varyans = sigma^2
                     [----- Negatif Değerler -----|+++++ Pozitif Değerler +++++]
                                 |
                          ReLU(z) = max(0, z)
                                 v
Son-aktivasyon x:    [-------- Sıfırlanan (0) ----|+++++ Korunan Değerler +++++]
                     Varyans tam yarı yarıya erir: Var(x) = 0.5 * Var(z)
```

Eğer 30 katmanlı bir ReLU ağı Xavier başlatmasıyla ($\text{Var}(W) = 1/n_{in}$) başlatılırsa:

$$
\text{Var}(x_L) = \left( \frac{1}{2} \right)^L \text{Var}(x_0) = \left( \frac{1}{2} \right)^{30} \approx 9.31 \times 10^{-10}
$$

İleri sinyal 9 büyüklük mertebesi eriyerek ağın tamamen donmasına neden olur.

### Kaiming / He Formülasyonunun Türetilmesi

2015 yılında Kaiming He, Xiangyu Zhang, Shaoqing Ren ve Jian Sun (*Delving Deep into Rectifiers*), bu kaybı telafi eden kesin çarpanı türettiler.

Girdisi $x_{l-1} = \phi(z_{l-1})$ olan $l$ katmanı için:

$$
z_{l, i} = \sum_{j=1}^{n_{in}} W_{l, ij} x_{l-1, j}
$$

$x_{l-1}$ bir ReLU çıktısı olduğundan ortalaması sıfır değildir ($\mathbb{E}[x_{l-1}] = \sqrt{\frac{\text{Var}(z_{l-1})}{2\pi}}$). Buna rağmen:

$$
\text{Var}(z_{l, i}) = n_{in} \cdot \text{Var}(W_l) \cdot \mathbb{E}[x_{l-1}^2]
$$

$\mathbb{E}[x_{l-1}^2] = \frac{1}{2} \text{Var}(z_{l-1})$ eşitliğini yerine yazarsak:

$$
\text{Var}(z_l) = n_{in} \cdot \text{Var}(W_l) \cdot \frac{1}{2} \text{Var}(z_{l-1}) = \left( \frac{1}{2} n_{in} \text{Var}(W_l) \right) \text{Var}(z_{l-1})
$$

Katmanlar boyunca sabit varyansı korumak için ($\text{Var}(z_l) = \text{Var}(z_{l-1})$):

$$
\frac{1}{2} n_{in} \text{Var}(W_l) = 1 \implies \text{Var}(W) = \frac{2}{n_{in}}
$$

Normal dağılım için:

$$
W \sim \mathcal{N}\left( 0, \sqrt{\frac{2}{n_{in}}} \right)
$$

Düzgün dağılım için $\mathcal{U}(-\sqrt{6/n_{in}}, +\sqrt{6/n_{in}})$. Eğim katsayısı $\alpha$ olan Leaky ReLU için ise:

$$
\text{Var}(W) = \frac{2}{(1 + \alpha^2) n_{in}}
$$

---

## 5. Residual Akış Patlaması ve DeepNorm Varyans Ölçeklemesi

Doğrusal izdüşümler matematiksel olarak mükemmel bir varyans korumasıyla başlatılsa dahi, residual (artık) bağlantılar sisteme kaçınılmaz bir eklemeli varyans birikimi sokar.

### Doğrusal Varyans Birikimi Yasası

Standart residual güncellemesini inceleyelim:

$$
x_{l+1} = x_l + F(x_l; W_l)
$$

Residual kol çıktısının $F(x_l)$, atlama bağlantısındaki $x_l$ ile korelasyonsuz olduğunu varsayarsak:

$$
\text{Var}(x_{l+1}) = \text{Var}(x_l) + \text{Var}(F(x_l))
$$

Eğer her alt katman varyansı koruyorsa ($\text{Var}(F(x_l)) \approx \text{Var}(x_0)$):

$$
\text{Var}(x_L) = \text{Var}(x_0) + \sum_{l=1}^L \text{Var}(F(x_l)) = (1 + L) \cdot \text{Var}(x_0)
$$

$L = 64$ katmanlı derin bir modelde residual akıştaki aktivasyon varyansı 65 katına çıkar!

<svg viewBox="0 0 560 184" role="img" aria-label="Residual akış boyunca varyansın büyümesi. x0&#x27;ın varyansı 1&#x27;dir. Her blok varyansı 1,0 olan F(x)&#x27;i ekler; x1&#x27;in varyansı 2 olur ve L bloktan sonra x L&#x27;nin varyansı 1 artı L&#x27;dir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="rs-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="rs-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="22" width="76" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="54.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x₀</text>
<text x="54.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Var = 1</text>
<rect x="150" y="22" width="76" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="188.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x₁</text>
<text x="188.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Var = 2</text>
<text x="306" y="48" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px">…</text>
<rect x="444" y="22" width="100" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="494.0" y="43.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x_L</text>
<text x="494.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Var = 1 + L</text>
<circle cx="112" cy="39" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="112" y="44" text-anchor="middle" style="fill:var(--c-text);font-size:15px">+</text>
<rect x="66" y="106" width="92" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="112.0" y="125.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">F(x₀)</text>
<text x="112" y="152" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Var = 1,0</text>
<line x1="112" y1="106" x2="112" y2="52" marker-end="url(#rs-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="262" cy="39" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="262" y="44" text-anchor="middle" style="fill:var(--c-text);font-size:15px">+</text>
<rect x="216" y="106" width="92" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="262.0" y="125.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">F(x₁)</text>
<text x="262" y="152" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Var = 1,0</text>
<line x1="262" y1="106" x2="262" y2="52" marker-end="url(#rs-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="412" cy="39" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="412" y="44" text-anchor="middle" style="fill:var(--c-text);font-size:15px">+</text>
<rect x="366" y="106" width="92" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="412.0" y="125.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">F(x_L−1)</text>
<text x="412" y="152" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Var = 1,0</text>
<line x1="412" y1="106" x2="412" y2="52" marker-end="url(#rs-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="92" y1="39" x2="99" y2="39" marker-end="url(#rs-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="123" y1="39" x2="148" y2="39" marker-end="url(#rs-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="226" y1="39" x2="249" y2="39" marker-end="url(#rs-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="273" y1="39" x2="296" y2="39" marker-end="url(#rs-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="314" y1="39" x2="399" y2="39" marker-end="url(#rs-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="423" y1="39" x2="442" y2="39" marker-end="url(#rs-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her blok varyansı 1 olan bir kol ekler</text>
</svg>

### $1/\sqrt{2L}$ Ölçekleme Kuralı (GPT-2)

İlk autoregressive Transformer'larda residual akışı dizginlemek amacıyla GPT-2 makalesi (Radford et al., 2019), artık izdüşüm katmanları için (self-attention içindeki $W_O$ ve MLP içindeki $W_2$) derinliğe bağlı bir başlatma çarpanı getirdi:

$$
W_{init} \sim \mathcal{N}\left( 0, \frac{\sigma_{base}}{\sqrt{2L}} \right)
$$

Buradaki $2L$, $L$ adet Transformer bloğundaki toplam residual ekleme sayısını temsil eder (katman başına bir attention ve bir feedforward bloğu).

Bu ölçekleme altında her alt katmanın eklediği varyans orantılı olarak bastırılır:

$$
\text{Var}(F(x_l)) \approx \frac{1}{2L} \text{Var}(x_0)
$$

$$
\text{Var}(x_L) \approx \text{Var}(x_0) + 2L \left( \frac{1}{2L} \text{Var}(x_0) \right) = 2 \cdot \text{Var}(x_0)
$$

Ağın ister 12, ister 128 katmanı olsun; aktivasyon varyansı $1.0$ ile $2.0$ arasında kesin olarak sınırlanır.

### DeepNorm (Wang et al., 2022)

Transformer mimarilerini patlama yaşamadan 1000 katmana kadar ölçekleyebilmek için Microsoft, **DeepNorm** yöntemini önerdi:

$$
x_{l+1} = x_l \cdot \alpha + F(x_l; W_l \cdot \beta)
$$

Buradaki $\alpha$ ve $\beta$ katsayıları, gradyan beklentilerini sınırlayacak şekilde analitik olarak türetilmiştir:

$$
\alpha = (2L)^{1/4}, \quad \beta = (8L)^{-1/4}
$$

DeepNorm, gradyan normunun derinlikle birlikte büyümesini engelleyerek 1.000 katmanlı Transformer'ları karmaşık öğrenme hızı ısıtma (warmup) döngülerine ihtiyaç duymadan eğitilebilir hale getirir.

---

## 6. Normalizasyon Taksonomisi: BatchNorm, LayerNorm ve GroupNorm

Başlatma, sıfırıncı adımı kurtarır. Ancak eğitim ilerledikçe milyonlarca ağırlık güncellenir ve ara aktivasyonların dağılımı kaçınılmaz olarak kayar — literatürde buna *internal covariate shift* denir.

Eğitim boyunca varyansı dinamik olarak yönetebilmek için mimarilere normalizasyon katmanları eklenir. Farklı normalizasyon aileleri arasındaki temel ayrım, **aktivasyon tensörünün hangi boyutları üzerinden istatistik toplandığıdır**.

```
   Tensör Boyutları: [Batch (B), Dizi/Uzamsal (T), Kanal/Gizli Boyut (D)]

   Batch Normalization (BN)          Layer Normalization (LN)         Group Normalization (GN)
      (B ve T üzerinden indirgeme)      (D boyutu üzerinden indirgeme)   (D alt grupları üzerinden)
          [ *  *  * ]                      [ *  .  . ]                      [ *  *  . ]
          [ *  *  * ]                      [ *  .  . ]                      [ *  *  . ]
          [ *  *  * ]                      [ *  .  . ]                      [ *  *  . ]
      Batch Boyutuna Bağımlı           Batch Boyutundan Bağımsız        Batch Boyutundan Bağımsız
```

### Karşılaştırmalı Taksonomi Tablosu

| Normalizasyon Türü | İndirgeme Boyutları | Değişmezlik (Invariance) | Birincil Kullanım Alanı | Kritik Zaafiyet / Arıza |
| :--- | :--- | :--- | :--- | :--- |
| **Batch Normalization (BN)** | Batch ($B$), Dizi/Uzam ($T$) | $W$ ölçeğine duyarsız; batch boyutuna bağımlı | Bilgisayarlı Görü (ResNet, CNN) | $B=1$ durumunda veya değişken token boylarında çöker |
| **Layer Normalization (LN)** | Gizli Boyut ($D$) | Batch boyutundan ve $x$ kaymasından bağımsız | NLP, Transformer (BERT, GPT-2) | Hesaplama yükü: ortalama $\mu$ indirgemesi gerektirir |
| **Group Normalization (GN)** | Kanal Grupları ($G$) | Batch boyutundan bağımsız; esnek gruplama | Küçük batch'li görsel modeller (Mask R-CNN) | Grup sayısı $G$ hiperparametresine hassas |
| **Instance Normalization (IN)** | Yalnızca Uzamsal ($T$) | Kontrast ve stil değişimlerine duyarsız | Stil transferi, GAN modelleri | Kanallar arası istatistiksel bilgiyi siler |
| **RMS Normalization (RMSNorm)** | Gizli Boyut ($D$) | Vektör ölçeğine duyarsız; sıfır ortalama varsayımı | Modern LLM'ler (LLaMA, Mistral, DeepSeek) | Ortalamayı sıfırlamaz ($\mu \neq 0$) |

### BatchNorm Büyük Dil Modellerinde Neden İflas Etti?

1. **Değişken Dizi Uzunlukları:** LLM ön-eğitimi 8k veya 128k gibi değişken uzunluktaki token dizilerini dinamik olarak paketler. BatchNorm istatistikleri batch içindeki tüm token'lar üzerinden hesaplandığından padding token'ları veya değişken uzunluklar ampirik batch istatistiklerini zehirler.
2. **Mikro-Batch ve Pipeline Paralelizmi:** 70B'lik modeller eğitilirken bellek kısıtları nedeniyle GPU başına mikro-batch boyutu genellikle $B=1$ veya $B=2$'ye kadar düşer. $B=1$ olduğunda batch içi varyans hesabı $\sigma_B^2$ tanımsızlaşır veya sıfıra iner; BatchNorm matematiksel olarak çalışamaz.
3. **Eğitim ve Çıkarım Tutarsızlığı:** BatchNorm eğitim esnasında koşan ortalama ve varyans ($\hat{\mu}, \hat{\sigma}^2$) biriktirir ve çıkarımda bunları dondurur. Oysa tek tek token üreten autoregressive çıkarımda (decode fazı) bu istatistikler tekil token temsilinden saparak kaliteyi düşürür.

LayerNorm ise istatistikleri her bir token vektörünün kendi gizli boyutu $D$ üzerinden toplar; bu sayede batch boyutundan ve komşu token'lardan tamamen bağımsız çalışır.

---

## 7. LayerNorm Mekaniği: Matematiksel ve Geri Yayılım Türevi

Jimmy Lei Ba, Jamie Ryan Kiros ve Geoffrey Hinton tarafından 2016'da tanıtılan Layer Normalization, $x \in \mathbb{R}^D$ özellik vektörü üzerinde şu adımları işletir:

### İleri Geçiş Formülasyonu

1. **Ortalama Hesabı:**

$$
\mu = \frac{1}{D} \sum_{i=1}^D x_i
$$

2. **Merkezlenmiş Varyans Hesabı:**

$$
\sigma^2 = \frac{1}{D} \sum_{i=1}^D (x_i - \mu)^2
$$

3. **Standartlaştırma:**

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

Buradaki $\epsilon > 0$, sıfıra bölme hatasını önleyen küçük bir kararlılık sabitidir (genellikle $10^{-5}$).

4. **Öğrenilebilir Afin Dönüşüm:**

$$
y_i = \gamma_i \hat{x}_i + \beta_i
$$

Burada $\gamma, \beta \in \mathbb{R}^D$ öğrenilebilir ölçek ve kaydırma parametreleridir. Başlangıçta $\gamma = \mathbf{1}$ ve $\beta = \mathbf{0}$ olarak ayarlanırlar.

### Tam Geri Yayılım Türevi

Sonraki katmanlardan gelen üst gradyan vektörü $\frac{\partial \mathcal{L}}{\partial y} \in \mathbb{R}^D$ olsun.

Öğrenilebilir parametrelere göre gradyanlar doğrudan zincir kuralıyla bulunur:

$$
\frac{\partial \mathcal{L}}{\partial \gamma_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \hat{x}_i, \quad \frac{\partial \mathcal{L}}{\partial \beta_i} = \frac{\partial \mathcal{L}}{\partial y_i}
$$

Standartlaştırılmış $\hat{x}_i$ bileşenine göre gradyan:

$$
\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \gamma_i
$$

Girdi gradyanı $\frac{\partial \mathcal{L}}{\partial x_i}$ için $\sigma^2$ ve $\mu$ üzerinden çok değişkenli zincir kuralını işletiriz:

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i} + \frac{\partial \mathcal{L}}{\partial \mu} \frac{\partial \mu}{\partial x_i}
$$

Parçalı türevleri tek tek hesapladığımızda:

$$
\frac{\partial \mathcal{L}}{\partial \sigma^2} = \sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial \hat{x}_j} (x_j - \mu) \left( -\frac{1}{2} (\sigma^2 + \epsilon)^{-3/2} \right) = -\frac{1}{2 (\sigma^2 + \epsilon)} \sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j
$$

$$
\frac{\partial \mathcal{L}}{\partial \mu} = \sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \left( -\frac{1}{\sqrt{\sigma^2 + \epsilon}} \right) + \frac{\partial \mathcal{L}}{\partial \sigma^2} \left( \frac{1}{D} \sum_{j=1}^D -2(x_j - \mu) \right) = -\frac{1}{\sqrt{\sigma^2 + \epsilon}} \sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial \hat{x}_j}
$$

Bu terimleri yerine koyup sadeleştirdiğimizde birleşik analitik geri yayılım formülü ortaya çıkar:

$$
\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{D \sqrt{\sigma^2 + \epsilon}} \left( D \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right)
$$

Bu ifadenin iki muazzam geometrik özelliği vardır:
1. $\sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial x_i} = 0$: Girdi gradyanlarının toplamı (ortalaması) kesin olarak sıfırdır.
2. $\sum_{j=1}^D \frac{\partial \mathcal{L}}{\partial x_i} \hat{x}_i = 0$: Gradyan vektörü, standartlaştırılmış girdi vektörüne $\hat{x}$ diktir (ortogonaldir).

---

## 8. RMSNorm Atılımı: Hız Uğruna Ortalamayı Çıkarmak

2019 yılında Biao Zhang ve Rico Sennrich, *Root Mean Square Layer Normalization* (NeurIPS 2019) çalışmasını yayınladılar. Makalenin sorduğu temel soru şuydu: **Ağı stabilize eden şey gerçekten ortalamayı çıkarmak ($\mu$) mıdır, yoksa sadece vektörün ölçeğini kontrol altında tutmak mıdır?**

### Temel Sezgi

Zhang & Sennrich, LayerNorm'un sağladığı eğitim kararlılığının neredeyse tamamen **ölçek değişmezliğinden (scale invariance)** kaynaklandığını, ortalamayı kaydırmanın (shift invariance) modele kayda değer bir regülarizasyon veya kararlılık avantajı sağlamadığını kanıtladılar.

RMSNorm, $\mu \equiv 0$ varsayarak karekök ortalama kare (Root Mean Square) istatistiğini hesaplar:

$$
\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}
$$

$$
\bar{x}_i = \frac{x_i}{\text{RMS}(x)}
$$

$$
y_i = \gamma_i \bar{x}_i
$$

Buradaki iki kritik yapısal değişime dikkat edin:
1. **Ortalama çıkarma adımı yok:** Pay doğrudan $x_i$ değeridir.
2. **Kaydırma parametresi $\beta$ yok:** Afin ölçekleme yalnızca $\gamma_i$ ile yapılır, $\beta \in \mathbb{R}^D$ parametresi mimariden tamamen atılmıştır.

<svg viewBox="0 0 560 210" role="img" aria-label="LayerNorm ile RMSNorm. LayerNorm x üzerinde 1. geçişte ortalama mu&#x27;yu hesaplar, 2. geçişte x eksi mu ile merkezler, 3. geçişte varyans sigma kareyi hesaplar, sonra normalize eder ve gamma ile beta afinini uygular. RMSNorm x&#x27;in karelerinin ortalamasının karekökünü tek geçişte hesaplar, sonra normalize eder ve yalnızca gamma uygular." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="lr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="20" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700;letter-spacing:.06em">LayerNorm</text>
<rect x="16" y="38" width="30" height="36" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="31.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x</text>
<rect x="60" y="30" width="88" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="104.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">ortalama μ</text>
<text x="104.0" y="68.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">1. geçiş</text>
<line x1="46" y1="56" x2="58" y2="56" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="158" y="30" width="88" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="202.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">x − μ</text>
<text x="202.0" y="68.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">2. geçiş</text>
<line x1="148" y1="56" x2="156" y2="56" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="256" y="30" width="88" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="300.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">varyans σ²</text>
<text x="300.0" y="68.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">3. geçiş</text>
<line x1="246" y1="56" x2="254" y2="56" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="354" y="30" width="88" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="398.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">normalize</text>
<line x1="344" y1="56" x2="352" y2="56" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="452" y="30" width="88" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="496.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">afin</text>
<text x="496.0" y="68.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">γ, β</text>
<line x1="442" y1="56" x2="450" y2="56" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="120" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">RMSNorm</text>
<rect x="16" y="138" width="30" height="36" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="31.0" y="160.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x</text>
<rect x="60" y="130" width="284" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="202.0" y="152.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">RMS = √(Σx²/D)</text>
<text x="202.0" y="168.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">x üzerinde TEK geçiş</text>
<line x1="46" y1="156" x2="58" y2="156" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="354" y="130" width="88" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="398.0" y="160.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">normalize</text>
<line x1="344" y1="156" x2="352" y2="156" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="452" y="130" width="88" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="496.0" y="152.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">afin</text>
<text x="496.0" y="168.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">yalnız γ</text>
<line x1="442" y1="156" x2="450" y2="156" marker-end="url(#lr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="202" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">x üç kez yerine bir kez okunur; ortalama ve β ortadan kalkar.</text>
</svg>

### Donanımsal ve Hesaplamalı Üstünlük

Standart GPU mimarisinde LayerNorm, gizli boyut $D$ üzerinde 2 veya 3 farklı global bellek indirgemesi (reduction pass) gerektirir:
1. $\mu$ değerini bulmak için $\sum x_i$.
2. $\sigma^2$ değerini bulmak için $\sum (x_i - \mu)^2$.
3. Standartlaştırma ve afin dönüşüm.

Bellek bant genişliği darboğazına takılan LLM iş yüklerinde, GPU SRAM ile High Bandwidth Memory (HBM) arasındaki bu veri transferleri gecikmeyi domine eder. RMSNorm ise tensör üzerinde **tek bir indirgeme geçişiyle** ($x_i^2$ kareler toplamı) tamamlanır.

RMSNorm, modelin yakınsama hızından ve doğruluk başarımından hiçbir şey kaybetmeden normalizasyon süresini **%7 ila %15** oranında kısaltır. Bu nedenle LLaMA, Mistral, Gemma, DeepSeek ve Qwen gibi modern modellerin tamamı RMSNorm'a geçmiştir.

---

## 9. Pre-LN, Post-LN ve Gemma 2 Çift Normalizasyon Mimarisi

Normalizasyonun residual toplama işlemine göre nereye yerleştirildiği, derin Transformer blokları boyunca gradyan akışını kökünden değiştirir.

<svg viewBox="0 0 560 318" role="img" aria-label="Post-LN ile Pre-LN. Orijinal 2017 Transformer&#x27;ı olan Post-LN&#x27;de x l hem alt katmana hem residual yola gider; ikisi toplanır ve LayerNorm toplamadan sonra ana yolun üstünde durur, x l artı 1&#x27;i üretir; her gradyan bir normalizasyondan geçmek zorundadır. 2019-2023&#x27;ün modern standardı Pre-LN&#x27;de kol önce LayerNorm ya da RMSNorm&#x27;u, sonra alt katmanı uygular; sonuç el değmemiş residual yola eklenir ve x l artı 1 çıkar; residual otoyolu temiz kalır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="pp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="pp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="8" width="256" height="302" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="30" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700;letter-spacing:.06em">Post-LN</text>
<text x="30" y="47" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">orijinal Transformer, 2017</text>
<rect x="52" y="60" width="68" height="28" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="86.0" y="78.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x_l</text>
<circle cx="86" cy="214" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="86" y="219" text-anchor="middle" style="fill:var(--c-text);font-size:15px">+</text>
<path d="M86 102 H192 V112" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="192" y1="112" x2="192" y2="114" marker-end="url(#pp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="86" y1="88" x2="86" y2="201" marker-end="url(#pp-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="138" y="116" width="108" height="32" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="192.0" y="136.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">SubLayer</text>
<path d="M192 148 V214 H99" marker-end="url(#pp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="36" y="238" width="100" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="86.0" y="257.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">LayerNorm</text>
<line x1="86" y1="225" x2="86" y2="236" marker-end="url(#pp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="52" y="276" width="68" height="26" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="86.0" y="293.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x_l+1</text>
<line x1="86" y1="268" x2="86" y2="274" marker-end="url(#pp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="144" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">norm otoyolun</text>
<text x="144" y="276" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tam üstünde</text>
<text x="78" y="160" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px"transform="rotate(-90 78 160)">residual</text>
<rect x="288" y="8" width="256" height="302" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="302" y="30" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">Pre-LN</text>
<text x="302" y="47" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">modern standart, 2019–2023</text>
<rect x="324" y="60" width="68" height="28" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="358.0" y="78.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x_l</text>
<circle cx="358" cy="214" r="11" style="fill:var(--c-surface-2);stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="358" y="219" text-anchor="middle" style="fill:var(--c-text);font-size:15px">+</text>
<path d="M358 102 H464 V112" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="464" y1="112" x2="464" y2="114" marker-end="url(#pp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="358" y1="88" x2="358" y2="201" marker-end="url(#pp-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="410" y="116" width="108" height="32" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="464.0" y="136.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">LN / RMSNorm</text>
<rect x="410" y="162" width="108" height="32" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="464.0" y="182.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">SubLayer</text>
<line x1="464" y1="148" x2="464" y2="160" marker-end="url(#pp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M464 194 V214 H371" marker-end="url(#pp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="324" y="250" width="68" height="26" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="358.0" y="267.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x_l+1</text>
<line x1="358" y1="225" x2="358" y2="248" marker-end="url(#pp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="416" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">otoyol el</text>
<text x="416" y="276" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">değmeden kalır</text>
<text x="350" y="160" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px"transform="rotate(-90 350 160)">residual</text>
</svg>

### 1. Post-LN (Vaswani et al., 2017)

Orijinal Transformer mimarisinde normalizasyon katmanı, residual toplamanın hemen sonrasına konulmuştur:

$$
x_{l+1} = \text{Norm}(x_l + F(x_l))
$$

Her katmanın çıktısı normalize edildiğinden, $l$ katmanındaki residual hat üzerinden geriye akan gradyan şu katsayıyla ölçeklenir:

$$
\frac{\partial x_{l+1}}{\partial x_l} \approx \frac{1}{\sqrt{\text{Var}(x_l + F(x_l))}}
$$

Derin katmanlarda çıkışa yakın gradyanlar çok büyükken, ilk katmanlara doğru indikçe gradyanlar üstel olarak sönümlenir ($O(1/\sqrt{L})$). Çok hassas bir öğrenme hızı ısıtma (warmup) takvimi uygulanmazsa Post-LN modelleri eğitimin ilk adımlarında patlayarak çöker.

### 2. Pre-LN (Radford et al., 2019; Wang et al., 2019)

Pre-LN mimarisi normalizasyonu alt katman kolunun içine taşır:

$$
x_{l+1} = x_l + F(\text{Norm}(x_l))
$$

$x_l$ girdisine göre gradyana bakalım:

$$
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \left( I + \frac{\partial F}{\partial \text{Norm}(x_l)} \frac{\partial \text{Norm}(x_l)}{\partial x_l} \right)
$$

Bunu $L$ katman boyunca açarsak:

$$
\frac{\partial \mathcal{L}}{\partial x_0} = \frac{\partial \mathcal{L}}{\partial x_L} + \sum_{l=0}^{L-1} \frac{\partial \mathcal{L}}{\partial x_{l+1}} \frac{\partial F}{\partial x_l}
$$

Buradaki birim matris $I$, kayıptan doğrudan ilk embedding katmanına kadar kesintisiz ve sönümlenmeyen bir gradyan otoyolu sunar. Pre-LN modelleri agresif ısınma takvimlerine ihtiyaç duymadan kararlı şekilde eğitilir.

### 3. Pre-LN Artık Akış Kayması (Residual Drift)

Fakat Pre-LN ikincil bir fiziksel kusur doğurur: **sınırsız artık akış büyümesi**. $x_{l+1} = x_l + F(\text{Norm}(x_l))$ formülü nedeniyle her alt katman, normalize edilmiş bir girdiden üretilen vektörü mevcut akışa ekler. Derinlik $L = 64$ veya $L = 80$'e çıktıkça residual akışın normu $\|x_l\|_2$ monoton şekilde büyür:

$$
\|x_L\| \approx \sqrt{L} \cdot \|x_0\|
$$

Bunun sonucunda en derin alt katmanların ana akışa yaptığı göreceli katkı gittikçe küçülür:

$$
\frac{\|F(\text{Norm}(x_l))\|}{\|x_l\|} \approx \frac{1}{\sqrt{l}}
$$

Son katmanlar temsili değiştirmekte güçlük çeker.

### 4. Gemma 2 Çift Normalizasyon (Dual-Norm)

Google'ın Gemma 2 (2024) modeli, 27 milyar parametreli ölçekte bu artık akış büyümesini durdurmak için **Çift Normalizasyon (Pre- ve Post-Normalizasyon)** yapısını tanıttı:

$$
x_{l+1} = x_l + \text{RMSNorm}_{post}(F(\text{RMSNorm}_{pre}(x_l)))
$$

Hem alt katmana girmeden önce hem de alt katmandan çıktıktan sonra normalizasyon uygulanır. Bu sayede alt katman çıktılarının normu sınırlanarak devasa modellerde eğitim dinamiği tam kontrol altına alınır.

---

## 10. Adım Adım Elle Tensör Hesabı ve Python Doğrulaması

Bu denklemleri somut sayılarla mühürlemek için 4 boyutlu bir aktivasyon vektörünü hem **LayerNorm** hem de **RMSNorm** adımlarından elle geçirip geri yayılım gradyanlarını hesaplayalım.

### Tensör Tanımı

Girdi vektörümüz:

$$
x = [2.0, -1.0, 4.0, 3.0], \quad D = 4, \quad \epsilon = 10^{-5}
$$

Afin parametreleri:

$$
\gamma = [1.0, 0.5, 1.5, 0.8], \quad \beta = [0.1, -0.2, 0.0, 0.5]
$$

---

### Adım 1: Elle LayerNorm İleri Geçişi

1. **Ortalama $\mu$:**

$$
\mu = \frac{2.0 + (-1.0) + 4.0 + 3.0}{4} = \frac{8.0}{4} = 2.000000
$$

2. **Merkezlenmiş Vektör $(x - \mu)$:**

$$
x - \mu = [2.0 - 2.0, -1.0 - 2.0, 4.0 - 2.0, 3.0 - 2.0] = [0.0, -3.0, 2.0, 1.0]
$$

3. **Varyans $\sigma^2$:**

$$
\sigma^2 = \frac{0.0^2 + (-3.0)^2 + 2.0^2 + 1.0^2}{4} = \frac{0 + 9 + 4 + 1}{4} = \frac{14.0}{4} = 3.500000
$$

4. **Standart Sapma $\sqrt{\sigma^2 + \epsilon}$:**

$$
\sqrt{3.500000 + 0.000010} = \sqrt{3.500010} \approx 1.870831
$$

5. **Standartlaştırılmış Vektör $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$:**

$$
\hat{x}_0 = \frac{0.0}{1.870831} = 0.000000
$$

$$
\hat{x}_1 = \frac{-3.0}{1.870831} \approx -1.603565
$$

$$
\hat{x}_2 = \frac{2.0}{1.870831} \approx 1.069043
$$

$$
\hat{x}_3 = \frac{1.0}{1.870831} \approx 0.534522
$$

6. **Afin Dönüşüm $y_i = \gamma_i \hat{x}_i + \beta_i$:**

$$
y_0 = 1.0 \times 0.000000 + 0.1 = 0.100000
$$

$$
y_1 = 0.5 \times (-1.603565) + (-0.2) = -0.801783 - 0.2 = -1.001783
$$

$$
y_2 = 1.5 \times 1.069043 + 0.0 = 1.603565
$$

$$
y_3 = 0.8 \times 0.534522 + 0.5 = 0.427618 + 0.5 = 0.927618
$$

---

### Adım 2: Elle RMSNorm İleri Geçişi

1. **Karelerin Ortalaması:**

$$
\frac{1}{D} \sum_{i=1}^4 x_i^2 = \frac{2.0^2 + (-1.0)^2 + 4.0^2 + 3.0^2}{4} = \frac{4 + 1 + 16 + 9}{4} = \frac{30.0}{4} = 7.500000
$$

2. **Karekök Ortalama Kare $\text{RMS}(x)$:**

$$
\text{RMS}(x) = \sqrt{7.500000 + 0.000010} = \sqrt{7.500010} \approx 2.738615
$$

3. **Normalize Vektör $\bar{x}_i = \frac{x_i}{\text{RMS}(x)}$:**

$$
\bar{x}_0 = \frac{2.0}{2.738615} \approx 0.730296
$$

$$
\bar{x}_1 = \frac{-1.0}{2.738615} \approx -0.365148
$$

$$
\bar{x}_2 = \frac{4.0}{2.738615} \approx 1.460593
$$

$$
\bar{x}_3 = \frac{3.0}{2.738615} \approx 1.095444
$$

4. **Afin Dönüşüm $y_{RMS, i} = \gamma_i \bar{x}_i$:**

$$
y_{RMS, 0} = 1.0 \times 0.730296 = 0.730296
$$

$$
y_{RMS, 1} = 0.5 \times (-0.365148) = -0.182574
$$

$$
y_{RMS, 2} = 1.5 \times 1.460593 = 2.190889
$$

$$
y_{RMS, 3} = 0.8 \times 1.095444 = 0.876356
$$

---

### Adım 3: Geri Yayılım Gradyan İncelemesi

Üst katmandan gelen gradyan vektörünün $\frac{\partial \mathcal{L}}{\partial y} = [0.5, -0.5, 1.0, -1.0]$ olduğunu varsayalım.

Bölüm 7 ve 8'deki analitik formülleri işlettiğimizde:

```
Metrik / Bileşen                 LayerNorm Gradyanı       RMSNorm Gradyanı
--------------------------------------------------------------------------
dL / dx_0                        +0.140312                +0.064510
dL / dx_1                        +0.077314                -0.032255
dL / dx_2                        +0.449572                +0.311593
dL / dx_3                        -0.667197                -0.469215
--------------------------------------------------------------------------
Girdi Gradyanları Toplamı:        0.000000                -0.125367
```

LayerNorm için girdi gradyanları toplamı tam olarak sıfırdır ($\sum \frac{\partial \mathcal{L}}{\partial x_i} = 0.000000$). RMSNorm'da ise ortalama sıfırlanmadığı için gradyanların toplamı sıfır olmak zorunda değildir; bu durum modelin genel ölçeği daha hızlı öğrenmesini sağlar.

---

### Adım 4: Bağımsız Python Doğrulama Betiği

Tüm bu hesaplamaları virgülden sonra 6 basamağa kadar doğrulayan saf Python betiği:

```python
import math

def verify_normalization():
    x = [2.0, -1.0, 4.0, 3.0]
    D = len(x)
    eps = 1e-5
    gamma = [1.0, 0.5, 1.5, 0.8]
    beta = [0.1, -0.2, 0.0, 0.5]
    dy = [0.5, -0.5, 1.0, -1.0]

    # --- 1. LayerNorm İleri ve Geri ---
    mu = sum(x) / D
    centered = [xi - mu for xi in x]
    var = sum(ci ** 2 for ci in centered) / D
    std = math.sqrt(var + eps)
    x_hat = [ci / std for ci in centered]
    y_ln = [g * xh + b for g, xh, b in zip(gamma, x_hat, beta)]

    dx_hat = [dyi * gi for dyi, gi in zip(dy, gamma)]
    sum_dx_hat = sum(dx_hat)
    sum_dx_hat_x_hat = sum(dh * xh for dh, xh in zip(dx_hat, x_hat))
    dx_ln = [
        (1.0 / (D * std)) * (D * dhi - sum_dx_hat - xhi * sum_dx_hat_x_hat)
        for dhi, xhi in zip(dx_hat, x_hat)
    ]

    # --- 2. RMSNorm İleri ve Geri ---
    mean_sq = sum(xi ** 2 for xi in x) / D
    rms = math.sqrt(mean_sq + eps)
    x_bar = [xi / rms for xi in x]
    y_rms = [g * xb for g, xb in zip(gamma, x_bar)]

    dx_bar = [dyi * gi for dyi, gi in zip(dy, gamma)]
    sum_dx_bar_x_bar = sum(dh * xb for dh, xb in zip(dx_bar, x_bar))
    dx_rms = [
        (1.0 / rms) * (dhi - (xb / D) * sum_dx_bar_x_bar)
        for dhi, xb in zip(dx_bar, x_bar)
    ]

    print("LayerNorm İleri y:", [round(v, 6) for v in y_ln])
    print("RMSNorm İleri y:  ", [round(v, 6) for v in y_rms])
    print("LayerNorm Geri dx:", [round(v, 6) for v in dx_ln])
    print("RMSNorm Geri dx:  ", [round(v, 6) for v in dx_rms])

if __name__ == "__main__":
    verify_normalization()
```

---

## 11. İki Mühendislik Gözü: Eğitim Çekirdekleri ve Çıkarımda Katlama

Normalizasyon katmanları parametre sayısı bakımından küçüktür; buna karşın GPU bellek bant genişliği darboğazları sebebiyle uçtan uca verimlilik üzerinde devasa bir etkiye sahiptir.

### 1. Eğitim Sistemleri: Triton ve CUDA Çekirdek Füzyonu

Derin öğrenme çatılarının naif uygulamalarında Pre-LayerNorm kullanan bir attention bloğu birden fazla bağımsız GPU çekirdeği (kernel) fırlatır:

<svg viewBox="0 0 560 190" role="img" aria-label="Birleştirilmemiş normalizasyon. Residual toplama sonucunu HBM&#x27;e yazar; ortalama indirgemesi onu geri okur ve yazar; varyans indirgemesi okur ve yazar; standartlaştırma ve afin okur ve yazar; QKV GEMM en sonunda normalize tensörü okur. Tek bir normalizasyon için dört HBM gidiş-dönüşü." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="uf-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="uf-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="18" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ayrı kernel&#x27;ler</text>
<rect x="16" y="26" width="96" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="64.0" y="48.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">residual</text>
<text x="64.0" y="64.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">toplama</text>
<rect x="124" y="26" width="96" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="172.0" y="48.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">ortalama</text>
<text x="172.0" y="64.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">indirgeme</text>
<rect x="232" y="26" width="96" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="48.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">varyans</text>
<text x="280.0" y="64.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">indirgeme</text>
<rect x="340" y="26" width="96" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="388.0" y="48.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">standartlaş.</text>
<text x="388.0" y="64.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+ afin</text>
<rect x="448" y="26" width="96" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="496.0" y="56.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">QKV GEMM</text>
<rect x="16" y="130" width="528" height="30" rx="6" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="280" y="149.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">HBM (çip dışı)</text>
<line x1="86" y1="78" x2="86" y2="128" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<line x1="150" y1="130" x2="150" y2="80" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="91" y="98" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yaz</text>
<text x="145" y="118" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">oku</text>
<line x1="194" y1="78" x2="194" y2="128" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<line x1="258" y1="130" x2="258" y2="80" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="199" y="98" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yaz</text>
<text x="253" y="118" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">oku</text>
<line x1="302" y1="78" x2="302" y2="128" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<line x1="366" y1="130" x2="366" y2="80" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="307" y="98" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yaz</text>
<text x="361" y="118" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">oku</text>
<line x1="410" y1="78" x2="410" y2="128" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<line x1="474" y1="130" x2="474" y2="80" marker-end="url(#uf-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="415" y="98" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yaz</text>
<text x="469" y="118" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">oku</text>
<text x="16" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tek bir normalizasyon için 4 HBM gidiş-dönüşü.</text>
</svg>

Normalizasyon işlemleri hesaplama-yoğun değil bellek-yoğun işlemler olduğundan (aritmetik yoğunluk $< 2 \text{ FLOP/byte}$), aktivasyonların GPU çip-içi SRAM'i ile HBM arasında sürekli gidip gelmesi katman süresinin %40'ını boşa harcar.

Modern eğitim motorları **Kaynaşmış Çekirdekler (Fused Kernels)** kullanır (OpenAI Triton veya CUTLASS ile):
1. **Fused Residual + RMSNorm:** $x_{l+1} = x_l + \text{sublayer}(x)$ artık toplama işlemi doğrudan SM yazmaçlarında (registers) yapılır.
2. Kareler toplamı $\sum x_i^2$, GPU warp-shuffle buyruklarıyla (`__shfl_xor_sync`) çip içinde birkaç saat çevriminde tamamlanır.
3. Normalize edilen vektör $\gamma$ ile çarpılıp HBM'e hiç yazılmadan doğrudan QKV GEMM matris çarpımının girdi tamponuna teslim edilir.

<svg viewBox="0 0 560 236" role="img" aria-label="Birleştirilmiş RMSNorm kernel&#x27;i. Tek bir GPU streaming multiprocessor içinde, SRAM ve yazmaçlarda: residual girdi yazmaçlarda toplanır, warp-shuffle indirgemesi RMS&#x27;i hesaplar, sonuç gamma ile çarpılır ve ölçekli çıktı hazır olur. Tek bir HBM yazımı onu doğrudan GEMM girdi tamponuna gönderir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="fu-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="fu-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<rect x="16" y="8" width="528" height="150" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="30" y="28" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU streaming multiprocessor (SM) · SRAM / yazmaçlar</text>
<rect x="32" y="40" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="102.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">residual girdi</text>
<line x1="172" y1="57" x2="208" y2="57" marker-end="url(#fu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="210" y="40" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">yazmaç içi toplam</text>
<line x1="350" y1="57" x2="386" y2="57" marker-end="url(#fu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="388" y="40" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="458.0" y="61.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">warp-shuffle RMS</text>
<path d="M458 74 V115 H352" marker-end="url(#fu-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="32" y="98" width="140" height="34" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="102.0" y="119.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">ölçekli çıktı</text>
<rect x="210" y="98" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="119.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">γ ile çarp</text>
<line x1="210" y1="115" x2="174" y2="115" marker-end="url(#fu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="530" y="150" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">sonuna kadar hiçbir şey çipten çıkmaz</text>
<rect x="16" y="198" width="528" height="30" rx="6" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="280" y="217.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">HBM · GEMM girdi tamponu</text>
<line x1="102" y1="132" x2="102" y2="196" marker-end="url(#fu-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="112" y="176" text-anchor="start" style="fill:var(--c-text);font-size:12px">TEK HBM yazımı</text>
</svg>

### 2. Çıkarım Sistemleri: Ağırlık Katlama (Weight Folding) ve Sıfır Ek Yük

Tek tek token üreten autoregressive decode fazında çıkarım gecikmesi, tek bir token'ın matris-vektör çarpımını (GEMV) yapmak için ağırlıkların HBM'den okunma süresiyle sınırlıdır.

1. **RMSNorm Ağırlık Katlama:** RMSNorm'u takip eden doğrusal katmanlarda (örneğin QKV izdüşümü $W_{QKV} \in \mathbb{R}^{D \times 3D}$), $\gamma \in \mathbb{R}^D$ parametresi servis başlamadan önce izdüşüm ağırlıklarının içine matematiksel olarak katlanabilir:

$$
y = \text{RMSNorm}(x) W = \left( \frac{x}{\text{RMS}(x)} \odot \gamma \right) W = \frac{x}{\text{RMS}(x)} (\text{diag}(\gamma) W)
$$

$$
W_{\text{katlanmış}} = \text{diag}(\gamma) W
$$

Ağırlık matrisinin her $i$ satırı model yüklenirken peşinen $\gamma_i$ ile çarpılarak servis sırasında bağımsız bir afin çarpım adımı tamamen ortadan kaldırılır.

2. **Çıkarım GEMV Füzyonu:** vLLM ve TensorRT-LLM gibi modern motorlarda RMSNorm indirgemesi, decode GEMV çekirdeğinin girişine (prologue) kaynaştırılır. Tekil token vektörü SRAM yazmaçlarına yüklenirken ilk iş parçacığı bloğu tarafından RMS hesaplanır ve normalizasyon bağımsız bir aşama olmaktan tamamen çıkar.

---

## Bütün hikâye altı satırda

- Derin ağlarda katman varyansı kesin bir izometriyle korunmadığı sürece sinyal ya sayısal sıfıra çöker ya da kayan nokta taşmasıyla patlar.
- Ağırlıkları sıfır başlatmak tüm nöronların aynı çıktıyı ve aynı gradyanı almasına yol açarak tüm katmanı tek bir nöronun etkinliğine hapseder.
- Orijin etrafında simetrik doğrusal aktivasyonlarda ileri ve geri sinyali korumak Xavier kuralı olan $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$ şartını zorunlu kılar.
- ReLU aktivasyonu sinyalin yarısını sıfırlayıp varyansı her katmanda yarıya indirdiğinden Kaiming başlatması $\text{Var}(W) = \frac{2}{n_{in}}$ çarpanını şart koşar.
- Residual bağlantıların katman derinliğiyle doğrusal artan varyansını ($Var(x_L) \approx L \cdot \sigma^2$) LayerNorm ve RMSNorm dinamik bir termostat gibi sınırlar.
- Ortalamayı çıkarma adımını atarak bellek bant genişliği darboğazını kıran RMSNorm, modern büyük dil modellerinin standart normalizasyon motoru haline gelmiştir.

---

## Terimler sözlüğü

- **Permütasyon Simetrisi (Permutation Symmetry)** — Katmandaki tüm nöronların özdeş çıktılar ve gradyanlar üretmesi sonucunda çok nöronlu bir yapının tek bir nörona çökmesi durumu.
- **Xavier / Glorot Başlatma** — Sigmoid ve Tanh gibi orijin etrafında simetrik aktivasyonlarda ileri ve geri varyansı koruyan $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$ başlatma stratejisi.
- **Kaiming / He Başlatma** — ReLU'nun sinyalin yarısını sıfırlamasını telafi ederek derin ağlarda varyans çöküşünü önleyen $\text{Var}(W) = \frac{2}{n_{in}}$ başlatma formülasyonu.
- **Residual Varyans Birikimi** — Atlama bağlantılarının ($x + F(x)$) katman sayısı arttıkça varyansı doğrusal olarak ($Var(x_L) \propto L$) şişirmesi olgusu.
- **Layer Normalization (LayerNorm)** — Her token'ın kendi $D$ özellik boyutu üzerinden ampirik ortalama $\mu$ ve varyans $\sigma^2$ hesaplayarak batch boyutundan bağımsız standartlaştırma yapan katman.
- **Root Mean Square Normalization (RMSNorm)** — Ortalama çıkarma adımını ve bias terimini kaldırarak yalnızca karekök ortalama kare ile ölçek değişmezliği sağlayan yüksek verimli normalizasyon.
- **Pre-LN vs Post-LN** — Normalizasyonun alt katmandan önce (Pre-LN, kesintisiz gradyan otoyolu) veya residual toplamadan sonra (Post-LN, sönümlenen gradyanlar) yerleştirildiği mimari düzenler.
- **Operatör Füzyonu (Operator Fusion)** — Normalizasyon, indirgeme ve bellek işlemlerini tek bir GPU çekirdeğinde birleştirerek HBM bellek transferlerini baypas eden derleyici optimizasyonu.

---

## Daha derine inmek için

- [Glorot & Bengio (AISTATS 2010): Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Xavier başlatmasının kurucu matematiksel temelleri.
- [He et al. (ICCV 2015): Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) — Kaiming (He) başlatmasının orijinal makalesi.
- [Ba, Kiros & Hinton (2016): Layer Normalization](https://arxiv.org/abs/1607.06450) — LayerNorm mimarisinin kurucu makalesi.
- [Zhang & Sennrich (NeurIPS 2019): Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm formülasyonunun ve hız kazanımlarının sunulduğu makale.
- Bu blogda: [Sinir Ağlarının Yapı Taşları (4): Optimizasyon Algoritmaları](post.html?slug=optimizasyon-algoritmalari-derinlemesine) — Ağırlıkların gradyanlarla nasıl güncellendiği ve AdamW mekaniği.
