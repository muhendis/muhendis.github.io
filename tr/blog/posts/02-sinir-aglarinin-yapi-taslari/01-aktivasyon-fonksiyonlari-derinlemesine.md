100 katmandan oluşan devasa bir derin yapay sinir ağı tasarladığınızı, GPU belleğinde milyarlarca kayan noktalı parametre tahsis ettiğinizi ve binlerce CUDA çekirdeğinde tensör çarpımlarını koşturduğunuzu hayal edin. Modeli petabaytlarca veriyle günlerce eğitiyorsunuz; ancak eğitim ne kadar sürerse sürsün model düz bir doğrusal karar sınırının ötesine geçemiyor ve en basit 2-bitlik XOR problemini dahi öğrenemeden iflas ediyor.

Bu arızanın sebebi bir yazılım hatası ya da sayısal kararsızlık değildir; bütünüyle temel lineer cebirin değişmez bir kuralıdır: matris çarpımı birleşmelidir (associative). Katmanların arasına doğrusal-olmayan (non-linear) bir aktivasyon fonksiyonu yerleştirmediğiniz sürece, art arda gelen tüm $W_L \dots W_2 W_1 x$ projeksiyonları matematiksel olarak tek bir $W_{\text{net}} x$ matrisine çöker. 100 katmanlı devasa ağınız, işlevsel açıdan tek bir doğrusal regresyon modeline indirgenir; yüksek boyutlu uzayı bükemez, katlayamaz ve karmaşık desenleri ayıramaz.

Aktivasyon fonksiyonları, sinir ağlarına evrensel fonksiyon yaklaştırıcı (Universal Approximation Theorem) olma gücünü kazandıran yegâne matematiksel bileşendir. Ancak onlarca yıl boyunca yanlış aktivasyon seçimleri; sönümlenen gradyanlar (vanishing gradients), ölü nöronlar (dead neurons) ve eğitim kararsızlıklarıyla derin öğrenmenin önündeki en büyük tıkanıklık oldu. Bu yazı, doğrusal-olmayan aktivasyonların geometrik zorunluluğunu ortaya koyuyor; erken dönem bağlantıcıların doymuş sigmoidlerinden GELU'nun olasılıksal kapılamasına uzanan evrimi inceliyor; LLaMA 3 ve DeepSeek-V3 gibi modern büyük dil modellerinin (LLM) neden istisnasız SwiGLU kullandığını çözümlüyor ve elle yapılan sayısal tensör yürüyüşünü doğrulanabilir kodlarla sunuyor.

**Bu yazıda**

- [1. En yalın hâliyle temel sezgi: Doğrusallık aktivasyonsuz neden çöker?](#1-en-yalın-hâliyle-temel-sezgi-doğrusallık-aktivasyonsuz-neden-çöker)
- [2. Klasik dönem: Sigmoid, Tanh ve gradyan doymasının kalkülüsü](#2-klasik-dönem-sigmoid-tanh-ve-gradyan-doymasının-kalkülüsü)
- [3. Derin öğrenme rönesansı: ReLU ve ölü nöron patolojisi](#3-derin-öğrenme-rönesansı-relu-ve-ölü-nöron-patolojisi)
- [4. Olasılıksal kapılama paradigması: GELU](#4-olasılıksal-kapılama-paradigması-gelu)
- [5. Modern LLM standardı: Gated Linear Units ve SwiGLU](#5-modern-llm-standardı-gated-linear-units-ve-swiglu)
- [6. Adım adım elle sayısal tensör yürüyüşü](#6-adım-adım-elle-sayısal-tensör-yürüyüşü)
- [7. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri](#7-iki-mühendislik-gözü-eğitim-ve-çıkarım-dinamikleri)
- [8. Python ve PyTorch ile doğrulama](#8-python-ve-pytorch-ile-doğrulama)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. En yalın hâliyle temel sezgi: Doğrusallık aktivasyonsuz neden çöker?

### Temsilin Origami İlkesi

Aktivasyon fonksiyonunun uzayda ne yaptığını anlamak için masanın üzerindeki düz bir kâğıt parçasını düşünün.

Kâğıdı yalnızca döndürmenize, ötelemenize, uzatıp kısaltmanıza ya da eğriltmenize izin verilirse —yani $W x + b$ afin dönüşümlerinin geometrik karşılıklarını uygularsanız— düz bir çizginin iki karşı tarafında kalan iki noktayı, aradaki tüm doğrudaş noktaları hareket ettirmeden asla yan yana getiremezsiniz. Yalnızca doğrusal kesitler kullanarak kâğıdın merkezindeki dairesel bir bölgeyi çevresinden ayıramazsınız.

Doğrusal-olmayan aktivasyon fonksiyonları kâğıda **katlama (fold)** yeteneği kazandırır. Uzayı bükerek, keserek veya burkarak birbirine çok uzak bölgeleri tek bir katlama hamlesiyle yan yana getirir ve konveks olmayan karmaşık karar sınırlarının çizilmesine olanak tanır.

<svg viewBox="0 0 560 300" role="img" aria-label="Derinlik neden doğrusal olmayana muhtaç. Solda doğrusal çöküş: girdi x, W1'li 1. katmandan, W2'li 2. katmandan, W_L'li L. katmana kadar geçer; birleşme özelliği yüzünden çıktı (W_L ... W1) x'tir, bu da tek bir W_net matrisinin x ile çarpımına eşittir, yani derin yığın tek bir doğrusal katmana çöker ve rankı darboğaz katmanıyla sınırlanır. Sağda doğrusal olmayan katlama: her katmanda W x artı b afin dönüşümünün ardından uzayı büken sigma aktivasyonu uygulanır; sonuç evrensel yaklaşım teoremi uyarınca konveks olmayan karmaşık manifoldları modelleyebilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="af-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="af-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="8" width="256" height="284" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="30" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">Doğrusal çöküş (Rank sınırı)</text>
<rect x="40" y="44" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="62.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">girdi: x ∈ ℝᵈ</text>
<line x1="144" y1="72" x2="144" y2="81" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="82" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">1. katman: W₁x</text>
<line x1="144" y1="110" x2="144" y2="119" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="120" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="138.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">2. katman: W₂h₁</text>
<line x1="144" y1="148" x2="144" y2="157" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="40" y="158" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="176.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">L. katman: W_L h_{L-1}</text>
<line x1="144" y1="186" x2="144" y2="236" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="30" y="238" width="228" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="144" y="256" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">(W_L … W₁) x = W_net x</text>
<text x="144" y="273" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bileşke kuralı: tek matrise çöker</text>
<rect x="288" y="8" width="256" height="284" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="302" y="30" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">Doğrusal olmayan katlama</text>
<rect x="312" y="44" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="62.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">girdi: x ∈ ℝᵈ</text>
<line x1="416" y1="72" x2="416" y2="81" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="82" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="100.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">afin: z₁ = W₁x + b₁</text>
<line x1="416" y1="110" x2="416" y2="119" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="120" width="208" height="28" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416.0" y="138.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">katlama: h₁ = σ(z₁)</text>
<line x1="416" y1="148" x2="416" y2="157" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="158" width="208" height="28" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416.0" y="176.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">afin: z₂ = W₂h₁ + b₂</text>
<line x1="416" y1="186" x2="416" y2="195" marker-end="url(#af-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="312" y="196" width="208" height="28" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416.0" y="214.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">katlama: h₂ = σ(z₂)</text>
<line x1="416" y1="224" x2="416" y2="236" marker-end="url(#af-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="302" y="238" width="228" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="256" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">evrensel yaklaşım manifoldu</text>
<text x="416" y="273" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">uzay bükülür: konveks olmayan sınırlar</text>
</svg>

### Matris Çöküşünün Matematiksel İspatı

Aktivasyon fonksiyonu barındırmayan $L$ katmanlı bir ileri beslemeli ağı matematiksel olarak tanımlayalım:

$$h_1 = W_1 x + b_1$$

$$h_2 = W_2 h_1 + b_2 = W_2 (W_1 x + b_1) + b_2 = (W_2 W_1) x + (W_2 b_1 + b_2)$$

Tüm $L$ katman boyunca tümevarım uygulandığında:

$$y = W_L (W_{L-1} \dots (W_1 x + b_1) \dots + b_{L-1}) + b_L = W_{\text{net}} x + b_{\text{net}}$$

Burada:

$$W_{\text{net}} = \prod_{l=L}^{1} W_l, \quad b_{\text{net}} = b_L + \sum_{i=1}^{L-1} \left( \prod_{j=L}^{i+1} W_j \right) b_i$$

Doğrusal dönüşümler kümesi bileşke işlemine göre kapalı olduğundan, $W_{\text{net}}$ ifadesi $d_{\text{out}} \times d_{\text{in}}$ boyutunda tek bir matristen ibarettir. Ayrıca matris rank eşitsizliği gereği:

$$\text{rank}(W_{\text{net}}) \le \min \left( \text{rank}(W_1), \text{rank}(W_2), \dots, \text{rank}(W_L) \right)$$

Ara katmanları ne kadar geniş veya derin tutarsanız tutun, doğrusal katmanlar darboğaz katmanının rank sınırını aşamaz. Daha da önemlisi, doğrusal dönüşümler mantıksal XOR fonksiyonunu bile modelleyemez; dilin sözdizimini çözemez, görsel nesne hiyerarşilerini ayıramaz veya anlamsal çıkarım yapamaz.

---

## 2. Klasik dönem: Sigmoid, Tanh ve gradyan doymasının kalkülüsü

Yapay sinir ağlarının erken döneminde biyolojik benzerlik mimari kararları belirliyordu. Biyolojik nöronların hücre zarı potansiyeli belirli bir eşiği aştığında aksiyon potansiyeli üretmesi, araştırmacıları pürüzsüz ve S-şekilli sıkıştırma fonksiyonlarına yöneltti.

### Lojistik Sigmoid ve Gradyan Sönümlenmesi

Standart lojistik sigmoid, $(-\infty, \infty)$ gerçel sayı doğrusunu $(0, 1)$ açık olasılık aralığına sıkıştırır:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Birinci türevi sade bir cebirsel özelliğe sahiptir:

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z) \left( 1 - \sigma(z) \right)$$

Bu türevin tepe noktası incelendiğinde mimari bir zaaf derhal göze çarpar:
* $z = 0$ iken, $\sigma(0) = 0.5$ ve $\sigma'(0) = 0.5 \times 0.5 = 0.25$ olur.
* $|z| \to \infty$ iken $\sigma(z) \to 0$ veya $1$ değerine yaklaşır ve türev $\sigma'(z) \to 0$ olur.

Sigmoid fonksiyonunun alabileceği en yüksek türev değeri kesin olarak **$0.25$**'tir. $L$ katmanlı bir ağda, geriye yayılımın zincir kuralı yerel Jacobian türevlerinin geriye doğru çarpılmasını zorunlu kılar:

$$\frac{\partial \mathcal{L}}{\partial z_1} = \frac{\partial \mathcal{L}}{\partial z_L} \prod_{l=1}^{L-1} \left( W_{l+1}^T \text{diag}(\sigma'(z_l)) \right)$$

Ağırlık matrislerinin spektral normunun 1 civarında başlatıldığını varsaysak bile, $L$ aktivasyon katmanından geriye akan gradyan sinyali üstel olarak ufalanır:

$$\left\| \prod_{l=1}^{L} \sigma'(z_l) \right\| \le (0.25)^L$$

Yalnızca 10 katmanlık mütevazı bir ağda dahi $(0.25)^{10} \approx 9.53 \times 10^{-7}$ çarpanı ortaya çıkar. En alt katmanlara ulaşan hata gradyanı yaklaşık milyonda birine iner. Sonuç olarak ilk katmanlardaki ağırlıklar neredeyse hiç güncellenemez ve ağ rastgele başlatıldığı ilk halinde donup kalır.

### Sıfır-Merkezli Olmama Çıkmazı

Sigmoid fonksiyonu ikinci bir yapısal probleme daha sahiptir: çıktıları strictly pozitiftir ($\sigma(z) > 0$).

Çıktısı bir sonraki $l+1$ katmanına giren bir nöron için $h^{(l)} > 0$ olur. Kaybın ilgili ağırlık vektörüne göre gradyanı:

$$\frac{\partial \mathcal{L}}{\partial w_i^{(l+1)}} = \frac{\partial \mathcal{L}}{\partial z_i^{(l+1)}} h^{(l)}$$

$h^{(l)}$ her zaman pozitif olduğundan, $\frac{\partial \mathcal{L}}{\partial w_i^{(l+1)}}$ vektörünün tüm elemanları yukarıdan gelen $\frac{\partial \mathcal{L}}{\partial z_i^{(l+1)}}$ skaler gradyanıyla aynı işareti taşımak zorundadır. Gradyan vektörü yalnızca tamamen pozitif ya da tamamen negatif yöne işaret edebilir; bu durum parametre uzayında gradyan inişini zikzaklar çizen, son derece verimsiz bir yörüngeye hapseder.

### Hiperbolik Tanjant (Tanh)

Sıfır-merkezli olmama problemini gidermek için araştırmacılar $(-\infty, \infty)$ aralığını $(-1, 1)$ aralığına simetrik olarak eşleyen hiperbolik tanjant ($\tanh$) fonksiyonuna yöneldi:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1$$

Türevi şu şekildedir:

$$\tanh'(z) = 1 - \tanh^2(z)$$

Başlangıç noktasında $\tanh'(0) = 1.0$ olması sigmoidin $0.25$ tavanını aşsa da, $\tanh(z)$ fonksiyonu da $|z| > 2$ olduğunda hızla doymaya (saturation) uğrar. Bir nöronun ön-aktivasyon değeri uç kuyruklara kaydığında ($|z| \ge 3 \implies \tanh'(z) \le 0.01$), yerel gradyan sıfırlanır ve derin modellerin ön-eğitimi kilitlenir.

| Aktivasyon | Matematiksel Form | Çıktı Aralığı | Maksimum Türev | Sıfır-Merkezli? | Doyma Riski |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $(0, 1)$ | $0.25$ ($z=0$) | Hayır | Şiddetli ($\vert z \vert > 3$) |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | $1.00$ ($z=0$) | Evet | Şiddetli ($\vert z \vert > 2$) |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | $1.00$ ($z>0$) | Hayır | $z>0$ için sıfır; $z<0$ için mutlak |
| **GELU** | $z \Phi(z)$ | $[-0.17, \infty)$ | $\approx 1.12$ ($z=0.75$) | Sıfıra yakın | Minimal (pürüzsüz kuyruk) |
| **SiLU (Swish)** | $z \sigma(z)$ | $[-0.278, \infty)$ | $\approx 1.10$ ($z=1.28$) | Sıfıra yakın | Minimal (pürüzsüz kuyruk) |

---

## 3. Derin öğrenme rönesansı: ReLU ve ölü nöron patolojisi

2010 yılında Vinod Nair ve Geoffrey Hinton tarafından tanıtılan Düzeltilmiş Doğrusal Birim (ReLU - Rectified Linear Unit), 2010'ların derin öğrenme devrimini hesaplama açısından mümkün kılan kırılma noktası oldu:

$$\text{ReLU}(z) = \max(0, z)$$

Türevi parçalı sabittir:

$$\frac{d}{dz}\text{ReLU}(z) = \begin{cases} 1 & \text{eğer } z > 0 \\ 0 & \text{eğer } z < 0 \end{cases}$$

### ReLU Derin Ağların Önünü Neden Açtı?

1. **Doymayan Gradyan Otoyolu:** Pozitif tüm değerler ($z > 0$) için yerel türev tam olarak $1.0$'dır. Hata gradyanları onlarca katman boyunca hiçbir zayıflamaya uğramadan geriye doğru akar.
2. **Hesaplama Tutumluluğu:** $\max(0, z)$ işlemini hesaplamak için üstel işlem ($e^z$) veya bölme gerekmez. Donanım seviyesinde modern ALU'lar bu işlemi tek bir ikili dallanma veya SIMD bit maskelemesi (`z & ~(z >> 31)`) ile tek bir saat döngüsünde yürütür.
3. **Biyolojik Seyreklik:** Biyolojik beyinlerde nöronların aynı anda yalnızca %1 ila %4'ü ateşlenir. ReLU doğal bir seyreklik üretir: negatif özellikler tam olarak sıfırlanır, bellek tamponları temizlenir ve gereksiz özellik etkileşimleri elenir.

### "Ölü Nöron" (Dying ReLU) Patolojisi

Tüm bu avantajlarına karşın ReLU, tek yönlü bir ölüm tuzağı barındırır: **geri döndürülemezlik**.

Bir nöronun ağırlıkları veya bias değeri, eğitim sırasında yüksek bir öğrenme oranı ya da gürültülü bir mini-batch nedeniyle büyük bir gradyan adımıyla güncellensin. Bu güncelleme sonucu nöronun parametreleri öyle bir noktaya gelebilir ki:

$$w_{\text{yeni}}^T x + b_{\text{yeni}} \le 0 \quad \forall x \in \mathcal{D}_{\text{eğitim}}$$

Eğitim verisindeki tüm örnekler için ön-aktivasyon negatif kaldığında, nöronun çıktısı sürekli 0 olur. Bunun doğrudan sonucu olarak geriye yayılım gradyanı da sürekli 0 olur:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \cdot x = 0 \cdot x = \mathbf{0}$$

Bu nörondan geriye bir daha asla gradyan akamaz. Ağırlıkları güncellenemez, bias değeri değişemez ve nöron kalıcı olarak felç olur: bu duruma **ölü nöron (dead neuron)** adı verilir. Kötü başlatılmış veya agresif eğitilen derin ağlarda, ilk epoch bitmeden nöronların %10 ila %40'ı kalıcı olarak ölebilir ve model kapasitesi boşa harcanır.

### Çözüm Arayışları: Leaky ReLU ve PReLU

Gradyanın tamamen kesilmesini engellemek için Andrew Maas ve ekibi (2013) **Leaky ReLU**'yu önerdi:

$$\text{LeakyReLU}(z) = \max(\alpha z, z) = \begin{cases} z & \text{eğer } z > 0 \\ \alpha z & \text{eğer } z \le 0 \end{cases}$$

Burada $\alpha$ genellikle $0.01$ gibi küçük bir sabit katsayıdır. Negatif bölgede gradyan $\alpha$ olarak kaldığından, nöron sıfırın soluna hapsolsa dahi küçük de olsa bir güncelleme sinyali alarak hayata dönebilir. Kaiming He ve ekibi (2015) bu eğimi öğrenilebilir bir parametre haline getirerek **PReLU (Parametric ReLU)** modelini geliştirdi.

---

## 4. Olasılıksal kapılama paradigması: GELU

ReLU sönümlenen gradyanları çözmüş, LeakyReLU ise ölü nöronları yamamış olsa da, her iki fonksiyon da $z = 0$ noktasında sert ve türevlenemeyen bir kırılmaya sahip parçalı doğrusal yaklaşımlardır.

2016 yılında Dan Hendrycks ve Kevin Gimpel, **Gaussian Error Linear Unit (GELU)** aktivasyonunu önererek nöron ateşlemesini deterministik bir eşik kontrolünden çıkarıp **stokastik bir kapılama mekanizmasına** dönüştürdü.

### Kavramsal Sıçrama: Stokastik Düzenlileştirme ile Doğrusalsızlığın Birleşimi

Modern derin öğrenmede iki kavram geleneksel olarak ayrı tutuluyordu:
1. **Deterministik Doğrusalsızlık:** ReLU ($\max(0, x)$) negatif girdileri keser.
2. **Stokastik Düzenlileştirme:** Dropout, nöronları $p$ olasılığıyla rastgele sıfırlar.

GELU bu iki mekanizmayı tek bir çatıda birleştirir. Girdileri rastgele düşürmek veya katı bir ikili kuralla sıfırlamak yerine, $x$ girdisini standart normal dağılıma sahip bir rastgele değişkenin ($X \sim \mathcal{N}(0, 1)$) $x$'ten küçük veya ona eşit olma olasılığıyla çarpar:

$$\text{GELU}(x) = x \cdot P(X \le x) = x \cdot \Phi(x)$$

Burada $\Phi(x)$, standart normal dağılımın kümülatif dağılım fonksiyonudur (CDF):

$$\Phi(x) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-\frac{t^2}{2}} dt = \frac{1}{2} \left[ 1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right]$$

Burada $\text{erf}(z) = \frac{2}{\sqrt{\pi}} \int_0^z e^{-t^2} dt$ Gauss hata fonksiyonudur.

<svg viewBox="0 0 560 196" role="img" aria-label="Bir kapı olarak GELU. Girdi x, standart normal bir değişkenin x'ten küçük ya da eşit olma olasılığıyla ölçeklenir ve çıktı y, x çarpı Phi(x) olur. Gauss birikimli dağılım fonksiyonu Phi 0'dan 1'e yumuşakça yükselir: çok negatif girdiler gradyanı söndürülerek bastırılır, çok pozitif girdiler özdeşlik geçişiyle neredeyse aynen geçer." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ge-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="20" width="96" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="64.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">girdi: x</text>
<rect x="150" y="14" width="260" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="38.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Stokastik Gauss kapısı: x · Φ(x)</text>
<text x="280.0" y="54.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">Φ(x) = P(X ≤ x),  X ~ 𝒩(0, 1)</text>
<rect x="448" y="20" width="96" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="496.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">y = GELU(x)</text>
<line x1="112" y1="42" x2="148" y2="42" marker-end="url(#ge-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="410" y1="42" x2="446" y2="42" marker-end="url(#ge-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="180" y="94" width="200" height="98" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="280" y1="94" x2="280" y2="72" marker-end="url(#ge-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M190 170 H370" style="stroke:var(--c-border);stroke-width:1"/>
<path d="M190.0 169.9 L193.0 169.9 L196.0 169.8 L199.0 169.8 L202.0 169.7 L205.0 169.6 L208.0 169.5 L211.0 169.4 L214.0 169.2 L217.0 168.9 L220.0 168.6 L223.0 168.3 L226.0 167.8 L229.0 167.3 L232.0 166.7 L235.0 166.0 L238.0 165.2 L241.0 164.2 L244.0 163.1 L247.0 161.9 L250.0 160.5 L253.0 159.0 L256.0 157.3 L259.0 155.5 L262.0 153.5 L265.0 151.5 L268.0 149.3 L271.0 147.1 L274.0 144.8 L277.0 142.4 L280.0 140.0 L283.0 137.6 L286.0 135.2 L289.0 132.9 L292.0 130.7 L295.0 128.5 L298.0 126.5 L301.0 124.5 L304.0 122.7 L307.0 121.0 L310.0 119.5 L313.0 118.1 L316.0 116.9 L319.0 115.8 L322.0 114.8 L325.0 114.0 L328.0 113.3 L331.0 112.7 L334.0 112.2 L337.0 111.7 L340.0 111.4 L343.0 111.1 L346.0 110.8 L349.0 110.6 L352.0 110.5 L355.0 110.4 L358.0 110.3 L361.0 110.2 L364.0 110.2 L367.0 110.1 L370.0 110.1" style="fill:none;stroke:var(--c-accent);stroke-width:2"/>
<text x="194" y="114" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Φ(x)</text>
<text x="370" y="186" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">Gauss CDF: x ∈ [-3, 3]</text>
<text x="16" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">x ≪ 0: Φ(x) → 0</text>
<text x="16" y="135" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyan söner, kapı kapalı</text>
<text x="544" y="120" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">x ≫ 0: Φ(x) → 1</text>
<text x="544" y="135" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">özdeşlik: GELU(x) ≈ x</text>
</svg>

Bu kurgu geometriye büyük üstünlükler sağlar:
* $x$ büyük ve pozitif olduğunda ($x > 2$), $\Phi(x) \to 1$ olur ve $\text{GELU}(x) \to x$ (ReLU'nun pozitif kimliğine yaklaşır).
* $x$ büyük ve negatif olduğunda ($x < -2$), $\Phi(x) \to 0$ olur ve $\text{GELU}(x) \to 0$ (ReLU gibi sıfıra yaklaşır).
* Sıfırın çevresindeki kritik geçiş bölgesinde GELU **pürüzsüz, monoton olmayan ve sonsuz kez türevlenebilir** ($C^\infty$) bir eğridir. Sıfırın altına hafifçe dalarak $x \approx -0.7517$ noktasında yaklaşık $-0.170$ minimum değerine ulaşır.

### Analitik Türev

Çarpım kuralı ile $\text{GELU}(x)$'in $x$'e göre türevi alındığında:

$$\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \cdot \Phi'(x) = \Phi(x) + x \cdot \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$

$x = 0$ noktasında türevin aldığı değere dikkat edin:
$$\frac{d}{dx}\text{GELU}(0) = \Phi(0) + 0 = 0.5000$$

Sıfır noktasındaki alt-gradyanı belirsiz olan ReLU'nun aksine GELU, başlangıç noktasında tam $0.5$ eğimle pürüzsüzce akar; bu yumuşak eğrilik Transformer mimarilerinin çok daha kararlı ve hızlı yakınsamasını sağlar.

### Hızlı Tanh Yaklaşımı

İlk nesil GPU'larda Gauss hata fonksiyonunu ($\text{erf}$) hesaplamak donanımsal olarak maliyetliydi. Hendrycks ve Gimpel, hiperbolik tanjanta dayalı son derece hassas bir polinom yaklaşımı geliştirdi:

$$\text{GELU}(x) \approx 0.5x \left( 1 + \tanh\left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)$$

Bu formülasyon orijinal BERT, GPT-2, GPT-3 ve ViT (Vision Transformer) modellerinde temel standart olarak yerleşti. Modern donanımlarda doğrudan hata fonksiyonu çekirdekleri bulunsa da, PyTorch ve Hugging Face kütüphaneleri geriye dönük kararlılık için `gelu_pytorch_tanh` desteğini sürdürmektedir (Gemma 2 ve Gemma 3 modellerinde olduğu gibi).

---

## 5. Modern LLM standardı: Gated Linear Units ve SwiGLU

2017 yılında Google Brain araştırmacıları (Prajit Ramachandran, Barret Zoph, Quoc V. Le), pekiştirmeli öğrenme tabanlı bir mimari arama algoritmasıyla binlerce matematiksel aday arasından en iyi aktivasyon fonksiyonunu taradı. Bu arama **Swish** fonksiyonunu ortaya çıkardı (literatürde **SiLU** — Sigmoid Linear Unit olarak da bilinir):

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

$\beta = 1$ için:

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

SiLU, GELU'nun pürüzsüz ve monoton olmayan kıvrımını paylaşır ($x \approx -1.28$ noktasında $-0.278$ çukuruna iner), ancak Gauss CDF'si yerine hesaplaması daha sade olan lojistik sigmoidi kullanır.

### Çift Doğrusal Kapılama Devrimi (GLU)

Eşzamanlı olarak Yann Dauphin ve ekibi (2016), dil modellerinde **Gated Linear Units (GLU)** mimarisinin başarısını kanıtladı.

Standart bir Çok Katmanlı Algılayıcıda (MLP/FFN) girdiler doğrusal olarak genişletilir, eleman bazlı aktivasyondan geçirilir ve tekrar daraltılır:

$$\text{FFN}_{\text{standart}}(x) = \sigma(x W_1 + b_1) W_2 + b_2$$

GLU bu tek projeksiyonu, çıktıları birbiriyle **çarpımsal** olarak etkileşen iki paralel doğrusal projeksiyonla değiştirir:

$$\text{GLU}(x, W, V) = \sigma(x W) \odot (x V)$$

Burada $\odot$ Hadamard (eleman bazlı) çarpımıdır.
* Birinci kol ($x W$), $0$ ile $1$ arasında değerler üreten sürekli ve yumuşak bir **kapı (gate)** işlevi görür.
* İkinci kol ($x V$), dönüştürülmemiş **doğrusal sinyali (genlik)** taşır.

### Shazeer'in SwiGLU Mimarisi

2020 yılında Noam Shazeer *"GLU Variants Improve Transformer"* makalesinde, GLU'daki klasik sigmoid kapısını modern aktivasyonlarla sistematik olarak değiştirdi: ReGLU (ReLU ile), GeGLU (GELU ile) ve **SwiGLU** (Swish/SiLU ile):

$$\text{SwiGLU}(x, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}) = \left( \text{Swish}(x W_{\text{gate}}) \odot (x W_{\text{up}}) \right) W_{\text{down}}$$

Burada:
* $W_{\text{gate}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$: Girdiyi kapılama alt uzayına yansıtır.
* $W_{\text{up}} \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$: Girdiyi kısıtlanmamış özellik alt uzayına yansıtır.
* $W_{\text{down}} \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$: Kapılanmış temsili tekrar model boyutuna daraltır.

<svg viewBox="0 0 560 196" role="img" aria-label="SwiGLU ileri besleme bloğu. d_model boyutlu girdi tensörü x iki kez yansıtılır: W_gate kolu SiLU/Swish aktivasyonuyla dinamik kapıyı üretir; W_up kolu doğrusal özellik sinyalini taşır. İki kol eleman bazında Hadamard çarpımıyla buluşur ve W_down tensörü d_model boyutuna geri indirir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="70" width="92" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="62.0" y="90.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">x</text>
<text x="62.0" y="106.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">(d_model)</text>
<rect x="140" y="14" width="110" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="195.0" y="36.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_gate</text>
<rect x="274" y="14" width="120" height="36" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="334.0" y="36.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">SiLU / Swish</text>
<rect x="140" y="138" width="254" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="267.0" y="160.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_up (doğrusal genlik)</text>
<path d="M108 94 H124 V32 H138" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M124 94 V156 H138" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="250" y1="32" x2="272" y2="32" marker-end="url(#sw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="436" cy="94" r="18" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="436" y="100" text-anchor="middle" style="fill:var(--c-text);font-size:18px">⊙</text>
<path d="M394 32 H436 V74" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M394 156 H436 V114" marker-end="url(#sw-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="460" y="70" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hadamard çarpımı</text>
<rect x="476" y="76" width="68" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="510.0" y="98.5" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">W_down</text>
<line x1="454" y1="94" x2="474" y2="94" marker-end="url(#sw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="476" y="134" width="68" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="510.0" y="154.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">y</text>
<text x="510.0" y="170.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">(d_model)</text>
<line x1="510" y1="112" x2="510" y2="132" marker-end="url(#sw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="140" y="76" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kapı kolu: Swish(x W_gate)</text>
<text x="140" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">doğrusal değer: x W_up</text>
</svg>

### Parametre Bütçesi Dengesi ($\frac{8}{3} d_{\text{model}}$)

Standart bir 2 katmanlı FFN iki matris içerir:
* $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$
* $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$
* Toplam parametre (bias hariç) $= 2 \times d_{\text{model}} \times d_{\text{ff}}$.
* Orijinal Transformer'da (Vaswani et al., 2017) $d_{\text{ff}} = 4 d_{\text{model}}$ seçildiğinden parametre sayısı $8 d_{\text{model}}^2$ olur.

SwiGLU ise **üç** ayrı projeksiyon matrisi gerektirir ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$). Eğer $d_{\text{ff}}$ yine $4 d_{\text{model}}$ olarak kalsaydı, parametre sayısı $3 \times d_{\text{model}} \times 4 d_{\text{model}} = 12 d_{\text{model}}^2$ değerine fırlar ve FLOP yükü %50 artardı.

Parametre ve hesaplama denkliğini korumak için Shazeer, ara gizli katman boyutunu yeniden ölçeklendirdi:

$$3 \times (d_{\text{model}} \times d_{\text{ff}}) \approx 2 \times (d_{\text{model}} \times 4 d_{\text{model}})$$

$$d_{\text{ff}} = \frac{8}{3} d_{\text{model}} \approx 2.667 d_{\text{model}}$$

Üretim modellerinde bu ara boyut, GPU Tensor Çekirdeklerinin bellek hizalamasını en üst düzeye çıkarmak için 256 veya 64'ün en yakın katına yuvarlanır:
* **LLaMA 3 8B:** $d_{\text{model}} = 4096 \implies \frac{8}{3} \times 4096 \approx 10922.6 \to \mathbf{14336}$ (kapasite artışı için yukarı yuvarlanmıştır).
* **Mistral 7B:** $d_{\text{model}} = 4096 \to d_{\text{ff}} = \mathbf{14336}$.
* **DeepSeek-V3:** Yönlendirilmiş MoE ve Paylaşımlı Uzmanların tamamında SwiGLU kullanır.

SwiGLU'nun dil modellerinde GELU ve ReLU'ya karşı mutlak üstünlük sağlamasının arkasındaki sır, çarpımsal kapılamanın **özellik seçiciliği üzerinde dinamik kontrol** sağlamasıdır. Sabit bir fonksiyonla tüm özellikleri aynı kalıba sokmak yerine bir kol, mevcut token bağlamına göre hangi bilginin bastırılacağını veya güçlendirileceğini anlık olarak belirler.

---

## 6. Adım adım elle sayısal tensör yürüyüşü

Formülleri zihinde somutlaştırmak için iki aşamalı, elle adım adım hesaplanan bir sayısal yürüyüş gerçekleştirelim:
1. **Bölüm A:** Negatif, sıfır ve pozitif noktalardan oluşan 1D vektörde skaler aktivasyonlar ve türevler.
2. **Bölüm B:** Küçük boyutlu matrislerle uçtan uca 2D SwiGLU katmanı ileri beslemesi.

### Bölüm A: 1D Eleman Bazlı Skaler Yürüyüş

Üç kritik noktayı test eden bir girdi vektörü alalım:

$$x = \begin{bmatrix} -1.5 & 0.0 & 2.0 \end{bmatrix}$$

Matematiksel formülleri adım adım uygulayalım:

#### Nokta 1: $x = -1.5$
* **Sigmoid:** $\sigma(-1.5) = \frac{1}{1 + e^{1.5}} = \frac{1}{1 + 4.481689} = \mathbf{0.182426}$
  * Türev: $\sigma'(-1.5) = 0.182426 \times (1 - 0.182426) = \mathbf{0.149146}$
* **Tanh:** $\tanh(-1.5) = \frac{e^{-1.5} - e^{1.5}}{e^{-1.5} + e^{1.5}} = \frac{0.223130 - 4.481689}{0.223130 + 4.481689} = \mathbf{-0.905148}$
  * Türev: $1 - (-0.905148)^2 = 1 - 0.819293 = \mathbf{0.180707}$
* **ReLU:** $\max(0, -1.5) = \mathbf{0.000000}$
  * Türev: $\mathbf{0.000000}$ *(Ölü nöron bölgesi)*
* **LeakyReLU ($\alpha=0.01$):** $0.01 \times (-1.5) = \mathbf{-0.015000}$
  * Türev: $\mathbf{0.010000}$
* **GELU (Analitik):** $\Phi(-1.5) = 0.5 \times [1 + \text{erf}(-1.5 / \sqrt{2})] = 0.5 \times [1 - 0.866386] = 0.066807$
  * Değer: $-1.5 \times 0.066807 = \mathbf{-0.100211}$
  * Türev: $\Phi(-1.5) + (-1.5) \times \frac{1}{\sqrt{2\pi}} e^{-(-1.5)^2/2} = 0.066807 - 1.5 \times 0.129518 = \mathbf{-0.127469}$
* **SiLU (Swish):** $-1.5 \times \sigma(-1.5) = -1.5 \times 0.182426 = \mathbf{-0.273638}$
  * Türev: $0.182426 \times [1 - 1.5 \times (1 - 0.182426)] = 0.182426 \times [1 - 1.226361] = \mathbf{-0.041294}$

#### Nokta 2: $x = 0.0$
* **Sigmoid:** $\sigma(0) = \mathbf{0.500000}$ | Türev: $0.5 \times 0.5 = \mathbf{0.250000}$
* **Tanh:** $\tanh(0) = \mathbf{0.000000}$ | Türev: $1 - 0^2 = \mathbf{1.000000}$
* **ReLU:** $\max(0, 0) = \mathbf{0.000000}$ | Türev: $\mathbf{0.000000}$
* **GELU (Analitik):** $0 \times \Phi(0) = \mathbf{0.000000}$ | Türev: $\Phi(0) + 0 = \mathbf{0.500000}$
* **SiLU (Swish):** $0 \times \sigma(0) = \mathbf{0.000000}$ | Türev: $\sigma(0) + 0 = \mathbf{0.500000}$

#### Nokta 3: $x = 2.0$
* **Sigmoid:** $\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135335} = \mathbf{0.880797}$ | Türev: $0.880797 \times (1 - 0.880797) = \mathbf{0.104994}$
* **Tanh:** $\tanh(2) = \mathbf{0.964028}$ | Türev: $1 - (0.964028)^2 = \mathbf{0.070651}$
* **ReLU:** $\max(0, 2) = \mathbf{2.000000}$ | Türev: $\mathbf{1.000000}$
* **GELU (Analitik):** $\Phi(2) = 0.977250 \implies 2 \times 0.977250 = \mathbf{1.954500}$ | Türev: $0.977250 + 2 \times 0.053991 = \mathbf{1.085232}$
* **SiLU (Swish):** $2 \times \sigma(2) = 2 \times 0.880797 = \mathbf{1.761594}$ | Türev: $0.880797 \times [1 + 2 \times 0.119203] = \mathbf{1.090784}$

| Girdi ($x$) | Aktivasyon Fonksiyonu | İleri Değer ($y$) | Geri Gradyan ($\frac{dy}{dx}$) | Kritik Mühendislik Davranışı |
| :--- | :--- | :--- | :--- | :--- |
| **$-1.5$** | Sigmoid | $0.182426$ | $0.149146$ | Pozitif yanlılık, baskılanmış gradyan |
| | Tanh | $-0.905148$ | $0.180707$ | Belirgin doyma başlangıcı |
| | ReLU | $0.000000$ | $0.000000$ | **Ölü nöron (gradyan = 0)** |
| | LeakyReLU | $-0.015000$ | $0.010000$ | %1'lik kurtarma gradyanını korur |
| | GELU (Analitik) | $-0.100211$ | $-0.127469$ | Pürüzsüz negatif çukur, canlı gradyan |
| | SiLU / Swish | $-0.273638$ | $-0.041294$ | Monoton olmayan yumuşak dip |
| **$0.0$** | Sigmoid | $0.500000$ | $0.250000$ | Maksimum gradyan bile sadece 0.25 |
| | Tanh | $0.000000$ | $1.000000$ | Sıfır-merkezli, birim eğim |
| | ReLU | $0.000000$ | $0.000000$ | Sınır noktasında türev süreksizliği |
| | GELU (Analitik) | $0.000000$ | $0.500000$ | Pürüzsüz 0.5 eğim |
| | SiLU / Swish | $0.000000$ | $0.500000$ | Pürüzsüz 0.5 eğim |
| **$2.0$** | Sigmoid | $0.880797$ | $0.104994$ | Doymuş bölge; gradyan ufalanır |
| | Tanh | $0.964028$ | $0.070651$ | İleri doyma; gradyan sıfırlanır |
| | ReLU | $2.000000$ | $1.000000$ | Kusursuz birim gradyan otoyolu |
| | GELU (Analitik) | $1.954500$ | $1.085232$ | Birime yakın hafif eğrilik takviyesi |
| | SiLU / Swish | $1.761594$ | $1.090784$ | Dinamik genlik takviyeli birim eğim |

```text
x = -1.5 Noktasında Skaler Türevler:
  ReLU:      0.000000  <-- Sinyal tamamen silindi.
  LeakyReLU: 0.010000  <-- Küçük bir 0.01 sızıntı korundu.
  GELU:     -0.127469  <-- Bilgilendirici negatif gradyan dinamikleri korur.
  SiLU:     -0.041294  <-- Negatif eğrilik gradyanı geriye bilgi taşır.
```

---

### Bölüm B: Tam SwiGLU 2D Katman Yürüyüşü

Şimdi 2 boyutlu bir vektör ve küçük boyutlu projeksiyon matrisleri ($d_{\text{in}} = 2, d_{\text{ff}} = 2, d_{\text{out}} = 2$) üzerinden eksiksiz bir SwiGLU katmanı yürütelim:

$$x = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}$$

Ağırlık matrisleri:

$$W_{\text{gate}} = \begin{bmatrix} 1.0 & -1.0 \\ 0.5 & 2.0 \end{bmatrix}, \quad W_{\text{up}} = \begin{bmatrix} 0.5 & 1.0 \\ -1.0 & 0.5 \end{bmatrix}, \quad W_{\text{down}} = \begin{bmatrix} 1.0 & 0.5 \\ 0.0 & 1.0 \end{bmatrix}$$

#### Adım 1: Gate ve Up Doğrusal Projeksiyonları

$$z_{\text{gate}} = x W_{\text{gate}} = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix} \begin{bmatrix} 1.0 & -1.0 \\ 0.5 & 2.0 \end{bmatrix}$$

$$z_{\text{gate}}[0] = (1.0)(1.0) + (-0.5)(0.5) = 1.0 - 0.25 = \mathbf{0.75}$$

$$z_{\text{gate}}[1] = (1.0)(-1.0) + (-0.5)(2.0) = -1.0 - 1.0 = \mathbf{-2.00}$$

$$z_{\text{gate}} = \begin{bmatrix} 0.75 & -2.00 \end{bmatrix}$$

Benzer şekilde $z_{\text{up}}$:

$$z_{\text{up}} = x W_{\text{up}} = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix} \begin{bmatrix} 0.5 & 1.0 \\ -1.0 & 0.5 \end{bmatrix}$$

$$z_{\text{up}}[0] = (1.0)(0.5) + (-0.5)(-1.0) = 0.5 + 0.5 = \mathbf{1.00}$$

$$z_{\text{up}}[1] = (1.0)(1.0) + (-0.5)(0.5) = 1.0 - 0.25 = \mathbf{0.75}$$

$$z_{\text{up}} = \begin{bmatrix} 1.00 & 0.75 \end{bmatrix}$$

#### Adım 2: Gate Projeksiyonuna SiLU Aktivasyonu Uygulama

$$\text{SiLU}(z) = z \cdot \sigma(z)$$

$z_{\text{gate}}[0] = 0.75$ için:
$$\sigma(0.75) = \frac{1}{1 + e^{-0.75}} = \frac{1}{1 + 0.472367} = 0.679179$$
$$\text{SiLU}(0.75) = 0.75 \times 0.679179 = \mathbf{0.509384}$$

$z_{\text{gate}}[1] = -2.00$ için:
$$\sigma(-2.00) = \frac{1}{1 + e^{2.00}} = \frac{1}{1 + 7.389056} = 0.119203$$
$$\text{SiLU}(-2.00) = -2.00 \times 0.119203 = \mathbf{-0.238406}$$

$$\text{gate\_act} = \begin{bmatrix} 0.509384 & -0.238406 \end{bmatrix}$$

#### Adım 3: Hadamard (Eleman Bazlı) Kapılama Etkileşimi

$$h = \text{gate\_act} \odot z_{\text{up}}$$

$$h[0] = 0.509384 \times 1.00 = \mathbf{0.509384}$$

$$h[1] = -0.238406 \times 0.75 = \mathbf{-0.178804}$$

$$h = \begin{bmatrix} 0.509384 & -0.178804 \end{bmatrix}$$

1. boyutun negatif kapı değeriyle hem zayıflatılıp hem ters çevrildiğine, 0. boyutun ise hafif bir sönümlemeyle geçtiğine dikkat edin.

#### Adım 4: Down-Projeksiyonu

$$\text{out} = h W_{\text{down}} = \begin{bmatrix} 0.509384 & -0.178804 \end{bmatrix} \begin{bmatrix} 1.0 & 0.5 \\ 0.0 & 1.0 \end{bmatrix}$$

$$\text{out}[0] = (0.509384)(1.0) + (-0.178804)(0.0) = \mathbf{0.509384}$$

$$\text{out}[1] = (0.509384)(0.5) + (-0.178804)(1.0) = 0.254692 - 0.178804 = \mathbf{0.075888}$$

$$\text{out} = \begin{bmatrix} 0.509384 & 0.075888 \end{bmatrix}$$

```text
Oyuncak SwiGLU İleri Besleme Özeti:
  Girdi Vektörü x:           [ 1.000000, -0.500000]
  Gate Projeksiyonu z_gate:  [ 0.750000, -2.000000]
  Up Projeksiyonu z_up:      [ 1.000000,  0.750000]
  SiLU Gate Aktivasyonu:     [ 0.509384, -0.238406]
  Hadamard Çarpımı h:        [ 0.509384, -0.178804]
  Nihai Çıktı Vektörü:       [ 0.509384,  0.075888]
```

---

## 7. İki mühendislik gözü: Eğitim ve çıkarım dinamikleri

Üretim sistemlerinde aktivasyon fonksiyonları seçilirken matematiksel zarafet, eğitim ve çıkarım aşamalarındaki somut donanım kısıtlarıyla dengelenmek zorundadır.

| Mühendislik Boyutu | Eğitim Rejimi (Ön-Eğitim / Fine-Tuning) | Çıkarım Rejimi (Sunum / Üretim) |
| :--- | :--- | :--- |
| **Baskın İş Yükü** | **Hesaplama ve Aktivasyon Belleği Kısıtlı** | **Bellek Bant Genişliği Kısıtlı (Decode Sırasında)** |
| **Aktivasyon Belleği** | Geri yayılım için ara tensörler ($z_{\text{gate}}, z_{\text{up}}, \text{act}$) VRAM'de saklanır | Geçicidir; ara SRAM yazmaçları anında serbest bırakılır |
| **Aritmetik Yoğunluk** | Yüksek batch boyutu ($B \times S \times D$) sayesinde yüksek FLOP/bayt oranı | Prefill: GEMM (Hesaplama kısıtlı); Decode: GEMV (Bant genişliği kısıtlı) |
| **Sayısal Format** | Bfloat16 / FP32 ana ağırlıklar (gradyan zincirlerinde taşmaları önler) | FP16, BF16 veya FP8/INT4 kuantize edilmiş matris ağırlıkları |
| **Donanım Darboğazı** | HBM VRAM kapasitesi ve dağıtık düğümler arası all-reduce bant genişliği | Ağırlık yükleme sırasında HBM'den SRAM'e bellek aktarım hızı |

### Eğitim Gözü: Aktivasyon Belleği Baskısı

Standart ReLU veya GELU MLP'lerinde ileri geçiş sırasında her token için geriye yayılımda kullanılmak üzere iki tensörün saklanması yeterlidir:
1. Ön-aktivasyon tensörü $z_1 = x W_1$ ($\sigma'(z_1)$ hesabı için).
2. Katman girdisi $x$ ($\frac{\partial \mathcal{L}}{\partial W_1} = x^T dZ_1$ hesabı için).

**SwiGLU** mimarisinde ise geriye yayılım bir Hadamard çarpımı üzerinden akar:

$$\frac{\partial \mathcal{L}}{\partial z_{\text{up}}} = dH \odot \text{SiLU}(z_{\text{gate}})$$

$$\frac{\partial \mathcal{L}}{\partial z_{\text{gate}}} = \left( dH \odot z_{\text{up}} \right) \odot \text{SiLU}'(z_{\text{gate}})$$

Bu durum çalışma zamanının her token için GPU VRAM'inde **üç ayrı ara tensörü** saklamasını gerektirir:
1. $z_{\text{gate}}$ (veya aktifleştirilmiş $\text{SiLU}(z_{\text{gate}})$).
2. $z_{\text{up}}$.
3. İlk projeksiyon girdisi $x$.

4096 bağlam uzunluğunda ve 32 batch boyutunda eğitilen 8 milyar parametreli bir modelde bu durum, geleneksel 2 matrisli FFN'e kıyasla aktivasyon belleğini **1.5 kat** artırır. Bellek yetersizliği (OOM) hatalarından kaçınmak için mühendisler **aktivasyon kontrol noktası (activation checkpointing)** kullanır veya doğrusal projeksiyon, SiLU ve Hadamard çarpımını HBM'e yazmadan doğrudan GPU SRAM üzerinde birleştiren **kaynaşmış (fused) CUDA/Triton çekirdekleri** yazar.

### Çıkarım Gözü: Prefill ve Decode Ayrımı

Çıkarım sırasında ara aktivasyonlar tamamen geçicidir. Çıktı üretildiği anda $z_{\text{gate}}$ ve $z_{\text{up}}$ GPU paylaşımlı belleğinde (SRAM) ezilir ve kalıcı VRAM tüketmez.

Ancak donanım çalışma profili iki üretim fazı arasında kesin bir şekilde ayrışır:
1. **Prefill Fazı (Prompt İşleme):**
   * Tüm prompt token'ları aynı anda paralel işlenir ($N_{\text{ctx}} \times d_{\text{model}}$).
   * Bu işlem Genel Matris Çarpımıdır (**GEMM**) ve yüksek aritmetik yoğunluğa sahiptir. SwiGLU'daki üçüncü matris Tensor Çekirdeklerini tepe verimle doyurur.
2. **Decode Fazı (Adım Adım Üretim):**
   * Her adımda yalnızca tek bir token üretilir ($N = 1$).
   * Matris çarpımı Genel Matris-Vektör çarpımına (**GEMV**) çöker.
   * GPU, tek bir token vektörünü işlemek için her üç matrisi ($W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$) HBM'den çip içi önbelleğe taşımak zorundadır. $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$ formülü sayesinde SwiGLU, klasik $4 d_{\text{model}}$ FFN ile tamamen aynı sayıda bayt transfer eder ve bellek bant genişliğine takılan gecikmeyi sabit tutar.

---

## 8. Python ve PyTorch ile doğrulama

Bölüm 6'da elle hesaplanan tüm skaler ve tensör değerlerini doğrulamak amacıyla aşağıdaki bağımsız Python scripti temel matematik kütüphanesini kullanarak fonksiyonları sıfırdan türetmekte ve yerel PyTorch işlemleriyle karşılaştırmaktadır:

```python
import math
import torch
import torch.nn.functional as F

# 1. Standart matematik kütüphanesi ile sıfırdan türetim
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)

def tanh_fn(x):
    return math.tanh(x)

def tanh_grad(x):
    t = math.tanh(x)
    return 1.0 - t * t

def relu(x):
    return max(0.0, x)

def relu_grad(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_grad(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def phi_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def gelu_exact(x):
    return x * phi_cdf(x)

def gelu_exact_grad(x):
    pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)
    return phi_cdf(x) + x * pdf

def silu(x):
    return x * sigmoid(x)

def silu_grad(x):
    s = sigmoid(x)
    return s + x * s * (1.0 - s)

# Skaler doğrulama
test_points = [-1.5, 0.0, 2.0]
print("=== 1D ELEMAN BAZLI AKTİVASYON DOĞRULAMASI ===")
for x in test_points:
    print(f"\n--- x = {x:.1f} Noktasında Değerlendirme ---")
    print(f"Sigmoid:       val = {sigmoid(x):.6f}, grad = {sigmoid_grad(x):.6f}")
    print(f"Tanh:          val = {tanh_fn(x):.6f}, grad = {tanh_grad(x):.6f}")
    print(f"ReLU:          val = {relu(x):.6f}, grad = {relu_grad(x):.6f}")
    print(f"LeakyReLU:     val = {leaky_relu(x):.6f}, grad = {leaky_relu_grad(x):.6f}")
    print(f"GELU (exact):  val = {gelu_exact(x):.6f}, grad = {gelu_exact_grad(x):.6f}")
    print(f"SiLU / Swish:  val = {silu(x):.6f}, grad = {silu_grad(x):.6f}")

# 2. SwiGLU ileri besleme doğrulaması
print("\n=== OYUNCAK SWIGLU SAYISAL DOĞRULAMASI ===")
x_vec = [1.0, -0.5]
W_gate = [[1.0, -1.0], [0.5, 2.0]]
W_up   = [[0.5,  1.0], [-1.0, 0.5]]
W_down = [[1.0,  0.5], [0.0, 1.0]]

# Elle Adım 1: Doğrusal projeksiyonlar
z_gate = [x_vec[0]*W_gate[0][0] + x_vec[1]*W_gate[1][0],
          x_vec[0]*W_gate[0][1] + x_vec[1]*W_gate[1][1]]
z_up   = [x_vec[0]*W_up[0][0] + x_vec[1]*W_up[1][0],
          x_vec[0]*W_up[0][1] + x_vec[1]*W_up[1][1]]

# Elle Adım 2: Kapılama
gate_act = [silu(z_gate[0]), silu(z_gate[1])]
h = [gate_act[0] * z_up[0], gate_act[1] * z_up[1]]

# Elle Adım 3: Down-projeksiyonu
out = [h[0]*W_down[0][0] + h[1]*W_down[1][0],
       h[0]*W_down[0][1] + h[1]*W_down[1][1]]

print(f"Hesaplanan z_gate:   [{z_gate[0]:.6f}, {z_gate[1]:.6f}]")
print(f"Hesaplanan z_up:     [{z_up[0]:.6f}, {z_up[1]:.6f}]")
print(f"Hesaplanan gate_act: [{gate_act[0]:.6f}, {gate_act[1]:.6f}]")
print(f"Hesaplanan h:        [{h[0]:.6f}, {h[1]:.6f}]")
print(f"Hesaplanan out:      [{out[0]:.6f}, {out[1]:.6f}]")

# PyTorch autograd denklik testi
x_tensor = torch.tensor([[1.0, -0.5]], dtype=torch.float32, requires_grad=True)
W_g = torch.tensor([[1.0, -1.0], [0.5, 2.0]], dtype=torch.float32)
W_u = torch.tensor([[0.5,  1.0], [-1.0, 0.5]], dtype=torch.float32)
W_d = torch.tensor([[1.0,  0.5], [0.0, 1.0]], dtype=torch.float32)

torch_gate = x_tensor @ W_g
torch_up = x_tensor @ W_u
torch_h = F.silu(torch_gate) * torch_up
torch_out = torch_h @ W_d
print(f"\nPyTorch Tensör Çıktısı: {torch_out.detach().numpy().round(6).tolist()}")
```

Doğrulama scripti çalıştırıldığında tüm ondalık basamaklarda kusursuz bir sayısal uyum elde edilmektedir:

```text
=== 1D ELEMAN BAZLI AKTİVASYON DOĞRULAMASI ===

--- x = -1.5 Noktasında Değerlendirme ---
Sigmoid:       val = 0.182426, grad = 0.149146
Tanh:          val = -0.905148, grad = 0.180707
ReLU:          val = 0.000000, grad = 0.000000
LeakyReLU:     val = -0.015000, grad = 0.010000
GELU (exact):  val = -0.100211, grad = -0.127469
SiLU / Swish:  val = -0.273638, grad = -0.041294

--- x = 0.0 Noktasında Değerlendirme ---
Sigmoid:       val = 0.500000, grad = 0.250000
Tanh:          val = 0.000000, grad = 1.000000
ReLU:          val = 0.000000, grad = 0.000000
LeakyReLU:     val = 0.000000, grad = 0.010000
GELU (exact):  val = 0.000000, grad = 0.500000
SiLU / Swish:  val = 0.000000, grad = 0.500000

--- x = 2.0 Noktasında Değerlendirme ---
Sigmoid:       val = 0.880797, grad = 0.104994
Tanh:          val = 0.964028, grad = 0.070651
ReLU:          val = 2.000000, grad = 1.000000
LeakyReLU:     val = 2.000000, grad = 1.000000
GELU (exact):  val = 1.954500, grad = 1.085232
SiLU / Swish:  val = 1.761594, grad = 1.090784

=== OYUNCAK SWIGLU SAYISAL DOĞRULAMASI ===
Hesaplanan z_gate:   [0.750000, -2.000000]
Hesaplanan z_up:     [1.000000, 0.750000]
Hesaplanan gate_act: [0.509384, -0.238406]
Hesaplanan h:        [0.509384, -0.178804]
Hesaplanan out:      [0.509384, 0.075888]

PyTorch Tensör Çıktısı: [[0.509384, 0.075888]]
```

---

## Bütün hikâye altı satırda

- Doğrusal-olmayan aktivasyonlar olmadan çok katmanlı ağlar matris birleşme özelliği nedeniyle tek bir regresyon matrisine çöker.
- Sigmoid ve Tanh çıktıları sınırlar ancak kuyruklarda üstel doyma yaratarak gradyanların geriye doğru yok olmasına yol açar.
- ReLU birim eğimi ve sıfıra yakın hesaplama maliyetiyle derin ağları açtı; ancak geri döndürülemez ölü nöron arızasına gebedir.
- GELU Gauss kümülatif dağılımını kullanarak olasılıksal kapılama getirdi; BERT, GPT ve ViT'in pürüzsüz standardı oldu.
- Modern LLM'ler (LLaMA 3, Gemma 2, DeepSeek-V3), dinamik özellik filtrelemesi sağlayan çift doğrusal SwiGLU mimarisinde birleşti.
- SwiGLU gerektirdiği üç matrisi ara katman boyutunu $\frac{8}{3} d_{\text{model}}$ değerine çekerek dengeler; parametre ve FLOP bütçesini eşitler.

---

## Terimler sözlüğü

- **Afin Dönüşüm (Affine Transformation)** — Doğrusal bir dönüşüm ($W x$) ile bir öteleme bias vektörünün ($+ b$) birleşimi; doğrudaşlığı korur.
- **Sönümlenen Gradyan (Vanishing Gradient)** — Kesirli türevlerin ($\le 0.25$) zincir kuralıyla geriye çarpılması sonucu hata sinyalinin ilk katmanlarda yok olması.
- **Ölü Nöron (Dead Neuron)** — Bir ReLU biriminin negatif bias adımı alarak tüm eğitim örneklerinde kalıcı olarak 0 çıktısı ve 0 gradyan üretmesi hali.
- **Evrensel Fonksiyon Yaklaştırma Teoremi** — Tek bir doğrusal-olmayan gizli katmana sahip ileri beslemeli bir ağın her sürekli fonksiyonu yaklaştırabileceğini kanıtlayan teorem.
- **GELU (Gaussian Error Linear Unit)** — Girdiyi standart Gauss kümülatif olasılığıyla ölçekleyen, stokastik düzenlileştirmeyi pürüzsüz eğrilikle birleştiren aktivasyon.
- **Hadamard Çarpımı ($\odot$)** — Aynı boyuttaki iki tensörün eleman bazlı çarpılması işlemi; GLU ve SwiGLU kapılamasının çekirdeğidir.
- **SwiGLU** — Paralel projeksiyonlar üzerinde Swish/SiLU non-lineerliğini kullanan ve modern dil modellerinde standartlaşan Gated Linear Unit türevi.
- **Aktivasyon Kontrol Noktası (Activation Checkpointing)** — GPU VRAM tasarrufu için ara aktivasyonları ileri geçişte silip geri yayılımda anlık tekrar hesaplayan yöntem.

---

## Daha derine inmek için

- [Nair & Hinton (ICML 2010): Rectified Linear Units Improve Restricted Boltzmann Machines](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) — ReLU'yu modern makine öğrenmesine kazandıran kurucu makale.
- [Hendrycks & Gimpel (arXiv:1606.08415, 2016): Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Olasılıksal kapılamanın temel teorisi.
- [Ramachandran, Zoph, & Le (arXiv:1710.05941, 2017): Searching for Activation Functions](https://arxiv.org/abs/1710.05941) — Google Brain'in pekiştirmeli öğrenme ile Swish/SiLU keşfi.
- [Noam Shazeer (arXiv:2002.05202, 2020): GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU, GeGLU ve Transformer FFN kapılamasının mimari kaynağı.
- Bu blogda: [Bir Prompt'un Yolculuğu (7): Transformer Bloğu](post.html?slug=transformer-blogu-derinlemesine) — SwiGLU ve FFN'lerin dil modellerinde nasıl bir ilişkisel anahtar-değer belleği oluşturduğu.
