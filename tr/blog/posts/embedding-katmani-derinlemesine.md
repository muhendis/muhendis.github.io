Bir dil modeline bir cümle verdiğinizde, tokenizer metni parçalar ve
her birine bir tamsayı damgası vurur: `[1054, 492, 281]`. Ancak bir
transformer tamsayılarla düşünemez; sinir ağlarının anlayabildiği tek
dil, sürekli vektör uzayları ve matris çarpımlarıdır. İşte **gömme katmanı
(embedding layer)** tam bu sınır çizgisinde durur: ayrık token numaralarını,
modelin üzerinde işlem yapabileceği zengin geometrik koordinatlara
dönüştüren ilk çevirmendir.

İlk bakışta bu katman şaşırtıcı derecede yalındır: prompt içeri girerken
tek bir çarpma bile yapmaz, yalnızca bellekten satır okur (sıfır FLOP).
Ancak bir transformer'ın en büyük ağırlık bloklarından biridir — kimi
modellerde toplam parametrelerin üçte birini tek başına tutar. Üstelik
sadece anlamı değil; kelimelerin cümle içindeki sırasını (pozisyonel
kodlama), sayısal kararlılığı ($\sqrt{d_{\text{model}}}$ ölçeklemesi) ve
çıkarım anındaki çıktı projeksiyonunu (`lm_head`) da bu mekanizma yönetir.

Bu yazı, tamsayıların geometriye dönüştüğü o ilk eşiğin anatomisini
çıkarıyor: lookup table mekaniği ve tensör boyutları, sessizce hatalara
yol açan $\sqrt{d_{\text{model}}}$ ölçekleme tuzağı, mutlak konumlardan
RoPE'a uzanan pozisyonel kodlama evrimi, ön eğitimdeki kayıp sıçramaları
(loss spikes) ve aynı ağırlık matrisinin eğitim ile çıkarım arasındaki
zıt yüzü.

**Bu yazıda**

- [1. Bir LLM neden gömme katmanına ihtiyaç duyar?](#1-bir-llm-neden-gömme-katmanına-ihtiyaç-duyar)
- [2. Arama tablosu mekaniği](#2-arama-tablosu-mekaniği)
- [3. Vektör ölçekleme ve gizlendiği yer](#3-vektör-ölçekleme-ve-gizlendiği-yer)
- [4. Pozisyonel kodlama nasıl evrildi?](#4-pozisyonel-kodlama-nasıl-evrildi)
  - [A. Öğrenilmiş mutlak pozisyon (GPT-2, BERT)](#a-öğrenilmiş-mutlak-pozisyon-gpt-2-bert)
  - [B. Sinüzoidal dalga (Vaswani et al., 2017)](#b-sinüzoidal-dalga-vaswani-et-al-2017)
  - [C. RoPE: Döner pozisyonel gömme (Gemma 3, Llama)](#c-rope-döner-pozisyonel-gömme-gemma-3-llama)
  - [Göreli mesafenin doğal kazanımı](#göreli-mesafenin-doğal-kazanımı)
  - [Üç yöntemin karşılaştırma tablosu](#üç-yöntemin-karşılaştırma-tablosu)
- [5. Kayıp sıçramaları ve WeSaR](#5-kayıp-sıçramaları-ve-wesar)
- [6. Eğitim maliyeti ile sunum maliyeti karşılaştırması](#6-eğitim-maliyeti-ile-sunum-maliyeti-karşılaştırması)
- [7. PyTorch adım adım inceleme](#7-pytorch-adım-adım-inceleme)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Bir LLM neden gömme katmanına ihtiyaç duyar?

Derin öğrenme modelleri matris çarpımları ve sürekli vektör uzayları
üzerinde çalışır. "kedi" metnini doğrudan bir matrisle çarpamazsınız.
Metni bu ağlara aktarmanın üç yolu denenmiştir:

- **Kelime düzeyi (Word-level).** Metni kelimelere bölmek yüz binlerce
  terimlik devasa bir sözlük gerektirir. Sözlükte bulunmayan her yeni
  kelime Out-Of-Vocabulary (OOV) hatası üretir, `[UNK]` token'ına
  indirgenir ve anlamsal içeriğini tamamen kaybeder.
- **Karakter düzeyi (Character-level).** Kelime dağarcığı küçüktür
  (~256 girdi) ve OOV problemi yaşanmaz; ancak diziler aşırı uzar.
  Tek bir harf neredeyse hiçbir anlamsal yük taşımadığından, modelin
  karesel dikkat (attention) penceresi anlamsız parçalarla hızla tükenir.
- **Alt kelime (Subword).** Modern LLM'lerin uzlaştığı dengedir (BPE,
  WordPiece, SentencePiece). Sık kullanılan kelimeler tek parça kalırken,
  nadir kelimeler anlamlı alt köklere ve eklere bölünür. Ayrıntılı
  işleyiş için bu blogdaki
  [Tokenizasyon nasıl çalışır](post.html?slug=tokenizasyon-nasil-calisir)
  rehberine bakabilirsiniz.

Tokenizasyon işlemi bize **token ID** adı verilen tamsayıları verir.
Örneğin `"the capital of united states"` girdisi için Gemma 3 şu diziyi
üretir: `[2, 1437, 5279, 529, 26974, 5022]`. Bu tamsayıların kendi
başlarına bir büyüklükleri, geometrik yönleri ya da birbirlerine
uzaklıkları yoktur: 5279 değerinin 1437'den sayısal olarak büyük olması
semantik açıdan hiçbir anlam taşımaz.

Gömme katmanı, bu ayrık tamsayıları çok boyutlu uzayda anlamsal
koordinatlar olarak işlev gören yoğun (dense) vektörlere dönüştürür.

---

## 2. Arama tablosu mekaniği

> **Gömme katmanı (`embed_tokens`)** = Kelime dağarcığındaki her girdi
> için bir satır tutan eğitilebilir bir tablodur. Görevi girdiyi
> matematiksel olarak dönüştürmek değil, her tamsayıyı işaret ettiği
> satırla *değiştirmektir* — aritmetik bir işlem yapmaktan çok bir
> sözlüğü sayfa numarasından açmaya benzer.

`embed_tokens`, $V \times d_{\text{model}}$ boyutunda bir ağırlık
matrisidir; burada $V$ kelime dağarcığı boyutu (vocabulary size),
$d_{\text{model}}$ ise gizli katman boyutudur (hidden dimension):

```mermaid
flowchart LR
    subgraph Girdi["Girdi Tensörü"]
        A["Token ID Dizisi<br>[batch_size, seq_len]<br>örn. [1, 6]"]
    end
    subgraph Bellek["Ağırlık Belleği"]
        B["embed_tokens tablosu<br>V × d_model<br>(262.144 × 1.152)"]
    end
    subgraph Cikti["Gömme Çıktısı"]
        C["Yoğun Tensör<br>[batch_size, seq_len, d_model]<br>örn. [1, 6, 1.152]"]
    end
    A -->|"O(1) gather<br>(satır okuma)"| B
    B -->|"0 FLOP<br>(aritmetik yok)"| C
```

- **Arama tablosu ($O(1)$) mantığı.** Bir matris çarpımı yapmak yerine
  `embed_tokens`, gelen token ID'sine karşılık gelen satırı doğrudan
  ağırlık matrisinden çeker — bu işlem bir **gather** operasyonudur; yani
  aritmetik değil, doğrudan bir bellek okumasıdır. Ders kitapları bunu
  one-hot vektör ile $E$ matrisini çarpmak olarak tanımlar; bu matematiksel
  olarak özdeş olsa da hesaplama açısından anlamsızdır, çünkü 262.144
  çarpmanın $d_{\text{model}}$ kadarı hariç tümü sıfırla yapılır.
- **Tensör dönüşümü.** `[batch_size, seq_len]` boyutundaki bir tamsayı
  dizisi, arama işleminden sonra `[batch_size, seq_len, d_model]`
  boyutunda semantik bir tensöre genişler.
- **Model örnekleri.** Gemma-3-1B matrisi $262.144 \times 1.152$
  boyutundadır; Gemma-3-27B modelinde ise $262.208 \times 5.376$ boyutuna
  ulaşır. Aradaki 64 satırlık fark gerçektir — büyük çok modlu
  (multimodal) modeller görüntü token'ları için ek satırlar barındırır.
- **Parametre maliyeti.** $262.144 \times 1.152 \approx 302\text{M}$
  parametre, 1B parametreli bir modelin yaklaşık %30'una denk gelir.
  27B modelinde ise aynı kelime dağarcığı toplam parametrelerin yaklaşık
  %5'ini oluşturur. Geniş kelime dağarcığı yalnızca küçük ölçekli
  modellerde ciddi bir VRAM baskısı yaratır.
- **Sıradan ve bağlamdan bağımsız.** Aynı ID her seferinde aynı vektörü
  döndürür. 5279 numaralı satır "river bank" (nehir kıyısı) ve "bank loan"
  (banka kredisi) ifadelerinde tamamen aynıdır; anlam ayrımını burada
  gömme katmanı değil, sonraki dikkat (attention) katmanları çözer.
- **`padding_idx`.** Toplu işlem (batching) eşit uzunluklar gerektirir,
  bu nedenle kısa diziler doldurma (pad) token'ı ile tamamlanır. Gemma
  bu durumu `padding_idx=config.pad_token_id` ile yönetir; bu parametre
  ilgili satırı gradyan güncellemelerinden muaf tutarak dolgu vektörünün
  eğitim boyunca sabit kalmasını sağlar.

---

## 3. Vektör ölçekleme ve gizlendiği yer

Gemma modelinde ve orijinal Transformer mimarisinde, gömme tablosundan
çıkan vektörler ilk bloğa ulaşmadan önce $\sqrt{d_{\text{model}}}$ ile
çarpılır (Gemma-3-1B için $\sqrt{1152} \approx 33,94$):

$$\mathbf{x}_{\text{ölçekli}} = \mathbf{x}_{\text{gömme}} \times \sqrt{d_{\text{model}}}$$

Şöyle okuyun: *Tablodan okunan ham satır vektörü, model boyunca sabit
kalan tek bir skaler katsayı ile çarpılır.*

- $\mathbf{x}_{\text{gömme}}$: Gömme tablosundan doğrudan çekilen ham satır
  vektörü.
- $\sqrt{d_{\text{model}}}$: Gizli katman boyutunun karekökü olan sabit
  çarpan (Gemma-3-1B'de $\sqrt{1152} \approx 33,94$).
- $\mathbf{x}_{\text{ölçekli}}$: İlk transformer bloğunun artık akışına
  (residual stream) teslim edilen ölçeklenmiş girdi vektörü.

Bu sayı nereden geliyor ve neden gereklidir? Tabloda depolanan tipik bir
ağırlık başlangıçta `0,02` civarındadır (Gemma `initializer_range = 0.02`
kullanır). Adım adım hesaplarsak:

```text
d_model               = 1152
ölçek katsayısı       = √1152 = 33,94
ham değer             = 0,02
ölçeklenmiş değer     = 0,02 × 33,94 = 0,68
```

Satırdaki tüm sayılar aynı 33,94 katsayısıyla çarpıldığından vektörün
uzaydaki yönü değişmez; yalnızca boyu (Öklid normu) 33,94 katına çıkar.

Bunun temel nedeni ağırlık bağlama (weight tying) yöntemidir (Bölüm 6).
Tek bir matris hem girdi tablosu hem de çıktı projeksiyonu (`lm_head`)
olarak görev yapar ve bu iki görev farklı ölçekler talep eder: Çıktı tarafı
bir softmax katmanını beslediği için ağırlıklarının küçük başlaması
şarttır; aksi takdirde softmax doygunluğa ulaşır ve gradyanlar sıfırlanır.
Ancak bu küçük değerler, girdi tarafındaki vektör normlarının transformer
bloğunun beklediği birim varyans seviyesinin çok altında kalmasına yol
açar. Sabit $\sqrt{d_{\text{model}}}$ katsayısı bu iki karşıt gereksinimi
birbirine bağlar. *Attention Is All You Need* makalesinin 3.4 numaralı
bölümü ağırlık paylaşımı ile $\sqrt{d_{\text{model}}}$ faktörünü aynı
cümlede tanımlar.

**PyTorch tuzağı.** Gemma mimarisinde bu çarpım model omurgasının genel
`forward` akışında değil, gömme modülünün kendi sınıfı içinde gerçekleşir:

```python
class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)
```

Dolayısıyla `embed_tokens(ids)` çağrısı **zaten ölçeklenmiş** vektör
döndürürken, standart `F.embedding(ids, weight)` fonksiyonu **ölçeklenmemiş**
ham değerleri verir. Dahası `embed_scale`, model kontrol noktasında
(checkpoint) yer almayan kalıcı olmayan bir arabellektir (non-persistent
buffer). Ağırlıkları safça yükleyip bu ölçek katsayısını gözden kaçıran
özel bir implementasyon hata vermeden çalışır fakat model performansını
sessizce çökertir.

Llama tarzı modeller ise ağırlık bağlama ve gömme ölçeklemesi yapmaz;
bunun yerine her blok girişindeki RMSNorm katmanının normalizasyonuna
güvenir.

---

## 4. Pozisyonel kodlama nasıl evrildi?

"Köpek adamı ısırdı" ve "Adam köpeği ısırdı" cümlelerini ele alalım.
Bölüm 2'de gösterildiği gibi, arama işlemi her iki cümle için de tamamen
aynı üç satırı döndürür; tablo sıra bilgisine sahip değildir. Dikkat
mekanizması da bunu tek başına çözemez: Dikkat katmanı token çiftlerini
iç çarpım (dot product) ile puanlar ve iç çarpım işlemi simetriktir
($q \cdot k = k \cdot q$). Hangi kelimenin önce geldiğini ayırt edemez.

Bu nedenle sıra bilgisinin dikkat katmanından önce doğrudan sayıların içine
yazılması gerekir. Buradaki asıl amaç bir token'a sadece "sen 4.517 numaralı
yuvadasın" demek değil, **iki token'ın birbirine ne kadar uzak olduğunu**
(göreli mesafeyi) kodlamaktır.

Bu alanda üç ana yaklaşım geliştirilmiştir:

- **A. Öğrenilmiş (Learned)** — Her pozisyon için ayrı bir satır içeren
  ikinci bir eğitilebilir tablo tutulur ve kelime vektörüne eklenir.
- **B. Sinüzoidal (Sinusoidal)** — Sabit trigonometrik dalga formülleriyle
  pozisyon koordinatları hesaplanır ve kelime vektörüne eklenir.
- **C. RoPE** — Vektör toplama işlemi tamamen terk edilir. Vektör, slot
  indeksi ve boyuta özgü frekansla belirlenen bir açıyla 2D koordinat
  düzlemlerinde döndürülür (rotasyon).

```text
A. Learned Absolute (GPT-2, BERT)   B. Sinusoidal (Vaswani 2017)   C. RoPE (Llama, Gemma 3)
   Vektör Toplama                      Trigonometrik Toplama          Vektör Döndürme
   x_i = TokenEmbed + PosEmbed         x_i = TokenEmbed + PE_pos      x_i = R(Θ, i) · TokenEmbed
   (Sabit tavan: L_max)                (Norm şişmesi / semantik kayma)(Kesin norm koruma, ||x|| = sabit)
```

Bu mekanizmaları üç yaklaşım arasında net biçimde karşılaştırmak için 4
boyutlu temsiller ($d = 4$) üzerinden 6 token'lık somut bir cümleyi izleyelim:

$$\text{Dizi: } [\text{"The"}, \ \text{"dog"}, \ \text{"chased"}, \ \text{"the"}, \ \text{"black"}, \ \text{"cat"}] \implies m \in \{0, 1, 2, 3, 4, 5\}$$

Arama tablosundan çekilen ham kelime gömmeleri şöyle olsun:

- **Slot 0 ("The"):** $\mathbf{x}_{(0)} = [0,80, \ 0,60, \ 0,50, \ 0,50]^T \implies \Vert{}\mathbf{x}_{(0)}\Vert{} = \sqrt{0,80^2 + 0,60^2 + 0,50^2 + 0,50^2} = \sqrt{1,50} \approx 1,2247$
- **Slot 1 ("dog"):** $\mathbf{x}_{(1)} = [0,70, \ 0,10, \ 0,40, \ 0,80]^T \implies \Vert{}\mathbf{x}_{(1)}\Vert{} = \sqrt{0,70^2 + 0,10^2 + 0,40^2 + 0,80^2} = \sqrt{1,30} \approx 1,1402$
- **Slot 2 ("chased"):** $\mathbf{x}_{(2)} = [0,50, \ 0,80, \ 0,30, \ 0,60]^T \implies \Vert{}\mathbf{x}_{(2)}\Vert{} = \sqrt{0,50^2 + 0,80^2 + 0,30^2 + 0,60^2} = \sqrt{1,34} \approx 1,1576$
- **Slot 3 ("the"):** $\mathbf{x}_{(3)} = [0,80, \ 0,60, \ 0,50, \ 0,50]^T \implies \Vert{}\mathbf{x}_{(3)}\Vert{} = \sqrt{0,80^2 + 0,60^2 + 0,50^2 + 0,50^2} = \sqrt{1,50} \approx 1,2247$
- **Slot 4 ("black"):** $\mathbf{x}_{(4)} = [0,60, \ 0,20, \ 0,70, \ 0,30]^T \implies \Vert{}\mathbf{x}_{(4)}\Vert{} = \sqrt{0,60^2 + 0,20^2 + 0,70^2 + 0,30^2} = \sqrt{0,98} \approx 0,9899$
- **Slot 5 ("cat"):** $\mathbf{x}_{(5)} = [0,30, \ 0,90, \ 0,60, \ 0,20]^T \implies \Vert{}\mathbf{x}_{(5)}\Vert{} = \sqrt{0,30^2 + 0,90^2 + 0,60^2 + 0,20^2} = \sqrt{1,30} \approx 1,1402$

### A. Öğrenilmiş mutlak pozisyon (GPT-2, BERT)

Parametre belleğinde $L_{\max} \times d_{\text{model}}$ boyutunda eğitilebilir
ikinci bir matris saklanır. Her token temsili, kelime vektörü ile pozisyon
vektörünün eleman düzeyinde toplamıdır:

$$\mathbf{x}_i = \text{TokenEmbed}(w_i) + \text{PosEmbed}(i)$$

Eğitim gradyanlarının öğrenilmiş pozisyon satırlarını şu değerlere getirdiğini
varsayalım:
- $\text{PosEmbed}(0) = [0,00, \ 0,20, \ 0,10, \ 0,00]$
- $\text{PosEmbed}(1) = [0,20, \ -0,10, \ 0,00, \ 0,10]$
- $\text{PosEmbed}(2) = [-0,10, \ 0,10, \ 0,20, \ -0,10]$
- $\text{PosEmbed}(3) = [0,10, \ 0,00, \ -0,10, \ 0,20]$
- $\text{PosEmbed}(4) = [0,00, \ -0,10, \ 0,10, \ 0,10]$
- $\text{PosEmbed}(5) = [-0,10, \ 0,20, \ -0,10, \ 0,00]$

Kelime temsillerini pozisyon vektörleriyle toplayalım:

```text
Slot 0 ("The"):
  TokenEmbed("The") = [ 0,80   0,60   0,50   0,50 ]
  PosEmbed(0)       = [ 0,00   0,20   0,10   0,00 ]
  x_0 (toplam)      = [ 0,80   0,80   0,60   0,50 ]  ──> Norm: 1,37  (1,22'den saptı)

Slot 1 ("dog"):
  TokenEmbed("dog") = [ 0,70   0,10   0,40   0,80 ]
  PosEmbed(1)       = [ 0,20  -0,10   0,00   0,10 ]
  x_1 (toplam)      = [ 0,90   0,00   0,40   0,90 ]  ──> Norm: 1,33  (1,14'ten saptı)

Slot 2 ("chased"):
  TokenEmbed("cha") = [ 0,50   0,80   0,30   0,60 ]
  PosEmbed(2)       = [-0,10   0,10   0,20  -0,10 ]
  x_2 (toplam)      = [ 0,40   0,90   0,50   0,50 ]  ──> Norm: 1,21  (1,16'dan saptı)

Slot 3 ("the"):
  TokenEmbed("the") = [ 0,80   0,60   0,50   0,50 ]
  PosEmbed(3)       = [ 0,10   0,00  -0,10   0,20 ]
  x_3 (toplam)      = [ 0,90   0,60   0,40   0,70 ]  ──> Norm: 1,35  (1,22'den saptı)

Slot 4 ("black"):
  TokenEmbed("bla") = [ 0,60   0,20   0,70   0,30 ]
  PosEmbed(4)       = [ 0,00  -0,10   0,10   0,10 ]
  x_4 (toplam)      = [ 0,60   0,10   0,80   0,40 ]  ──> Norm: 1,08  (0,99'dan saptı)

Slot 5 ("cat"):
  TokenEmbed("cat") = [ 0,30   0,90   0,60   0,20 ]
  PosEmbed(5)       = [-0,10   0,20  -0,10   0,00 ]
  x_5 (toplam)      = [ 0,20   1,10   0,50   0,20 ]  ──> Norm: 1,24  (1,14'ten saptı)
```

- **Dezavantaj 1 (Sert bağlam tavanı):** Model 2.048 yuva ile eğitildiyse,
  ağırlık matrisinde 2.049. satır yoktur. Model, ilklendirilmemiş yeni
  parametreler eklemeden $L_{\max}$ ötesindeki dizileri işleyemez.
- **Dezavantaj 2 (Yapısal göreli mesafe farkındalığı yok):** "dog" (Slot 1) ile
  "cat" (Slot 5) arasındaki 4 adımlık mesafe ($5 - 1 = 4$), 101 ve 105.
  yuvalarda geçen aynı 4 adımlık mesafeden tamamen bağımsız olarak öğrenilir.

### B. Sinüzoidal dalga (Vaswani et al., 2017)

Pozisyon koordinatları geometrik frekans serileri kullanılarak analitik
biçimde hesaplanır:

$$PE_{(pos, 2j)} = \sin\left(\frac{pos}{10000^{2j/d}}\right), \quad PE_{(pos, 2j+1)} = \cos\left(\frac{pos}{10000^{2j/d}}\right)$$

Burada $pos \in \{0, 1, 2, 3, 4, 5\}$, boyut indeksi $j \in \{0, 1\}$ ve model boyutu $d = 4$'tür:
- $j = 0$ için (0 ve 1. kanallar): $\text{bölen} = 10000^0 = 1,0 \implies \text{frekans} = 1,0\text{ rad/adım}$
- $j = 1$ için (2 ve 3. kanallar): $\text{bölen} = 10000^{2/4} = 100,0 \implies \text{frekans} = 0,01\text{ rad/adım}$

Her yuva için dalga koordinatları:

```text
Slot 0 (pos = 0):
  PE_0 = [ sin(0,0), cos(0,0), sin(0,00), cos(0,00) ]
       ≈ [ 0,0000,   1,0000,   0,0000,   1,0000   ]

Slot 1 (pos = 1):
  PE_1 = [ sin(1,0), cos(1,0), sin(0,01), cos(0,01) ]
       ≈ [ 0,8415,   0,5403,   0,0100,   1,0000   ]

Slot 2 (pos = 2):
  PE_2 = [ sin(2,0), cos(2,0), sin(0,02), cos(0,02) ]
       ≈ [ 0,9093,  -0,4161,   0,0200,   0,9998   ]

Slot 3 (pos = 3):
  PE_3 = [ sin(3,0), cos(3,0), sin(0,03), cos(0,03) ]
       ≈ [ 0,1411,  -0,9900,   0,0300,   0,9996   ]

Slot 4 (pos = 4):
  PE_4 = [ sin(4,0), cos(4,0), sin(0,04), cos(0,04) ]
       ≈ [-0,7568,  -0,6536,   0,0400,   0,9992   ]

Slot 5 (pos = 5):
  PE_5 = [ sin(5,0), cos(5,0), sin(0,05), cos(0,05) ]
       ≈ [-0,9589,   0,2837,   0,0500,   0,9988   ]
```

Vektör toplama işlemi uygulandığında ($\mathbf{x}_m = \text{TokenEmbed}(w_m) + PE_m$):

```text
Slot 0 ("The"):
  x_0 (toplam) = [ 0,80 + 0,0000, 0,60 + 1,0000, 0,50 + 0,0000, 0,50 + 1,0000 ]
               = [ 0,8000,        1,6000,        0,5000,        1,5000 ]      ──> Norm: 2,39  (+%95 şişti)

Slot 1 ("dog"):
  x_1 (toplam) = [ 0,70 + 0,8415, 0,10 + 0,5403, 0,40 + 0,0100, 0,80 + 1,0000 ]
               = [ 1,5415,        0,6403,        0,4100,        1,8000 ]      ──> Norm: 2,49  (+%118 şişti)

Slot 2 ("chased"):
  x_2 (toplam) = [ 0,50 + 0,9093, 0,80 - 0,4161, 0,30 + 0,0200, 0,60 + 0,9998 ]
               = [ 1,4093,        0,3839,        0,3200,        1,5998 ]      ──> Norm: 2,19  (+%89 şişti)

Slot 3 ("the"):
  x_3 (toplam) = [ 0,80 + 0,1411, 0,60 - 0,9900, 0,50 + 0,0300, 0,50 + 0,9996 ]
               = [ 0,9411,       -0,3900,        0,5300,        1,4996 ]      ──> Norm: 1,89  (+%54 şişti)

Slot 4 ("black"):
  x_4 (toplam) = [ 0,60 - 0,7568, 0,20 - 0,6536, 0,70 + 0,0400, 0,30 + 0,9992 ]
               = [-0,1568,       -0,4536,        0,7400,        1,2992 ]      ──> Norm: 1,57  (+%59 şişti)

Slot 5 ("cat"):
  x_5 (toplam) = [ 0,30 - 0,9589, 0,90 + 0,2837, 0,60 + 0,0500, 0,20 + 0,9988 ]
               = [-0,6589,        1,1837,        0,6500,        1,1988 ]      ──> Norm: 1,92  (+%69 şişti)
```

- **Dezavantaj (Ağır semantik bozulma ve norm kayması):** Vektör toplama işlemi
  vektör boylarını dramatik biçimde deforme eder. Birebir aynı sözlük vektörüne
  sahip olan "The" (Slot 0) ve "the" (Slot 3) kelimeleri tamamen farklı normlara
  ($2,39$ ve $1,89$) ulaşır; temel anlamsal sinyal cümlenin neresinde geçtiğine
  bağlı olarak bozulur.

### C. RoPE: Döner pozisyonel gömme (Gemma 3, Llama)

Öğrenilmiş ve sinüzoidal yaklaşımlar (A ve B yöntemleri) pozisyon bilgisini
vektör toplama ($\mathbf{x} + \mathbf{p}$) işlemiyle aktarır. Ancak gördüğümüz
gibi iki vektörü toplamak, kelimenin orijinal anlam vektörünün boyunu (normunu)
$1,22$'den $2,39$'a kadar şişirir; bu durum dikkat mekanizmasındaki iç
çarpımları yapay olarak büyüterek softmax dengesini bozar.

**RoPE'un temel çıkış noktası:** Bir vektörün büyüklüğünü (normunu) kesinlikle
değiştirmeden, sadece cümledeki sırasına göre uzaydaki yönünü değiştirebilir
miyiz? Evet: **Vektörü döndürerek (rotasyon)!** Masada duran 10 cm'lik bir
kalemi kendi ekseni etrafında kaç derece çevirirseniz çevirin, boyu daima 10 cm
kalır. RoPE, kelime vektörlerini uzatıp kısaltmak yerine pozisyonlarına göre
belirli açılarla döndürür.

$$\mathbf{x}_m = \mathbf{R}_{\Theta, m}^{d} \cdot \text{TokenEmbed}(w_m)$$

Şöyle okuyun: *Kelime tablosundan okunan ham anlamsal vektör, $m$. pozisyona ve
boyut frekanslarına göre oluşturulan ortogonal bir rotasyon matrisi ile çarpılır;
vektörün boyu korunurken yönü pozisyona göre döner.*

Bu formüldeki her terimin somut anlamı ve mimarideki görevi şöyledir:
- **$\text{TokenEmbed}(w_m)$:** Sözlük tablosundan okunan ham kelime vektörü
  ($d$ boyutlu). Pozisyon bilgisi içermez; "The" kelimesi cümlenin neresinde
  geçerse geçsin sözlükten aynı ham sayılarla çıkar.
- **$m$:** Token'ın cümle içindeki sıra numarası / yuva indeksidir ($m = 0, 1, 2,
  3, \dots$). RoPE'ta bu indeks, fiziksel sistemlerdeki "ayrık zaman adımı" gibi
  işler.
- **$\Theta = \{\theta_1, \theta_2, \dots, \theta_{d/2}\}$:** Her koordinat
  çiftine tahsis edilen temel açısal frekanslar (dönüş hızları) kümesidir.
- **$\mathbf{R}_{\Theta, m}^{d}$:** $m$. pozisyondaki token için hesaplanan
  $d \times d$ boyutundaki blok-köşegen döndürme matrisidir.
- **$\mathbf{x}_m$:** Döndürülmüş, hem anlamsal içeriği hem de pozisyon bilgisini
  aynı anda taşıyan nihai $d$ boyutlu vektördür.

**Neden $d$ boyutlu uzay $d/2$ adet iki boyutlu (2D) alt uzaya bölünür?**
128 boyutlu bir vektörü tek bir kütle gibi 128 boyutta döndürmek matematiksel
olarak muazzam karmaşık tensörler gerektirir. RoPE bunu dâhiyane bir yöntemle
çözer: $d$ boyutlu vektörü ikişerli koordinat çiftlerine böler:
$$(x_1, x_2), \ (x_3, x_4), \ \dots, \ (x_{d-1}, x_d)$$
Her koordinat çifti, bir saatin kadranı veya iki boyutlu bir pusula yüzeyi gibi
kendi 2D düzleminde bağımsız bir nokta oluşturur. 2D düzlemde bir noktayı
orijin etrafında $\phi$ açısıyla döndürmek ise lise geometrisidir:
$$\tilde{x}_1 = x_1 \cos\phi - x_2 \sin\phi$$
$$\tilde{x}_2 = x_1 \sin\phi + x_2 \cos\phi$$

**Normun kesinlikle korunmasının ispatı ($\cos^2\phi + \sin^2\phi = 1$):**
Döndürülmüş bileşenlerin kareleri toplamı hesaplandığında:
$$\tilde{x}_1^2 + \tilde{x}_2^2 = (x_1 \cos\phi - x_2 \sin\phi)^2 + (x_1 \sin\phi + x_2 \cos\phi)^2 = (x_1^2 + x_2^2)(\cos^2\phi + \sin^2\phi) = x_1^2 + x_2^2$$
Trigonometrinin temel özdeşliği sayesinde vektörün uzunluğu bir milimetre bile
şişmez; %0 kayma ile başlangıçtaki Öklid normunda kilitlenir.

**RoPE'ta "frekans" ($\theta_j$) tam olarak nedir ve ne işe yarar?**
Dalga fiziğinde frekans bir dalganın birim zamanda kaç kez tekrarladığını
gösterir. RoPE'ta ise frekans, **"Dizide 1 token ileri gittiğimizde bu 2D
koordinat çifti kaç radyan dönecek?"** sorusunun cevabıdır (açısal hız).
$j$. alt uzayın açısal hızı formülle belirlenir:

$$\theta_j = \text{base}^{-\frac{2(j-1)}{d}} = \frac{1}{\text{base}^{\frac{2(j-1)}{d}}} \quad [\text{token başına radyan}]$$

Formüldeki bileşenler:
- **$\text{base}$ (taban frekans):** Standart olarak $10.000$ seçilir. En hızlı
  dönen boyut ile en yavaş dönen boyut arasındaki hız makasını belirler.
- **$j$:** Koordinat çiftinin sıra numarasıdır ($j = 1, 2, \dots, d/2$).
- **$d$:** Vektör / başlık boyutudur ($d = 4$ ya da üretimde $128$).
- $j$ büyüdükçe (vektörün son kanallarına doğru gidildikçe) paydadaki üs büyür;
  dolayısıyla $\theta_j$ açısal hızı hızla küçülür. Yani **ilk kanallar fırıl
  fırıl dönerken, son kanallar neredeyse yerinden kıpırdamaz.**

**Neden tek bir hız yetmez? Saat ibreleri analojisi (Saniye, Yelkovan, Akrep):**
Eğer tüm boyutlar aynı hızda dönseydi dil modelleri çalışamazdı:
- Sadece hızlı dönselerdi: Birkaç kelime sonra $360^\circ$ tur atıp başa dönerler,
  50 token önceki kelimeyle 1 token önceki kelime aynı yöne bakarak birbirine
  karışırdı (faz çakışması).
- Sadece yavaş dönselerdi: Yan yana duran kelimeler ("dog" ile "chased")
  arasında neredeyse hiçbir açı farkı oluşmaz, model hangisinin önce geldiğini
  anlayamazdı.

Bir kol saatini düşünün:
- **Saniye ibresi (Hızlı frekans, $\theta_1 = 1,0\text{ rad} \approx 57,3^\circ$):**
  Her kelimede büyük bir açı fırlar. Görevi: **Yerel sözdizimini (local syntax)**
  çözmek; yan yana duran kelimeleri keskin biçimde ayırt etmek.
- **Yelkovan (Orta frekanslar):** Cümle ve yan tümce düzeyindeki mesafeleri
  ölçmek.
- **Akrep ibresi (Çok yavaş frekans, $\theta_2 = 0,01\text{ rad} \approx 0,57^\circ$):**
  Her kelimede ancak $0,57^\circ$ kıpırdar. Görevi: **Küresel sırayı (global order)**
  korumak; yüzlerce ve binlerce token sonra bile metnin başı ile sonu arasındaki
  düzeni canlı tutmak.

**Dönüş periyodu (dalga boyu $T_j$) nedir?**
Bir koordinat çiftinin tam bir $360^\circ$ ($2\pi$ radyan) tur atıp başladığı
yöne dönmesi için gereken token sayısıdır:
$$T_j = \frac{2\pi}{\theta_j} = 2\pi \cdot \text{base}^{\frac{2(j-1)}{d}}$$

4 boyutlu modelimiz ($d = 4$, $j \in \{1, 2\}$) ve standart $10.000$ tabanı için:
- **1. Çift ($j=1$, 1–2. kanallar):**
  $$\theta_1 = 10000^{-\frac{2(0)}{4}} = 10000^0 = 1,0 \text{ rad/adım} \quad (\approx 57,3^\circ/\text{adım})$$
  $$T_1 = \frac{2\pi}{1,0} \approx 6,28 \text{ token} \quad (\text{Her 6 kelimede bir tam tur})$$
- **2. Çift ($j=2$, 3–4. kanallar):**
  $$\theta_2 = 10000^{-\frac{2(1)}{4}} = 10000^{-0,5} = \frac{1}{\sqrt{10000}} = 0,01 \text{ rad/adım} \quad (\approx 0,573^\circ/\text{adım})$$
  $$T_2 = \frac{2\pi}{0,01} \approx 628,3 \text{ token} \quad (\text{Tam bir tur 628 token sürer})$$

| Alt Uzay | Açısal Hız ($\theta_j$) | Token Başına Dönüş | Periyot ($T_j = 2\pi/\theta_j$) | Mimari Görevi ("Ne işe yarar?") |
| :--- | :---: | :---: | :---: | :--- |
| **1. Çift ($j=1$, 1–2. kanallar)** | $1,0\text{ rad}$ | $\approx 57,3^\circ$ | $T_1 \approx 6,28\text{ token}$ | **Mikroskop / Saniye İbresi:** Hızlı döner; yan yana duran kelimelerin ("The" $\to$ "dog") sözdizimini ayırır. |
| **2. Çift ($j=2$, 3–4. kanallar)** | $0,01\text{ rad}$ | $\approx 0,57^\circ$ | $T_2 \approx 628,3\text{ token}$ | **Teleskop / Akrep İbresi:** Çok yavaş döner; yüzlerce token boyunca cümlenin başı ile sonu arasındaki sırayı korur. |

**`rope_theta` (taban frekans) neden 10.000'den 500.000 veya 1.000.000'a çıkarıldı?**
LLaMA 1 ve 2 döneminde bağlam penceresi 2.048 veya 4.096 token ile sınırlıydı ve
`base = 10000` yeterliydi. Fakat LLaMA 3 ve Gemma 3 bağlamı 131.072 (128k)
token'a uzattığında, `base = 10000` kullanılsaydı en yavaş akrep ibreleri bile
yüzlerce kez tam tur atarak yönlerini şaşırır ve **faz çakışmasına (phase
wrapping)** düşerdi. Tabanın `1.000.000` yapılması, en yavaş ibrenin periyodunu
milyonlarca token'a uzatarak devasa kitaplarda bile her pozisyona benzersiz bir
açısal damga vurulmasını sağlar.

**Frekans vektöre adım adım nasıl uygulanır? 4 adımlı dönüşüm hattı:**
$m$ pozisyonundaki bir $\mathbf{x} = [x_1, x_2, x_3, x_4]^T$ girdi vektörünü
dönüştürmek için şu 4 adım işletilir:

1. **2D çiftlere ayırma:** Kanallar bağımsız düzlemler halinde gruplanır:
   - 1. Çift: $(x_1, x_2)$
   - 2. Çift: $(x_3, x_4)$
2. **$m$ pozisyonu için açıları ($\phi_j$) hesaplama:**
   Her koordinat çiftinin dönüş açısı, pozisyon numarası ile açısal hızın çarpımıdır:
   $$\phi_1(m) = m \cdot \theta_1 = m \times 1,0 \text{ rad}, \quad \phi_2(m) = m \cdot \theta_2 = m \times 0,01 \text{ rad}$$
3. **Her çifti kendi düzleminde döndürme (2D rotasyon formülü):**
   $$\begin{pmatrix} \tilde{x}_1 \\ \tilde{x}_2 \end{pmatrix} = \begin{pmatrix} \cos(\phi_1) & -\sin(\phi_1) \\ \sin(\phi_1) & \cos(\phi_1) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} x_1 \cos(\phi_1) - x_2 \sin(\phi_1) \\ x_1 \sin(\phi_1) + x_2 \cos(\phi_1) \end{pmatrix}$$
   $$\begin{pmatrix} \tilde{x}_3 \\ \tilde{x}_4 \end{pmatrix} = \begin{pmatrix} \cos(\phi_2) & -\sin(\phi_2) \\ \sin(\phi_2) & \cos(\phi_2) \end{pmatrix} \begin{pmatrix} x_3 \\ x_4 \end{pmatrix} = \begin{pmatrix} x_3 \cos(\phi_2) - x_4 \sin(\phi_2) \\ x_3 \sin(\phi_2) + x_4 \cos(\phi_2) \end{pmatrix}$$
4. **Blok-köşegen rotasyon matrisi $\mathbf{R}_{\Theta, m}^{4}$ ile birleştirme:**
   Tüm 2D rotasyonlar bir arada tek bir matris çarpımı olarak ifade edilebilir:
   $$\mathbf{R}_{\Theta, m}^{4} = \begin{bmatrix} \cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 \\ \sin(m\theta_1) & \cos(m\theta_1) & 0 & 0 \\ 0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) \\ 0 & 0 & \sin(m\theta_2) & \cos(m\theta_2) \end{bmatrix}$$
   Bu matrisin köşegeninde iki adet $2 \times 2$'lik rotasyon bloğu yer alır;
   kalan tüm hücreler sıfırdır. Vektör bu matrisle çarpıldığında her koordinat
   çifti yalnızca kendi düzleminde döner, kanallar birbirine karışmaz.

Bu dönüşümü 6 token'lık dizimize uygulayalım:

**Slot 0 ($m = 0$): "The"**
- Açılar: $\phi_1 = 0 \times 1,0 = 0,0 \text{ rad}$ ($\cos = 1,0000, \sin = 0,0000$), $\phi_2 = 0 \times 0,01 = 0,00 \text{ rad}$ ($\cos = 1,00000, \sin = 0,0000$)
- Dönüşüm:
  $$x_{0, 1} = 0,80(1,0000) - 0,60(0,0000) = 0,8000, \quad x_{0, 2} = 0,80(0,0000) + 0,60(1,0000) = 0,6000$$
  $$x_{0, 3} = 0,50(1,00000) - 0,50(0,0000) = 0,5000, \quad x_{0, 4} = 0,50(0,0000) + 0,50(1,00000) = 0,5000$$
  $$\mathbf{x}_{\text{rotated}(0)} = [0,8000, \ 0,6000, \ 0,5000, \ 0,5000]^T \implies \text{Norm} = \mathbf{1,2247} \quad (\text{Korundu})$$

**Slot 1 ($m = 1$): "dog"**
- Açılar: $\phi_1 = 1 \times 1,0 = 1,0 \text{ rad}$ ($\cos \approx 0,5403, \sin \approx 0,8415$), $\phi_2 = 1 \times 0,01 = 0,01 \text{ rad}$ ($\cos \approx 0,99995, \sin \approx 0,0100$)
- Dönüşüm:
  $$x_{1, 1} = 0,70(0,5403) - 0,10(0,8415) = 0,2941, \quad x_{1, 2} = 0,70(0,8415) + 0,10(0,5403) = 0,6431$$
  $$x_{1, 3} = 0,40(0,99995) - 0,80(0,0100) = 0,3920, \quad x_{1, 4} = 0,40(0,0100) + 0,80(0,99995) = 0,8040$$
  $$\mathbf{x}_{\text{rotated}(1)} = [0,2941, \ 0,6431, \ 0,3920, \ 0,8040]^T \implies \text{Norm} = \mathbf{1,1402} \quad (\text{Korundu})$$

**Slot 2 ($m = 2$): "chased"**
- Açılar: $\phi_1 = 2 \times 1,0 = 2,0 \text{ rad}$ ($\cos \approx -0,4161, \sin \approx 0,9093$), $\phi_2 = 2 \times 0,01 = 0,02 \text{ rad}$ ($\cos \approx 0,99980, \sin \approx 0,0200$)
- Dönüşüm:
  $$x_{2, 1} = 0,50(-0,4161) - 0,80(0,9093) = -0,9355, \quad x_{2, 2} = 0,50(0,9093) + 0,80(-0,4161) = 0,1217$$
  $$x_{2, 3} = 0,30(0,99980) - 0,60(0,0200) = 0,2879, \quad x_{2, 4} = 0,30(0,0200) + 0,60(0,99980) = 0,6059$$
  $$\mathbf{x}_{\text{rotated}(2)} = [-0,9355, \ 0,1217, \ 0,2879, \ 0,6059]^T \implies \text{Norm} = \mathbf{1,1576} \quad (\text{Korundu})$$

**Slot 3 ($m = 3$): "the"**
- Açılar: $\phi_1 = 3 \times 1,0 = 3,0 \text{ rad}$ ($\cos \approx -0,9900, \sin \approx 0,1411$), $\phi_2 = 3 \times 0,01 = 0,03 \text{ rad}$ ($\cos \approx 0,99955, \sin \approx 0,0300$)
- Dönüşüm:
  $$x_{3, 1} = 0,80(-0,9900) - 0,60(0,1411) = -0,8767, \quad x_{3, 2} = 0,80(0,1411) + 0,60(-0,9900) = -0,4811$$
  $$x_{3, 3} = 0,50(0,99955) - 0,50(0,0300) = 0,4848, \quad x_{3, 4} = 0,50(0,0300) + 0,50(0,99955) = 0,5148$$
  $$\mathbf{x}_{\text{rotated}(3)} = [-0,8767, \ -0,4811, \ 0,4848, \ 0,5148]^T \implies \text{Norm} = \mathbf{1,2247} \quad (\text{Korundu})$$

**Slot 4 ($m = 4$): "black"**
- Açılar: $\phi_1 = 4 \times 1,0 = 4,0 \text{ rad}$ ($\cos \approx -0,6536, \sin \approx -0,7568$), $\phi_2 = 4 \times 0,01 = 0,04 \text{ rad}$ ($\cos \approx 0,99920, \sin \approx 0,0400$)
- Dönüşüm:
  $$x_{4, 1} = 0,60(-0,6536) - 0,20(-0,7568) = -0,2408, \quad x_{4, 2} = 0,60(-0,7568) + 0,20(-0,6536) = -0,5848$$
  $$x_{4, 3} = 0,70(0,99920) - 0,30(0,0400) = 0,6874, \quad x_{4, 4} = 0,70(0,0400) + 0,30(0,99920) = 0,3278$$
  $$\mathbf{x}_{\text{rotated}(4)} = [-0,2408, \ -0,5848, \ 0,6874, \ 0,3278]^T \implies \text{Norm} = \mathbf{0,9899} \quad (\text{Korundu})$$

**Slot 5 ($m = 5$): "cat"**
- Açılar: $\phi_1 = 5 \times 1,0 = 5,0 \text{ rad}$ ($\cos \approx 0,2837, \sin \approx -0,9589$), $\phi_2 = 5 \times 0,01 = 0,05 \text{ rad}$ ($\cos \approx 0,99875, \sin \approx 0,0500$)
- Dönüşüm:
  $$x_{5, 1} = 0,30(0,2837) - 0,90(-0,9589) = 0,9481, \quad x_{5, 2} = 0,30(-0,9589) + 0,90(0,2837) = -0,0324$$
  $$x_{5, 3} = 0,60(0,99875) - 0,20(0,0500) = 0,5893, \quad x_{5, 4} = 0,60(0,0500) + 0,20(0,99875) = 0,2297$$
  $$\mathbf{x}_{\text{rotated}(5)} = [0,9481, \ -0,0324, \ 0,5893, \ 0,2297]^T \implies \text{Norm} = \mathbf{1,1402} \quad (\text{Korundu})$$

**Referans PyTorch uygulaması: $O(d)$ sürede RoPE hesaplama.** Üretimdeki LLM'lerde
(LLaMA, Gemma, Mistral) $\mathbf{R}_{\Theta, m}^{d}$ yoğun matris çarpımı hiçbir zaman
açıkça oluşturulmaz. Bunun yerine döndürme işlemi, Query ($Q$) ve Key ($K$)
tensörlerine eleman düzeyinde vektör işlemleriyle doğrudan $O(d)$ karmaşıklığında
uygulanır:

```python
import torch

def get_rotary_position_encoding(
    input: torch.Tensor,
    base: float = 10000.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    [context_length, dimension] şeklindeki tensöre RoPE (Rotary Position Embedding) uygular.
    
    LLaMA ve Hugging Face standardı olan ikiye bölme (split-half) düzenini kullanır:
      - İlk yarı:  input[:, :dimension // 2]
      - İkinci yarı: input[:, dimension // 2:]
    """
    context_length, dimension = input.shape
    assert dimension % 2 == 0, "Boyut (dimension) çift sayı olmalıdır"

    half_dimension = dimension // 2

    # 1. Adım: Her 2D alt uzay için temel açısal frekanslar:
    # theta_j = 1 / (base ** (2 * j / dimension))
    freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (2.0 * freqs_indices / dimension))

    # 2. Adım: Dış çarpımla açı ızgarası: phi(m, j) = m * theta_j
    # Şekil: [context_length, 1] * [1, half_dimension] -> [context_length, half_dimension]
    positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)
    angles = positions * freqs

    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    # 3. Adım: Vektörü iki eşit yarıya ayırma (i. kanalı i + d/2 kanalıyla eşler):
    input_first = input[:, :half_dimension]
    input_second = input[:, half_dimension:]

    # 4. Adım: 2D rotasyon matrisi formülü:
    # [x1']   [cos  -sin] [x1]   [x1 * cos - x2 * sin]
    # [x2'] = [sin   cos] [x2] = [x1 * sin + x2 * cos]
    input_first_rotated = input_first * cos_angles - input_second * sin_angles
    input_second_rotated = input_first * sin_angles + input_second * cos_angles

    # 5. Adım: Döndürülmüş kanalları yeniden birleştirme
    input_rotated = torch.empty_like(input)
    input_rotated[:, :half_dimension] = input_first_rotated
    input_rotated[:, half_dimension:] = input_second_rotated

    return input_rotated

# Gösterim: 6 token'ın tamamında kesin norm koruma testi
torch.manual_seed(1)
context_length = 6
random_input = torch.randn(context_length, 4)

pos_rotary_encodings = get_rotary_position_encoding(random_input)

# Doğrulama: Normlar rotasyondan önce ve sonra birebir aynıdır
print("Orijinal normlar:  ", random_input.norm(dim=-1))
print("Döndürülmüş normlar:", pos_rotary_encodings.norm(dim=-1))
assert torch.allclose(random_input.norm(dim=-1), pos_rotary_encodings.norm(dim=-1))
```

> **Uygulama Notu (Aralıklı / Interleaved ve İkiye Bölme / Split-Half):** Orijinal
> RoFormer makalesi (Su et al.) ardışık boyutları $(x_0, x_1), (x_2, x_3)$ adım
> atlamalı dilimlerle (`input[:, 0::2]` ve `input[:, 1::2]`) eşleştirmiştir. Modern
> üretim kütüphaneleri (LLaMA, Hugging Face `rotate_half`) ise tensörü tam ortadan
> ikiye böler (`[:dimension//2]` ve `[dimension//2:]`). Her iki yöntem de özdeş
> matematiksel özelliklere sahip bağımsız $d/2$ adet 2D düzlem dönüşü gerçekleştirir;
> ancak ortadan ikiye bölme GPU bellek birleştirmesini (coalescing) koruduğundan
> çok daha hızlı çalışır.

### Göreli mesafenin doğal kazanımı

$m$ slotundaki Query ile $n$ slotundaki Key arasındaki dikkat iç çarpımı
hesaplandığında rotasyon matrislerinin ortogonal yapısı
($\mathbf{R}_m^T \mathbf{R}_n = \mathbf{R}_{n-m}$) devreye girer:

$$\langle \mathbf{R}_m Q_m, \ \mathbf{R}_n K_n \rangle = (\mathbf{R}_m Q_m)^T (\mathbf{R}_n K_n) = Q_m^T \mathbf{R}_m^T \mathbf{R}_n K_n = Q_m^T \mathbf{R}_{n-m} K_n$$

Şöyle okuyun: *Döndürülmüş iki vektörün iç çarpımı, mutlak bağlam
pozisyonlarından tamamen bağımsızdır; yalnızca aralarındaki göreli indeks
farkına ($(n - m)$) bağlıdır.*

Bunu cümlemiz üzerinden doğrudan doğrulayabiliriz. "dog" ($m = 1$)
sorgusunun "cat" ($n = 5$) anahtarına dikkatini ele alalım:
- Göreli mesafe: $n - m = 5 - 1 = 4$ yuva.
- Göreli rotasyon operatörü $\mathbf{R}_{5-1} = \mathbf{R}_4$ devreye girer:
  - 1. alt uzay göreli açısı: $\Delta\phi_1 = 4 \times 1,0 = 4,0 \text{ rad}$
  - 2. alt uzay göreli açısı: $\Delta\phi_2 = 4 \times 0,01 = 0,04 \text{ rad}$

Bu token çifti ister $1 \to 5$ indekslerinde, isterse 128k'lık bir belgenin
$10.001 \to 10.005$ indekslerinde yer alsın; $(n - m) = 4$ değişmez kalır.
Mutlak pozisyonlar ($m=1, n=5$) iç çarpım hesabından tamamen düşer; yalnızca
aralarındaki tam 4 adımlık göreli ilişki hesaplanır. Ayrıca yan yana duran
kelime çiftleri — örneğin "The" ($m=0$) $\to$ "dog" ($n=1$) ve "the" ($m=3$) $\to$
"black" ($n=4$) — her ikisi de $(n - m) = 1$ adım farkına sahiptir ve özdeş
pozisyonel rotasyon dönüşümünü paylaşır.

### Üç yöntemin karşılaştırma tablosu

| Metrik / Davranış | A. Learned Absolute | B. Sinusoidal | C. RoPE |
| :--- | :---: | :---: | :---: |
| **Slot 0 ("The") Çıktısı** | `[0,80, 0,80, 0,60, 0,50]` | `[0,80, 1,60, 0,50, 1,50]` | `[0,80, 0,60, 0,50, 0,50]` |
| **Slot 0 Normu** | 1,37 | 2,39 | **1,22 (%0 kayma)** |
| **Slot 1 ("dog") Çıktısı** | `[0,90, 0,00, 0,40, 0,90]` | `[1,54, 0,64, 0,41, 1,80]` | `[0,29, 0,64, 0,39, 0,80]` |
| **Slot 1 Normu** | 1,33 | 2,49 | **1,14 (%0 kayma)** |
| **Slot 2 ("chased") Normu** | 1,21 | 2,19 | **1,16 (%0 kayma)** |
| **Slot 3 ("the") Normu** | 1,35 | 1,89 | **1,22 (%0 kayma)** |
| **Slot 4 ("black") Normu** | 1,08 | 1,57 | **0,99 (%0 kayma)** |
| **Slot 5 ("cat") Normu** | 1,24 | 1,92 | **1,14 (%0 kayma)** |
| **Vektör Boyu Şişti mi?** | Evet | Evet (Aşırı) | **Hayır (Kesinlikle Korundu)** |
| **Göreli Mesafe Doğal mı?** | Hayır | Kısmen | **Evet (Tam $n-m$)** |
| **Eğitilebilir Parametre** | $L_{\max} \times d_{\text{model}}$ | 0 | **0** |

**Vektör boyu neden bozulmamalı (norm koruma).** Dikkat puanı iç çarpımla
hesaplanır: $\mathbf{u} \cdot \mathbf{v} = \Vert{}\mathbf{u}\Vert{} \Vert{}\mathbf{v}\Vert{} \cos(\theta)$.
B yöntemindeki gibi bir vektörün normunun $1,22$'den $2,39$'a şişmesi,
ölçeklenmemiş iç çarpımlarını yaklaşık 4 katına çıkararak softmax'i aşırı
doygunluğa (saturation) iter. Tek bir token dikkat dağılımını tekeline alırken
diğerlerinin gradyanı sıfırlanır. RoPE, vektörleri uzatıp bükmeden geometrik
yüzeylerde döndürür ve normu tertemiz başlangıç büyüklüğünde kilitler.

**Göreli mesafe neden doğal çıkmalı (bağlam genellemesi).** Sözdizimi mutlak
sıraya değil, göreli yer değiştirmeye dayanır: özne ("dog") ile fiil ("chased")
arasındaki 1 yuvalık ilişki, tümce ister 1. sayfada ister 500. sayfada geçsin
aynı kalır. Bu çift farkı ($(n - m)$) modele üç temel yetenek kazandırır:

1. **Kayma değişmezliği (translation invariance):** Dil yapıları kaymaya
   duyarsızdır. $(n - m) = 1$ tıpatıp aynı $\mathbf{R}_1$ dönüşünü ürettiğinden,
   model belgenin neresinde olursa olsun özdeş sözdizimsel dikkati uygular.
2. **Eğitim sınırının ötesine genelleme (extrapolation):** Öğrenilmiş mutlak
   tablolar 2.049. yuvada çöker, çünkü 2.049 indeksi için ağırlık tahsis
   edilmemiştir. RoPE'ta model 100.001 pozisyonunu asla bilinmeyen bir varlık
   olarak görmez; ön eğitimde milyarlarca kez gördüğü tanıdık $(n - m) = 2$
   aralığını değerlendirir.
3. **Doğal frekans sönümlenmesi (decay):** Hızlı dönen boyutlar
   ($\theta_1 = 1,0$) süratle salınarak yerel sözdizimini yalıtır. Yavaş dönen
   boyutlar ($\theta_2 = 0,01$) ise adım başına neredeyse $0,01$ radyan
   dönerek gürültüye dönüşmeden binlerce token boyunca uzun menzilli anlamsal
   bütünlüğü korur.

---

## 5. Kayıp sıçramaları ve WeSaR

Bir **kayıp sıçraması (loss spike)** — ön eğitim sırasında eğitim kaybının
aniden fırlaması ve optimizasyonun diverjans göstermesi — katman ölçekleme
dengesizlikleriyle yakından ilişkilidir.

```text
Artık bağlantı ölçeklemesi (1/sqrt(2N)) ──> Parametre normları küçülür (||W_d|| ≈ 0,002)
                                                          │
                                                          ▼
                               Yüksek bağıl güncelleme oranı (||ΔW|| / ||W||)
                                                          │
                                                          ▼
                                     Aşırı duyarlılık ve LOSS SPIKE
```

**Neden gerçekleşir (norm dengesizliği).** Derin ağlarda patlayan
gradyanları engellemek için artık (residual) dallar derinliğe bağlı
faktörlerle ($1/\sqrt{2N}$) ölçeklenir. Bu durum alt-projeksiyon
matrislerinin ($W_d$, $W_o$) normlarını başlangıçta çok küçük hale
getirir (`0,002`). Güncelleme adımlarını gradyan varyansına göre normalize
eden Adam optimizer'ı altında, parametreler taban büyüklüklerinden
bağımsız olarak yaklaşık aynı büyüklükte (`0,001`) güncellenir:

| Matris Durumu | Tipik Ağırlık Büyüklüğü | Adam Adım Büyüklüğü | Bağıl Değişim ($\Vert \Delta W \Vert / \Vert W \Vert$) |
| :--- | ---: | ---: | ---: |
| Standart başlatma | 0,02 | 0,001 | %5 |
| $1/\sqrt{2N}$ ile küçültülmüş | 0,002 | 0,001 | **%50** |

Tek bir optimizasyon adımında ağırlıkların %50 oranında değişmesi,
katmanlar arası aktivasyon dengesini bozar ve loss spike felaketine yol
açar.

**WeSaR çözümü.** Yeniden Parametreleştirme Yoluyla Ağırlık Ölçekleme
(Weight Scaling as Reparameterization - WeSaR), matrisin gerçek normunu
eğitilebilir bir $\alpha$ skaler katsayısı ile matrisin kendisinden ayırır:

$$\bar{W} = \alpha W, \quad \text{burada } W \sim \mathcal{N}(0, \sigma^2)$$

- $W$: Modelde standart başlatma ölçeğiyle ($\sigma \approx 0,02$) saklanan
  ve Adam tarafından güncellenen ana matris.
- $\alpha$: Matrisin yanında tutulan eğitilebilir tek bir skaler katsayı
  ($\alpha \approx 0,1$).
- $\bar{W}$: İleri geçişte aktivasyonlarla çarpılan efektif küçük ağırlık
  matrisi ($0,1 \times 0,02 = 0,002$).

```text
Geleneksel: doğrudan 0,002 sakla               → W = 0,002
WeSaR:      0,02 sakla, yanında α = 0,1 tut   → W̄ = 0,1 × 0,02 = 0,002
```

İleri geçiş (forward pass) yine $0,002$ efektif değerini görür; ancak Adam
optimizer $0,02$ ölçeğindeki bir matrisi günceller. Bağıl güncelleme oranı
%5 seviyesinde sabit kalarak eğitim kararlılığını korur.

---

## 6. Eğitim maliyeti ile sunum maliyeti karşılaştırması

Gömme katmanı eğitim ve çıkarım (inference) aşamalarında tamamen farklı
mühendislik darboğazları üretir:

| Boyut | Eğitim Profili (Training) | Çıkarım / Sunum Profili (Inference / Serving) |
| :--- | :--- | :--- |
| **Temel Darboğaz** | $E$ üzerindeki gradyan trafiği ve bant genişliği | VRAM bellek yerleşimi ve transfer hızı |
| **Gömme Hesaplama Yükü** | İhmal edilebilir ($O(1)$ gather) | İhmal edilebilir ($O(1)$ gather) |
| **Bağlı Bileşen (`lm_head`)** | Paylaşımlı geri yayılım | Üretilen token başına $2 \times d_{\text{model}} \times V$ FLOP |
| **Gemma-3-1B Üzerindeki Etki** | ~302M parametre + optimizer durumları | ~0,60 GB yerleşik VRAM |
| **Temel Optimizasyonlar** | Seyrek gradyanlar, dondurma (freezing) | Kuantizasyon, sözlük paralelleştirme |

- **Ağırlık Bağlama (Weight Tying).** Gemma 3 modelinde `tie_word_embeddings = True`
  olarak ayarlanmıştır. `embed_tokens.weight` ve `lm_head.weight` bellekte
  tamamen aynı işaretçiyi (`data_ptr()`) paylaşır. Eğitimde birine yapılan
  güncelleme diğerine doğrudan yansır.
- **Eğitim Gradyanları.** Bir eğitim grubu (batch) yalnızca 4.096 farklı
  token içerse bile, PyTorch varsayılan olarak $262.144 \times 1.152$
  boyutundaki tam yoğun gradyan tensörünü oluşturur. Bu da %99'u sıfırlardan
  oluşan büyük bir bellek bant genişliği yükü yaratır.
- **Sunum Hesaplama Yükü.** Çıkarım sırasında `embed_tokens` yalnızca
  indeks tabanlı bir bellek okumasıdır (0 FLOP). Ancak aynı ağırlığı
  paylaşan `lm_head`, üretilen her bir token için tüm kelime dağarcığıyla
  tam bir matris çarpımı yapar:
  - Gemma-3-1B için token başına: $2 \times 1.152 \times 262.144 \approx 0,60\text{ GFLOP}$.
  - Gemma-3-27B için token başına: $2 \times 5.376 \times 262.208 \approx 2,82\text{ GFLOP}$.
  Modern çıkarım motorları bu yükü `VocabParallelEmbedding` ve
  `ParallelLMHead` katmanları ile GPU kümeleri arasında paylaştırır.

---

## 7. PyTorch adım adım inceleme

Gemma-3-1B üzerinde arama işlemi, tensör şekilleri, ölçekleme katsayısı ve
ağırlık bağlama doğrulaması:

```python
import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM

# 1. Model ve tokenizer yükleme
model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = Gemma3ForCausalLM.from_pretrained(model_id)

prompt = "the capital of united states"
tokens = tokenizer.encode(prompt, return_tensors="pt")
# tokens: tensor([[2, 1437, 5279, 529, 26974, 5022]])

# 2. Gömme katmanından geçiş
# Gemma3TextScaledWordEmbedding forward içinde sqrt(d_model) ile çarpar
embed_layer = model.model.embed_tokens
embeddings = embed_layer(tokens)

# 3. Ölçeklenmemiş ham satır okuması (gather)
raw_lookup = torch.nn.functional.embedding(tokens, embed_layer.weight)

print("Tokens şekli         :", tokens.shape)      # torch.Size([1, 6])
print("Gömme çıktısı        :", embeddings.shape)  # torch.Size([1, 6, 1152])
print("Ham tablo okuması    :", raw_lookup.shape)  # torch.Size([1, 6, 1152])

# Ölçekleme faktörünü doğrula: sqrt(1152) ≈ 33.94
ratio = (embeddings.norm() / raw_lookup.norm()).item()
print(f"Norm oranı           : {ratio:.2f}")       # 33.94

# Bellek işaretçisi üzerinden ağırlık bağlamayı doğrula
print("Bağlama Yapılandırması:", model.config.text_config.tie_word_embeddings)
print("Aynı Bellek Alanı     :", embed_layer.weight.data_ptr() == model.lm_head.weight.data_ptr())
```

---

## Bütün hikâye altı satırda

- Gömme katmanı girdi aşamasında matris aritmetiği yapmaz; $O(1)$ maliyetli bir bellek satırı okuması (gather) gerçekleştirir.
- Bu tablo Gemma-3-1B modelinde parametrelerin yaklaşık %30'unu, Gemma-3-27B modelinde ise yalnızca %5'ini oluşturur.
- $\sqrt{d_{\text{model}}}$ faktörü, tied çıkış projeksiyonunun ölçeğini dengelemek için kullanılır ve Gemma'da modülün `forward` metodunun içinde yer alır.
- Modern mimarilerde pozisyon bilgisi bu katmandan ayrılmıştır: RoPE, dikkat katmanı içinde $Q$ ve $K$ vektörlerini döndürerek çalışır.
- Çıkarım (inference) anında gömme tablosu neredeyse hiç işlem gücü harcamaz; ancak bağlı ikizi olan `lm_head`, token başına $2 \times d_{\text{model}} \times V$ işlem yükü üretir.
- Ön eğitimdeki kayıp sıçramaları (loss spikes), optimizasyon hassasiyetini dengeleyen WeSaR gibi yeniden parametreleştirme yöntemleriyle kontrol altına alınabilir.

---

## Terimler sözlüğü

Yazının temel kavramları, sezgi ve mühendislik sonuçlarıyla:

- **gömme katmanı (`embed_tokens`)** — ayrık token ID'lerini ağın işleyebileceği yoğun vektörlere çeviren eğitilebilir tablo. Tıpkı sözlükte sayfa numarasına bakmak gibidir; 1B bir modelde ağırlıkların yaklaşık üçte birini tek başına kaplar.
- **lookup / gather** — satır indeksini bellekten doğrudan çekme işlemi ($O(1)$). One-hot matris çarpımıyla matematiksel olarak aynıdır ama 262.144 satırlık gereksiz sıfır çarpımını ve devasa tensör tahsisini önler.
- **$d_{\text{model}}$** — model boyunca korunan gizli katman vektör genişliği. Gemma-3-1B'de 1.152'dir; modelin anlamsal taşıma kapasitesini belirler.
- **$V$ (kelime dağarcığı boyutu)** — tokenizer tarafından tanımlanan toplam ayrık sembol sayısı. Gemma-3-1B'de 262.144'tür; geniş dağarcık dizileri kısaltır ama küçük modellerde parametre bütçesini zorlar.
- **vektör normu** — bir gömme vektörünün Öklid uzunluğu ($\sqrt{\sum x_i^2}$). Vektörün yönü semantiği, normu ise model içindeki sinyal gücünü temsil eder.
- **artık akış (residual stream)** — katmanlar boyunca akan ana durum tensörü. Her blok bu akıştan okur ve kendi katkısını üzerine toplar.
- **$\sqrt{d_{\text{model}}}$ ölçeklemesi** — başlangıç ağırlık varyansı ile artık akışın beklediği genliği dengeleyen sabit çarpan. Gemma'da checkpoint dışı gizli bir buffer olarak çalıştığından port ederken unutulması en yaygın arızadır.
- **kalıcı olmayan arabellek (non-persistent buffer)** — PyTorch modülünde bulunan fakat `state_dict` içine kaydedilmeyen tensör. Ağırlıkları dosyadakiyle birebir eşleseniz dahi çalışma anında sessizce kaybolabilir.
- **weight tying (ağırlık bağlama)** — girdi gömme tablosu ile çıktı lojit projeksiyon matrisinin aynı ağırlıkları paylaşması. Gemma-3-1B'de ~302M parametre tasarrufu sağlar.
- **`lm_head`** — son gizli durumu kelime olasılık skorlarına (lojiklere) dönüştüren doğrusal projeksiyon katmanı. Gömme tablosunun bağlı ikizidir ve çıkarımda token başına devasa bir matris çarpımı faturası keser.
- **`padding_idx`** — doldurma token'ına karşılık gelen ve gradyan güncellemesi alması engellenen satır indeksi. Dolgu vektörünün rastgele kaymasını önleyerek eğitim kararlılığı sağlar.
- **permütasyon simetrisi** — pozisyon bilgisi olmadan dikkat matris çarpımının token sırasına duyarsız olması durumu. "Köpek adamı ısırdı" ile "Adam köpeği ısırdı" arasındaki farkı yok eder.
- **RoPE (Rotary Position Embedding)** — $Q$ ve $K$ vektörlerini koordinat düzlemlerinde döndürerek pozisyonu kodlayan yöntem. Vektör normunu hiç bozmadan göreli mesafeyi ($n-m$) doğrudan iç çarpıma yansıtır.
- **loss spike (kayıp sıçraması)** — gradyan ve parametre normu dengesizliklerinden kaynaklanan ani eğitim kaybı sapması. Günlerce süren GPU eğitim koşularını tek bir adımda çöpe atabilir.
- **bağıl güncelleme oranı** — bir optimizasyon adımında parametrenin büyüklüğüne göre değişim oranı ($||\Delta W|| / ||W||$). Adam'ın küçük normlu katmanları aşırı güncellemesini ölçen kritik göstergedir.
- **WeSaR** — ağırlıkları skaler bir katsayı ile yeniden parametreleştirerek eğitim kararlılığını artıran mimari yöntem. İleri geçişin gördüğü ölçekle optimizer'ın güncellediği ölçeği birbirinden ayırır.
- **sözlük paralelleştirme (Vocabulary Parallelism)** — çok büyük gömme tablolarını GPU kümeleri arasında paylaştıran tensör paralelliği. Servis motorlarında `VocabParallelEmbedding` katmanı ile VRAM ve gecikmeyi dağıtır.

---

## Daha derine inmek için

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — Orijinal Transformer mimarisi, sinüzoidal kodlama ve $\sqrt{d_{\text{model}}}$ ölçeklemesi.
- Press & Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) (2016) — Ağırlık bağlama (weight tying) yönteminin temelleri.
- Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021) — RoPE döner pozisyonel gömme mekaniği.
- Nishida et al., [Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes](https://arxiv.org/abs/2410.05052) (2024) — WeSaR ve kayıp sıçramalarının analizi.
- Google DeepMind, [Gemma 3 technical report](https://arxiv.org/abs/2503.19786) (2025) — Gemma 3 mimarisi ve çok modlu token genişlemesi.
- Bu blogda: [Tokenizasyon nasıl çalışır](post.html?slug=tokenizasyon-nasil-calisir) — masanın bir önceki adımı: metinden token ID'sine —, [LLM nasıl çalışır](post.html?slug=llm-nasil-calisir) — vektörlerin katmanlar arasındaki yolculuğu — ve [Embedding'ler derinlemesine](post.html?slug=embeddingler-derinlemesine) — anlamsal arama ve vektör uzayları.
