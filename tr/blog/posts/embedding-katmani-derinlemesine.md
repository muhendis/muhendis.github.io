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

<svg viewBox="0 0 560 150" role="img" aria-label="Embedding lookup&#x27;ı. batch&#x27;e dizi uzunluğu boyutlu, örneğin 1&#x27;e 6, token ID girdi tensörü, V&#x27;ye d_model boyutlu embed_tokens ağırlık tablosunu indeksler; Gemma-3-1B&#x27;de 262.144&#x27;e 1.152. Bu adım satır okuyan O(1) bir gather&#x27;dır, sıfır FLOP, aritmetik yok; sonuç batch&#x27;e dizi uzunluğuna d_model boyutlu yoğun bir tensördür, örneğin 1&#x27;e 6&#x27;ya 1.152." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="lk-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="90" y="22" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">girdi tensörü</text>
<rect x="16" y="30" width="148" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="90.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">token ID&#x27;leri</text>
<text x="90.0" y="71.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">[batch, seq_len]</text>
<text x="90.0" y="87.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">örn. [1, 6]</text>
<text x="280" y="22" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">ağırlık belleği</text>
<rect x="206" y="30" width="148" height="74" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">embed_tokens</text>
<text x="280.0" y="71.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">V × d_model</text>
<text x="280.0" y="87.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">262.144 × 1.152</text>
<text x="470" y="22" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">gömme çıktısı</text>
<rect x="396" y="30" width="148" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="470.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">yoğun tensör</text>
<text x="470.0" y="71.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">[batch, seq, d_model]</text>
<text x="470.0" y="87.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">örn. [1, 6, 1.152]</text>
<line x1="164" y1="67" x2="204" y2="67" marker-end="url(#lk-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="184.0" y="122" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">O(1) gather</text>
<text x="184.0" y="136" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">(satır okuma)</text>
<line x1="354" y1="67" x2="394" y2="67" marker-end="url(#lk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="374.0" y="122" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">0 FLOP</text>
<text x="374.0" y="136" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">(aritmetik yok)</text>
</svg>

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

"Köpek adamı ısırdı" ile "adam köpeği ısırdı" cümlelerini ele alalım. Bölüm
2'de gördüğümüz gibi arama tablosu her iki cümle için de sözlükten tıpatıp aynı
üç satır vektörünü çeker; arama tablosunun sıra farkındalığı kesinlikle sıfırdır.
Öz-dikkat (self-attention) mekanizması da bunu tek başına çözemez: dikkat
ağırlıkları iç çarpımlarla hesaplanır ve iç çarpım değişmelidir ($q \cdot k = k \cdot q$). Hangi kelimenin önce, hangisinin sonra geldiğini ayırt edemez.

Dolayısıyla dizi sırası, dikkat katmanı çalışmadan önce doğrudan sayısal
vektörlerin içine enjekte edilmelidir. Amaç sadece token'a "sen 4.517. sıradasın"
etiketi yapıştırmak değil, iki kelimenin birbirinden kaç adım uzakta olduğunu
(göreli mesafeyi) doğal olarak hissettirmektir.

Tarihsel olarak üç farklı paradigma kullanılmıştır:
- **A. Öğrenilmiş mutlak:** Pozisyon başına ikinci bir eğitilebilir ağırlık
  tablosu tutulur ve kelime vektörüyle eleman düzeyinde toplanır.
- **B. Sinüzoidal:** Sabit trigonometrik dalga koordinatları hesaplanır ve
  kelime vektörüyle toplanır.
- **C. RoPE:** Vektör toplamayı tamamen terk eder. Vektörü 2D koordinat
  düzlemlerinde pozisyonuna ve boyut frekansına orantılı açılarla döndürür.

```text
A. Öğrenilmiş Mutlak (GPT-2, BERT)   B. Sinüzoidal (Vaswani 2017)   C. RoPE (Llama, Gemma 3)
   Vektör Toplama                       Trigonometrik Toplama          Vektör Döndürme
   x_i = TokenEmbed + PosEmbed          x_i = TokenEmbed + PE_pos      x_i = R(Θ, i) · TokenEmbed
   (Sert sınır: L_max)                  (Norm şişmesi / kayması)       (Tam norm koruma, ||x|| = sabit)
```

Bu üç mekanizmayı temiz biçimde karşılaştırmak için **8 boyutlu temsiller ($d = 8$)**
üzerinden 6 token'lık somut bir cümleyi adım adım izliyoruz:

$$\text{Dizi: } [\text{"The"}, \ \text{"dog"}, \ \text{"chased"}, \ \text{"the"}, \ \text{"black"}, \ \text{"cat"}] \implies m \in \{0, 1, 2, 3, 4, 5\}$$

Sözlük tablosundan okunan ham kelime vektörleri şunlar olsun:
- **Slot 0 ("The"):** $\mathbf{x}_{(0)} = [0,80, \ 0,60, \ 0,40, \ 0,20, \ 0,50, \ 0,10, \ 0,30, \ 0,20]^T \implies \Vert \mathbf{x}_{(0)}\Vert = \sqrt{1,59} \approx 1,2610$
- **Slot 1 ("dog"):** $\mathbf{x}_{(1)} = [0,70, \ 0,10, \ 0,30, \ 0,50, \ 0,20, \ 0,40, \ 0,60, \ 0,10]^T \implies \Vert \mathbf{x}_{(1)}\Vert = \sqrt{1,41} \approx 1,1874$
- **Slot 2 ("chased"):** $\mathbf{x}_{(2)} = [0,50, \ 0,80, \ 0,20, \ 0,30, \ 0,40, \ 0,20, \ 0,10, \ 0,50]^T \implies \Vert \mathbf{x}_{(2)}\Vert = \sqrt{1,48} \approx 1,2166$
- **Slot 3 ("the"):** $\mathbf{x}_{(3)} = [0,80, \ 0,60, \ 0,40, \ 0,20, \ 0,50, \ 0,10, \ 0,30, \ 0,20]^T \implies \Vert \mathbf{x}_{(3)}\Vert = \sqrt{1,59} \approx 1,2610$
- **Slot 4 ("black"):** $\mathbf{x}_{(4)} = [0,60, \ 0,20, \ 0,50, \ 0,10, \ 0,30, \ 0,60, \ 0,20, \ 0,40]^T \implies \Vert \mathbf{x}_{(4)}\Vert = \sqrt{1,31} \approx 1,1446$
- **Slot 5 ("cat"):** $\mathbf{x}_{(5)} = [0,30, \ 0,90, \ 0,10, \ 0,40, \ 0,50, \ 0,20, \ 0,40, \ 0,10]^T \implies \Vert \mathbf{x}_{(5)}\Vert = \sqrt{1,53} \approx 1,2369$

*(Dikkat ederseniz Slot 0 "The" ve Slot 3 "the" kelimeleri sözlükte aynı satıra denk geldiğinden tamamen özdeş ham vektörlere ve $1,2610$ normuna sahiptir.)*

### A. Öğrenilmiş mutlak pozisyon (GPT-2, BERT)

Parametre belleğinde $L_{\max} \times d_{\text{model}}$ boyutunda eğitilebilir
ikinci bir matris saklanır. Her token temsili, kelime vektörü ile pozisyon
vektörünün eleman düzeyinde toplamıdır:

$$\mathbf{x}_i = \text{TokenEmbed}(w_i) + \text{PosEmbed}(i)$$

Eğitim gradyanlarının öğrenilmiş 8 boyutlu pozisyon satırlarını şu değerlere
getirdiğini varsayalım:
- $\text{PosEmbed}(0) = [0,00, \ 0,20, \ 0,10, \ 0,00, \ 0,10, \ -0,10, \ 0,00, \ 0,10]$
- $\text{PosEmbed}(1) = [0,20, \ -0,10, \ 0,00, \ 0,10, \ -0,10, \ 0,10, \ 0,10, \ 0,00]$
- $\text{PosEmbed}(2) = [-0,10, \ 0,10, \ 0,20, \ -0,10, \ 0,00, \ 0,10, \ -0,10, \ 0,10]$
- $\text{PosEmbed}(3) = [0,10, \ 0,00, \ -0,10, \ 0,20, \ 0,10, \ 0,00, \ 0,10, \ -0,10]$
- $\text{PosEmbed}(4) = [0,00, \ -0,10, \ 0,10, \ 0,10, \ -0,10, \ 0,00, \ 0,10, \ 0,10]$
- $\text{PosEmbed}(5) = [-0,10, \ 0,20, \ -0,10, \ 0,00, \ 0,10, \ -0,10, \ 0,00, \ 0,10]$

Kelime temsillerini pozisyon vektörleriyle toplayalım:

```text
Slot 0 ("The"):
  TokenEmbed("The") = [ 0,80   0,60   0,40   0,20   0,50   0,10   0,30   0,20 ]
  PosEmbed(0)       = [ 0,00   0,20   0,10   0,00   0,10  -0,10   0,00   0,10 ]
  x_0 (toplam)      = [ 0,80   0,80   0,50   0,20   0,60   0,00   0,30   0,30 ]  ──> Norm: 1,45  (1,26'dan saptı)

Slot 1 ("dog"):
  TokenEmbed("dog") = [ 0,70   0,10   0,30   0,50   0,20   0,40   0,60   0,10 ]
  PosEmbed(1)       = [ 0,20  -0,10   0,00   0,10  -0,10   0,10   0,10   0,00 ]
  x_1 (toplam)      = [ 0,90   0,00   0,30   0,60   0,10   0,50   0,70   0,10 ]  ──> Norm: 1,42  (1,19'dan saptı)

Slot 2 ("chased"):
  TokenEmbed("cha") = [ 0,50   0,80   0,20   0,30   0,40   0,20   0,10   0,50 ]
  PosEmbed(2)       = [-0,10   0,10   0,20  -0,10   0,00   0,10  -0,10   0,10 ]
  x_2 (toplam)      = [ 0,40   0,90   0,40   0,20   0,40   0,30   0,00   0,60 ]  ──> Norm: 1,33  (1,22'den saptı)

Slot 3 ("the"):
  TokenEmbed("the") = [ 0,80   0,60   0,40   0,20   0,50   0,10   0,30   0,20 ]
  PosEmbed(3)       = [ 0,10   0,00  -0,10   0,20   0,10   0,00   0,10  -0,10 ]
  x_3 (toplam)      = [ 0,90   0,60   0,30   0,40   0,60   0,10   0,40   0,10 ]  ──> Norm: 1,40  (1,26'dan saptı)

Slot 4 ("black"):
  TokenEmbed("bla") = [ 0,60   0,20   0,50   0,10   0,30   0,60   0,20   0,40 ]
  PosEmbed(4)       = [ 0,00  -0,10   0,10   0,10  -0,10   0,00   0,10   0,10 ]
  x_4 (toplam)      = [ 0,60   0,10   0,60   0,20   0,20   0,60   0,30   0,50 ]  ──> Norm: 1,23  (1,14'ten saptı)

Slot 5 ("cat"):
  TokenEmbed("cat") = [ 0,30   0,90   0,10   0,40   0,50   0,20   0,40   0,10 ]
  PosEmbed(5)       = [-0,10   0,20  -0,10   0,00   0,10  -0,10   0,00   0,10 ]
  x_5 (toplam)      = [ 0,20   1,10   0,00   0,40   0,60   0,10   0,40   0,20 ]  ──> Norm: 1,41  (1,24'ten saptı)
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

Burada $pos \in \{0, 1, 2, 3, 4, 5\}$, çift indeksi $j \in \{0, 1, 2, 3\}$ ve model boyutu $d = 8$'dir:
- $j = 0$ için (0 ve 1. kanallar): $\text{bölen} = 10000^{0/8} = 10000^0 = 1,0 \implies \text{frekans} = 1,0\text{ rad/adım}$
- $j = 1$ için (2 ve 3. kanallar): $\text{bölen} = 10000^{2/8} = 10000^{1/4} = 10,0 \implies \text{frekans} = 0,1\text{ rad/adım}$
- $j = 2$ için (4 ve 5. kanallar): $\text{bölen} = 10000^{4/8} = 10000^{1/2} = 100,0 \implies \text{frekans} = 0,01\text{ rad/adım}$
- $j = 3$ için (6 ve 7. kanallar): $\text{bölen} = 10000^{6/8} = 10000^{3/4} = 1000,0 \implies \text{frekans} = 0,001\text{ rad/adım}$

Her yuva için 8 boyutlu dalga koordinatları:

```text
Slot 0 (pos = 0):
  PE_0 = [ sin(0,0), cos(0,0), sin(0,00), cos(0,00), sin(0,000), cos(0,000), sin(0,0000), cos(0,0000) ]
       ≈ [ 0,0000,   1,0000,   0,0000,   1,0000,   0,0000,    1,0000,    0,0000,     1,0000     ]

Slot 1 (pos = 1):
  PE_1 = [ sin(1,0), cos(1,0), sin(0,10), cos(0,10), sin(0,010), cos(0,010), sin(0,0010), cos(0,0010) ]
       ≈ [ 0,8415,   0,5403,   0,0998,   0,9950,   0,0100,    1,0000,    0,0010,     1,0000     ]

Slot 2 (pos = 2):
  PE_2 = [ sin(2,0), cos(2,0), sin(0,20), cos(0,20), sin(0,020), cos(0,020), sin(0,0020), cos(0,0020) ]
       ≈ [ 0,9093,  -0,4161,   0,1987,   0,9801,   0,0200,    0,9998,    0,0020,     1,0000     ]

Slot 3 (pos = 3):
  PE_3 = [ sin(3,0), cos(3,0), sin(0,30), cos(0,30), sin(0,030), cos(0,030), sin(0,0030), cos(0,0030) ]
       ≈ [ 0,1411,  -0,9900,   0,2955,   0,9553,   0,0300,    0,9996,    0,0030,     1,0000     ]

Slot 4 (pos = 4):
  PE_4 = [ sin(4,0), cos(4,0), sin(0,40), cos(0,40), sin(0,040), cos(0,040), sin(0,0040), cos(0,0040) ]
       ≈ [-0,7568,  -0,6536,   0,3894,   0,9211,   0,0400,    0,9992,    0,0040,     1,0000     ]

Slot 5 (pos = 5):
  PE_5 = [ sin(5,0), cos(5,0), sin(0,50), cos(0,50), sin(0,050), cos(0,050), sin(0,0050), cos(0,0050) ]
       ≈ [-0,9589,   0,2837,   0,4794,   0,8776,   0,0500,    0,9988,    0,0050,     1,0000     ]
```

Vektör toplama işlemi uygulandığında ($\mathbf{x}_m = \text{TokenEmbed}(w_m) + PE_m$):

```text
Slot 0 ("The"):
  x_0 (toplam) = [ 0,80+0,0000, 0,60+1,0000, 0,40+0,0000, 0,20+1,0000, 0,50+0,0000, 0,10+1,0000, 0,30+0,0000, 0,20+1,0000 ]
               = [ 0,8000,      1,6000,      0,4000,      1,2000,      0,5000,      1,1000,      0,3000,      1,2000 ]  ──> Norm: 2,79  (+%121 şişti)

Slot 1 ("dog"):
  x_1 (toplam) = [ 0,70+0,8415, 0,10+0,5403, 0,30+0,0998, 0,50+0,9950, 0,20+0,0100, 0,40+1,0000, 0,60+0,0010, 0,10+1,0000 ]
               = [ 1,5415,      0,6403,      0,3998,      1,4950,      0,2100,      1,4000,      0,6010,      1,1000 ]  ──> Norm: 2,96  (+%149 şişti)

Slot 2 ("chased"):
  x_2 (toplam) = [ 0,50+0,9093, 0,80-0,4161, 0,20+0,1987, 0,30+0,9801, 0,40+0,0200, 0,20+0,9998, 0,10+0,0020, 0,50+1,0000 ]
               = [ 1,4093,      0,3839,      0,3987,      1,2801,      0,4200,      1,1998,      0,1020,      1,5000 ]  ──> Norm: 2,79  (+%130 şişti)

Slot 3 ("the"):
  x_3 (toplam) = [ 0,80+0,1411, 0,60-0,9900, 0,40+0,2955, 0,20+0,9553, 0,50+0,0300, 0,10+0,9996, 0,30+0,0030, 0,20+1,0000 ]
               = [ 0,9411,     -0,3900,      0,6955,      1,1553,      0,5300,      1,0996,      0,3030,      1,2000 ]  ──> Norm: 2,42  (+%92 şişti)

Slot 4 ("black"):
  x_4 (toplam) = [ 0,60-0,7568, 0,20-0,6536, 0,50+0,3894, 0,10+0,9211, 0,30+0,0400, 0,60+0,9992, 0,20+0,0040, 0,40+1,0000 ]
               = [-0,1568,     -0,4536,      0,8894,      1,0211,      0,3400,      1,5992,      0,2040,      1,4000 ]  ──> Norm: 2,60  (+%127 şişti)

Slot 5 ("cat"):
  x_5 (toplam) = [ 0,30-0,9589, 0,90+0,2837, 0,10+0,4794, 0,40+0,8776, 0,50+0,0500, 0,20+0,9988, 0,40+0,0050, 0,10+1,0000 ]
               = [-0,6589,      1,1837,      0,5794,      1,2776,      0,5500,      1,1988,      0,4050,      1,1000 ]  ──> Norm: 2,63  (+%113 şişti)
```

- **Dezavantaj (Ağır semantik bozulma ve norm kayması):** Vektör toplama işlemi
  vektör boylarını dramatik biçimde deforme eder. Birebir aynı sözlük vektörüne
  sahip olan "The" (Slot 0) ve "the" (Slot 3) kelimeleri tamamen farklı normlara
  ($2,79$ ve $2,42$) ulaşır; temel anlamsal sinyal cümlenin neresinde geçtiğine
  bağlı olarak bozulur.

### C. RoPE: Döner pozisyonel gömme (Gemma 3, Llama)

Önceki iki yöntemin (Öğrenilmiş Mutlak ve Sinüzoidal) temel çıkmazını gördük: Pozisyon bilgisini kelime vektörüne **vektör toplama ($\mathbf{x} + \mathbf{p}$)** yoluyla ekliyorlardı. Bu toplama işlemi, kelimenin orijinal anlam vektörünün boyunu (normunu) $1,26$'dan $2,96$'ya kadar (%149'a varan oranda) şişirdi. Dikkat katmanında bu yapay şişme, iç çarpımları fırlatarak softmax dengesini altüst eder.

**RoPE'un dâhiyane çıkış noktası tek bir soruyla özetlenebilir:**
> *"Bir kelimenin anlamını (vektör boyunu) hiç bozmadan, sadece cümledeki sırasını nasıl kodlayabiliriz?"*

**Cevap: Vektöre bir şey ekleme, onu bir pusula iğnesi gibi döndür!**

Masada duran 10 cm uzunluğunda bir pusula iğnesini düşünün. İğneyi masanın üzerinde kaç derece çevirirseniz çevirin, boyu kuruşu kuruşuna 10 cm kalır. İşte **RoPE (Rotary Position Embedding)** tam olarak bunu yapar: Kelime vektörlerinin boyunu (öz anlamını) kesinlikle değiştirmeden, yalnızca uzaydaki yönünü token'ın cümledeki sırasına göre döndürür.

$$\mathbf{x}_m = \mathbf{R}_{\Theta, m}^{d} \cdot \text{TokenEmbed}(w_m)$$

**Formülü sade bir dille okuyalım:**
*Sözlük tablosundan okunan ham kelime vektörü ($\text{TokenEmbed}$), $m$. pozisyon için hesaplanan bir rotasyon (döndürme) matrisi ($\mathbf{R}$) ile çarpılır. Vektörün boyu milimetrik olarak korunurken, yönü kelimenin cümledeki sırasına göre döner.*

Bu formüldeki yapı taşları:
- **$\text{TokenEmbed}(w_m)$:** Sözlük tablosundan çekilen ham anlamsal vektör ($d = 8$ boyutlu). Pozisyon içermez; "The" kelimesi cümlenin neresinde geçerse geçsin aynı ham sayılarla gelir.
- **$m$:** Token'ın cümledeki sıra indeksidir ($m = 0, 1, 2, 3, \dots$). Fiziksel sistemlerdeki "ayrık zaman adımı" gibidir.
- **$\Theta = \{\theta_1, \theta_2, \dots, \theta_{d/2}\}$:** Boyut çiftlerine atanan açısal dönüş hızları kümesidir ($d = 8$ için 4 adet hız).
- **$\mathbf{R}_{\Theta, m}^{d}$:** $m$. yuva için oluşturulan $8 \times 8$ boyutundaki blok-köşegen döndürme matrisidir.
- **$\mathbf{x}_m$:** Boyu hiç bozulmamış, yönüne pozisyon bilgisi kusursuzca mühürlenmiş nihai vektördür.

---

#### 1. Temel Kural: Açı = Pozisyon × Açısal Hız

RoPE'ta bir kelimenin ne kadar döneceği, dönen bir saatin ibresi kadar yalındır:

$$\text{Dönüş Açısı} = \text{Pozisyon} \times \text{Açısal Hız} \implies \phi(m) = m \cdot \theta$$

- **0. pozisyondaki kelime ($m = 0$):** $0 \times \theta = 0^\circ$ döner. İbre tam başlangıç noktasındadır (12 yönü, orijinal doğrultusunda kalır).
- **1. pozisyondaki kelime ($m = 1$):** $1 \times \theta$ kadar ileri döner.
- **2. pozisyondaki kelime ($m = 2$):** $2 \times \theta$ kadar ileri döner.
- **50. pozisyondaki kelime ($m = 50$):** $50 \times \theta$ kadar ileri döner.

Kelime cümlede ne kadar ilerideyse, ibresi o kadar ileri dönmüştür. Kelimenin öz anlamı (vektörün boyu) zerre kadar değişmez; yalnızca yönü cümledeki sırasına göre adım adım kayar.

---

#### 2. İki Boyutlu (2D) Düzlem Hilesi: 8 Boyut Nasıl Döndürülür?

8 boyutlu bir vektörü 8 boyutta tek bir katı cisim gibi döndürmek yerine RoPE bunu **zarif bir parçalama hilesiyle** çözer:
8 boyutlu vektörü ikişerli sayılardan oluşan **4 bağımsız 2D çifte (kadran)** böler:

$$(x_1, x_2), \quad (x_3, x_4), \quad (x_5, x_6), \quad (x_7, x_8)$$

Bunu bir masa üzerinde yan yana duran **4 küçük saat kadranı** gibi hayal edin. Her kadranın üzerinde iki boyutlu bir $(x, y)$ oku vardır. İki boyutlu bir oku merkez etrafında $\phi$ açısıyla döndürmek ise lise geometrisidir:

$$\tilde{x}_1 = x_1 \cos\phi - x_2 \sin\phi$$
$$\tilde{x}_2 = x_1 \sin\phi + x_2 \cos\phi$$

*(Karmaşık sayılarla düşünenler için: Bu işlem, $z = x_1 + i x_2$ sayısını $e^{i\phi}$ ile çarpmaktan ibarettir; Euler özdeşliği $|e^{i\phi}| = 1$ olduğundan büyüklük asla değişmez!)*

**Vektör boyu neden asla bozulmaz? (Matematiksel İspat):**
Döndürülmüş bileşenlerin karelerini topladığımızda:
$$\tilde{x}_1^2 + \tilde{x}_2^2 = (x_1 \cos\phi - x_2 \sin\phi)^2 + (x_1 \sin\phi + x_2 \cos\phi)^2 = (x_1^2 + x_2^2)(\cos^2\phi + \sin^2\phi)$$
Trigonometrinin temel özdeşliği gereği $\cos^2\phi + \sin^2\phi = 1$ olduğundan:
$$\tilde{x}_1^2 + \tilde{x}_2^2 = x_1^2 + x_2^2$$
Vektörün uzunluğunda milimetrik bir sapma bile oluşmaz; 4 çiftin dördünde de norm kayması kesinlikle **%0**'dır!

---

#### 3. Neden Tek Bir Hız Yetmez? Saat İbreleri Analojisi

"Peki 4 kadranın hepsi aynı hızla ($\theta$) mı döner?"
**Kesinlikle hayır.** Bir kol saatini düşünün: Bir saatte neden hem saniye, hem yelkovan, hem de akrep ibresi vardır?

- **Eğer saatte sadece saniye ibresi olsaydı:** İbre her 60 saniyede bir tam tur atıp başa dönerdi. Öğlen 12:00 ile gece 24:00 birbirinin aynısı görünürdü (buna **faz çakışması / phase wrapping** denir).
- **Eğer saatte sadece akrep ibresi olsaydı:** İbre o kadar yavaş kıpırdardı ki, hangi saniyede olduğunuzu asla ayırt edemezdiniz.

RoPE, 8 boyutlu modelimizdeki 4 koordinat çiftine tam olarak bu mantıkla 4 farklı hız ($\theta_j$) atar:

Bu açısal hızlar geometrik frekans serisiyle formüle edilir ($d = 8$ ve $\text{base} = 10.000$ için):

$$\theta_j = \text{base}^{-\frac{2(j-1)}{d}} = \frac{1}{\text{base}^{\frac{2(j-1)}{d}}} \quad [\text{token başına radyan}]$$

Burada $10.000 = 10^4$ olduğundan, $d = 8$ boyutunda 4 çiftin üsleri tam olarak $10$'un katları şeklinde sadeleşir:

- **1. Çift ($j = 1$, kanallar 1–2):**  
  Üs: $\frac{2(1-1)}{8} = \frac{0}{8} = 0 \implies \theta_1 = \frac{1}{10000^0} = \frac{1}{1} = \mathbf{1{,}0\text{ rad/adım}} \ (\approx 57{,}3^\circ)$
- **2. Çift ($j = 2$, kanallar 3–4):**  
  Üs: $\frac{2(2-1)}{8} = \frac{2}{8} = \frac{1}{4} \implies \theta_2 = \frac{1}{10000^{1/4}} = \frac{1}{\sqrt[4]{10^4}} = \frac{1}{10} = \mathbf{0{,}1\text{ rad/adım}} \ (\approx 5{,}73^\circ)$
- **3. Çift ($j = 3$, kanallar 5–6):**  
  Üs: $\frac{2(3-1)}{8} = \frac{4}{8} = \frac{1}{2} \implies \theta_3 = \frac{1}{10000^{1/2}} = \frac{1}{\sqrt{10000}} = \frac{1}{100} = \mathbf{0{,}01\text{ rad/adım}} \ (\approx 0{,}57^\circ)$
- **4. Çift ($j = 4$, kanallar 7–8):**  
  Üs: $\frac{2(4-1)}{8} = \frac{6}{8} = \frac{3}{4} \implies \theta_4 = \frac{1}{10000^{3/4}} = \frac{1}{(\sqrt[4]{10^4})^3} = \frac{1}{10^3} = \frac{1}{1000} = \mathbf{0{,}001\text{ rad/adım}} \ (\approx 0{,}057^\circ)$

Standart $\text{base} = 10.000$ ile 4 çiftin açısal hızları ve tam tur süreleri:

| Alt Uzay | Açısal Hız ($\theta_j$) | Adım Başına Dönüş | Tam Tur Süresi ($T_j = 2\pi/\theta_j$) | İbre Analojisi ve Görevi |
| :--- | :---: | :---: | :---: | :--- |
| **1. Çift ($j=1$, 1–2. kanallar)** | $1,0\text{ rad}$ | $\approx 57,3^\circ$ | $T_1 = 2\pi / 1,0 \approx 6,28\text{ token}$ | **Saniye İbresi (Mikroskop):** Hızlı döner; komşu kelimeleri ("The" $\to$ "dog") keskin biçimde ayırır. |
| **2. Çift ($j=2$, 3–4. kanallar)** | $0,1\text{ rad}$ | $\approx 5,73^\circ$ | $T_2 = 2\pi / 0,1 \approx 62,83\text{ token}$ | **Hızlı Yelkovan:** Yan tümceler ve cümle içi mesafeleri (10–60 token) ölçer. |
| **3. Çift ($j=3$, 5–6. kanallar)** | $0,01\text{ rad}$ | $\approx 0,57^\circ$ | $T_3 = 2\pi / 0,01 \approx 628,3\text{ token}$ | **Yavaş Yelkovan:** Paragraf düzeyindeki ilişkileri korur. |
| **4. Çift ($j=4$, 7–8. kanallar)** | $0,001\text{ rad}$ | $\approx 0,057^\circ$ | $T_4 = 2\pi / 0,001 \approx 6283,2\text{ token}$ | **Akrep İbresi (Teleskop):** Çok yavaş döner; binlerce token boyunca küresel sırayı korur. |

Cümlemizin altı pozisyonunu her çiftten geçirince dört hız gözle görünür hâle gelir:

<svg viewBox="0 0 560 268" role="img" aria-label="RoPE&#x27;nin dört boyut çifti dört saat kadranı olarak çizilmiş; her biri ibresinin 0&#x27;dan 5&#x27;e kadar pozisyonlarda nereyi gösterdiğini işaretler. 1. çift token başına 1 radyan, yaklaşık 57 derece döner ve 5. pozisyona gelindiğinde kadranın büyük kısmını dolaşmıştır. 2. çift token başına 0,1 radyan döner, yaklaşık 29 derece ilerlemiştir. 3. çift 0,01 radyanla neredeyse kıpırdamamıştır, yaklaşık 3 derece. 4. çift 0,001 radyanla donmuş görünür. Hızlı çiftler komşuları ayırır, yavaş çiftler uzak sırayı korur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<circle cx="82" cy="112" r="56" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="82" y1="112" x2="82.0" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.25;stroke-width:1.6"/>
<text x="82.0" y="51.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">0</text>
<line x1="82" y1="112" x2="122.4" y2="86.1" style="stroke:var(--c-accent);stroke-opacity:0.37;stroke-width:1.6"/>
<text x="136.7" y="80.9" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1</text>
<line x1="82" y1="112" x2="125.6" y2="132.0" style="stroke:var(--c-accent);stroke-opacity:0.49;stroke-width:1.6"/>
<text x="141.1" y="143.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2</text>
<line x1="82" y1="112" x2="88.8" y2="159.5" style="stroke:var(--c-accent);stroke-opacity:0.61;stroke-width:1.6"/>
<text x="91.2" y="180.3" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">3</text>
<line x1="82" y1="112" x2="45.7" y2="143.4" style="stroke:var(--c-accent);stroke-opacity:0.73;stroke-width:1.6"/>
<text x="32.8" y="158.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4</text>
<line x1="82" y1="112" x2="36.0" y2="98.4" style="stroke:var(--c-accent-2);stroke-opacity:1.00;stroke-width:2.6"/>
<text x="19.7" y="97.6" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">5</text>
<circle cx="82" cy="112" r="3" style="fill:var(--c-text)"/>
<text x="82" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">1. çift</text>
<text x="82" y="213" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">θ = 1,0 rad</text>
<text x="82" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">saniye ibresi</text>
<circle cx="214" cy="112" r="56" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="214" y1="112" x2="214.0" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.25;stroke-width:1.6"/>
<line x1="214" y1="112" x2="218.8" y2="64.2" style="stroke:var(--c-accent);stroke-opacity:0.37;stroke-width:1.6"/>
<line x1="214" y1="112" x2="223.5" y2="65.0" style="stroke:var(--c-accent);stroke-opacity:0.49;stroke-width:1.6"/>
<line x1="214" y1="112" x2="228.2" y2="66.1" style="stroke:var(--c-accent);stroke-opacity:0.61;stroke-width:1.6"/>
<line x1="214" y1="112" x2="232.7" y2="67.8" style="stroke:var(--c-accent);stroke-opacity:0.73;stroke-width:1.6"/>
<line x1="214" y1="112" x2="237.0" y2="69.9" style="stroke:var(--c-accent-2);stroke-opacity:1.00;stroke-width:2.6"/>
<circle cx="214" cy="112" r="3" style="fill:var(--c-text)"/>
<text x="214" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">2. çift</text>
<text x="214" y="213" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">θ = 0,1 rad</text>
<text x="214" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">hızlı yelkovan</text>
<circle cx="346" cy="112" r="56" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="346" y1="112" x2="346.0" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.25;stroke-width:1.6"/>
<line x1="346" y1="112" x2="346.5" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.37;stroke-width:1.6"/>
<line x1="346" y1="112" x2="347.0" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.49;stroke-width:1.6"/>
<line x1="346" y1="112" x2="347.4" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.61;stroke-width:1.6"/>
<line x1="346" y1="112" x2="347.9" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.73;stroke-width:1.6"/>
<line x1="346" y1="112" x2="348.4" y2="64.1" style="stroke:var(--c-accent-2);stroke-opacity:1.00;stroke-width:2.6"/>
<circle cx="346" cy="112" r="3" style="fill:var(--c-text)"/>
<text x="346" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">3. çift</text>
<text x="346" y="213" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">θ = 0,01 rad</text>
<text x="346" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">yavaş yelkovan</text>
<circle cx="478" cy="112" r="56" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="478" y1="112" x2="478.0" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.25;stroke-width:1.6"/>
<line x1="478" y1="112" x2="478.0" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.37;stroke-width:1.6"/>
<line x1="478" y1="112" x2="478.1" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.49;stroke-width:1.6"/>
<line x1="478" y1="112" x2="478.1" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.61;stroke-width:1.6"/>
<line x1="478" y1="112" x2="478.2" y2="64.0" style="stroke:var(--c-accent);stroke-opacity:0.73;stroke-width:1.6"/>
<line x1="478" y1="112" x2="478.2" y2="64.0" style="stroke:var(--c-accent-2);stroke-opacity:1.00;stroke-width:2.6"/>
<circle cx="478" cy="112" r="3" style="fill:var(--c-text)"/>
<text x="478" y="196" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">4. çift</text>
<text x="478" y="213" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">θ = 0,001 rad</text>
<text x="478" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">akrep</text>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">m = 0 … 5 için ibre konumları (base = 10.000, d = 8); m = 5 mor</text>
<text x="16" y="258" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Aynı altı adım, dört hız: 1. çift kadranı neredeyse dolaşırken 4. çift gözle görülür kıpırdamaz.</text>
</svg>

**`rope_theta` Neden 10.000'den 1 Milyona Çıkarıldı?**
LLaMA 1 döneminde modeller yalnızca 2.048 token okuyabiliyordu; en yavaş ibrenin tam bir turu ($6.283$ token) bu uzunluk için yeterliydi. Fakat LLaMA 3 ve Gemma 3 bağlamı **131.072 (128k) token'a** çıkardığında, $10.000$ tabanı yetersiz kaldı; çünkü en yavaş akrep ibreleri bile onlarca kez tam tur atarak yönünü şaşırır ve **faz çakışmasına (phase wrapping)** düşerdi. Tabanın `1.000.000` yapılması, en yavaş akrebin bir tam turunu **milyonlarca token'a** yayar; böylece 128k'lık dev belgelerde bile hiçbir pozisyon bir diğeriyle aynı açıya denk gelmez.

---

#### 4. Frekansın Vektöre Uygulanışı: 4 Adımlı Dönüşüm Hattı

Bir token'ın ham 8 boyutlu vektörü $\mathbf{x} = [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8]^T$ verildiğinde, $m$. pozisyon için RoPE şu 4 adımla uygulanır:

1. **2D Çiftlere Ayır:** Vektörü 4 bağımsız düzlem halinde grupla:
   - 1. Çift: $(x_1, x_2)$
   - 2. Çift: $(x_3, x_4)$
   - 3. Çift: $(x_5, x_6)$
   - 4. Çift: $(x_7, x_8)$
2. **Açıları Hesapla:** Pozisyon numarasıyla açısal hızları çarp:
   $$\phi_1(m) = m \times 1,0, \quad \phi_2(m) = m \times 0,1, \quad \phi_3(m) = m \times 0,01, \quad \phi_4(m) = m \times 0,001$$
3. **Her Çifti Kendi Açısıyla Döndür (2D trigonometri):**
   $$\begin{pmatrix} \tilde{x}_1 \\ \tilde{x}_2 \end{pmatrix} = \begin{pmatrix} x_1 \cos(\phi_1) - x_2 \sin(\phi_1) \\ x_1 \sin(\phi_1) + x_2 \cos(\phi_1) \end{pmatrix}, \quad \begin{pmatrix} \tilde{x}_3 \\ \tilde{x}_4 \end{pmatrix} = \begin{pmatrix} x_3 \cos(\phi_2) - x_4 \sin(\phi_2) \\ x_3 \sin(\phi_2) + x_4 \cos(\phi_2) \end{pmatrix}$$
   $$\begin{pmatrix} \tilde{x}_5 \\ \tilde{x}_6 \end{pmatrix} = \begin{pmatrix} x_5 \cos(\phi_3) - x_6 \sin(\phi_3) \\ x_5 \sin(\phi_3) + x_6 \cos(\phi_3) \end{pmatrix}, \quad \begin{pmatrix} \tilde{x}_7 \\ \tilde{x}_8 \end{pmatrix} = \begin{pmatrix} x_7 \cos(\phi_4) - x_8 \sin(\phi_4) \\ x_7 \sin(\phi_4) + x_8 \cos(\phi_4) \end{pmatrix}$$
4. **Matris Olarak Birleştir:** Bu 4 bağımsız dönüş, tek bir blok-köşegen matris çarpımı olarak yazılabilir:
   $$\mathbf{R}_{\Theta, m}^{8} = \begin{bmatrix} \mathbf{R}_{\phi_1} & 0 & 0 & 0 \\ 0 & \mathbf{R}_{\phi_2} & 0 & 0 \\ 0 & 0 & \mathbf{R}_{\phi_3} & 0 \\ 0 & 0 & 0 & \mathbf{R}_{\phi_4} \end{bmatrix}, \quad \text{burada } \mathbf{R}_{\phi_j} = \begin{bmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{bmatrix}$$
   Köşegendeki 4 adet $2 \times 2$'lik döndürme bloğu dışında her yer sıfırdır; kanallar birbirine taşmadan sadece kendi düzleminde döner.

Bu dönüşümü 6 token'lık dizimize adım adım uygulayalım:

**Slot 0 ($m = 0$): "The" — Adım Adım Hesaplama**
- **Ham vektör:** $\mathbf{x}_{(0)} = [0,80, \ 0,60, \ 0,40, \ 0,20, \ 0,50, \ 0,10, \ 0,30, \ 0,20]^T$
- **Açıların hesabı ($m = 0$ olduğu için tüm açılar $0$ radyan):**
  - $\phi_1(0) = 0 \times 1,0 = 0,0\text{ rad} \implies \cos(0) = 1,0, \quad \sin(0) = 0,0$
  - $\phi_2(0) = 0 \times 0,1 = 0,0\text{ rad} \implies \cos(0) = 1,0, \quad \sin(0) = 0,0$
  - $\phi_3(0) = 0 \times 0,01 = 0,0\text{ rad} \implies \cos(0) = 1,0, \quad \sin(0) = 0,0$
  - $\phi_4(0) = 0 \times 0,001 = 0,0\text{ rad} \implies \cos(0) = 1,0, \quad \sin(0) = 0,0$
- **4 çiftin bağımsız 2D rotasyon hesabı:**
  - 1. Çift: $\tilde{x}_1 = 0,80(1,0) - 0,60(0,0) = 0,8000, \quad \tilde{x}_2 = 0,80(0,0) + 0,60(1,0) = 0,6000$
  - 2. Çift: $\tilde{x}_3 = 0,40(1,0) - 0,20(0,0) = 0,4000, \quad \tilde{x}_4 = 0,40(0,0) + 0,20(1,0) = 0,2000$
  - 3. Çift: $\tilde{x}_5 = 0,50(1,0) - 0,10(0,0) = 0,5000, \quad \tilde{x}_6 = 0,50(0,0) + 0,10(1,0) = 0,1000$
  - 4. Çift: $\tilde{x}_7 = 0,30(1,0) - 0,20(0,0) = 0,3000, \quad \tilde{x}_8 = 0,30(0,0) + 0,20(1,0) = 0,2000$
- **Döndürülmüş vektör ve korunan norm:**
  $$\mathbf{x}_{\text{rotated}(0)} = [0,8000, \ 0,6000, \ 0,4000, \ 0,2000, \ 0,5000, \ 0,1000, \ 0,3000, \ 0,2000]^T$$
  $$\Vert \mathbf{x}_{\text{rotated}(0)}\Vert = \sqrt{0,80^2 + 0,60^2 + 0,40^2 + 0,20^2 + 0,50^2 + 0,10^2 + 0,30^2 + 0,20^2} = \sqrt{1,59} \approx \mathbf{1,2610} \quad (\text{Kuruşu kuruşuna korundu})$$

**Slot 1 ($m = 1$): "dog" — Adım Adım Hesaplama**
- **Ham vektör:** $\mathbf{x}_{(1)} = [0,70, \ 0,10, \ 0,30, \ 0,50, \ 0,20, \ 0,40, \ 0,60, \ 0,10]^T$
- **Açıların hesabı ($m = 1$ adımı):**
  - $\phi_1(1) = 1 \times 1,0 = 1,0\text{ rad} \implies \cos(1,0) \approx 0,5403, \quad \sin(1,0) \approx 0,8415$
  - $\phi_2(1) = 1 \times 0,1 = 0,1\text{ rad} \implies \cos(0,1) \approx 0,9950, \quad \sin(0,1) \approx 0,0998$
  - $\phi_3(1) = 1 \times 0,01 = 0,01\text{ rad} \implies \cos(0,01) \approx 1,0000, \quad \sin(0,01) \approx 0,0100$
  - $\phi_4(1) = 1 \times 0,001 = 0,001\text{ rad} \implies \cos(0,001) \approx 1,0000, \quad \sin(0,001) \approx 0,0010$
- **4 çiftin bağımsız 2D rotasyon hesabı:**
  - 1. Çift: $\tilde{x}_1 = 0,70(0,5403) - 0,10(0,8415) = 0,2941, \quad \tilde{x}_2 = 0,70(0,8415) + 0,10(0,5403) = 0,6431$
  - 2. Çift: $\tilde{x}_3 = 0,30(0,9950) - 0,50(0,0998) = 0,2486, \quad \tilde{x}_4 = 0,30(0,0998) + 0,50(0,9950) = 0,5275$
  - 3. Çift: $\tilde{x}_5 = 0,20(1,0000) - 0,40(0,0100) = 0,1960, \quad \tilde{x}_6 = 0,20(0,0100) + 0,40(1,0000) = 0,4020$
  - 4. Çift: $\tilde{x}_7 = 0,60(1,0000) - 0,10(0,0010) = 0,5999, \quad \tilde{x}_8 = 0,60(0,0010) + 0,10(1,0000) = 0,1006$
- **Döndürülmüş vektör ve korunan norm:**
  $$\mathbf{x}_{\text{rotated}(1)} = [0,2941, \ 0,6431, \ 0,2486, \ 0,5275, \ 0,1960, \ 0,4020, \ 0,5999, \ 0,1006]^T \implies \Vert \mathbf{x}_{\text{rotated}(1)}\Vert = \sqrt{1,41} \approx \mathbf{1,1874} \quad (\text{Korundu})$$

**Slot 2 ($m = 2$): "chased"**
- Açılar: $\phi_1 = 2,0\text{ rad}, \ \phi_2 = 0,2\text{ rad}, \ \phi_3 = 0,02\text{ rad}, \ \phi_4 = 0,002\text{ rad}$
- $\mathbf{x}_{\text{rotated}(2)} = [-0,9355, \ 0,1217, \ 0,1364, \ 0,3338, \ 0,3959, \ 0,2080, \ 0,0990, \ 0,5002]^T \implies \text{Norm} = \mathbf{1.2166} \quad (\text{Korundu})$

**Slot 3 ($m = 3$): "the"**
- Açılar: $\phi_1 = 3,0\text{ rad}, \ \phi_2 = 0,3\text{ rad}, \ \phi_3 = 0,03\text{ rad}, \ \phi_4 = 0,003\text{ rad}$
- $\mathbf{x}_{\text{rotated}(3)} = [-0,8767, \ -0,4811, \ 0,3230, \ 0,3093, \ 0,4968, \ 0,1150, \ 0,2994, \ 0,2009]^T \implies \text{Norm} = \mathbf{1.2610} \quad (\text{Korundu})$

**Slot 4 ($m = 4$): "black"**
- Açılar: $\phi_1 = 4,0\text{ rad}, \ \phi_2 = 0,4\text{ rad}, \ \phi_3 = 0,04\text{ rad}, \ \phi_4 = 0,004\text{ rad}$
- $\mathbf{x}_{\text{rotated}(4)} = [-0,2408, \ -0,5848, \ 0,4216, \ 0,2868, \ 0,2758, \ 0,6115, \ 0,1984, \ 0,4008]^T \implies \text{Norm} = \mathbf{1.1446} \quad (\text{Korundu})$

**Slot 5 ($m = 5$): "cat"**
- Açılar: $\phi_1 = 5,0\text{ rad}, \ \phi_2 = 0,5\text{ rad}, \ \phi_3 = 0,05\text{ rad}, \ \phi_4 = 0,005\text{ rad}$
- $\mathbf{x}_{\text{rotated}(5)} = [0,9481, \ -0,0324, \ -0,1040, \ 0,3990, \ 0,4894, \ 0,2247, \ 0,3995, \ 0,1020]^T \implies \text{Norm} = \mathbf{1.2369} \quad (\text{Korundu})$

**Referans PyTorch uygulaması: $O(d)$ sürede RoPE hesaplama.** Üretimdeki LLM'lerde
(LLaMA, Gemma, Mistral) $\mathbf{R}_{\Theta, m}^{d}$ yoğun matrisi asla bellekte
oluşturulmaz. Bunun yerine rotasyon işlemi doğrudan vektör dilimleri üzerinde
$O(d)$ karmaşıklıkla işletilir. Makalemizdeki hesaplamaları doğrudan test etmek
isteyenler için metindeki 6 kelimelik 8 boyutlu mock tensörle birlikte kod şöyledir:

```python
import torch

def get_rotary_position_encoding(
    input: torch.Tensor,
    base: float = 10000.0,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Applies Rotary Position Embedding (RoPE) to a tensor of shape [context_length, dimension].
    Makaledeki ardışık 2D çiftler (interleaved) kurgusuyla birebir eşleşir:
      - 1. Çift: (x1, x2) = (input[:, 0], input[:, 1])
      - 2. Çift: (x3, x4) = (input[:, 2], input[:, 3])
      - 3. Çift: (x5, x6) = (input[:, 4], input[:, 5])
      - 4. Çift: (x7, x8) = (input[:, 6], input[:, 7])
    """
    context_length, dimension = input.shape
    assert dimension % 2 == 0, "Boyut çift sayı olmalıdır"

    half_dimension = dimension // 2

    # Adım 1: Her 2D alt uzay için temel açısal frekanslar (theta_j = 1 / base^(2j/d))
    freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)
    freqs = 1.0 / (base ** (2.0 * freqs_indices / dimension))

    # Adım 2: Pozisyonlar ile frekansların dış çarpımı: phi(m, j) = m * theta_j
    positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)
    angles = positions * freqs

    sin_angles = torch.sin(angles)
    cos_angles = torch.cos(angles)

    # Adım 3: Ardışık 2D çiftlerin (çift ve tek indekslerin) ayrılması:
    x_even = input[:, 0::2]  # kanallar 0, 2, 4, 6 (x1, x3, x5, x7)
    x_odd  = input[:, 1::2]  # kanallar 1, 3, 5, 7 (x2, x4, x6, x8)

    # 2D rotasyon formülü: [x1*cos - x2*sin, x1*sin + x2*cos]
    x_even_rotated = x_even * cos_angles - x_odd * sin_angles
    x_odd_rotated  = x_even * sin_angles + x_odd * cos_angles

    # Adım 4: Özgün kanal düzenine geri birleştirme
    input_rotated = torch.empty_like(input)
    input_rotated[:, 0::2] = x_even_rotated
    input_rotated[:, 1::2] = x_odd_rotated

    return input_rotated

# -------------------------------------------------------------
# Makaledeki 6 Token'lık 8 Boyutlu Mock Verilerle Doğrulama:
# ["The", "dog", "chased", "the", "black", "cat"]
# -------------------------------------------------------------
mock_embeddings = torch.tensor([
    [0.80, 0.60, 0.40, 0.20, 0.50, 0.10, 0.30, 0.20],  # Slot 0 ("The")
    [0.70, 0.10, 0.30, 0.50, 0.20, 0.40, 0.60, 0.10],  # Slot 1 ("dog")
    [0.50, 0.80, 0.20, 0.30, 0.40, 0.20, 0.10, 0.50],  # Slot 2 ("chased")
    [0.80, 0.60, 0.40, 0.20, 0.50, 0.10, 0.30, 0.20],  # Slot 3 ("the")
    [0.60, 0.20, 0.50, 0.10, 0.30, 0.60, 0.20, 0.40],  # Slot 4 ("black")
    [0.30, 0.90, 0.10, 0.40, 0.50, 0.20, 0.40, 0.10],  # Slot 5 ("cat")
], dtype=torch.float32)

# RoPE uygula:
pos_rotary_encodings = get_rotary_position_encoding(mock_embeddings)

print("Ham Girdi Vektörleri:\n", mock_embeddings)
print("\nDöndürülmüş Vektörler (RoPE):\n", pos_rotary_encodings.round(decimals=4))

# Doğrulama: Normlar rotasyondan önce ve sonra birebir aynıdır (%0 sapma)
print("\nOrijinal normlar:  ", mock_embeddings.norm(dim=-1).round(decimals=4))
print("Döndürülmüş normlar:", pos_rotary_encodings.norm(dim=-1).round(decimals=4))
assert torch.allclose(mock_embeddings.norm(dim=-1), pos_rotary_encodings.norm(dim=-1))
```

> **Üretim Ortamı Notu 1: Gemma 3 ve LLaMA'da RoPE Gerçekte Nereye Uygulanır?**  
> Bu bölümde sezgiyi canlı tutmak için RoPE'u doğrudan kelime gömme vektörlerine
> uyguladık. Ancak üretim modellerinde (Gemma 3, LLaMA, Mistral, Qwen) RoPE kelime
> arama tablosuna **kesinlikle uygulanmaz**. Bunun yerine, dikkat (attention)
> katmanındaki **Query ($Q$) ve Key ($K$)** vektörlerine doğrudan her dikkat
> başlığında (attention head) ayrı ayrı uygulanır; Değer ($V$) vektörleri ise
> döndürülmez. Böylece dilin öz anlamsal içeriği ile geometrik pozisyonu, dikkat
> iç çarpımının tam hesaplandığı âna kadar birbirinden tertemiz ayrılmış kalır.

> **Üretim Ortamı Notu 2: İki Farklı Dilimleme Deseni — Aralıklı (Interleaved) vs. İkiye Bölmeli (Split-Half)**  
> Pratik kodlamada 8 boyutlu (veya üretimdeki 128 boyutlu) vektörün 2D çiftlere nasıl
> bölündüğü konusunda LLM dünyasında iki farklı ekol vardır:
> 
> 1. **Aralıklı / Birer Boşluklu (Interleaved / GPT-J ve Orijinal RoFormer Stili):**
>    Vektörün ardışık kanalları yan yana çiftlenir: $(x_0, x_1), (x_2, x_3), (x_4, x_5), (x_6, x_7)$.
>    PyTorch'ta tek ve çift indeks dilimleriyle (`input[:, 0::2]` ve `input[:, 1::2]`) işletilir.
>    Su et al.'ın orijinal **RoFormer** makalesinde ve EleutherAI'ın **GPT-J** modelinde bu desen kullanılmıştır.
>    Matematiksel ve geometrik açıdan 2D rotasyon matrisini doğrudan yansıttığı için kavraması en kolay formdur
>    (makalemizdeki kod da bu sezgisel yapıyı temel almıştır).
> 
> 2. **İkiye Bölmeli / Yarı Yarıya (Split-Half / Rotate-Half / LLaMA, Gemma, Mistral Stili):**
>    Vektör tam ortadan ikiye kesilir: birinci yarı $[0 \dots d/2-1]$ ile ikinci yarı $[d/2 \dots d-1]$.
>    $i$. kanal ile $(i + d/2)$. kanal bir 2D çift oluşturur: $(x_i, x_{i + d/2})$.
>    PyTorch'ta `input[:, :d//2]` ve `input[:, d//2:]` şeklinde dilimlenir.
>    **LLaMA (Meta)**, **Gemma (Google)**, **Mistral**, **Qwen** ve Hugging Face Transformers
>    kütüphanesindeki meşhur `rotate_half` fonksiyonu bu deseni kullanır.
> 
> **Peki modern modeller neden ikiye bölmeyi (split-half) tercih etti?**  
> Cevap tamamen **GPU bellek mimarisidir (memory coalescing)**: GPU donanımında (CUDA ve Triton
> çekirdeklerinde) bellekte peş peşe duran bitişik (contiguous) blokları okumak (`[:d//2]`),
> atlamalı bellek adreslerini okumaktan (`0::2`) katbekat daha hızlıdır; önbellek satırlarını parçalamaz.
> 
> *Önemli Mühendislik Notu:* İki yöntem de uzayda $d/2$ adet bağımsız 2D rotasyon yaptığı için matematiksel
> olarak tamamen **izomorftur** (özdeş temsil gücüne sahiptir). Ancak kanal eşleşmeleri farklı olduğundan,
> "interleaved" formatında eğitilmiş ağırlıklar "split-half" çalıştıran bir motorda doğrudan kullanılamaz.
> İşte bu nedenle Hugging Face'in LLaMA ağırlık dönüştürücüsünde (`convert_llama_weights_to_hf.py`),
> tensör boyutlarını bu iki düzen arasında hizalayan özel bir matris permütasyon adımı (`permute`) yer alır.

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
  - 1. alt uzay göreli açısı: $\Delta\phi_1 = 4 \times 1,0 = 4,0\text{ rad}$
  - 2. alt uzay göreli açısı: $\Delta\phi_2 = 4 \times 0,1 = 0,4\text{ rad}$
  - 3. alt uzay göreli açısı: $\Delta\phi_3 = 4 \times 0,01 = 0,04\text{ rad}$
  - 4. alt uzay göreli açısı: $\Delta\phi_4 = 4 \times 0,001 = 0,004\text{ rad}$

Bu token çifti ister $1 \to 5$ indekslerinde, isterse 128k'lık bir belgenin
$10.001 \to 10.005$ indekslerinde yer alsın; $(n - m) = 4$ değişmez kalır.
Mutlak pozisyonlar ($m=1, n=5$) iç çarpım hesabından tamamen düşer; yalnızca
aralarındaki tam 4 adımlık göreli ilişki hesaplanır.

### Üç yöntemin karşılaştırma tablosu

| Metrik / Davranış | A. Learned Absolute | B. Sinusoidal | C. RoPE |
| :--- | :---: | :---: | :---: |
| **Slot 0 ("The") Çıktısı** | `[0,80, 0,80, 0,50, 0,20, 0,60, 0,00, 0,30, 0,30]` | `[0,80, 1,60, 0,40, 1,20, 0,50, 1,10, 0,30, 1,20]` | `[0,80, 0,60, 0,40, 0,20, 0,50, 0,10, 0,30, 0,20]` |
| **Slot 0 Normu** | 1,45 | 2,79 | **1,26 (%0 kayma)** |
| **Slot 1 ("dog") Çıktısı** | `[0,90, 0,00, 0,30, 0,60, 0,10, 0,50, 0,70, 0,10]` | `[1,54, 0,64, 0,40, 1,50, 0,21, 1,40, 0,60, 1,10]` | `[0,29, 0,64, 0,25, 0,53, 0,20, 0,40, 0,60, 0,10]` |
| **Slot 1 Normu** | 1,42 | 2,96 | **1,19 (%0 kayma)** |
| **Slot 2 ("chased") Normu** | 1,33 | 2,79 | **1,22 (%0 kayma)** |
| **Slot 3 ("the") Normu** | 1,40 | 2,42 | **1,26 (%0 kayma)** |
| **Slot 4 ("black") Normu** | 1,23 | 2,60 | **1,14 (%0 kayma)** |
| **Slot 5 ("cat") Normu** | 1,41 | 2,63 | **1,24 (%0 kayma)** |
| **Vektör Boyu Şişti mi?** | Evet | Evet (Aşırı) | **Hayır (Kesinlikle Korundu)** |
| **Göreli Mesafe Doğal mı?** | Hayır | Kısmen | **Evet (Tam $n-m$)** |
| **Eğitilebilir Parametre** | $L_{\max} \times d_{\text{model}}$ | 0 | **0** |

Tablodaki norm satırları, her token'ın ham normuyla birlikte çizildiğinde:

<svg viewBox="0 0 560 300" role="img" aria-label="The, dog, chased, the, black, cat token&#x27;larının üç pozisyon yöntemi altındaki vektör normlarını gösteren gruplu çubuk grafik; her token&#x27;ın ham normu kesikli çizgiyle işaretli. Öğrenilmiş mutlak toplama normları biraz yukarı kaydırır: 1,45, 1,42, 1,33, 1,40, 1,23, 1,41. Sinüzoidal toplama onları iki katından fazla büyütür: 2,79, 2,96, 2,79, 2,42, 2,60, 2,63. RoPE ham normları birebir korur: 1,26, 1,19, 1,22, 1,26, 1,14, 1,24. Aynı token olan The ve the, iki toplamalı yöntemde de farklı normlara ulaşır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="52" y1="236.0" x2="544" y2="236.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="44" y="240.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">0</text>
<line x1="52" y1="172.7" x2="544" y2="172.7" style="stroke:var(--c-border);stroke-width:1"/>
<text x="44" y="176.66666666666666" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">1</text>
<line x1="52" y1="109.3" x2="544" y2="109.3" style="stroke:var(--c-border);stroke-width:1"/>
<text x="44" y="113.33333333333333" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">2</text>
<line x1="52" y1="46.0" x2="544" y2="46.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="44" y="50.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">3</text>
<text x="16" y="30" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">vektör normu ‖x‖</text>
<rect x="62.0" y="144.2" width="18" height="91.8" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="84.0" y="59.3" width="18" height="176.7" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<rect x="106.0" y="156.2" width="18" height="79.8" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<line x1="59.0" y1="156.2" x2="127.0" y2="156.2" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="93.0" y="53.29999999999998" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">2,79</text>
<text x="93.0" y="252" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">The</text>
<rect x="144.0" y="146.1" width="18" height="89.9" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="166.0" y="48.5" width="18" height="187.5" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<rect x="188.0" y="160.6" width="18" height="75.4" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<line x1="141.0" y1="160.6" x2="209.0" y2="160.6" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="175.0" y="42.53333333333333" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">2,96</text>
<text x="175.0" y="252" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">dog</text>
<rect x="226.0" y="151.8" width="18" height="84.2" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="248.0" y="59.3" width="18" height="176.7" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<rect x="270.0" y="158.7" width="18" height="77.3" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<line x1="223.0" y1="158.7" x2="291.0" y2="158.7" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="257.0" y="53.29999999999998" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">2,79</text>
<text x="257.0" y="252" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">chased</text>
<rect x="308.0" y="147.3" width="18" height="88.7" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="330.0" y="82.7" width="18" height="153.3" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<rect x="352.0" y="156.2" width="18" height="79.8" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<line x1="305.0" y1="156.2" x2="373.0" y2="156.2" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="339.0" y="76.73333333333332" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">2,42</text>
<text x="339.0" y="252" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">the</text>
<rect x="390.0" y="158.1" width="18" height="77.9" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="412.0" y="71.3" width="18" height="164.7" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<rect x="434.0" y="163.8" width="18" height="72.2" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<line x1="387.0" y1="163.8" x2="455.0" y2="163.8" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="421.0" y="65.33333333333331" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">2,60</text>
<text x="421.0" y="252" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">black</text>
<rect x="472.0" y="146.7" width="18" height="89.3" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="494.0" y="69.4" width="18" height="166.6" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<rect x="516.0" y="157.5" width="18" height="78.5" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<line x1="469.0" y1="157.5" x2="537.0" y2="157.5" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="503.0" y="63.43333333333334" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">2,63</text>
<text x="503.0" y="252" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">cat</text>
<rect x="16" y="262" width="12" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<text x="34" y="272" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">öğrenilmiş mutlak</text>
<rect x="150" y="262" width="12" height="12" rx="2" style="fill:var(--c-danger);fill-opacity:.75"/>
<text x="168" y="272" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sinüzoidal</text>
<rect x="246" y="262" width="12" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<text x="264" y="272" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">RoPE</text>
<line x1="312" y1="268" x2="330" y2="268" style="stroke:var(--c-text);stroke-width:1.5;stroke-dasharray:3 2"/>
<text x="336" y="272" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ham norm</text>
<text x="16" y="294" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Pozisyon vektörü eklemek token&#x27;ı uzatır; döndürmek uzatmaz. The ile the eşit başlar.</text>
</svg>

**Vektör boyu neden bozulmamalı (norm koruma).** Dikkat puanı iç çarpımla
hesaplanır: $\mathbf{u} \cdot \mathbf{v} = \Vert \mathbf{u}\Vert \Vert \mathbf{v}\Vert \cos(\theta)$.
B yöntemindeki gibi bir vektörün normunun $1,26$'dan $2,96$'ya kadar şişmesi,
ölçeklenmemiş iç çarpımlarını yaklaşık 5 katına çıkararak softmax'i aşırı
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
2. **Bağlam dışdeğerleme (extrapolation):** Mutlak pozisyon tabloları eğitim
   uzunluğu ($L_{\max}$) ile sınırlıyken, göreli rotasyon açıları periyodik
   olarak sonsuz uzunluktaki diziler boyunca işlemeye devam eder.
3. **Doğal uzunluk sönümlenmesi (decay):** Hızlı dönen boyutlar
   ($\theta_1 = 1,0$) süratle salınarak yerel sözdizimini yalıtır. Yavaş dönen
   boyutlar ($\theta_4 = 0,001$) ise binlerce token boyunca genel anlamsal
   bağlamı canlı tutar.

## 5. Kayıp sıçramaları ve WeSaR

Bir **kayıp sıçraması (loss spike)** — ön eğitim sırasında eğitim kaybının
aniden fırlaması ve optimizasyonun diverjans göstermesi — katman ölçekleme
dengesizlikleriyle yakından ilişkilidir.

<svg viewBox="0 0 560 136" role="img" aria-label="Bir loss spike&#x27;ın oluşumu. 1 bölü karekök 2N ile residual ölçekleme, aşağı projeksiyon ağırlıklarının normlarını yaklaşık 0,002&#x27;ye küçültür. Adam&#x27;ın adımları yine yaklaşık 0,001 kalır, bu yüzden bağıl güncelleme oranı adım başına yüzde 5&#x27;ten yüzde 50&#x27;ye fırlar. Bu aşırı duyarlılık loss spike&#x27;ı doğurur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="14" width="120" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="76.0" y="47.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">residual ölçekleme</text>
<text x="76.0" y="63.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">1/√(2N)</text>
<line x1="136" y1="51" x2="150" y2="51" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="152" y="14" width="120" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="212.0" y="47.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">normlar küçülür</text>
<text x="212.0" y="63.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">‖W_d‖ ≈ 0,002</text>
<line x1="272" y1="51" x2="286" y2="51" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="288" y="14" width="120" height="74" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="348.0" y="39.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">bağıl güncelleme</text>
<text x="348.0" y="55.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">‖ΔW‖ / ‖W‖</text>
<text x="348.0" y="71.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">adımda %5 → %50</text>
<line x1="408" y1="51" x2="422" y2="51" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="424" y="14" width="120" height="74" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="484.0" y="47.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LOSS SPIKE</text>
<text x="484.0" y="63.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">aşırı duyarlılık</text>
<text x="16" y="112" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Adam&#x27;ın adımı ağırlık ölçeğinden bağımsız olarak 0,001 civarında kalır;</text>
<text x="16" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">küçük bir ağırlık bu yüzden oransal olarak dev bir adım atar.</text>
</svg>

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

Asimetri, tek matrisin iki rolü yan yana konunca en net görünür:

<svg viewBox="0 0 560 262" role="img" aria-label="Gemma 3&#x27;te bağlı ağırlıklar. 262.144&#x27;e 1.152 boyutlu, her vocabulary girdisi için bir satırı olan tek bir E matrisi aynı bellek işaretçisi üzerinden iki kez kullanılır. Girişte embed_tokens token ID k&#x27;yı alır ve yalnızca k. satırı okur: bir gather, sıfır FLOP. Çıkışta lm_head son gizli vektör h&#x27;yi alır ve her vocabulary girdisi için bir logit üretmek üzere E&#x27;nin bütün satırlarıyla çarpar: Gemma-3-1B&#x27;de üretilen token başına yaklaşık 0,60 GFLOP." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ty-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="ty-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="222" y="40" width="116" height="170" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="230" y1="50" x2="330" y2="50" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="58" x2="330" y2="58" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="66" x2="330" y2="66" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="74" x2="330" y2="74" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="82" x2="330" y2="82" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="90" x2="330" y2="90" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="98" x2="330" y2="98" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="106" x2="330" y2="106" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="114" x2="330" y2="114" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="122" x2="330" y2="122" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="130" x2="330" y2="130" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="138" x2="330" y2="138" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="146" x2="330" y2="146" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="154" x2="330" y2="154" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="162" x2="330" y2="162" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="170" x2="330" y2="170" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="178" x2="330" y2="178" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="186" x2="330" y2="186" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="194" x2="330" y2="194" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="230" y1="202" x2="330" y2="202" style="stroke:var(--c-border);stroke-width:1"/>
<text x="280" y="30" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">tek matris E</text>
<text x="280" y="228" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">262.144 × 1.152</text>
<text x="280" y="244" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">aynı data_ptr() iki kez</text>
<text x="16" y="30" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">GİRİŞ · embed_tokens</text>
<rect x="16" y="74" width="70" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="51.0" y="91.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ID k</text>
<line x1="86" y1="87" x2="220" y2="87" marker-end="url(#ty-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="224" y="82" width="112" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="150" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">k. satırı oku</text>
<text x="16" y="124" text-anchor="start" style="fill:var(--c-text);font-size:12px">tek satır gather</text>
<text x="16" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">0 FLOP, bir bellek okuması</text>
<text x="16" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">(Gemma&#x27;da sonra × √1152)</text>
<text x="544" y="30" text-anchor="end" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">ÇIKIŞ · lm_head</text>
<rect x="474" y="74" width="70" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="509.0" y="91.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<path d="M474 87 H406 V125 H342" marker-end="url(#ty-arr-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="338" y1="50" x2="352" y2="50" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="58" x2="352" y2="58" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="66" x2="352" y2="66" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="74" x2="352" y2="74" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="82" x2="352" y2="82" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="90" x2="352" y2="90" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="98" x2="352" y2="98" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="106" x2="352" y2="106" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="114" x2="352" y2="114" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="122" x2="352" y2="122" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="130" x2="352" y2="130" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="138" x2="352" y2="138" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="146" x2="352" y2="146" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="154" x2="352" y2="154" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="162" x2="352" y2="162" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="170" x2="352" y2="170" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="178" x2="352" y2="178" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="186" x2="352" y2="186" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="194" x2="352" y2="194" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<line x1="338" y1="202" x2="352" y2="202" style="stroke:var(--c-accent-2);stroke-opacity:.5;stroke-width:1"/>
<text x="472" y="118" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">h · her satır</text>
<line x1="352" y1="180" x2="470" y2="180" marker-end="url(#ty-arr-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="544" y="176" text-anchor="end" style="fill:var(--c-text);font-size:12px">262.144 logit</text>
<text x="544" y="196" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">2 × d × V ≈ 0,60 GFLOP</text>
<text x="544" y="212" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">üretilen token başına (1B)</text>
</svg>

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
- Bu blogda: [Bir Prompt'un Yolculuğu (1): Tokenizasyon](post.html?slug=tokenizasyon-nasil-calisir) — masanın bir önceki adımı: metinden token ID'sine —, [Bir Prompt'un Yolculuğu (3): Anlamsal Embedding'ler](post.html?slug=embeddingler-derinlemesine) — anlamsal arama ve vektör uzayları —, [Bir Prompt'un Yolculuğu (4): Self-Attention](post.html?slug=self-attention-derinlemesine) — $Z$ matrisinden $Q, K, V$ ve bağlam vektörlerine — ve [Büyük Resim (1): Baştan Sona Bir LLM](post.html?slug=llm-nasil-calisir) — vektörlerin katmanlar arasındaki yolculuğu.
