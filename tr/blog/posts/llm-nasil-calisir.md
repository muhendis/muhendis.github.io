Bu cümleyi bitirmeden zihniniz sıradaki kelimeyi tahmin etmeye başladı
bile. Kanıt mı? "Bir varmış, bir ___." Durduramadınız: boşluk, sizden izin
almadan kendini doldurdu.

O refleksin bir adı var — sıradaki-token tahmini (next-token prediction) — ve bu yazının tek bir
iddiası: bir dil modelinin size yazdığı her cümle, her deneme, her kod
parçası, az önce yaptığı hata için her özür, aynı refleksin milyarlarca
kat büyütülmüşüdür. Telefon klavyeniz "görüşürüz"den sonra *yarın*
önerdiğinde bu oyunun cep boyunu oynuyor. Peki bu kadar basit bir oyun
nasıl sınav geçer, nasıl yazılım yazar? Çünkü talebi acımasızdır: insan
metninde sıradaki kelimeyi *iyi* tahmin etmek için dil bilgisini,
olguları, üslubu ve akıl yürütmenin işleyen bir taklidini özümsemek
zorundasınızdır. Aşağıdaki her şey bu fikrin dipnotudur.

**Bu yazıda**

- [1. Metin sayıya dönüşür](#1-metin-sayıya-dönüşür)
- [2. Anlam taşıyan sayılar](#2-anlam-taşıyan-sayılar)
- [3. Transformer: bir bağlam makinesi](#3-transformer-bir-bağlam-makinesi) — [Q, K, V](#q-k-v-mekanizma-sayılarla) · [Tek yön](#tek-yön) · [Birçok kafa](#birçok-kafa)
- [4. Katmanlar: bilgi nerede yaşıyor](#4-katmanlar-bilgi-nerede-yaşıyor)
- [5. Eğitim ve ölçek](#5-eğitim-ve-ölçek)
- [6. Otomatik tamamlamadan asistana](#6-otomatik-tamamlamadan-asistana)
- [7. Üretim: plan değil, döngü](#7-üretim-plan-değil-döngü)
- [8. Sizi hatırlamaz](#8-sizi-hatırlamaz)
- [9. Neden uyduruyor](#9-neden-uyduruyor)
- [Bütün hikâye beş satırda](#bütün-hikâye-beş-satırda)
- [Daha derine inmek için](#daha-derine-inmek-için)

## 1. Metin sayıya dönüşür

**Tokenizer**, metni **token** denen parçalara böler ve her birine bir
kimlik numarası verir — "için" tek parça, "inanılmaz" belki
"inan + ılmaz". LEGO gibi: dil, en çok yeniden kullanılan parçalarına
ayrılır — yaygın kelimeler bütün kalır, nadirler parçalardan kurulur
(standart algoritma: **byte-pair encoding**, BPE). Pratik kural:
100 token ≈ 75 İngilizce kelime; "128K bağlam penceresi" (context window) kabaca bir roman eder. Model harfleri değil yalnız token kimliklerini gördüğü için
"strawberry"deki r'leri saymak meşhur biçimde zordu — bir tablonun
*fotoğrafındaki* fırça darbelerini saymak gibi.

## 2. Anlam taşıyan sayılar

Sonra her token bir **embedding**'e dönüşür: uzun bir sayı listesi —
anlam haritasındaki koordinatları. Binlerce kadran düşünün: biri
resmiyet, biri zaman, çoğu hiçbir insanın adlandırmadığı nitelikler
için. Haritadan üç kare:

- *kral*, *kraliçe*ye yakın, *hesap tablosu*na uzak durur;
- *Paris*ten *Fransa*ya giden ok, *Roma*dan *İtalya*ya giden okla paraleldir — bir "başkenti-olmak" yönü;
- *kral − erkek + kadın*, *kraliçe*nin yakınına düşer.

Binlerce kadrandan ikisi, çizimle:

<svg viewBox="0 0 480 310" role="img" aria-label="Kelimeler iki boyutlu anlam haritasında noktalar olarak: erkek-kadın ile kral-kraliçe okları paralel; Paris-Fransa ile Roma-İtalya okları da paralel; hesap tablosu uzakta" style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="emb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<line x1="40" y1="280" x2="470" y2="280" style="stroke:var(--c-border);stroke-width:1.5"/>
<line x1="40" y1="280" x2="40" y2="15" style="stroke:var(--c-border);stroke-width:1.5"/>
<text x="465" y="298" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">kadran 1</text>
<text x="20" y="160" transform="rotate(-90 20 160)" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">kadran 2</text>
<line x1="140" y1="215" x2="215" y2="155" marker-end="url(#emb-arr)" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="170" y1="115" x2="245" y2="55" marker-end="url(#emb-arr)" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="315" y1="130" x2="385" y2="95" marker-end="url(#emb-arr)" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="320" y1="235" x2="390" y2="200" marker-end="url(#emb-arr)" style="stroke:var(--c-accent);stroke-width:1.8"/>
<circle cx="140" cy="215" r="4.5" style="fill:var(--c-text)"/>
<circle cx="215" cy="155" r="4.5" style="fill:var(--c-text)"/>
<circle cx="170" cy="115" r="4.5" style="fill:var(--c-text)"/>
<circle cx="245" cy="55" r="4.5" style="fill:var(--c-text)"/>
<circle cx="315" cy="130" r="4.5" style="fill:var(--c-text)"/>
<circle cx="385" cy="95" r="4.5" style="fill:var(--c-text)"/>
<circle cx="320" cy="235" r="4.5" style="fill:var(--c-text)"/>
<circle cx="390" cy="200" r="4.5" style="fill:var(--c-text)"/>
<circle cx="428" cy="252" r="4.5" style="fill:var(--c-text-mute)"/>
<text x="140" y="233" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-style:italic">erkek</text>
<text x="215" y="143" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-style:italic">kadın</text>
<text x="160" y="110" text-anchor="end" style="fill:var(--c-text);font-size:13px;font-style:italic">kral</text>
<text x="245" y="43" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-style:italic">kraliçe</text>
<text x="309" y="147" text-anchor="end" style="fill:var(--c-text);font-size:13px;font-style:italic">Paris</text>
<text x="391" y="86" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-style:italic">Fransa</text>
<text x="318" y="253" text-anchor="end" style="fill:var(--c-text);font-size:13px;font-style:italic">Roma</text>
<text x="396" y="192" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-style:italic">İtalya</text>
<text x="428" y="240" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-style:italic">hesap tablosu</text>
<text x="48" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">aynı yön</text>
<text x="352" y="172" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">"başkenti-olmak" yönü</text>
</svg>

Haritayı kimse çizmedi — öğrenildi. Bir de her token'ın **konumu** işlenir, çünkü "köpek adamı
ısırdı", "adam köpeği ısırdı"dan farklı kalmalıdır.

## 3. Transformer: bir bağlam makinesi

Embedding tek başına "yüz"ün ne olduğunu söyleyemez — surattaki yüz mü,
sayı olan yüz mü? Anlam bağlamda yaşar ve **transformer** — GPT'deki
T — onu okumak için kurulmuş makinedir. Eski tasarımlar metni soldan
sağa, mesafeyle solan tek bir dar hafızadan geçirirdi. 2017 tarihli
*Attention Is All You Need* makalesi bunu atıp tek bir mekanizma tuttu:
**attention** — her token, diğer her token'a *doğrudan* ve aynı anda
bakar, neyin önemli olduğuna kendisi karar verir.

Bu kararı makalenin kendi örneğinde izleyin:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Tek kelime değişir, "o" taraf değiştirir. Siz bunu anında çözdünüz;
attention, modelin çözme biçimidir. Bütün numara tek cümledir: **bir
kelimenin yeni anlamı, diğer kelimelerin ağırlıklı karışımıdır — ve
attention'ın bütün işi ağırlıkları seçmektir.** Doğru ağırlık mesafeden
gelemez ("o"yu çözen kelime yirmi token geride olabilir); *içerikten*
hesaplanır — ve öğrenilir.

Makine, üst üste katmanlardan oluşur; her katmanda iki alt katman
vardır: her kelimenin vektörünü diğerlerinin ışığında yeniden yazdığı
**self-attention** ve sonra her kelimeyi tek başına sindiren küçük bir
**ileri beslemeli ağ**. Önce birlikte topla, sonra yalnız sindir. Bu
bölüm birinciyi açar; ikincisi 4. bölümün konusu.

### Q, K, V — mekanizma, sayılarla

Ağırlıklarını seçmek için her token aynı anda üç rol oynar — kendi embedding'inin öğrenilmiş üç küçük kılığı:

- **sorgu (Q)** — ne arıyorum?
- **anahtar (K)** — başkaları beni nasıl bulur?
- **değer (V)** — seçilirsem ne devrederim?

YouTube aynı üçlüyle çalışır: yazdığınız metin sorgudur, her videonun
başlığı anahtardır, videoların kendisi değerdir. Modelde üçü de
token'ın embedding'inden gelir: üç öğrenilmiş tabloyla çarpım — **W_Q,
W_K, W_V** — aynı kelime, üç kıyafet, üçü de aynı anda üstünde. Ham
embedding'ler yetmezdi: embedding kelimenin her şeyini karıştırır,
oysa arama tek seferde tek yön ister — "yorgun olabilir" üzerinden
eşleşen bir sorgu-anahtar çifti, "f ile başlar" üzerinden değil. Sonuç bir **yumuşak sözlüktür** (soft dictionary): her anahtar *kısmen* eşleşir, her değerden
orantılı bir dilim alınır.

Aritmetik iki hamledir: **iç çarpım** (dot product: iki vektörü konum konum çarpıp topla — ne kadar hizalıysa o kadar büyük: bir benzerlik ölçer) ve
**ağırlıklı toplam** (weighted sum: vektörleri yüzdelerle karıştır, tarif gibi).
Bunları dört adım çalıştırır. "Hızlı kahverengi tilki"yi, model
*tilki* üzerinde çalışırken izleyin:

**1. Adım — Puanla.** *Tilki*nin sorgusu her kelimenin anahtarıyla
buluşur: puanᵢ = Q(tilki) · K(kelimeᵢ).

**2. Adım — Ölçekle.** Her puan, anahtar vektörünün boyutu **√dₖ**'ye
bölünür; büyük vektörlerin softmax'ı ya-hep-ya-hiç ağırlıklara
kilitlemesi böyle önlenir: ölçekliᵢ = puanᵢ ÷ √dₖ.

**3. Adım — Softmax'la.** Her puan için *e* üssü alınır, toplama
bölünür; puanlar yüzdeye dönüşür: ağırlıkᵢ = e^puanᵢ ÷ (e^puan₁ + …).

| çift | puan | e^puan | pay |
|---|---|---|---|
| Q(tilki) · K(hızlı) | 2,1 | 8,2 | **%3** |
| Q(tilki) · K(kahverengi) | 4,0 | 54,6 | **%19** |
| Q(tilki) · K(tilki) | 5,4 | 221,4 | **%78** |

Son satırı doğrulayın: 221,4 ÷ 284,2 ≈ %78. Üstel fonksiyonun yaptığına bakın: 5,4, 4,0'ın yalnızca biraz üstünde; ama %78, %19'un dört katı — softmax liderleri ödüllendirir.

**4. Adım — Karıştır.** Yeni *tilki*, değerlerin ağırlıklı toplamıdır:
tilki_yeni = 0,03·V(hızlı) + 0,19·V(kahverengi) + 0,78·V(tilki) —
artık sözlükteki *tilki* değil, *bu-belirli-hızlı-kahverengi-tilki*.

Yukarıdaki her şey tek ünlü satırdır:

> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**

1. Adım QKᵀ, 2. Adım bölme, 3. Adım softmax, 4. Adım V ile çarpım.
Öğrenilen *tek* parça üç tablodur; gerisi sabit aritmetik — ve tüm
token'lar matris olarak üst üste konduğundan bu tek satır bütün
aramaları aynı anda koşturur: GPU'ların bayıldığı iş. Bedeli **O(n²)**: herkes herkesi puanlar — bağlamı ikiye katlayın, maliyet dörde katlanır.

Bütün mekanizma, tek resimde:

```mermaid
flowchart TD
    E["tilki'nin embedding'i"] -->|"× W_Q"| Q["Q — ne arıyorum?"]
    E -->|"× W_K"| K["K — nasıl bulunurum?"]
    E -->|"× W_V"| V["V — ne devrederim?"]
    Q --> S1["1. Adım · puan = Q · K"]
    K --> S1
    S1 --> S2["2. Adım · ölçek ÷ √dₖ"]
    S2 --> S3["3. Adım · softmax → yüzdeler"]
    S3 --> S4["4. Adım · karışım = Σ ağırlık × V"]
    V --> S4
    S4 --> OUT["yeni tilki — bu-belirli-hızlı-kahverengi-tilki"]
```

### Tek yön

Üretim sırasında token yalnızca *geriye* bakar: *tilki*, *kahverengi*yi
görür; *kahverengi*, *tilki*yi asla. Modeli *sıradaki*-token tahmincisi
yapan bu maskedir — ve geçmiş bir token'ın anahtarıyla değeri, bir kez
hesaplanınca bir daha değişmez. Bir kenara yazın; 7. bölümde KV cache
olacak. Maske aynı zamanda model ailelerinin anahtarıdır: maske*yle*
kurulan model yazar — **decoder** ailesi: GPT ve neredeyse tüm modern
LLM'ler; maske*siz* kurulan iki yönü görür ve sınıflandırır —
**encoder** ailesi: BERT.

### Birçok kafa

Katman başına tek ağırlıklama kaba kalırdı — kelimenin bir komşudan dil
bilgisine, başkasından göndergeye ihtiyacı var. Bu yüzden her katman,
her biri kendi Q/K/V mercekli, vektörün ince bir dilimiyle çalışan
birçok **kafa** koşturur (orijinal tasarımda 512 ÷ 8 kafa = 64; sekiz
kafa, kabaca tek kafa fiyatına). "O" kodlanırken bir kafa *hayvan*a,
öteki *yorgun*a kilitlenir — gönderge ve gerekçe, aynı anda.

Bütün 3. bölüm, tek kartta:

| soru | cevap |
|---|---|
| Self-attention nedir? | Cümlenin kendine dikkat etmesi: her token, vektörünü diğerlerinin ışığında yeniden yazar |
| Q, K, V nedir? | Token başına üç rol — sorgu: *ne arıyorum?* · anahtar: *nasıl bulunurum?* · değer: *ne devrederim?* |
| Neden üç ayrı vektör? | Embedding her yönü karıştırır; her rol yalnızca kendi işinin yönünü çeker |
| Hesap hangi sırayla akar? | puanla (Q·K) → ölçekle (÷√dₖ) → softmax'la → karıştır (×V) |
| İleri beslemeli ağ nedir? | Her token'ı tek başına sindiren ağ — 4. bölümün bilgi ambarı |

## 4. Katmanlar: bilgi nerede yaşıyor

Bir attention alt katmanıyla bir ileri beslemeli alt katman, birlikte bir **katman** oluşturur; transformer, bu katlardan örülü bir kuledir — onlarca, kimi zaman yüzü aşkın kat. Her katta aynı rutin işler:

- **Attention** — kütüphaneci — diğer token'lardan bağlamı toplar.
- **İleri beslemeli ağ** — ambar — onu tek başına sindirir; "Paris, Fransa ile eşleşir" gibi öğrenilmiş örüntüleri taşır.
- **Residual bağlantı** — bina kuralı — katın çıktısını girdisinin üstüne *ekler*; alttaki hiçbir şey silinmez.

*Tilki* kat kat *kahverengi-hızlı-tilki*ye, sonra *harekete geçmek üzere olan özne*ye büyür; alt katlar yazım ve dil bilgisini, üst katlar olguları ve mantığı üstlenir. Bilgi de çoğunlukla ambarlarda yaşar — tüm **parametrelerin**
kabaca üçte ikisi — cümle olarak değil, milyarlarca ağırlığa yayılmış
halde. Modeli büyütmek çoğunlukla ambarı büyütmektir: GPT-2'nin meşhur
1,5 milyar parametresi (2019) bugün trilyonlara vardı; **mixture of
experts (MoE)** ise her kata birçok ambar koyar, her token en iyi
bir-iki tanesine yönlendirilir.

Çatıda kule borcunu öder: bir *tahmin*. Son token'ın nihai vektörü — artık
bağlamın tamamını kodlar — modelin bildiği her token'la iç çarpıma
girer (GPT-2'de ~50.000):

> puan(aday) = son vektör · adayın embedding'i

**Softmax** puanları olasılığa çevirir. "Bir varmış bir"den sonra kütle
"yokmuş"a yığılır; "En sevdiğim şehir"den sonra yüzlerce şehre dağılır.
İki durumda da model tek sorusunu cevaplar: *sırada ne gelmesi muhtemel?*

## 5. Eğitim ve ölçek

Makinedeki her sayı rastgele gürültü olarak başlar. Onları **ön eğitim** (pretraining) ayarlar: modele trilyonlarca token gerçek metin gösterin,
sıradakini gizleyin, tahmin ettirin. Cevap anahtarı bedavadır — metinde
gerçekten sonra gelen token'dır; veri kendi kendini notlandırır. Her tahmini **kayıp** (loss) puanlar: kayıp = −log p(doğru token) — doğruya %90
vermek ≈ 0,1'e, %20 vermek ≈ 1,6'ya mal olur (karne: **perplexity** =
e^(ortalama kayıp)). **Gradyan inişi** (gradient descent) her parametreyi yokuş aşağı
minicik bir adım kaydırır — sisli bir iniş, trilyonlarca kez — ta ki model, eğitim verisinin JPEG gibi sıkıştırılmışı olana dek: resim kalır, pikseller kalmaz.

```mermaid
flowchart LR
    A["gerçek metni göster"] --> B["sıradaki token'ı gizle"]
    B --> C["model tahmin eder"]
    C --> D["kayıp = doğrunun −log p'si"]
    D --> E["gradyan inişi — minicik bir adım"]
    E -->|"trilyonlarca kez tekrar"| A
```

Ölçeğin getirisi öngörülebilirdir. **Ölçek yasaları** (scaling laws) — kayıp ≈
a · C^(−α), log-log kâğıdında düz çizgi — OpenAI'ın GPT-4'ün nihai
kaybını 10.000 kat küçük denemelerden öngörmesini sağladı; DeepMind'ın
**Chinchilla**sı tarifi parametre başına ~20 token'a sabitledi — 70
milyarlığı, 280 milyarlık Gopher'ı geçti. İki şerh: beceriler yine de
sıçramayla gelebilir (**beliren yetenekler**, emergent abilities) ve kaliteli açık metin
tükeniyor — hesap, cevap anına kayıyor: 7. bölümün akıl yürüten
modelleri.

## 6. Otomatik tamamlamadan asistana

Ön eğitimin ürünü bir **taban modeldir** (base model): metni sürdüren bir makine, o
kadar. "Fransa'nın başkenti nedir?" deyin; "Paris." alabilirsiniz — ya
da dokuz quiz sorusu daha — ya da "diye sordu öğretmen; kimse parmak
kaldırmadı." Hepsi sadık devamlardır; cevabı çekip çıkarmak bir
zamanlar başına kendiniz "S: … C:" yazmayı gerektirirdi — prompt
mühendisliği orada doğdu. İki ucuz aşama asistan yapar:

- **Talimat eğitimi** (instruction tuning) — on binlerce soru → ideal cevap çiftiyle eğitime devam edilir; ta ki yardımcı cevap en olası devam olana dek.
- **RLHF** — insanlar aday cevapları karşılaştırır, bir ödül modeli (reward model) zevklerini öğrenir, LLM ona doğru ayarlanır: ton, dürüstlük, ret — örneklerin yazamadığı. (Daha da ucuzu **LoRA**: modeli dondurup yanına minik adaptör matrisleri eğitir.)

Vurucu son: GPT-3, ChatGPT'den iki yılı aşkın süre önce
vardı. Devrim bu aşamalardı, daha büyük ağ değil.

## 7. Üretim: plan değil, döngü

Model olasılıkları hesaplar, bir token **örnekler** (sampling), ekler ve özel bir durdurma token'ına (stop token) dek tekrarlar — her yeni token, anında bir sonraki
tahminin girdisidir. Çekilişi üç kadran yönetir. "Gökyüzü ...ydi"den
sonra: *mavi* %60, *karanlık* %10, …, *patates* %0,0001.

- **Temperature**, softmax'tan önce her puanı T'ye böler — T = 0, açgözlü (greedy) ve neredeyse deterministik seçimdir; yüksek T, *karanlık* ile *gri*yi
  yarıştırır. SQL için düşük, beyin fırtınası için yüksek.
- **Top-k**, en olası k token'ı tutar — *patates* silindi.
- **Top-p**, olasılığın örneğin %90'ını örten en küçük kümeyi tutar —
  model eminse iki token, kararsızsa seksen.

Önce buda, sonra çek: cevaplar bu yüzden günden güne değişir ve gökyüzü bu yüzden asla patates olmaz.

Döngü, "adım adım düşün"ün sırrını da açıklar — sayfa, modelin tek
karalama defteridir. 17 × 24 tek hamlede istenirse cevabı tek tahminde
tutturmak zorundadır; 17 × 24 = 340 + 68 = 408 yazmasına izin verilirse
her ara adım bağlama katılır ve sonraki tahmini keskinleştirir. Akıl
yürüten modeller tam bunu sanayileştirir.

Tabloyu ekonomik bir gerçek tamamlar. Eğitim, belgeleri paralel işler;
sohbet, token'ları teker teker üretir — ve **KV cache**, 3. bölümün
sözünü bozdurur: geçmiş anahtarlarla değerler hiç değişmez, bir kez
hesaplanır ve saklanır. "Ben seni" bağlamıyla tek tur:

1. "Ben" ile "seni"nin önbellekteki K, V'leri taze sorgu Q₃ ile buluşur — ağırlıklar %30 / %70 düşer.
2. Karışım kulede yükselir; softmax *seviyorum* der (%85); çekiliş onu seçer.
3. "seviyorum" için K₃, V₃ hesaplanır ve önbelleğe katılır; döngü yeniden başlar. **Q hep taze hesaplanır; K ile V hep önbellekten
gelir** — bütün hikâye bu cümledir. Bunu siz de hissettiniz: uzun bir
istemin ilk kelimesinden önceki duraklama, önbelleği kuran
**prefill**'dir; sonrası akar. Fatura bellektir:

> cache = 2 × katman × bağlam × genişlik × bayt ≈ 2 × 32 × 100.000 × 4.096 × 2 ≈ **52 GB** — tek uzun sohbet

— önbelleğe alınmış girdinin daha ucuz fiyatlanması bundandır; öbür
büyük kaldıraç **quantization**'dır: ağırlıkları daha az bitle sakla
(16 → 8 → 4). Çıkarım matematikten çok bayt taşımaya takılır; küçük
ağırlık, hızlı ve ucuz cevap demektir.

## 8. Sizi hatırlamaz

Eğitimden sonra parametreler **donar**. Her mesaj, tüm konuşmayı ağdan
yeniden geçirir — uzun süreli hafızası olmayan, her sabah dosyanın
tamamı eline verilen parlak bir danışman. Hafıza sandığınız, bağlam
penceresidir. Öbür yüzü **bağlam içi öğrenmedir** (in-context learning): "deniz → mer, ev →
maison, kedi → ?" gösterin; *chat* çıkar — görev yalnızca istemden
öğrenildi, tek parametre değişmedi. Pratik prompt mühendisliği tam
budur: bağlamı, istenen devam en olası olacak şekilde dizmek.

## 9. Neden uyduruyor

2023'te *Mata v. Avianca* davasının avukatları, ChatGPT'nin uydurduğu
altı emsal kararı mahkemeye sundu — model, "gerçekler mi?" sorusuna
"evet" demişti. 5.000 dolarlık ceza, "yapay zekâ halüsinasyonu"nu
meşhur etti. Sır yok: model bir olasılık motorudur, veritabanı değil.
Eğitim verisinin zengin olduğu yerde en olası devam genellikle
doğrudur; ince olduğu yerde model yine de cevap *biçiminde* bir şey
üretir — optimize ettiği, doğru değil makuldür. Bu yalan değildir —
yalan, doğruyu bilmeyi gerektirir. Bu, cümleyi tamamlamaktır. Çözümler bir merdivendir — sırayla tırmanın, her basamak daha pahalı:

- **prompting** davranışı bağlamda biçimler;
- **RAG** taze bilgiyi cevap anında getirir;
- **fine-tuning** kalıcı olması gerekeni içine işler.

Ve kaynakları kendiniz kontrol edin: avukatların atladığı adım.

## Bütün hikâye beş satırda

1. Metin → **token** → **embedding** (anlamın koordinatları, konum dahil).
2. **Attention** (sorgu·anahtar·değer) her token'ın vektörünü bağlamıyla
   karıştırır — yalnızca geriye bakarak; bilgiyi ileri beslemeli katmanlar
   depolar.
3. **Ön eğitim** = ölçekli sıradaki-token tahmini; ölçek yasaları kazancı
   öngörülebilir kılar; talimat eğitimi + RLHF taban modeli asistana çevirir.
4. Üretim = örnekle, ekle, tekrarla — temperature, top-k, top-p çekilişi
   ayarlar; KV cache bunu ödenebilir kılar.
5. Donmuş ağırlıklar; hafıza bağlam penceresidir; akıcıdır çünkü *makul*ü
   optimize eder — uydurmasının sebebi de aynıdır.

Ve bir dahaki sefere biri bu modellerin nasıl çalıştığını sorduğunda — bir
mülakatçı, bir öğrenci ya da içinizdeki meraklı ses — modelin başladığı
yerden başlayın: sıradaki token'dan.

Son bir şey. Bu yazının ilk satırında zihniniz boşluğa "yokmuş" yazdı —
anında, emin, saf örüntüden. Artık bir makinenin aynısını nasıl yaptığını
tam olarak biliyorsunuz. Bütün hikâye bu.

## Daha derine inmek için

- Vaswani vd., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — orijinal Transformer makalesi; yorgundu/genişti örneği onlarındır.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — klasikleşmiş görsel anlatım.
- Ebrahim Pichka, [What are Query, Key, and Value in the Transformer Architecture?](https://medium.com/data-science/what-are-query-key-and-value-in-the-transformer-architecture-and-why-are-they-used-acbe73f731f2) — QKV sezgisinin, yumuşak sözlük bakışı dahil, özenli bir açılımı.
- Andrej Karpathy, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — bütün makinenin gözünüzün önünde kodla inşası.
