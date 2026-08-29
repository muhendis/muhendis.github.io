Bu cümleyi bitirmeden zihniniz sıradaki kelimeyi tahmin etmeye başladı
bile. Kanıt mı? "Bir varmış, bir ___." Durduramadınız: boşluk, sizden izin
almadan kendini doldurdu.

O refleksin bir adı var — sıradaki-token tahmini — ve bu yazının tek bir
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
100 token ≈ 75 İngilizce kelime; "128K bağlam penceresi" kabaca bir
roman eder. Model harfleri değil yalnız token kimliklerini gördüğü için
"strawberry"deki r'leri saymak meşhur biçimde zordu — bir tablonun
*fotoğrafındaki* fırça darbelerini saymak gibi.

## 2. Anlam taşıyan sayılar

Sonra her token bir **embedding**'e dönüşür: uzun bir sayı listesi —
anlam haritasındaki koordinatları. Binlerce kadran düşünün: biri
resmiyet, biri zaman, çoğu hiçbir insanın adlandırmadığı nitelikler
için. Bu haritada *kral*, *kraliçe*ye yakındır; *Paris*ten *Fransa*ya
giden ok, *Roma*dan *İtalya*ya giden okla paraleldir; *kral − erkek +
kadın*, *kraliçe*nin yakınına düşer. Haritayı kimse çizmedi —
öğrenildi. Bir de her token'ın **konumu** işlenir, çünkü "köpek adamı
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

Ağırlıklarını seçmek için her token aynı anda üç rol oynar — kendi
embedding'inin üç küçük öğrenilmiş kılığı:

- **sorgu (Q)** — ne arıyorum?
- **anahtar (K)** — başkaları beni nasıl bulur?
- **değer (V)** — seçilirsem ne devrederim?

YouTube aynı üçlüyle çalışır: yazdığınız metin sorgudur, her videonun
başlığı anahtardır, videoların kendisi değerdir. Modelde üçü de
token'ın embedding'inden gelir: üç öğrenilmiş tabloyla çarpım — **W_Q,
W_K, W_V** — aynı kelime, üç kıyafet, üçü de aynı anda üstünde. Ham
embedding'ler yetmezdi: embedding kelimenin her şeyini karıştırır,
oysa arama tek seferde tek yön ister — "yorgun olabilir" üzerinden
eşleşen bir sorgu-anahtar çifti, "f ile başlar" üzerinden değil. Sonuç
bir **yumuşak sözlüktür**: her anahtar *kısmen* eşleşir, her değerden
orantılı bir dilim alınır.

Aritmetik iki hamledir: **iç çarpım** (iki vektörü konum konum çarpıp
topla — ne kadar hizalıysa o kadar büyük: bir benzerlik ölçer) ve
**ağırlıklı toplam** (vektörleri yüzdelerle karıştır, tarif gibi).
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

Son satırı doğrulayın: 221,4 ÷ 284,2 ≈ %78. Üstele dikkat — 5,4, 4,0'ın
yalnızca biraz üstünde, ama %78, %19'un dört katı: softmax liderleri
ödüllendirir.

**4. Adım — Karıştır.** Yeni *tilki*, değerlerin ağırlıklı toplamıdır:
tilki_yeni = 0,03·V(hızlı) + 0,19·V(kahverengi) + 0,78·V(tilki) —
artık sözlükteki *tilki* değil, *bu-belirli-hızlı-kahverengi-tilki*.

Yukarıdaki her şey tek ünlü satırdır:

> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**

1. Adım QKᵀ, 2. Adım bölme, 3. Adım softmax, 4. Adım V ile çarpım.
Öğrenilen *tek* parça üç tablodur; gerisi sabit aritmetik — ve tüm
token'lar matris olarak üst üste konduğundan bu tek satır bütün
aramaları aynı anda koşturur: GPU'ların bayıldığı iş. Bedeli **O(n²)**:
herkes herkesi puanlar — bağlamı ikiye katla, maliyeti dörtle.

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

Bir attention alt katmanı artı bir ileri beslemeli alt katman, bir
**katman** eder; transformer, bunlardan örülü bir kuledir — onlarca,
yüzü aşkın kat. Her katta attention — kütüphaneci — bağlamı toplar;
ileri beslemeli ağ — ambar — onu tek başına sindirir ve "Paris, Fransa
ile eşleşir" gibi öğrenilmiş örüntüleri taşır. Katın çıktısı girdisinin
üstüne *eklenir* (**residual bağlantı**), hiçbir şey silinmez: *tilki*
kat kat *kahverengi-hızlı-tilki*ye, sonra *harekete geçmek üzere olan
özne*ye büyür. Alt katlar yazım ve dil bilgisi; üst katlar olgular ve
mantık. Bilgi de çoğunlukla ambarlarda yaşar — tüm **parametrelerin**
kabaca üçte ikisi — cümle olarak değil, milyarlarca ağırlığa yayılmış
halde. Modeli büyütmek çoğunlukla ambarı büyütmektir: GPT-2'nin meşhur
1,5 milyar parametresi (2019) bugün trilyonlara vardı; **mixture of
experts (MoE)** ise her kata birçok ambar koyar, her token en iyi
bir-iki tanesine yönlendirilir.

Çatıda bir *tahmin* borcu vardır. Son token'ın nihai vektörü — artık
bağlamın tamamını kodlar — modelin bildiği her token'la iç çarpıma
girer (GPT-2'de ~50.000):

> puan(aday) = son vektör · adayın embedding'i

**Softmax** puanları olasılığa çevirir. "Bir varmış bir"den sonra kütle
"yokmuş"a yığılır; "En sevdiğim şehir"den sonra yüzlerce şehre dağılır.
İki durumda da model tek sorusunu cevaplar: *sırada ne gelmesi
muhtemel?*

## 5. Eğitim ve ölçek

Makinedeki her sayı rastgele gürültü olarak başlar. Onları **ön
eğitim** ayarlar: modele trilyonlarca token gerçek metin gösterin,
sıradakini gizleyin, tahmin ettirin. Cevap anahtarı bedavadır — metinde
gerçekten sonra gelen token'dır; veri kendi kendini notlandırır. Her
tahmini **kayıp** puanlar: kayıp = −log p(doğru token) — doğruya %90
vermek ≈ 0,1'e, %20 vermek ≈ 1,6'ya mal olur (karne: **perplexity** =
e^(ortalama kayıp)). **Gradyan inişi** her parametreyi yokuş aşağı
minicik bir adım kaydırır — sisli bir iniş, trilyonlarca kez — ta ki
model, eğitim verisinin JPEG gibi sıkıştırılmışı olana dek: resim
kalır, pikseller kalmaz.

Ölçeğin getirisi öngörülebilirdir. **Ölçek yasaları** — kayıp ≈
a · C^(−α), log-log kâğıdında düz çizgi — OpenAI'ın GPT-4'ün nihai
kaybını 10.000 kat küçük denemelerden öngörmesini sağladı; DeepMind'ın
**Chinchilla**sı tarifi parametre başına ~20 token'a sabitledi — 70
milyarlığı, 280 milyarlık Gopher'ı geçti. İki şerh: beceriler yine de
sıçramayla gelebilir (**beliren yetenekler**) ve kaliteli açık metin
tükeniyor — hesap, cevap anına kayıyor: 7. bölümün akıl yürüten
modelleri.

## 6. Otomatik tamamlamadan asistana

Ön eğitimin ürünü bir **taban modeldir**: metni sürdüren bir makine, o
kadar. "Fransa'nın başkenti nedir?" deyin; "Paris." alabilirsiniz — ya
da dokuz quiz sorusu daha — ya da "diye sordu öğretmen; kimse parmak
kaldırmadı." Hepsi sadık devamlardır; cevabı çekip çıkarmak bir
zamanlar başına kendiniz "S: … C:" yazmayı gerektirirdi — prompt
mühendisliği orada doğdu. İki ucuz aşama asistan yapar. **Talimat
eğitimi**: on binlerce soru → ideal cevap çiftiyle eğitime devam, ta ki
yardımcı cevap en olası devam olana dek. **RLHF**: insanlar aday
cevapları karşılaştırır, bir ödül modeli zevklerini öğrenir, LLM ona
doğru ayarlanır — örneklerin yazamadığını yakalar: ton, dürüstlük, ret.
(Daha da ucuzu **LoRA**: modeli dondurup yanına minik adaptör
matrisleri eğitir — parametrelerin kırıntısıyla ince ayara yakın
kalite.) Vurucu son: GPT-3, ChatGPT'den iki yılı aşkın süre önce
vardı. Devrim bu aşamalardı, daha büyük ağ değil.

## 7. Üretim: plan değil, döngü

Model olasılıkları hesaplar, bir token **örnekler**, ekler ve özel bir
durdurma token'ına dek tekrarlar — her yeni token, anında bir sonraki
tahminin girdisidir. Çekilişi üç kadran yönetir. "Gökyüzü ...ydi"den
sonra: *mavi* %60, *karanlık* %10, …, *patates* %0,0001.

- **Temperature**, softmax'tan önce her puanı T'ye böler — T = 0
  açgözlü ve neredeyse deterministik; yüksek T, *karanlık* ile *gri*yi
  yarıştırır. SQL için düşük, beyin fırtınası için yüksek.
- **Top-k**, en olası k token'ı tutar — *patates* silindi.
- **Top-p**, olasılığın örneğin %90'ını örten en küçük kümeyi tutar —
  model eminse iki token, kararsızsa seksen.

Önce buda, sonra çek: cevaplar bu yüzden günden güne değişir ve gökyüzü
bu yüzden asla patates olmaz.

Döngü, "adım adım düşün"ün sırrını da açıklar — sayfa, modelin tek
karalama defteridir. 17 × 24 tek hamlede istenirse cevabı tek tahminde
tutturmak zorundadır; 17 × 24 = 340 + 68 = 408 yazmasına izin verilirse
her ara adım bağlama katılır ve sonraki tahmini keskinleştirir. Akıl
yürüten modeller tam bunu sanayileştirir.

Resmi bir ekonomik gerçek tamamlar. Eğitim, belgeleri paralel işler;
sohbet, token'ları teker teker üretir — ve **KV cache**, 3. bölümün
sözünü bozdurur: geçmiş anahtarlarla değerler hiç değişmez, bir kez
hesaplanır ve saklanır. "Ben seni" bağlamıyla tek turu izleyin:
"Ben" ile "seni"nin önbellekteki K, V'leri taze bir sorgu Q₃ ile
buluşur (ağırlıklar %30 / %70 düşer), karışım kulede yükselir, softmax
*seviyorum* der (%85), çekiliş onu seçer ve K₃, V₃ bir sonraki tur için
önbelleğe katılır. **Q hep taze hesaplanır; K ile V hep önbellekten
gelir** — bütün hikâye bu cümledir. Bunu hissettiniz de: uzun bir
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
penceresidir. Öbür yüzü **bağlam içi öğrenmedir**: "deniz → mer, ev →
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
yalan, doğruyu bilmeyi gerektirir. Bu, cümleyi tamamlamaktır. Çözümler
bir merdivendir: **prompting** davranışı bağlamda biçimler; **RAG**
taze bilgiyi cevap anında getirir; **fine-tuning** kalıcı olması
gerekeni içine işler. Bu sırayla tırmanın — her basamak daha pahalı —
ve kaynakları kendiniz kontrol edin: avukatların atladığı adım.

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
