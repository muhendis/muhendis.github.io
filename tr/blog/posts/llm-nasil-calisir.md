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

**Tokenizer** (standart algoritmanın adı **byte-pair encoding**, BPE), metni **token** denen parçalara böler — "için" tek parça,
"inanılmaz" belki "inan + ılmaz" — ve her birine bir kimlik numarası verir. LEGO gibi düşünün: dil, en çok yeniden kullanılan parçalarına ayrılır — yaygın kelimeler tek parça kalır, nadirler küçük parçalardan kurulur.
Kaba kural: 100 token ≈ 75 kelime; yani "128K bağlam penceresi" aşağı yukarı
bir roman tutar. Bilmeye değer bir sonuç: model harf görmez, yalnızca token
kimlikleri görür — "strawberry"deki r'leri saymanın meşhur zorluğu bundandır.
Bir tablonun *fotoğrafındaki* fırça darbelerini saymak gibidir.

## 2. Anlam taşıyan sayılar

Her token bir **embedding**'e dönüşür: binlerce sayıyı kadran gibi düşünün — biri resmiyet, biri zaman, biri duygu, çoğuysa hiçbir insanın adlandırmadığı nitelikler için. Bu kadranlar, bir anlam haritasındaki koordinatları
gibi davranan uzun bir sayı listesi. *Kral*, *kraliçe*ye yakın, *hesap tablosu*na uzaktır. Yönler de anlam taşır: *Paris*ten *Fransa*ya giden ok, *Roma*dan *İtalya*ya giden okla paraleldir — bir "başkenti-olmak" yönü. Kelime aritmetiği bile çalışır: *kral − erkek + kadın*, *kraliçe*nin yakınına düşer. Bu haritayı kimse çizmedi; öğrenildi. Bir torba koordinatta
sıra olmadığından, her token'ın **konumu** da işlenir: "köpek adamı ısırdı",
"adam köpeği ısırdı"dan farklı kalmalıdır.

## 3. Transformer: bir bağlam makinesi

Embedding tek başına "yüz"ün ne olduğunu söyleyemez — surattaki yüz mü, sayı
olan yüz mü? Anlam komşulara bağlıdır. **Transformer** — modern bütün
modellerin arkasındaki tasarım, GPT'deki T — komşuları okumak için
kurulmuştur. Önceki mimariler metni soldan sağa sindirir, o ana dek görülen
her şeyi mesafeyle solan tek bir dar akan hafızadan geçirirdi. 2017 makalesinin radikal hamlesi — ve başlığının, *Attention Is All You
Need*, "ihtiyacınız olan tek şey attention", gerçek anlamı — o makineyi
atıp tek bir mekanizmayı, **attention**'ı tutmaktı: her token, diğer her
token'a *doğrudan* ve aynı anda baksın, neyin önemli olduğuna kendisi
karar versin.

Bu kararı, orijinal makalenin kendi örneğinde izleyin:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Tek kelime değişir, "o" taraf değiştirir. Siz bunu anında çözdünüz;
**attention**, modelin çözme biçimidir. Bütün numara tek cümleye iner: **bir kelimenin bağlamı, diğer kelimelerin ağırlıklı bir karışımıdır ve attention'ın bütün işi ağırlıkları seçmektir.** Ve ağırlıklar *mesafeden* gelemez — "o"yu çözen kelime yirmi token geride olabilir, bitişikteki kelime ise gürültü. Ağırlıklar *içerikten* hesaplanmalı — ve öğrenilmelidir.

Transformer'ın cevabının belirgin bir biçimi var. Yığındaki her katman tam
iki alt katman taşır. Birincisi **self-attention** — "self" (öz), çünkü
cümle *kendine* dikkat eder: her kelime, aynı metnin diğer kelimelerine
bakar ve kendi vektörünü onların ışığında yeniden yazar. İkincisi **ileri
beslemeli ağ**: her konuma tek başına uygulanan, etrafına hiç bakmayan
küçük bir sinir ağı — önce bağlamı birlikte topla, sonra tek başına sindir.
Bu bölüm birinci alt katmanı açar; ikincisi sözü 4. bölümde alacak.

### Q, K, V — mekanizma, sayılarla

Bir token ağırlıklarını nasıl seçer? Aynı anda üç rol oynayarak — her biri
kendi vektörünün küçük, öğrenilmiş bir dönüşümü:

- **sorgu (query)** — ne arıyorum?
- **anahtar (key)** — başkaları beni nasıl bulsun?
- **değer (value)** — seçilirsem ne teslim ederim?

YouTube aynı üçlüyle çalışır: yazdığınız metin sorgudur, her videonun
başlığı bir anahtar, videoların kendisi ise değerdir. Modelde üçü de tek
kaynaktan gelir — token'ın embedding'inden — üç öğrenilmiş sayı tablosuyla,
yani ağırlık matrisleri **W_Q, W_K, W_V** ile çarpılarak (her biri sıradan
bir yoğun katmandır). Aynı kelime, üç kıyafet; ve her token üçünü aynı anda
giyer — hem arayıcı, hem bulunabilir, hem devredilecek içerik.

Peki kıyafetleri atlayıp ham embedding'leri karşılaştırsak? Olmaz; çünkü
embedding, kelimenin her yönünü birden karıştırır — dil bilgisi, anlam,
konum — oysa iyi bir aramaya bunların yalnızca biri lazımdır. Üç tablo,
*tam da bu işin gerektirdiği yönü* çeker: "yorgun olabilir" üzerinden
eşleşen bir sorgu-anahtar çifti, "f ile başlar" üzerinden değil. Değer de
bir seçimdir — seçilen token bütün embedding'ini değil, aktarılmaya değer
dilimi devreder; hangi dilim olduğuna W_V karar verir. Ortaya çıkan şey bir
**yumuşak sözlük** gibi davranır: gerçek sözlük anahtarı ya tam eşleştirir
ya da hiçbir şey döndürmez; attention her anahtarla *kısmen* eşleşir ve her
değerden oransal bir dilim alır.

Kadro bu. Arkasındaki aritmetik ise yalnızca iki hamle:

- **İç çarpım** — iki vektörü basamak basamak çarpıp hepsini topla:
  a · b = a₁b₁ + a₂b₂ + a₃b₃ + … Örneğin
  [2, 1, 0] · [3, 1, 4] = 6 + 1 + 0 = **7**. Tek bir sayı çıkar: iki vektör
  aynı yönü gösteriyorsa büyük, göstermiyorsa küçük. Bir benzerlik ölçer.
- **Ağırlıklı toplam** — birkaç vektörü yüzdelere göre karıştır; bir tarif
  gibi: şundan %60, bundan %30.

Şimdi çalıştıralım. "Hızlı kahverengi tilki"yi, model *tilki* üzerinde
çalışırken izleyin — dört adım; sonda karşılaşacağınız resmî formülün
aynısı, aynı sırayla.

**1. Adım — Puanla: bana kim önemli?** *Tilki*nin sorgusu, kendisininki
dahil her kelimenin anahtarıyla iç çarpıma girer:

> puanᵢ = Q(tilki) · K(kelimeᵢ)

| çift | iç çarpım | okunuşu |
|---|---|---|
| Q(tilki) · K(Hızlı) | 2,1 | biraz ilgili |
| Q(tilki) · K(kahverengi) | 4,0 | çok ilgili |
| Q(tilki) · K(tilki) | 5,4 | kendisi — en çok |

**2. Adım — Ölçekle: sayıları dizginle.** Yüzdelere geçmeden önce her puan,
anahtar vektörünün boyutu olan **√dₖ**'ye bölünür:

> ölçekliᵢ = puanᵢ ÷ √dₖ

Büyük vektörlerde iç çarpımlar devasalaşır; devasa puanlar softmax'ı
ya-hep-ya-hiç ağırlıklara doyurur ve öğrenmeyi durdururdu. *Ölçekli* iç
çarpım attention'daki "ölçekli" bu bölmedir. (Oyuncak sayılarımız okunur
kalsın diye tablodaki puanları ölçeklenmiş halleri sayın.)

**3. Adım — Yüzdele: puanları tarife çevir.** Bu, **softmax**'tır ve
yalnızca iki hamledir. Önce her puan için *e* üzeri puan alınır — bu, her
sayıyı pozitif yapar ve aralarını açar. Sonra her sonuç toplama bölünür —
artık toplamları tam %100'dür:

> ağırlıkᵢ = e^puanᵢ ÷ (e^puan₁ + e^puan₂ + … )

| çift | puan | e^puan | toplam içindeki pay |
|---|---|---|---|
| Q(tilki) · K(Hızlı) | 2,1 | 8,2 | **%3** |
| Q(tilki) · K(kahverengi) | 4,0 | 54,6 | **%19** |
| Q(tilki) · K(tilki) | 5,4 | 221,4 | **%78** |
| | | toplam ≈ 284 | %100 |

Son satırı formülle sınayın: 221,4 ÷ 284 ≈ 0,78 — %78. Üstelin yaptığına da dikkat edin: 5,4, 4,0'ın yalnızca biraz üstünde; ama %78,
%19'un dört katı — softmax öndekileri ödüllendirir, geridekileri aç bırakır.
Model az önce, sayılarla, her kelimenin ne kadar dikkati hak ettiğine karar
verdi. (Evet, token kendine de dikkat eder — genellikle en çok.)

**4. Adım — Karıştır: tarifi pişir.** Yeni *tilki* vektörü, *değerlerin*
ağırlıklı toplamıdır:

> yeni vektör = ağırlık₁ × V(kelime₁) + ağırlık₂ × V(kelime₂) + …
> tilki_yeni = 0,03 × V(Hızlı) + 0,19 × V(kahverengi) + 0,78 × V(tilki)

Sonuç artık sözlükteki *tilki* kelimesi değildir; *bu-belirli-hızlı-
kahverengi-tilki*dir ve bir sonraki katmana giden, bu zenginleşmiş
vektördür.

O formülden önce iki not. Birincisi, bu dansta *öğrenilmiş* olan tek şey
Q, K ve V'yi üreten üç tablodur; iç çarpımlar, softmax, ağırlıklı toplam —
sabit aritmetiktir, içlerinde öğrenme yoktur. Modele tilkilerin kahverengi
olduğunu hiçbir kural söylemedi: tablolar, trilyonlarca tahmin boyunca işe
yarar ağırlıklar kendiliğinden çıkana dek ayarlandı. İkincisi, bedeli: her
token diğer her token'ı puanladığından iş, uzunluğun *karesiyle* büyür —
O(n²). Bağlamı ikiye katlayın, maliyet dörde katlansın; milyon token'lık
pencereler bir kutucuk değil, mühendislik başarısıdır.

Şimdi ödül. Yukarıdaki her şey tek satıra sıkışır — 2017'den beri her
makalede basılan formül:

> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**

Soldan sağa okuyun; tilki izinin sembollere dökülmüşüdür: 1. Adım QKᵀ,
2. Adım √dₖ'ye bölme, 3. Adım softmax, 4. Adım V ile çarpma — aynı dört
hamle, aynı sıra, her seferinde. Büyük harfler son bir armağan daha taşır:
buradaki Q, K ve V birer matristir — bütün token'ların vektörleri tek
blokta üst üste — yani bu tek satır, aramayı *bütün* token'lar için aynı
anda, saf matris çarpımı olarak yürütür. Döngü yok; tam da GPU'ların silip
süpürdüğü iş biçimi.

### Tek yön

Sonuçları büyük bir ayrıntı: üretim sırasında her token yalnızca *geriye*
bakabilir. *Tilki*, *kahverengi*yi görür; *kahverengi*, *tilki*yi asla. Bu
maske, modeli bir *sıradaki*-token tahmincisi yapan şeydir — ve geçmiş bir
token'ın anahtarıyla değerinin, bir kez hesaplandıktan sonra asla
değişmemesinin sebebi de budur: sonradan gelen hiçbir şey onlara dokunamaz. Bir kenara yazın; 7. bölümde KV cache olacak.

Bu maske aynı zamanda iki ünlü model ailesinin anahtarıdır. Maskeyle kurulan model soldan sağa okur, bu yüzden *yazar* — bu, **decoder** ailesidir: GPT ve neredeyse bütün modern LLM'ler. Maskesiz kurulan model iki yönü birden görür ve yazmak yerine *sınıflandırır* — bu da **encoder** ailesidir: BERT. (Adlar, transformer'ın ikisini yan yana kullanan orijinal çeviri tasarımından kalmadır.) Tek mimari anahtar, bütün soy ağacı.

### Birçok kafa

Katman başına tek bir ağırlıklama kaba kalırdı — bir kelimenin bir
komşusundan dil bilgisine, başka birinden göndermeye ihtiyacı var. Bu yüzden
her katman, her biri kendi Q/K/V mercekleriyle donanmış birçok attention
**kafasını** paralel çalıştırır ve her kafa izleyeceği ilişkiyi kendisi
öğrenir: biri söz dizimini izler, biri "o"yu çözer, biri sıfatı isme bağlar.
Bu rolleri kimse atamaz; kendiliğinden belirir — çünkü her biri sıradakini
tahmine yarar. İki somut ayrıntı resmi tamamlar. Kafalar dilimler halinde
çalışır: orijinal tasarımda her kafa, 512 sayılık embedding'i 64'e indirir;
böylece sekiz kafa, tam genişlikte tek kafayla aşağı yukarı aynıya mal
olur. Ve makalenin kendi görselleştirmeleri iş bölümünü gösterir — "o"
kodlanırken bir kafa *hayvan*a, bir başkası *yorgun*a kilitlenir: gönderge
ve gerekçe, aynı anda izlenir.

Bütün 3. bölüm, tek kartta:

| soru | cevap |
|---|---|
| Self-attention nedir? | Cümlenin kendine dikkat etmesi: her token, vektörünü diğerlerinin ışığında yeniden yazar |
| Q, K, V nedir? | Token başına üç rol — sorgu: *ne arıyorum?* · anahtar: *nasıl bulunurum?* · değer: *ne devrederim?* |
| Nasıl üretilirler? | embedding × W_Q, W_K, W_V — üç öğrenilmiş lineer katman; aynı kelime, üç kıyafet |
| Neden üç ayrı vektör? | Embedding, kelimenin her yönünü karıştırır; her rol yalnızca kendi işinin yönünü çeker |
| Ağırlıklar neden öğrenilir? | Doğru ağırlık kelime mesafesinden çıkmaz; içerikten hesaplanır |
| Hesap hangi sırayla akar? | puanla (Q·K) → ölçekle (÷√dₖ) → softmax'la → karıştır (×V) |
| İleri beslemeli ağ nedir? | Bağlam toplandıktan sonra her token'ı tek başına sindiren küçük ağ — 4. bölümün bilgi ambarı |
| Nerede, ne zaman, kim? | Her katmanda, her kafada, tüm token'lar aynı anda — Vaswani vd., 2017 |

## 4. Katmanlar: bilgi nerede yaşıyor

Bir attention alt katmanı artı bir ileri beslemeli alt katman, bir
**katman** eder — transformer ise bunlardan örülü bir kuledir: aynı kat
planı, onlarca-yüzü aşkın kez üst üste. Bir token'ın vektörü zemin
kattan girer ve yukarı çıkar. Kulenin neden bu kadar yüksek olduğunu
görmek için onu izleyin.

Her katta aynı iki adımlık rutin işler. Önce attention — kütüphaneci —
diğer token'lardan bağlam getirir: *tilki*, *kahverengi* ve *hızlı*
olduğunu, sıçramanın ortasında olduğunu öğrenir. Sonra **ileri beslemeli
ağ** — ambar — toplanan malzemeyi tek başına, etrafına bakmadan sindirir
ve eğitimin bıraktığı örüntülerle zenginleştirir: "Paris, Fransa ile
eşleşir". Her katın çıktısını da **residual bağlantı** denen bir bina
kuralı yönetir: çıktı, girenin *yerine geçmez*, üstüne *eklenir* —
kitabın kenar boşluğuna düşülen notlar gibi. Alt katların yazdığı hiçbir
şey silinmez; üst katlara varıldığında *tilki*, sözlükteki kelimeden
önce *kahverengi-hızlı-tilki*ye, sonra *harekete geçmek üzere olan
özne*ye büyümüştür.

Katlar uzmanlaşır da — üstelik görevleri kimse dağıtmadan: alt katlar
yazımı ve dil bilgisini halleder, üst katlar olguları ve uzun menzilli
mantığı — banyoda beliren fotoğraf gibi: önce hatlar, sonra yüzler. Bu
iş bölümü eğitimden kendiliğinden doğar.

Peki bilgi bu kulenin neresinde yaşıyor? Çoğunlukla ambarlarda: modelin
**parametrelerinin** — öğrenilmiş sayılarının — kabaca üçte ikisi ileri
beslemeli katmanlardadır. Bölüm başlığının dürüst cevabı bu yüzden
şudur: parmakla gösterebileceğiniz hiçbir yerde. Bir olgu, kayıtlı bir
cümle değildir; milyarlarca ağırlığa yayılmıştır. Modeli büyütmek de
çoğunlukla ambarı büyütmektir — GPT-2, 2019'da 1,5 milyar parametreyle
manşet olmuştu; öncü modeller bugün trilyonlara varıyor. Modern bir
dokunuş olan **mixture of experts (MoE)**, kat başına birçok ambar kurar
ve küçük bir yönlendiriciyi her token'ı en iyi bir-iki tanesine
gönderecek şekilde eğitir: devasa toplam kapasite, ama her token bunun
yalnızca bir kesrini öder.

Bir durak kaldı: çatı. Kule, vektörleri hatırları için cilalamıyordu —
bize bir *tahmin* borçlu. En tepede model, **son** token'ın nihai
vektörünü alır — onca kattan sonra bu vektör tek bir kelimeyi değil,
oraya varan bağlamın tamamını kodlar — ve onu bildiği her token'la
puanlar; hamle, attention'ın kullandığının aynısıdır, bir iç çarpım.
Üstelik birçok modelde puanlama, 2. bölümdeki embedding tablosunun ta
kendisiyle yapılır, bu kez tersinden:

> puan(aday) = son vektör · adayın embedding'i — modelin bildiği her token'a bir puan (GPT-2'de ~50.000 tane)

**Softmax** — attention'daki yüzde çeviricinin aynısı (3. Adım) — bu
puanları, toplamı %100 eden olasılıklara çevirir. "Bir varmış bir"den
sonra kütle "yokmuş"a yığılır; "En sevdiğim şehir"den sonra yüzlerce
şehre dağılır. İkisi de modelin cevapladığı tek sorunun doğru cevabıdır:
*sırada ne gelmesi muhtemel?*

## 5. Eğitim ve ölçek

**Ön eğitim**, kulenin açık bıraktığı soruyu cevaplar: makinenin şu ana
dek gördüğünüz her parçası — embedding haritası, Q/K/V tabloları,
ambarlar — başlangıçta rastgele gürültüden ibaret bir sayı ızgarasıdır
ve kimse onları elle ayarlamaz. Ayarlayan şey, dersi olmayan, yalnızca
sınavı olan bir okul gibi çalışır. Trilyonlarca token gerçek metin alın
— web sayfaları, kitaplar, kod. Modele bir parça gösterin, sıradaki
token'ı gizleyin, bir tahmin isteyin. Cevap anahtarının maliyeti
sıfırdır, çünkü cevap, metinde gerçekten sonra gelen token'ın ta
kendisidir: veri kendi kendini notlandırır. Ham internet ölçeğinde
metnin yetmesinin sebebi budur — insan etiketi gerekmez.

Her tahmine bir not verilir: **kayıp** — doğru karşısındaki şaşkınlığın
ölçüsü:

> kayıp = −log p(doğru token)

Doğru token'a %90 olasılık verin, kayıp −log 0,9 ≈ 0,1'dir: neredeyse
hiç şaşırmamış. %20 verin, kayıp −log 0,2 ≈ 1,6'dır: fena şaşırmış.
Nottan sonra düzeltme gelir: **gradyan inişi**, her parametreyi kaybı
küçültecek yönde minicik bir adım kaydırır — sisli bir dağdan iner gibi:
vadiyi göremezsiniz, yalnızca ayağınızın altındaki eğimi hissedersiniz;
yokuş aşağı küçük bir adım, sonra bir trilyon tane daha. (Çok tahminin
karnesi **perplexity** = e^(ortalama kayıp)'tır. Ortalama kayıp 1,6 ise
e^1,6 ≈ 5 — beş eşit olası kelime arasında seçim yapıyormuş kadar
kararsız. Düşüğü iyidir; ama ölçtüğü tahmindir, işe yararlık değil.)

Bu döngüyü yeterince uzun döndürün, dikkate değer bir şey olur: kimsenin
programlamadığı beceriler belirir, çünkü işe yararlar. Polisiye romanın
son bölümünü tahmin etmek, kimin cinayet sebebi olduğunu izlemeyi
gerektirir — model izlemeyi öğrenir, çünkü izleyince kayıp düşer.
Sonunda elde kalan, eğitim verisinin JPEG gibi sıkıştırılmışıdır: resim
kalır, pikseller kalmaz.

Peki neden herkes daha büyüğünü kurmak için yarıştı? Çünkü getirisi
öngörülebilir. **Ölçek yasaları**, kaybın hesaplamayla birlikte
pürüzsüzce düştüğünü söyler:

> kayıp ≈ a · C^(−α) — C hesaplama; a ile α, küçük ve ucuz denemelerden ölçülen sabitler

Log-log kâğıdında bu düz bir çizgidir — çizgiyi uzatın ve bin kat büyük
bir modelin ne kadar iyi olacağını, *parasını ödemeden önce* okuyun. Yüz
milyon dolarlık eğitim koşularını kumardan plana çeviren budur: OpenAI,
GPT-4'ün nihai kaybını 10.000 kat küçük denemelerden önceden tahmin
etti. DeepMind'ın **Chinchilla** çalışması tarife denge kuralını ekledi:
parametre ve veri birlikte büyümeli, parametre başına yaklaşık 20 eğitim
token'ı — bu aritmetikle beslenen 70 milyarlık modelleri, kendi 280
milyarlık Gopher'larını geçti.

Hikâyeyi iki dürüst şerh kapatır. Kayıp eğrisi pürüzsüzdür ama beceriler
sıçramayla gelebilir: bir model üç basamaklı aritmetikte boy boy
başarısız olur, sonra birden güvenilir biçimde yapar — **beliren
yetenek**; eğri kaybı öngörür, yetenekleri değil. Yakıt da sonlu:
kaliteli açık metin tükenmek üzere; sınır bu yüzden sentetik veriye ve
hesaplamayı cevap anında harcamaya — 7. bölümün akıl yürüten
modellerine — kayıyor.

## 6. Otomatik tamamlamadan asistana

Ön eğitimin ürünü **taban modeldir**: metni sürdüren bir makine, başka
hiçbir şey değil. "Fransa'nın başkenti neresi?" deyin; "Paris."
alabilirsiniz — ya da dokuz quiz sorusu daha, çünkü internette quizler sürü
halinde gezer — ya da bir kurgu sahnesi: "diye sordu öğretmen, kimse parmak
kaldırmadı." Hepsi sadık birer devamdır. Bir zamanlar cevap koparmak
"Soru: … Cevap:" kalıbı yazmayı gerektirirdi — cevabı en olası devam yapmak
için; prompt mühendisliği orada doğdu.

Onu asistan yapan iki ucuz aşama var. **Talimat eğitimi**: on binlerce
yazılı soru → ideal cevap örneğiyle eğitime devam edin; "yardımcı biçimde
cevapla" en olası devam haline gelsin. **RLHF** (insan geri bildirimiyle
pekiştirmeli öğrenme): insanlar aday cevapları karşılaştırır, bir ödül
modeli bu zevki öğrenir, LLM ona doğru ayarlanır — karşılaştırmak, kusursuz
yazmaktan çok daha kolaydır ve karşılaştırmalar, örneklerin anlatamadığını
yakalar: ton, emin olmadığında dürüstlük, zararı geri çevirmek.

İki aşama da, ön eğitimin binlerce GPU'da geçen aylarının küçük bir kesrine
mal olur — o bile fazla geldiğinde **LoRA**, modeli dondurup yanına minicik
düşük ranklı adaptör matrisleri eğitir: parametrelerin kırıntısıyla
fine-tuning'e yakın kalite, lens gibi takılıp çıkarılan adaptörlerle.
Vurucu gerçek: GPT-3, ChatGPT'den iki yıldan fazla bir süre önce vardı. Devrim daha büyük ağ
değil, bu aşamalardı.

## 7. Üretim: plan değil, döngü

Model olasılıkları hesaplar, bir token **örnekler** (ağırlıklı çekiliş),
ekler ve tekrarlar — her yeni token anında bir sonraki tahminin
girdisidir — ta ki "bitirdim" diyen özel bir durdurma token'ına kadar.

Çekilişi üç düğme yönetir. "Gökyüzü"nden sonra liste şöyle olabilir:
*maviydi* %60, *karanlıktı* %10, …, *patatesti* %0,0001:

- **Temperature**, softmax'tan önce her puanı T'ye böler — çekiliş
  softmax(puan ÷ T) kullanır. T 1'in altındaysa
  aralar açılır, lider neredeyse her şeyi alır — T = 0 açgözlü çözümlemedir,
  neredeyse deterministik (yığınlama ve kayan nokta sırası yine küçük
  sapmalar bırakır). T 1'in üstündeyse aralar daralır; *karanlıktı* ve
  *griydi* yarışa girer. SQL için düşük, beyin fırtınası için yüksek.
- **Top-k**, yalnızca en olası k token'ı tutar, kuyruğu siler — *patatesti*
  dahil.
- **Top-p**, olasılığın örneğin %90'ını kapsayan en küçük kümeyi tutar —
  model eminken iki token, kararsızken seksen.

Önce kes, sonra kalanlar arasından çek: cevapların günden güne değişmesi ve
gökyüzünün asla patates olmaması bundandır.

Döngü, "adım adım düşün"ün neden işe yaradığını açıklar: sayfa, modelin tek
karalama defteridir. 17 × 24 tek hamlede istendiğinde cevabı tek tahminde
tutturmak zorundadır; "17 × 24 = 17 × 20 + 17 × 4 = 340 + 68 = 408" yazması
serbestse her ara adım bağlama katılır ve sonraki tahmini keskinleştirir.
Akıl yürüten modeller tam olarak bunu endüstrileştirir.

Resmi bir karşıtlık tamamlar. *Eğitimde* model, belgeleri bütün halinde
görür ve her token'ı paralel işler — transformer'ların GPU'ları doyurup
ölçeklenmesini sağlayan paralellik budur. *Çıkarımda* — sohbette — metin
token token gelir; **KV cache**, 3. bölümdeki asimetriyi paraya çevirerek
bu seri döngüyü ucuzlatmak için vardır. 1.000'inci token, sorgusunu önceki
999 anahtarla karşılaştırmak zorundadır — her adımda her şeyi yeniden
okumak gibi görünür. Değildir: geçmiş anahtarlar ve değerler hiç değişmez;
bir kez hesaplanıp saklanır.

Bu önbelleği hissettiniz. Uzun bir istemde ilk kelimeden önceki duraklama,
onu kuran **prefill**'dir; sonrasında kelimeler hızla akar, çünkü her biri
yalnızca kendi bedelini öder. Uzun sohbetlerin bellek yemesi de bundandır —
önbellek her token'la, her katmanda büyür. Hesap düşündürücüdür:

> önbellek = 2 (K ve V) × katman × bağlam uzunluğu × vektör genişliği × bayt

32 katmanlı, 4.096 genişlikli, 16-bit bir modelde 100K token:
2 × 32 × 100.000 × 4.096 × 2 bayt ≈ **52 GB** — tek bir sohbet için.
"Önbelleklenmiş girdinin" daha ucuz fiyatlanması da bundandır: bedeli
çoktan ödenmiştir. Servisin öbür büyük kolu **kuantizasyondur** —
ağırlıkları daha az bitle saklamak (16 → 8 → 4); çıkarım aritmetiğe değil
bayt taşımaya takılır, küçük ağırlıklar mütevazı bir doğruluk bedeliyle
daha hızlı ve ucuz cevap demektir.

İşte bütün makine tek küçük izde. Girdi: **"Ben seni"** — model sıradaki
token'ı üretecek.

1. **Önbellek kontrolü.** "Ben" ve "seni" zaten işlendi; anahtarları ve
   değerleri (K₁V₁, K₂V₂) KV cache'te duruyor.
2. **Taze sorgu.** Yeni konum için model bir sorgu hesaplar: Q₃ — fiilen
   "şu ana kadarki her şeye göre sırada ne olmalı?" sorusu.
3. **Eşle, ölçekle, softmax'la.** Q₃ önbellekteki anahtarlarla iç çarpıma
   girer, √dₖ ile ölçeklenir ve softmax'lanır — tilki izindeki 1–3.
   adımların aynısı — ve şuraya varır:

   | karşılaştırma | dikkat ağırlığı |
   |---|---|
   | Q₃ · K₁ ("Ben") | %30 |
   | Q₃ · K₂ ("seni") | %70 |

4. **Karışım.** 4. Adım'ın aynısı — saklı değerlerin ağırlıklı toplamı:
   0,30 × V₁ + 0,70 × V₂ — *bu bağlamı* temsil eden bir vektör.
5. **Tahmin.** Bu vektör son katmanlardan ve softmax'tan geçer:
   *seviyorum* %85, *gördüm* %7, *özledim* %5, … Çekiliş **"seviyorum"** der.
6. **Önbelleği uzat.** "seviyorum" için K₃ ve V₃ hesaplanıp eklenir; döngü,
   bağlam artık "Ben seni seviyorum" olarak yeniden başlar.

Q hep taze hesaplanır; K ve V hep önbellekten gelir. Bu tek cümle, KV
cache'in bütün hikâyesidir.

## 8. Sizi hatırlamaz

Eğitimden sonra parametreler **donar**. Her mesajda konuşmanın tamamı ağdan
yeniden geçer — her sabah bütün dava dosyasını isteyen, uzun süreli hafızası
olmayan parlak bir danışman gibi. Hafıza sanılan şey bağlam penceresidir — çok uzun sohbetlerin yavaşlaması ve başını unutması bundandır.
Madalyonun öbür yüzü **in-context learning**: "deniz → sea, ev → house,
kedi → ?" gösterin; model *cat* der — görevi yalnızca istemden öğrenmiştir, tek parametre değişmeden. Pratik prompt mühendisliğinin çoğu tam olarak budur: bağlamı, istenen devamı en olası devam yapacak şekilde düzenlemek.

## 9. Neden uyduruyor

2023'te *Mata v. Avianca* davasında avukatlar, ChatGPT'nin uydurduğu altı
içtihadı mahkemeye sundu — "gerçek mi?" diye sorduklarında da "evet" almıştı.
5.000 dolarlık ceza, "yapay zekâ halüsinasyonunu" meşhur etti. Mekanizma sır
değil: model bir olasılık motorudur, veritabanı değil. Eğitim verisinin zengin olduğu yerde en olası devam genellikle doğrudur. İnce olduğu yerde ise bulunamayacak kayıt yoktur — cevap *biçiminde* bir şey üretir — bu yalan değildir; yalan, doğruyu bilmeyi gerektirir. Bu, cümleyi tamamlamaktır;
çünkü optimize ettiği şey doğru değil, makuldür. Çözümler girdiyi değiştirir ve bir karar merdiveni oluşturur: **prompting** davranışı bağlam içinde biçimlendirir; **RAG** (retrieval), sık değişen bilgiyi getirir — ağırlıklara dokunmadan; **fine-tuning**, kalıcı olması gereken üslubu ya da alanı içine işler. Bu sırayla deneyin — her basamak daha pahalıdır. Araçları ekleyin ve kaynakları kendiniz kontrol edin: avukatların atladığı adım.

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
