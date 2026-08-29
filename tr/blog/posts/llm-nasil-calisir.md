Bir dil modelinin size verdiği her yanıt aynı yöntemle üretildi: bir sonraki
küçük metin parçasını tahmin ederek, tekrar tekrar. Telefon klavyeniz bunun
minik halini yapıyor — "görüşürüz" yazın, *yarın* önerir. LLM, aynı numaranın
milyarlarca kat büyütülmüşüdür ve ilginç olan, oyunun talep ettiği şeydir:
insan metninde sıradaki kelimeyi iyi tahmin etmek için model dil bilgisini,
olguları, üslubu ve akıl yürütmenin işleyen bir taklidini özümsemek zorunda.
Aşağıdaki her şey bu fikrin dipnotudur.

## 1. Metin sayıya dönüşür

**Tokenizer** (standart algoritmanın adı **byte-pair encoding**, BPE), metni **token** denen parçalara böler — "için" tek parça,
"inanılmaz" belki "inan + ılmaz" — ve her birine bir kimlik numarası verir.
Kaba kural: 100 token ≈ 75 kelime; yani "128K bağlam penceresi" aşağı yukarı
bir roman tutar. Bilmeye değer bir sonuç: model harf görmez, yalnızca token
kimlikleri görür — "strawberry"deki r'leri saymanın meşhur zorluğu bundandır.
Bir tablonun *fotoğrafındaki* fırça darbelerini saymak gibidir.

## 2. Anlam taşıyan sayılar

Her token bir **embedding**'e dönüşür: bir anlam haritasındaki koordinatları
gibi davranan uzun bir sayı listesi. *Kral*, *kraliçe*ye yakın, *hesap
tablosu*na uzaktır; yönler ilişkidir: *Paris*ten *Fransa*ya giden ok, *Roma*dan *İtalya*ya giden okla paraleldir — bir "başkenti-olmak" yönü — ve *kral − erkek + kadın*, *kraliçe*nin yakınına düşer. Bu haritayı kimse çizmedi; öğrenildi. Bir torba koordinatta
sıra olmadığından, her token'ın **konumu** da işlenir: "köpek adamı ısırdı",
"adam köpeği ısırdı"dan farklı kalmalıdır.

## 3. Transformer: bir bağlam makinesi

Embedding tek başına "yüz"ün ne olduğunu söyleyemez — surattaki yüz mü, sayı
olan yüz mü? Anlam komşulara bağlıdır. **Transformer** — modern bütün
modellerin arkasındaki tasarım, GPT'deki T — komşuları okumak için
kurulmuştur. Önceki mimariler metni soldan sağa sindirir, o ana dek görülen
her şeyi mesafeyle solan tek bir dar akan hafızadan geçirirdi. Transformer'ın
hamlesi: her token, diğer her token'a *doğrudan* ve aynı anda baksın, neyin
önemli olduğuna kendisi karar versin.

Bu kararı, orijinal makalenin kendi örneğinde izleyin:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Tek kelime değişir, "o" taraf değiştirir. Siz bunu anında çözdünüz;
**attention**, modelin çözme biçimidir. Bütün numara tek cümleye iner: **bir kelimenin bağlamı, diğer kelimelerin ağırlıklı bir karışımıdır ve attention'ın bütün işi ağırlıkları seçmektir.** Ve ağırlıklar yalnızca mesafeden gelemez — "o"yu çözen kelime yirmi token geride olabilir — içerikten hesaplanmak ve öğrenilmek zorundadırlar.

### Q, K, V — mekanizma, sayılarla

Ağırlıkları seçmek için her token'a, her biri vektörünün küçük öğrenilmiş
bir dönüşümü olan üç rol verilir:

- **sorgu (query)** — ne arıyorum?
- **anahtar (key)** — başkaları beni nasıl bulsun?
- **değer (value)** — seçilirsem ne teslim ederim?

Bu üçü nereden geliyor? Token'ın kendi vektöründen. Model, üç öğrenilmiş
sayı tablosu tutar; bir token'ın vektörünü her tabloyla çarpmak, onun
sorgusunu, anahtarını ve değerini üretir. Aynı kelime, üç kıyafet. (Ve YouTube'u düşünün: arama metniniz sorgu, her videonun başlığı anahtar, videoların kendisi değer.)

Peki neden doğrudan ham embedding'lerle karşılaştırma yapmıyoruz? Çünkü bir
embedding, kelimenin birçok yönünü aynı anda karıştırır — dil bilgisi, anlam,
konum. Üç tablo, modelin *tam da bu aramanın ihtiyacı olan yönü* çekmesine
izin verir: "yorgun olabilir" üzerinden eşleşen bir sorgu-anahtar çifti,
"f ile başlar" üzerinden değil. Birlikte bir **yumuşak sözlük** gibi
davranırlar: gerçek bir sözlük anahtarı ya tam eşleştirir ya da hiçbir şey
döndürmez; attention her anahtarla *kısmen* eşleşir ve her değerden oransal
bir dilim alır.

Bütün işi yalnızca iki matematik işlemi yapar:

- **İç çarpım** — iki vektörü basamak basamak çarpıp hepsini topla:
  a · b = a₁b₁ + a₂b₂ + a₃b₃ + … Örneğin
  [2, 1, 0] · [3, 1, 4] = 6 + 1 + 0 = **7**. Tek bir sayı çıkar: iki vektör
  aynı yönü gösteriyorsa büyük, göstermiyorsa küçük. Bir benzerlik ölçer.
- **Ağırlıklı toplam** — birkaç vektörü yüzdelere göre karıştır; bir tarif
  gibi: şundan %60, bundan %30.

Şimdi "Hızlı kahverengi tilki"yi, model *tilki* üzerinde çalışırken izleyin.

**1. Adım — Puanla: bana kim önemli?** *Tilki*nin sorgusu, kendisininki
dahil her kelimenin anahtarıyla iç çarpıma girer:

| çift | iç çarpım | okunuşu |
|---|---|---|
| Q(tilki) · K(Hızlı) | 2,1 | biraz ilgili |
| Q(tilki) · K(kahverengi) | 4,0 | çok ilgili |
| Q(tilki) · K(tilki) | 5,4 | kendisi — en çok |

**2. Adım — Yüzdele: puanları tarife çevir.** Bu, **softmax**'tır ve
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

**3. Adım — Karıştır: tarifi pişir.** Yeni *tilki* vektörü, *değerlerin*
ağırlıklı toplamıdır:

> tilki_yeni = 0,03 × V(Hızlı) + 0,19 × V(kahverengi) + 0,78 × V(tilki)

Sonuç artık sözlükteki *tilki* kelimesi değildir; *bu-belirli-hızlı-
kahverengi-tilki*dir ve bir sonraki katmana giden, bu zenginleşmiş
vektördür.

Elde tutmaya değer bir netlik: buradaki *öğrenilmiş* tek parça, Q, K ve
V'yi üreten üç tablodur. İç çarpımlar, softmax, ağırlıklı toplam — sabit
aritmetiktir, içlerinde öğrenme yoktur. Ve modele tilkilerin kahverengi
olduğunu hiçbir kural söylemedi: üç tablo, trilyonlarca tahmin boyunca, işe
yarar ağırlıklar kendiliğinden çıkana dek ayarlandı.

Bu mekanizmaya iki dipnot. Birincisi, ham puanlar softmax'tan önce
√(anahtar boyutu)'na bölünür — adındaki "*ölçekli* iç çarpım" budur: devasa
iç çarpımlar ağırlıkları ya-hep-ya-hiç'e doyurur ve öğrenmeyi durdururdu.
İkincisi, her token diğer her token'ı puanladığından iş, uzunluğun
*karesiyle* büyür — O(n²). Bağlamı ikiye katlayın, maliyet dörde katlansın;
milyon token'lık pencereler bir kutucuk değil, mühendislik başarısıdır.

Hepsini birleştirin; bu alt bölümün tamamı tek satırdır — 2017'den beri her
makalede basılan formül:

> **Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V**

Soldan sağa okuyun: her sorguyu her anahtarla iç çarpıma sok (QKᵀ), √dₖ ile
ölçekle, puanları softmax ile yüzdeye çevir, değerleri bu yüzdelerle
karıştır. İçindeki her sembol artık tanıdık.

### Tek yön

Sonuçları büyük bir ayrıntı: üretim sırasında her token yalnızca *geriye*
bakabilir. *Tilki*, *kahverengi*yi görür; *kahverengi*, *tilki*yi asla. Bu
maske, modeli bir *sıradaki*-token tahmincisi yapan şeydir — ve geçmiş bir
token'ın anahtarıyla değerinin, bir kez hesaplandıktan sonra asla
değişmemesinin sebebi de budur: sonradan gelen hiçbir şey onlara dokunamaz. Bir kenara yazın; 7. bölümde KV cache olacak.

Bu maske aynı zamanda alanın ayrım çizgisidir: maskeyle kurulan modeller (GPT tarzı **decoder**'lar) üretir; maskesiz kurulanlar (BERT tarzı **encoder**'lar) iki yöne birden bakar ve üretmek yerine sınıflandırır. Modern LLM'lerin neredeyse hepsi yalnız-decoder'dır.

### Birçok kafa

Katman başına tek bir ağırlıklama kaba kalırdı — bir kelimenin bir
komşusundan dil bilgisine, başka birinden göndermeye ihtiyacı var. Bu yüzden
her katman, her biri kendi Q/K/V mercekleriyle donanmış birçok attention
**kafasını** paralel çalıştırır ve her kafa izleyeceği ilişkiyi kendisi
öğrenir: biri söz dizimini izler, biri "o"yu çözer, biri sıfatı isme bağlar.
Bu rolleri kimse atamaz; kendiliğinden belirir — çünkü her biri sıradakini
tahmine yarar.

## 4. Katmanlar: bilgi nerede yaşıyor

Transformer, bu bloğun onlarca-yüzü aşkın kez üst üste konmuşudur — ve her
katman vektörleri değiştirmek yerine *düzenler*; anlam böylece birikir:
*tilki* önce *kahverengi-hızlı-tilki*, sonra *harekete geçmek üzere olan
özne* olur, katman katman. Her katmanın içinde attention bağlamı toplar
(kütüphaneci), **ileri beslemeli ağ** ise — her token'a tek tek uygulanan
küçük bir ağ — "Paris, Fransa ile eşleşir" gibi öğrenilmiş örüntüleri
depolar (ambar; **parametrelerin** çoğu burada yaşar). İlk katmanlar yazımı ve dil
bilgisini üstlenir; derin katmanlar olguları ve uzun menzilli mantığı.
GPT-2, 2019'da 1,5 milyar parametreyle manşet olmuştu; öncü modeller bugün
trilyonlara varıyor — modern bir dokunuş olan **mixture of experts (MoE)**
ise tek ambar yerine birçoğunu kurar ve bir yönlendirici her token'ı en iyi
bir-iki tanesine gönderir: devasa toplam kapasite, token başına bunun
yalnızca bir kesri hesaplama öder.

En tepede **softmax** — attention'ın kullandığı yüzde çeviricinin aynısı —
son puanları, modelin bildiği her token için bir olasılığa çevirir. "Bir
varmış bir"den sonra kütle "yokmuş"a yığılır. "En sevdiğim şehir"den sonra
yüzlerce şehre dağılır. İkisi de modelin cevapladığı tek sorunun doğru
cevabıdır: *sırada ne gelmesi muhtemel?*

## 5. Eğitim ve ölçek

**Ön eğitim**: modele trilyonlarca token gösterin, sıradakini gizleyin,
tahmin ettirin. **Kayıp fonksiyonu**, modelin doğru cevap karşısındaki
şaşkınlığını puanlar:

> kayıp = −log p(doğru token)

Model doğru token'a %90 olasılık vermişse kayıp −log 0,9 ≈ 0,1'dir —
neredeyse hiç şaşırmamış. %20 vermişse kayıp −log 0,2 ≈ 1,6'dır — fena
şaşırmış. **Gradyan inişi** de her parametreyi bu sayıyı küçültecek yönde
minicik bir adım kaydırır; bunu trilyonlarca kez tekrarlayın. (Standart
karne olan **perplexity**, ortalama kaybın üstelinden ibarettir: ortalama
1,6 ise e^1,6 ≈ 5 — beş eşit olası kelime arasında seçim yapıyormuş kadar
kararsız. Ön eğitimin merkezinde; gerçek görevler için zayıf bir vekil.)

İçine kural yazılmaz — polisiye romanın son bölümünü tahmin etmek kimin
cinayet sebebi olduğunu izlemeyi gerektirir, o yüzden izlemek öğrenilir.
Sonuç, eğitim verisinin JPEG gibi sıkıştırılmışıdır: resim kalır, pikseller
kalmaz.

Ölçek, **ölçek yasalarını** izler: kayıp, hesaplamanın kuvvet yasası olarak
düşer — kabaca kayıp ≈ a · C^(−α), log-log kâğıdında düz bir çizgi — yani
hesaplamayı ona katlamak öngörülebilir bir düşüş satın alır. Dokuz haneli
eğitim bütçelerini kumardan plana çeviren budur: GPT-4'ün nihai kaybı
10.000 kat küçük denemelerden önceden tahmin edildi. DeepMind'ın
**Chinchilla** çalışması denge kuralını ekledi: parametre ve veri birlikte
büyümeli (parametre başına ~20 token); 70 milyarlık modelleri, sırf bu
aritmetikle 280 milyarlık rakibini geçti.

Dürüst bir şerh: eğri pürüzsüzdür ama beceriler aniden gelebilir — bir
model üç basamaklı aritmetikte boy boy başarısız olup bir sonraki
sıçrayışta bunu güvenilir yapabilir: **beliren yetenek**. Ham madde de
sonlu: kaliteli açık metin tükenmek üzere; sınır bu yüzden sentetik veriye
ve hesaplamayı cevap anında harcamaya — aşağıdaki akıl yürüten modellere —
kayıyor.

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
Vurucu gerçek: GPT-3, ChatGPT'den iki yıl önce vardı. Devrim daha büyük ağ
değil, bu aşamalardı.

## 7. Üretim: plan değil, döngü

Model olasılıkları hesaplar, bir token **örnekler** (ağırlıklı çekiliş),
ekler ve tekrarlar — her yeni token anında bir sonraki tahminin
girdisidir — ta ki "bitirdim" diyen özel bir durdurma token'ına kadar.

Çekilişi üç düğme yönetir. "Gökyüzü"nden sonra liste şöyle olabilir:
*maviydi* %60, *karanlıktı* %10, …, *patatesti* %0,0001:

- **Temperature**, softmax'tan önce her puanı T'ye böler. T 1'in altındaysa
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
3. **Eşleştirme.** Q₃, önbellekteki anahtarlarla karşılaştırılır:

   | karşılaştırma | dikkat ağırlığı |
   |---|---|
   | Q₃ · K₁ ("Ben") | %30 |
   | Q₃ · K₂ ("seni") | %70 |

4. **Karışım.** Çıktı, önbellekteki değerlerden kurulur:
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
değil: model bir olasılık motorudur, veritabanı değil. Eğitim verisinin zengin olduğu yerde en olası devam genellikle doğrudur. İnce olduğu yerde ise bulunamayacak kayıt yoktur — cevap *biçiminde* bir şey üretir;
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

## Daha derine inmek için

- Vaswani vd., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — orijinal Transformer makalesi; yorgundu/genişti örneği onlarındır.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — klasikleşmiş görsel anlatım.
- Ebrahim Pichka, [What are Query, Key, and Value in the Transformer Architecture?](https://medium.com/data-science/what-are-query-key-and-value-in-the-transformer-architecture-and-why-are-they-used-acbe73f731f2) — QKV sezgisinin, yumuşak sözlük bakışı dahil, özenli bir açılımı.
- Andrej Karpathy, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — bütün makinenin gözünüzün önünde kodla inşası.
