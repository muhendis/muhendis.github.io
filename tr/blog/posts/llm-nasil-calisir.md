Büyük bir dil modelinin size verdiği her yanıt — her deneme, her kod parçası,
az önce yaptığı hata için her özür — aynı yöntemle üretildi: bir sonraki küçük
metin parçasını tahmin ederek. Tekrar tekrar, her seferinde tek parça.

Bu, sınav geçebilen ve yazılım yazabilen bir şeyi açıklamak için fazla basit
görünüyor. İşin ilginç kısmı, bir sonraki kelimeyi *iyi* tahmin etmenin neler
gerektirdiği. Bunu üst düzeyde yapabilmek için model dil bilgisini, olguları,
üslubu ve akıl yürütmenin işleyen bir taklidini özümsemek zorunda kalıyor —
çünkü insan metninde sıradaki parçayı kestirmek için bunların hepsi gerekiyor.
Bu fikri aklınızda tutun; aşağıdaki her şey onun dipnotu.

## Metin sayıya dönüşür: token'lar

Bilgisayarlar kelime okumaz. İlk adım **tokenizer**: metni token denen
parçalara böler ve her parçaya bir kimlik numarası verir. Bir token çoğu zaman
yaygın bir kelimenin tamamıdır ("model", "için"), bazen daha nadir bir
kelimenin parçasıdır ("kuantizasyon" belki "kuant" + "izasyon" olur), bazen de
noktalama işaretidir.

Bunun iki pratik sonucu var:

- **Bağlam sınırı kelimeyle değil token'la ölçülür.** Bir modelin "128K bağlam
  penceresi" varsa, aynı anda dikkate alabildiği token sayısı budur.
- **Tuhaf hatalar çoğu zaman token düzeyinde yaşar.** Bir kelimedeki harfleri
  saymak, harfleri hiç görmeyen bir model için gerçekten zordur — model token
  kimlikleri görür. "Çilek" modelin gözünde ç-i-l-e-k değildir; opak bir veya
  iki parçadır.

## Anlam taşıyan sayılar: embedding'ler

Sonra her token kimliği bir **embedding**'e eşlenir: token'ın anlamını temsil
eden, çoğu zaman binlerce boyutlu uzun bir sayı listesi — bir vektör. Bu
vektörler elle tasarlanmaz, öğrenilir; ve öyle örgütlenirler ki benzer anlamlar
birbirine yakın durur: *kral* ile *kraliçe*, *Paris* ile *Roma* gibi. İlişkiler
bile bu uzayda birer yön haline gelir — meşhur salon numarası, *kral − erkek +
kadın* vektör aritmetiğinin *kraliçe*nin yakınına düşmesidir.

Bu, modelin ana dilidir. Buradan sonrası vektörler üzerinde aritmetiktir.

Eksik bir şey kaldı: sıra. Bir torba vektör, "köpek adamı ısırdı" ile "adam
köpeği ısırdı" arasındaki farkı bilmez; bu yüzden her token'ın dizideki
**konumu** da temsiline işlenir. Kelime sırası, matematiğe yolculuğu sağ
salim atlatır.

## Attention: bağlam içinde okumak

Tek başına bir embedding "yüz" kelimesinin ne demek olduğunu söyleyemez —
surattaki yüz mü, sayı olan yüz mü, yüzme fiili mi? Kelimenin anlamı
komşularına bağlıdır. **Transformer** mimarisinin çözdüğü sorun budur ve temel
aracı **attention** (dikkat) mekanizmasıdır.

Mekanizmayı en iyi bir eşleştirme servisi gibi düşünebilirsiniz. Her token bir
**sorgu (query)** yayınlar — ne aradığının tarifi ("ben bir zamirim; daha önce
geçen bir kişiye ihtiyacım var"). Her token aynı zamanda bir **anahtar (key)**
ilan eder — ne olduğunun tarifi ("ben bir kişiyim, iki kelime önce geçtim").
Sorgu ile anahtar güçlü eşleştiğinde bilgi akar: eşleşen token **değerini
(value)** devreder ve zamirin vektörü onunla güncellenir. "Mühendis raporu
bitirdi çünkü teslim tarihi gelmişti" cümlesi işlenirken *teslim tarihi*ni
*rapor*a bağlayan şey budur.

Ve bu bir kez olmaz. Her katman, paralel çalışan birçok attention "kafası"
(head) barındırır ve her kafa farklı türden bir ilişkiyi izlemeyi öğrenir —
biri dil bilgisini takip eder, biri göndermeleri çözer, bir başkası hangi
sıfatın hangi ismi nitelediğini fark eder. Bu rolleri kimse programlamaz;
kendiliğinden ortaya çıkarlar, çünkü her biri sıradaki token'ı tahmin etmeye
yarar.

## Katmanlar: bilgi nerede yaşıyor

Transformer bu mekanizmayı **katmanlar** halinde üst üste diziyor — onlarcadan
yüzü aşkına kadar. Her katmanda iki blok var: attention (diğer token'lardan
bağlamı karıştır) ve **ileri beslemeli ağ** (her token'ın vektörünü kendi
başına dönüştür). Parametrelerin çoğunu ileri beslemeli bloklar taşır ve
kullanışlı bir zihinsel model şudur: modelin depoladığı *bilginin* önemli kısmı
burada yaşar — "Paris, Fransa ile eşleşir" gibi örüntüler — attention ise
eldeki cümlenin *hangi* bilgiyi çağırdığına karar verir.

İlk katmanlar yazımı ve dil bilgisini, derin katmanlar olguları, ilişkileri ve
daha uzun menzilli mantığı yakalar. *Büyük* dil modelindeki "büyük", bütün bu
katmanlardaki öğrenilmiş sayıları — **parametreleri** — sayar: milyarlardan
trilyonlara.

En tepede model, son vektörü sözlüğündeki her token için bir puana çevirir ve
puanları olasılığa sıkıştırır (**softmax** adımı). Her adımda modelin bütün
çıktısı budur: bir cümle değil, bir fikir değil — bildiği her token için bir
olasılık.

## Eğitim: bilgi nereden geliyor

Parametreler değerlerini nasıl alıyor? **Ön eğitimle (pretraining)**: modele
devasa miktarda metin gösterilir — açık internetin büyük bölümü, kitaplar,
kod — sıradaki token gizlenir ve model tahmin eder. **Kayıp fonksiyonu (loss)**
modelin doğru cevaba ne kadar şaşırdığını ölçer; **gradyan inişi** de her
parametreyi, modeli daha az şaşırtacak yönde minicik bir adım kaydırır. Bunu
trilyonlarca kez tekrarlayın.

Modelin içine kimse kural yazmaz. Dil bilgisi, coğrafya, kimya, Python — hepsi
tek bir oyunda ustalaşmanın yan etkisi olarak özümsenir: sıradaki token'ı bil.
Sonucu, eğitim verisinin kayıplı bir sıkıştırması gibi düşünebilirsiniz —
örüntüler saklanır, birebir kopyalar çoğunlukla saklanmaz.

Hikâyenin diğer yarısı ölçek. Modeli büyütün, daha çok veri verin, daha çok
hesaplama harcayın; kayıp, pürüzsüz ve neredeyse yasa gibi bir eğriyle düşer —
devasa eğitim koşularını meşrulaştıran **ölçek yasaları (scaling laws)**
bunlardır. Yol boyunca kimsenin hedeflemediği yetenekler belirir: çeviri,
aritmetik, çalışan kod. Ortaya çıkarlar, çünkü her biri modelin sahip olduğu
tek hedefe hizmet eder.

## Otomatik tamamlamadan asistana

Ön eğitimden çıkan model ham bir otomatik tamamlayıcıdır. "Fransa'nın başkenti
neresi?" diye sorarsanız cevap verebilir — ya da dokuz quiz sorusu daha
ekleyebilir; çünkü o da metnin makul bir devamıdır. Onu asistana çeviren iki
aşama daha vardır:

1. **Talimat eğitimi (instruction tuning)** — soru ve iyi cevap çiftlerinden
   oluşan örneklerle ek eğitim; yardımcı olmanın *biçimini* öğretir.
2. **İnsan tercihlerinden öğrenme** (RLHF ve akrabaları) — insanlar aday
   cevapları karşılaştırır, model insanların tercih ettiği yöne doğru
   ayarlanır: yardımcı, dürüst, zararsız.

Aynı mimari, aynı sıradaki-token makinesi — farklı davranış.

## Üretim: plan değil, döngü

Bir istem gönderdiğinizde model önce cevabı planlayıp sonra yazmaz. Olası her
sonraki token'ın olasılığını hesaplar, birini **örnekler**, metne ekler ve
tekrar eder — her yeni token, bir sonraki tahminin girdisine anında dahil
olur — ta ki bir durdurma token'ı üretene kadar.

Her zaman en olası token'ı seçmez. Hep birinci tercihi almak tekrarlı, kasılmış
metin üretir; bu yüzden araya kontrollü bir rastgelelik katılır.
**Temperature** bunu ölçekler — düşükte odaklı ve kararlı, yüksekte çeşitli ve
yaratıcı — top-p gibi ayarlar da örneklemeden önce gerçekten olasılıksız
seçenekleri budar. Aynı sorunun farklı günlerde farklı cevaplar almasının
sebebi budur.

Bu döngü, "adım adım düşün" komutunun neden işe yaradığını da açıklar. Modelin
kafasının içinde bir karalama defteri yoktur — tek çalışma belleği metnin
kendisidir. Ara adımları yazıya döktüğünde her adım bağlama girer ve sonraki
tahminleri iyileştirir. Akıl yürüten (reasoning) modeller tam olarak bunu
endüstrileştirir: cevaba bağlanmadan önce sesli düşünmeye bolca token harcamak
üzere eğitilirler.

## Model sizi hatırlamaz

Resmi tamamlayan son parça: eğitim bittikten sonra parametreler **donar**.
Sohbetiniz modeli yeniden eğitmez. Her mesaj gönderdiğinizde konuşmanın
*tamamı* ağdan yeniden geçirilir ve cevap oradan tahmin edilir. Hafıza gibi
hissettiren şey yalnızca bağlam penceresidir — çok uzun sohbetlerin
yavaşlamasının, sınıra dayanmasının ya da başını unutmasının sebebi de bu.

Madalyonun öbür yüzü, **in-context learning** denen gerçek bir süper güçtür:
istemin içinde bir görevin birkaç örneğini gösterin, model örüntüyü kapar ve
sürdürür — tek bir parametre değişmeden. Pratik prompt mühendisliğinin büyük
kısmı tam olarak budur: bağlamı, istenen devamı en olası devam haline
getirecek şekilde düzenlemek.

## Modeller neden uyduruyor

Parçalar artık en meşhur hata türünü açıklıyor. Model, arama tablosu olan bir
veritabanı değil; metin üzerinde bir olasılık makinesidir. Eğitim verisinde iyi
temsil edilen bir şey sorduğunuzda, en olası devam genellikle doğrudur. Nadir
bir şey sorduğunuzda ise bulunamayacak bir kayıt yoktur — makine yaptığı tek
işi yapmaya devam eder ve doğru bir cevabın *biçimini* taşıyan bir devam
üretir. Makul görünen bir kaynak. Kendinden emin bir tarih. Bu
**halüsinasyondur** ve sisteme sonradan bulaşmış bir hata değildir; sistemin
varsayılan davranışıdır — yukarıdaki eğitim aşamalarıyla evcilleştirilmiş ama
yok edilmemiştir.

Pratik çözümler çoğunlukla modeli değil girdiyi değiştirerek çalışır: retrieval
(gerçek belgeleri bağlama getirip modele oradan cevap verdirmek), araç
kullanımı (arama motoru çağırtmak, kod çalıştırtmak) ve kontrol edebileceğiniz
kaynaklar istemek.

## Saklamaya değer iskelet

1. Metin **token**'lara bölünür; her biri öğrenilmiş bir vektöre
   (**embedding**) dönüşür ve kelime sırası kaybolmasın diye **konum** bilgisi
   işlenir.
2. Üst üste dizilmiş **transformer** katmanları **attention** kullanır —
   sorgular anahtarlarla eşleşir, birçok kafa paralel çalışır — böylece her
   token'ın vektörü bağlamıyla beslenir; bilginin çoğunu ileri beslemeli
   bloklar depolar.
3. Devasa metin üzerinde sıradaki-token tahminiyle yapılan **ön eğitim**
   bilginin kaynağıdır; **ölçek yasaları** daha çok model, veri ve hesaplamanın
   öngörülebilir biçimde işe yaradığını söyler; talimat eğitimi ve insan geri
   bildirimi sonucu asistana biçimlendirir.
4. Üretim bir döngüdür: softmax her token'a bir olasılık verir, **örnekleme**
   birini seçer, seçim bir sonraki adımı besler. Rastgeleliği **temperature**
   kontrol eder; "adım adım düşünmek", modelin kendi çıktısını çalışma belleği
   olarak kullanmasıdır.
5. Eğitimden sonra parametreler **donmuştur** — hafıza sanılan şey bağlam
   penceresidir; **in-context learning**, örüntünün yalnızca istemden
   kapılmasıdır.
6. Model *doğru* için değil *makul* için optimize edilmiştir — akıcılığının da
   halüsinasyonunun da sebebi tam olarak budur.

Adımların hiçbiri tek başına sihir değil; sürpriz, bu kadar basit bir şey
yeterli ölçekte yapıldığında ortaya çıkanlarda. Ve bir dahaki sefere biri size
bu modellerin nasıl çalıştığını sorduğunda — masanın karşısındaki bir mülakatçı,
ders sonrası bir öğrenci ya da kendi içinizdeki meraklı ses — tam olarak
modelin başladığı yerden başlayabilirsiniz: sıradaki token'dan.
