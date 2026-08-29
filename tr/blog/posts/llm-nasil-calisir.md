Büyük bir dil modelinin size verdiği her yanıt — her deneme, her kod parçası,
az önce yaptığı hata için her özür — aynı yöntemle üretildi: bir sonraki küçük
metin parçasını tahmin ederek. Tekrar tekrar, her seferinde tek parça.

Telefonunuz bunun minik bir sürümünü zaten yapıyor. "Görüşürüz" yazın, klavye
*yarın*, *orada*, *akşam* önerir. Büyük dil modeli, aynı numaranın milyarlarca
kat büyütülmüş halidir — ve işin ilginç kısmı, bu oyunu o kadar iyi oynamanın
*neler gerektirdiği*. Bir sonraki kelimeyi üst düzeyde tahmin edebilmek için
model dil bilgisini, olguları, üslubu ve akıl yürütmenin işleyen bir taklidini
özümsemek zorunda kalıyor; çünkü insan metninde sıradaki parçayı kestirmek için
bunların hepsi gerekiyor. Bu fikri aklınızda tutun; aşağıdaki her şey onun
dipnotu.

## Metin sayıya dönüşür: token'lar

Bilgisayarlar kelime okumaz. İlk adım **tokenizer**: metni token denen
parçalara böler ve her parçaya bir kimlik numarası verir. Yaygın kelimeler tek
parça kurtulur — "için" bir token, "model" bir token. Nadir kelimeler bölünür:
"inanılmaz" belki "inan" + "ılmaz" olur, "kuantizasyon" belki "kuant" +
"izasyon". İngilizce için kaba bir kural: **100 token aşağı yukarı 75
kelimedir**; yani "128K bağlam penceresi", modelin aynı anda yaklaşık 96.000
kelimeyi — koca bir romanı — görebildiği anlamına gelir.

Bunun iki pratik sonucu var:

- **Bağlam sınırı kelimeyle ya da sayfayla değil, token'la ölçülür.**
- **Tuhaf hatalar çoğu zaman token düzeyinde yaşar.** Modeller yıllarca
  "strawberry" kelimesindeki r harflerini yanlış saymalarıyla meşhur oldu. Bu,
  meyve cahilliği değil — model harf görmez. Onun gözünde "çilek" ç-i-l-e-k
  değil, bir iki opak parçadır; ondan harf saymasını istemek, sizden bir
  tablonun *fotoğrafındaki* fırça darbelerini saymanızı istemek gibidir.

## Anlam taşıyan sayılar: embedding'ler

Sonra her token kimliği bir **embedding**'e eşlenir: çoğu zaman binlerce
boyutlu uzun bir sayı listesi — token'ın bir anlam haritasındaki koordinatları
gibi davranan bir vektör. Gerçek haritada Paris, Brüksel'e yakın, Sidney'e
uzaktır. Anlam haritasında *kral*, *kraliçe* ve *taht*ın yanında, *hesap
tablosu*ndan uzakta oturur. İlişkiler bile tutarlı yönlere dönüşür: *Paris*ten
*Fransa*ya giden ok, *Roma*dan *İtalya*ya giden okla aynı yönü gösterir — bir
"başkenti-olmak" yönü. Meşhur salon numarası da vektör aritmetiğidir: *kral −
erkek + kadın*, *kraliçe*nin yakınına düşer.

Bu haritayı kimse elle çizmedi; öğrenildi. Ve modelin ana dili budur — buradan
sonrası vektörler üzerinde aritmetiktir.

Eksik bir şey kaldı: sıra. Bir torba koordinat, "köpek adamı ısırdı" ile "adam
köpeği ısırdı" arasındaki farkı bilmez; bu yüzden her token'ın dizideki
**konumu** da temsiline işlenir. Kelime sırası, matematiğe yolculuğu sağ salim
atlatır.

## Attention: bağlam içinde okumak

Koordinatlar tek başına "yüz" kelimesinin ne demek olduğunu söyleyemez —
surattaki yüz mü, sayı olan yüz mü? Kelimenin anlamı komşularına bağlıdır.
**Transformer** mimarisinin çözdüğü sorun budur ve temel aracı **attention**
(dikkat) mekanizmasıdır.

En temiz gösterim, orijinal Transformer makalesinin kendi örneğinden gelir.
Karşılaştırın:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Sondaki tek kelimeyi değiştirin, "o" anlam değiştirsin — *yorgun* hayvanı
gösterir, *geniş* caddeyi. Siz bunu anında ve farkında olmadan çözdünüz.
Attention, modelin aynısını yapmasını sağlayan mekanizmadır.

Bir eşleştirme servisi gibi çalışır. Her token bir **sorgu (query)** yayınlar —
ne aradığının tarifi ("ben bir zamirim; daha önce geçen ve *yorgun* olabilecek
bir şeye ihtiyacım var"). Her token aynı zamanda bir **anahtar (key)** ilan
eder — ne olduğunun tarifi ("ben bir hayvanım, dört kelime önce geçtim").
Sorgu ile anahtar güçlü eşleştiğinde bilgi akar: eşleşen token **değerini
(value)** devreder ve zamirin vektörü fiilen *hayvan* anlamına gelecek şekilde
güncellenir.

Ve bu bir kez olmaz. Her katman, paralel çalışan birçok attention "kafası"
(head) barındırır ve her kafa farklı türden bir ilişkiyi izlemeyi öğrenir —
biri dil bilgisini takip eder, biri yukarıdaki gibi göndermeleri çözer, bir
başkası hangi sıfatın hangi ismi nitelediğini fark eder. Bu rolleri kimse
atamaz; kendiliğinden ortaya çıkarlar, çünkü her biri sıradaki token'ı tahmin
etmeye yarar.

## Katmanlar: bilgi nerede yaşıyor

Transformer bu mekanizmayı **katmanlar** halinde üst üste diziyor — onlarcadan
yüzü aşkına kadar. Her katmanda iki blok var: attention (diğer token'lardan
bağlamı karıştır) ve **ileri beslemeli ağ** (her token'ın vektörünü kendi
başına dönüştür). Kullanışlı bir zihinsel model: ileri beslemeli bloklar,
öğrenilmiş örüntülerin depolandığı ambardır — "Paris, Fransa ile eşleşir",
"`def`ten sonra gelen kod bir fonksiyon adıdır" — attention ise eldeki cümlenin
hangi rafa ihtiyacı olduğuna karar veren kütüphanecidir.

İlk katmanlar yazımı ve dil bilgisini, derin katmanlar olguları, ilişkileri ve
daha uzun menzilli mantığı yakalar. *Büyük* dil modelindeki "büyük", bütün bu
katmanlardaki öğrenilmiş sayıları — **parametreleri** — sayar. GPT-2, 2019'da
1,5 milyar parametreyle manşetlere çıkmıştı; bugünün öncü modelleri yüz
milyarlarla ve trilyonlarla ölçülüyor.

En tepede model, son vektörü sözlüğündeki her token için bir puana çevirir ve
puanları olasılığa sıkıştırır — **softmax** adımı. Her adımda modelin bütün
çıktısı budur: bir cümle değil, bir fikir değil — bildiği her token için bir
olasılık. "Bir varmış bir" dedikten sonra olasılığın neredeyse tamamı
"yokmuş"un üzerine yığılır. "En sevdiğim şehir" dedikten sonra yüzlerce makul
şehre dağılır. İkisi de, modelin cevapladığı tek sorunun doğru cevabıdır.

## Eğitim: bilgi nereden geliyor

Parametreler değerlerini nasıl alıyor? **Ön eğitimle (pretraining)**: modele
devasa miktarda metin gösterilir — açık internetin büyük bölümü, kitaplar,
kod; trilyonlarca token, bir insanın on bin ömürde okuyabileceğinden fazlası —
sıradaki token gizlenir ve model tahmin eder. **Kayıp fonksiyonu (loss)**
modelin doğru cevaba ne kadar şaşırdığını ölçer; **gradyan inişi** de her
parametreyi, modeli daha az şaşırtacak yönde minicik bir adım kaydırır. Bunu
trilyonlarca kez tekrarlayın.

Modelin içine kimse kural yazmaz. Dil bilgisi, coğrafya, kimya, Python — hepsi
tek bir oyunda ustalaşmanın yan etkisi olarak özümsenir: sıradaki token'ı bil.
Bir polisiye romanın son bölümünde sıradaki kelimeyi tahmin etmek için kimin
cinayet sebebi olduğunu takip etmiş olmak gerekir; bir fizik ders kitabının
sonraki satırı için biraz fizik içselleştirmiş olmak. Sonucu, eğitim verisinin
kayıplı bir sıkıştırması gibi düşünebilirsiniz — örüntüler saklanır, birebir
kopyalar çoğunlukla saklanmaz.

Hikâyenin diğer yarısı ölçek. Modeli büyütün, daha çok veri verin, daha çok
hesaplama harcayın; kayıp, pürüzsüz ve neredeyse yasa gibi bir eğriyle düşer —
devasa eğitim koşularını meşrulaştıran **ölçek yasaları (scaling laws)**
bunlardır. Yol boyunca kimsenin hedeflemediği yetenekler belirir: çeviri,
aritmetik, çalışan kod. Ortaya çıkarlar, çünkü her biri modelin sahip olduğu
tek hedefe hizmet eder.

## Otomatik tamamlamadan asistana

Ön eğitimden çıkan model ham bir otomatik tamamlayıcıdır ve öyle de davranır.
"Fransa'nın başkenti neresi?" diye sorarsanız cevap verebilir — ya da dokuz
quiz sorusu daha ekleyebilir; çünkü internette bir quiz sorusunun ardından
genellikle bir tane daha gelir. Onu asistana çeviren iki aşama daha vardır:

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

Her zaman en olası token'ı seçmez; hep birinci tercihi almak tekrarlı,
kasılmış metin üretir. Bunun yerine araya kontrollü bir rastgelelik katılır ve
bunu **temperature** ölçekler. Düşük sıcaklıkta "Gökyüzü" neredeyse her
seferinde *maviydi* diye devam eder — veri çıkarmak ya da SQL yazdırmak için
doğru ayar. Yüksek sıcaklıkta *limanın üzerinde çürük moruna çalıyordu* diye
devam edebilir — beyin fırtınası için doğru ayar. Top-p gibi ayarlar da
örneklemeden önce gerçekten olasılıksız seçenekleri budar. Aynı sorunun farklı
günlerde farklı cevaplar almasının sebebi de budur.

Bu döngü, "adım adım düşün" komutunun neden gerçekten işe yaradığını da
açıklar. Modelden 17 × 24'ü tek hamlede isteyin; cevabı tek bir
sıradaki-token tahminiyle tutturmak zorundadır. "17 × 24 = 17 × 20 + 17 × 4 =
340 + 68 = ..." diye yazmasına izin verin; her ara adım bağlama girer ve
sonraki tahminleri keskinleştirir — modelin kafasında karalama defteri yoktur,
o yüzden sayfayı defter olarak kullanır. Akıl yürüten (reasoning) modeller tam
olarak bunu endüstrileştirir: cevaba bağlanmadan önce sesli düşünmeye bolca
token harcamak üzere eğitilirler.

## Model sizi hatırlamaz

Resmi tamamlayan son parça: eğitim bittikten sonra parametreler **donar**.
Sohbetiniz modeli yeniden eğitmez. Uzun süreli hafızası olmayan parlak bir
danışmanla çalışmak gibidir — her sabah bütün dava dosyasını ona yeniden
vermeniz gerekir. Olan tam olarak budur: her mesaj gönderdiğinizde konuşmanın
*tamamı* ağdan yeniden geçirilir ve cevap oradan tahmin edilir. Hafıza gibi
hissettiren şey yalnızca bağlam penceresidir — çok uzun sohbetlerin
yavaşlamasının, sınıra dayanmasının ya da başını unutmasının sebebi de bu.

Madalyonun öbür yüzü, **in-context learning** denen gerçek bir süper güçtür.
Şunu bir isteme koyun:

> deniz → sea, ev → house, kedi → ?

Model *cat* der — görevi iki örnekten çıkarmıştır, tek bir parametre
değişmeden. Ona *acil* ya da *rutin* diye etiketlenmiş üç destek talebi
gösterin; dördüncüyü etiketler. Pratik prompt mühendisliğinin büyük kısmı tam
olarak budur: bağlamı, istenen devamı en olası devam haline getirecek şekilde
düzenlemek.

## Modeller neden uyduruyor

Parçalar artık en meşhur hata türünü açıklıyor. 2023'te New York'lu iki
avukat, *Mata v. Avianca* davasında altı havayolu içtihadına atıf yapan bir
dilekçe sundu — dava adları, dosya numaraları ve alıntılanabilir gerekçelerle
birlikte. Hiçbiri yoktu. Altısını da ChatGPT uydurmuştu; avukatlar davaların
gerçek olup olmadığını sorduklarında da onları gerçek olduklarına temin
etmişti. Mahkeme avukatlara 5.000 dolar ceza kesti ve "yapay zekâ
halüsinasyonu" ana akım sözlüğe girdi.

Yukarıdaki mekanizma bunu açıklıyor. Model, arama tablosu olan bir veritabanı
değil; metin üzerinde bir olasılık motorudur. Eğitim verisinde iyi temsil
edilen bir şey sorduğunuzda, en olası devam genellikle doğrudur. Var olmayan
bir içtihat istediğinizde ise bulunamayacak bir kayıt yoktur — makine yaptığı
tek işi yapmaya devam eder ve doğru bir cevabın *biçimini* taşıyan bir devam
üretir: makul isimler, makul atıflar, kendinden emin bir ton. **Halüsinasyon**
sisteme sonradan bulaşmış bir hata değildir; sistemin varsayılan davranışıdır —
yukarıdaki eğitim aşamalarıyla evcilleştirilmiş ama yok edilmemiştir.

Pratik çözümler çoğunlukla modeli değil girdiyi değiştirir: retrieval (gerçek
belgeleri bağlama getirip modele oradan cevap verdirmek), araç kullanımı
(arama motoru çağırtmak, kod çalıştırtmak) ve kendiniz kontrol edebileceğiniz
kaynaklar istemek — avukatların atladığı adım.

## Saklamaya değer iskelet

1. Metin **token**'lara bölünür; her biri öğrenilmiş bir vektöre
   (**embedding**) dönüşür — bir anlam haritasındaki koordinatlar — ve kelime
   sırası kaybolmasın diye **konum** bilgisi işlenir.
2. Üst üste dizilmiş **transformer** katmanları **attention** kullanır —
   sorgular anahtarlarla eşleşir, birçok kafa paralel çalışır — böylece her
   token'ın vektörü bağlamıyla beslenir ("çok yorgundu" ile "çok genişti");
   bilginin çoğunu ileri beslemeli bloklar depolar.
3. Devasa metin üzerinde sıradaki-token tahminiyle yapılan **ön eğitim**
   bilginin kaynağıdır; **ölçek yasaları** daha çok model, veri ve hesaplamanın
   öngörülebilir biçimde işe yaradığını söyler; talimat eğitimi ve insan geri
   bildirimi sonucu asistana biçimlendirir.
4. Üretim bir döngüdür: softmax her token'a bir olasılık verir, **örnekleme**
   birini seçer, seçim bir sonraki adımı besler. Rastgeleliği **temperature**
   kontrol eder; "adım adım düşünmek", modelin sayfayı karalama defteri olarak
   kullanmasıdır.
5. Eğitimden sonra parametreler **donmuştur** — hafıza sanılan şey bağlam
   penceresidir; **in-context learning**, örüntünün yalnızca istemden
   kapılmasıdır (*deniz → sea, kedi → cat*).
6. Model *doğru* için değil *makul* için optimize edilmiştir — akıcılığının
   sebebi de budur, altı sahte içtihadın bir federal mahkeme dilekçesine nasıl
   girdiğinin açıklaması da.

Adımların hiçbiri tek başına sihir değil; sürpriz, bu kadar basit bir şey
yeterli ölçekte yapıldığında ortaya çıkanlarda. Ve bir dahaki sefere biri size
bu modellerin nasıl çalıştığını sorduğunda — masanın karşısındaki bir
mülakatçı, ders sonrası bir öğrenci ya da kendi içinizdeki meraklı ses — tam
olarak modelin başladığı yerden başlayabilirsiniz: sıradaki token'dan.
