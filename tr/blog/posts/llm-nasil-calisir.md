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

Sonra her token kimliği bir **embedding**'e eşlenir — bir vektöre: çoğu zaman binlerce sayıdan oluşan, token'ın bir anlam haritasındaki koordinatları gibi davranan uzun bir liste. Gerçek haritada Paris, Brüksel'e yakın, Sidney'e
uzaktır. Anlam haritasında *kral*, *kraliçe* ve *taht*ın yanında, *hesap
tablosu*ndan uzakta oturur. İlişkiler bile tutarlı yönlere dönüşür: *Paris*ten
*Fransa*ya giden ok, *Roma*dan *İtalya*ya giden okla aynı yönü gösterir — bir
"başkenti-olmak" yönü. Meşhur gösteri numarası da vektör aritmetiğidir: *kral −
erkek + kadın*, *kraliçe*nin yakınına düşer.

Bu haritayı kimse elle çizmedi; öğrenildi. Ve modelin ana dili budur — buradan
sonrası vektörler üzerinde aritmetiktir.

Eksik bir şey kaldı: sıra. Bir torba koordinat, "köpek adamı ısırdı" ile "adam
köpeği ısırdı" arasındaki farkı bilmez; bu yüzden her token'ın dizideki **konumu** da vektörüne işlenir. Kelime sırası, matematiğe yolculuğu sağ salim
atlatır.

## Attention: bağlam içinde okumak

Koordinatlar tek başına "yüz" kelimesinin ne demek olduğunu söyleyemez —
surattaki yüz mü, sayı olan yüz mü? Kelimenin anlamı komşularına bağlıdır.
Bu sorunu **Transformer** çözer — modern bütün dil modellerinin arkasındaki sinir ağı tasarımı, GPT'deki T harfi. Temel aracı da **attention** (dikkat) mekanizmasıdır.

En temiz gösterim, orijinal Transformer makalesinin kendi örneğinden gelir.
Karşılaştırın:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Sondaki tek kelimeyi değiştirin, "o" anlam değiştirsin — *yorgun* hayvanı
gösterir, *geniş* caddeyi. Siz bunu anında ve farkında olmadan çözdünüz.
Attention, modelin aynısını yapmasını sağlayan mekanizmadır.

Peki nasıl? Modelin içinde her token, her biri vektörünün farklı küçük bir
dönüşümü olan üç şapka takar:

- bir **sorgu (query)**: "ben ne arıyorum?"
- bir **anahtar (key)**: "başkaları beni nasıl bulsun?"
- bir **değer (value)**: "seçilirsem ne teslim ederim?"

Günlük hayattaki en yakın makine bir arama motorudur. YouTube'a yazdığınız
metin sorgudur. Her videonun başlığı ve açıklaması birer anahtardır. Videoların
kendisi ise değerdir. Motor, sorgunuzu bütün anahtarlarla karşılaştırır,
eşleşmeleri puanlar ve en iyi eşleşenlerin arkasındaki içeriği önünüze koyar.

Attention bu aramayı her token için aynı anda çalıştırır. "O çok *yorgundu*"
cümlesinde *o* token'ı kabaca şöyle bir sorgu yayınlar: "daha önce geçen ve
yorgun olabilecek bir şey arıyorum." *Hayvan*ın anahtarı bu sorguyla güçlü
eşleşir; *cadde*nin anahtarı zayıf. Eşleşme puanları, toplamı yüzde yüz olan
ağırlıklara çevrilir ve *o*, vektörünü değerlerin bu ağırlıklarla karışımıyla
günceller — diyelim %85'i *hayvan*dan, %10'u *cadde*den, birazı da geri
kalandan. Hiçbir şey olduğu gibi kopyalanmaz: her güncelleme, ilgiye göre
ağırlıklanmış bir karışımdır. *Yorgun*u *geniş*le değiştirin; aynı mekanizma
ağırlıkları *cadde*ye çevirir.

Ve bu bir kez olmaz. Her katman, paralel çalışan birçok attention "kafası"
(head) barındırır ve her kafa farklı türden bir ilişkiyi izlemeyi öğrenir —
biri dil bilgisini takip eder, biri yukarıdaki gibi göndermeleri çözer, bir
başkası hangi sıfatın hangi ismi nitelediğini fark eder. Bu rolleri kimse
atamaz; kendiliğinden ortaya çıkarlar, çünkü her biri sıradaki token'ı tahmin
etmeye yarar.

## Katmanlar: bilgi nerede yaşıyor

Transformer bu mekanizmayı **katmanlar** halinde üst üste diziyor — onlarcadan
yüzü aşkına kadar. Her katmanda iki blok var: attention (diğer token'lardan
bağlamı karıştır) ve **ileri beslemeli ağ** — her token'ın vektörüne tek tek uygulanan küçük bir sinir ağı. Kullanışlı bir zihinsel model: ileri beslemeli bloklar,
öğrenilmiş örüntülerin depolandığı ambardır — "Paris, Fransa ile eşleşir",
"`def`ten sonra gelen kod bir fonksiyon adıdır" — attention ise eldeki cümlenin
hangi rafa ihtiyacı olduğuna karar veren kütüphanecidir.

İlk katmanlar yazımı ve dil bilgisini, derin katmanlar olguları, ilişkileri ve
daha uzun menzilli mantığı yakalar. *Büyük* dil modelindeki "büyük", bütün bu
katmanlardaki öğrenilmiş sayıları — **parametreleri** — sayar. GPT-2, 2019'da
1,5 milyar parametreyle manşetlere çıkmıştı; bugünün öncü modelleri yüz
milyarlarla ve trilyonlarla ölçülüyor.

En tepede model, son vektörü sözlüğündeki her token için bir puana çevirir ve
puanları toplamı yüzde yüz eden olasılıklara çevirir — **softmax** adımı; attention'ın içeride kullandığı puanı-yüzdeye-çevirme hamlesinin aynısı. Her adımda modelin bütün
çıktısı budur: bir cümle değil, bir fikir değil — bildiği her token için bir
olasılık. "Bir varmış bir" dedikten sonra olasılığın neredeyse tamamı
"yokmuş"un üzerine yığılır. "En sevdiğim şehir" dedikten sonra yüzlerce makul
şehre dağılır. İkisi de modelin cevapladığı tek sorunun doğru cevabıdır: sırada ne gelmesi muhtemel?

## Eğitim: bilgi nereden geliyor

Parametreler değerlerini nasıl alıyor? **Ön eğitimle (pretraining)**: modele
devasa miktarda metin gösterilir — açık internetin büyük bölümü, kitaplar,
kod; trilyonlarca token, bir insanın on bin ömürde okuyabileceğinden fazlası —
sıradaki token gizlenir ve model tahmin eder. **Kayıp fonksiyonu (loss)**
modelin doğru cevaba ne kadar şaşırdığını ölçer; **gradyan inişi** denen algoritma da her parametreyi, modeli daha az şaşırtacak yönde minicik bir adım kaydırır. Bunu
trilyonlarca kez tekrarlayın.

Modelin içine kimse kural yazmaz. Dil bilgisi, coğrafya, kimya, Python — hepsi
tek bir oyunda ustalaşmanın yan etkisi olarak özümsenir: sıradaki token'ı bil.
Bir polisiye romanın son bölümünde sıradaki kelimeyi tahmin etmek için kimin
cinayet sebebi olduğunu takip etmiş olmak gerekir; bir fizik ders kitabının
sonraki satırı için biraz fizik içselleştirmiş olmak. Sonucu, eğitim verisinin sıkıştırılmış bir kopyası gibi düşünebilirsiniz — bir JPEG'in fotoğrafı sıkıştırdığı gibi: resmin bütünü kalır, piksellerin birebir aynısı kalmaz.

Hikâyenin diğer yarısı ölçek. Modeli büyütün, daha çok veri verin, daha çok
hesaplama harcayın; kayıp, pürüzsüz ve neredeyse yasa gibi bir eğriyle düşer —
devasa eğitim yatırımlarını meşrulaştıran **ölçek yasaları (scaling laws)**
bunlardır. Yol boyunca kimsenin hedeflemediği yetenekler belirir: çeviri,
aritmetik, çalışan kod. Ortaya çıkarlar, çünkü her biri modelin sahip olduğu
tek hedefe hizmet eder.

## Otomatik tamamlamadan asistana

Ön eğitimin ürettiği şeye **taban model (base model)** denir ve ne olduğu
konusunda net olmakta fayda var: metni sürdüren bir makine — başka hiçbir şey
değil. Bir görev tanımı yoktur; kendisine yöneltilen bir sorunun kendisi
tarafından cevaplanması gerektiği fikri bile yoktur. Sadece interneti
okumuştur ve internette metin, metni izler.

Taban modele "Fransa'nın başkenti neresi?" diye sorun; "Paris." alabilirsiniz.
Aynı ihtimalle "Almanya'nın başkenti neresi? İspanya'nın başkenti neresi?" de
alabilirsiniz — internette quiz soruları sürü halinde gezer — ya da "diye
sordu öğretmen, kimse parmak kaldırmadı" diye sahneyi kurgu gibi sürdürmesi de
mümkündür. Üçü de sadık birer devamdır. Taban model çağında cevap koparmak,
"Soru: ... Cevap:" kalıbı yazmak gibi numaralar gerektirirdi — cevabı en
olası devam haline getirmek için. Prompt mühendisliği orada doğdu.

Bu ham malzemeyi asistana çevirmek iki aşama daha ister. İkisi de mimariyi
değiştirmez; ikisi de özenle seçilmiş metin üzerinde aynı sıradaki-token
eğitiminin devamıdır.

**Birinci aşama: talimat eğitimi (instruction tuning).** İnsanlar — giderek
artan ölçüde modellerin de yardımıyla — on binlerce örnek diyalog yazar: bir
talimat ve ideal cevabı.

> **Kullanıcı:** Bu e-postayı iki cümlede özetle.
> **Asistan:** (gerçekten iyi, iki cümlelik bir özet)

Bunlardan yeterince eğitin; "ben bir asistanım, soru cevaplanmak içindir,
yardımcı olmak böyle görünür" en olası devam haline gelir. Yardımcı olmanın
biçimi de her şey gibi öğrenilir — örneklerden.

**İkinci aşama: insan tercihlerinden öğrenme** — **RLHF** (insan geri
bildirimiyle pekiştirmeli öğrenme). Modele aynı isteme birden çok cevap
ürettirin. Çiftleri insan değerlendiricilere gösterin: *hangisi daha iyi?* Bu
yargıları tahmin etmeyi öğrenen ikinci bir model — **ödül modeli** — eğitin;
sonra LLM'i, ödül modelinin yüksek puan verdiği cevaplara doğru ayarlayın. Bu
dolambaç niye? Çünkü insanlar iki cevabı *karşılaştırmakta*, kusursuz cevap
yazmaktan çok daha iyidir; ve karşılaştırmalar, örneklerle anlatması zor
şeyleri yakalar: ton, emin olmadığında dürüstlük, zararlı istekleri geri
çevirmek.

İki yarının maliyeti orantısızdır: ön eğitim binlerce GPU üzerinde aylar
sürer; asistan aşamaları bunun küçük bir kesridir. Aradaki fark da bizzat
hissettiğiniz farktır. Taban model GPT-3, ChatGPT'den iki yıldan fazla önce
vardı. Bir araştırma merakını tarihin en hızlı büyüyen ürününe çeviren şey
daha büyük bir ağ değildi — aynı sıradaki-token makinesine eklenen bu iki
aşamaydı.

## Üretim: plan değil, döngü

Bir istem (prompt) gönderdiğinizde model önce cevabı planlayıp sonra yazmaz. Olası her
sonraki token'ın olasılığını hesaplar, birini **örnekler** — yani olasılığıyla orantılı biçimde rastgele çeker — metne ekler ve
tekrar eder — her yeni token, bir sonraki tahminin girdisine anında dahil
olur — ta ki "bitirdim" anlamına gelen özel bir durdurma token'ı üretene kadar.

Her zaman en olası token'ı seçmez — hep birinci tercihi almak tekrarlı,
kasılmış metin üretir — bu yüzden seçim, adını bilmeye değer üç düğmenin
yönettiği kontrollü bir çekiliştir. Somutlaştıralım: "Gökyüzü" girdisinden
sonra modelin listesi şöyle olabilir: *maviydi* %60, *açıktı* %20,
*karanlıktı* %10, *griydi* %5 ve binlerce token'lık minicik olasılıklı bir
kuyruk — çok aşağılarda bir yerde, %0,0001 ile *patatesti* de dahil.

- **Temperature**, çekilişten önce listeyi yeniden biçimlendirir. Düşük
  sıcaklık lideri abartır: *maviydi* neredeyse her seferinde kazanır — veri
  çıkarırken ya da SQL yazdırırken istediğiniz budur. Yüksek sıcaklık listeyi
  düzleştirir: *karanlıktı* ve *griydi* gerçek şans kazanır, arada bir de
  *limanın üzerinde çürük moruna çalıyordu* çıkar — beyin fırtınasında
  istediğiniz budur.
- **Top-k**, listeyi çekilişten önce sabit uzunlukta keser. k = 50 ise
  çekilişte yalnızca en olası 50 token kalır; kuyruk — *patatesti* dahil —
  düpedüz silinir.
- **Top-p**, listeyi sayıyla değil olasılık toplamıyla keser: yüzdeleri
  toplamı p'ye — diyelim %90'a — ulaşan en küçük token kümesini tut, gerisini
  at. İncelik şu ki bu küme kendini duruma göre ayarlar. Model eminse ("Bir
  varmış bir") %90'lık küme iki token olabilir; gerçekten kararsızsa ("En
  sevdiğim şehir") seksen token. Top-p'nin daha yaygın tercih olmasının sebebi
  bu uyarlanabilirliktir.

Önce kes, sonra kalanlar arasından çek. Aynı sorunun farklı günlerde farklı
cevaplar almasının sebebi budur — gökyüzünün asla bir patatesle
tamamlanmamasının sebebi de.

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

Pratik çözümler çoğunlukla modeli değil girdiyi değiştirir: retrieval (RAG diye de bilinir: gerçek belgeleri bağlama getirip modele oradan cevap verdirmek), araç kullanımı
(arama motoru çağırtmak, kod çalıştırtmak) ve kendiniz kontrol edebileceğiniz
kaynaklar istemek — avukatların atladığı adım.

## Saklamaya değer iskelet

1. Metin **token**'lara bölünür; her biri öğrenilmiş bir vektöre
   (**embedding**) dönüşür — bir anlam haritasındaki koordinatlar — ve kelime
   sırası kaybolmasın diye **konum** bilgisi işlenir.
2. Üst üste dizilmiş **transformer** katmanları **attention** kullanır —
   sorgular anahtarlarla eşleşir, değerler ilgiye göre karışır, birçok kafa paralel çalışır — böylece her
   token'ın vektörü bağlamıyla beslenir ("çok yorgundu" ile "çok genişti");
   bilginin çoğunu ileri beslemeli bloklar depolar.
3. Devasa metin üzerinde sıradaki-token tahminiyle yapılan **ön eğitim**
   bilginin kaynağıdır; **ölçek yasaları** daha çok model, veri ve hesaplamanın
   öngörülebilir biçimde işe yaradığını söyler. Ön eğitim tek başına ham bir
   **taban model** verir — çıplak otomatik tamamlama; talimat eğitimi ve RLHF
   onu asistana biçimlendirir.
4. Üretim bir döngüdür: softmax her token'a bir olasılık verir, **örnekleme**
   birini seçer, seçim bir sonraki adımı besler. Rastgeleliği **temperature**, **top-k** ve **top-p** kontrol eder; "adım adım düşünmek", modelin sayfayı karalama defteri olarak
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
