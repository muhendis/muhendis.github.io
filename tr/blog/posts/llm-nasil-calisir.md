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
birbirine yakın durur: *kral* ile *kraliçe*, *Paris* ile *Roma* gibi.

Bu, modelin ana dilidir. Buradan sonrası vektörler üzerinde aritmetiktir.

## Attention: bağlam içinde okumak

Tek başına bir embedding "yüz" kelimesinin ne demek olduğunu söyleyemez —
surattaki yüz mü, sayı olan yüz mü, yüzme fiili mi? Kelimenin anlamı
komşularına bağlıdır. **Transformer** mimarisinin çözdüğü sorun budur ve temel
aracı **attention** (dikkat) mekanizmasıdır.

Attention, girdideki her token'ın diğer bütün token'lara bakıp hangilerinin
kendi yorumu için önemli olduğuna karar vermesini sağlar. "Mühendis raporu
bitirdi çünkü teslim tarihi gelmişti" cümlesi işlenirken *teslim tarihi*ni
*rapor*a bağlayan şey attention'dır. Bu bağlantıları kimse programlamadı;
model, göndermeleri çözmenin bir sonraki token'ı tahmin etmeye yaradığını
öğrendi.

Transformer bunu **katmanlar** halinde üst üste diziyor — onlarcadan yüzü
aşkına kadar. Her katman vektörleri iki blokla inceltir: attention (diğer
token'lardan bağlamı karıştır) ve küçük bir ileri beslemeli ağ (her token'ın
vektörünü kendi başına dönüştür; modelin depoladığı bilginin önemli kısmı
burada yaşar). İlk katmanlar dil bilgisini ve yerel yapıyı, derin katmanlar
olguları, ilişkileri ve daha uzun menzilli mantığı yakalar. *Büyük* dil
modelindeki "büyük", bütün bu katmanlardaki öğrenilmiş sayıları —
**parametreleri** — sayar: milyarlardan trilyonlara.

En tepede model, son vektörü sözlüğündeki her token için bir puana çevirir ve
puanları olasılığa dönüştürür: "şimdiye kadarki her şey veriliyken, olası her
sonraki token'ın olasılığı şu."

## Eğitim: bilgi nereden geliyor

Parametreler değerlerini nasıl alıyor? **Ön eğitimle (pretraining)**: modele
devasa miktarda metin gösterilir — açık internetin büyük bölümü, kitaplar,
kod — sıradaki token gizlenir ve model tahmin eder. Yanlış mı tahmin etti? Her
parametre, doğru cevabı daha olası kılacak yönde minicik bir adım kaydırılır
(bu, gradyan inişidir). Bunu trilyonlarca kez tekrarlayın.

Modelin içine kimse kural yazmaz. Dil bilgisi, coğrafya, kimya, Python —
hepsi tek bir oyunda ustalaşmanın yan etkisi olarak özümsenir: sıradaki
token'ı bil.

Yalnız, ön eğitimden çıkan model ham bir otomatik tamamlayıcıdır. "Fransa'nın
başkenti neresi?" diye sorarsanız cevap verebilir — ya da dokuz quiz sorusu
daha ekleyebilir; çünkü o da metnin makul bir devamıdır. Onu asistana çeviren
iki aşama daha vardır:

1. **Talimat eğitimi (instruction tuning)** — soru ve iyi cevap çiftlerinden
   oluşan örneklerle ek eğitim; yardımcı olmanın *biçimini* öğretir.
2. **İnsan tercihlerinden öğrenme** (RLHF ve akrabaları) — insanlar aday
   cevapları karşılaştırır, model insanların tercih ettiği yöne doğru ayarlanır:
   yardımcı, dürüst, zararsız.

Aynı mimari, aynı sıradaki-token makinesi — farklı davranış.

## Üretim: her seferinde tek token

Bir istem gönderdiğinizde model önce cevabı planlayıp sonra yazmaz. Olası her
sonraki token'ın olasılığını hesaplar, birini **örnekler**, metne ekler ve
tekrar eder — her yeni token, bir sonraki tahminin girdisine anında dahil
olur — ta ki bir durdurma token'ı üretene kadar.

Her zaman en olası token'ı seçmez. Hep birinci tercihi almak tekrarlı, kasılmış
metin üretir; bu yüzden araya kontrollü bir rastgelelik katılır. **Temperature**
ayarı bunu ölçekler: düşük sıcaklık çıktıyı odaklı ve kararlı yapar, yüksek
sıcaklık çeşitli ve yaratıcı. Aynı sorunun farklı günlerde farklı cevaplar
almasının sebebi de budur.

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

## Bütün hikâye beş satırda

Bir mülakata hazırlanıyorsanız, saklamaya değer iskelet şu:

1. Metin **token**'lara bölünür; her biri öğrenilmiş bir vektöre
   (**embedding**) dönüşür.
2. Üst üste dizilmiş **transformer** katmanları **attention** ile her token'ın
   vektörünü çevresindeki bağlamla besler.
3. Devasa metin üzerinde sıradaki-token tahminiyle yapılan **ön eğitim** bütün
   bilginin kaynağıdır; talimat eğitimi ve insan geri bildirimi onu asistana
   biçimlendirir.
4. Üretim bir döngüdür: sonraki token için olasılıkları hesapla, birini
   **örnekle**, ekle, tekrarla. Rastgeleliği **temperature** kontrol eder.
5. Model *doğru* için değil *makul* için optimize edilmiştir — akıcılığının da
   halüsinasyonunun da sebebi tam olarak budur.

Adımların hiçbiri tek başına sihir değil. Son birkaç yılın sürprizi, bu kadar
basit bir şeyi yeterli ölçekte yaptığınızda ortaya çıkanlar.
