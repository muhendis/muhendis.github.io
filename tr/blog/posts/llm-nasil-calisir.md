Bir dil modelinin size verdiği her yanıt aynı yöntemle üretildi: bir sonraki
küçük metin parçasını tahmin ederek, tekrar tekrar. Telefon klavyeniz bunun
minik halini yapıyor — "görüşürüz" yazın, *yarın* önerir. LLM, aynı numaranın
milyarlarca kat büyütülmüşüdür ve ilginç olan, oyunun talep ettiği şeydir:
insan metninde sıradaki kelimeyi iyi tahmin etmek için model dil bilgisini,
olguları, üslubu ve akıl yürütmenin işleyen bir taklidini özümsemek zorunda.
Aşağıdaki her şey bu fikrin dipnotudur.

## 1. Metin sayıya dönüşür

**Tokenizer**, metni **token** denen parçalara böler — "için" tek parça,
"inanılmaz" belki "inan + ılmaz" — ve her birine bir kimlik numarası verir.
Kaba kural: 100 token ≈ 75 kelime; yani "128K bağlam penceresi" aşağı yukarı
bir roman tutar. Bilmeye değer bir sonuç: model harf görmez, yalnızca token
kimlikleri görür — "strawberry"deki r'leri saymanın meşhur zorluğu bundandır.
Bir tablonun *fotoğrafındaki* fırça darbelerini saymak gibidir.

## 2. Anlam taşıyan sayılar

Her token bir **embedding**'e dönüşür: bir anlam haritasındaki koordinatları
gibi davranan uzun bir sayı listesi. *Kral*, *kraliçe*ye yakın, *hesap
tablosu*na uzaktır; yönler ilişkidir — *kral − erkek + kadın*, *kraliçe*nin
yakınına düşer. Bu haritayı kimse çizmedi; öğrenildi. Bir torba koordinatta
sıra olmadığından, her token'ın **konumu** da işlenir: "köpek adamı ısırdı",
"adam köpeği ısırdı"dan farklı kalmalıdır.

## 3. Attention: bağlam içinde okumak

Orijinal Transformer makalesinden karşılaştırın:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Tek kelime değişir, "o" taraf değiştirir. Siz bunu anında çözdünüz;
**attention**, **Transformer**'ın — modern bütün modellerin arkasındaki
tasarım, GPT'deki T — aynısını yapma biçimidir. Her token üç rol oynar:
**sorgu** ("ne arıyorum?"), **anahtar** ("başkaları beni nasıl bulsun?"),
**değer** ("seçilirsem ne teslim ederim?"). YouTube'u düşünün: aramanız
sorgu, video başlıkları anahtar, videolar değer. "O çok yorgundu"da *o*,
daha önce geçen ve yorgun olabilecek bir şey arar; *hayvan*ın anahtarı güçlü
eşleşir, *cadde*ninki zayıf; puanlar yüzdeye çevrilir ve *o*, vektörünü
ağırlıklı bir karışım olarak yeniden kurar — diyelim %85 *hayvan*, %10
*cadde*. Her katman, her biri kendi ilişkisini izlemeyi öğrenen birçok
attention "kafasını" paralel çalıştırır: dil bilgisi, göndermeler, hangi
sıfat hangi ismin.

Bir asimetriye dikkat: sorgu bir kez ateşlenir; ama bir token'ın anahtarı ve
değeri, geriye bakan her sonraki token için geçerli kalır. Bunu aklınızda
tutun — birazdan paraya dönüşecek.

## 4. Katmanlar: bilgi nerede yaşıyor

Bunu onlarca-yüzü aşkın kez üst üste koyun. Her katmanda attention bağlamı
karıştırır (kütüphaneci), **ileri beslemeli ağ** ise "Paris, Fransa ile
eşleşir" gibi öğrenilmiş örüntüleri depolar (ambar — **parametrelerin** çoğu
burada yaşar). GPT-2, 2019'da 1,5 milyar parametreyle manşet olmuştu; öncü
modeller bugün trilyonlara varıyor. En tepede **softmax**, puanları toplamı
yüzde yüz olan olasılıklara çevirir — modelin bütün çıktısı, bildiği her
token için bir olasılıktır. "Bir varmış bir"den sonra kütle "yokmuş"a
yığılır; "En sevdiğim şehir"den sonra yüzlerce şehre dağılır. İkisi de
modelin cevapladığı tek sorunun doğru cevabıdır: *sırada ne gelmesi
muhtemel?*

## 5. Eğitim ve ölçek

**Ön eğitim**: modele trilyonlarca token gösterin, sıradakini gizleyin,
tahmin ettirin. **Kayıp fonksiyonu** şaşkınlığını puanlar; **gradyan inişi**
her parametreyi daha az şaşırma yönünde minicik bir adım kaydırır; bunu
trilyonlarca kez tekrarlayın. İçine kural yazılmaz — polisiye romanın son
bölümünü tahmin etmek kimin cinayet sebebi olduğunu izlemeyi gerektirir, o
yüzden izlemek öğrenilir. Sonuç, eğitim verisinin JPEG gibi
sıkıştırılmışıdır: resim kalır, pikseller kalmaz.

Ölçek, **ölçek yasalarını** izler: hesaplamayı ona katlayın, kayıp
öngörülebilir miktarda düşer — dokuz haneli eğitim bütçelerini kumardan plana
çeviren şey budur; GPT-4'ün nihai kaybı 10.000 kat küçük denemelerden
önceden tahmin edildi. DeepMind'ın **Chinchilla** çalışması denge kuralını
ekledi: parametre ve veri birlikte büyümeli (parametre başına ~20 token);
70 milyarlık modelleri, sırf bu aritmetikle 280 milyarlık rakibini geçti.
Dürüst bir şerh: eğri pürüzsüzdür ama beceriler aniden gelebilir — **beliren
yetenekler**.

## 6. Otomatik tamamlamadan asistana

Ön eğitimin ürünü **taban modeldir**: metni sürdüren bir makine, başka hiçbir
şey değil. "Fransa'nın başkenti neresi?" deyin; "Paris." alabilirsiniz — ya
da dokuz quiz sorusu daha, çünkü internette quizler sürü halinde gezer. Onu
asistan yapan iki ucuz aşama var. **Talimat eğitimi**: on binlerce yazılı
soru → ideal cevap örneğiyle eğitime devam edin; "yardımcı biçimde cevapla"
en olası devam haline gelsin. **RLHF** (insan geri bildirimiyle pekiştirmeli
öğrenme): insanlar aday cevapları karşılaştırır, bir ödül modeli bu zevki
öğrenir, LLM ona doğru ayarlanır — karşılaştırmak, kusursuz yazmaktan çok
daha kolaydır. Vurucu gerçek: GPT-3, ChatGPT'den iki yıl önce vardı. Devrim
daha büyük ağ değil, bu aşamalardı.

## 7. Üretim: plan değil, döngü

Model olasılıkları hesaplar, bir token **örnekler** (ağırlıklı çekiliş),
ekler, tekrarlar. Çekilişi üç düğme yönetir. "Gökyüzü"nden sonra: *maviydi*
%60, *karanlıktı* %10, … *patatesti* %0,0001. **Temperature** listeyi
sivriltir ya da düzleştirir (SQL için düşük, beyin fırtınası için yüksek);
**top-k** yalnızca en olası k token'ı tutar; **top-p**, toplamı örneğin
%90'ı bulan en küçük kümeyi tutar — model eminken iki, kararsızken seksen
token'a uyarlanır. Önce kes, sonra çek: cevapların günden güne değişmesi ve
gökyüzünün asla patates olmaması bundandır.

Döngünün iki sonucu: "adım adım düşün" işe yarar, çünkü modelin tek karalama
defteri sayfadır — "17 × 24 = 340 + 68" yazmak her sonraki tahmini
kolaylaştırır; akıl yürüten modeller bunu endüstrileştirir. Ve **KV cache**,
3. bölümdeki asimetriyi paraya çevirir: geçmiş anahtarlar ve değerler hiç
değişmez, bir kez hesaplanıp saklanır — uzun bir istemde ilk kelimeden
önceki duraklama, o önbelleği kuran **prefill**'dir; "önbelleklenmiş
girdinin" ucuz olması, bedelinin çoktan ödenmiş olmasındandır.

## 8. Sizi hatırlamaz

Eğitimden sonra parametreler **donar**. Her mesajda konuşmanın tamamı ağdan
yeniden geçer — her sabah bütün dava dosyasını isteyen, uzun süreli hafızası
olmayan parlak bir danışman gibi. Hafıza sanılan şey bağlam penceresidir.
Madalyonun öbür yüzü **in-context learning**: "deniz → sea, ev → house,
kedi → ?" gösterin; model *cat* der — görevi yalnızca istemden öğrenmiştir,
tek parametre değişmeden.

## 9. Neden uyduruyor

2023'te *Mata v. Avianca* davasında avukatlar, ChatGPT'nin uydurduğu altı
içtihadı mahkemeye sundu — "gerçek mi?" diye sorduklarında da "evet" almıştı.
5.000 dolarlık ceza, "yapay zekâ halüsinasyonunu" meşhur etti. Mekanizma sır
değil: model bir olasılık motorudur, veritabanı değil. Eğitim verisinin ince
olduğu yerde bulunamayacak kayıt yoktur — cevap *biçiminde* bir şey üretir;
çünkü optimize ettiği şey doğru değil, makuldür. Çözümler girdiyi değiştirir:
retrieval (RAG), araçlar ve kaynakları kendiniz kontrol etmek — avukatların
atladığı adım.

## Bütün hikâye beş satırda

1. Metin → **token** → **embedding** (anlamın koordinatları, konum dahil).
2. **Attention** (sorgu·anahtar·değer) her token'ın vektörünü bağlamıyla
   karıştırır; bilgiyi ileri beslemeli katmanlar depolar.
3. **Ön eğitim** = ölçekli sıradaki-token tahmini; ölçek yasaları kazancı
   öngörülebilir kılar; talimat eğitimi + RLHF taban modeli asistana çevirir.
4. Üretim = örnekle, ekle, tekrarla — temperature, top-k, top-p çekilişi
   ayarlar; KV cache bunu ödenebilir kılar.
5. Donmuş ağırlıklar; hafıza bağlam penceresidir; akıcıdır çünkü *makul*ü
   optimize eder — uydurmasının sebebi de aynıdır.

Ve bir dahaki sefere biri bu modellerin nasıl çalıştığını sorduğunda — bir
mülakatçı, bir öğrenci ya da içinizdeki meraklı ses — modelin başladığı
yerden başlayın: sıradaki token'dan.
