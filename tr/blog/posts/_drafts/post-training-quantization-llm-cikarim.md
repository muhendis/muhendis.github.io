> **Taslak.** Yüksek lisans tezimden bir başlangıç noktası olarak derlenmiştir.
> `posts.json` içinde `"draft": false` yapmadan önce düzenleyin.

Büyük bir dil modelini sunmak (serving) büyük ölçüde bir bellek problemidir.
Üretilen her bir token için parametrelerin HBM'den (yüksek bant genişlikli
bellek) hesaplama birimlerine taşınması gerekir ve modern GPU'larda beklediğiniz
şey aritmetik hesaplama değil, tam olarak bu veri aktarımıdır. Otoregresif
çözümleme (autoregressive decoding) bunu daha da ağırlaştırır: her adımda tek
bir token üretilirken modelin tüm ağırlık matrisi taşınır ve tensör çekirdekleri
ne kadar hızlı olursa olsun aritmetik yoğunluk (arithmetic intensity) düşük kalır.

Nicemleme (quantization) doğrudan bu darboğaza saldırır: Ağırlıkları daha az bit
ile saklayın, daha az bayt taşıyın, daha az bekleyin.

## Neden quantization-aware değil de post-training?

Eğitim sırasında nicemleme farkındalığı (quantization-aware training / QAT),
düşük bit genişliklerinde daha iyi doğruluk sağlar; ancak ince ayar yapmak için
eğitim hattına, verilere ve devasa hesaplama bütçesine ihtiyaç duyar. Kendi
eğitmediğiniz bir model için —ki pratikte modellerin büyük çoğunluğu böyledir—
bu imkân elinizde yoktur.

Eğitim sonrası nicemleme (post-training quantization / PTQ), önceden eğitilmiş
bir checkpoint ve küçük bir kalibrasyon veri seti üzerinden çalışır. Pratik ve
uygulanabilir seçenek budur; buradaki asıl kritik soru ne kadar doğruluk
kaybettiğiniz ve bu kaybın nereden kaynaklandığıdır.

## Bütün problem aykırı değerlerdir (outliers)

Transformer mimarilerinde naif PTQ'nun çökmesinin temel nedeni aktivasyon
aykırı değerleridir (activation outliers). Daha büyük modellerde, az sayıda
öznitelik boyutu diğerlerinin katbekat üzerinde değerler taşır ve bunlar
tutarlı bir biçimde hep aynı kanallarda ortaya çıkar.

Nicemleme, kayan noktalı (floating-point) bir aralığı küçük bir tamsayı
ızgarasına eşler. Tek bir aşırı uç değer bu aralığı fazlasıyla gerer; böylece
tüm sıradan değerler yalnızca birkaç basamağa sıkışır ve efektif hassasiyet yok
olur. Ortalama gayet iyidir; kuyruktaki uç değerler ise her şeyi bozar.

Tensör başına (per-tensor) ölçeklemenin yetersiz kalmasının ve işe yarar
yöntemlerin daha ince tanecikli —kanal başına (per-channel) veya grup başına
(per-group)— ölçeklerle çalışmasının ya da zorluğu tek bir yerde soğurmak yerine
ağırlıklar ile aktivasyonlar arasında paylaştırmasının nedeni tam olarak budur.

## Pratikte gerçekte ne önemlidir?

- **Ağırlıkları nicemlemek aktivasyonlara göre çok daha kolaydır.** 4-bit
  seviyesindeki salt-ağırlık (weight-only) şemaları nispeten bağışlayıcıdır.
  Asıl dikkatli çalışma gerektiren yer 8-bit aktivasyonlardır.
- **Kalibrasyon verisinin niteliği miktarından önemlidir.** Dağıtım ortamındaki
  gerçek kullanım dağılımına benzeyen birkaç yüz örnek, dağılımı tutmayan
  binlerce örnekten çok daha iyi sonuç verir.
- **Her şeyin işe yarayıp yaramayacağını kernel desteği belirler.** Hedef
  donanım mimariniz için verimli bir kernel'i olmayan bir nicemleme şeması,
  yalnızca kâğıt üzerinde kalan bir sonuçtur. Bellek tasarrufu gerçektir; ancak
  gecikme kazanımı, matris çarpımından önce kayan noktalı bir kopya oluşturmak
  yerine ters nicemleme (dequantization) yolunun kernel içinde kaynaştırılmış
  (fused) olmasına bağlıdır.

## Dürüstçe ölçmek

Perplexity (şaşkınlık) zayıf bir göstergedir. Görev doğruluğu ciddi biçimde
oynarken perplexity çok az değişir; bu da onu raporlaması rahat ama tek başına
güvenmesi tehlikeli bir metrik yapar. Asıl önemli olan sayı, fiilen sunduğunuz
görevler üzerindeki uçtan uca değerlendirmedir.

Aynı durum hız için de geçerlidir. Gecikmeyi gerçekçi bir batch boyutu ve dizi
uzunluğunda, prefill ile decode fazlarını birbirinden ayırarak uçtan uca
raporlayın — ikisi farklı donanım sınırlarına tabidir ve nicemleme ikisine eşit
derecede yardım etmez.

## Kaynak

Yöntemin tamamı ve ayrıntılı sonuçlar, Bahçeşehir Üniversitesi'nde Dr. Öğr.
Üyesi Fatih Kahraman danışmanlığında hazırladığım yüksek lisans tezimde yer
almaktadır: *Post-Training Quantization for Efficient Inference of Large
Language Models on Modern GPU Architectures*.

