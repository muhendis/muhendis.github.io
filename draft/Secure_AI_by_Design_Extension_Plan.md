# Secure AI by Design — Genişleme Planı

**Tarih:** 2026-11-15
**Kapsam:** 5 yeni bölüm (13–17), mevcut 4 bölüme ek, 2 gerçek ölçüm.
**Durum:** Plan. Hiçbir makale dosyasına dokunulmadı.

## 1. Neden

Seri bugün "kendi sunucusunda LLM çalıştıran bir destek asistanı"nı uçtan uca anlatıyor. Türkçe metinlerde yapılan anahtar kelime taraması (yaklaşık 85 bin kelime) şu boşlukları gösterdi:

| Konu | Bugünkü durum | Planda karşılığı |
| --- | --- | --- |
| Yönetilen model API'leri (Bedrock, Azure OpenAI, Vertex) | 0 satır | Bölüm 13 |
| Kod çalıştıran, tarayıcı ve bilgisayar kullanan ajanlar | 0 satır; kum havuzu 7 satır | Bölüm 14 |
| Çok ajanlı sistemler, ajanlar arası kimlik | 1 yazıda 5 satır | Bölüm 15 |
| Model çalma, üyelik çıkarımı, ince ayar hattı | 1'er satır; ince ayar dağınık 20 satır | Bölüm 16 |
| EU AI Act | 0 satır | Bölüm 17 |
| Çok kipli enjeksiyon (görüntü, ses, belge) | 5 satır | Bölüm 5'e ek |
| Guardrail ölçümü | Simülatör ve şablon | Bölüm 6'ya gerçek ölçüm |
| Türkçe saldırı kümesi | Anlatı var, sonuç yok | Bölüm 10'a gerçek ölçüm |

## 2. Yapısal kararlar

1. **Numaralandırma: sona ekle (13–17), araya sokma.** Mevcut metinler birbirine "Bölüm 8" gibi numarayla atıf yapıyor; araya eklemek 26 dosyada atıf düzeltmesi gerektirir. Bölüm 12 bugün final; yeni bölümler "ikinci halka" olarak sunulur ve Bölüm 12'nin kanıt paketi her yeni bölümden bir satır alacak şekilde genişletilir.
2. **Ortak örnek aynı kalır, büyür.** Destek asistanı her yeni bölümde bir yetenek kazanır; yeni risk o yetenekten doğar (ayrıntı §3).
3. **Sabit sayılar korunur.** p=(0,10; 0,20; 0,30), β=ρ=0,05, zarar eşiği %0,5, κ ≥ 0,75, FPR eşiği %3, politika `2026-10-29.3`, aday `rc-2026-11.2`, model `support-7b@a1b2c3`. Yeni bölümler bu değerleri yeniden tanımlamaz, atıf yapar.
4. **Tehdit kaydı T7–T11 ile uzar.** Her yeni bölüm bir tehdit ekler; artık risk R' Bölüm 2'deki yöntemle yazım sırasında hesaplanır ve Bölüm 2 ile Bölüm 12 tablolarına işlenir.
5. **Seri kimliği değişmez** (EN `secure-ai-by-design`, TR `guvenli-yapay-zeka-tasarimi`); `seriesPart` 13–17. Seri açıklamasındaki "12 bölüm" ifadesi güncellenir.

## 3. Ortak örneğin büyümesi

| Bölüm | Asistanın kazandığı yetenek | Doğan tehdit |
| --- | --- | --- |
| 13 | Yoğun saatte yönetilen bir model API'sine taşma (Bölüm 9'daki fallback'in dış sağlayıcıya çıkması) | T7: müşteri verisinin sağlayıcıda saklanması veya yanlış bölgeye çıkması |
| 14 | "Rapor üret" aracı: müşterinin CSV dökümü üzerinde Python çalıştırma; bilgi tabanı için web sayfası açma | T8: enjekte edilmiş içeriğin kod çalıştırıp veri sızdırması |
| 15 | Faturalama ekibinin ajanına iş devretme | T9: devredilen yetkinin genişlemesi (confused deputy) |
| 16 | `support-7b`'nin geçmiş destek kayıtlarıyla ince ayarlanması | T10: eğitim kaydının çıkarılması, ağırlığın çalınması, arka kapılı adaptör |
| 17 | AB'deki müşterilere açılma | T11: yükümlülüğün karşılanmadığının denetimde ortaya çıkması |

## 4. Yeni bölümler

Her bölüm mevcut şablonu izler: belirti-önce giriş, kavram, sayısal yürüyüş, çalıştırılmış kod ve çıktısı, sınırlılıklar, pratik çıktı, quiz ve sözlük. Figürler inline SVG, W=720, EN ve TR ayrı.

### Bölüm 13 — Yönetilen Model API'leriyle Güvenlik: Paylaşılan Sorumluluk

**Temel soru:** Modeli başkası çalıştırdığında hangi kontroller hâlâ bizde, hangilerini yalnızca sözleşmeyle alabiliriz?

**İçerik akışı**
1. Belirti: taşma trafiği dış sağlayıcıya gidiyor; Bölüm 9'daki sunum kontrollerinin yarısı artık uygulanamıyor.
2. Sorumluluk matrisi: kendi barındırma, yönetilen bulut modeli, doğrudan model API'si. Satırlar: ağırlık bütünlüğü, istem ve yanıt saklama, kötüye kullanım izleme, bölge, şifreleme anahtarı, günlükler, alt işleyenler.
3. Veri saklama: varsayılan saklama, sıfır veri saklama seçenekleri, kötüye kullanım izleme için insan incelemesi; hangisi ayarla, hangisi sözleşmeyle kapanır.
4. Bölge ve yurt dışı aktarım: bölgesel uç nokta ile "global" yönlendirmenin farkı; KVKK m.9 ve GDPR aktarım koşullarına bağlantı (ayrıntı Bölüm 17).
5. Kimlik ve anahtar: uzun ömürlü API anahtarı yerine kısa ömürlü iş yükü kimliği; gateway'de sağlayıcı başına anahtar ve bütçe; anahtar sızıntısının maliyeti.
6. Sağlayıcıya gitmeden önce: Bölüm 3'teki takma adlandırmanın gateway'de uygulanması; geri dönüşte yeniden eşleme.
7. Sağlayıcı tarafı önbellek ve çok kiracılılık: Bölüm 9'daki zamanlama yan kanalının yönetilen API'deki karşılığı.
8. Model sürüm kayması: sağlayıcı modeli güncellediğinde Bölüm 10'daki regresyon kapısının yeniden koşması; sürüm sabitleme.

**Sayısal yürüyüş:** Taşma oranı %15 ve PII taşıyan istek oranı verilince günde sağlayıcıya çıkan kişisel veri içeren istek sayısı; takma adlandırmanın kaçırma oranıyla artık maruziyet.

**Kod:** Sağlayıcı yönlendirme politikası (veri sınıfı × bölge × saklama → izin/ret) ve takma adlandırıp geri eşleyen küçük bir sarmalayıcı; çıktısıyla.

**Figürler (3):** sorumluluk matrisi; taşma yolunda veri akışı ve güven sınırı; saklama pencereleri zaman çizelgesi.

**Pratik çıktı:** Sağlayıcı değerlendirme formu (12 soru) ve yönlendirme politikası YAML'ı.

**Yazarken teyit edilecekler:** her sağlayıcının güncel saklama ve eğitimde kullanmama taahhüdü; kötüye kullanım izleme muafiyeti koşulları; bölgesel ve çapraz bölge çıkarım davranışı; sağlayıcı tarafı istem önbelleğinin kiracı yalıtımı.

### Bölüm 14 — Kod Çalıştıran ve Tarayıcı Kullanan Ajanlar: Yalıtım

**Temel soru:** Ajan kendi yazdığı kodu çalıştırıyor ve güvenilmeyen sayfaları açıyorsa, hata yaptığında zararı ne sınırlar?

**İçerik akışı**
1. Belirti: müşterinin yüklediği CSV'nin bir hücresindeki talimat, "rapor üret" aracına ortam değişkenlerini dışarı göndermesini söyletiyor.
2. Üç koşulun kesişimi: özel veriye erişim, güvenilmeyen içerik, dışarıya iletişim. Bölüm 8'deki `auto ⟺ r ≤ τ(c)` kuralının neden yetmediği: kod çalıştırma tek bir "araç" değil, sınırsız bir araç ailesi.
3. Yalıtım basamakları: süreç, konteyner, kullanıcı alanı çekirdeği, mikro sanal makine. Her basamağın saldırı yüzeyi ve soğuk başlatma maliyeti.
4. Ağ çıkışı: varsayılan kapalı, alan adı izin listesi, DNS üzerinden sızdırma; çıkış vekili.
5. Dosya sistemi ve sırlar: salt okunur kök, geçici çalışma dizini, kum havuzuna sır koymama; sonuçların ana ajana veri olarak dönmesi (τ=0).
6. Tarayıcı ve bilgisayar kullanan ajanlar: sayfa içeriğinin talimat olarak okunması, oturum çerezlerinin kapsamı, ayrı tarayıcı profili, ekran görüntüsü üzerinden enjeksiyon.
7. Kodlama ajanları: depo içi talimat dosyaları, bağımlılık kurulum betikleri, MCP yapılandırmasının değiştirilmesi; onay yorgunluğu.
8. Kaynak sınırları: CPU, bellek, süre, çıktı boyutu; Bölüm 8'deki bütçeyle birleşme.

**Sayısal yürüyüş:** Çıkış kapalıyken ve açıkken T8'in artık riski; yalıtım basamağı başına gecikme bütçesi (ölçülmüş değer, tahmin değil).

**Kod:** Çıkış izin listesi karar fonksiyonu; kum havuzu politikası (YAML) ve onu doğrulayan denetleyici; çıktısıyla. Mümkünse gerçek bir konteynerde "çıkış kapalı" kanıtı.

**Figürler (4):** üç koşul kesişimi; yalıtım basamakları; kum havuzlu araç çağrısının yolu; tarayıcı ajanında güven sınırı.

**Pratik çıktı:** Kum havuzu politika şablonu ve 10 maddelik çıkış denetim listesi.

**Yazarken teyit edilecekler:** seçilen kum havuzu teknolojilerinin güncel yalıtım modeli; atıf yapılacak gerçek olayların (kodlama ajanı ve asistan enjeksiyon CVE'leri) numaraları ve kapsamı; "üç koşul" ve "ikiden fazlası olmasın" kurallarının özgün kaynakları.

### Bölüm 15 — Çok Ajanlı Sistemlerde Güven, Kimlik ve Yetki Devri

**Temel soru:** Bir ajan başka bir ajana iş verdiğinde, kimin adına ve hangi yetkiyle çalıştığını nasıl kanıtlarız?

**İçerik akışı**
1. Belirti: destek asistanı faturalama ajanına "iade et" diyor; faturalama ajanı kendi geniş yetkisiyle işlemi yapıyor ve kayıtta kullanıcı görünmüyor.
2. Ajan bir güven sınırıdır: başka ajanın çıktısı τ=0 veridir. Bölüm 5'teki ilkenin ajanlar arasına taşınması.
3. Kimlik: ajan kimliği, kullanıcı kimliği ve "adına" zinciri; jetonun daraltılarak devredilmesi; Bölüm 8'deki Macaroon kısıtlarının devirde kullanımı.
4. Yetki devri kuralı: devredilen yetki ≤ devredenin yetkisi ∩ kullanıcının yetkisi; zincir uzadıkça yalnız daralır.
5. Ajan keşfi ve yetenek kartları: sahte ajan, yetenek abartısı, Bölüm 4'teki imza ve köken kontrollerinin karta uygulanması.
6. Enjeksiyonun yayılması: bir ajandaki enjeksiyonun zincirde ilerlemesi; her atlamada bağımsız kontrol ve Büyük Resim'deki ortak kör nokta hesabı.
7. Paylaşılan bellek ve mesaj kuyruğu: Bölüm 7'deki zehirlemenin çok ajanlı hâli.
8. Sorumluluk ve iz: Bölüm 11'deki iz kimliğinin ajanlar arasında taşınması; hangi karar hangi ajana ait.

**Sayısal yürüyüş:** k atlamalı zincirde enjeksiyonun uca ulaşma olasılığı; bağımsız kontrolle ve ortak kör noktayla (β=0,05) karşılaştırma.

**Kod:** Yetki daraltan devir fonksiyonu (kesişim, süre, tutar tavanı) ve genişleme denemesinin reddi; çıktısıyla.

**Figürler (3):** "adına" zinciri; yetkinin atlamalarda daralması; iz kimliğinin yayılması.

**Pratik çıktı:** Ajanlar arası sözleşme şablonu (kimlik, kapsam, tavan, iz alanları).

**Yazarken teyit edilecekler:** ajanlar arası protokolün güncel sürümü ve kimlik doğrulama modeli; jeton değişimi ve iş yükü kimliği standartlarının ilgili maddeleri; OWASP'ın ajan uygulamaları listesinin güncel maddeleri.

### Bölüm 16 — Modelin Kendisini Korumak: İnce Ayar Hattı, Çıkarım ve Çalma

**Temel soru:** Model bir varlıksa, eğitim verisini ele vermesini, kopyalanmasını ve içine arka kapı konmasını ne engeller?

**İçerik akışı**
1. Belirti: ince ayarlı `support-7b`, doğru önekle bir müşterinin destek kaydını kelimesi kelimesine tamamlıyor.
2. Ezberleme ve çıkarım: üyelik çıkarımı (kayıp eşiği, referans modelle oran), eğitim verisi çıkarma; Bölüm 3'teki DP-SGD'nin ε değeriyle bağlantısı.
3. İnce ayar hattının güvenliği: veri kümesinin dondurulması ve özeti, eğitim ortamının yalıtımı, adaptörün imzalanması; Bölüm 4'teki köken zincirine ince ayar halkası.
4. Arka kapı: tetikleyicili zehirleme, adaptör üzerinden taşınan arka kapı; tespit sınırları ve davranış testleri.
5. İnce ayarın güvenlik davranışını bozması: az sayıda örnekle ret davranışının kaybı; ince ayar sonrası Bölüm 10 kapısının zorunlu olması.
6. Model çalma: sorgu yoluyla damıtma, logit ve olasılık çıktısından katman çıkarma, ağırlık dosyasının sızması; hangi API alanlarının kapatılacağı.
7. Ağırlığın korunması: erişim, şifreleme, Bölüm 9'daki gizli hesaplamayla bağlantı; filigran ve parmak izinin sınırları.
8. Sınırlılık: hiçbir tespit yöntemi arka kapının yokluğunu kanıtlamaz.

**Sayısal yürüyüş:** Küçük bir örnekte üyelik çıkarımı: üye ve üye olmayan kayıpların dağılımı, eşik, TPR ve FPR, Wilson aralığı.

**Kod:** Küçük bir açık model üzerinde üyelik çıkarımı denemesi (gerçek koşu) veya, donanım yetmezse, açıkça "sentetik" etiketli kayıp dağılımlarıyla aynı hesap.

**Figürler (3):** ince ayar hattı ve imza noktaları; üye ile üye olmayan kayıp dağılımı; çalma yolları ve kapatılan API alanları.

**Pratik çıktı:** İnce ayar yayın denetim listesi ve API çıktı alanı politikası.

**Yazarken teyit edilecekler:** üyelik çıkarımı ve eğitim verisi çıkarma makalelerinin sayıları; üretim modelinden katman çıkarma çalışmasının kapsamı; ince ayarın güvenlik hizasını bozduğunu gösteren çalışmanın örnek sayısı; "uyuyan ajan" çalışmasının iddiasının sınırı.

### Bölüm 17 — Düzenlemeden Kanıta: EU AI Act ve KVKK

**Temel soru:** Seri boyunca ürettiğimiz kanıtlar hangi yasal yükümlülüğü karşılar, hangisi için yeni kanıt gerekir?

**İçerik akışı**
1. Belirti: AB müşterisinin satın alma ekibi "AI Act uygunluğunuzu gösterin" diyor; elimizde Bölüm 12'nin standart eşlemesi var ama yasa eşlemesi yok.
2. Rol ve sınıf: sağlayıcı, dağıtıcı, genel amaçlı model sağlayıcısı; destek asistanının hangi sınıfa girdiği ve hangi koşulda yüksek riske geçtiği.
3. Destek asistanına düşen yükümlülükler: şeffaflık (kullanıcının bir yapay zekâyla konuştuğunu bilmesi), kayıt tutma, insan gözetimi; yüksek risk olsaydı eklenecekler (risk yönetimi, doğruluk, sağlamlık, siber güvenlik).
4. Takvim: hangi yükümlülük hangi tarihte yürürlüğe giriyor; ertelemeler.
5. KVKK tarafı: işleme şartı, aydınlatma, yurt dışı aktarım, veri sahibinin silme talebi ve Bölüm 3'teki silme hattı; Kurul'un yapay zekâya ilişkin rehberleri.
6. Eşleme tablosu: yükümlülük → serideki kontrol → kanıt dosyası → sahip. Bölüm 12'deki `unowned_gap_count` kapısının yasal maddelere genişletilmesi.
7. Standartların rolü: uyumlaştırılmış standartlar ve Bölüm 12'deki ISO ve ETSI atıflarının yasayla ilişkisi.
8. Sınırlılık: bu yazı hukuki görüş değildir; sınıflandırma kararı hukuk ekibinindir.

**Sayısal yürüyüş:** Eşleme tablosundan kapsama oranı: N yükümlülükten kaçı kanıtlı, kaçı sahipsiz; kapının geçme koşulu.

**Kod:** Bölüm 12'deki kanıt paketi denetleyicisinin yasal madde listesiyle koşturulması; çıktısıyla.

**Figürler (3):** rol ve sınıf karar ağacı; yürürlük takvimi; yükümlülük–kontrol–kanıt eşlemesi.

**Pratik çıktı:** Yükümlülük eşleme tablosu (CSV) ve kullanıcıya gösterilecek şeffaflık metni örneği.

**Yazarken teyit edilecekler (hepsi birincil kaynaktan):** madde numaraları ve metinleri; yürürlük tarihleri ve varsa erteleme düzenlemesinin son durumu; KVKK'nın yurt dışı aktarım maddesinin güncel hâli; Kurul rehberlerinin adı ve tarihi. KVKK için "6 aylık imha süresi" daha önce doğrulanamadı, yazılmayacak.

## 5. Mevcut bölümlere ekler

| Bölüm | Ek | Boyut |
| --- | --- | --- |
| 5 | Yeni H2: çok kipli enjeksiyon. Görüntüdeki metin, belge içi gizli katman, ses; OCR ve dosya dönüştürme çıktısının τ=0 sayılması; bir yürüyüş, bir figür. | ~600 kelime |
| 6 | Ablasyon şablonunun yerine gerçek ölçüm (§6.1). | tablo + 300 kelime |
| 10 | Türkçe saldırı kümesinde gerçek koşu (§6.2). | tablo + 400 kelime |
| 2 | Tehdit kaydına T7–T11 satırları. | tablo |
| 12 | Kanıt paketine yeni bölümlerden birer satır; "seri burada bitiyor" ifadesinin "ikinci halka"ya bağlanması. | ~200 kelime |
| Büyük Resim (3 yazı) | Seri haritasına 13–17; katman figürüne yeni kutular gerekiyorsa SVG metni düzenlenir. | harita + figür |

## 6. Gerçek ölçümler

Serinin iddiası "kontrol ölçülür"; bugün ölçümler simülatörle yapılıyor. İki ölçüm bunu kapatır.

### 6.1 Guardrail ablasyonu (Bölüm 6)
- **Katmanlar:** regex, açık bir enjeksiyon sınıflandırıcısı, küçük bir karar modeli; tek tek ve birlikte.
- **Veri:** açık bir enjeksiyon kümesinden örneklem ile zararsız destek mesajları; boyut her hücrede Wilson aralığı ±%5'in altına inecek kadar.
- **Raporlanan:** kaçırma oranı, yanlış alarm oranı (eşik %3), p50 ve p95 gecikme; donanım ve sürümlerle birlikte.
- **Dürüstlük kuralı:** sonuç Büyük Resim'deki örnek p değerleriyle uyuşmazsa örnek değiştirilmez; "örnek sayılar öğretim amaçlıdır, ölçüm şudur" diye yan yana verilir.

### 6.2 Türkçe saldırı kümesi (Bölüm 10)
- **Küme:** Bölüm 6'daki üç Türkçe riskten (ek ile gizleme, diakritik, dil karıştırma) üretilmiş tohumlar; tohum başına tek varyant kuralı korunur.
- **Karşılaştırma:** aynı tohumların İngilizce karşılığı; sınıflandırıcının Türkçeyi değerlendirmediği notunun ölçümle sınanması.
- **Raporlanan:** dil başına kaçırma oranı ve aralığı; hakem tutarlılığı κ.
- **Yayın:** küme ve betik depoya konur (zararlı yük içermeyen, yalnız enjeksiyon kalıbı taşıyan örnekler).

**Ön koşul:** hangi donanımda koşulacağı (bkz. §9). GPU yoksa sınıflandırıcılar CPU'da koşar; karar modeli ölçümü küçültülür veya kapsam dışı bırakılır ve bu açıkça yazılır.

## 7. Çalışma sırası

| Sıra | İş | Gerekçe |
| --- | --- | --- |
| 1 | Bölüm 13 (yönetilen API) | En geniş okur kitlesinin boşluğu; Bölüm 9'a doğrudan bağlanıyor. |
| 2 | Bölüm 14 (yalıtım) | En yüksek riskli ajan sınıfı; Bölüm 8'in devamı. |
| 3 | §6.1 ve §6.2 ölçümleri | Serinin kendi iddiasını kanıtlar; sonraki bölümlerden bağımsız. |
| 4 | Bölüm 5 eki (çok kipli) | Küçük, bağımsız. |
| 5 | Bölüm 15 (çok ajanlı) | 14'teki yalıtım kavramlarına dayanır. |
| 6 | Bölüm 16 (model koruma) | Bölüm 3 ve 4'e dayanır; deney gerektirir. |
| 7 | Bölüm 17 (düzenleme) | En son: önceki bölümlerin kanıtlarını eşler. |
| 8 | Bölüm 2, 12 ve Büyük Resim güncellemeleri | Her yeni bölümle artımlı; sonda bütünlük geçişi. |

Her bölüm kendi başına yayınlanabilir; sıra 1 ve 2 bittiğinde seri "uçtan uca" iddiasının büyük kısmını karşılar.

## 8. Kabul ölçütleri (bölüm başına)

- EN ve TR birlikte yazıldı; `validate_post.sh` temiz (TOC, KaTeX, simetri), `validate_site.py` 0 hata.
- Her Python bloğu çalıştırıldı, çıktı makaledekiyle birebir.
- Her olgusal iddia birincil kaynağa bağlı; teyit edilemeyen iddia yazılmadı ya da açıkça işaretlendi.
- Sabit sayılarla çelişki yok; yeni tehdit Bölüm 2 ve 12 tablolarına işlendi.
- Figürler SVG, W=720, açık ve koyu temada kontrol edildi; EN figüründe Türkçe metin yok.
- `posts.json` (iki dil), `sitemap.xml` ve seri haritaları güncel.
- Engin "push" demeden commit ve push yok.

## 9. Açık kararlar

1. **Kapsam:** beş bölümün hepsi mi, yoksa yalnız 13 ve 14 mü? Öneri: 13, 14 ve ölçümlerle başla; 15–17'ye onlardan sonra karar ver.
2. **Bölüm 15'in bağımsızlığı:** çok ajanlı konu 14'ün içine bir H2 olarak da sığar. Öneri: ayrı bölüm; yetki devri hesabı kendi yürüyüşünü hak ediyor.
3. **Ölçüm donanımı:** ölçümler hangi makinede koşacak (GPU var mı, hangi model boyutu makul)?
4. **Sağlayıcı adları:** Bölüm 13 sağlayıcıları adıyla mı karşılaştırsın, yoksa "yönetilen bulut modeli" ve "doğrudan API" sınıflarıyla mı kalsın? Öneri: sınıflarla anlat, adları yalnız teyitli olgularda ve tarih vererek kullan; sağlayıcı koşulları hızlı eskiyor.
5. **Bölüm 17'nin derinliği:** yalnız destek asistanı (sınırlı risk) mi, yoksa yüksek riskli bir karşı örnek de mi? Öneri: karşı örnek yalnız bir tabloyla.
