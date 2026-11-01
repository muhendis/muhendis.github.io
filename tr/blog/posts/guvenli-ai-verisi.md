Bir müşteri, şirketten kişisel verilerinin silinmesini istiyor. CRM ekibi silme işlemini çalıştırıyor ve kayıt bir saniyeden kısa sürede ortadan kalkıyor. Üç hafta sonra bir çalışan destek asistanından eski bir şikâyet yazışmasını özetlemesini istiyor ve özet, o müşterinin telefon numarasını aktarıyor. CRM kaydı gerçekten silmişti. Ama aynı telefon numarası vektör dizinindeki bir parçada, semantik önbellekte, bir log dışa aktarımında, birinin "gerçekçi görünsün" diye gerçek şikâyetlerden kurduğu bir değerlendirme setinde ve gecelik yedekte de yaşıyordu.

Bu, yapay zekâ sistemlerinde en sık rastlanan veri güvenliği arızasıdır ve bir algoritma hatası değil, bir sistemik envanter ve yaşam döngüsü problemidir. Bir AI sistemine giren veri hızla çoğalır. Tek bir müşteri şikâyeti bir retrieval parçasına, bir embedding vektörüne, bir prompt'a, bir model yanıtına, bir trace kaydına, önbelleğe alınmış bir yanıta ve belki bir fine-tuning veri setine dönüşür. Her kopyanın kendi depolaması, kendi erişim kontrolü ve kendi saklama süresi vardır. Silme süreci bunların yalnızca bir kısmını biliyorsa, her adım ne kadar hızlı olursa olsun silme eksik kalır.

> **Yapay Zekâ Veri Güvenliğinin Temel Aksiyomu (Embedding Tersinirliği ve Model Kalıcılığı):**
> Vektör embedding'leri ($\mathbf{v} \in \mathbb{R}^d$) anonimleştirilmiş veri değildir; modern tersine çevirme teknikleriyle (Vec2Text) orijinal metne geri döndürülebilir ve ham veriyle aynı gizlilik sınıfında korunmalıdır. Ayrıca bir kişisel veri parçası model ağırlıklarına ince ayar (fine-tuning) ile bir kez işlendiğinde, onu seçici bir sorguyla silmek matematiksel olarak imkânsızdır. Kişisel veri modelden sonra değil, eğitime girmeden **önce** diferansiyel mahremiyet ve katı temizlikle arındırılmalıdır.

Bu yazı, veriyi destek asistanının içinde toplandığı andan ortadan kalkması gereken ana kadar adım adım izliyor. Bir AI sisteminin işlediği beş veri türünü ayıracak, Carlini ve Koh-Liang'ın etki fonksiyonları (*influence functions*) ile veri zehirlemenin gradyan seviyesindeki mekanizmasını açığa çıkaracak, Diferansiyel Mahremiyetin ($(\epsilon, \delta)$-DP) matematiksel garantilerini kuracak, sağlama toplamı (checksum) ve NER'i birleştiren bir PII dedektörü inşa edecek ve maskeleme, takma adlandırma (pseudonymization) ve anonimleştirme arasındaki hukuki ve teknik sınırları netleştireceğiz. Yazının sonunda elinizde dokuz evi kapsayan bir veri envanteri, bir PII işleme akışı ve test edilebilir bir yaşam döngüsü kontrol listesi olacak.

> **Önceki yazıdan devraldığımız fikir:** [İkinci bölümde](post.html?slug=ai-tehdit-modelleme-ve-guven-sinirlari) çizdiğimiz yedi güven sınırından ikisi doğrudan veriyle ilgiliydi: sağlayıcıya aktarılan loglar (TB6) ve dış göndericinin yazdığı şikâyet arşivi (TB4). Şimdi bu verilerin sistemin hangi depolarında çoğaldığını ve model ağırlıklarına sızmasını nasıl engelleyeceğimizi inceliyoruz.

> **Bu yazının sonunda**
>
> - Bir AI sisteminin işlediği beş veri türünü ve her birinin gerektirdiği mimari koruma katmanını sayabileceksiniz.
> - Temiz etiketli veri zehirlemenin etki fonksiyonuyla ($I_{\text{up,loss}}$) ve web ölçeğindeki bölünmüş görüş saldırılarıyla nasıl çalıştığını açıklayabileceksiniz.
> - Diferansiyel Mahremiyetin ($(\epsilon, \delta)$-DP) ezberlemeyi nasıl engellediğini ve embedding vektörlerinin neden düz metin gibi korunması gerektiğini savunabileceksiniz.
> - TC Kimlik numaraları için sağlama toplamına dayalı bir filtre yazıp, bağlamsal PII için ne zaman Türkçe NER gerektiğinin ayrımına varacaksınız.
> - Bağımsız $N$ kopyada silmenin eksiksiz olma olasılığını ($(1-q)^N$) hesaplayabilecek ve kriptografik imha (*crypto-shredding*) ile silmeyi tek anahtara indirebileceksiniz.

**Bu yazıda**

- [1. Tek kayıt, dokuz ev: AI verisinin çoğalma haritası](#1-tek-kayıt-dokuz-ev-ai-verisinin-çoğalma-haritası)
- [2. Beş veri türü, beş güvenlik ihtiyacı](#2-beş-veri-türü-beş-güvenlik-ihtiyacı)
- [3. Köken ve zehirleme: Etki fonksiyonları ve web ölçeğinde sabotaj](#3-köken-ve-zehirleme-etki-fonksiyonları-ve-web-ölçeğinde-sabotaj)
- [4. Diferansiyel mahremiyet ve embedding tersine çevirme saldırıları](#4-diferansiyel-mahremiyet-ve-embedding-tersine-çevirme-saldırıları)
- [5. Kişisel veriyi bulmak: Kalıplar, sağlama toplamları ve Türkçe NER](#5-kişisel-veriyi-bulmak-kalıplar-sağlama-toplamları-ve-türkçe-ner)
- [6. Maskeleme, takma adlandırma, anonimleştirme](#6-maskeleme-takma-adlandırma-anonimleştirme)
- [7. Saklama ve silme: Unutulan kopyaların matematiği ve kripto-imha](#7-saklama-ve-silme-unutulan-kopyaların-matematiği-ve-kripto-imha)
- [8. Mühendisliğin iki gözü: Kurulum zamanı ve çalışma zamanı](#8-mühendisliğin-iki-gözü-kurulum-zamanı-ve-çalışma-zamanı)
- [9. Pratik çıktı: Envanter, akış ve kontrol listesi](#9-pratik-çıktı-envanter-akış-ve-kontrol-listesi)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Tek kayıt, dokuz ev: AI verisinin çoğalma haritası

Yoğun bir kamu dairesine elden resmi bir dilekçe bıraktığınızı düşünün. Biri aslını arşive kaldırır. Bir başkası fotokopisini çekip masasına koyar. Üçüncü kişi önemli bir paragrafını yapışkan nota yazıp panoya asar; tarayıcı cihazı da belleğinde taranmış dijital kopyayı saklar. Dilekçe sahibi daha sonra gelip evrakının imha edilmesini istediğinde arşivdeki asıl nüshayı yakmak yetmez. Fotokopiyi, yapışkan notu ve tarayıcının önbelleğini de yok etmek zorundasınızdır.

Destek asistanımız tam olarak o dairedir. Bir müşterinin adını, telefon numarasını ve TC Kimlik numarasını içeren sentetik bir şikâyet metninin sistem içindeki yolculuğunu izleyelim:

<!-- fig:p3_homes -->
<svg viewBox="0 0 720 366" role="img" aria-label="Kaynak sistemdeki tek bir şikâyet, yani 1. ev, müşteri istediğinde silinir. Türetilmiş sekiz kopya başka yerlerde yaşar: 2 retrieval parçaları, 3 embedding vektörleri, 4 model sağlayıcısındaki prompt ve yanıtlar, 5 semantik önbellek, 6 log ve trace kayıtları, 7 dışa aktarımlar ve analitik, 8 eğitim ve değerlendirme setleri, 9 yedekler. Hikâyede 2, 5, 6, 8 ve 9. evler CRM silmesinden üç hafta sonra hâlâ telefon numarasını tutuyordu." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p3h-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Tek Bir Sentetik Şikâyet, Dokuz Ayrı Yaşam Alanı</text>
<rect x="16" y="112" width="170" height="84" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="26" y="122" width="22" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="37.0" y="133.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="56" y="135" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Kaynak Sistem</text>
<text x="26" y="160" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">şikâyet arşivi / CRM</text>
<text x="26" y="180" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">Talep üzerine silindi ✓</text>
<rect x="222" y="44" width="482" height="216" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="234" y="64" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">türetilmiş kopyalar: her birinin kendi sahibi, erişim listesi ve saklama süresi var</text>
<line x1="186" y1="154" x2="220" y2="154" marker-end="url(#p3h-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="234" y="76" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="242" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="252.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<text x="268" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Parçalar</text>
<text x="242" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">retrieval dizini</text>
<text x="242" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI ekibi</text>
<rect x="351" y="76" width="108" height="74" rx="6" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="359" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="369.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<text x="385" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Vektörler</text>
<text x="359" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tersinir temsil</text>
<text x="359" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI ekibi</text>
<rect x="468" y="76" width="108" height="74" rx="6" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="476" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="486.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="502" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Prompt'lar</text>
<text x="476" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sağlayıcıya gider</text>
<text x="476" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TB2, sağlayıcı</text>
<rect x="585" y="76" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="593" y="84" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="603.0" y="95.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="619" y="97" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Önbellek</text>
<text x="593" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">semantik yanıtlar</text>
<text x="593" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">platform, TTL?</text>
<rect x="234" y="164" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="242" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="252.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<text x="268" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Loglar</text>
<text x="242" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">trace, aktarım</text>
<text x="242" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">platform, sağlayıcı</text>
<rect x="351" y="164" width="108" height="74" rx="6" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="359" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="369.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<text x="385" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Aktarımlar</text>
<text x="359" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">rapor için CSV</text>
<text x="359" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">analist dizüstü</text>
<rect x="468" y="164" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="476" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="486.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="502" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Eğitim/Test</text>
<text x="476" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">“gerçekçi” set</text>
<text x="476" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI ekibi, süresiz</text>
<rect x="585" y="164" width="108" height="74" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<rect x="593" y="172" width="20" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="603.0" y="183.8" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<text x="619" y="185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Yedekler</text>
<text x="593" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gecelik görüntü</text>
<text x="593" y="222" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">operasyon, 35 gün</text>
<rect x="16" y="272" width="22" height="14" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:4 3"/>
<text x="46" y="283" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kırmızı, kesikli: CRM silmesinden üç hafta sonra veriyi tutmaya devam eden yetim kopyalar</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="294" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="313" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 VERİ ÇOĞALMASI İLKESİ: BİR KAYIT GİRER, DOKUZ FARKLI SİSTEMDE YAŞAR</text>
<text x="28" y="331" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Geleneksel veritabanından bir satırı silmek yapay zeka hattında veriyi yok etmez; embedding&#x27;ler ve loglar yetim kalır.</text>
<text x="28" y="347" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Uçtan uca silme garantisi ancak deterministik takma adlandırma ve kriptografik imha (crypto-shredding) ile sağlanabilir.</text>
</svg>

1. **Kaynak Sistem (Ev 1):** Şikâyet arşivindeki orijinal e-posta. Silme talebinde ilk ve çoğu zaman *yalnızca* burası temizlenir.
2. **Retrieval Parçaları (Ev 2):** RAG araması için metin parçalarına (chunk) bölünmüş kayıt.
3. **Embedding Vektörleri (Ev 3):** Her metin parçasının yüksek boyutlu uzaydaki ($\mathbb{R}^d$) sayısal izdüşümü.
4. **Prompt'lar ve Yanıtlar (Ev 4):** Modelin bağlam penceresinde ve model sağlayıcısına gönderilen API yükü (TB2).
5. **Semantik Önbellek (Ev 5):** Benzer şikâyetlerin özetlerini tekrar üretmemek için saklanan Redis veya vLLM prefix cache deposu.
6. **Log ve Trace Kayıtları (Ev 6):** Hata ayıklama ve telemetri için yakalanan OpenTelemetry span'leri.
7. **Dışa Aktarımlar ve Analitik (Ev 7):** Operasyon analistinin masaüstüne indirdiği CSV veya Excel raporları.
8. **Eğitim ve Değerlendirme Setleri (Ev 8):** Modeli ince ayar yapmak (fine-tuning) veya değerlendirmek için kopyalanmış altın veri setleri.
9. **Yedekler (Ev 9):** Yukarıdaki veritabanlarının gecelik anlık görüntüleri (*snapshots*).

Bu dokuz yaşam alanının her birinin farklı bir yöneticisi, farklı bir erişim listesi ve farklı bir saklama süresi vardır. Bir veri güvenliği mimarının görevi, bu dokuz evi birbirine bağlayan tekil bir kimlik anahtarı ve silme orkestrasyonu kurmaktır.

---

## 2. Beş veri türü, beş güvenlik ihtiyacı

Bir yapay zekâ sistemindeki bütün veriler aynı yaşam döngüsüne sahip değildir:

| Veri Türü | Destek Asistanında | Kim Yazar? | Başlıca Risk | Zorunlu Güvenlik Kontrolü |
| --- | --- | --- | --- | --- |
| **Ön Eğitim Verisi** | Temel modelin web derlemi | Sağlayıcı / Açık Topluluk | Denetlenemeyen arka kapılar ve telif ihlalleri | Model kartı incelemesi, kırmızı takım testleri (Bölüm 4) |
| **Fine-Tuning Verisi** | 5.000 adet şikâyet-özet çifti | Kurum içi veri mühendisleri | Model ağırlıklarına ezberlenen PII; veri zehirleme | **Eğitim öncesi PII arındırma**, $(\epsilon, \delta)$-DP-SGD, veri köken imzası |
| **Değerlendirme Verisi** | 400 adet altın test şikâyeti | AI güvenlik ve test ekibi | Eğitime sızma (*data contamination*); test loglarında PII | Sentetik veri kullanımı, CI/CD erişim kısıtlaması |
| **Çalışma Zamanı Kullanıcı Verisi** | Çalışan prompt'u, oturum geçmişi | Yetkili kullanıcı | Oturumlar arası sızıntı; aşırı uzun saklama | Kullanıcı izolasyonu, kısa TTL (24 saat), sıfır veri saklama (ZDR) |
| **Retrieval Verisi (RAG)** | Şikâyet arşivi, CRM, bilgi tabanı | Çok sayıda dış ve iç aktör | Dolaylı prompt injection; yetkisiz erişim | ACL etiketli chunking, kaynak güven sınıflandırması |

İki mimari gerçeklik sistem tasarımını belirler:

**En uzun yaşayan ve en tehlikeli olan fine-tuning verisidir.** Kişisel veri model ağırlıklarına ($W$) gradyan güncellemesi olarak girdiğinde, onu bir SQL `DELETE` komutuyla geri alamazsınız. Akademik dünyadaki *Machine Unlearning* (makineye unutturma) çalışmaları henüz kurumsal kesinlikte kanıtlanabilir garantiler sunamamaktadır. Pratik kural kesindir: **Eğer bir silme talebi geldiğinde modeli çöpe atıp baştan eğitmeyi göze almıyorsanız, ham kişisel veriyi fine-tuning setine asla sokmayın.**

**En heterojen ve güvensiz olan retrieval verisidir.** Şikâyet arşivini şirket dışındaki binlerce yabancı yazar. Bu veri hem bir mahremiyet nesnesidir (içinde müşteri sırları vardır) hem de potansiyel bir saldırı vektörüdür (içinde gizli talimatlar olabilir).

---

## 3. Köken ve zehirleme: Etki fonksiyonları ve web ölçeğinde sabotaj

**Köken** (*Data Provenance*), bir veri parçasının doğumundan model eğitimine kadar geçirdiği tüm dönüşümlerin, kaynak URL'lerinin, hash değerlerinin ve onay imzalarının kriptografik kaydıdır. Gıda endüstrisindeki soğuk zincir takip sistemine benzer: Çiftlikten sofraya her adım belgelenir ki zehirli bir parti çıktığında tüm ürünleri imha etmek yerine yalnızca sorunlu parti geri çağrılabilsin.

**Veri Zehirleme** (*Data Poisoning*), köken denetiminin savunmasız kaldığı anlarda gerçekleşir. Saldırgan, eğitim veya retrieval verisine özel tasarlanmış örnekler enjekte ederek modelin hedefli girdilerde yanlış veya saldırgan lehine kararlar almasını sağlar (`AML.T0020`, `AML.T0043`).

### Koh-Liang Etki Fonksiyonları (Influence Functions)

Temiz etiketli (*clean-label*) bir zehirleme saldırısında saldırgan etiketleri değiştirmez; metnin içine soktuğu hassas manipülasyonlarla modelin parametre gradyanlarını saptırır. Bir $z = (x, y)$ eğitim örneğinin, modelin belirli bir $z_{\text{test}}$ test girdisi üzerindeki kaybına ($L$) olan etkisi Koh ve Liang (2017) tarafından **Etki Fonksiyonu** formülasyonuyla türetilmiştir:

$$
I_{\text{up,loss}}(z, z_{\text{test}}) = -\nabla_\theta L(z_{\text{test}}, \hat{\theta})^\top H_{\hat{\theta}}^{-1} \nabla_\theta L(z, \hat{\theta})
$$

Şöyle okuyun: $\nabla_\theta L(z, \hat{\theta})$, $z$ eğitim örneğinin model ağırlıkları üzerindeki gradyan vektörüdür. $H_{\hat{\theta}}^{-1}$, tüm eğitim veri seti üzerinden hesaplanan ampirik kayıp fonksiyonunun ters Hessian matrisidir (ağırlık uzayının ikinci dereceden eğriliği). $\nabla_\theta L(z_{\text{test}}, \hat{\theta})$ ise modelin hedef test girdisindeki gradyanıdır.

Bu formül şu fiziksel anlamı taşır:
- Eğer $I_{\text{up,loss}}$ negatif ise, $z$ örneğini eğitim setine eklemek modelin $z_{\text{test}}$ üzerindeki kaybını **azaltır** (model o yönde öğrenir).
- Saldırgan, bu iç çarpımı tersine çevirerek $I_{\text{up,loss}}$ değerini maksimize eden sentetik $z$ belgeleri üretir; böylece çok küçük bir örnek grubuyla modelin hedef bir test vakasındaki kararını sessizce saptırır.

### Zehirleme İçin Kaç Belge Yeterlidir?

Endüstriyel ölçümler saldırganın büyük hacimlere ihtiyaç duymadığını göstermektedir:
1. **Web Ölçeğinde Bölünmüş Görüş Zehirlemesi (Split-View Poisoning):** Carlini ve ark. (2023), URL listeleriyle paylaşılan açık veri setlerinde süresi dolan alan adlarının satın alınarak LAION veya COYO derlemlerinin %0,01'inin yalnızca **60 dolar** maliyetle zehirlenebildiğini kanıtladı.
2. **Büyük Dil Modellerinde Arka Kapı Eşiği:** Anthropic ve UK AI Security Institute (Ekim 2025), 600M ile 13B parametre arasındaki modellerde sadece **250 zehirli belgenin** modeli kalıcı bir arka kapıyla mühürlemeye yettiğini belgeledi. 13B'lik bir model için bu miktar, eğitim külliyatının yalnızca **%0,00016'sına** (milyonda 1,6) denk gelmektedir.
3. **PoisonedRAG:** Zou ve ark. (2024), milyonlarca belgelik bir RAG bilgi tabanına hedef soru başına yalnızca **5 adet** düşmanca üretilmiş metin ekleyerek retrieval başarı oranını %90 seviyesinde manipüle etmiştir.

Bu sabotajı önleyen köken kontrolleri:
- **Toplama Anında SHA-256 Hash'leme:** Dış kaynaktan çekilen her dokümanın kriptografik özeti kaydedilir; kaynak sonradan değişirse belge derhal hattan düşürülür.
- **Değiştirilemez Veri Sürümleri:** Fine-tuning rastgele bir "en son arşiv" üzerinde değil; Git LFS veya DVC ile dondurulmuş, hash'lenmiş veri sürümleriyle çalıştırılır.

---

## 4. Diferansiyel mahremiyet ve embedding tersine çevirme saldırıları

Kişisel verilerin yapay zekâda korunması iki derin matematiksel yanılsamayı yıkmayı gerektirir: "Vektörler sayıdır, dolayısıyla gizlidir" ve "İsimleri silersek model ezberlemez".

### 4.1. Embedding Vektörlerinin Tersine Çevrilmesi (Vec2Text)

Pek çok mühendislik ekibi, müşteri şikâyetlerini parçalayıp bir embedding modelinden (örn. `text-embedding-3-large`) geçirdikten sonra ortaya çıkan $\mathbf{v} \in \mathbb{R}^{3072}$ vektörlerinin güvenli olduğunu varsayar.

Bu varsayım tehlikeli bir yanılgıdır. Morris ve ark. (2023) tarafından geliştirilen **Vec2Text** yöntemi, yoğun embedding vektörlerinin yüksek derecede tersinir (*invertible*) olduğunu kanıtlamıştır. Bir düzeltici dil modeli eğiterek embedding uzayından geriye doğru metin arayan bu yöntem:
- 32 token uzunluğundaki metinlerin **%92'sini harfi harfine (verbatim)** geri kurabilmektedir.
- Klinik sağlık kayıtlarındaki hasta isimlerini, teşhisleri ve ilaç adlarını %95'in üzerinde kesinlikle vektörden tekrar üretmiştir.

$$\text{Optimizasyon:} \quad \hat{x}^* = \arg\min_{\hat{x}} \|\text{Embed}(\hat{x}) - \mathbf{v}\|_2^2$$

> [!TIP]
> **Öğrenme Kontrolü: Embedding Vektörleri Anonim Değildir**
> Vektör veritabanınızda saklanan embedding dizileri, KVKK ve GDPR kapsamında ham müşteri metni ile **birebir aynı gizlilik derecesine** sahiptir. Vektör deposu açıkta bırakıldığında veya kiracılar arası izolasyon çöktüğünde, bir saldırgan vektörleri indirip saniyeler içinde orijinal müşteri kayıtlarına dönüştürebilir.

### 4.2. Diferansiyel Mahremiyet (Differential Privacy — DP-SGD)

Bir dil modelinin eğitim verisindeki nadir veya sıra dışı bir kişisel veriyi ezberlemesini engellemenin tek matematiksel kanıtı **Diferansiyel Mahremiyettir** (Dwork, 2006).

Bir $\mathcal{M}$ öğrenme algoritması, birbirine komşu olan (yalnızca tek bir müşterinin kaydı kadar farklılaşan, $\|D - D'\|_1 = 1$) herhangi iki veri seti üzerinde çalıştırıldığında çıktı dağılımı şu eşitsizliği sağlıyorsa **$(\epsilon, \delta)$-Diferansiyel Özeldir**:

$$
\mathcal{P}(\mathcal{M}(D) \in S) \le e^\epsilon \cdot \mathcal{P}(\mathcal{M}(D') \in S) + \delta
$$

Şöyle okuyun: $\epsilon$ (*epsilon*), mahremiyet kaybı bütçesidir; $\epsilon \to 0$ yaklaştıkça modelin o müşterinin verisi varken veya yokken üreteceği ağırlık dağılımları birbirinden ayırt edilemez hâle gelir. $\delta$ (*delta*), bu garantinin bozulma olasılığıdır ve veri seti büyüklüğünün çok altında ($\delta \ll 1/|D|$) seçilmelidir.

Derin öğrenmede bu garanti **DP-SGD** (Abadi ve ark., 2016) algoritmasıyla sağlanır:
1. **Gradyan Kırpma (Per-Sample Gradient Clipping):** Her örneğin gradyan normu sabit bir $C$ eşiğiyle sınırlandırılır:
   $$\bar{g}_t(x_i) = \frac{g_t(x_i)}{\max\left(1, \frac{\|g_t(x_i)\|_2}{C}\right)}$$
2. **Gaussian Gürültü Enjeksiyonu:** Kırpılmış gradyanların ortalamasına calibrated Gaussian gürültüsü eklenir:
   $$\tilde{g}_t = \frac{1}{B} \sum_{i=1}^B \bar{g}_t(x_i) + \mathcal{N}\left(0, \sigma^2 C^2 \mathbf{I}\right)$$

Bu gürültü, tek bir müşterinin ağırlıklar üzerindeki bireysel izini tamamen silerken kümenin genel istatistiksel örüntüsünün öğrenilmesini mümkün kılar.

---

## 5. Kişisel veriyi bulmak: Kalıplar, sağlama toplamları ve Türkçe NER

Kişisel veriyi (PII) koruyabilmek için önce onu yüksek doğrulukla tespit etmek gerekir. PII iki farklı formda karşımıza çıkar:

1. **Yapılandırılmış Tanımlayıcılar:** Sabit biçime sahip veriler (TCKN, IBAN, kredi kartı, telefon). Regex adayları bulur; **sağlama toplamı (checksum)** ise yanlış alarmları eler.
2. **Bağlamsal Tanımlayıcılar:** Serbest metindeki isimler, adresler, hastalık bilgileri. Bunlar **NER (Named Entity Recognition)** modelleri gerektirir.

### 5.1. TC Kimlik Numarası Algoritması ve Sağlama Toplamı Matematiği

TC Kimlik numarası 11 haneden oluşur; ilk hanesi sıfır olamaz. Son iki hane matematiksel kontrol haneleridir:

$$
d_{10} = \Big( 7\,(d_1 + d_3 + d_5 + d_7 + d_9) - (d_2 + d_4 + d_6 + d_8) \Big) \bmod 10
$$

$$
d_{11} = \Big( \sum_{i=1}^{10} d_i \Big) \bmod 10
$$

Şöyle okuyun: $d_i$, soldan $i$. basamaktaki rakamdır. 10. basamak, tek konumdaki rakamların toplamının 7 katından çift konumdaki rakamların toplamının çıkarılıp 10'a göre modunun alınmasıyla elde edilir. 11. basamak ise ilk 10 basamağın toplamının son basamağıdır.

### Adım Adım Sayısal Yürüyüş

İlk 9 hanesi `1 2 3 4 5 6 7 8 9` olan sentetik bir numara üzerinden hesaplayalım:

```text
Haneler:           d1 d2 d3 d4 d5 d6 d7 d8 d9
                   1  2  3  4  5  6  7  8  9

1. Adım: Konum toplamları
  Tek konumlar:   d1 + d3 + d5 + d7 + d9 = 1 + 3 + 5 + 7 + 9 = 25
  Çift konumlar:  d2 + d4 + d6 + d8      = 2 + 4 + 6 + 8     = 20

2. Adım: 10. hanenin hesabı
  Ara değer = 7 × 25 - 20 = 175 - 20 = 155
  d10 = 155 mod 10 = 5

3. Adım: 11. hanenin hesabı
  Toplam = (1+2+3+4+5+6+7+8+9) + d10 = 45 + 5 = 50
  d11 = 50 mod 10 = 0

Sonuç: Sentetik geçerli numara = 12345678950 (Geçerli biçim)
Tek bir haneyi boz:              12345678951 (d11=1 != 0 -> Geçersiz!)
```

### Sağlama Toplamının Gücü: Yanlış Pozitiflerin %99 Elenmesi

Yalnızca `[1-9]\d{10}` regex'i kullanan bir sistem, 11 haneli her sipariş veya barkod numarasını TCKN sanarak maskeler. Oysa rastgele 11 haneli bir dizide $d_{10}$'un tutma ihtimali $1/10$, $d_{11}$'in de tutma ihtimali $1/10$'dur:

$$
P(\text{rastgele dizi geçer}) = \frac{1}{10} \times \frac{1}{10} = 0{,}01
$$

Sağlama toplamı, işlemci maliyeti sıfıra yakın bir kontrolle **yanlış pozitiflerin %99'unu anında eler**.

```python
import random
import re

def tckn_valid(s: str) -> bool:
    """TC Kimlik numarası biçim doğrulaması."""
    if not re.fullmatch(r"[1-9]\d{10}", s):
        return False
    d = [int(c) for c in s]
    # Python'da negatif mod pozitiftir; diğer dillerde ((x % 10) + 10) % 10 gerekir
    d10 = (7 * (d[0] + d[2] + d[4] + d[6] + d[8]) - (d[1] + d[3] + d[5] + d[7])) % 10
    d11 = sum(d[:10]) % 10
    return d[9] == d10 and d[10] == d11

# Doğrulama
print("12345678950 geçerli mi?:", tckn_valid("12345678950"))
print("12345678951 geçerli mi?:", tckn_valid("12345678951"))

# 1.000.000 rastgele 11 haneli sayı testi
random.seed(42)
N = 1_000_000
hits = sum(tckn_valid(str(random.randint(10**10, 10**11 - 1))) for _ in range(N))
print(f"Rastgele 11 haneli dizilerin geçiş oranı: %{hits / N * 100:.2f}")
```

Konsol çıktısı:

```text
12345678950 geçerli mi?: True
12345678951 geçerli mi?: False
Rastgele 11 haneli dizilerin geçiş oranı: %1.00
```

### 5.2. Türkçe NER Zorlukları

Sağlama toplamı isimleri ve adresleri bulamaz. Burada devreye giren NER modelleri Türkçenin sondan eklemeli morfolojisiyle sınanır:
- İsim köklerine gelen çekim ekleri ("Ayşe'nin", "Mehmet'e", "Kadıköy'deki") kesme işareti olmadan yazıldığında kural tabanlı sözlükler çöker.
- Yaygın isimler aynı zamanda cins isimdir ("Deniz", "Kaya", "Umut", "Gül").
- Çözüm: BERT tabanlı Türkçe NER modellerinin (örn. `savasy/bert-base-turkish-ner`) çalışma zamanında regex katmanıyla zincirlenmesidir.

---

## 6. Maskeleme, takma adlandırma, anonimleştirme

Bu üç kavramın hukuki (GDPR / KVKK) ve teknik sonuçları taban tabana zıttır:

<!-- fig:p3_spectrum -->
<svg viewBox="0 0 720 328" role="img" aria-label="Daha az korumadan daha fazlasına dört sütun. Asıl değer: TCKN 12345678950; erişimi olan herkes okur; kişisel veridir. Maskelenmiş: 123******50; asıl değer saklanır; kişisel veridir. Takma adlı: HMAC ile üretilen ID; yalnızca anahtarla döner; hâlâ kişisel veridir. Anonim: hiçbir makul yolla geri döndürülemez; hukukun dışındadır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p3s-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Maskelemeden Anonimleştirmeye Koruma Yelpazesi</text>
<line x1="112" y1="46" x2="700" y2="46" marker-end="url(#p3s-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="704" y="40" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">daha az tanınabilir →</text>
<text x="16" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">Örnek</text>
<text x="16" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">Geri Döner mi?</text>
<text x="16" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">Hukuki Durum</text>
<rect x="112" y="58" width="140" height="182" rx="8" style="fill:var(--c-danger);fill-opacity:.07;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="182" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Asıl Değer</text>
<text x="182" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">12345678950</text>
<text x="182" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">erişimi olan</text>
<text x="182" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">herkes okur</text>
<text x="182" y="202" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:600">Kişisel Veri</text>
<rect x="260" y="58" width="140" height="182" rx="8" style="fill:var(--c-warn);fill-opacity:.07;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="330" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Maskelenmiş</text>
<text x="330" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">123******50</text>
<text x="330" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">asıl değer saklı;</text>
<text x="330" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yalnızca UI filtreli</text>
<text x="330" y="202" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">Kişisel Veri</text>
<rect x="408" y="58" width="140" height="182" rx="8" style="fill:var(--c-warn);fill-opacity:.07;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="478" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Takma Adlı</text>
<text x="478" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">ID_f0a155…</text>
<text x="478" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">ayrı tutulan</text>
<text x="478" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">anahtarla döner</text>
<text x="478" y="202" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">Kişisel Veri</text>
<text x="478" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">(GDPR Ger. 26)</text>
<rect x="556" y="58" width="140" height="182" rx="8" style="fill:var(--c-success);fill-opacity:.07;stroke:var(--c-success);stroke-width:1.3"/>
<text x="626" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Anonim</text>
<text x="626" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">bir müşteri</text>
<text x="626" y="150" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">makul hiçbir</text>
<text x="626" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yolla geri dönmez</text>
<text x="626" y="202" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:600">Hukukun Dışında</text>
<text x="626" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">(KVKK Yönetmeliği)</text>
<text x="16" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Yarı tanımlayıcılar (ilçe, işlem saati, tutar) temizlenmedikçe hiçbir veri hukuken anonim kabul edilemez.</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="256" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="275" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 KORUMA TAYFI İLKESİ: MASKELENMİŞ VE TAKMA ADLI VERİ HÂLÂ KİŞİSEL VERİDİR</text>
<text x="28" y="293" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Yıldızlama ve HMAC şifreleme veriyi anonim kılmaz; yarı tanımlayıcılar ve yeniden birleştirme riski devam eder.</text>
<text x="28" y="309" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Hukuken anonimlik ancak Vec2Text ters çevirme saldırılarına dirençli ve diferansiyel gizlilik (DP-SGD) garantili dönüşümlerle mümkündür.</text>
</svg>

- **Maskeleme:** Bilgiyi ekranda gizler (`123******50`), veri tabanında aslı durur. Hukuken kişisel veridir.
- **Takma Adlandırma (Pseudonymization):** Değeri ayrı bir kasada tutulan kriptografik bir anahtarla değiştirir:
  $$\text{Pseudonym} = \text{HMAC}_K(\text{TCKN})$$
  $K$ anahtarına sahip olan orijinal veriye dönebilir. GDPR Madde 4(5) ve Gerekçe 26 uyarınca **takma adlı veri hâlâ kişisel veridir**; çünkü anahtar mevcuttur.
- **Anonimleştirme:** Verinin makul hiçbir yöntemle, üçüncü taraf veri setleriyle birleştirilse dahi bir kişiye bağlanamamasıdır. Ancak o zaman KVKK ve GDPR kapsamından çıkar. Sweeney (2000), 5 haneli posta kodu, cinsiyet ve doğum tarihi üçlüsünün ABD nüfusunun %87'sini tekil olarak tanımladığını kanıtlamıştır. Dolayısıyla isimleri silmek anonimleştirme değildir; **yarı tanımlayıcıların (quasi-identifiers)** da genelletirilmesi şarttır.

---

## 7. Saklama ve silme: Unutulan kopyaların matematiği ve kripto-imha

Giriş hikâyesindeki başarısızlığın temel nedeni bir envanter kusurudur. Bir müşteri verisinin sistemde birbirinden bağımsız $N$ ayrı kopyası bulunduğunu varsayalım. Her kopyanın silme işlemi sırasında gözden kaçma, atlanma veya yetki hatasıyla silinememe olasılığı $q$ olsun. Silme işleminin **tamamen eksiksiz** tamamlanma olasılığı şudur:

$$
P_{\text{eksiksiz}} = (1 - q)^{N}
$$

Şöyle okuyun: $1 - q$, tek bir depodaki silme işleminin başarı oranıdır. $N$ depoda birden temizlik sağlanması gerektiğinden bu bağımsız olasılık $N$ defa birbiriyle çarpılır.

### Adım Adım Sayısal Yürüyüş

Depo başına %5 başarısızlık ihtimaliyle ($q = 0,05$) kopyalar arttıkça eksiksiz silme ihtimalinin çöküşünü izleyelim:

```text
Parametreler:
  q = 0,05   (Depo başına %5 gözden kaçırma payı)

1. Adım: Yalnızca CRM (N=1)
  P_eksiksiz = (1 - 0,05)^1 = 0,9500  → %95,0 başarı

2. Adım: CRM + Vektör Dizini + Loglar (N=3)
  P_eksiksiz = (0,95)^3 = 0,95 × 0,95 × 0,95
             = 0,9025 × 0,95 = 0,8574  → %85,7 başarı

3. Adım: Dokuz Yaşam Alanının Tümü (N=9)
  P_eksiksiz = (0,95)^9 ≈ 0,6302       → %63,0 başarı!
```

Dokuz kopyalı bir mimaride her üç silme talebinden biri (%37'si) geride yetim bir veri parçası bırakır!

### Çözüm: Kriptografik İmha (Crypto-Shredding)

$N$ depoyu tek tek dolaşıp silmek yerine mimariyi **kriptografik imha** (*crypto-shredding*) deseniyle kurarız:
1. Her müşterinin verisi, o müşteriye özel tekil bir anahtarla ($K_{\text{user}}$) şifrelenerek depolara dağıtılır.
2. Silme talebi geldiğinde 9 depoya dokunulmaz; yalnızca anahtar yönetim servisindeki (KMS) $K_{\text{user}}$ anahtarı kalıcı olarak imha edilir (`shred`).
3. Anahtar yok edildiği anda CRM'deki, vektör veritabanındaki, yedeklerdeki ve önbellekteki tüm kopyalar anında çözülemez anlamsız gürültüye dönüşür.

---

## 8. Mühendisliğin iki gözü: Kurulum zamanı ve çalışma zamanı

**Kurulum Zamanı (Build Time):**
- Veri zehirleme ve kalıcı ezberleme burada gerçekleşir.
- Kontroller: SHA-256 köken manifestoları, değiştirilemez veri sürümleri, veri setlerinden PII'ın eğitim öncesi regex+NER ile temizlenmesi ve $(\epsilon, \delta)$-DP-SGD kullanımıdır.

**Çalışma Zamanı (Run Time):**
- Verinin kopyalanarak dokuz eve dağılması burada gerçekleşir.
- Kontroller: Prompt'lara minimum PII aktarımı, telemetri loglarında içerik kaydının varsayılan kapalı olması, önbelleklerde kısa TTL (Time-To-Live) ve HMAC tabanlı takma adlandırma kalkanıdır.

---

## 9. Pratik çıktı: Envanter, akış ve kontrol listesi

### 9.1. PII İşleme Akışı

<!-- fig:p3_flow -->
<svg viewBox="0 0 720 426" role="img" aria-label="PII işleme akışı diyagramı. Gelen şikâyet iki kola ayrılır: Regex artı checksum yapılandırılmış veriyi, Türkçe NER ise bağlamsal veriyi bulur. İkisi Politika adımında birleşir. Politika hedefe göre dağıtır: retrieval dizinine takma ad artı ACL, prompt'a asgari veri, loglara sıfır içerik, eğitime anonim veri gider. Takma adlandırma ayrı tutulan HMAC anahtarını kullanır. Silme talebi her yerde takma ada göre çalışır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p3f-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p3f-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="p3f-r" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">PII İşleme Akışı: Bir Kez Tespit Et, Hedefe Göre Dönüştür</text>
<rect x="16" y="120" width="112" height="64" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="72.0" y="140.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Şikâyet</text>
<text x="72.0" y="156.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">ham metin</text>
<text x="72.0" y="172.5" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px">TB4: dış kaynak</text>
<rect x="158" y="76" width="150" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="233.0" y="92.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Regex + Checksum</text>
<text x="233.0" y="108.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">TCKN, IBAN (mod 97)</text>
<text x="233.0" y="122.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Luhn (kredi kartı)</text>
<rect x="158" y="172" width="150" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="233.0" y="188.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Türkçe NER Modeli</text>
<text x="233.0" y="204.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">isim, adres, kurum</text>
<text x="233.0" y="220.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">çekim eklerine duyarlı</text>
<path d="M128 152 H142 V104 H156" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M128 152 H142 V200 H156" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="338" y="112" width="132" height="80" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="404.0" y="136.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Politika Motoru</text>
<text x="404.0" y="154.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">hedefe göre kural:</text>
<text x="404.0" y="170.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10.5px">maskele / takma ad</text>
<path d="M308 104 H322 V140 H336" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M308 200 H322 V164 H336" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="510" y="52" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="522" y="73" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Retrieval Dizini</text>
<text x="522" y="90" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kimlikler takma adla + ACL</text>
<path d="M470 152 C 490 152, 490 77, 508 77" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="510" y="116" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="522" y="137" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Modele Giden Prompt</text>
<text x="522" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">görevin gerektirdiği asgari PII</text>
<path d="M470 152 C 490 152, 490 141, 508 141" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="510" y="180" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="522" y="201" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Log ve Telemetri</text>
<text x="522" y="218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">içerik yok; tuzlu token hash'i</text>
<path d="M470 152 C 490 152, 490 205, 508 205" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="510" y="244" width="194" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="522" y="265" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Eğitim Seti</text>
<text x="522" y="282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yalnızca sentetik / DP-SGD</text>
<path d="M470 152 C 490 152, 490 269, 508 269" marker-end="url(#p3f-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<path d="M148 277 V321 A85.0 7 0 0 0 318 321 V277" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="233.0" cy="277" rx="85.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="233.0" y="300.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Anahtar Kasası</text>
<text x="233.0" y="315.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">KMS / HMAC Anahtarı</text>
<path d="M404 192 V299 H320" marker-end="url(#p3f-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="410" y="250" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">takma adlar</text>
<rect x="510" y="312" width="194" height="30" rx="6" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:5 4"/>
<text x="607" y="331" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Kripto-İmha: Anahtarı sil, her yerde silinsin</text>
<path d="M318 312 H470 V327 H508" marker-end="url(#p3f-r)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5;stroke-dasharray:5 4"/>
<!-- Bottom Educational Banner -->
<rect x="16" y="354" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="373" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 MİMARİ ÇÖZÜM: KRİPTO-İMHA İLE ANINDA VE KESİN SİLME (CRYPTO-SHREDDING)</text>
<text x="28" y="391" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Dokuz ayrı depoda silme sorguları koşturmak yerine, her veri konusu için benzersiz bir anahtar kullanılır.</text>
<text x="28" y="407" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Silme talebinde yalnızca o şahsın anahtarı imha edilir; tüm vektörler ve loglar anında geri döndürülemez çöp baytlara dönüşür.</text>
</svg>

### 9.2. Veri Yaşam Döngüsü Kontrol Listesi

1. **Dokuz Yaşam Alanı Envanteri:** Verinin kopyalandığı tüm depolar (vektör, önbellek, log, yedek) taranmış ve atanmış bir veri sahibine bağlanmıştır.
2. **Kriptografik Köken Doğrulaması:** Eğitim ve RAG belgeleri toplanma anında SHA-256 ile özetlenmiş, değişmez sürüm etiketi almıştır.
3. **Fine-Tuning Öncesi PII Sıfırlama:** Model ağırlıklarına ham kişisel veri işlenmez; eğitim verisi diferansiyel mahremiyet süzgecinden geçirilir.
4. **Çift Aşamalı PII Tespiti:** Yapılandırılmış veriler regex ve sağlama toplamıyla, serbest metinler Türkçe NER ile tespit edilir.
5. **Embedding Güvenlik Sınıflandırması:** Vektör dizinleri ve önbellekler, düz metin PII ile aynı şifreleme ve ACL denetimine tabidir.
6. **Kriptografik İmha (Crypto-Shredding):** Kullanıcı verileri kullanıcıya özel anahtarla şifrelenir; silme talebinde anahtar yok edilerek tüm kopyalar tek hamlede imha edilir.
7. **Periyodik Silme Doğrulama Testi:** Her hafta sentetik bir kayıt silinir ve dokuz deponun tamamında sıfır veri kaldığı otomatik testle raporlanır.

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

- Neden bir CRM kaydını silmek, bir yapay zekâ asistanında o verinin tamamen unutulduğunu garanti etmez?
- Vec2Text saldırıları karşısında embedding vektörleri neden düz metin gibi korunmak zorundadır?
- Koh-Liang etki fonksiyonu ($I_{\text{up,loss}}$), temiz etiketli veri zehirlemenin gradyan mekanizmasını nasıl açıklar?
- TC Kimlik numarası sağlama toplamı, yanlış alarmları hangi olasılık hesabıyla %99 oranında eler?
- Bağımsız dokuz kopyalı bir mimaride eksiksiz silme olasılığı ($(1-q)^N$) neden çöker ve kripto-imha bu sorunu nasıl çözer?

---

## Bütün hikâye altı satırda

- AI sistemlerinde veri çoğalır: Tek bir müşteri şikâyeti, her biri farklı saklama süresine sahip dokuz ayrı depoda yaşar.
- Embedding vektörleri tersinirdir; Vec2Text saldırıları nedeniyle vektörler düz metinle aynı gizlilik sınıfında korunmalıdır.
- Kişisel veri ağırlıklardan seçilerek silinemez; bu yüzden fine-tuning setleri eğitime girmeden önce temizlenmelidir.
- Temiz etiketli zehirleme etki fonksiyonları ile çalışır; 250 adet zehirli belge 13B'lik bir modele arka kapı koymaya yetti.
- Sağlama toplamı TCKN için yanlış pozitifleri %99 eler; serbest metin ise Türkçede değerlendirilmiş NER gerektirir.
- Silmenin eksiksizliği $(1-q)^N$ ile hızla çöker; silmeyi prosedürel aramadan tek anahtarlı kripto-imhaya dönüştürün.

---

## Terimler sözlüğü

- **Köken (Data Provenance):** Verinin kaynağını, geçirdiği dönüşümleri ve doğrulama imzalarını gösteren denetlenebilir iz zinciri.
- **Etki Fonksiyonu (Influence Function):** Bir eğitim örneğinin model ağırlıkları ve hedef test kaybı üzerindeki gradyan bazlı türevi ($I_{\text{up,loss}}$).
- **Vec2Text (Embedding Inversion):** Sayısal embedding vektörlerini optimize ederek orijinal metni harfi harfine geri çeken saldırı yöntemi.
- **Diferansiyel Mahremiyet (DP-SGD):** Model ağırlıklarının tek bir eğitim örneğinin varlığına veya yokluğuna aşırı duyarlı olmamasını sağlayan gürültülü gradyan algoritması.
- **Sağlama Toplamı (Checksum):** Tanımlayıcının kendi hanelerinden modüler aritmetikle hesaplanan ve rastgele dizileri eleyen doğrulama basamakları.
- **Yarı Tanımlayıcı (Quasi-Identifier):** Tek başına kişiyi ele vermeyen ancak diğer veri setleriyle birleştiğinde tekilleştiren öznitelikler (posta kodu, yaş, cinsiyet).
- **Takma Adlandırma (Pseudonymization):** Veriyi ayrı bir kasada tutulan anahtarla token'a çevirme işlemi; anahtar mevcut olduğu için hukuken hâlâ kişisel veridir.
- **Kriptografik İmha (Crypto-Shredding):** Verinin şifrelendiği anahtarı silerek tüm türetilmiş kopyaları ve yedekleri anında okunamaz kılma yöntemi.
- **Semantik Önbellek (Semantic Cache):** LLM çıkarım maliyetini düşürmek için benzer prompt ve yanıtları saklayan önbellek katmanı.
- **Model Kartı (Model Card):** Modelin hangi veri seti sürümleriyle eğitildiğini ve güvenlik sınırlarını belgeleyen resmi kayıt.

---

## Daha derine inmek için

- [K1] N. Carlini ve ark., [*Poisoning Web-Scale Training Datasets is Practical*](https://arxiv.org/abs/2302.10149) (2023): URL bölünmüş görüş zehirlemesi.
- [K2] Anthropic, UK AI Security Institute ve Alan Turing Institute, [*A small number of samples can poison LLMs of any size*](https://www.anthropic.com/research/small-samples-poison) (Ekim 2025): 250 belgeyle model arka kapısı.
- [K3] P. W. Koh ve P. Liang, [*Understanding Black-box Predictions via Influence Functions*](https://arxiv.org/abs/1703.04730) (ICML 2017): Gradyan ve ters Hessian bazlı etki formülasyonu.
- [K4] J. X. Morris ve ark., [*Text Embeddings Reveal (Almost) As Much As Text*](https://arxiv.org/abs/2310.06816) (EMNLP 2023): Vec2Text embedding tersine mühendisliği.
- [K5] M. Abadi ve ark., [*Deep Learning with Differential Privacy*](https://arxiv.org/abs/1607.00133) (ACM CCS 2016): DP-SGD algoritması ve gradyan kırpma.
- [K6] W. Zou ve ark., [*PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation*](https://arxiv.org/abs/2402.07867) (2024): Bilgi tabanına hedefli zehir enjeksiyonu.
- [K7] L. Sweeney, *Simple Demographics Often Identify People Uniquely*, Carnegie Mellon University Data Privacy Working Paper 3 (2000).
- [K8] ISO/IEC, [*27701:2025 Privacy Information Management System*](https://www.iso.org/standard/27701) ve NIST, [*AI 600-1*](https://doi.org/10.6028/NIST.AI.600-1) (Veri Mahremiyeti Profili).
- [K9] Bu seriden: [Secure AI by Design (4): Modelden Bağımlılığa — AI Tedarik Zincirini Güvenceye Almak](post.html?slug=ai-tedarik-zinciri-guvenligi).
