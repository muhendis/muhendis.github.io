Masanızın altına yerleştirilmiş, 1.600 vatlık şebeke gücünü çekerken kapalı devre sıvı soğutma radyatörleriyle fısıltı sessizliğinde çalışan mat siyah bir yapay zekâ kulesi hayal edin. Masanızın üzerinde ise üreticinin dört sayfalık resmi dokümanı duruyor: *NVIDIA DGX Station Veri Sayfası* (`5543600-web`). Kapağın üzerindeki manşet sayılar göz alıcı bir puntoda parıldıyor: **20 PFLOPS FP4 AI Hesaplama Gücü**, **748 GB Birleşik Tutarlı Bellek**, **Masaüstünde 1 Trilyon Parametreli Model Çıkarımı** ve **800 Gb/s ConnectX-8 SuperNIC**.

Ekip arkadaşınız broşürü incelerken heyecanla yanınıza geliyor: *"Veri merkezindeki koca bir sunucu kabinini tek bir kasanın içine sıkıştırmışlar! Bulut kuyruklarında sıra beklemeden, 1 trilyon parametreli modelleri masamızın başında eğitebilir ve otonom akıl yürüten ajanlarımızı çalıştırabiliriz!"*

Peki bu makine kimin içindir? Bir ev veya oyun bilgisayarı mıdır, yoksa devasa bir veri merkezi sunucusu mu? Cevap: Ne biri, ne diğeri. NVIDIA bu kategoriye **"Deskside Supercomputer" (Masa Yanı Süper Bilgisayarı)** adını veriyor. İçindeki silikon %100 saf veri merkezi donanımıdır; ancak dışındaki kasa, tek fazlı güç (1.600W) ve sıvı soğutma tasarımı, özel bir sunucu odası olmayan bir Ar-Ge ofisinde masanın altına konulabilmesi için optimize edilmiştir.

Ancak bir sistem mühendisi olarak bir donanım veri sayfasına baktığınızda, onu garanti edilmiş bir performans listesi olarak değil, **fiziksel bir sınır değer problemi** olarak okursunuz: Bu 20 PFLOPS sayısının altına hangi varsayımlar gömülmüştür? 748 GB gerçekten tek parça, homojen bir bellek havuzu mudur; yoksa iki yarısı arasında 18 katlık bir bant genişliği uçurumu mu saklanmaktadır? Ve 1 trilyon parametreli bir model bu sınırın üzerine oturduğunda, gerçekten insan düşüncesi hızında token üretebilir mi; yoksa saniyede birkaç token'lık arka plan doğrulama hızına mı mahkûmdur?

**Bu yazının temel fikri:** Bir GPU datasheet'i donanımın üretim ortamında ne yapacağını değil, silikonun ulaşabileceği kuramsal tavanı ve performansın serbest düşüşe geçtiği mikromimari uçurumları tanımlar. Gerçek donanım okuryazarlığı; parlatılmış pazarlama metriklerinin arkasına geçerek gerçek aritmetik yoğunluk eşiklerini, bellek kısıtlı token gecikmelerini ve fiziksel termal sınırları hesaplayabilme disiplinidir.

> **Serideki yerimiz:** Bu makale, beş bölümlük *Yapay Zekâ Altyapısı* serimizi (SM mimarisi, iş yükleri, küme ağları, arıza teşhisi ve kurumsal platform tasarımı) taçlandıran özel pratik rehberdir (6. Bölüm). Kurduğumuz kuramsal modelleri doğrudan sahada çalışan bir üretim sistemine uyguluyoruz: NVIDIA'nın masaüstü amiral gemisi, GB300 Grace Blackwell Ultra Desktop Superchip ile güçlendirilmiş DGX Station'ın resmi veri sayfasını satır satır parçalıyoruz.

> **Bu yazının sonunda**
>
> - Herhangi bir GPU datasheet'ini üç temel mühendislik katmanına (kuramsal tepe değerler, mikromimari kapılar ve fiziksel sistem sınırları) ayırabileceksiniz.
> - Bir donanımın ev ortamı, veri merkezi ve Ar-Ge ofisi arasındaki fiziksel ve operasyonel sınırlarını net biçimde ayırt edebileceksiniz.
> - 2:4 yapısal seyreklik (sparsity) ile yoğun (dense) Tensor Core yürütmesini ayrıştırarak manşet sayıların ardındaki gerçek matematiksel kapasiteyi hesaplayabileceksiniz.
> - Heterojen birleşik bellek mimarilerinde (NVLink-C2C üzerinden HBM3e vs. LPDDR5X) yaşanan mimari bant genişliği uçurumunu sayısal olarak kanıtlayabileceksiniz.
> - Bellek bant genişliği kısıtlı token üretim formüllerini işleterek 1 trilyon parametreli bir modelin bu kasada interaktif hızda çalışıp çalışamayacağını matematiksel olarak gösterebileceksiniz.
> - Tek fazlı ofis elektriğinin (1.600W) ve küme ölçekleme ağının (800 Gb/s ConnectX-8) masaüstü süper bilgisayarların fiziksel sınırlarını nasıl çizdiğini gerekçelendirebileceksiniz.

**Bu yazıda**

- [1. Bir datasheet'in anatomisi: Tepe değerler vs. fiziksel sınırlar](#1-bir-datasheetin-anatomisi-tepe-değerler-vs-fiziksel-sınırlar)
  - [1.1. Teknik özelliklerin üç katmanı](#11-teknik-özelliklerin-üç-katmanı)
  - [1.2. Bir kısıt haritası olarak datasheet](#12-bir-kısıt-haritası-olarak-datasheet)
- [2. Paradigma değişimi: 4 ayrık GPU'dan tekil süperçipe](#2-paradigma-değişimi-4-ayrık-gpudan-tekil-süperçipe)
  - [2.1. DGX Station'ın evrimi (Volta'dan Blackwell'e)](#21-dgx-stationın-evrimi-voltadan-blackwelle)
  - [2.2. NVLink-C2C neden ayrık GPU topolojilerinin yerini aldı?](#22-nvlink-c2c-neden-ayrık-gpu-topolojilerinin-yerini-aldı)
- [3. Hesaplama gücünün perde arkası: 20 PFLOPS ve seyreklik illüzyonu](#3-hesaplama-gücünün-perde-arkası-20-pflops-ve-seyreklik-illüzyonu)
  - [3.1. Hassasiyet formatlarına göre tepe Tensor FLOP değerleri](#31-hassasiyet-formatlarına-göre-tepe-tensor-flop-değerleri)
  - [3.2. Yapısal seyreklik (2:4) vs. yoğun matematiksel gerçeklik](#32-yapısal-seyreklik-24-vs-yoğun-matematiksel-gerçeklik)
- [4. 748 GB birleşik bellek havuzu ve bant genişliği uçurumu](#4-748-gb-birleşik-bellek-havuzu-ve-bant-genişliği-uçurumu)
  - [4.1. Fiziksel ayrım: HBM3e vs. LPDDR5X](#41-fiziksel-ayrım-hbm3e-vs-lpddr5x)
  - [4.2. NVLink-C2C köprüsü ve 18 katlık bant genişliği uçurumu](#42-nvlink-c2c-köprüsü-ve-18-katlık-bant-genişliği-uçurumu)
- [5. 1 Trilyon parametre iddiası: Prefill ve decode matematiği](#5-1-trilyon-parametre-iddiası-prefill-ve-decode-matematiği)
  - [5.1. Model ayak izi ve KV cache bellek tahsisi](#51-model-ayak-izi-ve-kv-cache-bellek-tahsisi)
  - [5.2. Adım adım gecikme yürüyüşü (1T vs. 405B vs. 70B)](#52-adım-adım-gecikme-yürüyüşü-1t-vs-405b-vs-70b)
- [6. Roofline modeli ve operasyonel yoğunluk eşiği](#6-roofline-modeli-ve-operasyonel-yoğunluk-eşiği)
  - [6.1. Denge noktasını hesaplamak: Hesaplama vs. bant genişliği](#61-denge-noktasını-hesaplamak-hesaplama-vs-bant-genişliği)
  - [6.2. GEMM tepeye ulaşırken GEMV neden aç kalır?](#62-gemm-tepeye-ulaşırken-gemv-neden-aç-kalır)
- [7. Masaüstü gerçekliği: Ev mi, veri merkezi mi? 1600W güç ve ConnectX-8](#7-masaüstü-gerçekliği-ev-mi-veri-merkezi-mi-1600w-güç-ve-connectx-8)
  - [7.1. Üç dünyanın ayrımı: Ev mi, veri merkezi mi, Ar-Ge masası mı?](#71-üç-dünyanın-ayrımı-ev-mi-veri-merkezi-mi-ar-ge-masası-mı)
  - [7.2. 800 Gb/s SuperNIC neden bir masaüstü kasasında yer alır?](#72-800-gbs-supernic-neden-bir-masaüstü-kasasında-yer-alır)
- [8. Karşılaştırmalı evrim matrisi: V100'den GB300'e](#8-karşılaştırmalı-evrim-matrisi-v100den-gb300e)
  - [8.1. Mimarilerin nesilsel dökümü](#81-mimarilerin-nesilsel-dökümü)
  - [8.2. Donanım mimarları için temel çıkarımlar](#82-donanım-mimarları-için-temel-çıkarımlar)
- [9. Öğrendiklerimizi sınayalım](#9-öğrendiklerimizi-sınayalım)
  - [9.1. Soru 1: Yoğun vs. seyrek hesaplama](#91-soru-1-yoğun-vs-seyrek-hesaplama)
  - [9.2. Soru 2: Bellek uçurumu hesabı](#92-soru-2-bellek-uçurumu-hesabı)
  - [9.3. Soru 3: Doğru model ölçeğini seçmek](#93-soru-3-doğru-model-ölçeğini-seçmek)
- [Okuyucu artık hangi sorulara kendi başına yanıt verebilir?](#okuyucu-artık-hangi-sorulara-kendi-başına-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. Bir datasheet'in anatomisi: Tepe değerler vs. fiziksel sınırlar

Donanım veri sayfaları iki farklı amaca aynı anda hizmet eder: Üst düzey satın alma komitelerine çarpıcı manşetler sunmak isteyen satış organizasyonu ve elektriksel/mikromimari sınırları kayıt altına alan donanım mühendisliği ekibi. Bir veri sayfasından mühendislik hakikatini süzebilmek için her metriği üç aşamalı bir süzgeçten geçirmek gerekir.

### 1.1. Teknik özelliklerin üç katmanı

Herhangi bir GPU özelliğini incelerken sayıları şu üç katmana ayırarak okumalısınız:

1. **Üretici Tepe Değerleri (Vendor Peak Specs - Manşet):** Aritmetik işlem hatlarının %100'ünün, en düşük hassasiyet formatında, sıfır register taşması (spill), sıfır bellek beklemesi ve azami donanım ivmelendirmesiyle (örneğin 2:4 yapısal seyreklik) her saat döngüsünde ateşlendiği varsayılarak hesaplanan sayılardır. Örnek: *"20 PFLOPS FP4 AI Compute"*.
2. **Mikromimari Kapılar (Microarchitectural Gating - Silikon Gerçeği):** Veri erişilebilirliği, yazmaç tahsisi ve sayısal format kısıtlarının belirlediği gerçek işlem hacmidir. Algoritmanız BF16 formatında yoğun (dense) matris çarpımları yapıyorsa, o 20 PFLOPS bir anda **2,5 PFLOPS** seviyesine iner.
3. **Sistem ve Termal Sınırlar (Physical Limits - Fiziksel Kafes):** Şebeke prizi, güç kaynağı verimliliği, ısı dağıtımı ve ara bağlantı bant genişliği tarafından dayatılan sınırlardır. Sürekli olarak 1.200W güç çekebilen bir süperçip; ofis ortam sıcaklığı yükseldiğinde ya da sıvı soğutma radyatörleri doyuma ulaştığında saat frekanslarını kısmak (thermal throttling) zorundadır.

GB300'ün manşet sayısını bu üç süzgeçten geçirdiğimizde tablo şöyle değişir:

<svg viewBox="0 0 720 362" role="img" aria-label="20 PFLOPS manşetine uygulanan üç süzgeç. Birinci katman, manşet: 20 PFLOPS; FP4, 2:4 seyreklik ve her döngüde her ALU&#x27;nun çalıştığını varsayar. İkinci katman, mikromimari kapılar: seyreklik olmadan FP4 15 PFLOPS&#x27;a, yoğun FP8 5&#x27;e, eğitimin tavanı olan yoğun BF16 2,5&#x27;e iner; tek kullanıcılı decode bellek beklediği için kapasitenin %0,6&#x27;sından azını kullanır. Üçüncü katman, fiziksel kafes: 1.600 W tek faz güç, ısı ve saat kısma, 800 Gb/s ağ." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir manşet sayısı, üç süzgeç: 20 PFLOPS nereye iner?</text>
<rect x="16" y="40" width="688" height="62" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="16.0" y1="46.0" x2="16.0" y2="96.0" style="stroke:var(--c-accent);stroke-width:3"/>
<text x="30" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">1 · Manşet</text>
<text x="30" y="79" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">üretici tepe değeri</text>
<text x="30" y="94" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">tüm ALU&#x27;lar, her döngü</text>
<rect x="16" y="112" width="688" height="160" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="16.0" y1="118.0" x2="16.0" y2="266.0" style="stroke:var(--c-warn);stroke-width:3"/>
<text x="30" y="134" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">2 · Silikon kapıları</text>
<text x="30" y="151" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">format + seyreklik</text>
<text x="30" y="166" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">+ bellek besleyebilir mi?</text>
<rect x="16" y="282" width="688" height="66" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="16.0" y1="288.0" x2="16.0" y2="342.0" style="stroke:var(--c-danger);stroke-width:3"/>
<text x="30" y="304" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">3 · Fiziksel kafes</text>
<text x="30" y="321" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">priz, ısı, ağ</text>
<text x="30" y="336" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">dış sınırı çizer</text>
<rect x="236.0" y="58.0" width="440.0" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="246" y="75" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">20 PFLOPS · FP4 · 2:4 seyrek</text>
<rect x="236.0" y="124.0" width="440.0" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:0;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="236.0" y="124.0" width="330.0" height="22" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="246" y="139.5" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">FP4 yoğun (seyreklik yok) · 15 PFLOPS</text>
<text x="574.0" y="139.5" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">−%25</text>
<rect x="236.0" y="160.0" width="440.0" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:0;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="236.0" y="160.0" width="110.0" height="22" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="354.0" y="175.5" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">FP8 yoğun · 5 PFLOPS</text>
<rect x="236.0" y="196.0" width="440.0" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:0;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="236.0" y="196.0" width="55.0" height="22" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="299.0" y="211.5" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">BF16 yoğun · 2,5 PFLOPS — eğitimin tavanı</text>
<rect x="236.0" y="232.0" width="440.0" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:0;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="236.0" y="232.0" width="2.0" height="22" rx="4" style="fill:var(--c-danger);fill-opacity:.22;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="246" y="247.5" text-anchor="start" style="fill:var(--c-text);font-size:12px">decode, B = 1: %0,6&#x27;dan az dolu (belleği bekler)</text>
<rect x="236" y="296" width="140" height="26" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="306" y="313" text-anchor="middle" style="fill:var(--c-text);font-size:12px">1.600 W tek faz</text>
<rect x="388" y="296" width="140" height="26" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="458" y="313" text-anchor="middle" style="fill:var(--c-text);font-size:12px">ısı → saat kısma</text>
<rect x="540" y="296" width="140" height="26" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="610" y="313" text-anchor="middle" style="fill:var(--c-text);font-size:12px">ağ: 800 Gb/s</text>
<text x="236" y="338" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Bunlar tepeyi düşürmez; o tepeyi ne kadar süre tutabileceğinizi belirler.</text>
</svg>

### 1.2. Bir kısıt haritası olarak datasheet

Bir veri sayfası gelecekteki performansı tahmin eden bir falcı değildir; birbirine bağlı **lineer eşitsizlikler sistemidir**.

Modern bir silikon üzerinde çalışan büyük dil modelini düşünün. İşlem hacmi iki katı fiziksel sınır tarafından yönetilir:

$$\text{İşlem Hacmi} \le \min \left( \frac{\text{Hesaplama Kapasitesi (FLOP/s)}}{\text{Model Aritmetik Yoğunluğu (FLOP/Bayt)}}, \text{Bellek Bant Genişliği (Bayt/s)} \right)$$

Burada:
- $\text{Hesaplama Kapasitesi}$, seçilen veri tipinde SM'lerin ulaşabileceği tepe matematiksel hızdır.
- $\text{Model Aritmetik Yoğunluğu}$, taşınan her bayt veri başına algoritmanın kaç işlem yaptığını gösterir.
- $\text{Bellek Bant Genişliği}$, bellek alt sisteminin silikona saniyede besleyebileceği azami veri akışıdır.

Bir broşür petaFLOP değerlerini öne çıkardığında, soldaki terimin payını büyütür. Ancak iş yükünüz yeterli aritmetik yoğunluğu sağlayamıyorsa, sağdaki terim sistemi kilitler ve o devasa silikon kapılarının %95'i bellekten veri bekleyerek saat döngülerini boşa harcar.

---

## 2. Paradigma değişimi: 4 ayrık GPU'dan tekil süperçipe

GB300 DGX Station'ın mimarisini kavramak için "masaüstü AI süper bilgisayarı" kavramının üç donanım nesli boyunca geçirdiği evrimi izlemeliyiz.

### 2.1. DGX Station'ın evrimi (Volta'dan Blackwell'e)

2017 yılında ilk DGX Station; bir Intel Xeon E5-2698 v4 CPU etrafına yerleştirilmiş dört adet SXM2 Tesla V100 GPU (GPU başına 16 GB, daha sonra 32 GB HBM2) barındırıyordu. GPU'lar, devasa bir kapalı devre su soğutma bloğu eşliğinde, GPU başına 200 GB/s hız sunan hibrit küp örgülü (cube mesh) NVLink köprüleriyle konuşuyordu.

2020 yılında DGX Station A100 sahneye çıktı: İki adet dahili NVSwitch yongası üzerinden GPU başına 600 GB/s tam blokajsız (non-blocking) örgülü ağa sahip dört adet SXM4 A100 GPU (40 GB veya 80 GB HBM2e) ve 64 çekirdekli AMD EPYC 7742 CPU ile donatıldı.

Her iki sistem de ortak bir fiziksel mimariyi paylaşıyordu: **PCIe Gen3/Gen4 anahtarları üzerinden bir x86 ana işlemciye bağlı çoklu ayrık GPU soketleri**. GPU'ların kendi aralarındaki iletişim NVLink ile hızlıydı; ancak CPU ile GPU arasındaki veri transferi PCIe köprüsünün (32–64 GB/s) sınırlarına çarpıyordu.

GB300 DGX Station bu çoklu GPU paradigmasını tamamen rafa kaldırıyor:

<svg viewBox="0 0 720 352" role="img" aria-label="İki mimari yan yana. Solda 2017 ve 2020&#x27;nin DGX Station V100 ve A100&#x27;ü: kendi host RAM&#x27;i olan bir x86 CPU, yön başına 16–32 GB/s hızlı bir PCIe anahtarı üzerinden dört ayrık GPU&#x27;ya bağlanır; her GPU&#x27;nun kendi HBM&#x27;i vardır ve GPU&#x27;lar 200–600 GB/s NVLink ya da NVSwitch ile konuşur. Büyük bir model tensör ya da boru hattı paralelliği ve AllReduce ile bölünmek zorundadır. Sağda DGX Station GB300: tek bir pakette 396 GB/s hızlı 496 GB LPDDR5X&#x27;li Grace CPU ile 7,1 TB/s hızlı 252 GB HBM3e&#x27;li Blackwell Ultra GPU, toplam 900 GB/s, yani yön başına 450 GB/s NVLink-C2C ile birleşir. Tek bir 748 GB&#x27;lık tutarlı adres alanı vardır; model tek GPU&#x27;da yaşar. CPU–GPU köprüsü yön başına 16–32 GB/s&#x27;den 450 GB/s&#x27;ye, 14–28 kat büyür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="pd-arr-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker><marker id="pd-arr-s" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Dört ayrık GPU&#x27;dan tek bir süperçipe</text>
<rect x="16" y="40" width="336" height="262" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">2017–2020: 4 ayrık GPU + x86</text>
<rect x="30" y="74" width="140" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="100.0" y="92.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">x86 CPU</text>
<text x="100.0" y="108.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">Xeon / EPYC</text>
<rect x="198" y="74" width="140" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="268.0" y="92.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Host RAM</text>
<text x="268.0" y="108.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">ayrı havuz</text>
<line x1="170.0" y1="96.0" x2="198.0" y2="96.0" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="100.0" y1="118.0" x2="100.0" y2="142.0" style="stroke:var(--c-warn);stroke-width:2.5"/>
<rect x="30.0" y="142.0" width="308.0" height="26" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="184" y="159.5" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">PCIe anahtarı · yön başına 16–32 GB/s</text>
<line x1="64.0" y1="168.0" x2="64.0" y2="186.0" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="30" y="186" width="68" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="64.0" y="204.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">GPU</text>
<text x="64.0" y="220.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kendi HBM&#x27;i</text>
<line x1="64.0" y1="230.0" x2="64.0" y2="244.0" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="144.0" y1="168.0" x2="144.0" y2="186.0" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="110" y="186" width="68" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144.0" y="204.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">GPU</text>
<text x="144.0" y="220.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kendi HBM&#x27;i</text>
<line x1="144.0" y1="230.0" x2="144.0" y2="244.0" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="224.0" y1="168.0" x2="224.0" y2="186.0" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="190" y="186" width="68" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="224.0" y="204.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">GPU</text>
<text x="224.0" y="220.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kendi HBM&#x27;i</text>
<line x1="224.0" y1="230.0" x2="224.0" y2="244.0" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="304.0" y1="168.0" x2="304.0" y2="186.0" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="270" y="186" width="68" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="304.0" y="204.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">GPU</text>
<text x="304.0" y="220.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kendi HBM&#x27;i</text>
<line x1="304.0" y1="230.0" x2="304.0" y2="244.0" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="64.0" y1="244.0" x2="304.0" y2="244.0" style="stroke:var(--c-accent);stroke-width:3"/>
<text x="184" y="262" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">NVLink / NVSwitch: 200–600 GB/s</text>
<text x="184" y="290" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">büyük model → TP/PP + AllReduce</text>
<rect x="368" y="40" width="336" height="262" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="382" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">2025: GB300 — tek süperçip</text>
<rect x="382" y="74" width="308" height="168" rx="10" style="fill:none;stroke:var(--c-accent);stroke-width:1.3;stroke-dasharray:5 4"/>
<text x="682" y="90" text-anchor="end" style="fill:var(--c-accent);font-size:11px;font-weight:600">tek paket</text>
<rect x="394" y="100" width="122" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="455.0" y="119.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Grace CPU</text>
<text x="455.0" y="135.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">72 çekirdek</text>
<rect x="394" y="184" width="122" height="46" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="455.0" y="203.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LPDDR5X</text>
<text x="455.0" y="219.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">496 GB</text>
<rect x="566" y="100" width="114" height="46" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="623.0" y="119.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Blackwell Ultra</text>
<text x="623.0" y="135.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">2 kalıp · NV-HBI</text>
<rect x="566" y="184" width="114" height="46" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="623.0" y="203.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">HBM3e</text>
<text x="623.0" y="219.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">252 GB</text>
<line x1="455.0" y1="146.0" x2="455.0" y2="184.0" style="stroke:var(--c-text-mute);stroke-width:2"/>
<text x="462" y="169" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">396 GB/s</text>
<line x1="623.0" y1="146.0" x2="623.0" y2="184.0" style="stroke:var(--c-success);stroke-width:3"/>
<text x="630" y="169" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">7,1 TB/s</text>
<line x1="518" y1="123" x2="562" y2="123" marker-end="url(#pd-arr-s)" style="stroke:var(--c-success);stroke-width:2.5"/>
<line x1="562" y1="123" x2="518" y2="123" marker-end="url(#pd-arr-s)" style="stroke:var(--c-success);stroke-width:2.5"/>
<text x="541" y="115" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">C2C</text>
<text x="536" y="262" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">NVLink-C2C: toplam 900 GB/s = yön başına 450 GB/s</text>
<text x="536" y="290" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">tek, tutarlı 748 GB&#x27;lık adres alanı</text>
<text x="360" y="326" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">CPU ↔ GPU köprüsü, yön başına: 16–32 GB/s (PCIe) → 450 GB/s (C2C), 14–28 kat</text>
<text x="360" y="344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">Datasheet tuzağı: &quot;900 GB/s&quot; iki yönün toplamıdır; tek yön bunun yarısını taşır.</text>
</svg>

### 2.2. NVLink-C2C neden ayrık GPU topolojilerinin yerini aldı?

GB300 DGX Station **tek bir süperçip paketi** barındırır. NVIDIA, birden çok bağımsız GPU kartını kasa içi kartlarla bağlamak yerine, 72 çekirdekli Arm Neoverse V2 mimarili Grace CPU ile çift kalıplı (dual-die) Blackwell Ultra GPU'yu aynı alt katman (substrate) üzerinde birleştirdi.

NVIDIA masaüstünde 4 GPU'lu tasarımdan neden vazgeçti?
1. **NUMA ve Soketler Arası Gecikmelerin Sıfırlanması:** 4 GPU'lu bir sistemde büyük bir model çalıştırmak Tensör Paralelliği (TP=4) veya Boru Hattı Paralelliği (PP=4) gerektirir. Her attention kafasının çıktısı NVLink üzerinden `AllReduce` kolektifleriyle diğer GPU'lara senkronize edilmelidir. GB300'de ise GPU hesaplamasının tamamı, 10 TB/s bant genişliğine sahip çip içi ara bağlantıyla (NV-HBI) birbirine perçinlenmiş tek bir çift kalıp üzerinde koşar.
2. **Donanımsal Tutarlı Bellek Adresleme:** Grace CPU ile Blackwell GPU, **NVLink-C2C** köprüsü sayesinde çift yönlü 900 GB/s hızında tek bir birleşik fiziksel adres alanını paylaşır. GPU, işletim sistemi sürücülerine veya PCIe DMA transferlerine bulaşmadan, 496 GB'lık CPU belleğindeki verilere doğrudan atomik okuma ve yazma komutları gönderebilir.

---

## 3. Hesaplama gücünün perde arkası: 20 PFLOPS ve seyreklik illüzyonu

Datasheet kapağında büyük puntolarla yazan **"20 PFLOPS'a varan AI Hesaplama Gücü"** ifadesi, lineer cebir açısından gerçekte ne anlama gelir?

### 3.1. Hassasiyet formatlarına göre tepe Tensor FLOP değerleri

GB300'ün işlem kapasitesini tüm sayısal formatlar boyunca mercek altına alalım:

| Sayısal Format | Tensor Core Modu | Seyreklik (Sparsity) Durumu | Tepe FLOP/s | Yoğun BF16'ya Göre Oran |
| :--- | :--- | :--- | :--- | :--- |
| **NVFP4 (4-bit FP)** | 5. Nesil Tensor Core | **2:4 Yapısal Seyreklik** | **20,0 PFLOPS** | **8,0x** |
| **NVFP4 (4-bit FP)** | 5. Nesil Tensor Core | Yoğun (Dense) | 15,0 PFLOPS | 6,0x |
| **FP8 / FP6** | 5. Nesil Tensor Core | 2:4 Yapısal Seyreklik | 10,0 PFLOPS | 4,0x |
| **FP8 / FP6** | 5. Nesil Tensor Core | Yoğun (Dense) | 5,0 PFLOPS | 2,0x |
| **BF16 / FP16** | 5. Nesil Tensor Core | 2:4 Yapısal Seyreklik | 5,0 PFLOPS | 2,0x |
| **BF16 / FP16** | 5. Nesil Tensor Core | **Yoğun (Dense)** | **2,5 PFLOPS** | **1,0x (Referans)** |
| **TF32** | 5. Nesil Tensor Core | Yoğun (Dense) | 1,25 PFLOPS | 0,5x |
| **FP32** | Standart CUDA Çekirdekleri | Tensor Olmayan Vektör ALU | 80 TFLOPS | 0,032x |
| **FP64 / FP64 Tensor** | Çift Duyarlıklı ALU | Yoğun (Dense) | 1,3 TFLOPS | 0,0005x |

Bu tablonun nasıl okunacağını datasheet'in dipnotları söyler: Tensor Core değerleri aksi belirtilmedikçe seyreklik içerir ve yoğun değeri (15 PFLOPS) ayrıca yazılan tek format FP4'tür. Bu yüzden seyreklik FP8, BF16 ve TF32'yi ikiye katlarken FP4'ü yalnızca 1,33 kat büyütür:

<svg viewBox="0 0 720 318" role="img" aria-label="Dört Tensor Core formatı için çift çubuklar. Datasheet değerleri 2:4 seyreklik varsayar: FP4 20, FP8 ve FP6 10, BF16 ve FP16 5, TF32 2,5 PFLOPS. Yoğun değerler 15, 5, 2,5 ve 1,25 PFLOPS&#x27;tur. Seyreklik FP4 dışındaki her formatı ikiye katlar; FP4&#x27;te yoğun değer ayrıca 15 olarak verildiği için kazanç yalnızca 1,33 kattır. Kesikli çizgi, eğitimin gerçek tavanı olan 2,5 PFLOPS&#x27;luk yoğun BF16&#x27;yı işaretler." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Formata göre datasheet sayısı ve yoğun gerçek</text>
<rect x="130.0" y="40.0" width="26.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="162" y="50" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">datasheet değeri (2:4 seyrek, dipnot 2)</text>
<rect x="400.0" y="40.0" width="26.0" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="432" y="50" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">yoğun (kodların çoğunun aldığı)</text>
<text x="16" y="90" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVFP4</text>
<rect x="130.0" y="72.0" width="360.0" height="15" rx="3" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="496.0" y="84" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">20 PF</text>
<rect x="130.0" y="91.0" width="270.0" height="15" rx="3" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="406.0" y="103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">15 PF</text>
<text x="704" y="94" text-anchor="end" style="fill:var(--c-warn);font-size:12px;font-weight:700">seyreklik kazancı ×1,33</text>
<text x="16" y="140" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">FP8 / FP6</text>
<rect x="130.0" y="122.0" width="180.0" height="15" rx="3" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="316.0" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">10 PF</text>
<rect x="130.0" y="141.0" width="90.0" height="15" rx="3" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="226.0" y="153" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">5 PF</text>
<text x="704" y="144" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">seyreklik kazancı ×2</text>
<text x="16" y="190" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">BF16 / FP16</text>
<rect x="130.0" y="172.0" width="90.0" height="15" rx="3" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="226.0" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">5 PF</text>
<rect x="130.0" y="191.0" width="45.0" height="15" rx="3" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="181.0" y="203" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2,5 PF</text>
<text x="704" y="194" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">seyreklik kazancı ×2</text>
<text x="16" y="240" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">TF32</text>
<rect x="130.0" y="222.0" width="45.0" height="15" rx="3" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="181.0" y="234" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">2,5 PF</text>
<rect x="130.0" y="241.0" width="22.5" height="15" rx="3" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<text x="158.5" y="253" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">1,25 PF</text>
<text x="704" y="244" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">seyreklik kazancı ×2</text>
<text x="704" y="108" text-anchor="end" style="fill:var(--c-warn);font-size:11px">yoğun FP4 ayrıca verilmiş (dipnot 3)</text>
<line x1="175.0" y1="66.0" x2="175.0" y2="272.0" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="181.0" y="284" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">yoğun BF16 = eğitimin tavanı</text>
<text x="16" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Datasheet dipnot 2: &quot;Aksi belirtilmedikçe tüm Tensor Core değerleri seyreklikledir.&quot;</text>
</svg>

### 3.2. Yapısal seyreklik (2:4) vs. yoğun matematiksel gerçeklik

Gördüğünüz gibi, 20 PFLOPS manşeti **2:4 yapısal seyreklik** varsayımına dayanır.

2:4 seyreklik silikon seviyesinde nasıl çalışır? Ağırlık matrisinin iç boyutundaki her dört ardışık sayıdan tam olarak ikisinin matematiksel olarak sıfır olması şarttır:

<svg viewBox="0 0 720 352" role="img" aria-label="Sekiz ağırlıklık bir satır üzerinde 2:4 yapısal seyrekliğin işleyişi; satır dörderli iki gruba ayrılır. Birinci grup 0,42, 0, −1,85, 0; ikinci grup 0, 0,91, 0, −0,37 değerlerini taşır. Her grup yalnızca sıfır olmayan iki değerini ve iki adet 2 bitlik indeksi saklar: birinci grupta 0 ve 2, ikinci grupta 1 ve 3. İndeksler donanım çoğullayıcısına hangi aktivasyonları seçeceğini söyler: x0, x2, x5 ve x7. Tensor Core 0,42·x0 + (−1,85)·x2 ile 0,91·x5 + (−0,37)·x7&#x27;yi hesaplar; sekiz yerine dört çarpma yapar. Seyrek değerin yoğun değerin iki katı yazılmasının sebebi budur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sp-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">2:4 seyreklik: her dörtlükte iki sıfır, yarı yarıya çarpma</text>
<text x="16" y="66" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">ağırlık satırı</text>
<text x="16" y="81" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 değer, 4&#x27;ü sıfır</text>
<text x="16" y="146" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">saklanan</text>
<text x="16" y="161" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">değerler + 2 bit indeks</text>
<text x="16" y="226" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">aktivasyonlar</text>
<text x="16" y="241" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">mux indekse göre seçer</text>
<text x="16" y="302" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">çarpma</text>
<text x="16" y="317" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 yerine 4 MAC</text>
<rect x="164" y="44" width="246" height="44" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="416" y="40" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">grup 1: tam 2 sıfır</text>
<rect x="434" y="44" width="246" height="44" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="686" y="40" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">grup 2: tam 2 sıfır</text>
<rect x="170" y="52" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="197.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,42</text>
<rect x="230" y="52" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="257.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,00</text>
<rect x="290" y="52" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="317.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">−1,85</text>
<rect x="350" y="52" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="377.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,00</text>
<rect x="440" y="52" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="467.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,00</text>
<rect x="500" y="52" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="527.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,91</text>
<rect x="560" y="52" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="587.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,00</text>
<rect x="620" y="52" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="647.0" y="70.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">−0,37</text>
<rect x="170" y="134" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="197.0" y="152.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,42</text>
<line x1="197.0" y1="82" x2="197.0" y2="130" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="230" y="134" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="257.0" y="152.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">−1,85</text>
<line x1="317.0" y1="82" x2="257.0" y2="130" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="440" y="134" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="467.0" y="152.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,91</text>
<line x1="527.0" y1="82" x2="467.0" y2="130" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="500" y="134" width="54" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="527.0" y="152.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">−0,37</text>
<line x1="647.0" y1="82" x2="527.0" y2="130" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="290" y="134" width="114" height="28" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="347.0" y="152.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">indeks: 0, 2</text>
<rect x="560" y="134" width="114" height="28" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="617.0" y="152.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">indeks: 1, 3</text>
<rect x="170" y="214" width="54" height="28" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="197.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x0</text>
<rect x="230" y="214" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="257.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x1</text>
<rect x="290" y="214" width="54" height="28" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="317.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x2</text>
<rect x="350" y="214" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="377.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x3</text>
<rect x="440" y="214" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="467.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x4</text>
<rect x="500" y="214" width="54" height="28" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="527.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x5</text>
<rect x="560" y="214" width="54" height="28" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/><text x="587.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x6</text>
<rect x="620" y="214" width="54" height="28" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="647.0" y="232.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">x7</text>
<path d="M347 162 C 347 190, 318 190, 317 210" marker-end="url(#sp-arr-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M617 162 C 617 190, 647 190, 647 210" marker-end="url(#sp-arr-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<rect x="164" y="290" width="246" height="34" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="287" y="312" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,42·x0 + (−1,85)·x2</text>
<line x1="197.0" y1="244" x2="197.0" y2="287" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="317.0" y1="244" x2="317.0" y2="287" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="434" y="290" width="246" height="34" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="557" y="312" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">0,91·x5 + (−0,37)·x7</text>
<line x1="527.0" y1="244" x2="527.0" y2="287" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="647.0" y1="244" x2="647.0" y2="287" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="360" y="344" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Şart katıdır: her dörtlükte iki sıfır olmalı; ağırlıklar önce budanarak bu kalıba sokulmalıdır.</text>
</svg>

Matris çarpımı sırasında Tensor Core, bir donanım çoğullayıcı (multiplexer) kullanarak sıfır olmayan iki değeri karşılık gelen aktivasyonlarla eşleştirir; çarpmaların yarısını atlayarak matematiksel işlem hacmini ikiye katlar.

> [!WARNING]
> **Uygulamadaki Seyreklik Bedeli:**
> 1. **Eğitim Yoğundur (Dense):** Bir temel modeli sıfırdan 2:4 seyreklikle eğitemezsiniz; bu gradyan inişini (SGD/AdamW) kararsızlaştırır ve modelin şaşkınlık (perplexity) skorunu bozar. Model eğitimi **Yoğun BF16 (2,5 PFLOPS)** ya da **Yoğun FP8 (5,0 PFLOPS)** ile yürütülür.
> 2. **Budama ve İnce Ayar Yükü:** Var olan yoğun bir modeli 2:4 seyreklik kalıbına sokmak ciddi bir kalibrasyon ve ek ince ayar (fine-tuning) gerektirir. Açık ağırlıklı modellerin (LLaMA 3, Mistral, Qwen vb.) tamamı yoğun matrisler olarak dağıtılır. Modeli bu kalıba sokacak ek mühendislik emeği vermezseniz, o 2 katlık donanım ivmesi tamamen kilitli kalır.

---

## 4. 748 GB birleşik bellek havuzu ve bant genişliği uçurumu

GB300 veri sayfasında en çok dikkat çeken iddia şüphesiz **"748 GB Toplam Birleşik Bellek"** ifadesidir.

### 4.1. Fiziksel ayrım: HBM3e vs. LPDDR5X

Bu 748 GB'lık havuz, silikon seviyesinde iki taban tabana zıt bellek teknolojisinden oluşur:

1. **GPU Hızlı Belleği: 252 GB HBM3e**
   - Masaüstü süperçipinde, GPU kalıplarıyla aynı silikon ara katmanı (interposer) üzerine yerleştirilmiş yedi adet 36 GB 12-Hi HBM3e yığını bulunur.
   - Kuramsal tepe bant genişliği: **7,1 TB/s** ($7.100 \text{ GB/s}$).
2. **CPU Sistem Belleği: 496 GB LPDDR5X**
   - Grace CPU; 16 kanallı, lehimli ve ECC korumalı düşük güç tüketimli DDR5X belleğe bağlıdır.
   - Kuramsal tepe bant genişliği: **396 GB/s**.

### 4.2. NVLink-C2C köprüsü ve 18 katlık bant genişliği uçurumu

GPU'nun CPU belleğine, CPU'nun da GPU belleğine erişebilmesi için araya **çift yönlü 900 GB/s** hızında çalışan **NVLink-C2C** köprüsü yerleştirilmiştir.

Datasheet'in bu satırını okurken iki ayrıntı önemlidir. Birincisi, 900 GB/s iki yönün toplamıdır; GPU CPU belleğini okurken veri tek yönde akar ve en fazla 450 GB/s taşır. İkincisi, LPDDR5X'teki bir bayt C2C'ye ulaşmadan önce kendi 396 GB/s'lik veri yolundan çıkmak zorundadır. Bir zincir en yavaş halkası kadar hızlı olduğundan, GPU'nun LPDDR5X'ten görebileceği tavan $\min(450, 396) = 396$ GB/s'dir:

<svg viewBox="0 0 720 352" role="img" aria-label="GB300&#x27;ün içindeki bellek zinciri. 252 GB&#x27;lık HBM3e, Blackwell Ultra GPU&#x27;yu 7,1 TB/s ile besler. 496 GB&#x27;lık LPDDR5X, Grace CPU&#x27;yu 396 GB/s ile besler; CPU da GPU&#x27;ya toplam 900 GB/s, yön başına 450 GB/s hızlı NVLink-C2C ile ulaşır. Çubuklar SM&#x27;lerin gördüğü hızı gösterir: HBM3e&#x27;deki veri 7.100 GB/s ile gelir; LPDDR5X&#x27;teki veri 450 ile 396&#x27;nın küçüğü, yani 396 GB/s ile gelir ve 18 kat yavaştır. 900 GB/s etiketi iki yönün toplamıdır. Bir zincir en yavaş halkası kadar hızlıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mc-arr-s" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="mc-arr-d" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bant genişliği uçurumu: zincir, en yavaş halkası kadar hızlıdır</text>
<rect x="16" y="52" width="112" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="72.0" y="74.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">HBM3e</text>
<text x="72.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">252 GB</text>
<rect x="196" y="52" width="132" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="262.0" y="74.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Blackwell Ultra</text>
<text x="262.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">SM&#x27;ler</text>
<rect x="468" y="52" width="104" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="520.0" y="74.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Grace CPU</text>
<text x="520.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">72 çekirdek</text>
<rect x="616" y="52" width="88" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="660.0" y="74.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LPDDR5X</text>
<text x="660.0" y="90.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">496 GB</text>
<line x1="128" y1="78" x2="192" y2="78" marker-end="url(#mc-arr-s)" style="stroke:var(--c-success);stroke-width:3"/>
<text x="160" y="70" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:700">7,1 TB/s</text>
<line x1="616" y1="78" x2="576" y2="78" marker-end="url(#mc-arr-d)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<text x="594" y="70" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:700">396</text>
<line x1="468" y1="78" x2="332" y2="78" marker-end="url(#mc-arr-d)" style="stroke:var(--c-danger);stroke-width:2"/>
<text x="398" y="70" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:700">NVLink-C2C</text>
<text x="398" y="95" text-anchor="middle" style="fill:var(--c-text);font-size:11px">toplam 900 GB/s</text>
<text x="398" y="109" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yön başına 450 GB/s</text>
<text x="594" y="124" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:700">en dar halka</text>
<line x1="594.0" y1="112.0" x2="594.0" y2="84.0" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:3 3"/>
<text x="16" y="164" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">SM&#x27;lerin gördüğü bant genişliği: bayt nerede duruyor?</text>
<text x="16" y="198" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">bayt HBM3e&#x27;de</text>
<rect x="190.0" y="182.0" width="420.0" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="618.0" y="198" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">7.100 GB/s</text>
<text x="16" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">bayt LPDDR5X&#x27;te</text>
<rect x="190.0" y="222.0" width="23.4" height="24" rx="4" style="fill:var(--c-danger);fill-opacity:.3;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="221.42535211267605" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">min(450, 396) = 396 GB/s → 18 kat yavaş</text>
<text x="16" y="278" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">C2C etiketi</text>
<rect x="190.0" y="262.0" width="53.2" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:0;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="251.2394366197183" y="278" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">900 GB/s: iki yön toplanmış; tek yön 450</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">LPDDR5X&#x27;teki bir bayt önce kendi 396 GB/s&#x27;lik veri yolundan çıkmak zorundadır; daha hızlı bir C2C bunu değiştirmez.</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">252 GB&#x27;ı tek bir gigabayt bile aşmak, o gigabaytı yavaş tarafa koyar.</text>
</svg>

Bu sert düşüşe sistem mühendisliğinde **Bant Genişliği Uçurumu (Bandwidth Cliff)** denir. Model ağırlıklarınız 252 GB'lık HBM3e içine sığıyorsa, SM'ler saniyede 7,1 terabayt veriyle beslenir. Ancak modeliniz 252 GB sınırını tek bir gigabayt bile aştığı anda, o taşan katmanların yürütülmesi 396 GB/s'ye çakılır.

---

## 5. 1 Trilyon parametre iddiası: Prefill ve decode matematiği

Veri sayfası açıkça iddia ediyor: *"1 trilyon parametreye kadar modeller için yerel eğitim, ince ayar ve çıkarım desteği."*

1 trilyon parametreli bir model bu makinede gerçekten çalışabilir mi? Matematiği adım adım kuralım.

### 5.1. Model ayak izi ve KV cache bellek tahsisi

Özbağlanımlı (autoregressive) Transformer modellerinde bellek tüketimi iki ana kaleme ayrılır: Model ağırlıkları ve Key-Value (KV) cache.

1 trilyon ($1 \times 10^{12}$) parametreli bir model için ağırlık boyutları:
- **FP16 / BF16 (2 bayt/parametre):** $10^{12} \times 2 = 2.000 \text{ GB}$ (748 GB'lık toplam kasaya sığmaz!).
- **FP8 (1 bayt/parametre):** $10^{12} \times 1 = 1.000 \text{ GB}$ (Yine sığmaz!).
- **NVFP4 (0,5 bayt/parametre):** $10^{12} \times 0.5 = \mathbf{500 \text{ GB}}$.

NVFP4 formatında 500 GB tutan ağırlıklar, 748 GB'lık birleşik havuza **sığar**. Geriye işletim sistemi, aktivasyonlar ve KV cache için kalan alan:

$$748 \text{ GB} - 500 \text{ GB} = 248 \text{ GB Boş Alan (Aktivasyon ve KV Cache)}$$

Demek ki model **belleğe sığmaktadır**. Peki saniyede kaç token üretebilir?

### 5.2. Adım adım gecikme yürüyüşü (1T vs. 405B vs. 70B)

Token üretim evresinde (decode fazı), tekil kullanıcı ($B=1$) için her bir yeni token üretilirken modelin tüm ağırlıkları bellekten en az bir kez baştan sona okunmak zorundadır. Bu yüzden decode adımı tamamen **bellek bant genişliği darboğazındadır (memory-bandwidth bound)**:

$$T_{\text{token}} \ge \frac{\text{Model Ağırlık Boyutu (Bayt)}}{\text{Efektif Bellek Bant Genişliği (Bayt/s)}}$$

Şimdi üç farklı model sınıfı için bu matematiksel yürüyüşü gerçekleştirelim:

<svg viewBox="0 0 720 420" role="img" aria-label="Her modelin ağırlıklarının 748 GB&#x27;lık havuzda nereye düştüğü ve bunun token başına maliyeti. FP8&#x27;deki Llama 3 70B 70 GB&#x27;tır ve tamamen HBM3e&#x27;dedir: 70 GB bölü 7,1 TB/s 9,9 ms eder, tavan yaklaşık 101 token/sn&#x27;dir. NVFP4&#x27;teki Llama 3 405B 202,5 GB&#x27;tır, o da tamamen HBM3e&#x27;dedir: 28,5 ms, yaklaşık 35 token/sn. NVFP4&#x27;teki 1 trilyon parametreli model 500 GB&#x27;tır: 252 GB HBM3e&#x27;yi doldurur, 248 GB LPDDR5X&#x27;e taşar. Okuma 35,5 ms artı 626 ms, toplam 662 ms sürer; yaklaşık 1,5 token/sn. Ortak zaman ölçeğinde 1T ağırlıklarının yarısı LPDDR5X&#x27;tedir ama okuma süresinin yüzde 95&#x27;i orada geçer." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ağırlıklar nereye düşüyor, her token neye mal oluyor?</text>
<text x="240.74866310160428" y="50" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:700">HBM3e · 7,1 TB/s</text>
<text x="450.7486631016043" y="50" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:700">LPDDR5X · 396 GB/s</text>
<text x="311.49732620320856" y="250" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">252 GB HBM sınırı</text>
<line x1="311.5" y1="230.0" x2="311.5" y2="239.0" style="stroke:var(--c-text);stroke-width:1.2"/>
<text x="16" y="79" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Llama 3 70B</text>
<text x="16" y="95" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">FP8 · 70 GB</text>
<rect x="170.0" y="66.0" width="141.5" height="22" rx="3" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="311.5" y="66.0" width="278.5" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="170.0" y="66.0" width="39.3" height="22" rx="3" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1.2"/>
<line x1="311.5" y1="60.0" x2="311.5" y2="94.0" style="stroke:var(--c-text);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="170" y="106" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">70 GB ÷ 7,1 TB/s = 9,9 ms</text>
<text x="704" y="82" text-anchor="end" style="fill:var(--c-success);font-size:18px;font-weight:700">~101</text>
<text x="704" y="98" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">token/sn tavanı</text>
<text x="16" y="137" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Llama 3 405B</text>
<text x="16" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">NVFP4 · 202,5 GB</text>
<rect x="170.0" y="124.0" width="141.5" height="22" rx="3" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="311.5" y="124.0" width="278.5" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="170.0" y="124.0" width="113.7" height="22" rx="3" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1.2"/>
<line x1="311.5" y1="118.0" x2="311.5" y2="152.0" style="stroke:var(--c-text);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="170" y="164" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">202,5 GB ÷ 7,1 TB/s = 28,5 ms</text>
<text x="704" y="140" text-anchor="end" style="fill:var(--c-success);font-size:18px;font-weight:700">~35</text>
<text x="704" y="156" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">token/sn tavanı</text>
<text x="16" y="195" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">1T parametre</text>
<text x="16" y="211" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">NVFP4 · 500 GB</text>
<rect x="170.0" y="182.0" width="141.5" height="22" rx="3" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="311.5" y="182.0" width="278.5" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="170.0" y="182.0" width="141.5" height="22" rx="3" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="311.5" y="182.0" width="139.3" height="22" rx="3" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="311.5" y1="176.0" x2="311.5" y2="210.0" style="stroke:var(--c-text);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="170" y="222" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">252 GB ÷ 7,1 TB/s + 248 GB ÷ 396 GB/s = 35,5 + 626 = 662 ms</text>
<text x="704" y="198" text-anchor="end" style="fill:var(--c-danger);font-size:18px;font-weight:700">~1,5</text>
<text x="704" y="214" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">token/sn tavanı</text>
<text x="16" y="280" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Token başına okuma süresi (ms, aynı ölçek)</text>
<text x="150" y="305" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:600">70B</text>
<rect x="170.0" y="292.0" width="6.3" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1.2"/>
<text x="184.25558912386708" y="305" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">9,9 ms</text>
<text x="150" y="329" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:600">405B</text>
<rect x="170.0" y="316.0" width="18.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1.2"/>
<text x="196.08157099697885" y="329" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">28,5 ms</text>
<text x="150" y="353" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:600">1T</text>
<rect x="170.0" y="340.0" width="22.5" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="192.5" y="340.0" width="397.4" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="597.8731117824773" y="353" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">662 ms</text>
<text x="200.5226586102719" y="372" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:700">ağırlıkların yarısı, sürenin %95&#x27;i</text>
<text x="16" y="398" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Bunlar B = 1 için tavan değerlerdir: gerçek decode tepe bant genişliğinin bir kısmına ulaşır ve KV cache&#x27;i de okur.</text>
<text x="16" y="414" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Kural: token/sn ≤ bant genişliği ÷ ağırlık baytı. Önce HBM&#x27;e sığdırın, sonra hızı düşünün.</text>
</svg>

#### Senaryo A: 1 Trilyon Parametreli Model (NVFP4)
- **Ağırlık Dağılımı:** 252 GB'lık ilk kısım HBM3e'ye oturur; kalan 248 GB ise CPU LPDDR5X belleğinde kalır.
- **HBM3e Okuma Süresi:**
  $$T_{\text{HBM}} = \frac{252 \times 10^9 \text{ Bayt}}{7,1 \times 10^{12} \text{ Bayt/s}} \approx 35,5 \text{ ms}$$
- **LPDDR5X'ten Okuma Süresi:** Taşan kısım LPDDR5X veri yolundan (396 GB/s) çıkar ve NVLink-C2C'den (yön başına 450 GB/s) geçer; bu zincirin en yavaş halkası 396 GB/s'dir:
  $$T_{\text{LPDDR}} = \frac{248 \times 10^9 \text{ Bayt}}{396 \times 10^9 \text{ Bayt/s}} \approx 626,3 \text{ ms}$$
  *(900 GB/s'lik C2C değeriyle hesaplasaydık 275,6 ms ve 3,2 token/sn bulurduk. Ancak bu değer iki yönün toplamıdır ve LPDDR5X zaten daha yavaştır; dolayısıyla bu iyimser senaryo fiziksel olarak gerçekleşemez.)*
- **Tek Bir Token Üretim Süresi:** $T_{\text{token}} = 35,5 \text{ ms} + 626,3 \text{ ms} \approx 661,8 \text{ ms}$.
- **Üretim Hızı:** $1 / 0,6618 \approx \mathbf{1,51 \text{ token/saniye}}$. Ağırlıkların yarısı LPDDR5X'te dururken okuma süresinin %95'i orada geçer.

#### Senaryo B: LLaMA 3 405B (NVFP4)
- **Ağırlık Ayak İzi:** $405 \times 10^9 \times 0,5 \text{ Bayt} = \mathbf{202,5 \text{ GB}}$.
- **Yerleşim:** Modelin **%100'ü 252 GB'lık HBM3e içine sığar**! Geriye anlık KV cache için 49,5 GB ultra hızlı VRAM kalır. 496 GB'lık LPDDR5X ise milyonlarca token'lık ikincil bağlam önbelleği olarak görev yapar.
- **Decode Okuma Süresi:**
  $$T_{\text{decode}} = \frac{202,5 \times 10^9 \text{ Bayt}}{7,1 \times 10^{12} \text{ Bayt/s}} \approx 28,5 \text{ ms}$$
- **Üretim Hızı:** $1 / 0,0285 \approx \mathbf{35,1 \text{ token/saniye}}$.

#### Senaryo C: LLaMA 3 70B (FP8)
- **Ağırlık Ayak İzi:** $70 \times 10^9 \times 1,0 \text{ Bayt} = \mathbf{70,0 \text{ GB}}$.
- **Yerleşim:** HBM3e içine fazlasıyla oturur; 182 GB boş VRAM bırakır.
- **Decode Okuma Süresi:**
  $$T_{\text{decode}} = \frac{70,0 \times 10^9 \text{ Bayt}}{7,1 \times 10^{12} \text{ Bayt/s}} \approx 9,86 \text{ ms}$$
- **Üretim Hızı:** $1 / 0,00986 \approx \mathbf{101,4 \text{ token/saniye}}$.

> [!TIP]
> **Mühendislik Çıkarımı:** GB300 DGX Station'ın asıl güç merkezi (sweet spot) 1 trilyon parametreli pazarlama sınırı **değildir**. Asıl güç merkezi **NVFP4 formatında 405B (35 token/sn)** ve **FP8 formatında 70B (101 token/sn)** modellerdir. 1T modeli masada çalıştırmak çevrimdışı ajan döngüleri, sentetik veri üretimi ve toplu doğrulama için harikadır; ancak gerçek zamanlı interaktif bir sohbet için fazla yavaştır.

---

## 6. Roofline modeli ve operasyonel yoğunluk eşiği

Büyük matris işlemlerinin prompt işleme (prefill) ile token üretimi (decode) sırasında neden bambaşka rejimlere girdiğini anlamak için **Roofline Modeli**'ni uygulayalım.

### 6.1. Denge noktasını hesaplamak: Hesaplama vs. bant genişliği

Denge noktası—yani **Kritik Operasyonel Yoğunluk ($\alpha_{\text{crit}}$)**—bir algoritmanın aritmetik işlem hatlarını tam kapasiteyle doyurabilmesi için bellekten okuduğu her bayt başına yapması gereken asgari kayan nokta işlem sayısıdır:

$$\alpha_{\text{crit}} = \frac{\text{Tepe Hesaplama Kapasitesi (FLOP/s)}}{\text{Tepe Bellek Bant Genişliği (Bayt/s)}}$$

GB300'ün HBM3e alt sistemi için $\alpha_{\text{crit}}$ eşiklerini hesaplayalım:

$$\alpha_{\text{crit, BF16 Dense}} = \frac{2,5 \times 10^{15} \text{ FLOP/s}}{7,1 \times 10^{12} \text{ Bayt/s}} \approx \mathbf{352,1 \text{ FLOP/Bayt}}$$

$$\alpha_{\text{crit, FP4 Dense}} = \frac{15,0 \times 10^{15} \text{ FLOP/s}}{7,1 \times 10^{12} \text{ Bayt/s}} \approx \mathbf{2112,7 \text{ FLOP/Bayt}}$$

$$\alpha_{\text{crit, FP4 Sparse}} = \frac{20,0 \times 10^{15} \text{ FLOP/s}}{7,1 \times 10^{12} \text{ Bayt/s}} \approx \mathbf{2816,9 \text{ FLOP/Bayt}}$$

### 6.2. GEMM tepeye ulaşırken GEMV neden aç kalır?

Bu sayılar modern bellek duvarının boyutunu gözler önüne serer:
1. **Ön-Doldurma / Prefill (GEMM - Genel Matris-Matris Çarpımı):** Uzun bir prompt girildiğinde ($S=4096$), GPU $[D, D]$ boyutundaki ağırlık matrisini $[B, S, D]$ boyutundaki aktivasyon tensörüyle çarpar. Operasyonel yoğunluk dizi uzunluğuyla orantılı olarak büyür ($\sim O(S)$). $S \gg 352$ olduğundan, iş yükü Roofline grafiğinin sağındaki yatay tepeye yerleşir ve 2,5–15 PFLOPS'luk hesaplama gücünün tamamını kullanır.
2. **Kod Çözme / Decode (GEMV - Genel Matris-Vektör Çarpımı):** Teker teker token üretilirken ($S=1$), $[D, D]$ matrisi tek bir $[1, D]$ vektörüyle çarpılır. Her ağırlık bellekten yalnızca iki işlem (bir çarpma, bir toplama) yapmak için okunur. Bir ağırlık BF16'da 2 bayt, FP8'de 1 bayt, FP4'te yarım bayt tuttuğu için operasyonel yoğunluk:

$$\alpha_{\text{decode}} \approx 1\text{–}4 \text{ FLOP/Bayt} \ll 352,1 \text{ FLOP/Bayt}$$

BF16'da $\alpha \approx 1$ olduğundan decode en fazla $7,1 \text{ TB/s} \times 1 \text{ FLOP/Bayt} \approx 7,1$ TFLOP/s'ye ulaşabilir; bu, 2,5 PFLOPS'luk tavanın yalnızca **%0,3'ü** kadardır. SM'ler zamanlarının %99'undan fazlasını HBM3e önbellek satırlarının dolmasını bekleyerek geçirir.

Roofline grafiği, iki evrenin aynı donanımda nasıl bambaşka yerlere düştüğünü tek bakışta gösterir:

<svg viewBox="0 0 720 392" role="img" aria-label="GB300&#x27;ün log-log eksenlerde roofline&#x27;ı. Yatay eksende operasyonel yoğunluk 1&#x27;den 10.000 FLOP/Bayt&#x27;a, dikey eksende ulaşılabilir hız 100 GFLOP/s&#x27;den 100 PFLOP/s&#x27;ye uzanır. HBM3e&#x27;den beslenen eğim 7,1 TB/s çarpı yoğunluktur. Bu eğim 2,5 PFLOP/s&#x27;lik yoğun BF16 tavanına 352 FLOP/Bayt&#x27;ta, 15 PFLOP/s&#x27;lik yoğun FP4 tavanına 2.113&#x27;te ulaşır. 20 PFLOP/s&#x27;lik seyrek FP4 manşet tavanı kesikli çizilmiştir. Veri LPDDR5X&#x27;te ise eğim 396 GB/s&#x27;ye düşer ve BF16 tavanına ancak 6.300 civarında ulaşır. Tek kullanıcılı decode eğimin üzerinde 1–4 yoğunlukta, yaklaşık 7 TFLOP/s&#x27;de, BF16 tavanının yüzde 0,3&#x27;ü civarında oturur. 4.096 token&#x27;lık prompt&#x27;la prefill düz tavanın üzerindedir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">GB300 roofline&#x27;ı: decode eğimde, prefill tavanda yaşar</text>
<line x1="96.0" y1="330.0" x2="690.0" y2="330.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="334.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100 G</text>
<line x1="96.0" y1="282.7" x2="690.0" y2="282.7" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="286.6666666666667" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1 T</text>
<line x1="96.0" y1="235.3" x2="690.0" y2="235.3" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="239.33333333333331" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10 T</text>
<line x1="96.0" y1="188.0" x2="690.0" y2="188.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="192.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100 T</text>
<line x1="96.0" y1="140.7" x2="690.0" y2="140.7" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="144.66666666666666" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1 P</text>
<line x1="96.0" y1="93.3" x2="690.0" y2="93.3" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="97.33333333333334" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10 P</text>
<line x1="96.0" y1="46.0" x2="690.0" y2="46.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="90" y="50.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100 P</text>
<line x1="96.0" y1="330.0" x2="96.0" y2="46.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="96.0" y="345" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1</text>
<line x1="244.5" y1="330.0" x2="244.5" y2="46.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="244.5" y="345" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10</text>
<line x1="393.0" y1="330.0" x2="393.0" y2="46.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="393.0" y="345" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<line x1="541.5" y1="330.0" x2="541.5" y2="46.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="541.5" y="345" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1.000</text>
<line x1="690.0" y1="330.0" x2="690.0" y2="46.0" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="690.0" y="345" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10.000</text>
<line x1="96.0" y1="330.0" x2="690.0" y2="330.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="96.0" y1="330.0" x2="96.0" y2="46.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="393.0" y="364" text-anchor="middle" style="fill:var(--c-text);font-size:12px">operasyonel yoğunluk α (FLOP/Bayt, log)</text>
<text x="18" y="188.0" text-anchor="middle" style="fill:var(--c-text);font-size:12px" transform="rotate(-90 18 188.0)">ulaşılabilir FLOP/s (log)</text>
<path d="M96.0 301.7 L660.3 121.8 L690.0 121.8" style="fill:none;stroke:var(--c-danger);stroke-width:1.6;stroke-dasharray:6 4"/>
<path d="M608.3 79.1 L690.0 79.1" style="fill:none;stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:3 3"/>
<path d="M96.0 242.4 L589.7 85.0 L690.0 85.0" style="fill:none;stroke:var(--c-accent);stroke-width:2.2"/>
<path d="M474.2 121.8 L690.0 121.8" style="fill:none;stroke:var(--c-success);stroke-width:2.4"/>
<line x1="474.2" y1="121.8" x2="474.2" y2="330.0" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:2 3"/>
<circle cx="474.2" cy="121.8" r="4" style="fill:var(--c-success)"/>
<text x="478.17990472292234" y="324" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">352</text>
<line x1="589.7" y1="85.0" x2="589.7" y2="330.0" style="stroke:var(--c-accent);stroke-width:1;stroke-dasharray:2 3"/>
<circle cx="589.7" cy="85.0" r="4" style="fill:var(--c-accent)"/>
<text x="593.7384180984573" y="324" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">2.113</text>
<rect x="106" y="54" width="244" height="86" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="116.0" y1="68.0" x2="142.0" y2="68.0" style="stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:3 3"/>
<text x="150" y="72" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">manşet: FP4 seyrek 20 PF</text>
<line x1="116.0" y1="87.0" x2="142.0" y2="87.0" style="stroke:var(--c-accent);stroke-width:2.2"/>
<text x="150" y="91" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">FP4 yoğun tavan 15 PF</text>
<line x1="116.0" y1="106.0" x2="142.0" y2="106.0" style="stroke:var(--c-success);stroke-width:2.4"/>
<text x="150" y="110" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">BF16 yoğun tavan 2,5 PF</text>
<line x1="116.0" y1="125.0" x2="142.0" y2="125.0" style="stroke:var(--c-danger);stroke-width:1.6;stroke-dasharray:6 4"/>
<text x="150" y="129" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">veri LPDDR5X&#x27;teyse eğim</text>
<text x="333.9059087122024" y="158.54293190444025" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:700" transform="rotate(-17.7 333.9 158.5)">HBM3e: 7,1 TB/s × α</text>
<path d="M96.0 242.4 L185.4 213.9" style="fill:none;stroke:var(--c-accent-2);stroke-width:6"/>
<circle cx="96.0" cy="242.4" r="5" style="fill:var(--c-accent-2)"/>
<line x1="140.7" y1="224.1" x2="140.7" y2="182.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-dasharray:2 2"/>
<text x="106.0" y="162" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">decode, B = 1</text>
<text x="106.0" y="177" text-anchor="start" style="fill:var(--c-text);font-size:11px">α ≈ 1–4 → tavanın ~%0,3&#x27;ü</text>
<circle cx="632.4" cy="121.8" r="5" style="fill:var(--c-success)"/>
<text x="632.4354522732144" y="179.83083958952355" text-anchor="middle" style="fill:var(--c-success);font-size:12px;font-weight:700">prefill, S = 4.096</text>
<text x="632.4354522732144" y="194.83083958952355" text-anchor="middle" style="fill:var(--c-text);font-size:11px">α, S ile büyür → tavanda</text>
<line x1="632.4" y1="127.8" x2="632.4" y2="163.8" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:2 2"/>
<text x="303.5940912877976" y="229.5399341223826" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:600" transform="rotate(-17.7 303.6 229.5)">LPDDR5X: 396 GB/s × α</text>
<text x="466.17990472292234" y="249.58208646142845" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-weight:600">← bellek sınırlı</text>
<text x="482.17990472292234" y="249.58208646142845" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:600">hesaplama sınırlı →</text>
</svg>

---

## 7. Masaüstü gerçekliği: Ev mi, veri merkezi mi? 1600W güç ve ConnectX-8

Bir donanım teknik şartnamesi temelde elektrik kanunları, termodinamik ve kullanım ortamı tarafından dikte edilir. DGX Station broşürünü eline alan bir mühendisin aklına gelen ilk soru şudur: *"Bu cihazı evime koyabilir miyim, yoksa bu bir veri merkezi sunucusu mudur?"*

### 7.1. Üç dünyanın ayrımı: Ev mi, veri merkezi mi, Ar-Ge masası mı?

Bu sorunun yanıtı ne siyah ne de beyazdır. DGX Station, üç farklı dünyanın fiziksel kesişiminde durur:

<svg viewBox="0 0 720 446" role="img" aria-label="Üç dünyanın karşılaştırması. Ev: 24–32 GB&#x27;lık GeForce RTX 4090 ya da 5090, ev prizi ve hava soğutma; küçük modeller, LoRA ve prototipler için. Ar-Ge ofisi: 748 GB tutarlı belleği olan DGX Station GB300, masa altında 1.600 W tek faz ve sıvı soğutma; 70B–405B modeller veri şirket içinde kalarak yerelde çalışır. Veri merkezi: 480 V üç faz, megavatlar ve tesis soğutmasıyla DGX B200 ve GB200 NVL72 kabinleri; binlerce GPU&#x27;luk ön eğitim için. Altta, hangi hattın 1.600 W&#x27;ı sürekli taşıyabildiği: Kuzey Amerika&#x27;da 120 V 15 A hat 1.440 W&#x27;a izin verir ve yetmez; 20 A hat 1.920 W&#x27;a izin verir; Türkiye ve Avrupa&#x27;daki 230 V prizler 10 A&#x27;de 2.300 W, 16 A&#x27;de 3.680 W verir. 1.600 W&#x27;ın neredeyse tamamı odaya ısı olarak çıkar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ev mi, ofis mi, veri merkezi mi? DGX Station&#x27;ın yeri</text>
<rect x="16" y="40" width="224" height="214" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13.5px;font-weight:700">Ev / bireysel</text>
<text x="30" y="79" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">tüketici GPU&#x27;su</text>
<text x="30" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">DONANIM</text>
<text x="30" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12px">GeForce RTX 4090 / 5090</text>
<text x="30" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">BELLEK</text>
<text x="30" y="158" text-anchor="start" style="fill:var(--c-text);font-size:12px">24–32 GB VRAM</text>
<text x="30" y="180" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">GÜÇ / ORTAM</text>
<text x="30" y="196" text-anchor="start" style="fill:var(--c-text);font-size:12px">ev prizi, hava soğutma</text>
<text x="30" y="218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">TİPİK İŞ</text>
<text x="30" y="234" text-anchor="start" style="fill:var(--c-text);font-size:12px">küçük model, LoRA, prototip</text>
<rect x="248" y="40" width="224" height="214" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="248" y="40" width="224" height="214" rx="8" style="fill:var(--c-accent);fill-opacity:.06;stroke:none"/>
<text x="262" y="62" text-anchor="start" style="fill:var(--c-accent);font-size:13.5px;font-weight:700">Ar-Ge ofisi ★</text>
<text x="262" y="79" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">masa yanı süper bilgisayar</text>
<text x="262" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">DONANIM</text>
<text x="262" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">DGX Station GB300</text>
<text x="262" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">BELLEK</text>
<text x="262" y="158" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">748 GB tutarlı</text>
<text x="262" y="180" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">GÜÇ / ORTAM</text>
<text x="262" y="196" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">1.600 W tek faz, sıvı</text>
<text x="262" y="218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">TİPİK İŞ</text>
<text x="262" y="234" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">70B–405B yerelde, veri içeride</text>
<rect x="480" y="40" width="224" height="214" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="494" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13.5px;font-weight:700">Veri merkezi</text>
<text x="494" y="79" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">kabin ölçeği</text>
<text x="494" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">DONANIM</text>
<text x="494" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12px">DGX B200 / GB200 NVL72</text>
<text x="494" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">BELLEK</text>
<text x="494" y="158" text-anchor="start" style="fill:var(--c-text);font-size:12px">kabin başına TB&#x27;larca HBM</text>
<text x="494" y="180" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">GÜÇ / ORTAM</text>
<text x="494" y="196" text-anchor="start" style="fill:var(--c-text);font-size:12px">480 V üç faz, megavatlar</text>
<text x="494" y="218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.0px;font-weight:600">TİPİK İŞ</text>
<text x="494" y="234" text-anchor="start" style="fill:var(--c-text);font-size:12px">ön eğitim, binlerce GPU</text>
<text x="16" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Hat 1.600 W&#x27;ı sürekli taşıyabilir mi?</text>
<text x="16" y="311" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">K. Amerika 120 V · 15 A*</text>
<rect x="236.0" y="298.0" width="158.4" height="18" rx="3" style="fill:var(--c-danger);fill-opacity:.3;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="386.4" y="311" text-anchor="end" style="fill:var(--c-danger);font-size:12px;font-weight:700">1.440 W  ✗</text>
<text x="16" y="337" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">K. Amerika 120 V · 20 A*</text>
<rect x="236.0" y="324.0" width="211.2" height="18" rx="3" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="455.2" y="337" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">1.920 W  ✓</text>
<text x="16" y="363" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">TR / AB 230 V · 10 A</text>
<rect x="236.0" y="350.0" width="253.0" height="18" rx="3" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="497.0" y="363" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">2.300 W  ✓</text>
<text x="16" y="389" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">TR / AB 230 V · 16 A</text>
<rect x="236.0" y="376.0" width="404.8" height="18" rx="3" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="648.8" y="389" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">3.680 W  ✓</text>
<line x1="412.0" y1="292.0" x2="412.0" y2="402.0" style="stroke:var(--c-text);stroke-width:1.6;stroke-dasharray:5 3"/>
<text x="418.0" y="412" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:700">DGX Station: 1.600 W</text>
<text x="16" y="432" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">* %80 sürekli yük kuralı. Sığması yetmez: 1.600 W&#x27;ın neredeyse tamamı odaya ısı olarak geri çıkar.</text>
</svg>

1. **Ev Ortamı (Fiziksel Olarak Mümkün, Pratik Olarak Çılgınlık):**
   - *Priz Uyumluluğu:* Sistem 1.600W çeker. Türkiye ve Avrupa'daki standart ev prizleri 230V'ta 10A ila 16A ($2.300\text{W} - 3.680\text{W}$) akım sağlar (Kuzey Amerika'da 120V 15A hattında continuous load derating ile $1.440\text{W}$, 20A hatta $1.920\text{W}$). Yani Türkiye ve Avrupa'da teknik olarak bir su ısıtıcısı gibi ev prizine takılıp çalışır; Kuzey Amerika'daki standart 15A hat ise bu yükü sürekli taşıyamaz, 20A hat gerekir.
   - *Termodinamik Tuzak (1.600W Elektrikli Soba):* Termodinamiğin birinci kanunu gereği, kasaya giren 1.600W elektrik enerjisinin neredeyse tamamı **ısı** olarak odaya atılır. Bir model eğitirken odanızda sürekli son ayarda yanan bir elektrikli ısıtıcı çalışır; klimasız tipik bir ev odasını kısa sürede fırına çevirir.
   - *Fiyat ve Yazılım:* Tüketici sınıfı GeForce RTX 5090 kartların ($2.000 civarı) aksine, DGX Station serisi $50.000 - $150.000+ bandında kurumsal bütçeler gerektirir ve Windows yerine kurumsal Linux (NVIDIA AI Developer Tools'lu Ubuntu, CUDA-X araçları ve NVIDIA AI Enterprise lisansları) ile donatılmıştır.
2. **Veri Merkezi Ortamı (Form Faktörü Ayrışması):**
   - Fiziksel olarak standart 19 inçlik sunucu kabinlerine (rack) vidalanan yatay bir sunucu değildir; dik duran kule tipi bir kasadır.
   - Veri merkezlerinin gerektirdiği 480V üç fazlı endüstriyel elektrik altyapısı, yükseltilmiş taban hava kanalları veya harici tesisat suyu soğutması gerektirmez.
3. **Ar-Ge Ofisi ve Laboratuvar (Asıl Varoluş Sebebi):**
   - NVIDIA bu ürünü tam olarak bu iki dünyanın arasına, **"Deskside AI Supercomputer"** olarak yerleştirir.
   - Hedef kitle: Hassas şirket verisini genel bulut API'lerine göndermek istemeyen, veri merkezi GPU kuyruklarında (Slurm rezervasyonları, küme kotası) haftalarca beklemek istemeyen ve 70B ila 405B modeller üzerinde ofisinde anlık prototip geliştirmek isteyen yapay zekâ ekipleridir. Kısacası: **Masanızın altına sokulmuş, fısıltı sessizliğinde (< 45 dBA) bir veri merkezi düğümüdür.**

### 7.2. 800 Gb/s SuperNIC neden bir masaüstü kasasında yer alır?

Datasheet'te yer alan bir diğer şaşırtıcı donanım, iki adet QSFP112 portu üzerinden **800 Gb/s Ethernet veya InfiniBand** destekleyen **NVIDIA ConnectX-8 SuperNIC**'tir.

Masa altındaki tek bir bilgisayara neden 800 Gb/s ağ kartı konur?
1. **Veri Merkezi Kümelerine Dikişsiz Köprü:** Araştırmacılar modellerini, kontrol noktalarını ve eğitim döngülerini masalarındaki DGX Station üzerinde prototipler. Büyük ölçekli dağıtık eğitime geçmeye hazır olduklarında istasyon, şirketin RoCEv2 veya InfiniBand kümesine doğrudan bağlanarak veri merkezindeki DGX SuperPOD'larla yan yana çalışan aktif bir hesaplama düğümüne dönüşür.
2. **İstasyonlar Arası Kümeleme:** Datasheet'e göre iki DGX Station ConnectX-8 üzerinden doğrudan birbirine bağlanabilir; böylece çok düğümlü boru hattı paralelliği için 1,4 TB'ı aşan bellek ve 40 PFLOPS'a varan hesaplama bir araya gelir.

Yine de bu ağı bellekle karıştırmamak gerekir. Ağ hızları bit, bellek hızları bayt cinsinden yazılır (datasheet tablosu bile ağ satırına "800 GB/s" yazarak bu tuzağa düşer; portlar 2 × 400 Gb/s'dir):

<svg viewBox="0 0 720 344" role="img" aria-label="Masaüstü bir kasada neden 800 Gb/s ağ kartı var? Her biri 748 GB ve 20 PFLOPS olan iki DGX Station, iki adet 400 Gb/s port üzerinden doğrudan bağlanır; datasheet en fazla ikiye izin verir, toplam yaklaşık 1,5 TB ve 40 PFLOPS olur. Bir istasyon RoCEv2 ya da InfiniBand üzerinden veri merkezi kümesine de katılabilir. Çubuklar bir baytın nereden geldiğini tek doğrusal ölçekte GB/s olarak karşılaştırır: HBM3e 7.100, NVLink-C2C yön başına 450, LPDDR5X 396, ağ ise 800 Gb/s bölü 8, yani 100 GB/s; HBM&#x27;den 71 kat yavaş. İstasyonlar arasında katman sınırında az veri taşıyan boru hattı paralelliği uygundur; her katmanda AllReduce isteyen tensör paralelliği uygun değildir. Ağ hızları bit, bellek hızları bayt cinsinden yazılır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="nw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="nw-arr-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Masadan 800 Gb/s: bit, bayt değildir</text>
<rect x="16" y="46" width="150" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="91.0" y="67.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">DGX Station A</text>
<text x="91.0" y="83.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">748 GB · 20 PF</text>
<rect x="276" y="46" width="150" height="50" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="351.0" y="67.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">DGX Station B</text>
<text x="351.0" y="83.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">748 GB · 20 PF</text>
<rect x="520" y="46" width="184" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="612.0" y="67.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">veri merkezi kümesi</text>
<text x="612.0" y="83.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">RoCEv2 / InfiniBand</text>
<line x1="170" y1="71" x2="272" y2="71" marker-end="url(#nw-arr-a)" style="stroke:var(--c-accent);stroke-width:2.2"/>
<line x1="272" y1="71" x2="170" y2="71" marker-end="url(#nw-arr-a)" style="stroke:var(--c-accent);stroke-width:2.2"/>
<text x="221" y="63" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:700">2 × 400 Gb/s</text>
<line x1="430" y1="71" x2="516" y2="71" marker-end="url(#nw-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="473" y="63" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">aynı yazılım</text>
<text x="221" y="116" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">en fazla 2 istasyon: ≈1,5 TB, 40 PFLOPS</text>
<text x="16" y="152" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Bir bayt nereden geliyor? (GB/s, tek doğrusal ölçek)</text>
<text x="16" y="179" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">HBM3e</text>
<rect x="210.0" y="164.0" width="420.0" height="20" rx="3" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="638.0" y="179" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">7.100 GB/s</text>
<text x="16" y="209" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">NVLink-C2C (yön başına)</text>
<rect x="210.0" y="194.0" width="26.6" height="20" rx="3" style="fill:var(--c-text-mute);fill-opacity:.3;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="244.61971830985914" y="209" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">450 GB/s</text>
<text x="16" y="239" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">LPDDR5X</text>
<rect x="210.0" y="224.0" width="23.4" height="20" rx="3" style="fill:var(--c-danger);fill-opacity:.3;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="241.42535211267605" y="239" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">396 GB/s</text>
<text x="16" y="269" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">ConnectX-8 ağı</text>
<rect x="210.0" y="254.0" width="5.9" height="20" rx="3" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="223.91549295774647" y="269" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">800 Gb/s ÷ 8 = 100 GB/s</text>
<text x="16" y="302" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Ağ, HBM&#x27;den 71 kat yavaştır: istasyonlar arasında boru hattı paralelliği (katman sınırında az veri)</text>
<text x="16" y="320" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">uygundur; tensör paralelliği (her katmanda bir AllReduce) uygun değildir.</text>
<text x="16" y="338" text-anchor="start" style="fill:var(--c-warn);font-size:11.5px;font-weight:600">Birim tuzağı: ağ hızı bit (Gb/s), bellek hızı bayt (GB/s) ile yazılır. 8&#x27;e bölün.</text>
</svg>

---

## 8. Karşılaştırmalı evrim matrisi: V100'den GB300'e

GB300 DGX Station'ı tarihsel bir perspektife oturtmak için platformun üç neslini yan yana getirelim:

### 8.1. Mimarilerin nesilsel dökümü

| Özellik / Metrik | DGX Station V100 (2017) | DGX Station A100 (2020) | DGX Station GB300 (2025/2026) |
| :--- | :--- | :--- | :--- |
| **Hesaplama Motoru** | 4x Tesla V100 SXM2 | 4x A100 SXM4 | **1x GB300 Grace Blackwell Ultra** |
| **Ana İşlemci (CPU)** | Intel Xeon E5-2698 v4 (20C) | AMD EPYC 7742 (64C) | **NVIDIA Grace Neoverse V2 (72C)** |
| **Host Ara Bağlantısı** | PCIe Gen3 x16 (16 GB/s) | PCIe Gen4 x16 (32 GB/s) | **NVLink-C2C (900 GB/s Tutarlı)** |
| **GPU'lar Arası Ağ** | NVLink Örgüsü (200 GB/s) | NVSwitch Örgüsü (600 GB/s) | **Tek Soketli Süperçip** |
| **GPU VRAM Havuzu** | 64 GB / 128 GB HBM2 | 160 GB / 320 GB HBM2e | **252 GB HBM3e** |
| **Toplam Tutarlı Bellek** | Yok (Ayrık Host RAM) | Yok (Ayrık Host RAM) | **748 GB (252 GB HBM + 496 GB LPDDR)** |
| **GPU Bellek Bant Genişliği** | $4 \times 900 \text{ GB/s} = 3,6 \text{ TB/s}$ | $4 \times 2,0 \text{ TB/s} = 8,0 \text{ TB/s}$ | **7,1 TB/s (Tek Süperçip)** |
| **Tepe FP16/BF16 Tensor** | 500 TFLOPS (FP16 Yoğun) | 2.496 TFLOPS (FP16 Yoğun) | **2.500 TFLOPS (BF16 Yoğun)** |
| **Tepe Düşük Hassasiyet AI** | 500 TFLOPS (FP16) | 4.992 TFLOPS (INT8 Seyrek) | **20.000 TFLOPS (FP4 Seyrek)** |
| **Ölçekleme Ağı (Scale-Out)** | 4x 100 Gb/s InfiniBand | 4x 200 Gb/s InfiniBand | **1x 800 Gb/s ConnectX-8 (2x 400G)** |
| **Sistem Gücü** | 1.500W | 1.500W | **1.600W** |

### 8.2. Donanım mimarları için temel çıkarımlar

1. **Silikon Konsolidasyonu:** 2017'de 4 ayrı sıvı soğutmalı V100 kartı ve karmaşık devre kartı yollarıyla elde edilen hesaplama gücü, bugün FLOP başına harcanan çok daha az enerjiyle tek bir süperçip soketinde toplanmıştır.
2. **Platform İlkesi Olarak Birleşik Bellek:** Ayrık host/cihaz PCIe tamponlarından kesintisiz 900 GB/s tutarlı bellek ağına geçiş, büyük modellerin sunulma biçimini kalıcı olarak değiştirmiş; eskiden çok düğümlü kümeler gerektiren iş yüklerini tek bir masaüstü istasyonuna sığdırmıştır.

---

## 9. Öğrendiklerimizi sınayalım

Bu rehberdeki formülleri ve kavramları kullanarak donanım analiz yetkinliğinizi test edin.

### 9.1. Soru 1: Yoğun vs. seyrek hesaplama

<details>
<summary><strong>Bir üretici veri sayfası, GPU'nun 2:4 yapısal seyreklikle 10 PFLOPS FP8 hesaplama sunduğunu belirtiyor. Eğitim kod tabanınız yoğun (dense) FP8 matrisleri kullanıyorsa garanti edilen aritmetik kapasite nedir?</strong></summary>

**Cevap:** Tam olarak **5,0 PFLOPS**. 

2:4 yapısal seyreklik, ağırlık değerlerinin %50'sinin sıfır olduğunu ve donanım indeks kodlamasıyla atlandığını varsayar. Standart yoğun ön-eğitim ve ince ayar bu özelliği kullanamaz; dolayısıyla gerçek matematiksel yürütme hızı yarıya iner.
</details>

### 9.2. Soru 2: Bellek uçurumu hesabı

<details>
<summary><strong>GB300 üzerinde ağırlıkları 300 GB bellek gerektiren bir model devreye alıyorsunuz. Verinin ne kadarı 7,1 TB/s ile okunur, ne kadarı NVLink-C2C köprüsünden akar ve token başına decode okuma gecikmesi ne olur?</strong></summary>

**Cevap:**
1. İlk **252 GB** doğrudan HBM3e'den $7,1 \text{ TB/s}$ hızla okunur:
   $$T_{\text{HBM}} = \frac{252 \times 10^9}{7,1 \times 10^{12}} \approx 35,5 \text{ ms}$$
2. Kalan **48 GB** ($300 - 252$) LPDDR5X belleğine taşar. Bu veri NVLink-C2C'den geçer, ancak zincirin en yavaş halkası $396 \text{ GB/s}$'lik LPDDR5X veri yoludur (C2C'nin 900 GB/s'i iki yönün toplamıdır; tek yön 450 GB/s taşır):
   $$T_{\text{LPDDR}} = \frac{48 \times 10^9}{396 \times 10^9} \approx 121,2 \text{ ms}$$
3. Token başına toplam decode okuma süresi $35,5 \text{ ms} + 121,2 \text{ ms} = 156,7 \text{ ms}$ olur. Bu da yaklaşık **6,4 token/saniye** üretim hızına karşılık gelir. Ağırlıkların yalnızca %16'sı taşmıştır, ama sürenin %77'sini o kısım harcar.
</details>

### 9.3. Soru 3: Doğru model ölçeğini seçmek

<details>
<summary><strong>LLaMA 3 405B (NVFP4) DGX Station'da ~35 token/sn ile çalışırken, 1 trilyon parametreli bir model (NVFP4) neden ~1,5 token/sn hızına çakılır?</strong></summary>

**Cevap:** 

LLaMA 3 405B NVFP4 formatında **202,5 GB** yer kaplar ve **%100'ü 252 GB'lık HBM3e içine sığar**. Bellek okumaları tamamen 7,1 TB/s hızındaki bellek veri yolunda kalır. 

1 trilyon parametreli model ise NVFP4 formatında **500 GB** tutar ve **248 GB'ı LPDDR5X'e taşar**. Bu 248 GB önce 396 GB/s'lik LPDDR5X veri yolundan çıkıp ardından NVLink-C2C'den geçmek zorundadır; zincirin en yavaş halkası 396 GB/s olduğu için 18 katlık bant genişliği uçurumuna takılır.
</details>

---

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

- Bir yapay zekâ donanım şartnamesindeki üretici tepe iddiaları ile gerçek dünya mikromimari işlem hacmi arasındaki sınır çizgisi nerededir?
- Roofline modelini kullanarak yeni bir hızlandırıcı üzerinde bir algoritmanın compute-bound mu yoksa memory-bound mu çalışacağı nasıl hesaplanır?
- Heterojen fiziksel katmanlara sahip birleşik bir bellek havuzu neden kaçınılmaz olarak dik bir bant genişliği uçurumu yaratır?
- Yalnızca parametre sayısı, kuantizasyon formatı ve bellek veri yolu hızına bakarak bir LLM'in token üretim hızı analitik olarak nasıl hesaplanır?
- Şebeke elektriği sınırları (1.600W) kurumsal masaüstü süper bilgisayarların form faktörünü ve soğutma mimarisini nasıl şekillendirir?

---

## Bütün hikâye altı satırda

- Bir GPU datasheet'i çalışma zamanı performansını tahmin etmez; sistemin ulaşabileceği kuramsal fiziksel ve aritmetik sınır koşullarını belirler.
- Manşet PFLOPS sayıları 2:4 yapısal seyreklik şartına bağlıdır; yoğun eğitim ve standart açık ağırlıklı çıkarım bu tepe kapasiteyi BF16, FP8 ve TF32'de yarıya, FP4'te dörtte üçüne indirir.
- GB300 DGX Station, karmaşık çoklu GPU örgülerinin yerine 900 GB/s NVLink-C2C ile bağlanan tek soketli Grace Blackwell Ultra süperçipini koymuştur.
- 748 GB'lık birleşik bellek havuzu düz bir zemin değildir; 252 GB HBM3e'den 496 GB LPDDR5X'e geçiş 18 katlık sert bir bant genişliği uçurumu yaratır.
- 1 trilyon parametreli modeller NVFP4 ile belleğe sığsa bile, token üretim hızları 396 GB/s'lik LPDDR5X veri yolu darboğazı nedeniyle yaklaşık 1,5 token/sn ile sınırlanır.
- Masaüstü AI donanımının asıl tatlı noktası (sweet spot); HBM3e içine tam sığarak 35 ila 101 token/sn hızla akan 70B ila 405B modellerdir.

---

## Terimler sözlüğü

- **NVFP4 (NVIDIA 4-bit Floating Point):** Blackwell mimarisiyle tanıtılan, FP8'e kıyasla 2 kat bellek yoğunluğu sağlayan bir işaret, iki üs ve bir mantis (E2M1) bitli mikroskobik ölçekleme faktörlü 4-bit kayan nokta formatı.
- **2:4 Yapısal Seyreklik (Structured Sparsity):** Bir vektördeki her dört değerden ikisinin sıfır olduğu ve donanımsal indeks kodlamasıyla Tensor Core işlem hacmini ikiye katlayan matris kodlama tekniği.
- **NVLink-C2C (Chip-to-Chip):** Grace CPU ile Blackwell GPU'yu çip paketi üzerinde çift yönlü 900 GB/s hızla bağlayan ultra düşük gecikmeli ve donanımsal tutarlı ara bağlantı.
- **Bant Genişliği Uçurumu (Bandwidth Cliff):** Çalışma kümesinin çip üzerindeki ultra hızlı bellekten (HBM) çip dışındaki daha yavaş sistem belleğine (LPDDR/DDR) taştığında veri aktarım hızında yaşanan sert düşüş.
- **HBM3e (High-Bandwidth Memory 3 Extended):** Silikon ara katman üzerinde doğrudan GPU kalıplarının yanına 3D yığılmış ve 7,1 TB/s bant genişliğine ulaşan yüksek hızlı DRAM mimarisi.
- **Operasyonel Yoğunluk ($\alpha$):** Bir hesaplama çekirdeğinin bellekten okuduğu her bayt veri başına gerçekleştirdiği kayan nokta işlem sayısı ($\text{FLOP/Bayt}$).
- **Roofline Modeli:** Bir algoritmanın ulaşabileceği azami performansı operasyonel yoğunluk ve donanım bant genişliğinin fonksiyonu olarak haritalayan görsel ve analitik başarım modeli.
- **ConnectX-8 SuperNIC:** PCIe Gen6 için optimize edilmiş, RoCEv2 ve InfiniBand protokolleri üzerinden 800 Gb/s hızında çok düğümlü kolektif iletişimi yöneten ağ arabirim kartı.

---

## Kaynaklar ve doğrulama bağlantıları

- [K1] [NVIDIA DGX Station Datasheet (Belge ID: 5543600-web)](https://www.nvidia.com/en-us/data-center/dgx-station/): GB300 Grace Blackwell Ultra Desktop Superchip iş istasyonu için resmi donanım şartnamesi.
- [K2] [NVIDIA Blackwell Architecture Technical Brief](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/): Beşinci nesil Tensor Çekirdekleri, NVFP4 formatları ve çift kalıplı NV-HBI ölçeklemesini detaylandıran teknik rapor.
- [K3] [NVIDIA Grace CPU Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/grace-cpu/): 72 çekirdekli Arm Neoverse V2 mimarisi, LPDDR5X bellek alt sistemi ve NVLink-C2C tutarlı ara bağlantısı üzerine derinlemesine mimari rapor.
- [K4] [Williams, S., Waterman, A., & Patterson, D. (2009). Roofline: An Insightful Visual Performance Model for Multicore Architectures. Communications of the ACM](https://doi.org/10.1145/1498765.1498785): Aritmetik yoğunluk ve operasyonel roofline sınırlarını kuran seminal akademik makale.
- [K5] [GPU Mimarisi ve LLM Ölçekleme](post.html?slug=gpu-mimarisi-ve-llm-olcekleme): Yapay Zekâ Altyapısı serimizin SM'ler, warplar ve bellek hiyerarşisini ele alan 1. Bölümü.
- [K6] [Kurumsal AI Donanım Kümesi Nasıl Tasarlanır?](post.html?slug=kurumsal-ai-platform-mimarisi): Küme topolojisi, güç altyapısı ve 70B referans tasarımını inceleyen serimizin 5. Bölümü.
