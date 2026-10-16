Kâğıt üzerinde 127 adet DGX H100 düğümünden oluşan bir NVIDIA DGX SuperPOD muazzam bir mühendislik yapısıdır: 1.016 H100 GPU, 1,0 ExaFLOP/s yoğun (dense) BF16 hesaplama gücü, 81 TB HBM3 bellek, 51,2 TB/s biseksiyon bant genişliğine sahip bir InfiniBand omurga ağı ve 1,39 MW güç altyapısı. Ancak bu devasa makineyi kavramak, yalnızca tepe FLOP/s sayılarına bakmakla mümkün değildir. Bir yapay zekâ süper bilgisayarı; silikon düzeyindeki transistörlerden kasa içi çapraz bağlantılara, küme anahtarlama kumaşından doğrudan depolama yollarına kadar uzanan kusursuz bir donanım hiyerarşisidir.

Bu sistemin her katmanı bir sonrakine fiziksel olarak bağlanır: SM içindeki Tensor Core'lar HBM3 yığınlarından beslenir, GPU'lar kasa içinde 450 GB/s'lik NVLink hatlarıyla kenetlenir, düğümler raylı fat-tree kumaşıyla birbirine bağlanır ve depolama doğrudan DMA ile ana işlemciyi baypas eder. Bu parçaların her biri bağımsız birer teknoloji harikasıdır; ancak yapay zekâ iş yüklerinde asıl belirleyici olan, bu donanım katmanlarının birbirine nasıl bağlandığı ve verinin bu hiyerarşi içinde hangi fiziksel hızlarla aktığıdır.

Bu yazı, söz konusu donanım mimarisini tek ve bütüncül bir haritaya yerleştiriyor. Şekil 1.1'deki mimari poster; tek bir Streaming Multiprocessor'dan (SM) başlayarak H100 çipine, 8 GPU'lu DGX H100 kasasına, 127 düğümlük DGX SuperPOD'a ve 16.384 GPU'luk devasa bir veri merkezi altyapısına uzanan donanım anatomisini adım adım açıyor. Yazının görevi doğrudan donanım yapısını öğretmektir: makine nelerden oluşur, parçalar birbirine nasıl bağlanır ve her bayt hangi yoldan geçer. Serinin ikinci yazısı olan [Yapay Zekâ Altyapısı (2): Bir AI Süper Bilgisayarının Teşhisi — 1.024 H100 Neden %38 MFU'da Kalır?](post.html?slug=gpudan-ai-super-bilgisayarina) ise bu donanım yapısının büyük ölçekli model eğitimi altındaki davranışını, 4D paralelliği, operasyonel arızaları ve %38 MFU teşhisini ele almaktadır.

**Bu yazıda**

- [1. Haritayı okumak: üç sütun, beş bölüm](#1-haritayı-okumak-üç-sütun-beş-bölüm)
- [2. 1. Sütun — Hesaplama: SM'in ve H100 çipinin içi](#2-1-sütun-hesaplama-smin-ve-h100-çipinin-içi)
- [3. 2. Sütun — Ağ: NVLink kasasından SuperPOD omurgasına](#3-2-sütun-ağ-nvlink-kasasından-superpod-omurgasına)
- [4. 3. Sütun — Depolama: Doğrudan DMA ve yüksek başarımlı veri yolu](#4-3-sütun-depolama-doğrudan-dma-ve-yüksek-başarımlı-veri-yolu)
- [5. Sentez: Donanım hiyerarşisi ve hız farkları](#5-sentez-donanım-hiyerarşisi-ve-hız-farkları)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Haritayı okumak: üç sütun, beş bölüm

Modern bir yapay zekâ süper bilgisayarı, yalnızca tek tek güçlü GPU'ların yan yana konulmasıyla kurulamaz; silikon hesaplamadan kasa içi ara bağlantılara, küme ağı kumaşından yüksek hızlı depolama hatlarına kadar entegre bir **ortak tasarım (co-design)** bütünüdür. Bu devasa makine üç fiziksel donanım sütunu üzerinde yükselir:

1. **Hesaplama Sütunu (Compute):** Çip içi hiyerarşi — Streaming Multiprocessor (SM), Warp Scheduler'lar, Tensor Core'lar ve 3,35 TB/s bant genişliğine sahip HBM3 bellek yığınları.
2. **Ağ Sütunu (Fabric / Interconnect):** Kasa içi ölçekleme (scale-up) için 450 GB/s'lik (tek yön) NVLink 4 ve NVSwitch matrisi; kasalar arası ölçekleme (scale-out) için 50 GB/s'lik (400 Gb/s) non-blocking rail-optimized InfiniBand fat-tree kumaşı.
3. **Depolama Sütunu (Storage):** Ana işlemciyi ve host bellek kopyalarını atlayarak NVMe sürücülerden doğrudan GPU HBM'ine veri aktaran GPUDirect Storage (GDS) ve ayrık depolama ağı.

Bu üç sütun, mikroskobik thread düzeyinden binlerce sunuculuk veri merkezine kadar birbirini tamamlayan bir fiziksel ölçek merdiveninde birleşir:

| Seviye | Boyut | Donanım Bileşeni ve Temel Mimari Rolü |
| :--- | :--- | :--- |
| **Thread** | Programın en küçük mantıksal örneği | Bağımsız register alanına sahip skaler CUDA yürütme hattı |
| **Warp** | 32 thread | SM warp scheduler tarafından eşzamanlı (SIMT) yürütülen temel çizelgeleme birimi |
| **SM** | En çok 64 warp, 2.048 thread | 4 alt bölme (SMSP), Tensor Core'lar, 256 KiB register file ve 228 KiB L1/Shared Memory |
| **H100 SXM5** | 132 SM, 80 GB HBM3 | 990 TFLOP/s dense BF16 hesaplama, 50 MB L2 önbellek ve 3,35 TB/s HBM3 bant genişliği |
| **DGX H100** | 8 GPU, 4 NVSwitch | Kasa içinde 3,6 TB/s all-to-all NVLink crossbar kumaşı ve çift soket CPU |
| **Ölçeklenebilir Birim (SU)** | 32 DGX, 256 GPU | 8 bağımsız leaf switch düzlemine dağıtılan rail-aligned bağlantı |
| **SuperPOD** | 127 DGX, 1.016 GPU | İki katmanlı, engellemesiz (non-blocking) rail-optimized InfiniBand fat-tree |
| **Çok Pod'lu Küme** | 16.384 H100 | Veri merkezi ölçeğinde 3 katmanlı spine kumaşı ve ayrık depolama ağı |

Sayılar hakkında bir not: 32 sunuculu dört SU kâğıt üzerinde $128 \times 8 = 1.024$ GPU eder; ancak NVIDIA referans tasarımında bir sunucu yuvası küme yönetim ve UFM cihazlarına ayrıldığından, kurulu bir SuperPOD tam olarak 127 DGX düğümü ve 1.016 GPU barındırır.


Aşağıdaki poster üç sütunu tek bir referans sistem üzerinde çizer: NVIDIA H100 SXM5 → DGX H100 → InfiniBand fat-tree'li DGX SuperPOD. Başka her hızlandırıcı veya küme ya parantez içinde ya da kendine ait bölümde durur:

<svg viewBox="0 0 840 19412" role="img" aria-label="Yapay zekâ altyapısını anlatan beş bölümlük bir poster. A bölümü dört basamakta uzaklaşır. 1. basamak, H100 SXM5&#x27;teki 132 SM&#x27;den biri olan Streaming Multiprocessor; metindeki SM listesiyle aynı numaralarla: dört özdeş alt bölüm, her birinde (1) döngüde 1.024 BF16 ya da 2.048 FP8 FLOP yapan 4. nesil bir Tensor Core, (2) her döngü 16 warp&#x27;tan birinin komutunu veren bir warp scheduler, (3) 32 FP32 CUDA core ile 16 INT32 ve 16 FP64 birimi, (4) 16.384 register&#x27;lık 64 KiB&#x27;lık register file; dördü 256 KiB SMEM ve L1&#x27;i paylaşır, TMA kopya motoru bu alanı L2 ve HBM&#x27;den besler. 528 Tensor Core × 1.024 FLOP × 1,83 GHz yaklaşık 989 TFLOP/s eder. Hopper&#x27;da ölçülen yükleme gecikmeleri SMEM için yaklaşık 29, L1 için 41, L2 için 263, HBM için 479 döngüdür; scheduler bu beklemeyi her döngü hazır bir warp&#x27;a bedavaya geçerek gizler. Bir zaman çizgisi, 2 birim hesaplayıp 6 birim bekleyen dört warp&#x27;ın scheduler&#x27;ı hep meşgul tuttuğunu, iki warp&#x27;ın ise onu zamanın yarısında boşta bıraktığını gösterir; gereken warp yaklaşık 1 artı bekleme bölü hesaptır. Thread başına register arttıkça yerleşik warp azalır: 32 register 64 warp, 64 register 32, 128 register 16, 255 register 8 warp bırakır; FlashAttention-3 gibi hızlı Hopper çekirdekleri ise az warp ile çalışıp gecikmeyi TMA kopyaları ve wgmma ile gizler. Mutfak modelinde SM sipariş fişli bir hat aşçısı, register ve SMEM tezgâh, Tensor Core bıçak ve ocak, HBM ise ortak kilerdir. 2. basamak, metindeki bellek hiyerarşisi ve roofline alt bölümüne bağlı H100 SXM5 çipi: TSMC 4N&#x27;de 80 milyar transistörlü 814 milimetrekarelik GH100 kalıbı; her biri 9 TPC&#x27;li ve TPC başına 2 SM&#x27;li 8 GPC, 144 yuvanın 132&#x27;si verim için açık; 18 bağlantılı NVLink 4 kapısı ve PCIe Gen5 x16 kapısı; iki bölümlü 50 MB L2; 6 yuvada her biri iki 512-bit bellek denetleyicisinin arkasında 16 GB&#x27;lık 5 etkin HBM3 yığını. Tam çipte 144 SM, 60 MB L2 ve 6 yığın vardır. Pin başına yaklaşık 5,2 Gb/s ile 5.120 bit 3,35 TB/s eder. Bellek piramidi SM başına 256 KiB register ve 256 KiB SMEM ya da L1&#x27;den, çip genelinde her biri yaklaşık 33 MB, yaklaşık 263 döngü ve 5,5 TB/s&#x27;lik 50 MB L2&#x27;ye ve yaklaşık 479 döngü ve 3,35 TB/s&#x27;lik 80 GB HBM3&#x27;e iner. Log-log bir roofline, 990 ile 3,35 çarpı alfanın küçüğü, kırılma noktasını alfa 295,5 FLOP/bayt&#x27;ta yapar: alfa 0,17&#x27;deki BF16 toplama ve alfa yaklaşık 1&#x27;deki batch 1 decode bellek sınırlıdır, batch 128 420 TFLOP/s&#x27;e ulaşır, batch 4.096 990 tavanındadır. Mutfak modelinde GPU 132 aşçılı mutfak, L2 ortak raf, HBM 5 kapılı kiler, NVLink ve PCIe servis kapısıdır. 8 GPU bir DGX H100 düğümü eder: 4 NVSwitch, GPU başına 900 GB/s NVLink, 3,6 TB/s biseksiyon, PCIe Gen5 üzerinden 64 GB/s ile bağlı 2 TB RAM&#x27;li 2 Xeon işlemci, GPU başına bir 400 Gb/s hesaplama NIC&#x27;i, depolama ve yönetim için 2 çift portlu NIC ve 8 NVMe sürücü. Metindeki intra-node ağ alt bölümüne bağlı düğüm kartı 450 GB/s&#x27;i 25 GB/s&#x27;lik 18 bağlantıdan, 3,6 TB/s biseksiyonu 4 GPU çarpı 450 GB/s çarpı 2&#x27;den türetir; bir bağlantı matrisi her GPU&#x27;nun NVSwitch 0–3&#x27;e 5, 4, 4 ve 5 NVLink indirdiğini, switch&#x27;lerin 144 bağlantının 40, 32, 32 ve 40&#x27;ını aldığını gösterir; ağ engelsizdir, 64 portlu her NVSwitch çift yönde 3,2 TB/s taşır ve NVLink SHARP ile toplamayı kendisi yapabilir. GPU başına NVLink 450 GB/s, PCIe 64 GB/s, NIC 50 GB/s verir: TP AllReduce katmanda iki kez NVLink&#x27;i, PP aktivasyonları ve DP gradyanları leaf i&#x27;ye giden NIC&#x27;i, veri ve kontrol noktaları CX-7 depolama ağını, veri yükleyici aynı soketten PCIe&#x27;yi kullanır. Metindeki TP kuralına göre Y, NVLink&#x27;te 28.672 bölü 2.200, yaklaşık 13&#x27;e kadar çıkabilir ama ağda yalnızca 1,45&#x27;tir; TP 8 bu yüzden kasada kalır. Restoran modelinde düğüm sekiz mutfaklı bir restoran, NVLink iç koridor, CPU&#x27;lar iki müdür, NIC&#x27;ler kurye kapılarıdır. 32 düğüm bir ölçeklenebilir birim, 4 birim 127 düğümlük, 1.016 GPU&#x27;luk bir DGX SuperPOD eder: her düğümün i. NIC&#x27;inin i. leaf&#x27;e gittiği ray uyumlu fat-tree&#x27;de 32 leaf ve 16 spine switch, düğüm başına en az 40 GB/s veren ayrı bir depolama ağı ve in-band ile OOB Ethernet yönetim ağları. Metindeki scale-out alt bölümüne bağlı küme kartı, A düğümünün GPU 3&#x27;ünün B düğümünün GPU 3&#x27;üne leaf, spine ve leaf üzerinden 3 sıçramada ulaştığı raf üstü düzeni, her düğümün NIC i&#x27;sinin leaf i&#x27;ye gittiği ve aynı mesajın 1 sıçramada ulaştığı NVIDIA ray düzeniyle karşılaştırır; TP 8 kasada kaldığı için DP ve PP trafiği aynı numaralı GPU&#x27;lar arasında akar. Bir ölçek merdiveni BF16 hesabı SM başına 7,5 TFLOP/s&#x27;den H100&#x27;de 990&#x27;a, DGX&#x27;te 7,9 PFLOP/s&#x27;e, SU&#x27;da 253&#x27;e, SuperPOD&#x27;da 1,0 EFLOP/s&#x27;e ve 64 SU&#x27;da 16,2 EFLOP/s&#x27;e çıkarır; HBM 80 GB&#x27;tan 1,31 PB&#x27;a büyürken dışarıya açılan yol GPU başına 3,35 TB/s&#x27;den 450 ve 50 GB/s&#x27;e daralır. Restoran zinciri modelinde leaf&#x27;ler mahalle santrali, spine&#x27;lar şehirlerarası hat, depolama ve yönetim ise depo ve müdürlüktür. B bölümü Llama 3 405B&#x27;nin 16.384 GPU&#x27;daki tek bir eğitim adımını izler. Bir 4D paralellik kartı bölmeyi anlatır. Sorun: Llama 3 405B&#x27;yi tek bir H100&#x27;de eğitmek yaklaşık 3.000 yıl sürer ve 6.480 GB&#x27;lık eğitim durumu GPU&#x27;nun 80 GB&#x27;ının 81 katıdır. Dört adımın her biri bir sorunu çözer: TP 8 her katmanı bir düğümün 8 GPU&#x27;suna böler, GPU başına 6.480&#x27;den 810 GB&#x27;a, 3.000&#x27;den 380 yıla; PP 16 126 katmanı 16 düğüme dağıtıp 128 GPU&#x27;luk bir zincir kurar, 810&#x27;dan 50,6 GB&#x27;a ve 24 yıla; DP 128 zinciri kopyalar, 16.384 GPU ve yaklaşık 67 gün; FSDP optimizer durumunu kopyalara paylaştırır, 50,6&#x27;dan 12,9 GB&#x27;a. Bir özet grafiği her adımı 80 GB çizgisine karşı gösterir. TP 8 bir düğümdeki GPU&#x27;lar, PP 16 bir zincirdeki düğümler, DP 128 kopyalardır; her GPU&#x27;nun adresi [TP sırası, PP sırası, DP sırası] olur ve aynı TP sırasındaki GPU&#x27;lar aynı raydadır. Bir tablo beş boyutun her birinin neyi böldüğünü, bir GPU&#x27;nun ne yaptığını, nerede olduğunu, ne sıklıkla konuştuğunu ve neden gerektiğini, ne kadar sık konuşan o kadar içeride kuralıyla verir; MoE için EP ve uzun bağlam için CP kaynağa göre dördüncü boyuttur. Dört resim küçük örneği Llama 3&#x27;teki karşılığıyla karşılaştırır. Sekiz adım veriyi izler: 1 adım başına 67 MB token okunur; 2 ileri geçiş yaklaşık 2,1 s; 3 NVLink üzerinde 268 MB&#x27;lık tensör paralel AllReduce yaklaşık 500 kez, her biri 1 ms; 4 mikro-batch başına 34 MB&#x27;lık aktivasyon dilimi raydan gönderilir; 5 geri geçiş yaklaşık 4,1 s; 6 FSDP gradyan senkronu 12,7 artı 6,3 GB, yaklaşık 0,38 s ve örtüşür; 7 AdamW her GPU&#x27;nun 24,7 milyon parametresini HBM&#x27;de günceller; 8 yaklaşık 14 dakikada bir 7,5 TB&#x27;lık kayıt 250 GB/s&#x27;te ~31 s&#x27;de yazılır. Bütçe grafiği 6,2 s&#x27;lik hesabı tepe hızdaki 2,5 s, 0,53 s NVLink, 0,38 s ağ senkronu, 0,23 s kayıt payı ve 0,02 s boru hattı gönderimiyle karşılaştırır. C bölümü üç duvarı hesaplanmış eğrilerle gösterir. Bellek duvarı: H100&#x27;de Llama 3 70B MLP katmanı için elde edilen hız batch 1&#x27;de 3,3 TFLOP/s&#x27;den batch 128&#x27;de 420&#x27;ye çıkar ve batch 310 civarında 990 tavanına ulaşır; çare büyük batch, kernel füzyonu ve decode&#x27;da düşük bitli ağırlıktır. Ağ duvarı: GPU başına bant genişliği HBM&#x27;de 3,35 TB/s&#x27;den NVLink&#x27;te 450 GB/s&#x27;ye, düğüm dışında 50 GB/s&#x27;ye düşer; tensör paralelliği NVLink&#x27;te 13 GPU&#x27;ya, yani 8&#x27;e sığar, InfiniBand&#x27;de yalnızca 1,45&#x27;e; 16 aşamalı bir boru hattı 16 mikrobatch ile yüzde 48, 128 ile yüzde 10,5 boşta kalır; çare TP&#x27;yi düğüm içinde tutmak, SHARP ve iletişimi hesapla çakıştırmaktır. G/Ç duvarı: kayıt aralığına göre Young-Daly kayıp eğrileri; 25 GB/s&#x27;de en düşük kayıp 42,9 dakikada yüzde 24, 250 GB/s&#x27;de 13,6 dakikada yüzde 7,5&#x27;tir; çare GPUDirect Storage, asenkron yazma ve karekök aralığıdır. D bölümü aynı 140 GB&#x27;ın taşınma süresini verir: HBM&#x27;den 42 milisaniye, NVLink&#x27;ten 0,31 saniye, PCIe&#x27;den 2,2 saniye, GPU başına InfiniBand&#x27;den 2,8 saniye, bütün kümenin paylaştığı eski depolamadan 5,6 saniye; 8 H100&#x27;de decode saniyede yaklaşık 190 tokenle sınırlıdır. E bölümü Meta&#x27;nın Llama 3 405B eğitimini özetler: GPU başına 400 TFLOP/s ve yüzde 41 MFU ile 16.384 H100, 3.072 GPU&#x27;luk pod&#x27;lardan oluşan bir RoCE ağı, 2 TB/s sürekli ve 7 TB/s tepe hızlı 240 PB&#x27;lık Tectonic depolama ve 54 günde 466 kesinti, yani yaklaşık 3 saatte bir arıza. En yavaş halka bütün kümenin hızını belirler. Küme basamağından sonra bir yapı taşları kartı bir DGX&#x27;in dört ağını gösterir: 48 QM9700&#x27;lük hesaplama ağına 8 × 400 Gb/s InfiniBand (400 GB/s), 16 QM9700&#x27;lük depolama ağına 2 × 400 Gb/s (100 GB/s, hedef en az 40 GB/s), 8 SN4600C in-band switch&#x27;ine 2 × 100 GbE ve BMC&#x27;den 8 SN2201 OOB switch&#x27;ine 1 GbE; her biri trafiği AllReduce&#x27;u yavaşlatmasın diye ayrıdır. Parça listesi her adedi kaynağıyla verir: 4 SU × 32 yuvadan UFM çifti için bir eksik 127 DGX, 32 leaf artı 16 spine, 4 UFM cihazı, 5 yönetim sunucusu, 38 kabin ve 108 PDU. Güç ortalama 1,39 MW eder; bunun 1,295 MW&#x27;ı, yüzde 92,9&#x27;u 10,2 kW&#x27;lık 127 DGX&#x27;ten gelir. Depolama token değil kontrol noktası taşır: 1.016 GPU&#x27;da Llama 3 70B yalnız saniyede yaklaşık 3,8 MB token okur, NVIDIA&#x27;nın NLP hedefi 60 GB/s&#x27;tir; 1,12 TB&#x27;lık kayıt ise eğitimi 30, 80 ya da 250 GB/s&#x27;te 37, 14 ya da 4,5 saniye durdurur. Altı yazılım adımı bir işi taşır: Base Command Platform gönderir, Slurm düğümleri ayırır, Base Command Manager aynı imajı kurar, NGC ve AI Enterprise aynı yığını başlatır, Magnum IO veriyi NCCL, SHARP ve GPUDirect Storage ile taşır, UFM arızayı bulur; Llama 3&#x27;te 419 kesintinin yalnız 3&#x27;ü elle müdahale istedi. Restoran modelinde depolama tedarik hattı, in-band servis koridoru, OOB elektrik panosu, zamanlayıcılar ise vardiya amiridir. Düğüm basamağı NVIDIA&#x27;nın DGX H100 sistem topolojisini yeniden çizer: UPI ile bağlı CPU 0 ve CPU 1; CPU 0 GPU 0–3 ile NIC 0–3&#x27;ü, CPU 1 GPU 4–7 ile NIC 4–7&#x27;yi besler ve her GPU&#x27;nun kendi ağ kartı vardır; üstteki iki PCIe switch çift portlu depolama kartlarını ve 4&#x27;er NVMe diski taşır; CPU 0&#x27;da iki M.2 işletim sistemi diski, CPU 1&#x27;de bir 100 GbE kartı bulunur; her GPU 18 NVLink&#x27;ini 5, 4, 4 ve 5 olarak dört NVSwitch&#x27;e gönderir; alttaki PCIe switch NVSwitch&#x27;leri yönetim için CPU 0&#x27;a bağlar. Düğüm şemasındaki altı numaralı daire, altındaki altı soruyla eşleşir: kim kimi besler (iki NUMA yarısı), GPU ağa nasıl çıkar (GPUDirect RDMA ile kendi NIC&#x27;i), GPU&#x27;lar nasıl konuşur (dört NVSwitch&#x27;e 18 NVLink), depolama trafiği nereden geçer (üstteki PCIe switch&#x27;ler), iki yarıyı ne birleştirir (CPU iletişimi, UPI) ve NVSwitch&#x27;leri kim yönetir (CPU 0&#x27;dan PCIe ile Fabric Manager). Küme basamağı H100 SXM5, DGX H100, 32 DGX&#x27;lik ölçeklenebilir birim ve 4 birimli DGX SuperPOD hiyerarşisiyle açılır. Metindeki scale-out ve kolektif iletişim alt bölümlerine bağlı yakından inceleme kartı bir leaf&#x27;in portlarını sayar: 32&#x27;si düğümlere iner, 32&#x27;si 16 spine&#x27;ın her birine iki kabloyla çıkar, toplam 1.024 leaf–spine kablosu vardır ve ağ engelsizdir: 16 × 64 × 50 GB/s = 51,2 TB/s spine kapasitesine karşı GPU&#x27;lar 1.016 × 50 GB/s = 50,8 TB/s gönderir. Dört yol bedelleriyle karşılaştırılır: aynı SU&#x27;da aynı ray 1 switch, başka SU&#x27;da aynı ray ve raylar arası trafik 3 switch ister; NCCL&#x27;in PXN&#x27;i raylar arası trafiği bir NVSwitch geçişi artı 1 switch&#x27;e çevirir, DeepSeek-V3 de tokenları önce IB ile aynı indeksli GPU&#x27;ya göndererek bunun aynasını yapar. GPU başına 17,5 GB&#x27;lık Llama 3 70B gradyanı 127 rank arasında halka AllReduce ile 0,69 saniyede kabloya 34,7 GB, SHARP ile 0,35 saniyede 17,5 GB basar. İki katman 64 leaf × 32 = 2.048 GPU&#x27;da durur; 16 SU&#x27;dan itibaren core katmanı eklenir, 2.048 DGX 512 leaf, 512 spine ve 256 core switch kullanır ve en uzak yol 3&#x27;ten 5 switch&#x27;e uzar. Restoran zinciri modelinde raylar aynı numaralı mutfaklar, PXN koridor, SHARP santralde oy sayımı, core katmanı ise ülke omurgasıdır. Bir not SuperPOD&#x27;un neden 128 değil 127 DGX olduğunu açıklar: bir çift UFM cihazı bir DGX yuvasını alır; daha küçük kurulumlar tek birimden başlar, 31 ile 127 DGX arasındadır. Küme çizimi NVIDIA&#x27;nın hesaplama ağı şemasını izler: üstte Spine-01&#x27;den Spine-16&#x27;ya; Leaf-01-01–08 ve H100-001–032 düğümleriyle SU 1; Leaf-02-01–08 ve H100-033–064 düğümleriyle SU 2; aynı düzeni tekrarlayan ve çizimde atlanan SU 3; Leaf-04-01–08, H100-097–127 düğümleri ve C-UFM 1 ile 2 UFM çiftiyle SU 4; spine–leaf hatları 2 × 400 Gb/s, DGX–leaf hatları 400 Gb/s, UFM–leaf hatları 200 Gb/s&#x27;dir. Hiyerarşinin altındaki &#x27;ne neyi yapar&#x27; tablosu her basamağı açıklar: H100 matematiği yapar ve modelin kendine düşen payını HBM&#x27;de tutar; DGX 8 GPU&#x27;yu tensör paralelliğinin yaşadığı tek bir NVLink alanında birleştirir, işlemcileri işletim sistemini ve veri yükleyiciyi koşturur; SU, aynı numaralı GPU&#x27;ların tek sıçramada buluştuğu 32 DGX ve 8 leaf switch&#x27;lik tekrar eden birimdir; SuperPOD SU&#x27;ları spine ile birleştirir, depolama, yönetim ve yazılımı ekler ve veri paralelliğini bütün kümeye yayar. Bir tanım kutusu leaf ve spine&#x27;ı açıklar: leaf, sunucuların takıldığı alt katman switch&#x27;idir, SU başına 8 tane, her ray için bir tane, 32 portu aşağı 32 portu yukarı; spine, leaf&#x27;leri birbirine bağlayan ve hiç sunucu taşımayan üst katman switch&#x27;idir, toplam 16 tane, her biri 32 leaf&#x27;in hepsine bağlıdır; her leaf her spine&#x27;a ulaştığı için herhangi iki sunucu en fazla leaf, spine, leaf yoluyla buluşur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bp-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bp-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="bp-wrn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="bp-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="16" y="27" text-anchor="start" style="fill:var(--c-accent);font-size:17px;font-weight:700">Büyük Resim: Yapay Zekâ Altyapısının 3 Temel Sütunu</text>
<text x="16" y="48" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Silikon düzeyinden 16.384 GPU'luk dev kümeye: Donanım katmanları, veri akış yolları ve fiziksel darboğazlar</text>
<rect x="16" y="60" width="808" height="197" rx="10" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="40" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="108.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">A · Donanım</text>
<text x="108.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SM → SuperPOD</text>
<line x1="178.0" y1="96.0" x2="194.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="196" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="264.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">B · Eğitim Adımı</text>
<text x="264.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Uçtan Uca Veri Yolu</text>
<line x1="334.0" y1="96.0" x2="350.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="352" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="420.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">C · 3 Duvar</text>
<text x="420.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Memory · Fabric · I/O</text>
<line x1="490.0" y1="96.0" x2="506.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="508" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="576.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">D · Hız &amp; Gecikme</text>
<text x="576.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Bant Genişliği Farkı</text>
<line x1="646.0" y1="96.0" x2="662.0" y2="96.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="664" y="74" width="136" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="732.0" y="92" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">E · LLaMA 3 405B</text>
<text x="732.0" y="109" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16k GPU Üretim Kümesi</text>
<text x="40" y="144" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bu mimari harita; bir LLM eğitim iterasyonunda donanım bileşenlerinin mikroskobik ölçekten veri merkezi ölçeğine nasıl etkileşime girdiğini,</text>
<text x="40" y="161" text-anchor="start" style="fill:var(--c-text);font-size:12px">verinin hangi yollardan aktığını, kümenin hangi fiziksel sınırlara çarptığını ve bu engellerin endüstride nasıl aşıldığını modeller.</text>
<rect x="28.0" y="175.0" width="784.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Referans Mimari: NVIDIA H100 SXM5 (Çip) → DGX H100 (Sunucu Düğümü) → DGX SuperPOD (InfiniBand Fat-Tree).</text>
<text x="40" y="211" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Diyagramdaki tüm FLOP/s, bant genişliği ve gecikme metrikleri bu sisteme aittir; alternatif donanımlar kıyaslama amacıyla verilmiştir.</text>
<rect x="40.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="56" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hesaplama (Compute)</text>
<rect x="195.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="211.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Ağ ve Fabric (Network)</text>
<rect x="355.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="371.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Depolama ve I/O (Storage)</text>
<rect x="525.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="541.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Darboğaz (Wall / Bottleneck)</text>
<rect x="710.0" y="236" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="726.0" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Mimari Çözüm</text>
<rect x="16" y="269" width="808" height="12969" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="293" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="297.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">A</text>
<text x="56" y="298" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Yakınlaştır: bir SM&#x27;den bütün kümeye</text>
<text x="28" y="319" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her basamak öncekinin ×N kopyasıdır; koyu vurgu bir önceki basamağı gösterir.</text>
<rect x="28" y="331" width="784" height="1815" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="357" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">1 · SM: GPU&#x27;nun temel işçisi (Streaming Multiprocessor)</text>
<text x="40" y="377" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">H100 SXM5&#x27;te 132 SM vardır ve her SM 4 özdeş alt bölümden (SMSP) oluşur. Numaralı daireler, metinde SM&#x27;i anlatan 1–4</text>
<text x="40" y="394" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">numaralı maddelerdir: Tensor Core, warp scheduler, FP32 CUDA core ve register.</text>
<text x="40" y="424" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Donanım: bir alt bölümün içi</text>
<rect x="40" y="438" width="430" height="324" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="458" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">SM · 132&#x27;nin biri</text>
<text x="460" y="458" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">4 alt bölüm (SMSP 0–3)</text>
<rect x="50" y="468" width="410" height="186" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="60" y="486" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Alt bölüm 0 (SMSP 0) · yakından</text>
<rect x="82.0" y="496.0" width="368.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="266.0" y="510" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Warp scheduler + dispatch</text>
<text x="266.0" y="524" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">her döngü 16 warp&#x27;tan birinin komutunu verir</text>
<circle cx="68" cy="513" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="68" y="517.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<line x1="266.0" y1="531.0" x2="266.0" y2="541.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="82.0" y="542.0" width="368.0" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="266.0" y="556" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Register file · 16.384 × 32 bit = 64 KiB</text>
<text x="266.0" y="570" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">thread&#x27;lerin değişkenleri; işlenenler buradan okunur</text>
<circle cx="68" cy="559" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="68" y="563.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<line x1="187.0" y1="577.0" x2="187.0" y2="587.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="384.0" y1="577.0" x2="384.0" y2="587.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="82.0" y="588.0" width="210.0" height="56" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="187" y="604" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Tensor Core (4. nesil)</text>
<text x="187" y="620" text-anchor="middle" style="fill:var(--c-text);font-size:11px">BF16: 1.024 FLOP/döngü (8×8×8)</text>
<text x="187" y="635" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">FP8: 2.048 FLOP/döngü (yoğun)</text>
<circle cx="68" cy="616" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="68" y="620.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<rect x="318.0" y="588.0" width="132.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="384.0" y="601.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">32 × FP32 CUDA core</text>
<rect x="318.0" y="609.0" width="132.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="384.0" y="622.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">16 INT32 · 16 FP64</text>
<text x="384" y="641" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">GELU · SiLU · RMSNorm</text>
<circle cx="305" cy="597" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="305" y="601.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<rect x="50" y="662" width="132" height="50" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="60" y="678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Alt bölüm 1</text>
<rect x="58.0" y="686.0" width="40.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="78.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sched</text>
<rect x="102.0" y="686.0" width="46.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="125.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">64 KiB</text>
<rect x="152.0" y="686.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="165.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TC</text>
<rect x="189" y="662" width="132" height="50" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="199" y="678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Alt bölüm 2</text>
<rect x="197.0" y="686.0" width="40.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="217.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sched</text>
<rect x="241.0" y="686.0" width="46.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="264.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">64 KiB</text>
<rect x="291.0" y="686.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="304.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TC</text>
<rect x="328" y="662" width="132" height="50" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="338" y="678" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Alt bölüm 3</text>
<rect x="336.0" y="686.0" width="40.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="356.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sched</text>
<rect x="380.0" y="686.0" width="46.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="403.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">64 KiB</text>
<rect x="430.0" y="686.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="443.0" y="699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TC</text>
<rect x="50.0" y="720.0" width="410.0" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="255.0" y="734" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Shared memory (SMEM) / L1 · 256 KiB</text>
<text x="255.0" y="748" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 alt bölümün ortak alanı; en çok 228 KiB&#x27;ı SMEM olarak ayrılır</text>
<line x1="255.0" y1="765.0" x2="255.0" y2="785.0" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.6"/>
<text x="267" y="779" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">TMA: asenkron kopya</text>
<rect x="40.0" y="788.0" width="430.0" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="255.0" y="803.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SM&#x27;nin dışında: L2 (50 MB, bütün SM&#x27;lerin ortak rafı) → HBM3 (80 GB)</text>
<rect x="490" y="438" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="500" y="460" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">4</text>
<text x="500" y="477" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölüm</text>
<rect x="649" y="438" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="659" y="460" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1.024</text>
<text x="659" y="477" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/döngü (TC)</text>
<rect x="490" y="494" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="500" y="516" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">256 KiB</text>
<text x="500" y="533" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">register / SM</text>
<rect x="649" y="494" width="151" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="659" y="516" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">64</text>
<text x="659" y="533" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">resident warp</text>
<text x="490" y="570" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">132 SM ile çarpınca</text>
<text x="490" y="588" text-anchor="start" style="fill:var(--c-text);font-size:11px">528 Tensor Core · 16.896 CUDA core</text>
<text x="490" y="604" text-anchor="start" style="fill:var(--c-text);font-size:11px">~33 MB register + ~33 MB SMEM/L1</text>
<text x="490" y="620" text-anchor="start" style="fill:var(--c-text);font-size:11px">8.448 resident warp (270.336 thread)</text>
<rect x="490.0" y="634.0" width="310.0" height="128" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="502" y="654" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">990 TFLOP/s nereden çıkar?</text>
<text x="502" y="673" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">528 TC × 1.024 FLOP × 1,83 GHz</text>
<text x="502" y="689" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">≈ 989 TFLOP/s ≈ 990 (BF16, yoğun)</text>
<text x="502" y="708" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1,83 GHz, H100&#x27;ün Tensor Core yükündeki saat</text>
<text x="502" y="723" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hızıdır (CUDA core&#x27;lar 1,98 GHz&#x27;e çıkar). FP8&#x27;de</text>
<text x="502" y="738" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">döngü başına iş iki katıdır: ~1.979 TFLOP/s.</text>
<text x="502" y="753" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(B200: 148 SM, TC başına 2.048 FLOP/döngü.)</text>
<text x="490" y="784" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki karşılığı: 2. bölüm, “Silikon seviyesinde GPU</text>
<text x="490" y="799" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">anatomisi”.</text>
<line x1="480.0" y1="438.0" x2="480.0" y2="810.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="850" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Gecikme gizleme: warp scheduler nasıl çalışır?</text>
<text x="40" y="876" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Veri ne kadar uzakta? (bir yüklemenin gecikmesi, döngü)</text>
<text x="40" y="904" text-anchor="start" style="fill:var(--c-text);font-size:11px">register</text>
<text x="112" y="904" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yüklenmez; komut doğrudan okur</text>
<text x="40" y="928" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM</text>
<rect x="112" y="916" width="18.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="136.0" y="928" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">29</text>
<text x="40" y="952" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L1</text>
<rect x="112" y="940" width="25.2" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="143.2" y="952" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">41</text>
<text x="40" y="976" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2</text>
<rect x="112" y="964" width="163.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="281.1" y="976" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">263</text>
<text x="40" y="1000" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM3</text>
<rect x="112" y="988" width="296.9" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="414.9" y="1000" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">479</text>
<text x="40" y="1030" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Hopper (H800) üzerinde ölçülen değerler (Luo vd., 2024). HBM, SMEM&#x27;den ~16</text>
<text x="40" y="1045" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">kat uzaktadır.</text>
<text x="490" y="876" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metnin 2. maddesi: SM 64 warp&#x27;ı yerleşik</text>
<text x="490" y="893" text-anchor="start" style="fill:var(--c-text);font-size:12px">(resident) tutar ve aralarında sıfır ek yüklü</text>
<text x="490" y="910" text-anchor="start" style="fill:var(--c-text);font-size:12px">geçiş yapar. Bunun anlamı şudur: bir warp HBM&#x27;den</text>
<text x="490" y="927" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri istediğinde cevap ~480 döngü sonra gelir; bu</text>
<text x="490" y="944" text-anchor="start" style="fill:var(--c-text);font-size:12px">sürede tek bir Tensor Core 480 × 1.024 ≈ 490 bin</text>
<text x="490" y="961" text-anchor="start" style="fill:var(--c-text);font-size:12px">FLOP yapabilirdi. Scheduler beklemez, her döngü</text>
<text x="490" y="978" text-anchor="start" style="fill:var(--c-text);font-size:12px">hazır bir warp seçer. Bütün warp&#x27;ların</text>
<text x="490" y="995" text-anchor="start" style="fill:var(--c-text);font-size:12px">register&#x27;ları zaten çipte durduğu için CPU&#x27;daki</text>
<text x="490" y="1012" text-anchor="start" style="fill:var(--c-text);font-size:12px">gibi bağlam kaydedilip geri yüklenmez.</text>
<line x1="480.0" y1="864.0" x2="480.0" y2="1045.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1079" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Bir scheduler&#x27;ın havuzu: 16 warp (SM&#x27;de 4 × 16 = 64)</text>
<rect x="40.0" y="1091.0" width="220.0" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="150.0" y="1106.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">scheduler 0 → bu döngüde W5</text>
<rect x="40.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="61.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="87.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="108.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="134.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="155.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="181.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="202.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W3</text>
<rect x="228.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="249.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W4</text>
<rect x="275.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="296.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">W5</text>
<rect x="322.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="343.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W6</text>
<rect x="369.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="390.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W7</text>
<rect x="416.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="437.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W8</text>
<rect x="463.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="484.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W9</text>
<rect x="510.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="531.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W10</text>
<rect x="557.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="578.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W11</text>
<rect x="604.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="625.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W12</text>
<rect x="651.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="672.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W13</text>
<rect x="698.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="719.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W14</text>
<rect x="745.0" y="1131.0" width="42.0" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="766.0" y="1148.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">W15</text>
<path d="M150 1113 V1122 H296.0 V1129" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="40.0" y="1167.0" width="14.0" height="12" rx="5" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60" y="1177" text-anchor="start" style="fill:var(--c-text);font-size:11px">koşuyor</text>
<rect x="124.7" y="1167.0" width="14.0" height="12" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="144.7" y="1177" text-anchor="start" style="fill:var(--c-text);font-size:11px">hazır</text>
<rect x="197.2" y="1167.0" width="14.0" height="12" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="217.2" y="1177" text-anchor="start" style="fill:var(--c-text);font-size:11px">belleği bekliyor</text>
<text x="800" y="1177" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Durumlar temsilîdir. Warp, aynı komutu birlikte yürüten 32 thread&#x27;dir (SIMT).</text>
<text x="40" y="1213" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Zaman çizgisi: aynı iş, farklı sayıda warp</text>
<text x="40" y="1230" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Oyuncak örnek: her warp 2 birim hesaplar, sonra 6 birim belleği bekler.</text>
<text x="40" y="1251" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">4 warp: scheduler hiç boş kalmaz</text>
<text x="40" y="1273" text-anchor="start" style="fill:var(--c-text);font-size:11px">scheduler</text>
<text x="40" y="1297" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="112.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="149.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="256.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="293.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="40" y="1317" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="112.0" y="1311" width="36.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="148.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="185.0" y="1308" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="292.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="329.0" y="1308" width="70.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="40" y="1337" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="112.0" y="1331" width="72.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="184.0" y="1327" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="221.0" y="1328" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="328.0" y="1327" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="365.0" y="1328" width="34.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="40" y="1357" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W3</text>
<rect x="112.0" y="1351" width="108.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="220.0" y="1347" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="257.0" y="1348" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="364.0" y="1347" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="112.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="148.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="166.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="184.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="202.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="220.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="238.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W3</text>
<rect x="256.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="274.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="292.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="310.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="328.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="346.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W2</text>
<rect x="364.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="382.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W3</text>
<line x1="112.0" y1="1373.0" x2="400.0" y2="1373.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="112" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="256" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="400" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="400" y="1401" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">zaman →</text>
<text x="432" y="1251" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">2 warp: scheduler zamanın %50&#x27;sinde boşta</text>
<text x="432" y="1273" text-anchor="start" style="fill:var(--c-text);font-size:11px">scheduler</text>
<text x="432" y="1297" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="504.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="541.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="648.0" y="1287" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="685.0" y="1288" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="432" y="1317" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="504.0" y="1311" width="36.0" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="540.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="577.0" y="1308" width="106.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="684.0" y="1307" width="36.0" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="721.0" y="1308" width="70.0" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="504.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="522.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="540.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="558.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="576.0" y="1261.0" width="72.0" height="16" rx="5" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="612.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">boşta</text>
<rect x="648.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="666.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W0</text>
<rect x="684.0" y="1261.0" width="36.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="702.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">W1</text>
<rect x="720.0" y="1261.0" width="72.0" height="16" rx="5" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="756.0" y="1273.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">boşta</text>
<line x1="504.0" y1="1373.0" x2="792.0" y2="1373.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="504" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="648" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="792" y="1387" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="792" y="1401" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">zaman →</text>
<line x1="420.0" y1="1239.0" x2="420.0" y2="1401.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40" y="1417.0" width="18" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="64" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">hesap</text>
<rect x="118.5" y="1421.0" width="18" height="4" rx="2" style="fill:var(--c-success);fill-opacity:.55;stroke:var(--c-success);stroke-width:0"/>
<text x="142.5" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">hazır (sırada)</text>
<rect x="251.89999999999998" y="1418.0" width="18" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="275.9" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">belleği bekler</text>
<rect x="385.29999999999995" y="1417.0" width="18" height="12" rx="2" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1"/>
<text x="409.3" y="1427" text-anchor="start" style="fill:var(--c-text);font-size:11px">scheduler boşta</text>
<rect x="40.0" y="1441.0" width="760.0" height="24" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="420.0" y="1457.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">gereken warp ≈ 1 + bekleme ÷ hesap = 1 + 6 ÷ 2 = 4</text>
<text x="40" y="1489" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bir warp beklerken diğerleri sırayla iş vermelidir. Bekleme hesaba göre ne kadar uzunsa o kadar çok hazır warp gerekir;</text>
<text x="40" y="1506" text-anchor="start" style="fill:var(--c-text);font-size:12px">HBM&#x27;in ~480 döngüsü bu yüzden ya kalabalık bir havuz ya da önceden başlatılmış kopyalar ister.</text>
<text x="40" y="1546" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Darboğaz: havuz boşalınca</text>
<text x="40" y="1572" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Thread başına register arttıkça SM&#x27;e sığan warp azalır</text>
<text x="40" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:11px">32 register/thread</text>
<rect x="182" y="1584" width="190.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="378.0" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">64 warp · %100</text>
<text x="40" y="1620" text-anchor="start" style="fill:var(--c-text);font-size:11px">64 register/thread</text>
<rect x="182" y="1608" width="95.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="283.0" y="1620" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">32 warp · %50</text>
<text x="40" y="1644" text-anchor="start" style="fill:var(--c-text);font-size:11px">128 register/thread</text>
<rect x="182" y="1632" width="47.5" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="235.5" y="1644" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16 warp · %25</text>
<text x="40" y="1668" text-anchor="start" style="fill:var(--c-text);font-size:11px">255 register/thread</text>
<rect x="182" y="1656" width="23.8" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="211.8" y="1668" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">8 warp · %12,5</text>
<text x="40" y="1696" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">warp = 65.536 ÷ (register/thread × 32), en çok 64; 255 register 256&#x27;ya</text>
<text x="40" y="1711" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yuvarlanır.</text>
<text x="490" y="1572" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metnin 4. maddesindeki register&#x27;lar sınırlıdır: SM</text>
<text x="490" y="1589" text-anchor="start" style="fill:var(--c-text);font-size:12px">başına 65.536 tane. Her thread çok register</text>
<text x="490" y="1606" text-anchor="start" style="fill:var(--c-text);font-size:12px">isterse SM&#x27;e daha az warp sığar. Bütün warp&#x27;lar</text>
<text x="490" y="1623" text-anchor="start" style="fill:var(--c-text);font-size:12px">belleği bekliyorsa scheduler&#x27;ın verecek komutu</text>
<text x="490" y="1640" text-anchor="start" style="fill:var(--c-text);font-size:12px">kalmaz; SM durur (stall) ve Tensor Core&#x27;lar boşta</text>
<text x="490" y="1657" text-anchor="start" style="fill:var(--c-text);font-size:12px">döner. Profil araçlarındaki occupancy bu havuzun</text>
<text x="490" y="1674" text-anchor="start" style="fill:var(--c-text);font-size:12px">doluluğudur.</text>
<line x1="480.0" y1="1560.0" x2="480.0" y2="1711.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="1727.0" width="760.0" height="67" rx="5" style="fill:var(--c-success);fill-opacity:.08;stroke:var(--c-success);stroke-width:1.2"/>
<text x="52" y="1747" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-success);font-weight:700">Ama az warp her zaman kötü değildir.</tspan> Hopper&#x27;ın hızlı GEMM ve FlashAttention-3 çekirdekleri bilerek az warp ve çok</text>
<text x="52" y="1764" text-anchor="start" style="fill:var(--c-text);font-size:12px">register kullanır. Gecikmeyi warp sayısıyla değil, önceden başlatılan asenkron kopyalarla gizler: üretici warp&#x27;lar TMA</text>
<text x="52" y="1781" text-anchor="start" style="fill:var(--c-text);font-size:12px">ile veriyi HBM&#x27;den SMEM&#x27;e taşırken tüketici warp&#x27;lar wgmma ile Tensor Core&#x27;da hesap yapar.</text>
<text x="40" y="1834" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Zihinsel model: mutfak</text>
<text x="40" y="1852" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki benzetme: GPU, ortak bir HBM kilerini paylaşan 132 hat aşçısıdır (TPU v5p ise 128×128&#x27;lik dev tahtalarla çalışan iki baş</text>
<text x="40" y="1867" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">aşçıdır).</text>
<rect x="40.0" y="1879" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="1879" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Hat aşçısı</text>
<rect x="52.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SM + warp scheduler</text>
<text x="54.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">Önünde 64&#x27;e kadar sipariş</text>
<text x="54.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">fişi (warp) durur. Bir</text>
<text x="54.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">yemek kilerden malzeme</text>
<text x="54.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">beklerken hazır olan başka</text>
<text x="54.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">bir fişe geçer; malzemesi</text>
<text x="54.0" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:11px">tezgâhta durduğu için geçiş</text>
<text x="54.0" y="2035" text-anchor="start" style="fill:var(--c-text);font-size:11px">bedavadır.</text>
<text x="54.0" y="2068" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">64 warp · 1 komut/döngü</text>
<rect x="233.0" y="1879" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="1879" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="247.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Tezgâh</text>
<rect x="245.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="323.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">register + SMEM/L1</text>
<text x="247.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">Aşçının elinin altı.</text>
<text x="247.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">Register&#x27;daki malzeme</text>
<text x="247.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">anında, SMEM&#x27;deki ~29</text>
<text x="247.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">döngüde gelir; ama alan</text>
<text x="247.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">dardır: SM başına 256 KiB +</text>
<text x="247.0" y="2020" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB.</text>
<text x="247.0" y="2068" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">~29 döngü (SMEM)</text>
<rect x="426.0" y="1879" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="1879" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="440.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Bıçak ve ocak</text>
<rect x="438.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="516.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Tensor Core</text>
<text x="440.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">Tezgâhta malzeme varsa her</text>
<text x="440.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">döngüde 1.024 BF16 işlem</text>
<text x="440.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">yapar. Malzeme gelmezse ne</text>
<text x="440.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">kadar güçlü olursa olsun</text>
<text x="440.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">soğuk kalır.</text>
<text x="440.0" y="2068" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">4 × 1.024 FLOP/döngü</text>
<rect x="619.0" y="1879" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="1879" width="4" height="201" rx="2" style="fill:var(--c-danger);fill-opacity:.85"/>
<text x="633.0" y="1899" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ortak kiler</text>
<rect x="631.0" y="1907.0" width="157.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="709.5" y="1921.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM3</text>
<text x="633.0" y="1945" text-anchor="start" style="fill:var(--c-text);font-size:11px">80 GB&#x27;lık büyük depo, ama</text>
<text x="633.0" y="1960" text-anchor="start" style="fill:var(--c-text);font-size:11px">uzakta: bir gidiş-dönüş</text>
<text x="633.0" y="1975" text-anchor="start" style="fill:var(--c-text);font-size:11px">~480 döngü. 132 aşçının</text>
<text x="633.0" y="1990" text-anchor="start" style="fill:var(--c-text);font-size:11px">hepsi aynı kileri paylaşır;</text>
<text x="633.0" y="2005" text-anchor="start" style="fill:var(--c-text);font-size:11px">aradaki ortak raf L2&#x27;dir.</text>
<text x="633.0" y="2068" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">~480 döngü · 3,35 TB/s</text>
<rect x="36.0" y="2092.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="2109" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Altın kural: aşçı kilerden domates beklerken başka bir siparişin soğanını doğrar (warp geçişi). Tezgâhta iş kalmadıysa, yani bütün</text>
<text x="46" y="2124" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">siparişler kileri bekliyorsa, dünyanın en hızlı ocağı da soğuk kalır.</text>
<line x1="420.0" y1="2149" x2="420.0" y2="2171" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="2164" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 132 SM = 1 GPU</text>
<rect x="28" y="2174" width="784" height="1598" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2200" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">2 · GPU: H100 SXM5 çipi</text>
<text x="40" y="2220" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">132 SM, ikiye bölünmüş 50 MB L2 ve 5 HBM3 yığını tek pakette durur. Bu kart metindeki “Bellek hiyerarşisi ve aritmetik</text>
<text x="40" y="2237" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">yoğunluk roofline modeli” alt bölümünü çizer: piramit, α_crit ≈ 295 ve batch&#x27;in etkisi.</text>
<text x="40" y="2267" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Donanım: çipin haritası</text>
<rect x="40" y="2281" width="480" height="316" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="52" y="2299" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Paket · GH100 kalıbı: 814 mm², 80 milyar transistör, TSMC 4N</text>
<rect x="50.0" y="2309.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73" y="2347" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="73" y="2363" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="101.0" y="2309.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="118.0" y="2355.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="50.0" y="2406.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73" y="2444" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="73" y="2460" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="101.0" y="2406.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="118.0" y="2452.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="50.0" y="2503.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73" y="2541" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="73" y="2557" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="101.0" y="2503.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="118.0" y="2549.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="466.0" y="2309.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="489" y="2347" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="489" y="2363" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="427.0" y="2309.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="444.0" y="2355.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="466.0" y="2406.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="489" y="2444" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">HBM3</text>
<text x="489" y="2460" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 GB</text>
<rect x="427.0" y="2406.0" width="34.0" height="84" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="444.0" y="2452.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 MC</text>
<rect x="466.0" y="2503.0" width="46.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.04;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="489" y="2541" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">boş</text>
<text x="489" y="2557" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yuva</text>
<rect x="140" y="2309" width="282" height="278" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="150.0" y="2317.0" width="262.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="281.0" y="2330.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink 4 · 18 bağlantı (diğer GPU&#x27;lara)</text>
<rect x="148" y="2343" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2356" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 0</text>
<rect x="155" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<rect x="155" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="285" y="2343" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2356" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 1</text>
<rect x="292" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2361" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2372" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2361" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2372" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="148" y="2391" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2404" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 2</text>
<rect x="155" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="155" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2409" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="259" y="2420" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="285" y="2391" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2404" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 3</text>
<rect x="292" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2409" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="383" y="2420" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2409" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2420" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="148" y="2473" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2486" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 4</text>
<rect x="155" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="155" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="285" y="2473" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2486" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 5</text>
<rect x="292" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2491" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2502" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2491" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2502" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="148" y="2521" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="154" y="2534" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 6</text>
<rect x="155" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="155" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="168" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="181" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="194" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="207" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="220" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="233" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="246" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="259" y="2539" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="259" y="2550" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="285" y="2521" width="129" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="291" y="2534" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPC 7</text>
<rect x="292" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="292" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="305" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="318" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="331" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="344" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="357" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="370" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2539" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="383" y="2550" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="396" y="2539" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="396" y="2550" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<rect x="148.0" y="2439.0" width="129.0" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="212.5" y="2457.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · bölüm 0</text>
<rect x="285.0" y="2439.0" width="129.0" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="349.5" y="2457.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · bölüm 1</text>
<rect x="150.0" y="2567.0" width="262.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="281.0" y="2580.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe Gen5 ×16 (CPU&#x27;ya)</text>
<rect x="40" y="2606" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.35"/>
<text x="57" y="2615" text-anchor="start" style="fill:var(--c-text);font-size:11px">etkin SM</text>
<rect x="123.8" y="2606" width="11" height="9" rx="1.5" style="fill:none;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:2 1.5"/>
<text x="140.8" y="2615" text-anchor="start" style="fill:var(--c-text);font-size:11px">verim için kapalı</text>
<rect x="262.5" y="2606" width="11" height="9" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<text x="279.5" y="2615" text-anchor="start" style="fill:var(--c-text);font-size:11px">1. basamaktaki SM</text>
<text x="40" y="2631" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">MC: 512-bit bellek denetleyicisi</text>
<text x="40" y="2651" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Her sütun bir TPC&#x27;dir (2 SM): 8 GPC × 9 TPC × 2 = 144 yuva, ürün 132&#x27;sini açar. Hangi</text>
<text x="40" y="2666" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">6 TPC&#x27;nin kapalı olduğu çipten çipe değişir; çizim bir örnektir. Soldaki GPC&#x27;ler</text>
<text x="40" y="2681" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">L2&#x27;nin 0. bölümüne, sağdakiler 1. bölümüne yakındır.</text>
<rect x="540" y="2281" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="2303" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">132</text>
<text x="550" y="2320" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SM</text>
<rect x="674" y="2281" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="2303" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">990</text>
<text x="684" y="2320" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (BF16)</text>
<rect x="540" y="2337" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="2359" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">80 GB</text>
<text x="550" y="2376" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM3</text>
<rect x="674" y="2337" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="2359" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">3,35 TB/s</text>
<text x="684" y="2376" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM hızı</text>
<text x="540" y="2415" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Tam çip ve ürün</text>
<text x="712" y="2433" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GH100 (tam)</text>
<text x="800" y="2433" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">H100 SXM5</text>
<line x1="540.0" y1="2439.0" x2="800.0" y2="2439.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="540" y="2455" text-anchor="start" style="fill:var(--c-text);font-size:11px">SM</text>
<text x="712" y="2455" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">144</text>
<text x="800" y="2455" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">132</text>
<text x="540" y="2471" text-anchor="start" style="fill:var(--c-text);font-size:11px">L2</text>
<text x="712" y="2471" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">60 MB</text>
<text x="800" y="2471" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">50 MB</text>
<text x="540" y="2487" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM yığını</text>
<text x="712" y="2487" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">6</text>
<text x="800" y="2487" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="540" y="2503" text-anchor="start" style="fill:var(--c-text);font-size:11px">bellek denetleyicisi</text>
<text x="712" y="2503" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">12</text>
<text x="800" y="2503" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700">10</text>
<text x="540" y="2527" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Kalıp çok büyük olduğu için kusursuz çıkanı</text>
<text x="540" y="2542" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">azdır; birkaç birim kapatılarak verim (yield)</text>
<text x="540" y="2557" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">artırılır.</text>
<rect x="540.0" y="2573.0" width="260.0" height="128" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="2593" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">3,35 TB/s nereden çıkar?</text>
<text x="552" y="2612" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5 yığın × 1.024 bit = 5.120 bit</text>
<text x="552" y="2628" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">× ~5,2 Gb/s ÷ 8 ≈ 3,35 TB/s</text>
<text x="552" y="2647" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Pin hızı resmi bant genişliğinden</text>
<text x="552" y="2662" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">türetildi: yığın başına ~670 GB/s ve</text>
<text x="552" y="2677" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 GB. Kartın güç sınırı 700 W&#x27;tır. (B200:</text>
<text x="552" y="2692" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">148 SM, 192 GB, 8 TB/s.)</text>
<text x="540" y="2723" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki karşılığı: 2. bölüm, “Bellek</text>
<text x="540" y="2738" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">hiyerarşisi ve aritmetik yoğunluk roofline</text>
<text x="540" y="2753" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">modeli”.</text>
<line x1="530.0" y1="2281.0" x2="530.0" y2="2753.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2793" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Bellek hiyerarşisi: veri çipte nerede durur?</text>
<text x="205" y="2829" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">küçük ve hızlı</text>
<text x="382" y="2829" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">gecikme · hız</text>
<polygon points="130.0,2839 280.0,2839 299.0,2881 111.0,2881" style="fill:var(--c-accent);fill-opacity:0.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205" y="2857" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Register file</text>
<text x="205" y="2872" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">256 KiB/SM · ~33 MB</text>
<line x1="305.0" y1="2860.0" x2="374.0" y2="2860.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2857" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">komutun işleneni</text>
<polygon points="108.0,2885 302.0,2885 321.0,2927 89.0,2927" style="fill:var(--c-accent);fill-opacity:0.15;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205" y="2903" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SMEM / L1</text>
<text x="205" y="2918" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">256 KiB/SM · ~33 MB</text>
<line x1="327.0" y1="2906.0" x2="374.0" y2="2906.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2903" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">~29–41 döngü</text>
<polygon points="86.0,2931 324.0,2931 343.0,2973 67.0,2973" style="fill:var(--c-accent);fill-opacity:0.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205" y="2949" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">L2 (2 bölüm)</text>
<text x="205" y="2964" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">50 MB</text>
<line x1="349.0" y1="2952.0" x2="374.0" y2="2952.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2949" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">~263 döngü</text>
<text x="382" y="2964" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">~5,5 TB/s</text>
<polygon points="64.0,2977 346.0,2977 365.0,3019 45.0,3019" style="fill:var(--c-danger);fill-opacity:0.25;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="205" y="2995" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">HBM3</text>
<text x="205" y="3010" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB</text>
<line x1="371.0" y1="2998.0" x2="374.0" y2="2998.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="382" y="2995" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">~479 döngü</text>
<text x="382" y="3010" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">3,35 TB/s</text>
<text x="205" y="3031" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">büyük ve yavaş</text>
<text x="205" y="3047" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kapasite (çip geneli)</text>
<text x="540" y="2817" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metindeki “dik piramit” budur: üstteki</text>
<text x="540" y="2834" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmanlar küçük ama hızlı, alttakiler</text>
<text x="540" y="2851" text-anchor="start" style="fill:var(--c-text);font-size:12px">büyük ama yavaştır. L2&#x27;den HBM&#x27;e inince</text>
<text x="540" y="2868" text-anchor="start" style="fill:var(--c-text);font-size:12px">kapasite 1.600 kat büyür, gecikme ise</text>
<text x="540" y="2885" text-anchor="start" style="fill:var(--c-text);font-size:12px">neredeyse iki katına çıkar. SM&#x27;lerin</text>
<text x="540" y="2902" text-anchor="start" style="fill:var(--c-text);font-size:12px">içindeki SRAM&#x27;in toplamı (~33 MB register</text>
<text x="540" y="2919" text-anchor="start" style="fill:var(--c-text);font-size:12px">+ ~33 MB SMEM/L1) L2&#x27;den bile büyüktür;</text>
<text x="540" y="2936" text-anchor="start" style="fill:var(--c-text);font-size:12px">FlashAttention gibi iyi çekirdekler veriyi</text>
<text x="540" y="2953" text-anchor="start" style="fill:var(--c-text);font-size:12px">bu katmanlarda tutup HBM&#x27;e az gider. (B200</text>
<text x="540" y="2970" text-anchor="start" style="fill:var(--c-text);font-size:12px">bu piramide SM başına 256 KiB&#x27;lık TMEM&#x27;i</text>
<text x="540" y="2987" text-anchor="start" style="fill:var(--c-text);font-size:12px">ekler.)</text>
<line x1="530.0" y1="2805.0" x2="530.0" y2="3047.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3087" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Roofline: hızı bellek mi belirler, hesap mı?</text>
<text x="40" y="3143.0" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (log)</text>
<line x1="88.0" y1="3153.0" x2="88.0" y2="3317.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="88.0" y1="3317.0" x2="498.0" y2="3317.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="88.0" y1="3277.0" x2="498.0" y2="3277.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3281.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="88.0" y1="3237.0" x2="498.0" y2="3237.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3241.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<line x1="88.0" y1="3197.0" x2="498.0" y2="3197.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3201.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<line x1="88.0" y1="3157.0" x2="498.0" y2="3157.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="83" y="3161.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.000</text>
<text x="88" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0,1</text>
<text x="170" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="252" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10</text>
<text x="334" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100</text>
<text x="416" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.000</text>
<text x="498" y="3332" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10.000</text>
<text x="498.0" y="3348" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">aritmetik yoğunluk α (FLOP/bayt, log)</text>
<polyline points="88.0,3296.0 372.6,3157.2 498.0,3157.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.4"/>
<line x1="372.6" y1="3157.2" x2="372.6" y2="3317.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="372.6" cy="3157.2" r="4" style="fill:var(--c-success)"/>
<text x="372.6" y="3148.2" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">α_crit ≈ 295,5</text>
<text x="276.7" y="3247" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">bellek sınırlı</text>
<text x="498.0" y="3209.2" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700">hesap sınırlı</text>
<circle cx="106.2" cy="3287.1" r="4.5" style="fill:var(--c-danger);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="115.2" y="3302.1" text-anchor="start" style="fill:var(--c-text);font-size:11px">toplama (BF16) · α ≈ 0,17</text>
<circle cx="170.0" cy="3256.0" r="4.5" style="fill:var(--c-danger);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="179.0" y="3271.0" text-anchor="start" style="fill:var(--c-text);font-size:11px">decode, B = 1 · α ≈ 1</text>
<circle cx="342.1" cy="3172.1" r="4.5" style="fill:var(--c-danger);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="333.1" y="3165.1" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (%42)</text>
<circle cx="448.5" cy="3157.2" r="4.5" style="fill:var(--c-accent);stroke:var(--c-surface);stroke-width:1.2"/>
<text x="448.5" y="3175.2" text-anchor="middle" style="fill:var(--c-text);font-size:11px">B = 4.096: tavanda</text>
<rect x="40.0" y="3361.0" width="480.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="3377.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">α_crit = C_peak ÷ W_mem = 990 ÷ 3,35 ≈ 295,5 FLOP/bayt</text>
<text x="40" y="3405" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Noktalar Llama 3 70B MLP boyutlarıyla (D = 8.192, F = 28.672) hesaplandı; BF16</text>
<text x="40" y="3420" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">toplama 1 FLOP / 6 bayttır. Tavan çizgisi min(990, 3,35 · α).</text>
<text x="540" y="3111" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metindeki formül: çip saniyede 990 trilyon</text>
<text x="540" y="3128" text-anchor="start" style="fill:var(--c-text);font-size:12px">işlem yapar ama HBM&#x27;den saniyede yalnızca</text>
<text x="540" y="3145" text-anchor="start" style="fill:var(--c-text);font-size:12px">3,35 TB okur. Okunan her bayt ~295 işlemi</text>
<text x="540" y="3162" text-anchor="start" style="fill:var(--c-text);font-size:12px">beslemezse Tensor Core&#x27;lar bekler. Matris</text>
<text x="540" y="3179" text-anchor="start" style="fill:var(--c-text);font-size:12px">çarpımında α(B) = BDF / (B(D+F) + DF):</text>
<text x="540" y="3196" text-anchor="start" style="fill:var(--c-text);font-size:12px">decode&#x27;da (B = 1) α ≈ 1 olur ve GPU</text>
<text x="540" y="3213" text-anchor="start" style="fill:var(--c-text);font-size:12px">gücünün %0,3&#x27;ünü kullanır; B = 4.096 gibi</text>
<text x="540" y="3230" text-anchor="start" style="fill:var(--c-text);font-size:12px">büyük batch&#x27;lerde α ≈ 2.500 ile tavana</text>
<text x="540" y="3247" text-anchor="start" style="fill:var(--c-text);font-size:12px">dayanır. C bölümündeki bellek duvarı aynı</text>
<text x="540" y="3264" text-anchor="start" style="fill:var(--c-text);font-size:12px">eğrinin batch&#x27;e göre çizilmiş hâlidir.</text>
<line x1="530.0" y1="3099.0" x2="530.0" y2="3420.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3460" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Zihinsel model: mutfak</text>
<text x="40" y="3478" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki benzetmenin devamı: SM bir hat aşçısıysa GPU, 132 aşçının birlikte çalıştığı mutfaktır.</text>
<rect x="40.0" y="3490" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3490" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Mutfak</text>
<rect x="52.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU çipi · 132 SM</text>
<text x="54.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">132 hat aşçısı aynı</text>
<text x="54.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">mutfakta, her biri kendi</text>
<text x="54.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">tezgâhında çalışır.</text>
<text x="54.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">Fabrikada 144 tezgâh</text>
<text x="54.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">kurulur, kusurlu</text>
<text x="54.0" y="3631" text-anchor="start" style="fill:var(--c-text);font-size:11px">çıkabilecek 12&#x27;si</text>
<text x="54.0" y="3646" text-anchor="start" style="fill:var(--c-text);font-size:11px">kapatılır.</text>
<text x="54.0" y="3679" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">132 aşçı · 990 TFLOP/s</text>
<rect x="233.0" y="3490" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="3490" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="247.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ortak raf</text>
<rect x="245.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="323.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">L2 · 50 MB</text>
<text x="247.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">Sık kullanılan malzeme</text>
<text x="247.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">kilere gitmeden buradan</text>
<text x="247.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">alınır. Raf ikiye</text>
<text x="247.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">bölünmüştür; mutfağın öbür</text>
<text x="247.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">yarısındaki raf biraz daha</text>
<text x="247.0" y="3631" text-anchor="start" style="fill:var(--c-text);font-size:11px">uzaktadır.</text>
<text x="247.0" y="3679" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">~263 döngü · ~5,5 TB/s</text>
<rect x="426.0" y="3490" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="3490" width="4" height="201" rx="2" style="fill:var(--c-danger);fill-opacity:.85"/>
<text x="440.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kiler</text>
<rect x="438.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="516.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM3 · 80 GB</text>
<text x="440.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">Bütün malzemenin durduğu</text>
<text x="440.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">büyük depo. 5 kapısı</text>
<text x="440.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">(yığın) var ve hepsi</text>
<text x="440.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">birlikte saniyede 3,35 TB</text>
<text x="440.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">malzeme geçirir.</text>
<text x="440.0" y="3679" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">~479 döngü · 3,35 TB/s</text>
<rect x="619.0" y="3490" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="3490" width="4" height="201" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="633.0" y="3510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Servis kapısı</text>
<rect x="631.0" y="3518.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="709.5" y="3532.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink · PCIe</text>
<text x="633.0" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:11px">Diğer mutfaklara açılan</text>
<text x="633.0" y="3571" text-anchor="start" style="fill:var(--c-text);font-size:11px">kapı: NVLink ile 450 GB/s,</text>
<text x="633.0" y="3586" text-anchor="start" style="fill:var(--c-text);font-size:11px">CPU&#x27;ya PCIe ile 64 GB/s</text>
<text x="633.0" y="3601" text-anchor="start" style="fill:var(--c-text);font-size:11px">(tek yön). Kilerden 7 kat</text>
<text x="633.0" y="3616" text-anchor="start" style="fill:var(--c-text);font-size:11px">dardır; sonraki basamak</text>
<text x="633.0" y="3631" text-anchor="start" style="fill:var(--c-text);font-size:11px">burayı anlatır.</text>
<text x="633.0" y="3679" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">450 GB/s · 64 GB/s</text>
<rect x="36.0" y="3703.0" width="768.0" height="59" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="3720" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Altın kural: kilerin kapısı saniyede 3,35 TB geçirir, aşçılar saniyede 990 trilyon iş yapar. Kilerden gelen her bayt malzemeyle</text>
<text x="46" y="3735" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">~295 iş yapılmazsa aşçılar kapıda bekler. Pek çok optimizasyonun özü bu yüzden aynıdır: HBM&#x27;den az oku, okuduğun her baytla çok</text>
<text x="46" y="3750" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">hesapla.</text>
<line x1="420.0" y1="3775" x2="420.0" y2="3797" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="3790" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 8 GPU = 1 düğüm</text>
<rect x="28" y="3800" width="784" height="2641" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3826" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">3 · Düğüm: DGX H100 kasasının içi</text>
<text x="40" y="3846" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">8 GPU, 4 NVSwitch, 2 CPU ve 10 ConnectX-7 ağ kartı tek kasada. Bu kart metindeki “Intra-node ağ mimarisi: NVLink ve NVSwitch</text>
<text x="40" y="3863" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">fabric&#x27;leri” alt bölümünü çizer; şemadaki numaralı daireler aşağıdaki altı soruyla eşleşir.</text>
<text x="40" y="3893" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Donanım: NVIDIA&#x27;nın sistem topolojisi</text>
<rect x="36" y="3907" width="488" height="418" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="44.0" y="3919" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="76.0" y="3933" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="76.0" y="3947" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 veri</text>
<rect x="130.0" y="3919" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="158.0" y="3933" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="158.0" y="3947" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="206.0" y="3919" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="3933" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="239.0" y="3947" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çift port</text>
<rect x="288.0" y="3919" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="321.0" y="3933" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="321.0" y="3947" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çift port</text>
<rect x="374.0" y="3919" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="402.0" y="3933" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="402.0" y="3947" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="452.0" y="3919" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="484.0" y="3933" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="484.0" y="3947" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 veri</text>
<line x1="109.0" y1="3937" x2="128.0" y2="3937" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="187.0" y1="3937" x2="204.0" y2="3937" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="356.0" y1="3937" x2="372.0" y2="3937" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="432.0" y1="3937" x2="450.0" y2="3937" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="158.0" y1="3957" x2="158.0" y2="3973" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="402.0" y1="3957" x2="402.0" y2="3973" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="120.0" y="3975" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158" y="3995" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="158" y="4011" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="364.0" y="3975" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="402" y="3995" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="402" y="4011" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="210.0" y="3977" width="62.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="241.0" y="3991" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 × M.2 OS</text>
<line x1="198.0" y1="3987" x2="208.0" y2="3987" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="288.0" y="3977" width="62.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="319.0" y="3991" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<line x1="352.0" y1="3987" x2="362.0" y2="3987" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="199.0" y1="4009" x2="361.0" y2="4009" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<path d="M196 4019H280V4144A8 18 0 0 1 280 4180V4241" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="40" y="4047" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="286" y="4047" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<path d="M132 4023V4031H69V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M148 4023V4038H127V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M164 4023V4038H185V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M180 4023V4031H243V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M376 4023V4031H317V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M392 4023V4038H375V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M408 4023V4038H433V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M424 4023V4031H491V4053" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="46.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="69.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="69.0" y1="4081" x2="69.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="44.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.34;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="69.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="104.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="127.0" y1="4081" x2="127.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="102.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="127.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="162.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="185.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="185.0" y1="4081" x2="185.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="160.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="185.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="220.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="243.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="243.0" y1="4081" x2="243.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="218.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="243.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="294.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="317.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="317.0" y1="4081" x2="317.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="292.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<rect x="352.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="375.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="375.0" y1="4081" x2="375.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="350.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<rect x="410.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="433.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="433.0" y1="4081" x2="433.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="408.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="433.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<rect x="468.0" y="4055" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="491.0" y="4071" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="491.0" y1="4081" x2="491.0" y2="4095" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="466.0" y="4097" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="491.0" y="4117.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<!-- GPU 0 (center 69): 4 NVLink drop lines -->
<line x1="60" y1="4129" x2="60" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="66" y1="4129" x2="66" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="72" y1="4129" x2="72" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="78" y1="4129" x2="78" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 1 (center 127): 4 NVLink drop lines -->
<line x1="118" y1="4129" x2="118" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="124" y1="4129" x2="124" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="130" y1="4129" x2="130" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="136" y1="4129" x2="136" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 2 (center 185): 4 NVLink drop lines -->
<line x1="176" y1="4129" x2="176" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="182" y1="4129" x2="182" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="188" y1="4129" x2="188" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="194" y1="4129" x2="194" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 3 (center 243): 4 NVLink drop lines -->
<line x1="234" y1="4129" x2="234" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="240" y1="4129" x2="240" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="246" y1="4129" x2="246" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="252" y1="4129" x2="252" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 4 (center 317): 4 NVLink drop lines -->
<line x1="308" y1="4129" x2="308" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="314" y1="4129" x2="314" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="320" y1="4129" x2="320" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="326" y1="4129" x2="326" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 5 (center 375): 4 NVLink drop lines -->
<line x1="366" y1="4129" x2="366" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="372" y1="4129" x2="372" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="378" y1="4129" x2="378" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="384" y1="4129" x2="384" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 6 (center 433): 4 NVLink drop lines -->
<line x1="424" y1="4129" x2="424" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="430" y1="4129" x2="430" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="436" y1="4129" x2="436" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="442" y1="4129" x2="442" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 7 (center 491): 4 NVLink drop lines -->
<line x1="482" y1="4129" x2="482" y2="4150" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="488" y1="4129" x2="488" y2="4158" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="494" y1="4129" x2="494" y2="4166" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="500" y1="4129" x2="500" y2="4174" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- 4 Crossbar Horizontal Rails spanning across all 8 GPUs -->
<line x1="56" y1="4150" x2="486" y2="4150" style="stroke:#e11d48;stroke-width:1.8"/>
<line x1="62" y1="4158" x2="492" y2="4158" style="stroke:#0284c7;stroke-width:1.8"/>
<line x1="68" y1="4166" x2="498" y2="4166" style="stroke:#ea580c;stroke-width:1.8"/>
<line x1="74" y1="4174" x2="504" y2="4174" style="stroke:#9333ea;stroke-width:1.8"/>
<!-- Feed bundles into 4 NVSwitches from respective rails -->
<!-- Rail 0 (Pink) -> NVSwitch 0 (5 links) -->
<line x1="88" y1="4150" x2="88" y2="4194" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="92" y1="4150" x2="92" y2="4194" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="96" y1="4150" x2="96" y2="4194" style="stroke:#e11d48;stroke-width:1.4"/>
<line x1="100" y1="4150" x2="100" y2="4194" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="104" y1="4150" x2="104" y2="4194" style="stroke:#e11d48;stroke-width:1.2"/>
<!-- Rail 1 (Blue) -> NVSwitch 1 (4 links) -->
<line x1="194" y1="4158" x2="194" y2="4194" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="198" y1="4158" x2="198" y2="4194" style="stroke:#0284c7;stroke-width:1.3"/>
<line x1="202" y1="4158" x2="202" y2="4194" style="stroke:#0284c7;stroke-width:1.3"/>
<line x1="206" y1="4158" x2="206" y2="4194" style="stroke:#0284c7;stroke-width:1.2"/>
<!-- Rail 2 (Orange) -> NVSwitch 2 (4 links) -->
<line x1="354" y1="4166" x2="354" y2="4194" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="358" y1="4166" x2="358" y2="4194" style="stroke:#ea580c;stroke-width:1.3"/>
<line x1="362" y1="4166" x2="362" y2="4194" style="stroke:#ea580c;stroke-width:1.3"/>
<line x1="366" y1="4166" x2="366" y2="4194" style="stroke:#ea580c;stroke-width:1.2"/>
<!-- Rail 3 (Purple) -> NVSwitch 3 (5 links) -->
<line x1="456" y1="4174" x2="456" y2="4194" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="460" y1="4174" x2="460" y2="4194" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="464" y1="4174" x2="464" y2="4194" style="stroke:#9333ea;stroke-width:1.4"/>
<line x1="468" y1="4174" x2="468" y2="4194" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="472" y1="4174" x2="472" y2="4194" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- 4 NVSwitch Chips (Green, matching NVIDIA official diagram) -->
<rect x="52.0" y="4194" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="96.0" y="4211" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 0</text>
<rect x="156.0" y="4194" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="200.0" y="4211" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 1</text>
<rect x="316.0" y="4194" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="360.0" y="4211" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 2</text>
<rect x="420.0" y="4194" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="464.0" y="4211" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 3</text>
<!-- Bottom PCIe Switch (Purple, connecting CPU 0 to NVSwitches) -->
<rect x="250.0" y="4244" width="60.0" height="26" rx="5" style="fill:#8b5cf6;fill-opacity:.18;stroke:#8b5cf6;stroke-width:1.3"/>
<text x="280" y="4256" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">PCIe</text>
<text x="280" y="4267" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">switch</text>
<!-- PCIe Management lines from PCIe Switch UP to NVSwitches (Fabric Manager control plane) -->
<path d="M250 4266H96V4222" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M250 4255H200V4222" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M310 4255H360V4222" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M310 4266H464V4222" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<!-- Explanatory note -->
<text x="280" y="4283" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-style:italic">PCIe Kontrol Yolu: CPU 0 → PCIe switch → 4 NVSwitch (Fabric Manager kontrol düzlemi)</text>
<!-- Component Legend inside Node Box matching NVIDIA official diagram -->
<!-- Legend Row 1: Chips & Modules -->
<rect x="42" y="4294" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1"/>
<text x="56" y="4302" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">ConnectX-7</text>
<rect x="120" y="4294" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.3;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="134" y="4302" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">CX-7 Modül</text>
<rect x="200" y="4294" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.4;stroke:var(--c-warn);stroke-width:1"/>
<text x="214" y="4302" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVMe</text>
<rect x="256" y="4294" width="10" height="10" rx="2" style="fill:#8b5cf6;fill-opacity:.3;stroke:#8b5cf6;stroke-width:1"/>
<text x="270" y="4302" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">PCIe Switch</text>
<rect x="336" y="4294" width="10" height="10" rx="2" style="fill:#10b981;fill-opacity:.3;stroke:#10b981;stroke-width:1"/>
<text x="350" y="4302" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVSwitch</text>
<rect x="406" y="4294" width="10" height="10" rx="2" style="fill:#ec4899;fill-opacity:.3;stroke:#ec4899;stroke-width:1"/>
<text x="420" y="4302" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">100 GbE</text>
<!-- Legend Row 2: Interconnects & Fabrics -->
<line x1="50" y1="4316" x2="72" y2="4316" style="stroke:#8b5cf6;stroke-width:1.5"/>
<text x="76" y="4319" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">PCIe Gen5</text>
<line x1="140" y1="4314" x2="162" y2="4314" style="stroke:#e11d48;stroke-width:1.5"/>
<line x1="140" y1="4316" x2="162" y2="4316" style="stroke:#0284c7;stroke-width:1.5"/>
<line x1="140" y1="4318" x2="162" y2="4318" style="stroke:#ea580c;stroke-width:1.5"/>
<line x1="140" y1="4320" x2="162" y2="4320" style="stroke:#9333ea;stroke-width:1.5"/>
<text x="166" y="4319" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVLink (4 Ray)</text>
<line x1="250" y1="4316" x2="272" y2="4316" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="276" y="4319" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">CPU İletişimi (UPI)</text>
<!-- Question Callout Circles (Topoloji Soruları) -->
<circle cx="102" cy="3999" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="102" y="4003" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<circle cx="52" cy="4035" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="52" y="4039" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="98" cy="4159" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="98" y="4163" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="239" cy="3966" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="239" y="3970" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="280" cy="4009" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="4013" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<circle cx="280" cy="4226" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="4230" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="540" y="3907" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="3929" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">8</text>
<text x="550" y="3946" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU · 640 GB</text>
<rect x="674" y="3907" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="3929" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">450 GB/s</text>
<text x="684" y="3946" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink, tek yön</text>
<rect x="540" y="3963" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="3985" text-anchor="start" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">64 GB/s</text>
<text x="550" y="4002" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe, CPU yolu</text>
<rect x="674" y="3963" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="3985" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">50 GB/s</text>
<text x="684" y="4002" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağ kartı / GPU</text>
<text x="540" y="4033" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 NVSwitch · biseksiyon 3,6 TB/s · 2 × Xeon</text>
<text x="540" y="4048" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8480C (112 çekirdek), 2 TB RAM · 8 × 3,84 TB</text>
<text x="540" y="4063" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVMe veri · güç 10,2 kW</text>
<rect x="540.0" y="4081" width="260.0" height="177" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="4103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Örnek yol: bir gradyanın düğümdeki</text>
<text x="552" y="4120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">yolculuğu</text>
<text x="552" y="4143" text-anchor="start" style="fill:var(--c-text);font-size:12px">Komşu düğümden gelen bir gradyan NIC</text>
<text x="552" y="4160" text-anchor="start" style="fill:var(--c-text);font-size:12px">5&#x27;e girer ve GPUDirect RDMA ile</text>
<text x="552" y="4177" text-anchor="start" style="fill:var(--c-text);font-size:12px">doğrudan GPU 5&#x27;in belleğine yazılır;</text>
<text x="552" y="4194" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU 1&#x27;e hiç uğramaz. GPU 5 bu gradyanı</text>
<text x="552" y="4211" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU 2 ile paylaşacaksa ağa ya da</text>
<text x="552" y="4228" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU&#x27;lara değil, NVLink üzerinden</text>
<text x="552" y="4245" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch&#x27;lere gider.</text>
<rect x="540.0" y="4274.0" width="260.0" height="143" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="4294" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">450 GB/s nereden çıkar?</text>
<text x="552" y="4313" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">18 bağlantı × 25 GB/s = 450 GB/s</text>
<text x="552" y="4329" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">biseksiyon: 4 × 450 × 2 = 3,6 TB/s</text>
<text x="552" y="4348" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her NVLink her yönde 25 GB/s taşır; çift</text>
<text x="552" y="4363" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yönde GPU başına 900 GB/s eder. Kasa ikiye</text>
<text x="552" y="4378" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">bölününce (GPU 0–3 | 4–7) aradan 3,6 TB/s</text>
<text x="552" y="4393" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">geçer. (GB200 NVL72: 72 GPU tek NVLink</text>
<text x="552" y="4408" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alanında, GPU başına 900 GB/s tek yön.)</text>
<text x="540" y="4439" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki karşılığı: 3. bölüm, “Intra-node ağ</text>
<text x="540" y="4454" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">mimarisi: NVLink ve NVSwitch fabric&#x27;leri”.</text>
<rect x="36" y="4468" width="768" height="517" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="4490" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Lejant: şemadaki her parça ne?</text>
<text x="48" y="4508" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Adlar NVIDIA&#x27;nın diyagram lejantından, renkler posterin sütun renkleridir. Görevler DGX H100 kullanım kılavuzundan gelir; “olası”</text>
<text x="48" y="4523" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yazanlar çıkarımdır.</text>
<rect x="48.0" y="4541" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="4555" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CX-7</text>
<text x="126" y="4555" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7” · üstte, 2 adet</text>
<text x="126" y="4571" text-anchor="start" style="fill:var(--c-text);font-size:11px">Çift portlu ağ kartı; kılavuza göre depolama ağı</text>
<text x="126" y="4586" text-anchor="start" style="fill:var(--c-text);font-size:11px">ve in-band yönetim için. Her biri üstteki bir PCIe</text>
<text x="126" y="4601" text-anchor="start" style="fill:var(--c-text);font-size:11px">switch&#x27;e bağlı.</text>
<rect x="48.0" y="4615" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="4629" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0–7</text>
<text x="126" y="4629" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7 Network Module” · 8 adet</text>
<text x="126" y="4645" text-anchor="start" style="fill:var(--c-text);font-size:11px">Tek portlu 400 Gb/s hesaplama ağı kartları, GPU</text>
<text x="126" y="4660" text-anchor="start" style="fill:var(--c-text);font-size:11px">başına bir tane; kümeye 4 OSFP portundan çıkar.</text>
<text x="126" y="4675" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gri çerçeveler bu kartları taşıyan modüllerdir</text>
<text x="126" y="4690" text-anchor="start" style="fill:var(--c-text);font-size:11px">(olası).</text>
<rect x="48.0" y="4704" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="4718" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe</text>
<text x="126" y="4718" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · üstte, 4 + 4</text>
<text x="126" y="4734" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri diskleri: 8 × 3,84 TB U.2 NVMe, RAID 0</text>
<text x="126" y="4749" text-anchor="start" style="fill:var(--c-text);font-size:11px">önbellek; okuma ve kontrol noktası için yerel</text>
<text x="126" y="4764" text-anchor="start" style="fill:var(--c-text);font-size:11px">alan.</text>
<rect x="48.0" y="4778" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="4792" text-anchor="middle" style="fill:var(--c-text);font-size:11px">M.2 OS</text>
<text x="126" y="4792" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · CPU 0&#x27;ın yanında</text>
<text x="126" y="4808" text-anchor="start" style="fill:var(--c-text);font-size:11px">Büyük olasılıkla 2 × 1,92 TB M.2 işletim sistemi</text>
<text x="126" y="4823" text-anchor="start" style="fill:var(--c-text);font-size:11px">diski (RAID 1); diyagram veri diskleriyle aynı</text>
<text x="126" y="4838" text-anchor="start" style="fill:var(--c-text);font-size:11px">simgeyi kullanır.</text>
<rect x="48.0" y="4852" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="4866" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe sw</text>
<text x="126" y="4866" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe Switches” · 3 adet</text>
<text x="126" y="4882" text-anchor="start" style="fill:var(--c-text);font-size:11px">Bir CPU hattını birçok cihaza dağıtır: üstteki</text>
<text x="126" y="4897" text-anchor="start" style="fill:var(--c-text);font-size:11px">ikisi NVMe ve CX-7&#x27;yi taşır, alttaki NVSwitch&#x27;leri</text>
<text x="126" y="4912" text-anchor="start" style="fill:var(--c-text);font-size:11px">CPU 0&#x27;a bağlar.</text>
<rect x="48.0" y="4926" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="4940" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<text x="126" y="4940" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “100 GbE” · CPU 1&#x27;e bağlı</text>
<text x="126" y="4956" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ek bir Ethernet kartı; görevi diyagramda yazmıyor</text>
<text x="126" y="4971" text-anchor="start" style="fill:var(--c-text);font-size:11px">(olası: servis ya da yönetim ağı).</text>
<rect x="432.0" y="4541" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="465.0" y="4555" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVSwitch</text>
<text x="510" y="4555" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVSwitch” · 4 adet</text>
<text x="510" y="4571" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;lar arası NVLink anahtarı; her GPU 18</text>
<text x="510" y="4586" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlantısını bu dördüne 5 + 4 + 4 + 5 olarak</text>
<text x="510" y="4601" text-anchor="start" style="fill:var(--c-text);font-size:11px">dağıtır.</text>
<rect x="432.0" y="4615" width="66.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="4629" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CPU</text>
<text x="510" y="4629" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU 0 / CPU 1”</text>
<text x="510" y="4645" text-anchor="start" style="fill:var(--c-text);font-size:11px">Intel Xeon 8480C (56&#x27;şar çekirdek); her biri dört</text>
<text x="510" y="4660" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;yu ve onların NIC&#x27;lerini PCIe ile besler.</text>
<line x1="432.0" y1="4684" x2="498.0" y2="4684" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="510" y="4688" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU communication”</text>
<text x="510" y="4704" text-anchor="start" style="fill:var(--c-text);font-size:11px">İki soketi birbirine bağlayan hat; Intel Xeon&#x27;da</text>
<text x="510" y="4719" text-anchor="start" style="fill:var(--c-text);font-size:11px">bu bağlantının adı UPI&#x27;dir.</text>
<line x1="432.0" y1="4743" x2="498.0" y2="4743" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="510" y="4747" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe”</text>
<text x="510" y="4763" text-anchor="start" style="fill:var(--c-text);font-size:11px">CPU&#x27;ları NIC&#x27;lere, GPU&#x27;lara, disklere ve</text>
<text x="510" y="4778" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVSwitch&#x27;lere bağlayan genel yol (PCIe Gen5).</text>
<line x1="432.0" y1="4802" x2="498.0" y2="4802" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="510" y="4806" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA lejantında yok: NVLink</text>
<text x="510" y="4822" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;ların altındaki renkli demetler NVLink</text>
<text x="510" y="4837" text-anchor="start" style="fill:var(--c-text);font-size:11px">hatlarıdır: GPU başına 18 bağlantı, toplam</text>
<text x="510" y="4852" text-anchor="start" style="fill:var(--c-text);font-size:11px">900 GB/s çift yön.</text>
<line x1="420.0" y1="4537" x2="420.0" y2="4971" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5011" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Topolojiyi okuma: altı soru</text>
<text x="40" y="5030" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Şemadaki numaralı daireleri sırayla izleyin; her biri bir soruyu cevaplar.</text>
<circle cx="49" cy="5056" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="5060" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="5060" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Kim kimi besler?</text>
<text x="66" y="5080" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğüm birbirinin aynası olan iki yarıdan oluşur. CPU 0,</text>
<text x="66" y="5097" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU 0–3&#x27;ü ve NIC 0–3&#x27;ü PCIe ile besler; CPU 1 de GPU</text>
<text x="66" y="5114" text-anchor="start" style="fill:var(--c-text);font-size:12px">4–7&#x27;yi ve NIC 4–7&#x27;yi. Her yarının kendi belleği vardır;</text>
<text x="66" y="5131" text-anchor="start" style="fill:var(--c-text);font-size:12px">buna NUMA alanı denir.</text>
<text x="66" y="5150" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> GPU 5&#x27;e veri hazırlayan süreç CPU 1&#x27;de</text>
<text x="66" y="5167" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">koşmalı ve CPU 1&#x27;in belleğini kullanmalıdır; yoksa her</text>
<text x="66" y="5184" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">batch önce CPU&#x27;lar arasındaki hattı geçer.</text>
<circle cx="49" cy="5210" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="5214" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="5214" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">GPU ağa nasıl çıkar?</text>
<text x="66" y="5234" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU&#x27;nun hemen üstünde kendine ait 400 Gb/s&#x27;lik bir</text>
<text x="66" y="5251" text-anchor="start" style="fill:var(--c-text);font-size:12px">NIC durur (GPU i ↔ NIC i). GPUDirect RDMA sayesinde</text>
<text x="66" y="5268" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri GPU belleğinden doğrudan NIC&#x27;e geçer, CPU</text>
<text x="66" y="5285" text-anchor="start" style="fill:var(--c-text);font-size:12px">belleğine hiç uğramaz.</text>
<text x="66" y="5304" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> kasadan 8 × 50 = 400 GB/s çıkar. Kümede NIC</text>
<text x="66" y="5321" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">i, i. leaf switch&#x27;e takılır; farklı düğümlerdeki aynı</text>
<text x="66" y="5338" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">numaralı GPU&#x27;lar bu yüzden tek sıçramada buluşur.</text>
<circle cx="49" cy="5364" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="5368" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="5368" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">GPU&#x27;lar birbirleriyle nasıl konuşur?</text>
<text x="66" y="5388" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU&#x27;nun 18 NVLink&#x27;i aşağıdaki dört NVSwitch&#x27;e iner:</text>
<text x="66" y="5405" text-anchor="start" style="fill:var(--c-text);font-size:12px">ikisine 5&#x27;er, ikisine 4&#x27;er hat. Herhangi iki GPU tek</text>
<text x="66" y="5422" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir switch sıçramasıyla, her yönde 450 GB/s ile</text>
<text x="66" y="5439" text-anchor="start" style="fill:var(--c-text);font-size:12px">konuşur.</text>
<text x="66" y="5458" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> bu yol ağdan 9 kat geniştir; her katmanda</text>
<text x="66" y="5475" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">konuşan tensör paralelliği bu yüzden bu sekiz GPU&#x27;nun</text>
<text x="66" y="5492" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">içinde tutulur.</text>
<circle cx="441" cy="5056" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="5060" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="458" y="5060" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Depolama trafiği nereden geçer?</text>
<text x="458" y="5080" text-anchor="start" style="fill:var(--c-text);font-size:12px">Üstteki iki PCIe switch, çift portlu CX-7 kartlarını</text>
<text x="458" y="5097" text-anchor="start" style="fill:var(--c-text);font-size:12px">(depolama ağı ve in-band yönetim) ve 4&#x27;er U.2 NVMe veri</text>
<text x="458" y="5114" text-anchor="start" style="fill:var(--c-text);font-size:12px">diskini taşır. CPU 0&#x27;ın yanında ayrıca iki M.2 işletim</text>
<text x="458" y="5131" text-anchor="start" style="fill:var(--c-text);font-size:12px">sistemi diski durur.</text>
<text x="458" y="5150" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> veri okumak ve kontrol noktası yazmak,</text>
<text x="458" y="5167" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">gradyan trafiğinden farklı kartları ve kabloları</text>
<text x="458" y="5184" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">kullanır; büyük bir kayıt AllReduce&#x27;u yavaşlatmaz.</text>
<circle cx="441" cy="5210" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="5214" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="5214" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">İki yarıyı ne birleştirir?</text>
<text x="458" y="5234" text-anchor="start" style="fill:var(--c-text);font-size:12px">İki CPU, NVIDIA&#x27;nın “CPU communication” dediği hatla</text>
<text x="458" y="5251" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Xeon&#x27;da UPI) bağlıdır. Bu hat CPU&#x27;ların kendi trafiği</text>
<text x="458" y="5268" text-anchor="start" style="fill:var(--c-text);font-size:12px">içindir; iki yarıdaki GPU&#x27;lar birbirleriyle CPU&#x27;lar</text>
<text x="458" y="5285" text-anchor="start" style="fill:var(--c-text);font-size:12px">üzerinden değil, NVLink üzerinden konuşur.</text>
<text x="458" y="5304" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> bir CPU öbür yarının belleğine erişebilir ama</text>
<text x="458" y="5321" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">daha yavaş; NUMA sabitlemesi bu cezadan kaçınmak</text>
<text x="458" y="5338" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">içindir.</text>
<circle cx="441" cy="5364" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="5368" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="5368" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVSwitch&#x27;leri kim yönetir?</text>
<text x="458" y="5388" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU 0&#x27;dan inen PCIe hattı, alttaki PCIe switch</text>
<text x="458" y="5405" text-anchor="start" style="fill:var(--c-text);font-size:12px">üzerinden dört NVSwitch&#x27;e ulaşır. Bu bir kontrol</text>
<text x="458" y="5422" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoludur: NVIDIA Fabric Manager switch&#x27;lerin yönlendirme</text>
<text x="458" y="5439" text-anchor="start" style="fill:var(--c-text);font-size:12px">tablolarını buradan kurar; GPU verisi bu hattan akmaz.</text>
<text x="458" y="5458" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> Fabric Manager çalışmazsa GPU&#x27;lar tek tek</text>
<text x="458" y="5475" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sağlıklı görünse bile NVLink ağı kurulmaz ve çok GPU&#x27;lu</text>
<text x="458" y="5492" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">işler başlamaz.</text>
<line x1="420.0" y1="5044" x2="420.0" y2="5492" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5536" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">NVSwitch ağı: bağlantıları sayalım</text>
<text x="156" y="5566" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 0</text>
<text x="240" y="5566" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 1</text>
<text x="324" y="5566" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 2</text>
<text x="408" y="5566" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">NVSwitch 3</text>
<text x="487" y="5566" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">toplam</text>
<line x1="40.0" y1="5572.0" x2="520.0" y2="5572.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="5593" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 0</text>
<rect x="118.0" y="5580.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5580.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5580.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5580.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5593.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5593" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5615" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 1</text>
<rect x="118.0" y="5602.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5615.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5602.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5615.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5602.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5615.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5602.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5615.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5615" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5637" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 2</text>
<rect x="118.0" y="5624.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5637.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5624.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5637.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5624.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5637.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5624.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5637.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5637" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5659" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 3</text>
<rect x="118.0" y="5646.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5659.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5646.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5659.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5646.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5659.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5646.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5659.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5659" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<line x1="40.0" y1="5669.0" x2="520.0" y2="5669.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="40" y="5687" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 4</text>
<rect x="118.0" y="5674.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5687.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5674.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5687.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5674.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5687.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5674.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5687.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5687" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5709" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 5</text>
<rect x="118.0" y="5696.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5709.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5696.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5709.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5696.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5709.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5696.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5709.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5731" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 6</text>
<rect x="118.0" y="5718.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5731.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5718.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5731.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5718.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5731.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5718.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5731.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5731" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<text x="40" y="5753" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">GPU 7</text>
<rect x="118.0" y="5740.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="5753.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="202.0" y="5740.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="240.0" y="5753.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="286.0" y="5740.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="5753.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="370.0" y="5740.0" width="76.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="408.0" y="5753.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<text x="487" y="5753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">18</text>
<line x1="40.0" y1="5763.0" x2="520.0" y2="5763.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="5783" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">toplam</text>
<text x="156" y="5783" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">40</text>
<text x="240" y="5783" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">32</text>
<text x="324" y="5783" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">32</text>
<text x="408" y="5783" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700;font-family:var(--font-mono)">40</text>
<text x="487" y="5783" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">144</text>
<line x1="40.0" y1="5806.0" x2="62.0" y2="5806.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="68" y="5810" text-anchor="start" style="fill:var(--c-text);font-size:11px">kesit: GPU 0–3 | GPU 4–7 → 4 × 450 × 2 = 3,6 TB/s</text>
<text x="40" y="5828" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Her hücre bir GPU&#x27;dan bir NVSwitch&#x27;e inen NVLink sayısıdır (Hot Chips 34).</text>
<text x="540" y="5560" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metindeki hesap: her GPU&#x27;nun 18 NVLink&#x27;i</text>
<text x="540" y="5577" text-anchor="start" style="fill:var(--c-text);font-size:12px">dört NVSwitch&#x27;e 5 + 4 + 4 + 5 olarak</text>
<text x="540" y="5594" text-anchor="start" style="fill:var(--c-text);font-size:12px">dağılır. Switch&#x27;lere toplam 144 bağlantı</text>
<text x="540" y="5611" text-anchor="start" style="fill:var(--c-text);font-size:12px">iner, bu yüzden 8 GPU&#x27;nun hepsi aynı anda</text>
<text x="540" y="5628" text-anchor="start" style="fill:var(--c-text);font-size:12px">tam hızla gönderebilir: ağ engelsizdir ve</text>
<text x="540" y="5645" text-anchor="start" style="fill:var(--c-text);font-size:12px">her GPU diğer yedisine tek switch</text>
<text x="540" y="5662" text-anchor="start" style="fill:var(--c-text);font-size:12px">sıçramasıyla ulaşır. Her NVSwitch çipinde</text>
<text x="540" y="5679" text-anchor="start" style="fill:var(--c-text);font-size:12px">64 NVLink portu ve 3,2 TB/s çift yönlü</text>
<text x="540" y="5696" text-anchor="start" style="fill:var(--c-text);font-size:12px">kapasite vardır; dördü birlikte tek yönde</text>
<text x="540" y="5713" text-anchor="start" style="fill:var(--c-text);font-size:12px">6,4 TB/s eder. NVSwitch toplamayı kendi</text>
<text x="540" y="5730" text-anchor="start" style="fill:var(--c-text);font-size:12px">içinde de yapabilir (NVLink SHARP);</text>
<text x="540" y="5747" text-anchor="start" style="fill:var(--c-text);font-size:12px">AllReduce&#x27;ta veri switch&#x27;e bir kez gider.</text>
<line x1="530.0" y1="5548.0" x2="530.0" y2="5828.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5868" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Hangi trafik hangi yoldan geçer?</text>
<text x="40" y="5894" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU başına bant genişliği, tek yön</text>
<text x="40" y="5918" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (NVSwitch)</text>
<rect x="178" y="5906" width="240.0" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="424.0" y="5918" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">450 GB/s</text>
<text x="40" y="5942" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU)</text>
<rect x="178" y="5930" width="34.1" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="218.1" y="5942" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">64 GB/s</text>
<text x="282.1" y="5942" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7</text>
<text x="40" y="5966" text-anchor="start" style="fill:var(--c-text);font-size:11px">NIC (ağ, ray i)</text>
<rect x="178" y="5954" width="26.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="210.7" y="5966" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">50 GB/s</text>
<text x="274.7" y="5966" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<text x="40" y="6000" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">iş</text>
<text x="232" y="6000" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">yol</text>
<text x="370" y="6000" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">ne sıklıkla</text>
<line x1="40.0" y1="6006.0" x2="520.0" y2="6006.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="6024" text-anchor="start" style="fill:var(--c-text);font-size:11px">TP AllReduce</text>
<text x="232" y="6024" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVLink</text>
<text x="370" y="6024" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her katmanda 2×</text>
<text x="40" y="6042" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP aktivasyonu · DP gradyanı</text>
<text x="232" y="6042" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NIC → leaf i</text>
<text x="370" y="6042" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">mikrobatch · adım</text>
<text x="40" y="6060" text-anchor="start" style="fill:var(--c-text);font-size:11px">veri okuma · kontrol noktası</text>
<text x="232" y="6060" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7 → depolama ağı</text>
<text x="370" y="6060" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">adım · ~14 dk</text>
<text x="40" y="6078" text-anchor="start" style="fill:var(--c-text);font-size:11px">veri yükleyici · kopyalar</text>
<text x="232" y="6078" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">CPU → PCIe</text>
<text x="370" y="6078" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sürekli (aynı soket)</text>
<text x="40" y="6104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">B bölümündeki 8 adımın her biri bu yollardan birini kullanır.</text>
<rect x="540.0" y="5882.0" width="260.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="670.0" y="5897.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVLink: Y ≤ 28.672 ÷ 2.200 ≈ 13 ✓</text>
<rect x="540.0" y="5910.0" width="260.0" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="670.0" y="5925.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">ağ: Y ≤ 28.672 ÷ 19.800 ≈ 1,45 ✗</text>
<text x="540" y="5954" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metindeki TP kuralı: bir katman Y GPU&#x27;ya</text>
<text x="540" y="5971" text-anchor="start" style="fill:var(--c-text);font-size:12px">bölününce iletişim ancak Y ≤ F ÷ (C/W)</text>
<text x="540" y="5988" text-anchor="start" style="fill:var(--c-text);font-size:12px">iken hesabın arkasına saklanır (C = 990</text>
<text x="540" y="6005" text-anchor="start" style="fill:var(--c-text);font-size:12px">TFLOP/s, W = yolun bant genişliği).</text>
<text x="540" y="6022" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVLink&#x27;te C/W = 2.200, ağda 19.800&#x27;dür.</text>
<text x="540" y="6039" text-anchor="start" style="fill:var(--c-text);font-size:12px">Llama 3 70B&#x27;de (F = 28.672) TP = 8 kasaya</text>
<text x="540" y="6056" text-anchor="start" style="fill:var(--c-text);font-size:12px">sığar; kasadan taşan TP hesabı aç bırakır.</text>
<text x="540" y="6073" text-anchor="start" style="fill:var(--c-text);font-size:12px">Kasanın sınırı bu yüzden TP&#x27;nin sınırıdır.</text>
<line x1="530.0" y1="5880.0" x2="530.0" y2="6104.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="6144" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Zihinsel model: restoran</text>
<text x="40" y="6162" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki benzetmenin devamı: GPU bir mutfaksa DGX H100, sekiz mutfaklı bir restorandır.</text>
<rect x="40.0" y="6174" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="6174" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="6194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Restoran</text>
<rect x="52.0" y="6202.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="6216.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DGX H100 · 8 GPU</text>
<text x="54.0" y="6240" text-anchor="start" style="fill:var(--c-text);font-size:11px">Sekiz mutfak (GPU) aynı</text>
<text x="54.0" y="6255" text-anchor="start" style="fill:var(--c-text);font-size:11px">binada çalışır; her birinin</text>
<text x="54.0" y="6270" text-anchor="start" style="fill:var(--c-text);font-size:11px">kendi kileri (80 GB HBM3)</text>
<text x="54.0" y="6285" text-anchor="start" style="fill:var(--c-text);font-size:11px">vardır. Binanın toplam gücü</text>
<text x="54.0" y="6300" text-anchor="start" style="fill:var(--c-text);font-size:11px">10,2 kW&#x27;tır.</text>
<text x="54.0" y="6363" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">8 mutfak · 640 GB</text>
<rect x="233.0" y="6174" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="6174" width="4" height="201" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="247.0" y="6194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">İç koridor</text>
<rect x="245.0" y="6202.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.5" y="6216.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink + 4 NVSwitch</text>
<text x="247.0" y="6240" text-anchor="start" style="fill:var(--c-text);font-size:11px">Mutfakları birbirine</text>
<text x="247.0" y="6255" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlayan geniş koridor: her</text>
<text x="247.0" y="6270" text-anchor="start" style="fill:var(--c-text);font-size:11px">mutfak diğer yedisine tek</text>
<text x="247.0" y="6285" text-anchor="start" style="fill:var(--c-text);font-size:11px">kapıdan, 450 GB/s ile</text>
<text x="247.0" y="6300" text-anchor="start" style="fill:var(--c-text);font-size:11px">ulaşır. En sık konuşan iş</text>
<text x="247.0" y="6315" text-anchor="start" style="fill:var(--c-text);font-size:11px">(TP) burada kalır.</text>
<text x="247.0" y="6363" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">450 GB/s · tek sıçrama</text>
<rect x="426.0" y="6174" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="6174" width="4" height="201" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="440.0" y="6194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Müdür odaları</text>
<rect x="438.0" y="6202.0" width="157.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="516.5" y="6216.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 CPU + 2 TB DRAM</text>
<text x="440.0" y="6240" text-anchor="start" style="fill:var(--c-text);font-size:11px">İki müdür vardır, her biri</text>
<text x="440.0" y="6255" text-anchor="start" style="fill:var(--c-text);font-size:11px">dört mutfağa bakar (NUMA).</text>
<text x="440.0" y="6270" text-anchor="start" style="fill:var(--c-text);font-size:11px">Siparişi hazırlar (veri</text>
<text x="440.0" y="6285" text-anchor="start" style="fill:var(--c-text);font-size:11px">yükleyici); kendi yarısında</text>
<text x="440.0" y="6300" text-anchor="start" style="fill:var(--c-text);font-size:11px">hızlıdır, öbür yarıya</text>
<text x="440.0" y="6315" text-anchor="start" style="fill:var(--c-text);font-size:11px">koşmak zaman alır.</text>
<text x="440.0" y="6363" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">PCIe 64 GB/s · UPI</text>
<rect x="619.0" y="6174" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="6174" width="4" height="201" rx="2" style="fill:var(--c-danger);fill-opacity:.85"/>
<text x="633.0" y="6194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kurye kapıları</text>
<rect x="631.0" y="6202.0" width="157.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="709.5" y="6216.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">8 NIC · 400 Gb/s</text>
<text x="633.0" y="6240" text-anchor="start" style="fill:var(--c-text);font-size:11px">Her mutfağın şubelere</text>
<text x="633.0" y="6255" text-anchor="start" style="fill:var(--c-text);font-size:11px">(diğer düğümlere) açılan</text>
<text x="633.0" y="6270" text-anchor="start" style="fill:var(--c-text);font-size:11px">kendi kurye kapısı vardır.</text>
<text x="633.0" y="6285" text-anchor="start" style="fill:var(--c-text);font-size:11px">Koridordan 9 kat dardır;</text>
<text x="633.0" y="6300" text-anchor="start" style="fill:var(--c-text);font-size:11px">ağır iş buraya yalnızca</text>
<text x="633.0" y="6315" text-anchor="start" style="fill:var(--c-text);font-size:11px">katman ve adım sınırlarında</text>
<text x="633.0" y="6330" text-anchor="start" style="fill:var(--c-text);font-size:11px">düşer.</text>
<text x="633.0" y="6363" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">8 × 50 = 400 GB/s</text>
<rect x="36.0" y="6387.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="6404" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Altın kural: en sık konuşan iş iç koridorda kalır. TP her katmanda konuştuğu için kasadan çıkmaz; kurye kapısına yalnızca katman</text>
<text x="46" y="6419" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">sınırlarındaki (PP) ve adım sonundaki (DP) daha seyrek trafik düşer.</text>
<line x1="420.0" y1="6444" x2="420.0" y2="6466" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="6459" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 32 düğüm = 1 SU · × 4 SU = SuperPOD</text>
<rect x="28" y="6469" width="784" height="2484" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="6495" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4 · Küme: DGX SuperPOD</text>
<text x="40" y="6515" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">4 SU × 32 yuva, bir yuva UFM&#x27;e: 127 DGX, 1.016 H100. Bu kart metindeki “Düğümler ötesi ölçekleme (Scale-out): SuperPod ve</text>
<text x="40" y="6532" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">rail-optimized fat-tree topolojisi” alt bölümünü çizer; ağın port hesabı ve trafik yolları 4b kartındadır.</text>
<text x="40" y="6562" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Hiyerarşiyi hatırla: parçadan bütüne</text>
<rect x="40.0" y="6572" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="6586" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="126.0" y="6600" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">tek GPU modülü</text>
<line x1="214.0" y1="6592" x2="234.0" y2="6592" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="236.0" y="6572" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="322.0" y="6586" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="322.0" y="6600" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPU&#x27;lu sunucu</text>
<line x1="410.0" y1="6592" x2="430.0" y2="6592" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="432.0" y="6572" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="518.0" y="6586" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="518.0" y="6600" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 DGX = 256 GPU</text>
<line x1="606.0" y1="6592" x2="626.0" y2="6592" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="628.0" y="6572" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="714.0" y="6586" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="714.0" y="6600" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 SU = 127 DGX</text>
<text x="40" y="6632" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">H100 SXM5 modülleri anakarta değil, 8 GPU&#x27;yu ve 4 NVSwitch&#x27;i taşıyan HGX taban kartına takılır; DGX H100&#x27;ün iki işlemcisi Intel Xeon</text>
<text x="40" y="6647" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">8480C&#x27;dir. SuperPOD 32 DGX&#x27;lik birimlerden (SU) kurulur; referans tasarım 4 SU&#x27;dur.</text>
<text x="40" y="6679" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Leaf ve spine nedir?</text>
<rect x="40" y="6691" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="48.0" y="6699" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="162.0" y="6714.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf (yaprak) switch</text>
<text x="52" y="6741" text-anchor="start" style="fill:var(--c-text);font-size:12px">Sunucuların doğrudan takıldığı alt</text>
<text x="52" y="6758" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmandır; ağacın uçları, yani</text>
<text x="52" y="6775" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;lara en yakın switch&#x27;lerdir.</text>
<text x="52" y="6798" text-anchor="start" style="fill:var(--c-text);font-size:12px">SuperPOD&#x27;da: SU başına 8 tane, her</text>
<text x="52" y="6815" text-anchor="start" style="fill:var(--c-text);font-size:12px">ray için bir tane. 64 portunun</text>
<text x="52" y="6832" text-anchor="start" style="fill:var(--c-text);font-size:12px">32&#x27;si DGX&#x27;lere iner, 32&#x27;si</text>
<text x="52" y="6849" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine&#x27;lara çıkar.</text>
<rect x="296" y="6691" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="304.0" y="6699" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="418.0" y="6714.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Spine (omurga) switch</text>
<text x="308" y="6741" text-anchor="start" style="fill:var(--c-text);font-size:12px">Leaf&#x27;leri birbirine bağlayan üst</text>
<text x="308" y="6758" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmandır; spine&#x27;a hiçbir sunucu</text>
<text x="308" y="6775" text-anchor="start" style="fill:var(--c-text);font-size:12px">takılmaz.</text>
<text x="308" y="6798" text-anchor="start" style="fill:var(--c-text);font-size:12px">SuperPOD&#x27;da: toplam 16 tane; her</text>
<text x="308" y="6815" text-anchor="start" style="fill:var(--c-text);font-size:12px">biri 32 leaf&#x27;in hepsine ikişer</text>
<text x="308" y="6832" text-anchor="start" style="fill:var(--c-text);font-size:12px">hatla bağlıdır ve 64 portu böyle</text>
<text x="308" y="6849" text-anchor="start" style="fill:var(--c-text);font-size:12px">dolar.</text>
<text x="552" y="6705" text-anchor="start" style="fill:var(--c-text);font-size:12px">İki katmanın kuralı basittir: her leaf</text>
<text x="552" y="6722" text-anchor="start" style="fill:var(--c-text);font-size:12px">her spine&#x27;a bağlanır. Böylece herhangi</text>
<text x="552" y="6739" text-anchor="start" style="fill:var(--c-text);font-size:12px">iki sunucu en fazla leaf → spine → leaf</text>
<text x="552" y="6756" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoluyla buluşur ve küme büyüdükçe yol</text>
<text x="552" y="6773" text-anchor="start" style="fill:var(--c-text);font-size:12px">uzamaz. Sunucu eklemek için leaf, bant</text>
<text x="552" y="6790" text-anchor="start" style="fill:var(--c-text);font-size:12px">genişliği eklemek için spine eklenir. Bu</text>
<text x="552" y="6807" text-anchor="start" style="fill:var(--c-text);font-size:12px">düzen telefon santrallerinden gelir</text>
<text x="552" y="6824" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Clos, 1953); yukarı çıktıkça</text>
<text x="552" y="6841" text-anchor="start" style="fill:var(--c-text);font-size:12px">bağlantıların kalınlaştığı hâline</text>
<text x="552" y="6858" text-anchor="start" style="fill:var(--c-text);font-size:12px">fat-tree topolojisi denir</text>
<text x="552" y="6875" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Leiserson, 1985).</text>
<text x="552" y="6897" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Benzetme: leaf bir mahallenin santralidir,</text>
<text x="552" y="6912" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">spine şehirler arasındaki ana hat.</text>
<text x="40" y="6950" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Ne neyi yapar?</text>
<text x="168" y="6976" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Görevi</text>
<text x="384" y="6976" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">İçinde</text>
<text x="600" y="6976" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Bağlantı ve paralellik</text>
<line x1="40.0" y1="6985" x2="800.0" y2="6985" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="6998" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="7014" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="168" y="7014" text-anchor="start" style="fill:var(--c-text);font-size:12px">modelin matematiğini yapar; 132</text>
<text x="168" y="7031" text-anchor="start" style="fill:var(--c-text);font-size:12px">SM&#x27;deki Tensor Core&#x27;lar matris</text>
<text x="168" y="7048" text-anchor="start" style="fill:var(--c-text);font-size:12px">çarpımlarını yürütür.</text>
<text x="384" y="7014" text-anchor="start" style="fill:var(--c-text);font-size:12px">ağırlıkların, aktivasyonların ve</text>
<text x="384" y="7031" text-anchor="start" style="fill:var(--c-text);font-size:12px">optimizer durumunun bu GPU&#x27;ya</text>
<text x="384" y="7048" text-anchor="start" style="fill:var(--c-text);font-size:12px">düşen payı, 80 GB HBM3&#x27;te.</text>
<text x="600" y="7014" text-anchor="start" style="fill:var(--c-text);font-size:12px">18 NVLink ile kasadaki diğer</text>
<text x="600" y="7031" text-anchor="start" style="fill:var(--c-text);font-size:12px">7 GPU&#x27;ya, PCIe ile CPU&#x27;ya ve</text>
<text x="600" y="7048" text-anchor="start" style="fill:var(--c-text);font-size:12px">kendi NIC&#x27;ine ulaşır; her</text>
<text x="600" y="7065" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;nun bir TP, bir PP ve bir DP</text>
<text x="600" y="7082" text-anchor="start" style="fill:var(--c-text);font-size:12px">koordinatı vardır.</text>
<line x1="40.0" y1="7092" x2="800.0" y2="7092" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="7106" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="7122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="168" y="7122" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 GPU&#x27;yu tek bir NVLink alanında</text>
<text x="168" y="7139" text-anchor="start" style="fill:var(--c-text);font-size:12px">birleştirir; iki Xeon işlemci</text>
<text x="168" y="7156" text-anchor="start" style="fill:var(--c-text);font-size:12px">işletim sistemini, veri</text>
<text x="168" y="7173" text-anchor="start" style="fill:var(--c-text);font-size:12px">yükleyiciyi ve GPU&#x27;lara iş</text>
<text x="168" y="7190" text-anchor="start" style="fill:var(--c-text);font-size:12px">gönderen süreçleri koşturur.</text>
<text x="384" y="7122" text-anchor="start" style="fill:var(--c-text);font-size:12px">HGX taban kartında 8 GPU ve 4</text>
<text x="384" y="7139" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch; ayrıca 2 TB RAM,</text>
<text x="384" y="7156" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 × 3,84 TB NVMe önbellek ve 8 +</text>
<text x="384" y="7173" text-anchor="start" style="fill:var(--c-text);font-size:12px">2 ağ kartı.</text>
<text x="600" y="7122" text-anchor="start" style="fill:var(--c-text);font-size:12px">tensör paralelliği (TP 8) bu</text>
<text x="600" y="7139" text-anchor="start" style="fill:var(--c-text);font-size:12px">kasanın içinde kalır; kasa ağa</text>
<text x="600" y="7156" text-anchor="start" style="fill:var(--c-text);font-size:12px">her GPU için bir ray olmak üzere</text>
<text x="600" y="7173" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 rayla çıkar.</text>
<line x1="40.0" y1="7200" x2="800.0" y2="7200" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="7214" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="7230" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="168" y="7230" text-anchor="start" style="fill:var(--c-text);font-size:12px">kurulumun ve ağın tekrar eden</text>
<text x="168" y="7247" text-anchor="start" style="fill:var(--c-text);font-size:12px">birimidir: 32 DGX (8 kabin × 4)</text>
<text x="168" y="7264" text-anchor="start" style="fill:var(--c-text);font-size:12px">ve onları bağlayan 8 leaf</text>
<text x="168" y="7281" text-anchor="start" style="fill:var(--c-text);font-size:12px">switch.</text>
<text x="384" y="7230" text-anchor="start" style="fill:var(--c-text);font-size:12px">256 GPU; her ray tek bir</text>
<text x="384" y="7247" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf&#x27;tir, yani Leaf-xx-0i bütün</text>
<text x="384" y="7264" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümlerin NIC i&#x27;sini toplar.</text>
<text x="600" y="7230" text-anchor="start" style="fill:var(--c-text);font-size:12px">aynı numaralı GPU&#x27;lar tek</text>
<text x="600" y="7247" text-anchor="start" style="fill:var(--c-text);font-size:12px">sıçramada buluşur; boru hattı ve</text>
<text x="600" y="7264" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri paralelliğinin düğümler</text>
<text x="600" y="7281" text-anchor="start" style="fill:var(--c-text);font-size:12px">arası trafiği önce burada, ray</text>
<text x="600" y="7298" text-anchor="start" style="fill:var(--c-text);font-size:12px">ray akar.</text>
<line x1="40.0" y1="7308" x2="800.0" y2="7308" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="7322" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="7338" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="168" y="7338" text-anchor="start" style="fill:var(--c-text);font-size:12px">SU&#x27;ları spine katmanıyla tek bir</text>
<text x="168" y="7355" text-anchor="start" style="fill:var(--c-text);font-size:12px">engelsiz ağda birleştirir;</text>
<text x="168" y="7372" text-anchor="start" style="fill:var(--c-text);font-size:12px">yanına ayrı depolama ve yönetim</text>
<text x="168" y="7389" text-anchor="start" style="fill:var(--c-text);font-size:12px">ağlarını, UFM&#x27;i ve Base Command</text>
<text x="168" y="7406" text-anchor="start" style="fill:var(--c-text);font-size:12px">ile Slurm gibi yazılımları</text>
<text x="168" y="7423" text-anchor="start" style="fill:var(--c-text);font-size:12px">ekler.</text>
<text x="384" y="7338" text-anchor="start" style="fill:var(--c-text);font-size:12px">127 DGX = 1.016 GPU; tek bir</text>
<text x="384" y="7355" text-anchor="start" style="fill:var(--c-text);font-size:12px">eğitim işi bütün kümeye</text>
<text x="384" y="7372" text-anchor="start" style="fill:var(--c-text);font-size:12px">yayılabilir.</text>
<text x="600" y="7338" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri paralelliği bütün kümeye</text>
<text x="600" y="7355" text-anchor="start" style="fill:var(--c-text);font-size:12px">yayılır, raylar arası trafik bir</text>
<text x="600" y="7372" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine üzerinden döner; Slurm işi</text>
<text x="600" y="7389" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümlere yerleştirir ve arızalı</text>
<text x="600" y="7406" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümü devreden çıkarır.</text>
<rect x="40.0" y="7441" width="760.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="7461" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Neden 128 değil 127?</tspan> Kâğıt üstünde 4 SU × 32 = 128 DGX eder. NVIDIA&#x27;nın yerleşim düzeninde ise bir çift UFM cihazı bir</text>
<text x="52" y="7478" text-anchor="start" style="fill:var(--c-text);font-size:12px">DGX&#x27;in yerini alır (NVIDIA&#x27;nın şemasında SU 4&#x27;teki C-UFM 1 ve 2); tam bir SuperPOD bu yüzden en fazla 127 DGX, yani</text>
<text x="52" y="7495" text-anchor="start" style="fill:var(--c-text);font-size:12px">1.016 GPU&#x27;dur. Kablo tablosu da bunu gösterir: 127 × 8 + 4 UFM = 1.020 kablo. Daha küçük kurulumlar tek SU&#x27;dan başlar</text>
<text x="52" y="7512" text-anchor="start" style="fill:var(--c-text);font-size:12px">(31–127 DGX).</text>
<rect x="36" y="7537" width="768" height="392" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="214.0" y="7551" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="252.0" y="7566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-01</text>
<rect x="318.0" y="7551" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="7566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-02</text>
<rect x="530.0" y="7551" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="568.0" y="7566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-16</text>
<text x="462" y="7566" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700">⋯</text>
<text x="620" y="7566" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700</text>
<rect x="46" y="7607" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="52.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="83.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-01</text>
<rect x="120.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="151.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-02</text>
<rect x="206.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-08</text>
<text x="194" y="7634" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="89.0" y1="7693" x2="83.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="7693" x2="151.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="7693" x2="237.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="54" y="7693" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="89" y="7709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-001</text>
<text x="89" y="7724" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="175.0" y1="7693" x2="83.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="7693" x2="151.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="7693" x2="237.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="140" y="7693" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="175" y="7709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-032</text>
<text x="175" y="7724" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="132" y="7716" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="56" y="7771" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 1</text>
<text x="98" y="7771" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-001 … 032 · 8 leaf</text>
<rect x="290" y="7607" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="296.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="327.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-01</text>
<rect x="364.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="395.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-02</text>
<rect x="450.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="481.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-08</text>
<text x="438" y="7634" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="333.0" y1="7693" x2="327.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="7693" x2="395.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="7693" x2="481.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="298" y="7693" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="333" y="7709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-033</text>
<text x="333" y="7724" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="419.0" y1="7693" x2="327.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="7693" x2="395.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="7693" x2="481.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="384" y="7693" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="419" y="7709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-064</text>
<text x="419" y="7724" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="376" y="7716" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="300" y="7771" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 2</text>
<text x="342" y="7771" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-033 … 064 · 8 leaf</text>
<rect x="574" y="7607" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="580.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="611.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-01</text>
<rect x="648.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="679.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-02</text>
<rect x="734.0" y="7619" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="765.0" y="7633" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-08</text>
<text x="722" y="7634" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="617.0" y1="7693" x2="611.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="7693" x2="679.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="7693" x2="765.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="582" y="7693" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="617" y="7709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-097</text>
<text x="617" y="7724" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="703.0" y1="7693" x2="611.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="7693" x2="679.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="7693" x2="765.0" y2="7639" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="668" y="7693" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="703" y="7709" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-127</text>
<text x="703" y="7724" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="660" y="7716" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="584" y="7771" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 4</text>
<text x="626" y="7771" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-097 … 127 + C-UFM</text>
<text x="545" y="7699" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">⋯</text>
<text x="545" y="7719" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SU 3</text>
<path d="M746 7708H743V7639" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="7699" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="7712" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 1</text>
<path d="M746 7732H740V7639" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="7723" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="7736" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 2</text>
<line x1="252.0" y1="7573" x2="83.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="151.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="237.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="327.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="395.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="481.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="611.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="679.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="7573" x2="765.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="83.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="151.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="237.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="327.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="395.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="481.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="611.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="679.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="7573" x2="765.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="83.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="151.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="237.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="327.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="395.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="481.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="611.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="679.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="7573" x2="765.0" y2="7619" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="48.0" y1="7805" x2="70.0" y2="7805" style="stroke:var(--c-danger);stroke-width:2.4"/>
<text x="76" y="7809" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine–leaf: 2 × 400 Gb/s</text>
<line x1="300.0" y1="7805" x2="322.0" y2="7805" style="stroke:var(--c-warn);stroke-width:2.4"/>
<text x="328" y="7809" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX–leaf: 400 Gb/s</text>
<line x1="552.0" y1="7805" x2="574.0" y2="7805" style="stroke:var(--c-accent);stroke-width:2.4"/>
<text x="580" y="7809" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM–leaf: 200 Gb/s</text>
<text x="48" y="7833" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Hat renkleri NVIDIA&#x27;nın şemasındaki gibidir ve burada kablo türünü gösterir (posterin sütun renkleri değil). Adlar da oradan:</text>
<text x="48" y="7848" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Leaf-SU-ray; Leaf-04-08, SU 4&#x27;ün 8. rayıdır ve her DGX&#x27;in NIC 0–7&#x27;si Leaf-xx-01 ile 08 arasına gider. SU 3 (H100-065 … 096) aynı</text>
<text x="48" y="7863" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">düzeni tekrarlar.</text>
<rect x="44.0" y="7877" width="368.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="54" y="7893" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">depolama ağı (ayrı InfiniBand)</text>
<text x="54" y="7909" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 switch · düğüm başına ≥ 40 GB/s</text>
<rect x="428.0" y="7877" width="368.0" height="40" rx="5" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="438.0" y="7893" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">yönetim ağları (Ethernet)</text>
<text x="438.0" y="7909" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">in-band 100 GbE · OOB 1 GbE</text>
<rect x="40.0" y="7941" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="7963" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">127</text>
<text x="50.0" y="7980" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX düğümü</text>
<rect x="234.0" y="7941" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="244.0" y="7963" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1.016</text>
<text x="244.0" y="7980" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<rect x="428.0" y="7941" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="438.0" y="7963" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">32 + 16</text>
<text x="438.0" y="7980" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf + spine</text>
<rect x="622.0" y="7941" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="632.0" y="7963" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">4</text>
<text x="632.0" y="7980" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ayrı ağ</text>
<text x="40" y="8011" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">127 DGX (UFM çifti SU 4&#x27;te)</text>
<text x="420" y="8011" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ray içi: SU&#x27;daki 31 düğüme tek sıçrama</text>
<text x="40" y="8027" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesaplama: 32 leaf + 16 spine</text>
<text x="420" y="8027" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her leaf her spine&#x27;a 2 × 400 Gb/s</text>
<text x="40" y="8043" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:1 fat-tree · SHARPv3</text>
<text x="420" y="8043" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 ağ: hesaplama, depolama, 2 yönetim</text>
<text x="40" y="8087" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Raylar: neden her düğümün NIC i&#x27;si leaf i&#x27;ye gider?</text>
<text x="40" y="8117" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">Raf üstü (ToR) düzen</text>
<rect x="96.0" y="8127.0" width="120.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="8140.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="75.0" y1="8169.0" x2="156.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="156.0" y1="8169.0" x2="156.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="237.0" y1="8169.0" x2="156.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="47.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="55.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="63.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="71.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="79.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="87.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="95.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="103.0" y1="8261.0" x2="75.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="128.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="136.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="144.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="152.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="160.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="168.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="176.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="184.0" y1="8261.0" x2="156.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="209.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="217.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="225.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="233.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="241.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="249.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="257.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="265.0" y1="8261.0" x2="237.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<polyline points="71.0,8261.0 75.0,8187.0 75.0,8169.0 156.0,8145.0 156.0,8169.0 156.0,8187.0 152.0,8261.0" style="fill:none;stroke:var(--c-danger);stroke-width:2.4"/>
<rect x="51.0" y="8169.0" width="48.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="75.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf A</text>
<rect x="132.0" y="8169.0" width="48.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="156.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf B</text>
<rect x="213.0" y="8169.0" width="48.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf C</text>
<rect x="40" y="8235" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="75" y="8251" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX A</text>
<circle cx="47" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="55" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="63" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="71" cy="8261" r="2.6" style="fill:var(--c-danger);fill-opacity:1"/>
<circle cx="79" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="87" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="95" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="103" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="121" y="8235" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="156" y="8251" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX B</text>
<circle cx="128" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="136" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="144" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="152" cy="8261" r="2.6" style="fill:var(--c-danger);fill-opacity:1"/>
<circle cx="160" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="168" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="176" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="184" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="202" y="8235" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="237" y="8251" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX C</text>
<circle cx="209" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="217" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="225" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="233" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="241" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="249" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="257" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="265" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<text x="40" y="8291" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">GPU 3 (A) → GPU 3 (B):</text>
<text x="40" y="8306" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">leaf → spine → leaf, 3 sıçrama</text>
<text x="288" y="8117" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">Ray düzeni (NVIDIA)</text>
<rect x="344.0" y="8127.0" width="120.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="404.0" y="8140.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="301.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="330.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="359.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="388.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="417.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="446.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="475.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="504.0" y1="8169.0" x2="404.0" y2="8145.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.45"/>
<line x1="295.0" y1="8261.0" x2="301.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="303.0" y1="8261.0" x2="330.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="311.0" y1="8261.0" x2="359.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="319.0" y1="8261.0" x2="388.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="327.0" y1="8261.0" x2="417.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="335.0" y1="8261.0" x2="446.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="343.0" y1="8261.0" x2="475.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="351.0" y1="8261.0" x2="504.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="376.0" y1="8261.0" x2="301.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="384.0" y1="8261.0" x2="330.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="392.0" y1="8261.0" x2="359.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="400.0" y1="8261.0" x2="388.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="408.0" y1="8261.0" x2="417.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="416.0" y1="8261.0" x2="446.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="424.0" y1="8261.0" x2="475.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="432.0" y1="8261.0" x2="504.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="457.0" y1="8261.0" x2="301.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="465.0" y1="8261.0" x2="330.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="473.0" y1="8261.0" x2="359.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="481.0" y1="8261.0" x2="388.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="489.0" y1="8261.0" x2="417.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="497.0" y1="8261.0" x2="446.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="505.0" y1="8261.0" x2="475.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<line x1="513.0" y1="8261.0" x2="504.0" y2="8187.0" style="stroke:var(--c-accent-2);stroke-width:0.9;stroke-opacity:.35"/>
<polyline points="319.0,8261.0 388.0,8187.0 400.0,8261.0" style="fill:none;stroke:var(--c-success);stroke-width:2.4"/>
<rect x="288.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="301.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L0</text>
<rect x="317.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L1</text>
<rect x="346.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="359.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2</text>
<rect x="375.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="388.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L3</text>
<rect x="404.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="417.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L4</text>
<rect x="433.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="446.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L5</text>
<rect x="462.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="475.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L6</text>
<rect x="491.0" y="8169.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="504.0" y="8182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L7</text>
<rect x="288" y="8235" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="323" y="8251" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX A</text>
<circle cx="295" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="303" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="311" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="319" cy="8261" r="2.6" style="fill:var(--c-success);fill-opacity:1"/>
<circle cx="327" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="335" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="343" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="351" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="369" y="8235" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="404" y="8251" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX B</text>
<circle cx="376" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="384" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="392" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="400" cy="8261" r="2.6" style="fill:var(--c-success);fill-opacity:1"/>
<circle cx="408" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="416" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="424" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="432" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<rect x="450" y="8235" width="70" height="34" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="485" y="8251" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX C</text>
<circle cx="457" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="465" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="473" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="481" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="489" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="497" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="505" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<circle cx="513" cy="8261" r="2.6" style="fill:var(--c-accent-2);fill-opacity:0.6"/>
<text x="288" y="8291" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">GPU 3 (A) → GPU 3 (B): tek leaf, 1</text>
<text x="288" y="8306" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">sıçrama</text>
<text x="40" y="8328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Çizimde SU başına 3 düğüm ve düğüm başına 8 NIC gösterildi; yeşil ve kırmızı hat aynı</text>
<text x="40" y="8343" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">mesajın yoludur.</text>
<text x="540" y="8111" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metindeki “rail-optimized” kablolama: her</text>
<text x="540" y="8128" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümdeki i. GPU&#x27;nun NIC&#x27;i aynı leaf i&#x27;ye</text>
<text x="540" y="8145" text-anchor="start" style="fill:var(--c-text);font-size:12px">takılır. TP 8 kasanın içinde kaldığı için</text>
<text x="540" y="8162" text-anchor="start" style="fill:var(--c-text);font-size:12px">her düğümde GPU i modelin aynı dilimini</text>
<text x="540" y="8179" text-anchor="start" style="fill:var(--c-text);font-size:12px">taşır; DP gradyanları ve PP aktivasyonları</text>
<text x="540" y="8196" text-anchor="start" style="fill:var(--c-text);font-size:12px">da bu yüzden aynı numaralı GPU&#x27;lar</text>
<text x="540" y="8213" text-anchor="start" style="fill:var(--c-text);font-size:12px">arasında akar. Ray düzeninde bu trafik SU</text>
<text x="540" y="8230" text-anchor="start" style="fill:var(--c-text);font-size:12px">içinde tek bir leaf&#x27;ten geçer ve spine&#x27;a</text>
<text x="540" y="8247" text-anchor="start" style="fill:var(--c-text);font-size:12px">hiç çıkmaz; klasik raf üstü düzende aynı</text>
<text x="540" y="8264" text-anchor="start" style="fill:var(--c-text);font-size:12px">mesaj leaf → spine → leaf yolunu izler.</text>
<text x="540" y="8281" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bedeli raylar arası trafiktir (GPU 3 → GPU</text>
<text x="540" y="8298" text-anchor="start" style="fill:var(--c-text);font-size:12px">5): o spine&#x27;dan geçer ya da NCCL&#x27;in PXN&#x27;i</text>
<text x="540" y="8315" text-anchor="start" style="fill:var(--c-text);font-size:12px">onu önce NVLink ile doğru raya taşır (4b).</text>
<line x1="530.0" y1="8099.0" x2="530.0" y2="8343.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="8383" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Ölçek merdiveni: SM&#x27;den 16.384 GPU&#x27;ya</text>
<text x="40" y="8413" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">basamak</text>
<text x="252" y="8413" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPU</text>
<text x="268" y="8413" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">BF16 hesap (log)</text>
<text x="676" y="8413" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">HBM</text>
<text x="690" y="8413" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">bir üste çıkış</text>
<line x1="40.0" y1="8420.0" x2="800.0" y2="8420.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="8431.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="8445.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SM</text>
<text x="252" y="8445" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1/132</text>
<rect x="268" y="8433" width="29.8" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="303.8" y="8445" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">7,5 TFLOP/s</text>
<text x="676" y="8445" text-anchor="end" style="fill:var(--c-text);font-size:11px">512 KiB SRAM</text>
<text x="690" y="8445" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">L2 · HBM (ortak)</text>
<rect x="40.0" y="8457.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="8471.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100</text>
<text x="146" y="8471" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 132 SM</text>
<text x="252" y="8471" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1</text>
<rect x="268" y="8459" width="101.9" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="375.9" y="8471" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">990 TFLOP/s</text>
<text x="676" y="8471" text-anchor="end" style="fill:var(--c-text);font-size:11px">80 GB</text>
<text x="690" y="8471" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink 450 GB/s</text>
<rect x="40.0" y="8483.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="8497.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="146" y="8497" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 8</text>
<text x="252" y="8497" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8</text>
<rect x="268" y="8485" width="132.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="406.6" y="8497" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">7,9 PFLOP/s</text>
<text x="676" y="8497" text-anchor="end" style="fill:var(--c-text);font-size:11px">640 GB</text>
<text x="690" y="8497" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağa 400 GB/s</text>
<rect x="40.0" y="8509.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="8523.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="146" y="8523" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 32</text>
<text x="252" y="8523" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">256</text>
<rect x="268" y="8511" width="183.7" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="457.7" y="8523" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">253 PFLOP/s</text>
<text x="676" y="8523" text-anchor="end" style="fill:var(--c-text);font-size:11px">20,5 TB</text>
<text x="690" y="8523" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine&#x27;a 12,8 TB/s</text>
<rect x="40.0" y="8535.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="8549.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SuperPOD</text>
<text x="146" y="8549" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 4 − 1</text>
<text x="252" y="8549" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1.016</text>
<rect x="268" y="8537" width="204.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="478.1" y="8549" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">1,0 EFLOP/s</text>
<text x="676" y="8549" text-anchor="end" style="fill:var(--c-text);font-size:11px">81 TB</text>
<text x="690" y="8549" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine 51,2 TB/s</text>
<rect x="40.0" y="8561.0" width="96.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="8575.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">64 SU</text>
<text x="146" y="8575" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">× 16</text>
<text x="252" y="8575" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16.384</text>
<rect x="268" y="8563" width="245.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="519.1" y="8575" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16,2 EFLOP/s</text>
<text x="676" y="8575" text-anchor="end" style="fill:var(--c-text);font-size:11px">1,31 PB</text>
<text x="690" y="8575" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">core katmanı</text>
<text x="40" y="8601" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Hesap BF16 yoğun tepe değeridir (990 TFLOP/s × GPU sayısı); gerçek eğitimde MFU %40 civarıdır (E). 64 SU, B ve C bölümlerinin düşündüğü</text>
<text x="40" y="8616" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">ölçektir: 2.048 DGX = 16.384 H100. Her basamakta hesap N kat büyür, dışarıya açılan yol ise GPU başına daralır:</text>
<text x="40" y="8631" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">3,35 TB/s → 450 GB/s → 50 GB/s.</text>
<text x="40" y="8671" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Zihinsel model: restoran zinciri</text>
<text x="40" y="8689" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki benzetmenin devamı: düğüm sekiz mutfaklı bir restoransa SuperPOD bir restoran zinciridir.</text>
<rect x="40.0" y="8701" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="8701" width="4" height="186" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="8721" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Restoran zinciri</text>
<rect x="52.0" y="8729.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="8743.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SuperPOD · 127 DGX</text>
<text x="54.0" y="8767" text-anchor="start" style="fill:var(--c-text);font-size:11px">32 restoran bir mahalleyi</text>
<text x="54.0" y="8782" text-anchor="start" style="fill:var(--c-text);font-size:11px">(SU) oluşturur, zincirde 4</text>
<text x="54.0" y="8797" text-anchor="start" style="fill:var(--c-text);font-size:11px">mahalle vardır. Bir yuva</text>
<text x="54.0" y="8812" text-anchor="start" style="fill:var(--c-text);font-size:11px">genel müdürlüğe (UFM)</text>
<text x="54.0" y="8827" text-anchor="start" style="fill:var(--c-text);font-size:11px">ayrıldığı için 128 değil</text>
<text x="54.0" y="8842" text-anchor="start" style="fill:var(--c-text);font-size:11px">127 restoran çalışır.</text>
<text x="54.0" y="8875" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">127 DGX · 1.016 GPU</text>
<rect x="233.0" y="8701" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="8701" width="4" height="186" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="247.0" y="8721" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Mahalle santrali</text>
<rect x="245.0" y="8729.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.5" y="8743.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf · 32 adet</text>
<text x="247.0" y="8767" text-anchor="start" style="fill:var(--c-text);font-size:11px">Her mahallede 8 santral</text>
<text x="247.0" y="8782" text-anchor="start" style="fill:var(--c-text);font-size:11px">vardır, her biri bir rayın</text>
<text x="247.0" y="8797" text-anchor="start" style="fill:var(--c-text);font-size:11px">hattını toplar: bütün</text>
<text x="247.0" y="8812" text-anchor="start" style="fill:var(--c-text);font-size:11px">restoranların 3 numaralı</text>
<text x="247.0" y="8827" text-anchor="start" style="fill:var(--c-text);font-size:11px">mutfağı aynı santrale</text>
<text x="247.0" y="8842" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlıdır.</text>
<text x="247.0" y="8875" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">SU başına 8 · ray başına 1</text>
<rect x="426.0" y="8701" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="8701" width="4" height="186" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="440.0" y="8721" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Şehirlerarası hat</text>
<rect x="438.0" y="8729.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="516.5" y="8743.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine · 16 adet</text>
<text x="440.0" y="8767" text-anchor="start" style="fill:var(--c-text);font-size:11px">Mahalleleri birbirine</text>
<text x="440.0" y="8782" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlar; her santral her ana</text>
<text x="440.0" y="8797" text-anchor="start" style="fill:var(--c-text);font-size:11px">hatta iki kabloyla ulaşır.</text>
<text x="440.0" y="8812" text-anchor="start" style="fill:var(--c-text);font-size:11px">Aşağıdan gelen kadar yukarı</text>
<text x="440.0" y="8827" text-anchor="start" style="fill:var(--c-text);font-size:11px">çıkabildiği için tıkanmaz</text>
<text x="440.0" y="8842" text-anchor="start" style="fill:var(--c-text);font-size:11px">(1:1).</text>
<text x="440.0" y="8875" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">2 × 400 Gb/s · 51,2 TB/s</text>
<rect x="619.0" y="8701" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="8701" width="4" height="186" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="633.0" y="8721" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Depo ve müdürlük</text>
<rect x="631.0" y="8729.0" width="157.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="709.5" y="8743.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">depolama + yönetim ağları</text>
<text x="633.0" y="8767" text-anchor="start" style="fill:var(--c-text);font-size:11px">Malzeme (veri, kontrol</text>
<text x="633.0" y="8782" text-anchor="start" style="fill:var(--c-text);font-size:11px">noktası) ayrı yoldan merkez</text>
<text x="633.0" y="8797" text-anchor="start" style="fill:var(--c-text);font-size:11px">depodan gelir; müdürlük</text>
<text x="633.0" y="8812" text-anchor="start" style="fill:var(--c-text);font-size:11px">(UFM, Base Command) hatları</text>
<text x="633.0" y="8827" text-anchor="start" style="fill:var(--c-text);font-size:11px">izler, arızalı restoranı</text>
<text x="633.0" y="8842" text-anchor="start" style="fill:var(--c-text);font-size:11px">devreden çıkarır.</text>
<text x="633.0" y="8875" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">≥ 40 GB/s/düğüm · UFM</text>
<rect x="36.0" y="8899.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="8916" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Altın kural: aynı numaralı mutfaklar aynı santrale bağlı olduğu için en sık toplantı (DP AllReduce) mahalle içinde tek sıçramada</text>
<text x="46" y="8931" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">biter; şehirlerarası hatta yalnızca raylar ve mahalleler arası trafik çıkar.</text>
<line x1="420.0" y1="8956" x2="420.0" y2="8978" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="8971" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">SuperPOD&#x27;un ağını yakından inceleyelim</text>
<rect x="28" y="8981" width="784" height="1913" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="9007" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4b · SuperPOD&#x27;u yakından incele</text>
<text x="40" y="9027" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Aynı 127 DGX&#x27;i bu kez ağ gözüyle okuyalım: neden tıkanmaz, trafik hangi yolu izler, bir AllReduce ne kadar sürer, ağ nereye</text>
<text x="40" y="9044" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">kadar büyür? Kart metindeki “Düğümler ötesi ölçekleme” ve “Kolektif iletişim (Collectives): Ring algoritmalarından</text>
<text x="40" y="9061" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">donanımsal SHARP&#x27;a” alt bölümlerini çizer.</text>
<text x="40" y="9091" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Ağ neden tıkanmaz (1:1)? Bir leaf&#x27;in 64 portunu sayalım</text>
<text x="40" y="9117" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf i (ray i): 32 port aşağı, 32 port yukarı, her spine&#x27;a 2 kablo</text>
<text x="40" y="9144" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16 spine</text>
<line x1="110.5" y1="9149.0" x2="110.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="123.5" y1="9149.0" x2="123.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="136.5" y1="9149.0" x2="136.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="149.5" y1="9149.0" x2="149.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="162.5" y1="9149.0" x2="162.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="175.5" y1="9149.0" x2="175.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="188.5" y1="9149.0" x2="188.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="201.5" y1="9149.0" x2="201.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="214.5" y1="9149.0" x2="214.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="227.5" y1="9149.0" x2="227.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="240.5" y1="9149.0" x2="240.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="253.5" y1="9149.0" x2="253.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="266.5" y1="9149.0" x2="266.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="279.5" y1="9149.0" x2="279.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="292.5" y1="9149.0" x2="292.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="305.5" y1="9149.0" x2="305.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="318.5" y1="9149.0" x2="318.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="331.5" y1="9149.0" x2="331.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="344.5" y1="9149.0" x2="344.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="357.5" y1="9149.0" x2="357.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="370.5" y1="9149.0" x2="370.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="383.5" y1="9149.0" x2="383.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="396.5" y1="9149.0" x2="396.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="409.5" y1="9149.0" x2="409.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="422.5" y1="9149.0" x2="422.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="435.5" y1="9149.0" x2="435.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="448.5" y1="9149.0" x2="448.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="461.5" y1="9149.0" x2="461.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="474.5" y1="9149.0" x2="474.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="487.5" y1="9149.0" x2="487.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="500.5" y1="9149.0" x2="500.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="513.5" y1="9149.0" x2="513.5" y2="9187.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="106.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="117.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="132.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="143.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="158.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="169.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="184.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="195.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="210.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="221.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="236.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="247.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="262.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="273.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<rect x="288.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="299.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<rect x="314.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="325.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<rect x="340.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="351.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">10</text>
<rect x="366.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="377.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">11</text>
<rect x="392.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="403.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">12</text>
<rect x="418.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="429.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">13</text>
<rect x="444.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="455.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">14</text>
<rect x="470.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="481.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">15</text>
<rect x="496.0" y="9131.0" width="22.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="9144.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16</text>
<line x1="110.5" y1="9225.0" x2="110.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="123.5" y1="9225.0" x2="123.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="136.5" y1="9225.0" x2="136.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="149.5" y1="9225.0" x2="149.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="162.5" y1="9225.0" x2="162.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="175.5" y1="9225.0" x2="175.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="188.5" y1="9225.0" x2="188.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="201.5" y1="9225.0" x2="201.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="214.5" y1="9225.0" x2="214.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="227.5" y1="9225.0" x2="227.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="240.5" y1="9225.0" x2="240.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="253.5" y1="9225.0" x2="253.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="266.5" y1="9225.0" x2="266.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="279.5" y1="9225.0" x2="279.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="292.5" y1="9225.0" x2="292.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="305.5" y1="9225.0" x2="305.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="318.5" y1="9225.0" x2="318.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="331.5" y1="9225.0" x2="331.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="344.5" y1="9225.0" x2="344.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="357.5" y1="9225.0" x2="357.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="370.5" y1="9225.0" x2="370.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="383.5" y1="9225.0" x2="383.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="396.5" y1="9225.0" x2="396.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="409.5" y1="9225.0" x2="409.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="422.5" y1="9225.0" x2="422.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="435.5" y1="9225.0" x2="435.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="448.5" y1="9225.0" x2="448.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="461.5" y1="9225.0" x2="461.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="474.5" y1="9225.0" x2="474.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="487.5" y1="9225.0" x2="487.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="500.5" y1="9225.0" x2="500.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="513.5" y1="9225.0" x2="513.5" y2="9253.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="104.0" y="9183.0" width="416.0" height="46" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="106.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="106.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="119.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="119.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="132.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="132.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="145.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="145.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="158.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="158.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="171.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="171.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="184.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="184.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="197.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="197.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="210.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="210.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="223.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="223.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="236.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="236.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="249.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="262.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="275.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="275.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="288.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="288.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="301.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="301.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="314.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="314.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="327.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="327.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="340.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="340.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="353.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="353.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="366.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="366.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="379.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="379.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="392.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="405.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="405.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="418.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="418.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="431.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="431.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="444.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="444.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="457.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="457.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="470.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="470.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="483.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="483.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="496.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="496.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="509.0" y="9187" width="9" height="8" rx="1" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="509.0" y="9217" width="9" height="8" rx="1" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="312" y="9210" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">QM9700 · 64 × 400 Gb/s</text>
<text x="40" y="9210" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">leaf i</text>
<rect x="105.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="118.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="131.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="144.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="157.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="170.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="183.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="196.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="209.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="222.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="235.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="248.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="261.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="274.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="287.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="300.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="313.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="326.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="339.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="352.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="365.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="378.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="391.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="404.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="417.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="430.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="443.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="456.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="469.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="482.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="495.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="508.0" y="9253.0" width="11.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="9266" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">32 DGX</text>
<text x="104" y="9287" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX 1</text>
<text x="520" y="9287" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">DGX 32</text>
<text x="312" y="9287" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">her birinin NIC i&#x27;si · 400 Gb/s</text>
<rect x="40.0" y="9301.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="9316.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 leaf × 32 yukarı port = 1.024 kablo = 16 spine × 64 port</text>
<text x="40" y="9345" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Kablo sayısı (NVIDIA tablosu): düğümlerden leaf&#x27;lere 1.020 (127 × 8 + 4 UFM),</text>
<text x="40" y="9360" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf&#x27;lerden spine&#x27;lara 1.024. Küçük kurulumlar aynı kuralı izler: 1 SU = 8 leaf +</text>
<text x="40" y="9375" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 spine, 2 SU = 16 leaf + 8 spine.</text>
<text x="540" y="9117" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her leaf&#x27;in 64 portunun 32&#x27;si düğümlere</text>
<text x="540" y="9134" text-anchor="start" style="fill:var(--c-text);font-size:12px">iner, 32&#x27;si spine&#x27;lara çıkar. Aşağıdan ne</text>
<text x="540" y="9151" text-anchor="start" style="fill:var(--c-text);font-size:12px">kadar trafik girerse yukarı o kadar</text>
<text x="540" y="9168" text-anchor="start" style="fill:var(--c-text);font-size:12px">çıkabilir; bu yüzden aşırı abonelik</text>
<text x="540" y="9185" text-anchor="start" style="fill:var(--c-text);font-size:12px">(oversubscription) yoktur, oran 1:1&#x27;dir.</text>
<text x="540" y="9202" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her leaf her spine&#x27;a bağlı olduğu için</text>
<text x="540" y="9219" text-anchor="start" style="fill:var(--c-text);font-size:12px">herhangi iki GPU en fazla</text>
<text x="540" y="9236" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf → spine → leaf yoluyla buluşur.</text>
<rect x="540.0" y="9252.0" width="260.0" height="188" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="9272" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">51,2 TB/s nereden çıkar?</text>
<text x="552" y="9291" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16 × 64 × 50 GB/s = 51,2 TB/s</text>
<text x="552" y="9307" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1.016 × 50 GB/s = 50,8 TB/s</text>
<text x="552" y="9326" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Port başına 400 Gb/s ÷ 8 = 50 GB/s, tek</text>
<text x="552" y="9341" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yön. Spine katmanının taşıyabildiği, bütün</text>
<text x="552" y="9356" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU&#x27;ların basabildiğinden az değildir.</text>
<text x="552" y="9371" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Karıştırma: NVIDIA&#x27;nın QM9700</text>
<text x="552" y="9386" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kılavuzundaki “51,2 Tb/s” tek switch&#x27;in</text>
<text x="552" y="9401" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">iki yönlü toplamıdır (bit); buradaki</text>
<text x="552" y="9416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">51,2 TB/s (bayt) 16 spine&#x27;ın tek yönlü</text>
<text x="552" y="9431" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">toplamıdır.</text>
<line x1="530.0" y1="9103.0" x2="530.0" y2="9440.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="9480" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Trafik hangi yolu izler? Dört yol, dört bedel</text>
<text x="40" y="9506" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı ray, aynı SU · DP ve PP</text>
<rect x="455.6" y="9494.0" width="64.4" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="487.8" y="9507.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">1 switch</text>
<rect x="40.0" y="9518.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9534.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9518.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9534.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="126.0" y1="9530.0" x2="137.0" y2="9530.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="238.0" y="9518.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="9534.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · B</text>
<line x1="225.0" y1="9530.0" x2="236.0" y2="9530.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9572" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı ray, başka SU · SU&#x27;lar arası DP</text>
<rect x="455.6" y="9560.0" width="64.4" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="487.8" y="9573.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">3 switch</text>
<rect x="40.0" y="9584.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9600.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9584.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9600.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="126.0" y1="9596.0" x2="137.0" y2="9596.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="238.0" y="9584.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="9600.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="225.0" y1="9596.0" x2="236.0" y2="9596.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="337.0" y="9584.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="379.0" y="9600.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3 · SU 2</text>
<line x1="324.0" y1="9596.0" x2="335.0" y2="9596.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="436.0" y="9584.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="478.0" y="9600.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · C</text>
<line x1="423.0" y1="9596.0" x2="434.0" y2="9596.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9638" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Farklı ray, düz yol · MoE AllToAll</text>
<rect x="455.6" y="9626.0" width="64.4" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="487.8" y="9639.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">3 switch</text>
<rect x="40.0" y="9650.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9666.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9650.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9666.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="126.0" y1="9662.0" x2="137.0" y2="9662.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="238.0" y="9650.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="9666.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="225.0" y1="9662.0" x2="236.0" y2="9662.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="337.0" y="9650.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="379.0" y="9666.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="324.0" y1="9662.0" x2="335.0" y2="9662.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="436.0" y="9650.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="478.0" y="9666.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<line x1="423.0" y1="9662.0" x2="434.0" y2="9662.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9704" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Farklı ray, PXN ile</text>
<rect x="401.1" y="9692.0" width="118.9" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="460.6" y="9705.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink + 1 switch</text>
<rect x="40.0" y="9716.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82.0" y="9732.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<rect x="139.0" y="9716.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.24;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="181.0" y="9732.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">NVSwitch</text>
<line x1="126.0" y1="9728.0" x2="137.0" y2="9728.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="238.0" y="9716.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="9732.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · A</text>
<line x1="225.0" y1="9728.0" x2="236.0" y2="9728.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="337.0" y="9716.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="379.0" y="9732.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="324.0" y1="9728.0" x2="335.0" y2="9728.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="436.0" y="9716.0" width="84.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="478.0" y="9732.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<line x1="423.0" y1="9728.0" x2="434.0" y2="9728.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="40" y="9762" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">A ve B aynı SU&#x27;da, C başka bir SU&#x27;da. Mor ok NVLink (kasanın içi), gri ok InfiniBand</text>
<text x="40" y="9777" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(400 Gb/s).</text>
<text x="540" y="9506" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA&#x27;ya göre bir raydaki trafik SU&#x27;daki</text>
<text x="540" y="9523" text-anchor="start" style="fill:var(--c-text);font-size:12px">diğer 31 düğüme her zaman tek sıçramadır;</text>
<text x="540" y="9540" text-anchor="start" style="fill:var(--c-text);font-size:12px">SU&#x27;lar ya da raylar arası trafik spine</text>
<text x="540" y="9557" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmanından geçer. NCCL 2.12&#x27;deki PXN,</text>
<text x="540" y="9574" text-anchor="start" style="fill:var(--c-text);font-size:12px">veriyi önce NVSwitch üzerinden hedefle</text>
<text x="540" y="9591" text-anchor="start" style="fill:var(--c-text);font-size:12px">aynı raydaki GPU&#x27;ya taşır, oradan rayı</text>
<text x="540" y="9608" text-anchor="start" style="fill:var(--c-text);font-size:12px">terk etmeden gönderir. Bir düğümün sekiz</text>
<text x="540" y="9625" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;sunun aynı hedefe giden mesajları</text>
<text x="540" y="9642" text-anchor="start" style="fill:var(--c-text);font-size:12px">böylece tek GPU&#x27;da toplanıp birlikte</text>
<text x="540" y="9659" text-anchor="start" style="fill:var(--c-text);font-size:12px">yollanır; NVIDIA bununla all2all</text>
<text x="540" y="9676" text-anchor="start" style="fill:var(--c-text);font-size:12px">başarımının iki katına çıktığını raporlar.</text>
<rect x="540.0" y="9692.0" width="260.0" height="104" rx="5" style="fill:var(--c-success);fill-opacity:.08;stroke:var(--c-success);stroke-width:1.2"/>
<text x="552" y="9711" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-success);font-weight:700">Gerçek örnek:</tspan> DeepSeek-V3 (2.048 H800)</text>
<text x="552" y="9726" text-anchor="start" style="fill:var(--c-text);font-size:11px">bunun aynasını yapar: token önce IB ile</text>
<text x="552" y="9741" text-anchor="start" style="fill:var(--c-text);font-size:11px">hedef düğümde aynı yerel indeksli GPU&#x27;ya,</text>
<text x="552" y="9756" text-anchor="start" style="fill:var(--c-text);font-size:11px">yani kendi rayından gider, orada NVLink</text>
<text x="552" y="9771" text-anchor="start" style="fill:var(--c-text);font-size:11px">ile uzmanın GPU&#x27;suna aktarılır. Her token</text>
<text x="552" y="9786" text-anchor="start" style="fill:var(--c-text);font-size:11px">en fazla 4 düğüme gönderilir.</text>
<line x1="530.0" y1="9492.0" x2="530.0" y2="9796.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="9836" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Bir AllReduce bu ağda ne kadar sürer? Halka ve SHARP</text>
<text x="40" y="9862" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Örnek: Llama 3 70B, TP 8 × DP 127. GPU başına 70 milyar ÷ 8 = 8,75 milyar</text>
<text x="40" y="9879" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">parametre, BF16 gradyan B = 17,5 GB; DP AllReduce ray i&#x27;deki 127 GPU arasında</text>
<text x="40" y="9896" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">koşar.</text>
<text x="40" y="9924" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">Halka (ring): parçalar GPU&#x27;dan GPU&#x27;ya döner</text>
<line x1="163.6" y1="9946.5" x2="180.4" y2="9951.9" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="193.6" y1="9961.2" x2="204.0" y2="9975.4" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="209.3" y1="9990.7" x2="209.3" y2="10008.3" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="204.6" y1="10023.8" x2="194.2" y2="10038.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="181.3" y1="10047.8" x2="164.6" y2="10053.2" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="148.4" y1="10053.5" x2="131.6" y2="10048.1" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="118.4" y1="10038.8" x2="108.0" y2="10024.6" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="102.7" y1="10009.3" x2="102.7" y2="9991.7" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<line x1="107.4" y1="9976.2" x2="118.4" y2="9961.2" style="stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:4 3;stroke-opacity:.7"/>
<line x1="130.7" y1="9952.2" x2="147.4" y2="9946.8" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<circle cx="156.0" cy="9944.0" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="188.9" cy="9954.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="209.3" cy="9982.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="209.3" cy="10017.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="188.9" cy="10045.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="156.0" cy="10056.0" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="123.1" cy="10045.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="102.7" cy="10017.3" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="102.7" cy="9982.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="123.1" cy="9954.7" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="156" y="9998" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">ray i</text>
<text x="156" y="10013" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">127 GPU</text>
<text x="40" y="10084" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 × 126 adım; her adımda</text>
<text x="40" y="10099" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">B/127 ≈ 138 MB&#x27;lık bir parça komşuya</text>
<text x="40" y="10114" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">geçer.</text>
<text x="288" y="9924" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">SHARP: toplamayı switch yapar</text>
<line x1="318.0" y1="9984.0" x2="383.0" y2="9962.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="302.0" y1="10040.0" x2="311.6" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="318.0" y1="10040.0" x2="318.0" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="334.0" y1="10040.0" x2="324.4" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="378.0" y1="9984.0" x2="397.0" y2="9962.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="362.0" y1="10040.0" x2="371.6" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="378.0" y1="10040.0" x2="378.0" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="394.0" y1="10040.0" x2="384.4" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="438.0" y1="9984.0" x2="411.0" y2="9962.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="422.0" y1="10040.0" x2="431.6" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="438.0" y1="10040.0" x2="438.0" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="454.0" y1="10040.0" x2="444.4" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="498.0" y1="9984.0" x2="425.0" y2="9962.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.4"/>
<line x1="482.0" y1="10040.0" x2="491.6" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="498.0" y1="10040.0" x2="498.0" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="514.0" y1="10040.0" x2="504.4" y2="10014.0" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="364.0" y="9940.0" width="80.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="404.0" y="9954.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Σ spine</text>
<rect x="292.0" y="9988.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="318.0" y="10003.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 1</text>
<circle cx="302" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="318" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="334" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="352.0" y="9988.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="378.0" y="10003.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 2</text>
<circle cx="362" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="378" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="394" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="412.0" y="9988.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="438.0" y="10003.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 3</text>
<circle cx="422" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="438" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="454" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<rect x="472.0" y="9988.0" width="52.0" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="498.0" y="10003.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Σ SU 4</text>
<circle cx="482" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="498" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<circle cx="514" cy="10046" r="5" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="520" y="10068" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">↓ toplam herkese</text>
<text x="288" y="10084" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her GPU B&#x27;yi bir kez yukarı yollar;</text>
<text x="288" y="10099" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf&#x27;ler SU toplamını, spine dört SU&#x27;nun</text>
<text x="288" y="10114" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">toplamını alır, sonuç aynı yoldan aşağı</text>
<text x="288" y="10129" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">çoğaltılır.</text>
<line x1="280.0" y1="9912.0" x2="280.0" y2="10129.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="10147.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="160" y="10162" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">T halka ≈ 2 · (N−1)/N · B/W</text>
<text x="400" y="10162" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">T SHARP ≈ B/W</text>
<line x1="280.0" y1="10151.0" x2="280.0" y2="10165.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10195" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU başına kabloya çıkan veri ve süre (W = 50 GB/s)</text>
<text x="40" y="10219" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Halka</text>
<rect x="104" y="10207" width="260.0" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="370.0" y="10219" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">34,7 GB · 0,69 s</text>
<text x="40" y="10243" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">SHARP</text>
<rect x="104" y="10231" width="131.0" height="16" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="241.0" y="10243" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">17,5 GB · 0,35 s</text>
<text x="540" y="9862" text-anchor="start" style="fill:var(--c-text);font-size:12px">Metindeki formüller: halka AllReduce önce</text>
<text x="540" y="9879" text-anchor="start" style="fill:var(--c-text);font-size:12px">ReduceScatter, sonra AllGather yapar; her</text>
<text x="540" y="9896" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU kabloya 2 · (N−1)/N · B bayt basar,</text>
<text x="540" y="9913" text-anchor="start" style="fill:var(--c-text);font-size:12px">süre ≈ 2B/W. SHARP&#x27;ta toplamayı Quantum-2</text>
<text x="540" y="9930" text-anchor="start" style="fill:var(--c-text);font-size:12px">switch&#x27;lerinin ASIC&#x27;i yapar: GPU</text>
<text x="540" y="9947" text-anchor="start" style="fill:var(--c-text);font-size:12px">gradyanını bir kez yukarı yollar, toplamı</text>
<text x="540" y="9964" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir kez geri alır; trafik yarıya,</text>
<text x="540" y="9981" text-anchor="start" style="fill:var(--c-text);font-size:12px">süre ≈ B/W&#x27;ye iner. Ray düzeni bu ağaca</text>
<text x="540" y="9998" text-anchor="start" style="fill:var(--c-text);font-size:12px">doğal olarak oturur: SU&#x27;daki GPU i&#x27;lerin</text>
<text x="540" y="10015" text-anchor="start" style="fill:var(--c-text);font-size:12px">toplamı leaf i&#x27;de, dört SU&#x27;nun ara toplamı</text>
<text x="540" y="10032" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir spine&#x27;da birleşir.</text>
<text x="540" y="10054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Gerçek eğitimde gradyanlar kovalara bölünüp</text>
<text x="540" y="10069" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">geri yayılımla örtüştürülür; bu süreler adıma</text>
<text x="540" y="10084" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">doğrudan eklenmez. Meta&#x27;nın 405B kümesi RoCE</text>
<text x="540" y="10099" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">kullandığı için SHARP adımı SuperPOD</text>
<text x="540" y="10114" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">senaryosuna aittir (metin, 5. bölüm).</text>
<line x1="530.0" y1="9848.0" x2="530.0" y2="10247.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10287" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Ağ nereye kadar büyür? Üçüncü katman</text>
<text x="40" y="10317" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">SU</text>
<text x="132" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">DGX</text>
<text x="200" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPU</text>
<text x="256" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">leaf</text>
<text x="312" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">spine</text>
<text x="364" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">core</text>
<text x="424" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">katman</text>
<text x="520" y="10317" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">en uzak yol</text>
<line x1="40.0" y1="10324.0" x2="520.0" y2="10324.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="10345" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="132" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">31</text>
<text x="200" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">248</text>
<text x="256" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<text x="312" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="364" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">–</text>
<text x="424" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px">2</text>
<text x="520" y="10345" text-anchor="end" style="fill:var(--c-text);font-size:11px">3 switch</text>
<rect x="36.0" y="10353.0" width="488.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="10367" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="132" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">127</text>
<text x="200" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1.016</text>
<text x="256" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32</text>
<text x="312" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="364" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">–</text>
<text x="424" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px">2</text>
<text x="520" y="10367" text-anchor="end" style="fill:var(--c-text);font-size:11px">3 switch</text>
<text x="40" y="10389" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="132" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="200" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">2.048</text>
<text x="256" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="312" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32</text>
<text x="364" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">–</text>
<text x="424" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px">2</text>
<text x="520" y="10389" text-anchor="end" style="fill:var(--c-text);font-size:11px">3 switch</text>
<line x1="40.0" y1="10399.0" x2="520.0" y2="10399.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="40" y="10413" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">iki katmanın tavanı: 2.048 GPU</text>
<text x="40" y="10433" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16</text>
<text x="132" y="10433" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">512</text>
<text x="200" y="10433" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">4.096</text>
<text x="256" y="10433" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">128</text>
<text x="312" y="10433" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">128</text>
<text x="364" y="10433" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="424" y="10433" text-anchor="end" style="fill:var(--c-text);font-size:11px">3</text>
<text x="520" y="10433" text-anchor="end" style="fill:var(--c-text);font-size:11px">5 switch</text>
<text x="40" y="10455" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">32</text>
<text x="132" y="10455" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.024</text>
<text x="200" y="10455" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8.192</text>
<text x="256" y="10455" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="312" y="10455" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="364" y="10455" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-family:var(--font-mono)">128</text>
<text x="424" y="10455" text-anchor="end" style="fill:var(--c-text);font-size:11px">3</text>
<text x="520" y="10455" text-anchor="end" style="fill:var(--c-text);font-size:11px">5 switch</text>
<text x="40" y="10477" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">64</text>
<text x="132" y="10477" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2.048</text>
<text x="200" y="10477" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16.384</text>
<text x="256" y="10477" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">512</text>
<text x="312" y="10477" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">512</text>
<text x="364" y="10477" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="424" y="10477" text-anchor="end" style="fill:var(--c-text);font-size:11px">3</text>
<text x="520" y="10477" text-anchor="end" style="fill:var(--c-text);font-size:11px">5 switch</text>
<line x1="40.0" y1="10487.0" x2="520.0" y2="10487.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="10507" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">1 ve 4 SU satırları NVIDIA&#x27;nın SuperPOD tablosundan (bir yuva UFM&#x27;e); 8 SU ve üstü</text>
<text x="40" y="10522" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">büyük kurulum tablosundan, orada SU başına 32 DGX sayılır. O tablo son satırı “56 SU”</text>
<text x="40" y="10537" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">diye yazar; 2.048 ÷ 32 = 64.</text>
<rect x="540.0" y="10301.0" width="260.0" height="189" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="10321" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2.048 nereden çıkar?</text>
<text x="552" y="10340" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">64 port = 32 aşağı + 32 yukarı</text>
<text x="552" y="10356" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2 katman: 64 leaf × 32 = 2.048</text>
<text x="552" y="10372" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3 katman: 64 × 32 × 32 = 65.536</text>
<text x="552" y="10391" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Bir spine&#x27;ın 64 portu en fazla 64 leaf&#x27;e</text>
<text x="552" y="10406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">uzanır, her leaf 32 GPU taşır. 8 SU</text>
<text x="552" y="10421" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(64 leaf) bu tavana tam oturur; 16 SU&#x27;da</text>
<text x="552" y="10436" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine&#x27;ların üstüne core katmanı gelir; en</text>
<text x="552" y="10451" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">uzak yol 5 switch&#x27;e (leaf, spine, core,</text>
<text x="552" y="10466" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine, leaf) uzar ve GPU başına bir kablo</text>
<text x="552" y="10481" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">daha eklenir.</text>
<text x="540" y="10512" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Karşılaştırma: Meta&#x27;nın Llama 3 kümesi (RoCE,</text>
<text x="540" y="10527" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">24K GPU) de üç katmanlıdır ama baştan sona</text>
<text x="540" y="10542" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">engelsiz değildir: 3.072 GPU&#x27;luk pod&#x27;lar kendi</text>
<text x="540" y="10557" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">içinde 1:1, pod&#x27;lar arası 1:7.</text>
<line x1="530.0" y1="10299.0" x2="530.0" y2="10557.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10597" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Zihinsel model: zincirin haberleşmesi</text>
<text x="40" y="10615" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Restoran zinciri benzetmesinin devamı: bu kez mutfaklar arasında kimin, hangi hattan, nasıl konuştuğuna bakıyoruz.</text>
<rect x="40.0" y="10627" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="10627" width="4" height="201" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="10647" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı numaralı mutfaklar</text>
<rect x="52.0" y="10655.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.5" y="10669.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ray · leaf i</text>
<text x="54.0" y="10693" text-anchor="start" style="fill:var(--c-text);font-size:11px">Her restoranın 3 numaralı</text>
<text x="54.0" y="10708" text-anchor="start" style="fill:var(--c-text);font-size:11px">mutfağı aynı santrale</text>
<text x="54.0" y="10723" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlıdır. Hepsi aynı tarifi</text>
<text x="54.0" y="10738" text-anchor="start" style="fill:var(--c-text);font-size:11px">(modelin aynı dilimini)</text>
<text x="54.0" y="10753" text-anchor="start" style="fill:var(--c-text);font-size:11px">pişirdiği için en sık onlar</text>
<text x="54.0" y="10768" text-anchor="start" style="fill:var(--c-text);font-size:11px">konuşur ve konuşmaları</text>
<text x="54.0" y="10783" text-anchor="start" style="fill:var(--c-text);font-size:11px">mahalleden çıkmaz.</text>
<text x="54.0" y="10816" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">1 switch · DP, PP</text>
<rect x="233.0" y="10627" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="10627" width="4" height="201" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="247.0" y="10647" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Restoran içi koridor</text>
<rect x="245.0" y="10655.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.5" y="10669.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PXN · NVSwitch</text>
<text x="247.0" y="10693" text-anchor="start" style="fill:var(--c-text);font-size:11px">5 numaralı mutfağa gidecek</text>
<text x="247.0" y="10708" text-anchor="start" style="fill:var(--c-text);font-size:11px">not önce restoranın</text>
<text x="247.0" y="10723" text-anchor="start" style="fill:var(--c-text);font-size:11px">içindeki koridordan kendi 5</text>
<text x="247.0" y="10738" text-anchor="start" style="fill:var(--c-text);font-size:11px">numaralı mutfağına taşınır,</text>
<text x="247.0" y="10753" text-anchor="start" style="fill:var(--c-text);font-size:11px">oradan doğru santrale</text>
<text x="247.0" y="10768" text-anchor="start" style="fill:var(--c-text);font-size:11px">verilir; şehirlerarası</text>
<text x="247.0" y="10783" text-anchor="start" style="fill:var(--c-text);font-size:11px">hatta çıkmaz.</text>
<text x="247.0" y="10816" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">NVLink + 1 switch</text>
<rect x="426.0" y="10627" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="10627" width="4" height="201" rx="2" style="fill:var(--c-success);fill-opacity:.85"/>
<text x="440.0" y="10647" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Santralde oy sayımı</text>
<rect x="438.0" y="10655.0" width="157.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="516.5" y="10669.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP · switch ASIC</text>
<text x="440.0" y="10693" text-anchor="start" style="fill:var(--c-text);font-size:11px">Herkes oyunu (gradyanını)</text>
<text x="440.0" y="10708" text-anchor="start" style="fill:var(--c-text);font-size:11px">elden ele dolaştırmak</text>
<text x="440.0" y="10723" text-anchor="start" style="fill:var(--c-text);font-size:11px">yerine santrale bir kez</text>
<text x="440.0" y="10738" text-anchor="start" style="fill:var(--c-text);font-size:11px">verir; santral sayar,</text>
<text x="440.0" y="10753" text-anchor="start" style="fill:var(--c-text);font-size:11px">sonucu herkese bir kez</text>
<text x="440.0" y="10768" text-anchor="start" style="fill:var(--c-text);font-size:11px">yollar. Hatta dolaşan kâğıt</text>
<text x="440.0" y="10783" text-anchor="start" style="fill:var(--c-text);font-size:11px">yarıya iner.</text>
<text x="440.0" y="10816" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:700">2B yerine B</text>
<rect x="619.0" y="10627" width="181.0" height="201" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="10627" width="4" height="201" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="633.0" y="10647" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ülke omurgası</text>
<rect x="631.0" y="10655.0" width="157.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="709.5" y="10669.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">core · 3. katman</text>
<text x="633.0" y="10693" text-anchor="start" style="fill:var(--c-text);font-size:11px">Zincir 8 mahalleyi aşınca</text>
<text x="633.0" y="10708" text-anchor="start" style="fill:var(--c-text);font-size:11px">şehirlerarası hatların</text>
<text x="633.0" y="10723" text-anchor="start" style="fill:var(--c-text);font-size:11px">üstüne bir omurga eklenir.</text>
<text x="633.0" y="10738" text-anchor="start" style="fill:var(--c-text);font-size:11px">En uzak sipariş artık 3</text>
<text x="633.0" y="10753" text-anchor="start" style="fill:var(--c-text);font-size:11px">değil 5 santralden geçer.</text>
<text x="633.0" y="10816" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">2.048 GPU&#x27;dan sonra</text>
<rect x="36.0" y="10840.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="10857" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Altın kural: trafiği en geniş yola koy. Önce kasanın içi (NVLink, 450 GB/s), sonra ray (tek leaf, 50 GB/s), en son spine ve core.</text>
<text x="46" y="10872" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">TP kasada kalır, DP ve PP rayda akar, MoE&#x27;nin AllToAll&#x27;unu PXN raya çevirir, AllReduce&#x27;u SHARP switch&#x27;te toplar.</text>
<line x1="420.0" y1="10897" x2="420.0" y2="10919" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="10912" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">kümeyi ayakta tutan yapı taşları</text>
<rect x="28" y="10922" width="784" height="2259" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="10948" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">SuperPOD&#x27;un yapı taşları: her biri neden var?</text>
<text x="40" y="10968" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">NVIDIA&#x27;nın 4 SU&#x27;luk referans tasarımı (127 DGX); adetler, güç ve görevler NVIDIA belgelerinden. Şerit rengi parçanın hangi</text>
<text x="40" y="10985" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sütuna hizmet ettiğini gösterir; gri yönetim ve altyapıdır. Kart, metnin 4. bölümündeki depolama alt bölümlerini ve “Cluster</text>
<text x="40" y="11002" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">MTBF” alt bölümünü SuperPOD&#x27;un parçalarına bağlar.</text>
<text x="40" y="11032" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Bir DGX&#x27;in dört kablosu: neden dört ayrı ağ?</text>
<rect x="40" y="11048" width="132" height="218" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="106" y="11068" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100</text>
<rect x="48.0" y="11080.0" width="116.0" height="34" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="106.0" y="11094" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8 × CX-7</text>
<text x="106.0" y="11108" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × 400 Gb/s IB</text>
<line x1="166.0" y1="11097.0" x2="300.0" y2="11097.0" style="stroke:var(--c-accent-2);stroke-width:4"/>
<text x="214" y="11090" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">400 GB/s</text>
<rect x="300.0" y="11080.0" width="220.0" height="34" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="410.0" y="11094" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Hesaplama ağı</text>
<text x="410.0" y="11108" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">48 × QM9700 · 1:1</text>
<rect x="48.0" y="11126.0" width="116.0" height="34" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="106.0" y="11140" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2 × CX-7 portu</text>
<text x="106.0" y="11154" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2 × 400 Gb/s IB</text>
<line x1="166.0" y1="11143.0" x2="300.0" y2="11143.0" style="stroke:var(--c-warn);stroke-width:3"/>
<text x="214" y="11136" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">100 GB/s</text>
<rect x="300.0" y="11126.0" width="220.0" height="34" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="410.0" y="11140" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Depolama ağı</text>
<text x="410.0" y="11154" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700 · ~4:3</text>
<rect x="48.0" y="11172.0" width="116.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="106.0" y="11186" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2 × CX-7 portu</text>
<text x="106.0" y="11200" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2 × 100 GbE</text>
<line x1="166.0" y1="11189.0" x2="300.0" y2="11189.0" style="stroke:var(--c-text-mute);stroke-width:2"/>
<text x="214" y="11182" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">25 GB/s</text>
<rect x="300.0" y="11172.0" width="220.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="410.0" y="11186" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">In-band yönetim</text>
<text x="410.0" y="11200" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × SN4600C · Ethernet</text>
<rect x="48.0" y="11218.0" width="116.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="106.0" y="11232" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">BMC</text>
<text x="106.0" y="11246" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 × 1 GbE bakır</text>
<line x1="166.0" y1="11235.0" x2="300.0" y2="11235.0" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="214" y="11228" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0,125 GB/s</text>
<rect x="300.0" y="11218.0" width="220.0" height="34" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="410.0" y="11232" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">OOB yönetim</text>
<text x="410.0" y="11246" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × SN2201 · Ethernet</text>
<text x="40" y="11288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Kablolar (NVIDIA tablosu): hesaplama 2.040 NDR (1.016 düğüm + 1.024 leaf–spine),</text>
<text x="40" y="11303" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">depolama düğüm başına 2 (254), in-band 254 × 100 GbE, OOB 127 × 1 GbE. Depolama hattı</text>
<text x="40" y="11318" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">100 GB/s taşısa da NVIDIA&#x27;nın hedefi düğüm başına ≥ 40 GB/s&#x27;tir.</text>
<text x="540" y="11058" text-anchor="start" style="fill:var(--c-text);font-size:12px">Hesaplama ağı yalnız GPU&#x27;lar arası trafiği</text>
<text x="540" y="11075" text-anchor="start" style="fill:var(--c-text);font-size:12px">taşır: B&#x27;nin 4., 5. ve 6. adımları</text>
<text x="540" y="11092" text-anchor="start" style="fill:var(--c-text);font-size:12px">(aktivasyon, geri geçiş, gradyan</text>
<text x="540" y="11109" text-anchor="start" style="fill:var(--c-text);font-size:12px">AllReduce). Bir kontrol noktası yazımı ya</text>
<text x="540" y="11126" text-anchor="start" style="fill:var(--c-text);font-size:12px">da konteyner indirmesi aynı kablodan</text>
<text x="540" y="11143" text-anchor="start" style="fill:var(--c-text);font-size:12px">geçseydi AllReduce&#x27;u yavaşlatırdı. Bu</text>
<text x="540" y="11160" text-anchor="start" style="fill:var(--c-text);font-size:12px">yüzden depolama ağı veri okuma ve kayıt</text>
<text x="540" y="11177" text-anchor="start" style="fill:var(--c-text);font-size:12px">yazma (1. ve 8. adımlar), in-band ağ</text>
<text x="540" y="11194" text-anchor="start" style="fill:var(--c-text);font-size:12px">işletim sistemi, kullanıcı girişi ve</text>
<text x="540" y="11211" text-anchor="start" style="fill:var(--c-text);font-size:12px">konteyner indirmeleri, OOB ağ ise BMC</text>
<text x="540" y="11228" text-anchor="start" style="fill:var(--c-text);font-size:12px">üzerinden güç ve konsol erişimi için</text>
<text x="540" y="11245" text-anchor="start" style="fill:var(--c-text);font-size:12px">ayrıdır. Dört hat 400, 100, 25 ve</text>
<text x="540" y="11262" text-anchor="start" style="fill:var(--c-text);font-size:12px">0,125 GB/s taşır; her biri kendi trafiğine</text>
<text x="540" y="11279" text-anchor="start" style="fill:var(--c-text);font-size:12px">boyutlanır.</text>
<line x1="530.0" y1="11044.0" x2="530.0" y2="11318.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="11358" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Donanım: kim, kaç tane, neden?</text>
<rect x="36" y="11372" width="248" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="11372" width="4" height="158" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="11391" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100 sistemi</text>
<rect x="237.8" y="11378.0" width="38.2" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="256.9" y="11391.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×127</text>
<text x="48" y="11408" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 H100 · 640 GB HBM3 · 10,2 kW</text>
<text x="48" y="11426" text-anchor="start" style="fill:var(--c-text);font-size:11px">Modelin bütün matematiği burada</text>
<text x="48" y="11441" text-anchor="start" style="fill:var(--c-text);font-size:11px">yapılır; kümedeki diğer her parça bu</text>
<text x="48" y="11456" text-anchor="start" style="fill:var(--c-text);font-size:11px">kutuları beslemek için vardır.</text>
<line x1="48.0" y1="11504.0" x2="272.0" y2="11504.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="11520" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">4 SU × 32 yuva − 1 (UFM) = 127</text>
<rect x="296" y="11372" width="248" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="11372" width="4" height="158" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="308" y="11391" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Hesaplama ağı · QM9700</text>
<rect x="503.9" y="11378.0" width="32.2" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="519.9" y="11391.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×48</text>
<text x="308" y="11408" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 × 400 Gb/s port · ray uyumlu</text>
<text x="308" y="11422" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">fat-tree</text>
<text x="308" y="11440" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gradyan ve aktivasyon trafiği düğümler</text>
<text x="308" y="11455" text-anchor="start" style="fill:var(--c-text);font-size:11px">arasında tıkanmadan akmalı; bu yüzden</text>
<text x="308" y="11470" text-anchor="start" style="fill:var(--c-text);font-size:11px">her GPU&#x27;nun kendi 400 Gb/s&#x27;lik rayı</text>
<text x="308" y="11485" text-anchor="start" style="fill:var(--c-text);font-size:11px">vardır (4 ve 4b).</text>
<line x1="308.0" y1="11504.0" x2="532.0" y2="11504.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="308" y="11520" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">32 leaf (4 SU × 8 ray) + 16 spine</text>
<rect x="556" y="11372" width="248" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="11372" width="4" height="158" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="568" y="11391" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Depolama ağı · QM9700</text>
<rect x="763.9" y="11378.0" width="32.2" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="779.9" y="11391.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×16</text>
<text x="568" y="11408" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">düğüm başına ≥ 40 GB/s · ~4:3</text>
<text x="568" y="11426" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri okuma ve kontrol noktası yazma</text>
<text x="568" y="11441" text-anchor="start" style="fill:var(--c-text);font-size:11px">hesaplama ağındaki AllReduce&#x27;a</text>
<text x="568" y="11456" text-anchor="start" style="fill:var(--c-text);font-size:11px">karışmaz; hızı depolama dizisinin tepe</text>
<text x="568" y="11471" text-anchor="start" style="fill:var(--c-text);font-size:11px">hızına göre seçilir.</text>
<line x1="568.0" y1="11504.0" x2="792.0" y2="11504.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="568" y="11520" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">düğüm başına 2 × 400 Gb/s</text>
<rect x="36" y="11542" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="11542" width="4" height="144" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="48" y="11561" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">UFM cihazı</text>
<rect x="249.9" y="11548.0" width="26.1" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="262.9" y="11561.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×4</text>
<text x="48" y="11578" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM 3.1 · InfiniBand telemetrisi</text>
<text x="48" y="11596" text-anchor="start" style="fill:var(--c-text);font-size:11px">Binlerce port ve kabloyu izleyip</text>
<text x="48" y="11611" text-anchor="start" style="fill:var(--c-text);font-size:11px">bozulan bağlantıyı bulur; Llama 3&#x27;te</text>
<text x="48" y="11626" text-anchor="start" style="fill:var(--c-text);font-size:11px">419 beklenmeyen kesintinin 35&#x27;i (%8,4)</text>
<text x="48" y="11641" text-anchor="start" style="fill:var(--c-text);font-size:11px">switch ve kablodandı.</text>
<line x1="48.0" y1="11660.0" x2="272.0" y2="11660.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="11676" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">bir çifti 128. DGX yuvasını alır</text>
<rect x="296" y="11542" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="11542" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="11561" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">In-band yönetim · SN4600C</text>
<rect x="509.9" y="11548.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="522.9" y="11561.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="308" y="11578" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 port × 100 GbE (Ethernet)</text>
<text x="308" y="11596" text-anchor="start" style="fill:var(--c-text);font-size:11px">Düğüm kurulumu, kullanıcı girişi, dosya</text>
<text x="308" y="11611" text-anchor="start" style="fill:var(--c-text);font-size:11px">sistemi ve dış erişim (NGC, paket</text>
<text x="308" y="11626" text-anchor="start" style="fill:var(--c-text);font-size:11px">depoları) bu ağdan akar; hesaplama ağı</text>
<text x="308" y="11641" text-anchor="start" style="fill:var(--c-text);font-size:11px">yalnız GPU trafiğine kalır.</text>
<line x1="308.0" y1="11660.0" x2="532.0" y2="11660.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="308" y="11676" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">düğüm başına 2 × 100 GbE</text>
<rect x="556" y="11542" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="11542" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="568" y="11561" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">OOB yönetim · SN2201</text>
<rect x="769.9" y="11548.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="782.9" y="11561.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="568" y="11578" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">48 port × 1 GbE bakır</text>
<text x="568" y="11596" text-anchor="start" style="fill:var(--c-text);font-size:11px">Her sunucunun BMC yönetim çipine ve</text>
<text x="568" y="11611" text-anchor="start" style="fill:var(--c-text);font-size:11px">PDU&#x27;lara ulaşır; işletim sistemi</text>
<text x="568" y="11626" text-anchor="start" style="fill:var(--c-text);font-size:11px">kilitlense bile düğüm uzaktan kapatılıp</text>
<text x="568" y="11641" text-anchor="start" style="fill:var(--c-text);font-size:11px">açılabilir.</text>
<line x1="568.0" y1="11660.0" x2="792.0" y2="11660.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="568" y="11676" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">düğüm başına 1 × 1 GbE</text>
<rect x="36" y="11698" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="11698" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="11717" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Yönetim sunucuları</text>
<rect x="249.9" y="11704.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="262.9" y="11717.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×5</text>
<text x="48" y="11734" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 soket · 24+ çekirdek · 384 GB RAM</text>
<text x="48" y="11752" text-anchor="start" style="fill:var(--c-text);font-size:11px">Küme yönetimi (Base Command Manager) ve</text>
<text x="48" y="11767" text-anchor="start" style="fill:var(--c-text);font-size:11px">iş zamanlama (Slurm) ayrı makinelerde</text>
<text x="48" y="11782" text-anchor="start" style="fill:var(--c-text);font-size:11px">koşar; GPU düğümleri yalnızca eğitime</text>
<text x="48" y="11797" text-anchor="start" style="fill:var(--c-text);font-size:11px">kalır.</text>
<line x1="48.0" y1="11816.0" x2="272.0" y2="11816.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="11832" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">4 × HDR200 · 7,68 TB NVMe</text>
<rect x="296" y="11698" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="11698" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="11717" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kabin ve güç</text>
<rect x="473.6" y="11704.0" width="62.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="504.8" y="11717.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">38 kabin</text>
<text x="308" y="11734" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">108 PDU (96 + 12) · kabin başına 4 DGX</text>
<text x="308" y="11752" text-anchor="start" style="fill:var(--c-text);font-size:11px">Kabin başına 4 DGX = 40,8 kW; bir</text>
<text x="308" y="11767" text-anchor="start" style="fill:var(--c-text);font-size:11px">SU&#x27;nun 8 DGX kabini 326,4 kW çeker. Güç</text>
<text x="308" y="11782" text-anchor="start" style="fill:var(--c-text);font-size:11px">ve soğutma kümenin görünmeyen dördüncü</text>
<text x="308" y="11797" text-anchor="start" style="fill:var(--c-text);font-size:11px">sütunudur.</text>
<line x1="308.0" y1="11816.0" x2="532.0" y2="11816.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="308" y="11832" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">toplam ≈ 1,39 MW (aşağıda)</text>
<text x="40" y="11882" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Güç: 1,39 MW nereden çıkar?</text>
<rect x="40.0" y="11900" width="446.1" height="22" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:.8"/>
<rect x="486.1" y="11900" width="22.7" height="22" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:.8"/>
<rect x="508.8" y="11900" width="7.6" height="22" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:.8"/>
<rect x="516.4" y="11900" width="1.3" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="517.7" y="11900" width="1.2" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="518.9" y="11900" width="0.8" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="519.7" y="11900" width="0.3" height="22" style="fill:var(--c-text-mute);fill-opacity:.40;stroke:var(--c-text-mute);stroke-width:.8"/>
<text x="50" y="11915" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100 · %92,9</text>
<text x="40" y="11946" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">parça</text>
<text x="300" y="11946" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">adet × güç</text>
<text x="400" y="11946" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">toplam</text>
<text x="460" y="11946" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">pay</text>
<line x1="40.0" y1="11952.0" x2="520.0" y2="11952.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40" y="11959" width="10" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.55;stroke:var(--c-accent);stroke-width:1"/>
<text x="56" y="11968" text-anchor="start" style="fill:var(--c-text);font-size:11px">DGX H100</text>
<text x="300" y="11968" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">127 × 10.200 W</text>
<text x="400" y="11968" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1.295,4 kW</text>
<text x="460" y="11968" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%92,9</text>
<rect x="40" y="11977" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="56" y="11986" text-anchor="start" style="fill:var(--c-text);font-size:11px">QM9700 · hesaplama</text>
<text x="300" y="11986" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">48 × 1.376 W</text>
<text x="400" y="11986" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">66,0 kW</text>
<text x="460" y="11986" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%4,7</text>
<rect x="40" y="11995" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.55;stroke:var(--c-warn);stroke-width:1"/>
<text x="56" y="12004" text-anchor="start" style="fill:var(--c-text);font-size:11px">QM9700 · depolama</text>
<text x="300" y="12004" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16 × 1.376 W</text>
<text x="400" y="12004" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">22,0 kW</text>
<text x="460" y="12004" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%1,6</text>
<rect x="40" y="12013" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12022" text-anchor="start" style="fill:var(--c-text);font-size:11px">SN4600C · in-band</text>
<text x="300" y="12022" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × 466 W</text>
<text x="400" y="12022" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3,7 kW</text>
<text x="460" y="12022" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%0,3</text>
<rect x="40" y="12031" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12040" text-anchor="start" style="fill:var(--c-text);font-size:11px">Yönetim sunucusu</text>
<text x="300" y="12040" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">5 × 704 W</text>
<text x="400" y="12040" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3,5 kW</text>
<text x="460" y="12040" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%0,3</text>
<rect x="40" y="12049" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12058" text-anchor="start" style="fill:var(--c-text);font-size:11px">UFM 3.1</text>
<text x="300" y="12058" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4 × 600 W</text>
<text x="400" y="12058" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">2,4 kW</text>
<text x="460" y="12058" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%0,2</text>
<rect x="40" y="12067" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.55;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="56" y="12076" text-anchor="start" style="fill:var(--c-text);font-size:11px">SN2201 · OOB</text>
<text x="300" y="12076" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × 98 W</text>
<text x="400" y="12076" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0,8 kW</text>
<text x="460" y="12076" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">%0,1</text>
<line x1="40.0" y1="12084.0" x2="520.0" y2="12084.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="56" y="12098" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Toplam (ortalama)</text>
<text x="400" y="12098" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1.393,9 kW</text>
<text x="40" y="12122" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">NVIDIA&#x27;nın planlama tablosundaki ortalama (EAP) değerler; tablo yalnız BT ekipmanını</text>
<text x="40" y="12137" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">sayar, soğutma bunun üstüne gelir.</text>
<rect x="540.0" y="11896.0" width="260.0" height="174" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="11916" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">1,39 MW nereden çıkar?</text>
<text x="552" y="11935" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">127 × 10,2 kW = 1,295 MW</text>
<text x="552" y="11951" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">+ ağ ve yönetim = 98,5 kW</text>
<text x="552" y="11967" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">≈ 1,39 MW (tepe 1,42 MW)</text>
<text x="552" y="11986" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX&#x27;ler AI yükünde sürekli tepeye yakın</text>
<text x="552" y="12001" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">çalıştığı için NVIDIA ortalama ile tepeyi</text>
<text x="552" y="12016" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">aynı (10,2 kW) alır. Kabin başına</text>
<text x="552" y="12031" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 DGX = 40,8 kW; bir SU&#x27;nun 8 DGX kabini</text>
<text x="552" y="12046" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">326,4 kW eder. Gücün her 100 W&#x27;ının 93&#x27;ü</text>
<text x="552" y="12061" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU kutularına gider.</text>
<line x1="530.0" y1="11894.0" x2="530.0" y2="12137.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="12177" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Depolama ağı ne taşır? Token değil, kontrol noktası</text>
<text x="40" y="12203" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Okuma: token akışı ve NVIDIA&#x27;nın hedefi (4 SU, log ölçek)</text>
<text x="40" y="12227" text-anchor="start" style="fill:var(--c-text);font-size:11px">token akışı (70B)</text>
<rect x="168" y="12215" width="29.2" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="203.2" y="12227" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3,8 MB/s</text>
<text x="40" y="12251" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVIDIA “Good” · NLP</text>
<rect x="168" y="12239" width="238.9" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="412.9" y="12251" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">60 GB/s</text>
<text x="40" y="12275" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVIDIA “Best”</text>
<rect x="168" y="12263" width="284.9" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.9" y="12275" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">500 GB/s</text>
<line x1="168.0" y1="12291.0" x2="468.0" y2="12291.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="168.0" y1="12291.0" x2="168.0" y2="12295.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="168" y="12307" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 MB/s</text>
<line x1="268.0" y1="12291.0" x2="268.0" y2="12295.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="268" y="12307" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100 MB/s</text>
<line x1="368.0" y1="12291.0" x2="368.0" y2="12295.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="368" y="12307" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">10 GB/s</text>
<line x1="468.0" y1="12291.0" x2="468.0" y2="12295.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="468" y="12307" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 TB/s</text>
<text x="40" y="12335" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Yazma: 1,12 TB&#x27;lık kontrol noktası eğitimi kaç saniye durdurur?</text>
<text x="40" y="12359" text-anchor="start" style="fill:var(--c-text);font-size:11px">“Good” · 30 GB/s</text>
<rect x="168" y="12347" width="250.0" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="424.0" y="12359" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">37 s</text>
<text x="40" y="12383" text-anchor="start" style="fill:var(--c-text);font-size:11px">“Better” · 80 GB/s</text>
<rect x="168" y="12371" width="93.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="267.8" y="12383" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">14 s</text>
<text x="40" y="12407" text-anchor="start" style="fill:var(--c-text);font-size:11px">“Best” · 250 GB/s</text>
<rect x="168" y="12395" width="30.0" height="16" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="204.0" y="12407" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">4,5 s</text>
<text x="40" y="12433" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Okuma ve yazma hedefleri NVIDIA&#x27;nın 4 SU için “Good / Better / Best” yönergesidir.</text>
<text x="40" y="12448" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">NLP en düşük sınıftadır, çünkü metin veri seti çoğunlukla düğümün yerel NVMe</text>
<text x="40" y="12463" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">önbelleğine sığar.</text>
<rect x="540.0" y="12191.0" width="260.0" height="144" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="12211" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">3,8 MB/s nereden çıkar?</text>
<text x="552" y="12230" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1.016 × 990 TFLOP/s × 0,40</text>
<text x="552" y="12246" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">÷ (6 × 70 milyar) ≈ 958 bin/s</text>
<text x="552" y="12262" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">× 4 bayt ≈ 3,8 MB/s</text>
<text x="552" y="12281" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Metindeki 6·N FLOP/token, E&#x27;deki ~%40 MFU</text>
<text x="552" y="12296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ve int32 token kimliği (128K sözlük</text>
<text x="552" y="12311" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">uint16&#x27;ya sığmaz). 16.384 GPU&#x27;luk 405B</text>
<text x="552" y="12326" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">eğitiminde bile akış ~11 MB/s&#x27;tir.</text>
<text x="540" y="12355" text-anchor="start" style="fill:var(--c-text);font-size:12px">Kontrol noktası ise 70 milyar × 16</text>
<text x="540" y="12372" text-anchor="start" style="fill:var(--c-text);font-size:12px">bayt ≈ 1,12 TB&#x27;tır (metnin 4. bölümündeki</text>
<text x="540" y="12389" text-anchor="start" style="fill:var(--c-text);font-size:12px">formül). NVIDIA&#x27;ya göre kayıt eşzamanlı</text>
<text x="540" y="12406" text-anchor="start" style="fill:var(--c-text);font-size:12px">yazılır ve eğitimi durdurur; depolama ağı</text>
<text x="540" y="12423" text-anchor="start" style="fill:var(--c-text);font-size:12px">bu kısa patlamaya göre boyutlanır.</text>
<text x="540" y="12440" text-anchor="start" style="fill:var(--c-text);font-size:12px">Duraklamanın bedelini C&#x27;deki G/Ç duvarı</text>
<text x="540" y="12457" text-anchor="start" style="fill:var(--c-text);font-size:12px">Young-Daly ile hesaplar.</text>
<line x1="530.0" y1="12189.0" x2="530.0" y2="12463.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="12503" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Yazılım: bir eğitim işinin yolculuğu</text>
<text x="40" y="12521" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Altı yazılım parçası bir işi gönderildiği andan ilk arızaya kadar taşır; numara sırayı, şerit rengi hizmet ettiği sütunu gösterir.</text>
<rect x="40.0" y="12535" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="12535" width="4" height="118" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<circle cx="62.0" cy="12553" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="62.0" y="12557.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="78.0" y="12557" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">İşi gönder</text>
<rect x="211.7" y="12565.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="239.8" y="12578.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="52.0" y="12577" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Base Command Platform</text>
<text x="52.0" y="12595" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri bilimci işi, veri setini ve</text>
<text x="52.0" y="12610" text-anchor="start" style="fill:var(--c-text);font-size:11px">konteyneri seçer; alttaki donanımı</text>
<text x="52.0" y="12625" text-anchor="start" style="fill:var(--c-text);font-size:11px">bilmesi gerekmez.</text>
<rect x="302.0" y="12535" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="302.0" y="12535" width="4" height="118" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="324.0" cy="12553" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="324.0" y="12557.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="340.0" y="12557" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Düğümleri ayır</text>
<rect x="461.6" y="12565.0" width="68.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="495.8" y="12578.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesaplama</text>
<text x="314.0" y="12577" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">Slurm</text>
<text x="314.0" y="12595" text-anchor="start" style="fill:var(--c-text);font-size:11px">Çok düğümlü işin hangi GPU&#x27;larda ve</text>
<text x="314.0" y="12610" text-anchor="start" style="fill:var(--c-text);font-size:11px">hangi sırayla koşacağını belirler;</text>
<text x="314.0" y="12625" text-anchor="start" style="fill:var(--c-text);font-size:11px">arızalı düğümü devreden çıkarıp işi</text>
<text x="314.0" y="12640" text-anchor="start" style="fill:var(--c-text);font-size:11px">yeniden kuyruğa alabilir.</text>
<line x1="279.0" y1="12553.0" x2="299.0" y2="12553.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="564.0" y="12535" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="564.0" y="12535" width="4" height="118" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<circle cx="586.0" cy="12553" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="586.0" y="12557.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="602.0" y="12557" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı imajı kur</text>
<rect x="735.6" y="12565.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="763.8" y="12578.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="576.0" y="12577" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Base Command Manager</text>
<text x="576.0" y="12595" text-anchor="start" style="fill:var(--c-text);font-size:11px">Bütün düğümlere aynı işletim sistemi</text>
<text x="576.0" y="12610" text-anchor="start" style="fill:var(--c-text);font-size:11px">imajını kurar, günceller ve izler;</text>
<text x="576.0" y="12625" text-anchor="start" style="fill:var(--c-text);font-size:11px">127 makineyi elle yönetmek gerçekçi</text>
<text x="576.0" y="12640" text-anchor="start" style="fill:var(--c-text);font-size:11px">değildir.</text>
<line x1="541.0" y1="12553.0" x2="561.0" y2="12553.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<path d="M682.0 12653 V12665 H158.0 V12675" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="40.0" y="12677" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="12677" width="4" height="118" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="62.0" cy="12695" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="62.0" y="12699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="78.0" y="12699" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı yazılımı başlat</text>
<rect x="199.6" y="12707.0" width="68.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="233.8" y="12720.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesaplama</text>
<text x="52.0" y="12719" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">NGC · AI Enterprise</text>
<text x="52.0" y="12737" text-anchor="start" style="fill:var(--c-text);font-size:11px">CUDA, NCCL ve PyTorch birbirine uyan</text>
<text x="52.0" y="12752" text-anchor="start" style="fill:var(--c-text);font-size:11px">konteynerler hâlinde gelir; AI</text>
<text x="52.0" y="12767" text-anchor="start" style="fill:var(--c-text);font-size:11px">Enterprise destekli araç ve</text>
<text x="52.0" y="12782" text-anchor="start" style="fill:var(--c-text);font-size:11px">orkestrasyon ekler.</text>
<rect x="302.0" y="12677" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="302.0" y="12677" width="4" height="118" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="324.0" cy="12695" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="324.0" y="12699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="340.0" y="12699" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Veriyi taşı</text>
<rect x="437.4" y="12707.0" width="92.7" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="483.7" y="12720.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ + depolama</text>
<text x="314.0" y="12719" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">Magnum IO</text>
<text x="314.0" y="12737" text-anchor="start" style="fill:var(--c-text);font-size:11px">NCCL hesaplama ağında (SHARP ile),</text>
<text x="314.0" y="12752" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPUDirect Storage depolama ağında</text>
<text x="314.0" y="12767" text-anchor="start" style="fill:var(--c-text);font-size:11px">çalışır: B&#x27;deki 1., 3., 4., 6. ve 8.</text>
<text x="314.0" y="12782" text-anchor="start" style="fill:var(--c-text);font-size:11px">adımlar.</text>
<line x1="279.0" y1="12695.0" x2="299.0" y2="12695.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="564.0" y="12677" width="236.0" height="118" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="564.0" y="12677" width="4" height="118" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="586.0" cy="12695" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="586.0" y="12699.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="602.0" y="12699" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">İzle ve onar</text>
<rect x="765.9" y="12707.0" width="26.1" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="778.9" y="12720.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ</text>
<text x="576.0" y="12719" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">UFM</text>
<text x="576.0" y="12737" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ağdaki bozuk port ve kabloyu bulur;</text>
<text x="576.0" y="12752" text-anchor="start" style="fill:var(--c-text);font-size:11px">Slurm düğümü çıkarır, iş son kontrol</text>
<text x="576.0" y="12767" text-anchor="start" style="fill:var(--c-text);font-size:11px">noktasından döner.</text>
<line x1="541.0" y1="12695.0" x2="561.0" y2="12695.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="40.0" y="12809.0" width="760.0" height="50" rx="5" style="fill:var(--c-success);fill-opacity:.08;stroke:var(--c-success);stroke-width:1.2"/>
<text x="52" y="12829" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-success);font-weight:700">Neden önemli:</tspan> Llama 3&#x27;ün 54 günlük kesitinde 419 beklenmeyen kesinti oldu (~3 saatte bir); yalnız 3&#x27;ü ciddi elle</text>
<text x="52" y="12846" text-anchor="start" style="fill:var(--c-text);font-size:12px">müdahale istedi, gerisini otomasyon çözdü ve etkin eğitim süresi %90&#x27;ın üstünde kaldı (E).</text>
<text x="40" y="12899" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Zihinsel model: restoranın arka tarafı</text>
<text x="40" y="12917" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Restoran zinciri benzetmesinin devamı: müşterinin görmediği ama her şeyin ona bağlı olduğu kısımlar.</text>
<rect x="40.0" y="12929" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="12929" width="4" height="186" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="54.0" y="12949" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Tedarik hattı</text>
<rect x="52.0" y="12957.0" width="157.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="130.5" y="12971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">depolama ağı</text>
<text x="54.0" y="12995" text-anchor="start" style="fill:var(--c-text);font-size:11px">Malzeme (token) az ama</text>
<text x="54.0" y="13010" text-anchor="start" style="fill:var(--c-text);font-size:11px">düzenli gelir: saniyede</text>
<text x="54.0" y="13025" text-anchor="start" style="fill:var(--c-text);font-size:11px">birkaç MB. Asıl yük, gün</text>
<text x="54.0" y="13040" text-anchor="start" style="fill:var(--c-text);font-size:11px">sonunda kasanın (kontrol</text>
<text x="54.0" y="13055" text-anchor="start" style="fill:var(--c-text);font-size:11px">noktası) toplu</text>
<text x="54.0" y="13070" text-anchor="start" style="fill:var(--c-text);font-size:11px">taşınmasıdır.</text>
<text x="54.0" y="13103" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:700">1,12 TB · 4,5–37 s</text>
<rect x="233.0" y="12929" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="233.0" y="12929" width="4" height="186" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="247.0" y="12949" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Servis koridoru</text>
<rect x="245.0" y="12957.0" width="157.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="323.5" y="12971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">in-band yönetim</text>
<text x="247.0" y="12995" text-anchor="start" style="fill:var(--c-text);font-size:11px">Personel girişi, tarif</text>
<text x="247.0" y="13010" text-anchor="start" style="fill:var(--c-text);font-size:11px">kitapları (konteynerler) ve</text>
<text x="247.0" y="13025" text-anchor="start" style="fill:var(--c-text);font-size:11px">vardiya listesi bu</text>
<text x="247.0" y="13040" text-anchor="start" style="fill:var(--c-text);font-size:11px">koridordan geçer; müşteri</text>
<text x="247.0" y="13055" text-anchor="start" style="fill:var(--c-text);font-size:11px">salonunu (hesaplama ağı)</text>
<text x="247.0" y="13070" text-anchor="start" style="fill:var(--c-text);font-size:11px">meşgul etmez.</text>
<text x="247.0" y="13103" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">2 × 100 GbE</text>
<rect x="426.0" y="12929" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="426.0" y="12929" width="4" height="186" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="440.0" y="12949" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Elektrik panosu</text>
<rect x="438.0" y="12957.0" width="157.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="516.5" y="12971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">OOB · BMC</text>
<text x="440.0" y="12995" text-anchor="start" style="fill:var(--c-text);font-size:11px">Mutfak kilitlense bile</text>
<text x="440.0" y="13010" text-anchor="start" style="fill:var(--c-text);font-size:11px">panoya ayrı yoldan</text>
<text x="440.0" y="13025" text-anchor="start" style="fill:var(--c-text);font-size:11px">ulaşılır; restoran uzaktan</text>
<text x="440.0" y="13040" text-anchor="start" style="fill:var(--c-text);font-size:11px">kapatılıp açılabilir.</text>
<text x="440.0" y="13103" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">1 GbE · 8 × SN2201</text>
<rect x="619.0" y="12929" width="181.0" height="186" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="619.0" y="12929" width="4" height="186" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="633.0" y="12949" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Vardiya amiri</text>
<rect x="631.0" y="12957.0" width="157.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="709.5" y="12971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Slurm · BCM · UFM</text>
<text x="633.0" y="12995" text-anchor="start" style="fill:var(--c-text);font-size:11px">Kimin hangi mutfakta</text>
<text x="633.0" y="13010" text-anchor="start" style="fill:var(--c-text);font-size:11px">çalışacağını belirler,</text>
<text x="633.0" y="13025" text-anchor="start" style="fill:var(--c-text);font-size:11px">arızalı mutfağı kapatır ve</text>
<text x="633.0" y="13040" text-anchor="start" style="fill:var(--c-text);font-size:11px">işi son kayıttan yeniden</text>
<text x="633.0" y="13055" text-anchor="start" style="fill:var(--c-text);font-size:11px">başlatır.</text>
<text x="633.0" y="13103" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">419 kesinti, 3 müdahale</text>
<rect x="36.0" y="13127.0" width="768.0" height="44" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="13144" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Altın kural: gücün %93&#x27;ü DGX&#x27;lere gider; kalan her parça o gücün boşa gitmemesi için vardır. GPU&#x27;yu bekleten her şey (tıkanan ağ,</text>
<text x="46" y="13159" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">yavaş kayıt, elle onarım) bu parçalarla önlenir.</text>
<rect x="28.0" y="13193" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="13213" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: GPU&#x27;dan uzaklaştıkça yol daralır: HBM 3,35 TB/s → NVLink 450 GB/s → ağ 50 GB/s.</text>
<text x="420.0" y="13258" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Parçaları tanıdık. Şimdi bir eğitim adımında birlikte nasıl çalıştıklarını izleyelim.</text>
<line x1="420.0" y1="13266" x2="420.0" y2="13286" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="13290" width="808" height="4265" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="13314" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="13318.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">B</text>
<text x="56" y="13319" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Bir eğitim adımını izle: veri nereden geçer?</text>
<text x="28" y="13340" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">64 SU&#x27;luk H100 SuperPOD&#x27;da Llama 3 405B: TP 8 × PP 16 × DP 128 = 16.384 GPU</text>
<text x="28" y="13358" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Metindeki “LLM ölçekleme için 4D paralellik roofline sınırları” ve “Uçtan uca cluster analizi” alt bölümlerini izler; dağılım Llama 3</text>
<text x="28" y="13373" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">raporunun Table 4&#x27;ünden, sayılar metindeki formüllerden.</text>
<rect x="28" y="13387" width="784" height="2838" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="13413" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4D paralellik: modeli binlerce GPU&#x27;ya nasıl bölüyoruz?</text>
<text x="40" y="13433" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Önce sorun, sonra çözüm. Her paralellik tek bir sorunu çözer ve sıra içten dışa gider: düğümün içinde TP, düğümler arasında</text>
<text x="40" y="13450" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">PP, en dışta DP. MoE ve çok uzun bağlam için iki ek boyut daha vardır: EP ve CP.</text>
<text x="40" y="13484" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Sorun: tek bir GPU hem çok yavaş hem çok küçük</text>
<rect x="36" y="13496" width="374" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="13522" text-anchor="start" style="fill:var(--c-danger);font-size:18px;font-weight:700">~3.000 yıl</text>
<text x="48" y="13542" text-anchor="start" style="fill:var(--c-text);font-size:12px">Llama 3 405B&#x27;nin eğitimi (3,8 × 10²⁵ FLOP) tek bir</text>
<text x="48" y="13559" text-anchor="start" style="fill:var(--c-text);font-size:12px">H100&#x27;de, 400 TFLOP/s hızla bu kadar sürerdi.</text>
<rect x="422" y="13496" width="374" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="434" y="13522" text-anchor="start" style="fill:var(--c-danger);font-size:18px;font-weight:700">6,5 TB &gt; 80 GB</text>
<text x="434" y="13542" text-anchor="start" style="fill:var(--c-text);font-size:12px">Modelin eğitim durumu (parametre başına 16 bayt) tek</text>
<text x="434" y="13559" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;nun belleğinin 81 katı.</text>
<text x="40" y="13590" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Çözüm işi GPU&#x27;lara bölmek; ama neyi böldüğün, GPU&#x27;ların ne sıklıkla konuşacağını belirler.</text>
<text x="40" y="13630" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Çözüm adım adım: Llama 3 405B&#x27;yi 16.384 GPU&#x27;ya bölelim</text>
<text x="40" y="13648" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Her adım bir sorunu çözer ve yeni bir sorun bırakır. Sağdaki tablo o adımın neyi değiştirdiğini gösterir: GPU sayısı, GPU başına eğitim</text>
<text x="40" y="13663" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">durumu (80 GB&#x27;a sığmalı) ve eğitim süresi.</text>
<rect x="36" y="13664" width="768" height="180" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="13664" width="4" height="180" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="58" cy="13686" r="11" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="58" y="13690.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">1</text>
<text x="78" y="13691" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Sorun:</tspan> model tek GPU&#x27;ya sığmıyor. Eğitim</text>
<text x="78" y="13708" text-anchor="start" style="fill:var(--c-text);font-size:12px">durumu 6.480 GB, GPU&#x27;nun belleği 80 GB.</text>
<rect x="78.0" y="13720.0" width="167.2" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="161.6" y="13734.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TP 8 · tensör paralelliği</text>
<text x="78" y="13760" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Çözüm:</tspan> her katmanın matrislerini</text>
<text x="78" y="13777" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümdeki 8 GPU&#x27;ya böl. Her GPU her</text>
<text x="78" y="13794" text-anchor="start" style="fill:var(--c-text);font-size:12px">matrisin sekizde birini taşır ve</text>
<text x="78" y="13811" text-anchor="start" style="fill:var(--c-text);font-size:12px">hesaplar; kısmi sonuçlar her katmanda</text>
<text x="78" y="13828" text-anchor="start" style="fill:var(--c-text);font-size:12px">toplanır.</text>
<text x="346" y="13688" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">W · bir katmanın matrisi, 8 dilim</text>
<rect x="348.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="362.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<line x1="362.0" y1="13715.0" x2="362.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="379.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="393.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<line x1="393.0" y1="13715.0" x2="393.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="410.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.30;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="424.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<line x1="424.0" y1="13715.0" x2="424.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="441.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.30;stroke:var(--c-success);stroke-width:1.2"/>
<text x="455.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<line x1="455.0" y1="13715.0" x2="455.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="472.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.30;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="486.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<line x1="486.0" y1="13715.0" x2="486.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="503.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="517.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<line x1="517.0" y1="13715.0" x2="517.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="534.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.30;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="548.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<line x1="548.0" y1="13715.0" x2="548.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="565.0" y="13694.0" width="29.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.30;stroke:var(--c-success);stroke-width:1.2"/>
<text x="579.5" y="13708.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<line x1="579.0" y1="13715.0" x2="579.0" y2="13727.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="346" y="13728" width="250" height="30" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="350.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="363.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G0</text>
<rect x="381.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="394.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G1</text>
<rect x="412.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="425.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G2</text>
<rect x="443.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="456.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G3</text>
<rect x="474.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="487.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G4</text>
<rect x="505.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="518.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G5</text>
<rect x="536.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="549.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G6</text>
<rect x="567.0" y="13734.0" width="27.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="580.5" y="13747.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G7</text>
<text x="346" y="13776" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">dilim i, düğümdeki GPU i&#x27;ye gider</text>
<text x="612" y="13686" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">önce → sonra</text>
<text x="612" y="13704" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<text x="796" y="13704" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1 → 8</text>
<text x="612" y="13721" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına</text>
<text x="796" y="13721" text-anchor="end" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">6.480 → 810 GB</text>
<text x="612" y="13738" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">süre</text>
<text x="796" y="13738" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">3.000 → 380 yıl</text>
<text x="612" y="13763" text-anchor="start" style="fill:var(--c-text);font-size:11px">Neden önce TP: her katmanda</text>
<text x="612" y="13778" text-anchor="start" style="fill:var(--c-text);font-size:11px">konuştuğu için en hızlı yola,</text>
<text x="612" y="13793" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğüm içindeki NVLink&#x27;e ihtiyaç</text>
<text x="612" y="13808" text-anchor="start" style="fill:var(--c-text);font-size:11px">duyar. Düğümde 8 GPU olduğu için</text>
<text x="612" y="13823" text-anchor="start" style="fill:var(--c-text);font-size:11px">8&#x27;de durur.</text>
<line x1="600.0" y1="13676.0" x2="600.0" y2="13832.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="13845.0" x2="420.0" y2="13857.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="36" y="13858" width="768" height="190" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="13858" width="4" height="190" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="58" cy="13880" r="11" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="58" y="13884.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">2</text>
<text x="78" y="13885" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Sorun:</tspan> hâlâ sığmıyor (810 GB) ve TP 8&#x27;in</text>
<text x="78" y="13902" text-anchor="start" style="fill:var(--c-text);font-size:12px">ötesine geçemez.</text>
<rect x="78.0" y="13914.0" width="197.5" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="176.8" y="13928.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PP 16 · boru hattı paralelliği</text>
<text x="78" y="13954" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Çözüm:</tspan> modelin 126 katmanını 16 düğüme</text>
<text x="78" y="13971" text-anchor="start" style="fill:var(--c-text);font-size:12px">sırayla dağıt. Her düğüm ~8 katmanı işler</text>
<text x="78" y="13988" text-anchor="start" style="fill:var(--c-text);font-size:12px">ve çıktısını küçük bir tepsiyle (34 MB)</text>
<text x="78" y="14005" text-anchor="start" style="fill:var(--c-text);font-size:12px">sonraki düğüme verir.</text>
<text x="346" y="13882" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">aşama 0</text>
<text x="594" y="13882" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">aşama 15</text>
<rect x="346.0" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="361.6" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="377.2" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="392.8" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="408.4" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="424.0" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="439.6" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="455.2" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="470.8" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="486.4" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="502.0" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="517.6" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="533.2" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="548.8" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="564.4" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="580.0" y="13888.0" width="14.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.30;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="346.0" y1="13924.0" x2="594.0" y2="13924.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="346" y="13944" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 düğüm sırayla, her biri ~8 katman</text>
<text x="612" y="13880" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">önce → sonra</text>
<text x="612" y="13898" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<text x="796" y="13898" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8 → 128</text>
<text x="612" y="13915" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına</text>
<text x="796" y="13915" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">810 → 50,6 GB</text>
<text x="612" y="13932" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">süre</text>
<text x="796" y="13932" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">380 → 24 yıl</text>
<text x="612" y="13957" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 düğüm × 8 GPU = 128 GPU&#x27;luk</text>
<text x="612" y="13972" text-anchor="start" style="fill:var(--c-text);font-size:11px">bir zincir. Nerede: düğümler</text>
<text x="612" y="13987" text-anchor="start" style="fill:var(--c-text);font-size:11px">arası, ray; tepsi küçük olduğu</text>
<text x="612" y="14002" text-anchor="start" style="fill:var(--c-text);font-size:11px">için yavaş yol yeter. Bedeli hat</text>
<text x="612" y="14017" text-anchor="start" style="fill:var(--c-text);font-size:11px">dolarken bekleyen aşamalar</text>
<text x="612" y="14032" text-anchor="start" style="fill:var(--c-text);font-size:11px">(kabarcık).</text>
<line x1="600.0" y1="13870.0" x2="600.0" y2="14036.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="14049.0" x2="420.0" y2="14061.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="36" y="14062" width="768" height="180" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="14062" width="4" height="180" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="58" cy="14084" r="11" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="58" y="14088.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">3</text>
<text x="78" y="14089" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Sorun:</tspan> model sığdı, ama tek zincirle</text>
<text x="78" y="14106" text-anchor="start" style="fill:var(--c-text);font-size:12px">eğitim ~24 yıl sürer.</text>
<rect x="78.0" y="14118.0" width="167.2" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="161.6" y="14132.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DP 128 · veri paralelliği</text>
<text x="78" y="14158" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Çözüm:</tspan> bu 128 GPU&#x27;luk zinciri 128 kez</text>
<text x="78" y="14175" text-anchor="start" style="fill:var(--c-text);font-size:12px">kopyala. Her kopya başka verilerle</text>
<text x="78" y="14192" text-anchor="start" style="fill:var(--c-text);font-size:12px">çalışır; adım sonunda kopyalar</text>
<text x="78" y="14209" text-anchor="start" style="fill:var(--c-text);font-size:12px">öğrendiklerini (gradyanları) ortalar ve</text>
<text x="78" y="14226" text-anchor="start" style="fill:var(--c-text);font-size:12px">aynı güncellemeyi yapar.</text>
<rect x="346.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent);fill-opacity:0.8"/>
<rect x="353.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14080" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="346.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="353.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14089" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="346.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="353.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14098" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="346.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="353.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="361.6" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="369.4" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="377.2" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="385.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="392.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="400.6" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="408.4" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="416.2" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="424.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="431.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="439.6" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="447.4" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="455.2" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="463.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="470.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="478.6" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="486.4" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="494.2" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="502.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="509.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="517.6" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="525.4" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="533.2" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="541.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="548.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="556.6" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="564.4" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="572.2" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="580.0" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<rect x="587.8" y="14107" width="6.6" height="7" rx="1" style="fill:var(--c-accent-2);fill-opacity:0.25"/>
<text x="346" y="14132" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her kare 128 GPU&#x27;luk bir zincir: 128 kopya</text>
<text x="612" y="14084" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">önce → sonra</text>
<text x="612" y="14102" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<text x="796" y="14102" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">128 → 16.384</text>
<text x="612" y="14119" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına</text>
<text x="796" y="14119" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">50,6 → 50,6 GB</text>
<text x="612" y="14136" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">süre</text>
<text x="796" y="14136" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">24 yıl → 67 gün</text>
<text x="612" y="14161" text-anchor="start" style="fill:var(--c-text);font-size:11px">Nerede: tüm küme, ray + spine.</text>
<text x="612" y="14176" text-anchor="start" style="fill:var(--c-text);font-size:11px">Adımda bir konuştukları için en</text>
<text x="612" y="14191" text-anchor="start" style="fill:var(--c-text);font-size:11px">dıştadırlar. Bellek değişmez:</text>
<text x="612" y="14206" text-anchor="start" style="fill:var(--c-text);font-size:11px">her kopya aynı durumu taşır.</text>
<line x1="600.0" y1="14074.0" x2="600.0" y2="14230.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="14243.0" x2="420.0" y2="14255.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="36" y="14256" width="768" height="197" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="14256" width="4" height="197" rx="2" style="fill:var(--c-success);fill-opacity:.85"/>
<circle cx="58" cy="14278" r="11" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.3"/>
<text x="58" y="14282.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">4</text>
<text x="78" y="14283" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-danger);font-weight:700">Sorun:</tspan> 128 kopyanın her biri aynı</text>
<text x="78" y="14300" text-anchor="start" style="fill:var(--c-text);font-size:12px">optimizer durumunu ayrı ayrı tutuyor;</text>
<text x="78" y="14317" text-anchor="start" style="fill:var(--c-text);font-size:12px">aktivasyonlara yalnız ~29 GB kalıyor.</text>
<rect x="78.0" y="14329.0" width="155.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="155.6" y="14343.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">FSDP · durumu paylaştır</text>
<text x="78" y="14369" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-success);font-weight:700">Çözüm:</tspan> kopyalar optimizer durumunu</text>
<text x="78" y="14386" text-anchor="start" style="fill:var(--c-text);font-size:12px">(parametre başına 12 bayt) aralarında</text>
<text x="78" y="14403" text-anchor="start" style="fill:var(--c-text);font-size:12px">128&#x27;e böler. Her GPU yalnız kendi payını</text>
<text x="78" y="14420" text-anchor="start" style="fill:var(--c-text);font-size:12px">günceller; güncel ağırlıklar sonra</text>
<text x="78" y="14437" text-anchor="start" style="fill:var(--c-text);font-size:12px">herkese dağıtılır (all-gather).</text>
<rect x="346" y="14270" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="375.0" y="14283" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">kopya 0</text>
<rect x="350.0" y="14289.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="14301.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="350.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<rect x="362.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="375.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="387.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="410" y="14270" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="439.0" y="14283" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">kopya 1</text>
<rect x="414.0" y="14289.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="439.0" y="14301.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="414.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="426.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<rect x="439.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="451.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="474" y="14270" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="503.0" y="14283" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">kopya 2</text>
<rect x="478.0" y="14289.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="503.0" y="14301.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="478.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="490.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="503.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<rect x="515.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="538" y="14270" width="58" height="58" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="567.0" y="14283" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">kopya 3</text>
<rect x="542.0" y="14289.0" width="50.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="567.0" y="14301.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">model</text>
<rect x="542.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="554.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="567.0" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-text-mute);fill-opacity:0.1;stroke:var(--c-text-mute);stroke-width:.8"/>
<rect x="579.5" y="14310" width="11" height="12" rx="1.5" style="fill:var(--c-success);fill-opacity:0.75;stroke:var(--c-success);stroke-width:.8"/>
<text x="346" y="14346" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her kopya modeli (ağırlık + gradyan) tam</text>
<text x="346" y="14360" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tutar; yeşil kare optimizer&#x27;dan tuttuğu pay</text>
<text x="346" y="14374" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(burada ¼, Llama 3&#x27;te 1/128).</text>
<text x="612" y="14278" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">önce → sonra</text>
<text x="612" y="14296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<text x="796" y="14296" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16.384 → 16.384</text>
<text x="612" y="14313" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına</text>
<text x="796" y="14313" text-anchor="end" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">50,6 → 12,9 GB</text>
<text x="612" y="14330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">süre</text>
<text x="796" y="14330" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">67 → 67 gün</text>
<text x="612" y="14355" text-anchor="start" style="fill:var(--c-text);font-size:11px">Llama 3 tam olarak bu düzeni</text>
<text x="612" y="14370" text-anchor="start" style="fill:var(--c-text);font-size:11px">kullandı: TP 8 × PP 16 × DP 128,</text>
<text x="612" y="14385" text-anchor="start" style="fill:var(--c-text);font-size:11px">FSDP ile. Gradyanları FP32</text>
<text x="612" y="14400" text-anchor="start" style="fill:var(--c-text);font-size:11px">biriktirdiği için gerçek pay</text>
<text x="612" y="14415" text-anchor="start" style="fill:var(--c-text);font-size:11px">~19 GB&#x27;tır.</text>
<line x1="600.0" y1="14268.0" x2="600.0" y2="14441.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="14493" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Özet: her adım neyi düzeltti?</text>
<text x="40" y="14517" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">adım</text>
<text x="186" y="14517" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPU</text>
<text x="214" y="14517" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">GPU başına eğitim durumu (log ölçek)</text>
<text x="800" y="14517" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">eğitim süresi</text>
<line x1="40.0" y1="14524.0" x2="800.0" y2="14524.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="40" y="14548" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">başlangıç</text>
<text x="186" y="14548" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">1</text>
<rect x="214" y="14535" width="243.9" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="463.9" y="14548" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">6.480 GB</text>
<text x="800" y="14548" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~3.000 yıl</text>
<text x="40" y="14574" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">+ TP 8</text>
<text x="186" y="14574" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">8</text>
<rect x="214" y="14561" width="186.1" height="18" rx="2" style="fill:var(--c-danger);fill-opacity:.40;stroke:var(--c-danger);stroke-width:1"/>
<text x="406.1" y="14574" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">810 GB</text>
<text x="800" y="14574" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~380 yıl</text>
<text x="40" y="14600" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">+ PP 16</text>
<text x="186" y="14600" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">128</text>
<rect x="214" y="14587" width="109.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="343.8" y="14600" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">50,6 GB</text>
<text x="800" y="14600" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~24 yıl</text>
<text x="40" y="14626" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">+ DP 128</text>
<text x="186" y="14626" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16.384</text>
<rect x="214" y="14613" width="109.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="343.8" y="14626" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">50,6 GB</text>
<text x="800" y="14626" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~67 gün</text>
<text x="40" y="14652" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">+ FSDP</text>
<text x="186" y="14652" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">16.384</text>
<rect x="214" y="14639" width="71.1" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.40;stroke:var(--c-success);stroke-width:1"/>
<text x="343.8" y="14652" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">12,9 GB</text>
<text x="800" y="14652" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">~67 gün</text>
<line x1="335.8" y1="14529.0" x2="335.8" y2="14661.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="339.8" y="14675" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">H100 · 80 GB</text>
<text x="40" y="14697" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">TP ve PP belleği küçültür, DP süreyi kısaltır, FSDP kopyaların tekrar eden payını siler. Süre 400 TFLOP/s&#x27;te kesintisiz hesaptır;</text>
<text x="40" y="14712" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">bellek parametre başına 16 bayttır (metin, 4. bölüm), aktivasyonlar hariç.</text>
<text x="40" y="14752" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Sayılar neye denk gelir? Her GPU&#x27;nun bir adresi var</text>
<rect x="40.0" y="14764" width="244.0" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="14764" width="4" height="72" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="54.0" y="14790" text-anchor="start" style="fill:var(--c-accent);font-size:18px;font-weight:700">TP 8</text>
<text x="54.0" y="14810" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir düğümdeki GPU sayısı</text>
<text x="54.0" y="14827" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sıra 0–7 = düğümdeki GPU numarası</text>
<rect x="298.0" y="14764" width="244.0" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="298.0" y="14764" width="4" height="72" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="312.0" y="14790" text-anchor="start" style="fill:var(--c-accent-2);font-size:18px;font-weight:700">PP 16</text>
<text x="312.0" y="14810" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir zincirdeki düğüm sayısı</text>
<text x="312.0" y="14827" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sıra 0–15 = zincirdeki aşama</text>
<rect x="556.0" y="14764" width="244.0" height="72" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556.0" y="14764" width="4" height="72" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="570.0" y="14790" text-anchor="start" style="fill:var(--c-accent-2);font-size:18px;font-weight:700">DP 128</text>
<text x="570.0" y="14810" text-anchor="start" style="fill:var(--c-text);font-size:12px">zincirin kopya sayısı</text>
<text x="570.0" y="14827" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sıra 0–127 = kopya numarası</text>
<rect x="40.0" y="14848.0" width="760.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="420.0" y="14864.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700;font-family:var(--font-mono)">8 GPU × 16 düğüm × 128 kopya = 16.384 GPU  ·  16 × 128 = 2.048 DGX</text>
<text x="40" y="14906" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Örnek adres: [TP sırası, PP sırası, DP sırası]</text>
<text x="40" y="14944" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">[</text>
<rect x="56.0" y="14920.0" width="30.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="71.0" y="14941.8" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700;font-family:var(--font-mono)">3</text>
<text x="89" y="14944" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">,</text>
<rect x="102.0" y="14920.0" width="30.0" height="32" rx="5" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="117.0" y="14941.8" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700;font-family:var(--font-mono)">5</text>
<text x="135" y="14944" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">,</text>
<rect x="148.0" y="14920.0" width="42.0" height="32" rx="5" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="169.0" y="14941.8" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-weight:700;font-family:var(--font-mono)">42</text>
<text x="193" y="14944" text-anchor="start" style="fill:var(--c-text);font-size:22px;font-weight:700;font-family:var(--font-mono)">]</text>
<rect x="40" y="14966" width="10" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="56" y="14976" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP sırası 3: düğümün GPU 3&#x27;ü</text>
<rect x="40" y="14984" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="56" y="14994" text-anchor="start" style="fill:var(--c-text);font-size:12px">PP sırası 5: zincirin 5. aşaması</text>
<rect x="40" y="15002" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="56" y="15012" text-anchor="start" style="fill:var(--c-text);font-size:12px">DP sırası 42: 42. kopya</text>
<rect x="300.0" y="14908.0" width="500.0" height="50" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="310" y="14925" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">kopya 42 · 16 düğümlük zincir (aşama 0–15)</text>
<rect x="310.0" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="323.0" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<rect x="340.3" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="353.3" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="370.6" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="383.6" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="400.9" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="413.9" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="431.2" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="444.2" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="461.5" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="474.5" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">5</text>
<rect x="491.8" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="504.8" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="522.1" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="535.1" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<rect x="552.4" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="565.4" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<rect x="582.7" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="595.7" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<rect x="613.0" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="626.0" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">10</text>
<rect x="643.3" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="656.3" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">11</text>
<rect x="673.6" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="686.6" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">12</text>
<rect x="703.9" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="716.9" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">13</text>
<rect x="734.2" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="747.2" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">14</text>
<rect x="764.5" y="14933.0" width="26.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="777.5" y="14946.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">15</text>
<line x1="461.5" y1="14951.0" x2="420.0" y2="14988.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<line x1="487.5" y1="14951.0" x2="800.0" y2="14988.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="420" y="14988" width="380" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="430" y="15005" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">aşama 5&#x27;in düğümü · 8 GPU</text>
<rect x="430.0" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="450.0" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="475.5" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="495.5" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="521.0" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="541.0" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="566.5" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.50;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="586.5" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 3</text>
<rect x="612.0" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="632.0" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="657.5" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="677.5" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="703.0" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="723.0" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="748.5" y="15013.0" width="40.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="768.5" y="15026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<rect x="40.0" y="15056.0" width="760.0" height="65" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="52" y="15076" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Aynı TP sırası, aynı ray:</tspan> [3, 5, 42]&#x27;nin PP komşusu [3, 6, 42], DP eşi [3, 5, 43]. Üçü de GPU 3&#x27;tür ve 3. raydadır; PP</text>
<text x="52" y="15093" text-anchor="start" style="fill:var(--c-text);font-size:12px">ve DP trafiği bu yüzden raylar üzerinden akar (4). Sıralar 0&#x27;dan sayılır; Llama 3 aynı adresi [TP, CP, PP, DP]</text>
<text x="52" y="15110" text-anchor="start" style="fill:var(--c-text);font-size:12px">vektörüyle yazar.</text>
<text x="40" y="15165" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Kim ne yapar? Beş boyut tek tabloda</text>
<text x="40" y="15185" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kural: ne kadar sık konuşuyorsa o kadar içeride, en hızlı yolda durur.</text>
<text x="88" y="15207" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Ne böler?</text>
<text x="214" y="15207" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Bir GPU ne yapar?</text>
<text x="392" y="15207" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Nerede?</text>
<text x="520" y="15207" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Ne sıklıkla konuşur?</text>
<text x="648" y="15207" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:700">Neden gerekir?</text>
<line x1="40.0" y1="15214.0" x2="800.0" y2="15214.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="15225.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="59.0" y="15238.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">TP</text>
<text x="88" y="15238" text-anchor="start" style="fill:var(--c-text);font-size:11px">bir katmanın</text>
<text x="88" y="15253" text-anchor="start" style="fill:var(--c-text);font-size:11px">matrislerini</text>
<text x="214" y="15238" text-anchor="start" style="fill:var(--c-text);font-size:11px">her katmanın 1/8&#x27;ini</text>
<text x="214" y="15253" text-anchor="start" style="fill:var(--c-text);font-size:11px">hesaplar</text>
<text x="392" y="15238" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğüm içi · NVLink</text>
<text x="520" y="15238" text-anchor="start" style="fill:var(--c-text);font-size:11px">her katmanda</text>
<text x="648" y="15238" text-anchor="start" style="fill:var(--c-text);font-size:11px">tek katman bile sığmaz</text>
<rect x="40.0" y="15267.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="59.0" y="15280.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PP</text>
<text x="88" y="15280" text-anchor="start" style="fill:var(--c-text);font-size:11px">katmanları</text>
<text x="214" y="15280" text-anchor="start" style="fill:var(--c-text);font-size:11px">~8 katmanı işleyip sonrakine</text>
<text x="214" y="15295" text-anchor="start" style="fill:var(--c-text);font-size:11px">verir</text>
<text x="392" y="15280" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğümler arası ·</text>
<text x="392" y="15295" text-anchor="start" style="fill:var(--c-text);font-size:11px">ray</text>
<text x="520" y="15280" text-anchor="start" style="fill:var(--c-text);font-size:11px">her mikro-batch&#x27;te</text>
<text x="648" y="15280" text-anchor="start" style="fill:var(--c-text);font-size:11px">model tek düğüme sığmaz</text>
<rect x="40.0" y="15309.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="59.0" y="15322.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DP</text>
<text x="88" y="15322" text-anchor="start" style="fill:var(--c-text);font-size:11px">veriyi</text>
<text x="214" y="15322" text-anchor="start" style="fill:var(--c-text);font-size:11px">tam zinciri başka veriyle</text>
<text x="214" y="15337" text-anchor="start" style="fill:var(--c-text);font-size:11px">çalıştırır</text>
<text x="392" y="15322" text-anchor="start" style="fill:var(--c-text);font-size:11px">tüm küme · ray +</text>
<text x="392" y="15337" text-anchor="start" style="fill:var(--c-text);font-size:11px">spine</text>
<text x="520" y="15322" text-anchor="start" style="fill:var(--c-text);font-size:11px">adımda bir</text>
<text x="648" y="15322" text-anchor="start" style="fill:var(--c-text);font-size:11px">tek zincir çok yavaş</text>
<line x1="40.0" y1="15347.0" x2="800.0" y2="15347.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="800" y="15360" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">gerekirse</text>
<rect x="40.0" y="15371.0" width="38.0" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="59.0" y="15384.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">EP</text>
<text x="88" y="15384" text-anchor="start" style="fill:var(--c-text);font-size:11px">MoE uzmanlarını</text>
<text x="214" y="15384" text-anchor="start" style="fill:var(--c-text);font-size:11px">birkaç uzmanı taşır, gelen</text>
<text x="214" y="15399" text-anchor="start" style="fill:var(--c-text);font-size:11px">tokenları işler</text>
<text x="392" y="15384" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğümler arası</text>
<text x="520" y="15384" text-anchor="start" style="fill:var(--c-text);font-size:11px">her MoE katmanında</text>
<text x="648" y="15384" text-anchor="start" style="fill:var(--c-text);font-size:11px">uzmanlar sığmaz (yalnız</text>
<text x="648" y="15399" text-anchor="start" style="fill:var(--c-text);font-size:11px">MoE)</text>
<rect x="40.0" y="15413.0" width="38.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="59.0" y="15426.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CP</text>
<text x="88" y="15426" text-anchor="start" style="fill:var(--c-text);font-size:11px">uzun diziyi</text>
<text x="214" y="15426" text-anchor="start" style="fill:var(--c-text);font-size:11px">dizinin bir parçasını işler</text>
<text x="392" y="15426" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğüm içi + ağ</text>
<text x="520" y="15426" text-anchor="start" style="fill:var(--c-text);font-size:11px">her dikkat</text>
<text x="520" y="15441" text-anchor="start" style="fill:var(--c-text);font-size:11px">katmanında</text>
<text x="648" y="15426" text-anchor="start" style="fill:var(--c-text);font-size:11px">uzun dizinin aktivasyonu</text>
<text x="648" y="15441" text-anchor="start" style="fill:var(--c-text);font-size:11px">sığmaz</text>
<text x="40" y="15467" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Kaynaklar 4D&#x27;nin dördüncü boyutunu EP ya da CP olarak sayar: MoE modelleri uzmanları EP ile böler (DeepSeek-V3: 256 uzman, EP 64),</text>
<text x="40" y="15482" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Llama 3 128K bağlamda diziyi CP ile böler (CP 16).</text>
<text x="40" y="15522" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Nasıl çalışır? Dört resim: küçük örnek ve Llama 3&#x27;teki karşılığı</text>
<rect x="36" y="15536" width="374" height="317" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="422" y="15536" width="374" height="317" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="15557" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">DP · veri paralelliği</text>
<rect x="48.0" y="15570" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="88.8" y="15585" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">kopya 0</text>
<rect x="54.0" y="15592.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.8" y="15605.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">tam model</text>
<rect x="54.0" y="15615.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="88.8" y="15628.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">veri 1/4</text>
<line x1="88.8" y1="15640.0" x2="88.8" y2="15654.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="137.5" y="15570" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="178.2" y="15585" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">kopya 1</text>
<rect x="143.5" y="15592.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="178.2" y="15605.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">tam model</text>
<rect x="143.5" y="15615.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="178.2" y="15628.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">veri 2/4</text>
<line x1="178.2" y1="15640.0" x2="178.2" y2="15654.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="227.0" y="15570" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="267.8" y="15585" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">kopya 2</text>
<rect x="233.0" y="15592.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="267.8" y="15605.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">tam model</text>
<rect x="233.0" y="15615.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="267.8" y="15628.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">veri 3/4</text>
<line x1="267.8" y1="15640.0" x2="267.8" y2="15654.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="316.5" y="15570" width="81.5" height="70" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="357.2" y="15585" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">kopya 3</text>
<rect x="322.5" y="15592.0" width="69.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.20;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="357.2" y="15605.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">tam model</text>
<rect x="322.5" y="15615.0" width="69.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="357.2" y="15628.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">veri 4/4</text>
<line x1="357.2" y1="15640.0" x2="357.2" y2="15654.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="88.8" y1="15654.0" x2="357.2" y2="15654.0" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<text x="223.0" y="15671" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">gradyan AllReduce · adımda bir</text>
<text x="48" y="15694" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Resimde:</tspan> 4 kopya aynı modeli taşır, her biri verinin dörtte</text>
<text x="48" y="15709" text-anchor="start" style="fill:var(--c-text);font-size:11px">birini işler; adım sonunda gradyanlar ortalanır ve herkes aynı</text>
<text x="48" y="15724" text-anchor="start" style="fill:var(--c-text);font-size:11px">güncellemeyi yapar.</text>
<text x="48" y="15744" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Llama 3&#x27;te:</tspan> model tek GPU&#x27;ya sığmadığı için bir kopya tek GPU</text>
<text x="48" y="15759" text-anchor="start" style="fill:var(--c-text);font-size:11px">değil, 128 GPU&#x27;luk bir zincirdir (TP 8 × PP 16). DP 128 bu</text>
<text x="48" y="15774" text-anchor="start" style="fill:var(--c-text);font-size:11px">zincirden 128 tane demektir. Gradyanı aynı adresteki GPU&#x27;lar</text>
<text x="48" y="15789" text-anchor="start" style="fill:var(--c-text);font-size:11px">ortalar: [3, 5, 0] … [3, 5, 127], her biri kendi 3,16 milyar</text>
<text x="48" y="15804" text-anchor="start" style="fill:var(--c-text);font-size:11px">parametrelik payı için.</text>
<text x="48" y="15824" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Bedeli:</tspan> her kopya modelin tamamını taşır (FSDP bunu</text>
<text x="48" y="15839" text-anchor="start" style="fill:var(--c-text);font-size:11px">hafifletir); iletişim adımda bir, gradyan boyutunda.</text>
<text x="434" y="15557" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">TP · tensör paralelliği</text>
<rect x="434.0" y="15570.0" width="34.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="451.0" y="15616.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700;font-family:var(--font-mono)">X</text>
<line x1="470.0" y1="15579.0" x2="482.0" y2="15579.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15570.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="528.0" y="15583.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in sütun 0</text>
<line x1="574.0" y1="15579.0" x2="584.0" y2="15579.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15570.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="630.0" y="15583.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out satır 0</text>
<line x1="674.0" y1="15579.0" x2="690.0" y2="15612.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="470.0" y1="15601.0" x2="482.0" y2="15601.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15592.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="528.0" y="15605.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in sütun 1</text>
<line x1="574.0" y1="15601.0" x2="584.0" y2="15601.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15592.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="630.0" y="15605.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out satır 1</text>
<line x1="674.0" y1="15601.0" x2="690.0" y2="15612.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="470.0" y1="15623.0" x2="482.0" y2="15623.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15614.0" width="88.0" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="528.0" y="15627.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in sütun 2</text>
<line x1="574.0" y1="15623.0" x2="584.0" y2="15623.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15614.0" width="88.0" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="630.0" y="15627.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out satır 2</text>
<line x1="674.0" y1="15623.0" x2="690.0" y2="15612.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="470.0" y1="15645.0" x2="482.0" y2="15645.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="484.0" y="15636.0" width="88.0" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="528.0" y="15649.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_in sütun 3</text>
<line x1="574.0" y1="15645.0" x2="584.0" y2="15645.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="586.0" y="15636.0" width="88.0" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="630.0" y="15649.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">W_out satır 3</text>
<line x1="674.0" y1="15645.0" x2="690.0" y2="15612.0" style="stroke:var(--c-accent-2);stroke-width:1.2"/>
<circle cx="702" cy="15612" r="12" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="702" y="15617" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">Σ</text>
<line x1="715.0" y1="15612.0" x2="736.0" y2="15612.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<rect x="738.0" y="15602.0" width="30.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="753.0" y="15616.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700;font-family:var(--font-mono)">Y</text>
<text x="434" y="15674" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her satır bir GPU; Σ = AllReduce.</text>
<text x="434" y="15694" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Resimde:</tspan> 4 GPU bir katmanı paylaşır. X herkeste aynıdır; her</text>
<text x="434" y="15709" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU W_in&#x27;in bir sütun dilimini ve W_out&#x27;un bir satır dilimini</text>
<text x="434" y="15724" text-anchor="start" style="fill:var(--c-text);font-size:11px">taşır, kısmi çıktılar Σ ile (AllReduce) toplanır.</text>
<text x="434" y="15744" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Llama 3&#x27;te:</tspan> TP 8: düğümün 8 GPU&#x27;su her katmanı paylaşır.</text>
<text x="434" y="15759" text-anchor="start" style="fill:var(--c-text);font-size:11px">405B&#x27;nin 16.384 × 53.248&#x27;lik W_in&#x27;inden her GPU</text>
<text x="434" y="15774" text-anchor="start" style="fill:var(--c-text);font-size:11px">16.384 × 6.656&#x27;lık dilimi taşır; dikkat katmanında 128 başın</text>
<text x="434" y="15789" text-anchor="start" style="fill:var(--c-text);font-size:11px">16&#x27;sı her GPU&#x27;ya düşer.</text>
<text x="434" y="15809" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent);font-weight:700">Bedeli:</tspan> her katmanda ileri 2, geri 2 toplama (405B&#x27;de her biri</text>
<text x="434" y="15824" text-anchor="start" style="fill:var(--c-text);font-size:11px">~1 ms); bu yüzden yalnız NVLink&#x27;te.</text>
<rect x="36" y="15865" width="374" height="342" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="422" y="15865" width="374" height="342" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="15886" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">PP · boru hattı paralelliği</text>
<text x="48" y="15911" text-anchor="start" style="fill:var(--c-text);font-size:11px">aşama 0</text>
<rect x="107.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="116.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="127.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="136.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="147.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="167.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="187.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="207.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="227.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="247.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="267.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="287.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="307.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="327.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="336.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="347.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="367.0" y="15899.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="376.0" y="15911.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="48" y="15931" text-anchor="start" style="fill:var(--c-text);font-size:11px">aşama 1</text>
<rect x="107.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="127.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="136.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="147.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="167.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="187.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="196.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="207.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="227.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="247.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="267.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="287.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="296.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="307.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="327.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="336.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="347.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="15931.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="367.0" y="15919.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="48" y="15951" text-anchor="start" style="fill:var(--c-text);font-size:11px">aşama 2</text>
<rect x="107.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="127.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="147.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="167.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="187.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="196.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="207.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="216.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="227.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="247.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="267.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="276.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="287.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="296.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="307.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="327.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="336.0" y="15951.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="347.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="367.0" y="15939.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="48" y="15971" text-anchor="start" style="fill:var(--c-text);font-size:11px">aşama 3</text>
<rect x="107.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="127.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="147.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="167.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="176.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="187.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="196.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="207.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="216.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="227.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="247.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="256.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="267.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="276.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="287.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="296.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="307.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="316.0" y="15971.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="327.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="347.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="367.0" y="15959.0" width="18.0" height="16" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="48.0" y="15983.0" width="14.0" height="12" rx="5" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="68" y="15993" text-anchor="start" style="fill:var(--c-text);font-size:11px">ileri</text>
<rect x="112.2" y="15983.0" width="14.0" height="12" rx="5" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="132.2" y="15993" text-anchor="start" style="fill:var(--c-text);font-size:11px">geri</text>
<rect x="170.4" y="15983.0" width="14.0" height="12" rx="5" style="fill:var(--c-text-mute);fill-opacity:.04;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="190.4" y="15993" text-anchor="start" style="fill:var(--c-text);font-size:11px">kabarcık</text>
<text x="398" y="15993" text-anchor="end" style="fill:var(--c-danger);font-size:11px;font-weight:700">P = 4, M = 4 → 3/7 ≈ %43</text>
<text x="48" y="16017" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent-2);font-weight:700">Resimde:</tspan> 4 aşama (düğüm) ve 4 mikro-batch. Aşama 0 birinci</text>
<text x="48" y="16032" text-anchor="start" style="fill:var(--c-text);font-size:11px">mikro-batch&#x27;i bitirince aşama 1&#x27;e verir; ileri geçişler</text>
<text x="48" y="16047" text-anchor="start" style="fill:var(--c-text);font-size:11px">bitince geri geçiş ters yönde döner.</text>
<text x="48" y="16067" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent-2);font-weight:700">Llama 3&#x27;te:</tspan> PP 16: 16 düğüm, her biri ~8 katman; her kopya</text>
<text x="48" y="16082" text-anchor="start" style="fill:var(--c-text);font-size:11px">adımda 16 dizi (mikro-batch) işler. Bu basit düzende kabarcık</text>
<text x="48" y="16097" text-anchor="start" style="fill:var(--c-text);font-size:11px">%48 olurdu; Llama 3 interleaved düzenle her düğüme birden çok</text>
<text x="48" y="16112" text-anchor="start" style="fill:var(--c-text);font-size:11px">küçük katman parçası verir ve kabarcığı bu parça sayısı kadar</text>
<text x="48" y="16127" text-anchor="start" style="fill:var(--c-text);font-size:11px">küçültür.</text>
<text x="48" y="16147" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-accent-2);font-weight:700">Bedeli:</tspan> hat dolup boşalırken boşta kalan dilimler (kabarcık);</text>
<text x="48" y="16162" text-anchor="start" style="fill:var(--c-text);font-size:11px">mikro-batch sayısı arttıkça küçülür.</text>
<text x="434" y="15886" text-anchor="start" style="fill:var(--c-warn);font-size:13px;font-weight:700">EP · uzman paralelliği</text>
<line x1="448.0" y1="15929.0" x2="474.8" y2="15965.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="466.0" y1="15929.0" x2="564.2" y2="15965.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="484.0" y1="15929.0" x2="653.8" y2="15965.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="502.0" y1="15929.0" x2="743.2" y2="15965.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<line x1="537.5" y1="15929.0" x2="474.8" y2="15965.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="555.5" y1="15929.0" x2="564.2" y2="15965.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="573.5" y1="15929.0" x2="653.8" y2="15965.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="591.5" y1="15929.0" x2="743.2" y2="15965.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<line x1="627.0" y1="15929.0" x2="474.8" y2="15965.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="645.0" y1="15929.0" x2="564.2" y2="15965.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="663.0" y1="15929.0" x2="653.8" y2="15965.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="681.0" y1="15929.0" x2="743.2" y2="15965.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<line x1="716.5" y1="15929.0" x2="474.8" y2="15965.0" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<line x1="734.5" y1="15929.0" x2="564.2" y2="15965.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="752.5" y1="15929.0" x2="653.8" y2="15965.0" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<line x1="770.5" y1="15929.0" x2="743.2" y2="15965.0" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="434.0" y="15899" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="474.8" y="15912" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 0</text>
<circle cx="448.0" cy="15921" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="466.0" cy="15921" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="484.0" cy="15921" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="502.0" cy="15921" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="434.0" y="15965.0" width="81.5" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="474.8" y="15978" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 0</text>
<text x="474.8" y="15991" text-anchor="middle" style="fill:var(--c-text);font-size:11px">uzman 0–1</text>
<rect x="523.5" y="15899" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="564.2" y="15912" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 1</text>
<circle cx="537.5" cy="15921" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="555.5" cy="15921" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="573.5" cy="15921" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="591.5" cy="15921" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="523.5" y="15965.0" width="81.5" height="30" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="564.2" y="15978" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 1</text>
<text x="564.2" y="15991" text-anchor="middle" style="fill:var(--c-text);font-size:11px">uzman 2–3</text>
<rect x="613.0" y="15899" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="653.8" y="15912" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 2</text>
<circle cx="627.0" cy="15921" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="645.0" cy="15921" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="663.0" cy="15921" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="681.0" cy="15921" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="613.0" y="15965.0" width="81.5" height="30" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="653.8" y="15978" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 2</text>
<text x="653.8" y="15991" text-anchor="middle" style="fill:var(--c-text);font-size:11px">uzman 4–5</text>
<rect x="702.5" y="15899" width="81.5" height="30" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="743.2" y="15912" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 3</text>
<circle cx="716.5" cy="15921" r="3.5" style="fill:var(--c-accent);fill-opacity:.85"/>
<circle cx="734.5" cy="15921" r="3.5" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<circle cx="752.5" cy="15921" r="3.5" style="fill:var(--c-warn);fill-opacity:.85"/>
<circle cx="770.5" cy="15921" r="3.5" style="fill:var(--c-success);fill-opacity:.85"/>
<rect x="702.5" y="15965.0" width="81.5" height="30" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="743.2" y="15978" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU 3</text>
<text x="743.2" y="15991" text-anchor="middle" style="fill:var(--c-text);font-size:11px">uzman 6–7</text>
<text x="434" y="16013" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Renk tokenin gideceği uzmanı gösterir: her GPU her GPU&#x27;ya</text>
<text x="434" y="16028" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yollar (AllToAll).</text>
<text x="434" y="16048" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-warn);font-weight:700">Resimde:</tspan> 8 uzman 4 GPU&#x27;ya dağıtılmış (GPU başına 2). Her</text>
<text x="434" y="16063" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;daki tokenlar renklerine göre uzmanlarının GPU&#x27;suna gider</text>
<text x="434" y="16078" text-anchor="start" style="fill:var(--c-text);font-size:11px">(dispatch) ve sonuçla geri döner (combine).</text>
<text x="434" y="16098" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-warn);font-weight:700">DeepSeek-V3&#x27;te:</tspan> katman başına 256 uzman, EP 64: 8 düğümdeki</text>
<text x="434" y="16113" text-anchor="start" style="fill:var(--c-text);font-size:11px">64 GPU&#x27;nun her biri 4 uzman taşır; her token 8 uzmana, en</text>
<text x="434" y="16128" text-anchor="start" style="fill:var(--c-text);font-size:11px">fazla 4 düğüme gider. Llama 3 yoğun (dense) bir model olduğu</text>
<text x="434" y="16143" text-anchor="start" style="fill:var(--c-text);font-size:11px">için EP kullanmaz.</text>
<text x="434" y="16163" text-anchor="start" style="fill:var(--c-text);font-size:11px"><tspan style="fill:var(--c-warn);font-weight:700">Bedeli:</tspan> her MoE katmanında herkes herkese yollar (AllToAll) ve</text>
<text x="434" y="16178" text-anchor="start" style="fill:var(--c-text);font-size:11px">ağa yük bindirir; bir uzmana çok token düşerse diğerleri</text>
<text x="434" y="16193" text-anchor="start" style="fill:var(--c-text);font-size:11px">bekler.</text>
<rect x="28" y="16237" width="784" height="863" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="16263" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">Veri nereden geçer? Sekiz adım, dört yol</text>
<text x="40" y="16283" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Düzen hazır; şimdi bu 16.384 GPU&#x27;da tek bir eğitim adımını izleyelim. Haritadaki numaralar aşağıdaki adımlardır; sayılar GPU</text>
<text x="40" y="16300" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">başına ve bir adım içindir (mikro-batch = bir dizi, 8.192 token).</text>
<rect x="290.0" y="16322" width="260.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="16337" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine → diğer 127 DP kopyası</text>
<rect x="48" y="16370" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="58" y="16386" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">düğüm A · PP aşaması s</text>
<rect x="88.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="150.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="212.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="274.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="88.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="150.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="212.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="274.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="112.0" y1="16414" x2="112.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="112.0" y1="16440" x2="112.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="16414" x2="174.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="16440" x2="174.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="16414" x2="236.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="16440" x2="236.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="16414" x2="298.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="16440" x2="298.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="88.0" y="16422" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="205.0" y="16435" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<circle cx="64" cy="16404" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="64" cy="16431" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="64" y="16435" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="64" cy="16458" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<rect x="502" y="16370" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="512" y="16386" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">düğüm B · aşama s+1</text>
<rect x="530.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="592.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="654.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="716.0" y="16394" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="16408" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="530.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="592.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="654.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="716.0" y="16448" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="16462" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="554.0" y1="16414" x2="554.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="554.0" y1="16440" x2="554.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="16414" x2="616.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="16440" x2="616.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="16414" x2="678.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="16440" x2="678.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="16414" x2="740.0" y2="16422" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="16440" x2="740.0" y2="16448" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="530.0" y="16422" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="16435" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<rect x="396.0" y="16380" width="48.0" height="96" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="16438" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">leaf</text>
<circle cx="420.0" cy="16406" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="420.0" y="16410" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="420.0" cy="16460" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="420.0" y="16464" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<line x1="338.0" y1="16406" x2="396.0" y2="16406" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<line x1="444.0" y1="16406" x2="500.0" y2="16406" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="502.0" y1="16460" x2="444.0" y2="16460" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="396.0" y1="16460" x2="340.0" y2="16460" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<line x1="420.0" y1="16378" x2="420.0" y2="16346" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="438.0" cy="16362" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438.0" y="16366" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="48.0" y="16522" width="744.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="420.0" y="16538" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">depolama sistemi (ör. Lustre · Weka)</text>
<text x="420.0" y="16554" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ayrı depolama ağı · düğüm başına ≥ 40 GB/s</text>
<line x1="108.0" y1="16520" x2="108.0" y2="16489" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<circle cx="126" cy="16504" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="126" y="16508" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<line x1="208.0" y1="16489" x2="208.0" y2="16520" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5;stroke-dasharray:5 4"/>
<circle cx="226" cy="16504" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="226" y="16508" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="647.0" y="16506" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(B düğümü de aynı yolu kullanır)</text>
<line x1="40.0" y1="16576.0" x2="800.0" y2="16576.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="46.0" y1="16604.0" x2="46.0" y2="16978.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="46" cy="16604" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="46" y="16608.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="16609" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Veriyi oku</text>
<rect x="66.0" y="16618.0" width="177.4" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="154.7" y="16631.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">depolama → HBM · 67 MB/adım</text>
<text x="66" y="16654" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Adımın 16,8 milyon tokeni (128 × 16 dizi × 8.192) int32</text>
<text x="66" y="16670" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">olarak 67 MB eder, saniyede ~11 MB. Yalnız ilk aşama</text>
<text x="66" y="16686" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">token okur; mmap ya da GDS araya CPU kopyası sokmaz.</text>
<circle cx="46" cy="16718" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="46" y="16722.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="16723" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">İleri geçiş</text>
<rect x="66.0" y="16732.0" width="165.2" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="148.6" y="16745.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Tensor Core · ~2,1 s/adım</text>
<text x="66" y="16768" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GPU başına 3,16 milyar parametre: 8.192 tokenlik</text>
<text x="66" y="16784" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">mikro-batch başına ağırlıklar bir kez okunur (6,3 GB,</text>
<text x="66" y="16800" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">~2 ms) ama 52 TFLOP iş yapılır (~130 ms). Hesap</text>
<text x="66" y="16816" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">baskındır.</text>
<circle cx="46" cy="16848" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="16852.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="16853" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP AllReduce</text>
<rect x="66.0" y="16862.0" width="135.0" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="133.5" y="16875.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink · ~500 × 1 ms</text>
<text x="66" y="16898" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her katman ileri geçişte 2, geri geçişte 2 kez</text>
<text x="66" y="16914" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">268 MB&#x27;lık aktivasyonu 8 GPU arasında toplar: ~8</text>
<text x="66" y="16930" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">katman × 4 × 16 mikro-batch ≈ 500 kez, her biri ~1 ms.</text>
<text x="66" y="16946" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">TP bu yüzden kasadan çıkmaz.</text>
<circle cx="46" cy="16978" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="16982.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="66" y="16983" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aktivasyonu gönder (PP)</text>
<rect x="66.0" y="16992.0" width="153.2" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="142.6" y="17005.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ray · 34 MB/mikro-batch</text>
<text x="66" y="17028" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Aşamanın çıktısı 8.192 × 16.384 × 2 bayt = 268 MB; her</text>
<text x="66" y="17044" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">TP GPU&#x27;su sekizde birini kendi rayından sonraki düğümün</text>
<text x="66" y="17060" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">aynı numaralı GPU&#x27;suna yollar: 34 MB, ~0,7 ms.</text>
<line x1="438.0" y1="16604.0" x2="438.0" y2="16994.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="438" cy="16604" r="10" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="438" y="16608.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="16609" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Geri geçiş</text>
<rect x="458.0" y="16618.0" width="195.5" height="19" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="555.8" y="16631.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">aynı hatlar, ters yön · ~4,1 s</text>
<text x="458" y="16654" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Gradyanlar aynı yoldan geri akar; 3. ve 4. adımlar</text>
<text x="458" y="16670" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tersine tekrarlanır ve iş ileri geçişin ~2 katıdır.</text>
<text x="458" y="16686" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Gradyanlar mikro-batch&#x27;ler boyunca FP32 biriktirilir.</text>
<circle cx="438" cy="16718" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438" y="16722.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="16723" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Gradyan senkronu (DP)</text>
<rect x="458.0" y="16732.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="543.6" y="16745.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ray + spine · ~0,38 s/adım</text>
<text x="458" y="16768" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">FSDP, 128 kopyanın FP32 gradyanlarını reduce-scatter</text>
<text x="458" y="16784" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ile toplar (12,7 GB) ve güncel BF16 ağırlıkları</text>
<text x="458" y="16800" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">all-gather ile dağıtır (6,3 GB): 50 GB/s&#x27;te ~0,38 s,</text>
<text x="458" y="16816" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">geri geçişle örtüşür. AllReduce kullanan düzenlerde</text>
<text x="458" y="16832" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">SHARP toplamayı switch&#x27;te yapar (4b).</text>
<circle cx="438" cy="16864" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="438" y="16868.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<text x="458" y="16869" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Optimizer adımı</text>
<rect x="458.0" y="16878.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="543.6" y="16891.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM · GPU başına 25M param</text>
<text x="458" y="16914" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her GPU yalnız kendi 1/128 payını günceller: 24,7</text>
<text x="458" y="16930" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">milyon parametre ve 0,3 GB AdamW durumu (FP32 ana</text>
<text x="458" y="16946" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ağırlık, momentum, varyans); HBM&#x27;de milisaniyenin</text>
<text x="458" y="16962" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">altında biter.</text>
<circle cx="438" cy="16994" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="438" y="16998.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="458" y="16999" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kontrol noktası</text>
<rect x="458.0" y="17008.0" width="171.3" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="543.6" y="17021.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">depolama · ~130 adımda bir</text>
<text x="458" y="17044" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">~14 dakikada bir 7,5 TB durum yazılır (GPU başına</text>
<text x="458" y="17060" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">~460 MB): 250 GB/s ile ~31 s, eski 25 GB/s ile ~5 dk.</text>
<text x="458" y="17076" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Arızadan sonra eğitim buradan devam eder.</text>
<line x1="420.0" y1="16588.0" x2="420.0" y2="17082.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="17112" width="784" height="369" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="17138" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">Bir adımın bütçesi: hangi yol ne kadar sürer?</text>
<text x="40" y="17158" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Sekiz adımın süreleri yan yana, GPU başına ve seri toplanmış hâlde: zaman nereye gidiyor?</text>
<text x="40" y="17199" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">hesap (400 TFLOP/s)</text>
<rect x="200.0" y="17186" width="86.7" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="206" y="17199" text-anchor="start" style="fill:var(--c-text);font-size:11px">ileri 2,1</text>
<rect x="286.7" y="17186" width="173.3" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.40;stroke:var(--c-accent);stroke-width:1"/>
<text x="292.7" y="17199" text-anchor="start" style="fill:var(--c-text);font-size:11px">geri 4,1</text>
<text x="466.0" y="17199" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">6,2 s</text>
<text x="40" y="17225" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">hesap, tepe hızda (990)</text>
<rect x="200" y="17212" width="105.1" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:4 3"/>
<text x="311.1" y="17225" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">2,5 s</text>
<line x1="40.0" y1="17238.0" x2="520.0" y2="17238.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="17259" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink · TP AllReduce</text>
<rect x="200" y="17246" width="22.0" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="228.0" y="17259" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0,53 s</text>
<text x="40" y="17285" text-anchor="start" style="fill:var(--c-text);font-size:11px">ağ · DP senkronu</text>
<rect x="200" y="17272" width="15.8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="221.8" y="17285" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0,38 s</text>
<text x="40" y="17311" text-anchor="start" style="fill:var(--c-text);font-size:11px">depolama · kayıt payı</text>
<rect x="200" y="17298" width="9.8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="215.8" y="17311" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0,23 s</text>
<text x="40" y="17337" text-anchor="start" style="fill:var(--c-text);font-size:11px">ağ · PP aktivasyonu</text>
<rect x="200" y="17324" width="2.0" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.40;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="208" y="17337" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">0,02 s</text>
<text x="40" y="17363" text-anchor="start" style="fill:var(--c-text);font-size:11px">depolama · token okuma</text>
<rect x="200" y="17350" width="2.0" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.40;stroke:var(--c-warn);stroke-width:1"/>
<text x="208" y="17363" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">~0</text>
<text x="40" y="17386" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">İletişim süreleri seri toplanmıştır; gerçek adımda çoğu hesapla örtüşür. Kayıt payı:</text>
<text x="40" y="17401" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">31 s ÷ ~130 adım. PP en az 32 gönderimdir; interleaved düzende artar ama yine küçük</text>
<text x="40" y="17416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">kalır.</text>
<rect x="540.0" y="17174.0" width="260.0" height="144" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="17194" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">6,2 s nereden çıkar?</text>
<text x="552" y="17213" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6 × 405 milyar × 16,8M token</text>
<text x="552" y="17229" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">÷ (16.384 × 400 TFLOP/s)</text>
<text x="552" y="17245" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">≈ 6,2 s/adım</text>
<text x="552" y="17264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16,8M = 128 DP × 16 dizi × 8.192 token</text>
<text x="552" y="17279" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(Table 4); saniyede 2,7 milyon token</text>
<text x="552" y="17294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">işlenir. 400 TFLOP/s, H100&#x27;ün 990&#x27;ının</text>
<text x="552" y="17309" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">%41&#x27;idir (MFU).</text>
<text x="540" y="17340" text-anchor="start" style="fill:var(--c-text);font-size:12px">İletişim seri toplansa ~0,9 s eder ve çoğu</text>
<text x="540" y="17357" text-anchor="start" style="fill:var(--c-text);font-size:12px">hesabın arkasına saklanır: DP senkronu</text>
<text x="540" y="17374" text-anchor="start" style="fill:var(--c-text);font-size:12px">geri geçişle, PP gönderimi asenkron koşar.</text>
<text x="540" y="17391" text-anchor="start" style="fill:var(--c-text);font-size:12px">Aynı iş tepe hızda 2,5 s sürerdi; 6,2 s</text>
<text x="540" y="17408" text-anchor="start" style="fill:var(--c-text);font-size:12px">ile arasındaki farkı pipeline kabarcığı,</text>
<text x="540" y="17425" text-anchor="start" style="fill:var(--c-text);font-size:12px">açık kalan TP iletişimi ve kernel verimi</text>
<text x="540" y="17442" text-anchor="start" style="fill:var(--c-text);font-size:12px">açıklar. C&#x27;deki duvarlar bunları tek tek</text>
<text x="540" y="17459" text-anchor="start" style="fill:var(--c-text);font-size:12px">ele alır.</text>
<line x1="530.0" y1="17174.0" x2="530.0" y2="17459.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="17493.0" width="784.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="17513" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: bir adımda veri dört ayrı yoldan geçer; en sık konuşan en hızlı yola yerleşir (TP → NVLink,</text>
<text x="40" y="17530" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">PP → ray, DP → ağ, kayıt → depolama) ve iyi yerleşimde iletişim hesabın arkasına saklanır.</text>
<text x="420.0" y="17575" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Peki bu yollar nerede tıkanıyor? Her sütunun bir duvarı var.</text>
<line x1="420.0" y1="17583" x2="420.0" y2="17603" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="17607" width="808" height="1050" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="17631" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="17635.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">C</text>
<text x="56" y="17636" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Üç duvar: belirti → neden → çare</text>
<text x="28" y="17657" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Eğriler makaledeki formüllerden hesaplandı; kırmızı darboğaz, yeşil çaredir.</text>
<rect x="28" y="17669" width="784" height="295" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="17693" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Bellek duvarı</text>
<text x="800" y="17693" text-anchor="end" style="fill:var(--c-accent);font-size:11px;font-weight:700">1. sütun · hesaplama</text>
<line x1="80.0" y1="17713" x2="80.0" y2="17839" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="17839" x2="512.0" y2="17839" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="17720.2" x2="512.0" y2="17720.2" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="17724.2" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">990</text>
<text x="76" y="17843" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="40" y="17712.2" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s</text>
<line x1="377.9" y1="17720.2" x2="377.9" y2="17839" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<polyline points="80,17838.6 84.5,17838.6 89,17838.5 93.5,17838.5 98,17838.4 102.5,17838.4 107,17838.3 111.5,17838.3 116,17838.2 120.5,17838.1 125,17838 129.5,17838 134,17837.9 138.5,17837.8 143,17837.6 147.5,17837.5 152,17837.4 156.5,17837.2 161,17837.1 165.5,17836.9 170,17836.7 174.5,17836.5 179,17836.3 183.5,17836.1 188,17835.8 192.5,17835.5 197,17835.2 201.5,17834.8 206,17834.5 210.5,17834 215,17833.6 219.5,17833.1 224,17832.6 228.5,17832 233,17831.4 237.5,17830.7 242,17829.9 246.5,17829.1 251,17828.2 255.5,17827.3 260,17826.2 264.5,17825 269,17823.8 273.5,17822.4 278,17820.9 282.5,17819.3 287,17817.5 291.5,17815.6 296,17813.5 300.5,17811.2 305,17808.8 309.5,17806.1 314,17803.1 318.5,17799.9 323,17796.4 327.5,17792.7 332,17788.6 336.5,17784.1 341,17779.2 345.5,17774 350,17768.2 354.5,17762 359,17755.3 363.5,17748 368,17740.1 372.5,17731.5 377,17722.2 381.5,17720.2 386,17720.2 390.5,17720.2 395,17720.2 399.5,17720.2 404,17720.2 408.5,17720.2 413,17720.2 417.5,17720.2 422,17720.2 426.5,17720.2 431,17720.2 435.5,17720.2 440,17720.2 444.5,17720.2 449,17720.2 453.5,17720.2 458,17720.2 462.5,17720.2 467,17720.2 471.5,17720.2 476,17720.2 480.5,17720.2 485,17720.2 489.5,17720.2 494,17720.2 498.5,17720.2 503,17720.2 507.5,17720.2 512,17720.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.2"/>
<text x="80.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="152.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="224.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="296.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="368.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="440.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.024</text>
<text x="512.0" y="17853" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4.096</text>
<text x="512" y="17869" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">batch B (token, log ölçek)</text>
<text x="150" y="17736.2" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">bellek sınırlı</text>
<text x="470" y="17821" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">hesap sınırlı</text>
<circle cx="80.0" cy="17838.6" r="4" style="fill:var(--c-danger)"/>
<text x="88.0" y="17829" text-anchor="start" style="fill:var(--c-text);font-size:11px">B = 1: 3,3 TFLOP/s (%0,3)</text>
<circle cx="332.0" cy="17788.6" r="4" style="fill:var(--c-accent)"/>
<text x="324.0" y="17782.6" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (%42)</text>
<circle cx="377.9" cy="17720.2" r="4" style="fill:var(--c-success)"/>
<text x="385.9" y="17736.2" text-anchor="start" style="fill:var(--c-text);font-size:11px">B ≈ 310: tavan 990</text>
<text x="40" y="17887" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP (D = 8.192, F = 28.672), H100 BF16: hız = min(990, 3,35·α)</text>
<text x="40" y="17903" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: batch büyüdükçe okunan her bayt daha çok işe yarar; eğri B ≈ 310&#x27;da</text>
<text x="40" y="17918" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">tavana değer.</text>
<text x="552" y="17723" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="552" y="17741" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU meşgul görünür, ama gücünün yalnızca</text>
<text x="552" y="17758" text-anchor="start" style="fill:var(--c-text);font-size:12px">%1–2&#x27;si kullanılır.</text>
<text x="552" y="17786" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="552" y="17804" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her bayta düşen iş az: α = BDF / (B(D+F)</text>
<text x="552" y="17821" text-anchor="start" style="fill:var(--c-text);font-size:12px">+ DF), H100&#x27;ün eşiği olan 295&#x27;in altında</text>
<text x="552" y="17838" text-anchor="start" style="fill:var(--c-text);font-size:12px">kaldıkça Tensor Core&#x27;lar veri bekler.</text>
<text x="552" y="17866" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="552.0" y="17876" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="17890" text-anchor="middle" style="fill:var(--c-text);font-size:11px">batch ≥ ~310 token</text>
<rect x="552.0" y="17902" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="17916" text-anchor="middle" style="fill:var(--c-text);font-size:11px">füzyon / FlashAttention</text>
<rect x="552.0" y="17928" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="17942" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode: 4–8 bit ağırlık</text>
<line x1="540.0" y1="17709" x2="540.0" y2="17948" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="17974" width="784" height="321" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="17998" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Ağ duvarı</text>
<text x="800" y="17998" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">2. sütun · ağ</text>
<line x1="184.0" y1="18016" x2="184.0" y2="18100" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="184" y="18114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="288.0" y1="18016" x2="288.0" y2="18100" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="288" y="18114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="392.0" y1="18016" x2="392.0" y2="18100" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="392" y="18114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="496.0" y1="18016" x2="496.0" y2="18100" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="496" y="18114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="40" y="18034" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM (çip içi)</text>
<rect x="184" y="18022" width="262.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="452.6" y="18034" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3,35 TB/s</text>
<text x="40" y="18060" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="184" y="18048" width="171.9" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="361.9" y="18060" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="420.7" y="18060" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7,4</text>
<text x="40" y="18086" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (düğüm dışı)</text>
<rect x="184" y="18074" width="72.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="262.7" y="18086" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="314.9" y="18086" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<rect x="40.0" y="18126" width="480.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="18140" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink: Y ≤ F / (C/W) = 28.672 / 2.200 ≈ 13 → TP = 8 sığar ✓ (F: Llama 3 70B)</text>
<rect x="40.0" y="18152" width="480.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="18166" text-anchor="middle" style="fill:var(--c-text);font-size:11px">InfiniBand: Y ≤ 28.672 / 19.800 ≈ 1,45 → TP düğümden taşarsa hesap aç kalır ✗</text>
<text x="40" y="18190" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP = 16 aşama, kabarcık = 15 / (15 + M): M = 16 → %48 boşta · M = 128 → %10,5</text>
<text x="40" y="18208" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: her çubuk bir H100&#x27;ün o hattaki bant genişliği; düğümden çıkınca çubuk</text>
<text x="40" y="18223" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">9 kat kısalır.</text>
<text x="552" y="18028" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="552" y="18046" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğüm ekledikçe adım uzar; GPU&#x27;lar</text>
<text x="552" y="18063" text-anchor="start" style="fill:var(--c-text);font-size:12px">konuşmak için bekler.</text>
<text x="552" y="18091" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="552" y="18109" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğümden çıkan veri 9 kat daha dar bir</text>
<text x="552" y="18126" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoldan geçer; TP gibi her katmanda</text>
<text x="552" y="18143" text-anchor="start" style="fill:var(--c-text);font-size:12px">konuşan işler bu yolda boğulur.</text>
<text x="552" y="18171" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="552.0" y="18181" width="104.8" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="604.4" y="18195" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TP düğüm içinde</text>
<rect x="552.0" y="18207" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="18221" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PP ve DP düğümler arası</text>
<rect x="552.0" y="18233" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="18247" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP: 2·B/W → B/W</text>
<rect x="552.0" y="18259" width="165.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="634.6" y="18273" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DualPipe: iletişimi gizle</text>
<line x1="540.0" y1="18014" x2="540.0" y2="18279" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="18305" width="784" height="295" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="18329" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">G/Ç duvarı</text>
<text x="800" y="18329" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:700">3. sütun · depolama</text>
<line x1="80.0" y1="18353" x2="80.0" y2="18475" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="18475" x2="512.0" y2="18475" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="18445" x2="512.0" y2="18445" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18449" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%10</text>
<line x1="80.0" y1="18415" x2="512.0" y2="18415" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18419" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%20</text>
<line x1="80.0" y1="18385" x2="512.0" y2="18385" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18389" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%30</text>
<line x1="80.0" y1="18355" x2="512.0" y2="18355" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="18359" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%40</text>
<polyline points="80,18475 512,18400" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<polyline points="148.4,18355.3 149.6,18357 150.8,18358.6 152,18360.1 153.2,18361.6 154.4,18363 155.6,18364.4 156.8,18365.7 158,18366.9 159.2,18368.2 160.4,18369.3 161.6,18370.5 162.8,18371.6 164,18372.6 165.2,18373.7 166.4,18374.7 167.6,18375.6 168.8,18376.6 170,18377.5 171.2,18378.3 172.4,18379.2 173.6,18380 174.8,18380.8 176,18381.5 177.2,18382.3 178.4,18383 179.6,18383.7 180.8,18384.4 182,18385 183.2,18385.6 184.4,18386.3 185.6,18386.8 186.8,18387.4 188,18388 189.2,18388.5 190.4,18389.1 191.6,18389.6 192.8,18390.1 194,18390.5 195.2,18391 196.4,18391.5 197.6,18391.9 198.8,18392.3 200,18392.7 201.2,18393.1 202.4,18393.5 203.6,18393.9 204.8,18394.3 206,18394.6 207.2,18395 208.4,18395.3 209.6,18395.6 210.8,18395.9 212,18396.2 213.2,18396.5 214.4,18396.8 215.6,18397.1 216.8,18397.4 218,18397.6 219.2,18397.9 220.4,18398.1 221.6,18398.3 222.8,18398.6 224,18398.8 225.2,18399 226.4,18399.2 227.6,18399.4 228.8,18399.6 230,18399.8 231.2,18400 232.4,18400.2 233.6,18400.3 234.8,18400.5 236,18400.7 237.2,18400.8 238.4,18401 239.6,18401.1 240.8,18401.2 242,18401.4 243.2,18401.5 244.4,18401.6 245.6,18401.7 246.8,18401.8 248,18401.9 249.2,18402.1 250.4,18402.1 251.6,18402.2 252.8,18402.3 254,18402.4 255.2,18402.5 256.4,18402.6 257.6,18402.7 258.8,18402.7 260,18402.8 261.2,18402.9 262.4,18402.9 263.6,18403 264.8,18403 266,18403.1 267.2,18403.1 268.4,18403.2 269.6,18403.2 270.8,18403.2 272,18403.3 273.2,18403.3 274.4,18403.3 275.6,18403.3 276.8,18403.4 278,18403.4 279.2,18403.4 280.4,18403.4 281.6,18403.4 282.8,18403.4 284,18403.4 285.2,18403.4 286.4,18403.4 287.6,18403.4 288.8,18403.4 290,18403.4 291.2,18403.4 292.4,18403.4 293.6,18403.4 294.8,18403.4 296,18403.4 297.2,18403.3 298.4,18403.3 299.6,18403.3 300.8,18403.3 302,18403.2 303.2,18403.2 304.4,18403.2 305.6,18403.2 306.8,18403.1 308,18403.1 309.2,18403 310.4,18403 311.6,18403 312.8,18402.9 314,18402.9 315.2,18402.8 316.4,18402.8 317.6,18402.7 318.8,18402.7 320,18402.6 321.2,18402.6 322.4,18402.5 323.6,18402.4 324.8,18402.4 326,18402.3 327.2,18402.3 328.4,18402.2 329.6,18402.1 330.8,18402.1 332,18402 333.2,18401.9 334.4,18401.9 335.6,18401.8 336.8,18401.7 338,18401.6 339.2,18401.6 340.4,18401.5 341.6,18401.4 342.8,18401.3 344,18401.2 345.2,18401.2 346.4,18401.1 347.6,18401 348.8,18400.9 350,18400.8 351.2,18400.7 352.4,18400.6 353.6,18400.6 354.8,18400.5 356,18400.4 357.2,18400.3 358.4,18400.2 359.6,18400.1 360.8,18400 362,18399.9 363.2,18399.8 364.4,18399.7 365.6,18399.6 366.8,18399.5 368,18399.4 369.2,18399.3 370.4,18399.2 371.6,18399.1 372.8,18399 374,18398.9 375.2,18398.8 376.4,18398.7 377.6,18398.6 378.8,18398.5 380,18398.3 381.2,18398.2 382.4,18398.1 383.6,18398 384.8,18397.9 386,18397.8 387.2,18397.7 388.4,18397.6 389.6,18397.4 390.8,18397.3 392,18397.2 393.2,18397.1 394.4,18397 395.6,18396.8 396.8,18396.7 398,18396.6 399.2,18396.5 400.4,18396.4 401.6,18396.2 402.8,18396.1 404,18396 405.2,18395.9 406.4,18395.7 407.6,18395.6 408.8,18395.5 410,18395.4 411.2,18395.2 412.4,18395.1 413.6,18395 414.8,18394.9 416,18394.7 417.2,18394.6 418.4,18394.5 419.6,18394.3 420.8,18394.2 422,18394.1 423.2,18393.9 424.4,18393.8 425.6,18393.7 426.8,18393.5 428,18393.4 429.2,18393.3 430.4,18393.1 431.6,18393 432.8,18392.9 434,18392.7 435.2,18392.6 436.4,18392.4 437.6,18392.3 438.8,18392.2 440,18392 441.2,18391.9 442.4,18391.7 443.6,18391.6 444.8,18391.5 446,18391.3 447.2,18391.2 448.4,18391 449.6,18390.9 450.8,18390.7 452,18390.6 453.2,18390.5 454.4,18390.3 455.6,18390.2 456.8,18390 458,18389.9 459.2,18389.7 460.4,18389.6 461.6,18389.4 462.8,18389.3 464,18389.1 465.2,18389 466.4,18388.8 467.6,18388.7 468.8,18388.5 470,18388.4 471.2,18388.2 472.4,18388.1 473.6,18387.9 474.8,18387.8 476,18387.6 477.2,18387.5 478.4,18387.3 479.6,18387.2 480.8,18387 482,18386.9 483.2,18386.7 484.4,18386.6 485.6,18386.4 486.8,18386.3 488,18386.1 489.2,18385.9 490.4,18385.8 491.6,18385.6 492.8,18385.5 494,18385.3 495.2,18385.2 496.4,18385 497.6,18384.8 498.8,18384.7 500,18384.5 501.2,18384.4 502.4,18384.2 503.6,18384.1 504.8,18383.9 506,18383.7 507.2,18383.6 508.4,18383.4 509.6,18383.3 510.8,18383.1 512,18382.9" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<line x1="286.1" y1="18403.4" x2="286.1" y2="18475" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="286.1" cy="18403.4" r="4" style="fill:var(--c-danger)"/>
<text x="286.1" y="18489" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">42,9</text>
<polyline points="87.2,18371.4 88.4,18385.8 89.6,18396.5 90.8,18404.9 92,18411.5 93.2,18416.9 94.4,18421.3 95.6,18425 96.8,18428.2 98,18430.9 99.2,18433.3 100.4,18435.3 101.6,18437.1 102.8,18438.7 104,18440.1 105.2,18441.4 106.4,18442.5 107.6,18443.5 108.8,18444.4 110,18445.2 111.2,18446 112.4,18446.6 113.6,18447.2 114.8,18447.8 116,18448.3 117.2,18448.7 118.4,18449.1 119.6,18449.5 120.8,18449.8 122,18450.2 123.2,18450.4 124.4,18450.7 125.6,18450.9 126.8,18451.1 128,18451.3 129.2,18451.5 130.4,18451.6 131.6,18451.8 132.8,18451.9 134,18452 135.2,18452.1 136.4,18452.1 137.6,18452.2 138.8,18452.3 140,18452.3 141.2,18452.3 142.4,18452.4 143.6,18452.4 144.8,18452.4 146,18452.4 147.2,18452.4 148.4,18452.3 149.6,18452.3 150.8,18452.3 152,18452.3 153.2,18452.2 154.4,18452.2 155.6,18452.1 156.8,18452.1 158,18452 159.2,18451.9 160.4,18451.9 161.6,18451.8 162.8,18451.7 164,18451.6 165.2,18451.6 166.4,18451.5 167.6,18451.4 168.8,18451.3 170,18451.2 171.2,18451.1 172.4,18451 173.6,18450.9 174.8,18450.8 176,18450.7 177.2,18450.5 178.4,18450.4 179.6,18450.3 180.8,18450.2 182,18450.1 183.2,18449.9 184.4,18449.8 185.6,18449.7 186.8,18449.6 188,18449.4 189.2,18449.3 190.4,18449.2 191.6,18449 192.8,18448.9 194,18448.7 195.2,18448.6 196.4,18448.5 197.6,18448.3 198.8,18448.2 200,18448 201.2,18447.9 202.4,18447.7 203.6,18447.6 204.8,18447.4 206,18447.3 207.2,18447.1 208.4,18447 209.6,18446.8 210.8,18446.7 212,18446.5 213.2,18446.3 214.4,18446.2 215.6,18446 216.8,18445.9 218,18445.7 219.2,18445.5 220.4,18445.4 221.6,18445.2 222.8,18445 224,18444.9 225.2,18444.7 226.4,18444.5 227.6,18444.4 228.8,18444.2 230,18444 231.2,18443.9 232.4,18443.7 233.6,18443.5 234.8,18443.4 236,18443.2 237.2,18443 238.4,18442.8 239.6,18442.7 240.8,18442.5 242,18442.3 243.2,18442.1 244.4,18442 245.6,18441.8 246.8,18441.6 248,18441.4 249.2,18441.3 250.4,18441.1 251.6,18440.9 252.8,18440.7 254,18440.6 255.2,18440.4 256.4,18440.2 257.6,18440 258.8,18439.8 260,18439.7 261.2,18439.5 262.4,18439.3 263.6,18439.1 264.8,18438.9 266,18438.7 267.2,18438.6 268.4,18438.4 269.6,18438.2 270.8,18438 272,18437.8 273.2,18437.6 274.4,18437.5 275.6,18437.3 276.8,18437.1 278,18436.9 279.2,18436.7 280.4,18436.5 281.6,18436.3 282.8,18436.2 284,18436 285.2,18435.8 286.4,18435.6 287.6,18435.4 288.8,18435.2 290,18435 291.2,18434.8 292.4,18434.7 293.6,18434.5 294.8,18434.3 296,18434.1 297.2,18433.9 298.4,18433.7 299.6,18433.5 300.8,18433.3 302,18433.1 303.2,18432.9 304.4,18432.8 305.6,18432.6 306.8,18432.4 308,18432.2 309.2,18432 310.4,18431.8 311.6,18431.6 312.8,18431.4 314,18431.2 315.2,18431 316.4,18430.8 317.6,18430.6 318.8,18430.5 320,18430.3 321.2,18430.1 322.4,18429.9 323.6,18429.7 324.8,18429.5 326,18429.3 327.2,18429.1 328.4,18428.9 329.6,18428.7 330.8,18428.5 332,18428.3 333.2,18428.1 334.4,18427.9 335.6,18427.7 336.8,18427.5 338,18427.4 339.2,18427.2 340.4,18427 341.6,18426.8 342.8,18426.6 344,18426.4 345.2,18426.2 346.4,18426 347.6,18425.8 348.8,18425.6 350,18425.4 351.2,18425.2 352.4,18425 353.6,18424.8 354.8,18424.6 356,18424.4 357.2,18424.2 358.4,18424 359.6,18423.8 360.8,18423.6 362,18423.4 363.2,18423.2 364.4,18423 365.6,18422.8 366.8,18422.6 368,18422.4 369.2,18422.2 370.4,18422 371.6,18421.8 372.8,18421.6 374,18421.5 375.2,18421.3 376.4,18421.1 377.6,18420.9 378.8,18420.7 380,18420.5 381.2,18420.3 382.4,18420.1 383.6,18419.9 384.8,18419.7 386,18419.5 387.2,18419.3 388.4,18419.1 389.6,18418.9 390.8,18418.7 392,18418.5 393.2,18418.3 394.4,18418.1 395.6,18417.9 396.8,18417.7 398,18417.5 399.2,18417.3 400.4,18417.1 401.6,18416.9 402.8,18416.7 404,18416.5 405.2,18416.3 406.4,18416.1 407.6,18415.9 408.8,18415.7 410,18415.5 411.2,18415.3 412.4,18415.1 413.6,18414.9 414.8,18414.7 416,18414.5 417.2,18414.3 418.4,18414.1 419.6,18413.9 420.8,18413.7 422,18413.5 423.2,18413.3 424.4,18413.1 425.6,18412.9 426.8,18412.7 428,18412.5 429.2,18412.3 430.4,18412.1 431.6,18411.9 432.8,18411.7 434,18411.5 435.2,18411.3 436.4,18411.1 437.6,18410.9 438.8,18410.7 440,18410.5 441.2,18410.3 442.4,18410 443.6,18409.8 444.8,18409.6 446,18409.4 447.2,18409.2 448.4,18409 449.6,18408.8 450.8,18408.6 452,18408.4 453.2,18408.2 454.4,18408 455.6,18407.8 456.8,18407.6 458,18407.4 459.2,18407.2 460.4,18407 461.6,18406.8 462.8,18406.6 464,18406.4 465.2,18406.2 466.4,18406 467.6,18405.8 468.8,18405.6 470,18405.4 471.2,18405.2 472.4,18405 473.6,18404.8 474.8,18404.6 476,18404.4 477.2,18404.2 478.4,18404 479.6,18403.8 480.8,18403.6 482,18403.4 483.2,18403.2 484.4,18403 485.6,18402.8 486.8,18402.6 488,18402.4 489.2,18402.2 490.4,18402 491.6,18401.8 492.8,18401.5 494,18401.3 495.2,18401.1 496.4,18400.9 497.6,18400.7 498.8,18400.5 500,18400.3 501.2,18400.1 502.4,18399.9 503.6,18399.7 504.8,18399.5 506,18399.3 507.2,18399.1 508.4,18398.9 509.6,18398.7 510.8,18398.5 512,18398.3" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<line x1="145.2" y1="18452.4" x2="145.2" y2="18475" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="145.2" cy="18452.4" r="4" style="fill:var(--c-success)"/>
<text x="145.2" y="18489" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">13,6</text>
<text x="80.0" y="18489" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="224.0" y="18489" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30</text>
<text x="368.0" y="18489" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">60</text>
<text x="512.0" y="18489" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">90</text>
<text x="512" y="18505" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">kayıt aralığı τ (dakika)</text>
<text x="40" y="18347" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıp</text>
<rect x="300" y="18354" width="12" height="10" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="318" y="18363" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">A · 25 GB/s: en az %24 kayıp</text>
<rect x="300" y="18371" width="12" height="10" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="318" y="18380" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">B · 250 GB/s (GDS): en az %7,5</text>
<text x="353.6" y="18443.5" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">geri sarma τ/2M (ortak)</text>
<text x="40" y="18523" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">M = 3 sa (16.384 H100&#x27;de ~3 saatte bir arıza) · kayıp = δ/τ + τ/2M</text>
<text x="40" y="18539" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: sık kayıt yazmaya, seyrek kayıt geri sarmaya zaman kaybettirir; en iyi</text>
<text x="40" y="18554" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">aralık çukurun dibidir.</text>
<text x="552" y="18359" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="552" y="18377" text-anchor="start" style="fill:var(--c-text);font-size:12px">Küme her kayıtta durur; arızada son</text>
<text x="552" y="18394" text-anchor="start" style="fill:var(--c-text);font-size:12px">kayda geri dönülür.</text>
<text x="552" y="18422" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="552" y="18440" text-anchor="start" style="fill:var(--c-text);font-size:12px">7,5 TB&#x27;lık kayıt CPU sayfa önbelleğinden</text>
<text x="552" y="18457" text-anchor="start" style="fill:var(--c-text);font-size:12px">25 GB/s ile yazılırsa δ = 307 s sürer ve</text>
<text x="552" y="18474" text-anchor="start" style="fill:var(--c-text);font-size:12px">küme bu sürede bekler.</text>
<text x="552" y="18502" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="552.0" y="18512" width="116.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="610.4" y="18526" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPUDirect Storage</text>
<rect x="552.0" y="18538" width="147.1" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="625.5" y="18552" text-anchor="middle" style="fill:var(--c-text);font-size:11px">asenkron / anlık yazma</text>
<rect x="552.0" y="18564" width="98.7" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="601.4" y="18578" text-anchor="middle" style="fill:var(--c-text);font-size:11px">τ = √(2δM) seç</text>
<line x1="540.0" y1="18345" x2="540.0" y2="18584" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="18610" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="18630" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: üç çarenin ortak fikri aynı: veriyi az taşı, taşırken hesabı durdurma.</text>
<text x="420.0" y="18677" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Bu farkları sezgiyle görelim: aynı yükü her yoldan geçirelim.</text>
<line x1="420.0" y1="18685" x2="420.0" y2="18705" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="18709" width="808" height="333" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="18733" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="18737.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">D</text>
<text x="56" y="18738" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Hız sezgisi: 140 GB&#x27;ı taşımak ne kadar sürer?</text>
<text x="28" y="18759" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 70B&#x27;nin BF16 ağırlıkları = 140 GB; aynı yük her hattan (log ölçek).</text>
<rect x="28" y="18771" width="784" height="214" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="200.0" y1="18785" x2="200.0" y2="18909" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="200" y="18923" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 ms</text>
<line x1="292.0" y1="18785" x2="292.0" y2="18909" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="18923" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 ms</text>
<line x1="384.0" y1="18785" x2="384.0" y2="18909" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="384" y="18923" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 s</text>
<line x1="476.0" y1="18785" x2="476.0" y2="18909" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="476" y="18923" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 s</text>
<text x="40" y="18803" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3 (tek H100)</text>
<rect x="200" y="18791" width="57.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="263.1" y="18803" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">42 ms</text>
<text x="40" y="18827" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="200" y="18815" width="137.3" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="343.3" y="18827" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0,31 s</text>
<text x="40" y="18851" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU yolu)</text>
<rect x="200" y="18839" width="215.3" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="421.3" y="18851" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,2 s</text>
<text x="40" y="18875" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (GPU başına)</text>
<rect x="200" y="18863" width="225.1" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="431.1" y="18875" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,8 s</text>
<text x="40" y="18899" text-anchor="start" style="fill:var(--c-text);font-size:11px">eski depolama (tüm küme)</text>
<rect x="200" y="18887" width="252.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.8" y="18899" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5,6 s</text>
<text x="40" y="18951" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Çubuk uzadıkça yavaşlar: her çubuk aynı 140 GB&#x27;ın o hattan geçme süresidir.</text>
<rect x="544.0" y="18785" width="256.0" height="80" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554" y="18803" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">İnsan ölçeği: HBM&#x27;den 1 saniyede gelen</text>
<text x="554" y="18819" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">veri</text>
<text x="554" y="18835" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink&#x27;ten 7,4 s, PCIe&#x27;den 52 s,</text>
<text x="554" y="18851" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand&#x27;den 67 s&#x27;de gelir.</text>
<rect x="544.0" y="18875" width="256.0" height="96" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="554" y="18893" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Decode&#x27;un tavanı: her token bütün</text>
<text x="554" y="18909" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">ağırlıkları bir kez okur</text>
<text x="554" y="18925" text-anchor="start" style="fill:var(--c-text);font-size:11px">8 H100&#x27;de 140 GB ÷</text>
<text x="554" y="18941" text-anchor="start" style="fill:var(--c-text);font-size:11px">(8 × 3,35 TB/s) ≈ 5,2 ms → batch 1&#x27;de en</text>
<text x="554" y="18957" text-anchor="start" style="fill:var(--c-text);font-size:11px">fazla ~190 token/s</text>
<line x1="532.0" y1="18785" x2="532.0" y2="18971" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="18997" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="19017" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: HBM&#x27;de milisaniyeler, ağda saniyeler. İyi tasarım veriyi hızlı katmanda tutar.</text>
<text x="420.0" y="19062" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Son durak: bütün bunlar gerçek bir kümede nasıl görünüyor?</text>
<line x1="420.0" y1="19070" x2="420.0" y2="19090" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="19094" width="808" height="310" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="19118" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="19122.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">E</text>
<text x="56" y="19123" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Gerçek küme: Meta&#x27;nın Llama 3 405B eğitimi</text>
<text x="28" y="19144" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Aynı 16.384 H100, Meta&#x27;nın kendi kümesinde (ağ: RoCE); sayılar Llama 3 raporundan.</text>
<rect x="28.0" y="19156" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="19164" width="237.3" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.7" y="19178" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Hesaplama</text>
<text x="42.0" y="19202" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16.384 H100</text>
<text x="42.0" y="19217" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP 8 × PP 16 × DP 128</text>
<text x="42.0" y="19232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">adım başına 16M token</text>
<text x="42.0" y="19247" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına 400 TFLOP/s</text>
<text x="42.0" y="19262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BF16 MFU %41</text>
<rect x="293.33333333333337" y="19156" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="301.3" y="19164" width="237.3" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="19178" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Ağ</text>
<text x="307.3" y="19202" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">RoCE · 400 Gb/s/GPU</text>
<text x="307.3" y="19217" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod = 3.072 GPU</text>
<text x="307.3" y="19232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod içi tam biseksiyon</text>
<text x="307.3" y="19247" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 pod arası 1:7</text>
<text x="307.3" y="19262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(küçük modeller: InfiniBand)</text>
<rect x="558.6666666666667" y="19156" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="566.7" y="19164" width="237.3" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="685.3" y="19178" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Depolama</text>
<text x="572.7" y="19202" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Tectonic · 240 PB</text>
<text x="572.7" y="19217" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7.500 SSD sunucusu</text>
<text x="572.7" y="19232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 TB/s sürekli</text>
<text x="572.7" y="19247" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7 TB/s tepe</text>
<text x="572.7" y="19262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıt: 1 MB–4 GB/GPU</text>
<text x="40" y="19298" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">MFU: 8.192 H100&#x27;de %43, 16.384&#x27;te %41, 128K bağlamda %38; ölçek büyüdükçe iletişim artar.</text>
<rect x="28.0" y="19312" width="386.0" height="80" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="38" y="19330" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">54 günde 466 kesinti; 419&#x27;u beklenmeyen → ortalama ~3 saatte</text>
<text x="38" y="19346" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">bir arıza</text>
<text x="38" y="19362" text-anchor="start" style="fill:var(--c-text);font-size:11px">%78 donanım (GPU %30,1 · HBM3 %17,2); otomasyonla etkin eğitim</text>
<text x="38" y="19378" text-anchor="start" style="fill:var(--c-text);font-size:11px">süresi &gt; %90</text>
<rect x="426.0" y="19312" width="386.0" height="80" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="436.0" y="19330" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ders: üç sütun birlikte tasarlanır.</text>
<text x="436.0" y="19347" text-anchor="start" style="fill:var(--c-text);font-size:11px">En yavaş halka (HBM, ağ ya da disk) bütün kümenin hızını</text>
<text x="436.0" y="19363" text-anchor="start" style="fill:var(--c-text);font-size:11px">belirler; MFU onun aynasıdır.</text>
</svg>

*Şekil 1.1: Üç sütunun büyük resmi. A: SM → GPU → DGX düğümü (sistem topolojisiyle) → SuperPOD ölçek merdiveni, SuperPOD ağının yakından incelemesi (port hesabı, trafik yolları, ölçek) ve kümeyi ayakta tutan 14 yapı taşı; B: bir eğitim adımında verinin yolu; C: üç duvar ve hesaplanmış eğrileri; D: 140 GB'lık yükle hız sezgisi; E: Llama 3 405B'nin gerçek sayıları. Kaynaklar: NVIDIA DGX H100 ve DGX SuperPOD belgeleri, The Llama 3 Herd of Models (2024).*

### Poster nasıl kurulu?

Şekil 1.1'deki mimari posteri yukarıdan aşağıya doğru bir donanım hiyerarşisi olarak okuyun; her bölüm tek bir mimari katmanı açar ve donanım bileşenlerinin nasıl bağlandığını gösterir:

| Bölüm | Odak Noktası | Akılda Tutulacak Mimari Değer | Detaylı İnceleme |
| :--- | :--- | :--- | :--- |
| **A · Donanım** | SM silikonundan 16.384 GPU'luk kümeye fiziksel hiyerarşi | GPU başına 990 TFLOP/s; 3,35 TB/s (HBM) → 450 GB/s (NVLink) → 50 GB/s (Ağ) | 2., 3. ve 4. bölümler |
| **B · Veri Akışı** | Donanım hatlarında tensörlerin izlediği fiziksel yollar | HBM okuma → NVLink kasa içi → Rail-IB ağ kumaşı → GDS depolama | 3. ve 4. bölümler |
| **C · Donanım Sınırları** | Çip içi ve ağ üzeri bant genişliği dengesi | Kritik aritmetik yoğunluk $\alpha_{\text{crit}} \approx 295$ FLOP/bayt | 2. ve 3. bölümler |
| **D · Hız ve Gecikme** | 140 GB verinin bağlantı türlerine göre aktarım süreleri | HBM'de 42 ms iken PCIe'de 2,2 s, InfiniBand'de 2,8 s | 5. bölüm |
| **E · Üretim Ölçeği** | 16.384 GPU'lu devasa bir veri merkezi altyapısı | 51,2 TB/s omurga bant genişliği, 1,39 MW güç altyapısı | 3. ve 5. bölümler |

A bölümü dört basamakta uzaklaşır ve her basamak bir öncekinin düzenli bir modülüdür: 132 SM bir H100 çipini, 8 H100 bir DGX düğümünü, 32 düğüm bir ölçeklenebilir birimi (SU), 4 SU ise 127 düğümlük bir SuperPOD'u oluşturur. Ardından gelen yapı taşları ise kümeyi ayakta tutan donanım katmanlarını ve 1,39 MW'lık güç altyapısını belgeler.

> [!TIP]
> **Serinin 2. Bölümü: Sistem Teşhisi**
> Bu makale donanımın fiziksel anatomisini öğretir. Bu yapının üretimde neden beklenen hızın gerisinde kaldığını, MFU kayıplarını ve küme arızalarını operasyonel bir teşhis masasında incelemek için serinin ikinci yazısına geçebilirsiniz: [Yapay Zekâ Altyapısı (2): Bir AI Süper Bilgisayarının Teşhisi — 1.024 H100 Neden %38 MFU'da Kalır?](post.html?slug=gpudan-ai-super-bilgisayarina).

---

## 2. 1. Sütun — Hesaplama: SM'in ve H100 çipinin içi

Bu bölüm A'daki 1. ve 2. basamakların ve C'deki bellek duvarının lejantıdır: bir SM tek döngüde ne yapar, neden 64 warp'ı aynı anda havada tutar ve neden yaklaşık 300 token'lık bir batch bekleyen bir GPU'yu çalışan bir GPU'dan ayırır.

### Silikon seviyesinde GPU anatomisi: SM, warp scheduler ve tensor core

GPU'yu "daha güçlü bir işlemci" diye tanımlamak işin özünü kaçırır. Bir CPU silikonunu dallanması bol, düzensiz ve gecikmeye duyarlı komut akışları için yapılmış az sayıda gelişmiş, düşük gecikmeli çekirdeğe harcar. Bir GPU ise silikonunun büyük kısmını yürütme birimlerine ayırır ki çok sayıda benzer işlemi aynı anda koşturabilsin. Bu takas yoğun matris çarpımında kazandırır; iş çok küçük ya da düzensiz olduğunda ise koordinasyon yükü paralelliği yediği için kazandırmaz.

GPU'ların TPU'lardan neden bu kadar farklı davrandığını kavramak için bir restoran mutfağı analojisi kuralım. Bir Google TPU v5p, sadece iki devasa baş aşçı (TensorCore birimleri) tarafından yönetilen lüks bir mutfak gibidir. Her baş aşçı, 128x128 matrisleri lockstep içinde tek hamlede işleyebilen devasa kesme tahtalarına (Matrix Multiply Unit / MXU) sahiptir; küçük siparişler veya asimetrik görevler geldiğinde bu dev tahtalar boş (idle) kalır. Modern bir NVIDIA GPU (örneğin Hopper H100 veya Blackwell B200) ise 132 ila 148 adet bağımsız hat aşçısından oluşan bir ordudur. Bu aşçıların her birine **Streaming Multiprocessor (SM)** denir. Her hat aşçısı kendi küçük tezgâhında tamamen bağımsız siparişler hazırlar ve hepsi ortak bir HBM kilerine bağlıdır.

Fiziksel donanımda her bir SM, **SM subpartition (alt bölüm, SMSP)** adı verilen 4 özdeş çeyreğe ayrılmıştır. Her çeyrekte posterin 1. basamağında 1–4 diye numaralanan şu bileşenler bulunur:
1. **Bir Tensor Core (TC):** bir matris çarp-topla (multiply-accumulate) motoru. Tensor Core yükünde 1,83 GHz saat hızında çalışan bir H100 SXM5 üzerinde her Tensor Core döngü başına 1.024 yoğun BF16 kayan nokta işlemi (FLOP) yapar; bu, 8×8×8'lik bir matris çarpımının işidir. FP8'de bu sayı iki katına (2.048) çıkar. Blackwell B200'de her Tensor Core döngü başına 2.048 BF16 FLOP yaparak yoğun BF16'da 2.250 TFLOP/s seviyesine ulaşır.
2. **Bir Warp Scheduler ve Dağıtım Birimi (Dispatch Unit):** **Warp** adı verilen 32 thread'lik kümelere her döngüde bir komut dağıtır ve bunu yerleşik 16 warp'tan biri arasından seçerek yapar. Dört scheduler sayesinde tek bir SM aynı anda 64 adede kadar resident (yerleşik) warp barındırır; bellek erişim gecikmelerini gizlemek (latency hiding) için her döngüde warp'lar arasında sıfır ek yüklü (zero-overhead) geçiş yapar.
3. **32 FP32 CUDA Core (ALU)** ile 16 INT32 ve 16 FP64 birimi: vektör aktivasyonlarını (GELU, SiLU), normalizasyonları (RMSNorm) ve element-wise toplamaları yürüten genel amaçlı aritmetik birimler. 132 SM'e sahip bir H100 çipinde ~1,98 GHz boost saat hızında çalışan 16.896 adet CUDA çekirdeği bulunur.
4. **16.384 Adet 32-bit Register:** subpartition başına 64 KiB (SM başına 256 KiB, çip genelinde ~33 MB) ultra hızlı register file. Her thread'in değişkenleri burada yaşar ve her komut operandlarını buradan okur.

Dört alt bölüm 256 KiB'lık ortak bir L1 veri önbelleği ve paylaşımlı bellek (SMEM) alanını paylaşır; bunun en çok 228 KiB'ı SMEM olarak ayrılabilir. **Tensor Memory Accelerator (TMA)** ise L2'den veya HBM'den SMEM'e tile'ları asenkron olarak kopyalar. Döngü başına işi çarpınca manşet sayı ortaya çıkar:

$$C_{\text{peak}} = N_{\text{TC}} \times \text{FLOP/döngü} \times f = 528 \times 1.024 \times 1{,}83 \times 10^{9} \approx 989 \times 10^{12} \text{ FLOP/s}$$

Şöyle okuyun: $N_{\text{TC}} = 132 \times 4 = 528$ Tensor Core'un her biri döngü başına 1.024 BF16 FLOP yapar ve saat hızı $f$ 1,83 GHz'tir. Bu yazıda her yerde kullanılan "990 TFLOP/s" budur; FP8 bunu ~1.979 TFLOP/s'ye ikiye katlar.

Bu sayılar SXM modülüne aittir; H100'ün PCIe sürümünde 114 SM vardır. Bir SM'in içindeki dört parça türünün işleri birbirinden net biçimde ayrıdır:

| Birim | Temel görevi | Mutfak modelindeki karşılığı |
| :--- | :--- | :--- |
| **Warp scheduler** | operandları hazır olan bir warp'ın sıradaki komutunu verir | aşçının önündeki sipariş rafı |
| **CUDA core / ALU** | toplama, çarpma, element-wise işlemler ve genel aritmetik | genel çalışma tezgâhı |
| **Tensor Core** | küçük tile'lar üzerinde birleşik matris çarp-topla, $D = A \times B + C$ | bıçak ve ocak |
| **Register / paylaşımlı bellek** | kullanımdaki operandlar, en düşük gecikmeyle | tezgâh |

Bir Transformer'ın doğrusal projeksiyonları ve ileri beslemeli katmanları yoğun matris çarpımlarıdır; 990 TFLOP/s'nin neredeyse tamamı bu yüzden Tensor Core'lardan gelir. CUDA core ile Tensor Core farklı fiziksel birimlerdir; `nvidia-smi`'nin %99 GPU kullanımı göstermesinin Tensor Core'ların verimli kullanılıp kullanılmadığı, hatta hiç kullanılıp kullanılmadığı hakkında bir şey söylememesinin nedeni budur.

Modern bir GPU'nun genel çip yerleşimi ve tek bir SM'in iç anatomisi aşağıdaki mimari diyagramda gösterilmektedir:

<svg viewBox="0 0 560 440" role="img" aria-label="NVIDIA GPU Streaming Multiprocessor mimarisi, 4 alt bölme, Tensor Core, Warp Scheduler ve bellek hiyerarşisi diyagramı." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sm-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sm-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="sm-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">GPU SM Mimarisi ve Bellek Hiyerarşisi</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hopper (H100) &amp; Blackwell (B200)</text>
<rect x="16" y="48" width="140" height="240" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU Silikon Çipi</text>
<text x="86" y="80" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">H100: 132 SM</text>
<text x="86" y="94" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">B200: 148 SM</text>
<rect x="26" y="104" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="43" y="117" text-anchor="middle" style="fill:var(--c-accent);font-size:9px;font-family:var(--font-mono)">SM0</text>
<rect x="66" y="104" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM1</text>
<rect x="106" y="104" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM2</text>
<rect x="26" y="128" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="43" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM3</text>
<rect x="66" y="128" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM4</text>
<rect x="106" y="128" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM5</text>
<rect x="26" y="152" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="43" y="165" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM6</text>
<rect x="66" y="152" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="165" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM7</text>
<rect x="106" y="152" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="165" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM8</text>
<rect x="26" y="176" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="43" y="189" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM9</text>
<rect x="66" y="176" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="83" y="189" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM10</text>
<rect x="106" y="176" width="34" height="18" rx="3" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="123" y="189" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9px;font-family:var(--font-mono)">SM11</text>
<text x="86" y="218" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Bağımsız SM Birimleri</text>
<text x="86" y="234" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">H100: 990 TFLOP/s BF16</text>
<text x="86" y="250" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">B200: 2250 TFLOP/s BF16</text>
<text x="86" y="268" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px;font-weight:600">GPU: 132+ SM (Esnek)</text>
<line x1="156" y1="112" x2="174" y2="112" marker-end="url(#sm-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="176" y="48" width="368" height="240" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Streaming Multiprocessor (SM) Anatomisi</text>
<text x="360" y="80" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 Özdeş Alt Bölme · 256 KiB Register Dosyası</text>
<rect x="184" y="88" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="270.0" y="103" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Alt Bölme 0</text>
<rect x="190" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="228.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="274" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="312.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="190" y="131" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="228.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="274" y="131" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="312.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Register</text>
<rect x="364" y="88" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="450.0" y="103" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Alt Bölme 1</text>
<rect x="370" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="408.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="454" y="110" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="492.0" y="122.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="370" y="131" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="408.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="454" y="131" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="492.0" y="143.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Register</text>
<rect x="184" y="158" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="270.0" y="173" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Alt Bölme 2</text>
<rect x="190" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="228.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="274" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="312.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="190" y="201" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="228.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="274" y="201" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="312.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Register</text>
<rect x="364" y="158" width="172" height="66" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="450.0" y="173" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Alt Bölme 3</text>
<rect x="370" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="408.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="454" y="180" width="76" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="492.0" y="192.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">Warp Sched</text>
<rect x="370" y="201" width="76" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="408.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">32x CUDA Core</text>
<rect x="454" y="201" width="76" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="492.0" y="213.2" text-anchor="middle" style="fill:var(--c-text);font-size:9px;font-family:var(--font-mono)">16k Register</text>
<rect x="184" y="230" width="172" height="48" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="270" y="248" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">SMEM / L1 Cache (256 KiB)</text>
<text x="270" y="266" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Hızlı paylaşımlı çip içi SRAM</text>
<rect x="364" y="230" width="172" height="48" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="450" y="248" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">TMEM (256 KiB Blackwell)</text>
<text x="450" y="266" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Tensor Memory matris deposu</text>
<line x1="270" y1="280" x2="270" y2="304" marker-end="url(#sm-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="450" y1="280" x2="450" y2="304" marker-end="url(#sm-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="306" width="528" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280" y="326" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Paylaşımlı L2 Cache (~50 MB Hopper / 126 MB Blackwell)</text>
<text x="280" y="344" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Tüm SM&#x27;ler arası ortak bağlantı · ~5,5 TB/s çift yönlü bant genişliği</text>
<line x1="180" y1="358" x2="180" y2="380" marker-end="url(#sm-arr-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="380" y1="358" x2="380" y2="380" marker-end="url(#sm-arr-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<rect x="16" y="382" width="528" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280" y="402" text-anchor="middle" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">HBM3 / HBM3e Yüksek Bant Genişlikli Bellek</text>
<text x="280" y="418" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB @ 3,35 TB/s (H100) · 141 GB @ 4,8 TB/s (H200) · 192 GB @ 8,0 TB/s (B200)</text>
</svg>

### SIMT yürütme dinamikleri: Thread, warp ve blok hiyerarşisi

TPU'ların saf SIMD (Single Instruction, Multiple Data) yürütme modelinin aksine, NVIDIA GPU'ları **SIMT (Single Instruction, Multiple Threads)** paradigmasıyla çalışır. Bir warp içindeki 32 CUDA core belirli bir saat döngüsünde aynı komutu yürütse de, her thread kendi program counter'ına ve bağımsız register alanına sahiptir.

Birimler iç içe geçer. **Thread**, bir programın en küçük mantıksal örneğidir: tek bir program yazarsınız, örneğin bir bulanıklaştırma filtresi, GPU da onun binlerce kopyasını her piksel için bir tane olacak şekilde başlatır. Thread'ler **thread block**'lar hâlinde gruplanır ve donanım her bloğu birlikte zamanlanan 32 thread'lik **warp**'lara keser. Bir blok tek bir SM'de koşar; register'ları, paylaşımlı belleği ve warp yuvaları yettiği sürece bir SM aynı anda birkaç bloğu barındırabilir. SIMT, her thread'in kendi register'larını ve durumunu tutması, her komutun getirilmesi ve dağıtılmasının ise bütün warp tarafından paylaşılması demektir: aynı formülü bir tablonun 32 satırına aynı anda uygulayan 32 memur gibi.

<svg viewBox="0 0 720 392" role="img" aria-label="Üstte: 128 thread&#x27;lik bir block, 32&#x27;şer thread&#x27;lik dört warp&#x27;a (warp 0–3) bölünür. Block tek bir SM&#x27;de çalışır; SM, register, shared memory ve warp yuvası yettiği kadar birden çok block barındırabilir. Altta: warp divergence. &#x27;x 0&#x27;dan küçükse A, değilse B&#x27; kodunda aynı warp önce 16 şerit etkin, 16 şerit maskeli olarak A yolunu, sonra diğer 16 şeritle B yolunu yürütür; bu eşit maliyetli örnekte her geçişte şeritlerin yarısı boştadır, ama bu evrensel bir %50 kayıp değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="bw-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="bw-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Thread block, 32&#x27;lik warp&#x27;lara bölünür</text>
<rect x="16" y="38" width="430" height="176" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">thread block (128 thread)</text>
<text x="28" y="87" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 0</text>
<rect x="92" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="102" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="112" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="122" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="132" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="142" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="152" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="162" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="172" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="182" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="192" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="202" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="212" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="222" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="232" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="242" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="252" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="272" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="282" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="292" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="302" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="312" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="322" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="332" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="342" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="352" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="362" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="372" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="382" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="392" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="402" y="74" width="8" height="18" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="28" y="121" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 1</text>
<rect x="92" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="102" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="112" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="122" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="132" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="142" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="152" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="162" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="172" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="182" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="192" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="202" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="212" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="222" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="232" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="242" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="252" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="262" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="272" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="282" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="292" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="302" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="312" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="322" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="332" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="342" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="352" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="362" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="372" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="382" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="392" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="402" y="108" width="8" height="18" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="28" y="155" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 2</text>
<rect x="92" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="102" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="112" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="122" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="132" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="142" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="152" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="162" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="172" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="182" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="192" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="202" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="212" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="222" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="232" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="242" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="252" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="262" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="272" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="282" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="292" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="302" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="312" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="322" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="332" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="342" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="352" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="362" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="372" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="382" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="392" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<rect x="402" y="142" width="8" height="18" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="28" y="189" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Warp 3</text>
<rect x="92" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="102" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="112" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="122" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="132" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="142" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="152" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="162" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="172" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="182" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="192" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="202" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="212" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="222" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="232" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="242" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="252" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="262" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="272" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="282" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="292" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="302" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="312" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="322" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="332" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="342" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="352" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="362" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="372" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="382" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="392" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<rect x="402" y="176" width="8" height="18" rx="2" style="fill:var(--c-accent-2);fill-opacity:.55"/>
<text x="92" y="208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her kare = bir thread · warp başına 32</text>
<line x1="448" y1="126" x2="478" y2="126" marker-end="url(#bw-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="480" y="38" width="224" height="176" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="492" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">tek SM</text>
<rect x="492" y="70" width="200" height="34" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="592" y="91" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">block 0</text>
<rect x="492" y="112" width="200" height="34" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="592" y="133" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">block 1</text>
<rect x="492" y="154" width="200" height="34" rx="5" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="592" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+ kaynak yettikçe fazlası</text>
<text x="592" y="204" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">register · shared memory · warp yuvası</text>
<text x="16" y="252" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Warp divergence: tek warp, iki yol, sırayla</text>
<text x="16" y="276" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">if (x &lt; 0) { A } else { B }</text>
<text x="16" y="307" text-anchor="start" style="fill:var(--c-text);font-size:12px">1. geçiş: if → A</text>
<rect x="176" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="188" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="200" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="212" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="224" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="236" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="248" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="260" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="272" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="284" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="296" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="308" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="320" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="332" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="344" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="356" y="293" width="10" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="368" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="380" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="392" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="404" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="416" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="428" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="440" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="452" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="464" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="476" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="488" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="500" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="512" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="524" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="536" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="548" y="293" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<text x="16" y="341" text-anchor="start" style="fill:var(--c-text);font-size:12px">2. geçiş: else → B</text>
<rect x="176" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="188" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="200" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="212" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="224" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="236" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="248" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="260" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="272" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="284" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="296" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="308" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="320" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="332" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="344" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="356" y="327" width="10" height="20" rx="2" style="fill:var(--c-text-mute);fill-opacity:.15"/>
<rect x="368" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="380" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="392" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="404" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="416" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="428" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="440" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="452" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="464" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="476" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="488" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="500" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="512" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="524" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="536" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="548" y="327" width="10" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<text x="176" y="362" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">her geçişte 16 şerit maskeli → bu eşit maliyetli örnekte şeritlerin yarısı boşta</text>
<text x="176" y="378" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(evrensel %50 kayıp değil: dal maliyetine ve zamanlamaya bağlı)</text>
</svg>

Bu esneklik, donanımda **warp divergence (sapma)** adı verilen tehlikeli bir başarım cezası doğurur. Kernel koşullu bir dallanma içeriyorsa (örneğin MoE expert routing kararını kontrol eden bir `if/else` veya dinamik sequence padding maskesi), dallara ayrılan thread'ler bağımsızca ilerleyemez. Warp scheduler önce `if` yolunu yürütürken diğer execution lane'leri maskeleyip durdurur (stall eder); ardından `else` yolunu yürütürken ilk grubu bekletir:

32 thread'in 16'sı A dalına, 16'sı B dalına giderse ve iki dalın maliyeti eşitse, her geçişte şeritlerin yarısı boş kalır ve o warp'ın verimi yarıya iner. Bu %50 eşit maliyetli duruma aittir, evrensel bir kural değildir: gerçek ceza her dalın uzunluğuna, komutların ne kadar bağımsız iş içerdiğine ve derleyicinin onları nasıl zamanladığına bağlıdır.

### Gecikme gizleme ve doluluk (occupancy): bir SM neden 64 warp tutar?

Sapma tek bir warp'ın içindeki şeritleri boşa harcar; asıl büyük kayıp ise scheduler'ın verecek hiçbir warp'ı kalmamasıdır. Mutfak modelinde kilerden domates isteyen aşçı elini kolunu bağlayıp beklemez, başka bir siparişin soğanını doğrar. Siparişler warp'lardır, aşçının önündeki tezgâh register file ve SMEM'dir, kiler ise HBM'dir. Hopper üzerinde ölçüldüğünde tek bir yükleme SMEM'den yaklaşık 29, L1'den 41, L2'den 263, HBM'den ise 479 döngü sürer; yani HBM'e bir gidiş, tek bir Tensor Core'un ~490 bin FLOP'luk işine denk gelen zamandır. Yerleşik her warp'ın register'ları zaten çipin üzerinde durduğu için başka bir warp'a geçmenin bedeli yoktur; bir CPU'daki bağlam değişiminde olduğu gibi kaydedilip geri yüklenecek bir durum kalmaz.

Scheduler'ın hiç boş kalmaması için kaç warp gerekir? Her warp $t_c$ döngü hesaplayıp ardından bellek için $t_w$ döngü bekliyorsa, bu bekleyişi diğer warp'lar doldurmalıdır:

$$N_{\text{warp}} \approx 1 + \frac{t_w}{t_c}$$

Şöyle okuyun: şu anda hesaplayan bir warp ve o warp'ın bekleme süresini kendi hesaplama süresi cinsinden dolduracak kadar warp. Posterdeki oyuncak zaman çizelgesinde her warp 2 birim hesaplayıp 6 birim bekler; $1 + 6/2 = 4$ warp scheduler'ı sürekli meşgul tutarken 2 warp onu zamanın yarısında boşta bırakır.

Warp havuzunu 4. maddedeki register file sınırlar: SM başına 65.536 register, yerleşik bütün thread'ler arasında paylaşılır. Thread başına $R$ register isteyen bir kernel için sığan warp sayısı şudur:

$$\text{SM başına warp} = \min\left(64, \frac{65.536}{32 \cdot R}\right)$$

```text
register/thread    SM başına warp        doluluk
  32               65.536 / 1.024 = 64   %100
  64               65.536 / 2.048 = 32   %50
 128               65.536 / 4.096 = 16   %25
 255 → 256         65.536 / 8.192 =  8   %12,5
```

Profil araçlarındaki doluluk (occupancy) tam olarak bu orandır. Yine de az warp her zaman kötü değildir. Hopper'ın en hızlı GEMM ve FlashAttention-3 kernel'leri bilerek az sayıda ama çok register'lı warp çalıştırır ve gecikmeyi başka bir yolla gizler: üretici (producer) warp'lar TMA kopyalarını HBM'den SMEM'e önceden başlatırken tüketici (consumer) warp'lar Tensor Core'larda `wgmma` ile hesaplar. Her iki yolda da kural aynıdır: bir iş bellek beklerken bir başkası çalışmaya hazır olmalıdır.

Bu noktada üç fikir sık sık birbirine karışır. **Eşzamanlılık (concurrency)**, aynı zaman aralığında birçok işin sürmekte olmasıdır. **Paralel yürütme**, farklı donanım motorlarının, örneğin birkaç SM'in ya da Tensor Core'un, tam olarak aynı döngüde çalışmasıdır. **Gecikme gizleme** ise bir scheduler'ın diğerleri belleği beklerken hazır warp'ları araya sokmasıdır. Yüksek doluluk yalnızca scheduler'ın seçebileceği çok aday olduğunu söyler; Tensor Core'ların yararlı FLOP ürettiğini söylemez.

Tablodaki warp sayıları yalnızca register bütçesinden çıkan üst sınırlardır. Blok başına paylaşımlı bellek, blok boyutu ve ayırma tanecikliği onları daha da düşürür; profil aracının bildirdiği doluluğun genellikle daha düşük olmasının nedeni budur ve bu sınırlar doğrusal olmayan biçimde etkileşir. 256 thread'lik bloklarla SM başına yalnızca bir blok sığdırabilen bir kernel, 128 thread'lik bloklarla daha çok etkin warp tutabilir; ama bu daha hızlı bir kernel garantisi değildir, çünkü blok içindeki veri yeniden kullanımı düşebilir ve zamanlama çekişmesi artabilir. Register taşmalarını (spill), paylaşımlı belleği, sapmayı ve bellek erişim desenlerini önce profillemeden "daha çok thread" başlatmak bir kernel'i çoğu zaman hızlandırmaz, yavaşlatır.

### H100 çipi: 132 SM, L2 önbellek ve HBM3 yığınları

Posterin 2. basamağı SXM5 paketinin haritasını çıkarır. GH100 kalıbı (die) TSMC 4N üzerinde 80 milyar transistörlü, 814 mm²'lik bir silikondur ve her biri 2'şer SM'li 9 TPC'den oluşan 8 GPC olarak kurulmuştur: toplam 144 SM yuvası vardır, H100 SXM5 bunların 132'sini etkinleştirir. Bu kadar büyük bir kalıp nadiren kusursuz çıkar; birkaç birimi kapatmak verimi (yield) artırır ve hangi altı TPC'nin kapalı olduğu çipten çipe değişir. 50 MB'lık L2 iki bölüme ayrılmıştır ve her yandaki GPC'ler bunlardan birine daha yakındır. Paketin altı HBM3 yuvasından beşi etkindir; her biri iki adet 512-bit bellek denetleyicisinin arkasında duran 16 GB'lık bir yığındır. Bant genişliği veri yolu genişliğinden çıkar:

$$W_{\text{HBM}} = \frac{5 \times 1.024 \text{ bit} \times 5{,}23 \text{ Gb/s}}{8 \text{ bit/bayt}} \approx 3{,}35 \text{ TB/s}$$

Şöyle okuyun: her biri 1.024-bit arayüzlü beş yığın 5.120-bit'lik bir veri yolu verir; pin başına ~5,23 Gb/s ile bu yol saniyede 3,35 TB taşır, yığın başına yaklaşık 670 GB/s. Tam GH100'de 144 SM, 60 MB L2 ve altı yığın bulunur; SXM5 modülünün güç sınırı 700 W'tır.

**Bellek hiyerarşisi ve aritmetik yoğunluk dengesi:**

Hesaplama gücü, işlem birimlerini besleyecek bellek bant genişliği olmadan bir hiçtir. GPU bellek alt sistemi üstü küçük ve hızlı, altı büyük ve yavaş olan dik bir piramit oluşturur:
- **Register File:** SM başına 256 KiB (çip genelinde ~33 MB). Operandları komut doğrudan okur; ayrı bir yükleme gerekmez.
- **Shared Memory (SMEM) / L1 Cache:** SM başına 256 KiB (çip genelinde ~33 MB). FlashAttention algoritmalarında attention tile'larını önbelleğe almak için programatik olarak yönetilen SRAM; SMEM gecikmesi ~29 döngü (yaklaşık 16 ns), L1 gecikmesi ~41 döngüdür.
- **Tensor Memory (TMEM):** Blackwell B200 ile tanıtılan SM başına 256 KiB on-chip bellek. B200 Tensor Core birimlerinin ürettiği matris çarpım akümülatörleri klasik register file kapasitesini aştığı için çip üstünde dedicated matris accumulator deposu sağlar.
- **Shared L2 Cache:** Hopper H100'de 50 MB (~263 döngü, yaklaşık 5,5 TB/s), Blackwell B200'de 126 MB.
- **High-Bandwidth Memory (HBM3 / HBM3e):** H100'de 80 GB (~479 döngü, 3,35 TB/s), H200'de 141 GB (4,8 TB/s), B200'de 192 GB (8,0 TB/s).

L2'den HBM'e inerken kapasite 1.600 kat büyür, gecikme ise neredeyse ikiye katlanır. SM'lerin içindeki SRAM (~33 MB register ile ~33 MB SMEM/L1) toplamda L2'den bile büyüktür; iyi kernel'lerin çalışma kümesini bu üst katmanlarda tutup HBM'e olabildiğince seyrek gitmesinin nedeni budur.

"Hızlı" kelimesinin içinde iki farklı soru saklıdır. **Bant genişliği (bandwidth)** bir yolun saniyede kaç bayt taşıdığıdır; **gecikme (latency)** ise bir isteğin ilk baytının ne kadar sürede vardığıdır. 500 km uzunluğunda sekiz şeritli bir otoyolun hem bant genişliği hem gecikmesi yüksektir. Büyük ve bitişik tensörleri akıtmayı bant genişliği, işaretçi kovalamayı (pointer chasing), düzensiz aramaları ve ince taneli bağımlılıkları ise gecikme yönetir. Fiziksel uzaklık işin yalnızca bir parçasıdır: HBM kalıpla aynı paketin üzerindedir, ama bir warp için register'daki bir operand ile HBM'deki bir operand bambaşka dünyalardır; çünkü protokol yükü, hakemlik (arbitration), bank çakışmaları, önbellek isabetleri ve aktarım genişliği "yakın"ın bedeline eklenir.

Bant genişliğini erişim deseniyle hak etmek de gerekir. **Bellek birleştirme (memory coalescing)**, bir warp'ın 32 thread'inin bitişik ve hizalı adresler istemesi, böylece bellek denetleyicisinin bütün warp'a birkaç geniş, 32 ya da 128 baytlık işlemle hizmet edebilmesidir; dağınık ya da hizasız erişimler bus döngülerini kimsenin kullanmadığı baytlara harcar. Tepe değer de sürdürülen değer değildir: ondalık GB ikili GiB değildir ve hiçbir kernel 3,35 TB/s'nin tamamını akıtmaz.

Hesaplama kapasitesi ile bellek bant genişliği arasındaki denge **Roofline Modeli** ile tanımlanır:

$$\alpha_{\text{crit}} = \frac{C_{\text{peak}}}{W_{\text{mem}}}$$

Şöyle okuyun: $\alpha_{\text{crit}}$ bayt başına FLOP cinsinden kritik aritmetik yoğunluk (arithmetic intensity), $C_{\text{peak}}$ saniyedeki tepe kayan nokta işlem sayısı (peak FLOP/s), $W_{\text{mem}}$ ise saniyedeki bellek bant genişliğidir (bytes/s). Yoğunluğu $\alpha$ olan bir kernel $\min(C_{\text{peak}}, W_{\text{mem}} \cdot \alpha)$ hızında koşar; posterin 2. basamağında çizilen çatı budur.

BF16 Tensor Core çalıştıran bir H100 SXM5 için:
$$C_{\text{peak}} = 990 \times 10^{12} \text{ FLOP/s}, \quad W_{\text{mem}} = 3,35 \times 10^{12} \text{ B/s}$$
$$\alpha_{\text{crit}} = \frac{990 \times 10^{12}}{3,35 \times 10^{12}} \approx 295,5 \text{ FLOP/bayt}$$

Bu sayı, yoğun BF16 Tensor Core matematiğinin HBM'e karşı makine dengesidir; her kernel'in bayt başına 295 FLOP yapması gerektiği anlamına gelmez. Veri tipini değiştirirseniz dirsek tepe değerle birlikte kayar: ~1.979 TFLOP/s'lik FP8 Tensor Core'lar onu $1.979 / 3{,}35 \approx 591$ FLOP/bayt'a taşır, CUDA çekirdeklerindeki FP32 ise çok daha aşağıya çeker.

$X \in \mathbb{R}^{B \times D}$ aktivasyon matrisi ile $W \in \mathbb{R}^{D \times F}$ ağırlık matrisinin 16-bit duyarlıkta çarpımı için:
- **Toplam Matematiksel İşlem:** $2 B D F \text{ FLOP}$
- **Toplam Bellek Transferi:** $2(BD + DF + BF) \text{ bayt}$
- **Operasyonel Yoğunluk (Operational Intensity):**

$$\alpha(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

Batch boyutu $B$ çok küçük olduğunda ($B \to 1$, örneğin tek tokenli ardışık çıkarım / autoregressive decoding), ağırlık matrisi bellek trafiğini domine eder ($DF \gg B(D+F)$):
$$\alpha(1) \approx \frac{D F}{D F} \cdot B \approx B = 1 \text{ FLOP/bayt} \ll 295,5 \text{ FLOP/bayt}$$

Bu rejimde GPU aşırı derecede **memory-bandwidth bound** (bellek bant genişliği kısıtlı) çalışır ve tepe hesaplama kapasitesinin %1'inden azını kullanabilir. **Compute-bound** (hesaplama kısıtlı) sınırına geçmek için gereken batch boyutu şartı şudur:

$$B \ge \frac{\alpha_{\text{crit}} \cdot D F}{D F - \alpha_{\text{crit}}(D + F)} \approx \alpha_{\text{crit}} \approx 295 \text{ token}$$

Posterdeki roofline ve bellek duvarı eğrisi Llama 3 70B'nin MLP boyutlarını kullanır: $D = 8.192$ ve $F = 28.672$. Batch'i adım adım büyütmek üç rejimi gösterir:

```text
B = 1      α = 8.192·28.672 / (1·36.864 + 234,9M)     ≈ 1,0     → 3,35 × 1,0   ≈   3,3 TFLOP/s  (%0,3)
B = 8      α = 1,88G / (0,29M + 234,9M)               ≈ 8,0     → 3,35 × 8,0   ≈  26,8 TFLOP/s  (%2,7)
B = 32     α = 7,52G / (1,18M + 234,9M)               ≈ 31,8    → 3,35 × 31,8  ≈ 106,7 TFLOP/s  (%11)
B = 128    α = 30,06G / (4,72M + 234,9M)              ≈ 125,5   → 3,35 × 125,5 ≈ 420   TFLOP/s  (%42)
B = 310    α = 72,8G / (11,4M + 234,9M)               ≈ 295,6   → dirsekte         ≈ 990   TFLOP/s
B = 512    α = 120,3G / (18,9M + 234,9M)              ≈ 473,9   → çatının altında  = 990   TFLOP/s
B = 4.096  α = 962G / (151M + 234,9M)                 ≈ 2.493   → çatıda           = 990   TFLOP/s
```

Hem Hopper H100 hem de Blackwell B200'de roofline eğrisinin dirsek noktasını (roofline knee) aşıp compute-bound bölgesine geçebilmek için bir matris çarpımının kabaca 280 ila 300 token'lık bir batch ile çalışması gerekir; bu Llama 3 70B boyutlarında ~310'dur. Karşılaştırma için, BF16'da element-wise bir toplama 6 bayt başına 1 FLOP yapar ($\alpha \approx 0{,}17$); bu tür işlemlerin tek başına koşturulmak yerine komşu kernel'lerle birleştirilmesinin (fusion) nedeni budur.

Bu noktalar profillenmiş kernel'lerden değil, idealleştirilmiş bir veri taşıma modelinden gelir: tiling, L2'de kalma, register yeniden kullanımı, aktivasyon yerleşimi ve nicemleme gerçek kırılma noktalarını kaydırır. Model yine de iş yüklerini üç aileye ayırmaya yeter:

| İş yükü | Ağırlık ya da veri yeniden kullanımı | Beklenen darboğaz |
| :--- | :--- | :--- |
| Büyük matris-vektör çarpımı (GEMV, $B = 1$) | sıfıra yakın | HBM bant genişliği |
| Batch'li matris-matris çarpımı (GEMM, büyük $B$) | yüksek | Tensor Core'lar (hesaplama) |
| Arka arkaya element-wise kernel'ler | düşük, parçalı | kernel başlatmaları ve tekrarlanan HBM gidiş-dönüşleri |

Daha hızlı HBM üçüne de yardım eder, ama en büyük kazanç veriyi çip üstü SRAM'e bir kez getirip geri yazmadan önce defalarca kullanan algoritmalardan gelir; kernel fusion ve FlashAttention'ın arkasındaki fikir budur.

### Düşük duyarlıklı sayısal formatlar: FP32'den FP8 ve NVFP4'e

Bellek baskısını hafifletmek ve aritmetik işlem hızını katlamak için modern GPU mimarileri sayısal duyarlığı kademeli olarak düşürmüştür. FP32'den (32 bit: 1 işaret, 8 üs, 23 mantis) 16-bit formatlara (FP16 ve BF16) geçiş, bellek ayak izini yarıya indirip FLOP oranlarını ikiye katlamıştır.

Mimari nesilleri boyunca Tensor Core hesaplama motorlarının evrimi ve donanım sınırları aşağıda özetlenmiştir. Son sütun roofline formülünü her nesle uygular ve 1. bölümdeki bellek duvarını açıklar: hesaplama bant genişliğinden hızlı büyüdüğü için dirsek yukarı kaymıştır.

| Mimari Nesli | Hesaplama Motoru | HBM Bellek & Bant Genişliği | BF16 / FP16 TFLOP/s | FP8 / INT8 TFLOP/s | FP4 TFLOP/s | Dirsek $\alpha_{\text{crit}}$ (16-bit) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Volta (V100)** | 1. Nesil Tensor Core | 32 GB @ 0,9 TB/s | 125 | — | — | ~139 |
| **Ampere (A100)** | 3. Nesil Tensor Core | 80 GB @ 2,0 TB/s | 312 | 624 | — | ~156 |
| **Hopper (H100)** | 4. Nesil Tensor Core | 80 GB @ 3,35 TB/s | 990 | 1.980 | — | ~295 |
| **Hopper (H200)** | 4. Nesil Tensor Core | 141 GB @ 4,8 TB/s | 990 | 1.980 | — | ~206 |
| **Blackwell (B200)** | 5. Nesil Tensor Core | 192 GB @ 8,0 TB/s | 2.250 | 4.500 | 9.000 | ~281 |

Bütün hız değerleri yoğun (2:4 seyreklik olmadan) değerlerdir. DGX B200 sistemi B200'ün HBM3e'sinin GPU başına 180 GB'ını kullanıma açar. H200, H100'ün hesaplama gücünü korurken %43 daha fazla bant genişliği ekler; aynı decode iş yükünün onda daha hızlı koşmasının nedeni budur.

Hopper H100 ile birlikte donanıma iki farklı FP8 (8-bit kayan nokta) formatı eklendi:
1. **E4M3 (1 işaret, 4 üs, 3 mantis biti):** Dinamik aralığın kararlı olduğu ileri geçiş (forward pass) aktivasyonları ve ağırlıklar için daha yüksek sayısal hassasiyet sunar.
2. **E5M2 (1 işaret, 5 üs, 2 mantis biti):** FP16'nın dinamik aralığını koruyarak geri yayılım (backward pass) gradyanlarında taşma veya sıfıra yuvarlanmayı (underflow) engeller.

| Sayısal Format | İşaret Biti | Üs Biti | Mantis Biti | Dinamik Aralık ($E_{\text{min}} \dots E_{\text{max}}$) | LLM Ölçeklemesindeki Temel Rolü |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IEEE FP16** | 1 | 5 | 10 | $\approx 6 \times 10^{-5} \dots 65.504$ | Geleneksel çıkarım ve temel işlem |
| **Google BF16** | 1 | 8 | 7 | $\approx 1 \times 10^{-38} \dots 3 \times 10^{38}$ | Kararlı temel model ön eğitimi |
| **FP8 (E4M3)** | 1 | 4 | 3 | $\approx \pm 448$ | Forward pass aktivasyonları ve ağırlıklar |
| **FP8 (E5M2)** | 1 | 5 | 2 | $\approx \pm 57.344$ | Backward pass gradyan iletimi |
| **NVFP4 (E2M1)** | 1 | 2 | 1 | Blok ölçekli | Blackwell ultra yoğun GEMM hızlandırması |

Blackwell B200 bir adım daha atarak blok ölçekli 4-bit bir format olan **NVFP4**'ü getirdi. Her eleman bir E2M1 sayısıdır; her 16 elemanlık blok tek bir FP8 (E4M3) ölçek katsayısını paylaşır, en üstte de tensör başına bir FP32 ölçek durur. Küçük bloklar ölçeğin yerel aykırı değerleri izlemesini sağlar; 4 biti kullanılabilir kılan da budur. B200 bu formatta yoğun 9.000 TFLOP/s'ye ulaşır: FP8 hızının iki, BF16 hızının dört katı. Üstelik her ağırlık BF16'daki baytların dörtte birini kaplar; bu da hızı FLOP'ların değil baytların belirlediği decode evresine ayrıca yardım eder.

---

## 3. 2. Sütun — Ağ: NVLink kasasından SuperPOD omurgasına

Bu bölüm A'daki 3., 4. ve 4b basamaklarının, B bölümünün ve C'deki ağ duvarının lejantıdır. Tek fikri bant genişliği merdivenidir: GPU başına HBM'de 3,35 TB/s, NVLink'te 450 GB/s, ağda 50 GB/s. Aşağıdaki her paralellik kararı aslında bir trafik parçasının bu merdivenin hangi basamağını kullanmasına izin verildiğine dair bir karardır.

### Intra-node ağ mimarisi: NVLink ve NVSwitch fabric'leri

Tek bir çipin sınırlarını aşarken standart PCIe veri yolları (PCIe Gen 5'te tek yönde yalnızca 64 GB/s) aşılması imkânsız bir darboğaza dönüşür. NVIDIA bu engeli **NVLink** ve **NVSwitch** ile aşar.

Standart bir DGX H100 veya HGX H100 kasasında 8 GPU, HGX taban kartı üzerindeki 4 adet özel NVSwitch 4 ASIC'i ile eşleştirilmiştir. Her H100 çipi 18 adet NVLink 4 portuna sahiptir. Her port çift yönde 50 GB/s (tek yönde 25 GB/s) aktarım yaparak çip başına şu egress hızını üretir:
$$W_{\text{GPU egress}} = 18 \times 25 \text{ GB/s} = 450 \text{ GB/s tek yönlü } (900 \text{ GB/s çift yönlü})$$

Her NVSwitch çipinde 64 NVLink portu vardır; yani her biri yön başına $64 \times 25 = 1{,}6$ TB/s (iki yönde 3,2 TB/s) taşır. 4 NVSwitch birlikte non-blocking (engellemesiz) bir crossbar fabric oluşturur ve kasa içindeki toplam anahtarlama kapasitesi şöyledir:
$$4 \text{ switch} \times 1,6 \text{ TB/s} = 6,4 \text{ TB/s anahtarlama bant genişliği (tek yön)}$$

8 GPU'nun her biri 450 GB/s egress yapabildiğinden, kasanın 4 GPU'ya karşı 4 GPU şeklindeki herhangi bir eşit ikiye bölünüşünde (bisection):
$$\text{Bisection Bandwidth} = 4 \text{ GPU} \times 450 \text{ GB/s} \times 2 \text{ yön} = 3,6 \text{ TB/s}$$

Düğüm içi (intra-node) crossbar ağ mimarisi aşağıdaki yapısal diyagramda gösterilmektedir:

<svg viewBox="0 0 560 410" role="img" aria-label="8 GPU&#x27;lu DGX düğümünde NVSwitch çapraz bağlantı mimarisi ve bant genişlikleri." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="nv-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="nv-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="nv-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Düğüm İçi NVLink &amp; NVSwitch Mimarisi</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8x GPU NVLink Etki Alanı (Hopper / Blackwell)</text>
<rect x="16" y="48" width="528" height="276" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="66" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX H100 / B200 Düğümü (Node)</text>
<text x="532" y="66" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">Düğüm Biseksiyonu: 3,6 TB/s</text>
<rect x="32" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="81.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 0</text>
<text x="81.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="160" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="209.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 1</text>
<text x="209.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="288" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="337.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 2</text>
<text x="337.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="416" y="76" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="97" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 3</text>
<text x="465.0" y="114" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="32" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 0</text>
<text x="81.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1,6 TB/s Tek Yön</text>
<rect x="160" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="209.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 1</text>
<text x="209.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1,6 TB/s Tek Yön</text>
<rect x="288" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="337.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 2</text>
<text x="337.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1,6 TB/s Tek Yön</text>
<rect x="416" y="160" width="98" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="465.0" y="178" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">NVSwitch 3</text>
<text x="465.0" y="193" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1,6 TB/s Tek Yön</text>
<rect x="32" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="81.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 4</text>
<text x="81.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="160" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="209.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 5</text>
<text x="209.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="288" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="337.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 6</text>
<text x="337.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<rect x="416" y="232" width="98" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="253" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU 7</text>
<text x="465.0" y="270" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">450 GB/s Egress</text>
<line x1="81.0" y1="132" x2="81.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="81.0" y1="204" x2="81.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="209.0" y1="132" x2="209.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="209.0" y1="204" x2="209.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="337.0" y1="132" x2="337.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="337.0" y1="204" x2="337.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="465.0" y1="132" x2="465.0" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="465.0" y1="204" x2="465.0" y2="230" marker-end="url(#nv-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<line x1="97" y1="132" x2="195" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="225" y1="132" x2="323" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="353" y1="132" x2="451" y2="158" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="97" y1="230" x2="195" y2="204" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="225" y1="230" x2="323" y2="204" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="353" y1="230" x2="451" y2="204" marker-end="url(#nv-arr)" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="16" y="332" width="528" height="66" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="352" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">NVSwitch Crossbar: 8 GPU arasında engelsiz tam bağlantı (all-to-all).</text>
<text x="28" y="370" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her GPU diğer 7 GPU&#x27;nun belleğine doğrudan 450 GB/s tek yönlü,</text>
<text x="28" y="386" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">çift yönde 900 GB/s tam biseksiyon bant genişliğiyle erişir.</text>
</svg>

Posterdeki bağlantı matrisi 18 hattın nasıl dağıldığını sayar. Her GPU NVSwitch 0–3'e sırasıyla 5, 4, 4 ve 5 NVLink gönderir; switch'lere de 40, 32, 32 ve 40 hat ulaşır, toplamda $8 \times 18 = 144$. Her GPU'nun 450 GB/s'sinin tamamı switch'lere vardığı için sekiz GPU aynı anda tam hızla gönderebilir ve herhangi bir GPU diğerine tek bir switch sıçramasıyla ulaşır. NVSwitch toplamayı kendisi de yapabilir (**NVLink SHARP**); böylece bir AllReduce'ta veri switch'e yalnızca bir kez gider.

Blackwell nesliyle birlikte NVIDIA, **GB200 NVL72** mimarisiyle bu denklemi kökten değiştirdi. NVLink domain'ini 8 GPU'lu tek bir kasayla sınırlamak yerine, sıvı soğutmalı rack'ler 72 adet B200 GPU'yu 18 adet NVSwitch 5 ASIC'i üzerinden bakır arka panelde (copper backplane) birleştirir:

NVL72'de 72 GPU'nun tümü, herhangi bir çipin diğerinin HBM belleğine 900 GB/s hızla erişebildiği tek bir NVLink etki alanını paylaşır ve tek node sınırını 9 kat genişletir.

### DGX H100'ün içi: iki NUMA yarısı ve dört yol

NVLink, bir GPU'dan çıkan yollardan yalnızca biridir. Posterin 3. basamağı NVIDIA'nın DGX H100 sistem topolojisini yeniden çizer; numaralı altı sorusunu düzyazıyla cevaplamaya değer, çünkü her biri bir yapılandırma kuralına dönüşür.

1. **Kim kimi besler?** Düğüm birbirinin ayna görüntüsü olan iki yarıdan oluşur. CPU 0 (56 çekirdekli bir Xeon 8480C) GPU 0–3'ü ve 0–3 numaralı ağ kartlarını PCIe Gen5 üzerinden besler; CPU 1 ise GPU 4–7'yi ve 4–7 numaralı kartları. Her yarının kendi belleği, yani kendi **NUMA** alanı vardır; iki soket UPI ile birbirine bağlıdır. Bu yüzden GPU 5 için veri hazırlayan süreç CPU 1'de koşmalı ve CPU 1'in belleğini kullanmalıdır.
2. **Bir GPU ağa nasıl ulaşır?** Her GPU'nun hemen üstünde kendine ait tek portlu, 400 Gb/s'lik bir ConnectX-7 kartı durur (GPU $i$ ↔ NIC $i$). **GPUDirect RDMA** ile veri GPU belleğinden doğrudan ağ kartına gider ve CPU belleğine hiç uğramaz. Bu sekiz kart kasadan dışarı $8 \times 50 = 400$ GB/s çıkarır.
3. **GPU'lar birbirleriyle nasıl konuşur?** Yukarıda sayılan, dört NVSwitch'e giden 18 NVLink üzerinden.
4. **Depolama trafiği nereden akar?** Üstteki iki PCIe switch, her biri iki çift portlu ConnectX-7 kartını (depolama ağı ve in-band yönetim) ve dört U.2 NVMe sürücüsünü taşır; toplamda yerel bir RAID 0 önbelleği olarak kullanılan 8 × 3,84 TB. Veri okuma ve kontrol noktası yazma işleri bu sayede gradyan trafiğinden farklı kartları ve kabloları kullanır.
5. **İki yarıyı ne birleştirir?** UPI, CPU'ların kendi trafiğini taşır. Farklı yarılardaki GPU'lar onun üzerinden hiç konuşmaz; NVLink'i kullanırlar.
6. **NVSwitch'leri kim yönetir?** CPU 0'dan çıkan bir PCIe kontrol yolu üçüncü bir PCIe switch üzerinden dört NVSwitch'e ulaşır. **NVIDIA Fabric Manager** switch'lerin yönlendirme tablolarını bu yoldan kurar; üzerinden hiçbir GPU verisi akmaz. Fabric Manager çalışmıyorsa NVLink ağı hiç ayağa kalkmaz ve her GPU tek başına sağlıklı görünse bile çok GPU'lu işler başlamaz.

Hepsi bir araya geldiğinde tek bir GPU'nun dört yolu vardır ve B bölümündeki eğitim döngüsünün her adımı bunlardan birini seçer:

| Yol (GPU başına, tek yön) | Bant genişliği | Bu yolu kullanan trafik | Sıklık |
| :--- | :--- | :--- | :--- |
| NVSwitch üzerinden NVLink | 450 GB/s | Tensor paralel AllReduce | katman başına iki kez, ileri ve geri |
| Hesaplama NIC'i → leaf switch $i$ | 50 GB/s | Pipeline aktivasyonları, veri paralel gradyanlar | mikro-batch başına, adım başına |
| Depolama CX-7 → depolama ağı | paylaşımlı, düğüm başına ≥ 40 GB/s | Veri okuma, kontrol noktası yazma | adım başına, ~14 dakikada bir |
| CPU'ya PCIe Gen5 | 64 GB/s | Veri yükleyici, host kopyaları | sürekli, aynı soketten |

Sonraki alt bölümlerdeki tensor paralellik kuralı ilk satırın gerekçesini verir: NVLink ağdan dokuz kat geniştir; bu yüzden her katmanda konuşan trafik bu sekiz GPU'nun içinde kalır ve kasa duvarı aynı zamanda tensor paralellik duvarıdır.

Bu yollar hakkında iki incelik var. PCIe Gen5 ×16, 128b/130b kodlamayla 16 şeridi 32 GT/s'de çalıştırır; bu, protokol yükünden önce yön başına yaklaşık 63 GB/s eder, yani posterdeki 64 GB/s yuvarlanmış hat hızıdır. Ayrıca **GPUDirect RDMA**, **GPUDirect Storage** değildir: RDMA, düğümler arasında gradyanları ve aktivasyonları taşıyan NIC ↔ HBM yoludur; GDS ise 4. bölümde açılan depolama ↔ HBM yoludur. Depolama sistemleri altta çoğu zaman RDMA taşımaları kullanır, ama ikisi farklı alt sistemlere hizmet eder.

NVSwitch alanı aynı zamanda **scale-up**'ın bittiği yerdir. Bu alanın içinde GPU'lar sıkı bağlı, yüksek bant genişlikli tek bir alan oluşturur; onun ötesine büyümek, bağımsız düğümleri hesaplama ağı üzerinden birbirine bağlayan **scale-out** demektir. H100 sistemlerinde bu sınır 8 GPU'luk kasadır.

### Düğümler ötesi ölçekleme (Scale-out): SuperPod ve rail-optimized fat-tree topolojisi

Yüz milyarlarca parametreli modelleri eğitmek için cluster'ın binlerce GPU'ya ölçeklenmesi gerekir. Bu scale-out katmanı **InfiniBand** (veya RoCE v2 Ethernet) tarafından yönetilir.

1.016 GPU'lu bir DGX H100 SuperPod cluster'ında (127 node; 128. yuvayı bir çift UFM cihazı alır):
- Cluster, her biri 32 DGX node (256 GPU) içeren **Scalable Unit (SU)** bloklarına ayrılır.
- Her node'da 8 adet ConnectX-7 400 Gbps (NDR) NIC bulunur; bu da node başına $8 \times 50 \text{ GB/s} = 400 \text{ GB/s}$ (GPU başına 50 GB/s) scale-out network egress sağlar.
- Her SU içinde 8 Leaf switch bulunur. 4 SU genelinde 16 merkezi Spine switch, ağı 51,2 TB/s aggregate switching capacity'ye sahip non-blocking 1:1 **Fat-Tree topolojisinde** birleştirir.

**Leaf**, sunucuların takıldığı alt katman switch'tir; **spine** ise leaf'leri birbirine bağlayan ve kendisine hiçbir sunucu takılmayan üst katman switch'tir. Her leaf her spine'a bağlıdır; bu yüzden herhangi iki sunucu en fazla leaf → spine → leaf yoluyla buluşur ve küme büyüdükçe bu yol uzamaz: sunucu eklemek için leaf, bant genişliği eklemek için spine eklersiniz. Desen telefon santrallerinden gelir (Clos, 1953); bağlantılar yukarı doğru kalınlaştığında buna fat-tree denir (Leiserson, 1985).

Bir **sıçrama (hop)**, bir switch ASIC'inden bir kez geçmektir. **Biseksiyon bant genişliği (bisection bandwidth)**, ağın mümkün olan en kötü ikiye bölünüşünde iki eşit yarısı arasındaki kapasitedir; engellemesiz ya da tam biseksiyonlu bir ağ, dengeli yönlendirmede hiçbir trafik permütasyonunun yukarı bağlantılar tarafından kısılmamasını garanti eder. Pratikte akışlar arası hash çakışmaları, dolan switch tamponları ve kısa mikro patlamalar (microburst) yine de kuyruk gecikmesi ekleyebilir; trafik desenlerinin varsayılmak yerine ölçülmesinin nedeni budur.

Kritik olarak modern cluster'lar **Rail-Optimized kablolama** kullanır. Bu topolojide her node'daki $i$. GPU rank'ı doğrudan $i$. Leaf switch'e, yani $i$. rayın (rail) switch'ine bağlanır. Aynı SU'daki node'ların aynı indeksli GPU'ları arasında Data Parallelism (DP) veya Pipeline Parallelism (PP) kolektifleri yürütülürken paketler spine katmanına çıkmadan doğrudan kendi rail switch'i içinde kalır; böylece spine hop'ları ortadan kalkar ve fabric congestion önlenir. Bir düğümün sekiz NIC'inin aynı leaf'e gittiği klasik top-of-rack düzeninde A düğümünün 3. GPU'su B düğümünün 3. GPU'suna leaf → spine → leaf üzerinden üç sıçramada ulaşırdı; raylı düzende aynı mesaj tek sıçramada gider. Bunun işe yaramasının nedeni tensor paralelliğin kasa içinde kalmasıdır: her düğümün $i$. GPU'su modelin aynı dilimini tutar, dolayısıyla pipeline ve veri paralel ortakları da başka düğümlerdeki $i$. GPU'lardır.

<svg viewBox="0 0 560 440" role="img" aria-label="1.016 GPU&#x27;lu DGX H100 SuperPod Fat-Tree InfiniBand ağ topolojisi diyagramı." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sp-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="sp-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">SuperPod Fat-Tree Ölçekleme Topolojisi</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1.016 GPU · Tam Biseksiyon Bant Genişliği</text>
<rect x="16" y="48" width="528" height="62" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280" y="68" text-anchor="middle" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">16x InfiniBand Spine (Omurga) Switch</text>
<text x="280" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Quantum-2 400 Gbps NDR · 51,2 TB/s Toplam Anahtarlama Kapasitesi</text>
<text x="280" y="99" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">Her SU&#x27;ya tam biseksiyonlu 12,8 TB/s bağlantı</text>
<rect x="16" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="78.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 0</text>
<text x="78.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switch</text>
<text x="78.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12,8 TB/s Egress</text>
<rect x="150" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="212.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 1</text>
<text x="212.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switch</text>
<text x="212.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12,8 TB/s Egress</text>
<rect x="284" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="346.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 2</text>
<text x="346.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switch</text>
<text x="346.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12,8 TB/s Egress</text>
<rect x="418" y="144" width="124" height="68" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="480.0" y="162" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SU 3</text>
<text x="480.0" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x Leaf Switch</text>
<text x="480.0" y="194" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">12,8 TB/s Egress</text>
<line x1="78.0" y1="112" x2="78.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="212.0" y1="112" x2="212.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="346.0" y1="112" x2="346.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="480.0" y1="112" x2="480.0" y2="142" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="280" y="131" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">Tam Biseksiyon (1:1 Engellemesiz)</text>
<rect x="16" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="78.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Düğüm (256 Çip)</text>
<text x="78.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Her düğümde 8 GPU</text>
<text x="78.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NIC</text>
<text x="78.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<rect x="150" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="212.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Düğüm (256 Çip)</text>
<text x="212.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Her düğümde 8 GPU</text>
<text x="212.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NIC</text>
<text x="212.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<rect x="284" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="346.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Düğüm (256 Çip)</text>
<text x="346.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Her düğümde 8 GPU</text>
<text x="346.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NIC</text>
<text x="346.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<rect x="418" y="248" width="124" height="76" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="480.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">31 Düğüm (248 Çip)</text>
<text x="480.0" y="282" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Her düğümde 8 GPU</text>
<text x="480.0" y="298" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8x 400 Gbps CX7 NIC</text>
<text x="480.0" y="314" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Egress: 400 GB/s</text>
<line x1="78.0" y1="214" x2="78.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="212.0" y1="214" x2="212.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="346.0" y1="214" x2="346.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<line x1="480.0" y1="214" x2="480.0" y2="246" marker-end="url(#sp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="280" y="234" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">Düğüm Başına 400 GB/s Egress</text>
<rect x="16" y="336" width="528" height="88" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="356" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Fat-Tree: Katmanlar yükseldikçe kablo sayısı bant genişliğini korur.</text>
<text x="28" y="374" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Düğüm içinde GPU başına 450 GB/s NVLink bulunurken,</text>
<text x="28" y="390" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">düğümler arasında her düğüm için 400 GB/s InfiniBand korunur.</text>
<text x="28" y="408" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Oversubscription yoktur; kolektif iletişim tüm ağda sabit hızda akar.</text>
</svg>

**"Engellemesiz" neden tutar: tek bir leaf'in portlarını sayın.** Her leaf, 400 Gb/s'lik 64 portu olan bir Quantum-2 QM9700'dür. 32 port aşağıya, SU'sundaki 32 düğümün her birinin $i$. NIC'ine gider; 32 port ise yukarıya, 16 spine'ın her birine ikişer kabloyla çıkar. Aşağıdan giren her şey yukarıdan çıkabildiği için aşırı abonelik (oversubscription) oranı 1:1'dir. Ağacın tepesinde aynı hesap şöyledir:

$$\underbrace{16 \times 64 \times 50 \text{ GB/s}}_{\text{spine kapasitesi}} = 51{,}2 \text{ TB/s} \;\ge\; \underbrace{1.016 \times 50 \text{ GB/s}}_{\text{bütün GPU'lar gönderirken}} = 50{,}8 \text{ TB/s}$$

Şöyle okuyun: her biri 50 GB/s'lik (400 Gb/s ÷ 8) 64 portu olan 16 spine, 1.016 GPU'nun aynı anda ağa itebileceği her şeyi taşıyabilir. 32 leaf × 32 yukarı bağlantı = 1.024 kablo, 16 spine × 64 portu tam olarak doldurur. Bunu NVIDIA'nın QM9700 kılavuzundaki "51,2 Tb/s" ile karıştırmayın: o değer tek bir switch'in iki yöndeki toplamıdır ve bit cinsindendir; buradaki 51,2 TB/s ise 16 spine'ın tek yöndeki toplamıdır ve bayt cinsindendir.

**Neden 128 değil de 127 düğüm?** Kâğıt üzerinde 4 SU × 32 = 128 DGX eder. NVIDIA'nın kurulum deseninde InfiniBand ağını izleyen bir çift UFM cihazı SU 4'te bir DGX'in yerini alır; bu yüzden tam bir SuperPOD en çok 127 DGX, yani 1.016 GPU barındırır. NVIDIA'nın kablo tablosu da bunu gösterir: 127 × 8 + 4 UFM portu = düğümden leaf'e 1.020 kablo. Daha küçük kurulumlar tek bir SU'dan başlar ve 31 ile 127 DGX arasında herhangi bir boyutta olabilir.

**Dört yol, dört fiyat.** Raylı düzen bazı trafikleri ucuz, bazılarını pahalı yapar; posterin 4b kartı her yolun fiyatını çıkarır:

| Yol | Tipik trafik | Geçilen switch |
| :--- | :--- | :--- |
| Aynı ray, aynı SU | Veri paralel gradyanlar, pipeline aktivasyonları | 1 (leaf $i$) |
| Aynı ray, başka SU | SU'lar arası veri paralellik | 3 (leaf → spine → leaf) |
| Farklı ray, doğrudan | MoE AllToAll | 3 (leaf → spine → leaf) |
| Farklı ray, PXN ile | NCCL ≥ 2.12'de MoE AllToAll | bir NVLink sıçraması + 1 switch |

NCCL 2.12'deki PXN (PCIe × NVLink), raylar arası bir mesajı önce NVSwitch üzerinden aynı düğümde hedefin rayında duran GPU'ya taşır, sonra o raydan hiç çıkmadan gönderir. Böylece bir düğümün sekiz GPU'su aynı hedefe giden mesajlarını tek bir GPU'da toplayıp birlikte yollar; NVIDIA bunun all2all başarımını ikiye katladığını bildirir. 2.048 H800 üzerinde eğitilen DeepSeek-V3 bunun ayna görüntüsünü yapar: bir token önce InfiniBand üzerinden hedef düğümde aynı düğüm içi indekse sahip GPU'ya, kendi rayından çıkmadan gider, ardından NVLink üzerinden uzmanın (expert) GPU'suna aktarılır.

**Ağ ne kadar büyüyebilir?** Bir spine'ın 64 portu en çok 64 leaf'e ulaşır ve her leaf 32 GPU taşır; bu yüzden iki katman $64 \times 32 = 2.048$ GPU'da, tam olarak 8 SU'da durur. 16 SU'dan itibaren NVIDIA spine'ların üstüne bir core katmanı ekler ve en uzun yol 3 switch'ten (leaf, spine, leaf) 5 switch'e (leaf, spine, core, spine, leaf) uzar:

| SU | DGX | GPU | Leaf | Spine | Core | Katman | En uzun yol |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 31 | 248 | 8 | 4 | – | 2 | 3 switch |
| 4 | 127 | 1.016 | 32 | 16 | – | 2 | 3 switch |
| 8 | 256 | 2.048 | 64 | 32 | – | 2 | 3 switch |
| 16 | 512 | 4.096 | 128 | 128 | 64 | 3 | 5 switch |
| 64 | 2.048 | 16.384 | 512 | 512 | 256 | 3 | 5 switch |

1 ve 4 SU satırları, bir yuvanın UFM'e gittiği NVIDIA SuperPOD tablosundan gelir; büyük satırlar ise SU başına 32 DGX sayan ölçekleme tablosundandır (o tablo son satırı "56 SU" diye etiketler, oysa $2.048 \div 32 = 64$). Üç katman $64 \times 32 \times 32 = 65.536$ GPU'ya kadar çıkar. Meta'nın RoCE Ethernet üzerine kurulu 24K GPU'luk Llama 3 kümesi de üç katmanlıdır ama uçtan uca engellemesiz değildir: 3.072 GPU'luk pod'ları içeride 1:1, pod'lar arasında ise 1:7'dir. Meta'nın paralellik yerleşimini ve iş zamanlayıcısını ağ topolojisinden haberdar ederek en ağır trafiği bir pod'un içinde tutmasının nedeni budur.

Posterin ölçek merdiveni bu uzaklaşmayı özetler: her basamak BF16 hesaplama gücünü katlar (SM başına 7,5 TFLOP/s, H100 başına 990, DGX başına 7,9 PFLOP/s, SU başına 253, SuperPOD başına 1,0 EFLOP/s ve 64 SU için 16,2 EFLOP/s), GPU başına dışarı çıkan yol ise 3,35 TB/s'den 450 GB/s'ye, oradan 50 GB/s'ye daralır.

### Ağ içi donanım hızlandırması: Mellanox SHARP ve PXN

Kümeler arası ölçeklemede en büyük zorluk, binlerce GPU arasında paylaşılan model ve gradyan tensörlerinin kumaş (fabric) üzerindeki trafik yüküdür. Geleneksel kolektif iletişimde (NCCL kütüphanesinin kullandığı AllReduce gibi primitiflerde), veriler GPU'lar arasında mantıksal halkalar (ring) üzerinden iki aşamada aktarılır: önce parçalı toplamları oluşturan ReduceScatter, ardından nihai tensörü çoğaltan AllGather. $N$ adet GPU ve $W$ çıkış bant genişliği ile klasik bir halka AllReduce, GPU başına yaklaşık $2B$ bayt veri taşır:

$$T_{\text{AllReduce, Ring}} \approx 2 \cdot \frac{B}{W}$$

Bu aktarım, tüm tensör dilimlerinin leaf ve spine switch omurgasından iki kez geçmesini zorunlu kılar. Bu $2\times$ bant genişliği maliyetini ortadan kaldırmak için modern InfiniBand ağları **Mellanox SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** teknolojisini kullanır. SHARP, floating-point toplama mantığını doğrudan Quantum switch ASIC'lerine gömer:

<svg viewBox="0 0 560 400" role="img" aria-label="Halka tabanlı kolektif iletişim ile Quantum switch SHARP ağ içi indirgeme karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="col-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="col-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="col-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="col-arr-suc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Kolektif İletişim Primitifleri</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Ring AllGather vs SHARP Donanımsal AllReduce</text>
<rect x="16" y="48" width="256" height="276" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="68" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Klasik Ring Kolektifi</text>
<text x="144" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">AllGather &amp; ReduceScatter (NCCL)</text>
<rect x="48" y="102" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="78" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 0</text>
<rect x="180" y="102" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="210" y="127" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 1</text>
<rect x="180" y="202" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="210" y="227" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 2</text>
<rect x="48" y="202" width="60" height="42" rx="4" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="78" y="227" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">GPU 3</text>
<line x1="110" y1="123" x2="178" y2="123" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="144" y="117" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Parça</text>
<line x1="210" y1="146" x2="210" y2="200" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="224" y="176" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Parça</text>
<line x1="178" y1="223" x2="110" y2="223" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="144" y="237" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Parça</text>
<line x1="78" y1="200" x2="78" y2="146" marker-end="url(#col-arr-acc)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="64" y="176" text-anchor="middle" style="fill:var(--c-accent);font-size:9.5px">B / 4 Parça</text>
<text x="144" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Süre: T ≈ 2B / W_egress</text>
<text x="144" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2(N-1) adımda halka dolaşımı</text>
<text x="144" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">AllReduce için 2x transfer (RS + AG)</text>
<rect x="288" y="48" width="256" height="276" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="68" text-anchor="middle" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">Ağ İçi İndirgeme (SHARP)</text>
<text x="416" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Quantum-2 Donanımsal Toplama</text>
<rect x="348" y="104" width="136" height="46" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="123" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Quantum Switch</text>
<text x="416" y="138" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:600">ASIC İçinde Toplama (FP)</text>
<rect x="300" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="326" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G0</text>
<rect x="358" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="384" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G1</text>
<rect x="416" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="442" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G2</text>
<rect x="474" y="204" width="52" height="38" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="500" y="226" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">G3</text>
<line x1="326" y1="202" x2="380" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<line x1="384" y1="202" x2="380" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<line x1="442" y1="202" x2="450" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<line x1="500" y1="202" x2="450" y2="152" marker-end="url(#col-arr-suc)" style="stroke:var(--c-success);stroke-width:1.3"/>
<text x="416" y="180" text-anchor="middle" style="fill:var(--c-success);font-size:10.5px;font-weight:600">Tek adımda topla ve dağıt</text>
<text x="416" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Süre: T ≈ B / W_egress (1x)</text>
<text x="416" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Ağ trafiğini %50 azaltır</text>
<text x="416" y="302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">AllReduce gecikmesini yarıya indirir</text>
<rect x="16" y="334" width="528" height="56" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="354" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Düğüm içinde NVLink ile AllGather süresi: T ≈ B / 450 GB/s (H100).</text>
<text x="28" y="372" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">SHARP devredeyken AllReduce, klasik 2x (RS + AG) yerine tek geçiş süresinde tamamlanır.</text>
</svg>

Tensörleri halkada elden ele dolaştırmak yerine her GPU tensörünü switch'e bir kez basar. Switch ASIC'i sayıları hatta akarken (in-flight) toplar ve sonucu tek geçişte tüm GPU'lara geri dağıtır:

$$T_{\text{AllReduce, SHARP}} \approx \frac{B}{W}$$

Büyük ve bant genişliğine bağlı iletimlerde SHARP, her GPU'nun hatta bastığı baytı yarıya indirir ve AllReduce süresini doğrudan %50'ye yaklaştırır. Raylı düzen bu ağaç yapısına kusursuz oturur: bir SU'daki her düğümün $i$. GPU'su leaf $i$'de toplanır, SU'lar arası kısmi toplamlar da spine switch'lerde buluşur.

**Raylar Arası Köprü: PXN (PCIe Acceleration / NVLink)**

Ray optimizasyonunda her GPU doğrudan kendi numarasına ait leaf switch'e bağlanır ($i$. GPU $\rightarrow$ $i$. leaf). Ancak raylar arasında çapraz bir iletişim gerektiğinde (örneğin GPU 0'ın başka bir sunucudaki GPU 3 ile konuşması istendiğinde), standart bir paket leaf switch'ten spine switch'e tırmanmak, oradan diğer rayın leaf switch'ine inmek zorunda kalırdı. Bu durum spine katmanında gereksiz tıkanıklık (congestion) yaratır.

NVIDIA NCCL kütüphanesinin **PXN (PCIe Acceleration / NVLink)** mekanizması bu darboğazı düğüm içi topolojiyi akıllıca kullanarak çözer: paket önce kasa içindeki ultra hızlı 450 GB/s NVLink üzerinden hedef rayın yerel GPU'suna aktarılır; ardından o GPU'nun doğrudan bağlı olduğu NIC üzerinden hedef sunucuya tek sıçramada gönderilir. Böylece omurga (spine) switch'leri hiç meşgul edilmez.

Düğüm içindeki bu PCIe, NVLink ve NUMA ara bağlantı hiyerarşisi `nvidia-smi topo -m` komutuyla doğrudan doğrulanabilir:

```text
        GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7  NIC0  NIC1  CPU0  CPU1
GPU0     X    NV18  NV18  NV18  NV18  NV18  NV18  NV18  PIX   PXB   SYS   SYS
GPU1    NV18   X    NV18  NV18  NV18  NV18  NV18  NV18  PIX   PXB   SYS   SYS
GPU2    NV18  NV18   X    NV18  NV18  NV18  NV18  NV18  PXB   PIX   SYS   SYS
GPU3    NV18  NV18  NV18   X    NV18  NV18  NV18  NV18  PXB   PIX   SYS   SYS
GPU4    NV18  NV18  NV18  NV18   X    NV18  NV18  NV18  SYS   SYS   SYS   SYS
```

Burada `NV18`, GPU'lar arasındaki 18 hatlı 4. nesil NVLink bağlantısını (çift yönde 900 GB/s), `PIX` aynı PCIe anahtarı üzerinden doğrudan erişimi, `SYS` ise NUMA soketleri arası geçişi temsil eder. Donanım topolojisinin bu hiyerarşik netliği, kumaşın en üst noktasından en alt silikonuna kadar tüm veri yollarının neden milisaniyeler değil nanosaniyeler mertebesinde koordine edildiğini açıklar.

---

## 4. 3. Sütun — Depolama: Doğrudan DMA ve yüksek başarımlı veri yolu

Bir yapay zekâ süper bilgisayarında depolama alt sistemi, birbirinden bütünüyle farklı iki temel iş yüküne cevap vermek zorundadır: eğitim veri setinin kesintisiz akışı ve model kontrol noktalarının (checkpoint snapshot) kumaşı dondurmadan kalıcı ortama yazılması.

### Düğüm başına dört ağ: Depolama neden kendi ağına sahiptir?

Bir DGX H100 sunucusunun arkasına baktığınızda tek bir kablo türü değil, fiziksel olarak yalıtılmış dört farklı ağ kumaşı görürsünüz. Hesaplama ağı yalnızca GPU'dan GPU'ya olan trafiği taşır: forward aktivasyonları, backward pass ve gradyan AllReduce transferleri. Bir kontrol noktası yazımı ya da işletim sistemi container indirmesi bu kabloları paylaşsaydı, hesaplama trafiği anında gecikme dalgalanmalarına (jitter) boğulurdu. Bu nedenle her trafik sınıfı bağımsız kablolarla ayrılır:

| Ağ | DGX başına | Switch'ler (4 SU) | Düğüm başına bant genişliği | Taşıdığı |
| :--- | :--- | :--- | :--- | :--- |
| Hesaplama ağı | 8 × 400 Gb/s InfiniBand | 48 × QM9700, 1:1 | 400 GB/s | Düğümler arası GPU trafiği |
| Depolama ağı | 2 × 400 Gb/s InfiniBand | 16 × QM9700, ~4:3 | 100 GB/s | Veri okuma, kontrol noktası yazma |
| In-band yönetim | 2 × 100 GbE | 8 × SN4600C | 25 GB/s | İşletim sistemi, cluster servisleri |
| Out-of-band yönetim | 1 × 1 GbE (BMC) | 8 × SN2201 | 0,125 GB/s | Donanım telemetrisi ve güç yönetimi |

GPU'lara en yakın katmanda her düğüm ayrıca yerel RAID 0 önbelleği olarak 8 × 3,84 TB U.2 NVMe sürücü taşır. Metin tabanlı ön eğitim veri setleri çoğu zaman bu yerel NVMe katmanında önbelleklenir. Dışarıdaki paylaşımlı yüksek başarımlı depolama kümesi ise asıl olarak yüzlerce gigabaytlık ani kontrol noktası patlamalarını karşılamak üzere boyutlandırılır.

### Geleneksel I/O darboğazı: CPU bounce buffer engeli

Geleneksel sunucu mimarilerinde bir depolama transferi, çok sekmeli (multi-hop) bir CPU dolambaçlı yolunu izler:
1. **NVMe'den Okuma:** Veri yerel SSD'den PCIe üzerinden host CPU'nun root complex birimine akar.
2. **Kernel Page Cache Kopyası:** Linux çekirdeği veriyi sistem DDR5 RAM'indeki page cache alanına kopyalar.
3. **User Space Bounce Buffer:** Uygulama veya I/O kütüphanesi veriyi kernel alanından kullanıcı belleğine `memcpy` ile taşır.
4. **GPU'ya İletim:** Host CPU, PCIe Gen5 hattı üzerinden veriyi GPU High Bandwidth Memory (HBM) içine aktarmak için ikinci bir DMA komutu verir.

Bu dolambaçlı yol iki kritik ceza üretir: veri PCIe veri yolunu iki kez katetmek zorunda kalır ve gigabaytlarca bellek kopyalaması host CPU çekirdeklerini kilitleyerek veri hazırlığı yerine kopyalamayla meşgul eder. Sonuç olarak I/O akışının hızı, en yavaş `memcpy` döngüsüne düşer.

### GPUDirect Storage (GDS): NVMe'den GPU belleğine doğrudan DMA

Modern yapay zekâ donanım tasarımları bu darboğazı **GPUDirect Storage (GDS)** mimarisiyle ortadan kaldırır:

<svg viewBox="0 0 560 420" role="img" aria-label="Geleneksel CPU bellek kopyalı G/Ç ile GPUDirect Storage doğrudan DMA mimarisi karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="gds-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="gds-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="gds-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="gds-arr-suc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
<marker id="gds-arr-warn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Storage Architecture: CPU Bounce vs GPUDirect Storage</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Checkpointing ve Dataset Streaming Pipeline'ları</text>
<rect x="16" y="48" width="256" height="286" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="68" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Geleneksel I/O Yolu</text>
<text x="144" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">CPU Page Cache &amp; Bounce Buffer'lar</text>
<rect x="32" y="252" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="144" y="274" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">NVMe Depolama (SSD)</text>
<rect x="32" y="150" width="224" height="54" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="144" y="170" text-anchor="middle" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">Host CPU &amp; DDR5 RAM</text>
<text x="144" y="188" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Kernel Page Cache (50-100 GB/s)</text>
<rect x="32" y="96" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="144" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU HBM3 (3,35 TB/s)</text>
<line x1="144" y1="250" x2="144" y2="206" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">I/O Adım 1</text>
<line x1="144" y1="148" x2="144" y2="134" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">I/O Adım 2</text>
<rect x="24" y="302" width="240" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="144.0" y="316.3" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Darboğaz: CPU &amp; RAM Bounce</text>
<rect x="288" y="48" width="256" height="286" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="68" text-anchor="middle" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">GPUDirect Storage (GDS)</text>
<text x="416" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Direct DMA (cuFile API)</text>
<rect x="304" y="252" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="416" y="274" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">NVMe Depolama (SSD)</text>
<rect x="304" y="162" width="224" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="180" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:700">PCIe Gen 5 Switch / DMA</text>
<text x="416" y="196" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Direct Donanım Yolu (64+ GB/s)</text>
<rect x="304" y="96" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU HBM3 (3,35 TB/s)</text>
<line x1="416" y1="250" x2="416" y2="206" marker-end="url(#gds-arr-suc)" style="stroke:var(--c-success);stroke-width:2"/>
<line x1="416" y1="160" x2="416" y2="134" marker-end="url(#gds-arr-suc)" style="stroke:var(--c-success);stroke-width:2"/>
<text x="424" y="232" text-anchor="start" style="fill:var(--c-success);font-size:10px;font-weight:600">Direct DMA</text>
<rect x="296" y="302" width="240" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="416.0" y="316.3" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Sıfır CPU Yükü · Direct DMA</text>
<rect x="16" y="344" width="528" height="64" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="364" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Checkpointing Zorluğu: Büyük modellerde devasa state periyodik kaydedilir.</text>
<text x="28" y="382" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GDS ile depolama, sistem RAM'ini baypas ederek doğrudan GPU HBM'e akar,</text>
<text x="28" y="398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">CPU bounce gecikmesi elenir ve eğitim duraklamaları önlenir.</text>
</svg>

GDS, NVIDIA `cuFile` API'si ve `nvidia-fs.ko` çekirdek modülü üzerinden çalışır. Yerel NVMe sürücüler (veya NVMe-oF üzerinden ağ depolaması) ile GPU HBM belleği arasında doğrudan bir PCIe DMA kanalı kurar. Bu sayede:
- Veri, PCIe hattını yalnızca bir kez geçer; host belleğindeki kopyalama adımları ortadan kalkar.
- Host CPU çekirdekleri `memcpy` yükünden kurtulur ve veri ön işleme görevlerine odaklanır.
- Gecikme varyansı düşer ve I/O veri yolu donanımın fiziksel PCIe Gen5 tavanına (yön başına ~64 GB/s) yaklaşır.

DGX H100 sisteminde depolama NIC'leri ve yerel NVMe sürücüler, kendi NUMA yarılarındaki GPU'larla aynı PCIe anahtarına bağlıdır. Bu sayede doğrudan DMA transferleri soketler arası veri yoluna hiç taşmadan yerel PCIe anahtarı içinde tamamlanır. Donanım seviyesindeki bu doğrudan DMA erişimi, `cuFile` API'si ile kod seviyesinde şöyle yapılandırılır:

```cpp
// cuFile API: Doğrudan NVMe'den GPU HBM'e DMA transferi (CPU bounce buffer baypas edilir)
CUfileDescr_t cf_descr;
cf_descr.handle.fd = fd;
cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
cuFileHandleRegister(&cf_handle, &cf_descr);

// Host RAM'e kopyalamadan doğrudan GPU belleğine okuma
cuFileRead(cf_handle, dev_ptr, size, file_offset, 0);
cuFileHandleDeregister(cf_handle);
```

---

## 5. Sentez: Donanım hiyerarşisi ve hız farkları

Üç sütunun mimari kararları, model eğitimi ile yüksek hacimli çıkarım (inference) sunumu arasında donanımı bambaşka rejimlere sokar:

### İki mühendislik merceği: Training ve inference donanım rejimleri

Donanım kararları model eğitimi ile yüksek hacimli çıkarım (inference) sunumu arasında taban tabana zıttır:

| Mimari Boyut | Eğitim (Pre-training / SFT) | Çıkarım (Prefill & Decode) |
| :--- | :--- | :--- |
| **Birincil Bellek Ayak İzi** | Model Ağırlıkları ($2B$) + Gradyanlar ($2B$) + AdamW Durumları ($12-16B$) + Aktivasyonlar | Model Ağırlıkları ($4-8$ bit quantized) + Dinamik tahsis edilen KV Cache |
| **Hesaplama / Bellek Rejimi** | Yoğun biçimde **Compute-Bound** (Batch başına binlerce token içeren GEMM) | Prefill evresi **Compute-Bound**; Decode evresi kesinlikle **Memory-Bandwidth-Bound** ($B=1$, GEMV) |
| **Baskın Ağ Primitifi** | InfiniBand üzerinde yüksek throughput'lu **AllReduce** & **ReduceScatter** | NVLink üzerinde, token başına ve katman başına bir ya da iki kez koşan düşük gecikmeli tensor paralel **AllReduce** |
| **Temel Performans Metrikleri** | Model FLOPs Utilization (MFU), Step Time (ms), Gradyan Kararlılığı | Time to First Token (TTFT), Inter-Token Latency (TPOT), Eşzamanlı İstek Sayısı |
| **Kritik Hata Modları** | Loss spike'lar, silent data corruption (SDC), pipeline bubble'lar | KV Cache bellek patlaması (OOM), bellek parçalanması, tail latency sıçramaları |

Posterin D bölümü çıkarım sütununu elle tutulur hâle getirir. 140 GB'lık BF16 model ağırlıklarını donanımın her katmanından geçirdiğinizde hız farkları mertebelerle birbirinden ayrılır:

```text
hat                          140 GB'ın süresi
HBM3 (tek H100, 3,35 TB/s)   140 ÷ 3.350  ≈ 0,042 s  =  42 ms
NVLink (450 GB/s)            140 ÷ 450    ≈ 0,31 s
PCIe Gen5 (64 GB/s)          140 ÷ 64     ≈ 2,2 s
InfiniBand (GPU başına 50)   140 ÷ 50     =  2,8 s
eski depolama (25 GB/s)      140 ÷ 25     =  5,6 s
```

İnsan ölçeğine çevirirsek, HBM'den bir saniyede gelen veri NVLink üzerinden 7,4 saniyede, PCIe üzerinden 52 saniyede, InfiniBand üzerinden 67 saniyede gelir. Decode evresi doğrudan ilk satıra bağlıdır: üretilen her token tüm model ağırlıklarını en az bir kez bellekten okur. Bu donanım hiyerarşisi, büyük modellerin binlerce GPU'luk süper bilgisayarlara nasıl ölçeklendiğini, gerçek küme koşullarında MFU kayıplarının nasıl yönetildiğini ve operasyonel arıza mekanizmalarını serinin ikinci makalesine bağlar: [Yapay Zekâ Altyapısı (2): Bir AI Süper Bilgisayarının Teşhisi — 1.024 H100 Neden %38 MFU'da Kalır?](post.html?slug=gpudan-ai-super-bilgisayarina).

---

## Bütün hikâye altı satırda

- **Üç Sütunun Birlikte Tasarımı (Co-design):** Bir yapay zekâ hızlandırıcısı tek başına çalışmaz; hesaplama (SM/Tensor Core), ağ (NVLink/InfiniBand) ve depolama (NVMe/GDS) birbirine denk bant genişlikleriyle dengelenmek zorundadır.
- **SM ve Tensor Core Anatomisi:** H100 çipi 132 SM ve 528 Tensor Core ile 990 TFLOP/s BF16 sunar; bellek duvarını aşmak için her SM 64 warp'a kadar donanımsal bağlam tutarak ~480 döngülük HBM3 gecikmesini saklar.
- **Kasa İçi NVLink Crossbar'ı:** Sekiz H100 GPU'su 4 adet NVSwitch ASIC'i üzerinden tam engellemesiz (non-blocking) bir crossbar ile bağlanır; GPU başına 450 GB/s tek yönlü bant genişliği katman içi tensör paylaşımını mümkün kılar.
- **SuperPOD ve Ray Hizalı Fat-Tree:** Kümeler 32 DGX'lik Ölçeklenebilir Birimler (SU) halinde 2:1 veya 1:1 engellemesiz iki katmanlı fat-tree topolojisiyle örülür; ray optimizasyonu sayesinde aynı dilimi tutan GPU'lar tek switch sıçramasında buluşur.
- **Ağ İçi Donanım Hızlandırması (SHARP & PXN):** Quantum-2 switch ASIC'leri AllReduce toplamalarını hatta akarken (in-network) hesaplayarak ağ trafiğini %50 azaltır; PXN ise raylar arası paketleri NVLink üzerinden aktararak omurga tıkanıklığını önler.
- **Depolamada Doğrudan DMA (GDS):** Düğüm başına dört bağımsız ağ ayrılır; GPUDirect Storage, CPU page cache ve host RAM bounce buffer'larını atlayarak NVMe sürücüler ile HBM arasında doğrudan PCIe DMA kanalı kurar.

---

## Terimler sözlüğü

- **Streaming Multiprocessor (SM)** — NVIDIA GPU'larının Tensor Core, Warp Scheduler, CUDA çekirdekleri ve on-chip SRAM barındıran temel çok iş parçacıklı hesaplama hücresi.
- **Warp** — NVIDIA mimarilerinde bir SM warp scheduler tarafından eşzamanlı olarak yürütülen 32 paralel thread'den oluşan temel çizelgeleme birimi.
- **Warp Divergence** — Bir warp içindeki thread'lerin farklı yürütme dallarına sapması sonucu inaktif çekirdeklerin beklemesiyle oluşan başarım kaybı.
- **Occupancy (doluluk)** — Bir kernel'in bir SM'deki 64 warp yuvasından ne kadarını doldurduğu; thread başına istenen register arttıkça düşer ve SM'in bellek gecikmesini gizleme yoludur.
- **Kritik aritmetik yoğunluk ($\alpha_{\text{crit}}$)** — Donanımın tavan hesaplama gücünün bellek bant genişliğine oranı ($C_{\text{peak}} / W_{\text{mem}} \approx 295$ FLOP/bayt); bu sınırın altında kalan işlemler bellek hızında koşar.
- **NVLink** — H100 GPU başına tek yönde 450 GB/s (çift yönde 900 GB/s) bellek aktarımı sağlayan NVIDIA'nın tescilli yüksek bant genişlikli interconnect teknolojisi.
- **NVSwitch** — Bir HGX şasisindeki 8 GPU arasında non-blocking, all-to-all iletişim sağlayan node içi crossbar ASIC'i.
- **Fabric Manager** — NVSwitch yönlendirme tablolarını bir PCIe kontrol yolu üzerinden programlayan NVIDIA servisi; o olmadan NVLink ağı ayağa kalkmaz.
- **NUMA alanı** — Kendi belleği ve ona bağlı GPU'larla NIC'leri olan bir CPU soketi; bir DGX H100'de iki tane vardır ve veri yolları bu yarılar üzerinden optimize edilir.
- **Scalable Unit (SU)** — Bir DGX SuperPOD'un tekrar eden yapı taşı: 32 DGX düğümü (256 GPU) ve onları birleştiren 8 leaf switch.
- **Leaf ve spine switch** — Bir fat-tree'nin iki anahtarlama katmanı: sunucular leaf'lere takılır, leaf'ler her spine'a bağlanır ve herhangi iki sunucu en çok üç switch sıçramasında buluşur.
- **Rail-optimized kablolama (raylı kablolama)** — Her düğümün $i$. NIC'ini $i$. leaf'e bağlamak; modelin aynı dilimini tutan aynı numaralı GPU'lar böylece tek sıçramada buluşur.
- **PXN (PCIe Acceleration / NVLink)** — Raylar arası bir mesajı önce yerel NVLink üzerinden hedef raydaki GPU'ya taşıyarak spine switch trafiğini atlatan donanım/NCCL özelliği.
- **Mellanox SHARP** — Scalable Hierarchical Aggregation and Reduction Protocol; Quantum switch donanımında floating-point AllReduce işlemlerini hatta akarken (in-network reduction) toplayan donanım protokolü.
- **GPUDirect Storage (GDS)** — NVMe depolama birimlerini host CPU page cache'lerini baypas ederek doğrudan GPU High Bandwidth Memory'ye bağlayan direct DMA teknolojisi.

---

## Daha derine inmek için

- Bu blogda: [Yapay Zekâ Altyapısı (2): Bir AI Süper Bilgisayarının Teşhisi — 1.024 H100 Neden %38 MFU'da Kalır?](post.html?slug=gpudan-ai-super-bilgisayarina).
- Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12.
- SemiAnalysis (2025). [NVIDIA Tensor Core Evolution from Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/). Mikro-mimari retrospektifi.
- NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Resmi mikro-mimari kılavuzu: GPC'ler, TPC'ler, SM alt bölümleri, HBM3 yığınları ve bellek denetleyicileri.
- NVIDIA Corporation. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html). Şekil 1.1'deki SM sınırları: SM başına 64 warp ve 64K register, thread başına en çok 255 register, en çok 228 KB shared memory ve TMA kopya motoru.
- Weile Luo vd. (2024). [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). arXiv:2402.13499. Şekil 1.1'deki ölçülmüş Hopper yükleme gecikmeleri: SMEM ~29, L1 ~41, L2 ~263 ve global bellek ~479 döngü.
- NVIDIA Corporation. [DGX SuperPOD Reference Architecture (DGX H100): Key Components](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-components.html) ve [Network Fabrics](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html). Ölçeklenebilir birimler, dört ağ, UFM yuvalı 127 düğüm ve ayrı depolama ağı; Şekil 1.1'deki diyagramlara ilham verdi.
- NVIDIA Corporation. [Introduction to NVIDIA DGX H100/H200 Systems](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). Düğüm donanımı: işlemciler, bellek, NIC'ler ve NVMe sürücüler; DGX H100/H200 sistem topolojisi diyagramı Şekil 1.1'deki düğüm basamağına ilham verdi.
- NVIDIA Corporation. [Fabric Manager for NVIDIA NVSwitch Systems](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). Her H100'ün 18 NVLink'inin dört NVSwitch'e 5 + 4 + 4 + 5 olarak dağılması ve Fabric Manager olmadan NVLink ağının neden kurulmadığı.
- NVIDIA Corporation. [DGX SuperPOD Architecture (DGX H100)](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html). Ölçek tablosu: 8 SU'ya kadar iki katman, 16 SU'dan itibaren core katmanı; 2.048 DGX için 512 leaf, 512 spine ve 256 core switch.
- NVIDIA Corporation. [QM97X0 NDR 400Gb/s InfiniBand Switch Systems User Manual: Introduction](https://docs.nvidia.com/networking/display/qm97x0um/introduction). Quantum-2 QM9700: 64 adet NDR 400 Gb/s port, switch başına iki yönde toplam 51,2 Tb/s, SHARP ile ağ içinde toplama.
- Charles Clos (1953). [A Study of Non-Blocking Switching Networks](https://doi.org/10.1002/j.1538-7305.1953.tb01433.x). Bell System Technical Journal 32(2), 406–424. Bugünkü leaf–spine ağlarının dayandığı çok katlı anahtarlama fikri.
- Charles E. Leiserson (1985). [Fat-trees: Universal Networks for Hardware-Efficient Supercomputing](https://doi.org/10.1109/TC.1985.6312192). IEEE Transactions on Computers C-34(10), 892–901. Fat-tree adının kökeni: yukarı çıktıkça kalınlaşan bağlantılar.
- NVIDIA Developer Blog (2022). [Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/). PXN: raylar arası trafiği önce NVLink'e taşıyıp spine'ı atlamak.

