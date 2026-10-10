Kâğıt üzerinde 1.024 adet NVIDIA H100 GPU'dan oluşan bir küme yenilmez görünür: 1 Eksaflopp'un üzerinde 8-bit tensör hesaplama gücü, 80 Terabayt yüksek bant genişlikli bellek ve saniyede petabitleri aşan ara bağlantı omurgası. Ancak 70 milyar parametreli bir temel modelin ön eğitimini başlattığınız anda telemetri panoları sizi soğuk bir gerçekle yüzleştirir: Model FLOP Kullanımı (MFU) %38'in altında sürünür, yüksek maliyetli Streaming Multiprocessor çekirdekleri döngülerinin %40'ını yüksek bant genişlikli bellek kuyruklarında bekleyerek harcar, InfiniBand hatları gradyan senkronizasyon kabarcıklarıyla tıkanır ve eşzamanlı kontrol noktası (checkpoint) kayıtları birkaç saatte bir tüm kümeyi beş dakika boyunca dondurur.

Bu darboğaz rastgele bir aksaklık ya da yazılım hatası değildir. Modern bir GPU süper bilgisayarını homojen, tek parça devasa bir işlem havuzu sanmanın fiziksel bedelidir. Statik ve eşzamanlı iletişim döngülerine sahip 2D veya 3D torus ağlarıyla birbirine bağlı Google TPU'larının aksine, modern yapay zekâ süper bilgisayarları derinlemesine hiyerarşik ve üç temel fiziksel sütun tarafından yönetilen yapılardır: **Hesaplama (Compute)**, **Ağ (Network)** ve **Depolama (Storage)**. Her bir silikon çipin içinde yüzden fazla bağımsız Streaming Multiprocessor dar bir L2 önbellek için yarışır. 8 GPU'lu tek bir kasa içinde NVSwitch çapraz bağlantıları çip başına çift yönde 900 GB/s gibi akıl almaz bir bant genişliği sunar. Fakat kolektif bir işlem düğüm sınırını aşıp InfiniBand ağına adım attığı anda erişilebilir bant genişliği neredeyse bir mertebe birden düşer; depolama alt sisteminiz ise ana işlemcileri (CPU) boğmadan terabaytlarca kontrol noktasını ve trilyonlarca tokeni kesintisiz akıtamazsa, milyonlarca dolarlık tüm hesaplama omurganız kilitlenir.

Bu yazı, modern temel modellerin ölçekleme mimarisini bu üç kurucu sütun üzerinden adım adım inceliyor. Bağımsız warp scheduler birimlerinin silikon üzerindeki komut dağıtımından NVLink ve rail-optimized InfiniBand fat-tree ağ topolojilerine, Mellanox SHARP gibi ağ içi donanımsal hızlandırmalardan GPUDirect Storage ve Young-Daly kontrol noktası formülüne kadar tüm yığını titiz matematiksel modellemeler ve ampirik ölçümlerle ortaya koyuyor.

**Bu yazıda**

- [1. Modern yapay zekâ üçlüsü: Hesaplama, ağ ve depolama](#1-modern-yapay-zekâ-üçlüsü-hesaplama-ağ-ve-depolama)
- [2. 1. Sütun — Hesaplama: Streaming Multiprocessor, mikro-mimari ve yürütme dinamikleri](#2-1-sütun-hesaplama-streaming-multiprocessor-mikro-mimari-ve-yürütme-dinamikleri)
- [3. 2. Sütun — Ağ: Ara bağlantı omurgaları, kolektifler ve çok boyutlu paralellik](#3-2-sütun-ağ-ara-bağlantı-omurgaları-kolektifler-ve-çok-boyutlu-paralellik)
- [4. 3. Sütun — Depolama: Temel model veri setleri, kontrol noktaları ve yüksek verimli G/Ç](#4-3-sütun-depolama-temel-model-veri-setleri-kontrol-noktaları-ve-yüksek-verimli-gç)
- [5. Sentez: Eğitim ve çıkarım arasındaki operasyonel dengeler](#5-sentez-eğitim-ve-çıkarım-arasındaki-operasyonel-dengeler)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Modern yapay zekâ üçlüsü: Hesaplama, ağ ve depolama

Dağıtık temel model eğitimi, uç sınırlarda bir ortak tasarım (co-design) mühendisliğidir. Binlerce hızlandırıcı üzerinde bir büyük dil modeli ölçeklemeye çalışan mühendisler kaçınılmaz olarak üç fiziksel duvara çarpar:

1. **Bellek Duvarı (Hesaplama):** Son on yılda ham kayan nokta matris hesaplama gücü $1.000\times$'den fazla artarken, yüksek bant genişlikli bellek (HBM) hızları $30\times$'dan daha az büyüdü. GPU içindeki aritmetik birimler, matris boyutları donanım roofline modelinin kırılma noktasını aşacak düzeyde veri yeniden kullanımı sunmadığı sürece aç kalır.
2. **Ağ Duvarı (İletişim):** İletişim bant genişliği küme boyutuyla doğrusal ölçeklenmez. 8 GPU'lu tek bir kasa içinde NVLink üzerinden aktivasyon ve gradyan aktarmak, yüzlerce sunucu kabinine yayılan optik InfiniBand veya RoCE v2 hatlarına kıyasla neredeyse bir mertebe daha hızlıdır. Model boyutlarını bu ara bağlantı hiyerarşisine uyduramamak felaket gecikmelere yol açar.
3. **Depolama Duvarı (G/Ç):** Temel modeller 15 trilyondan fazla token ile eğitilir; 405 milyar parametreli bir modelin parametre ve optimizer durumları tek bir kontrol noktasında 7,5 Terabayttan fazla yer kaplar. Depolama erişimi geleneksel işletim sistemi önbelleklerine ve CPU bellek kopyalamalarına dayanırsa, host bellek veri yolları tıkanır ve her kontrol noktası yazımında tüm küme donar.

Donanımın boşta kalmasını engellemek ve Model FLOP Kullanımını (MFU) maksimize etmek için bir yapay zekâ altyapı mühendisi bu üç sütunu eşzamanlı olarak yönetmek zorundadır:

<svg viewBox="0 0 560 10849" role="img" aria-label="Yapay zekâ altyapısını anlatan beş bölümlük bir poster. A bölümü dört basamakta uzaklaşır. 1. basamak, Streaming Multiprocessor: dört özdeş alt bölme; her birinde döngüde 1.024 BF16 FLOP yapan bir Tensor Core, 32 thread&#x27;lik warp&#x27;ları dağıtan bir warp scheduler, 32 FP32 ALU ve 16.384 register; ayrıca 256 KiB SMEM ya da L1. Bir hat aşçısı ve tezgâhı gibidir. 132 SM bir H100 SXM5 GPU eder: 814 milimetrekare kalıp, 80 milyar transistör, 50 MB L2, 6 yuvanın 5&#x27;inde 3,35 TB/s hızlı 80 GB HBM3, 700 W&#x27;ta 990 TFLOP/s BF16; kileri HBM olan bir mutfak gibidir. 8 GPU bir DGX H100 düğümü eder: 4 NVSwitch, GPU başına 900 GB/s NVLink, 3,6 TB/s biseksiyon, PCIe Gen5 üzerinden 64 GB/s ile bağlı 2 TB RAM&#x27;li 2 Xeon işlemci, GPU başına bir 400 Gb/s hesaplama NIC&#x27;i, depolama ve yönetim için 2 çift portlu NIC ve 8 NVMe sürücü. 32 düğüm bir ölçeklenebilir birim, 4 birim 127 düğümlük, 1.016 GPU&#x27;luk bir DGX SuperPOD eder: her düğümün i. NIC&#x27;inin i. leaf&#x27;e gittiği ray uyumlu fat-tree&#x27;de 32 leaf ve 16 spine switch, düğüm başına en az 40 GB/s veren ayrı bir depolama ağı ve in-band ile OOB Ethernet yönetim ağları. B bölümü TP 8, PP 16, DP 128 düzenli Llama 3 405B&#x27;nin tek bir eğitim adımını izler: 1 depolamadan HBM&#x27;e 16 milyon token okunur; 2 ileri geçişte HBM&#x27;den Tensor Core&#x27;lara veri akar; 3 NVLink üzerinde katman başına iki tensör paralel AllReduce yapılır; 4 aktivasyonlar ağ üzerinden sonraki boru hattı aşamasına gönderilir; 5 geri geçiş aynı hatları kullanır; 6 gradyanlar ray switch&#x27;leri üzerinden diğer 127 kopyayla AllReduce edilir; 7 AdamW optimizer adımı HBM&#x27;de yapılır; 8 her 14 dakikada 7,5 TB&#x27;lık kontrol noktası depolamaya yazılır. C bölümü üç duvarı hesaplanmış eğrilerle gösterir. Bellek duvarı: H100&#x27;de Llama 3 70B MLP katmanı için elde edilen hız batch 1&#x27;de 3,3 TFLOP/s&#x27;den batch 128&#x27;de 420&#x27;ye çıkar ve batch 310 civarında 990 tavanına ulaşır; çare büyük batch, kernel füzyonu ve decode&#x27;da düşük bitli ağırlıktır. Ağ duvarı: GPU başına bant genişliği HBM&#x27;de 3,35 TB/s&#x27;den NVLink&#x27;te 450 GB/s&#x27;ye, düğüm dışında 50 GB/s&#x27;ye düşer; tensör paralelliği NVLink&#x27;te 13 GPU&#x27;ya, yani 8&#x27;e sığar, InfiniBand&#x27;de yalnızca 1,45&#x27;e; 16 aşamalı bir boru hattı 16 mikrobatch ile yüzde 48, 128 ile yüzde 10,5 boşta kalır; çare TP&#x27;yi düğüm içinde tutmak, SHARP ve iletişimi hesapla çakıştırmaktır. G/Ç duvarı: kayıt aralığına göre Young-Daly kayıp eğrileri; 25 GB/s&#x27;de en düşük kayıp 42,9 dakikada yüzde 24, 250 GB/s&#x27;de 13,6 dakikada yüzde 7,5&#x27;tir; çare GPUDirect Storage, asenkron yazma ve karekök aralığıdır. D bölümü aynı 140 GB&#x27;ın taşınma süresini verir: HBM&#x27;den 42 milisaniye, NVLink&#x27;ten 0,31 saniye, PCIe&#x27;den 2,2 saniye, GPU başına InfiniBand&#x27;den 2,8 saniye, bütün kümenin paylaştığı eski depolamadan 5,6 saniye; 8 H100&#x27;de decode saniyede yaklaşık 190 tokenle sınırlıdır. E bölümü Meta&#x27;nın Llama 3 405B eğitimini özetler: GPU başına 400 TFLOP/s ve yüzde 41 MFU ile 16.384 H100, 3.072 GPU&#x27;luk pod&#x27;lardan oluşan bir RoCE ağı, 2 TB/s sürekli ve 7 TB/s tepe hızlı 240 PB&#x27;lık Tectonic depolama ve 54 günde 466 kesinti, yani yaklaşık 3 saatte bir arıza. En yavaş halka bütün kümenin hızını belirler. Küme basamağından sonra bir kart SuperPOD&#x27;un yapı taşlarını ve her birinin neden var olduğunu sıralar: 127 DGX H100 sistemi matematiği yapar; 48 QM9700 switch her GPU&#x27;ya bir ray veren hesaplama ağını, 16 tanesi düğüm başına en az 40 GB/s&#x27;lik ayrı depolama ağını kurar; 4 UFM cihazı InfiniBand ağını izler; 8 SN4600C ve 8 SN2201 switch in-band ve OOB yönetimi taşır; 5 yönetim sunucusu kontrol yazılımlarını koşturur; 38 kabin ve 108 PDU yaklaşık 1,3 MW güç dağıtır. Yazılımda Base Command Manager düğümleri kurar, Slurm işleri zamanlar, Magnum IO veriyi NCCL, GPUDirect ve SHARP ile taşır, NGC uyumlu konteynerler sunar, Base Command Platform veri bilimcilere hizmet eder, AI Enterprise destekli araçlar ekler. Düğüm basamağı NVIDIA&#x27;nın DGX H100 sistem topolojisini yeniden çizer: UPI ile bağlı CPU 0 ve CPU 1; CPU 0 GPU 0–3 ile NIC 0–3&#x27;ü, CPU 1 GPU 4–7 ile NIC 4–7&#x27;yi besler ve her GPU&#x27;nun kendi ağ kartı vardır; üstteki iki PCIe switch çift portlu depolama kartlarını ve 4&#x27;er NVMe diski taşır; CPU 0&#x27;da iki M.2 işletim sistemi diski, CPU 1&#x27;de bir 100 GbE kartı bulunur; her GPU 18 NVLink&#x27;ini 5, 4, 4 ve 5 olarak dört NVSwitch&#x27;e gönderir; alttaki PCIe switch NVSwitch&#x27;leri yönetim için CPU 0&#x27;a bağlar. Düğüm şemasındaki altı numaralı daire, altındaki altı soruyla eşleşir: kim kimi besler (iki NUMA yarısı), GPU ağa nasıl çıkar (GPUDirect RDMA ile kendi NIC&#x27;i), GPU&#x27;lar nasıl konuşur (dört NVSwitch&#x27;e 18 NVLink), depolama trafiği nereden geçer (üstteki PCIe switch&#x27;ler), iki yarıyı ne birleştirir (CPU iletişimi, UPI) ve NVSwitch&#x27;leri kim yönetir (CPU 0&#x27;dan PCIe ile Fabric Manager). Küme basamağı H100 SXM5, DGX H100, 32 DGX&#x27;lik ölçeklenebilir birim ve 4 birimli DGX SuperPOD hiyerarşisiyle açılır. Yakından inceleme kartı portları sayar: 64 portlu her leaf 32 portunu düğümlere, 32 portunu spine&#x27;lara verir, toplam 1.024 leaf–spine kablosu vardır ve ağ engelsizdir; aynı raydaki trafik tek leaf&#x27;ten geçer, raylar arası trafik ise NCCL&#x27;in PXN&#x27;i önce NVLink&#x27;e taşımadıkça spine&#x27;dan geçer; bütün GPU&#x27;lar 50,8 TB/s gönderebilir, spine kapasitesi 51,2 TB/s&#x27;dir; 16 birimden itibaren bir core katmanı eklenir ve 2.048 DGX 512 leaf, 512 spine ve 256 core switch kullanır. Bir not SuperPOD&#x27;un neden 128 değil 127 DGX olduğunu açıklar: bir çift UFM cihazı bir DGX yuvasını alır; daha küçük kurulumlar tek birimden başlar, 31 ile 127 DGX arasındadır. Küme çizimi NVIDIA&#x27;nın hesaplama ağı şemasını izler: üstte Spine-01&#x27;den Spine-16&#x27;ya; Leaf-01-01–08 ve H100-001–032 düğümleriyle SU 1; Leaf-04-01–08, H100-097–127 düğümleri ve C-UFM 1 ile 2 UFM çiftiyle SU 4; spine–leaf hatları 2 × 400 Gb/s, DGX–leaf hatları 400 Gb/s, UFM–leaf hatları 200 Gb/s&#x27;dir. Hiyerarşinin altındaki &#x27;ne neyi yapar&#x27; tablosu her basamağı açıklar: H100 matematiği yapar ve modelin kendine düşen payını HBM&#x27;de tutar; DGX 8 GPU&#x27;yu tensör paralelliğinin yaşadığı tek bir NVLink alanında birleştirir, işlemcileri işletim sistemini ve veri yükleyiciyi koşturur; SU, aynı numaralı GPU&#x27;ların tek sıçramada buluştuğu 32 DGX ve 8 leaf switch&#x27;lik tekrar eden birimdir; SuperPOD SU&#x27;ları spine ile birleştirir, depolama, yönetim ve yazılımı ekler ve veri paralelliğini bütün kümeye yayar. Bir tanım kutusu leaf ve spine&#x27;ı açıklar: leaf, sunucuların takıldığı alt katman switch&#x27;idir, SU başına 8 tane, her ray için bir tane, 32 portu aşağı 32 portu yukarı; spine, leaf&#x27;leri birbirine bağlayan ve hiç sunucu taşımayan üst katman switch&#x27;idir, toplam 16 tane, her biri 32 leaf&#x27;in hepsine bağlıdır; her leaf her spine&#x27;a ulaştığı için herhangi iki sunucu en fazla leaf, spine, leaf yoluyla buluşur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bp-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bp-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="bp-wrn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="bp-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="16" y="26" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">Büyük Resim: Yapay Zekâ Altyapısının 3 Sütunu</text>
<text x="16" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Bir H100&#x27;ün içinden 16.384 H100&#x27;lük kümeye: parçalar nerede, veri nasıl akar, nerede</text>
<text x="16" y="63" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tıkanır?</text>
<rect x="16" y="75" width="528" height="225" rx="10" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="87" width="86" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="71" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">A · Parçalar</text>
<text x="71" y="120" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SM → küme</text>
<line x1="116.0" y1="108.0" x2="130.0" y2="108.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="132" y="87" width="86" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="175" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">B · Bir adım</text>
<text x="175" y="120" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">verinin yolu</text>
<line x1="220.0" y1="108.0" x2="234.0" y2="108.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="236" y="87" width="86" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="279" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">C · Duvarlar</text>
<text x="279" y="120" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">neden tıkanır</text>
<line x1="324.0" y1="108.0" x2="338.0" y2="108.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="340" y="87" width="86" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="383" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">D · Hız</text>
<text x="383" y="120" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">sezgi</text>
<line x1="428.0" y1="108.0" x2="442.0" y2="108.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="444" y="87" width="86" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="487" y="104" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">E · Llama 3</text>
<text x="487" y="120" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">gerçek küme</text>
<text x="28" y="153" text-anchor="start" style="fill:var(--c-text);font-size:12px">Önce parçaları tanıyacağız, sonra bir eğitim adımında birlikte nasıl</text>
<text x="28" y="170" text-anchor="start" style="fill:var(--c-text);font-size:12px">çalıştıklarını izleyeceğiz. Ardından nerede tıkandıklarını görecek, hızlar için</text>
<text x="28" y="187" text-anchor="start" style="fill:var(--c-text);font-size:12px">sezgi kuracak ve hepsini gerçek bir kümeyle sınayacağız.</text>
<rect x="28.0" y="201.0" width="504.0" height="65" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="220" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Referans sistem: H100 SXM5 → DGX H100 → DGX SuperPOD (InfiniBand). Ana</text>
<text x="40" y="237" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">sayıların hepsi bu sisteme aittir; başka GPU&#x27;lar ve başka kümeler yalnızca</text>
<text x="40" y="254" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">parantez içinde geçer.</text>
<rect x="28.0" y="279" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="44" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesaplama</text>
<rect x="109.0" y="279" width="11" height="11" rx="2" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="125.0" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağ</text>
<rect x="151.9" y="279" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="167.9" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">depolama</text>
<rect x="227.5" y="279" width="11" height="11" rx="2" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="243.5" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">duvar / darboğaz</text>
<rect x="346.6" y="279" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="362.6" y="289" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">çözüm</text>
<rect x="16" y="312" width="528" height="7322" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="336" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="340.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">A</text>
<text x="56" y="341" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Yakınlaştır: bir SM&#x27;den bütün kümeye</text>
<text x="28" y="362" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her basamak öncekinin ×N kopyasıdır; koyu vurgu bir önceki basamağı gösterir.</text>
<rect x="28" y="374" width="504" height="332" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="398" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">1 · SM: GPU&#x27;nun temel işçisi</text>
<rect x="40" y="410" width="228" height="152" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="48" y="426" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">SM</text>
<rect x="48" y="434" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="56" y="449" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 0</text>
<rect x="56.0" y="456.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="100.0" y="469.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="156" y="434" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="164" y="449" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 1</text>
<rect x="164.0" y="456.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="208.0" y="469.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48" y="484" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="56" y="499" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 2</text>
<rect x="56.0" y="506.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="100.0" y="519.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="156" y="484" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="164" y="499" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 3</text>
<rect x="164.0" y="506.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="208.0" y="519.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48.0" y="536.0" width="212.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.0" y="550.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM / L1 · 256 KiB</text>
<rect x="282" y="410" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="292" y="432" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">4</text>
<text x="292" y="449" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme</text>
<rect x="407" y="410" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="417" y="432" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1.024</text>
<text x="417" y="449" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/döngü (TC)</text>
<rect x="282" y="466" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="292" y="488" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">256 KiB</text>
<text x="292" y="505" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">register / SM</text>
<rect x="407" y="466" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="417" y="488" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">64</text>
<text x="417" y="505" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">warp aynı anda</text>
<text x="282" y="534" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her alt bölme: 1 Tensor Core,</text>
<text x="282" y="550" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1 warp scheduler, 32 FP32 ALU,</text>
<text x="282" y="566" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16.384 register · 256 KiB SMEM/L1</text>
<text x="282" y="582" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">warp = birlikte koşan 32 thread</text>
<text x="40" y="608" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="40" y="626" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bir warp bellekten veri beklerken scheduler hemen başka bir warp&#x27;a geçer.</text>
<text x="40" y="643" text-anchor="start" style="fill:var(--c-text);font-size:12px">SM&#x27;ler bekleme süresini böyle gizler; ama bütün warp&#x27;lar bekliyorsa</text>
<text x="40" y="660" text-anchor="start" style="fill:var(--c-text);font-size:12px">gizleyecek iş kalmaz.</text>
<rect x="36.0" y="674.0" width="488.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="689" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Mutfak benzetmesi: SM bir hat aşçısıdır; register ve SMEM onun tezgâhıdır.</text>
<line x1="280.0" y1="709.0" x2="280.0" y2="731.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="292" y="724" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 132 SM = 1 GPU</text>
<rect x="28" y="734" width="504" height="348" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="758" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">2 · GPU: H100 SXM5 çipi</text>
<rect x="40" y="770" width="228" height="180" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="46.0" y="778.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="807.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="46.0" y="834.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="46.0" y="890.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="919.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="778.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.0" y="807.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="834.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.0" y="863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="890.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.04;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="80" y="776" width="148" height="168" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="88" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<rect x="100" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="782" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="793" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="804" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="815" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="826" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="837" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="872" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="883" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="894" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="905" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="916" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="927" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88.0" y="850.0" width="129.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152.5" y="863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · 50 MB</text>
<rect x="282" y="770" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="292" y="792" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">132</text>
<text x="292" y="809" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SM</text>
<rect x="407" y="770" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="417" y="792" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">990</text>
<text x="417" y="809" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (BF16)</text>
<rect x="282" y="826" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="292" y="848" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">80 GB</text>
<text x="292" y="865" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM3 bellek</text>
<rect x="407" y="826" width="117" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="417" y="848" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">3,35 TB/s</text>
<text x="417" y="865" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM hızı</text>
<text x="282" y="894" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">814 mm² kalıp · 80 milyar transistör</text>
<text x="282" y="910" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">6 HBM yuvası, 5&#x27;i etkin · 700 W</text>
<text x="282" y="926" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ortak L2: 50 MB (~5,5 TB/s)</text>
<text x="282" y="942" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FP8: 1.980 TFLOP/s (yoğun)</text>
<text x="282" y="958" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(B200: 148 SM · 192 GB · 8 TB/s)</text>
<text x="40" y="984" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="40" y="1002" text-anchor="start" style="fill:var(--c-text);font-size:12px">Çip saniyede 990 trilyon işlem yapabilir ama bellekten saniyede yalnızca</text>
<text x="40" y="1019" text-anchor="start" style="fill:var(--c-text);font-size:12px">3,35 TB okuyabilir. Okunan her bayt için ~295 işlem yapılmazsa Tensor</text>
<text x="40" y="1036" text-anchor="start" style="fill:var(--c-text);font-size:12px">Core&#x27;lar boşta kalır.</text>
<rect x="36.0" y="1050.0" width="488.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="1065" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">132 hat aşçısı tek mutfakta çalışır; HBM mutfağın kileri, L2 ortak raftır.</text>
<line x1="280.0" y1="1085.0" x2="280.0" y2="1107.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="292" y="1100" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 8 GPU = 1 düğüm</text>
<rect x="28" y="1110" width="504" height="2358" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1134" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">3 · Düğüm: DGX H100 kasasının içi</text>
<rect x="36" y="1146" width="488" height="418" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="44.0" y="1158.0" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="76.0" y="1172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="76.0" y="1186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 veri</text>
<rect x="130.0" y="1158.0" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="158.0" y="1172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="158.0" y="1186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="206.0" y="1158.0" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="1172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="239.0" y="1186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çift port</text>
<rect x="288.0" y="1158.0" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="321.0" y="1172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="321.0" y="1186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çift port</text>
<rect x="374.0" y="1158.0" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="402.0" y="1172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="402.0" y="1186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="452.0" y="1158.0" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="484.0" y="1172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="484.0" y="1186" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 veri</text>
<line x1="109.0" y1="1176.0" x2="128.0" y2="1176.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="187.0" y1="1176.0" x2="204.0" y2="1176.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="356.0" y1="1176.0" x2="372.0" y2="1176.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="432.0" y1="1176.0" x2="450.0" y2="1176.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="158.0" y1="1196.0" x2="158.0" y2="1212.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="402.0" y1="1196.0" x2="402.0" y2="1212.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="120.0" y="1214.0" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158" y="1234" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="158" y="1250" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="364.0" y="1214.0" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="402" y="1234" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="402" y="1250" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="210.0" y="1216.0" width="62.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="241.0" y="1230.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 × M.2 OS</text>
<line x1="198.0" y1="1226.0" x2="208.0" y2="1226.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="288.0" y="1216.0" width="62.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="319.0" y="1230.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<line x1="352.0" y1="1226.0" x2="362.0" y2="1226.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="199.0" y1="1248.0" x2="361.0" y2="1248.0" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<path d="M196 1258 H280 V1480" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="40" y="1286" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="286" y="1286" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<path d="M132 1262 V1270 H69 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M148 1262 V1277 H127 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M164 1262 V1277 H185 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M180 1262 V1270 H243 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M376 1262 V1270 H317 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M392 1262 V1277 H375 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M408 1262 V1277 H433 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M424 1262 V1270 H491 V1292" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="46.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="69.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="69.0" y1="1320.0" x2="69.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="44.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.34;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="69.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="104.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="127.0" y1="1320.0" x2="127.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="102.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="127.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="162.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="185.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="185.0" y1="1320.0" x2="185.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="160.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="185.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="220.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="243.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="243.0" y1="1320.0" x2="243.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="218.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="243.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="294.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="317.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="317.0" y1="1320.0" x2="317.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="292.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<rect x="352.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="375.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="375.0" y1="1320.0" x2="375.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="350.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<rect x="410.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="433.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="433.0" y1="1320.0" x2="433.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="408.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="433.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<rect x="468.0" y="1294.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="491.0" y="1310.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="491.0" y1="1320.0" x2="491.0" y2="1334.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="466.0" y="1336.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="491.0" y="1356.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<line x1="49.0" y1="1370.0" x2="49.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="54.0" y1="1370.0" x2="54.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="59.0" y1="1370.0" x2="59.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="64.0" y1="1370.0" x2="64.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="69.0" y1="1370.0" x2="69.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="74.0" y1="1370.0" x2="74.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="79.0" y1="1370.0" x2="79.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="84.0" y1="1370.0" x2="84.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="89.0" y1="1370.0" x2="89.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="107.0" y1="1370.0" x2="107.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="112.0" y1="1370.0" x2="112.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="117.0" y1="1370.0" x2="117.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="122.0" y1="1370.0" x2="122.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="127.0" y1="1370.0" x2="127.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="132.0" y1="1370.0" x2="132.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="137.0" y1="1370.0" x2="137.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="142.0" y1="1370.0" x2="142.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="147.0" y1="1370.0" x2="147.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="165.0" y1="1370.0" x2="165.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="170.0" y1="1370.0" x2="170.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="175.0" y1="1370.0" x2="175.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="180.0" y1="1370.0" x2="180.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="185.0" y1="1370.0" x2="185.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="190.0" y1="1370.0" x2="190.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="195.0" y1="1370.0" x2="195.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="200.0" y1="1370.0" x2="200.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="205.0" y1="1370.0" x2="205.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="223.0" y1="1370.0" x2="223.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="228.0" y1="1370.0" x2="228.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="233.0" y1="1370.0" x2="233.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="238.0" y1="1370.0" x2="238.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="243.0" y1="1370.0" x2="243.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="248.0" y1="1370.0" x2="248.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="253.0" y1="1370.0" x2="253.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="258.0" y1="1370.0" x2="258.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="263.0" y1="1370.0" x2="263.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="297.0" y1="1370.0" x2="297.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="302.0" y1="1370.0" x2="302.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="307.0" y1="1370.0" x2="307.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="312.0" y1="1370.0" x2="312.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="317.0" y1="1370.0" x2="317.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="322.0" y1="1370.0" x2="322.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="327.0" y1="1370.0" x2="327.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="332.0" y1="1370.0" x2="332.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="337.0" y1="1370.0" x2="337.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="355.0" y1="1370.0" x2="355.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="360.0" y1="1370.0" x2="360.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="365.0" y1="1370.0" x2="365.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="370.0" y1="1370.0" x2="370.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="375.0" y1="1370.0" x2="375.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="380.0" y1="1370.0" x2="380.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="385.0" y1="1370.0" x2="385.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="390.0" y1="1370.0" x2="390.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="395.0" y1="1370.0" x2="395.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="413.0" y1="1370.0" x2="413.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="418.0" y1="1370.0" x2="418.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="423.0" y1="1370.0" x2="423.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="428.0" y1="1370.0" x2="428.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="433.0" y1="1370.0" x2="433.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="438.0" y1="1370.0" x2="438.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="443.0" y1="1370.0" x2="443.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="448.0" y1="1370.0" x2="448.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="453.0" y1="1370.0" x2="453.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="471.0" y1="1370.0" x2="471.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="476.0" y1="1370.0" x2="476.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="481.0" y1="1370.0" x2="481.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="486.0" y1="1370.0" x2="486.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="491.0" y1="1370.0" x2="491.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="496.0" y1="1370.0" x2="496.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="501.0" y1="1370.0" x2="501.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="506.0" y1="1370.0" x2="506.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="511.0" y1="1370.0" x2="511.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="44.0" y1="1408.0" x2="516.0" y2="1408.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<line x1="78.0" y1="1408.0" x2="78.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="84.0" y1="1408.0" x2="84.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="90.0" y1="1408.0" x2="90.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="96.0" y1="1408.0" x2="96.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="102.0" y1="1408.0" x2="102.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="108.0" y1="1408.0" x2="108.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="114.0" y1="1408.0" x2="114.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="52.0" y="1430.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="1447.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 0</text>
<line x1="182.0" y1="1408.0" x2="182.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="188.0" y1="1408.0" x2="188.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="194.0" y1="1408.0" x2="194.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="200.0" y1="1408.0" x2="200.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="206.0" y1="1408.0" x2="206.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="212.0" y1="1408.0" x2="212.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="218.0" y1="1408.0" x2="218.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="156.0" y="1430.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="200.0" y="1447.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 1</text>
<line x1="342.0" y1="1408.0" x2="342.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="348.0" y1="1408.0" x2="348.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="354.0" y1="1408.0" x2="354.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="360.0" y1="1408.0" x2="360.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="366.0" y1="1408.0" x2="366.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="372.0" y1="1408.0" x2="372.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="378.0" y1="1408.0" x2="378.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="316.0" y="1430.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="360.0" y="1447.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 2</text>
<line x1="446.0" y1="1408.0" x2="446.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="452.0" y1="1408.0" x2="452.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="458.0" y1="1408.0" x2="458.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="464.0" y1="1408.0" x2="464.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="470.0" y1="1408.0" x2="470.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="476.0" y1="1408.0" x2="476.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="482.0" y1="1408.0" x2="482.0" y2="1430.0" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="420.0" y="1430.0" width="88.0" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="464.0" y="1447.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch 3</text>
<rect x="250.0" y="1482.0" width="60.0" height="30" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="280" y="1495" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="280" y="1508" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<path d="M96 1456 V1504 H248" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M200 1456 V1491 H248" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M360 1456 V1491 H312" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M464 1456 V1504 H312" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="280" y="1528" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">PCIe: CPU 0 → NVSwitch&#x27;ler (yönetim)</text>
<line x1="48.0" y1="1548.0" x2="72.0" y2="1548.0" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="78" y="1552" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe</text>
<line x1="126.8" y1="1548.0" x2="150.8" y2="1548.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="156.8" y="1552" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink</text>
<line x1="218.0" y1="1548.0" x2="242.0" y2="1548.0" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="248.0" y="1552" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">CPU iletişimi (UPI)</text>
<circle cx="102" cy="1238" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="102" y="1242" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<circle cx="52" cy="1274" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="52" y="1278" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="98" cy="1394" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="98" y="1398" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="239" cy="1205" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="239" y="1209" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="280" cy="1248" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="1252" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<circle cx="320" cy="1469" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="320" y="1473" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="36" y="1574" width="488" height="733" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Lejant: şemadaki her parça ne?</text>
<text x="48" y="1614" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Adlar NVIDIA&#x27;nın diyagram lejantından, renkler posterin sütun renkleridir.</text>
<text x="48" y="1629" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Görevler DGX H100 kullanım kılavuzundan gelir; “olası” yazanlar çıkarımdır.</text>
<rect x="48.0" y="1643.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1657.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CX-7</text>
<text x="126" y="1657" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7” · üstte, 2 adet</text>
<text x="126" y="1673" text-anchor="start" style="fill:var(--c-text);font-size:11px">Çift portlu ağ kartı; kılavuza göre depolama ağı ve in-band yönetim</text>
<text x="126" y="1688" text-anchor="start" style="fill:var(--c-text);font-size:11px">için. Her biri üstteki bir PCIe switch&#x27;e bağlı.</text>
<rect x="48.0" y="1702.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="1716.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0–7</text>
<text x="126" y="1716" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7 Network Module” · 8 adet</text>
<text x="126" y="1732" text-anchor="start" style="fill:var(--c-text);font-size:11px">Tek portlu 400 Gb/s hesaplama ağı kartları, GPU başına bir tane;</text>
<text x="126" y="1747" text-anchor="start" style="fill:var(--c-text);font-size:11px">kümeye 4 OSFP portundan çıkar. Gri çerçeveler bu kartları taşıyan</text>
<text x="126" y="1762" text-anchor="start" style="fill:var(--c-text);font-size:11px">modüllerdir (olası).</text>
<rect x="48.0" y="1776.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1790.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe</text>
<text x="126" y="1790" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · üstte, 4 + 4</text>
<text x="126" y="1806" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri diskleri: 8 × 3,84 TB U.2 NVMe, RAID 0 önbellek; okuma ve</text>
<text x="126" y="1821" text-anchor="start" style="fill:var(--c-text);font-size:11px">kontrol noktası için yerel alan.</text>
<rect x="48.0" y="1835.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1849.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">M.2 OS</text>
<text x="126" y="1849" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · CPU 0&#x27;ın yanında</text>
<text x="126" y="1865" text-anchor="start" style="fill:var(--c-text);font-size:11px">Büyük olasılıkla 2 × 1,92 TB M.2 işletim sistemi diski (RAID 1);</text>
<text x="126" y="1880" text-anchor="start" style="fill:var(--c-text);font-size:11px">diyagram veri diskleriyle aynı simgeyi kullanır.</text>
<rect x="48.0" y="1894.0" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="1908.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe sw</text>
<text x="126" y="1908" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe Switches” · 3 adet</text>
<text x="126" y="1924" text-anchor="start" style="fill:var(--c-text);font-size:11px">Bir CPU hattını birçok cihaza dağıtır: üstteki ikisi NVMe ve CX-7&#x27;yi</text>
<text x="126" y="1939" text-anchor="start" style="fill:var(--c-text);font-size:11px">taşır, alttaki NVSwitch&#x27;leri CPU 0&#x27;a bağlar.</text>
<rect x="48.0" y="1953.0" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="1967.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<text x="126" y="1967" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “100 GbE” · CPU 1&#x27;e bağlı</text>
<text x="126" y="1983" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ek bir Ethernet kartı; görevi diyagramda yazmıyor (olası: servis ya</text>
<text x="126" y="1998" text-anchor="start" style="fill:var(--c-text);font-size:11px">da yönetim ağı).</text>
<rect x="48.0" y="2012.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="2026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVSwitch</text>
<text x="126" y="2026" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVSwitch” · 4 adet</text>
<text x="126" y="2042" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;lar arası NVLink anahtarı; her GPU 18 bağlantısını bu dördüne 5</text>
<text x="126" y="2057" text-anchor="start" style="fill:var(--c-text);font-size:11px">+ 4 + 4 + 5 olarak dağıtır.</text>
<rect x="48.0" y="2071.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="81.0" y="2085.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CPU</text>
<text x="126" y="2085" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU 0 / CPU 1”</text>
<text x="126" y="2101" text-anchor="start" style="fill:var(--c-text);font-size:11px">Intel Xeon 8480C (56&#x27;şar çekirdek); her biri dört GPU&#x27;yu ve onların</text>
<text x="126" y="2116" text-anchor="start" style="fill:var(--c-text);font-size:11px">NIC&#x27;lerini PCIe ile besler.</text>
<line x1="48.0" y1="2140.0" x2="114.0" y2="2140.0" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="126" y="2144" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU communication”</text>
<text x="126" y="2160" text-anchor="start" style="fill:var(--c-text);font-size:11px">İki soketi birbirine bağlayan hat; Intel Xeon&#x27;da bu bağlantının adı</text>
<text x="126" y="2175" text-anchor="start" style="fill:var(--c-text);font-size:11px">UPI&#x27;dir.</text>
<line x1="48.0" y1="2199.0" x2="114.0" y2="2199.0" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="126" y="2203" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe”</text>
<text x="126" y="2219" text-anchor="start" style="fill:var(--c-text);font-size:11px">CPU&#x27;ları NIC&#x27;lere, GPU&#x27;lara, disklere ve NVSwitch&#x27;lere bağlayan</text>
<text x="126" y="2234" text-anchor="start" style="fill:var(--c-text);font-size:11px">genel yol (PCIe Gen5).</text>
<line x1="48.0" y1="2258.0" x2="114.0" y2="2258.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="126" y="2262" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA lejantında yok: NVLink</text>
<text x="126" y="2278" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;ların altındaki renkli demetler NVLink hatlarıdır: GPU başına 18</text>
<text x="126" y="2293" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlantı, toplam 900 GB/s çift yön.</text>
<rect x="40" y="2319" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50" y="2341" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">8</text>
<text x="50" y="2358" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU · 640 GB</text>
<rect x="162" y="2319" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="172" y="2341" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">450 GB/s</text>
<text x="172" y="2358" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink, tek yön</text>
<rect x="284" y="2319" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="2341" text-anchor="start" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">64 GB/s</text>
<text x="294" y="2358" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe, CPU yolu</text>
<rect x="406" y="2319" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="2341" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">50 GB/s</text>
<text x="416" y="2358" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağ kartı / GPU</text>
<text x="40" y="2393" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Topolojiyi okuma: altı soru</text>
<text x="40" y="2412" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Şemadaki numaralı daireleri sırayla izleyin; her biri bir soruyu cevaplar.</text>
<circle cx="49" cy="2438" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2442" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="2442" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Kim kimi besler?</text>
<text x="66" y="2462" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğüm birbirinin aynası olan iki yarıdan oluşur. CPU 0, GPU 0–3&#x27;ü ve NIC</text>
<text x="66" y="2479" text-anchor="start" style="fill:var(--c-text);font-size:12px">0–3&#x27;ü PCIe ile besler; CPU 1 de GPU 4–7&#x27;yi ve NIC 4–7&#x27;yi. Her yarının</text>
<text x="66" y="2496" text-anchor="start" style="fill:var(--c-text);font-size:12px">kendi belleği vardır; buna NUMA alanı denir.</text>
<text x="66" y="2515" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> GPU 5&#x27;e veri hazırlayan süreç CPU 1&#x27;de koşmalı ve CPU 1&#x27;in</text>
<text x="66" y="2532" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">belleğini kullanmalıdır; yoksa her batch önce CPU&#x27;lar arasındaki hattı</text>
<text x="66" y="2549" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">geçer.</text>
<circle cx="49" cy="2575" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2579" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="2579" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">GPU ağa nasıl çıkar?</text>
<text x="66" y="2599" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU&#x27;nun hemen üstünde kendine ait 400 Gb/s&#x27;lik bir NIC durur (GPU i ↔</text>
<text x="66" y="2616" text-anchor="start" style="fill:var(--c-text);font-size:12px">NIC i). GPUDirect RDMA sayesinde veri GPU belleğinden doğrudan NIC&#x27;e</text>
<text x="66" y="2633" text-anchor="start" style="fill:var(--c-text);font-size:12px">geçer, CPU belleğine hiç uğramaz.</text>
<text x="66" y="2652" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> kasadan 8 × 50 = 400 GB/s çıkar. Kümede NIC i, i. leaf switch&#x27;e</text>
<text x="66" y="2669" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">takılır; farklı düğümlerdeki aynı numaralı GPU&#x27;lar bu yüzden tek</text>
<text x="66" y="2686" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sıçramada buluşur.</text>
<circle cx="49" cy="2712" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2716" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="2716" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">GPU&#x27;lar birbirleriyle nasıl konuşur?</text>
<text x="66" y="2736" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU&#x27;nun 18 NVLink&#x27;i aşağıdaki dört NVSwitch&#x27;e iner: ikisine 5&#x27;er,</text>
<text x="66" y="2753" text-anchor="start" style="fill:var(--c-text);font-size:12px">ikisine 4&#x27;er hat. Herhangi iki GPU tek bir switch sıçramasıyla, her yönde</text>
<text x="66" y="2770" text-anchor="start" style="fill:var(--c-text);font-size:12px">450 GB/s ile konuşur.</text>
<text x="66" y="2789" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> bu yol ağdan 9 kat geniştir; her katmanda konuşan tensör</text>
<text x="66" y="2806" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">paralelliği bu yüzden bu sekiz GPU&#x27;nun içinde tutulur.</text>
<circle cx="49" cy="2832" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2836" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="66" y="2836" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Depolama trafiği nereden geçer?</text>
<text x="66" y="2856" text-anchor="start" style="fill:var(--c-text);font-size:12px">Üstteki iki PCIe switch, çift portlu CX-7 kartlarını (depolama ağı ve</text>
<text x="66" y="2873" text-anchor="start" style="fill:var(--c-text);font-size:12px">in-band yönetim) ve 4&#x27;er U.2 NVMe veri diskini taşır. CPU 0&#x27;ın yanında</text>
<text x="66" y="2890" text-anchor="start" style="fill:var(--c-text);font-size:12px">ayrıca iki M.2 işletim sistemi diski durur.</text>
<text x="66" y="2909" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> veri okumak ve kontrol noktası yazmak, gradyan trafiğinden</text>
<text x="66" y="2926" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">farklı kartları ve kabloları kullanır; büyük bir kayıt AllReduce&#x27;u</text>
<text x="66" y="2943" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">yavaşlatmaz.</text>
<circle cx="49" cy="2969" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2973" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="66" y="2973" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">İki yarıyı ne birleştirir?</text>
<text x="66" y="2993" text-anchor="start" style="fill:var(--c-text);font-size:12px">İki CPU, NVIDIA&#x27;nın “CPU communication” dediği hatla (Xeon&#x27;da UPI)</text>
<text x="66" y="3010" text-anchor="start" style="fill:var(--c-text);font-size:12px">bağlıdır. Bu hat CPU&#x27;ların kendi trafiği içindir; iki yarıdaki GPU&#x27;lar</text>
<text x="66" y="3027" text-anchor="start" style="fill:var(--c-text);font-size:12px">birbirleriyle CPU&#x27;lar üzerinden değil, NVLink üzerinden konuşur.</text>
<text x="66" y="3046" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> bir CPU öbür yarının belleğine erişebilir ama daha yavaş; NUMA</text>
<text x="66" y="3063" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sabitlemesi bu cezadan kaçınmak içindir.</text>
<circle cx="49" cy="3089" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="3093" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="66" y="3093" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVSwitch&#x27;leri kim yönetir?</text>
<text x="66" y="3113" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU 0&#x27;dan inen PCIe hattı, alttaki PCIe switch üzerinden dört NVSwitch&#x27;e</text>
<text x="66" y="3130" text-anchor="start" style="fill:var(--c-text);font-size:12px">ulaşır. Bu bir kontrol yoludur: NVIDIA Fabric Manager switch&#x27;lerin</text>
<text x="66" y="3147" text-anchor="start" style="fill:var(--c-text);font-size:12px">yönlendirme tablolarını buradan kurar; GPU verisi bu hattan akmaz.</text>
<text x="66" y="3166" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> Fabric Manager çalışmazsa GPU&#x27;lar tek tek sağlıklı görünse bile</text>
<text x="66" y="3183" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">NVLink ağı kurulmaz ve çok GPU&#x27;lu işler başlamaz.</text>
<rect x="36.0" y="3205.0" width="488.0" height="102" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="48" y="3225" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Örnek yol: bir gradyanın düğümdeki yolculuğu</text>
<text x="48" y="3243" text-anchor="start" style="fill:var(--c-text);font-size:12px">Komşu düğümden gelen bir gradyan NIC 5&#x27;e girer ve GPUDirect RDMA ile</text>
<text x="48" y="3260" text-anchor="start" style="fill:var(--c-text);font-size:12px">doğrudan GPU 5&#x27;in belleğine yazılır; CPU 1&#x27;e hiç uğramaz. GPU 5 bu</text>
<text x="48" y="3277" text-anchor="start" style="fill:var(--c-text);font-size:12px">gradyanı GPU 2 ile paylaşacaksa ağa ya da CPU&#x27;lara değil, NVLink üzerinden</text>
<text x="48" y="3294" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch&#x27;lere gider.</text>
<text x="40" y="3331" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 NVSwitch · biseksiyon 3,6 TB/s · 2 × Xeon 8480C (112 çekirdek), 2 TB RAM ·</text>
<text x="40" y="3346" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 × 3,84 TB NVMe veri · güç 10,2 kW</text>
<text x="40" y="3370" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="40" y="3388" text-anchor="start" style="fill:var(--c-text);font-size:12px">Kasa içinde GPU&#x27;lar NVLink ile birbirinin belleğine doğrudan erişir. CPU&#x27;ya</text>
<text x="40" y="3405" text-anchor="start" style="fill:var(--c-text);font-size:12px">giden PCIe yolu 7 kat, kasadan çıkan ağ yolu 9 kat daha dardır; bu yüzden en</text>
<text x="40" y="3422" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoğun konuşma kasanın içinde tutulur.</text>
<rect x="36.0" y="3436.0" width="488.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="3451" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">8 mutfaklı bir restoran: NVLink mutfakları birbirine bağlayan iç koridordur.</text>
<line x1="280.0" y1="3471.0" x2="280.0" y2="3493.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="292" y="3486" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 32 düğüm = 1 SU · × 4 SU = SuperPOD</text>
<rect x="28" y="3496" width="504" height="1993" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="3520" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4 · Küme: DGX SuperPOD</text>
<text x="40" y="3542" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Hiyerarşiyi hatırla: parçadan bütüne</text>
<rect x="40.0" y="3552.0" width="108.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="94.0" y="3566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="94.0" y="3580" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">tek GPU modülü</text>
<line x1="150.0" y1="3572.0" x2="162.0" y2="3572.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="164.0" y="3552.0" width="108.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="218.0" y="3566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="218.0" y="3580" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPU&#x27;lu sunucu</text>
<line x1="274.0" y1="3572.0" x2="286.0" y2="3572.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="288.0" y="3552.0" width="108.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="342.0" y="3566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="342.0" y="3580" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 DGX = 256 GPU</text>
<line x1="398.0" y1="3572.0" x2="410.0" y2="3572.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="412.0" y="3552.0" width="108.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="466.0" y="3566" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="466.0" y="3580" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 SU = 127 DGX</text>
<text x="40" y="3620" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Leaf ve spine nedir?</text>
<rect x="40" y="3632" width="236" height="165" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="48.0" y="3640.0" width="220.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="158.0" y="3655.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf (yaprak) switch</text>
<text x="52" y="3682" text-anchor="start" style="fill:var(--c-text);font-size:12px">Sunucuların doğrudan takıldığı alt</text>
<text x="52" y="3699" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmandır; ağacın uçları, yani</text>
<text x="52" y="3716" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;lara en yakın switch&#x27;lerdir.</text>
<text x="52" y="3739" text-anchor="start" style="fill:var(--c-text);font-size:12px">SuperPOD&#x27;da: SU başına 8 tane, her</text>
<text x="52" y="3756" text-anchor="start" style="fill:var(--c-text);font-size:12px">ray için bir tane. 64 portunun</text>
<text x="52" y="3773" text-anchor="start" style="fill:var(--c-text);font-size:12px">32&#x27;si DGX&#x27;lere iner, 32&#x27;si</text>
<text x="52" y="3790" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine&#x27;lara çıkar.</text>
<rect x="284" y="3632" width="236" height="165" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="292.0" y="3640.0" width="220.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="402.0" y="3655.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Spine (omurga) switch</text>
<text x="296" y="3682" text-anchor="start" style="fill:var(--c-text);font-size:12px">Leaf&#x27;leri birbirine bağlayan üst</text>
<text x="296" y="3699" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmandır; spine&#x27;a hiçbir sunucu</text>
<text x="296" y="3716" text-anchor="start" style="fill:var(--c-text);font-size:12px">takılmaz.</text>
<text x="296" y="3739" text-anchor="start" style="fill:var(--c-text);font-size:12px">SuperPOD&#x27;da: toplam 16 tane; her</text>
<text x="296" y="3756" text-anchor="start" style="fill:var(--c-text);font-size:12px">biri 32 leaf&#x27;in hepsine ikişer</text>
<text x="296" y="3773" text-anchor="start" style="fill:var(--c-text);font-size:12px">hatla bağlıdır ve 64 portu böyle</text>
<text x="296" y="3790" text-anchor="start" style="fill:var(--c-text);font-size:12px">dolar.</text>
<text x="40" y="3819" text-anchor="start" style="fill:var(--c-text);font-size:12px">İki katmanın kuralı basittir: her leaf her spine&#x27;a bağlanır. Böylece herhangi</text>
<text x="40" y="3836" text-anchor="start" style="fill:var(--c-text);font-size:12px">iki sunucu en fazla leaf → spine → leaf yoluyla buluşur ve küme büyüdükçe yol</text>
<text x="40" y="3853" text-anchor="start" style="fill:var(--c-text);font-size:12px">uzamaz. Sunucu eklemek için leaf, bant genişliği eklemek için spine eklenir.</text>
<text x="40" y="3870" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bu düzen telefon santrallerinden gelir (Clos, 1953); yukarı çıktıkça</text>
<text x="40" y="3887" text-anchor="start" style="fill:var(--c-text);font-size:12px">bağlantıların kalınlaştığı hâline fat-tree, yani şişman ağaç denir</text>
<text x="40" y="3904" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Leiserson, 1985).</text>
<text x="40" y="3926" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Benzetme: leaf bir mahallenin santralidir, spine şehirler arasındaki ana hat.</text>
<text x="40" y="3962" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Ne neyi yapar?</text>
<rect x="40.0" y="3976.0" width="96.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="3992.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="148" y="3992" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Görevi:</tspan> modelin matematiğini yapar; 132 SM&#x27;deki Tensor</text>
<text x="148" y="4009" text-anchor="start" style="fill:var(--c-text);font-size:12px">Core&#x27;lar matris çarpımlarını yürütür.</text>
<text x="148" y="4028" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">İçinde:</tspan> ağırlıkların, aktivasyonların ve optimizer durumunun</text>
<text x="148" y="4045" text-anchor="start" style="fill:var(--c-text);font-size:12px">bu GPU&#x27;ya düşen payı, 80 GB HBM3&#x27;te.</text>
<text x="148" y="4064" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Bağlantı ve paralellik:</tspan> 18 NVLink ile kasadaki diğer</text>
<text x="148" y="4081" text-anchor="start" style="fill:var(--c-text);font-size:12px">7 GPU&#x27;ya, PCIe ile CPU&#x27;ya ve kendi NIC&#x27;ine ulaşır; her</text>
<text x="148" y="4098" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;nun bir TP, bir PP ve bir DP koordinatı vardır.</text>
<line x1="40.0" y1="4108.0" x2="520.0" y2="4108.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="4122.0" width="96.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="4138.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="148" y="4138" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Görevi:</tspan> 8 GPU&#x27;yu tek bir NVLink alanında birleştirir; iki</text>
<text x="148" y="4155" text-anchor="start" style="fill:var(--c-text);font-size:12px">Xeon işlemci işletim sistemini, veri yükleyiciyi ve GPU&#x27;lara</text>
<text x="148" y="4172" text-anchor="start" style="fill:var(--c-text);font-size:12px">iş gönderen süreçleri koşturur.</text>
<text x="148" y="4191" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">İçinde:</tspan> HGX taban kartında 8 GPU ve 4 NVSwitch; ayrıca 2 TB</text>
<text x="148" y="4208" text-anchor="start" style="fill:var(--c-text);font-size:12px">RAM, 8 × 3,84 TB NVMe önbellek ve 8 + 2 ağ kartı.</text>
<text x="148" y="4227" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Bağlantı ve paralellik:</tspan> tensör paralelliği (TP 8) bu kasanın</text>
<text x="148" y="4244" text-anchor="start" style="fill:var(--c-text);font-size:12px">içinde kalır; kasa ağa her GPU için bir ray olmak üzere 8</text>
<text x="148" y="4261" text-anchor="start" style="fill:var(--c-text);font-size:12px">rayla çıkar.</text>
<line x1="40.0" y1="4271.0" x2="520.0" y2="4271.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="4285.0" width="96.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="4301.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="148" y="4301" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Görevi:</tspan> kurulumun ve ağın tekrar eden birimidir: 32 DGX</text>
<text x="148" y="4318" text-anchor="start" style="fill:var(--c-text);font-size:12px">(8 kabin × 4) ve onları bağlayan 8 leaf switch.</text>
<text x="148" y="4337" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">İçinde:</tspan> 256 GPU; her ray tek bir leaf&#x27;tir, yani Leaf-xx-0i</text>
<text x="148" y="4354" text-anchor="start" style="fill:var(--c-text);font-size:12px">bütün düğümlerin NIC i&#x27;sini toplar.</text>
<text x="148" y="4373" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Bağlantı ve paralellik:</tspan> aynı numaralı GPU&#x27;lar tek sıçramada</text>
<text x="148" y="4390" text-anchor="start" style="fill:var(--c-text);font-size:12px">buluşur; boru hattı ve veri paralelliğinin düğümler arası</text>
<text x="148" y="4407" text-anchor="start" style="fill:var(--c-text);font-size:12px">trafiği önce burada, ray ray akar.</text>
<line x1="40.0" y1="4417.0" x2="520.0" y2="4417.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="4431.0" width="96.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="88.0" y="4447.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="148" y="4447" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Görevi:</tspan> SU&#x27;ları spine katmanıyla tek bir engelsiz ağda</text>
<text x="148" y="4464" text-anchor="start" style="fill:var(--c-text);font-size:12px">birleştirir; yanına ayrı depolama ve yönetim ağlarını, UFM&#x27;i</text>
<text x="148" y="4481" text-anchor="start" style="fill:var(--c-text);font-size:12px">ve Base Command ile Slurm gibi yazılımları ekler.</text>
<text x="148" y="4500" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">İçinde:</tspan> 127 DGX = 1.016 GPU; tek bir eğitim işi bütün kümeye</text>
<text x="148" y="4517" text-anchor="start" style="fill:var(--c-text);font-size:12px">yayılabilir.</text>
<text x="148" y="4536" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent-2);font-weight:700">Bağlantı ve paralellik:</tspan> veri paralelliği bütün kümeye</text>
<text x="148" y="4553" text-anchor="start" style="fill:var(--c-text);font-size:12px">yayılır, raylar arası trafik bir spine üzerinden döner;</text>
<text x="148" y="4570" text-anchor="start" style="fill:var(--c-text);font-size:12px">Slurm işi düğümlere yerleştirir ve arızalı düğümü devreden</text>
<text x="148" y="4587" text-anchor="start" style="fill:var(--c-text);font-size:12px">çıkarır.</text>
<text x="40" y="4613" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">H100 SXM5 modülleri anakarta değil, 8 GPU&#x27;yu ve 4 NVSwitch&#x27;i taşıyan HGX taban</text>
<text x="40" y="4628" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">kartına takılır; DGX H100&#x27;ün iki işlemcisi Intel Xeon 8480C&#x27;dir. SuperPOD 32 DGX&#x27;lik</text>
<text x="40" y="4643" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">birimlerden (SU) kurulur; referans tasarım 4 SU&#x27;dur.</text>
<rect x="40.0" y="4655.0" width="480.0" height="118" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="4675" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Neden 128 değil 127?</tspan> Kâğıt üstünde 4 SU × 32 = 128 DGX eder. NVIDIA&#x27;nın</text>
<text x="52" y="4692" text-anchor="start" style="fill:var(--c-text);font-size:12px">yerleşim düzeninde ise bir çift UFM cihazı bir DGX&#x27;in yerini alır</text>
<text x="52" y="4709" text-anchor="start" style="fill:var(--c-text);font-size:12px">(NVIDIA&#x27;nın şemasında SU 4&#x27;teki C-UFM 1 ve 2); tam bir SuperPOD bu yüzden</text>
<text x="52" y="4726" text-anchor="start" style="fill:var(--c-text);font-size:12px">en fazla 127 DGX, yani 1.016 GPU&#x27;dur. Kablo tablosu da bunu gösterir:</text>
<text x="52" y="4743" text-anchor="start" style="fill:var(--c-text);font-size:12px">127 × 8 + 4 UFM = 1.020 kablo. Daha küçük kurulumlar tek SU&#x27;dan başlar</text>
<text x="52" y="4760" text-anchor="start" style="fill:var(--c-text);font-size:12px">(31–127 DGX).</text>
<rect x="36" y="4785" width="488" height="415" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="122.0" y="4799.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="160.0" y="4814.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-01</text>
<rect x="206.0" y="4799.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="244.0" y="4814.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-02</text>
<rect x="338.0" y="4799.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="376.0" y="4814.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-16</text>
<text x="309" y="4814" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700">⋯</text>
<text x="424" y="4814" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700</text>
<rect x="40" y="4851" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="46.0" y="4863.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="77.0" y="4877.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-01</text>
<rect x="114.0" y="4863.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="145.0" y="4877.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-02</text>
<rect x="200.0" y="4863.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="231.0" y="4877.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-08</text>
<text x="188" y="4878" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="83.0" y1="4937.0" x2="77.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="83.0" y1="4937.0" x2="145.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="83.0" y1="4937.0" x2="231.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="48" y="4937" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="83" y="4953" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-001</text>
<text x="83" y="4968" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="169.0" y1="4937.0" x2="77.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="169.0" y1="4937.0" x2="145.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="169.0" y1="4937.0" x2="231.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="134" y="4937" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="169" y="4953" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-032</text>
<text x="169" y="4968" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="126" y="4960" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="50" y="5015" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 1</text>
<text x="92" y="5015" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-001 … 032 · 8 leaf</text>
<rect x="294" y="4851" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="300.0" y="4863.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="331.0" y="4877.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-01</text>
<rect x="368.0" y="4863.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="399.0" y="4877.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-02</text>
<rect x="454.0" y="4863.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="485.0" y="4877.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-08</text>
<text x="442" y="4878" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="337.0" y1="4937.0" x2="331.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="337.0" y1="4937.0" x2="399.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="337.0" y1="4937.0" x2="485.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="302" y="4937" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="337" y="4953" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-097</text>
<text x="337" y="4968" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="423.0" y1="4937.0" x2="331.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="423.0" y1="4937.0" x2="399.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="423.0" y1="4937.0" x2="485.0" y2="4883.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="388" y="4937" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="423" y="4953" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-127</text>
<text x="423" y="4968" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="380" y="4960" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="304" y="5015" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 4</text>
<text x="346" y="5015" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-097 … 127 + C-UFM</text>
<path d="M466 4952 H463 V4883" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="466.0" y="4943.0" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="492.0" y="4956.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 1</text>
<path d="M466 4976 H460 V4883" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="466.0" y="4967.0" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="492.0" y="4980.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 2</text>
<text x="280" y="4960" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700">⋯</text>
<line x1="160.0" y1="4821.0" x2="77.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="160.0" y1="4821.0" x2="145.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="160.0" y1="4821.0" x2="231.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="160.0" y1="4821.0" x2="331.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="160.0" y1="4821.0" x2="399.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="160.0" y1="4821.0" x2="485.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="244.0" y1="4821.0" x2="77.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="244.0" y1="4821.0" x2="145.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="244.0" y1="4821.0" x2="231.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="244.0" y1="4821.0" x2="331.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="244.0" y1="4821.0" x2="399.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="244.0" y1="4821.0" x2="485.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="376.0" y1="4821.0" x2="77.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="376.0" y1="4821.0" x2="145.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="376.0" y1="4821.0" x2="231.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="376.0" y1="4821.0" x2="331.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="376.0" y1="4821.0" x2="399.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="376.0" y1="4821.0" x2="485.0" y2="4863.0" style="stroke:var(--c-danger);stroke-width:1.5;stroke-opacity:.6"/>
<line x1="48.0" y1="5045.0" x2="70.0" y2="5045.0" style="stroke:var(--c-danger);stroke-width:2.4"/>
<text x="76" y="5049" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine–leaf: 2 × 400 Gb/s</text>
<line x1="284.0" y1="5045.0" x2="306.0" y2="5045.0" style="stroke:var(--c-warn);stroke-width:2.4"/>
<text x="312" y="5049" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX–leaf: 400 Gb/s</text>
<line x1="48.0" y1="5063.0" x2="70.0" y2="5063.0" style="stroke:var(--c-accent);stroke-width:2.4"/>
<text x="76" y="5067" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM–leaf: 200 Gb/s</text>
<text x="48" y="5089" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Hat renkleri NVIDIA&#x27;nın şemasındaki gibidir ve burada kablo türünü gösterir</text>
<text x="48" y="5104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(posterin sütun renkleri değil). Adlar da oradan: Leaf-SU-ray; Leaf-04-08, SU 4&#x27;ün</text>
<text x="48" y="5119" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">8. rayıdır ve her DGX&#x27;in NIC 0–7&#x27;si Leaf-xx-01 ile 08 arasına gider. SU 2 ve SU 3</text>
<text x="48" y="5134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">aynı düzeni tekrarlar.</text>
<rect x="44.0" y="5148.0" width="232.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="54" y="5164" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">depolama ağı (ayrı InfiniBand)</text>
<text x="54" y="5180" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 switch · düğüm başına ≥ 40 GB/s</text>
<rect x="284.0" y="5148.0" width="232.0" height="40" rx="5" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="294" y="5164" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">yönetim ağları (Ethernet)</text>
<text x="294" y="5180" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">in-band 100 GbE · OOB 1 GbE</text>
<rect x="40" y="5212" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50" y="5234" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">127</text>
<text x="50" y="5251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX düğümü</text>
<rect x="162" y="5212" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="172" y="5234" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1.016</text>
<text x="172" y="5251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<rect x="284" y="5212" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="5234" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">32 + 16</text>
<text x="294" y="5251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf + spine</text>
<rect x="406" y="5212" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="5234" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">4</text>
<text x="416" y="5251" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ayrı ağ</text>
<text x="40" y="5282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">127 DGX (UFM çifti SU 4&#x27;te)</text>
<text x="284" y="5282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ray içi: SU&#x27;daki 31 düğüme tek sıçrama</text>
<text x="40" y="5298" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesaplama: 32 leaf + 16 spine</text>
<text x="284" y="5298" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her leaf her spine&#x27;a 2 × 400 Gb/s</text>
<text x="40" y="5314" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:1 fat-tree · SHARPv3</text>
<text x="284" y="5314" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 ağ: hesaplama, depolama, 2 yönetim</text>
<text x="40" y="5340" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="40" y="5358" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her düğümün NIC 0&#x27;ı Leaf-xx-01&#x27;e, NIC 7&#x27;si Leaf-xx-08&#x27;e takılır; bu sekiz</text>
<text x="40" y="5375" text-anchor="start" style="fill:var(--c-text);font-size:12px">şerit raylardır ve aynı numaralı GPU&#x27;lar tek sıçramada buluşur. Her leaf de</text>
<text x="40" y="5392" text-anchor="start" style="fill:var(--c-text);font-size:12px">her spine&#x27;a ikişer 400 Gb/s hatla ulaşır; raylar arası trafik bu yüzden hep</text>
<text x="40" y="5409" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir spine üzerinden döner. Depolama ve yönetim trafiği hesaplama ağını</text>
<text x="40" y="5426" text-anchor="start" style="fill:var(--c-text);font-size:12px">meşgul etmesin diye kendi ağlarından akar. Aynı desen 64 SU&#x27;ya büyür:</text>
<text x="40" y="5443" text-anchor="start" style="fill:var(--c-text);font-size:12px">2.048 DGX, yani 16.384 H100; B ve C bölümleri bu ölçekte düşünür.</text>
<rect x="36.0" y="5457.0" width="488.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="5472" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Restoran zinciri: InfiniBand şubeler arası kurye, depolama merkez depo.</text>
<line x1="280.0" y1="5492.0" x2="280.0" y2="5514.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="292" y="5507" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">SuperPOD&#x27;un ağını yakından inceleyelim</text>
<rect x="28" y="5517" width="504" height="930" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5541" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4b · SuperPOD&#x27;u yakından incele</text>
<text x="40" y="5561" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Aynı 127 DGX&#x27;i bu kez ağ gözüyle okuyalım: ağ neden engelsiz, trafik nasıl</text>
<text x="40" y="5578" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">yol bulur ve ne kadar büyüyebilir?</text>
<text x="40" y="5608" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">1 · Ağ neden engelsiz (1:1)? Portları sayalım</text>
<rect x="40.0" y="5634.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="105.0" y="5648" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">32 DGX</text>
<text x="105.0" y="5662" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">her birinin NIC i&#x27;si</text>
<line x1="173.0" y1="5654.0" x2="212.0" y2="5654.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.8"/>
<text x="192" y="5646" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 port</text>
<rect x="215.0" y="5634.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="5648" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">leaf i · QM9700</text>
<text x="280.0" y="5662" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">64 × 400 Gb/s port</text>
<line x1="348.0" y1="5654.0" x2="387.0" y2="5654.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.8"/>
<text x="367" y="5646" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 port</text>
<rect x="390.0" y="5634.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="455.0" y="5648" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">16 spine</text>
<text x="455.0" y="5662" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">64&#x27;er port</text>
<text x="40" y="5700" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her leaf&#x27;in 64 portunun 32&#x27;si düğümlere iner, 32&#x27;si spine&#x27;lara çıkar.</text>
<text x="40" y="5717" text-anchor="start" style="fill:var(--c-text);font-size:12px">Aşağıdan ne kadar trafik girerse yukarı o kadar çıkabilir; bu yüzden aşırı</text>
<text x="40" y="5734" text-anchor="start" style="fill:var(--c-text);font-size:12px">abonelik yoktur (1:1).</text>
<rect x="40.0" y="5746.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="5761.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 leaf × 32 yukarı port = 1.024 kablo = 16 spine × 64 port</text>
<text x="40" y="5788" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Kablo sayısı (NVIDIA tablosu): düğümlerden leaf&#x27;lere 1.020 (127 × 8 + 4 UFM),</text>
<text x="40" y="5805" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">leaf&#x27;lerden spine&#x27;lara 1.024.</text>
<text x="40" y="5839" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">2 · Trafik nasıl yol bulur?</text>
<text x="40" y="5863" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı ray: A&#x27;nın GPU 3&#x27;ü → B&#x27;nin GPU 3&#x27;ü</text>
<rect x="400.3" y="5851.0" width="119.7" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="460.1" y="5864.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">1 leaf, spine yok</text>
<rect x="40.0" y="5877.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="5893.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="5889.0" x2="138.0" y2="5889.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="140.0" y="5877.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="180.0" y="5893.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="222.0" y1="5889.0" x2="238.0" y2="5889.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="5877.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="5893.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · B</text>
<text x="40" y="5929" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Farklı ray, düz yol: GPU 3 → GPU 5</text>
<rect x="412.5" y="5917.0" width="107.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="466.2" y="5930.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine&#x27;dan geçer</text>
<rect x="40.0" y="5943.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="5959.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="5955.0" x2="138.0" y2="5955.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="140.0" y="5943.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="180.0" y="5959.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="222.0" y1="5955.0" x2="238.0" y2="5955.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="5943.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="5959.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="322.0" y1="5955.0" x2="338.0" y2="5955.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="340.0" y="5943.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="380.0" y="5959.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="422.0" y1="5955.0" x2="438.0" y2="5955.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="440.0" y="5943.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="480.0" y="5959.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<text x="40" y="5995" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Farklı ray, PXN ile: önce NVLink, sonra tek leaf</text>
<rect x="449.1" y="5983.0" width="70.9" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="484.6" y="5996.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine yok</text>
<rect x="40.0" y="6009.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="6025.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="6021.0" x2="138.0" y2="6021.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="140.0" y="6009.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="180.0" y="6025.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · A</text>
<line x1="222.0" y1="6021.0" x2="238.0" y2="6021.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="6009.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="6025.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="322.0" y1="6021.0" x2="338.0" y2="6021.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="340.0" y="6009.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="380.0" y="6025.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<text x="40" y="6059" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA&#x27;ya göre bir raydaki trafik SU içindeki diğer 31 düğüme her zaman tek</text>
<text x="40" y="6076" text-anchor="start" style="fill:var(--c-text);font-size:12px">sıçramadır; raylar arası trafik ise spine katmanından geçer. NCCL 2.12&#x27;deki</text>
<text x="40" y="6093" text-anchor="start" style="fill:var(--c-text);font-size:12px">PXN, veriyi önce NVLink ile doğru raydaki GPU&#x27;ya taşıyıp spine&#x27;ı atlar;</text>
<text x="40" y="6110" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA bu sayede all2all veriminin iki katına çıktığını raporlar.</text>
<text x="40" y="6144" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">3 · Ölçek ve bant genişliği</text>
<rect x="40" y="6156" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50" y="6178" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">400 GB/s</text>
<text x="50" y="6195" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">düğüm başına</text>
<rect x="162" y="6156" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="172" y="6178" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">12,8 TB/s</text>
<text x="172" y="6195" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SU başına</text>
<rect x="284" y="6156" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="6178" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">50,8 TB/s</text>
<text x="294" y="6195" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tüm GPU&#x27;lar</text>
<rect x="406" y="6156" width="114" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="416" y="6178" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">51,2 TB/s</text>
<text x="416" y="6195" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine kapasitesi</text>
<text x="40" y="6226" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bir düğüm ağa 8 × 50 GB/s = 400 GB/s basar; 32 düğümlük bir SU 12,8 TB/s</text>
<text x="40" y="6243" text-anchor="start" style="fill:var(--c-text);font-size:12px">eder. 1.016 GPU&#x27;nun birlikte gönderebildiği (50,8 TB/s) ile 16 spine&#x27;ın</text>
<text x="40" y="6260" text-anchor="start" style="fill:var(--c-text);font-size:12px">taşıyabildiği (16 × 64 × 50 GB/s = 51,2 TB/s) neredeyse eşittir; engelsiz</text>
<text x="40" y="6277" text-anchor="start" style="fill:var(--c-text);font-size:12px">ağın sayılarla tanımı budur.</text>
<text x="40" y="6311" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">4 · Daha büyüğü: üçüncü katman</text>
<text x="40" y="6331" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA&#x27;nın tablosunda 8 SU&#x27;ya kadar iki katman (leaf + spine) yeter;</text>
<text x="40" y="6348" text-anchor="start" style="fill:var(--c-text);font-size:12px">16 SU&#x27;dan itibaren üstlerine bir core katmanı eklenir. 2.048 DGX&#x27;lik</text>
<text x="40" y="6365" text-anchor="start" style="fill:var(--c-text);font-size:12px">(16.384 H100) kurulum 512 leaf, 512 spine ve 256 core switch kullanır; B ve C</text>
<text x="40" y="6382" text-anchor="start" style="fill:var(--c-text);font-size:12px">bölümleri bu ölçekte düşünür.</text>
<text x="40" y="6408" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> NCCL kanallarını raylara göre kurar: aynı numaralı GPU&#x27;lar tek leaf</text>
<text x="40" y="6425" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">üzerinden konuşur, raylar arası trafik ise PXN ile önce NVLink&#x27;e taşınır.</text>
<line x1="280.0" y1="6450.0" x2="280.0" y2="6472.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="292" y="6465" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">kümeyi ayakta tutan yapı taşları</text>
<rect x="28" y="6475" width="504" height="1085" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="6499" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">SuperPOD&#x27;un yapı taşları: her biri neden var?</text>
<text x="40" y="6518" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVIDIA&#x27;nın 4 SU&#x27;luk referans tasarımı (127 DGX); adetler ve görevler NVIDIA</text>
<text x="40" y="6533" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">belgelerinden. Şerit rengi parçanın hangi sütuna hizmet ettiğini gösterir; gri</text>
<text x="40" y="6548" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">yönetim ve altyapıdır.</text>
<text x="40" y="6572" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Donanım</text>
<rect x="36" y="6582" width="240" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="6582" width="4" height="128" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="6601" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100 sistemi</text>
<rect x="229.8" y="6588.0" width="38.2" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.9" y="6601.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×127</text>
<text x="48" y="6618" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 H100 · 640 GB HBM3 · 900 GB/s NVLink</text>
<text x="48" y="6636" text-anchor="start" style="fill:var(--c-text);font-size:11px">Modelin bütün matematiği burada</text>
<text x="48" y="6651" text-anchor="start" style="fill:var(--c-text);font-size:11px">yapılır; kümedeki diğer her parça bu</text>
<text x="48" y="6666" text-anchor="start" style="fill:var(--c-text);font-size:11px">kutuları beslemek için vardır.</text>
<rect x="284" y="6582" width="240" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="6582" width="4" height="128" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="296" y="6601" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Hesaplama ağı · QM9700</text>
<rect x="483.9" y="6588.0" width="32.2" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="499.9" y="6601.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×48</text>
<text x="296" y="6618" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">32 leaf + 16 spine · düğüm başına</text>
<text x="296" y="6632" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 × 400 Gb/s</text>
<text x="296" y="6650" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gradyan ve aktivasyon trafiği düğümler</text>
<text x="296" y="6665" text-anchor="start" style="fill:var(--c-text);font-size:11px">arasında tıkanmadan akmalı; bu yüzden</text>
<text x="296" y="6680" text-anchor="start" style="fill:var(--c-text);font-size:11px">her GPU&#x27;nun kendi 400 Gb/s&#x27;lik rayı</text>
<text x="296" y="6695" text-anchor="start" style="fill:var(--c-text);font-size:11px">vardır.</text>
<rect x="36" y="6718" width="240" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="6718" width="4" height="144" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="48" y="6737" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Depolama ağı · QM9700</text>
<rect x="235.8" y="6724.0" width="32.2" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="251.9" y="6737.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×16</text>
<text x="48" y="6754" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">düğüm başına ≥ 40 GB/s · ~4:3</text>
<text x="48" y="6772" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri okuma ve kontrol noktası yazma,</text>
<text x="48" y="6787" text-anchor="start" style="fill:var(--c-text);font-size:11px">hesaplama ağındaki AllReduce trafiğine</text>
<text x="48" y="6802" text-anchor="start" style="fill:var(--c-text);font-size:11px">karışmasın; hızı depolama dizisinin</text>
<text x="48" y="6817" text-anchor="start" style="fill:var(--c-text);font-size:11px">tepe hızına göre seçilir.</text>
<rect x="284" y="6718" width="240" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="6718" width="4" height="144" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="296" y="6737" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">UFM cihazı</text>
<rect x="489.9" y="6724.0" width="26.1" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="502.9" y="6737.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×4</text>
<text x="296" y="6754" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">InfiniBand telemetrisi ve analitiği</text>
<text x="296" y="6772" text-anchor="start" style="fill:var(--c-text);font-size:11px">Binlerce port ve kabloyu izleyip</text>
<text x="296" y="6787" text-anchor="start" style="fill:var(--c-text);font-size:11px">bozulan bağlantıyı bulur (Llama 3</text>
<text x="296" y="6802" text-anchor="start" style="fill:var(--c-text);font-size:11px">eğitiminde kesintilerin %8,4&#x27;ü switch</text>
<text x="296" y="6817" text-anchor="start" style="fill:var(--c-text);font-size:11px">ve kablodan geldi). Bir çifti bir</text>
<text x="296" y="6832" text-anchor="start" style="fill:var(--c-text);font-size:11px">DGX&#x27;in yerini alır, küme bu yüzden 127</text>
<text x="296" y="6847" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğümdür.</text>
<rect x="36" y="6870" width="240" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="6870" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="6889" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">In-band yönetim · SN4600C</text>
<rect x="241.9" y="6876.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="255.0" y="6889.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="48" y="6906" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 port × 100 GbE (Ethernet)</text>
<text x="48" y="6924" text-anchor="start" style="fill:var(--c-text);font-size:11px">Düğüm kurulumu, kullanıcı girişi,</text>
<text x="48" y="6939" text-anchor="start" style="fill:var(--c-text);font-size:11px">dosya sistemi ve dış erişim (NGC,</text>
<text x="48" y="6954" text-anchor="start" style="fill:var(--c-text);font-size:11px">paket depoları) bu ağdan akar;</text>
<text x="48" y="6969" text-anchor="start" style="fill:var(--c-text);font-size:11px">hesaplama ağı yalnız GPU trafiğine</text>
<text x="48" y="6984" text-anchor="start" style="fill:var(--c-text);font-size:11px">kalır.</text>
<rect x="284" y="6870" width="240" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="6870" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="296" y="6889" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">OOB yönetim · SN2201</text>
<rect x="489.9" y="6876.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="502.9" y="6889.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="296" y="6906" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">48 port × 1 GbE bakır</text>
<text x="296" y="6924" text-anchor="start" style="fill:var(--c-text);font-size:11px">Her sunucunun BMC yönetim çipine ve</text>
<text x="296" y="6939" text-anchor="start" style="fill:var(--c-text);font-size:11px">PDU&#x27;lara ulaşır; işletim sistemi</text>
<text x="296" y="6954" text-anchor="start" style="fill:var(--c-text);font-size:11px">kilitlense bile düğüm uzaktan</text>
<text x="296" y="6969" text-anchor="start" style="fill:var(--c-text);font-size:11px">kapatılıp açılabilir. Kullanıcı</text>
<text x="296" y="6984" text-anchor="start" style="fill:var(--c-text);font-size:11px">ağlarından ayrıdır.</text>
<rect x="36" y="7007" width="240" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="7007" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="7026" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Yönetim sunucuları</text>
<rect x="241.9" y="7013.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="255.0" y="7026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×5</text>
<text x="48" y="7043" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 soket · 24+ çekirdek · 384 GB RAM</text>
<text x="48" y="7061" text-anchor="start" style="fill:var(--c-text);font-size:11px">Küme yönetimi ve iş zamanlama</text>
<text x="48" y="7076" text-anchor="start" style="fill:var(--c-text);font-size:11px">yazılımları ayrı makinelerde koşar;</text>
<text x="48" y="7091" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU düğümleri yalnızca eğitime kalır.</text>
<rect x="284" y="7007" width="240" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="7007" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="296" y="7026" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kabin ve güç</text>
<rect x="453.6" y="7013.0" width="62.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="484.8" y="7026.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">38 kabin</text>
<text x="296" y="7043" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">38 kabin · 108 PDU</text>
<text x="296" y="7061" text-anchor="start" style="fill:var(--c-text);font-size:11px">Kabin başına 4 DGX (40,8 kW), SU</text>
<text x="296" y="7076" text-anchor="start" style="fill:var(--c-text);font-size:11px">başına 8 DGX kabini; 127 DGX tek</text>
<text x="296" y="7091" text-anchor="start" style="fill:var(--c-text);font-size:11px">başına 127 × 10,2 kW ≈ 1,3 MW çeker.</text>
<text x="296" y="7106" text-anchor="start" style="fill:var(--c-text);font-size:11px">Güç ve soğutma kümenin görünmeyen</text>
<text x="296" y="7121" text-anchor="start" style="fill:var(--c-text);font-size:11px">dördüncü sütunudur.</text>
<text x="40" y="7162" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Yazılım</text>
<rect x="36" y="7172" width="240" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="7172" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="7191" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Base Command Manager</text>
<rect x="211.7" y="7178.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="239.8" y="7191.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="48" y="7208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">küme yönetimi · otomatik kurulum</text>
<text x="48" y="7226" text-anchor="start" style="fill:var(--c-text);font-size:11px">Binlerce düğüme aynı işletim sistemi</text>
<text x="48" y="7241" text-anchor="start" style="fill:var(--c-text);font-size:11px">imajını kurar, günceller ve izler; 127</text>
<text x="48" y="7256" text-anchor="start" style="fill:var(--c-text);font-size:11px">makineyi elle yönetmek gerçekçi</text>
<text x="48" y="7271" text-anchor="start" style="fill:var(--c-text);font-size:11px">değildir.</text>
<rect x="284" y="7172" width="240" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="7172" width="4" height="114" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="296" y="7191" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Slurm</text>
<rect x="447.6" y="7178.0" width="68.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="481.8" y="7191.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesaplama</text>
<text x="296" y="7208" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">toplu iş zamanlayıcı</text>
<text x="296" y="7226" text-anchor="start" style="fill:var(--c-text);font-size:11px">Çok düğümlü bir eğitim işinin hangi</text>
<text x="296" y="7241" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;larda ve hangi sırayla koşacağını</text>
<text x="296" y="7256" text-anchor="start" style="fill:var(--c-text);font-size:11px">belirler; arızalı düğümü devreden</text>
<text x="296" y="7271" text-anchor="start" style="fill:var(--c-text);font-size:11px">çıkarıp işi yeniden kuyruğa alabilir.</text>
<rect x="36" y="7294" width="240" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="7294" width="4" height="128" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="48" y="7313" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Magnum IO</text>
<rect x="175.3" y="7300.0" width="92.7" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="221.7" y="7313.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ + depolama</text>
<text x="48" y="7330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NCCL · GPUDirect RDMA / Storage ·</text>
<text x="48" y="7344" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SHARP</text>
<text x="48" y="7362" text-anchor="start" style="fill:var(--c-text);font-size:11px">Verinin GPU&#x27;lar ve diskler arasındaki</text>
<text x="48" y="7377" text-anchor="start" style="fill:var(--c-text);font-size:11px">hareketini hızlandıran kütüphane</text>
<text x="48" y="7392" text-anchor="start" style="fill:var(--c-text);font-size:11px">yığını; B&#x27;deki 1., 3. ve 6. adımlar bu</text>
<text x="48" y="7407" text-anchor="start" style="fill:var(--c-text);font-size:11px">katmanla yapılır.</text>
<rect x="284" y="7294" width="240" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="7294" width="4" height="128" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="296" y="7313" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">NGC kataloğu</text>
<rect x="447.6" y="7300.0" width="68.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="481.8" y="7313.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesaplama</text>
<text x="296" y="7330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU için optimize konteynerler</text>
<text x="296" y="7348" text-anchor="start" style="fill:var(--c-text);font-size:11px">CUDA, NCCL ve PyTorch gibi çerçeveler</text>
<text x="296" y="7363" text-anchor="start" style="fill:var(--c-text);font-size:11px">birbirine uyan tek bir paket hâlinde</text>
<text x="296" y="7378" text-anchor="start" style="fill:var(--c-text);font-size:11px">gelir; herkes aynı yazılım yığınıyla</text>
<text x="296" y="7393" text-anchor="start" style="fill:var(--c-text);font-size:11px">başlar.</text>
<rect x="36" y="7430" width="240" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="7430" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="7449" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Base Command Platform</text>
<rect x="211.7" y="7436.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="239.8" y="7449.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="48" y="7466" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI eğitimi için yazılım hizmeti</text>
<text x="48" y="7484" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri bilimcilerin iş gönderip veri</text>
<text x="48" y="7499" text-anchor="start" style="fill:var(--c-text);font-size:11px">setlerini ve sonuçları yönettiği</text>
<text x="48" y="7514" text-anchor="start" style="fill:var(--c-text);font-size:11px">katman; alttaki donanımı bilmeden</text>
<text x="48" y="7529" text-anchor="start" style="fill:var(--c-text);font-size:11px">kümeyi kullandırır.</text>
<rect x="284" y="7430" width="240" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="284" y="7430" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="296" y="7449" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">NVIDIA AI Enterprise</text>
<rect x="459.6" y="7436.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="487.8" y="7449.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="296" y="7466" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">araçlar, çerçeveler, orkestrasyon</text>
<text x="296" y="7484" text-anchor="start" style="fill:var(--c-text);font-size:11px">Geliştiriciye çerçeve ve araçları, BT</text>
<text x="296" y="7499" text-anchor="start" style="fill:var(--c-text);font-size:11px">ekibine destekli yönetim ve</text>
<text x="296" y="7514" text-anchor="start" style="fill:var(--c-text);font-size:11px">orkestrasyonu verir.</text>
<rect x="28.0" y="7572.0" width="504.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="7592" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: GPU&#x27;dan uzaklaştıkça yol daralır: HBM 3,35 TB/s → NVLink</text>
<text x="40" y="7609" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">450 GB/s → ağ 50 GB/s.</text>
<text x="280" y="7654" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Parçaları tanıdık. Şimdi bir eğitim adımında birlikte nasıl çalıştıklarını</text>
<text x="280" y="7671" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">izleyelim.</text>
<line x1="280.0" y1="7679.0" x2="280.0" y2="7699.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="7703" width="528" height="1016" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="7727" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="7731.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">B</text>
<text x="56" y="7732" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Bir eğitim adımını izle: veri nereden geçer?</text>
<text x="28" y="7753" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">64 SU&#x27;luk H100 SuperPOD&#x27;da Llama 3 405B: TP 8 × PP 16 × DP 128 = 16.384 GPU</text>
<rect x="28" y="7765" width="504" height="264" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="170.0" y="7777.0" width="220.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="7792.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine → diğer 127 DP kopyası</text>
<rect x="40" y="7825" width="200" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50" y="7841" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">düğüm A · PP aşaması s</text>
<rect x="70.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G0</text>
<rect x="112.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G1</text>
<rect x="154.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="172.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G2</text>
<rect x="196.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="214.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G3</text>
<rect x="70.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="88.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G4</text>
<rect x="112.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G5</text>
<rect x="154.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="172.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G6</text>
<rect x="196.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="214.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G7</text>
<line x1="88.0" y1="7869.0" x2="88.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="88.0" y1="7895.0" x2="88.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="130.0" y1="7869.0" x2="130.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="130.0" y1="7895.0" x2="130.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="172.0" y1="7869.0" x2="172.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="172.0" y1="7895.0" x2="172.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="214.0" y1="7869.0" x2="214.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="214.0" y1="7895.0" x2="214.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="70.0" y="7877.0" width="162.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="151.0" y="7890.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<circle cx="54" cy="7859" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="54" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="54" cy="7886" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="54" y="7890.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="54" cy="7913" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="54" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<rect x="320" y="7825" width="200" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="330" y="7841" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">düğüm B · aşama s+1</text>
<rect x="338.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="356.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G0</text>
<rect x="380.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="398.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G1</text>
<rect x="422.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="440.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G2</text>
<rect x="464.0" y="7849.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="482.0" y="7863.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G3</text>
<rect x="338.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="356.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G4</text>
<rect x="380.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="398.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G5</text>
<rect x="422.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="440.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G6</text>
<rect x="464.0" y="7903.0" width="36.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="482.0" y="7917.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">G7</text>
<line x1="356.0" y1="7869.0" x2="356.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="356.0" y1="7895.0" x2="356.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="398.0" y1="7869.0" x2="398.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="398.0" y1="7895.0" x2="398.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="440.0" y1="7869.0" x2="440.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="440.0" y1="7895.0" x2="440.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="482.0" y1="7869.0" x2="482.0" y2="7877.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="482.0" y1="7895.0" x2="482.0" y2="7903.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="338.0" y="7877.0" width="162.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="419.0" y="7890.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<rect x="256.0" y="7835.0" width="48.0" height="96" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280" y="7893" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">leaf</text>
<circle cx="280" cy="7861" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="280" y="7865.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="280" cy="7915" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="280" y="7919.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<line x1="240.0" y1="7861.0" x2="256.0" y2="7861.0" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<line x1="304.0" y1="7861.0" x2="318.0" y2="7861.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="320.0" y1="7915.0" x2="304.0" y2="7915.0" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="256.0" y1="7915.0" x2="242.0" y2="7915.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<line x1="280.0" y1="7833.0" x2="280.0" y2="7801.0" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="298" cy="7817" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="298" y="7821.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="40.0" y="7977.0" width="480.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="280" y="7993" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">depolama sistemi (ör. Lustre · Weka)</text>
<text x="280" y="8009" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ayrı depolama ağı · düğüm başına ≥ 40 GB/s</text>
<line x1="90.0" y1="7975.0" x2="90.0" y2="7944.0" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<circle cx="108" cy="7959" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="108" y="7963.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<line x1="180.0" y1="7944.0" x2="180.0" y2="7975.0" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5;stroke-dasharray:5 4"/>
<circle cx="198" cy="7959" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="198" y="7963.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="400" y="7961" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(B düğümü de aynı yolu kullanır)</text>
<rect x="28" y="8039" width="504" height="606" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="46.0" y1="8065.0" x2="46.0" y2="8585.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="46" cy="8065" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="46" y="8069.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="8070" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Veriyi oku</text>
<rect x="356.8" y="8057.0" width="165.2" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="439.4" y="8070.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">depolama → HBM · her adım</text>
<text x="66" y="8089" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Bu adımın 16 milyon tokeni depodan HBM&#x27;e gelir; GDS ya da mmap sayesinde</text>
<text x="66" y="8105" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">araya CPU kopyası girmez.</text>
<circle cx="46" cy="8137" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="46" y="8141.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="8142" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">İleri geçiş</text>
<rect x="350.7" y="8129.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="436.4" y="8142.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM 3,35 TB/s · her katman</text>
<text x="66" y="8161" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her katmanda SM&#x27;ler ağırlıkları HBM&#x27;den çekip Tensor Core&#x27;larda çarpar;</text>
<text x="66" y="8177" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">okunan bayt başına ~295 işlem yapılırsa çekirdekler doyar.</text>
<circle cx="46" cy="8209" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="8213.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="8214" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP AllReduce</text>
<rect x="332.5" y="8201.0" width="189.5" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="427.3" y="8214.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink 450 GB/s · katmanda 2×</text>
<text x="66" y="8233" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her matris 8 GPU&#x27;ya bölündüğü için parçalı sonuçlar NVLink&#x27;te toplanır;</text>
<text x="66" y="8249" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">TP bu yüzden kasanın dışına taşmaz.</text>
<circle cx="46" cy="8281" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="8285.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="66" y="8286" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aktivasyonu gönder (PP)</text>
<rect x="338.6" y="8273.0" width="183.4" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="430.3" y="8286.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">50 GB/s/GPU · her mikrobatch</text>
<text x="66" y="8305" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Düğümün katmanları bitince çıktı (~16 MB) ağdan sonraki düğüme geçer; yol</text>
<text x="66" y="8321" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">9 kat dardır ama giden veri küçüktür.</text>
<circle cx="46" cy="8353" r="10" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="46" y="8357.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="66" y="8358" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Geri geçiş</text>
<rect x="380.9" y="8345.0" width="141.1" height="19" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="451.5" y="8358.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">aynı hatlar, ters yön</text>
<text x="66" y="8377" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Hata sinyali (gradyan) aynı yolu geri döner; 3. ve 4. adımlar ters yönde</text>
<text x="66" y="8393" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tekrarlanır ve iş ileri geçişin ~2 katıdır.</text>
<circle cx="46" cy="8425" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="8429.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="66" y="8430" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Gradyan AllReduce (DP)</text>
<rect x="350.7" y="8417.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="436.4" y="8430.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ray switch&#x27;leri · her adım</text>
<text x="66" y="8449" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">128 kopyanın gradyanları ortalanır ki hepsi aynı güncellemeyi yapsın;</text>
<text x="66" y="8465" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">InfiniBand SHARP toplamayı switch içinde yapıp süreyi 2·B/W&#x27;den B/W&#x27;ye</text>
<text x="66" y="8481" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">indirir.</text>
<circle cx="46" cy="8513" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="46" y="8517.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<text x="66" y="8518" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Optimizer adımı</text>
<rect x="380.9" y="8505.0" width="141.1" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="451.5" y="8518.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM içinde · her adım</text>
<text x="66" y="8537" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">AdamW ağırlıkları günceller ve her parametre için 12 bayt FP32 durum</text>
<text x="66" y="8553" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tutar: ana ağırlık, momentum ve varyans.</text>
<circle cx="46" cy="8585" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="46" y="8589.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="66" y="8590" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kontrol noktası</text>
<rect x="326.5" y="8577.0" width="195.5" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="424.2" y="8590.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM → depolama · ~14 dk&#x27;da bir</text>
<text x="66" y="8609" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">~14 dakikada bir 7,5 TB durum diske yazılır: 250 GB/s ile ~31 s, eski</text>
<text x="66" y="8625" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">25 GB/s ile ~5 dk sürer; arızadan sonra buradan devam edilir.</text>
<rect x="28.0" y="8657.0" width="504.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="8677" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: bir adımda veri dört ayrı yoldan geçer (HBM, NVLink, ağ,</text>
<text x="40" y="8694" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">depolama); adım ancak en yavaş yol kadar hızlıdır.</text>
<text x="280" y="8739" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Peki bu yollar nerede tıkanıyor? Her sütunun bir duvarı var.</text>
<line x1="280.0" y1="8747.0" x2="280.0" y2="8767.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="8771" width="528" height="1240" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="8795" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="8799.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">C</text>
<text x="56" y="8800" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Üç duvar: belirti → neden → çare</text>
<text x="28" y="8821" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Eğriler makaledeki formüllerden hesaplandı; kırmızı darboğaz, yeşil çaredir.</text>
<rect x="28" y="8833" width="504" height="370" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="8857" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Bellek duvarı</text>
<text x="520" y="8857" text-anchor="end" style="fill:var(--c-accent);font-size:11px;font-weight:700">1. sütun · hesaplama</text>
<text x="40" y="8878" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="112" y="8878" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU meşgul görünür, ama gücünün yalnızca %1–2&#x27;si kullanılır.</text>
<text x="40" y="8896" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="112" y="8896" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her bayta düşen iş az: α = BDF / (B(D+F) + DF), H100&#x27;ün eşiği olan</text>
<text x="112" y="8913" text-anchor="start" style="fill:var(--c-text);font-size:12px">295&#x27;in altında kaldıkça Tensor Core&#x27;lar veri bekler.</text>
<line x1="80.0" y1="8928.0" x2="80.0" y2="9054.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="9054.0" x2="512.0" y2="9054.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="8935.2" x2="512.0" y2="8935.2" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="8939.2" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">990</text>
<text x="76" y="9058" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="40" y="8927.2" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s</text>
<line x1="377.9" y1="8935.2" x2="377.9" y2="9054.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<polyline points="80.0,9053.6 84.5,9053.6 89.0,9053.5 93.5,9053.5 98.0,9053.4 102.5,9053.4 107.0,9053.3 111.5,9053.3 116.0,9053.2 120.5,9053.1 125.0,9053.0 129.5,9053.0 134.0,9052.9 138.5,9052.8 143.0,9052.6 147.5,9052.5 152.0,9052.4 156.5,9052.2 161.0,9052.1 165.5,9051.9 170.0,9051.7 174.5,9051.5 179.0,9051.3 183.5,9051.1 188.0,9050.8 192.5,9050.5 197.0,9050.2 201.5,9049.8 206.0,9049.5 210.5,9049.0 215.0,9048.6 219.5,9048.1 224.0,9047.6 228.5,9047.0 233.0,9046.4 237.5,9045.7 242.0,9044.9 246.5,9044.1 251.0,9043.2 255.5,9042.3 260.0,9041.2 264.5,9040.0 269.0,9038.8 273.5,9037.4 278.0,9035.9 282.5,9034.3 287.0,9032.5 291.5,9030.6 296.0,9028.5 300.5,9026.2 305.0,9023.8 309.5,9021.1 314.0,9018.1 318.5,9014.9 323.0,9011.4 327.5,9007.7 332.0,9003.6 336.5,8999.1 341.0,8994.2 345.5,8989.0 350.0,8983.2 354.5,8977.0 359.0,8970.3 363.5,8963.0 368.0,8955.1 372.5,8946.5 377.0,8937.2 381.5,8935.2 386.0,8935.2 390.5,8935.2 395.0,8935.2 399.5,8935.2 404.0,8935.2 408.5,8935.2 413.0,8935.2 417.5,8935.2 422.0,8935.2 426.5,8935.2 431.0,8935.2 435.5,8935.2 440.0,8935.2 444.5,8935.2 449.0,8935.2 453.5,8935.2 458.0,8935.2 462.5,8935.2 467.0,8935.2 471.5,8935.2 476.0,8935.2 480.5,8935.2 485.0,8935.2 489.5,8935.2 494.0,8935.2 498.5,8935.2 503.0,8935.2 507.5,8935.2 512.0,8935.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.2"/>
<text x="80.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="152.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="224.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="296.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="368.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="440.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.024</text>
<text x="512.0" y="9068" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4.096</text>
<text x="512" y="9084" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">batch B (token, log ölçek)</text>
<text x="150" y="8951.2" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">bellek sınırlı</text>
<text x="470" y="9036" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">hesap sınırlı</text>
<circle cx="80.0" cy="9053.6" r="4" style="fill:var(--c-danger)"/>
<text x="88.0" y="9044" text-anchor="start" style="fill:var(--c-text);font-size:11px">B = 1: 3,3 TFLOP/s (%0,3)</text>
<circle cx="332.0" cy="9003.6" r="4" style="fill:var(--c-accent)"/>
<text x="324.0" y="8997.6" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (%42)</text>
<circle cx="377.9" cy="8935.2" r="4" style="fill:var(--c-success)"/>
<text x="385.9" y="8951.2" text-anchor="start" style="fill:var(--c-text);font-size:11px">B ≈ 310: tavan 990</text>
<text x="40" y="9102" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP (D = 8.192, F = 28.672), H100 BF16: hız = min(990, 3,35·α)</text>
<text x="40" y="9118" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: batch büyüdükçe okunan her bayt daha çok işe yarar; eğri B ≈ 310&#x27;da</text>
<text x="40" y="9133" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">tavana değer.</text>
<text x="40" y="9159" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="104.0" y="9145.0" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="165.4" y="9159.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">batch ≥ ~310 token</text>
<rect x="232.9" y="9145.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="309.5" y="9159.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">füzyon / FlashAttention</text>
<rect x="104.0" y="9171.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="180.6" y="9185.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode: 4–8 bit ağırlık</text>
<rect x="28" y="9213" width="504" height="370" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="9237" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Ağ duvarı</text>
<text x="520" y="9237" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">2. sütun · ağ</text>
<text x="40" y="9258" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="112" y="9258" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğüm ekledikçe adım uzar; GPU&#x27;lar konuşmak için bekler.</text>
<text x="40" y="9276" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="112" y="9276" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğümden çıkan veri 9 kat daha dar bir yoldan geçer; TP gibi her</text>
<text x="112" y="9293" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmanda konuşan işler bu yolda boğulur.</text>
<line x1="184.0" y1="9306.0" x2="184.0" y2="9390.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="184" y="9404" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="288.0" y1="9306.0" x2="288.0" y2="9390.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="288" y="9404" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="392.0" y1="9306.0" x2="392.0" y2="9390.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="392" y="9404" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="496.0" y1="9306.0" x2="496.0" y2="9390.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="496" y="9404" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="40" y="9324" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM (çip içi)</text>
<rect x="184" y="9312" width="262.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="452.6" y="9324" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3,35 TB/s</text>
<text x="40" y="9350" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="184" y="9338" width="171.9" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="361.9" y="9350" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="420.7" y="9350" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7,4</text>
<text x="40" y="9376" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (düğüm dışı)</text>
<rect x="184" y="9364" width="72.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="262.7" y="9376" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="314.9" y="9376" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<rect x="40.0" y="9416.0" width="480.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="9430.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink: Y ≤ F / (C/W) = 28.672 / 2.200 ≈ 13 → TP = 8 sığar ✓ (F: Llama 3 70B)</text>
<rect x="40.0" y="9442.0" width="480.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="9456.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">InfiniBand: Y ≤ 28.672 / 19.800 ≈ 1,45 → TP düğümden taşarsa hesap aç kalır ✗</text>
<text x="40" y="9480" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP = 16 aşama, kabarcık = 15 / (15 + M): M = 16 → %48 boşta · M = 128 → %10,5</text>
<text x="40" y="9498" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: her çubuk bir H100&#x27;ün o hattaki bant genişliği; düğümden çıkınca çubuk</text>
<text x="40" y="9513" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">9 kat kısalır.</text>
<text x="40" y="9539" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="104.0" y="9525.0" width="104.8" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="156.4" y="9539.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TP düğüm içinde</text>
<rect x="214.8" y="9525.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="291.3" y="9539.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PP ve DP düğümler arası</text>
<rect x="373.9" y="9525.0" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="435.3" y="9539.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP: 2·B/W → B/W</text>
<rect x="104.0" y="9551.0" width="165.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="186.6" y="9565.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DualPipe: iletişimi gizle</text>
<rect x="28" y="9593" width="504" height="344" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="9617" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">G/Ç duvarı</text>
<text x="520" y="9617" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:700">3. sütun · depolama</text>
<text x="40" y="9638" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="112" y="9638" text-anchor="start" style="fill:var(--c-text);font-size:12px">Küme her kayıtta durur; arızada son kayda geri dönülür.</text>
<text x="40" y="9656" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="112" y="9656" text-anchor="start" style="fill:var(--c-text);font-size:12px">7,5 TB&#x27;lık kayıt CPU sayfa önbelleğinden 25 GB/s ile yazılırsa</text>
<text x="112" y="9673" text-anchor="start" style="fill:var(--c-text);font-size:12px">δ = 307 s sürer ve küme bu sürede bekler.</text>
<line x1="80.0" y1="9692.0" x2="80.0" y2="9814.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="9814.0" x2="512.0" y2="9814.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="9784.0" x2="512.0" y2="9784.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="9788.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%10</text>
<line x1="80.0" y1="9754.0" x2="512.0" y2="9754.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="9758.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%20</text>
<line x1="80.0" y1="9724.0" x2="512.0" y2="9724.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="9728.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%30</text>
<line x1="80.0" y1="9694.0" x2="512.0" y2="9694.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="9698.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%40</text>
<polyline points="80.0,9814.0 512.0,9739.0" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<polyline points="148.4,9694.3 149.6,9696.0 150.8,9697.6 152.0,9699.1 153.2,9700.6 154.4,9702.0 155.6,9703.4 156.8,9704.7 158.0,9705.9 159.2,9707.2 160.4,9708.3 161.6,9709.5 162.8,9710.6 164.0,9711.6 165.2,9712.7 166.4,9713.7 167.6,9714.6 168.8,9715.6 170.0,9716.5 171.2,9717.3 172.4,9718.2 173.6,9719.0 174.8,9719.8 176.0,9720.5 177.2,9721.3 178.4,9722.0 179.6,9722.7 180.8,9723.4 182.0,9724.0 183.2,9724.6 184.4,9725.3 185.6,9725.8 186.8,9726.4 188.0,9727.0 189.2,9727.5 190.4,9728.1 191.6,9728.6 192.8,9729.1 194.0,9729.5 195.2,9730.0 196.4,9730.5 197.6,9730.9 198.8,9731.3 200.0,9731.7 201.2,9732.1 202.4,9732.5 203.6,9732.9 204.8,9733.3 206.0,9733.6 207.2,9734.0 208.4,9734.3 209.6,9734.6 210.8,9734.9 212.0,9735.2 213.2,9735.5 214.4,9735.8 215.6,9736.1 216.8,9736.4 218.0,9736.6 219.2,9736.9 220.4,9737.1 221.6,9737.3 222.8,9737.6 224.0,9737.8 225.2,9738.0 226.4,9738.2 227.6,9738.4 228.8,9738.6 230.0,9738.8 231.2,9739.0 232.4,9739.2 233.6,9739.3 234.8,9739.5 236.0,9739.7 237.2,9739.8 238.4,9740.0 239.6,9740.1 240.8,9740.2 242.0,9740.4 243.2,9740.5 244.4,9740.6 245.6,9740.7 246.8,9740.8 248.0,9740.9 249.2,9741.1 250.4,9741.1 251.6,9741.2 252.8,9741.3 254.0,9741.4 255.2,9741.5 256.4,9741.6 257.6,9741.7 258.8,9741.7 260.0,9741.8 261.2,9741.9 262.4,9741.9 263.6,9742.0 264.8,9742.0 266.0,9742.1 267.2,9742.1 268.4,9742.2 269.6,9742.2 270.8,9742.2 272.0,9742.3 273.2,9742.3 274.4,9742.3 275.6,9742.3 276.8,9742.4 278.0,9742.4 279.2,9742.4 280.4,9742.4 281.6,9742.4 282.8,9742.4 284.0,9742.4 285.2,9742.4 286.4,9742.4 287.6,9742.4 288.8,9742.4 290.0,9742.4 291.2,9742.4 292.4,9742.4 293.6,9742.4 294.8,9742.4 296.0,9742.4 297.2,9742.3 298.4,9742.3 299.6,9742.3 300.8,9742.3 302.0,9742.2 303.2,9742.2 304.4,9742.2 305.6,9742.2 306.8,9742.1 308.0,9742.1 309.2,9742.0 310.4,9742.0 311.6,9742.0 312.8,9741.9 314.0,9741.9 315.2,9741.8 316.4,9741.8 317.6,9741.7 318.8,9741.7 320.0,9741.6 321.2,9741.6 322.4,9741.5 323.6,9741.4 324.8,9741.4 326.0,9741.3 327.2,9741.3 328.4,9741.2 329.6,9741.1 330.8,9741.1 332.0,9741.0 333.2,9740.9 334.4,9740.9 335.6,9740.8 336.8,9740.7 338.0,9740.6 339.2,9740.6 340.4,9740.5 341.6,9740.4 342.8,9740.3 344.0,9740.2 345.2,9740.2 346.4,9740.1 347.6,9740.0 348.8,9739.9 350.0,9739.8 351.2,9739.7 352.4,9739.6 353.6,9739.6 354.8,9739.5 356.0,9739.4 357.2,9739.3 358.4,9739.2 359.6,9739.1 360.8,9739.0 362.0,9738.9 363.2,9738.8 364.4,9738.7 365.6,9738.6 366.8,9738.5 368.0,9738.4 369.2,9738.3 370.4,9738.2 371.6,9738.1 372.8,9738.0 374.0,9737.9 375.2,9737.8 376.4,9737.7 377.6,9737.6 378.8,9737.5 380.0,9737.3 381.2,9737.2 382.4,9737.1 383.6,9737.0 384.8,9736.9 386.0,9736.8 387.2,9736.7 388.4,9736.6 389.6,9736.4 390.8,9736.3 392.0,9736.2 393.2,9736.1 394.4,9736.0 395.6,9735.8 396.8,9735.7 398.0,9735.6 399.2,9735.5 400.4,9735.4 401.6,9735.2 402.8,9735.1 404.0,9735.0 405.2,9734.9 406.4,9734.7 407.6,9734.6 408.8,9734.5 410.0,9734.4 411.2,9734.2 412.4,9734.1 413.6,9734.0 414.8,9733.9 416.0,9733.7 417.2,9733.6 418.4,9733.5 419.6,9733.3 420.8,9733.2 422.0,9733.1 423.2,9732.9 424.4,9732.8 425.6,9732.7 426.8,9732.5 428.0,9732.4 429.2,9732.3 430.4,9732.1 431.6,9732.0 432.8,9731.9 434.0,9731.7 435.2,9731.6 436.4,9731.4 437.6,9731.3 438.8,9731.2 440.0,9731.0 441.2,9730.9 442.4,9730.7 443.6,9730.6 444.8,9730.5 446.0,9730.3 447.2,9730.2 448.4,9730.0 449.6,9729.9 450.8,9729.7 452.0,9729.6 453.2,9729.5 454.4,9729.3 455.6,9729.2 456.8,9729.0 458.0,9728.9 459.2,9728.7 460.4,9728.6 461.6,9728.4 462.8,9728.3 464.0,9728.1 465.2,9728.0 466.4,9727.8 467.6,9727.7 468.8,9727.5 470.0,9727.4 471.2,9727.2 472.4,9727.1 473.6,9726.9 474.8,9726.8 476.0,9726.6 477.2,9726.5 478.4,9726.3 479.6,9726.2 480.8,9726.0 482.0,9725.9 483.2,9725.7 484.4,9725.6 485.6,9725.4 486.8,9725.3 488.0,9725.1 489.2,9724.9 490.4,9724.8 491.6,9724.6 492.8,9724.5 494.0,9724.3 495.2,9724.2 496.4,9724.0 497.6,9723.8 498.8,9723.7 500.0,9723.5 501.2,9723.4 502.4,9723.2 503.6,9723.1 504.8,9722.9 506.0,9722.7 507.2,9722.6 508.4,9722.4 509.6,9722.3 510.8,9722.1 512.0,9721.9" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<line x1="286.1" y1="9742.4" x2="286.1" y2="9814.0" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="286.1" cy="9742.4" r="4" style="fill:var(--c-danger)"/>
<text x="286.1" y="9828" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">42,9</text>
<polyline points="87.2,9710.4 88.4,9724.8 89.6,9735.5 90.8,9743.9 92.0,9750.5 93.2,9755.9 94.4,9760.3 95.6,9764.0 96.8,9767.2 98.0,9769.9 99.2,9772.3 100.4,9774.3 101.6,9776.1 102.8,9777.7 104.0,9779.1 105.2,9780.4 106.4,9781.5 107.6,9782.5 108.8,9783.4 110.0,9784.2 111.2,9785.0 112.4,9785.6 113.6,9786.2 114.8,9786.8 116.0,9787.3 117.2,9787.7 118.4,9788.1 119.6,9788.5 120.8,9788.8 122.0,9789.2 123.2,9789.4 124.4,9789.7 125.6,9789.9 126.8,9790.1 128.0,9790.3 129.2,9790.5 130.4,9790.6 131.6,9790.8 132.8,9790.9 134.0,9791.0 135.2,9791.1 136.4,9791.1 137.6,9791.2 138.8,9791.3 140.0,9791.3 141.2,9791.3 142.4,9791.4 143.6,9791.4 144.8,9791.4 146.0,9791.4 147.2,9791.4 148.4,9791.3 149.6,9791.3 150.8,9791.3 152.0,9791.3 153.2,9791.2 154.4,9791.2 155.6,9791.1 156.8,9791.1 158.0,9791.0 159.2,9790.9 160.4,9790.9 161.6,9790.8 162.8,9790.7 164.0,9790.6 165.2,9790.6 166.4,9790.5 167.6,9790.4 168.8,9790.3 170.0,9790.2 171.2,9790.1 172.4,9790.0 173.6,9789.9 174.8,9789.8 176.0,9789.7 177.2,9789.5 178.4,9789.4 179.6,9789.3 180.8,9789.2 182.0,9789.1 183.2,9788.9 184.4,9788.8 185.6,9788.7 186.8,9788.6 188.0,9788.4 189.2,9788.3 190.4,9788.2 191.6,9788.0 192.8,9787.9 194.0,9787.7 195.2,9787.6 196.4,9787.5 197.6,9787.3 198.8,9787.2 200.0,9787.0 201.2,9786.9 202.4,9786.7 203.6,9786.6 204.8,9786.4 206.0,9786.3 207.2,9786.1 208.4,9786.0 209.6,9785.8 210.8,9785.7 212.0,9785.5 213.2,9785.3 214.4,9785.2 215.6,9785.0 216.8,9784.9 218.0,9784.7 219.2,9784.5 220.4,9784.4 221.6,9784.2 222.8,9784.0 224.0,9783.9 225.2,9783.7 226.4,9783.5 227.6,9783.4 228.8,9783.2 230.0,9783.0 231.2,9782.9 232.4,9782.7 233.6,9782.5 234.8,9782.4 236.0,9782.2 237.2,9782.0 238.4,9781.8 239.6,9781.7 240.8,9781.5 242.0,9781.3 243.2,9781.1 244.4,9781.0 245.6,9780.8 246.8,9780.6 248.0,9780.4 249.2,9780.3 250.4,9780.1 251.6,9779.9 252.8,9779.7 254.0,9779.6 255.2,9779.4 256.4,9779.2 257.6,9779.0 258.8,9778.8 260.0,9778.7 261.2,9778.5 262.4,9778.3 263.6,9778.1 264.8,9777.9 266.0,9777.7 267.2,9777.6 268.4,9777.4 269.6,9777.2 270.8,9777.0 272.0,9776.8 273.2,9776.6 274.4,9776.5 275.6,9776.3 276.8,9776.1 278.0,9775.9 279.2,9775.7 280.4,9775.5 281.6,9775.3 282.8,9775.2 284.0,9775.0 285.2,9774.8 286.4,9774.6 287.6,9774.4 288.8,9774.2 290.0,9774.0 291.2,9773.8 292.4,9773.7 293.6,9773.5 294.8,9773.3 296.0,9773.1 297.2,9772.9 298.4,9772.7 299.6,9772.5 300.8,9772.3 302.0,9772.1 303.2,9771.9 304.4,9771.8 305.6,9771.6 306.8,9771.4 308.0,9771.2 309.2,9771.0 310.4,9770.8 311.6,9770.6 312.8,9770.4 314.0,9770.2 315.2,9770.0 316.4,9769.8 317.6,9769.6 318.8,9769.5 320.0,9769.3 321.2,9769.1 322.4,9768.9 323.6,9768.7 324.8,9768.5 326.0,9768.3 327.2,9768.1 328.4,9767.9 329.6,9767.7 330.8,9767.5 332.0,9767.3 333.2,9767.1 334.4,9766.9 335.6,9766.7 336.8,9766.5 338.0,9766.4 339.2,9766.2 340.4,9766.0 341.6,9765.8 342.8,9765.6 344.0,9765.4 345.2,9765.2 346.4,9765.0 347.6,9764.8 348.8,9764.6 350.0,9764.4 351.2,9764.2 352.4,9764.0 353.6,9763.8 354.8,9763.6 356.0,9763.4 357.2,9763.2 358.4,9763.0 359.6,9762.8 360.8,9762.6 362.0,9762.4 363.2,9762.2 364.4,9762.0 365.6,9761.8 366.8,9761.6 368.0,9761.4 369.2,9761.2 370.4,9761.0 371.6,9760.8 372.8,9760.6 374.0,9760.5 375.2,9760.3 376.4,9760.1 377.6,9759.9 378.8,9759.7 380.0,9759.5 381.2,9759.3 382.4,9759.1 383.6,9758.9 384.8,9758.7 386.0,9758.5 387.2,9758.3 388.4,9758.1 389.6,9757.9 390.8,9757.7 392.0,9757.5 393.2,9757.3 394.4,9757.1 395.6,9756.9 396.8,9756.7 398.0,9756.5 399.2,9756.3 400.4,9756.1 401.6,9755.9 402.8,9755.7 404.0,9755.5 405.2,9755.3 406.4,9755.1 407.6,9754.9 408.8,9754.7 410.0,9754.5 411.2,9754.3 412.4,9754.1 413.6,9753.9 414.8,9753.7 416.0,9753.5 417.2,9753.3 418.4,9753.1 419.6,9752.9 420.8,9752.7 422.0,9752.5 423.2,9752.3 424.4,9752.1 425.6,9751.9 426.8,9751.7 428.0,9751.5 429.2,9751.3 430.4,9751.1 431.6,9750.9 432.8,9750.7 434.0,9750.5 435.2,9750.3 436.4,9750.1 437.6,9749.9 438.8,9749.7 440.0,9749.5 441.2,9749.3 442.4,9749.0 443.6,9748.8 444.8,9748.6 446.0,9748.4 447.2,9748.2 448.4,9748.0 449.6,9747.8 450.8,9747.6 452.0,9747.4 453.2,9747.2 454.4,9747.0 455.6,9746.8 456.8,9746.6 458.0,9746.4 459.2,9746.2 460.4,9746.0 461.6,9745.8 462.8,9745.6 464.0,9745.4 465.2,9745.2 466.4,9745.0 467.6,9744.8 468.8,9744.6 470.0,9744.4 471.2,9744.2 472.4,9744.0 473.6,9743.8 474.8,9743.6 476.0,9743.4 477.2,9743.2 478.4,9743.0 479.6,9742.8 480.8,9742.6 482.0,9742.4 483.2,9742.2 484.4,9742.0 485.6,9741.8 486.8,9741.6 488.0,9741.4 489.2,9741.2 490.4,9741.0 491.6,9740.8 492.8,9740.5 494.0,9740.3 495.2,9740.1 496.4,9739.9 497.6,9739.7 498.8,9739.5 500.0,9739.3 501.2,9739.1 502.4,9738.9 503.6,9738.7 504.8,9738.5 506.0,9738.3 507.2,9738.1 508.4,9737.9 509.6,9737.7 510.8,9737.5 512.0,9737.3" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<line x1="145.2" y1="9791.4" x2="145.2" y2="9814.0" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="145.2" cy="9791.4" r="4" style="fill:var(--c-success)"/>
<text x="145.2" y="9828" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">13,6</text>
<text x="80.0" y="9828" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="224.0" y="9828" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30</text>
<text x="368.0" y="9828" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">60</text>
<text x="512.0" y="9828" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">90</text>
<text x="512" y="9844" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">kayıt aralığı τ (dakika)</text>
<text x="40" y="9686" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıp</text>
<rect x="300" y="9693" width="12" height="10" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="318" y="9702" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">A · 25 GB/s: en az %24 kayıp</text>
<rect x="300" y="9710" width="12" height="10" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="318" y="9719" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">B · 250 GB/s (GDS): en az %7,5</text>
<text x="353.6" y="9782.5" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">geri sarma τ/2M (ortak)</text>
<text x="40" y="9862" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">M = 3 sa (16.384 H100&#x27;de ~3 saatte bir arıza) · kayıp = δ/τ + τ/2M</text>
<text x="40" y="9878" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: sık kayıt yazmaya, seyrek kayıt geri sarmaya zaman kaybettirir; en iyi</text>
<text x="40" y="9893" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">aralık çukurun dibidir.</text>
<text x="40" y="9919" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="104.0" y="9905.0" width="116.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="162.4" y="9919.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPUDirect Storage</text>
<rect x="226.9" y="9905.0" width="147.1" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="300.4" y="9919.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">asenkron / anlık yazma</text>
<rect x="380.0" y="9905.0" width="98.7" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="429.3" y="9919.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">τ = √(2δM) seç</text>
<rect x="28.0" y="9947.0" width="504.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="9967" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: üç çarenin ortak fikri aynı: veriyi az taşı, taşırken hesabı</text>
<text x="40" y="9984" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">durdurma.</text>
<text x="280" y="10031" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Bu farkları sezgiyle görelim: aynı yükü her yoldan geçirelim.</text>
<line x1="280.0" y1="10039.0" x2="280.0" y2="10059.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="10063" width="528" height="408" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="10087" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="10091.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">D</text>
<text x="56" y="10092" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Hız sezgisi: 140 GB&#x27;ı taşımak ne kadar sürer?</text>
<text x="28" y="10113" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 70B&#x27;nin BF16 ağırlıkları = 140 GB; aynı yük her hattan (log ölçek).</text>
<rect x="28" y="10125" width="504" height="272" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="200.0" y1="10139.0" x2="200.0" y2="10263.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="200" y="10277" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 ms</text>
<line x1="292.0" y1="10139.0" x2="292.0" y2="10263.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="10277" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 ms</text>
<line x1="384.0" y1="10139.0" x2="384.0" y2="10263.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="384" y="10277" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 s</text>
<line x1="476.0" y1="10139.0" x2="476.0" y2="10263.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="476" y="10277" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 s</text>
<text x="40" y="10157" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3 (tek H100)</text>
<rect x="200" y="10145" width="57.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="263.1" y="10157" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">42 ms</text>
<text x="40" y="10181" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="200" y="10169" width="137.3" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="343.3" y="10181" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0,31 s</text>
<text x="40" y="10205" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU yolu)</text>
<rect x="200" y="10193" width="215.3" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="421.3" y="10205" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,2 s</text>
<text x="40" y="10229" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (GPU başına)</text>
<rect x="200" y="10217" width="225.1" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="431.1" y="10229" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,8 s</text>
<text x="40" y="10253" text-anchor="start" style="fill:var(--c-text);font-size:11px">eski depolama (tüm küme)</text>
<rect x="200" y="10241" width="252.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.8" y="10253" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5,6 s</text>
<text x="40" y="10299" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Çubuk uzadıkça yavaşlar: her çubuk aynı 140 GB&#x27;ın o hattan geçme süresidir.</text>
<rect x="40.0" y="10309.0" width="480.0" height="38" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="50" y="10324" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">İnsan ölçeği: HBM&#x27;den 1 saniyede gelen veri</text>
<text x="50" y="10340" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink&#x27;ten 7,4 s, PCIe&#x27;den 52 s, InfiniBand&#x27;den 67 s&#x27;de gelir.</text>
<rect x="40.0" y="10353.0" width="480.0" height="38" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="50" y="10368" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Decode&#x27;un tavanı: her token bütün ağırlıkları bir kez okur</text>
<text x="50" y="10384" text-anchor="start" style="fill:var(--c-text);font-size:11px">8 H100&#x27;de 140 GB ÷ (8 × 3,35 TB/s) ≈ 5,2 ms → batch 1&#x27;de en fazla ~190 token/s</text>
<rect x="28.0" y="10409.0" width="504.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="10429" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: HBM&#x27;de milisaniyeler, ağda saniyeler. İyi tasarım veriyi</text>
<text x="40" y="10446" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">hızlı katmanda tutar.</text>
<text x="280" y="10491" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Son durak: bütün bunlar gerçek bir kümede nasıl görünüyor?</text>
<line x1="280.0" y1="10499.0" x2="280.0" y2="10519.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="10523" width="528" height="318" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="10547" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="10551.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">E</text>
<text x="56" y="10552" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Gerçek küme: Meta&#x27;nın Llama 3 405B eğitimi</text>
<text x="28" y="10573" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Aynı 16.384 H100, Meta&#x27;nın kendi kümesinde (ağ: RoCE); sayılar Llama 3 raporundan.</text>
<rect x="28" y="10585" width="160" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="10593.0" width="144.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="108.0" y="10607.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Hesaplama</text>
<text x="38" y="10631" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16.384 H100</text>
<text x="38" y="10646" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP 8 × PP 16 × DP 128</text>
<text x="38" y="10661" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">adım başına 16M token</text>
<text x="38" y="10676" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına 400 TFLOP/s</text>
<text x="38" y="10691" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BF16 MFU %41</text>
<rect x="200" y="10585" width="160" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="208.0" y="10593.0" width="144.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="10607.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Ağ</text>
<text x="210" y="10631" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">RoCE · 400 Gb/s/GPU</text>
<text x="210" y="10646" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod = 3.072 GPU</text>
<text x="210" y="10661" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod içi tam biseksiyon</text>
<text x="210" y="10676" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 pod arası 1:7</text>
<text x="210" y="10691" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(küçük modeller: InfiniBand)</text>
<rect x="372" y="10585" width="160" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="380.0" y="10593.0" width="144.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="452.0" y="10607.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Depolama</text>
<text x="382" y="10631" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Tectonic · 240 PB</text>
<text x="382" y="10646" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7.500 SSD sunucusu</text>
<text x="382" y="10661" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 TB/s sürekli</text>
<text x="382" y="10676" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7 TB/s tepe</text>
<text x="382" y="10691" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıt: 1 MB–4 GB/GPU</text>
<rect x="28.0" y="10717.0" width="504.0" height="42" rx="5" style="fill:var(--c-danger);fill-opacity:.10;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="10734" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">54 günde 466 kesinti; 419&#x27;u beklenmeyen → ortalama ~3 saatte bir arıza</text>
<text x="40" y="10750" text-anchor="start" style="fill:var(--c-text);font-size:11px">%78 donanım (GPU %30,1 · HBM3 %17,2); otomasyonla etkin eğitim süresi &gt; %90</text>
<text x="28" y="10775" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">MFU: 8.192 H100&#x27;de %43, 16.384&#x27;te %41, 128K bağlamda %38; ölçek büyüdükçe iletişim artar.</text>
<rect x="28.0" y="10787.0" width="504.0" height="42" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="40" y="10804" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ders: üç sütun birlikte tasarlanır.</text>
<text x="40" y="10820" text-anchor="start" style="fill:var(--c-text);font-size:11px">En yavaş halka (HBM, ağ ya da disk) bütün kümenin hızını belirler; MFU onun aynasıdır.</text>
</svg>

*Şekil 1.1: Üç sütunun büyük resmi. A: SM → GPU → DGX düğümü (sistem topolojisiyle) → SuperPOD ölçek merdiveni, SuperPOD ağının yakından incelemesi (port hesabı, trafik yolları, ölçek) ve kümeyi ayakta tutan 14 yapı taşı; B: bir eğitim adımında verinin yolu; C: üç duvar ve hesaplanmış eğrileri; D: 140 GB'lık yükle hız sezgisi; E: Llama 3 405B'nin gerçek sayıları. Kaynaklar: NVIDIA DGX H100 ve DGX SuperPOD belgeleri, The Llama 3 Herd of Models (2024).*

Şekli yukarıdan aşağı okuyun. A bölümü bir SM'den 1.016 GPU'luk SuperPOD'a dört basamakta uzaklaşır ve donanım ile yazılım yapı taşlarının her birinin neden gerektiğini anlatır; B bölümü tek bir eğitim adımında verinin uğradığı sekiz durağı numaralar. C bölümündeki üç eğri bu yazının formüllerinden hesaplandı: roofline, bant genişliği basamakları ve Young-Daly kayıp eğrisi. D bölümü aynı 140 GB'ı her hattan geçirir, E bölümü ise üç sütunun Meta'nın Llama 3 405B kümesinde ölçülmüş hâlini verir. Sonraki üç bölüm sütunları sırayla açıyor.

---

## 2. 1. Sütun — Hesaplama: Streaming Multiprocessor, mikro-mimari ve yürütme dinamikleri

### Silikon seviyesinde GPU anatomisi: SM, warp scheduler ve tensor core

GPU'ların TPU'lardan neden bu kadar farklı davrandığını kavramak için bir restoran mutfağı analojisi kuralım. Bir Google TPU v5p, sadece iki devasa baş aşçı (TensorCore birimleri) tarafından yönetilen lüks bir Fransız restoranı gibidir. Her baş aşçı, 128x128 matrisleri tek hamlede doğrayabilen devasa kesme tahtalarına (Matrix Multiply Unit / MXU) sahiptir; küçük siparişler veya asimetrik görevler geldiğinde bu dev tahtalar boş kalır. Modern bir NVIDIA GPU (örneğin Hopper H100 veya Blackwell B200) ise 132 ila 148 adet bağımsız hat aşçısından oluşan koca bir ordudur. Bu aşçıların her birine **Streaming Multiprocessor (SM)** denir. Her hat aşçısı kendi küçük tezgâhında tamamen bağımsız siparişler hazırlar ve hepsi ortak bir kilere bağlıdır.

Fiziksel donanımda her bir SM, **SM alt bölmesi (subpartition)** adı verilen 4 özdeş çeyreğe ayrılmıştır. Her çeyrekte şu bileşenler bulunur:
1. **Bir Tensor Core (TC):** Özel sistolik matris çarpım motoru. 1,76 GHz saat hızında çalışan bir H100 üzerinde her bir Tensor Core, saat döngüsü başına yaklaşık 1.024 adet BF16 kayan nokta işlemi (kabaca 8x8x8 matris çarpımı) yürütür. 132 SM üzerindeki 528 Tensor Core bir araya geldiğinde 990 TFLOP/s BF16 yoğun hesaplama gücü üretilir. Blackwell B200'de bu güç döngü başına 2.048 işleme çıkarak yoğun BF16'da 2.250 TFLOP/s seviyesine ulaşır.
2. **Bir Warp Scheduler ve Dağıtım Birimi:** **Warp** adı verilen 32 iş parçacıklı (thread) kümelere komut dağıtır. Tek bir SM aynı anda 64 adede kadar yerleşik (resident) warp barındırabilir; bellek gecikmelerini gizlemek için saat döngüleri arasında warp'lar arasında ışık hızında bağlam değiştirir.
3. **32 FP32 CUDA Core (ALU):** Vektör aktivasyonlarını (GELU, SiLU), normalizasyonları (RMSNorm) ve toplamaları yürüten genel amaçlı aritmetik birimler. 132 SM'e sahip bir H100 çipinde ~1,98 GHz boost saat hızında çalışan 16.896 adet CUDA çekirdeği bulunur.
4. **16.384 Adet 32-bit Register:** Alt bölme başına 64 KiB (SM başına 256 KiB, çip genelinde ~33 MB) ultra hızlı register dosyası sağlar.

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

NVIDIA'nın GH100 silikon yerleşimi ve mikro-mimari şemaları bu işlem birimlerinin fiziksel konumunu doğrulamaktadır:

![NVIDIA GPU Çip Yerleşimi ve SM Mikro-mimarisi](../../assets/img/gpu/gpu-diagram.png)
*Şekil 2.1: JAX-ML Scaling Book'ta açıklandığı gibi HBM belleğe bağlı 132 ila 148 Streaming Multiprocessor barındıran H100/B200 GPU silikon yerleşimi.*

![NVIDIA H100 SM Alt Bölme Anatomisi](../../assets/img/gpu/blackwell-sm.png)
*Şekil 2.2: Dört alt bölmeyi, register dosyalarını, Warp Scheduler birimlerini, Tensor Core motorlarını ve paylaşımlı 256 KiB L1/SMEM önbelleğini gösteren H100 SM iç şeması.*

### SIMT yürütme dinamikleri ve warp divergence tehdidi

TPU'ların saf SIMD (Single Instruction, Multiple Data) yürütme modelinin aksine, NVIDIA GPU'ları **SIMT (Single Instruction, Multiple Threads)** paradigmasıyla çalışır. Bir warp içindeki 32 CUDA çekirdeğinin tümü belirli bir döngüde aynı komutu yürütse de, her iş parçacığı kendi program sayacına ve özel register alanına sahiptir.

Bu esneklik, donanımda **warp divergence (sapma)** adı verilen tehlikeli bir başarım kaybı yaratır. Çekirdeğiniz (kernel) koşullu bir dallanma içeriyorsa (örneğin MoE uzman yönlendirmesini kontrol eden bir `if/else` veya dinamik dolgu maskesi), dallara ayrılan iş parçacıkları bağımsızca ilerleyemez. Warp scheduler önce `if` yolunu yürütürken diğer çekirdekleri maskeleyip durdurur; ardından `else` yolunu yürütürken ilk grubu bekletir:

![NVIDIA GPU'larda Warp Sapması](../../assets/img/gpu/warp-divergence.png)
*Şekil 2.3: Warp sapmasının donanım zaman çizelgesi. 32 iş parçacıklı bir warp dallara ayrıldığında aktif olmayan hatlar maskelenir ve donanım verimi yarıya düşer.*

32 iş parçacığının 16'sı A dalına, 16'sı B dalına giderse, yürütme süresi iki katına çıkar ve efektif aritmetik verim tam %50 düşer.

<details>
<summary>Quiz: Warp sapmasının donanım hesaplama verimine etkisi nedir?</summary>

Bir warp içindeki 32 iş parçacığının yarısı `then` dalına, diğer yarısı `else` dalına saptığında donanım bu iki yolu sırayla yürütmek zorundadır. İlk aşamada 16 ALU çalışırken 16'sı boş bekler; ikinci aşamada durum tersine döner. Toplam saat süresi iki katına çıkar ve o warp için efektif hesaplama verimi tam %50 oranında çöker.
</details>

### Bellek hiyerarşisi ve aritmetik yoğunluk roofline modeli

Hesaplama gücü, işlem birimlerini besleyecek bellek bant genişliği olmadan bir hiçtir. GPU bellek alt sistemi dik bir piramit oluşturur:
- **Register Dosyası:** SM başına 256 KiB (çip genelinde ~33 MB). Tek döngüde erişim gecikmesi.
- **Paylaşımlı Bellek (SMEM) / L1 Önbellek:** SM başına 256 KiB. FlashAttention algoritmalarında dikkat bloklarını önbelleğe almak için kullanılan programlanabilir SRAM (10 ns altı gecikme).
- **Tensor Belleği (TMEM):** Blackwell B200 ile tanıtılan SM başına 256 KiB özel bellek. B200 Tensor Core birimlerinin ürettiği matris çarpım akümülatörleri klasik register kapasitesini aştığı için çip üstünde özel matris deposu sağlar.
- **Paylaşımlı L2 Önbellek:** Hopper H100'de 50 MB, Blackwell B200'de 126 MB. Çift yönde yaklaşık 5,5 TB/s bant genişliği sunar.
- **Yüksek Bant Genişlikli Bellek (HBM3 / HBM3e):** H100'de 80 GB (3,35 TB/s), H200'de 141 GB (4,8 TB/s), B200'de 192 GB (8,0 TB/s).

Hesaplama kapasitesi ile bellek bant genişliği arasındaki denge **Roofline Modeli** ile tanımlanır:

$$\alpha_{\text{crit}} = \frac{C_{\text{peak}}}{W_{\text{mem}}}$$

Şöyle okuyun: $\alpha_{\text{crit}}$ bayt başına FLOP cinsinden kritik aritmetik yoğunluk, $C_{\text{peak}}$ saniyedeki tepe kayan nokta işlem sayısı, $W_{\text{mem}}$ ise saniyedeki bellek bant genişliğidir.

BF16 Tensor Core çalıştıran bir H100 SXM5 için:
$$C_{\text{peak}} = 990 \times 10^{12} \text{ FLOP/s}, \quad W_{\text{mem}} = 3,35 \times 10^{12} \text{ B/s}$$
$$\alpha_{\text{crit}} = \frac{990 \times 10^{12}}{3,35 \times 10^{12}} \approx 295,5 \text{ FLOP/bayt}$$

$X \in \mathbb{R}^{B \times D}$ aktivasyon matrisi ile $W \in \mathbb{R}^{D \times F}$ ağırlık matrisinin 16-bit duyarlıkta çarpımı için:
- **Toplam Matematiksel İşlem:** $2 B D F \text{ FLOP}$
- **Toplam Bellek Transferi:** $2(BD + DF + BF) \text{ bayt}$
- **Operasyonel Yoğunluk:**

$$\alpha(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

Batch boyutu $B$ çok küçük olduğunda ($B \to 1$, örneğin tek tokenli ardışık çıkarım üretimi), ağırlık matrisi bellek trafiğini domine eder ($DF \gg B(D+F)$):
$$\alpha(1) \approx \frac{D F}{D F} \cdot B \approx B = 1 \text{ FLOP/bayt} \ll 295,5 \text{ FLOP/bayt}$$

Bu rejimde GPU tamamen bellek bant genişliğine takılır (memory-bandwidth bound) ve tepe hesaplama kapasitesinin %1'inden azını kullanabilir. Hesaplama sınırına (compute-bound) geçmek için gereken batch boyutu şartı şudur:

$$B \ge \frac{\alpha_{\text{crit}} \cdot D F}{D F - \alpha_{\text{crit}}(D + F)} \approx \alpha_{\text{crit}} \approx 295 \text{ token}$$

Hem Hopper H100 hem de Blackwell B200'de roofline eğrisinin dirsek noktasını aşıp hesaplama sınırına geçebilmek için SM başına en az 280 ila 300 tokenlik bir batch boyutu gerekir.

<details>
<summary>Quiz: Otoregresif token üretiminde MFU neden %2'nin altında kalır?</summary>

Tek tokenli çıkarımda ($B=1$) GPU tek bir vektörü işlemek için tüm model ağırlıklarını HBM'den register'lara çekmek zorundadır. Aritmetik yoğunluk $\alpha(1) \approx 1 \text{ FLOP/bayt}$ seviyesindedir. H100 Tensor Core birimlerini doyurmak için $\alpha_{\text{crit}} \approx 295,5 \text{ FLOP/bayt}$ gerektiğinden, donanım döngülerinin %99'undan fazlasını bellek bekleyerek harcar ve MFU %1-2 bandına sıkışır.
</details>

### Düşük duyarlıklı sayısal formatlar: FP32'den FP8 ve NVFP4'e

Bellek baskısını hafifletmek ve aritmetik işlem hızını katlamak için modern GPU mimarileri sayısal duyarlığı kademeli olarak düşürmüştür. FP32'den (32 bit: 1 işaret, 8 üs, 23 mantis) 16-bit formatlara (FP16 ve BF16) geçiş, bellek ayak izini yarıya indirip FLOP oranlarını ikiye katlamıştır.

Mimari nesilleri boyunca Tensor Core hesaplama motorlarının evrimi ve donanım sınırları aşağıda özetlenmiştir:

| Mimari Nesli | Hesaplama Motoru | HBM Bellek & Bant Genişliği | BF16 / FP16 TFLOP/s | FP8 / INT8 TFLOP/s | FP4 TFLOP/s |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Volta (V100)** | 1. Nesil Tensor Core | 32 GB @ 0,9 TB/s | 125 | — | — |
| **Ampere (A100)** | 3. Nesil Tensor Core | 80 GB @ 2,0 TB/s | 312 | 624 | — |
| **Hopper (H100)** | 4. Nesil Tensor Core | 80 GB @ 3,35 TB/s | 990 | 1.980 | — |
| **Hopper (H200)** | 4. Nesil Tensor Core | 141 GB @ 4,8 TB/s | 990 | 1.980 | — |
| **Blackwell (B200)** | 5. Nesil Tensor Core | 192 GB @ 8,0 TB/s | 2.250 | 4.500 | 9.000 |

Hopper H100 ile birlikte donanıma iki farklı FP8 (8-bit kayan nokta) formatı eklendi:
1. **E4M3 (1 işaret, 4 üs, 3 mantis biti):** Dinamik aralığın kararlı olduğu ileri geçiş aktivasyonları ve ağırlıklar için daha yüksek sayısal hassasiyet sunar.
2. **E5M2 (1 işaret, 5 üs, 2 mantis biti):** FP16'nın dinamik aralığını koruyarak geri yayılım gradyanlarında taşma veya sıfıra yuvarlanmayı (underflow) engeller.

| Sayısal Format | İşaret Biti | Üs Biti | Mantis Biti | Dinamik Aralık ({\text{min}} \dots E_{\text{max}}$) | LLM Ölçeklemesindeki Temel Rolü |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IEEE FP16** | 1 | 5 | 10 | $\approx 6 \times 10^{-5} \dots 65.504$ | Geleneksel çıkarım ve temel işlem |
| **Google BF16** | 1 | 8 | 7 | $\approx 1 \times 10^{-38} \dots 3 \times 10^{38}$ | Kararlı temel model ön eğitimi |
| **FP8 (E4M3)** | 1 | 4 | 3 | $\approx \pm 448$ | İleri geçiş aktivasyonları ve ağırlıklar |
| **FP8 (E5M2)** | 1 | 5 | 2 | $\approx \pm 57.344$ | Geri yayılım gradyan iletimi |
| **NVFP4 (E2M1)** | 1 | 2 | 1 | Blok ölçekli | Blackwell ultra yoğun GEMM hızlandırması |

Blackwell B200'de ise **NVFP4 (4-bit Microscaling)** mimarisine geçilmiştir. 16 elemanlı vektör blokları üzerinde paylaşılan 8-bit veya 16-bit ölçekleme katsayılarıyla desteklenen 4-bit sayılar, çip başına 4.500 TFLOP/s yoğun hesaplama üreterek devasa matrislerin doğrudan TMEM ve L2 önbellek sınırlarında kalmasını sağlar.

---

## 3. 2. Sütun — Ağ: Ara bağlantı omurgaları, kolektifler ve çok boyutlu paralellik

### Düğüm seviyesinde ağ: NVLink ve NVSwitch mimarisi

Tek bir çipin sınırlarını aşarken standart PCIe veri yolları (PCIe Gen 5'te tek yönde yalnızca 64 GB/s) aşılması imkânsız bir darboğaza dönüşür. NVIDIA bu engeli **NVLink** ve **NVSwitch** ile aşar.

Standart bir DGX H100 veya HGX H100 kasasında 8 GPU, 4 adet özel NVSwitch 4 ASIC'i ile eşleştirilmiştir. Her H100 çipi 18 adet NVLink 4 portuna sahiptir. Her port çift yönde 50 GB/s (tek yönde 25 GB/s) aktarım yaparak çip başına şu çıkış hızını üretir:
$$W_{\text{GPU egress}} = 18 \times 25 \text{ GB/s} = 450 \text{ GB/s tek yönlü } (900 \text{ GB/s çift yönlü})$$

4 adet merkezi NVSwitch, engellemesiz (non-blocking) bir çapraz bağlantı (crossbar) oluşturur. Kasa içindeki toplam anahtarlama kapasitesi şöyledir:
$$4 \text{ switch} \times 1,6 \text{ TB/s} = 6,4 \text{ TB/s anahtarlama bant genişliği}$$

8 GPU'nun her biri 450 GB/s çıkış yapabildiğinden, kasanın 4 GPU'ya karşı 4 GPU şeklindeki herhangi bir eşit ikiye bölünüşünde (bisection):
$$\text{Biseksiyon Bant Genişliği} = 4 \text{ GPU} \times 450 \text{ GB/s} \times 2 \text{ yön} = 3,6 \text{ TB/s}$$

Düğüm içi çapraz bağlantı ağ mimarisi aşağıdaki yapısal diyagramda gösterilmektedir:

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

Bu 8 GPU'lu anakartların fiziksel yerleşimi ve NVSwitch 4 ASIC'inin iç yapısı şöyledir:

![NVIDIA HGX Düğüm İçi NVLink Topolojisi](../../assets/img/gpu/nvlink-nodes.png)
*Şekil 3.1: GPU'lar ile 4 yerleşik NVSwitch arasındaki yüksek yoğunluklu NVLink hatlarını yönlendiren HGX 8 GPU taban kartı mimarisi.*

![NVIDIA NVLink 4 Switch Crossbar Mimarisi](../../assets/img/gpu/nvlink4.png)
*Şekil 3.2: Hot Chips 34'te sunulan, donanımsal indirgeme yeteneğine sahip ve çift yönde 3,2 TB/s yönlendirebilen 64 portlu NVLink 4 switch ASIC'inin iç şeması.*

Blackwell nesliyle birlikte NVIDIA, **GB200 NVL72** mimarisiyle bu denklemi kökten değiştirdi. NVLink etki alanını 8 GPU'lu bir kasayla sınırlamak yerine, sıvı soğutmalı kabinler 72 adet B200 GPU'yu 18 adet NVSwitch 5 ASIC'i üzerinden bakır arka panelde birleştirir:

![GB200 NVL72 Kabin Mimarisi](../../assets/img/gpu/b200-node.png)
*Şekil 3.3: 72 GPU ve 18 NVLink switch'i tek bir 900 GB/s NVLink etki alanında birleştiren GB200 NVL72 kabin yapısı.*

NVL72'de 72 GPU'nun tümü, herhangi bir çipin diğerinin belleğine 900 GB/s hızla erişebildiği tek bir NVLink etki alanını paylaşır ve tek düğüm sınırını 9 kat genişletir.

<details>
<summary>Quiz: Bir H100 SXM5 GPU'da kaç NVLink portu bulunur ve toplam bant genişliği nedir?</summary>

H100 SXM5 çipinde 18 adet NVLink 4 portu bulunur. Her port tek yönde 25 GB/s hızla çalışarak GPU başına tek yönde 450 GB/s, çift yönde 900 GB/s toplam bant genişliği sağlar.
</details>

### Düğümler ötesi ölçekleme: SuperPod ve rail-optimized fat-tree topolojisi

Yüz milyarlarca parametreli modelleri eğitmek için kümenin binlerce GPU'ya ölçeklenmesi gerekir. Bu katman **InfiniBand** (veya RoCE v2 Ethernet) tarafından yönetilir.

1.024 GPU'lu bir DGX H100 SuperPod kümesinde:
- Küme, her biri 32 DGX düğümü (256 GPU) içeren **Ölçeklenebilir Birimlere (Scalable Unit / SU)** ayrılır.
- Her düğümde 8 adet ConnectX-7 400 Gbps (NDR) NIC bulunur; bu da düğüm başına $8 \times 50 \text{ GB/s} = 400 \text{ GB/s}$ (GPU başına 50 GB/s) dış ağ çıkışı sağlar.
- Her SU içinde 8 Leaf switch bulunur. 4 SU genelinde 16 merkezi Spine switch, ağı 51,2 TB/s anahtarlama kapasitesine sahip 1:1 **Fat-Tree (Şişman Ağaç)** topolojisinde birleştirir.

Kritik olarak modern kümeler **Rail-Optimized (Ray Uyumlu)** kablolama kullanır. Bu topolojide her düğümdeki $i$. GPU rank'ı doğrudan $i$. Leaf switch'e bağlanır. Aynı indeksli GPU'lar arasında veri paraleli veya boru hattı kolektifleri yürütülürken paketler spine katmanına çıkmadan kendi ray switch'i içinde kalır; gecikme ve omurga tıkanıklığı önlenir:

<svg viewBox="0 0 560 440" role="img" aria-label="1024 GPU&#x27;lu DGX H100 SuperPod Fat-Tree InfiniBand ağ topolojisi diyagramı." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="sp-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="sp-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">SuperPod Fat-Tree Ölçekleme Topolojisi</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1024 GPU · Tam Biseksiyon Bant Genişliği</text>
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
<text x="480.0" y="266" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">32 Düğüm (256 Çip)</text>
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

H100 ve GB200 SuperPod kümelerinin resmi topolojileri aşağıda belgelenmiştir:

![NVIDIA SuperPod InfiniBand Ağ Topolojisi](../../assets/img/gpu/superpod-diagram.png)
*Şekil 3.4: Hesaplama düğümleri ile depolama omurgası arasındaki leaf-spine bağlantılarını gösteren mantıksal SuperPod yerleşimi.*

![DGX H100 SuperPod Küme Mimarisi](../../assets/img/gpu/h100-superpod.png)
*Şekil 3.5: 128 DGX düğümü, ray uyumlu InfiniBand switch'leri ve yönetim kabinlerinden oluşan 1.024 GPU'lu fiziksel H100 SuperPod kurulumu.*

![GB200 NVL72 SuperPod Mimarisi](../../assets/img/gpu/gb200-superpod.png)
*Şekil 3.6: 576 B200 GPU'yu Quantum-X800 800 Gbps InfiniBand switch'leriyle bağlayan Blackwell GB200 NVL72 çok kabinli ölçekleme topolojisi.*

### Kolektif iletişim: Halka algoritmalarından donanımsal SHARP'a

Dağıtık eğitim, NCCL (NVIDIA Collective Communications Library) içinde uygulanan temel kolektif iletişim primitiflerine dayanır:
1. **AllGather:** Her rank $B/N$ boyutunda bir parça tutar; tüm parçalar toplanarak her GPU'da $B$ boyutunda tam tensör oluşturulur.
2. **ReduceScatter:** Her rank $B$ boyutunda girdi tutar; elemanlar rank'lar arasında toplanır ve $i$. rank yalnızca $B/N$ boyutundaki $i$. parçayı saklar.
3. **AllReduce:** ReduceScatter ve ardından AllGather işlemine eşdeğerdir. Tüm rank'lar arasında eleman bazında toplama yapar ve $B$ boyutundaki nihai tensörü $N$ GPU'nun tamamına çoğaltır.
4. **AllToAll:** Her rank diğer tüm rank'lara farklı bir tensör dilimi gönderir; MoE uzman dağıtımının temelini oluşturur.

AllGather kolektifinin adım adım akışı aşağıdaki animasyonda gösterilmektedir:

![AllGather Kolektif İletişim Adımı](../../assets/img/gpu/all-gather.gif)
*Şekil 3.7: Dört GPU arasında bir AllGather kolektifinin adım adım iletişim modeli.*

Ağ çıkış bant genişliği $W$ olan $N$ cihazlık klasik bir **Ring AllGather** algoritmasında:
- Her cihaz tensörünü $B/N$ boyutunda $N$ parçaya böler.
- Her adımda her rank bir parçayı komşusuna aktarırken diğerinden yenisini alır.
- Halka $N-1$ adımda tamamlanır:

$$T_{\text{AllGather, Ring}} = (N - 1) \cdot \frac{B / N}{W} = \frac{N - 1}{N} \cdot \frac{B}{W} \approx \frac{B}{W}$$

Bir AllReduce hem ReduceScatter hem AllGather gerektirdiğinden, klasik halka AllReduce işlemi $2 \times \frac{N-1}{N} B$ bayt aktarır ve şu süreyi alır:
$$T_{\text{AllReduce, Ring}} \approx 2 \cdot \frac{B}{W}$$

Bu $2\times$ maliyetini ortadan kaldırmak için modern InfiniBand switch'leri **Mellanox SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** teknolojisini kullanır. SHARP, kayan nokta toplama mantığını doğrudan Quantum switch ASIC'lerine gömer:

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
<text x="144" y="270" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Süre: T ≈ B / W_egress</text>
<text x="144" y="286" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">N-1 adımda halka dolaşımı</text>
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

Tensörleri halkada dolaştırmak yerine her GPU tensörünü switch'e bir kez basar. Switch ASIC'i sayıları hatta akarken toplar ve sonucu tek geçişte tüm GPU'lara geri dağıtır:

$$T_{\text{AllReduce, SHARP}} \approx \frac{B}{W}$$

SHARP ağ trafiğini tam %50 azaltır ve AllReduce gecikmesini yarıya indirir.

GPU ve TPU kümelerinde farklı tensör boyutları için ölçülen ampirik bant genişliği eğrileri şöyledir:

![GPU AllGather Bant Genişliği](../../assets/img/gpu/gpu-all-gather-bw.png)
*Şekil 3.8: H100 kümelerinde tensör boyutuna bağlı AllGather verimi (16 MB üzerinde doyuma ulaşır).*

![GPU AllReduce Bant Genişliği](../../assets/img/gpu/gpu-all-reduce-bw.png)
*Şekil 3.9: Düğüm içi NVLink ve düğümler arası InfiniBand üzerinde AllReduce verimi.*

![TPU AllGather Bant Genişliği](../../assets/img/gpu/tpu-all-gather-bw.png)
*Şekil 3.10: Karşılaştırma için Google TPU v4/v5p 3D torus ağında AllGather bant genişliği.*

![TPU AllReduce Bant Genişliği](../../assets/img/gpu/tpu-all-reduce-bw.png)
*Şekil 3.11: Google TPU kümelerinde AllReduce bant genişliği.*

Parçalı AllGather gecikmesi ve SHARP donanımsal hızlandırmasının ölçümleri aşağıda karşılaştırılmıştır:

![Parçalı AllGather İletişim Maliyeti](../../assets/img/gpu/sharded-all-gather-cost.png)
*Şekil 3.12: Büyüyen küme ölçeklerinde parçalı AllGather işlemlerinin analitik gecikme dağılımı.*

![Mellanox SHARP Ağ İçi İndirgeme Ağacı](../../assets/img/gpu/sharp-algorithm.png)
*Şekil 3.13: Quantum switch ASIC'leri tarafından bir SHARP AllReduce sırasında yürütülen ağaç tabanlı donanım toplama yapısı.*

![SHARP AllReduce Başarım Karşılaştırması](../../assets/img/gpu/sharp-all-reduce-cost.png)
*Şekil 3.14: SHARP ağ içi hesaplamanın klasik iki geçişli halka algoritmasına kıyasla sağladığı gecikme kazancı.*

MoE modellerinde seyrek token yönlendirmesi için düzensiz **AllToAll** iletişimi kullanılır:

![Seyrek MoE AllToAll Yaklaşımı](../../assets/img/gpu/all-to-all-approx.png)
*Şekil 3.15: Seyrek MoE token yönlendirmesinde tam kombinatorik beklenti ile analitik yaklaşım arasındaki uyum.*

<details>
<summary>Quiz: Mellanox SHARP klasik bir halka AllReduce'a kıyasla ağ trafiğini ne kadar azaltır?</summary>

Klasik halka AllReduce iki aşamadan oluşur (yaklaşık $B$ bayt aktaran ReduceScatter ve $B$ bayt aktaran AllGather); yani GPU başına yaklaşık $2B$ bayt veri taşınır. SHARP indirgemeyi doğrudan switch ASIC'inde tek geçişte yaptığı için GPU başına yalnızca $B$ bayt aktarılır ve toplam ağ trafiği tam %50 azalır.
</details>

### LLM ölçekleme için 4D paralellik roofline sınırları

Yüz milyarlarca parametreli modelleri eğitmek dört temel paralellik boyutunun harmanlanmasını gerektirir: **Tensör Paralelliği (TP)**, **Boru Hattı Paralelliği (PP)**, **Uzman Paralelliği (EP)** ve **Veri Paralelliği (DP / FSDP)**.

#### Tensör Paralelliği (TP) ve 8 GPU Tavanı

Megatron-LM tarzı tensör paralelliğinde Transformer MLP bloğu $Y$ parçaya bölünür: $W_{\text{in}}$ sütunlara ($[D, F/Y]$), $W_{\text{out}}$ satırlara ($[F/Y, D]$) paylaştırılır.
- **İleri Hesaplama Süresi:** $T_{\text{math}} = \frac{4 B D F}{Y \cdot C_{\text{peak}}}$
- **İletişim Süresi:** Aktivasyonlar üzerinde bir AllReduce: $T_{\text{comms}} = \frac{4 B D}{W_{\text{collective}}}$

Hesaplamanın iletişimden hızlı kalması ($T_{\text{math}} \ge T_{\text{comms}}$) için:

$$Y \le \frac{F \cdot W_{\text{collective}}}{C_{\text{peak}}} = \frac{F}{\alpha_{\text{crit}}}$$

Düğüm içi NVLink kullanan bir H100 için ($W_{\text{collective}} = 450 \text{ GB/s}$, $C_{\text{peak}} = 990 \text{ TFLOP/s}$):
$$\frac{C_{\text{peak}}}{W_{\text{collective}}} = \frac{990 \times 10^{12}}{450 \times 10^9} = 2.200$$

LLaMA-3 70B ($F = 28.672$) için:
$$Y \le \frac{28.672}{2.200} \approx 13,03 \implies Y = 8 \text{ GPU}$$

Bu matematiksel sınır, **Tensör Paralelliğinin neden kesinlikle 8 GPU'luk tek bir NVLink kasasına hapsedildiğini** açıklar. TP'yi InfiniBand üzerinden 16 GPU'ya çıkarmaya kalktığınızda ($W_{\text{collective}} = 400 \text{ GB/s}$ düğüm çıkışı, yani GPU başına 50 GB/s), $C/W$ oranı 19.800'e fırlar, izin verilen $Y$ değeri 1,5'in altına düşer ve donanım verimi çöker.

#### Boru Hattı Paralelliği (PP) ve Kabarcık Kaybı

Boru hattı paralelliği model katmanlarını $P$ aşamaya böler. Komşu aşamalar mikrobatch başına yalnızca $2BD / N_{\text{MB}}$ boyutunda aktivasyon tensörleri takas ettiğinden ağ trafiği önemsizdir.

Ancak boru hattı, aşamaların birbirini beklemesinden kaynaklanan **kabarcık (bubble)** kaybı yaratır:

$$\text{Kabarcık Oranı} = \frac{P - 1}{P - 1 + M}$$

Şöyle okuyun: $P$ boru hattı aşama sayısı, $M$ ise küresel batch içindeki mikrobatch sayısıdır.

![1F1B Çizelgesinde Boru Hattı Kabarcıkları](../../assets/img/gpu/pipeline-bubble.png)
*Şekil 3.16: 2 aşamalı ve 2 mikrobatchli boru hattı çizelgesi. Beyaz boşluklar kabarcık kayıplarını, 'G' ise gradyan senkronizasyonunu temsil eder.*

![Aktivasyon Aktarım Gecikmesinin Kabarcığa Etkisi](../../assets/img/gpu/pipeline-transfer.png)
*Şekil 3.17: Aşamalar arasındaki aktivasyon iletim gecikmesi boru hattını kaydırarak kabarcık oranını artırır.*

#### Veri Paralelliği (DP & FSDP) ve Kritik Batch Boyutu

Veri Paralelliğinde (ZeRO-3 / FSDP dahil) parametreler ve gradyanlar $X$ cihaza dağıtılır:
$$T_{\text{math}} = \frac{6 B D F}{X \cdot C_{\text{peak}}}, \quad T_{\text{comms}} = \frac{2 \cdot 2 \cdot DF \cdot (X - 1)}{X \cdot W_{\text{collective}}}$$

$T_{\text{math}} \ge T_{\text{comms}}$ şartından **Kritik Yerel Batch Boyutu** çıkar:

$$\frac{B}{X} \ge \frac{2}{3} \cdot \frac{C_{\text{peak}}}{W_{\text{collective}}} \approx 2.475 \text{ token / GPU}$$

GPU başına düşen batch boyutu yaklaşık 2.500 tokenin altına indiğinde, InfiniBand üzerindeki gradyan senkronizasyonu hesaplamadan daha uzun sürer ve GPU'lar boş bekler.

$E$ toplam uzman ve token başına $k$ aktif uzmana sahip MoE modellerinde bu eşik felaket biçimde büyür:

$$\left(\frac{B}{X}\right)_{\text{crit, MoE}} \ge \frac{E}{k} \cdot \frac{C_{\text{peak}}}{W_{\text{collective}}}$$

$E = 128$ ve $k = 4$ olan bir modelde kritik batch boyutu $32\times$ katlanarak GPU başına 79.000 tokene fırlar; bu patlama ancak Uzman Paralelliği (EP) ile çözülebilir.

Dört paralellik boyutunun karar matrisi aşağıdaki yapısal diyagramda gösterilmektedir:

<svg viewBox="0 0 560 440" role="img" aria-label="LLM 4D paralellik matrisi: Tensör, Boru Hattı, Uzman ve Veri Paralelliği roofline sınırları." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="par-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="par-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="par-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">LLM Paralellik Matrisi ve Roofline Dengesi</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP · PP · EP · DP / FSDP Karar Ağacı</text>
<rect x="16" y="48" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="30" y="68" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Tensör Paralelliği (TP)</text>
<text x="30" y="84" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: Tek Düğüm (NVLink)</text>
<rect x="30" y="92" width="228" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="144.0" y="104.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">AllGather + ReduceScatter</text>
<text x="30" y="128" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: Y ≤ F / 2475 (LLaMA: Y ≤ 8)</text>
<text x="30" y="146" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Aşılırsa iletişim hesabı yutar</text>
<text x="30" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="288" y="48" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="302" y="68" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">Boru Hattı Paralelliği (PP)</text>
<text x="302" y="84" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: Çok Düğümlü Katmanlar</text>
<rect x="302" y="92" width="228" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="416.0" y="104.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">P2P Aktivasyon (Microbatch)</text>
<text x="302" y="128" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: Kabarcık = (P-1) / (P-1+M)</text>
<text x="302" y="146" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Düşük ağ yükü, zor çizelgeleme</text>
<text x="302" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="16" y="192" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="30" y="212" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:700">Uzman Paralelliği (EP)</text>
<text x="30" y="228" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: MoE (1-8 Düğüm)</text>
<rect x="30" y="236" width="228" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="144.0" y="248.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Ragged AllToAll İletişimi</text>
<text x="30" y="272" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: F &lt; 8C/W ise 1-2 düğüm</text>
<text x="30" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">MoE ağırlık patlamasını çözer</text>
<text x="30" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="288" y="192" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="302" y="212" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Veri Paralelliği (DP / FSDP)</text>
<text x="302" y="228" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: Tüm Küme (ZeRO-1 / 3)</text>
<rect x="302" y="236" width="228" height="19" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="416.0" y="248.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">AllReduce / ReduceScatter</text>
<text x="302" y="272" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: B_yerel ≥ 2.500 token / GPU</text>
<text x="302" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">MoE&#x27;de eşik E/k kat büyür</text>
<text x="302" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="16" y="336" width="528" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="356" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Modern Üretim Modellerinin Paralellik Reçeteleri:</text>
<text x="28" y="374" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px;font-family:var(--font-mono)">DeepSeek-V3 (2048x H800): 64-way EP (8 düğüm) + 16-way PP + 2-way DP (ZeRO-1)</text>
<text x="28" y="392" text-anchor="start" style="fill:var(--c-accent);font-size:10.5px;font-family:var(--font-mono)">LLaMA-3 405B (16k H100):  8-way TP (1 düğüm) + 16-way PP + 128-way DP (ZeRO-1)</text>
<text x="28" y="410" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">Temel kural: TP düğüm içine hapsedilir, PP ve EP düğüm sınırlarını aşar, DP ise ölçeği tamamlar.</text>
</svg>

Öncü modeller bu alanı şu reçetelerle fetheder:
- **LLaMA-3 405B (16.384 H100):** 8-way düğüm içi TP $\times$ 16-way PP $\times$ 128-way DP (ZeRO-1).
- **DeepSeek-V3 (2.048 H800):** 64-way EP (InfiniBand üzerinde 8 düğüme yayılan) $\times$ 16-way PP $\times$ 2-way DP (ZeRO-1), GPU başına 30.720 token.

Megatron-LM 3D paralellik uzayı aşağıda özetlenmiştir:

![Megatron-LM 3D Paralellik Ayar Uzayı](../../assets/img/gpu/megatron-hparams.png)
*Şekil 3.18: Megatron-LM araştırmasında TP, PP ve DP eksenlerinin farklı küme boyutlarındaki başarım alanları.*

### Öncü mühendislik: DeepSeek-V3 DualPipe ve iletişim-hesaplama çakışması

64-way Uzman Paralelliği düğümler arasına yayıldığında, AllToAll token yönlendirmesi InfiniBand hatlarını tıkama tehlikesi yaratır. DeepSeek-V3 bu darboğazı **DualPipe** ile çözmüştür.

DualPipe, ileri ve geri yönlerde aynı anda iki bağımsız boru hattı çizelgesi yürütür. Katmanları ince taneli bloklara bölerek, bir bloğun düğümler arası AllToAll token transferini arka planda yürütürken GPU Tensor Core birimlerinde diğer bloğun yoğun dikkat ve GEMM hesaplamalarını çalıştırır. İletişimi hesaplamayla kusursuz biçimde çakıştırarak (overlap) DeepSeek, ağ gecikmesinin %95'ten fazlasını gizler ve kısıtlı H800 kümelerinde dahi tepe MFU'ya ulaşır.

---

## 4. 3. Sütun — Depolama: Temel model veri setleri, kontrol noktaları ve yüksek verimli G/Ç

### Modern temel modellerin ön eğitimindeki depolama krizi

Hesaplama ve ağ mimarisi ön planda olsa da, büyük dil modeli kümeleri depolama G/Ç (I/O) yetersizlikleri yüzünden binlerce saat boyunca felç olur. Ön eğitimde depolama alt sistemi iki devasa yük altındadır:

1. **Sürekli Veri Besleme (Data Streaming):** LLaMA 3 gibi bir temel modeli eğitmek 15 trilyondan fazla tokenin okunmasını gerektirir. 16.384 GPU'luk bir kümede veri besleme hattı, PyTorch iş parçacıklarını düşürmeden saniyede 2 ila 4 Terabayt kesintisiz okuma hızını korumalıdır.
2. **Kontrol Noktası (Checkpointing) Krizi:** Donanım arızalarına karşı model durumları periyodik olarak kalıcı depolamaya yazılmalıdır. 405 milyar parametreli bir modelin eğitim sırasındaki bellek faturası şöyledir:

$$S_{\text{state}} = S_{\text{weights}} + S_{\text{gradients}} + S_{\text{AdamW}} = 2M + 2M + 12M = 16M \text{ bayt}$$

$M = 405 \times 10^9$ parametre için:
- **BF16 Model Ağırlıkları:** $405 \times 10^9 \times 2 \text{ bayt} = 810 \text{ GB}$
- **BF16 Gradyanlar:** $405 \times 10^9 \times 2 \text{ bayt} = 810 \text{ GB}$
- **FP32 AdamW Optimizer Durumu:** Master ağırlıklar (4 bayt) + Momentum (4 bayt) + Varyans (4 bayt) = $405 \times 10^9 \times 12 \text{ bayt} = 4.860 \text{ GB} = 4,86 \text{ TB}$

Bunların toplamı $6.480 \text{ GB} \approx 6,48 \text{ TB}$ eder. Aktivasyon tamponları, rastgele sayı üreteci (RNG) durumları ve adım meta verileri eklendiğinde tek bir kontrol noktası boyutu **~7,5 Terabayta** ulaşır.

Eğer küme bu 7,5 TB veriyi geleneksel 25 GB/s'lik bir paylaşımlı depolama omurgasına eşzamanlı olarak yazmaya kalkarsa, **16.384 GPU'luk dev süper bilgisayar her kayıtta 5 dakikadan fazla ($307 \text{ saniye}$) tamamen donar**. Haftalar süren bir eğitim maratonunda bu duraklamalar milyonlarca dolarlık GPU döngüsünü çöpe atar.

### Geleneksel G/Ç darboğazı: Host CPU sayfa önbelleği ve bellek kopyalama

Geleneksel kümelerdeki G/Ç darboğazının temel nedeni işletim sisteminin çok sekmeli mimarisidir:
1. **NVMe Okuma/Yazma:** Veri NVMe SSD'den PCIe üzerinden ana sunucu CPU root complex birimine gelir.
2. **Kernel Sayfa Önbelleği Kopyası:** Linux depolama yığını veriyi ana sunucunun DDR5 sistem belleğindeki sayfa önbelleğine kopyalar.
3. **Kullanıcı Alanı Tamponu:** PyTorch veya kontrol noktası yöneticisi veriyi çekirdek alanından kullanıcı alanı RAM'ine kopyalar.
4. **Host-to-Device Aktarımı:** Host CPU, PCIe Gen 5 üzerinden veriyi GPU HBM belleğine basmak için ikinci bir DMA transferi başlatır.

Bu geleneksel yol, her baytın PCIe veri yolunu iki kez geçmesini ve iki ayrı CPU bellek kopyalamasına (`memcpy`) uğramasını zorunlu kılar. Çift soketli bir modern sunucu CPU'su %100 çekirdek yüküyle 50-100 GB/s hızda bellek bant genişliğini tüketir ve depolamayı felç eder.

### GPUDirect Storage (GDS) ve doğrudan DMA mimarisi

Modern yapay zekâ süper bilgisayarları bu darboğazı **GPUDirect Storage (GDS)** ile aşar:

<svg viewBox="0 0 560 420" role="img" aria-label="Geleneksel CPU bellek kopyalı G/Ç ile GPUDirect Storage doğrudan DMA mimarisi karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="gds-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="gds-arr-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="gds-arr-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="gds-arr-suc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
<marker id="gds-arr-warn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="22" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Depolama Mimarisi: CPU Bounce vs GPUDirect Storage</text>
<text x="16" y="38" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Checkpointing ve Veri Besleme Boru Hattı</text>
<rect x="16" y="48" width="256" height="286" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="68" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">Geleneksel G/Ç Yolu</text>
<text x="144" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">CPU Sayfa Önbelleği &amp; Bounce Buffer</text>
<rect x="32" y="252" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="144" y="274" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">NVMe Depolama (SSD)</text>
<rect x="32" y="150" width="224" height="54" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="144" y="170" text-anchor="middle" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">Host CPU &amp; DDR5 RAM</text>
<text x="144" y="188" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Kernel Sayfa Önbelleği (50-100 GB/s)</text>
<rect x="32" y="96" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="144" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU HBM3 (3,35 TB/s)</text>
<line x1="144" y1="250" x2="144" y2="206" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="232" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">G/Ç Adım 1</text>
<line x1="144" y1="148" x2="144" y2="134" marker-end="url(#gds-arr-warn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">G/Ç Adım 2</text>
<rect x="24" y="302" width="240" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="144.0" y="316.3" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Darboğaz: CPU ve RAM Kopyası</text>
<rect x="288" y="48" width="256" height="286" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="68" text-anchor="middle" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">GPUDirect Storage (GDS)</text>
<text x="416" y="84" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Doğrudan DMA (cuFile API)</text>
<rect x="304" y="252" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="416" y="274" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">NVMe Depolama (SSD)</text>
<rect x="304" y="162" width="224" height="42" rx="5" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="416" y="180" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:700">PCIe Gen 5 Switch / DMA</text>
<text x="416" y="196" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Doğrudan Donanım Yolu (64+ GB/s)</text>
<rect x="304" y="96" width="224" height="36" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="416" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">GPU HBM3 (3,35 TB/s)</text>
<line x1="416" y1="250" x2="416" y2="206" marker-end="url(#gds-arr-suc)" style="stroke:var(--c-success);stroke-width:2"/>
<line x1="416" y1="160" x2="416" y2="134" marker-end="url(#gds-arr-suc)" style="stroke:var(--c-success);stroke-width:2"/>
<text x="424" y="232" text-anchor="start" style="fill:var(--c-success);font-size:10px;font-weight:600">Doğrudan DMA</text>
<rect x="296" y="302" width="240" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="416.0" y="316.3" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">CPU Baypas · Doğrudan HBM DMA</text>
<rect x="16" y="344" width="528" height="64" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="364" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Checkpointing Zorluğu: 405B modelinde 7,5 TB durum kaydedilir.</text>
<text x="28" y="382" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GDS sayesinde depolama birimleri CPU ve sistem RAM&#x27;ini tamamen baypas ederek,</text>
<text x="28" y="398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">doğrudan GPU HBM belleğine veri basar; GPU bekleme süresi sıfıra yaklaşır.</text>
</svg>

GDS, `cuFile` API'si ve `nvidia-fs.ko` çekirdek modülü üzerinden çalışır. Yerel NVMe sürücüler (veya InfiniBand üzerinden NVMe-oF) ile GPU HBM belleği arasında PCIe switch katmanında doğrudan bir DMA (Direct Memory Access) köprüsü kurar; ana sunucu CPU'sunu, kullanıcı belleğini ve Linux sayfa önbelleğini tamamen baypas eder.

CPU bellek kopyalamaları ortadan kalktığında:
- Depolama verimi doğrudan PCIe Gen 5 hat hızına (**denetleyici başına 64+ GB/s**) ulaşır.
- Uçtan uca G/Ç gecikmesi **%80** azalır.
- Host CPU yükü **%100'den neredeyse %0'a** düşerek CPU çekirdeklerini veri işleme ve orkestrasyona bırakır.

<details>
<summary>Quiz: GPUDirect Storage (GDS) kontrol noktası yazarken CPU yükünü neden sıfıra indirir?</summary>

Geleneksel G/Ç'de ana sunucu CPU'su çekirdek önbelleği ile kullanıcı belleği arasında gigabaytlarca veriyi taşımak için sürekli `memcpy` komutları yürütür. GDS ise PCIe DMA motorlarını doğrudan programlayarak veriyi NVMe denetleyicisi ile GPU HBM arasında CPU ve DDR5 RAM'e hiç uğramadan akıtır; böylece CPU çekirdekleri tamamen boşta kalır.
</details>

### Eksabayt ölçeğinde dağıtık dosya sistemleri: Meta Tectonic vaka analizi

Binlerce düğümlü devasa bir kümede yerel diskler global kontrol noktalarını barındıramaz. Kümeler eksabayt ölçeğinde dağıtık dosya sistemlerine bağlanır.

Meta'nın resmi LLaMA 3 altyapı raporuna göre 16.384 GPU'luk küme Meta'nın **Tectonic** dağıtık dosya sistemiyle beslenmiştir:
- **Ölçek:** 240 Petabayt depolama kapasitesi sunan 7.500 adet özel SSD depolama sunucusu.
- **Sürekli Okuma Verimi:** 2.048 hesaplama düğümündeki veri besleme süreçlerini beslemek için kesintisiz **2 TB/s** okuma bant genişliği.
- **Anlık Yazma Verimi:** **7 TB/s anlık tepe yazma bant genişliği**; bu sayede 7,5 TB'lık global kontrol noktası tüm rank'lar tarafından asenkron kopyalama ile yaklaşık **1 saniyede** kalıcı depolamaya dökülebilir.
- **Güvenilirlik:** Sürücü ve kabin arızalarına karşı veriyi kaybetmemek için Reed-Solomon silme kodlaması ($9+3$ parça) kullanır.

### Yüksek verimli veri besleme: İkili token indeksleme ve sıfır kopya mmap

Milyonlarca tokeni GPU'lara kesintisiz beslemek özel veri yükleyici (dataloader) mühendisliği gerektirir:
- **Standart DataLoader Neden Çöker:** Ham metin veya JSONL dosyaları yüklemek, Python iş parçacıklarının sürekli string parse etmesine ve dinamik nesneler üretmesine yol açarak Python Global Interpreter Lock (GIL) kilitlenmelerine neden olur.
- **Bitişik İkili Düzen (.bin / .idx):** Üretim kütüphaneleri (Megatron-LM, Lit-GPT) veri setini diskte bitişik `uint16` veya `int32` token dizileri halinde tek bir ikili dosyada (`.bin`) saklar; yanında doküman sınırlarını gösteren bir indeks dosyası (`.idx`) tutulur.
- **Sıfır Kopya Bellek Eşleme (`mmap`):** İş parçacıkları POSIX `mmap()` çağrısıyla dosyayı doğrudan sanal adres alanına eşler. İşletim sistemi token bloklarını ihtiyaç duyuldukça kullanıcı belleğine ek kopyalama yapmadan sayfalayarak getirir.
- **NUMA Uyumlu Ön Getirme:** Düğümlerde iki ayrı CPU soketi bulunur ve her soket 4 GPU'ya fiziksel olarak bağlıdır (Şekil 1.1'deki düğüm topolojisinde CPU 0 GPU 0–3'ü, CPU 1 GPU 4–7'yi besler). DataLoader süreçleri doğrudan hedef GPU'nun PCIe root complex birimine bağlı CPU soketine kilitlenir (NUMA affinity) ve CUDA asenkron akışlarıyla (`pin_memory=True`) NUMA hatlarını geçmeden doğrudan HBM'e aktarılır.

### Küme hata toleransı, MTBF ve Young-Daly kontrol noktası formülü

16.384 GPU'luk (2.048 düğüm) bir kümede donanım arızası istisna değil, sürekli bir çalışma koşuludur:
- Tek bir DGX düğümünün Arızalar Arası Ortalama Süresi (MTBF) 3 yıl (~26.000 saat) olsa bile, kümenin efektif MTBF'si şöyledir:

$$M_{\text{küme}} = \frac{26.000 \text{ saat}}{2.048 \text{ düğüm}} \approx 12,7 \text{ saat}$$

Pratikte sessiz veri bozulmaları (SDC), HBM3 ECC bellek hataları, optik alıcı-verici düşmeleri ve ağ zaman aşımları eklendiğinde 16k GPU'luk kümede her **2 ila 3 saatte bir** ölümcül bir arıza yaşanır.

Bir düğüm çöktüğünde tüm küme son kaydedilen kontrol noktasından yeniden başlamak zorundadır. Bu durum temel bir matematiksel ödünleşme (trade-off) doğurur:
- Çok sık kontrol noktası alırsanız zamanınızı diske yazarak harcarsınız ($\delta$).
- Çok seyrek kontrol noktası alırsanız bir çöküşte saatlerce geriye düşüp hesaplamayı baştan yaparsınız.

Matematiksel olarak en uygun kayıt aralığı klasik **Young-Daly Formülü** ile hesaplanır:

$$\tau_{\text{opt}} = \sqrt{2 \cdot \delta \cdot M}$$

Burada:
- $\delta$ tek bir kontrol noktasını kaydetmek için geçen süre (saniye).
- $M$ kümenin Arızalar Arası Ortalama Süresi (MTBF, saniye).
- $\tau_{\text{opt}}$ ardışık iki kayıt arasındaki en uygun süredir.

LLaMA-3 405B için bu dengeyi hesaplayalım ($M = 3 \text{ saat} = 10.800 \text{ saniye}$):
- **Senaryo A: Geleneksel Depolama (25 GB/s küme yazma hızı):**
  $$\delta = \frac{7,5 \times 1024 \text{ GB}}{25 \text{ GB/s}} \approx 307,2 \text{ s} \approx 5,12 \text{ dakika}$$
  $$\tau_{\text{opt}} = \sqrt{2 \cdot 307,2 \cdot 10.800} \approx \sqrt{6.635.520} \approx 2.576 \text{ s} \approx 42,9 \text{ dakika}$$
  Her 43 dakikada bir kayıt almak $\frac{5,12}{42,9} \approx \%11,9$ doğrudan kayıt kaybı ve $\frac{\tau}{2M} \approx \%11,9$ beklenen geri sarma kaybı yaratır; **yani süper bilgisayarın hesaplama gücünün neredeyse %24'ü çöpe gider!**
- **Senaryo B: GPUDirect Storage & Tectonic Anlık Yazma (250 GB/s yazma hızı):**
  $$\delta = \frac{7,5 \times 1024 \text{ GB}}{250 \text{ GB/s}} \approx 30,72 \text{ s} \approx 0,51 \text{ dakika}$$
  $$\tau_{\text{opt}} = \sqrt{2 \cdot 30,72 \cdot 10.800} \approx \sqrt{663.552} \approx 814,6 \text{ s} \approx 13,6 \text{ dakika}$$
  Kayıt kaybı $\frac{0,51}{13,6} \approx \%3,8$, beklenen geri sarma kaybı da $\frac{\tau}{2M} \approx \%3,8$ olur; toplam kayıp **%24'ten yaklaşık %7,5'e** iner ve milyonlarca dolarlık hesaplama zamanı kurtarılır. Yazma 10 kat hızlanınca kayıp $\sqrt{10} \approx 3,2$ kat azalır, çünkü $\tau_{\text{opt}}$ noktasında iki kayıp terimi de $\sqrt{\delta}$ ile ölçeklenir.

<details>
<summary>Quiz: MTBF = 4 saat ve kayıt süresi = 2 dakika olan bir kümede ideal kontrol noktası alma sıklığı nedir?</summary>

Young-Daly formülü uygulandığında:
$\tau_{\text{opt}} = \sqrt{2 \cdot \delta \cdot M} = \sqrt{2 \cdot 2 \text{ dk} \cdot 240 \text{ dk}} = \sqrt{960} \approx 30,98 \text{ dakika}$.
Yaklaşık her 31 dakikada bir kontrol noktası almak toplam zaman kaybını en aza indirir.
</details>

---

## 5. Sentez: Eğitim ve çıkarım arasındaki operasyonel dengeler

### İki mühendislik gözü: Eğitim ve çıkarım

Donanım kararları model eğitimi ile yüksek hacimli çıkarım sunumu arasında taban tabana zıttır:

| Mimari Boyut | Eğitim Tarafı (Pre-training / SFT) | Çıkarım Tarafı (Prefill & Decode) |
| :--- | :--- | :--- |
| **Temel Bellek Faturası** | Ağırlıklar ($2B$) + Gradyanlar ($2B$) + AdamW Durumları ($12-16B$) + Aktivasyonlar | Model Ağırlıkları ($4-8$ bit nicemlenmiş) + Dinamik tahsis edilen KV Cache |
| **Hesap / Bellek Kısıtı** | Yoğun biçimde **Compute-Bound** (Binlerce tokenlik GEMM matris çarpımları) | Prefill evresi **Compute-Bound**; Decode evresi kesinlikle **Memory-Bandwidth-Bound** ($B=1$, GEMV) |
| **Baskın Ağ Primitifi** | InfiniBand üzerinde yüksek verimli **AllReduce** & **ReduceScatter** | NVLink üzerinde ultra düşük gecikmeli **AllGather** & **P2P Tensör Paralelliği** |
| **Kritik Metrikler** | Model FLOP Kullanımı (MFU), Adım Süresi (ms), Gradyan Kararlılığı | İlk Token Gecikmesi (TTFT), Token Başına Üretim Süresi (TPOT), Eşzamanlı İstek |
| **Kritik Riskler** | Loss spike'lar, sessiz veri bozulmaları (SDC), kabarcık tıkanıklıkları | KV Cache bellek patlaması (OOM), bellek parçalanması, istek kesintileri |

### Uçtan uca küme yürüyüşü: 16 bin GPU'da 70B ve 405B modeller

Üç sütunumuzu somutlaştırmak için NVIDIA H100 SuperPod üzerinde LLaMA-3 70B ve 405B modellerini inceleyelim (Meta'nın gerçek 405B kümesi InfiniBand değil, GPU başına 400 Gb/s RoCE kullandı; aşağıdaki SHARP adımı SuperPod senaryosuna aittir):

1. **8x H100 DGX Düğümünde LLaMA-3 70B:**
   - 70 milyar parametre BF16 formatında 140 GB ağırlık belleği kaplar.
   - 8 H100 GPU toplamda $8 \times 80 = 640 \text{ GB}$ HBM3 sunar.
   - Tensör Paralelliği $Y = 8$ olarak ayarlanır; tüm ağırlıklar ve aktivasyonlar 3,6 TB/s biseksiyonlu NVLink alanında kalır.
   - Her Transformer katmanı NVLink üzerinden iki AllReduce yürütür (kolektif gecikmesi: $<15 \, \mu\text{s}$).
   - MFU %48-52 bandında çalışır.
2. **16.384 H100 GPU'da LLaMA-3 405B (2.048 Düğüm):**
   - 4D Paralellik: 8-way düğüm içi TP $\times$ 16-way PP $\times$ 128-way DP (ZeRO-1).
   - TP kesinlikle 8 GPU'luk kasalar içine hapsedilir.
   - 16 boru hattı aşaması InfiniBand üzerinden komşu aşamalarla aktivasyon takas eder (P2P paket boyutu: $\sim 16 \text{ MB}$).
   - 128-way ZeRO-1 veri paraleli rank'ları Mellanox SHARP hızlandırmalı AllReduce ile ray switch'leri üzerinde toplanır.
   - Depolama alt sistemi `mmap` sıfır kopya ile 2 TB/s sürekli veri akıtırken, GPUDirect Storage 7,5 TB'lık kontrol noktalarını her 15 dakikada bir yaklaşık 30 saniyelik anlık yazma ile kaydeder.

---

## Bütün hikâye altı satırda

- Modern bir GPU süper bilgisayarı birbirine bağlı üç fiziksel sütun tarafından yönetilir: Streaming Multiprocessor hesaplama, çok katmanlı ağ omurgası ve yüksek verimli depolama G/Ç.
- Roofline modeli H100 üzerinde ~295 FLOP/baytlık kritik aritmetik yoğunluk dikte eder; bellek darboğazından kurtulmak için SM başına ~300 tokenlik batch gerekir.
- Düğüm içi NVLink çapraz bağlantıları GPU başına 450 GB/s tek yönlü bant genişliği (3,6 TB/s biseksiyon) sunarak 8-way Tensör Paralelliğini hesaplama sınırında tutar.
- Düğümler arası InfiniBand fat-tree ağı binlerce GPU arasında tam biseksiyon sunar; bant genişliği çip başına 50 GB/s (düğüm başına 400 GB/s) seviyesine iner.
- Mellanox SHARP, kayan nokta toplamayı doğrudan switch ASIC'leri içinde yürüterek küme genelindeki AllReduce trafiğini ve gecikmesini yarıya indirir.
- GPUDirect Storage, CPU bellek kopyalamalarını ortadan kaldırarak 7,5 TB'lık kontrol noktalarını saniyeler içinde kaydeder ve Young-Daly ideal aralığını koruyarak MFU'yu kurtarır.

---

## Terimler sözlüğü

- **Streaming Multiprocessor (SM)** — NVIDIA GPU'larının Tensor Core, Warp Scheduler, CUDA çekirdeği ve yerleşik SRAM barındıran temel çok iş parçacıklı hesaplama hücresi.
- **Warp** — NVIDIA mimarilerinde bir SM warp scheduler tarafından eşzamanlı olarak yürütülen 32 paralel iş parçacığından oluşan temel çizelgeleme birimi.
- **Warp Divergence (Warp Sapması)** — Bir warp içindeki iş parçacıklarının farklı dallara sapması sonucu aktif olmayan çekirdeklerin duraklamasıyla oluşan başarım kaybı.
- **NVLink** — H100 GPU başına tek yönde 450 GB/s (çift yönde 900 GB/s) bellek aktarımı sağlayan NVIDIA'nın tescilli yüksek bant genişlikli ara bağlantısı.
- **NVSwitch** — Bir HGX kasasındaki 8 GPU arasında engelsiz ve tam bağlantılı (all-to-all) iletişim kuran düğüm içi çapraz bağlantı ASIC'i.
- **InfiniBand Fat-Tree** — Küme düğümleri arasında aşırı abonelik (oversubscription) olmadan tam biseksiyon bant genişliği sağlayan hiyerarşik çok katmanlı ağ topolojisi.
- **Mellanox SHARP** — Switch donanımında kayan nokta AllReduce işlemlerini hatta akarken toplayan ölçeklenebilir hiyerarşik toplama protokolü.
- **Model FLOPs Utilization (MFU)** — Teorik olarak gereken model FLOP sayısının donanımın fiilen sürdürdüğü kayan nokta işlem gücüne oranı.
- **Tensör Paralelliği (TP)** — Tekil ağırlık matrislerini GPU'lar arasında bölen ve kesinlikle NVLink bant genişliğiyle sınırlı olan katman içi paralellik boyutu.
- **Boru Hattı Paralelliği (PP)** — Ardışık Transformer katmanlarını farklı GPU'lara atayan ve kabarcık (bubble) kaybına yol açan katmanlar arası paralellik boyutu.
- **GPUDirect Storage (GDS)** — NVMe depolama birimlerini ana sunucu CPU önbelleklerini baypas ederek doğrudan GPU HBM belleğine bağlayan doğrudan DMA teknolojisi.
- **Young-Daly Formülü** — Kayıt süresi ve küme MTBF değerine dayanarak ideal kontrol noktası alma aralığını belirleyen matematiksel bağıntı ($\tau = \sqrt{2 \delta M}$).

---

## Daha derine inmek için

- Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12.
- SemiAnalysis (2025). [NVIDIA Tensor Core Evolution from Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/). Mikro-mimari retrospektifi.
- DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437. 2.048 H800 GPU üzerinde 64-way Uzman Paralelliği ve DualPipe uygulaması.
- Meta AI (2024). [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). arXiv:2407.21783. 16.384 GPU 4D paralellik, Tectonic depolama ve eğitim altyapısı.
- NVIDIA Corporation. [DGX SuperPOD Reference Architecture (DGX H100): Key Components](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-components.html) ve [Network Fabrics](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html). Ölçeklenebilir birimler, dört ağ, UFM yuvalı 127 düğüm ve ayrı depolama ağı; Şekil 1.1'deki diyagramlara ilham verdi.
- NVIDIA Corporation. [Introduction to NVIDIA DGX H100/H200 Systems](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). Düğüm donanımı: işlemciler, bellek, NIC'ler ve NVMe sürücüler; DGX H100/H200 sistem topolojisi diyagramı Şekil 1.1'deki düğüm basamağına ilham verdi.
- NVIDIA Corporation. [Fabric Manager for NVIDIA NVSwitch Systems](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). Her H100'ün 18 NVLink'inin dört NVSwitch'e 5 + 4 + 4 + 5 olarak dağılması ve Fabric Manager olmadan NVLink ağının neden kurulmadığı.
- NVIDIA Corporation. [DGX SuperPOD Architecture (DGX H100)](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/dgx-superpod-architecture.html). Ölçek tablosu: 8 SU'ya kadar iki katman, 16 SU'dan itibaren core katmanı; 2.048 DGX için 512 leaf, 512 spine ve 256 core switch.
- NVIDIA Corporation. [DGX SuperPOD Data Center Design (H100): Planning](https://docs.nvidia.com/dgx-superpod/design-guides/dgx-superpod-data-center-design-h100/latest/planning.html). Tam bir SuperPOD'un neden 127 DGX olduğu (bir çift UFM cihazı bir yuvayı alır), kurulum başına 31–127 DGX ve kabin başına 4 DGX (40,8 kW).
- NVIDIA Developer Blog (2022). [Doubling all2all Performance with NVIDIA Collective Communication Library 2.12](https://developer.nvidia.com/blog/doubling-all2all-performance-with-nvidia-collective-communication-library-2-12/). PXN: raylar arası trafiği önce NVLink'e taşıyıp spine'ı atlamak.
- Deepak Narayanan vd. (2021). [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). arXiv:2104.04473. Tensör paralelliğinin neden sunucu içinde NVLink'te kaldığı, boru hattı ve veri paralelliğinin neden sunucular arasında koştuğu.
- Charles Clos (1953). [A Study of Non-Blocking Switching Networks](https://doi.org/10.1002/j.1538-7305.1953.tb01433.x). Bell System Technical Journal 32(2), 406–424. Bugünkü leaf–spine ağlarının dayandığı çok katlı anahtarlama fikri.
- Charles E. Leiserson (1985). [Fat-trees: Universal Networks for Hardware-Efficient Supercomputing](https://doi.org/10.1109/TC.1985.6312192). IEEE Transactions on Computers C-34(10), 892–901. Fat-tree adının kökeni: yukarı çıktıkça kalınlaşan bağlantılar.
- NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Resmi mikro-mimari kılavuzu.
- Bu blogda: [Bir Prompt'un Yolculuğu (8): Çıktı Katmanı (LM Head) — Megatron-LM ve Bellek Duvarı](post.html?slug=cikti-katmani-derinlemesine).
- Bu blogda: [Bir Prompt'un Yolculuğu (9): Bağlam Penceresinin İçi — 2K Duvarından FlashAttention'a](post.html?slug=baglam-penceresi-ve-dikkat).
