Kâğıt üzerinde 1.024 adet NVIDIA H100 GPU'dan oluşan bir küme yenilmez görünür: 1 ExaFLOP'un üzerinde 8-bit tensor compute kapasitesi, 80 Terabayt High-Bandwidth Memory (HBM) ve multi-terabit interconnect fabric omurgası. Ancak 70 milyar parametreli bir temel modelin (foundation model) ön eğitimini (pre-training) başlattığınız anda telemetri panoları sizi soğuk bir gerçekle yüzleştirir: Model FLOPs Utilization (MFU) %38'in altında sürünür; yüksek maliyetli Streaming Multiprocessor (SM) birimleriniz saat döngülerinin %40'ını HBM bellek kuyruklarında stall olarak harcar; scale-out InfiniBand linkleri gradient synchronization bubble'ları ile boğulur ve senkronize checkpoint yazımları birkaç saatte bir tüm cluster'ı beş dakika boyunca dondurur.

Bu darboğaz rastgele bir aksaklık ya da yazılım hatası değildir. Modern bir GPU süper bilgisayarını homojen, tek parça devasa bir işlem havuzu sanmanın fiziksel bedelidir. Statik ve eşzamanlı iletişim döngülerine sahip 2D veya 3D torus ağlarıyla birbirine bağlı Google TPU'larının aksine, modern yapay zekâ süper bilgisayarları derinlemesine hiyerarşik ve üç temel fiziksel sütun tarafından yönetilen sistemlerdir: **Hesaplama (Compute)**, **Ağ (Network)** ve **Depolama (Storage)**. Her bir silikon çipin içinde yüzden fazla bağımsız Streaming Multiprocessor dar bir L2 cache için yarışır. 8 GPU'lu tek bir kasa (chassis) içinde tescilli NVSwitch crossbar fabric'i çip başına çift yönde 900 GB/s gibi akıl almaz bir bant genişliği sunar. Fakat kolektif bir işlem (collective operation) node sınırını aşıp InfiniBand scale-out fabric'e adım attığı anda erişilebilir bant genişliği neredeyse bir büyüklük mertebesi birden düşer; depolama (I/O) alt sisteminiz ise host CPU'ları boğmadan onlarca terabayt checkpoint ve trilyonlarca token stream edemezse, multi-milyon dolarlık tüm hesaplama omurganız kilitlenir.

Bu yazı, modern foundation model'ların ölçekleme mimarisini bu üç kurucu sütun üzerinden adım adım inceliyor. Bağımsız warp scheduler birimlerinin silikon üzerindeki komut dağıtımından NVLink ve rail-optimized InfiniBand fat-tree ağ topolojilerine, Mellanox SHARP gibi ağ içi donanımsal hızlandırmalardan (in-network reduction) GPUDirect Storage ve Young-Daly checkpoint formülüne kadar tüm yığını titiz matematiksel modellemeler ve ampirik benchmark'larla ortaya koyuyor.

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

Dağıtık foundation model eğitimi, uç sınırlarda bir ortak tasarım (co-design) mühendisliğidir. Binlerce hızlandırıcı üzerinde bir büyük dil modeli ölçeklemeye çalışan mühendisler kaçınılmaz olarak üç fiziksel duvara çarpar:

1. **Memory Wall (Hesaplama / Compute):** Son on yılda ham floating-point matrix compute gücü $1.000\times$'den fazla artarken, High-Bandwidth Memory (HBM) bant genişliği $30\times$'dan daha az büyüdü. GPU içindeki aritmetik birimler, operasyonel matris boyutları donanım roofline modelinin kırılma noktasını (roofline knee) aşacak düzeyde veri yeniden kullanımı (data reuse) sunmadığı sürece aç kalır (starve eder).
2. **Fabric Wall (İletişim / Network):** İletişim bant genişliği cluster boyutuyla doğrusal ölçeklenmez. 8 GPU'lu tek bir kasa (chassis) içinde NVLink üzerinden aktivasyon ve gradyan aktarmak, yüzlerce sunucu kabinine yayılan optik InfiniBand veya RoCE v2 fabric hatlarına kıyasla neredeyse bir mertebe daha hızlıdır. Model boyutlarını bu interconnect hiyerarşisine uyduramamak felaket iletişim duraklamalarına (communication stalls) yol açar.
3. **I/O Wall (Depolama / Storage):** Foundation model'lar 15 trilyondan fazla yapılandırılmamış metin ve multimodal token ile eğitilir; 405 milyar parametreli bir modelin parametre durumları ve optimizer buffer'ları tek bir checkpoint snapshot'ında 7,5 Terabayttan fazla yer kaplar. Depolama erişimi geleneksel işletim sistemi kernel page cache mekanizmalarına ve CPU bellek kopyalamalarına (`memcpy`) dayanırsa, host bellek veri yolları doyuma ulaşır (saturate olur) ve her checkpoint aralığında tüm cluster donar.

Donanımın boşta (idle) kalmasını engellemek ve Model FLOPs Utilization (MFU) oranını maksimize etmek için bir yapay zekâ altyapı mühendisi bu üç sütunu eşzamanlı olarak yönetmek zorundadır:

<svg viewBox="0 0 840 8652" role="img" aria-label="Yapay zekâ altyapısını anlatan beş bölümlük bir poster. A bölümü dört basamakta uzaklaşır. 1. basamak, Streaming Multiprocessor: dört özdeş alt bölme; her birinde döngüde 1.024 BF16 FLOP yapan bir Tensor Core, 32 thread&#x27;lik warp&#x27;ları dağıtan bir warp scheduler, 32 FP32 ALU ve 16.384 register; ayrıca 256 KiB SMEM ya da L1. Bir hat aşçısı ve tezgâhı gibidir. 132 SM bir H100 SXM5 GPU eder: 814 milimetrekare kalıp, 80 milyar transistör, 50 MB L2, 6 yuvanın 5&#x27;inde 3,35 TB/s hızlı 80 GB HBM3, 700 W&#x27;ta 990 TFLOP/s BF16; kileri HBM olan bir mutfak gibidir. 8 GPU bir DGX H100 düğümü eder: 4 NVSwitch, GPU başına 900 GB/s NVLink, 3,6 TB/s biseksiyon, PCIe Gen5 üzerinden 64 GB/s ile bağlı 2 TB RAM&#x27;li 2 Xeon işlemci, GPU başına bir 400 Gb/s hesaplama NIC&#x27;i, depolama ve yönetim için 2 çift portlu NIC ve 8 NVMe sürücü. 32 düğüm bir ölçeklenebilir birim, 4 birim 127 düğümlük, 1.016 GPU&#x27;luk bir DGX SuperPOD eder: her düğümün i. NIC&#x27;inin i. leaf&#x27;e gittiği ray uyumlu fat-tree&#x27;de 32 leaf ve 16 spine switch, düğüm başına en az 40 GB/s veren ayrı bir depolama ağı ve in-band ile OOB Ethernet yönetim ağları. B bölümü TP 8, PP 16, DP 128 düzenli Llama 3 405B&#x27;nin tek bir eğitim adımını izler: 1 depolamadan HBM&#x27;e 16 milyon token okunur; 2 ileri geçişte HBM&#x27;den Tensor Core&#x27;lara veri akar; 3 NVLink üzerinde katman başına iki tensör paralel AllReduce yapılır; 4 aktivasyonlar ağ üzerinden sonraki boru hattı aşamasına gönderilir; 5 geri geçiş aynı hatları kullanır; 6 gradyanlar ray switch&#x27;leri üzerinden diğer 127 kopyayla AllReduce edilir; 7 AdamW optimizer adımı HBM&#x27;de yapılır; 8 her 14 dakikada 7,5 TB&#x27;lık kontrol noktası depolamaya yazılır. C bölümü üç duvarı hesaplanmış eğrilerle gösterir. Bellek duvarı: H100&#x27;de Llama 3 70B MLP katmanı için elde edilen hız batch 1&#x27;de 3,3 TFLOP/s&#x27;den batch 128&#x27;de 420&#x27;ye çıkar ve batch 310 civarında 990 tavanına ulaşır; çare büyük batch, kernel füzyonu ve decode&#x27;da düşük bitli ağırlıktır. Ağ duvarı: GPU başına bant genişliği HBM&#x27;de 3,35 TB/s&#x27;den NVLink&#x27;te 450 GB/s&#x27;ye, düğüm dışında 50 GB/s&#x27;ye düşer; tensör paralelliği NVLink&#x27;te 13 GPU&#x27;ya, yani 8&#x27;e sığar, InfiniBand&#x27;de yalnızca 1,45&#x27;e; 16 aşamalı bir boru hattı 16 mikrobatch ile yüzde 48, 128 ile yüzde 10,5 boşta kalır; çare TP&#x27;yi düğüm içinde tutmak, SHARP ve iletişimi hesapla çakıştırmaktır. G/Ç duvarı: kayıt aralığına göre Young-Daly kayıp eğrileri; 25 GB/s&#x27;de en düşük kayıp 42,9 dakikada yüzde 24, 250 GB/s&#x27;de 13,6 dakikada yüzde 7,5&#x27;tir; çare GPUDirect Storage, asenkron yazma ve karekök aralığıdır. D bölümü aynı 140 GB&#x27;ın taşınma süresini verir: HBM&#x27;den 42 milisaniye, NVLink&#x27;ten 0,31 saniye, PCIe&#x27;den 2,2 saniye, GPU başına InfiniBand&#x27;den 2,8 saniye, bütün kümenin paylaştığı eski depolamadan 5,6 saniye; 8 H100&#x27;de decode saniyede yaklaşık 190 tokenle sınırlıdır. E bölümü Meta&#x27;nın Llama 3 405B eğitimini özetler: GPU başına 400 TFLOP/s ve yüzde 41 MFU ile 16.384 H100, 3.072 GPU&#x27;luk pod&#x27;lardan oluşan bir RoCE ağı, 2 TB/s sürekli ve 7 TB/s tepe hızlı 240 PB&#x27;lık Tectonic depolama ve 54 günde 466 kesinti, yani yaklaşık 3 saatte bir arıza. En yavaş halka bütün kümenin hızını belirler. Küme basamağından sonra bir kart SuperPOD&#x27;un yapı taşlarını ve her birinin neden var olduğunu sıralar: 127 DGX H100 sistemi matematiği yapar; 48 QM9700 switch her GPU&#x27;ya bir ray veren hesaplama ağını, 16 tanesi düğüm başına en az 40 GB/s&#x27;lik ayrı depolama ağını kurar; 4 UFM cihazı InfiniBand ağını izler; 8 SN4600C ve 8 SN2201 switch in-band ve OOB yönetimi taşır; 5 yönetim sunucusu kontrol yazılımlarını koşturur; 38 kabin ve 108 PDU yaklaşık 1,3 MW güç dağıtır. Yazılımda Base Command Manager düğümleri kurar, Slurm işleri zamanlar, Magnum IO veriyi NCCL, GPUDirect ve SHARP ile taşır, NGC uyumlu konteynerler sunar, Base Command Platform veri bilimcilere hizmet eder, AI Enterprise destekli araçlar ekler. Düğüm basamağı NVIDIA&#x27;nın DGX H100 sistem topolojisini yeniden çizer: UPI ile bağlı CPU 0 ve CPU 1; CPU 0 GPU 0–3 ile NIC 0–3&#x27;ü, CPU 1 GPU 4–7 ile NIC 4–7&#x27;yi besler ve her GPU&#x27;nun kendi ağ kartı vardır; üstteki iki PCIe switch çift portlu depolama kartlarını ve 4&#x27;er NVMe diski taşır; CPU 0&#x27;da iki M.2 işletim sistemi diski, CPU 1&#x27;de bir 100 GbE kartı bulunur; her GPU 18 NVLink&#x27;ini 5, 4, 4 ve 5 olarak dört NVSwitch&#x27;e gönderir; alttaki PCIe switch NVSwitch&#x27;leri yönetim için CPU 0&#x27;a bağlar. Düğüm şemasındaki altı numaralı daire, altındaki altı soruyla eşleşir: kim kimi besler (iki NUMA yarısı), GPU ağa nasıl çıkar (GPUDirect RDMA ile kendi NIC&#x27;i), GPU&#x27;lar nasıl konuşur (dört NVSwitch&#x27;e 18 NVLink), depolama trafiği nereden geçer (üstteki PCIe switch&#x27;ler), iki yarıyı ne birleştirir (CPU iletişimi, UPI) ve NVSwitch&#x27;leri kim yönetir (CPU 0&#x27;dan PCIe ile Fabric Manager). Küme basamağı H100 SXM5, DGX H100, 32 DGX&#x27;lik ölçeklenebilir birim ve 4 birimli DGX SuperPOD hiyerarşisiyle açılır. Yakından inceleme kartı portları sayar: 64 portlu her leaf 32 portunu düğümlere, 32 portunu spine&#x27;lara verir, toplam 1.024 leaf–spine kablosu vardır ve ağ engelsizdir; aynı raydaki trafik tek leaf&#x27;ten geçer, raylar arası trafik ise NCCL&#x27;in PXN&#x27;i önce NVLink&#x27;e taşımadıkça spine&#x27;dan geçer; bütün GPU&#x27;lar 50,8 TB/s gönderebilir, spine kapasitesi 51,2 TB/s&#x27;dir; 16 birimden itibaren bir core katmanı eklenir ve 2.048 DGX 512 leaf, 512 spine ve 256 core switch kullanır. Bir not SuperPOD&#x27;un neden 128 değil 127 DGX olduğunu açıklar: bir çift UFM cihazı bir DGX yuvasını alır; daha küçük kurulumlar tek birimden başlar, 31 ile 127 DGX arasındadır. Küme çizimi NVIDIA&#x27;nın hesaplama ağı şemasını izler: üstte Spine-01&#x27;den Spine-16&#x27;ya; Leaf-01-01–08 ve H100-001–032 düğümleriyle SU 1; Leaf-02-01–08 ve H100-033–064 düğümleriyle SU 2; aynı düzeni tekrarlayan ve çizimde atlanan SU 3; Leaf-04-01–08, H100-097–127 düğümleri ve C-UFM 1 ile 2 UFM çiftiyle SU 4; spine–leaf hatları 2 × 400 Gb/s, DGX–leaf hatları 400 Gb/s, UFM–leaf hatları 200 Gb/s&#x27;dir. Hiyerarşinin altındaki &#x27;ne neyi yapar&#x27; tablosu her basamağı açıklar: H100 matematiği yapar ve modelin kendine düşen payını HBM&#x27;de tutar; DGX 8 GPU&#x27;yu tensör paralelliğinin yaşadığı tek bir NVLink alanında birleştirir, işlemcileri işletim sistemini ve veri yükleyiciyi koşturur; SU, aynı numaralı GPU&#x27;ların tek sıçramada buluştuğu 32 DGX ve 8 leaf switch&#x27;lik tekrar eden birimdir; SuperPOD SU&#x27;ları spine ile birleştirir, depolama, yönetim ve yazılımı ekler ve veri paralelliğini bütün kümeye yayar. Bir tanım kutusu leaf ve spine&#x27;ı açıklar: leaf, sunucuların takıldığı alt katman switch&#x27;idir, SU başına 8 tane, her ray için bir tane, 32 portu aşağı 32 portu yukarı; spine, leaf&#x27;leri birbirine bağlayan ve hiç sunucu taşımayan üst katman switch&#x27;idir, toplam 16 tane, her biri 32 leaf&#x27;in hepsine bağlıdır; her leaf her spine&#x27;a ulaştığı için herhangi iki sunucu en fazla leaf, spine, leaf yoluyla buluşur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
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
<rect x="16" y="269" width="808" height="5594" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="293" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="297.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">A</text>
<text x="56" y="298" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Yakınlaştır: bir SM&#x27;den bütün kümeye</text>
<text x="28" y="319" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her basamak öncekinin ×N kopyasıdır; koyu vurgu bir önceki basamağı gösterir.</text>
<rect x="28" y="331" width="784" height="260" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="355" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">1 · SM: GPU&#x27;nun temel işçisi</text>
<rect x="40" y="367" width="228" height="152" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="48" y="383" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">SM</text>
<rect x="48" y="391" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="56" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 0</text>
<rect x="56.0" y="413.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="100.0" y="426.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="156" y="391" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="164" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 1</text>
<rect x="164.0" y="413.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="208.0" y="426.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48" y="441" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="56" y="456" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 2</text>
<rect x="56.0" y="463.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="100.0" y="476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="156" y="441" width="104" height="46" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="164" y="456" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme 3</text>
<rect x="164.0" y="463.0" width="88.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="208.0" y="476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48.0" y="493.0" width="212.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.0" y="507.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM / L1 · 256 KiB</text>
<rect x="288" y="367" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="389" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">4</text>
<text x="298" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">alt bölme</text>
<rect x="416" y="367" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="389" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1.024</text>
<text x="426" y="406" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/döngü (TC)</text>
<rect x="288" y="423" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="445" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">256 KiB</text>
<text x="298" y="462" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">register / SM</text>
<rect x="416" y="423" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="445" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">64</text>
<text x="426" y="462" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">warp aynı anda</text>
<text x="288" y="493" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her alt bölme: 1 Tensor Core,</text>
<text x="288" y="509" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1 warp scheduler, 32 FP32 ALU,</text>
<text x="288" y="525" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16.384 register · 256 KiB SMEM/L1</text>
<text x="288" y="541" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">warp = birlikte koşan 32 thread</text>
<text x="552" y="381" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="552" y="399" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bir warp bellekten veri beklerken</text>
<text x="552" y="416" text-anchor="start" style="fill:var(--c-text);font-size:12px">scheduler hemen başka bir warp&#x27;a geçer.</text>
<text x="552" y="433" text-anchor="start" style="fill:var(--c-text);font-size:12px">SM&#x27;ler bekleme süresini böyle gizler;</text>
<text x="552" y="450" text-anchor="start" style="fill:var(--c-text);font-size:12px">ama bütün warp&#x27;lar bekliyorsa gizleyecek</text>
<text x="552" y="467" text-anchor="start" style="fill:var(--c-text);font-size:12px">iş kalmaz.</text>
<line x1="540.0" y1="367.0" x2="540.0" y2="519.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="559.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="574" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Mutfak benzetmesi: SM bir hat aşçısıdır; register ve SMEM onun tezgâhıdır.</text>
<line x1="420.0" y1="594.0" x2="420.0" y2="616.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="609" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 132 SM = 1 GPU</text>
<rect x="28" y="619" width="784" height="276" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="643" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">2 · GPU: H100 SXM5 çipi</text>
<rect x="40" y="655" width="228" height="180" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="46.0" y="663.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="692.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="46.0" y="719.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="748.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="46.0" y="775.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="60.0" y="804.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="663.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.0" y="692.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="719.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="248.0" y="748.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<rect x="234.0" y="775.0" width="28.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.04;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="80" y="661" width="148" height="168" rx="4" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="88" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.95"/>
<rect x="100" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="667" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="678" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="689" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="700" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="711" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="722" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="757" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="768" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="779" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="790" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="801" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="100" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="112" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="124" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="136" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="148" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="160" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="172" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="184" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="196" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="208" y="812" width="9" height="8" rx="1.5" style="fill:var(--c-accent);fill-opacity:.28"/>
<rect x="88.0" y="735.0" width="129.0" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152.5" y="748.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2 · 50 MB</text>
<rect x="288" y="655" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="677" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">132</text>
<text x="298" y="694" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SM</text>
<rect x="416" y="655" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="677" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">990</text>
<text x="426" y="694" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (BF16)</text>
<rect x="288" y="711" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="733" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">80 GB</text>
<text x="298" y="750" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM3 bellek</text>
<rect x="416" y="711" width="120" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="426" y="733" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">3,35 TB/s</text>
<text x="426" y="750" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM hızı</text>
<text x="288" y="781" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">814 mm² kalıp · 80 milyar transistör</text>
<text x="288" y="797" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">6 HBM yuvası, 5&#x27;i etkin · 700 W</text>
<text x="288" y="813" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ortak L2: 50 MB (~5,5 TB/s)</text>
<text x="288" y="829" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FP8: 1.980 TFLOP/s (yoğun)</text>
<text x="288" y="845" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(B200: 148 SM · 192 GB · 8 TB/s)</text>
<text x="552" y="669" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="552" y="687" text-anchor="start" style="fill:var(--c-text);font-size:12px">Çip saniyede 990 trilyon işlem yapabilir</text>
<text x="552" y="704" text-anchor="start" style="fill:var(--c-text);font-size:12px">ama bellekten saniyede yalnızca 3,35 TB</text>
<text x="552" y="721" text-anchor="start" style="fill:var(--c-text);font-size:12px">okuyabilir. Okunan her bayt için ~295</text>
<text x="552" y="738" text-anchor="start" style="fill:var(--c-text);font-size:12px">işlem yapılmazsa Tensor Core&#x27;lar boşta</text>
<text x="552" y="755" text-anchor="start" style="fill:var(--c-text);font-size:12px">kalır.</text>
<line x1="540.0" y1="655.0" x2="540.0" y2="835.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="863.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="878" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">132 hat aşçısı tek mutfakta çalışır; HBM mutfağın kileri, L2 ortak raftır.</text>
<line x1="420.0" y1="898.0" x2="420.0" y2="920.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="913" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 8 GPU = 1 düğüm</text>
<rect x="28" y="923" width="784" height="1609" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="947" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">3 · Düğüm: DGX H100 kasasının içi</text>
<rect x="36" y="959" width="488" height="418" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="44.0" y="971.0" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="76.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="76.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 veri</text>
<rect x="130.0" y="971.0" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="158.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="158.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="206.0" y="971.0" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="239.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çift port</text>
<rect x="288.0" y="971.0" width="66.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="321.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">CX-7</text>
<text x="321.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çift port</text>
<rect x="374.0" y="971.0" width="56.0" height="36" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="402.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">PCIe</text>
<text x="402.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">switch</text>
<rect x="452.0" y="971.0" width="64.0" height="36" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="484.0" y="985" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4 × NVMe</text>
<text x="484.0" y="999" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">U.2 veri</text>
<line x1="109.0" y1="989.0" x2="128.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="187.0" y1="989.0" x2="204.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="356.0" y1="989.0" x2="372.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="432.0" y1="989.0" x2="450.0" y2="989.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="158.0" y1="1009.0" x2="158.0" y2="1025.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="402.0" y1="1009.0" x2="402.0" y2="1025.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="120.0" y="1027.0" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158" y="1047" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="158" y="1063" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="364.0" y="1027.0" width="76.0" height="48" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="402" y="1047" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="402" y="1063" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">Xeon 8480C</text>
<rect x="210.0" y="1029.0" width="62.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="241.0" y="1043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">2 × M.2 OS</text>
<line x1="198.0" y1="1039.0" x2="208.0" y2="1039.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="288.0" y="1029.0" width="62.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="319.0" y="1043.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<line x1="352.0" y1="1039.0" x2="362.0" y2="1039.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="199.0" y1="1061.0" x2="361.0" y2="1061.0" marker-start="url(#bp-acc)" marker-end="url(#bp-acc)" style="stroke:var(--c-accent);stroke-width:2"/>
<path d="M196 1071 H280 V1196 A 8 18 0 0 1 280 1232 V1293" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="40" y="1099" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="286" y="1099" width="234" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<path d="M132 1075 V1083 H69 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M148 1075 V1090 H127 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M164 1075 V1090 H185 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M180 1075 V1083 H243 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M376 1075 V1083 H317 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M392 1075 V1090 H375 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M408 1075 V1090 H433 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<path d="M424 1075 V1083 H491 V1105" marker-end="url(#bp-mut)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="46.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="69.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="69.0" y1="1133.0" x2="69.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="44.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.34;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="69.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="104.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="127.0" y1="1133.0" x2="127.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="102.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="127.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="162.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="185.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="185.0" y1="1133.0" x2="185.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="160.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="185.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="220.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="243.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="243.0" y1="1133.0" x2="243.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="218.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="243.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="294.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="317.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="317.0" y1="1133.0" x2="317.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="292.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<rect x="352.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="375.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="375.0" y1="1133.0" x2="375.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="350.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="375.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<rect x="410.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="433.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="433.0" y1="1133.0" x2="433.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="408.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="433.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<rect x="468.0" y="1107.0" width="46.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="491.0" y="1123.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="491.0" y1="1133.0" x2="491.0" y2="1147.0" marker-start="url(#bp-mut)" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="466.0" y="1149.0" width="50.0" height="32" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="491.0" y="1169.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<!-- GPU 0 (center 69): 4 NVLink drop lines -->
<line x1="60" y1="1181" x2="60" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="66" y1="1181" x2="66" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="72" y1="1181" x2="72" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="78" y1="1181" x2="78" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 1 (center 127): 4 NVLink drop lines -->
<line x1="118" y1="1181" x2="118" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="124" y1="1181" x2="124" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="130" y1="1181" x2="130" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="136" y1="1181" x2="136" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 2 (center 185): 4 NVLink drop lines -->
<line x1="176" y1="1181" x2="176" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="182" y1="1181" x2="182" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="188" y1="1181" x2="188" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="194" y1="1181" x2="194" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 3 (center 243): 4 NVLink drop lines -->
<line x1="234" y1="1181" x2="234" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="240" y1="1181" x2="240" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="246" y1="1181" x2="246" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="252" y1="1181" x2="252" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 4 (center 317): 4 NVLink drop lines -->
<line x1="308" y1="1181" x2="308" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="314" y1="1181" x2="314" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="320" y1="1181" x2="320" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="326" y1="1181" x2="326" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 5 (center 375): 4 NVLink drop lines -->
<line x1="366" y1="1181" x2="366" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="372" y1="1181" x2="372" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="378" y1="1181" x2="378" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="384" y1="1181" x2="384" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 6 (center 433): 4 NVLink drop lines -->
<line x1="424" y1="1181" x2="424" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="430" y1="1181" x2="430" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="436" y1="1181" x2="436" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="442" y1="1181" x2="442" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- GPU 7 (center 491): 4 NVLink drop lines -->
<line x1="482" y1="1181" x2="482" y2="1202" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="488" y1="1181" x2="488" y2="1210" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="494" y1="1181" x2="494" y2="1218" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="500" y1="1181" x2="500" y2="1226" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- 4 Crossbar Horizontal Rails spanning across all 8 GPUs -->
<line x1="56" y1="1202" x2="486" y2="1202" style="stroke:#e11d48;stroke-width:1.8"/>
<line x1="62" y1="1210" x2="492" y2="1210" style="stroke:#0284c7;stroke-width:1.8"/>
<line x1="68" y1="1218" x2="498" y2="1218" style="stroke:#ea580c;stroke-width:1.8"/>
<line x1="74" y1="1226" x2="504" y2="1226" style="stroke:#9333ea;stroke-width:1.8"/>
<!-- Feed bundles into 4 NVSwitches from respective rails -->
<!-- Rail 0 (Pink) -> NVSwitch 0 (5 links) -->
<line x1="88" y1="1202" x2="88" y2="1246" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="92" y1="1202" x2="92" y2="1246" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="96" y1="1202" x2="96" y2="1246" style="stroke:#e11d48;stroke-width:1.4"/>
<line x1="100" y1="1202" x2="100" y2="1246" style="stroke:#e11d48;stroke-width:1.2"/>
<line x1="104" y1="1202" x2="104" y2="1246" style="stroke:#e11d48;stroke-width:1.2"/>
<!-- Rail 1 (Blue) -> NVSwitch 1 (4 links) -->
<line x1="194" y1="1210" x2="194" y2="1246" style="stroke:#0284c7;stroke-width:1.2"/>
<line x1="198" y1="1210" x2="198" y2="1246" style="stroke:#0284c7;stroke-width:1.3"/>
<line x1="202" y1="1210" x2="202" y2="1246" style="stroke:#0284c7;stroke-width:1.3"/>
<line x1="206" y1="1210" x2="206" y2="1246" style="stroke:#0284c7;stroke-width:1.2"/>
<!-- Rail 2 (Orange) -> NVSwitch 2 (4 links) -->
<line x1="354" y1="1218" x2="354" y2="1246" style="stroke:#ea580c;stroke-width:1.2"/>
<line x1="358" y1="1218" x2="358" y2="1246" style="stroke:#ea580c;stroke-width:1.3"/>
<line x1="362" y1="1218" x2="362" y2="1246" style="stroke:#ea580c;stroke-width:1.3"/>
<line x1="366" y1="1218" x2="366" y2="1246" style="stroke:#ea580c;stroke-width:1.2"/>
<!-- Rail 3 (Purple) -> NVSwitch 3 (5 links) -->
<line x1="456" y1="1226" x2="456" y2="1246" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="460" y1="1226" x2="460" y2="1246" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="464" y1="1226" x2="464" y2="1246" style="stroke:#9333ea;stroke-width:1.4"/>
<line x1="468" y1="1226" x2="468" y2="1246" style="stroke:#9333ea;stroke-width:1.2"/>
<line x1="472" y1="1226" x2="472" y2="1246" style="stroke:#9333ea;stroke-width:1.2"/>
<!-- 4 NVSwitch Chips (Green, matching NVIDIA official diagram) -->
<rect x="52.0" y="1246.0" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="96.0" y="1263.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 0</text>
<rect x="156.0" y="1246.0" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="200.0" y="1263.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 1</text>
<rect x="316.0" y="1246.0" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="360.0" y="1263.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 2</text>
<rect x="420.0" y="1246.0" width="88.0" height="26" rx="5" style="fill:#10b981;fill-opacity:.20;stroke:#10b981;stroke-width:1.3"/>
<text x="464.0" y="1263.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono);font-weight:700">NVSwitch 3</text>
<!-- Bottom PCIe Switch (Purple, connecting CPU 0 to NVSwitches) -->
<rect x="250.0" y="1296.0" width="60.0" height="26" rx="5" style="fill:#8b5cf6;fill-opacity:.18;stroke:#8b5cf6;stroke-width:1.3"/>
<text x="280" y="1308" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">PCIe</text>
<text x="280" y="1319" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">switch</text>
<!-- PCIe Management lines from PCIe Switch UP to NVSwitches (Fabric Manager control plane) -->
<path d="M250 1318 H96 V1274" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M250 1307 H200 V1274" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M310 1307 H360 V1274" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<path d="M310 1318 H464 V1274" marker-end="url(#bp-mut)" style="fill:none;stroke:#8b5cf6;stroke-width:1.3"/>
<!-- Explanatory note -->
<text x="280" y="1335" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-style:italic">PCIe Kontrol Yolu: CPU 0 → PCIe switch → 4 NVSwitch (Fabric Manager kontrol düzlemi)</text>
<!-- Component Legend inside Node Box matching NVIDIA official diagram -->
<!-- Legend Row 1: Chips & Modules -->
<rect x="42" y="1346" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1"/>
<text x="56" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">ConnectX-7</text>
<rect x="120" y="1346" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.3;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="134" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">CX-7 Modül</text>
<rect x="200" y="1346" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.4;stroke:var(--c-warn);stroke-width:1"/>
<text x="214" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVMe</text>
<rect x="256" y="1346" width="10" height="10" rx="2" style="fill:#8b5cf6;fill-opacity:.3;stroke:#8b5cf6;stroke-width:1"/>
<text x="270" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">PCIe Switch</text>
<rect x="336" y="1346" width="10" height="10" rx="2" style="fill:#10b981;fill-opacity:.3;stroke:#10b981;stroke-width:1"/>
<text x="350" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVSwitch</text>
<rect x="406" y="1346" width="10" height="10" rx="2" style="fill:#ec4899;fill-opacity:.3;stroke:#ec4899;stroke-width:1"/>
<text x="420" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">100 GbE</text>
<!-- Legend Row 2: Interconnects & Fabrics -->
<line x1="50" y1="1368" x2="72" y2="1368" style="stroke:#8b5cf6;stroke-width:1.5"/>
<text x="76" y="1371" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">PCIe Gen5</text>
<line x1="140" y1="1366" x2="162" y2="1366" style="stroke:#e11d48;stroke-width:1.5"/>
<line x1="140" y1="1368" x2="162" y2="1368" style="stroke:#0284c7;stroke-width:1.5"/>
<line x1="140" y1="1370" x2="162" y2="1370" style="stroke:#ea580c;stroke-width:1.5"/>
<line x1="140" y1="1372" x2="162" y2="1372" style="stroke:#9333ea;stroke-width:1.5"/>
<text x="166" y="1371" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">NVLink (4 Ray)</text>
<line x1="250" y1="1368" x2="272" y2="1368" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="276" y="1371" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">CPU İletişimi (UPI)</text>
<!-- Question Callout Circles (Topoloji Soruları) -->
<circle cx="102" cy="1051" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="102" y="1055" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<circle cx="52" cy="1087" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="52" y="1091" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="98" cy="1211" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="98" y="1215" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="239" cy="1018" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="239" y="1022" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="280" cy="1061" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="1065" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<circle cx="280" cy="1278" r="8" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="280" y="1282" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="540" y="959" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="981" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">8</text>
<text x="550" y="998" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU · 640 GB</text>
<rect x="674" y="959" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="981" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">450 GB/s</text>
<text x="684" y="998" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink, tek yön</text>
<rect x="540" y="1015" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="550" y="1037" text-anchor="start" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">64 GB/s</text>
<text x="550" y="1054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe, CPU yolu</text>
<rect x="674" y="1015" width="126" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="684" y="1037" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">50 GB/s</text>
<text x="684" y="1054" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağ kartı / GPU</text>
<text x="540" y="1085" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 NVSwitch · biseksiyon 3,6 TB/s · 2 × Xeon</text>
<text x="540" y="1100" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8480C (112 çekirdek), 2 TB RAM · 8 × 3,84 TB</text>
<text x="540" y="1115" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVMe veri · güç 10,2 kW</text>
<rect x="540.0" y="1133.0" width="260.0" height="177" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="552" y="1155" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Örnek yol: bir gradyanın düğümdeki</text>
<text x="552" y="1172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">yolculuğu</text>
<text x="552" y="1195" text-anchor="start" style="fill:var(--c-text);font-size:12px">Komşu düğümden gelen bir gradyan NIC</text>
<text x="552" y="1212" text-anchor="start" style="fill:var(--c-text);font-size:12px">5&#x27;e girer ve GPUDirect RDMA ile</text>
<text x="552" y="1229" text-anchor="start" style="fill:var(--c-text);font-size:12px">doğrudan GPU 5&#x27;in belleğine yazılır;</text>
<text x="552" y="1246" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU 1&#x27;e hiç uğramaz. GPU 5 bu gradyanı</text>
<text x="552" y="1263" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU 2 ile paylaşacaksa ağa ya da</text>
<text x="552" y="1280" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU&#x27;lara değil, NVLink üzerinden</text>
<text x="552" y="1297" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch&#x27;lere gider.</text>
<rect x="36" y="1389" width="768" height="517" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="48" y="1411" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Lejant: şemadaki her parça ne?</text>
<text x="48" y="1429" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Adlar NVIDIA&#x27;nın diyagram lejantından, renkler posterin sütun renkleridir. Görevler DGX H100 kullanım kılavuzundan gelir; “olası”</text>
<text x="48" y="1444" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yazanlar çıkarımdır.</text>
<rect x="48.0" y="1462.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CX-7</text>
<text x="126" y="1476" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7” · üstte, 2 adet</text>
<text x="126" y="1492" text-anchor="start" style="fill:var(--c-text);font-size:11px">Çift portlu ağ kartı; kılavuza göre depolama ağı</text>
<text x="126" y="1507" text-anchor="start" style="fill:var(--c-text);font-size:11px">ve in-band yönetim için. Her biri üstteki bir PCIe</text>
<text x="126" y="1522" text-anchor="start" style="fill:var(--c-text);font-size:11px">switch&#x27;e bağlı.</text>
<rect x="48.0" y="1536.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="81.0" y="1550.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0–7</text>
<text x="126" y="1550" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “ConnectX-7 Network Module” · 8 adet</text>
<text x="126" y="1566" text-anchor="start" style="fill:var(--c-text);font-size:11px">Tek portlu 400 Gb/s hesaplama ağı kartları, GPU</text>
<text x="126" y="1581" text-anchor="start" style="fill:var(--c-text);font-size:11px">başına bir tane; kümeye 4 OSFP portundan çıkar.</text>
<text x="126" y="1596" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gri çerçeveler bu kartları taşıyan modüllerdir</text>
<text x="126" y="1611" text-anchor="start" style="fill:var(--c-text);font-size:11px">(olası).</text>
<rect x="48.0" y="1625.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1639.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe</text>
<text x="126" y="1639" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · üstte, 4 + 4</text>
<text x="126" y="1655" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri diskleri: 8 × 3,84 TB U.2 NVMe, RAID 0</text>
<text x="126" y="1670" text-anchor="start" style="fill:var(--c-text);font-size:11px">önbellek; okuma ve kontrol noktası için yerel</text>
<text x="126" y="1685" text-anchor="start" style="fill:var(--c-text);font-size:11px">alan.</text>
<rect x="48.0" y="1699.0" width="66.0" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="81.0" y="1713.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">M.2 OS</text>
<text x="126" y="1713" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVMe” · CPU 0&#x27;ın yanında</text>
<text x="126" y="1729" text-anchor="start" style="fill:var(--c-text);font-size:11px">Büyük olasılıkla 2 × 1,92 TB M.2 işletim sistemi</text>
<text x="126" y="1744" text-anchor="start" style="fill:var(--c-text);font-size:11px">diski (RAID 1); diyagram veri diskleriyle aynı</text>
<text x="126" y="1759" text-anchor="start" style="fill:var(--c-text);font-size:11px">simgeyi kullanır.</text>
<rect x="48.0" y="1773.0" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="1787.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe sw</text>
<text x="126" y="1787" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe Switches” · 3 adet</text>
<text x="126" y="1803" text-anchor="start" style="fill:var(--c-text);font-size:11px">Bir CPU hattını birçok cihaza dağıtır: üstteki</text>
<text x="126" y="1818" text-anchor="start" style="fill:var(--c-text);font-size:11px">ikisi NVMe ve CX-7&#x27;yi taşır, alttaki NVSwitch&#x27;leri</text>
<text x="126" y="1833" text-anchor="start" style="fill:var(--c-text);font-size:11px">CPU 0&#x27;a bağlar.</text>
<rect x="48.0" y="1847.0" width="66.0" height="20" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="81.0" y="1861.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">100 GbE</text>
<text x="126" y="1861" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “100 GbE” · CPU 1&#x27;e bağlı</text>
<text x="126" y="1877" text-anchor="start" style="fill:var(--c-text);font-size:11px">Ek bir Ethernet kartı; görevi diyagramda yazmıyor</text>
<text x="126" y="1892" text-anchor="start" style="fill:var(--c-text);font-size:11px">(olası: servis ya da yönetim ağı).</text>
<rect x="432.0" y="1462.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="465.0" y="1476.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVSwitch</text>
<text x="510" y="1476" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “NVSwitch” · 4 adet</text>
<text x="510" y="1492" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;lar arası NVLink anahtarı; her GPU 18</text>
<text x="510" y="1507" text-anchor="start" style="fill:var(--c-text);font-size:11px">bağlantısını bu dördüne 5 + 4 + 4 + 5 olarak</text>
<text x="510" y="1522" text-anchor="start" style="fill:var(--c-text);font-size:11px">dağıtır.</text>
<rect x="432.0" y="1536.0" width="66.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="1550.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">CPU</text>
<text x="510" y="1550" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU 0 / CPU 1”</text>
<text x="510" y="1566" text-anchor="start" style="fill:var(--c-text);font-size:11px">Intel Xeon 8480C (56&#x27;şar çekirdek); her biri dört</text>
<text x="510" y="1581" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;yu ve onların NIC&#x27;lerini PCIe ile besler.</text>
<line x1="432.0" y1="1605.0" x2="498.0" y2="1605.0" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="510" y="1609" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “CPU communication”</text>
<text x="510" y="1625" text-anchor="start" style="fill:var(--c-text);font-size:11px">İki soketi birbirine bağlayan hat; Intel Xeon&#x27;da</text>
<text x="510" y="1640" text-anchor="start" style="fill:var(--c-text);font-size:11px">bu bağlantının adı UPI&#x27;dir.</text>
<line x1="432.0" y1="1664.0" x2="498.0" y2="1664.0" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<text x="510" y="1668" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA: “PCIe”</text>
<text x="510" y="1684" text-anchor="start" style="fill:var(--c-text);font-size:11px">CPU&#x27;ları NIC&#x27;lere, GPU&#x27;lara, disklere ve</text>
<text x="510" y="1699" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVSwitch&#x27;lere bağlayan genel yol (PCIe Gen5).</text>
<line x1="432.0" y1="1723.0" x2="498.0" y2="1723.0" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="510" y="1727" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NVIDIA lejantında yok: NVLink</text>
<text x="510" y="1743" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;ların altındaki renkli demetler NVLink</text>
<text x="510" y="1758" text-anchor="start" style="fill:var(--c-text);font-size:11px">hatlarıdır: GPU başına 18 bağlantı, toplam</text>
<text x="510" y="1773" text-anchor="start" style="fill:var(--c-text);font-size:11px">900 GB/s çift yön.</text>
<line x1="420.0" y1="1458.0" x2="420.0" y2="1892.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1932" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Topolojiyi okuma: altı soru</text>
<text x="40" y="1951" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Şemadaki numaralı daireleri sırayla izleyin; her biri bir soruyu cevaplar.</text>
<circle cx="49" cy="1977" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="1981" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="1981" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Kim kimi besler?</text>
<text x="66" y="2001" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğüm birbirinin aynası olan iki yarıdan oluşur. CPU 0,</text>
<text x="66" y="2018" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU 0–3&#x27;ü ve NIC 0–3&#x27;ü PCIe ile besler; CPU 1 de GPU</text>
<text x="66" y="2035" text-anchor="start" style="fill:var(--c-text);font-size:12px">4–7&#x27;yi ve NIC 4–7&#x27;yi. Her yarının kendi belleği vardır;</text>
<text x="66" y="2052" text-anchor="start" style="fill:var(--c-text);font-size:12px">buna NUMA alanı denir.</text>
<text x="66" y="2071" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> GPU 5&#x27;e veri hazırlayan süreç CPU 1&#x27;de</text>
<text x="66" y="2088" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">koşmalı ve CPU 1&#x27;in belleğini kullanmalıdır; yoksa her</text>
<text x="66" y="2105" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">batch önce CPU&#x27;lar arasındaki hattı geçer.</text>
<circle cx="49" cy="2131" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2135" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="2135" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">GPU ağa nasıl çıkar?</text>
<text x="66" y="2155" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU&#x27;nun hemen üstünde kendine ait 400 Gb/s&#x27;lik bir</text>
<text x="66" y="2172" text-anchor="start" style="fill:var(--c-text);font-size:12px">NIC durur (GPU i ↔ NIC i). GPUDirect RDMA sayesinde</text>
<text x="66" y="2189" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri GPU belleğinden doğrudan NIC&#x27;e geçer, CPU</text>
<text x="66" y="2206" text-anchor="start" style="fill:var(--c-text);font-size:12px">belleğine hiç uğramaz.</text>
<text x="66" y="2225" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> kasadan 8 × 50 = 400 GB/s çıkar. Kümede NIC</text>
<text x="66" y="2242" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">i, i. leaf switch&#x27;e takılır; farklı düğümlerdeki aynı</text>
<text x="66" y="2259" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">numaralı GPU&#x27;lar bu yüzden tek sıçramada buluşur.</text>
<circle cx="49" cy="2285" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="49" y="2289" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="2289" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">GPU&#x27;lar birbirleriyle nasıl konuşur?</text>
<text x="66" y="2309" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU&#x27;nun 18 NVLink&#x27;i aşağıdaki dört NVSwitch&#x27;e iner:</text>
<text x="66" y="2326" text-anchor="start" style="fill:var(--c-text);font-size:12px">ikisine 5&#x27;er, ikisine 4&#x27;er hat. Herhangi iki GPU tek</text>
<text x="66" y="2343" text-anchor="start" style="fill:var(--c-text);font-size:12px">bir switch sıçramasıyla, her yönde 450 GB/s ile</text>
<text x="66" y="2360" text-anchor="start" style="fill:var(--c-text);font-size:12px">konuşur.</text>
<text x="66" y="2379" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> bu yol ağdan 9 kat geniştir; her katmanda</text>
<text x="66" y="2396" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">konuşan tensör paralelliği bu yüzden bu sekiz GPU&#x27;nun</text>
<text x="66" y="2413" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">içinde tutulur.</text>
<circle cx="441" cy="1977" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="1981" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="458" y="1981" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Depolama trafiği nereden geçer?</text>
<text x="458" y="2001" text-anchor="start" style="fill:var(--c-text);font-size:12px">Üstteki iki PCIe switch, çift portlu CX-7 kartlarını</text>
<text x="458" y="2018" text-anchor="start" style="fill:var(--c-text);font-size:12px">(depolama ağı ve in-band yönetim) ve 4&#x27;er U.2 NVMe veri</text>
<text x="458" y="2035" text-anchor="start" style="fill:var(--c-text);font-size:12px">diskini taşır. CPU 0&#x27;ın yanında ayrıca iki M.2 işletim</text>
<text x="458" y="2052" text-anchor="start" style="fill:var(--c-text);font-size:12px">sistemi diski durur.</text>
<text x="458" y="2071" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> veri okumak ve kontrol noktası yazmak,</text>
<text x="458" y="2088" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">gradyan trafiğinden farklı kartları ve kabloları</text>
<text x="458" y="2105" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">kullanır; büyük bir kayıt AllReduce&#x27;u yavaşlatmaz.</text>
<circle cx="441" cy="2131" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="2135" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="2135" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">İki yarıyı ne birleştirir?</text>
<text x="458" y="2155" text-anchor="start" style="fill:var(--c-text);font-size:12px">İki CPU, NVIDIA&#x27;nın “CPU communication” dediği hatla</text>
<text x="458" y="2172" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Xeon&#x27;da UPI) bağlıdır. Bu hat CPU&#x27;ların kendi trafiği</text>
<text x="458" y="2189" text-anchor="start" style="fill:var(--c-text);font-size:12px">içindir; iki yarıdaki GPU&#x27;lar birbirleriyle CPU&#x27;lar</text>
<text x="458" y="2206" text-anchor="start" style="fill:var(--c-text);font-size:12px">üzerinden değil, NVLink üzerinden konuşur.</text>
<text x="458" y="2225" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> bir CPU öbür yarının belleğine erişebilir ama</text>
<text x="458" y="2242" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">daha yavaş; NUMA sabitlemesi bu cezadan kaçınmak</text>
<text x="458" y="2259" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">içindir.</text>
<circle cx="441" cy="2285" r="9" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.8"/>
<text x="441" y="2289" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="2289" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVSwitch&#x27;leri kim yönetir?</text>
<text x="458" y="2309" text-anchor="start" style="fill:var(--c-text);font-size:12px">CPU 0&#x27;dan inen PCIe hattı, alttaki PCIe switch</text>
<text x="458" y="2326" text-anchor="start" style="fill:var(--c-text);font-size:12px">üzerinden dört NVSwitch&#x27;e ulaşır. Bu bir kontrol</text>
<text x="458" y="2343" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoludur: NVIDIA Fabric Manager switch&#x27;lerin yönlendirme</text>
<text x="458" y="2360" text-anchor="start" style="fill:var(--c-text);font-size:12px">tablolarını buradan kurar; GPU verisi bu hattan akmaz.</text>
<text x="458" y="2379" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> Fabric Manager çalışmazsa GPU&#x27;lar tek tek</text>
<text x="458" y="2396" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sağlıklı görünse bile NVLink ağı kurulmaz ve çok GPU&#x27;lu</text>
<text x="458" y="2413" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">işler başlamaz.</text>
<line x1="420.0" y1="1965.0" x2="420.0" y2="2413.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2447" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="40" y="2465" text-anchor="start" style="fill:var(--c-text);font-size:12px">Kasa içinde GPU&#x27;lar NVLink ile birbirinin belleğine doğrudan erişir. CPU&#x27;ya giden PCIe yolu 7 kat, kasadan çıkan ağ yolu 9</text>
<text x="40" y="2482" text-anchor="start" style="fill:var(--c-text);font-size:12px">kat daha dardır; bu yüzden en yoğun konuşma kasanın içinde tutulur.</text>
<rect x="36.0" y="2500.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="2515" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">8 mutfaklı bir restoran: NVLink mutfakları birbirine bağlayan iç koridordur.</text>
<line x1="420.0" y1="2535.0" x2="420.0" y2="2557.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="2550" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">× 32 düğüm = 1 SU · × 4 SU = SuperPOD</text>
<rect x="28" y="2560" width="784" height="1674" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="2584" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4 · Küme: DGX SuperPOD</text>
<text x="40" y="2606" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Hiyerarşiyi hatırla: parçadan bütüne</text>
<rect x="40.0" y="2616.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="2630" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="126.0" y="2644" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">tek GPU modülü</text>
<line x1="214.0" y1="2636.0" x2="234.0" y2="2636.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="236.0" y="2616.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="322.0" y="2630" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="322.0" y="2644" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPU&#x27;lu sunucu</text>
<line x1="410.0" y1="2636.0" x2="430.0" y2="2636.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="432.0" y="2616.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="518.0" y="2630" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="518.0" y="2644" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 DGX = 256 GPU</text>
<line x1="606.0" y1="2636.0" x2="626.0" y2="2636.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="628.0" y="2616.0" width="172.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="714.0" y="2630" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="714.0" y="2644" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4 SU = 127 DGX</text>
<text x="40" y="2676" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">H100 SXM5 modülleri anakarta değil, 8 GPU&#x27;yu ve 4 NVSwitch&#x27;i taşıyan HGX taban kartına takılır; DGX H100&#x27;ün iki işlemcisi Intel Xeon</text>
<text x="40" y="2691" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">8480C&#x27;dir. SuperPOD 32 DGX&#x27;lik birimlerden (SU) kurulur; referans tasarım 4 SU&#x27;dur.</text>
<text x="40" y="2723" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Leaf ve spine nedir?</text>
<rect x="40" y="2735" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="48.0" y="2743.0" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="162.0" y="2758.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Leaf (yaprak) switch</text>
<text x="52" y="2785" text-anchor="start" style="fill:var(--c-text);font-size:12px">Sunucuların doğrudan takıldığı alt</text>
<text x="52" y="2802" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmandır; ağacın uçları, yani</text>
<text x="52" y="2819" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;lara en yakın switch&#x27;lerdir.</text>
<text x="52" y="2842" text-anchor="start" style="fill:var(--c-text);font-size:12px">SuperPOD&#x27;da: SU başına 8 tane, her</text>
<text x="52" y="2859" text-anchor="start" style="fill:var(--c-text);font-size:12px">ray için bir tane. 64 portunun</text>
<text x="52" y="2876" text-anchor="start" style="fill:var(--c-text);font-size:12px">32&#x27;si DGX&#x27;lere iner, 32&#x27;si</text>
<text x="52" y="2893" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine&#x27;lara çıkar.</text>
<rect x="296" y="2735" width="244" height="167" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="304.0" y="2743.0" width="228.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="418.0" y="2758.3" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">Spine (omurga) switch</text>
<text x="308" y="2785" text-anchor="start" style="fill:var(--c-text);font-size:12px">Leaf&#x27;leri birbirine bağlayan üst</text>
<text x="308" y="2802" text-anchor="start" style="fill:var(--c-text);font-size:12px">katmandır; spine&#x27;a hiçbir sunucu</text>
<text x="308" y="2819" text-anchor="start" style="fill:var(--c-text);font-size:12px">takılmaz.</text>
<text x="308" y="2842" text-anchor="start" style="fill:var(--c-text);font-size:12px">SuperPOD&#x27;da: toplam 16 tane; her</text>
<text x="308" y="2859" text-anchor="start" style="fill:var(--c-text);font-size:12px">biri 32 leaf&#x27;in hepsine ikişer</text>
<text x="308" y="2876" text-anchor="start" style="fill:var(--c-text);font-size:12px">hatla bağlıdır ve 64 portu böyle</text>
<text x="308" y="2893" text-anchor="start" style="fill:var(--c-text);font-size:12px">dolar.</text>
<text x="552" y="2749" text-anchor="start" style="fill:var(--c-text);font-size:12px">İki katmanın kuralı basittir: her leaf</text>
<text x="552" y="2766" text-anchor="start" style="fill:var(--c-text);font-size:12px">her spine&#x27;a bağlanır. Böylece herhangi</text>
<text x="552" y="2783" text-anchor="start" style="fill:var(--c-text);font-size:12px">iki sunucu en fazla leaf → spine → leaf</text>
<text x="552" y="2800" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoluyla buluşur ve küme büyüdükçe yol</text>
<text x="552" y="2817" text-anchor="start" style="fill:var(--c-text);font-size:12px">uzamaz. Sunucu eklemek için leaf, bant</text>
<text x="552" y="2834" text-anchor="start" style="fill:var(--c-text);font-size:12px">genişliği eklemek için spine eklenir. Bu</text>
<text x="552" y="2851" text-anchor="start" style="fill:var(--c-text);font-size:12px">düzen telefon santrallerinden gelir</text>
<text x="552" y="2868" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Clos, 1953); yukarı çıktıkça</text>
<text x="552" y="2885" text-anchor="start" style="fill:var(--c-text);font-size:12px">bağlantıların kalınlaştığı hâline</text>
<text x="552" y="2902" text-anchor="start" style="fill:var(--c-text);font-size:12px">fat-tree topolojisi denir</text>
<text x="552" y="2919" text-anchor="start" style="fill:var(--c-text);font-size:12px">(Leiserson, 1985).</text>
<text x="552" y="2941" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Benzetme: leaf bir mahallenin santralidir,</text>
<text x="552" y="2956" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">spine şehirler arasındaki ana hat.</text>
<text x="40" y="2994" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Ne neyi yapar?</text>
<text x="168" y="3020" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Görevi</text>
<text x="384" y="3020" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">İçinde</text>
<text x="600" y="3020" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">Bağlantı ve paralellik</text>
<line x1="40.0" y1="3029.0" x2="800.0" y2="3029.0" style="stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="3042.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="3058.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100 SXM5</text>
<text x="168" y="3058" text-anchor="start" style="fill:var(--c-text);font-size:12px">modelin matematiğini yapar; 132</text>
<text x="168" y="3075" text-anchor="start" style="fill:var(--c-text);font-size:12px">SM&#x27;deki Tensor Core&#x27;lar matris</text>
<text x="168" y="3092" text-anchor="start" style="fill:var(--c-text);font-size:12px">çarpımlarını yürütür.</text>
<text x="384" y="3058" text-anchor="start" style="fill:var(--c-text);font-size:12px">ağırlıkların, aktivasyonların ve</text>
<text x="384" y="3075" text-anchor="start" style="fill:var(--c-text);font-size:12px">optimizer durumunun bu GPU&#x27;ya</text>
<text x="384" y="3092" text-anchor="start" style="fill:var(--c-text);font-size:12px">düşen payı, 80 GB HBM3&#x27;te.</text>
<text x="600" y="3058" text-anchor="start" style="fill:var(--c-text);font-size:12px">18 NVLink ile kasadaki diğer</text>
<text x="600" y="3075" text-anchor="start" style="fill:var(--c-text);font-size:12px">7 GPU&#x27;ya, PCIe ile CPU&#x27;ya ve</text>
<text x="600" y="3092" text-anchor="start" style="fill:var(--c-text);font-size:12px">kendi NIC&#x27;ine ulaşır; her</text>
<text x="600" y="3109" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU&#x27;nun bir TP, bir PP ve bir DP</text>
<text x="600" y="3126" text-anchor="start" style="fill:var(--c-text);font-size:12px">koordinatı vardır.</text>
<line x1="40.0" y1="3136.0" x2="800.0" y2="3136.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3150.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="96.0" y="3166.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX H100</text>
<text x="168" y="3166" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 GPU&#x27;yu tek bir NVLink alanında</text>
<text x="168" y="3183" text-anchor="start" style="fill:var(--c-text);font-size:12px">birleştirir; iki Xeon işlemci</text>
<text x="168" y="3200" text-anchor="start" style="fill:var(--c-text);font-size:12px">işletim sistemini, veri</text>
<text x="168" y="3217" text-anchor="start" style="fill:var(--c-text);font-size:12px">yükleyiciyi ve GPU&#x27;lara iş</text>
<text x="168" y="3234" text-anchor="start" style="fill:var(--c-text);font-size:12px">gönderen süreçleri koşturur.</text>
<text x="384" y="3166" text-anchor="start" style="fill:var(--c-text);font-size:12px">HGX taban kartında 8 GPU ve 4</text>
<text x="384" y="3183" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVSwitch; ayrıca 2 TB RAM,</text>
<text x="384" y="3200" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 × 3,84 TB NVMe önbellek ve 8 +</text>
<text x="384" y="3217" text-anchor="start" style="fill:var(--c-text);font-size:12px">2 ağ kartı.</text>
<text x="600" y="3166" text-anchor="start" style="fill:var(--c-text);font-size:12px">tensör paralelliği (TP 8) bu</text>
<text x="600" y="3183" text-anchor="start" style="fill:var(--c-text);font-size:12px">kasanın içinde kalır; kasa ağa</text>
<text x="600" y="3200" text-anchor="start" style="fill:var(--c-text);font-size:12px">her GPU için bir ray olmak üzere</text>
<text x="600" y="3217" text-anchor="start" style="fill:var(--c-text);font-size:12px">8 rayla çıkar.</text>
<line x1="40.0" y1="3244.0" x2="800.0" y2="3244.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3258.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="3274.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">SU</text>
<text x="168" y="3274" text-anchor="start" style="fill:var(--c-text);font-size:12px">kurulumun ve ağın tekrar eden</text>
<text x="168" y="3291" text-anchor="start" style="fill:var(--c-text);font-size:12px">birimidir: 32 DGX (8 kabin × 4)</text>
<text x="168" y="3308" text-anchor="start" style="fill:var(--c-text);font-size:12px">ve onları bağlayan 8 leaf</text>
<text x="168" y="3325" text-anchor="start" style="fill:var(--c-text);font-size:12px">switch.</text>
<text x="384" y="3274" text-anchor="start" style="fill:var(--c-text);font-size:12px">256 GPU; her ray tek bir</text>
<text x="384" y="3291" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf&#x27;tir, yani Leaf-xx-0i bütün</text>
<text x="384" y="3308" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümlerin NIC i&#x27;sini toplar.</text>
<text x="600" y="3274" text-anchor="start" style="fill:var(--c-text);font-size:12px">aynı numaralı GPU&#x27;lar tek</text>
<text x="600" y="3291" text-anchor="start" style="fill:var(--c-text);font-size:12px">sıçramada buluşur; boru hattı ve</text>
<text x="600" y="3308" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri paralelliğinin düğümler</text>
<text x="600" y="3325" text-anchor="start" style="fill:var(--c-text);font-size:12px">arası trafiği önce burada, ray</text>
<text x="600" y="3342" text-anchor="start" style="fill:var(--c-text);font-size:12px">ray akar.</text>
<line x1="40.0" y1="3352.0" x2="800.0" y2="3352.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="40.0" y="3366.0" width="112.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="96.0" y="3382.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">DGX SuperPOD</text>
<text x="168" y="3382" text-anchor="start" style="fill:var(--c-text);font-size:12px">SU&#x27;ları spine katmanıyla tek bir</text>
<text x="168" y="3399" text-anchor="start" style="fill:var(--c-text);font-size:12px">engelsiz ağda birleştirir;</text>
<text x="168" y="3416" text-anchor="start" style="fill:var(--c-text);font-size:12px">yanına ayrı depolama ve yönetim</text>
<text x="168" y="3433" text-anchor="start" style="fill:var(--c-text);font-size:12px">ağlarını, UFM&#x27;i ve Base Command</text>
<text x="168" y="3450" text-anchor="start" style="fill:var(--c-text);font-size:12px">ile Slurm gibi yazılımları</text>
<text x="168" y="3467" text-anchor="start" style="fill:var(--c-text);font-size:12px">ekler.</text>
<text x="384" y="3382" text-anchor="start" style="fill:var(--c-text);font-size:12px">127 DGX = 1.016 GPU; tek bir</text>
<text x="384" y="3399" text-anchor="start" style="fill:var(--c-text);font-size:12px">eğitim işi bütün kümeye</text>
<text x="384" y="3416" text-anchor="start" style="fill:var(--c-text);font-size:12px">yayılabilir.</text>
<text x="600" y="3382" text-anchor="start" style="fill:var(--c-text);font-size:12px">veri paralelliği bütün kümeye</text>
<text x="600" y="3399" text-anchor="start" style="fill:var(--c-text);font-size:12px">yayılır, raylar arası trafik bir</text>
<text x="600" y="3416" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine üzerinden döner; Slurm işi</text>
<text x="600" y="3433" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümlere yerleştirir ve arızalı</text>
<text x="600" y="3450" text-anchor="start" style="fill:var(--c-text);font-size:12px">düğümü devreden çıkarır.</text>
<rect x="40.0" y="3485.0" width="760.0" height="84" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="52" y="3505" text-anchor="start" style="fill:var(--c-text);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Neden 128 değil 127?</tspan> Kâğıt üstünde 4 SU × 32 = 128 DGX eder. NVIDIA&#x27;nın yerleşim düzeninde ise bir çift UFM cihazı bir</text>
<text x="52" y="3522" text-anchor="start" style="fill:var(--c-text);font-size:12px">DGX&#x27;in yerini alır (NVIDIA&#x27;nın şemasında SU 4&#x27;teki C-UFM 1 ve 2); tam bir SuperPOD bu yüzden en fazla 127 DGX, yani</text>
<text x="52" y="3539" text-anchor="start" style="fill:var(--c-text);font-size:12px">1.016 GPU&#x27;dur. Kablo tablosu da bunu gösterir: 127 × 8 + 4 UFM = 1.020 kablo. Daha küçük kurulumlar tek SU&#x27;dan başlar</text>
<text x="52" y="3556" text-anchor="start" style="fill:var(--c-text);font-size:12px">(31–127 DGX).</text>
<rect x="36" y="3581" width="768" height="392" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="214.0" y="3595.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="252.0" y="3610.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-01</text>
<rect x="318.0" y="3595.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="356.0" y="3610.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-02</text>
<rect x="530.0" y="3595.0" width="76.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.20;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="568.0" y="3610.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Spine-16</text>
<text x="462" y="3610" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700">⋯</text>
<text x="620" y="3610" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16 × QM9700</text>
<rect x="46" y="3651" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="52.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="83.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-01</text>
<rect x="120.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="151.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-02</text>
<rect x="206.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-01-08</text>
<text x="194" y="3678" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="89.0" y1="3737.0" x2="83.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="3737.0" x2="151.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="89.0" y1="3737.0" x2="237.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="54" y="3737" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="89" y="3753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-001</text>
<text x="89" y="3768" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="175.0" y1="3737.0" x2="83.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="3737.0" x2="151.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="175.0" y1="3737.0" x2="237.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="140" y="3737" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="175" y="3753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-032</text>
<text x="175" y="3768" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="132" y="3760" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="56" y="3815" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 1</text>
<text x="98" y="3815" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-001 … 032 · 8 leaf</text>
<rect x="290" y="3651" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="296.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="327.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-01</text>
<rect x="364.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="395.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-02</text>
<rect x="450.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="481.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-02-08</text>
<text x="438" y="3678" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="333.0" y1="3737.0" x2="327.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="3737.0" x2="395.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="333.0" y1="3737.0" x2="481.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="298" y="3737" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="333" y="3753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-033</text>
<text x="333" y="3768" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="419.0" y1="3737.0" x2="327.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="3737.0" x2="395.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="419.0" y1="3737.0" x2="481.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="384" y="3737" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="419" y="3753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-064</text>
<text x="419" y="3768" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="376" y="3760" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="300" y="3815" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 2</text>
<text x="342" y="3815" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-033 … 064 · 8 leaf</text>
<rect x="574" y="3651" width="226" height="176" rx="8" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="580.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="611.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-01</text>
<rect x="648.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="679.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-02</text>
<rect x="734.0" y="3663.0" width="62.0" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="765.0" y="3677.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Leaf-04-08</text>
<text x="722" y="3678" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<line x1="617.0" y1="3737.0" x2="611.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="3737.0" x2="679.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="617.0" y1="3737.0" x2="765.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="582" y="3737" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="617" y="3753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-097</text>
<text x="617" y="3768" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<line x1="703.0" y1="3737.0" x2="611.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="3737.0" x2="679.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<line x1="703.0" y1="3737.0" x2="765.0" y2="3683.0" style="stroke:var(--c-warn);stroke-width:1.3;stroke-opacity:.75"/>
<rect x="668" y="3737" width="70" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="703" y="3753" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">H100-127</text>
<text x="703" y="3768" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 × H100</text>
<text x="660" y="3760" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-weight:700">⋯</text>
<text x="584" y="3815" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">SU 4</text>
<text x="626" y="3815" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100-097 … 127 + C-UFM</text>
<text x="545" y="3743" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px;font-weight:700">⋯</text>
<text x="545" y="3763" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">SU 3</text>
<path d="M746 3752 H743 V3683" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="3743.0" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="3756.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 1</text>
<path d="M746 3776 H740 V3683" style="fill:none;stroke:var(--c-accent);stroke-width:1.4"/>
<rect x="746.0" y="3767.0" width="52.0" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="772.0" y="3780.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">C-UFM 2</text>
<line x1="252.0" y1="3617.0" x2="83.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="151.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="237.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="327.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="395.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="481.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="611.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="679.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="252.0" y1="3617.0" x2="765.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="83.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="151.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="237.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="327.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="395.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="481.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="611.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="679.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="356.0" y1="3617.0" x2="765.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="83.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="151.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="237.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="327.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="395.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="481.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="611.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="679.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="568.0" y1="3617.0" x2="765.0" y2="3663.0" style="stroke:var(--c-danger);stroke-width:1.4;stroke-opacity:.55"/>
<line x1="48.0" y1="3849.0" x2="70.0" y2="3849.0" style="stroke:var(--c-danger);stroke-width:2.4"/>
<text x="76" y="3853" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine–leaf: 2 × 400 Gb/s</text>
<line x1="300.0" y1="3849.0" x2="322.0" y2="3849.0" style="stroke:var(--c-warn);stroke-width:2.4"/>
<text x="328" y="3853" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX–leaf: 400 Gb/s</text>
<line x1="552.0" y1="3849.0" x2="574.0" y2="3849.0" style="stroke:var(--c-accent);stroke-width:2.4"/>
<text x="580" y="3853" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">UFM–leaf: 200 Gb/s</text>
<text x="48" y="3877" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Hat renkleri NVIDIA&#x27;nın şemasındaki gibidir ve burada kablo türünü gösterir (posterin sütun renkleri değil). Adlar da oradan:</text>
<text x="48" y="3892" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Leaf-SU-ray; Leaf-04-08, SU 4&#x27;ün 8. rayıdır ve her DGX&#x27;in NIC 0–7&#x27;si Leaf-xx-01 ile 08 arasına gider. SU 3 (H100-065 … 096) aynı</text>
<text x="48" y="3907" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">düzeni tekrarlar.</text>
<rect x="44.0" y="3921.0" width="368.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="54" y="3937" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">depolama ağı (ayrı InfiniBand)</text>
<text x="54" y="3953" text-anchor="start" style="fill:var(--c-text);font-size:11px">16 switch · düğüm başına ≥ 40 GB/s</text>
<rect x="428.0" y="3921.0" width="368.0" height="40" rx="5" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="438.0" y="3937" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">yönetim ağları (Ethernet)</text>
<text x="438.0" y="3953" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">in-band 100 GbE · OOB 1 GbE</text>
<rect x="40.0" y="3985" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="4007" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">127</text>
<text x="50.0" y="4024" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DGX düğümü</text>
<rect x="234.0" y="3985" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="244.0" y="4007" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">1.016</text>
<text x="244.0" y="4024" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU</text>
<rect x="428.0" y="3985" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="438.0" y="4007" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">32 + 16</text>
<text x="438.0" y="4024" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">leaf + spine</text>
<rect x="622.0" y="3985" width="178.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="632.0" y="4007" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">4</text>
<text x="632.0" y="4024" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ayrı ağ</text>
<text x="40" y="4055" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">127 DGX (UFM çifti SU 4&#x27;te)</text>
<text x="420" y="4055" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ray içi: SU&#x27;daki 31 düğüme tek sıçrama</text>
<text x="40" y="4071" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesaplama: 32 leaf + 16 spine</text>
<text x="420" y="4071" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her leaf her spine&#x27;a 2 × 400 Gb/s</text>
<text x="40" y="4087" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1:1 fat-tree · SHARPv3</text>
<text x="420" y="4087" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">4 ağ: hesaplama, depolama, 2 yönetim</text>
<text x="40" y="4115" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Neden önemli?</text>
<text x="40" y="4133" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her düğümün NIC 0&#x27;ı Leaf-xx-01&#x27;e, NIC 7&#x27;si Leaf-xx-08&#x27;e takılır; bu sekiz şerit raylardır ve aynı numaralı GPU&#x27;lar tek</text>
<text x="40" y="4150" text-anchor="start" style="fill:var(--c-text);font-size:12px">sıçramada buluşur. Her leaf de her spine&#x27;a ikişer 400 Gb/s hatla ulaşır; raylar arası trafik bu yüzden hep bir spine</text>
<text x="40" y="4167" text-anchor="start" style="fill:var(--c-text);font-size:12px">üzerinden döner. Depolama ve yönetim trafiği hesaplama ağını meşgul etmesin diye kendi ağlarından akar. Aynı desen 64 SU&#x27;ya</text>
<text x="40" y="4184" text-anchor="start" style="fill:var(--c-text);font-size:12px">büyür: 2.048 DGX, yani 16.384 H100; B ve C bölümleri bu ölçekte düşünür.</text>
<rect x="36.0" y="4202.0" width="768.0" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="4217" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Restoran zinciri: InfiniBand şubeler arası kurye, depolama merkez depo.</text>
<line x1="420.0" y1="4237.0" x2="420.0" y2="4259.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="4252" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">SuperPOD&#x27;un ağını yakından inceleyelim</text>
<rect x="28" y="4262" width="784" height="719" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="4286" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">4b · SuperPOD&#x27;u yakından incele</text>
<text x="40" y="4306" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Aynı 127 DGX&#x27;i bu kez ağ gözüyle okuyalım: ağ neden engelsiz, trafik nasıl yol bulur ve ne kadar büyüyebilir?</text>
<text x="40" y="4336" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">1 · Ağ neden engelsiz (1:1)? Portları sayalım</text>
<rect x="40.0" y="4362.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="105.0" y="4376" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">32 DGX</text>
<text x="105.0" y="4390" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">her birinin NIC i&#x27;si</text>
<line x1="173.0" y1="4382.0" x2="212.0" y2="4382.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.8"/>
<text x="192" y="4374" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 port</text>
<rect x="215.0" y="4362.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4376" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">leaf i · QM9700</text>
<text x="280.0" y="4390" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">64 × 400 Gb/s port</text>
<line x1="348.0" y1="4382.0" x2="387.0" y2="4382.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.8"/>
<text x="367" y="4374" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">32 port</text>
<rect x="390.0" y="4362.0" width="130.0" height="40" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="455.0" y="4376" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">16 spine</text>
<text x="455.0" y="4390" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">64&#x27;er port</text>
<rect x="40.0" y="4418.0" width="480.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.10;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4433.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 leaf × 32 yukarı port = 1.024 kablo = 16 spine × 64 port</text>
<text x="40" y="4462" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Kablo sayısı (NVIDIA tablosu): düğümlerden leaf&#x27;lere 1.020 (127 × 8 + 4 UFM),</text>
<text x="40" y="4479" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">leaf&#x27;lerden spine&#x27;lara 1.024.</text>
<text x="552" y="4374" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her leaf&#x27;in 64 portunun 32&#x27;si düğümlere</text>
<text x="552" y="4391" text-anchor="start" style="fill:var(--c-text);font-size:12px">iner, 32&#x27;si spine&#x27;lara çıkar. Aşağıdan</text>
<text x="552" y="4408" text-anchor="start" style="fill:var(--c-text);font-size:12px">ne kadar trafik girerse yukarı o kadar</text>
<text x="552" y="4425" text-anchor="start" style="fill:var(--c-text);font-size:12px">çıkabilir; bu yüzden aşırı abonelik</text>
<text x="552" y="4442" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoktur (1:1).</text>
<line x1="540.0" y1="4360.0" x2="540.0" y2="4479.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="4515" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">2 · Trafik nasıl yol bulur?</text>
<text x="40" y="4539" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aynı ray: A&#x27;nın GPU 3&#x27;ü → B&#x27;nin GPU 3&#x27;ü</text>
<rect x="400.3" y="4527.0" width="119.7" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="460.1" y="4540.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">1 leaf, spine yok</text>
<rect x="40.0" y="4553.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="4569.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="4565.0" x2="138.0" y2="4565.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="140.0" y="4553.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="180.0" y="4569.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="222.0" y1="4565.0" x2="238.0" y2="4565.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="4553.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="280.0" y="4569.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · B</text>
<text x="40" y="4605" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Farklı ray, düz yol: GPU 3 → GPU 5</text>
<rect x="412.5" y="4593.0" width="107.5" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="466.2" y="4606.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine&#x27;dan geçer</text>
<rect x="40.0" y="4619.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="4635.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="4631.0" x2="138.0" y2="4631.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="140.0" y="4619.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="180.0" y="4635.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 3</text>
<line x1="222.0" y1="4631.0" x2="238.0" y2="4631.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="4619.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4635.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine</text>
<line x1="322.0" y1="4631.0" x2="338.0" y2="4631.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="340.0" y="4619.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="380.0" y="4635.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="422.0" y1="4631.0" x2="438.0" y2="4631.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="440.0" y="4619.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="480.0" y="4635.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<text x="40" y="4671" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Farklı ray, PXN ile: önce NVLink, sonra tek leaf</text>
<rect x="449.1" y="4659.0" width="70.9" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="484.6" y="4672.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine yok</text>
<rect x="40.0" y="4685.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="80.0" y="4701.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 3 · A</text>
<line x1="122.0" y1="4697.0" x2="138.0" y2="4697.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:2.2"/>
<rect x="140.0" y="4685.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="180.0" y="4701.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · A</text>
<line x1="222.0" y1="4697.0" x2="238.0" y2="4697.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="240.0" y="4685.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="280.0" y="4701.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">leaf 5</text>
<line x1="322.0" y1="4697.0" x2="338.0" y2="4697.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.4"/>
<rect x="340.0" y="4685.0" width="80.0" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="380.0" y="4701.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU 5 · B</text>
<text x="552" y="4539" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA&#x27;ya göre bir raydaki trafik SU</text>
<text x="552" y="4556" text-anchor="start" style="fill:var(--c-text);font-size:12px">içindeki diğer 31 düğüme her zaman tek</text>
<text x="552" y="4573" text-anchor="start" style="fill:var(--c-text);font-size:12px">sıçramadır; raylar arası trafik ise</text>
<text x="552" y="4590" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine katmanından geçer. NCCL 2.12&#x27;deki</text>
<text x="552" y="4607" text-anchor="start" style="fill:var(--c-text);font-size:12px">PXN, veriyi önce NVLink ile doğru</text>
<text x="552" y="4624" text-anchor="start" style="fill:var(--c-text);font-size:12px">raydaki GPU&#x27;ya taşıyıp spine&#x27;ı atlar;</text>
<text x="552" y="4641" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA bu sayede all2all veriminin iki</text>
<text x="552" y="4658" text-anchor="start" style="fill:var(--c-text);font-size:12px">katına çıktığını raporlar.</text>
<line x1="540.0" y1="4527.0" x2="540.0" y2="4709.0" style="stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="4745" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">3 · Ölçek ve bant genişliği</text>
<rect x="40.0" y="4757" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="4779" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">400 GB/s</text>
<text x="50.0" y="4796" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">düğüm başına</text>
<rect x="228.0" y="4757" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="238.0" y="4779" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">12,8 TB/s</text>
<text x="238.0" y="4796" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SU başına</text>
<rect x="40.0" y="4813" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="50.0" y="4835" text-anchor="start" style="fill:var(--c-accent);font-size:16px;font-weight:700">50,8 TB/s</text>
<text x="50.0" y="4852" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">tüm GPU&#x27;lar</text>
<rect x="228.0" y="4813" width="180.0" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="238.0" y="4835" text-anchor="start" style="fill:var(--c-accent-2);font-size:16px;font-weight:700">51,2 TB/s</text>
<text x="238.0" y="4852" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">spine kapasitesi</text>
<text x="40" y="4891" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bir düğüm ağa 8 × 50 GB/s = 400 GB/s basar; 32 düğümlük bir</text>
<text x="40" y="4908" text-anchor="start" style="fill:var(--c-text);font-size:12px">SU 12,8 TB/s eder. 1.016 GPU&#x27;nun birlikte gönderebildiği</text>
<text x="40" y="4925" text-anchor="start" style="fill:var(--c-text);font-size:12px">(50,8 TB/s) ile 16 spine&#x27;ın taşıyabildiği</text>
<text x="40" y="4942" text-anchor="start" style="fill:var(--c-text);font-size:12px">(16 × 64 × 50 GB/s = 51,2 TB/s) neredeyse eşittir; engelsiz</text>
<text x="40" y="4959" text-anchor="start" style="fill:var(--c-text);font-size:12px">ağın sayılarla tanımı budur.</text>
<text x="432" y="4745" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">4 · Daha büyüğü: üçüncü katman</text>
<text x="432" y="4771" text-anchor="start" style="fill:var(--c-text);font-size:12px">NVIDIA&#x27;nın tablosunda 8 SU&#x27;ya kadar iki katman (leaf +</text>
<text x="432" y="4788" text-anchor="start" style="fill:var(--c-text);font-size:12px">spine) yeter; 16 SU&#x27;dan itibaren üstlerine bir core katmanı</text>
<text x="432" y="4805" text-anchor="start" style="fill:var(--c-text);font-size:12px">eklenir. 2.048 DGX&#x27;lik (16.384 H100) kurulum 512 leaf,</text>
<text x="432" y="4822" text-anchor="start" style="fill:var(--c-text);font-size:12px">512 spine ve 256 core switch kullanır; B ve C bölümleri bu</text>
<text x="432" y="4839" text-anchor="start" style="fill:var(--c-text);font-size:12px">ölçekte düşünür.</text>
<text x="432" y="4865" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px"><tspan style="fill:var(--c-accent);font-weight:700">Pratikte:</tspan> NCCL kanallarını raylara göre kurar: aynı numaralı</text>
<text x="432" y="4882" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GPU&#x27;lar tek leaf üzerinden konuşur, raylar arası trafik ise</text>
<text x="432" y="4899" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">PXN ile önce NVLink&#x27;e taşınır.</text>
<line x1="420.0" y1="4757.0" x2="420.0" y2="4959.0" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="420.0" y1="4984.0" x2="420.0" y2="5006.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="432.0" y="4999" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700">kümeyi ayakta tutan yapı taşları</text>
<rect x="28" y="5009" width="784" height="797" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="5033" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700">SuperPOD&#x27;un yapı taşları: her biri neden var?</text>
<text x="40" y="5052" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVIDIA&#x27;nın 4 SU&#x27;luk referans tasarımı (127 DGX); adetler ve görevler NVIDIA belgelerinden. Şerit rengi parçanın hangi sütuna hizmet</text>
<text x="40" y="5067" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ettiğini gösterir; gri yönetim ve altyapıdır.</text>
<text x="40" y="5091" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Donanım</text>
<rect x="36" y="5101" width="248" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5101" width="4" height="128" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="5120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DGX H100 sistemi</text>
<rect x="237.8" y="5107.0" width="38.2" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="256.9" y="5120.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×127</text>
<text x="48" y="5137" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 H100 · 640 GB HBM3 · 900 GB/s NVLink</text>
<text x="48" y="5155" text-anchor="start" style="fill:var(--c-text);font-size:11px">Modelin bütün matematiği burada</text>
<text x="48" y="5170" text-anchor="start" style="fill:var(--c-text);font-size:11px">yapılır; kümedeki diğer her parça bu</text>
<text x="48" y="5185" text-anchor="start" style="fill:var(--c-text);font-size:11px">kutuları beslemek için vardır.</text>
<rect x="296" y="5101" width="248" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5101" width="4" height="128" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="308" y="5120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Hesaplama ağı · QM9700</text>
<rect x="503.9" y="5107.0" width="32.2" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="519.9" y="5120.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×48</text>
<text x="308" y="5137" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">32 leaf + 16 spine · düğüm başına</text>
<text x="308" y="5151" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 × 400 Gb/s</text>
<text x="308" y="5169" text-anchor="start" style="fill:var(--c-text);font-size:11px">Gradyan ve aktivasyon trafiği düğümler</text>
<text x="308" y="5184" text-anchor="start" style="fill:var(--c-text);font-size:11px">arasında tıkanmadan akmalı; bu yüzden</text>
<text x="308" y="5199" text-anchor="start" style="fill:var(--c-text);font-size:11px">her GPU&#x27;nun kendi 400 Gb/s&#x27;lik rayı</text>
<text x="308" y="5214" text-anchor="start" style="fill:var(--c-text);font-size:11px">vardır.</text>
<rect x="556" y="5101" width="248" height="128" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5101" width="4" height="128" rx="2" style="fill:var(--c-warn);fill-opacity:.85"/>
<text x="568" y="5120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Depolama ağı · QM9700</text>
<rect x="763.9" y="5107.0" width="32.2" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="779.9" y="5120.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×16</text>
<text x="568" y="5137" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">düğüm başına ≥ 40 GB/s · ~4:3</text>
<text x="568" y="5155" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri okuma ve kontrol noktası yazma,</text>
<text x="568" y="5170" text-anchor="start" style="fill:var(--c-text);font-size:11px">hesaplama ağındaki AllReduce trafiğine</text>
<text x="568" y="5185" text-anchor="start" style="fill:var(--c-text);font-size:11px">karışmasın; hızı depolama dizisinin</text>
<text x="568" y="5200" text-anchor="start" style="fill:var(--c-text);font-size:11px">tepe hızına göre seçilir.</text>
<rect x="36" y="5237" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5237" width="4" height="144" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="48" y="5256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">UFM cihazı</text>
<rect x="249.9" y="5243.0" width="26.1" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="262.9" y="5256.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×4</text>
<text x="48" y="5273" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">InfiniBand telemetrisi ve analitiği</text>
<text x="48" y="5291" text-anchor="start" style="fill:var(--c-text);font-size:11px">Binlerce port ve kabloyu izleyip</text>
<text x="48" y="5306" text-anchor="start" style="fill:var(--c-text);font-size:11px">bozulan bağlantıyı bulur (Llama 3</text>
<text x="48" y="5321" text-anchor="start" style="fill:var(--c-text);font-size:11px">eğitiminde kesintilerin %8,4&#x27;ü switch</text>
<text x="48" y="5336" text-anchor="start" style="fill:var(--c-text);font-size:11px">ve kablodan geldi). Bir çifti bir</text>
<text x="48" y="5351" text-anchor="start" style="fill:var(--c-text);font-size:11px">DGX&#x27;in yerini alır, küme bu yüzden 127</text>
<text x="48" y="5366" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğümdür.</text>
<rect x="296" y="5237" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5237" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="5256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">In-band yönetim · SN4600C</text>
<rect x="509.9" y="5243.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="522.9" y="5256.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="308" y="5273" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">64 port × 100 GbE (Ethernet)</text>
<text x="308" y="5291" text-anchor="start" style="fill:var(--c-text);font-size:11px">Düğüm kurulumu, kullanıcı girişi, dosya</text>
<text x="308" y="5306" text-anchor="start" style="fill:var(--c-text);font-size:11px">sistemi ve dış erişim (NGC, paket</text>
<text x="308" y="5321" text-anchor="start" style="fill:var(--c-text);font-size:11px">depoları) bu ağdan akar; hesaplama ağı</text>
<text x="308" y="5336" text-anchor="start" style="fill:var(--c-text);font-size:11px">yalnız GPU trafiğine kalır.</text>
<rect x="556" y="5237" width="248" height="144" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5237" width="4" height="144" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="568" y="5256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">OOB yönetim · SN2201</text>
<rect x="769.9" y="5243.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="782.9" y="5256.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×8</text>
<text x="568" y="5273" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">48 port × 1 GbE bakır</text>
<text x="568" y="5291" text-anchor="start" style="fill:var(--c-text);font-size:11px">Her sunucunun BMC yönetim çipine ve</text>
<text x="568" y="5306" text-anchor="start" style="fill:var(--c-text);font-size:11px">PDU&#x27;lara ulaşır; işletim sistemi</text>
<text x="568" y="5321" text-anchor="start" style="fill:var(--c-text);font-size:11px">kilitlense bile düğüm uzaktan kapatılıp</text>
<text x="568" y="5336" text-anchor="start" style="fill:var(--c-text);font-size:11px">açılabilir. Kullanıcı ağlarından</text>
<text x="568" y="5351" text-anchor="start" style="fill:var(--c-text);font-size:11px">ayrıdır.</text>
<rect x="36" y="5389" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5389" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="5408" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Yönetim sunucuları</text>
<rect x="249.9" y="5395.0" width="26.1" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="262.9" y="5408.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">×5</text>
<text x="48" y="5425" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 soket · 24+ çekirdek · 384 GB RAM</text>
<text x="48" y="5443" text-anchor="start" style="fill:var(--c-text);font-size:11px">Küme yönetimi ve iş zamanlama</text>
<text x="48" y="5458" text-anchor="start" style="fill:var(--c-text);font-size:11px">yazılımları ayrı makinelerde koşar; GPU</text>
<text x="48" y="5473" text-anchor="start" style="fill:var(--c-text);font-size:11px">düğümleri yalnızca eğitime kalır.</text>
<rect x="296" y="5389" width="248" height="129" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5389" width="4" height="129" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="5408" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kabin ve güç</text>
<rect x="473.6" y="5395.0" width="62.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="504.8" y="5408.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">38 kabin</text>
<text x="308" y="5425" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">38 kabin · 108 PDU</text>
<text x="308" y="5443" text-anchor="start" style="fill:var(--c-text);font-size:11px">Kabin başına 4 DGX (40,8 kW), SU başına</text>
<text x="308" y="5458" text-anchor="start" style="fill:var(--c-text);font-size:11px">8 DGX kabini; 127 DGX tek başına</text>
<text x="308" y="5473" text-anchor="start" style="fill:var(--c-text);font-size:11px">127 × 10,2 kW ≈ 1,3 MW çeker. Güç ve</text>
<text x="308" y="5488" text-anchor="start" style="fill:var(--c-text);font-size:11px">soğutma kümenin görünmeyen dördüncü</text>
<text x="308" y="5503" text-anchor="start" style="fill:var(--c-text);font-size:11px">sütunudur.</text>
<text x="40" y="5544" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Yazılım</text>
<rect x="36" y="5554" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5554" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="48" y="5573" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Base Command Manager</text>
<rect x="219.7" y="5560.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="247.8" y="5573.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="48" y="5590" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">küme yönetimi · otomatik kurulum</text>
<text x="48" y="5608" text-anchor="start" style="fill:var(--c-text);font-size:11px">Binlerce düğüme aynı işletim sistemi</text>
<text x="48" y="5623" text-anchor="start" style="fill:var(--c-text);font-size:11px">imajını kurar, günceller ve izler; 127</text>
<text x="48" y="5638" text-anchor="start" style="fill:var(--c-text);font-size:11px">makineyi elle yönetmek gerçekçi</text>
<text x="48" y="5653" text-anchor="start" style="fill:var(--c-text);font-size:11px">değildir.</text>
<rect x="296" y="5554" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5554" width="4" height="114" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="308" y="5573" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Slurm</text>
<rect x="467.6" y="5560.0" width="68.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="501.8" y="5573.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesaplama</text>
<text x="308" y="5590" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">toplu iş zamanlayıcı</text>
<text x="308" y="5608" text-anchor="start" style="fill:var(--c-text);font-size:11px">Çok düğümlü bir eğitim işinin hangi</text>
<text x="308" y="5623" text-anchor="start" style="fill:var(--c-text);font-size:11px">GPU&#x27;larda ve hangi sırayla koşacağını</text>
<text x="308" y="5638" text-anchor="start" style="fill:var(--c-text);font-size:11px">belirler; arızalı düğümü devreden</text>
<text x="308" y="5653" text-anchor="start" style="fill:var(--c-text);font-size:11px">çıkarıp işi yeniden kuyruğa alabilir.</text>
<rect x="556" y="5554" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5554" width="4" height="114" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<text x="568" y="5573" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Magnum IO</text>
<rect x="703.4" y="5560.0" width="92.7" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="749.7" y="5573.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ + depolama</text>
<text x="568" y="5590" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NCCL · GPUDirect RDMA / Storage · SHARP</text>
<text x="568" y="5608" text-anchor="start" style="fill:var(--c-text);font-size:11px">Verinin GPU&#x27;lar ve diskler arasındaki</text>
<text x="568" y="5623" text-anchor="start" style="fill:var(--c-text);font-size:11px">hareketini hızlandıran kütüphane</text>
<text x="568" y="5638" text-anchor="start" style="fill:var(--c-text);font-size:11px">yığını; B&#x27;deki 1., 3. ve 6. adımlar bu</text>
<text x="568" y="5653" text-anchor="start" style="fill:var(--c-text);font-size:11px">katmanla yapılır.</text>
<rect x="36" y="5676" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="36" y="5676" width="4" height="114" rx="2" style="fill:var(--c-accent);fill-opacity:.85"/>
<text x="48" y="5695" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">NGC kataloğu</text>
<rect x="207.6" y="5682.0" width="68.5" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="241.8" y="5695.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesaplama</text>
<text x="48" y="5712" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU için optimize konteynerler</text>
<text x="48" y="5730" text-anchor="start" style="fill:var(--c-text);font-size:11px">CUDA, NCCL ve PyTorch gibi çerçeveler</text>
<text x="48" y="5745" text-anchor="start" style="fill:var(--c-text);font-size:11px">birbirine uyan tek bir paket hâlinde</text>
<text x="48" y="5760" text-anchor="start" style="fill:var(--c-text);font-size:11px">gelir; herkes aynı yazılım yığınıyla</text>
<text x="48" y="5775" text-anchor="start" style="fill:var(--c-text);font-size:11px">başlar.</text>
<rect x="296" y="5676" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="296" y="5676" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="308" y="5695" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Base Command Platform</text>
<rect x="479.6" y="5682.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="507.8" y="5695.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="308" y="5712" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AI eğitimi için yazılım hizmeti</text>
<text x="308" y="5730" text-anchor="start" style="fill:var(--c-text);font-size:11px">Veri bilimcilerin iş gönderip veri</text>
<text x="308" y="5745" text-anchor="start" style="fill:var(--c-text);font-size:11px">setlerini ve sonuçları yönettiği</text>
<text x="308" y="5760" text-anchor="start" style="fill:var(--c-text);font-size:11px">katman; alttaki donanımı bilmeden</text>
<text x="308" y="5775" text-anchor="start" style="fill:var(--c-text);font-size:11px">kümeyi kullandırır.</text>
<rect x="556" y="5676" width="248" height="114" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="556" y="5676" width="4" height="114" rx="2" style="fill:var(--c-text-mute);fill-opacity:.85"/>
<text x="568" y="5695" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">NVIDIA AI Enterprise</text>
<rect x="739.6" y="5682.0" width="56.4" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.14;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="767.8" y="5695.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yönetim</text>
<text x="568" y="5712" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">araçlar, çerçeveler, orkestrasyon</text>
<text x="568" y="5730" text-anchor="start" style="fill:var(--c-text);font-size:11px">Geliştiriciye çerçeve ve araçları, BT</text>
<text x="568" y="5745" text-anchor="start" style="fill:var(--c-text);font-size:11px">ekibine destekli yönetim ve</text>
<text x="568" y="5760" text-anchor="start" style="fill:var(--c-text);font-size:11px">orkestrasyonu verir.</text>
<rect x="28.0" y="5818.0" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="5838" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: GPU&#x27;dan uzaklaştıkça yol daralır: HBM 3,35 TB/s → NVLink 450 GB/s → ağ 50 GB/s.</text>
<text x="420.0" y="5883" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Parçaları tanıdık. Şimdi bir eğitim adımında birlikte nasıl çalıştıklarını izleyelim.</text>
<line x1="420.0" y1="5891.0" x2="420.0" y2="5911.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="5915" width="808" height="880" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="5939" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="5943.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">B</text>
<text x="56" y="5944" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Bir eğitim adımını izle: veri nereden geçer?</text>
<text x="28" y="5965" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">64 SU&#x27;luk H100 SuperPOD&#x27;da Llama 3 405B: TP 8 × PP 16 × DP 128 = 16.384 GPU</text>
<rect x="28" y="5977" width="784" height="264" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="290.0" y="5989.0" width="260.0" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="6004.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">spine → diğer 127 DP kopyası</text>
<rect x="48" y="6037" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="58" y="6053" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">düğüm A · PP aşaması s</text>
<rect x="88.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="150.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="212.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="274.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="88.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="112.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="150.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="174.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="212.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="274.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="298.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="112.0" y1="6081.0" x2="112.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="112.0" y1="6107.0" x2="112.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="6081.0" x2="174.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="174.0" y1="6107.0" x2="174.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="6081.0" x2="236.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="236.0" y1="6107.0" x2="236.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="6081.0" x2="298.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="298.0" y1="6107.0" x2="298.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="88.0" y="6089.0" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="205.0" y="6102.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<circle cx="64" cy="6071" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<circle cx="64" cy="6098" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="64" y="6102.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<circle cx="64" cy="6125" r="9" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="64" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<rect x="502" y="6037" width="290" height="116" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<text x="512" y="6053" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">düğüm B · aşama s+1</text>
<rect x="530.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 0</text>
<rect x="592.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 1</text>
<rect x="654.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 2</text>
<rect x="716.0" y="6061.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="6075.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 3</text>
<rect x="530.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 4</text>
<rect x="592.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="616.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 5</text>
<rect x="654.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="678.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 6</text>
<rect x="716.0" y="6115.0" width="48.0" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="740.0" y="6129.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU 7</text>
<line x1="554.0" y1="6081.0" x2="554.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="554.0" y1="6107.0" x2="554.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="6081.0" x2="616.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="616.0" y1="6107.0" x2="616.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="6081.0" x2="678.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="678.0" y1="6107.0" x2="678.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="6081.0" x2="740.0" y2="6089.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="740.0" y1="6107.0" x2="740.0" y2="6115.0" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="530.0" y="6089.0" width="234.0" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="6102.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVSwitch</text>
<rect x="396.0" y="6047.0" width="48.0" height="96" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="6105" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">leaf</text>
<circle cx="420.0" cy="6073" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="420.0" y="6077.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<circle cx="420.0" cy="6127" r="9" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="420.0" y="6131.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<line x1="338.0" y1="6073.0" x2="396.0" y2="6073.0" style="stroke:var(--c-accent-2);stroke-width:1.6"/>
<line x1="444.0" y1="6073.0" x2="500.0" y2="6073.0" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="502.0" y1="6127.0" x2="444.0" y2="6127.0" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:4 3"/>
<line x1="396.0" y1="6127.0" x2="340.0" y2="6127.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<line x1="420.0" y1="6045.0" x2="420.0" y2="6013.0" marker-start="url(#bp-gen)" marker-end="url(#bp-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<circle cx="438.0" cy="6029" r="9" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438.0" y="6033.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<rect x="48.0" y="6189.0" width="744.0" height="40" rx="5" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="420.0" y="6205" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">depolama sistemi (ör. Lustre · Weka)</text>
<text x="420.0" y="6221" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ayrı depolama ağı · düğüm başına ≥ 40 GB/s</text>
<line x1="108.0" y1="6187.0" x2="108.0" y2="6156.0" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<circle cx="126" cy="6171" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="126" y="6175.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<line x1="208.0" y1="6156.0" x2="208.0" y2="6187.0" marker-end="url(#bp-wrn)" style="stroke:var(--c-warn);stroke-width:1.5;stroke-dasharray:5 4"/>
<circle cx="226" cy="6171" r="9" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="226" y="6175.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="647.0" y="6173" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">(B düğümü de aynı yolu kullanır)</text>
<rect x="28" y="6251" width="784" height="470" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="46.0" y1="6277.0" x2="46.0" y2="6587.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="46" cy="6277" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="46" y="6281.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">1</text>
<text x="66" y="6282" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Veriyi oku</text>
<rect x="66.0" y="6291.0" width="165.2" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="148.6" y="6304.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">depolama → HBM · her adım</text>
<text x="66" y="6327" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Bu adımın 16 milyon tokeni depodan HBM&#x27;e gelir; GDS ya</text>
<text x="66" y="6343" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">da mmap sayesinde araya CPU kopyası girmez.</text>
<circle cx="46" cy="6375" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="46" y="6379.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">2</text>
<text x="66" y="6380" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">İleri geçiş</text>
<rect x="66.0" y="6389.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="151.7" y="6402.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM 3,35 TB/s · her katman</text>
<text x="66" y="6425" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her katmanda SM&#x27;ler ağırlıkları HBM&#x27;den çekip Tensor</text>
<text x="66" y="6441" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Core&#x27;larda çarpar; okunan bayt başına ~295 işlem</text>
<text x="66" y="6457" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">yapılırsa çekirdekler doyar.</text>
<circle cx="46" cy="6489" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="6493.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">3</text>
<text x="66" y="6494" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP AllReduce</text>
<rect x="66.0" y="6503.0" width="189.5" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="160.7" y="6516.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink 450 GB/s · katmanda 2×</text>
<text x="66" y="6539" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her matris 8 GPU&#x27;ya bölündüğü için parçalı sonuçlar</text>
<text x="66" y="6555" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">NVLink&#x27;te toplanır; TP bu yüzden kasanın dışına taşmaz.</text>
<circle cx="46" cy="6587" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="46" y="6591.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">4</text>
<text x="66" y="6592" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Aktivasyonu gönder (PP)</text>
<rect x="66.0" y="6601.0" width="183.4" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="157.7" y="6614.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">50 GB/s/GPU · her mikrobatch</text>
<text x="66" y="6637" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Düğümün katmanları bitince çıktı (~16 MB) ağdan sonraki</text>
<text x="66" y="6653" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">düğüme geçer; yol 9 kat dardır ama giden veri küçüktür.</text>
<line x1="438.0" y1="6277.0" x2="438.0" y2="6619.0" style="stroke:var(--c-border);stroke-width:2"/>
<circle cx="438" cy="6277" r="10" style="fill:var(--c-text-mute);fill-opacity:.18;stroke:var(--c-text-mute);stroke-width:1.3"/>
<text x="438" y="6281.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">5</text>
<text x="458" y="6282" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Geri geçiş</text>
<rect x="458.0" y="6291.0" width="141.1" height="19" rx="5" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="528.5" y="6304.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">aynı hatlar, ters yön</text>
<text x="458" y="6327" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Hata sinyali (gradyan) aynı yolu geri döner; 3. ve 4.</text>
<text x="458" y="6343" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">adımlar ters yönde tekrarlanır ve iş ileri geçişin ~2</text>
<text x="458" y="6359" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">katıdır.</text>
<circle cx="438" cy="6391" r="10" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="438" y="6395.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">6</text>
<text x="458" y="6396" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Gradyan AllReduce (DP)</text>
<rect x="458.0" y="6405.0" width="171.3" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="543.6" y="6418.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ray switch&#x27;leri · her adım</text>
<text x="458" y="6441" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">128 kopyanın gradyanları ortalanır ki hepsi aynı</text>
<text x="458" y="6457" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">güncellemeyi yapsın; InfiniBand SHARP toplamayı switch</text>
<text x="458" y="6473" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">içinde yapıp süreyi 2·B/W&#x27;den B/W&#x27;ye indirir.</text>
<circle cx="438" cy="6505" r="10" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="438" y="6509.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">7</text>
<text x="458" y="6510" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Optimizer adımı</text>
<rect x="458.0" y="6519.0" width="141.1" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="528.5" y="6532.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM içinde · her adım</text>
<text x="458" y="6555" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">AdamW ağırlıkları günceller ve her parametre için 12</text>
<text x="458" y="6571" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">bayt FP32 durum tutar: ana ağırlık, momentum ve</text>
<text x="458" y="6587" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">varyans.</text>
<circle cx="438" cy="6619" r="10" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="438" y="6623.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">8</text>
<text x="458" y="6624" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Kontrol noktası</text>
<rect x="458.0" y="6633.0" width="195.5" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="555.8" y="6646.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px">HBM → depolama · ~14 dk&#x27;da bir</text>
<text x="458" y="6669" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">~14 dakikada bir 7,5 TB durum diske yazılır: 250 GB/s</text>
<text x="458" y="6685" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ile ~31 s, eski 25 GB/s ile ~5 dk sürer; arızadan sonra</text>
<text x="458" y="6701" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">buradan devam edilir.</text>
<line x1="420.0" y1="6265.0" x2="420.0" y2="6707.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="6733.0" width="784.0" height="50" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="6753" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: bir adımda veri dört ayrı yoldan geçer (HBM, NVLink, ağ, depolama); adım ancak en yavaş yol kadar</text>
<text x="40" y="6770" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">hızlıdır.</text>
<text x="420.0" y="6815" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Peki bu yollar nerede tıkanıyor? Her sütunun bir duvarı var.</text>
<line x1="420.0" y1="6823.0" x2="420.0" y2="6843.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="6847" width="808" height="1050" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="6871" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="6875.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">C</text>
<text x="56" y="6876" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Üç duvar: belirti → neden → çare</text>
<text x="28" y="6897" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Eğriler makaledeki formüllerden hesaplandı; kırmızı darboğaz, yeşil çaredir.</text>
<rect x="28" y="6909" width="784" height="295" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="6933" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Bellek duvarı</text>
<text x="800" y="6933" text-anchor="end" style="fill:var(--c-accent);font-size:11px;font-weight:700">1. sütun · hesaplama</text>
<line x1="80.0" y1="6953.0" x2="80.0" y2="7079.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="7079.0" x2="512.0" y2="7079.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="6960.2" x2="512.0" y2="6960.2" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="6964.2" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">990</text>
<text x="76" y="7083" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="40" y="6952.2" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s</text>
<line x1="377.9" y1="6960.2" x2="377.9" y2="7079.0" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<polyline points="80.0,7078.6 84.5,7078.6 89.0,7078.5 93.5,7078.5 98.0,7078.4 102.5,7078.4 107.0,7078.3 111.5,7078.3 116.0,7078.2 120.5,7078.1 125.0,7078.0 129.5,7078.0 134.0,7077.9 138.5,7077.8 143.0,7077.6 147.5,7077.5 152.0,7077.4 156.5,7077.2 161.0,7077.1 165.5,7076.9 170.0,7076.7 174.5,7076.5 179.0,7076.3 183.5,7076.1 188.0,7075.8 192.5,7075.5 197.0,7075.2 201.5,7074.8 206.0,7074.5 210.5,7074.0 215.0,7073.6 219.5,7073.1 224.0,7072.6 228.5,7072.0 233.0,7071.4 237.5,7070.7 242.0,7069.9 246.5,7069.1 251.0,7068.2 255.5,7067.3 260.0,7066.2 264.5,7065.0 269.0,7063.8 273.5,7062.4 278.0,7060.9 282.5,7059.3 287.0,7057.5 291.5,7055.6 296.0,7053.5 300.5,7051.2 305.0,7048.8 309.5,7046.1 314.0,7043.1 318.5,7039.9 323.0,7036.4 327.5,7032.7 332.0,7028.6 336.5,7024.1 341.0,7019.2 345.5,7014.0 350.0,7008.2 354.5,7002.0 359.0,6995.3 363.5,6988.0 368.0,6980.1 372.5,6971.5 377.0,6962.2 381.5,6960.2 386.0,6960.2 390.5,6960.2 395.0,6960.2 399.5,6960.2 404.0,6960.2 408.5,6960.2 413.0,6960.2 417.5,6960.2 422.0,6960.2 426.5,6960.2 431.0,6960.2 435.5,6960.2 440.0,6960.2 444.5,6960.2 449.0,6960.2 453.5,6960.2 458.0,6960.2 462.5,6960.2 467.0,6960.2 471.5,6960.2 476.0,6960.2 480.5,6960.2 485.0,6960.2 489.5,6960.2 494.0,6960.2 498.5,6960.2 503.0,6960.2 507.5,6960.2 512.0,6960.2" style="fill:none;stroke:var(--c-accent);stroke-width:2.2"/>
<text x="80.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1</text>
<text x="152.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4</text>
<text x="224.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="296.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">64</text>
<text x="368.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">256</text>
<text x="440.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.024</text>
<text x="512.0" y="7093" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4.096</text>
<text x="512" y="7109" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">batch B (token, log ölçek)</text>
<text x="150" y="6976.2" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700">bellek sınırlı</text>
<text x="470" y="7061" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700">hesap sınırlı</text>
<circle cx="80.0" cy="7078.6" r="4" style="fill:var(--c-danger)"/>
<text x="88.0" y="7069" text-anchor="start" style="fill:var(--c-text);font-size:11px">B = 1: 3,3 TFLOP/s (%0,3)</text>
<circle cx="332.0" cy="7028.6" r="4" style="fill:var(--c-accent)"/>
<text x="324.0" y="7022.6" text-anchor="end" style="fill:var(--c-text);font-size:11px">B = 128: 420 TFLOP/s (%42)</text>
<circle cx="377.9" cy="6960.2" r="4" style="fill:var(--c-success)"/>
<text x="385.9" y="6976.2" text-anchor="start" style="fill:var(--c-text);font-size:11px">B ≈ 310: tavan 990</text>
<text x="40" y="7127" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP (D = 8.192, F = 28.672), H100 BF16: hız = min(990, 3,35·α)</text>
<text x="40" y="7143" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: batch büyüdükçe okunan her bayt daha çok işe yarar; eğri B ≈ 310&#x27;da</text>
<text x="40" y="7158" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">tavana değer.</text>
<text x="552" y="6963" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="552" y="6981" text-anchor="start" style="fill:var(--c-text);font-size:12px">GPU meşgul görünür, ama gücünün yalnızca</text>
<text x="552" y="6998" text-anchor="start" style="fill:var(--c-text);font-size:12px">%1–2&#x27;si kullanılır.</text>
<text x="552" y="7026" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="552" y="7044" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her bayta düşen iş az: α = BDF / (B(D+F)</text>
<text x="552" y="7061" text-anchor="start" style="fill:var(--c-text);font-size:12px">+ DF), H100&#x27;ün eşiği olan 295&#x27;in altında</text>
<text x="552" y="7078" text-anchor="start" style="fill:var(--c-text);font-size:12px">kaldıkça Tensor Core&#x27;lar veri bekler.</text>
<text x="552" y="7106" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="552.0" y="7116.0" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="7130.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">batch ≥ ~310 token</text>
<rect x="552.0" y="7142.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="7156.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">füzyon / FlashAttention</text>
<rect x="552.0" y="7168.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="7182.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode: 4–8 bit ağırlık</text>
<line x1="540.0" y1="6949.0" x2="540.0" y2="7188.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="7214" width="784" height="321" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="7238" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">Ağ duvarı</text>
<text x="800" y="7238" text-anchor="end" style="fill:var(--c-accent-2);font-size:11px;font-weight:700">2. sütun · ağ</text>
<line x1="184.0" y1="7256.0" x2="184.0" y2="7340.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="184" y="7354" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="288.0" y1="7256.0" x2="288.0" y2="7340.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="288" y="7354" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="392.0" y1="7256.0" x2="392.0" y2="7340.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="392" y="7354" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="496.0" y1="7256.0" x2="496.0" y2="7340.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="496" y="7354" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="40" y="7274" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM (çip içi)</text>
<rect x="184" y="7262" width="262.6" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="452.6" y="7274" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3,35 TB/s</text>
<text x="40" y="7300" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="184" y="7288" width="171.9" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="361.9" y="7300" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="420.7" y="7300" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷7,4</text>
<text x="40" y="7326" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (düğüm dışı)</text>
<rect x="184" y="7314" width="72.7" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="262.7" y="7326" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="314.9" y="7326" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:700">÷9</text>
<rect x="40.0" y="7366.0" width="480.0" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="280.0" y="7380.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVLink: Y ≤ F / (C/W) = 28.672 / 2.200 ≈ 13 → TP = 8 sığar ✓ (F: Llama 3 70B)</text>
<rect x="40.0" y="7392.0" width="480.0" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="7406.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">InfiniBand: Y ≤ 28.672 / 19.800 ≈ 1,45 → TP düğümden taşarsa hesap aç kalır ✗</text>
<text x="40" y="7430" text-anchor="start" style="fill:var(--c-text);font-size:11px">PP = 16 aşama, kabarcık = 15 / (15 + M): M = 16 → %48 boşta · M = 128 → %10,5</text>
<text x="40" y="7448" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: her çubuk bir H100&#x27;ün o hattaki bant genişliği; düğümden çıkınca çubuk</text>
<text x="40" y="7463" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">9 kat kısalır.</text>
<text x="552" y="7268" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="552" y="7286" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğüm ekledikçe adım uzar; GPU&#x27;lar</text>
<text x="552" y="7303" text-anchor="start" style="fill:var(--c-text);font-size:12px">konuşmak için bekler.</text>
<text x="552" y="7331" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="552" y="7349" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düğümden çıkan veri 9 kat daha dar bir</text>
<text x="552" y="7366" text-anchor="start" style="fill:var(--c-text);font-size:12px">yoldan geçer; TP gibi her katmanda</text>
<text x="552" y="7383" text-anchor="start" style="fill:var(--c-text);font-size:12px">konuşan işler bu yolda boğulur.</text>
<text x="552" y="7411" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="552.0" y="7421.0" width="104.8" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="604.4" y="7435.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">TP düğüm içinde</text>
<rect x="552.0" y="7447.0" width="153.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="628.6" y="7461.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PP ve DP düğümler arası</text>
<rect x="552.0" y="7473.0" width="122.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="613.5" y="7487.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">SHARP: 2·B/W → B/W</text>
<rect x="552.0" y="7499.0" width="165.2" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="634.6" y="7513.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">DualPipe: iletişimi gizle</text>
<line x1="540.0" y1="7254.0" x2="540.0" y2="7519.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28" y="7545" width="784" height="295" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="40" y="7569" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700">G/Ç duvarı</text>
<text x="800" y="7569" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:700">3. sütun · depolama</text>
<line x1="80.0" y1="7593.0" x2="80.0" y2="7715.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="7715.0" x2="512.0" y2="7715.0" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="80.0" y1="7685.0" x2="512.0" y2="7685.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7689.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%10</text>
<line x1="80.0" y1="7655.0" x2="512.0" y2="7655.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7659.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%20</text>
<line x1="80.0" y1="7625.0" x2="512.0" y2="7625.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7629.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%30</text>
<line x1="80.0" y1="7595.0" x2="512.0" y2="7595.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="76" y="7599.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">%40</text>
<polyline points="80.0,7715.0 512.0,7640.0" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<polyline points="148.4,7595.3 149.6,7597.0 150.8,7598.6 152.0,7600.1 153.2,7601.6 154.4,7603.0 155.6,7604.4 156.8,7605.7 158.0,7606.9 159.2,7608.2 160.4,7609.3 161.6,7610.5 162.8,7611.6 164.0,7612.6 165.2,7613.7 166.4,7614.7 167.6,7615.6 168.8,7616.6 170.0,7617.5 171.2,7618.3 172.4,7619.2 173.6,7620.0 174.8,7620.8 176.0,7621.5 177.2,7622.3 178.4,7623.0 179.6,7623.7 180.8,7624.4 182.0,7625.0 183.2,7625.6 184.4,7626.3 185.6,7626.8 186.8,7627.4 188.0,7628.0 189.2,7628.5 190.4,7629.1 191.6,7629.6 192.8,7630.1 194.0,7630.5 195.2,7631.0 196.4,7631.5 197.6,7631.9 198.8,7632.3 200.0,7632.7 201.2,7633.1 202.4,7633.5 203.6,7633.9 204.8,7634.3 206.0,7634.6 207.2,7635.0 208.4,7635.3 209.6,7635.6 210.8,7635.9 212.0,7636.2 213.2,7636.5 214.4,7636.8 215.6,7637.1 216.8,7637.4 218.0,7637.6 219.2,7637.9 220.4,7638.1 221.6,7638.3 222.8,7638.6 224.0,7638.8 225.2,7639.0 226.4,7639.2 227.6,7639.4 228.8,7639.6 230.0,7639.8 231.2,7640.0 232.4,7640.2 233.6,7640.3 234.8,7640.5 236.0,7640.7 237.2,7640.8 238.4,7641.0 239.6,7641.1 240.8,7641.2 242.0,7641.4 243.2,7641.5 244.4,7641.6 245.6,7641.7 246.8,7641.8 248.0,7641.9 249.2,7642.1 250.4,7642.1 251.6,7642.2 252.8,7642.3 254.0,7642.4 255.2,7642.5 256.4,7642.6 257.6,7642.7 258.8,7642.7 260.0,7642.8 261.2,7642.9 262.4,7642.9 263.6,7643.0 264.8,7643.0 266.0,7643.1 267.2,7643.1 268.4,7643.2 269.6,7643.2 270.8,7643.2 272.0,7643.3 273.2,7643.3 274.4,7643.3 275.6,7643.3 276.8,7643.4 278.0,7643.4 279.2,7643.4 280.4,7643.4 281.6,7643.4 282.8,7643.4 284.0,7643.4 285.2,7643.4 286.4,7643.4 287.6,7643.4 288.8,7643.4 290.0,7643.4 291.2,7643.4 292.4,7643.4 293.6,7643.4 294.8,7643.4 296.0,7643.4 297.2,7643.3 298.4,7643.3 299.6,7643.3 300.8,7643.3 302.0,7643.2 303.2,7643.2 304.4,7643.2 305.6,7643.2 306.8,7643.1 308.0,7643.1 309.2,7643.0 310.4,7643.0 311.6,7643.0 312.8,7642.9 314.0,7642.9 315.2,7642.8 316.4,7642.8 317.6,7642.7 318.8,7642.7 320.0,7642.6 321.2,7642.6 322.4,7642.5 323.6,7642.4 324.8,7642.4 326.0,7642.3 327.2,7642.3 328.4,7642.2 329.6,7642.1 330.8,7642.1 332.0,7642.0 333.2,7641.9 334.4,7641.9 335.6,7641.8 336.8,7641.7 338.0,7641.6 339.2,7641.6 340.4,7641.5 341.6,7641.4 342.8,7641.3 344.0,7641.2 345.2,7641.2 346.4,7641.1 347.6,7641.0 348.8,7640.9 350.0,7640.8 351.2,7640.7 352.4,7640.6 353.6,7640.6 354.8,7640.5 356.0,7640.4 357.2,7640.3 358.4,7640.2 359.6,7640.1 360.8,7640.0 362.0,7639.9 363.2,7639.8 364.4,7639.7 365.6,7639.6 366.8,7639.5 368.0,7639.4 369.2,7639.3 370.4,7639.2 371.6,7639.1 372.8,7639.0 374.0,7638.9 375.2,7638.8 376.4,7638.7 377.6,7638.6 378.8,7638.5 380.0,7638.3 381.2,7638.2 382.4,7638.1 383.6,7638.0 384.8,7637.9 386.0,7637.8 387.2,7637.7 388.4,7637.6 389.6,7637.4 390.8,7637.3 392.0,7637.2 393.2,7637.1 394.4,7637.0 395.6,7636.8 396.8,7636.7 398.0,7636.6 399.2,7636.5 400.4,7636.4 401.6,7636.2 402.8,7636.1 404.0,7636.0 405.2,7635.9 406.4,7635.7 407.6,7635.6 408.8,7635.5 410.0,7635.4 411.2,7635.2 412.4,7635.1 413.6,7635.0 414.8,7634.9 416.0,7634.7 417.2,7634.6 418.4,7634.5 419.6,7634.3 420.8,7634.2 422.0,7634.1 423.2,7633.9 424.4,7633.8 425.6,7633.7 426.8,7633.5 428.0,7633.4 429.2,7633.3 430.4,7633.1 431.6,7633.0 432.8,7632.9 434.0,7632.7 435.2,7632.6 436.4,7632.4 437.6,7632.3 438.8,7632.2 440.0,7632.0 441.2,7631.9 442.4,7631.7 443.6,7631.6 444.8,7631.5 446.0,7631.3 447.2,7631.2 448.4,7631.0 449.6,7630.9 450.8,7630.7 452.0,7630.6 453.2,7630.5 454.4,7630.3 455.6,7630.2 456.8,7630.0 458.0,7629.9 459.2,7629.7 460.4,7629.6 461.6,7629.4 462.8,7629.3 464.0,7629.1 465.2,7629.0 466.4,7628.8 467.6,7628.7 468.8,7628.5 470.0,7628.4 471.2,7628.2 472.4,7628.1 473.6,7627.9 474.8,7627.8 476.0,7627.6 477.2,7627.5 478.4,7627.3 479.6,7627.2 480.8,7627.0 482.0,7626.9 483.2,7626.7 484.4,7626.6 485.6,7626.4 486.8,7626.3 488.0,7626.1 489.2,7625.9 490.4,7625.8 491.6,7625.6 492.8,7625.5 494.0,7625.3 495.2,7625.2 496.4,7625.0 497.6,7624.8 498.8,7624.7 500.0,7624.5 501.2,7624.4 502.4,7624.2 503.6,7624.1 504.8,7623.9 506.0,7623.7 507.2,7623.6 508.4,7623.4 509.6,7623.3 510.8,7623.1 512.0,7622.9" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<line x1="286.1" y1="7643.4" x2="286.1" y2="7715.0" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="286.1" cy="7643.4" r="4" style="fill:var(--c-danger)"/>
<text x="286.1" y="7729" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:700;font-family:var(--font-mono)">42,9</text>
<polyline points="87.2,7611.4 88.4,7625.8 89.6,7636.5 90.8,7644.9 92.0,7651.5 93.2,7656.9 94.4,7661.3 95.6,7665.0 96.8,7668.2 98.0,7670.9 99.2,7673.3 100.4,7675.3 101.6,7677.1 102.8,7678.7 104.0,7680.1 105.2,7681.4 106.4,7682.5 107.6,7683.5 108.8,7684.4 110.0,7685.2 111.2,7686.0 112.4,7686.6 113.6,7687.2 114.8,7687.8 116.0,7688.3 117.2,7688.7 118.4,7689.1 119.6,7689.5 120.8,7689.8 122.0,7690.2 123.2,7690.4 124.4,7690.7 125.6,7690.9 126.8,7691.1 128.0,7691.3 129.2,7691.5 130.4,7691.6 131.6,7691.8 132.8,7691.9 134.0,7692.0 135.2,7692.1 136.4,7692.1 137.6,7692.2 138.8,7692.3 140.0,7692.3 141.2,7692.3 142.4,7692.4 143.6,7692.4 144.8,7692.4 146.0,7692.4 147.2,7692.4 148.4,7692.3 149.6,7692.3 150.8,7692.3 152.0,7692.3 153.2,7692.2 154.4,7692.2 155.6,7692.1 156.8,7692.1 158.0,7692.0 159.2,7691.9 160.4,7691.9 161.6,7691.8 162.8,7691.7 164.0,7691.6 165.2,7691.6 166.4,7691.5 167.6,7691.4 168.8,7691.3 170.0,7691.2 171.2,7691.1 172.4,7691.0 173.6,7690.9 174.8,7690.8 176.0,7690.7 177.2,7690.5 178.4,7690.4 179.6,7690.3 180.8,7690.2 182.0,7690.1 183.2,7689.9 184.4,7689.8 185.6,7689.7 186.8,7689.6 188.0,7689.4 189.2,7689.3 190.4,7689.2 191.6,7689.0 192.8,7688.9 194.0,7688.7 195.2,7688.6 196.4,7688.5 197.6,7688.3 198.8,7688.2 200.0,7688.0 201.2,7687.9 202.4,7687.7 203.6,7687.6 204.8,7687.4 206.0,7687.3 207.2,7687.1 208.4,7687.0 209.6,7686.8 210.8,7686.7 212.0,7686.5 213.2,7686.3 214.4,7686.2 215.6,7686.0 216.8,7685.9 218.0,7685.7 219.2,7685.5 220.4,7685.4 221.6,7685.2 222.8,7685.0 224.0,7684.9 225.2,7684.7 226.4,7684.5 227.6,7684.4 228.8,7684.2 230.0,7684.0 231.2,7683.9 232.4,7683.7 233.6,7683.5 234.8,7683.4 236.0,7683.2 237.2,7683.0 238.4,7682.8 239.6,7682.7 240.8,7682.5 242.0,7682.3 243.2,7682.1 244.4,7682.0 245.6,7681.8 246.8,7681.6 248.0,7681.4 249.2,7681.3 250.4,7681.1 251.6,7680.9 252.8,7680.7 254.0,7680.6 255.2,7680.4 256.4,7680.2 257.6,7680.0 258.8,7679.8 260.0,7679.7 261.2,7679.5 262.4,7679.3 263.6,7679.1 264.8,7678.9 266.0,7678.7 267.2,7678.6 268.4,7678.4 269.6,7678.2 270.8,7678.0 272.0,7677.8 273.2,7677.6 274.4,7677.5 275.6,7677.3 276.8,7677.1 278.0,7676.9 279.2,7676.7 280.4,7676.5 281.6,7676.3 282.8,7676.2 284.0,7676.0 285.2,7675.8 286.4,7675.6 287.6,7675.4 288.8,7675.2 290.0,7675.0 291.2,7674.8 292.4,7674.7 293.6,7674.5 294.8,7674.3 296.0,7674.1 297.2,7673.9 298.4,7673.7 299.6,7673.5 300.8,7673.3 302.0,7673.1 303.2,7672.9 304.4,7672.8 305.6,7672.6 306.8,7672.4 308.0,7672.2 309.2,7672.0 310.4,7671.8 311.6,7671.6 312.8,7671.4 314.0,7671.2 315.2,7671.0 316.4,7670.8 317.6,7670.6 318.8,7670.5 320.0,7670.3 321.2,7670.1 322.4,7669.9 323.6,7669.7 324.8,7669.5 326.0,7669.3 327.2,7669.1 328.4,7668.9 329.6,7668.7 330.8,7668.5 332.0,7668.3 333.2,7668.1 334.4,7667.9 335.6,7667.7 336.8,7667.5 338.0,7667.4 339.2,7667.2 340.4,7667.0 341.6,7666.8 342.8,7666.6 344.0,7666.4 345.2,7666.2 346.4,7666.0 347.6,7665.8 348.8,7665.6 350.0,7665.4 351.2,7665.2 352.4,7665.0 353.6,7664.8 354.8,7664.6 356.0,7664.4 357.2,7664.2 358.4,7664.0 359.6,7663.8 360.8,7663.6 362.0,7663.4 363.2,7663.2 364.4,7663.0 365.6,7662.8 366.8,7662.6 368.0,7662.4 369.2,7662.2 370.4,7662.0 371.6,7661.8 372.8,7661.6 374.0,7661.5 375.2,7661.3 376.4,7661.1 377.6,7660.9 378.8,7660.7 380.0,7660.5 381.2,7660.3 382.4,7660.1 383.6,7659.9 384.8,7659.7 386.0,7659.5 387.2,7659.3 388.4,7659.1 389.6,7658.9 390.8,7658.7 392.0,7658.5 393.2,7658.3 394.4,7658.1 395.6,7657.9 396.8,7657.7 398.0,7657.5 399.2,7657.3 400.4,7657.1 401.6,7656.9 402.8,7656.7 404.0,7656.5 405.2,7656.3 406.4,7656.1 407.6,7655.9 408.8,7655.7 410.0,7655.5 411.2,7655.3 412.4,7655.1 413.6,7654.9 414.8,7654.7 416.0,7654.5 417.2,7654.3 418.4,7654.1 419.6,7653.9 420.8,7653.7 422.0,7653.5 423.2,7653.3 424.4,7653.1 425.6,7652.9 426.8,7652.7 428.0,7652.5 429.2,7652.3 430.4,7652.1 431.6,7651.9 432.8,7651.7 434.0,7651.5 435.2,7651.3 436.4,7651.1 437.6,7650.9 438.8,7650.7 440.0,7650.5 441.2,7650.3 442.4,7650.0 443.6,7649.8 444.8,7649.6 446.0,7649.4 447.2,7649.2 448.4,7649.0 449.6,7648.8 450.8,7648.6 452.0,7648.4 453.2,7648.2 454.4,7648.0 455.6,7647.8 456.8,7647.6 458.0,7647.4 459.2,7647.2 460.4,7647.0 461.6,7646.8 462.8,7646.6 464.0,7646.4 465.2,7646.2 466.4,7646.0 467.6,7645.8 468.8,7645.6 470.0,7645.4 471.2,7645.2 472.4,7645.0 473.6,7644.8 474.8,7644.6 476.0,7644.4 477.2,7644.2 478.4,7644.0 479.6,7643.8 480.8,7643.6 482.0,7643.4 483.2,7643.2 484.4,7643.0 485.6,7642.8 486.8,7642.6 488.0,7642.4 489.2,7642.2 490.4,7642.0 491.6,7641.8 492.8,7641.5 494.0,7641.3 495.2,7641.1 496.4,7640.9 497.6,7640.7 498.8,7640.5 500.0,7640.3 501.2,7640.1 502.4,7639.9 503.6,7639.7 504.8,7639.5 506.0,7639.3 507.2,7639.1 508.4,7638.9 509.6,7638.7 510.8,7638.5 512.0,7638.3" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<line x1="145.2" y1="7692.4" x2="145.2" y2="7715.0" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<circle cx="145.2" cy="7692.4" r="4" style="fill:var(--c-success)"/>
<text x="145.2" y="7729" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:700;font-family:var(--font-mono)">13,6</text>
<text x="80.0" y="7729" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">0</text>
<text x="224.0" y="7729" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">30</text>
<text x="368.0" y="7729" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">60</text>
<text x="512.0" y="7729" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">90</text>
<text x="512" y="7745" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">kayıt aralığı τ (dakika)</text>
<text x="40" y="7587" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıp</text>
<rect x="300" y="7594" width="12" height="10" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="318" y="7603" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">A · 25 GB/s: en az %24 kayıp</text>
<rect x="300" y="7611" width="12" height="10" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="318" y="7620" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">B · 250 GB/s (GDS): en az %7,5</text>
<text x="353.6" y="7683.5" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">geri sarma τ/2M (ortak)</text>
<text x="40" y="7763" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">M = 3 sa (16.384 H100&#x27;de ~3 saatte bir arıza) · kayıp = δ/τ + τ/2M</text>
<text x="40" y="7779" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">Nasıl okunur: sık kayıt yazmaya, seyrek kayıt geri sarmaya zaman kaybettirir; en iyi</text>
<text x="40" y="7794" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-style:italic">aralık çukurun dibidir.</text>
<text x="552" y="7599" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">Belirti</text>
<text x="552" y="7617" text-anchor="start" style="fill:var(--c-text);font-size:12px">Küme her kayıtta durur; arızada son</text>
<text x="552" y="7634" text-anchor="start" style="fill:var(--c-text);font-size:12px">kayda geri dönülür.</text>
<text x="552" y="7662" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Neden</text>
<text x="552" y="7680" text-anchor="start" style="fill:var(--c-text);font-size:12px">7,5 TB&#x27;lık kayıt CPU sayfa önbelleğinden</text>
<text x="552" y="7697" text-anchor="start" style="fill:var(--c-text);font-size:12px">25 GB/s ile yazılırsa δ = 307 s sürer ve</text>
<text x="552" y="7714" text-anchor="start" style="fill:var(--c-text);font-size:12px">küme bu sürede bekler.</text>
<text x="552" y="7742" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Çare</text>
<rect x="552.0" y="7752.0" width="116.9" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="610.4" y="7766.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPUDirect Storage</text>
<rect x="552.0" y="7778.0" width="147.1" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="625.5" y="7792.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">asenkron / anlık yazma</text>
<rect x="552.0" y="7804.0" width="98.7" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="601.4" y="7818.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">τ = √(2δM) seç</text>
<line x1="540.0" y1="7585.0" x2="540.0" y2="7824.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="7850.0" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="7870" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: üç çarenin ortak fikri aynı: veriyi az taşı, taşırken hesabı durdurma.</text>
<text x="420.0" y="7917" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Bu farkları sezgiyle görelim: aynı yükü her yoldan geçirelim.</text>
<line x1="420.0" y1="7925.0" x2="420.0" y2="7945.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="7949" width="808" height="333" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="7973" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="7977.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">D</text>
<text x="56" y="7978" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Hız sezgisi: 140 GB&#x27;ı taşımak ne kadar sürer?</text>
<text x="28" y="7999" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Llama 3 70B&#x27;nin BF16 ağırlıkları = 140 GB; aynı yük her hattan (log ölçek).</text>
<rect x="28" y="8011" width="784" height="214" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<line x1="200.0" y1="8025.0" x2="200.0" y2="8149.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="200" y="8163" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 ms</text>
<line x1="292.0" y1="8025.0" x2="292.0" y2="8149.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="8163" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 ms</text>
<line x1="384.0" y1="8025.0" x2="384.0" y2="8149.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="384" y="8163" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 s</text>
<line x1="476.0" y1="8025.0" x2="476.0" y2="8149.0" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="476" y="8163" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 s</text>
<text x="40" y="8043" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3 (tek H100)</text>
<rect x="200" y="8031" width="57.1" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="263.1" y="8043" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">42 ms</text>
<text x="40" y="8067" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="200" y="8055" width="137.3" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="343.3" y="8067" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0,31 s</text>
<text x="40" y="8091" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU yolu)</text>
<rect x="200" y="8079" width="215.3" height="16" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="421.3" y="8091" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,2 s</text>
<text x="40" y="8115" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (GPU başına)</text>
<rect x="200" y="8103" width="225.1" height="16" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="431.1" y="8115" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2,8 s</text>
<text x="40" y="8139" text-anchor="start" style="fill:var(--c-text);font-size:11px">eski depolama (tüm küme)</text>
<rect x="200" y="8127" width="252.8" height="16" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="458.8" y="8139" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5,6 s</text>
<text x="40" y="8191" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">Çubuk uzadıkça yavaşlar: her çubuk aynı 140 GB&#x27;ın o hattan geçme süresidir.</text>
<rect x="544.0" y="8025.0" width="256.0" height="80" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="554" y="8043" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">İnsan ölçeği: HBM&#x27;den 1 saniyede gelen</text>
<text x="554" y="8059" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">veri</text>
<text x="554" y="8075" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink&#x27;ten 7,4 s, PCIe&#x27;den 52 s,</text>
<text x="554" y="8091" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand&#x27;den 67 s&#x27;de gelir.</text>
<rect x="544.0" y="8115.0" width="256.0" height="96" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="554" y="8133" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Decode&#x27;un tavanı: her token bütün</text>
<text x="554" y="8149" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">ağırlıkları bir kez okur</text>
<text x="554" y="8165" text-anchor="start" style="fill:var(--c-text);font-size:11px">8 H100&#x27;de 140 GB ÷</text>
<text x="554" y="8181" text-anchor="start" style="fill:var(--c-text);font-size:11px">(8 × 3,35 TB/s) ≈ 5,2 ms → batch 1&#x27;de en</text>
<text x="554" y="8197" text-anchor="start" style="fill:var(--c-text);font-size:11px">fazla ~190 token/s</text>
<line x1="532.0" y1="8025.0" x2="532.0" y2="8211.0" style="stroke:var(--c-border);stroke-width:1"/>
<rect x="28.0" y="8237.0" width="784.0" height="33" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="40" y="8257" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Akılda kalsın: HBM&#x27;de milisaniyeler, ağda saniyeler. İyi tasarım veriyi hızlı katmanda tutar.</text>
<text x="420.0" y="8302" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px;font-style:italic">Son durak: bütün bunlar gerçek bir kümede nasıl görünüyor?</text>
<line x1="420.0" y1="8310.0" x2="420.0" y2="8330.0" marker-end="url(#bp-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="8334" width="808" height="310" rx="12" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.4"/>
<circle cx="36" cy="8358" r="12" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="36" y="8362.7" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">E</text>
<text x="56" y="8363" text-anchor="start" style="fill:var(--c-text);font-size:15px;font-weight:700">Gerçek küme: Meta&#x27;nın Llama 3 405B eğitimi</text>
<text x="28" y="8384" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Aynı 16.384 H100, Meta&#x27;nın kendi kümesinde (ağ: RoCE); sayılar Llama 3 raporundan.</text>
<rect x="28.0" y="8396" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="36.0" y="8404.0" width="237.3" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="154.7" y="8418.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Hesaplama</text>
<text x="42.0" y="8442" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">16.384 H100</text>
<text x="42.0" y="8457" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TP 8 × PP 16 × DP 128</text>
<text x="42.0" y="8472" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">adım başına 16M token</text>
<text x="42.0" y="8487" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına 400 TFLOP/s</text>
<text x="42.0" y="8502" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BF16 MFU %41</text>
<rect x="293.33333333333337" y="8396" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="301.3" y="8404.0" width="237.3" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="420.0" y="8418.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Ağ</text>
<text x="307.3" y="8442" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">RoCE · 400 Gb/s/GPU</text>
<text x="307.3" y="8457" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod = 3.072 GPU</text>
<text x="307.3" y="8472" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">pod içi tam biseksiyon</text>
<text x="307.3" y="8487" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 pod arası 1:7</text>
<text x="307.3" y="8502" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">(küçük modeller: InfiniBand)</text>
<rect x="558.6666666666667" y="8396" width="253.33333333333334" height="122" rx="8" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<rect x="566.7" y="8404.0" width="237.3" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="685.3" y="8418.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700">Depolama</text>
<text x="572.7" y="8442" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">Tectonic · 240 PB</text>
<text x="572.7" y="8457" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7.500 SSD sunucusu</text>
<text x="572.7" y="8472" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2 TB/s sürekli</text>
<text x="572.7" y="8487" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">7 TB/s tepe</text>
<text x="572.7" y="8502" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıt: 1 MB–4 GB/GPU</text>
<text x="40" y="8538" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">MFU: 8.192 H100&#x27;de %43, 16.384&#x27;te %41, 128K bağlamda %38; ölçek büyüdükçe iletişim artar.</text>
<rect x="28.0" y="8552.0" width="386.0" height="80" rx="5" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="38" y="8570" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">54 günde 466 kesinti; 419&#x27;u beklenmeyen → ortalama ~3 saatte</text>
<text x="38" y="8586" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">bir arıza</text>
<text x="38" y="8602" text-anchor="start" style="fill:var(--c-text);font-size:11px">%78 donanım (GPU %30,1 · HBM3 %17,2); otomasyonla etkin eğitim</text>
<text x="38" y="8618" text-anchor="start" style="fill:var(--c-text);font-size:11px">süresi &gt; %90</text>
<rect x="426.0" y="8552.0" width="386.0" height="80" rx="5" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="436.0" y="8570" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Ders: üç sütun birlikte tasarlanır.</text>
<text x="436.0" y="8587" text-anchor="start" style="fill:var(--c-text);font-size:11px">En yavaş halka (HBM, ağ ya da disk) bütün kümenin hızını</text>
<text x="436.0" y="8603" text-anchor="start" style="fill:var(--c-text);font-size:11px">belirler; MFU onun aynasıdır.</text>
</svg>

*Şekil 1.1: Üç sütunun büyük resmi. A: SM → GPU → DGX düğümü (sistem topolojisiyle) → SuperPOD ölçek merdiveni, SuperPOD ağının yakından incelemesi (port hesabı, trafik yolları, ölçek) ve kümeyi ayakta tutan 14 yapı taşı; B: bir eğitim adımında verinin yolu; C: üç duvar ve hesaplanmış eğrileri; D: 140 GB'lık yükle hız sezgisi; E: Llama 3 405B'nin gerçek sayıları. Kaynaklar: NVIDIA DGX H100 ve DGX SuperPOD belgeleri, The Llama 3 Herd of Models (2024).*

Şekli yukarıdan aşağı okuyun; çoğu kartta çizim solda, açıklaması sağda durur. A bölümü bir SM'den 1.016 GPU'luk SuperPOD'a dört basamakta uzaklaşır ve donanım ile yazılım yapı taşlarının her birinin neden gerektiğini anlatır; B bölümü tek bir eğitim adımında verinin uğradığı sekiz durağı numaralar. C bölümündeki üç eğri bu yazının formüllerinden hesaplandı: roofline, bant genişliği basamakları ve Young-Daly kayıp eğrisi. D bölümü aynı 140 GB'ı her hattan geçirir, E bölümü ise üç sütunun Meta'nın Llama 3 405B kümesinde ölçülmüş hâlini verir. Sonraki üç bölüm sütunları sırayla açıyor.

---

## 2. 1. Sütun — Hesaplama: Streaming Multiprocessor, mikro-mimari ve yürütme dinamikleri

### Silikon seviyesinde GPU anatomisi: SM, warp scheduler ve tensor core

GPU'ların TPU'lardan neden bu kadar farklı davrandığını kavramak için bir restoran mutfağı analojisi kuralım. Bir Google TPU v5p, sadece iki devasa baş aşçı (TensorCore birimleri) tarafından yönetilen lüks bir mutfak gibidir. Her baş aşçı, 128x128 matrisleri lockstep içinde tek hamlede işleyebilen devasa kesme tahtalarına (Matrix Multiply Unit / MXU) sahiptir; küçük siparişler veya asimetrik görevler geldiğinde bu dev tahtalar boş (idle) kalır. Modern bir NVIDIA GPU (örneğin Hopper H100 veya Blackwell B200) ise 132 ila 148 adet bağımsız hat aşçısından oluşan bir ordudur. Bu aşçıların her birine **Streaming Multiprocessor (SM)** denir. Her hat aşçısı kendi küçük tezgâhında tamamen bağımsız siparişler hazırlar ve hepsi ortak bir HBM kilerine bağlıdır.

Fiziksel donanımda her bir SM, **SM subpartition (alt bölüm)** adı verilen 4 özdeş çeyreğe ayrılmıştır. Her çeyrekte şu bileşenler bulunur:
1. **Bir Tensor Core (TC):** Özel sistolik matris çarpım motoru. 1,76 GHz saat hızında çalışan bir H100 SXM5 üzerinde her bir Tensor Core, saat döngüsü (clock cycle) başına yaklaşık 1.024 adet BF16 kayan nokta işlemi (FLOP) yürütür (kabaca 8x8x8 matris çarpımı). 132 SM üzerindeki 528 Tensor Core bir araya geldiğinde tepe 990 TFLOP/s BF16 yoğun (dense) hesaplama gücü üretilir. Blackwell B200'de bu güç döngü başına 2.048 FLOP'a çıkarak yoğun BF16'da 2.250 TFLOP/s seviyesine ulaşır.
2. **Bir Warp Scheduler ve Dağıtım Birimi (Dispatch Unit):** **Warp** adı verilen 32 thread'lik kümelere komut dağıtır. Tek bir SM aynı anda 64 adede kadar resident (yerleşik) warp barındırabilir; bellek erişim gecikmelerini gizlemek (latency hiding) için saat döngüleri arasında warp'lar arasında sıfır ek yüklü (zero-overhead) bağlam değiştirme (context-switching) yapar.
3. **32 FP32 CUDA Core (ALU):** Vektör aktivasyonlarını (GELU, SiLU), normalizasyonları (RMSNorm) ve element-wise toplamaları yürüten genel amaçlı aritmetik birimler. 132 SM'e sahip bir H100 çipinde ~1,98 GHz boost saat hızında çalışan 16.896 adet CUDA çekirdeği bulunur.
4. **16.384 Adet 32-bit Register:** Subpartition başına 64 KiB (SM başına 256 KiB, çip genelinde ~33 MB) ultra hızlı register file sağlar.

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

TPU'ların saf SIMD (Single Instruction, Multiple Data) yürütme modelinin aksine, NVIDIA GPU'ları **SIMT (Single Instruction, Multiple Threads)** paradigmasıyla çalışır. Bir warp içindeki 32 CUDA core belirli bir saat döngüsünde aynı komutu yürütse de, her thread kendi program counter'ına ve bağımsız register alanına sahiptir.

Bu esneklik, donanımda **warp divergence (sapma)** adı verilen tehlikeli bir başarım cezası doğurur. Kernel koşullu bir dallanma içeriyorsa (örneğin MoE expert routing kararını kontrol eden bir `if/else` veya dinamik sequence padding maskesi), dallara ayrılan thread'ler bağımsızca ilerleyemez. Warp scheduler önce `if` yolunu yürütürken diğer execution lane'leri maskeleyip durdurur (stall eder); ardından `else` yolunu yürütürken ilk grubu bekletir:

![NVIDIA GPU'larda Warp Sapması](../../assets/img/gpu/warp-divergence.png)
*Şekil 2.3: Warp divergence donanım zaman çizelgesi. 32 thread'lik bir warp dallara ayrıldığında aktif olmayan execution lane'ler maskelenir ve donanım verimi yarıya düşer.*

32 thread'in 16'sı A dalına, 16'sı B dalına giderse, wall-clock yürütme süresi iki katına çıkar ve o warp için sustained arithmetic throughput tam %50 düşer.

<details>
<summary>Quiz: Warp sapmasının donanım hesaplama verimine etkisi nedir?</summary>

Bir warp içindeki 32 thread'in yarısı `then` dalına, diğer yarısı `else` dalına saptığında donanım bu iki yolu serileştirmek (serialize etmek) zorundadır. İlk aşamada 16 ALU çalışırken 16'sı maskelenip boş bekler; ikinci aşamada durum tersine döner. Toplam saat süresi iki katına çıkar ve o warp için sürdürülen hesaplama verimi tam %50 oranında çöker.
</details>

### Bellek hiyerarşisi ve aritmetik yoğunluk roofline modeli

Hesaplama gücü, işlem birimlerini besleyecek bellek bant genişliği olmadan bir hiçtir. GPU bellek alt sistemi dik bir piramit oluşturur:
- **Register File:** SM başına 256 KiB (çip genelinde ~33 MB). Single-cycle (tek döngüde) erişim gecikmesi.
- **Shared Memory (SMEM) / L1 Cache:** SM başına 256 KiB. FlashAttention algoritmalarında attention tile'larını önbelleğe almak için programatik olarak yönetilen SRAM (sub-10 ns gecikme).
- **Tensor Memory (TMEM):** Blackwell B200 ile tanıtılan SM başına 256 KiB on-chip bellek. B200 Tensor Core birimlerinin ürettiği matris çarpım akümülatörleri klasik register file kapasitesini aştığı için çip üstünde dedicated matris accumulator deposu sağlar.
- **Shared L2 Cache:** Hopper H100'de 50 MB, Blackwell B200'de 126 MB. Çift yönde yaklaşık 5,5 TB/s bant genişliği sunar.
- **High-Bandwidth Memory (HBM3 / HBM3e):** H100'de 80 GB (3,35 TB/s), H200'de 141 GB (4,8 TB/s), B200'de 192 GB (8,0 TB/s).

Hesaplama kapasitesi ile bellek bant genişliği arasındaki denge **Roofline Modeli** ile tanımlanır:

$$\alpha_{\text{crit}} = \frac{C_{\text{peak}}}{W_{\text{mem}}}$$

Şöyle okuyun: $\alpha_{\text{crit}}$ bayt başına FLOP cinsinden kritik aritmetik yoğunluk (arithmetic intensity), $C_{\text{peak}}$ saniyedeki tepe kayan nokta işlem sayısı (peak FLOP/s), $W_{\text{mem}}$ ise saniyedeki bellek bant genişliğidir (bytes/s).

BF16 Tensor Core çalıştıran bir H100 SXM5 için:
$$C_{\text{peak}} = 990 \times 10^{12} \text{ FLOP/s}, \quad W_{\text{mem}} = 3,35 \times 10^{12} \text{ B/s}$$
$$\alpha_{\text{crit}} = \frac{990 \times 10^{12}}{3,35 \times 10^{12}} \approx 295,5 \text{ FLOP/bayt}$$

$X \in \mathbb{R}^{B \times D}$ aktivasyon matrisi ile $W \in \mathbb{R}^{D \times F}$ ağırlık matrisinin 16-bit duyarlıkta çarpımı için:
- **Toplam Matematiksel İşlem:** $2 B D F \text{ FLOP}$
- **Toplam Bellek Transferi:** $2(BD + DF + BF) \text{ bayt}$
- **Operasyonel Yoğunluk (Operational Intensity):**

$$\alpha(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

Batch boyutu $B$ çok küçük olduğunda ($B \to 1$, örneğin tek tokenli ardışık çıkarım / autoregressive decoding), ağırlık matrisi bellek trafiğini domine eder ($DF \gg B(D+F)$):
$$\alpha(1) \approx \frac{D F}{D F} \cdot B \approx B = 1 \text{ FLOP/bayt} \ll 295,5 \text{ FLOP/bayt}$$

Bu rejimde GPU aşırı derecede **memory-bandwidth bound** (bellek bant genişliği kısıtlı) çalışır ve tepe hesaplama kapasitesinin %1'inden azını kullanabilir. **Compute-bound** (hesaplama kısıtlı) sınırına geçmek için gereken batch boyutu şartı şudur:

$$B \ge \frac{\alpha_{\text{crit}} \cdot D F}{D F - \alpha_{\text{crit}}(D + F)} \approx \alpha_{\text{crit}} \approx 295 \text{ token}$$

Hem Hopper H100 hem de Blackwell B200'de roofline eğrisinin dirsek noktasını (roofline knee) aşıp compute-bound bölgesine geçebilmek için SM başına en az 280 ila 300 tokenlik bir batch boyutu gerekir.

<details>
<summary>Quiz: Otoregresif token üretiminde MFU neden %2'nin altında kalır?</summary>

Single-token decoding evresinde ($B=1$) GPU tek bir vektörü işlemek için tüm model ağırlıklarını HBM'den register file'a çekmek zorundadır. Aritmetik yoğunluk $\alpha(1) \approx 1 \text{ FLOP/bayt}$ seviyesindedir. H100 Tensor Core birimlerini doyurmak için $\alpha_{\text{crit}} \approx 295,5 \text{ FLOP/bayt}$ gerektiğinden, donanım saat döngülerinin %99'undan fazlasını bellek bekleyerek (memory stall) harcar ve MFU %1-2 bandına sıkışır.
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
1. **E4M3 (1 işaret, 4 üs, 3 mantis biti):** Dinamik aralığın kararlı olduğu ileri geçiş (forward pass) aktivasyonları ve ağırlıklar için daha yüksek sayısal hassasiyet sunar.
2. **E5M2 (1 işaret, 5 üs, 2 mantis biti):** FP16'nın dinamik aralığını koruyarak geri yayılım (backward pass) gradyanlarında taşma veya sıfıra yuvarlanmayı (underflow) engeller.

| Sayısal Format | İşaret Biti | Üs Biti | Mantis Biti | Dinamik Aralık ($E_{\text{min}} \dots E_{\text{max}}$) | LLM Ölçeklemesindeki Temel Rolü |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IEEE FP16** | 1 | 5 | 10 | $\approx 6 \times 10^{-5} \dots 65.504$ | Geleneksel çıkarım ve temel işlem |
| **Google BF16** | 1 | 8 | 7 | $\approx 1 \times 10^{-38} \dots 3 \times 10^{38}$ | Kararlı temel model ön eğitimi |
| **FP8 (E4M3)** | 1 | 4 | 3 | $\approx \pm 448$ | Forward pass aktivasyonları ve ağırlıklar |
| **FP8 (E5M2)** | 1 | 5 | 2 | $\approx \pm 57.344$ | Backward pass gradyan iletimi |
| **NVFP4 (E2M1)** | 1 | 2 | 1 | Blok ölçekli | Blackwell ultra yoğun GEMM hızlandırması |

Blackwell B200'de ise **NVFP4 (4-bit Microscaling)** mimarisine geçilmiştir. 16 elemanlı vektör blokları üzerinde paylaşılan 8-bit veya 16-bit ölçekleme katsayılarıyla desteklenen 4-bit sayılar, çip başına 4.500 TFLOP/s dense compute üreterek devasa GEMM matrislerinin doğrudan TMEM ve L2 cache sınırlarında kalmasını sağlar.

---

## 3. 2. Sütun — Ağ: Ara bağlantı omurgaları, kolektifler ve çok boyutlu paralellik

### Intra-node ağ mimarisi: NVLink ve NVSwitch fabric'leri

Tek bir çipin sınırlarını aşarken standart PCIe veri yolları (PCIe Gen 5'te tek yönde yalnızca 64 GB/s) aşılması imkânsız bir darboğaza dönüşür. NVIDIA bu engeli **NVLink** ve **NVSwitch** ile aşar.

Standart bir DGX H100 veya HGX H100 kasasında 8 GPU, 4 adet özel NVSwitch 4 ASIC'i ile eşleştirilmiştir. Her H100 çipi 18 adet NVLink 4 portuna sahiptir. Her port çift yönde 50 GB/s (tek yönde 25 GB/s) aktarım yaparak çip başına şu egress hızını üretir:
$$W_{\text{GPU egress}} = 18 \times 25 \text{ GB/s} = 450 \text{ GB/s tek yönlü } (900 \text{ GB/s çift yönlü})$$

4 adet merkezi NVSwitch, non-blocking (engellemesiz) bir crossbar fabric oluşturur. Kasa içindeki toplam switching kapasitesi şöyledir:
$$4 \text{ switch} \times 1,6 \text{ TB/s} = 6,4 \text{ TB/s anahtarlama bant genişliği}$$

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

Bu 8 GPU'lu anakartların fiziksel yerleşimi ve NVSwitch 4 ASIC'inin iç yapısı şöyledir:

![NVIDIA HGX Düğüm İçi NVLink Topolojisi](../../assets/img/gpu/nvlink-nodes.png)
*Şekil 3.1: GPU'lar ile 4 yerleşik NVSwitch arasındaki yüksek yoğunluklu NVLink hatlarını yönlendiren HGX 8 GPU taban kartı mimarisi.*

![NVIDIA NVLink 4 Switch Crossbar Mimarisi](../../assets/img/gpu/nvlink4.png)
*Şekil 3.2: Hot Chips 34'te sunulan, donanımsal in-network reduction yeteneğine sahip ve çift yönde 3,2 TB/s yönlendirebilen 64 portlu NVLink 4 switch ASIC'inin iç şeması.*

Blackwell nesliyle birlikte NVIDIA, **GB200 NVL72** mimarisiyle bu denklemi kökten değiştirdi. NVLink domain'ini 8 GPU'lu tek bir kasayla sınırlamak yerine, sıvı soğutmalı rack'ler 72 adet B200 GPU'yu 18 adet NVSwitch 5 ASIC'i üzerinden bakır arka panelde (copper backplane) birleştirir:

![GB200 NVL72 Kabin Mimarisi](../../assets/img/gpu/b200-node.png)
*Şekil 3.3: 72 GPU ve 18 NVLink switch'i tek bir 900 GB/s NVLink domain'inde birleştiren GB200 NVL72 rack mimarisi.*

NVL72'de 72 GPU'nun tümü, herhangi bir çipin diğerinin HBM belleğine 900 GB/s hızla erişebildiği tek bir NVLink etki alanını paylaşır ve tek node sınırını 9 kat genişletir.

<details>
<summary>Quiz: Bir H100 SXM5 GPU'da kaç NVLink portu bulunur ve toplam bant genişliği nedir?</summary>

H100 SXM5 çipinde 18 adet NVLink 4 portu bulunur. Her port tek yönde 25 GB/s hızla çalışarak GPU başına tek yönde 450 GB/s, çift yönde 900 GB/s toplam bant genişliği sağlar.
</details>

### Düğümler ötesi ölçekleme (Scale-out): SuperPod ve rail-optimized fat-tree topolojisi

Yüz milyarlarca parametreli modelleri eğitmek için cluster'ın binlerce GPU'ya ölçeklenmesi gerekir. Bu scale-out katmanı **InfiniBand** (veya RoCE v2 Ethernet) tarafından yönetilir.

1.024 GPU'lu bir DGX H100 SuperPod cluster'ında:
- Cluster, her biri 32 DGX node (256 GPU) içeren **Scalable Unit (SU)** bloklarına ayrılır.
- Her node'da 8 adet ConnectX-7 400 Gbps (NDR) NIC bulunur; bu da node başına $8 \times 50 \text{ GB/s} = 400 \text{ GB/s}$ (GPU başına 50 GB/s) scale-out network egress sağlar.
- Her SU içinde 8 Leaf switch bulunur. 4 SU genelinde 16 merkezi Spine switch, ağı 51,2 TB/s aggregate switching capacity'ye sahip non-blocking 1:1 **Fat-Tree topolojisinde** birleştirir.

Kritik olarak modern cluster'lar **Rail-Optimized kablolama** kullanır. Bu topolojide her node'daki $i$. GPU rank'ı doğrudan $i$. Leaf switch'e (rail switch) bağlanır. Aynı indeksli GPU'lar arasında Data Parallelism (DP) veya Pipeline Parallelism (PP) kolektifleri yürütülürken paketler spine katmanına çıkmadan doğrudan kendi rail switch'i içinde kalır; böylece spine hop'ları ortadan kalkar ve fabric congestion önlenir:

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
*Şekil 3.5: 128 DGX node'u, rail-optimized InfiniBand switch'leri ve yönetim rack'lerinden oluşan 1.024 GPU'luk fiziksel H100 SuperPod kurulumu.*

![GB200 NVL72 SuperPod Mimarisi](../../assets/img/gpu/gb200-superpod.png)
*Şekil 3.6: 576 B200 GPU'yu Quantum-X800 800 Gbps InfiniBand switch'leriyle bağlayan Blackwell GB200 NVL72 çok kabinli ölçekleme topolojisi.*

### Kolektif iletişim (Collectives): Ring algoritmalarından donanımsal SHARP'a

Dağıtık eğitim, NCCL (NVIDIA Collective Communications Library) içinde uygulanan temel kolektif iletişim (collective communication) primitiflerine dayanır:
1. **AllGather:** Her rank $B/N$ boyutunda bir tensör parçası (slice) tutar; tüm parçalar toplanarak her GPU'da $B$ boyutunda tam tensör oluşturulur.
2. **ReduceScatter:** Her rank $B$ boyutunda girdi tensörü tutar; elemanlar rank'lar arasında toplanır (element-wise sum) ve $i$. rank yalnızca $B/N$ boyutundaki $i$. parçayı saklar.
3. **AllReduce:** ReduceScatter ve ardından AllGather işlemine eşdeğerdir. Tüm rank'lar arasında element-wise toplama yapar ve $B$ boyutundaki nihai tensörü $N$ GPU'nun tamamına replike eder.
4. **AllToAll:** Her rank diğer tüm rank'lara farklı bir tensör slice'ı gönderir; Mixture of Experts (MoE) token routing mekanizmasının temelini oluşturur.

AllGather kolektifinin adım adım akışı aşağıdaki animasyonda gösterilmektedir:

![AllGather Kolektif İletişim Adımı](../../assets/img/gpu/all-gather.gif)
*Şekil 3.7: Dört GPU arasında bir AllGather kolektifinin adım adım iletişim modeli.*

Ağ çıkış bant genişliği (network egress bandwidth) $W$ olan $N$ cihazlık klasik bir **Ring AllGather** algoritmasında:
- Her cihaz tensörünü $B/N$ boyutunda $N$ parçaya (chunk) böler.
- Her adımda her rank bir parçayı komşusuna aktarırken diğerinden yenisini alır.
- Ring dolaşımı $N-1$ adımda tamamlanır:

$$T_{\text{AllGather, Ring}} = (N - 1) \cdot \frac{B / N}{W} = \frac{N - 1}{N} \cdot \frac{B}{W} \approx \frac{B}{W}$$

Bir AllReduce işlemi hem ReduceScatter hem AllGather gerektirdiğinden, klasik ring AllReduce $2 \times \frac{N-1}{N} B$ bayt aktarır ve şu süreyi alır:
$$T_{\text{AllReduce, Ring}} \approx 2 \cdot \frac{B}{W}$$

Bu $2\times$ maliyetini ortadan kaldırmak için modern InfiniBand switch'leri **Mellanox SHARP (Scalable Hierarchical Aggregation and Reduction Protocol)** teknolojisini kullanır. SHARP, floating-point reduction mantığını doğrudan Quantum switch ASIC'lerine gömer:

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

Frontier modelleri eğitmek dört temel paralellik boyutunun harmanlanmasını gerektirir: **Tensor Parallelism (TP)**, **Pipeline Parallelism (PP)**, **Expert Parallelism (EP)** ve **Data Parallelism (DP / FSDP)**.

#### Tensor Parallelism (TP) ve 8 GPU NVLink Tavanı

Megatron-LM tarzı tensor parallelism'de Transformer MLP bloğu $Y$ parçaya bölünür: $W_{\text{in}}$ column-wise ($[D, F/Y]$), $W_{\text{out}}$ row-wise ($[F/Y, D]$) paylaştırılır.
- **İleri Hesaplama Süresi (Forward Math Time):** $T_{\text{math}} = \frac{4 B D F}{Y \cdot C_{\text{peak}}}$
- **İletişim Süresi (Communication Time):** Aktivasyonlar üzerinde bir AllReduce: $T_{\text{comms}} = \frac{4 B D}{W_{\text{collective}}}$

Hesaplamanın iletişimden hızlı kalması ($T_{\text{math}} \ge T_{\text{comms}}$) için:

$$Y \le \frac{F \cdot W_{\text{collective}}}{C_{\text{peak}}} = \frac{F}{\alpha_{\text{crit}}}$$

Intra-node NVLink kullanan bir H100 SXM5 için ($W_{\text{collective}} = 450 \text{ GB/s}$, $C_{\text{peak}} = 990 \text{ TFLOP/s}$):
$$\frac{C_{\text{peak}}}{W_{\text{collective}}} = \frac{990 \times 10^{12}}{450 \times 10^9} = 2.200$$

LLaMA-3 70B ($F = 28.672$) için:
$$Y \le \frac{28.672}{2.200} \approx 13,03 \implies Y = 8 \text{ GPU}$$

Bu matematiksel sınır, **Tensor Parallelism'in neden kesinlikle 8 GPU'luk tek bir NVLink kasasına hapsedildiğini** kanıtlar. TP'yi InfiniBand üzerinden cross-node olarak 16 GPU'ya çıkarmaya kalktığınızda ($W_{\text{collective}} = 400 \text{ GB/s}$ node egress, yani GPU başına 50 GB/s), $C/W$ oranı 19.800'e fırlar, izin verilen $Y$ değeri 1,5'in altına düşer ve donanım verimi çöker.

#### Pipeline Parallelism (PP) ve Pipeline Kabarcıkları (Pipeline Bubbles)

Pipeline parallelism model katmanlarını $P$ pipeline stage'ine böler. Komşu stage'ler microbatch başına yalnızca $2BD / N_{\text{MB}}$ boyutunda boundary aktivasyon tensörleri takas eder (basit peer-to-peer / P2P hatlar üzerinden), bu nedenle network veri hacmi önemsizdir.

Ancak pipelining, stage'lerin microbatch'leri beklemesinden kaynaklanan **Pipeline Bubble** (kabarcık) cezası yaratır:

$$\text{Bubble Fraction} = \frac{P - 1}{P - 1 + M}$$

Şöyle okuyun: $P$ pipeline stage sayısı, $M$ ise global batch içindeki microbatch sayısıdır.

![1F1B Çizelgesinde Boru Hattı Kabarcıkları](../../assets/img/gpu/pipeline-bubble.png)
*Şekil 3.16: 2 stage ve 2 microbatch'li 1F1B pipeline çizelgesi. Beyaz boşluklar idle pipeline bubble döngülerini, 'G' ise gradyan senkronizasyonunu temsil eder.*

![Aktivasyon Aktarım Gecikmesinin Kabarcığa Etkisi](../../assets/img/gpu/pipeline-transfer.png)
*Şekil 3.17: Stage'ler arasındaki aktivasyon iletim gecikmesi pipeline yürütmesini kaydırarak bubble oranını artırır.*

#### Data Parallelism (DP & FSDP) ve Kritik Batch Boyutu

Data Parallelism'de (ZeRO-3 / FSDP dahil) parametreler ve gradyanlar $X$ cihaza dağıtılır:
$$T_{\text{math}} = \frac{6 B D F}{X \cdot C_{\text{peak}}}, \quad T_{\text{comms}} = \frac{2 \cdot 2 \cdot DF \cdot (X - 1)}{X \cdot W_{\text{collective}}}$$

$T_{\text{math}} \ge T_{\text{comms}}$ şartından **Critical Local Batch Size (Kritik Yerel Batch Boyutu)** çıkar:

$$\frac{B}{X} \ge \frac{2}{3} \cdot \frac{C_{\text{peak}}}{W_{\text{collective}}} \approx 2.475 \text{ token / GPU}$$

GPU başına düşen batch boyutu yaklaşık 2.500 tokenin altına indiğinde, scale-out gradient senkronizasyonu forward-backward hesaplamasından daha uzun sürer ve GPU'lar idle bekler.

$E$ toplam uzman ve token başına $k$ aktif uzmana sahip Mixture of Experts (MoE) modellerinde bu eşik dramatik biçimde büyür:

$$\left(\frac{B}{X}\right)_{\text{crit, MoE}} \ge \frac{E}{k} \cdot \frac{C_{\text{peak}}}{W_{\text{collective}}}$$

$E = 128$ ve $k = 4$ olan bir modelde kritik batch boyutu $32\times$ katlanarak GPU başına 79.000 tokene fırlar; bu patlama ancak Expert Parallelism (EP) ile çözülebilir.

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
<text x="30" y="68" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:700">Tensor Parallelism (TP)</text>
<text x="30" y="84" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: Tek Node (NVLink)</text>
<rect x="30" y="92" width="228" height="19" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="144.0" y="104.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">AllGather + ReduceScatter</text>
<text x="30" y="128" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: Y ≤ F / 2475 (LLaMA: Y ≤ 8)</text>
<text x="30" y="146" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Cross-node TP hızla comms-bound olur</text>
<text x="30" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="288" y="48" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="302" y="68" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:700">Pipeline Parallelism (PP)</text>
<text x="302" y="84" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: Cross-Node Katmanlar</text>
<rect x="302" y="92" width="228" height="19" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="416.0" y="104.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">P2P Activation Passing</text>
<text x="302" y="128" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: Bubble = (P-1) / (P-1+M)</text>
<text x="302" y="146" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Minimum hat maliyeti, karmaşık takvim</text>
<text x="302" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="16" y="192" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="30" y="212" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:700">Expert Parallelism (EP)</text>
<text x="30" y="228" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: MoE (1-8 Node)</text>
<rect x="30" y="236" width="228" height="19" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="144.0" y="248.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">Ragged AllToAll Routing</text>
<text x="30" y="272" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: 1-2 node (küçük F); multi-node (büyük)</text>
<text x="30" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">MoE parametre patlamasını çözer</text>
<text x="30" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="288" y="192" width="256" height="134" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="302" y="212" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">Data Parallelism (DP / FSDP)</text>
<text x="302" y="228" text-anchor="start" style="fill:var(--c-text);font-size:10.5px;font-weight:600">Etki Alanı: Tüm Cluster (ZeRO-1 / 3)</text>
<rect x="302" y="236" width="228" height="19" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="416.0" y="248.8" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">AllReduce / ReduceScatter</text>
<text x="302" y="272" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Kural: B_yerel ≥ 2.500 token / GPU</text>
<text x="302" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">MoE&#x27;de kritik batch boyutu E/k kat büyür</text>
<text x="302" y="308" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">Roofline: Compute vs Comms</text>
<rect x="16" y="336" width="528" height="92" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="356" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Modern Üretim Modellerinin Paralellik Reçeteleri:</text>
<text x="28" y="374" text-anchor="start" style="fill:var(--c-accent-2);font-size:10.5px;font-family:var(--font-mono)">DeepSeek-V3 (2048x H800): 64-way EP (8 node) + 16-way PP + 2-way DP (ZeRO-1)</text>
<text x="28" y="392" text-anchor="start" style="fill:var(--c-accent);font-size:10.5px;font-family:var(--font-mono)">LLaMA-3 405B (16k H100):  8-way TP (1 node) + 16-way PP + 128-way DP (ZeRO-1)</text>
<text x="28" y="410" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">Temel kural: TP intra-node kalır, PP ve EP node sınırlarını aşar, DP ise cluster genelinde ölçeklenir.</text>
</svg>

Frontier modeller bu tasarım uzayını şu üretim reçeteleriyle fetheder:
- **LLaMA-3 405B (16.384 H100):** 8-way intra-node TP $\times$ 16-way PP $\times$ 128-way DP (ZeRO-1).
- **DeepSeek-V3 (2.048 H800):** 64-way EP (InfiniBand üzerinde 8 node'a yayılan) $\times$ 16-way PP $\times$ 2-way DP (ZeRO-1), GPU başına 30.720 token.

Megatron-LM 3D paralellik uzayı aşağıda özetlenmiştir:

![Megatron-LM 3D Paralellik Ayar Uzayı](../../assets/img/gpu/megatron-hparams.png)
*Şekil 3.18: Megatron-LM araştırmasında farklı cluster boyutlarında TP, PP ve DP parametre verimlilik alanları.*

### Frontier engineering: DeepSeek-V3 DualPipe ve iletişim-hesaplama çakışması (communication-computation overlap)

64-way Expert Parallelism (EP) node'lar arasına yayıldığında, AllToAll token routing scale-out InfiniBand linklerini tıkama tehlikesi yaratır. DeepSeek-V3 bu darboğazı **DualPipe** ile çözmüştür.

DualPipe, ileri ve geri yönlerde (forward ve backward) aynı anda iki bağımsız pipeline schedule yürütür. Katmanları ince taneli chunk'lara bölerek, bir chunk'ın cross-node AllToAll token transferini arka planda yürütürken GPU Tensor Core birimlerinde diğer chunk'ın yoğun attention ve GEMM hesaplamalarını çalıştırır. İletişimi hesaplamayla kusursuz biçimde çakıştırarak (communication-computation overlap), DeepSeek ağ gecikmesinin %95'ten fazlasını gizler (latency hiding) ve kısıtlı H800 cluster'larında dahi tepe MFU'ya ulaşır.

---

## 4. 3. Sütun — Depolama (Storage): Foundation model veri setleri, checkpointing ve yüksek verimli I/O

### Foundation model ön eğitiminde depolama krizi

Hesaplama ve ağ mimarisi ön planda olsa da, foundation model eğitim cluster'ları depolama I/O yetersizlikleri yüzünden binlerce saat boyunca felç olur. Modern ön eğitimde depolama alt sistemi iki devasa zorlukla karşı karşıyadır:

1. **Sürekli Veri Besleme (Continuous Dataset Ingestion):** LLaMA 3 gibi bir frontier modeli eğitmek 15 trilyondan fazla yapılandırılmamış metin, kod ve multimodal tokenin stream edilmesini gerektirir. 16.384 GPU'luk bir cluster'da dataloader alt sistemi, PyTorch worker iş parçacıklarını düşürmeden veya bekletmeden saniyede 2 ila 4 Terabayt kesintisiz okuma throughput'unu korumalıdır.
2. **Checkpointing Krizi:** Donanım arızalarına karşı model durumları (states) periyodik olarak kalıcı depolamaya snapshot olarak yazılmalıdır. 405 milyar parametreli bir modelin eğitim sırasındaki tam bellek faturası şöyledir:

$$S_{\text{state}} = S_{\text{weights}} + S_{\text{gradients}} + S_{\text{AdamW}} = 2M + 2M + 12M = 16M \text{ bayt}$$

$M = 405 \times 10^9$ parametre için:
- **BF16 Model Ağırlıkları:** $405 \times 10^9 \times 2 \text{ bayt} = 810 \text{ GB}$
- **BF16 Gradyanlar:** $405 \times 10^9 \times 2 \text{ bayt} = 810 \text{ GB}$
- **FP32 AdamW Optimizer State:** Master weights (4 bayt) + Momentum (4 bayt) + Variance (4 bayt) = $405 \times 10^9 \times 12 \text{ bayt} = 4.860 \text{ GB} = 4,86 \text{ TB}$

Bunların toplamı $6.480 \text{ GB} \approx 6,48 \text{ TB}$ eder. Aktivasyon buffer'ları, rastgele sayı üreteci (RNG) state'leri ve optimizer step metadata'sı eklendiğinde tek bir checkpoint snapshot boyutu **~7,5 Terabayta** ulaşır.

Eğer küme bu 7,5 TB snapshot'ı geleneksel 25 GB/s'lik bir paylaşımlı depolama fabric'ine senkron olarak yazmaya kalkarsa, **16.384 GPU'luk süper bilgisayar her kayıtta 5 dakikadan fazla ($307 \text{ saniye}$) tamamen donar (freeze)**. Haftalar süren bir eğitim maratonunda bu checkpoint duraklamaları milyonlarca dolarlık GPU döngüsünü çöpe atar.

### Geleneksel I/O darboğazı: Host CPU page cache ve bounce buffer'lar

Geleneksel GPU cluster'larındaki depolama duraklamalarının temel nedeni çok sekmeli (multi-hop) geleneksel I/O mimarisidir:
1. **NVMe Okuma/Yazma:** Veri NVMe SSD'den PCIe üzerinden host CPU root complex birimine gelir.
2. **Kernel Page Cache Kopyası:** Linux depolama yığını veriyi host sistemin DDR5 RAM'indeki kernel page cache'e kopyalar.
3. **User Space Bounce Buffer:** PyTorch veya checkpoint yöneticisi veriyi kernel space'ten user space host RAM'e kopyalar.
4. **Host-to-Device Aktarımı:** Host CPU, PCIe Gen 5 üzerinden veriyi GPU High Bandwidth Memory (HBM) içine basmak için ikinci bir DMA transferi başlatır.

Bu geleneksel yol, her baytın PCIe veri yolunu iki kez geçmesini ve iki ayrı CPU bellek kopyalamasına (`memcpy`) uğramasını zorunlu kılar. Çift soketli modern bir host CPU %100 çekirdek yüküyle 50-100 GB/s hızda bellek veri yolunu tüketir ve depolama I/O'sunu felç edici bir darboğaza dönüştürür.

### GPUDirect Storage (GDS) ve doğrudan DMA pipeline'ları

Modern yapay zekâ süper bilgisayarları bu darboğazı **GPUDirect Storage (GDS)** ile aşar:

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
<text x="28" y="364" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Checkpointing Zorluğu: 405B modelinde ~7,5 TB state periyodik kaydedilir.</text>
<text x="28" y="382" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GDS ile depolama, sistem RAM&#x27;ini baypas ederek doğrudan GPU HBM&#x27;e akar,</text>
<text x="28" y="398" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">CPU bounce gecikmesi elenir ve eğitim duraklamaları önlenir.</text>
</svg>

GDS, `cuFile` API'si ve `nvidia-fs.ko` çekirdek modülü üzerinden çalışır. Yerel NVMe sürücüler (veya InfiniBand üzerinden NVMe-oF) ile GPU HBM belleği arasında PCIe switch fabric üzerinde doğrudan bir DMA (Direct Memory Access) kanalı kurar; host CPU'yu, user space bounce buffer'larını ve Linux kernel page cache'ini tamamen baypas eder.

CPU bellek kopyalamaları ortadan kalktığında:
- Depolama throughput'u PCIe Gen 5 hat hızına (**denetleyici başına 64+ GB/s**) ulaşır.
- Uçtan uca I/O gecikmesi **%80** azalır.
- Host CPU yükü **%100'den neredeyse %0'a** düşerek CPU çekirdeklerini token ön işleme ve orkestrasyona bırakır.

<details>
<summary>Quiz: GPUDirect Storage (GDS) checkpointing sırasında CPU yükünü neden neredeyse sıfıra indirir?</summary>

Geleneksel I/O, host CPU'nun kernel page cache buffer'ları ile user space belleği arasında gigabaytlarca veriyi taşımak için `memcpy` komutları yürütmesini gerektirir. GDS ise PCIe DMA motorlarını doğrudan programlayarak veriyi NVMe depolama denetleyicileri ile GPU HBM arasında, host CPU çekirdeklerine ve sistem DDR5 RAM'ine hiç uğramadan stream eder.
</details>

### Exabyte ölçeğinde dağıtık dosya sistemleri: Meta Tectonic vaka analizi

Binlerce node'a yayılan bir üretim cluster'ında yerel NVMe sürücüler global checkpoint'leri barındıramaz. Bunun yerine cluster'lar exabyte ölçeğindeki dağıtık dosya sistemlerine bağlanır.

Meta'nın resmi LLaMA 3 altyapı raporunda 16.384 GPU'luk cluster, Meta'nın dağıtık dosya sistemi **Tectonic** tarafından desteklenmiştir:
- **Ölçek:** 240 Petabayt depolama kapasitesi sunan 7.500 adet dedicated SSD depolama sunucusu.
- **Sürekli Okuma Throughput'u:** 2.048 compute node'undaki dataloader worker süreçlerini beslemek için kesintisiz **2 TB/s** okuma bant genişliği.
- **Burst Checkpoint Throughput'u:** **7 TB/s anlık tepe (burst) yazma bant genişliği**; bu sayede 7,5 TB'lık checkpoint snapshot'ı tüm rank'lar tarafından asenkron copy-on-write buffer'ları kullanılarak yaklaşık **bir saniyede** kalıcı depolamaya kaydedilebilir.
- **Güvenilirlik:** Sürücü ve rack düzeyindeki depolama sunucusu arızalarında model state'ini korumak için Reed-Solomon erasure coding ($9+3$ parça) kullanır.

### Yüksek verimli dataloading: Token binary indexing ve zero-copy mmap

Binlerce GPU'ya kesintisiz token stream etmek özel dataloader mühendisliği gerektirir:
- **Standart PyTorch DataLoader Neden Yetersiz Kalır:** Ham JSONL veya metin dosyaları yüklemek, Python worker'larının sürekli string parse etmesine, metin tokenize etmesine ve dinamik Python nesneleri allocate etmesine yol açar; bu da Python Global Interpreter Lock (GIL) kilitlenmelerine ve worker thread'lerinin duraklamasına neden olur.
- **Bitişik İkili Düzen (Contiguous Binary Layout - .bin / .idx):** Üretim framework'leri (Megatron-LM ve Lit-GPT gibi) veri setlerini diskte bitişik `uint16` veya `int32` token ID array'leri halinde tek bir ikili dosyada (`.bin`) önceden tokenize eder; yanında dokümanların bayt offset'lerini eşleyen bir indeks dosyası (`.idx`) tutulur.
- **Zero-Copy Memory Mapping (`mmap`):** Worker süreçleri POSIX `mmap()` çağrısıyla ikili veri setini doğrudan process sanal adres uzayına eşler. İşletim sistemi token bloklarını ihtiyaç duyuldukça user space buffer allocation'ı yapmadan şeffaf bir biçimde belleğe sayfalar (demand paging).
- **NUMA-Aware Prefetching:** Compute node'larında iki CPU soketi bulunur ve her soket dedicated bir NUMA node'una ve 4 GPU'ya fiziksel olarak bağlıdır (Şekil 1.1'deki node topolojisinde CPU 0 GPU 0–3'ü, CPU 1 GPU 4–7'yi besler). Worker'lar doğrudan hedef GPU'nun PCIe root complex birimine bağlı CPU soketine kilitlenir (NUMA affinity), böylece NUMA interconnect'lerini geçmeden asenkron CUDA stream'leri (`pin_memory=True`) üzerinden token'ları doğrudan HBM'e stream eder.

### Cluster MTBF, fault tolerance ve Young-Daly checkpoint formülü

16.384 GPU'luk (2.048 node) bir cluster'da donanım arızaları nadir istisnalar değil, sürekli operasyonel koşullardır:
- Tek bir DGX node'unun Arızalar Arası Ortalama Süresi (Mean Time Between Failures - MTBF) 3 yıl (~26.000 saat) olsa bile, toplam küme MTBF'si şöyledir:

$$M_{\text{küme}} = \frac{26.000 \text{ saat}}{2.048 \text{ node}} \approx 12,7 \text{ saat}$$

Pratikte sessiz veri bozulmaları (Silent Data Corruption - SDC), HBM3 ECC bellek hataları, optik alıcı-verici (transceiver) düşmeleri ve ağ zaman aşımları eklendiğinde, 16k GPU'luk kümede her **2 ila 3 saatte bir** fatal bir kesinti yaşanır.

Bir node çöktüğünde tüm cluster son kaydedilen checkpoint snapshot'ından yeniden başlamak zorundadır. Bu durum temel bir matematiksel trade-off doğurur:
- Çok sık checkpoint alırsanız zamanınızı diske veri yazarak harcarsınız ($\delta$).
- Çok seyrek checkpoint alırsanız bir çöküşte saatlerce geriye düşüp kayıp adımları baştan hesaplamakla vakit kaybedersiniz.

Matematiksel olarak optimal checkpoint aralığı klasik **Young-Daly Formülü** ile yönetilir:

$$\tau_{\text{opt}} = \sqrt{2 \cdot \delta \cdot M}$$

Burada:
- $\delta$ bir checkpoint snapshot'ını kaydetmek için gereken süredir (saniye).
- $M$ kümenin Arızalar Arası Ortalama Süresidir (cluster MTBF, saniye).
- $\tau_{\text{opt}}$ ardışık iki checkpoint arasındaki optimal süredir.

LLaMA-3 405B için bu trade-off'u 16k GPU üzerinde inceleyelim ($M = 3 \text{ saat} = 10.800 \text{ saniye}$):
- **Senaryo A: Geleneksel Depolama Fabric'i (25 GB/s küme toplam yazma hızı):**
  $$\delta = \frac{7,5 \times 1024 \text{ GB}}{25 \text{ GB/s}} \approx 307,2 \text{ s} \approx 5,12 \text{ dakika}$$
  $$\tau_{\text{opt}} = \sqrt{2 \cdot 307,2 \cdot 10.800} \approx \sqrt{6.635.520} \approx 2.576 \text{ s} \approx 42,9 \text{ dakika}$$
  Her 43 dakikada bir checkpoint almak $\frac{5,12}{42,9} \approx \%11,9$ doğrudan checkpoint yükü ve $\frac{\tau}{2M} \approx \%11,9$ beklenen yeniden hesaplama kaybı yaratır; **yani süper bilgisayarın toplam hesaplama kapasitesinin neredeyse %24'ü çöpe gider!**
- **Senaryo B: GPUDirect Storage & Tectonic Burst (250 GB/s yazma fabric'i):**
  $$\delta = \frac{7,5 \times 1024 \text{ GB}}{250 \text{ GB/s}} \approx 30,72 \text{ s} \approx 0,51 \text{ dakika}$$
  $$\tau_{\text{opt}} = \sqrt{2 \cdot 30,72 \cdot 10.800} \approx \sqrt{663.552} \approx 814,6 \text{ s} \approx 13,6 \text{ dakika}$$
  Checkpoint yükü $\frac{0,51}{13,6} \approx \%3,8$ ve beklenen yeniden hesaplama kaybı $\frac{\tau}{2M} \approx \%3,8$ olur; toplam hesaplama kaybı **%24'ten yaklaşık %7,5'e** iner ve milyonlarca dolarlık verimli eğitim süresi kurtarılır. Yazma 10 kat hızlandığında kayıp $\sqrt{10} \approx 3,2$ kat azalır, çünkü $\tau_{\text{opt}}$ noktasında her iki kayıp terimi de $\sqrt{\delta}$ ile ölçeklenir.

<details>
<summary>Quiz: MTBF = 4 saat ve checkpoint süresi = 2 dakika olan bir kümede optimal checkpoint aralığı nedir?</summary>

Young-Daly formülü uygulandığında:
$\tau_{\text{opt}} = \sqrt{2 \cdot \delta \cdot M} = \sqrt{2 \cdot 2 \text{ dk} \cdot 240 \text{ dk}} = \sqrt{960} \approx 30,98 \text{ dakika}$.
Yaklaşık her 31 dakikada bir checkpoint almak toplam kaybedilen eğitim süresini minimize eder.
</details>

---

## 5. Sentez: Training ve inference arasındaki operasyonel dengeler

### İki mühendislik merceği: Training ve inference karşılaştırması

Donanım kararları model eğitimi ile yüksek hacimli çıkarım (inference) sunumu arasında taban tabana zıttır:

| Mimari Boyut | Eğitim (Pre-training / SFT) | Çıkarım (Prefill & Decode) |
| :--- | :--- | :--- |
| **Birincil Bellek Ayak İzi** | Model Ağırlıkları ($2B$) + Gradyanlar ($2B$) + AdamW Durumları ($12-16B$) + Aktivasyonlar | Model Ağırlıkları ($4-8$ bit quantized) + Dinamik tahsis edilen KV Cache |
| **Hesaplama / Bellek Rejimi** | Yoğun biçimde **Compute-Bound** (Batch başına binlerce token içeren GEMM) | Prefill evresi **Compute-Bound**; Decode evresi kesinlikle **Memory-Bandwidth-Bound** ($B=1$, GEMV) |
| **Baskın Ağ Primitifi** | InfiniBand üzerinde yüksek throughput'lu **AllReduce** & **ReduceScatter** | NVLink üzerinde ultra düşük gecikmeli **AllGather** & **P2P Tensor Parallelism** |
| **Temel Performans Metrikleri** | Model FLOPs Utilization (MFU), Step Time (ms), Gradyan Kararlılığı | Time to First Token (TTFT), Inter-Token Latency (TPOT), Eşzamanlı İstek Sayısı |
| **Kritik Hata Modları** | Loss spike'lar, silent data corruption (SDC), pipeline bubble'lar | KV Cache bellek patlaması (OOM), bellek parçalanması, tail latency sıçramaları |

### Uçtan uca cluster analizi: 16k GPU'da 70B ve 405B modeller

Üç temel sütunumuzu birleştirmek için bir NVIDIA H100 SuperPod üzerinde LLaMA-3 70B ve 405B modellerinin ön eğitimini inceleyelim (Meta'nın gerçek 405B kümesi InfiniBand yerine GPU başına 400 Gb/s RoCE kullandı; aşağıdaki SHARP adımı SuperPod senaryosuna aittir):

1. **8x H100 DGX Node'unda LLaMA-3 70B:**
   - BF16 formatında 70 milyar parametre, model ağırlıkları için 140 GB yer kaplar.
   - 8 H100 SXM5 GPU toplamda $8 \times 80 = 640 \text{ GB}$ HBM3 sunar.
   - Tensor Parallelism $Y = 8$ olarak ayarlanır; ağırlıklar ve aktivasyonlar tamamen 3,6 TB/s bisection bant genişliğine sahip NVLink crossbar alanında kalır.
   - Her Transformer katmanı NVLink üzerinden iki AllReduce yürütür (collective gecikmesi: $<15 \, \mu\text{s}$).
   - Sürdürülen MFU %48-52 bandına ulaşır.
2. **16.384 H100 GPU'da LLaMA-3 405B (2.048 Node):**
   - 4D Paralellik: 8-way intra-node TP $\times$ 16-way PP $\times$ 128-way DP (ZeRO-1).
   - TP kesinlikle 8 GPU'luk bireysel şasiler içine NVLink üzerinde hapsedilir.
   - 16 pipeline stage'i, scale-out InfiniBand üzerinden microbatch aktivasyonlarını aktarır (P2P mesaj boyutu: $\sim 16 \text{ MB}$).
   - 128-way ZeRO-1 data-parallel rank'ları, leaf switch'ler üzerinde Mellanox SHARP donanım hızlandırmalı AllReduce ile optimizer gradyanlarını senkronize eder.
   - Depolama katmanı zero-copy `mmap` ile kesintisiz 2 TB/s token akışı sağlarken, GPUDirect Storage 7,5 TB'lık model checkpoint'lerini yaklaşık her 15 dakikada bir ~30 saniyelik burst yazma duraklamalarıyla kaydeder.

---

## Bütün hikâye altı satırda

- Modern bir GPU süper bilgisayarı birbirine bağlı üç fiziksel sütun tarafından yönetilir: Streaming Multiprocessor hesaplama kapasitesi, çok katmanlı ağ mimarisi ve yüksek verimli depolama I/O'su.
- Roofline modeli H100 üzerinde ~295 FLOP/baytlık kritik aritmetik yoğunluk dikte eder; memory-bound gecikmeden kaçmak için SM başına ~300 tokenlik bir batch size gerekir.
- Intra-node NVLink crossbar'ları GPU başına 450 GB/s tek yönlü bant genişliği (3,6 TB/s bisection) sunarak 8-way Tensor Parallelism'i compute-bound sınırında tutar.
- Scale-out InfiniBand fat-tree topolojisi binlerce GPU arasında tam bisection bant genişliği garanti eder; hat hızları çip başına 50 GB/s'ye (node başına 400 GB/s) iner.
- Mellanox SHARP, kayan nokta toplama işlemlerini doğrudan switch ASIC crossbar'larında yürüterek küme genelindeki AllReduce trafiğini ve gecikmesini yarıya indirir.
- GPUDirect Storage, host CPU bounce buffer'larını ortadan kaldırarak 7,5 TB'lık checkpoint'leri saniyeler içinde kaydeder ve Young-Daly optimal aralığıyla küme MFU'sunu korur.

---

## Terimler sözlüğü

- **Streaming Multiprocessor (SM)** — NVIDIA GPU'larının Tensor Core, Warp Scheduler, CUDA çekirdekleri ve on-chip SRAM barındıran temel çok iş parçacıklı hesaplama hücresi.
- **Warp** — NVIDIA mimarilerinde bir SM warp scheduler tarafından eşzamanlı olarak yürütülen 32 paralel thread'den oluşan temel çizelgeleme birimi.
- **Warp Divergence** — Bir warp içindeki thread'lerin farklı yürütme dallarına sapması sonucu inaktif çekirdeklerin beklemesiyle oluşan başarım kaybı.
- **NVLink** — H100 GPU başına tek yönde 450 GB/s (çift yönde 900 GB/s) bellek aktarımı sağlayan NVIDIA'nın tescilli yüksek bant genişlikli interconnect teknolojisi.
- **NVSwitch** — Bir HGX şasisindeki 8 GPU arasında non-blocking, all-to-all iletişim sağlayan node içi crossbar ASIC'i.
- **InfiniBand Fat-Tree** — Küme node'ları arasında oversubscription (aşırı abonelik) olmadan tam bisection bant genişliği sağlayan hiyerarşik çok katmanlı ağ topolojisi.
- **Mellanox SHARP** — Scalable Hierarchical Aggregation and Reduction Protocol; switch donanımında floating-point AllReduce işlemlerini hatta akarken (in-network reduction) toplayan donanım protokolü.
- **Model FLOPs Utilization (MFU)** — Teorik olarak gereken model FLOP sayısının, donanımın fiilen sürdürdüğü kayan nokta işlem throughput'una oranı.
- **Tensor Parallelism (TP)** — Tekil ağırlık matrislerini GPU'lar arasında bölen ve katı biçimde NVLink bant genişliğiyle sınırlı olan katman içi paralellik yöntemi.
- **Pipeline Parallelism (PP)** — Ardışık Transformer katmanlarını farklı GPU'lara atayan ve pipeline bubble yükü oluşturan katmanlar arası paralellik yöntemi.
- **GPUDirect Storage (GDS)** — NVMe depolama birimlerini host CPU page cache'lerini baypas ederek doğrudan GPU High Bandwidth Memory'ye bağlayan direct DMA teknolojisi.
- **Young-Daly Formülü** — Snapshot yazma süresi ve küme MTBF değerine dayanarak optimal checkpoint aralığını belirleyen matematiksel formülasyon ($\tau = \sqrt{2 \delta M}$).

---

## Daha derine inmek için

- Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12.
- SemiAnalysis (2025). [NVIDIA Tensor Core Evolution from Volta to Blackwell](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/). Mikro-mimari retrospektifi.
- DeepSeek-AI (2024). [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437). arXiv:2412.19437. 2.048 H800 GPU üzerinde 64-way Expert Parallelism ve DualPipe uygulaması.
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
