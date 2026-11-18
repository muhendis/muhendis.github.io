Bir eğitim koşusu %38 MFU’da ilerliyor. GPU utilization %100 görünüyor. Checkpoint birkaç dakikada yazılıyor ve bazı rank’ler diğerlerini bekliyor. Sorun GPU mu, ağ mı, depolama mı?

**Bu yazının temel fikri:** AI kümesi teşhisi tek metriğe bakmak değil, beklemenin fiziksel yerini kanıtla bulmaktır.

> **Önceki yazıdan devraldığımız fikir:** [Üçüncü yazıda](post.html?slug=coklu-gpu-aglari-ve-dagitik-egitim) paralellik gruplarını ve tensör yollarını kurduk. Şimdi bu yollardan biri yavaşladığında belirtileri kök nedene bağlayacağız.

> **Bu yazının sonunda**
>
> - Storage stall ile GPU/kernel darboğazını profiler üzerinde ayırt edebileceksiniz.
> - MTBF ve Young–Daly formülüyle optimum checkpoint aralığını ve kayıp iş bütçesini hesaplayabileceksiniz.
> - GPU utilization, MFU ve HFU arasındaki kavramsal ve sayısal farkları açıklayabileceksiniz.
> - Düşük MFU (%38) arkasındaki eksik %62'lik payı 6 fiziksel bileşene ayırabileceksiniz.
> - Bir AI altyapı mimarisinde straggler, silent data corruption ve fabric tıkanıklığı senaryolarını metodolojik olarak çözebileceksiniz.

**Bu yazıda**

- [1. Depolama GPU’yu nasıl bekletir?](#1-depolama-gpuyu-nasıl-bekletir)
- [2. Büyük kümelerde arıza neden normaldir?](#2-büyük-kümelerde-arıza-neden-normaldir)
- [3. GPU utilization, MFU ve HFU farkı](#3-gpu-utilization-mfu-ve-hfu-farkı)
- [4. Düşük MFU nasıl teşhis edilir?](#4-düşük-mfu-nasıl-teşhis-edilir)
- [5. Beş belirti, beş deney](#5-beş-belirti-beş-deney)
- [6. Maliyet ve enerji hangi metriğe bağlanır?](#6-maliyet-ve-enerji-hangi-metriğe-bağlanır)
- [7. Sistem tasarımı için yedi temel soru](#7-sistem-tasarımı-için-yedi-temel-soru)
- [Okuyucu artık hangi sorulara yanıt verebilir?](#okuyucu-artık-hangi-sorulara-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. Depolama GPU’yu nasıl bekletir?

Bir yapay zekâ süper bilgisayarında tensör hesaplamaları HBM ile Tensor Core'lar arasında pik hızda dönerken, sistemin en yavaş bileşeni olan depolama katmanı (SSD/NVMe ve paralel dosya sistemleri) GPU'ları kolayca açlığa (starvation) sürükleyebilir. Bir baytın depolamadan Tensor Core'a uzanan yolculuğunda çarpabileceği üç fiziksel duvar vardır: **I/O Duvarı**, **Ağ Duvarı** ve **Bellek Duvarı**.

<!-- restored-svg:tr_full:1 -->
<svg viewBox="0 0 720 240" role="img" aria-label="Neden-sonuç haritası. Bir byte depolamadan CPU ve host RAM üzerinden HBM&#x27;e, oradan SM ve Tensor Core&#x27;lara gider ve loss ya da sonraki token&#x27;ı üretir. Depolama ile CPU arasında I/O duvarı, HBM ile SM arasında bellek duvarı vardır. Birden çok GPU varsa sonuçlar NVLink veya InfiniBand kolektifleriyle tekrar HBM&#x27;e döner: bu da ağ duvarıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="mp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="mp-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker><marker id="mp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir byte&#x27;ın yolculuğu ve çarpabileceği üç duvar</text>
<rect x="16" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="76.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Depolama</text>
<text x="76.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">veri, checkpoint</text>
<rect x="158" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="218.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU + host RAM</text>
<text x="218.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">DataLoader, PCIe</text>
<rect x="300" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">HBM3</text>
<text x="360.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80 GB · 3,35 TB/s</text>
<rect x="442" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="502.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">SM + Tensor Core</text>
<text x="502.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">~990 TFLOP/s BF16</text>
<rect x="584" y="44" width="120" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="644.0" y="68.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Sonuç</text>
<text x="644.0" y="83.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">loss / sonraki token</text>
<line x1="136" y1="71" x2="156" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="278" y1="71" x2="298" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="420" y1="71" x2="440" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="562" y1="71" x2="582" y2="71" marker-end="url(#mp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="107.12" y="104" width="79.76" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="147.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">I/O duvarı</text>
<rect x="380.956" y="104" width="100.088" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="431.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Bellek duvarı</text>
<line x1="522" y1="98" x2="522" y2="134" marker-end="url(#mp-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="330" y="136" width="210" height="46" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="435.0" y="156.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">NVLink / InfiniBand</text>
<text x="435.0" y="171.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kolektifler (AllReduce …)</text>
<line x1="360" y1="136" x2="360" y2="100" marker-end="url(#mp-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="530" y="122" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">birden çok GPU mu?</text>
<rect x="398.508" y="190" width="72.984" height="18" rx="9" style="fill:var(--c-warn);fill-opacity:.14;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="435.0" y="202.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Ağ duvarı</text>
<text x="16" y="228" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her bölüm aynı soruyu sorar: bu byte nerede bekliyor ve onu kim bekliyor?</text>
</svg>

### 1.1. Okuma ve yazma farklı darboğazlardır

Büyük dil modellerinin eğitiminde depolama katmanına binen yük simetrik değildir; okuma ve yazma operasyonları taban tabana zıt profillere sahiptir:

1. **Dataloader Okuma Akışı (Sürekli ve Düşük Bant Genişliği):**
   - Küresel ölçekte $16\text{ milyon}$ token işleyen devasa bir LLM adımı düşünelim. Token ID'leri uint16 (2 bayt) veya int32 (4 bayt) olarak saklanır.
   - Global step başına okunan ham veri:
     $$16 \times 10^6 \times 4\text{ bayt} = 64\text{ MB}$$
   - Eğer bir step $1.5\text{ saniye}$ sürüyorsa, tüm kümenin ihtiyaç duyduğu net veri okuma hızı yalnızca $\approx 42.6\text{ MB/s}$'dir. 1.024 GPU'lu bir sistemde GPU başına düşen okuma yükü saniyede birkaç kilobayttır!
   - *Gerçek Darboğaz:* Ham bant genişliği değil, binlerce worker sürecinin milyonlarca küçük dosyayı eşzamanlı açmasıyla oluşan dosya sistemi metadata kilitlenmeleri (POSIX open/stat kilitleri) ve CPU dataloader jitter'ıdır.
2. **Checkpoint Yazma Dalgası (Senkron ve Devasa Ani Patlama):**
   - Checkpoint anında her GPU kendi model ağırlıklarını, optimizer durumlarını (FP32 master ağırlıklar, momentum, varyans) ve scheduler parametrelerini diske dökmek zorundadır.
   - 405 milyar parametreli bir model için küme genelinde yazılan tek bir snapshot'ın boyutu **$7.5\text{ TB}$**'ı bulur!
   - 16.384 GPU aynı saniyede bu devasa tensör kütlesini paylaşımlı depolama havuzuna (Lustre/Ceph) fırlattığında sistemde yüzlerce gigabaytlık ani yazma tsunamisi ($> 250\text{ GB/s}$) patlar.

<!-- restored-svg:tr_full:37 -->
<svg viewBox="0 0 720 300" role="img" aria-label="1. adımda token verisinin yolu: dağıtık dosya sistemi veya obje deposu, örnek seçen, sıralayan, padding/packing yapan CPU DataLoader worker&#x27;larınca okunur; batch pinned host RAM&#x27;e iner ya da direkt I/O kullanılır; ardından PCIe üzerinden H2D kopyayla GPU HBM&#x27;e gelir ve embedding ile Transformer ileri geçişi başlar. Yanında boyut kıyası: 16 milyon token × 4 byte, global step başına yalnızca 64 MB token ID eder; yaklaşık 7,5 TB&#x27;lık checkpoint dalgasının yanında küçücüktür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dp2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dp2-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">1. adımın içi: token&#x27;lar HBM&#x27;e nasıl ulaşır?</text>
<rect x="16" y="42" width="300" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="166.0" y="68.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Dağıtık dosya sistemi / obje deposu</text>
<line x1="166" y1="84" x2="166" y2="98" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="95" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">dosya okuma</text>
<rect x="16" y="100" width="300" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="166.0" y="126.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU DataLoader worker&#x27;ları</text>
<line x1="166" y1="142" x2="166" y2="156" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="153" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">örnek seçme, sıralama, padding/packing</text>
<rect x="16" y="158" width="300" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="166.0" y="184.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Pinned host RAM (veya direkt I/O)</text>
<line x1="166" y1="200" x2="166" y2="214" marker-end="url(#dp2-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="176" y="211" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">PCIe üzerinden H2D kopya</text>
<rect x="16" y="216" width="300" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="166.0" y="242.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM</text>
<text x="326" y="242" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">→ embedding → Transformer forward</text>
<rect x="380" y="42" width="324" height="158" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="392" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Global step başına (Llama 3 405B ölçeği)</text>
<text x="392" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">16 × 10⁶ token × 4 B</text>
<text x="392" y="106" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">= 64 MB token ID</text>
<rect x="392" y="116" width="3" height="16" rx="2" style="fill:var(--c-accent);fill-opacity:.8"/>
<text x="400" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">checkpoint yanında küçücük:</text>
<rect x="392" y="140" width="300" height="16" rx="2" style="fill:var(--c-danger);fill-opacity:.45"/>
<text x="392" y="174" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:600">~7,5 TB checkpoint dalgası</text>
<text x="392" y="190" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px;font-style:italic">(çubuklar ölçekli değil)</text>
<text x="16" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Metin pretraining&#x27;de zor olan ham token baytı değil, erişim düzeni ve checkpoint dalgasıdır; prefetch okumayı hesabın arkasına saklar.</text>
</svg>

### 1.2. Checkpoint alt sınırı

Geleneksel senkron (blocking) checkpointing yönteminde, snapshot süresince kümedeki tüm GPU hesaplamaları dondurulur. Checkpoint yazma süresi depolama kumaşının yazma bant genişliğine katı bir şekilde bağımlıdır:

$$
T_{\mathrm{chk}} = \frac{M_{\mathrm{state}}}{B_{\mathrm{storage}}}
$$

**Sayısal Örnek:**
$M_{\mathrm{state}} = 7.5\text{ TB}$ ($7.500\text{ GB}$) olan bir modelde depolama altyapısının sürdürülebilir toplu yazma hızının $25\text{ GB/s}$ olduğunu varsayalım:
$$T_{\mathrm{chk}} = \frac{7.500\text{ GB}}{25\text{ GB/s}} = 300\text{ saniye} = 5.0\text{ dakika}$$
Bu, kümedeki 16.384 GPU'nun her checkpoint adımında tam 5 dakika boyunca tamamen boşta (idle) beklemesi demektir. Saatlik amorti edilen maliyeti milyonlarca dolar olan bir kümede bu kayıp kabul edilemez.

### 1.3. Asenkron checkpointing

Bu 5 dakikalık donmayı ortadan kaldırmak için modern altyapılar **Asenkron İki Kademeli Checkpointing (Asynchronous / Non-blocking Checkpoint)** uygular:

1. **Bellek İçi Snapshot (Copy-on-Write):** Checkpoint adımı geldiğinde GPU'lar eğitimi yalnızca 1–2 saniye durdurur. GPU HBM'sindeki optimizer ve ağırlık tensörleri, yerel sunucudaki CPU host RAM'ine veya yerel NVMe sürücüsüne yüksek hızlı PCIe Gen5 DMA ($> 50\text{ GB/s}$) ile kopyalanır.
2. **Arka Plan Boşaltma (Background Drain):** Bellek içi kopya tamamlanır tamamlanmaz GPU'lar anında bir sonraki eğitim step'ine (Forward GEMM) başlar.
3. **Asenkron Dağıtık Yazma:** CPU tarafındaki arka plan süreçleri (daemon), yerel RAM'deki snapshot'ı ağ üzerinden merkezi dosya sistemine asenkron olarak yazar. GPU hesaplama döngüsü depolama I/O gecikmesinden tamamen izole edilir.

### 1.4. GPUDirect Storage neyi çözmez?

NVIDIA GPUDirect Storage (GDS), veri transferlerinde CPU host belleğini aradan çıkararak doğrudan NVMe sürücüler veya ağ depolama kartları ile GPU HBM'si arasında DMA (Direct Memory Access) bağlantısı kurar (`cuFile` API).

<!-- restored-svg:tr_full:42 -->
<svg viewBox="0 0 720 308" role="img" aria-label="Depolamadan GPU HBM&#x27;e iki veri yolu. Klasik: veri depodan CPU RAM&#x27;deki bounce buffer&#x27;a kopyalanır, sonra PCIe üzerinden yeniden HBM&#x27;e kopyalanır. GPUDirect Storage: veri cuFile ile PCIe veya NVMe-oF üzerinden DMA&#x27;yla doğrudan HBM&#x27;e gider; CPU yalnızca kontrolü yürütür. GDS bir kopyayı kaldırır, depolama sınırını değil: 25 GB/s&#x27;lik hedef 25 GB/s kalır; compatibility mode CPU yoluna geri dönebilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="gd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="gd-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Klasik bounce buffer yolu ve GPUDirect Storage</text>
<text x="16" y="56" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">1 · klasik: CPU RAM üzerinden kopya</text>
<rect x="16" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Depolama</text>
<text x="86.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVMe / paralel FS</text>
<rect x="564" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="634.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM3</text>
<text x="634.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="290" y="68" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="90.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">CPU RAM</text>
<text x="360.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bounce buffer</text>
<line x1="156" y1="92" x2="288" y2="92" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="430" y1="92" x2="562" y2="92" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="222" y="86" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kopya 1</text>
<text x="496" y="86" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kopya 2 (PCIe)</text>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">2 · GDS: HBM&#x27;e doğrudan DMA</text>
<rect x="16" y="182" width="140" height="48" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="86.0" y="204.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Depolama</text>
<text x="86.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVMe / paralel FS</text>
<rect x="564" y="182" width="140" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="634.0" y="204.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU HBM3</text>
<text x="634.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<line x1="156" y1="206" x2="562" y2="206" marker-end="url(#gd-o)" style="stroke:var(--c-success);stroke-width:2"/>
<text x="360" y="200" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">PCIe / NVMe-oF üzerinden DMA (cuFile)</text>
<rect x="290" y="240" width="140" height="34" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="360.0" y="261.75" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">CPU: yalnızca kontrol</text>
<line x1="360" y1="240" x2="360" y2="212" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="16" y="296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GDS bir kopyayı kaldırır, depolama sınırını değil: 25 GB/s&#x27;lik hedef 25 GB/s kalır. Compatibility mode CPU yoluna geri dönebilir.</text>
</svg>

GDS'in sağladığı ve sağlayamadığı faydalar arasındaki 4 adımlı mühendislik ayrımı:

1. **CPU Bounce Buffer'ı Kaldırır:** Klasik I/O'da veri depolamadan CPU RAM'indeki ara belleğe (bounce buffer) kopyalanır, ardından PCIe üzerinden HBM'e aktarılır. GDS bu çift kopyayı tek kopyaya indirir.
2. **CPU ve PCIe Rekabetini Azaltır:** CPU çekirdekleri memory-copy döngülerinden kurtulur; PCIe üzerindeki gereksiz trafik yarıya iner.
3. **Neyi Çözmez? (Fiziksel Depolama Sınırı):** Eğer arkadaki NVMe havuzu veya Lustre storage sunucuları fiziksel olarak en fazla $25\text{ GB/s}$ veri sağlayabiliyorsa, GDS kullanmak bu hattı $100\text{ GB/s}$ yapamaz. Borunun giriş hızı neyse tavan odur.
4. **Uyumluluk Modu Tuzağı (Compatibility Mode):** Eğer çekirdek sürücüsü veya dosya sistemi GDS ile tam uyumlu yapılandırılmamışsa, GDS sessizce fallback yaparak geleneksel CPU kopyalama yoluna geri döner; profiler'da GDS aktif görünse de performans fırlamaz.

### 1.5. Veri seti mi checkpoint mi?

Eğitim sırasında GPU'ların I/O yüzünden durakladığını (Input Stall) fark ettiğinizde sorunun veri seti okumasından mı yoksa checkpoint yazmasından mı kaynaklandığını izole etmek için şu sentetik deneyi yapın:

- **Sentetik Dataloader Deneyi:** Gerçek disk dosyalarını okumak yerine, GPU HBM'sinde önceden ayrılmış rastgele tensörlerle (in-memory dummy tensors) 100 adım eğitim çalıştırın.
- *Sonuç A:* Eğer step süresi belirgin şekilde düşüyor ve MFU fırlıyorsa, sorun kesinlikle veri seti okuma pipeline'ındadır (dataloader worker sayısı, disk IOPS veya CPU token preprocessing yetersizdir).
- *Sonuç B:* Eğer step süresi değişmiyorsa, depolama okuması masumdur; gecikme hesaplama kernel'larında veya NCCL kolektif iletişim ağındadır.
- *Checkpoint İzolasyonu:* Checkpoint aralığını iki katına çıkarın (örneğin 30 dakikadan 60 dakikaya). Saat başına ortalama step süresi iyileşiyorsa, checkpoint yazma dalgaları küme MFU'sunu aşağı çekmektedir.

---

## 2. Büyük kümelerde arıza neden normaldir?

Küçük bir sunucuda donanım arızası nadir bir istisnadır. Ancak 16.384 GPU'luk bir süper bilgisayarda arıza istisna değil, **sistemin sürekli çalışma durumudur**.

<!-- restored-svg:tr_full:43 -->
<svg viewBox="0 0 720 196" role="img" aria-label="Llama 3 405B pretraining&#x27;inin 54 gününde yaşanan 466 iş kesintisinin yığılmış çubuğu: 47 planlı, 419 beklenmedik; beklenmediklerin yaklaşık %78&#x27;i, kabaca 327&#x27;si doğrulanmış veya şüpheli donanım sorunlarına, 92 kadarı diğer nedenlere bağlandı. Bu, ortalama 3,09 saatte bir beklenmedik kesinti demektir. Yalnızca 3&#x27;ü ciddi manuel müdahale gerektirdi; kalanını otomasyon çözdü." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="li-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Llama 3 405B pretraining: 54 günde 466 iş kesintisi</text>
<rect x="17" y="44" width="67.3906" height="36" rx="2" style="fill:var(--c-text-mute);fill-opacity:.25"/>
<text x="50.69527896995708" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">47 planlı</text>
<rect x="86.3906" y="44" width="480.781" height="36" rx="2" style="fill:var(--c-danger);fill-opacity:.45"/>
<text x="326.78111587982835" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">~327 donanım (beklenmedik olanların ≈%78&#x27;i)</text>
<rect x="569.172" y="44" width="133.828" height="36" rx="2" style="fill:var(--c-warn);fill-opacity:.4"/>
<text x="636.0858369098713" y="67" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">~92 diğer</text>
<path d="M86.39055793991416 88 V94 H703 V88" style="fill:none;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="394.6952789699571" y="110" text-anchor="middle" style="fill:var(--c-danger);font-size:12px;font-weight:700">419 beklenmedik</text>
<rect x="16" y="124" width="688" height="60" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="146" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">54 gün × 24 sa ÷ 419 ≈ beklenmedik kesintiler arası 3,09 sa</text>
<text x="28" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca 3&#x27;ü ciddi manuel müdahale gerektirdi; kalanını otomasyon çözdü. “Her 3 saatte bir H100 bozuldu” demek değildir.</text>
</svg>

### 2.1. Küme MTBF sezgisi

Bir sistemdeki bileşen sayısı arttıkça, bağımsız arıza olasılıkları toplanır. Bir kümenin Ortalama Arıza Aralığı (Mean Time Between Failures - MTBF), tek bir bileşenin MTBF değerinin bileşen sayısına bölünmesiyle hesaplanır:

$$
\mathrm{MTBF}_{\mathrm{cluster}} = \frac{\mathrm{MTBF}_{\mathrm{node}}}{N_{\mathrm{nodes}}}
$$

Tek bir DGX H100 sunucusunun (içindeki 8 GPU, CPU'lar, bellekler, güç kaynakları ve 8 InfiniBand NIC ile birlikte) son derece güvenilir olduğunu ve yıllık donanım arıza oranının yalnızca %10 olduğunu varsayalım:
- Düğüm MTBF'si $\approx 10\text{ yıl} \approx 87.600\text{ saat}$.
- 2.048 adet DGX sunucusundan oluşan devasa bir kümede (16.384 GPU):
  $$\mathrm{MTBF}_{\mathrm{cluster}} = \frac{87.600\text{ saat}}{2.048} \approx 42.7\text{ saat}$$
Ancak bu sadece sunucu arızasıdır! Buna optik alıcı-vericiler (transceivers), kablo bükülmeleri, switch portları, fanlar, soğutma sıvı kaçakları ve yazılım kernel crash'leri eklendiğinde arıza aralığı saatler seviyesine iner.

**Meta Llama 3 405B Verisi:**
Meta'nın teknik raporuna göre, 54 günlük ön eğitim sürecinde tam **466 kez iş kesintisi (job interruption)** yaşanmıştır. Bunların 419'u beklenmedik arızalardır. Bu veri, kümede ortalama her **3.09 saatte bir** beklenmedik bir arıza yaşandığını belgeler!

### 2.2. Checkpoint sıklığı nasıl seçilir?

Arızalar bu kadar sık yaşanıyorsa ne sıklıkla checkpoint alınmalıdır?
- **Çok sık alırsanız:** Zamanınızın çoğu checkpoint yazma duraklamalarıyla boşa gider.
- **Çok seyrek alırsanız:** Bir arıza yaşandığında saatlerce geriye düşer, son checkpoint'ten arıza anına kadar yapılan tüm hesaplamaları çöpe atıp yeniden yapmak zorunda kalırsınız.

Bu dengeyi matematiksel olarak çözen model **Young–Daly Optimum Checkpoint Formülü**'dür:

$$
\tau_{\mathrm{opt}} \approx \sqrt{2 \times \delta \times M}
$$

Burada:
- $\tau_{\mathrm{opt}}$: İki checkpoint arasındaki optimum süre (dakika veya saat),
- $\delta$: Bir checkpoint'i diske yazmanın gerektirdiği duraklama süresi,
- $M$: Kümenin ortalama arıza aralığı ($\mathrm{MTBF}_{\mathrm{cluster}}$).

<!-- restored-svg:tr_full:44 -->
<svg viewBox="0 0 720 326" role="img" aria-label="7,5 TB checkpoint ve 3 saatlik ortalama kesinti aralığı için toplam kaybın checkpoint aralığına göre Young–Daly eğrileri. 25 GB/s depolamada bloklayıcı yazma 300 saniye sürer; eğrinin dibi 42,4 dakikada, %23,6 kayıptadır. 250 GB/s&#x27;de yazma 30 saniye sürer; optimum 13,4 dakikada, %7,5 kayıptadır. Çok kısa aralıklarda yazma, çok uzun aralıklarda arıza sonrası yeniden hesaplama baskındır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="yd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Young–Daly: toplam kayıp W(τ) = δ/τ + τ/2M; M = 3 sa, 7,5 TB checkpoint</text>
<line x1="70" y1="280" x2="690" y2="280" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="284.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%0</text>
<line x1="70" y1="234" x2="690" y2="234" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="238.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%10</text>
<line x1="70" y1="188" x2="690" y2="188" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="192.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%20</text>
<line x1="70" y1="142" x2="690" y2="142" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="146.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%30</text>
<line x1="70" y1="96" x2="690" y2="96" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="100.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%40</text>
<line x1="70" y1="50" x2="690" y2="50" style="stroke:var(--c-border);stroke-width:0.7"/>
<text x="64" y="54.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">%50</text>
<text x="70.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">0</text>
<text x="173.33333333333331" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">15</text>
<text x="276.66666666666663" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">30</text>
<text x="380.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">45</text>
<text x="483.3333333333333" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">60</text>
<text x="586.6666666666667" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">75</text>
<text x="690.0" y="296" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">90</text>
<line x1="70" y1="280" x2="690" y2="280" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="70" y1="50" x2="70" y2="280" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="380.0" y="314" text-anchor="middle" style="fill:var(--c-text);font-size:12px">checkpoint aralığı τ (dakika)</text>
<path d="M144.4 53.2 L145.9 57.3 L147.5 61.2 L149.1 64.9 L150.6 68.5 L152.2 71.9 L153.7 75.2 L155.2 78.3 L156.8 81.4 L158.3 84.3 L159.9 87.1 L161.4 89.8 L163.0 92.4 L164.6 94.9 L166.1 97.3 L167.7 99.6 L169.2 101.9 L170.8 104.0 L172.3 106.1 L173.8 108.2 L175.4 110.1 L176.9 112.0 L178.5 113.8 L180.1 115.6 L181.6 117.3 L183.1 119.0 L184.7 120.6 L186.2 122.1 L187.8 123.6 L189.4 125.1 L190.9 126.5 L192.4 127.9 L194.0 129.2 L195.6 130.5 L197.1 131.8 L198.7 133.0 L200.2 134.2 L201.8 135.3 L203.3 136.4 L204.8 137.5 L206.4 138.5 L207.9 139.6 L209.5 140.5 L211.1 141.5 L212.6 142.4 L214.2 143.3 L215.7 144.2 L217.2 145.1 L218.8 145.9 L220.3 146.7 L221.9 147.5 L223.4 148.3 L225.0 149.0 L226.6 149.8 L228.1 150.5 L229.7 151.1 L231.2 151.8 L232.8 152.5 L234.3 153.1 L235.9 153.7 L237.4 154.3 L238.9 154.9 L240.5 155.4 L242.1 156.0 L243.6 156.5 L245.2 157.1 L246.7 157.6 L248.2 158.0 L249.8 158.5 L251.3 159.0 L252.9 159.4 L254.4 159.9 L256.0 160.3 L257.5 160.7 L259.1 161.1 L260.6 161.5 L262.2 161.9 L263.8 162.3 L265.3 162.6 L266.9 163.0 L268.4 163.3 L270.0 163.7 L271.5 164.0 L273.1 164.3 L274.6 164.6 L276.1 164.9 L277.7 165.2 L279.2 165.5 L280.8 165.7 L282.4 166.0 L283.9 166.3 L285.4 166.5 L287.0 166.7 L288.6 167.0 L290.1 167.2 L291.6 167.4 L293.2 167.6 L294.8 167.8 L296.3 168.0 L297.9 168.2 L299.4 168.4 L300.9 168.6 L302.5 168.7 L304.1 168.9 L305.6 169.0 L307.1 169.2 L308.7 169.3 L310.2 169.5 L311.8 169.6 L313.4 169.8 L314.9 169.9 L316.4 170.0 L318.0 170.1 L319.6 170.2 L321.1 170.3 L322.6 170.4 L324.2 170.5 L325.8 170.6 L327.3 170.7 L328.9 170.8 L330.4 170.9 L331.9 170.9 L333.5 171.0 L335.1 171.1 L336.6 171.1 L338.1 171.2 L339.7 171.2 L341.2 171.3 L342.8 171.3 L344.4 171.4 L345.9 171.4 L347.4 171.4 L349.0 171.5 L350.6 171.5 L352.1 171.5 L353.6 171.5 L355.2 171.5 L356.8 171.6 L358.3 171.6 L359.9 171.6 L361.4 171.6 L362.9 171.6 L364.5 171.6 L366.1 171.6 L367.6 171.6 L369.1 171.5 L370.7 171.5 L372.2 171.5 L373.8 171.5 L375.4 171.5 L376.9 171.4 L378.4 171.4 L380.0 171.4 L381.6 171.4 L383.1 171.3 L384.6 171.3 L386.2 171.2 L387.8 171.2 L389.3 171.2 L390.9 171.1 L392.4 171.1 L393.9 171.0 L395.5 170.9 L397.0 170.9 L398.6 170.8 L400.1 170.8 L401.7 170.7 L403.2 170.6 L404.8 170.6 L406.3 170.5 L407.9 170.4 L409.4 170.4 L411.0 170.3 L412.6 170.2 L414.1 170.1 L415.6 170.0 L417.2 170.0 L418.8 169.9 L420.3 169.8 L421.9 169.7 L423.4 169.6 L424.9 169.5 L426.5 169.4 L428.1 169.3 L429.6 169.2 L431.2 169.1 L432.7 169.0 L434.2 168.9 L435.8 168.8 L437.4 168.7 L438.9 168.6 L440.5 168.5 L442.0 168.4 L443.6 168.3 L445.1 168.2 L446.6 168.1 L448.2 168.0 L449.8 167.8 L451.3 167.7 L452.9 167.6 L454.4 167.5 L455.9 167.4 L457.5 167.2 L459.1 167.1 L460.6 167.0 L462.1 166.9 L463.7 166.7 L465.2 166.6 L466.8 166.5 L468.4 166.3 L469.9 166.2 L471.4 166.1 L473.0 165.9 L474.5 165.8 L476.1 165.7 L477.6 165.5 L479.2 165.4 L480.8 165.2 L482.3 165.1 L483.8 165.0 L485.4 164.8 L486.9 164.7 L488.5 164.5 L490.1 164.4 L491.6 164.2 L493.1 164.1 L494.7 163.9 L496.2 163.8 L497.8 163.6 L499.4 163.5 L500.9 163.3 L502.4 163.1 L504.0 163.0 L505.6 162.8 L507.1 162.7 L508.7 162.5 L510.2 162.4 L511.8 162.2 L513.3 162.0 L514.9 161.9 L516.4 161.7 L518.0 161.5 L519.5 161.4 L521.0 161.2 L522.6 161.0 L524.1 160.9 L525.7 160.7 L527.2 160.5 L528.8 160.4 L530.4 160.2 L531.9 160.0 L533.5 159.8 L535.0 159.7 L536.5 159.5 L538.1 159.3 L539.6 159.2 L541.2 159.0 L542.8 158.8 L544.3 158.6 L545.9 158.4 L547.4 158.3 L549.0 158.1 L550.5 157.9 L552.0 157.7 L553.6 157.5 L555.1 157.4 L556.7 157.2 L558.2 157.0 L559.8 156.8 L561.3 156.6 L562.9 156.4 L564.5 156.2 L566.0 156.1 L567.5 155.9 L569.1 155.7 L570.6 155.5 L572.2 155.3 L573.8 155.1 L575.3 154.9 L576.9 154.7 L578.4 154.5 L580.0 154.3 L581.5 154.1 L583.0 154.0 L584.6 153.8 L586.1 153.6 L587.7 153.4 L589.2 153.2 L590.8 153.0 L592.4 152.8 L593.9 152.6 L595.5 152.4 L597.0 152.2 L598.5 152.0 L600.1 151.8 L601.6 151.6 L603.2 151.4 L604.8 151.2 L606.3 151.0 L607.9 150.8 L609.4 150.6 L611.0 150.4 L612.5 150.2 L614.0 150.0 L615.6 149.8 L617.1 149.6 L618.7 149.3 L620.2 149.1 L621.8 148.9 L623.4 148.7 L624.9 148.5 L626.5 148.3 L628.0 148.1 L629.5 147.9 L631.1 147.7 L632.6 147.5 L634.2 147.3 L635.8 147.1 L637.3 146.8 L638.9 146.6 L640.4 146.4 L642.0 146.2 L643.5 146.0 L645.0 145.8 L646.6 145.6 L648.1 145.4 L649.7 145.1 L651.2 144.9 L652.8 144.7 L654.4 144.5 L655.9 144.3 L657.5 144.1 L659.0 143.8 L660.5 143.6 L662.1 143.4 L663.6 143.2 L665.2 143.0 L666.8 142.8 L668.3 142.5 L669.9 142.3 L671.4 142.1 L673.0 141.9 L674.5 141.7 L676.0 141.4 L677.6 141.2 L679.1 141.0 L680.7 140.8 L682.2 140.6 L683.8 140.3 L685.4 140.1 L686.9 139.9 L688.5 139.7" style="fill:none;stroke:var(--c-danger);stroke-width:2.2"/>
<circle cx="362.271" cy="171.577" r="5" style="fill:var(--c-danger);fill-opacity:.9"/>
<line x1="362.271" y1="171.577" x2="362.271" y2="280" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="370.27080289043965" y="193.5769602180627" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:700">τ* = 42,4 dk → %23,6</text>
<line x1="428.222" y1="110.4" x2="450.222" y2="110.4" style="stroke:var(--c-danger);stroke-width:2.2"/>
<text x="456.2222222222222" y="114.4" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">25 GB/s → δ = 300 sn</text>
<path d="M77.8 74.1 L79.3 107.9 L80.8 132.0 L82.4 149.9 L84.0 163.8 L85.5 174.9 L87.0 183.9 L88.6 191.4 L90.2 197.6 L91.7 203.0 L93.2 207.5 L94.8 211.5 L96.3 215.0 L97.9 218.0 L99.5 220.7 L101.0 223.1 L102.5 225.3 L104.1 227.2 L105.7 228.9 L107.2 230.5 L108.8 231.9 L110.3 233.2 L111.8 234.4 L113.4 235.4 L115.0 236.4 L116.5 237.3 L118.0 238.1 L119.6 238.9 L121.2 239.5 L122.7 240.2 L124.2 240.7 L125.8 241.3 L127.3 241.7 L128.9 242.2 L130.4 242.6 L132.0 242.9 L133.6 243.3 L135.1 243.6 L136.7 243.9 L138.2 244.1 L139.8 244.3 L141.3 244.6 L142.8 244.7 L144.4 244.9 L145.9 245.1 L147.5 245.2 L149.1 245.3 L150.6 245.4 L152.2 245.5 L153.7 245.5 L155.2 245.6 L156.8 245.6 L158.3 245.7 L159.9 245.7 L161.4 245.7 L163.0 245.7 L164.6 245.7 L166.1 245.7 L167.7 245.7 L169.2 245.6 L170.8 245.6 L172.3 245.5 L173.8 245.5 L175.4 245.4 L176.9 245.3 L178.5 245.3 L180.1 245.2 L181.6 245.1 L183.1 245.0 L184.7 244.9 L186.2 244.8 L187.8 244.7 L189.4 244.6 L190.9 244.5 L192.4 244.3 L194.0 244.2 L195.6 244.1 L197.1 244.0 L198.7 243.8 L200.2 243.7 L201.8 243.5 L203.3 243.4 L204.8 243.2 L206.4 243.1 L207.9 242.9 L209.5 242.8 L211.1 242.6 L212.6 242.4 L214.2 242.3 L215.7 242.1 L217.2 241.9 L218.8 241.8 L220.3 241.6 L221.9 241.4 L223.4 241.2 L225.0 241.0 L226.6 240.8 L228.1 240.7 L229.7 240.5 L231.2 240.3 L232.8 240.1 L234.3 239.9 L235.9 239.7 L237.4 239.5 L238.9 239.3 L240.5 239.1 L242.1 238.9 L243.6 238.7 L245.2 238.5 L246.7 238.3 L248.2 238.0 L249.8 237.8 L251.3 237.6 L252.9 237.4 L254.4 237.2 L256.0 237.0 L257.5 236.8 L259.1 236.5 L260.6 236.3 L262.2 236.1 L263.8 235.9 L265.3 235.7 L266.9 235.4 L268.4 235.2 L270.0 235.0 L271.5 234.8 L273.1 234.5 L274.6 234.3 L276.1 234.1 L277.7 233.8 L279.2 233.6 L280.8 233.4 L282.4 233.2 L283.9 232.9 L285.4 232.7 L287.0 232.4 L288.6 232.2 L290.1 232.0 L291.6 231.7 L293.2 231.5 L294.8 231.3 L296.3 231.0 L297.9 230.8 L299.4 230.5 L300.9 230.3 L302.5 230.1 L304.1 229.8 L305.6 229.6 L307.1 229.3 L308.7 229.1 L310.2 228.8 L311.8 228.6 L313.4 228.4 L314.9 228.1 L316.4 227.9 L318.0 227.6 L319.6 227.4 L321.1 227.1 L322.6 226.9 L324.2 226.6 L325.8 226.4 L327.3 226.1 L328.9 225.9 L330.4 225.6 L331.9 225.4 L333.5 225.1 L335.1 224.9 L336.6 224.6 L338.1 224.4 L339.7 224.1 L341.2 223.8 L342.8 223.6 L344.4 223.3 L345.9 223.1 L347.4 222.8 L349.0 222.6 L350.6 222.3 L352.1 222.1 L353.6 221.8 L355.2 221.5 L356.8 221.3 L358.3 221.0 L359.9 220.8 L361.4 220.5 L362.9 220.3 L364.5 220.0 L366.1 219.7 L367.6 219.5 L369.1 219.2 L370.7 219.0 L372.2 218.7 L373.8 218.4 L375.4 218.2 L376.9 217.9 L378.4 217.7 L380.0 217.4 L381.6 217.1 L383.1 216.9 L384.6 216.6 L386.2 216.3 L387.8 216.1 L389.3 215.8 L390.9 215.5 L392.4 215.3 L393.9 215.0 L395.5 214.8 L397.0 214.5 L398.6 214.2 L400.1 214.0 L401.7 213.7 L403.2 213.4 L404.8 213.2 L406.3 212.9 L407.9 212.6 L409.4 212.4 L411.0 212.1 L412.6 211.8 L414.1 211.6 L415.6 211.3 L417.2 211.0 L418.8 210.8 L420.3 210.5 L421.9 210.2 L423.4 210.0 L424.9 209.7 L426.5 209.4 L428.1 209.2 L429.6 208.9 L431.2 208.6 L432.7 208.4 L434.2 208.1 L435.8 207.8 L437.4 207.5 L438.9 207.3 L440.5 207.0 L442.0 206.7 L443.6 206.5 L445.1 206.2 L446.6 205.9 L448.2 205.7 L449.8 205.4 L451.3 205.1 L452.9 204.8 L454.4 204.6 L455.9 204.3 L457.5 204.0 L459.1 203.8 L460.6 203.5 L462.1 203.2 L463.7 203.0 L465.2 202.7 L466.8 202.4 L468.4 202.1 L469.9 201.9 L471.4 201.6 L473.0 201.3 L474.5 201.0 L476.1 200.8 L477.6 200.5 L479.2 200.2 L480.8 200.0 L482.3 199.7 L483.8 199.4 L485.4 199.1 L486.9 198.9 L488.5 198.6 L490.1 198.3 L491.6 198.0 L493.1 197.8 L494.7 197.5 L496.2 197.2 L497.8 196.9 L499.4 196.7 L500.9 196.4 L502.4 196.1 L504.0 195.8 L505.6 195.6 L507.1 195.3 L508.7 195.0 L510.2 194.8 L511.8 194.5 L513.3 194.2 L514.9 193.9 L516.4 193.7 L518.0 193.4 L519.5 193.1 L521.0 192.8 L522.6 192.5 L524.1 192.3 L525.7 192.0 L527.2 191.7 L528.8 191.4 L530.4 191.2 L531.9 190.9 L533.5 190.6 L535.0 190.3 L536.5 190.1 L538.1 189.8 L539.6 189.5 L541.2 189.2 L542.8 189.0 L544.3 188.7 L545.9 188.4 L547.4 188.1 L549.0 187.9 L550.5 187.6 L552.0 187.3 L553.6 187.0 L555.1 186.7 L556.7 186.5 L558.2 186.2 L559.8 185.9 L561.3 185.6 L562.9 185.4 L564.5 185.1 L566.0 184.8 L567.5 184.5 L569.1 184.3 L570.6 184.0 L572.2 183.7 L573.8 183.4 L575.3 183.1 L576.9 182.9 L578.4 182.6 L580.0 182.3 L581.5 182.0 L583.0 181.7 L584.6 181.5 L586.1 181.2 L587.7 180.9 L589.2 180.6 L590.8 180.4 L592.4 180.1 L593.9 179.8 L595.5 179.5 L597.0 179.2 L598.5 179.0 L600.1 178.7 L601.6 178.4 L603.2 178.1 L604.8 177.8 L606.3 177.6 L607.9 177.3 L609.4 177.0 L611.0 176.7 L612.5 176.5 L614.0 176.2 L615.6 175.9 L617.1 175.6 L618.7 175.3 L620.2 175.1 L621.8 174.8 L623.4 174.5 L624.9 174.2 L626.5 173.9 L628.0 173.7 L629.5 173.4 L631.1 173.1 L632.6 172.8 L634.2 172.5 L635.8 172.3 L637.3 172.0 L638.9 171.7 L640.4 171.4 L642.0 171.1 L643.5 170.9 L645.0 170.6 L646.6 170.3 L648.1 170.0 L649.7 169.7 L651.2 169.5 L652.8 169.2 L654.4 168.9 L655.9 168.6 L657.5 168.3 L659.0 168.1 L660.5 167.8 L662.1 167.5 L663.6 167.2 L665.2 166.9 L666.8 166.7 L668.3 166.4 L669.9 166.1 L671.4 165.8 L673.0 165.5 L674.5 165.3 L676.0 165.0 L677.6 164.7 L679.1 164.4 L680.7 164.1 L682.2 163.8 L683.8 163.6 L685.4 163.3 L686.9 163.0 L688.5 162.7" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<circle cx="162.424" cy="245.714" r="5" style="fill:var(--c-success);fill-opacity:.9"/>
<line x1="162.424" y1="245.714" x2="162.424" y2="280" style="stroke:var(--c-success);stroke-width:1;stroke-dasharray:4 3"/>
<text x="170.42414306999132" y="267.7136243450032" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:700">τ* = 13,4 dk → %7,5</text>
<line x1="428.222" y1="124.2" x2="450.222" y2="124.2" style="stroke:var(--c-success);stroke-width:2.2"/>
<text x="456.2222222222222" y="128.2" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:600">250 GB/s → δ = 30 sn</text>
<text x="194.0" y="63.80000000000001" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">kısa τ: yazma baskın</text>
<text x="428.2222222222222" y="91.4" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-weight:600">uzun τ: arıza sonrası yeniden hesap baskın</text>
</svg>

**Sayısal Doğrulama ve Young–Daly Eğrisi:**
$M = 3.0\text{ saat} = 180\text{ dakika}$ olan bir kümede iki farklı depolama hızını kıyaslayalım:

- **Senaryo A (Yavaş Depolama - $\delta = 5.0\text{ dakika} = 300\text{ saniye}$):**
  $$\tau_{\mathrm{opt}} = \sqrt{2 \times 5.0 \times 180} = \sqrt{1.800} \approx 42.4\text{ dakika}$$
  Bu kümede her 42.4 dakikada bir checkpoint alınmalıdır. Toplam kayıp bütçesi (yazma maliyeti + beklenen yeniden hesaplama) çalışma süresinin **%23.6**'sını tüketir!
- **Senaryo B (Hızlı / Asenkron Depolama - $\delta = 0.5\text{ dakika} = 30\text{ saniye}$):**
  $$\tau_{\mathrm{opt}} = \sqrt{2 \times 0.5 \times 180} = \sqrt{180} \approx 13.4\text{ dakika}$$
  Optimum checkpoint aralığı 13.4 dakikaya iner ve toplam kayıp bütçesi **%7.5** seviyesine düşer. Depolamayı hızlandırmak yalnızca yazma süresini değil, arıza anında kaybedilen iş miktarını da radikal şekilde azaltır!

### 2.3. Checkpoint tek başına yetmez

Checkpoint almak arıza yönetiminin yalnızca ilk adımıdır. Bir GPU veya optik modül arızalandığında sistemi hızla kurtarmak için otomatik bir **Düğüm Karantina ve Kurtarma Döngüsü (Quarantine Loop)** gerekir:

<!-- restored-svg:tr_full:45 -->
<svg viewBox="0 0 720 342" role="img" aria-label="Düğüm karantina döngüsü. Sağlıklı havuzdaki bir düğüm Xid, ECC veya link hatası gibi telemetriyle ya da yavaş step süresiyle işaretlenir. Boşaltılıp batch havuzundan izole edilir, ardından kademeli olarak doğrulanır: DCGM tanılama, tek düğüm NCCL testi ve çok düğüm NCCL testi. Geçerse havuza kademeli olarak döner; kalırsa onarıma veya RMA&#x27;ya gider ve sonra yeniden test edilir. Arızalı düğüm hâlâ işteyken checkpoint&#x27;ten yeniden başlamak aynı arızayı tekrar üretir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="qr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="qr-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="qr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Düğüm karantina döngüsü: tespit → izole et → sağlığı kanıtla → geri al</text>
<rect x="16" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="91.0" y="72.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Sağlıklı havuz</text>
<text x="91.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">işler burada çalışır</text>
<rect x="206" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="281.0" y="64.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Tespit</text>
<text x="281.0" y="79.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Xid · ECC · link hatası</text>
<text x="281.0" y="93.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yavaş step süresi</text>
<rect x="396" y="44" width="150" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="471.0" y="72.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Boşalt + izole et</text>
<text x="471.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">batch havuzundan çıkar</text>
<line x1="166" y1="74" x2="204" y2="74" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="356" y1="74" x2="394" y2="74" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="380" y="128" width="324" height="132" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="392" y="148" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Doğrula (kademeli)</text>
<rect x="392" y="158" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="175.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">1 · DCGM tanılama</text>
<rect x="392" y="190" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="207.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">2 · tek düğüm NCCL testi</text>
<rect x="392" y="222" width="300" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.1;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="542.0" y="239.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">3 · çok düğüm NCCL testi</text>
<line x1="471" y1="104" x2="471" y2="126" marker-end="url(#qr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M380 200 H91 V106" marker-end="url(#qr-o)" style="fill:none;stroke:var(--c-success);stroke-width:1.5"/>
<text x="236" y="192" text-anchor="middle" style="fill:var(--c-success);font-size:11.5px;font-weight:600">geçti → kademeli dönüş</text>
<path d="M542 260 V286 H360" marker-end="url(#qr-b)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="206" y="270" width="152" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="282.0" y="295.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Onarım / RMA</text>
<text x="452" y="300" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">kaldı</text>
<path d="M282 270 V244 H378" marker-end="url(#qr-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="300" y="238" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yeniden test</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Arızalı düğüm hâlâ işteyken checkpoint&#x27;ten yeniden başlamak aynı arızayı tekrar üretir.</text>
</svg>

Karantina döngüsünün 4 kritik aşaması:

1. **Hata Tespiti:** Sistem DCGM Xid olayları, PCIe CRC hataları veya eğitim step süresinin eşik değeri aşmasıyla arızalı rank'i tespit eder.
2. **Karantinaya Alma ve Boşaltma:** Arızalı düğüm iş havuzundan derhal çıkarılır ve karantina havuzuna (triage pool) aktarılır.
3. **Kademeli Doğrulama (Triage Diagnostics):** Karantinadaki düğümde sırasıyla seviye-1 DCGM testleri, tek düğüm cuBLAS GEMM stres testleri ve komşu düğümlerle NCCL bant genişliği testleri otomatik çalıştırılır.
4. **Hotswap ve Yeniden Başlatma:** Küme, arızalı düğümü yedek (spare) sağlıklı bir düğümle değiştirir; son checkpoint'ten eğitim 2–3 dakika içinde otomatik olarak devam eder. Arızalı düğüm ise onarım ve RMA için donanım ekibine bilet açar.

### 2.4. Silent Data Corruption

En tehlikeli donanım arızası makinenin çökmesi değildir; makinenin çöktüğünü belli etmeden yanlış hesap yapmasıdır. Buna **Sessiz Veri Bozulması (Silent Data Corruption - SDC)** veya **Silent Data Errors (SDE)** denir.

- **Mekanizma:** SM içindeki bir ALU register'ında veya L2 cache veri yolunda bellek ECC'si tarafından yakalanamayan tek bir bit ters döner ($0 \to 1$).
- **Etki:** İşletim sistemi veya CUDA sürücüsü hiçbir hata vermez. Ancak hesaplanan matris çarpımında bir değer $1.0$ yerine $10^9$ olur. Bu anomali geriye yayılımda loss patlamasına (NaN / Inf loss) veya modelin dil yeteneklerinin sessizce çökmesine yol açar.
- **Teşhis:** Düzenli aralıklarla her düğümde deterministik küçük bir GEMM kernel'ı çalıştırılarak referans sağlama toplamı (checksum) ile kıyaslanır; gradient normlarında ani sıçramalar kaydedildiğinde o rank'in GPU çekirdekleri izole edilir.

### 2.5. Checkpoint aralığının kaybı nasıl hesaplanır?

Bir kümede arızalar ve checkpoint'ler yüzünden kaybedilen toplam zaman oranı ($W_{\mathrm{total}}$) üç bileşenin toplamıdır:

$$
W_{\mathrm{total}} = \frac{\delta}{\tau} + \frac{\tau}{2M} + \frac{R}{M}
$$

Burada:
- $\frac{\delta}{\tau}$: Checkpoint yazmak için harcanan zaman oranı,
- $\frac{\tau}{2M}$: Bir arıza anında ortalama olarak son checkpoint'ten bu yana kaybedilen ve yeniden hesaplanması gereken iş oranı,
- $\frac{R}{M}$: Her arıza sonrasında kümenin yeniden başlatılması, ortamın ayağa kaldırılması ve ağırlıkların okunması için geçen kurtarma süresi ($R$).

Bu formül, depolama hızının ($\delta$) ve hızlı yeniden başlatmanın ($R$) küme verimliliği üzerindeki doğrudan matematiksel etkisini kanıtlar.

---

## 3. GPU utilization, MFU ve HFU farkı

Bir yapay zekâ altyapı mühendisinin yapabileceği en büyük hata `nvidia-smi` çıktısındaki %100 değerine bakıp sistemin verimli çalıştığını zannetmektir.

| Metrik | Ne Ölçer? | Tipik Yanıltıcı Durum | Ne Zaman Güvenilir? |
|---|---|---|---|
| **GPU Utilization (`nvidia-smi`)** | GPU çekirdeklerinin en az birinde zaman dilimi boyunca kernel çalıştı mı? | Bellek bekleyen, uncoalesced erişim yapan veya spin-lock bekleyen GPU %100 görünür | Asla tek başına yeterli bir performans metriği değildir |
| **Model FLOPs Utilization (MFU)** | Gerçekleşen yararlı teorik model FLOP'larının donanımın tepe kapasitesine oranı | Aktivasyon recomputation FLOP'larını saymaz; gerçek algoritma verimini gösterir | Dağıtık eğitim verimliliğini ölçen altın standart |
| **Hardware FLOPs Utilization (HFU)** | Donanımın fiziksel olarak yürüttüğü tüm FLOP'ların (recomputation dahil) tepe değere oranı | Aktivasyonları tekrar tekrar hesaplayarak HFU suni olarak şişirilebilir | Donanım çekirdeklerinin ne kadar doymuş olduğunu anlamak için |

### 3.1. GPU utilization

`nvidia-smi` tarafından raporlanan kullanım yüzdesi bir örnekleme (sampling) metriğidir. Son 1 saniyelik zaman penceresinde, GPU üzerindeki 132 SM'den yalnızca bir tanesinde 1 mikrosaniyelik bir CUDA çekirdeği aktif olduysa, o örnekleme aralığı "aktif" sayılır.
GPU belleğe erişmek için 500 döngü boyunca SM'leri boşta bekletse dahi `nvidia-smi` **%100 Utilization** gösterebilir. Bu metrik bir hız göstergesi değil, yalnızca GPU'nun tamamen uykuda olmadığını belirten bir varlık sinyalidir.

### 3.2. MFU

**Model FLOPs Utilization (MFU)**, bir eğitim sürecinin teorik olarak minimum kaç kayan nokta işlemi (FLOP) gerektirdiğini hesaplayıp, bunu donanımın teorik tepe hesaplama kapasitesine bölen orandır:

$$
\mathrm{MFU} = \frac{\text{Saniyede İşlenen Token Sayısı} \times \text{Token Başına Model FLOP'u}}{N_{\mathrm{GPU}} \times \text{GPU Başına Tepe FLOP/s}}
$$

Standart bir yoğun Transformer modelinin eğitiminde (aktivasyon recomputation hariç), her bir parametre ($P$) için token başına yaklaşık **$6P$ FLOP** harcanır ($2P$ forward geçişinde, $4P$ backward geçişinde):
$$\text{FLOPs per token} \approx 6P$$
MFU, algoritmik verimliliği doğrudan yansıttığı için endüstrinin kabul ettiği en dürüst kıyaslama metriğidir.

### 3.3. HFU

**Hardware FLOPs Utilization (HFU)** ise donanımın fiziksel Tensor Core'larında fiilen dönen tüm matris işlemlerini hesaba katar.
Eğer bellek yetersizliği nedeniyle **Activation Checkpointing (Recomputation)** kullanıyorsanız, backward sırasında bazı katmanların forward hesaplaması baştan bir kez daha yapılır ($+2P$ ek FLOP).
Bu durumda donanım token başına $6P$ yerine $8P$ FLOP hesaplar:
$$\mathrm{HFU} \approx \frac{8P}{6P} \times \mathrm{MFU} \approx 1.33 \times \mathrm{MFU}$$
HFU donanımın matematiksel sınırlarına ne kadar yaklaştığını gösterir; ancak bu hesaplamaların bir kısmı mimari bellek yetersizliğinin faturasıdır.

### 3.4. %38 MFU kontrol hesabı

Sektörde 1.024 adet NVIDIA H100 GPU ile 70B model eğiten bir ekip **%38 MFU** elde ettiğinde bu bir başarı mıdır?
H100 SXM5 GPU'nun yoğun BF16 tepe hesaplama gücü $989.5 \approx 990\text{ TFLOP/s}$'dir. 1.024 GPU'luk bir kümenin teorik tepe kapasitesi:
$$1.024 \times 990\text{ TFLOP/s} \approx 1.014\text{ PFLOP/s} = 1.014\text{ EFLOP/s}$$
%38 MFU elde edildiğinde, saniyede fiilen faydalı modele dönüşen hesaplama:
$$\text{Faydalı Güç} = 0.38 \times 1.014\text{ EFLOP/s} \approx 0.385\text{ EFLOP/s}$$

<!-- restored-svg:tr_full:46 -->
<svg viewBox="0 0 720 226" role="img" aria-label="1.024 H100&#x27;ün yaklaşık 1,014 EFLOP/s&#x27;lik teorik BF16 yoğun tepe değerinin çubuğu. %38&#x27;i, yaklaşık 0,385 EFLOP/s, faydalı model FLOP&#x27;una dönüşür. Kalan %62 faydalı model hesabına dönüşmeyen potansiyeldir; GPU&#x27;ların %62&#x27;sinin kapalı olduğu anlamına gelmez. Payları profil çıkarılmadan bilinmeyen olası nedenler: memory-bound kernel&#x27;lar, kolektif beklemeler, pipeline bubble, küçük veya birleşmemiş kernel&#x27;lar, HFU&#x27;da sayılıp MFU&#x27;da sayılmayan recomputation ve veri ya da checkpoint duraklamaları." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">%38 MFU&#x27;da 1.024 H100: %62 nedir, ne değildir?</text>
<text x="16" y="50" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">teorik tepe: 1.024 × 990 TFLOP/s ≈ 1,014 EFLOP/s (BF16 yoğun)</text>
<rect x="16" y="60" width="261.44" height="40" rx="4" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="146.72" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">faydalı model FLOP&#x27;u</text>
<text x="146.72" y="94" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%38 ≈ 0,385 EFLOP/s</text>
<rect x="277.44" y="60" width="426.56" height="40" rx="4" style="fill:var(--c-text-mute);fill-opacity:.06;stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="490.71999999999997" y="78" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">%62 faydalı model hesabına dönüşmeyen</text>
<text x="490.71999999999997" y="94" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">GPU&#x27;ların %62&#x27;si kapalı DEMEK DEĞİL</text>
<text x="277.44" y="124" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">olası nedenler; paylar profil çıkarılmadan bilinmez:</text>
<rect x="277.44" y="134" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.44" y="148.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">memory-bound kernel</text>
<rect x="491.44" y="134" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594.44" y="148.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">kolektif bekleme</text>
<rect x="277.44" y="162" width="206" height="22" rx="4" style="fill:var(--c-warn);fill-opacity:.1;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.44" y="176.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">pipeline bubble</text>
<rect x="491.44" y="162" width="206" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:.1;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="594.44" y="176.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">küçük / birleşmemiş kernel</text>
<rect x="277.44" y="190" width="206" height="22" rx="4" style="fill:var(--c-text-mute);fill-opacity:.1;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="380.44" y="204.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">recomputation (HFU&#x27;da, MFU&#x27;da değil)</text>
<rect x="491.44" y="190" width="206" height="22" rx="4" style="fill:var(--c-danger);fill-opacity:.1;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="594.44" y="204.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">veri / checkpoint duraklaması</text>
</svg>

Peki geriye kalan **%62'lik devasa potansiyel nereye gitmiştir?** GPU'ların %62'si kapalı mıdır?
Hayır! Kalan %62'lik pay şu 6 fiziksel faktör arasında dağılır:

1. **Bellek Sınırındaki (Memory-Bound) Katmanlar (%10–12):** LayerNorm, RMSNorm, RoPE, Softmax ve Bias ekleme gibi işlemler düşük aritmetik yoğunluğa sahiptir; Tensor Core'ları değil HBM bant genişliğini beklerler.
2. **Örtülemeyen Kolektif İletişim (%12–15):** TP katman içi AllReduce'ları, PP sınır geçişleri ve backward altına sığmayan FSDP ReduceScatter kuyrukları.
3. **Pipeline Parallelism Kabarcığı (%8–12):** Stage'ler arasındaki 1F1B microbatch doldurma ve boşaltma ($F_{\mathrm{bubble}}$) süreleri.
4. **HFU ile MFU Arasındaki Recomputation Payı (%10–14):** Bellekten tasarruf etmek için tekrar hesaplanan aktivasyon GEMM'leri (HFU'da sayılır, MFU formülünde sayılmaz).
5. **Dataloader, Checkpoint ve Straggler Jitter'ı (%5–8):** Disk okuma duraklamaları, snapshot durmaları ve termal olarak yavaşlayan tekil rank beklemeleri.
6. **Tensor Core Alt-Doygunluğu (%4–6):** Matris boyutlarının ($M, N, K$) 16 veya 32'nin tam katı olmamasından kaynaklanan döşeme (tiling) verimsizlikleri.

Bu muhasebe yapıldığında %38 MFU'nun aslında son derece rafine ve optimize bir mühendislik başarısı olduğu anlaşılır.

### 3.5. MFU hesabı hangi varsayımlara duyarlıdır?

MFU hesaplarken yapılan üç yaygın hata ve duyarlılık noktası:
1. **Sparsity (Seyreltiklik) Tuzağı:** NVIDIA belgelerinde H100 için bazen $1.979\text{ TFLOP/s}$ değeri görürsünüz. Bu değer 2:4 yapısal seyreltiklik (structural sparsity) varsayar. Yoğun LLM eğitiminde bu özellik kullanılamaz; daima yoğun tepe değer olan $989.5\text{ TFLOP/s}$ baz alınmalıdır.
2. **Padding Token'ları:** Eğer dataset içindeki padding token'ları hesaplamaya dahil edilip FLOP sayısına eklenirse MFU suni olarak yüksek çıkar; yalnızca faydalı içerik token'ları sayılmalıdır.
3. **Dikkat Matrisi Formülü:** Uzun dizilerde ($S > 8\text{K}$) Self-Attention katmanının gerektirdiği ek $4 \times L \times H \times S$ FLOP'u ihmal etmek payı küçülterek MFU'yu olduğundan %3–5 daha düşük gösterebilir.

---

## 4. Düşük MFU nasıl teşhis edilir?

Bir kümede MFU hedeflenen seviyenin altına düştüğünde panik halinde rastgele parametre değiştirmek durumu daha da kötüleştirir. Sistematik bir teşhis protokolü izlenmelidir.

### 4.1. Önce karşılaştırmayı sabitleyin

Herhangi bir teşhise başlamadan önce sistemin temel değişkenlerini dondurun:
- Global batch size ve microbatch size sabitlenmelidir.
- Sequence length ve veri tipi (BF16) kilitlenmelidir.
- Optimizer (AdamW) ve paralel plan ($TP, PP, DP$) değiştirilmemelidir.
- Test koşusu en az 100 adım sürmeli; ilk 10 adım (warmup) analizden çıkarılmalıdır.

### 4.2. Beklemeyi sınıflandırın

Bir GPU'nun step süresi içindeki durumunu 4 ana bekleme sınıfına ayırın:

1. **Compute Stall:** GPU matris çarpıyor ama kernel verimsiz (küçük matris, register spilling).
2. **Communication Stall:** GPU bir NCCL kolektifinde (AllReduce, AllToAll) başka bir rank'i bekliyor.
3. **Memory Bandwidth Stall:** Tensor Core'lar boş, SM'ler HBM'den veri gelmesini bekliyor.
4. **I/O & Input Stall:** GPU hazır ama Dataloader yeni batch'i HBM'e teslim edememiş.

<!-- restored-svg:tr_full:41 -->
<svg viewBox="0 0 720 252" role="img" aria-label="Step süresi evrelerinin zaman çizelgesi. Kritik yolda: forward, backward, açıkta kalan iletişim dilimi, optimizer ve sonraki adım. Altında arka plan işleri: gradyan AllReduce bucket&#x27;ları backward&#x27;ın son kısmıyla örtüşür, veri prefetch hesap sürerken sonraki batch&#x27;i yükler, asenkron checkpoint kısa bir snapshot duraklamasından sonra durumu kalıcı yazar. Backward&#x27;ın altına sığmayan iletişim kritik yola düşer." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sp-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Step süresinin evreleri: kritik yol ve yanına gizlenebilenler</text>
<text x="16" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">kritik yol</text>
<rect x="250" y="46" width="135" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="317.5" y="64.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">forward</text>
<rect x="385" y="46" width="180" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="475.0" y="64.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">backward</text>
<rect x="565" y="46" width="45" height="28" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="587.5" y="63.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">açıkta</text>
<rect x="610" y="46" width="45" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="632.5" y="63.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">optim.</text>
<rect x="655" y="46" width="49.5" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="679.75" y="63.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">sonraki</text>
<text x="16" y="103" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">grad AllReduce</text>
<text x="16" y="118" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">bucket&#x27;lar geç backward&#x27;la örtüşür</text>
<rect x="475" y="92" width="135" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="16" y="147" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">veri prefetch</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">hesap sürerken sonraki batch</text>
<rect x="272.5" y="136" width="180" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="16" y="191" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">async checkpoint</text>
<text x="16" y="206" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">kısa snapshot sonrası kalıcı yazma</text>
<rect x="637" y="180" width="67.5" height="24" rx="4" style="fill:var(--c-danger);fill-opacity:.22;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="565" y1="40" x2="565" y2="226" style="stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="704" y="240" text-anchor="end" style="fill:var(--c-danger);font-size:11px">backward&#x27;ın altına sığmayan iletişim kritik yola düşer</text>
</svg>

<!-- restored-svg:tr_full:47 -->
<svg viewBox="0 0 720 254" role="img" aria-label="8 ms hesap ve 5 ms iletişimi olan bir katman. Seri çalışırsa 13 ms sürer. Bağımsız iletişim hesapla tam örtüşürse kritik yol en az max(8, 5) yani 8 ms olur. Bu, step süresinde %38 kısalma ya da aynı GPU&#x27;larda 1,63 kat throughput demektir; 5 ms&#x27;lik iletişim yine yaşanır, yalnızca kritik yolun dışına çıkar. Bağımlılıklar, ortak SM ve HBM kullanımı ve ağ rekabeti yüzünden gerçek örtüşme kısmidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="om-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="om-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">8 ms hesap + 5 ms iletişim: seri ve tam örtüşmeli</text>
<text x="16" y="74" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">seri</text>
<rect x="150" y="58" width="288" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="76.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">hesap 8 ms</text>
<rect x="438" y="58" width="180" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="528.0" y="76.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">iletişim 5 ms</text>
<text x="626" y="77" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">= 13 ms</text>
<text x="16" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">örtüşmeli</text>
<rect x="150" y="120" width="288" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="294.0" y="138.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">hesap 8 ms</text>
<rect x="222" y="152" width="180" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="312.0" y="167.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">iletişim 5 ms (gizli)</text>
<text x="446" y="139" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">≥ max(8, 5) = 8 ms</text>
<line x1="438" y1="50" x2="438" y2="184" style="stroke:var(--c-success);stroke-width:1.2;stroke-dasharray:4 3"/>
<line x1="618" y1="50" x2="618" y2="96" style="stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="16" y="196" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.2"/>
<text x="28" y="216" text-anchor="start" style="fill:var(--c-text);font-size:12px">Step süresi −%38 (13 → 8 ms) = aynı GPU&#x27;larda 1,63× throughput; 5 ms yine yaşanır, yalnızca kritik yoldan çıkar.</text>
<text x="28" y="233" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Gerçek örtüşme kısmidir: bağımlılıklar, ortak SM/HBM ve ağ rekabeti kazancı azaltır.</text>
</svg>

### 4.3. Üç gözlem katmanı

Sağlıklı bir teşhis, veriyi üç farklı telemetri katmanından eşzamanlı toplayarak korelasyon kurar:

1. **Katman 1 — Donanım ve Sürücü Telemetrisi (DCGM):**
   - GPU sıcaklıkları ve termal kısıtlama (throttling bayrakları).
   - Çekirdek ve bellek saat hızları (clock throttling).
   - PCIe Gen5 link hızı ve genişliği ($x16$ çalışması gerekirken $x8$'e düşen linkler).
   - DRAM tek-bit ve çift-bit ECC hataları.
2. **Katman 2 — Ağ ve Kumaş Telemetrisi:**
   - InfiniBand port sayaçları: `PortXmitWait` (tıkanıklık), `SymbolError` (kablo arızası).
   - RoCE Ethernet sayaçları: PFC Pause frame sayıları, ECN işaretli paketler, buffer taşmaları.
   - NCCL Telemetrisi: Kolektif gecikme dağılımları ve straggler tespiti.
3. **Katman 3 — Framework ve Profiler İzi:**
   - PyTorch Profiler / NVIDIA Nsight Systems (nsys) zaman çizelgeleri.
   - CUDA stream izleri, kernel süreleri, exposed communication blokları.

<!-- restored-svg:tr_full:48 -->
<svg viewBox="0 0 720 302" role="img" aria-label="Gözlemlenebilirlik yığını. Dört sinyal kaynağı: DCGM&#x27;den güç, ECC, Xid ve saat gibi GPU ve sunucu telemetrisi; NIC, RDMA sayaçları ve NCCL RAS&#x27;tan ağ telemetrisi; step süresi, MFU, loss ve restart gibi eğitim işi metrikleri; TTFT, TPOT, kuyruk ve KV kullanımı gibi serving metrikleri. Bunlar DCGM exporter ve Prometheus ya da OpenTelemetry üzerinden, istek başına tek trace_id ile ilişkilendirilen metrik ve trace deposuna akar; oradan aksiyon doğar: SLO burn-rate alarmı, düğüm karantinası, kapasite planlama ve faydalı token maliyeti. DCGM exporter yalnızca GPU alanlarını verir; geri kalanı ayrıca enstrümante edilmelidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ob-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ob-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Gözlemlenebilirlik: dört sinyal kaynağı, ilişkilendirmek için tek yer</text>
<rect x="16" y="44" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="126.0" y="66.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">GPU &amp; sunucu</text>
<text x="126.0" y="80.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">DCGM: güç, ECC, Xid, saat</text>
<path d="M236 68 C 262 68, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="102" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="126.0" y="124.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Ağ</text>
<text x="126.0" y="138.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NIC, RDMA sayaçları, NCCL RAS</text>
<path d="M236 126 C 262 126, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="160" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="126.0" y="182.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Eğitim işi</text>
<text x="126.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">step süresi, MFU, loss, restart</text>
<path d="M236 184 C 262 184, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="218" width="220" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="126.0" y="240.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Serving</text>
<text x="126.0" y="254.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">TTFT, TPOT, kuyruk, KV kullanımı</text>
<path d="M236 242 C 262 242, 262 160, 290 160" marker-end="url(#ob-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="292" y="116" width="190" height="88" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="387.0" y="143.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Metrik + trace</text>
<text x="387.0" y="158.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">DCGM exporter →</text>
<text x="387.0" y="172.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Prometheus / OpenTelemetry</text>
<text x="387.0" y="187.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">istek başına tek trace_id</text>
<line x1="482" y1="160" x2="512" y2="160" marker-end="url(#ob-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="514" y="106" width="190" height="108" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="609.0" y="136.25" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Aksiyon</text>
<text x="609.0" y="150.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">SLO burn-rate alarmı</text>
<text x="609.0" y="165.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">düğüm karantinası</text>
<text x="609.0" y="179.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kapasite planlama</text>
<text x="609.0" y="194.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">faydalı token maliyeti</text>
<text x="16" y="290" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DCGM exporter yalnızca GPU alanlarını verir; ağ, uygulama ve kullanıcı deneyimi metrikleri ayrıca enstrümante edilmelidir.</text>
</svg>

### 4.4. Tek hipotez, tek deney

Mühendislik disiplininin temeli: **Aynı anda birden fazla değişkeni asla değiştirmeyin.**
- *Hipotez:* "Step süresinin uzun olmasının sebebi dataloader okuma gecikmesidir."
- *Deney:* Yalnızca dataloader'ı sentetik in-memory moda alın. Ağ veya model ayarlarına dokunmayın.
- *Karar:* Step süresi düzelirse hipotez doğrulanmıştır; düzelmezse hipotez elenir ve bir sonraki katmana geçilir.

### 4.5. Teşhis ağacının sırası neden önemlidir?

Teşhis ağacı daima **aşağıdan yukarıya (fiziksel katmandan yazılıma)** doğru işletilmelidir:

```
[ Donanım Sağlığı (DCGM / ECC) ] -> [ Ağ Kumaşı (RDMA / Link) ] -> [ I/O & Dataloader ] -> [ Kernel & MFU ]
```

Neden?
Eğer bir GPU'nun soğutma bloğu gevşemişse ve termal kısıtlama yüzünden çekirdek frekansını $1.980\text{ MHz}$'den $800\text{ MHz}$'e düşürüyorsa, siz günlerce PyTorch kodunda FlashAttention kernel'ı optimize etmeye çalışsanız da hiçbir sonuç alamazsınız! Önce fiziksel temelin sağlam olduğu kanıtlanmalıdır.

<!-- restored-svg:tr_full:49 -->
<svg viewBox="0 0 720 364" role="img" aria-label="Kök neden teşhis ağacı. SLO veya MFU&#x27;nun düştüğü belirtisinden başlayın. Önce hata oranı arttı mı diye sorun: Xid, ECC, yeniden başlatma. Evetse bunu donanım veya sürücü arızası sayıp düğümü karantinaya alın. Hayırsa profil zaman çizelgesinde beklemenin yerini bulun: kuyruktaysa kabul politikası ve havuz kapasitesi; GPU hesabındaysa kernel belleği ve batch boyutu; ağdaysa NCCL kolektifleri ve NIC sayaçları; depolama ve I/O&#x27;daysa dataloader ve checkpoint yazımı. Ardından tek değişkenle tek hipotezi test edin." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dt-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dt-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Kök neden teşhisi: bir şey bozuk mu, yoksa bekliyor mu?</text>
<rect x="16" y="128" width="150" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="91.0" y="154.0" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">SLO / MFU düştü</text>
<text x="91.0" y="169.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">belirti</text>
<rect x="212" y="52" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="307.0" y="78.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Hata oranı arttı mı?</text>
<text x="307.0" y="92.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Xid, ECC, yeniden başlatma</text>
<rect x="212" y="204" width="190" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="307.0" y="230.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Bekleme nerede?</text>
<text x="307.0" y="244.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">profil zaman çizelgesi</text>
<path d="M166 146 C 190 146, 190 80, 210 80" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M166 166 C 190 166, 190 232, 210 232" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="52" width="254" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="577.0" y="78.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Donanım / sürücü arızası</text>
<text x="577.0" y="92.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">düğümü karantinaya al (§17.5)</text>
<line x1="402" y1="80" x2="448" y2="80" marker-end="url(#dt-b)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<text x="425" y="74" text-anchor="middle" style="fill:var(--c-danger);font-size:10.5px;font-weight:600">evet</text>
<path d="M307 108 V198" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="315" y="160" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">hayır</text>
<rect x="450" y="136" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="154.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Kuyruk</text>
<text x="577.0" y="169.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kabul politikası, havuz kapasitesi</text>
<path d="M402 232 C 426 232, 426 157, 448 157" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="186" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="577.0" y="204.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU hesap</text>
<text x="577.0" y="219.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">kernel belleği, batch boyutu</text>
<path d="M402 232 C 426 232, 426 207, 448 207" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="236" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="577.0" y="254.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Ağ</text>
<text x="577.0" y="269.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NCCL kolektifleri, NIC sayaçları</text>
<path d="M402 232 C 426 232, 426 257, 448 257" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="450" y="286" width="254" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="577.0" y="304.75" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">Depolama / I/O</text>
<text x="577.0" y="319.25" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">dataloader, checkpoint yazımı</text>
<path d="M402 232 C 426 232, 426 307, 448 307" marker-end="url(#dt-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="16" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Sonra her seferinde aynı disiplin: tek hipotez, tek değişken, önce/sonra ölçümü.</text>
</svg>

---

## 5. Beş belirti, beş deney

Üretim ortamlarında en sık karşılaşılan 5 patolojik belirti ve bunların kesin kök neden tespit deneyleri:

### 5.1. `nvidia-smi` %100, token/s düşük
- **Kök Neden:** Aritmetik yoğunluğu aşırı düşük kernel'lar (örneğin aşırı küçük microbatch size yüzünden Tensor Core'ların devreye girememesi) veya birleşmemiş (unfused) çok sayıda küçük element-wise operasyon.
- **Deney:** Nsight Systems ile profil alın. Kernel sürelerine bakın. Eğer SM'lerin vaktinin çoğu `at::native::elementwise_kernel` gibi bellek darboğazlı ufak fonksiyonlarda geçiyorsa, `torch.compile()` kullanarak kernel fusion uygulayın ve microbatch boyutunu artırın.

### 5.2. AllReduce beklenenden yavaş
- **Kök Neden:** Gruptaki rank'lerden biri yavaşlamıştır (straggler) veya sunucu içi GPU-NIC eşleşmesi yanlış NUMA düğümüne düşerek UPI köprüsünü boğmaktadır.
- **Deney:** `NCCL_DEBUG=INFO` ve `NCCL_TOPO_DUMP=1` ortam değişkenlerini açarak GPU-NIC afinitesini denetleyin. Tek tek her düğüm çifti arasında `nccl-tests` çalıştırarak tek bir raydaki veya switch portundaki bant genişliği düşüşünü izole edin.

### 5.3. Ortalama TTFT iyi, p99 kötü
- **Kök Neden:** Inference servislerinde yeni istekler geldiğinde KV cache belleği tahsisinde yaşanan bellek kilitlenmeleri (allocation lock) veya sunucu kuyruğuna aynı anda yığılan devasa prompt'ların prefill fazında decode işlemlerini aç bırakması.
- **Deney:** Chunked Prefill özelliğini aktif edin (uzun prompt'ları dilimler halinde işleme) ve vLLM PagedAttention bellek tahsis blok boyutunu 16'dan 32'ye optimize edin.

### 5.4. Checkpoint dakikalar sürüyor
- **Kök Neden:** Tüm rank'lerin aynı anda paylaşımlı depolama sisteminin aynı dizinine senkron yazmaya çalışması sonucu metadata sunucusunun kilitlenmesi veya depolama ağının doyması.
- **Deney:** Asenkron iki kademeli checkpointing'e geçin; tensörleri önce yerel sunucu NVMe'lerine yazın (`/tmp` veya yerel mount), ardından arka planda merkezi depolamaya aktarın.

### 5.5. Birkaç GPU sürekli geride
- **Kök Neden:** Termal kısıtlama (Thermal Throttling) veya PCIe bağlantısının $x16$'dan $x8$ veya $x4$ moduna düşmesi.
- **Deney:** `nvidia-smi -q -d PERFORMANCE,CLOCK` komutunu çalıştırarak `Clocks Throttle Reasons` alanını inceleyin. `SW Thermal Slowdown` veya `PCIe Link Width: 8x` görüyorsanız doğrudan veri merkezi teknisyenine bilet açarak donanım müdahalesi isteyin.

### 5.6. Vaka: MFU %42’den %22’ye neden düştü?

**Gerçek Olay İncelemesi (RCA):**
1.024 GPU'lu bir kümede haftalardır %42 MFU ile devam eden bir eğitim koşusu, planlı bir bakım sonrasında aniden %22 MFU'ya çakıldı. Token üretim hızı yarı yarıya düştü.

Adım adım dedektiflik:
1. *Donanım Taraması:* DCGM testleri temiz çıktı; hiçbir GPU termal throttling yemiyordu.
2. *Dataloader Taraması:* Sentetik dataloader deneyi yapıldı; MFU hâlâ %22'de kaldı (depolama masum).
3. *Ağ Analizi:* NCCL profilinde backward sonundaki AllReduce süresinin $12\text{ ms}$'den $48\text{ ms}$'ye fırladığı görüldü.
4. *Kök Neden:* Bakım sırasında bir switch rafının yeniden başlatılmasıyla, 16 adet ConnectX-7 NIC'inin bağlı olduğu leaf switch portlarında MTU ayarının $9.000$ (Jumbo Frames) yerine varsayılan $1.500$ bayta döndüğü keşfedildi! Paketler parçalanıyor (packet fragmentation), ağda paket patlaması yaşanıyor ve RoCE PFC kilitlenmeleri tetikleniyordu.
5. *Çözüm:* Switch portlarında MTU 9000 düzeltildi; MFU derhal %42'ye geri yükseldi.

### 5.7. Olay müdahalesinin kısa şablonu

Üretim kümesinde MFU düşüşü veya arıza alarmı çaldığında şu 4 adımlı müdahale protokolünü uygulayın:
1. **İzole Et (Contain):** Arızalı veya şüpheli düğümleri tespit edip karantinaya alın; sağlam düğümlerle son temiz checkpoint'ten eğitimi devam ettirin.
2. **Koru (Preserve):** Düğümü hemen yeniden başlatmayın! DCGM loglarını, `dmesg`, NCCL debug dökümünü ve switch port sayaçlarını kaydedin.
3. **Yeniden Üret (Reproduce):** Karantina altındaki izole düğümde sentetik `nccl-tests` ve `cuBLAS` stres testleri koşturarak hatayı tetikleyin.
4. **Onar ve Doğrula (Remediate & Verify):** Parça değişimi veya konfigürasyon düzeltmesi yapıldıktan sonra seviye 1–4 kabul testlerinden geçirip havuza kademeli olarak iade edin.

---

## 6. Maliyet ve enerji hangi metriğe bağlanır?

Dağıtık sistemlerde performans sadece saniyedeki token hızıyla ölçülemez; o token'ların şirkete kaç dolara ve kaç kilowatt-saate mal olduğu FinOps disiplininin konusudur.

### 6.1. GPU-saat tek başına yeterli değildir

Geleneksel faturalandırma "GPU-Saat" üzerinden yapılır. Ancak bu metrik kördür.
- Mühendislik Ekibi A: 1.000 GPU-saat harcayarak %20 MFU ile 100 milyar token eğitti.
- Mühendislik Ekibi B: 1.000 GPU-saat harcayarak %40 MFU ile 200 milyar token eğitti.
Her iki ekibin bulut faturası kuruşu kuruşuna aynıdır! Ancak Ekip B'nin elde ettiği **Yararlı Token Başına Maliyeti** tam yarı yarıyadır. Bu yüzden altyapı başarısı harcanan GPU saatiyle değil, üretilen faydalı token/dolar oranıyla ölçülür.

### 6.2. TCO bileşenleri

Bir AI süper bilgisayarının Toplam Sahip Olma Maliyeti (Total Cost of Ownership - TCO) sadece GPU yongalarının satın alma bedeli (CapEx) değildir:

$$
\mathrm{TCO} = \mathrm{CapEx}_{\mathrm{GPU+Servers}} + \mathrm{CapEx}_{\mathrm{Network+Storage}} + \mathrm{OpEx}_{\mathrm{Energy}} \times \mathrm{PUE} + \mathrm{OpEx}_{\mathrm{Facility+Cooling}}
$$

- **Enerji ve PUE Çarpanı:** 1.024 adet H100 GPU sunucusu soğutma ve ağ ile birlikte yaklaşık **$1.5 - 2.0\text{ Megawatt}$** sürekli güç çeker. Veri merkezinin Güç Kullanım Verimliliği (PUE) 1.3 ise, her 1 MW'lık GPU gücü için 0.3 MW soğutma enerjisi harcanır.
- 3 yıllık bir kullanım döngüsünde harcanan elektrik ve soğutma faturası, donanımın ilk satın alma maliyetini aşabilir!

### 6.3. Tokens per joule

Sürdürülebilir yapay zekâ mühendisliğinin en saf verimlilik metriği Joule başına üretilen token sayısıdır:

$$
\text{Energy Efficiency} = \frac{\text{Toplam Üretilen Token Sayısı}}{\text{Tüketilen Toplam Enerji (Joule)}}
$$

MFU'yu %25'ten %40'a çıkarmak, aynı modelin eğitiminde harcanan megawatt-saatlerce elektrik enerjisini ve veri merkezinin karbon ayak izini doğrudan %37.5 azaltır.

### 6.4. Yüksek MFU her zaman en iyi sonuç değildir

Bir mühendisin düşebileceği son tuzak, MFU metriğini fetişleştirmektir.
MFU'yu artırmak için global batch size'ı yapay olarak devasa boyutlara çıkarabilirsiniz (örneğin 4 milyon token'dan 32 milyon token'a). Matrisler büyüdüğü için Tensor Core verimliliği artar ve MFU %50'ye fırlayabilir.
Ancak **Gradient Noise Scale** sınırını aştığınız için model her adımda çok az şey öğrenir; kayıp fonksiyonunun (loss) yakınsaması için gereken toplam adım sayısı katlanır. MFU yüksek çıksa bile modeli eğitmek haftalarca daha uzun sürer ve fatura katlanır! Asıl amaç yüksek MFU değil, **en düşük toplam maliyet ve enerjiyle hedef loss değerine ulaşmaktır**.

---

## 7. Sistem tasarımı için yedi temel soru

### 7.1. GPU utilization ve MFU farkı?

`nvidia-smi` tarafından raporlanan GPU Utilization, belirli bir zaman penceresinde GPU çekirdeklerinde en az bir CUDA kernel'ının aktif olup olmadığını gösteren bir varlık metriğidir. Bellek bekleyen, spin-lock'ta dönen veya uncoalesced bellek erişimiyle Tensor Core'ları boş bırakan bir GPU da %100 utilization gösterebilir.

MFU (Model FLOPs Utilization) ise modelin mimarisine göre yapılması gereken faydalı teorik kayan nokta işlemlerinin, donanımın teorik tepe hesaplama gücüne bölünmesiyle elde edilen gerçek algoritmik verimlilik oranıdır. Birinde motorun çalışıp çalışmadığına, diğerinde tekerleklerin arabayı ne hızla götürdüğüne bakılır.

### 7.2. MFU neden %100 olamaz?

Bir Transformer modelinde MFU'nun %100'e ulaşmasını engelleyen 5 temel fiziksel sınır vardır:
1. LayerNorm, Softmax ve RoPE gibi katmanlar aritmetik yoğunluğu düşük bellek-darboğazlı (memory-bound) operasyonlardır; Tensor Core'ları doyuramazlar.
2. Dağıtık eğitimde katman içi TP AllReduce'ları, PP kabarcıkları ve ağ senkronizasyon bariyerleri hesaplamayı duraklatır.
3. Donanımın SRAM ve L2 cache kapasiteleri sınırlı olduğu için veri HBM'den döşeme (tile) halinde taşınır; hafıza bant genişliği gecikmesi yaşanır.
4. Matris boyutları donanım Tensor Core döşeme boyutlarının (örneğin 16x16x16) tam katı olmadığında padding ve artık hesaplama israfı oluşur.
5. Dataloader ve checkpoint I/O duraklamaları kritik yola düşebilir.

### 7.3. HFU neden MFU’dan yüksek olabilir?

HFU (Hardware FLOPs Utilization), donanım yongalarında fiilen dönen tüm kayan nokta işlemlerini sayar. Eğer model tek bir GPU'nun belleğine sığmadığı için **Activation Recomputation (Checkpointing)** kullanıyorsanız, ara aktivasyonlar bellekte saklanmaz; backward geçişinde katmanların forward hesaplaması bir kez daha baştan çalıştırılır.

Bu ek forward hesaplaması Tensor Core'larda fiziksel FLOP üretir ve HFU'yu artırır; ancak bu işlemler modele yeni bir matematiksel fayda sağlamadığından MFU payına eklenmez. Bu nedenle aktivasyon recomputation kullanılan sistemlerde HFU, MFU'dan %30–35 daha yüksek çıkar.

### 7.4. Checkpoint’i hızlandırmak neden iki kez kazandırır?

Checkpoint süresini örneğin 5 dakikadan 30 saniyeye indirmek sisteme iki farklı kanaldan kazanç sağlar:
1. **Doğrudan Yazma Tasarrufu:** Her checkpoint periyodunda GPU'ların boşta beklediği duraklama süresi doğrudan 4.5 dakika kısalır.
2. **Kayıp İş Azaltımı (Young-Daly Etkisi):** Checkpoint yazma süresi ($\delta$) kısaldığında, Young-Daly formülüne ($\tau_{\mathrm{opt}} \approx \sqrt{2\delta M}$) göre küme çok daha sık aralıklarla güvenle checkpoint alabilir. Daha sık checkpoint alındığında, beklenmedik bir arıza patladığında geriye dönülüp çöpe atılan ve yeniden hesaplanması gereken adım sayısı radikal biçimde azalır.

### 7.5. Büyük kümede ilk bakılacak metrik nedir?

Büyük bir kümede performans düşüşü incelenirken bakılacak ilk metrik asla uygulama logları değildir; ilk bakılacak sinyal **DCGM Donanım Hata ve Kısıtlama Sayaçlarıdır (Hardware Health & Throttling Flags)**.

Herhangi bir GPU'da termal kısıtlama (SW Thermal Slowdown), PCIe link genişliği düşmesi ($x16 \to x8$) veya bellek ECC çift-bit hatası varsa, o GPU tüm kümedeki kolektif bariyerleri kilitleyen bir straggler'a dönüşür. Donanım katmanının kusursuz olduğu doğrulanmadan üst katmanlardaki hiçbir metrik anlam ifade etmez.

### 7.6. Straggler nasıl bulunur?

Binlerce GPU arasından sürüyü yavaşlatan aylak rank'i (straggler) bulmak için üç aşamalı eleme uygulanır:
1. **Rank Bazlı Step Süresi Dağılımı:** Tüm rank'lerin forward ve backward tamamlanma süreleri histograma dökülür; P-99 kuyruğunda sürekli sağa sarkan uç değer rank'ler tespit edilir.
2. **NCCL Kernel Süresi Analizi:** Profiler'da AllReduce bariyerine ilk giren GPU'lar masumdur (onlar bekleyenlerdir); AllReduce'a en son katılan ve bariyeri başlatan GPU gerçek gecikmenin kaynağıdır.
3. **Fiziksel Telemetri Eşleştirmesi:** Geciken rank'in bulunduğu sunucunun DCGM saat frekansları ve InfiniBand switch portundaki `PortXmitWait` / FCS hata sayaçları incelenerek donanımsal arıza doğrulanır.

### 7.7. Benchmark neden production performansı değildir?

Sentetik benchmark'lar (örneğin sentetik tensörlerle koşturulan `nccl-tests` veya izole GEMM testleri) üretim ortamının karmaşık dinamiklerini yansıtmaz:
1. Benchmark'lar gerçek diskten veri okumaz; dosya sistemi metadata kilitlenmelerini ve dataloader gecikmesini simüle edemez.
2. Benchmark testleri genellikle 1–2 dakika sürer; saatler içinde biriken bellek parçalanmasını (fragmentation), termal ısınmayı ve sızıntıları (leak) göstermez.
3. Gerçek üretimde optimizer adımı, dinamik prompt uzunlukları, aktivasyon birikimi ve checkpoint dalgaları gibi heterojen fazlar bir arada yaşanır.

---

### Okuyucu artık hangi sorulara yanıt verebilir?

- Bir eğitim profilinde storage stall ile ağ kolektif beklemesini nasıl birbirinden ayırırsınız?
- Young-Daly formülünü kullanarak küme MTBF'sine göre en karlı checkpoint frekansını nasıl belirlersiniz?
- %38 MFU elde eden bir kümede kaybolan %62'lik hesaplama potansiyelini yöneticinize hangi 6 kalemle açıklarsınız?
- `nvidia-smi` %100 utilization gösterirken bir sistemin aslında neden sürünüyor olabileceğini nasıl kanıtlarsınız?
- Karantina döngüsü ile arızalı düğümlerin küme genelinde kilitlenme yaratmasını nasıl önlersiniz?

---

## Bütün hikâye altı satırda

- AI kümesi teşhisi tek bir soyut metrikle değil, beklemenin fiziksel adresini kanıtla bularak yapılır.
- Dataloader okuması metadata kilitlenmesiyle, checkpoint ise senkron yazma tsunamisiyle sistemi kilitler.
- Binlerce GPU'luk sistemlerde arıza kaçınılmazdır; optimum checkpoint sıklığı Young-Daly formülüyle yönetilir.
- GPU utilization sadece varlık gösterir; gerçek algoritmik verimlilik yalnızca MFU ile ölçülür.
- %38 MFU'nun arkasındaki eksik %62; memory-bound katmanlar, kolektifler, pipeline bubble ve recomputation payıdır.
- Teşhis daima fiziksel donanım katmanından başlayıp yazılıma doğru tek değişkenli deneylerle işletilir.

---

## Terimler sözlüğü

- **MFU (Model FLOPs Utilization):** Gerçekleştirilen faydalı algoritmik FLOP'ların donanımın teorik tepe gücüne oranı.
- **HFU (Hardware FLOPs Utilization):** Aktivasyon recomputation dahil donanımda fiilen dönen tüm FLOP'ların tepe güce oranı.
- **MTBF (Mean Time Between Failures):** Kümedeki ardışık iki donanım veya sistem arızası arasında geçen ortalama süre.
- **Young–Daly Formülü:** Checkpoint yazma süresi ve MTBF'ye bağlı olarak optimum checkpoint periyodunu hesaplayan denklem.
- **Straggler:** Termal kısıtlama veya donanım arızası nedeniyle yavaşlayıp tüm kolektif bariyeri kilitleyen GPU.
- **GPUDirect Storage (GDS):** Veriyi CPU RAM'ine kopyalamadan NVMe depolama ile GPU HBM arasında doğrudan DMA ile taşıyan teknoloji.
- **Silent Data Corruption (SDC):** İşletim sistemi veya donanım hata vermeden bir bitin ters dönmesiyle hesaplamanın bozulması.
- **Quarantine Loop:** Arızalı düğümleri otomatik tespit edip kümeden izole eden ve testten geçiren altyapı mekanizması.
- **Tokens per Joule:** Tüketilen birim enerji başına üretilen faydalı token sayısını ifade eden yeşil yapay zekâ metriği.

---

## Kaynaklar ve doğrulama bağlantıları

- Önceki yazı: [Yapay Zekâ Altyapısı (3): Çoklu GPU Ağları ve Dağıtık Eğitim](post.html?slug=coklu-gpu-aglari-ve-dagitik-egitim)
- Sonraki yazı: [Yapay Zekâ Altyapısı (5): Kurumsal AI Platform Mimarisi — Donanım, Maliyet ve Operasyon](post.html?slug=kurumsal-ai-platform-mimarisi)
- **[K1] Llama 3 405B Küme Deneyimi:** Meta, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783) (Bölüm 3: Altyapı, MTBF ve Hata Analizi).
- **[K2] Checkpointing Teorisi:** J. T. Daly, [*A higher order estimate of the optimum checkpoint interval for restart dumps*](https://doi.org/10.1016/j.jpdc.2004.02.004).
- **[K3] MFU ve Palm Analizi:** Google Research, [*PaLM: Scaling Language Modeling with Pathways*](https://arxiv.org/abs/2204.02311).
- **[K4] GPUDirect Storage:** NVIDIA, [*GPUDirect Storage Architecture Guide*](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html).
- **[K5] DCGM ve Gözlemlenebilirlik:** NVIDIA, [*DCGM User Guide*](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html).
