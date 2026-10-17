Bir GPU’nun veri sayfasında yüzlerce TFLOP/s veya PFLOP/s yazabilir; yine de kurduğunuz büyük dil modeli beklenenin çok altında bir hızla çalışabilir. Kâğıt üzerindeki tepe hesaplama gücüyle üretim ortamında elde edilen gerçek başarım arasındaki bu devasa uçurum, modern yapay zekâ mühendisliğinin en temel gerçeğidir: GPU performansı yalnızca aritmetik işlem birimlerinin hızından değil; verinin register, shared memory, önbellekler, HBM, NVLink, PCIe ve küme ağı boyunca ne kadar hızlı ve kesintisiz aktarıldığından doğar.

Bir veri merkezinde sekiz adet NVIDIA H100 GPU barındıran bir sunucunun önünde durduğunuzu hayal edin. Ekranda `nvidia-smi` komutunu çalıştırıyorsunuz; GPU kullanımı %100 görünüyor. Fakat modelin token üretim hızını ölçtüğünüzde, donanımın teorik kapasitesinin ancak %2'sini alabildiğinizi fark ediyorsunuz. GPU meşgul görünmektedir; fanlar son hızda dönmekte, güç kaynakları kilovatlarca enerji çekmektedir. Peki o zaman hesaplama nerede kaybolmaktadır? Cevap "GPU boşta bekliyor" kadar basit değildir; GPU silikonunun içindeki yürütme hatları bellekten bir sonraki sayıyı beklerken saat döngülerini boşa harcamaktadır.

**Bu yazının temel fikri:** Büyük ölçekli yapay zekâ, yalnızca hesaplama gücü problemi değildir. Aynı zamanda veriyi doğru yere, doğru zamanda, doğru bellek ve bağlantı katmanları boyunca en düşük gecikmeyle ulaştırma problemidir. Bir GPU'yu ve bir AI süper bilgisayarını anlamanın en kestirme yolu; bir tensörün depolamadan GPU'ya gelişini, Streaming Multiprocessor (SM) içinde işlenmesini ve diğer GPU'lara aktarılmasını adım adım izlemektir.

> **Serideki yerimiz:** Bu makale, beş bölümlük *Yapay Zekâ Altyapısı* serisinin kurucu birinci parçasıdır. Burada silikon seviyesindeki fiziksel haritayı (SM mimarisi, bellek hiyerarşisi, Roofline modeli ve düğüm içi ara bağlantıları) kuruyoruz. Sonraki bölümler bu haritayı sırasıyla iş yükleri (training/prefill/decode), küme ağları (InfiniBand/paralellik), arıza teşhisi (MFU/depolama) ve kurumsal platform tasarımı açısından ele alacaktır.

> **Bu yazının sonunda**
>
> - Thread, warp, SM ve Tensor Core arasındaki hiyerarşik ilişkiyi silikon düzeyinde açıklayabileceksiniz.
> - Bellek kapasitesi ile bellek bant genişliği arasındaki farkı ve coalesced bellek erişiminin önemini kavrayacaksınız.
> - Roofline modelini kullanarak bir iş yükünün neden compute-bound veya memory-bound olduğunu matematiksel olarak hesaplayabileceksiniz.
> - Training/prefill evresi ile autoregressive decode evresinin neden GPU donanımını taban tabana zıt rejimlerde çalıştırdığını kanıtlayabileceksiniz.
> - Bir sunucu düğümü içinde PCIe Gen5 ile NVLink 4 ara bağlantılarının görev dağılımını ve NUMA topolojisini sayısal verilerle savunabileceksiniz.

**Bu yazıda**

- [1. GPU hesaplamayı nasıl düzenler?](#1-gpu-hesaplamayı-nasıl-düzenler)
- [2. Bellek hiyerarşisi neden performansı belirler?](#2-bellek-hiyerarşisi-neden-performansı-belirler)
- [3. Roofline modeli: GPU gerçekte neyi bekliyor?](#3-roofline-modeli-gpu-gerçekte-neyi-bekliyor)
- [4. Bir düğümde GPU’lar nasıl bağlanır?](#4-bir-düğümde-gpular-nasıl-bağlanır)
- [5. Düğümden kümeye: Serideki yol haritamız](#5-düğümden-kümeye-serideki-yol-haritamız)
- [6. Sistem tasarımı için beş temel soru](#6-sistem-tasarımı-için-beş-temel-soru)
- [Okuyucu artık hangi sorulara kendi başına yanıt verebilir?](#okuyucu-artık-hangi-sorulara-kendi-başına-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. GPU hesaplamayı nasıl düzenler?

Modern yapay zekâ altyapısının kalbinde devasa matris işlemlerini paralel olarak yürüten GPU silikonu yer alır. Bir CPU çekirdeği; dallanması bol, tahmin edilmesi zor ve gecikmeye son derece duyarlı seri komut dizilerini en yüksek tek çekirdek performansıyla yürütmek üzere tasarlanmıştır. Bu yüzden CPU silikonunun büyük kısmı devasa önbelleklere (L2/L3 cache), karmaşık dal tahmin (branch prediction) mekanizmalarına ve sıra dışı yürütme (out-of-order execution) mantığına ayrılır.

GPU ise tamamen farklı bir felsefeyle üretilir: Silikon alanının ezici çoğunluğu veri işleme birimlerine (ALU ve Tensor Core) ayrılır. Binlerce benzer işlemi aynı anda koşturarak toplam hesaplama hacmini (throughput) maksimize eder. Fakat bu muazzam hesaplama gücünü verimli kullanabilmek için iş parçacıklarının son derece katı bir donanım hiyerarşisinde düzenlenmesi şarttır.

> [!TIP]
> **Mimari Harita: Bir AI Süper Bilgisayarının Büyük Resmi**
> Tek bir SM silikonundan 16.384 GPU'luk devasa bir süper bilgisayara uzanan 5 basamaklı uçtan uca mimari haritayı (A: SM'den SuperPOD'a Donanım Hiyerarşisi, B: Tek Bir Eğitim Adımında Veri Yolu, C: Bellek, Ağ ve Depolama Duvarları, D: Bağlantı Türlerine Göre Hız Merdiveni, E: 16k GPU LLaMA 3 405B Üretim Sistemi) tam boyutta ve yüksek çözünürlükte incelemek için:
> [👉 **Büyük Resmi Yeni Sayfada Açmak İçin Buraya Tıklayın (İnteraktif Mimari Posteri) ↗**](/assets/diagrams/ai-supercomputer-poster-tr.html)

Aşağıdaki özet tablo, yukarıdaki büyük resim haritasının ana katmanlarını ve bu makalede açacağımız donanım parametrelerini özetler:

| Katman | Fiziksel Kapsam | Kritik Donanım Metriği | Bu Makaledeki Yeri |
| :--- | :--- | :--- | :--- |
| **A · Donanım Hiyerarşisi** | SM → H100 Çipi → DGX Düğümü → SuperPOD | GPU başına 990 TFLOP/s BF16; 3,35 TB/s HBM3 | Bölüm 1 ve Bölüm 4 |
| **B · Veri Akış Hattı** | HBM → SRAM → Register → NVLink → Ağ | 3,35 TB/s (HBM) → 450 GB/s (NVLink) → 50 GB/s (Ağ) | Bölüm 2 ve Bölüm 4 |
| **C · Donanım Duvarları** | Hesaplama vs Bellek Transfer Dengesi | Kritik Aritmetik Yoğunluk $\alpha_{\text{crit}} \approx 295,4$ FLOP/bayt | Bölüm 3 |
| **D · Taşıma Merdiveni** | 140 GB model ağırlığının aktarım süreleri | HBM'de 42 ms iken PCIe'de 2,2 s, ağda 2,8 s | Bölüm 3 |
| **E · Üretim Ölçeği** | 16.384 GPU LLaMA 3 405B eğitim altyapısı | %38–41 MFU, 2 TB/s depolama akışı, 54 günde 466 arıza | Bölüm 5 ve Serinin Devamı |

### 1.1. Thread → warp → SM → GPU hiyerarşisi

GPU mimarisini sezgisel olarak kavramak için bir restoran mutfağı analojisi kuralım. Modern bir veri merkezi GPU'su (örneğin NVIDIA Hopper H100 veya Blackwell B200), tek bir dev aşçı değil; aynı mutfakta yan yana çalışan 132 ila 148 adet bağımsız hat aşçısından oluşan devasa bir tugaydır. Bu hat aşçılarının her birine **Streaming Multiprocessor (SM)** denir.

Her hat aşçısının (SM) kendi tezgâhı (Register File ve Shared Memory) ve kendi özel bıçak takımları (Tensor Core ve CUDA Core) bulunur. Aşçılar siparişleri tek tek değil, 32 kişilik ekipler halinde organize edilmiş sipariş fişleriyle alır. Bu 32 kişilik çalışma grubuna donanımda **warp** adı verilir. Tüm aşçılar malzemeleri restoranın ortak kilerinden (**High-Bandwidth Memory / HBM**) temin eder.

Aşağıdaki mimari diyagram, modern bir GPU'nun çip düzeyindeki yerleşimini ve tek bir SM'in iç anatomisini detaylandırmaktadır:

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

*Şekil 1.1: NVIDIA GPU Streaming Multiprocessor mimarisi. Solda silikon çip üzerindeki 132 SM'lik matris; sağda 4 alt bölmeye (SMSP), 256 KiB register dosyasına, 4 Tensor Core birimine ve ortak paylaşımlı bellek (SMEM/L1) alanına sahip tek bir SM'in iç yapısı.*

Fiziksel donanımda her bir SM, **SM subpartition (alt bölüm, SMSP)** adı verilen 4 özdeş çeyreğe ayrılmıştır:
1. **Dört Adet Tensor Core (TC):** Her alt bölümde bir adet bulunan özel donanımsal matris çarp-topla motoru. Tensor Core yükünde yaklaşık 1,83 GHz saat hızında çalışan bir H100 SXM5 üzerinde her Tensor Core döngü başına 1.024 yoğun BF16 kayan nokta işlemi (FLOP) yürütür ($8 \times 8 \times 8$ matris çarpımı). FP8 formatında bu sayı döngü başına 2.048 FLOP'a çıkar. Blackwell B200'de ise her Tensor Core döngü başına 2.048 BF16 FLOP yaparak çip genelinde 2.250 TFLOP/s seviyesine ulaşır.
2. **Dört Warp Scheduler ve Dağıtım Birimi (Dispatch Unit):** Her alt bölümdeki scheduler, kendisine atanmış 16 warp arasından yürütmeye hazır olan birini seçer ve her döngüde o warp'a ait komutu ilgili birimlere sevk eder. Dört alt bölüm sayesinde tek bir SM aynı anda **64 adede kadar yerleşik (resident) warp** (yani $64 \times 32 = 2.048$ thread) barındırabilir.
3. **128 Adet FP32 CUDA Core:** Alt bölüm başına 32 FP32 ALU, 16 INT32 ve 16 FP64 birimi bulunur. Bu birimler aktivasyon fonksiyonlarını (GELU, SiLU), normalizasyon katmanlarını (RMSNorm) ve element-wise tensör işlemlerini yürütür. 132 SM'e sahip bir H100'de toplam 16.896 adet FP32 CUDA çekirdeği görev yapar.
4. **256 KiB Register Dosyası:** Alt bölüm başına 64 KiB (SM genelinde 256 KiB, çip genelinde yaklaşık 33 MB) ultra hızlı register file yer alır. Thread değişkenleri burada saklanır ve tüm komutlar işlenenlerini buradan okur.

Dört alt bölüm, SM'in tabanında bulunan 256 KiB'lık birleşik L1 veri önbelleği ve paylaşımlı bellek (**Shared Memory / SMEM**) alanını ortaklaşa kullanır. Bu alanın en çok 228 KiB'ı programatik olarak SMEM olarak tahsis edilebilir. **Tensor Memory Accelerator (TMA)** ise L2 önbellekten veya HBM'den SMEM'e tensör bloklarını CPU veya SM çekirdeklerini yormadan doğrudan asenkron DMA ile kopyalar.

Bir H100 GPU'nun kâğıt üzerindeki meşhur tepe hız formülünü hesaplayalım:

$$C_{\text{peak}} = N_{\text{SM}} \times N_{\text{TC/SM}} \times \text{FLOP/döngü} \times f$$

Şöyle okuyun: $N_{\text{SM}}$ çipteki aktif SM sayısı (132), $N_{\text{TC/SM}}$ SM başına Tensor Core adedi (4), $\text{FLOP/döngü}$ Tensor Core başına döngüsel işlem sayısı (BF16 için 1.024), $f$ ise Tensor Core yükündeki saat hızıdır ($1{,}83 \times 10^9 \text{ Hz}$).

Sayıları yerine koyduğumuzda:

```text
528 TC × 1.024 FLOP × 1,83 GHz = 989.489.152.000 FLOP/s ≈ 989,5 TFLOP/s ≈ 990 TFLOP/s
```

Bu hesaplama, üretici veri sayfalarındaki "990 TFLOP/s BF16 non-sparse" değerinin silikon seviyesindeki doğrudan kaynağıdır.

### 1.2. Tensor Core neden farklıdır? Matris çarpım motorunun silikon mantığı

Geleneksel bir CPU veya GPU CUDA çekirdeği skaler bir mantıkla çalışır: Tek bir saat döngüsünde iki sayıyı çarpar ve bir sayıya ekler ($a \times b + c$, Multiply-Accumulate / MAC). 32 thread'lik bir warp bu komutu yürüttüğünde döngü başına 32 adet skaler sonuç üretilir.

Tensor Core ise skaler bir birim değildir; doğrudan matris düzeyinde çalışan özelleşmiş bir sistolik dizi (systolic-like array) mimarisidir. Tek bir komutla ($mma.sync$ veya Hopper mimarisindeki asenkron $wgmma$) küçük matris bloklarını tek hamlede çarpar ve akümülatöre yazar:

$$D = A \times B + C$$

Burada $A$ ($M \times K$), $B$ ($K \times N$) ve $C, D$ ($M \times N$) tensör parçalarıdır. 4. nesil Hopper Tensor Core'unda standart bir donanım mikro-adımı $8 \times 8 \times 8$ matris çarpımıdır. $8 \times 8 \times 8$ boyutunda bir matris çarpımında $8 \times 8 \times 8 = 512$ adet çarpma ve 512 adet toplama yapılır; yani tek bir işlem tam **1.024 FLOP** üretir!

Tensor Core'ların sunduğu devasa hızlanmanın ikinci anahtarı sayısal hassasiyetin (precision) düşürülmesidir. Sayısal bit genişliği azaldıkça silikon üzerindeki çarpıcı mantık kapıları (multiplier trees) karesel olarak küçülür. Aynı silikon alanına kat kat daha fazla işlem birimi sığdırılabilir:

| Sayısal Format | Toplam Bit | İşaret | Üs (Exponent) | Mantis (Fraction) | Sayısal Aralık | LLM Altyapısındaki Rolü | H100 SXM Tepe Hızı |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 | 1 | 8 | 23 | $\pm 10^{-38} \dots 10^{38}$ | Master ağırlıklar, optimizer state, loss | 60 TFLOP/s (Vektör) |
| **BF16** | 16 | 1 | 8 | 7 | $\pm 10^{-38} \dots 10^{38}$ | Temel model ön eğitimi (pre-training) | 990 TFLOP/s (TC) |
| **FP16** | 16 | 1 | 5 | 10 | $\pm 6 \times 10^{-5} \dots 65.504$ | Geleneksel çıkarım ve ince ayar (fine-tuning) | 990 TFLOP/s (TC) |
| **FP8 (E4M3)** | 8 | 1 | 4 | 3 | $\pm 448$ | Forward pass aktivasyonları ve ağırlıklar | 1.980 TFLOP/s (TC) |
| **FP8 (E5M2)** | 8 | 1 | 5 | 2 | $\pm 57.344$ | Backward pass gradyan iletimi | 1.980 TFLOP/s (TC) |
| **NVFP4 (E2M1)**| 4 | 1 | 2 | 1 | Blok ölçekli | Blackwell ultra yoğun GEMM çıkarımı | 4.500 TFLOP/s (B200) |

Hopper mimarisi iki farklı FP8 formatı sunar: **E4M3**, 3 mantis biti ile daha yüksek sayısal hassasiyet sağlar ve dinamik aralığın dar olduğu ileri geçiş (forward pass) için idealdir. **E5M2** ise FP16 ile aynı üs genişliğini (5 bit) koruyarak geniş bir dinamik aralık sunar; böylece geri yayılımda (backward pass) patlayan veya sıfıra yuvarlanan gradyanları korur.

### 1.3. Gecikme gizleme (latency hiding) ve Little's Law

Bir GPU'nun en büyük donanımsal gücü, gecikmeleri (latency) CPU gibi devasa önbelleklerle yok etmeye çalışmak yerine, paralel iş parçacıkları arasında ışık hızında geçiş yaparak gecikmeyi **gizlemesidir (latency hiding)**.

Bunu anlamak için donanımdaki fiziksel saat gecikmelerine bakalım:
- Register erişimi: $\sim 1$ döngü ($<0,5 \text{ ns}$)
- Paylaşımlı bellek (SMEM): $\sim 29$ döngü ($\sim 15 \text{ ns}$)
- L1 veri önbelleği: $\sim 41$ döngü ($\sim 22 \text{ ns}$)
- L2 önbellek: $\sim 263$ döngü ($\sim 140 \text{ ns}$)
- HBM3 genel bellek: $\sim 479$ döngü ($\sim 260 \text{ ns}$)

Bir warp, HBM'den global bir tensör verisi talep ettiğinde (global memory load), bu verinin HBM'den gelmesi yaklaşık 480 saat döngüsü sürer! Eğer SM tek bir warp çalıştırıyor olsaydı, saat döngülerinin %99'unda boşta (stall) bekleyecekti.

İşte Warp Scheduler burada devreye girer: Bir warp bellek yükleme komutu verip beklemeye (memory stall) girdiği anda, scheduler o warp'ı bekleme kuyruğuna çeker ve sonraki saat döngüsünde yürütmeye hazır (eligible) durumdaki başka bir warp'ın aritmetik komutunu sevk eder. Bu geçiş **sıfır ek yüklüdür (zero-overhead)**; çünkü tüm warp'ların register'ları zaten donanımsal register dosyasında fiziksel olarak ayrılmış durumdadır (CPU'daki gibi bağlam kaydetme / context switch maliyeti yoktur).

Sistemin sürekli meşgul kalabilmesi için gereken asgari warp sayısı **Little's Law** ile hesaplanır:

$$W_{\text{gerekli}} \ge 1 + \frac{L_{\text{gecikme}}}{T_{\text{hesap}}}$$

Şöyle okuyun: $L_{\text{gecikme}}$ bellek erişiminin saat döngüsü maliyeti, $T_{\text{hesap}}$ ise bir komutun aritmetik boru hattında harcadığı saat döngüsüdür.

Basit bir sayısal yürüyüş yapalım: Bir komutun 2 saat döngüsü hesaplama yaptığını ve ardından 6 döngülük bir önbellek gecikmesine girdiğini varsayalım:

```text
Warp 0: [Hesap: 2 döngü] -> [Bekleme: 6 döngü...................]
Warp 1:                 -> [Hesap: 2 döngü] -> [Bekleme: 6 döngü]
Warp 2:                                     -> [Hesap: 2 döngü]
Warp 3:                                                         -> [Hesap: 2 döngü]
Toplam gereken warp = 1 + (6 / 2) = 4 warp.
```

Eğer SM'de yalnızca 2 warp olsaydı, scheduler zamanın yarısında işsiz kalacaktı. 4 warp olduğunda ise boru hattı tek bir boş döngü dahi vermeden %100 meşgul tutulur. 480 döngülük HBM gecikmesini tamamen gizleyebilmek için bir SM'in onlarca aktif warp'ı havada tutması gerekir. 64 yerleşik warp kapasitesi işte bu fiziksel zorunluluktan doğmuştur.

> [!NOTE]
> **Öğrenme Kontrolü:** *"Daha çok thread açmak her zaman performansı artırır mı?"* Hayır. Bir kernel gereğinden fazla thread veya warp açtığında, thread başına düşen register ve shared memory miktarı daralır. Donanım sınırları aşıldığında occupancy çöker veya register spilling başlar. Önemli olan körü körüne thread sayısını artırmak değil; gecikmeyi gizlemeye yetecek kritik eşiği yakalamaktır.

### 1.4. Warp divergence, doluluk (occupancy) ve register baskısı

GPU'nun thread yürütme modeli **SIMT (Single Instruction, Multiple Threads)** olarak adlandırılır. Bir warp içindeki 32 thread, aynı saat döngüsünde aynı program komutunu yürütür; her thread komutu kendi özel verisi üzerinde çalıştırır.

Bu mimari, koşullu dallanmalarda (**branch divergence**) ciddi bir donanım darboğazına yol açar. Kernel kodunuzda şöyle bir ifade olduğunu düşünün:

```cuda
if (threadIdx.x < 16) {
    yol_A(); // 16 thread
} else {
    yol_B(); // 16 thread
}
```

Warp'taki 32 thread bağımsız CPU çekirdekleri gibi ikiye ayrılıp aynı anda farklı fonksiyonları çalıştıramaz. Donanım bu dalları serileştirmek (serialize etmek) zorundadır:

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

*Şekil 1.2: SIMT yürütme modeli ve Warp Divergence. Üstte 128 thread'lik bir bloğun 4 warp'a ayrılması; altta koşullu dallanmada aktif olmayan execution lane'lerin maskelenerek durdurulması ve yürütme süresinin iki katına çıkması.*

Donanım önce bir **yürütme maskesi (execution mask)** oluşturur: İlk 16 lane aktif edilir, son 16 lane maskelenip durdurulur ve `yol_A()` yürütülür. Ardından maske ters çevrilir; ilk 16 lane durdurulur ve son 16 lane ile `yol_B()` yürütülür. Sonuç: Saat süresi iki katına çıkar, o warp için efektif hesaplama verimi tam **%50 düşer**!

İkinci kritik gerilim **register baskısı (register pressure)** ve **doluluk (occupancy)** dengesidir. Bir SM'de 65.536 adet 32-bit register (256 KiB) bulunur. Bir SM'in teorik tavanı olan 64 warp'ı (2.048 thread) aynı anda barındırabilmesi için thread başına düşen register sayısı son derece sınırlıdır:

$$\text{Maksimum Register/Thread} = \frac{65.536 \text{ register}}{2.048 \text{ thread}} = 32 \text{ register/thread}$$

Kernel karmaşıklaştıkça ve derleyici optimizasyonları döngüleri açtıkça (loop unrolling), thread başına gereken register sayısı hızla artar. Thread başına register sayısı arttıkça SM'de barınabilen warp sayısı basamak basamak çöker:

| Thread Başına Register | SM'deki Aktif Warp Sayısı | Teorik Doluluk (Occupancy) | Donanımsal Etki |
| :--- | :--- | :--- | :--- |
| **$\le 32$ register** | 64 warp (2.048 thread) | **%100** | Kusursuz gecikme gizleme potansiyeli |
| **64 register** | 32 warp (1.024 thread) | **%50** | Gecikme gizleme marjı daralır |
| **128 register** | 16 warp (512 thread) | **%25** | HBM gecikmelerinde SM stall riski artar |
| **255 register (maksimum)** | 8 warp (256 thread) | **%12,5** | Kritik eşik; boru hattı boşlukları başlar |

Eğer bir kernel thread başına 255 register sınırını da aşarsa ne olur? **Register spilling** gerçekleşir: Sığmayan yerel değişkenler sessizce **local memory** alanına dökülür. "Local memory" adı yanıltıcıdır; fiziksel olarak SM içinde değil, yüzlerce döngü uzaktaki yüksek gecikmeli HBM içinde yer alır! Bir kernel spilling'e düştüğünde işlem hızı dramatik biçimde yere çakılır.

> [!NOTE]
> **Düşünce Deneyi: Düşük Occupancy Her Zaman Kötü müdür?**
> Klasik CUDA optimizasyonunda hedef daima "%100 occupancy" idi. Ancak modern Hopper mimarisinde geliştirilen **FlashAttention-3** ve gelişmiş GEMM çekirdekleri bu kuralı yıktı. Bu çekirdekler yalnızca %15–25 occupancy ile (8–16 warp) çalışır! Sır şudur: **Warp Uzmanlaşması (Warp Specialization)**. SM içindeki 1–2 warp yalnızca asenkron TMA kopyalarını başlatmakla görevlendirilirken (producer), kalan warp'lar durmaksızın Tensor Core $wgmma$ komutlarını yürütür (consumer). Veri TMA tarafından önceden SMEM'e getirildiği için consumer warp'lar asla beklemez. Bekleme yoksa gecikmeyi gizlemek için onlarca yedek warp tutmaya da gerek kalmaz!

---

## 2. Bellek hiyerarşisi neden performansı belirler?

Büyük dil modelleri ile çalışırken en sık karşılaşılan yanılgılardan biri, bellek ihtiyacını yalnızca "gigabayt" cinsinden düşünmektir. Oysa sistem performansını belirleyen asıl sınır, verinin nerede durduğu ve oradan ne kadar hızlı çekilebildiğidir.

### 2.1. Kapasite ve bant genişliği aynı şey değildir

Bir kütüphane düşünün: Kütüphanenin raflarında 10 milyon kitap bulunabilir (yüksek kapasite). Ancak kütüphanenin koridorları tek bir kişinin geçebileceği kadar darsa ve ödünç alma masasında sadece bir görevli çalışıyorsa (düşük bant genişliği), okuyucular dakikada ancak 5 kitap çıkarabilir.

Bir H100 SXM5 GPU'nun 80 GB HBM3 belleği vardır. BF16 formatında 70 milyar parametreli bir model 140 GB yer kaplar; yani tek bir GPU'ya sığmaz bile. Ancak model ağırlıkları 4-bit quantize edilip 35 GB'a indirilse ve 80 GB belleğe rahatça sığsa dahi sistem anında hızlanmaz. Çünkü çıkarım esnasında tek bir token üretebilmek için bu 35 GB'lık ağırlık matrisinin tamamı HBM'den okunup Tensor Core'lara taşınmak zorundadır!

GPU bellek hiyerarşisi, kapasite ile bant genişliği arasındaki bu fiziksel ödünleşimi yönetmek üzere dik bir piramit olarak inşa edilmiştir:

| Bellek Seviyesi | Fiziksel Konum | H100 SXM Kapasitesi | Bant Genişliği (Bandwidth) | Tipik Gecikme (Latency) |
| :--- | :--- | :--- | :--- | :--- |
| **Register File** | SM çekirdeğinde | SM başına 256 KiB (~33 MB) | **$\sim 30+$ TB/s** (çip geneli) | $\sim 1$ saat döngüsü ($<0,5 \text{ ns}$) |
| **Shared Memory / L1**| SM çekirdeğinde | SM başına 256 KiB (~33 MB) | **$\sim 15+$ TB/s** (çip geneli) | $\sim 29$ saat döngüsü ($\sim 15 \text{ ns}$) |
| **L2 Cache** | Çip merkezinde | 50 MB (iki bölüm) | **$\sim 5,5$ TB/s** | $\sim 263$ saat döngüsü ($\sim 140 \text{ ns}$) |
| **HBM3 Bellek** | Çip üzerinde (interposer)| 80 GB (5 aktif yığın) | **3,35 TB/s** | $\sim 479$ saat döngüsü ($\sim 260 \text{ ns}$) |
| **Host Bellek (DDR5)**| Anakart üzerinde | Düğüm başına 2 TB | 64 GB/s (PCIe Gen5 x16) | $\sim 1.000+$ saat döngüsü ($>500 \text{ ns}$)|

Veri piramidin tepesinden tabanına doğru indikçe kapasite 2.400 kat büyür (33 MB'tan 80 GB'a); ancak bant genişliği 10 kat düşer ve erişim gecikmesi 500 kat uzar! Yüksek MFU elde etmenin tek yolu, veriyi mümkün olduğunca piramidin tepesinde tutmaktır.

### 2.2. Birleşik (coalesced) erişim ve önbellek satırı matematiği

HBM bellek denetleyicileri bayt bayt veri transferi yapmaz. Veri, 32 baytlık veya 128 baytlık **önbellek satırları (cache lines)** halinde taşınır.

Bir warp içindeki 32 thread global bellekten 4 baytlık kayan noktalı sayılar okumak istediğinde iki senaryo oluşur:

**Senaryo A: Birleşik (Coalesced) Erişim:**
32 thread, bellekte yan yana duran ardışık adresleri talep eder ($A[0], A[1], \dots, A[31]$).
- Talep edilen toplam faydalı veri: $32 \times 4 = 128 \text{ bayt}$.
- Bellek alt sistemi bu talebi **tek bir 128 baytlık bellek işlemiyle** karşılar.
- Veri yolu verimliliği: $128 / 128 = \mathbf{\%100}$!

**Senaryo B: Aralıklı (Strided / Non-Coalesced) Erişim:**
Bir matrisin sütunları boyunca okuma yapıldığında her thread ardışık değil, örneğin 32 eleman aralıklı adresleri talep eder ($A[0], A[32], A[64], \dots$).
- Her thread farklı bir 128 baytlık bellek satırına denk gelir.
- Bellek denetleyicisi 32 ayrı 128 baytlık transfer başlatmak zorunda kalır.
- Taşınan toplam veri: $32 \times 128 = 4.096 \text{ bayt}$.
- Talep edilen faydalı veri: 128 bayt.
- Veri yolu verimliliği: $128 / 4.096 = \mathbf{\%3,1}$!

Bant genişliğinin %96,9'u gereksiz baytları taşırken çöpe gider. Yanlış bellek yerleşimi içeren bir kernel, hesaplama birimleri ne kadar güçlü olursa olsun HBM veri yolunu kilitler.

### 2.3. Tiling (döşeme) ve paylaşımlı bellek: Attention neden SRAM ister?

Klasik iki matris çarpımını ele alalım ($C = A \times B$, her biri $N \times N$). Naif üçlü döngü ile hesaplama yapıldığında her bir eleman için global bellekten satır ve sütunlar tekrar tekrar okunur; toplam $O(N^3)$ global bellek erişimi gerçekleşir. $N=4.096$ için bu trilyonlarca baytlık veri hareketi demektir ve HBM anında çöker.

**Tiling (Döşeme)** tekniği bu darboğazı çözer: Matrisler $B_k \times B_k$ boyutunda küçük bloklara (tile) bölünür. Bir blok HBM'den **bir kez** okunup SM içindeki ultra hızlı paylaşımlı belleğe (SMEM / SRAM) yüklenir. Tensor Core bu bloğu SRAM üzerinde defalarca çarparak HBM erişim sayısını $B_k$ kat azaltır.

Bu prensibin modern yapay zekâdaki en devasa zaferi **FlashAttention** algoritmasıdır. Standart self-attention algoritması, $N$ uzunluğundaki bir bağlam için $N \times N$ boyutunda devasa bir dikkat matrisi oluşturur ve bunu HBM'e yazıp tekrar okur ($O(N^2)$ bellek I/O). FlashAttention ise tiling ve online softmax matematiğini kullanarak dikkat matrisini asla HBM'e yazmaz; blokları SMEM/SRAM içinde hesaplar, indirger ve yalnızca nihai çıktıyı HBM'e yazar. Sonuç: Bellek erişimi $O(N^2)$'den $O(N)$'e iner ve işlem hızı 3–5 kat artar.

---

## 3. Roofline modeli: GPU gerçekte neyi bekliyor?

Bir derin öğrenme modelinin GPU üzerinde ne kadar hızlı çalışabileceğini ve hangi donanım sınırına takıldığını anlamanın en kesin matematiksel aracı **Williams vd. (2009)** tarafından geliştirilen **Roofline Modeli**dir.

### 3.1. Kritik aritmetik yoğunluk ve dirsek noktası

Roofline modeli, bir sistemin elde edebileceği azami işlem hızını ($P$, FLOP/s cinsinden) operasyonel yoğunluğun ($I$, FLOP/bayt cinsinden) bir fonksiyonu olarak tanımlar:

$$P = \min(P_{\text{tepe}}, \, I \times B_{\text{bellek}})$$

Şöyle okuyun: $P_{\text{tepe}}$ donanımın teorik tepe hesaplama gücü (FLOP/s), $B_{\text{bellek}}$ donanımın tepe bellek bant genişliği (bayt/s), $I$ ise algoritmanın her bir bayt bellek transferi başına yürüttüğü matematiksel işlem sayısıdır (aritmetik yoğunluk):

$$I = \frac{\text{Toplam Yürütülen FLOP}}{\text{HBM'den/HBM'e Taşınan Toplam Bayt}}$$

Sistemin bellek darboğazından hesaplama darboğazına geçtiği kırılma noktasına **kritik aritmetik yoğunluk (dirsek noktası / ridge point, $\alpha_{\text{crit}}$)** denir:

$$\alpha_{\text{crit}} = \frac{P_{\text{tepe}}}{B_{\text{bellek}}}$$

BF16 çalışan bir NVIDIA H100 SXM5 için kritik eşiği hesaplayalım:

$$\alpha_{\text{crit}} = \frac{989{,}5 \times 10^{12} \text{ FLOP/s}}{3{,}35 \times 10^{12} \text{ B/s}} \approx \mathbf{295,4 \text{ FLOP/bayt}}$$

Bu sayı modern sistem mühendisliğinin en kritik pusulasıdır:
- Eğer iş yükünüzün aritmetik yoğunluğu **$I < 295,4$ FLOP/bayt** ise: Sistem **Memory-Bound (Bellek Kısıtlı)** bölgesindedir. GPU'nuzun Tensor Core'ları boştadır; performans tamamen HBM'in hızıyla sınırlanır.
- Eğer iş yükünüzün aritmetik yoğunluğu **$I \ge 295,4$ FLOP/bayt** ise: Sistem **Compute-Bound (Hesaplama Kısıtlı)** bölgesindedir. Bellek veri akışı hesaplama birimlerini doyurmaya yeter; GPU teorik 990 TFLOP/s tavanına yaklaşır.

### 3.2. Training ve prefill ile decode neden taban tabana zıttır?

Büyük dil modellerinde matris çarpımlarını ($X \times W$) ele alalım. $X \in \mathbb{R}^{B \times D}$ aktivasyon matrisi, $W \in \mathbb{R}^{D \times F}$ ağırlık matrisi olsun (16-bit formatta, eleman başına 2 bayt):
- **Toplam FLOP:** $2 B D F$
- **Toplam Bellek Transferi:** $2(BD + DF + BF) \text{ bayt}$

Aritmetik yoğunluk formülü:

$$I(B) = \frac{2 B D F}{2(BD + DF + BF)} = \frac{B D F}{B(D + F) + D F}$$

LLaMA 3 70B modelinin MLP katmanını somut örnek alalım ($D = 8.192, F = 28.672$):
$D F = 234.881.024$ parametre ($469,8 \text{ MB}$).

**Durum 1: Ön Eğitim ve Prefill ($B = 512$ token):**
Büyük batch ve uzun prompt bağlamında hesaplama:
- Ağırlık matrisi: $469,8 \text{ MB}$
- Aktivasyonlar: $2 \times 512 \times (8.192 + 28.672) = 37,7 \text{ MB}$
- Aritmetik yoğunluk:

$$I(512) = \frac{512 \times 8.192 \times 28.672}{512 \times 36.864 + 234.881.024} = \frac{120.259.084.288}{253.755.392} \approx \mathbf{473,9 \text{ FLOP/bayt}}$$

$473,9 > 295,4$ olduğu için sistem açıkça **Compute-Bound** rejimdedir! H100 Tensor Core birimleri doyar ve sürdürülen hesaplama 600–800 TFLOP/s seviyesine (yüksek MFU) rahatlıkla ulaşır.

**Durum 2: Otoregresif Decode ($B = 1$ token):**
Kullanıcıya teker teker token üretilirken:
- $B=1$ olduğunda $D F \gg B(D+F)$ olur. Ağırlık matrisi bellek trafiğini tamamen domine eder.
- Aritmetik yoğunluk:

$$I(1) \approx \frac{1 \times D F}{D F} = \mathbf{1 \text{ FLOP/bayt}}$$

Kritik eşik 295,4 iken sistem 1 FLOP/bayt seviyesindedir! H100 bu rejimde ne kadar FLOP üretebilir?

$$P_{\text{decode}} = 1 \text{ FLOP/bayt} \times 3{,}35 \text{ TB/s} = \mathbf{3,35 \text{ TFLOP/s}}$$

990 TFLOP/s kapasiteli bir canavar, tekil çıkarımda sadece **3,35 TFLOP/s** üretebilir; yani donanım potansiyelinin yalnızca **%0,34'ünü** kullanabilir! Tensor Core'lar zamanın %99,66'sında HBM'den ağırlıkların akmasını bekler.

Bu matematiksel gerçek; çıkarım motorlarının (vLLM vb.) neden continuous batching, PagedAttention ve Speculative Decoding gibi tekniklerle batch boyutunu büyütmeye ve ağırlık okuma maliyetini birden fazla istek arasında paylaştırmaya çalıştığını kesin olarak kanıtlar.

### 3.3. 140 GB taşıma hesabı: Donanım sınırları merdiveni

Bir modelin ağırlıklarını ya da KV cache tensörlerini taşırken donanım katmanlarının sunduğu gerçek aktarım hızlarını bilmek hayati önem taşır. 70 milyar parametreli bir modelin BF16 ağırlıklarını (tam **140 GB**) farklı veri yolları üzerinden taşımanın saf fiziksel süresini hesaplayalım:

| Veri Yolu / Bağlantı | Bant Genişliği (Tek Yön) | 140 GB Aktarım Süresi | Donanımsal Gerçeklik |
| :--- | :--- | :--- | :--- |
| **HBM3 (GPU İçi)** | 3.350 GB/s (3,35 TB/s) | **41,8 milisaniye** | Çip üstü silikon hatları; referans hız tavanı |
| **NVLink 4 (Kasa İçi)** | 450 GB/s (GPU başına) | **311 milisaniye** | 8 GPU'lu DGX şasisi içi engelsiz crossbar |
| **PCIe Gen5 x16** | 32 GB/s (tek yön) / 64 GB/s | **2,19 – 4,38 saniye**| Host CPU, NVMe ve standart adaptör veri yolu |
| **InfiniBand NDR 400G** | 50 GB/s (GPU başına) | **2,80 saniye** | Düğümler arası scale-out küme ağı kumaşı |
| **100 GbE Kurumsal Ağ**| 12,5 GB/s | **11,2 saniye** | Standart veri merkezi yönetim ve servis ağı |
| **10 GbE Standart Hat** | 1,25 GB/s | **112 saniye (~1,9 dk)** | Paylaşımlı eski ağ veya uzak NFS depolama |

Bu merdiven sistem mimarisinin temel kuralını çizer:
Veriyi HBM'den çıkarmak **7 kat**, tek bir sunucu kasasından (NVLink alanından) çıkarıp ağa dökmek ise **70 kat** yavaşlamaya neden olur!

### 3.4. Mühendisler için üç adımlı Roofline reçetesi

Bir kernel veya çıkarım hattı optimize edilirken şu 3 adım izlenir:
1. **Aritmetik Yoğunluğu Ölçün:** `ncu` (NVIDIA Nsight Compute) profil aracını çalıştırarak `dram__bytes.sum` ve `sm__ops_path_tensor_src_fp16.sum` değerlerini alın. İkisini oranlayarak $I$ değerini bulun.
2. **Kritik Eşik ile Kıyaslayın:** $I < 295$ ise iş yükü bellek sınırlıdır; $I \ge 295$ ise hesaplama sınırlıdır.
3. **Doğru Optimizasyon Aracını Seçin:**
   - *Memory-Bound ise:* Çözüm daha hızlı bir GPU almak değildir! Çözüm; kernel füzyonu (birden fazla işlemi tek kernel'de birleştirip HBM ara yazmalarını yok etmek), ağırlık quantizasyonu (FP8/INT4 ile bellek ayak izini küçültmek) veya batch boyutunu büyütmektir.
   - *Compute-Bound ise:* Çözüm; düşük bitli Tensor Core komutlarına geçmek ($wgmma$, FP8), tensor layout optimizasyonları ve boru hattı örtüştürmesidir.

---

## 4. Bir düğümde GPU’lar nasıl bağlanır?

Tek bir H100 çipi 80 GB bellek ve 990 TFLOP/s sunar. Ancak günümüz frontier modelleri yüzlerce gigabayt, hatta terabaytlarca bellek gerektirir. Bu nedenle endüstri standardı temel yapı taşı, 8 adet GPU'nun tek bir şaside birleştirildiği sunucu düğümleridir (örneğin NVIDIA DGX H100 veya HGX H100).

### 4.1. PCIe Gen5 ile NVLink 4'ün görev bölüşümü

Bir GPU sunucusunda iki temel veri yolu omurgası bulunur ve görevleri tamamen farklıdır:

**PCIe Gen5 (Genel Amaçlı I/O):**
- CPU, sistem belleği (RAM), NVMe SSD sürücüler ve Ağ Arayüz Kartları (NIC) arasındaki iletişimi sağlar.
- PCIe Gen5 x16 yuvası çift yönde 64 GB/s (tek yönde 32 GB/s) teorik bant genişliği sunar.
- Çok amaçlıdır; ancak GPU'lar arası yoğun tensör senkronizasyonu için çok yavaştır.

**NVLink 4 (GPU-to-GPU Özel Kumaşı):**
- Yalnızca GPU'ların kendi aralarında konuşması için tasarlanmış özel, donanımsal önbellek tutarlı (cache-coherent) bir bağlantıdır.
- Her H100 GPU'da 18 adet NVLink 4 bağlantısı bulunur. Her bağlantı tek yönde 25 GB/s (çift yönde 50 GB/s) taşır.
- GPU başına toplam bant genişliği: $18 \times 25 = \mathbf{450 \text{ GB/s}}$ (çift yönde **900 GB/s**).
- PCIe Gen5'ten tam **14 kat daha hızlıdır**!

### 4.2. DGX H100 NUMA topolojisi ve GPU–NIC yakınlığı

Bir DGX H100 sunucusunun içinde iki adet Intel Xeon Platinum CPU bulunur (CPU 0 ve CPU 1). İki işlemci birbirine **UPI (Ultra Path Interconnect, ~32–40 GB/s)** köprüsüyle bağlıdır. Bu durum klasik bir **NUMA (Non-Uniform Memory Access)** mimarisi yaratır:

```text
[ CPU 0 ] <====== UPI (~36 GB/s) ======> [ CPU 1 ]
    |                                        |
  PCIe Switch                              PCIe Switch
  +-- GPU 0, 1, 2, 3                       +-- GPU 4, 5, 6, 7
  +-- NIC 0, 1, 2, 3 (Compute)             +-- NIC 4, 5, 6, 7 (Compute)
  +-- Storage NIC (GDS)                    +-- Storage NIC (GDS)
  +-- 4x NVMe SSD                          +-- 4x NVMe SSD
```

Bu topolojinin pratik sonucu şudur:
- GPU 0, kendi soketindeki NIC 0 ile konuşurken doğrudan yerel PCIe switch üzerinden tam 50 GB/s hızla ağa çıkar.
- Ancak GPU 0 yanlış yapılandırma sonucu CPU 1 soketindeki NIC 4 ile konuşturulursa, trafik CPU'lar arasındaki daracık UPI köprüsünden geçmek zorunda kalır! Bant genişliği 50 GB/s'den UPI sınırına düşer, gecikme katlanır. Dağıtık eğitimde worker süreçlerinin GPU ve NIC NUMA düğümlerine doğru kilitlenmesi (process pinning) bu yüzden zorunludur.

### 4.3. GPUDirect RDMA ve bellek kopyalama maliyeti

Geleneksel bir Linux sunucusunda ağ üzerinden gelen veri şu yolu izlerdi:
1. Veri NIC'e gelir.
2. NIC veriyi host işletim sistemi çekirdek belleğine (CPU page cache) yazar.
3. Uygulama veriyi kullanıcı belleğine kopyalar (bounce buffer).
4. CUDA sürücüsü veriyi PCIe üzerinden GPU HBM belleğine aktarır.

Bu zincirde CPU belleği iki kez kopyalanır, CPU çekirdekleri meşgul edilir ve onlarca mikrosaniyelik gecikme eklenir.

**GPUDirect RDMA (Remote Direct Memory Access)** bu engeli tamamen ortadan kaldırır. Ağ kartı (NIC), PCIe anahtarı üzerinden doğrudan GPU'nun HBM belleğine doğrudan DMA (Direct Memory Access) yapar. CPU işletim sistemi ve RAM tamamen baypas edilir. Gecikme onlarca mikrosaniyeden **1–2 mikrosaniyeye** iner.

### 4.4. Düğüm içi NVSwitch crossbar mimarisi

8 GPU'yu doğrudan birbirine kablolamaya kalksaydınız ($8 \times 7 / 2 = 28$ bağlantı), bağlantı karmaşası silikonu kilitlerdi. NVIDIA bu sorunu şasiye 4 adet bağımsız **NVSwitch** yongası yerleştirerek çözer.

Her GPU'nun 18 adet NVLink bağlantısı, bu 4 NVSwitch yongasına dengeli olarak dağıtılır ($5 + 4 + 4 + 5 = 18$). NVSwitch'ler tam bir donanımsal crossbar anahtar görevi görür:
- Herhangi bir GPU, diğer 7 GPU'nun belleğine aynı anda engelsiz 450 GB/s tek yönlü hızla erişebilir.
- Düğümün toplam biseksiyon bant genişliği tam **3,6 TB/s**'dir ($4 \text{ GPU} \times 450 \text{ GB/s} \times 2$).

Bu 3,6 TB/s'lik devasa otoban sayesinde, model katmanlarını 8 GPU'ya bölen **Tensör Paralelliği (TP=8)** sıfıra yakın iletişim cezasıyla kasada çalıştırılabilir.

---

## 5. Düğümden kümeye: Serideki yol haritamız

Sekiz GPU'lu bir DGX düğümü muazzam bir güçtür: 640 GB HBM3 belleği ve yaklaşık 7,9 PFLOP/s BF16 hesaplama kapasitesi sunar. Fakat sınırları çok nettir.

### 5.1. Düğüm sınırları ve ölçekleme zorunluluğu

Frontier bir temel modeli ele alalım: **LLaMA 3 405B**.
- Model parametreleri (BF16): 810 GB.
- Gradyanlar (BF16): 810 GB.
- AdamW Optimizer durumları (FP32 master ağırlıklar, momentum ve varyans): $405 \times 12 = 4.860 \text{ GB}$.
- Toplam asgari eğitim belleği: **$\approx 6.480 \text{ GB}$ (6,48 Terabayt)**!

Sekiz GPU'lu bir sunucunun 640 GB'lık belleği, bu modelin eğitim durumunun ancak **onda birini** alabilir. Modeli tek bir sunucuda eğitmek imkânsızdır; binlerce GPU'ya yayılmak zorundayız.

Ancak sunucu kasasından çıktığımız an fiziksel kurallar sertleşir:
- Bant genişliği NVLink'in 450 GB/s seviyesinden, InfiniBand ağının **50 GB/s** seviyesine (9 kat) düşer.
- İletişim gecikmesi 1 mikrosaniyenin altından, anahtarlar arası sıçramalarla **birkaç mikrosaniyeye** uzar.

Bu fiziksel uçurum, yapay zekâ altyapısının altın kuralını doğurur:
> **"En sık konuşan tensörler en içeride (NVLink'te) kalmalı; daha seyrek konuşan tensörler dışarıya (InfiniBand ağına) taşmalıdır."**

### 5.2. Sonraki bölümlere köprü: İş yükünden kurumsal platforma

Bu makalede kurduğumuz donanım anatomisi, bir sonraki adımların temelini oluşturur. Beş bölümlük serimizin devamında bu temeli katman katman büyüteceğiz:

1. **Yapay Zekâ Altyapısı (2): LLM GPU'yu Nasıl Kullanır? — Training, Prefill ve Decode**  
   Aynı GPU silikonunun eğitimde compute-bound, tekil çıkarımda ise neden acımasızca memory-bound çalıştığını; KV cache patlamasını, PagedAttention mimarisini, prefix caching mantığını ve vLLM motorunun üretim dinamiklerini inceleyeceğiz.  
   *[Yazıya Git: Yapay Zekâ Altyapısı (2)](post.html?slug=gpu-is-yukleri-ve-uretim-cikarimi)*

2. **Yapay Zekâ Altyapısı (3): GPU'lar Nasıl Birlikte Çalışır? — NVLink, InfiniBand ve Paralellik**  
   Düğümler arası ağ kumaşını (Quantum-2 InfiniBand, rail-optimized fat-tree topolojisi, SHARP ağ içi indirgeme) ve bir modeli binlerce GPU'ya bölen 4D paralellik boyutlarını (TP, PP, DP/FSDP, CP, EP ve DeepSeek DualPipe) göreceğiz.  
   *[Yazıya Git: Yapay Zekâ Altyapısı (3)](post.html?slug=coklu-gpu-aglari-ve-dagitik-egitim)*

3. **Yapay Zekâ Altyapısı (4): AI Kümesi Neden Yavaşlar? — Depolama, Arızalar ve MFU**  
   16.384 GPU'luk bir süper bilgisayarın kâğıt üzerindeki hızına rağmen neden %38 MFU seviyesinde kaldığını; GPUDirect Storage (GDS), checkpoint duraklamaları, Young-Daly arıza formülü ve küme telemetrisi üzerinden teşhis edeceğiz.  
   *[Yazıya Git: Yapay Zekâ Altyapısı (4)](post.html?slug=gpudan-ai-super-bilgisayarina)*

4. **Yapay Zekâ Altyapısı (5): Kurumsal AI Platform Mimarisi — Güç, FinOps ve Orkestrasyon**  
   Megavatlarca elektrik çeken veri merkezi güç ve sıvı soğutma tasarımlarını, Slurm/Kubernetes orkestrasyonunu, GPU maliyet modellemesini (FinOps) ve 70B kurumsal referans mimarisini uçtan uca inşa edeceğiz.  
   *[Yazıya Git: Yapay Zekâ Altyapısı (5)](post.html?slug=kurumsal-ai-platform-mimarisi)*

---

## 6. Sistem tasarımı için beş temel soru

### 6.1. GPU neden skaler CPU kodunda değil de matris çarpımında bin kat hızlıdır?

Bir CPU, saat frekansı 4–5 GHz olan ve karmaşık dal tahmincileri, derin önbellekleri ve sıra dışı yürütme motorları olan az sayıda (8–64) gelişmiş çekirdeğe sahiptir. Karmaşık mantık içeren seri kodları çok hızlı yürütür; ancak saat döngüsü başına üretebileceği aritmetik işlem sayısı sınırlıdır.

GPU ise saat frekansı daha düşük (~1,8 GHz) ancak on binlerce paralel aritmetik birime (16.896 CUDA core, 528 Tensor Core) sahip bir işlemci ordusudur. Matris çarpımı (GEMM), bağımsız elemanların birbirleriyle çarpılıp toplanmasına dayanan mükemmel bir paralel problemdir (embarrassingly parallel). Tensor Core'lar tek bir saat döngüsünde sistolik dizi mantığıyla yüzlerce çarp-topla işlemi yürüterek CPU'ların silikon mimarisi gereği asla ulaşamayacağı bir işlem yoğunluğu üretir.

### 6.2. nvidia-smi'da %100 GPU kullanımı görmek modelin hızlı çalıştığını gösterir mi?

Kesinlikle hayır. `nvidia-smi` aracının raporladığı "GPU Utilization" metriği, örnekleme periyodu boyunca SM'lerin en az bir iş parçacığı tarafından meşgul edilip edilmediğini gösterir. Bir SM, 480 saat döngüsü boyunca HBM'den veri gelmesini beklerken (memory stall) bile GPU utilization %100 görünebilir!

Önemli olan donanımın aktif görünmesi değil, **Model FLOPs Utilization (MFU)** değeridir. MFU; donanımın teorik tepe FLOP kapasitesine oranla modelin gerçekte ne kadar faydalı matematiksel işlem ürettiğini ölçer. Çıkarım esnasında GPU %100 meşgul görünürken MFU %1'in altında kalabilir.

### 6.3. NVLink neden PCIe yerine sunucunun her yerinde ve tüm kümede kullanılmıyor?

NVLink; aşırı yüksek frekansta çalışan, çok sayıda fiziksel iletim hattı (trace) gerektiren ve mesafe uzadıkça sinyal kaybı (attenuation) hızla artan pahalı bir silikon teknolojisidir. Bir PCB üzerinde yalnızca 20–30 santimetrelik mesafelerde çalışabilir.

Sunucu anakartında CPU, disk ve ağ kartları gibi farklı üreticilerden gelen genel amaçlı bileşenlerin konuşabilmesi için standartlaşmış, esnek ve geriye dönük uyumlu PCIe protokolü şarttır. Düğümler arasında (onlarca metre) ise bakır veya optik kablolar üzerinden sinyal taşıyabilen InfiniBand veya Ethernet gibi serileştirilmiş ölçeklenebilir ağ protokolleri kullanılmak zorundadır.

### 6.4. Model ağırlıkları HBM'e tamamen sığıyorsa neden hâlâ bellek duvarına takılır?

Model ağırlıklarının HBM'e sığması sadece **kapasite** probleminin çözüldüğünü gösterir; **bant genişliği** problemini çözmez.

Otoregresif token üretiminde (decode) her bir token üretilirken modelin tüm parametreleri HBM'den okunup register'lara çekilmek zorundadır. $B=1$ için aritmetik yoğunluk sadece 1 FLOP/bayttır. HBM bant genişliği (3,35 TB/s) saniyede akabilecek veri miktarını kısıtladığı için Tensor Core'lar döngülerin %99'unda boş bekler. Model belleğe sığsa bile bellek bant genişliği duvarına çarpmaktan kurtulamaz.

### 6.5. Bir kernel'i hızlandırmak için bloktaki thread sayısını artırmak neden bazen performansı düşürür?

Bir thread bloğundaki thread sayısı artırıldığında (örneğin 128'den 1.024'e), o bloğun ihtiyaç duyduğu toplam register ve shared memory miktarı katlanarak büyür. Bir SM'in fiziksel register (256 KiB) ve paylaşımlı bellek (228 KiB) tavanı sabittir.

Eğer tek bir blok SM kaynaklarının büyük kısmını tüketirse, SM aynı anda yalnızca 1 blok barındırabilir. Bu durum yerleşik warp sayısını düşürerek occupancy oranını çökertir. Daha da kötüsü, derleyici thread başına register kısıtını tutturamazsa register spilling gerçekleşir; değişkenler HBM'e dökülür ve kernel başarımında dramatik bir çöküş yaşanır.

---

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

1. Bir tensör HBM'den SM'e gelirken sırasıyla hangi bellek katmanlarından geçer ve her adımda gecikme kaç kat artar?
2. 990 TFLOP/s tepe hızına sahip bir H100 GPU'nun tekil decode evresinde neden 3,35 TFLOP/s'te tıkandığını matematiksel olarak nasıl kanıtlarsınız?
3. Warp divergence yaşandığında donanımsal yürütme maskesi nasıl çalışır ve yürütme süresi neden uzar?
4. Neden 8 GPU'lu bir DGX düğümünde Tensör Paralelliği (TP) rahatça uygulanabilirken, düğümler arasında uygulanması performansı öldürür?
5. Bir kernel geliştirirken occupancy oranını artırmak yerine ne zaman warp uzmanlaşmasına (warp specialization) odaklanmalısınız?

---

## Bütün hikâye altı satırda

- Bir GPU'nun teorik FLOP/s değeri, veriyi o işlem birimlerine besleyecek bellek ve bağlantı bant genişliği olmadığında hiçbir anlam ifade etmez.
- Streaming Multiprocessor (SM), 4 alt bölümünde yerleşik 64 warp arasında sıfır ek yükle geçiş yaparak yüzlerce döngülük HBM gecikmelerini gizler.
- Tensor Core birimleri skaler değil sistolik matris motorlarıdır; FP32'den FP8 ve NVFP4'e indikçe silikona döngü başına iki ve dört kat daha fazla matematik sığar.
- H100 için kritik aritmetik yoğunluk dirsek noktası 295,4 FLOP/bayttır; eğitim ve prefill bu sınırın üstünde (compute-bound), tekil decode ise altında (memory-bound) yaşar.
- Düğüm içindeki 8 GPU, PCIe'den 14 kat daha hızlı olan NVSwitch crossbar sayesinde 3,6 TB/s biseksiyon bant genişliğiyle tek bir sanal GPU gibi konuşur.
- 140 GB model ağırlığını HBM içinde taşımak 42 milisaniye sürerken, PCIe'de 2,2 saniye, küme ağında ise 2,8 saniye sürer; en yavaş hat tüm sistemin hızını belirler.

---

## Terimler sözlüğü

- **Streaming Multiprocessor (SM):** GPU üzerindeki temel bağımsız hesaplama birimi. H100'de 132, B200'de 148 adet bulunur; içinde scheduler, register, Tensor Core ve paylaşımlı bellek barındırır.
- **Warp:** Donanım scheduler tarafından tek bir SIMT komutuyla lockstep olarak yürütülen 32 thread'lik temel yürütme grubu.
- **Warp Divergence:** Bir warp içindeki thread'lerin koşullu dallanmalarda farklı yollara sapması sonucu donanımın yolları maskelerle serileştirmesi ve verimin düşmesi durumu.
- **Latency Hiding:** Bir warp bellekten veri beklerken (stall), scheduler'ın sıfır ek yükle yürütmeye hazır başka bir warp'a geçerek işlem boru hattını meşgul tutması mekanizması.
- **Occupancy (Doluluk):** Bir SM üzerinde aynı anda barınan aktif warp sayısının, SM'in fiziksel olarak barındırabileceği teorik azami warp sayısına (H100 için 64) oranı.
- **Roofline Modeli:** Bir algoritmanın aritmetik yoğunluğuna (FLOP/bayt) bağlı olarak sistemin bellek kısıtlı mı yoksa hesaplama kısıtlı mı çalıştığını gösteren analitik başarım modeli.
- **Critical Arithmetic Intensity ($\alpha_{\text{crit}}$):** Tepe FLOP/s değerinin tepe bellek bant genişliğine bölünmesiyle bulunan ve compute-bound rejimine geçişi belirleyen dirsek noktası eşiği (H100 BF16 için ~295,4 FLOP/bayt).
- **GPUDirect RDMA:** Ağ arayüz kartının (NIC), ana işlemciyi (CPU) ve sistem RAM'ini baypas ederek doğrudan GPU HBM belleğine veri okuyup yazmasını sağlayan doğrudan bellek erişim teknolojisi.
- **NVSwitch:** Bir sunucu şasisi içindeki birden fazla GPU'nun NVLink hatlarını engelsiz bir crossbar mimarisinde tam all-to-all olarak bağlayan donanımsal anahtarlama yongası.

---

## Kaynaklar ve doğrulama bağlantıları

- [K1] NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Resmi mikro-mimari kılavuzu: GPC'ler, TPC'ler, SM alt bölümleri, HBM3 bellek denetleyicileri ve TMA motoru.
- [K2] NVIDIA Corporation. [NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html). Şekil 1.1'deki SM sınırları: SM başına 64 warp, 65.536 register, thread başına en çok 255 register, en çok 228 KB shared memory.
- [K3] Weile Luo vd. (2024). [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). arXiv:2402.13499. Hopper yükleme gecikmeleri: SMEM ~29, L1 ~41, L2 ~263 ve HBM ~479 döngü.
- [K4] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao (2024). [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608). arXiv:2407.08608. Hopper'da warp uzmanlaşması (warp specialization), TMA asenkron kopyaları ve wgmma.
- [K5] Samuel Williams, Andrew Waterman, David Patterson (2009). [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785). Communications of the ACM, 52(4), 65–76. Kritik aritmetik yoğunluk ve dirsek noktası formülasyonu.
- [K6] Jacob Austin, Swapnil Patil, Adam Paszke, Reiner Pope (2025). [How to Scale Your Model: How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/). JAX-ML Scaling Book, Chapter 12. Donanım hiyerarşisi ve hız farkları.
- [K7] NVIDIA Corporation. [Introduction to NVIDIA DGX H100/H200 Systems](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html). Düğüm donanımı: NUMA topolojisi, PCIe anahtarları, çift soket Xeon mimarisi ve NVMe veri yolları.
- [K8] NVIDIA Corporation. [Fabric Manager for NVIDIA NVSwitch Systems](https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html). NVSwitch çapraz bağlantı mimarisi, 18 NVLink hattının 5+4+4+5 dağılımı ve 3,6 TB/s biseksiyon bant genişliği.
- [K9] Bu blogda: [Yapay Zekâ Altyapısı (2): LLM GPU'yu Nasıl Kullanır? — Training, Prefill ve Decode](post.html?slug=gpu-is-yukleri-ve-uretim-cikarimi).
- [K10] Bu blogda: [Yapay Zekâ Altyapısı (3): GPU'lar Nasıl Birlikte Çalışır? — NVLink, InfiniBand ve Paralellik](post.html?slug=coklu-gpu-aglari-ve-dagitik-egitim).
