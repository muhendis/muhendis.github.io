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

<svg viewBox="0 0 560 1764" role="img" aria-label="Yapay zekâ altyapısının büyük resmi, üst üste üç sütun olarak. 1. sütun, hesaplama, tek GPU&#x27;nun içi: bir Streaming Multiprocessor 4 özdeş alt bölmeden oluşur; her birinde bir Tensor Core, bir warp scheduler, 32 FP32 ALU ve 16.384 register bulunur. H100&#x27;de 132 SM ve 990 TFLOP/s BF16, B200&#x27;de 148 SM ve 2.250 TFLOP/s vardır. Bellek hiyerarşisi register&#x27;lardan (SM başına 256 KiB), paylaşımlı bellek ya da L1&#x27;den (256 KiB), B200&#x27;deki TMEM&#x27;den ve L2&#x27;den (50 ya da 126 MB, yaklaşık 5,5 TB/s) HBM&#x27;e (H100&#x27;de 80 GB ve 3,35 TB/s, B200&#x27;de 192 GB ve 8 TB/s) iner. Roofline dirseği 990 bölü 3,35, yaklaşık bayt başına 295,5 FLOP&#x27;tur: batch 1 ile decode&#x27;un yoğunluğu 1 civarındadır ve MFU yüzde 1-2&#x27;de kalır; yaklaşık 295 tokenin üzerindeki eğitim batch&#x27;leri hesap sınırlıdır. Bellek duvarı: on yılda FLOP 1.000 kattan fazla, HBM bant genişliği 30 kattan az arttı. GPU&#x27;lar birbirleriyle NVLink üzerinden 450 GB/s, CPU ile PCIe Gen5 üzerinden 64 GB/s konuşur. 2. sütun, ağ: düğüm içinde 8 GPU, 6,4 TB/s anahtarlama ve 3,6 TB/s biseksiyon sunan 4 NVSwitch ile bağlanır; 18 port çarpı 25 GB/s, GPU başına 450 GB/s eder; NVL72 bunu 72 GPU&#x27;ya genişletir. Düğümler arasında bir InfiniBand fat-tree, 16 spine switch&#x27;i ve 256&#x27;şar GPU&#x27;luk 4 ölçeklenebilir birimi, toplam 1.024 GPU&#x27;yu birleştirir; her düğümde 8 adet 400 Gb/s NIC vardır, düğüm başına 400 GB/s, GPU başına 50 GB/s eder ve i. GPU i. leaf&#x27;e bağlanır. Tensör paralelliği NVLink içinde kalır: 28.672 bölü 2.200 yaklaşık 13&#x27;tür, yani 8 GPU. Boru hattı paralelliği katmanları böler, kabarcık oranı P eksi 1 bölü P eksi 1 artı M&#x27;dir; uzman paralelliği MoE uzmanlarını AllToAll ile dağıtır, DeepSeek-V3&#x27;teki 64 yollu EP gibi; veri paralelliği gradyanları AllReduce ile toplar, LLaMA-3 405B&#x27;deki 128 yollu ZeRO-1 gibi. Halka AllReduce yaklaşık 2B bölü W sürer; SHARP toplamayı switch içinde yapar, yaklaşık B bölü W sürer ve trafiği yarıya indirir. Ağ duvarı: düğümden çıkınca GPU başına bant genişliği 450&#x27;den 50 GB/s&#x27;ye, 9 kat düşer. Depolama tokenleri yukarı 2 TB/s ile akıtır ve 7,5 TB&#x27;lık kontrol noktalarını alır. 3. sütun, depolama: geleneksel yol NVMe&#x27;den sayfa önbelleği ve kullanıcı RAM&#x27;i üzerinden HBM&#x27;e iki memcpy ile gider ve CPU yüzde 100 meşgul olur; GPUDirect Storage NVMe-oF&#x27;tan PCIe switch üzerinden doğrudan DMA ile HBM&#x27;e gider ve gecikmeyi yüzde 80 azaltır. Veri besleme 15 trilyon tokeni .bin ve .idx dosyalarından sıfır kopya mmap ile akıtır; Meta Tectonic 7.500 sunucu, 240 PB, 2 TB/s okuma ve 7 TB/s anlık yazma sunar. 405B modelin kontrol noktası parametre başına 16 bayttır: 0,81 TB ağırlık, 0,81 TB gradyan, 4,86 TB AdamW durumu; aktivasyon ve RNG ile yaklaşık 7,5 TB. Young-Daly aralığı 2 çarpı delta çarpı M&#x27;nin kareköküdür; M 3 saatken 25 GB/s ile delta 307 saniye, tau 42,9 dakika ve kayıp yaklaşık yüzde 24; 250 GB/s ile delta 31 saniye, tau 13,6 dakika ve kayıp yaklaşık yüzde 7,5&#x27;tir. G/Ç duvarı: 7,5 TB&#x27;lık kayıt 25 GB/s&#x27;de 16.384 GPU&#x27;yu 307 saniye bekletir. En altta GPU başına log ölçekli bant genişliği merdiveni: L2 yaklaşık 5,5 TB/s, HBM 3,35 TB/s, NVLink 450 GB/s, PCIe 64 GB/s, InfiniBand 50 GB/s ve tüm küme için eski depolama 25 GB/s. En yavaş basamak bütün kümenin hızını belirler." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tri-a-gen" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="tri-a-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="tri-a-wrn" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
<marker id="tri-a-bad" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker>
<marker id="tri-a-ok" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700">Büyük Resim: Yapay Zekâ Altyapısının 3 Sütunu</text>
<text x="16" y="42" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Bir baytın yolu: depolama → HBM → SM; GPU&#x27;dan GPU&#x27;ya NVLink → InfiniBand</text>
<rect x="16" y="52" width="11" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="32" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesaplama</text>
<rect x="97.005" y="52" width="11" height="11" rx="2" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="113.005" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağ</text>
<rect x="139.89499999999998" y="52" width="11" height="11" rx="2" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="155.89499999999998" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">depolama</text>
<rect x="215.45499999999998" y="52" width="11" height="11" rx="2" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="231.45499999999998" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">duvar / darboğaz</text>
<rect x="334.575" y="52" width="11" height="11" rx="2" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="350.575" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">çözüm</text>
<rect x="16" y="74" width="528" height="420" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.4"/>
<text x="28" y="98" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700">1 · HESAPLAMA</text>
<text x="532" y="98" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">tek GPU&#x27;nun içi · H100 / B200</text>
<text x="28" y="118" text-anchor="start" style="fill:var(--c-text);font-size:12px">Soru: Tensor Core&#x27;lar veriyle yeterince hızlı beslenebiliyor mu?</text>
<rect x="28" y="130" width="236" height="200" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="148" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Streaming Multiprocessor</text>
<text x="38" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">H100: 132 SM · 990 TFLOP/s BF16</text>
<text x="38" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">B200: 148 SM · 2.250 TFLOP/s BF16</text>
<path d="M56 310 H222 V206" style="fill:none;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:3 3;stroke-opacity:.7"/>
<path d="M52 306 H218 V202" style="fill:none;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:3 3;stroke-opacity:.7"/>
<path d="M48 302 H214 V198" style="fill:none;stroke:var(--c-accent);stroke-width:1;stroke-dasharray:3 3;stroke-opacity:.7"/>
<rect x="38" y="188" width="172" height="110" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="46" y="203" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:600">1 alt bölme</text>
<rect x="46" y="210" width="156" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="124.0" y="223.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="46" y="231" width="156" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="124.0" y="244.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Warp Scheduler</text>
<rect x="46" y="252" width="156" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="124.0" y="265.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">32 × FP32 ALU</text>
<rect x="46" y="273" width="156" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="124.0" y="286.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16.384 register</text>
<text x="236" y="254" text-anchor="middle" style="fill:var(--c-accent);font-size:15px;font-weight:700">×4</text>
<text x="38" y="322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her SM = 4 özdeş alt bölme</text>
<rect x="276" y="130" width="256" height="200" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="286" y="148" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Bellek hiyerarşisi</text>
<text x="286" y="164" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">↑ hızlı ve küçük · ↓ büyük ve yavaş</text>
<rect x="286" y="174" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.28;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="190.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Register</text>
<text x="358" y="190" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB/SM · çipte ~33 MB</text>
<rect x="286" y="202" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="218.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SMEM/L1</text>
<text x="358" y="218" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB/SM · &lt;10 ns</text>
<rect x="286" y="230" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="318.0" y="246.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">TMEM</text>
<text x="358" y="246" text-anchor="start" style="fill:var(--c-text);font-size:11px">256 KiB/SM · yalnız B200</text>
<rect x="286" y="258" width="64" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="274.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">L2</text>
<text x="358" y="274" text-anchor="start" style="fill:var(--c-text);font-size:11px">50 / 126 MB · ~5,5 TB/s</text>
<rect x="286" y="286" width="64" height="36" rx="5" style="fill:var(--c-accent);fill-opacity:.08;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="318.0" y="308.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">HBM</text>
<text x="358" y="301" text-anchor="start" style="fill:var(--c-text);font-size:11px">80 GB · 3,35 TB/s (H100)</text>
<text x="358" y="316" text-anchor="start" style="fill:var(--c-text);font-size:11px">192 GB · 8,0 TB/s (B200)</text>
<rect x="28" y="338" width="504" height="114" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="358" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Roofline: α = taşınan bayt başına FLOP</text>
<text x="40" y="378" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">α_crit = C / W = 990 / 3,35</text>
<text x="40" y="394" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">≈ 295,5 FLOP/bayt (H100)</text>
<rect x="40" y="402" width="300" height="20" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="190.0" y="416.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">decode (B = 1): α ≈ 1 → MFU %1-2</text>
<rect x="40" y="426" width="300" height="20" rx="5" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="190.0" y="440.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">eğitim (B ≳ 295 token): hesap sınırlı</text>
<line x1="372" y1="350" x2="372" y2="436" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="372" y1="436" x2="520" y2="436" style="stroke:var(--c-text-mute);stroke-width:1"/>
<path d="M372 436 L444 374 L516 374" style="fill:none;stroke:var(--c-text);stroke-width:1.8"/>
<line x1="444" y1="374" x2="444" y2="436" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<text x="378" y="356" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">FLOP/s</text>
<text x="450" y="392" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">α_crit ≈ 295</text>
<circle cx="384" cy="425.7" r="4" style="fill:var(--c-danger)"/>
<text x="384" y="448" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">decode</text>
<circle cx="496" cy="374" r="4" style="fill:var(--c-success)"/>
<text x="496" y="366" text-anchor="middle" style="fill:var(--c-success);font-size:11px">eğitim</text>
<rect x="28" y="460" width="504" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="475.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Bellek duvarı: 10 yılda FLOP 1.000×&#x27;ten fazla, HBM bant genişliği 30×&#x27;ten az arttı</text>
<line x1="100" y1="497" x2="100" y2="525" marker-start="url(#tri-a-gen)" marker-end="url(#tri-a-gen)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="110" y="515" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">GPU ↔ GPU: NVLink 450 GB/s</text>
<line x1="338" y1="497" x2="338" y2="525" marker-start="url(#tri-a-mut)" marker-end="url(#tri-a-mut)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="348" y="515" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU ↔ CPU: PCIe Gen5 64 GB/s</text>
<rect x="16" y="528" width="528" height="488" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="28" y="552" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700">2 · AĞ</text>
<text x="532" y="552" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">GPU&#x27;dan GPU&#x27;ya · düğüm → küme</text>
<text x="28" y="572" text-anchor="start" style="fill:var(--c-text);font-size:12px">Soru: GPU&#x27;lar ara sonuçları ne kadar hızlı paylaşabiliyor?</text>
<rect x="28" y="584" width="244" height="196" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="602" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Düğüm içi: NVLink</text>
<text x="38" y="618" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 GPU + 4 NVSwitch (crossbar)</text>
<rect x="38" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="63.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU0</text>
<rect x="96" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="121.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU1</text>
<rect x="154" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU2</text>
<rect x="212" y="628" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU3</text>
<rect x="38" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="63.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU4</text>
<rect x="96" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="121.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU5</text>
<rect x="154" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU6</text>
<rect x="212" y="710" width="50" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU7</text>
<line x1="63" y1="650" x2="63" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="63" y1="692" x2="63" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="121" y1="650" x2="121" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="121" y1="692" x2="121" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="179" y1="650" x2="179" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="179" y1="692" x2="179" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="237" y1="650" x2="237" y2="668" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<line x1="237" y1="692" x2="237" y2="710" style="stroke:var(--c-accent-2);stroke-width:1.4"/>
<rect x="38" y="668" width="224" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="150.0" y="684.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4 × NVSwitch · 6,4 TB/s</text>
<text x="38" y="750" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">18 × 25 GB/s = 450 GB/s/GPU</text>
<text x="38" y="768" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">biseksiyon 3,6 TB/s · NVL72: 72 GPU</text>
<rect x="288" y="584" width="244" height="196" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="298" y="602" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Düğümler arası: InfiniBand</text>
<text x="298" y="618" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">fat-tree · 4 SU × 256 = 1.024 GPU</text>
<rect x="298" y="628" width="224" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="410.0" y="643.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16 spine switch · 51,2 TB/s</text>
<line x1="324.0" y1="670" x2="309.2" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="324.0" y1="670" x2="326.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="324.0" y1="670" x2="342.8" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="298.0" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="324.0" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU0</text>
<line x1="324.0" y1="692" x2="324.0" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="298.0" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="324.0" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<line x1="381.3" y1="670" x2="365.2" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="381.3" y1="670" x2="382.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="381.3" y1="670" x2="398.8" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="355.3" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="381.3" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU1</text>
<line x1="381.3" y1="692" x2="381.3" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="355.3" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="381.3" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<line x1="438.6" y1="670" x2="421.20000000000005" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="438.6" y1="670" x2="438.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="438.6" y1="670" x2="454.79999999999995" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="412.6" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="438.6" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU2</text>
<line x1="438.6" y1="692" x2="438.6" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="412.6" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.6" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<line x1="495.9" y1="670" x2="477.20000000000005" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="495.9" y1="670" x2="494.0" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<line x1="495.9" y1="670" x2="510.79999999999995" y2="650" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="469.9" y="670" width="52" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="495.9" y="685.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">SU3</text>
<line x1="495.9" y1="692" x2="495.9" y2="710" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="469.9" y="710" width="52" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="495.9" y="725.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">256 GPU</text>
<text x="298" y="750" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">8 × 400 Gb/s NIC = düğüm başına 400 GB/s</text>
<text x="298" y="768" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU başına 50 GB/s · GPU i → leaf i</text>
<rect x="28" y="788" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="806" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">TP · tensör paralelliği</text>
<rect x="212.4" y="794" width="53.6" height="18" rx="5" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="239.20000000000002" y="807.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">NVLink</text>
<text x="38" y="822" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her ağırlık matrisini böler</text>
<text x="38" y="838" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Y ≤ 28.672 / 2.200 ≈ 13 → 8</text>
<rect x="284" y="788" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="806" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">PP · boru hattı</text>
<rect x="494.8" y="794" width="27.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="508.40000000000003" y="807.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">IB</text>
<text x="294" y="822" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">katmanları aşamalara böler</text>
<text x="294" y="838" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">kabarcık = (P−1) / (P−1+M)</text>
<rect x="28" y="852" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="870" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">EP · uzman paralelliği</text>
<rect x="238.8" y="858" width="27.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="252.4" y="871.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">IB</text>
<text x="38" y="886" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">MoE uzmanlarını dağıtır · AllToAll</text>
<text x="38" y="902" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">DeepSeek-V3: 64-yol, 8 düğüm</text>
<rect x="284" y="852" width="248" height="58" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="870" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">DP / FSDP · veri</text>
<rect x="494.8" y="858" width="27.2" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="508.40000000000003" y="871.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">IB</text>
<text x="294" y="886" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyanları AllReduce ile toplar</text>
<text x="294" y="902" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">LLaMA-3 405B: 128-yol ZeRO-1</text>
<rect x="28" y="920" width="248" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="939" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Halka AllReduce</text>
<text x="38" y="958" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">RS + AG → T ≈ 2·B / W</text>
<rect x="284" y="920" width="248" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="294" y="939" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">SHARP: toplama switch&#x27;in içinde</text>
<text x="294" y="958" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">T ≈ B / W · trafik −%50</text>
<rect x="28" y="982" width="504" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="997.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">Ağ duvarı: düğümden çıkınca GPU başına 450 → 50 GB/s (9×); TP 8 GPU&#x27;da kalır</text>
<line x1="100" y1="1047" x2="100" y2="1019" marker-end="url(#tri-a-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="110" y="1037" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">okuma: 2 TB/s token akışı</text>
<line x1="338" y1="1019" x2="338" y2="1047" marker-end="url(#tri-a-wrn)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<text x="348" y="1037" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">yazma: 7,5 TB kontrol noktası</text>
<rect x="16" y="1050" width="528" height="500" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-warn);stroke-width:1.4"/>
<text x="28" y="1074" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700">3 · DEPOLAMA</text>
<text x="532" y="1074" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">veri setleri ve kontrol noktaları</text>
<text x="28" y="1094" text-anchor="start" style="fill:var(--c-text);font-size:12px">Soru: Veri ve kayıtlar GPU&#x27;yu bekletmeden akabiliyor mu?</text>
<rect x="28" y="1106" width="504" height="124" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1124" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Bir bayt diskten HBM&#x27;e nasıl gider?</text>
<text x="40" y="1142" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:600">Geleneksel: CPU sayfa önbelleği üzerinden · 2 × memcpy, CPU %100</text>
<rect x="40" y="1150" width="96" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="88.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe SSD</text>
<rect x="162" y="1150" width="104" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="214.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">sayfa önbelleği</text>
<rect x="292" y="1150" width="104" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="344.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">kullanıcı RAM</text>
<rect x="422" y="1150" width="98" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="471.0" y="1165.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU HBM</text>
<line x1="137" y1="1161" x2="160" y2="1161" marker-end="url(#tri-a-bad)" style="stroke:var(--c-danger);stroke-width:1.4"/>
<line x1="267" y1="1161" x2="290" y2="1161" marker-end="url(#tri-a-bad)" style="stroke:var(--c-danger);stroke-width:1.4"/>
<line x1="397" y1="1161" x2="420" y2="1161" marker-end="url(#tri-a-bad)" style="stroke:var(--c-danger);stroke-width:1.4"/>
<text x="40" y="1194" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">GPUDirect Storage (cuFile): CPU ve RAM atlanır · gecikme −%80, CPU ≈ %0</text>
<rect x="40" y="1202" width="96" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="88.0" y="1217.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NVMe / NVMe-oF</text>
<rect x="162" y="1202" width="234" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="279.0" y="1217.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">PCIe switch · doğrudan DMA · 64+ GB/s</text>
<rect x="422" y="1202" width="98" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="471.0" y="1217.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">GPU HBM</text>
<line x1="137" y1="1213" x2="160" y2="1213" marker-end="url(#tri-a-ok)" style="stroke:var(--c-success);stroke-width:1.4"/>
<line x1="397" y1="1213" x2="420" y2="1213" marker-end="url(#tri-a-ok)" style="stroke:var(--c-success);stroke-width:1.4"/>
<rect x="28" y="1238" width="248" height="148" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="38" y="1256" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Veri besleme</text>
<text x="38" y="1274" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">15+ trilyon token (LLaMA-3)</text>
<text x="38" y="1290" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">.bin + .idx · sıfır kopya mmap</text>
<text x="38" y="1306" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NUMA: her soket kendi 4 GPU&#x27;sunu</text>
<text x="38" y="1322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">16K GPU: 2–4 TB/s sürekli okuma</text>
<rect x="38" y="1336" width="228" height="42" rx="5" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="46" y="1353" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:600">Meta Tectonic: 7.500 sunucu · 240 PB</text>
<text x="46" y="1370" text-anchor="start" style="fill:var(--c-text);font-size:11px">2 TB/s okuma · 7 TB/s anlık yazma</text>
<rect x="284" y="1238" width="248" height="148" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="294" y="1256" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Kontrol noktası · 405B model</text>
<text x="294" y="1274" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16 bayt/param = 2 + 2 + 12</text>
<rect x="294.0" y="1284" width="24.6" height="18" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<rect x="318.6" y="1284" width="24.6" height="18" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<rect x="343.2" y="1284" width="147.7" height="18" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<rect x="491.0" y="1284" width="31.0" height="18" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<rect x="294" y="1313" width="10" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="310" y="1322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ağırlık BF16 · 0,81 TB</text>
<rect x="294" y="1329" width="10" height="10" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="310" y="1338" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyan BF16 · 0,81 TB</text>
<rect x="294" y="1345" width="10" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="310" y="1354" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">AdamW FP32 · 4,86 TB</text>
<rect x="294" y="1361" width="10" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.12;stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:3 2"/>
<text x="310" y="1370" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:600">+ aktivasyon, RNG → ~7,5 TB</text>
<rect x="28" y="1394" width="504" height="116" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="40" y="1414" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Young-Daly: en uygun kayıt aralığı</text>
<text x="520" y="1414" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">τ = √(2 · δ · M)</text>
<text x="40" y="1432" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">küme MTBF: 26.000 sa ÷ 2.048 ≈ 12,7 sa; pratikte 2–3 sa (hesapta M = 3 sa)</text>
<rect x="40" y="1442" width="100" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="90.0" y="1457.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">A · 25 GB/s</text>
<text x="150" y="1457" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">δ = 307 s → τ ≈ 42,9 dk</text>
<rect x="340" y="1446" width="119.5" height="14" rx="2" style="fill:var(--c-danger);fill-opacity:.45;stroke:var(--c-danger);stroke-width:1"/>
<text x="465.5" y="1457" text-anchor="start" style="fill:var(--c-danger);font-size:11px;font-weight:600">kayıp ≈ %24</text>
<rect x="40" y="1470" width="100" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="90.0" y="1485.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">B · 250 GB/s</text>
<text x="150" y="1485" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">δ = 31 s → τ ≈ 13,6 dk</text>
<rect x="340" y="1474" width="37.5" height="14" rx="2" style="fill:var(--c-success);fill-opacity:.45;stroke:var(--c-success);stroke-width:1"/>
<text x="383.5" y="1485" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">kayıp ≈ %7,5</text>
<text x="40" y="1502" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıp = δ/τ (yazma) + τ/2M (geri sarma) · 10× hızlı yazma → √10 ≈ 3× az</text>
<rect x="28" y="1516" width="504" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="280.0" y="1531.0" text-anchor="middle" style="fill:var(--c-text);font-size:11px">G/Ç duvarı: 7,5 TB&#x27;lık kayıt 25 GB/s&#x27;de 307 s sürer; 16.384 GPU boşta bekler</text>
<rect x="16" y="1560" width="528" height="196" rx="10" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="1582" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Bant genişliği merdiveni · GPU başına, H100 (log ölçek)</text>
<line x1="196" y1="1592" x2="196" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="196" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 GB/s</text>
<line x1="292" y1="1592" x2="292" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="292" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100 GB/s</text>
<line x1="388" y1="1592" x2="388" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="388" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1 TB/s</text>
<line x1="484" y1="1592" x2="484" y2="1712" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:4 3"/>
<text x="484" y="1726" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10 TB/s</text>
<text x="28" y="1607" text-anchor="start" style="fill:var(--c-text);font-size:11px">L2 önbellek</text>
<rect x="196" y="1596" width="263.1" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="465.0748181914474" y="1607" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">~5,5 TB/s</text>
<text x="28" y="1626" text-anchor="start" style="fill:var(--c-text);font-size:11px">HBM3</text>
<rect x="196" y="1615" width="242.4" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1"/>
<text x="444.4043014755371" y="1626" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3,35 TB/s</text>
<text x="28" y="1645" text-anchor="start" style="fill:var(--c-text);font-size:11px">NVLink (düğüm içi)</text>
<rect x="196" y="1634" width="158.7" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="360.708401322433" y="1645" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">450 GB/s</text>
<text x="28" y="1664" text-anchor="start" style="fill:var(--c-text);font-size:11px">PCIe Gen5 (CPU)</text>
<rect x="196" y="1653" width="77.4" height="14" rx="2" style="fill:var(--c-text-mute);fill-opacity:.45;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="279.3932775024532" y="1664" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">64 GB/s</text>
<text x="28" y="1683" text-anchor="start" style="fill:var(--c-text);font-size:11px">InfiniBand (düğüm dışı)</text>
<rect x="196" y="1672" width="67.1" height="14" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="269.1011204162578" y="1683" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">50 GB/s</text>
<text x="28" y="1702" text-anchor="start" style="fill:var(--c-text);font-size:11px">eski depolama (tüm küme)</text>
<rect x="196" y="1691" width="38.2" height="14" rx="2" style="fill:var(--c-warn);fill-opacity:.45;stroke:var(--c-warn);stroke-width:1"/>
<text x="240.2022408325156" y="1702" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">25 GB/s</text>
<text x="28" y="1746" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">En yavaş basamak bütün kümenin hızını belirler: üç sütun birlikte tasarlanır.</text>
</svg>

*Şekil 1.1: Üç sütunun büyük resmi. Hesaplama (SM anatomisi, bellek hiyerarşisi, roofline), ağ (NVLink düğümü, InfiniBand fat-tree, 4D paralellik, SHARP) ve depolama (GDS yolu, veri besleme, kontrol noktası, Young-Daly); en altta GPU başına bant genişliği merdiveni.*

Şekli yukarıdan aşağı okuyun. Her sütun bir soruyla açılır ve en altındaki kırmızı şeritte o sütunun duvarı yazar; sütunlar arasındaki oklar onları birbirine bağlayan hatlardır. En alttaki merdiven bütün hatları tek eksene koyar: bant genişliği HBM'den NVLink'e yaklaşık 7 kat, NVLink'ten InfiniBand'e 9 kat düşer; eski bir depolama sisteminde ise bütün küme, tek bir GPU'nun InfiniBand hattından bile yavaş bir boruyu paylaşır. Sonraki üç bölüm bu sütunları sırayla açıyor.

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
- **NUMA Uyumlu Ön Getirme:** Düğümlerde iki ayrı CPU soketi bulunur ve her soket 4 GPU'ya fiziksel olarak bağlıdır. DataLoader süreçleri doğrudan hedef GPU'nun PCIe root complex birimine bağlı CPU soketine kilitlenir (NUMA affinity) ve CUDA asenkron akışlarıyla (`pin_memory=True`) NUMA hatlarını geçmeden doğrudan HBM'e aktarılır.

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

Üç sütunumuzu somutlaştırmak için NVIDIA H100 SuperPod üzerinde LLaMA-3 70B ve 405B modellerini inceleyelim:

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
- NVIDIA Corporation (2022). [NVIDIA H100 Tensor Core GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/hopper-architecture-whitepaper.pdf). Resmi mikro-mimari kılavuzu.
- Bu blogda: [Bir Prompt'un Yolculuğu (8): Çıktı Katmanı (LM Head) — Megatron-LM ve Bellek Duvarı](post.html?slug=cikti-katmani-derinlemesine).
- Bu blogda: [Bir Prompt'un Yolculuğu (9): Bağlam Penceresinin İçi — 2K Duvarından FlashAttention'a](post.html?slug=baglam-penceresi-ve-dikkat).
