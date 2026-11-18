Bir AI donanım platformu, yan yana dizilmiş GPU sunucularından ibaret değildir. İş yükünün hesaplama, HBM, bağlantı, depolama ve güç üzerindeki baskısını aynı anda taşıyabilen dengeli bir sistemdir.

**Bu yazının temel fikri şu:** Doğru donanım, “en hızlı GPU” seçilerek değil; training, fine-tuning ve inference sırasında tensörlerin nerede üretildiği, nerede tutulduğu ve hangi bağlantıdan geçtiği hesaplanarak seçilir.

Önce iş yükünü fiziksel kaynak talebine çevireceğiz. Ardından GPU düğümünü, scale-out ağı, depolamayı, güç altyapısını ve hata alanlarını bir araya getirip 70B model üzerinden somut bir referans tasarım kuracağız.

> **Önceki yazıdan devraldığımız fikir:** [Küme teşhisi yazısında](post.html?slug=gpudan-ai-super-bilgisayarina) düşük MFU, yavaş checkpoint ve kuyruk patlaması gibi belirtileri fiziksel bekleme noktalarına bağladık. Serinin son yazısında aynı ölçümleri satın alma ve yerleşim kararlarına dönüştürüyoruz.

> **Bu yazının sonunda**
>
> - Training, full fine-tuning, PEFT ve inference iş yüklerini donanım talepleriyle analitik olarak eşleştirebileceksiniz.
> - HBM kapasitesi, HBM bant genişliği, NVLink, scale-out ağ, depolama ve güç bütçesini tek bir mimari formülle okuyabileceksiniz.
> - 70B model için serving ve fine-tuning düğümlerinin neden tamamen farklı tasarlandığını matematiksel olarak hesaplayabileceksiniz.
> - Bir GPU kümesini satın almadan önce hangi katmanlarda doğrulama ve benchmark yapmanız gerektiğini belirleyebileceksiniz.

**Bu yazıda**

- [1. İş yükünü donanım imzasına çevirmek](#1-iş-yükünü-donanım-imzasına-çevirmek)
- [2. GPU düğümünü tasarlamak](#2-gpu-düğümünü-tasarlamak)
- [3. Düğümden kümeye: Ağ ve fiziksel yerleşim](#3-düğümden-kümeye-ağ-ve-fiziksel-yerleşim)
- [4. GPU dışındaki fiziksel sınırlar: Depolama, güç ve arızalar](#4-gpu-dışındaki-fiziksel-sınırlar-depolama-güç-ve-arızalar)
- [5. 70B model için donanım referans tasarımı](#5-70b-model-için-donanım-referans-tasarımı)
- [6. Satın almadan önce doğrulama merdiveni](#6-satın-almadan-önce-doğrulama-merdiveni)
- [7. Öğrendiklerimizi sınayalım](#7-öğrendiklerimizi-sınayalım)
- [Okuyucu artık hangi sorulara kendi başına yanıt verebilir?](#okuyucu-artık-hangi-sorulara-kendi-başına-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. İş yükünü donanım imzasına çevirmek

Yapay zekâ altyapısı tasarlarken yapılan en pahalı hata, tüm iş yüklerini (pre-training, fine-tuning, inference) homojen birer "GPU ihtiyacı" olarak görmektir. Oysa bir iş yükünün donanım üzerindeki fiziksel ayak izi (hardware signature); hesaplama yoğunluğu, bellek ayak izi ve ağ iletişim frekansı açısından dramatik farklılıklar gösterir.

Aşağıdan yukarıya doğru ölçek merdiveninde her basamak bir alttaki bileşeni çoğaltırken, sistemin darboğazı da fiziksel olarak yer değiştirir:

```
Thread -> Warp (32) -> SM -> GPU (HBM3) -> DGX Düğümü (NVLink) -> SU (Leaf) -> SuperPOD (Spine) -> Çoklu Pod Kümesi
```

<!-- restored-svg:tr_full:2 -->
<svg viewBox="0 0 720 388" role="img" aria-label="Aşağıdan yukarı ölçek merdiveni: thread; 32 thread&#x27;lik warp; en fazla 64 warp ya da 2.048 thread barındıran SM; 132 SM, 80 GB HBM3 ve 3,35 TB/s&#x27;lik H100 SXM; NVSwitch&#x27;li 8 GPU&#x27;luk DGX H100; 32 DGX yani 256 GPU&#x27;luk SU; 4 SU ve 127 DGX yani 1.016 GPU&#x27;luk SuperPOD; Llama 3 405B&#x27;deki gibi 16.384 H100&#x27;lük çoklu pod kümesi. Her basamağın yanında tipik darboğazı yazar: dal ayrışması, gecikme gizleme, register ve shared memory, HBM bant genişliği, NVLink&#x27;te TP kolektifleri, leaf switch ve rail&#x27;ler, InfiniBand&#x27;de DP ve PP, en üstte depolama, arıza ve güç." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ölçek merdiveni: her basamak alttakini çoğaltır</text>
<text x="704" y="24" text-anchor="end" style="fill:var(--c-text-mute);font-size:11.5px;font-weight:600">tipik darboğaz burada yaşar</text>
<rect x="70" y="44" width="386" height="32" rx="6" style="fill:var(--c-danger);fill-opacity:.10;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="82" y="64" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Çoklu pod kümesi</text>
<text x="210.4" y="64" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">16.384 H100 (ör. Llama 3 405B, RoCE)</text>
<text x="60" y="64" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)"></text>
<line x1="462" y1="60" x2="494" y2="60" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12px">depolama, arıza, güç</text>
<rect x="80" y="84" width="376" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="92" y="104" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SuperPOD</text>
<text x="163.2" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">4 SU · 127 DGX = 1.016 GPU</text>
<text x="70" y="104" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×4</text>
<line x1="462" y1="100" x2="494" y2="100" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="104" text-anchor="start" style="fill:var(--c-text);font-size:12px">InfiniBand&#x27;de DP/PP</text>
<rect x="90" y="124" width="366" height="32" rx="6" style="fill:var(--c-warn);fill-opacity:.10;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="102" y="144" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Scalable Unit</text>
<text x="208.95" y="144" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">32 DGX = 256 GPU</text>
<text x="80" y="144" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×32</text>
<line x1="462" y1="140" x2="494" y2="140" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="144" text-anchor="start" style="fill:var(--c-text);font-size:12px">leaf switch, rail&#x27;ler</text>
<rect x="100" y="164" width="356" height="32" rx="6" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="112" y="184" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX H100</text>
<text x="183.2" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">8 GPU · NVSwitch · 900 GB/s NVLink</text>
<text x="90" y="184" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×8</text>
<line x1="462" y1="180" x2="494" y2="180" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="184" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP kolektifleri (NVLink)</text>
<rect x="110" y="204" width="346" height="32" rx="6" style="fill:var(--c-success);fill-opacity:.10;stroke:var(--c-success);stroke-width:1.2"/>
<text x="122" y="224" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">H100 SXM</text>
<text x="193.2" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">132 SM · 80 GB HBM3 · 3,35 TB/s</text>
<text x="100" y="224" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×132</text>
<line x1="462" y1="220" x2="494" y2="220" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px">HBM bant genişliği (Roofline)</text>
<rect x="120" y="244" width="336" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="132" y="264" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SM</text>
<text x="160.3" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">≤ 64 warp = 2.048 thread</text>
<text x="110" y="264" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×64</text>
<line x1="462" y1="260" x2="494" y2="260" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="264" text-anchor="start" style="fill:var(--c-text);font-size:12px">register, shared memory</text>
<rect x="130" y="284" width="326" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="142" y="304" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Warp</text>
<text x="184.6" y="304" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">32 thread, ortak komut akışı</text>
<text x="120" y="304" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">×32</text>
<line x1="462" y1="300" x2="494" y2="300" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="304" text-anchor="start" style="fill:var(--c-text);font-size:12px">gecikme gizleme</text>
<rect x="140" y="324" width="316" height="32" rx="6" style="fill:var(--c-accent);fill-opacity:.10;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152" y="344" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Thread</text>
<text x="208.9" y="344" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">tek yürütme örneği</text>
<text x="130" y="344" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px;font-family:var(--font-mono)">1</text>
<line x1="462" y1="340" x2="494" y2="340" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="500" y="344" text-anchor="start" style="fill:var(--c-text);font-size:12px">dal ayrışması</text>
<text x="16" y="376" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">1.024 = 128 × 8, 4 SU&#x27;nun tasarım kapasitesidir; referans SuperPOD bir yuvayı UFM&#x27;e verir: 127 DGX = 1.016 GPU.</text>
</svg>

### 1.1. Pretraining neden ağ tasarımıyla başlar?

Sıfırdan temel model eğitimi (Pre-training), sistem mimarisinin en acımasız stres testidir.
- **Hesaplama Profili:** Milyarlarca token boyunca sürekli $6P$ FLOP/token hesaplama döner. Tensor Core'lar haftalarca %100 yük altında tutulur.
- **İletişim Talebi:** Katman içi Tensor Parallelism her Transformer bloğunda iki kez AllReduce üretirken, Pipeline Parallelism düğümler arasında sürekli aktivasyon fırlatır; Data Parallelism ise adım sonunda yüzlerce megabaytlık gradient senkronizasyonu yapar.
- **Ağ Önceliği:** Pre-training işinde ağ sonradan eklenen bir kablolama detayı değildir; sistemin omurgasıdır. Eğer raylar arasında 1:1 non-blocking bisection bant genişliği kuramazsanız, kümedeki binlerce H100 GPU vaktinin yarısını kolektif iletişim bariyerlerinde kilitli geçirir ve MFU %20'nin altına çakılır.

### 1.2. Full fine-tuning ve PEFT neden farklı kapasite modellerine sahiptir?

Bir kuruluşa 70 milyar parametreli bir modeli uyarlama görevi verildiğinde iki ana yol vardır: **Tam İnce Ayar (Full Fine-Tuning)** ve **Parametre Verimli İnce Ayar (PEFT / LoRA)**. Bu iki yöntemin donanım faturası taban tabana zıttır:

<!-- restored-svg:tr_full:39 -->
<svg viewBox="0 0 720 294" role="img" aria-label="405B modelin eğitim durumunun yığılmış çubuğu, parametre başına 16 byte: BF16 ağırlık 2 byte yani 0,81 TB, BF16 gradyan 2 byte yani 0,81 TB, FP32 Adam birinci ve ikinci momentleri 4&#x27;er byte yani 1,62&#x27;şer TB, opsiyonel FP32 master kopya 4 byte yani 1,62 TB; toplam 6,48 TB. GPU başına: TP 8 × PP 16 ile 128 GPU&#x27;ya ideal bölünürse 50,6 GB; optimizer durumu DP 128&#x27;e de bölünürse yaklaşık 12,96 GB. İkisi de 80 GB HBM çizgisinin altındadır, ama aktivasyon, tampon ve fragmentasyon hariçtir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="m4-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">405B modelin eğitim durumu: parametre başına 16 byte</text>
<rect x="17" y="40" width="84" height="54" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<text x="59.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">BF16 ağırlık</text>
<text x="59.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2 B/param</text>
<text x="59.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,81 TB</text>
<rect x="103" y="40" width="84" height="54" rx="2" style="fill:var(--c-success);fill-opacity:.45"/>
<text x="145.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">BF16 gradyan</text>
<text x="145.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2 B/param</text>
<text x="145.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0,81 TB</text>
<rect x="189" y="40" width="170" height="54" rx="2" style="fill:var(--c-warn);fill-opacity:.38"/>
<text x="274.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Adam m (FP32)</text>
<text x="274.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="274.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,62 TB</text>
<rect x="361" y="40" width="170" height="54" rx="2" style="fill:var(--c-warn);fill-opacity:.38"/>
<text x="446.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Adam v (FP32)</text>
<text x="446.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="446.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,62 TB</text>
<rect x="533" y="40" width="170" height="54" rx="2" style="fill:var(--c-accent-2);fill-opacity:.45"/>
<text x="618.0" y="58" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:600">FP32 master (opsiyonel)</text>
<text x="618.0" y="74" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4 B/param</text>
<text x="618.0" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1,62 TB</text>
<text x="704" y="112" text-anchor="end" style="fill:var(--c-text);font-size:12.5px;font-weight:700">toplam 16 B × 405 milyar = 6,48 TB</text>
<text x="16" y="140" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">GPU başına (aktivasyon ve tampon hariç):</text>
<text x="240" y="171" text-anchor="end" style="fill:var(--c-text);font-size:11.5px">÷ 128 (TP 8 × PP 16)</text>
<rect x="250" y="158" width="238.118" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="494.11764705882354" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">50,6 GB</text>
<text x="240" y="207" text-anchor="end" style="fill:var(--c-text);font-size:11.5px">+ optimizer DP 128&#x27;e bölünür</text>
<rect x="250" y="194" width="60.9882" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="316.98823529411766" y="208" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">≈ 12,96 GB</text>
<line x1="626.471" y1="148" x2="626.471" y2="226" style="stroke:var(--c-danger);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="626.4705882352941" y="240" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">80 GB HBM</text>
<text x="16" y="266" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">İdealize bölüşüm: AllGather tamponları, katman dengesizliği, aktivasyon ve fragmentasyon üstüne eklenir;</text>
<text x="16" y="282" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">FSDP kesin bir sayı garanti etmez.</text>
</svg>

Bu iki yaklaşımın bellek ve donanım matematiği:

1. **Full Fine-Tuning (Parametre Başına 16 Bayt):**
   Modelin tüm parametreleri eğitilir. FP16/BF16 karma hassasiyetli AdamW optimizer için statik bellek:
   - Model Ağırlıkları (FP16/BF16): $2\text{ bayt/param}$
   - Gradyanlar (FP16/BF16): $2\text{ bayt/param}$
   - AdamW Durumları (FP32 master ağırlık, momentum, varyans): $4 + 4 + 4 = 12\text{ bayt/param}$
   - **Toplam Statik Durum:** Parametre başına tam **$16\text{ bayt}$**!
   70B model için statik bellek:
   $$M_{\mathrm{static}} = 70 \times 10^9 \times 16\text{ bayt} \approx 1.120\text{ GB} = 1.12\text{ TB}$$
   Aktivasyon tensörleri ve geçici arabellekler eklendiğinde en az $1.5\text{ TB}$ HBM gerekir. Bu iş yükü en az iki adet 8-GPU'lu DGX H100 sunucusu ($16 \times 80\text{ GB} = 1.280\text{ GB}$) ve FSDP/ZeRO-3 gerektirir.
2. **PEFT / LoRA (Low-Rank Adaptation):**
   Orijinal model ağırlıkları dondurulur (frozen, $2\text{ bayt/param}$). Yalnızca araya eklenen düşük dereceli matrisler ($A$ ve $B$, rank $r=16$ veya $64$) eğitilir.
   - Dondurulmuş Ağırlıklar (FP16): $70\text{B} \times 2\text{ bayt} = 140\text{ GB}$ (veya 4-bit QLoRA ile $35\text{ GB}$).
   - Eğitilebilir Parametreler: Genellikle modelin %0.1 ile %0.5'i kadardır ($\approx 350\text{ milyon}$ parametre).
   - Optimizer Durumu: Yalnızca adaptörler için tutulur ($350\text{M} \times 16\text{ bayt} \approx 5.6\text{ GB}$).
   - **Sonuç:** LoRA ile 70B bir model, tek bir 8-GPU sunucuda hatta QLoRA ile tek veya iki adet 80 GB GPU üzerinde rahatlıkla eğitilebilir!

### 1.3. Inference neden ağırlıkları sığdırmaktan fazlasıdır?

Üretim çıkarımında (Production Inference) yaygın bir tuzak: "70B model FP16'da $140\text{ GB}$ yer tutar; elimde iki adet 80 GB GPU ($160\text{ GB}$) var, öyleyse bu modeli rahatlıkla sunabilirim."
Bu varsayım **üretimde anında felaketle sonuçlanır**.

Çünkü çıkarım belleği üç farklı bileşenden oluşur:
$$
M_{\mathrm{inference}} = M_{\mathrm{weights}} + M_{\mathrm{KV\_cache}} + M_{\mathrm{runtime\_buffers}}
$$

1. **Ağırlıklar ($M_{\mathrm{weights}}$):** $140\text{ GB}$.
2. **Çalışma Zamanı Tamponları ($M_{\mathrm{runtime\_buffers}}$):** CUDA graph'ler, NCCL arabellekleri, aktivasyon tensörleri için en az $10 - 15\text{ GB}$.
3. **Kalan Alan ($M_{\mathrm{KV\_cache}}$):** İki GPU'da $160 - 140 - 10 = 10\text{ GB}$ kalır!
   - 70B bir modelde (Grouped-Query Attention, $8$ KV kafası, $128$ kafa boyutu, $80$ katman) 1 token'lık KV cache belleği:
     $$\text{KV per token} = 2 \times 2\text{ bayt} \times 80\text{ katman} \times 8\text{ kafa} \times 128 = 327.680\text{ bayt} \approx 320\text{ KB/token}$$
   - $10\text{ GB}$'lık arabellek yalnızca $32.000$ token barındırabilir. 8 kullanıcının $4\text{K}$ bağlamla sorgu attığı anda sistem Out-of-Memory (OOM) vererek çöker!
4. **Çözüm:** 70B model çıkarımı için iki değil, en az **dört adet 80 GB GPU** ($320\text{ GB}$ HBM) tahsis edilmelidir; böylece $160\text{ GB}$ net dinamik KV cache alanı kalır ve onlarca eşzamanlı kullanıcı yüksek hızla sunulabilir.

---

## 2. GPU düğümünü tasarlamak

Bir yapay zekâ sunucusu seçilirken broşürdeki TFLOP/s değerine değil, düğüm içi bileşenlerin birbirine nasıl dengelendiğine bakılır.

### 2.1. HBM kapasitesini HBM bant genişliğinden ayırın

HBM'in iki boyutu vardır ve ikisi farklı iş yüklerini kısıtlar:
- **HBM Kapasitesi (GB):** Modele ne kadar ağırlığın ve ne büyüklükte bir KV cache havuzunun sığacağını belirler (Kapasite tavanı).
- **HBM Bant Genişliği (TB/s):** Bellekteki ağırlıkların Tensor Core'lara ne hızla pompalanacağını belirler (Hız tavanı).

İkinci yazımızda gördüğümüz gibi:
- **Prefill Fazı:** Büyük prompt batch'leri matris-matris çarpımı (GEMM) üretir; yüksek aritmetik yoğunluğa sahiptir ve **Compute-Bound**'dur (Tensor Core TFLOP/s gücü belirleyicidir).
- **Decode Fazı:** Model token'ları tek tek üretirken her adımda tüm $140\text{ GB}$ ağırlık matrisini HBM'den SM'lere baştan sona okumak zorundadır (GEMV). Aritmetik yoğunluk düşüktür ($\approx 1$) ve sistem tamamen **Memory-Bandwidth Bound** rejimindedir.
- Bir H100 SXM5 ($3.35\text{ TB/s}$ HBM) ile H100 NVL ($3.9\text{ TB/s}$) veya H200 ($4.8\text{ TB/s}$) arasındaki çıkarım hız farkını belirleyen şey TFLOP/s değil, doğrudan bu bellek bant genişliğidir.

### 2.2. Precision bir format seçimi değil, veri yolu kararıdır

FP32'den FP16/BF16'ya, oradan FP8 ve NVFP4'e geçiş yalnızca bir doğruluk (accuracy) tercihi değildir; tüm fiziksel veri yollarının yükünü değiştiren bir mimari karardır:

$$
\text{Bant Genişliği İhtiyacı} \propto \text{Hassasiyet (Bayt)}
$$

- FP16 ($2\text{ bayt}$) yerine FP8 ($1\text{ bayt}$) kullanıldığında:
  1. HBM'den okunan ağırlık hacmi yarıya iner; decode fazındaki token/s üretim hızı teorik olarak ikiye katlanır.
  2. NVLink ve PCIe üzerinden taşınan aktivasyon tensörleri %50 küçülür; kolektif iletişim bariyerleri hızlanır.
  3. Tensor Core'ların FP8 matris birimleri BF16'ya göre tam iki kat fazla FLOP/s üretir ($990 \to 1.979\text{ TFLOP/s}$).

### 2.3. NVLink alanı hangi işleri düğüm içinde tutmalı?

NVIDIA NVSwitch, sunucu içindeki 8 GPU arasında çift yönlü $900\text{ GB/s}$ all-to-all bant genişliği ve $0.5\ \mu\text{s}$ gecikme sunar. Bu alan kıymetli bir fiziksel kaynaktır:
- **Tutulması Zorunlu Olanlar:** Tensor Parallelism (TP). Her Transformer bloğunda iki kez tetiklenen yüksek frekanslı AllReduce operasyonları kesinlikle bu $900\text{ GB/s}$ alanında kalmalıdır.
- **Dışarı Çıkarılabilecekler:** Pipeline Parallelism (PP) ve Data Parallelism (DP/FSDP). Bu operasyonlar katman sınırlarında veya step sonunda çalıştığı için InfiniBand / RoCE üzerinden düğüm dışına taşınabilir.

### 2.4. MIG, MPS ve time-slicing hangi fiziksel sorunu çözer?

Kurumsal platformlarda her iş yükü 8 GPU'luk tam bir düğümü doldurmaz. Küçük embedding modelleri, değerlendirme (eval) testleri veya geliştirme ortamları için GPU sanallaştırma stratejileri:
1. **MIG (Multi-Instance GPU):** H100 GPU'yu 7 adede kadar donanımsal olarak izole bağımsız instance'a böler. Her instance kendi ayrılmış SM'lerine, bellek kontrolcülerine ve crossbar yollarına sahiptir. Güvenliğin ve hata izolasyonunun katı olduğu çok kiracılı (multi-tenant) kurumsal platformlar için altın standarttır.
2. **MPS (Multi-Process Service):** Birden fazla sürecin aynı CUDA context'ini paylaşarak tek bir GPU üzerinde eşzamanlı çalışmasını sağlar. Küçük batch'li çıkarım servislerinde GPU kaynaklarını doyurmak için idealdir.
3. **Time-Slicing:** Klasik zaman paylaşımlı çalıştırma; derin öğrenmede büyük bağlam değiştirme (context-switch) gecikmeleri nedeniyle önerilmez.

### 2.5. Bir düğüm seçimini hangi çalışma kâğıdıyla savunursunuz?

Bir GPU sunucusu seçimi yönetim kuruluna sunulurken şu 5 boyutlu doğrulama tablosu doldurulmalıdır:

| Değerlendirme Boyutu | Minimum Eşik Değeri | Kritik Gerekçe |
|---|---|---|
| **GPU Form Faktörü** | SXM5 (PCIe yerine) | $900\text{ GB/s}$ NVSwitch tam bağlılık (PCIe $128\text{ GB/s}$ ile sınırlıdır) |
| **GPU Başı HBM** | $\ge 80\text{ GB}$ (HBM3/HBM3e) | 70B modellerin TP=4 veya TP=8 ile KV cache taşmadan çalışabilmesi |
| **Host Belleği (RAM)** | $\ge 2\text{ TB}$ (2x GPU HBM) | Dataloader önbellekleme ve iki kademeli asenkron checkpoint snapshot |
| **NIC / Kumaş Bant Genişliği** | 8x 400G OSFP ($400\text{ Gb/s}$ per GPU) | Ray-optimize fat-tree ağında GPUDirect RDMA ile tek atlama gecikme |
| **Güç ve Soğutma** | $10.2\text{ kW}$ sürekli güç desteği | Düğümün termal kısıtlamaya girmeden tepe frekansta çalışabilmesi |

---

## 3. Düğümden kümeye: Ağ ve fiziksel yerleşim

Sunucu düzeyinde doğru seçimleri yaptıktan sonra, bu sunucuları veri merkezinde birbirine bağlayan ağ kumaşını tasarlamak gerekir.

### 3.1. Paralellik boyutu fiziksel trafik üretir

Paralellik soyut bir algoritma tercihi değildir; fiziksel kablolarda akan bit akıntılarıdır:
- $TP$ her katmanda yüzlerce megabaytlık yerel AllReduce fırtınası üretir.
- $PP$ ardışık sunucular arasında mikro-batch aktivasyon aktarımları üretir.
- $DP / FSDP$ tüm küme genelinde yüzlerce gigabaytlık gradient AllGather ve ReduceScatter trafiği basar.
- $EP$ (MoE) kümedeki tüm sunucuları birbirine bağlayan devasa bir AllToAll matris transpozesi yaratır.

### 3.2. GPU–NIC yakınlığı neden kablo ayrıntısı değildir?

Birinci yazımızda gördüğümüz gibi, bir DGX H100 sunucusunda her GPU doğrudan kendi yerel PCIe Gen5 switch'i üzerinden belirli bir ConnectX-7 NIC'e bağlıdır:
- GPU 0 $\to$ NIC 0 (NUMA Düğümü 0)
- GPU 7 $\to$ NIC 7 (NUMA Düğümü 1)
Eğer veri merkezi teknisyeni kablolamayı yaparken veya Kubernetes CNI ağ eklentisi yapılandırılırken GPU 0'ın trafiği yanlışlıkla NIC 7'ye atanırsa, paketler CPU'lar arası UPI köprüsüne takılır. Ağ kartı $400\text{ Gb/s}$ olsa dahi gecikme 3 katına çıkar ve bisection bant genişliği felç olur.

<!-- restored-svg:tr_full:24 -->
<svg viewBox="0 0 720 304" role="img" aria-label="Tek bir DGX H100 dört ayrı ağa bağlanır. Compute ağı, 400 Gb/s InfiniBand&#x27;li sekiz ConnectX-7, GPU kolektiflerini ve dağıtık eğitimi taşır; AllReduce ve AllToAll depolamayla boğuşmasın diye ayrıdır. Depolama ağı, iki çift portlu ConnectX-7, veri kümelerini ve checkpoint&#x27;leri taşır; yüksek hacimli burst I/O ayrı planlanır. 100 GbE in-band yönetim ağı işletim sistemi, servis ve scheduler trafiğini taşır. 1 GbE out-of-band yönetim ağı BMC&#x27;ye ulaşır; açma-kapama, konsol ve arıza müdahalesi içindir ve işletim sistemi bozulsa bile çalışır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="fn-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Tek DGX, dört ayrı ağ (NVIDIA DGX H100 SuperPOD referansı)</text>
<rect x="16" y="44" width="120" height="228" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="76" y="150" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:700">DGX H100</text>
<text x="76" y="168" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8 GPU</text>
<line x1="136" y1="69" x2="168" y2="69" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="44" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="182" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Compute ağı</text>
<text x="182" y="82" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × CX-7 · 400 Gb/s IB</text>
<text x="380" y="64" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU kolektifleri, dağıtık eğitim</text>
<text x="380" y="82" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">AllReduce/AllToAll depolamayla boğuşmasın</text>
<line x1="136" y1="127" x2="168" y2="127" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="102" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="182" y="122" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Depolama ağı</text>
<text x="182" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">2 × çift portlu CX-7</text>
<text x="380" y="122" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">veri kümeleri, checkpoint</text>
<text x="380" y="140" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">yüksek hacimli burst I/O ayrı planlansın</text>
<line x1="136" y1="185" x2="168" y2="185" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="160" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="182" y="180" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">In-band yönetim</text>
<text x="182" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">100 GbE</text>
<text x="380" y="180" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">işletim sistemi, servisler, scheduler</text>
<text x="380" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">günlük operasyon trafiği ayrı yönetilsin</text>
<line x1="136" y1="243" x2="168" y2="243" marker-end="url(#fn-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="170" y="218" width="534" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="182" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Out-of-band (BMC)</text>
<text x="182" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1 GbE</text>
<text x="380" y="238" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">açma-kapama, konsol, arıza müdahalesi</text>
<text x="380" y="256" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-style:italic">işletim sistemi bozulsa bile erişim kalsın</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">BMC = anakart yönetim denetleyicisi; UFM InfiniBand ağını yönetir ve izler; Slurm gibi bir scheduler işleri yerleştirir.</text>
</svg>

### 3.3. Rail hizalama ve bisection bant genişliği

Büyük bir AI kümesinde ağ trafiğinin omurgayı (spine switch'leri) tıkamasını önlemenin tek yolu **Rail Hizalaması (Rail Optimization)** kurgulamaktır:

1. **Dört Ayrı Fiziksel Ağ:**
   Bir DGX sunucusu veri merkezinde dört bağımsız ağa bağlanır:
   - **Compute Fabric (Hesaplama Ağı):** 8 adet $400\text{ Gb/s}$ InfiniBand / RoCE portu. Sadece GPU kolektiflerini taşır; asla depolama veya yönetim trafiğiyle karışmaz!
   - **Storage Fabric (Depolama Ağı):** 2 adet çift portlu $400\text{ Gb/s}$ NIC ($800\text{ Gb/s}$ toplam). Checkpoint yazma ve veri okuma dalgalarını compute kumaşından izole eder.
   - **In-Band Management:** $100\text{ GbE}$ Ethernet. İşletim sistemi, Kubernetes API, scheduler ve log telemetrisini taşır.
   - **Out-of-Band (OOB) BMC:** $1\text{ GbE}$ bağımsız yönetim ağı. Sunucunun IPMI/BMC arayüzüne bağlanır; işletim sistemi çökse dahi uzaktan güç yönetimi ve konsol erişimi sağlar.
2. **1:1 Non-Blocking Clos:**
   Compute ağında Scalable Unit (SU) içindeki leaf switch'ler ile spine katmanı arasında 1:1 oversubscription oranı korunur. Bu sayede kümedeki 1.024 GPU'dan herhangi ikisi aynı anda tam $50\text{ GB/s}$ hızında iletişim kurabilir.

### 3.4. Ağ rezervi neden ortalama kullanıma göre seçilmez?

Web uygulamalarında ağ kapasitesi "ortalama bant genişliğine" göre planlanır. Yapay zekâ altyapısında bunu yapmak intihardır.
Çünkü dağıtık eğitim trafiği homojen değildir; **anlık patlamalı (bursty)** ve senkronizedir:
- Backward pass bittiği anda tüm 1.024 GPU aynı milisaniyede ağa yüklenir.
- Eğer ağ kapasitesini ortalama kullanıma göre (örneğin $50\text{ Gb/s}$) boyutlandırırsanız, bu senkron patlama anında switch arabellekleri taşar, paketler düşer, PFC duraklatma çerçeveleri tüm rayları kilitler ve eğitim adımı dakikalarca duraklar. Ağ daima **tepe ani patlama (peak burst) kapasitesine** göre boyutlandırılmalıdır.

---

## 4. GPU dışındaki fiziksel sınırlar: Depolama, güç ve arızalar

Bir veri merkezinde yapay zekâ platformunun büyümesini engelleyen asıl sınır genellikle GPU tedariki değil, tesisin fiziksel altyapısıdır.

### 4.1. Depolama iki ayrı iş yapar

Dördüncü yazımızda kanıtladığımız gibi depolama mimarisi iki farklı iş yüküne bölünmelidir:
1. **Veri Kümesi Okuma (Data Ingestion):** Yüksek IOPS, düşük gecikme ve sıfır metadata kilitlenmesi ister (GPFS, Lustre veya NVMe-oF önbellekleri).
2. **Checkpoint Yazma (Snapshot Drain):** Yüzlerce gigabaytlık ani yazma bant genişliği ister ($> 100 - 250\text{ GB/s}$ sürdürülebilir write throughput).

### 4.2. Güç ve soğutma GPU sayısından önce kontrol edilmelidir

Bir DGX H100 sunucusunun maksimum güç tüketimi **$10.2\text{ kW}$**'tır!
- Standart bir kurumsal sunucu rafı (rack) 4 adet DGX H100 barındırır:
  $$\text{Raf Gücü} = 4 \times 10.2\text{ kW} = 40.8\text{ kW}$$
- Geleneksel kurumsal veri merkezleri raf başına $5 - 10\text{ kW}$ soğutma kapasitesine göre inşa edilmiştir. $40\text{ kW}$'lık bir AI rafını hava ile soğutmak neredeyse imkânsızdır; fanlar maksimum devirde dönse dahi GPU'lar termal kısıtlamaya (thermal throttling) girer.
- **Sıvı Soğutma Zorunluluğu:** Yeni nesil AI veri merkezleri doğrudan yongaya sıvı soğutma (Direct-to-Chip Liquid Cooling - DLC) veya daldırmalı soğutma (Immersion Cooling) kullanmak zorundadır.
- **PUE (Power Usage Effectiveness):** Veri merkezi PUE değeri 1.35 ise, $1\text{ MW}$'lık GPU hesaplama gücü çalıştırmak için tesisten $1.35\text{ MW}$ elektrik çekilmelidir. Elektrik trafosu kapasitesi hesaplanmadan GPU satın alınamaz!

### 4.3. Hata alanlarını paralellik gruplarıyla birlikte düşünün

Bir veri merkezinde arıza alanları (Failure Domains / Blast Radius) hiyerarşiktir:
`Tek GPU -> PCIe Riser -> Sunucu Güç Kaynağı -> PDU (Güç Dağıtım Birimi) -> Raf -> Switch -> Sıvı Soğutma Döngüsü`

Paralellik grupları bu hata alanlarına göre yerleştirilmelidir:
- Tek bir model kopyasının Pipeline Parallelism zincirini tek bir PDU'ya veya tek bir rafa bağlarsanız, o raftaki tek bir güç kesintisi tüm model kopyasını yok eder.
- Data Parallelism kopyaları bağımsız raflara ve bağımsız güç beslemelerine dağıtılmalıdır.

### 4.4. Yedek kapasite boşta duran israf değildir

1.024 GPU'luk bir kümede ortalama her 10–20 saatte bir donanım arızası yaşanır.
Eğer kümenizde yedek (spare) düğüm yoksa, tek bir sunucu bozulduğunda arızalı parça gelene kadar (RMA süreci: 3–7 gün) tüm 1.024 GPU'luk eğitim işi durur!
Küme kapasitesinin **%5 ile %10'unun (örneğin 1.024 GPU için 10–15 adet DGX sunucusunun)** sıcak yedek (hot-spare) olarak ayrılması israf değildir; sistemin aylık çalışma verimliliğini (availability) %70'ten %95'e çıkaran en kritik yatırımdır.

---

## 5. 70B model için donanım referans tasarımı

Tüm bu teoriyi kurumsal ölçekte somut bir mühendislik projesine dönüştürelim: **70 Milyar Parametreli Bir LLM için Uçtan Uca Platform Tasarımı**.

### 5.1. Varsayımları önce yazın

- **Model:** Llama 3 70B ($L=80\text{ katman}, H=8192, A=64\text{ attention head}, K=8\text{ KV head (GQA)}$).
- **Hassasiyet:** BF16 ($2\text{ bayt/parametre}$).
- **Bağlam Penceresi:** $8.192$ token (Standart) ile $32.768$ token (Genişletilmiş).

### 5.2. Serving düğümü

Amaç: Düşük gecikme (TTFT $< 300\text{ ms}$, TPOT $< 25\text{ ms/token}$) ve yüksek eşzamanlılık.
- **Donanım:** 1 adet sunucu, 4x NVIDIA H100 NVLink ($320\text{ GB}$ HBM3).
- **Paralellik:** $TP = 4$ (Tüm katman AllReduce'ları yerel NVSwitch üzerinden akar).
- **Bellek Ayrımı:**
  - Model Ağırlıkları (BF16): $140\text{ GB}$ ($35\text{ GB/GPU}$).
  - Runtime ve Aktivasyon Tamponları: $20\text{ GB}$.
  - Kalan Dinamik KV Cache Alanı: $320 - 140 - 20 = \mathbf{160\text{ GB}}$!
- **Kapasite Hesabı:** GQA sayesinde token başına KV cache yalnızca $320\text{ KB}$ tutar. $160\text{ GB}$ bellek, aynı anda **$500.000$ aktif token** barındırabilir! Bu kapasite, sistemin 120'den fazla eşzamanlı kullanıcıyı $4\text{K}$ bağlamla rahatlıkla sunmasını sağlar.

### 5.3. Fine-tuning düğümü

Amaç: Kurumsal verilerle haftalık LoRA ve dönemsel Full Fine-Tuning çalıştırmak.
- **Donanım:** 2 adet DGX H100 sunucusu ($16 \times \text{H100 } 80\text{ GB} = 1.280\text{ GB}$ HBM).
- **Paralellik:** $TP = 8$ (Her sunucu kendi içinde bir TP grubu), $DP = 2$ (İki sunucu arasında PyTorch FSDP / ZeRO-3).
- **Full Fine-Tuning Başarımı:** 70B modelin $1.120\text{ GB}$'lık statik durumu 16 GPU'ya bölünür ($70\text{ GB/GPU}$). FlashAttention ve activation checkpointing ile $1.280\text{ GB}$ toplam HBM içine tam olarak sığar; hiçbir CPU offload gerektirmeden maksimum MFU ile eğitilir.

### 5.4. Pretraining havuzu

Amaç: Şirkete özel 70B modeli sıfırdan veya geniş çaplı pretraining ile eğitmek.
- **Donanım:** 64 adet DGX H100 sunucusu (512 GPU, 8 Scalable Unit).
- **Ağ Kumaşı:** 8-rail InfiniBand Quantum-2 ($400\text{ Gb/s}$ non-blocking fat-tree).
- **Paralellik:** $TP = 8, PP = 4, DP = 16 \implies 8 \times 4 \times 16 = 512\text{ GPU}$.
- **Depolama:** Saniyede en az $120\text{ GB/s}$ kesintisiz yazma hızına sahip Lustre paralel depolama kümesi (iki kademeli asenkron checkpoint ile step kaybı %5'in altında tutulur).

### 5.5. Üç havuzu neden otomatik olarak birleştirmiyoruz?

"Tüm GPU'ları tek bir Kubernetes kümesine koyalım, gündüz serving yapsın, gece training yapsın" fikri pratikte neden çöker?
1. **SLA ve Determinizm Çatışması:** Serving işleri anlık yanıt gecikmesi (P-99 latency) garantisi ister; training işleri ise haftalarca süren kesintisiz kaynak rezervasyonu (Gang Scheduling) ister.
2. **Ağ Topolojisi Farkı:** Serving sunucusu sadece düğüm içi NVLink'e ihtiyaç duyar, scale-out InfiniBand kumaşına gerek yoktur; training havuzu ise milyonlarca dolarlık non-blocking rail kumaşı gerektirir. Serving'i InfiniBand kumaşında çalıştırmak pahalı ağ portlarını boşa harcamaktır.
3. **Güvenlik ve İzolasyon:** Dış dünyaya açık serving API'si ile şirketin en değerli varlığı olan temel model eğitim ortamının aynı kernel/ağ katmanını paylaşması kurumsal güvenlik politikalarını ihlal eder.

### 5.6. 70B tasarımında kararlar nasıl birbirine bağlanır?

Uçtan uca mimari ilişki matrisi:
- *İş Yükü:* Llama 3 70B Çıkarımı $\to$ *Darboğaz:* Decode fazı HBM bant genişliği $\to$ *Karar:* $TP=4$ H100 SXM5 ($13.4\text{ TB/s}$ toplam bant genişliği).
- *İş Yükü:* 70B Full Fine-Tuning $\to$ *Darboğaz:* AdamW bellek patlaması ($1.12\text{ TB}$) $\to$ *Karar:* 2x DGX (16 GPU) + FSDP ZeRO-3.
- *İş Yükü:* 70B Pre-training $\to$ *Darboğaz:* Kolektif iletişim ve MTBF $\to$ *Karar:* 512 GPU rail-optimized InfiniBand + Young-Daly asenkron checkpointing.

---

## 6. Satın almadan önce doğrulama merdiveni

Bir yapay zekâ kümesi satın alırken satıcının broşüründeki iddialara güvenilmez; katı bir **Doğrulama Merdiveni (Validation Ladder)** uygulanır:

<!-- restored-svg:tr_full:29 -->
<svg viewBox="0 0 720 306" role="img" aria-label="Aşağıdan yukarı doğrulanan altı katmanlı ölçüm piramidi: 1, fiziksel hatlar; link durumu, optik ve hata sayaçları. 2, host veri yolu; PCIe Gen5, NUMA ve GPU↔NIC yakınlığı, nvidia-smi topo -m ve lspci ile. 3, noktadan noktaya bant genişliği ve gecikme için RDMA mikrobenchmark. 4, all_reduce_perf busbw, AllGather ve AllToAll gibi NCCL kolektifleri. 5, model step süresi ve hesap/iletişim örtüşmesi. 6, time-to-quality, MFU ve p99 dahil iş yükü SLO&#x27;su ve maliyet. Her katmanın baz çizgisi ancak alttaki katman temizse anlamlıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="mp2-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Ağ ölçüm piramidi: aşağıdan yukarı doğrula</text>
<path d="M150 40 L190 40 L210 76 L130 76 Z" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<line x1="214" y1="58" x2="322" y2="58" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="55" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">6 · İş yükü SLO &amp; maliyet</text>
<text x="330" y="71" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">time-to-quality, MFU, p99</text>
<path d="M130 80 L210 80 L230 116 L110 116 Z" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="234" y1="98" x2="322" y2="98" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">5 · Model step süresi</text>
<text x="330" y="111" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hesap/iletişim örtüşmesi</text>
<path d="M110 120 L230 120 L250 156 L90 156 Z" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<line x1="254" y1="138" x2="322" y2="138" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="135" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">4 · NCCL kolektifleri</text>
<text x="330" y="151" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">all_reduce_perf busbw, AllGather, AllToAll</text>
<path d="M90 160 L250 160 L270 196 L70 196 Z" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<line x1="274" y1="178" x2="322" y2="178" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="175" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">3 · RDMA mikrobenchmark</text>
<text x="330" y="191" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">noktadan noktaya bant genişliği ve gecikme</text>
<path d="M70 200 L270 200 L290 236 L50 236 Z" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<line x1="294" y1="218" x2="322" y2="218" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="215" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">2 · Host veri yolu</text>
<text x="330" y="231" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PCIe Gen5, NUMA, GPU↔NIC: nvidia-smi topo -m, lspci</text>
<path d="M50 240 L290 240 L310 276 L30 276 Z" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.2"/>
<line x1="314" y1="258" x2="322" y2="258" style="stroke:var(--c-border);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="330" y="255" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">1 · Fiziksel hatlar</text>
<text x="330" y="271" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">link durumu, optik, hata sayaçları</text>
<text x="16" y="294" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her katmanın baz çizgisi ancak alttaki katman temizse anlamlıdır; mesaj boyutlarını tarayın, yükseltme öncesi ve sonrası tekrarlayın.</text>
</svg>

### 6.1. Altı doğrulama katmanı

Tedarik öncesi ve kabul testlerinde işletilen 6 seviyeli doğrulama hiyerarşisi:

1. **Katman 1 — Fiziksel ve Optik Seviye:** Tüm optik modüllerin (transceiver) sinyal güçleri, kablo bükülme kayıpları ve switch port hata sayaçları (`SymbolError`, `LinkDown`) taranır. Sıfır hata şarttır.
2. **Katman 2 — Host Veri Yolu ve NUMA:** `nvidia-smi topo -m` ve `lspci -vvv` ile tüm GPU'ların PCIe Gen5 $x16$ ($32\text{ GT/s}$) hızında çalıştığı, hiçbir linkin $x8$'e düşmediği ve GPU-NIC NUMA afinitesinin doğru eşleştiği kanıtlanır.
3. **Katman 3 — P2P RDMA Microbenchmark:** `ibv_rc_pingpong` ile düğümler arası tek yön gecikmenin $< 1.5\ \mu\text{s}$ olduğu ve `ibv_rc_bw` ile tek bir rayın $\ge 47.5\text{ GB/s}$ ($380\text{ Gb/s}$) bant genişliğine ulaştığı doğrulanır.
4. **Katman 4 — NCCL Kolektifleri:** `nccl-tests` paketi çalıştırılır. 512 GPU arasında AllReduce bus bandwidth değerinin en az $360\text{ GB/s}$ (NVSwitch için) ve $45\text{ GB/s}$ (InfiniBand için) çıktığı test edilir.
5. **Katman 5 — Model Step ve Overlap:** Gerçek Llama 3 mimarisiyle 100 adımlık sentetik eğitim koşturulur. PyTorch Profiler ile backward hesaplaması ile gradient AllReduce'un ne kadar örtüştüğü ölçülür.
6. **Katman 6 — İş Yükü SLO'su ve FinOps:** 48 saatlik kesintisiz çalışma (soak run) yapılır. Saniyedeki faydalı token sayısı, MFU stabilitesi, joule başına token verimi ve amorti edilen maliyet doğrulanır.

### 6.2. Karar tablosunda hangi kanıt bulunmalı?

Tedarikçi seçiminde teklif dosyasına yalnızca fiyat değil, şu kanıtlar eklenmelidir:
- 1.024 GPU ölçeğinde `all_reduce_perf` bant genişliği histogramı (P-50, P-99 ve minimum değer).
- 24 saatlik sentetik soak testinde kaydedilen Xid hata sayısı ve termal kısıtlama grafiği.
- Depolama sisteminin 16.384 GPU simülasyonunda sunduğu doğrulanmış burst yazma IOPS ve throughput raporu.

### 6.3. En sık yapılan kapsam hatası

En yaygın bütçe hatası: **"Bütçenin %90'ını GPU sunucularına ayırıp, ağ ve tesisi ihmal etmek."**
Tipik bir AI süper bilgisayarında bütçe dağılımı kabaca şöyle olmalıdır:
- GPU Hesaplama Sunucuları: %55–60
- Yüksek Hızlı Ağ Kumaşı (Leaf, Spine, Kablolar, Optikler): %20–25
- Paralel Depolama Altyapısı: %10–15
- Tesis, Soğutma ve Güç Altyapısı: %10

Ağ ve depolamadan kısıp daha çok GPU almak, Formula 1 motorunu traktör şasisine takmaya benzer; araç hızlanmaz, parçalanır.

### 6.4. Bir kabul testinin başarısızlık ölçütü nedir?

Bir küme kabul testinde şu üç kırmızı çizgiden biri bile aşılırsa fatura onaylanmamalıdır:
1. **Flapping Link:** Tek bir InfiniBand veya RoCE linkinin test sırasında port hızını düşürmesi veya saniyede birden fazla hata paketi üretmesi.
2. **Straggler Varlığı:** NCCL testinde en yavaş düğümün ortalamadan %5'ten daha fazla sapması (P-100 gecikmesinin P-50'den %10 yüksek olması).
3. **Termal Düşüş:** 1 saatlik cuBLAS stres testinde tek bir GPU'nun bile çekirdek saat hızını nominal frekansın altına düşürmesi (`SW Thermal Throttling`).

---

## 7. Öğrendiklerimizi sınayalım

### 7.1. 70B BF16 ağırlıklar kaç GB tutar?
70 milyar parametre $\times 2\text{ bayt/parametre} = 140\times 10^9\text{ bayt} \approx \mathbf{140\text{ GB}}$ (İkili tabanda $\approx 130.4\text{ GiB}$).

### 7.2. Dört adet 80 GB GPU neden “180 GB boş KV alanı” demek değildir?
Toplam HBM $4 \times 80 = 320\text{ GB}$'tır. Ağırlıklar çıkarıldığında kağıt üzerinde $320 - 140 = 180\text{ GB}$ kalır.
Ancak:
- CUDA context ve runtime arabellekleri: GPU başına $\approx 1.5 - 2\text{ GB}$ ($8\text{ GB}$ toplam).
- PyTorch / vLLM geçici aktivasyon tamponları: $\approx 10 - 15\text{ GB}$.
- Bellek parçalanması (fragmentation) ve güvenlik marjı: $\approx 15 - 20\text{ GB}$.
Net kullanılabilir KV cache alanı en fazla **$135 - 145\text{ GB}$** kalır. Bu pay hesaba katılmadan kapasite planlanırsa sunucu OOM ile çöker.

### 7.3. Full fine-tuning neden serving’den çok daha fazla HBM ister?
Serving sırasında model yalnızca ileri geçiş (forward pass) yapar; ağırlıklar sabittir ($2\text{ B/param}$) ve geriye yayılım yapılmaz.
Full fine-tuning'de ise:
- Gradyanlar ($2\text{ B/param}$),
- FP32 AdamW master ağırlıklar, momentum ve varyans ($12\text{ B/param}$),
- Backward pass için saklanan ara aktivasyon tensörleri ($M_{\mathrm{act}}$) bellekte tutulmak zorundadır.
Toplam durum $16\text{ B/parametre}$ artı aktivasyonlara ulaşır; yani serving'in tam **8 ile 10 katı** daha fazla HBM talep eder!

### 7.4. NCCL testi iyi, training yavaş: Bu çelişki midir?
Kesinlikle çelişki değildir!
`nccl-tests` izole bir mikrobenchmark'tır; GPU belleğinde önceden hazır bekleyen sentetik tensörleri aktarır.
Gerçek eğitimde ise:
1. Diskten token okuyan Dataloader worker'ları CPU veya dosya sistemi metadata kilitlerine takılıyor olabilir (I/O stall).
2. Matris boyutları küçük olduğu için Tensor Core'lar döşeme verimsizliği yaşıyor olabilir.
3. Backward hesaplaması ile gradient iletişimi örtüşemiyor (overlap başarısızlığı) olabilir.
Sentetik ağ testi ağın çalıştığını kanıtlar; uygulamanın verimli çalıştığını kanıtlamaz.

---

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

- 70B bir model için çıkarım, ince ayar ve ön eğitim havuzlarının neden tamamen farklı topolojilerle tasarlandığını nasıl açıklarsınız?
- Bir DGX H100 sunucusunun $10.2\text{ kW}$'lık güç tüketiminin veri merkezi soğutma ve PUE planlamasına etkisini nasıl hesaplarsınız?
- Bellek muhasebesinde $16\text{ bayt/parametre}$ kuralını kullanarak bir modelin kaç GPU'ya sığacağını nasıl belirlersiniz?
- Tedarik aşamasında bir yapay zekâ kümesini doğrulamak için hangi 6 seviyeli ölçüm merdivenini işletirsiniz?
- MFU'yu artırmak adına batch size'ı körlemesine büyütmenin FinOps ve yakınsama açısından neden tehlikeli olduğunu nasıl savunursunuz?

---

## Bütün hikâye altı satırda

- Doğru AI donanımı en hızlı GPU seçilerek değil, tensörlerin nerede üretilip nerede tutulduğu hesaplanarak tasarlanır.
- Pre-training ağ odaklı, Full fine-tuning bellek odaklı, Inference ise HBM bant genişliği odaklı iş yükleridir.
- 70B model çıkarımı için 4 GPU, tam ince ayar için 16 GPU, pretraining için ise yüzlerce rail-optimize GPU gerekir.
- GPU utilization varlık gösterir; gerçek tasarruf ve mühendislik başarısı MFU ve joule başına token ile ölçülür.
- Raf başına $40\text{ kW}$ güç çeken AI sunucuları veri merkezinde sıvı soğutmayı bir lüks değil fiziksel zorunluluk yapar.
- Sağlıklı bir platform satın alması fiziksel optiklerden model SLO'suna uzanan 6 katmanlı test merdiveniyle kanıtlanır.

---

## Terimler sözlüğü

- **Hardware Signature:** Bir yapay zekâ iş yükünün hesaplama, bellek, ağ ve depolama kaynakları üzerindeki karakteristik baskı profili.
- **Full Fine-Tuning:** Modelin tüm ağırlıklarının geriye yayılım ve optimizer durumlarıyla birlikte güncellendiği, $16\text{ B/param}$ isteyen yöntem.
- **PEFT / LoRA:** Orijinal ağırlıkları dondurup yalnızca araya eklenen küçük adaptör matrislerini eğiten bellek dostu ince ayar yöntemi.
- **HBM Bant Genişliği:** Bellekteki ağırlıkların Tensor Core'lara aktarılma hızı ($3.35\text{ TB/s}$); çıkarımın decode fazını belirler.
- **MIG (Multi-Instance GPU):** Tek bir GPU'yu donanımsal olarak izole bağımsız örneklere bölen sanallaştırma teknolojisi.
- **Ray (Rail) Hizalaması:** Kümedeki tüm sunucuların aynı indeksli NIC'lerini aynı leaf switch'e bağlayarak trafiği tek atlamada tutan mimari.
- **DLC (Direct-to-Chip Liquid Cooling):** Raf başına $40\text{ kW}$'ı aşan yüksek yoğunluklu AI sunucularını soğutmak için sıvıyı doğrudan yonga üzerine ileten sistem.
- **PUE (Power Usage Effectiveness):** Veri merkezine giren toplam gücün BT ekipmanına ulaşan güce oranı; tesisin enerji verimliliğini gösterir.
- **Doğrulama Merdiveni:** Donanım kabulünde fizikten uygulama SLO'suna kadar işletilen 6 seviyeli katı test metodolojisi.

---

## Kaynaklar ve doğrulama bağlantıları

- Önceki yazı: [Yapay Zekâ Altyapısı (4): AI Kümesi Neden Yavaşlar? — Depolama, Arızalar ve MFU](post.html?slug=gpudan-ai-super-bilgisayarina)
- Birinci yazı: [Yapay Zekâ Altyapısı (1): GPU Mimarisi ve LLM Ölçekleme — SM, Tensör Çekirdekleri ve Bellek Duvarı](post.html?slug=gpu-mimarisi-ve-llm-olcekleme)
- **[K1] DGX H100 Sistem Mimarisi:** NVIDIA, [*DGX H100 Architecture Whitepaper*](https://resources.nvidia.com/en-us-dgx-systems/dgx-h100-system-architecture).
- **[K2] Llama 3 Altyapı ve Veri Merkezi Tasarımı:** Meta AI, [*The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783).
- **[K3] Kurumsal Veri Merkezi Güç ve Soğutma:** Uptime Institute, [*Data Center High-Density Cooling and Power Trends*](https://uptimeinstitute.com).
- **[K4] vLLM ve PagedAttention:** Woosuk Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180).
- **[K5] Megatron-LM ve 3D Paralellik:** NVIDIA, [*Efficient Large-Scale Language Model Training on GPU Clusters*](https://arxiv.org/abs/2104.04473).
