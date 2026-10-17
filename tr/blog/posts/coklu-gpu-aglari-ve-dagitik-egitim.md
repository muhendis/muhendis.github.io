Tek GPU’nun belleği veya hesaplama gücü yetmediğinde modeli birden çok GPU’ya böleriz. Fakat ikinci GPU yalnızca ek kapasite getirmez; senkronizasyon, mesaj gecikmesi ve fiziksel topoloji de getirir.

**Bu yazının temel fikri:** Dağıtık eğitim performansı, modeli matematiksel olarak bölmek ile oluşan tensör trafiğini doğru bağlantılara yerleştirmek arasındaki uyuma bağlıdır.

> **Önceki yazıdan devraldığımız fikir:** [İkinci yazıda](post.html?slug=gpu-is-yukleri-ve-uretim-cikarimi) training, prefill ve decode’un farklı kaynak profillerini gördük. Şimdi tek GPU’nun sınırını aşınca hangi verinin NVLink, PCIe ve scale-out ağ üzerinden geçtiğini izleyeceğiz.

> **Bu yazının sonunda**
>
> - Ring AllReduce veri hacmini ve gecikme adımlarını analitik olarak hesaplayabileceksiniz.
> - TP, PP, DP/FSDP, CP ve EP arasındaki farkları, tensör dönüşümlerini ve bellek kazanımlarını açıklayabileceksiniz.
> - Paralellik gruplarını fiziksel topolojiye (NVLink, Leaf, Spine) doğru hiyerarşiyle yerleştirebileceksiniz.
> - Bir training step’inde compute, kolektif iletişim, dataloader ve checkpoint akışını kritik yol analiziyle izleyebileceksiniz.
> - Dağıtık eğitim mimarisinde scaling verimliliğini, straggler etkisini ve fabric darboğazlarını sayılarla savunabileceksiniz.

**Bu yazıda**

- [1. GPU’lar hangi fiziksel yollarla konuşur?](#1-gpular-hangi-fiziksel-yollarla-konuşur)
- [2. Kolektif iletişim ne kadar veri taşır?](#2-kolektif-iletişim-ne-kadar-veri-taşır)
- [3. Model hangi eksenlerden bölünür?](#3-model-hangi-eksenlerden-bölünür)
- [4. Paralellik topolojiye nasıl yerleştirilir?](#4-paralellik-topolojiye-nasıl-yerleştirilir)
- [5. Bir training step’inde veri nereye gider?](#5-bir-training-stepinde-veri-nereye-gider)
- [6. Scheduler, paylaşım ve doğrulama](#6-scheduler-paylaşım-ve-doğrulama)
- [7. 16 GPU’luk tasarım örneği](#7-16-gpuluk-tasarım-örneği)
- [8. Sistem tasarımı için yedi temel soru](#8-sistem-tasarımı-için-yedi-temel-soru)
- [Okuyucu artık hangi sorulara yanıt verebilir?](#okuyucu-artık-hangi-sorulara-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. GPU’lar hangi fiziksel yollarla konuşur?

Dağıtık bir yapay zekâ kümesinde soyut bir “GPU sayısı” yoktur; GPU’ların birbirine hangi bant genişliği, gecikme ve anahtarlama katmanlarıyla bağlandığını gösteren fiziksel bir topoloji grafiği vardır. Hesaplama ne kadar hızlı olursa olsun, tensörlerin aktığı boru hattı daraldığında veya geciktiğinde Tensor Core’lar boş döngülerde kilitlenir.

| Yol | Kapsam | Çift Yönlü Bant Genişliği | Tipik Gecikme ($\alpha$) | En Uygun Trafik |
|---|---|---|---|---|
| **HBM3 / HBM3e** | Tek GPU içi | $3.35 - 8.0 \text{ TB/s}$ | $\approx 30 - 50 \text{ ns}$ | Ağırlık, aktivasyon, yerel gradient matrisleri |
| **NVLink 4 / NVSwitch** | Düğüm içi (8 GPU) | $900 \text{ GB/s}$ / GPU | $\approx 0.5 - 0.8\ \mu\text{s}$ | Tensor Parallelism (TP), yüksek frekanslı all-reduce |
| **PCIe Gen5** | GPU ↔ Host / NIC | $128 \text{ GB/s}$ ($x16$) | $\approx 1.0 - 1.5\ \mu\text{s}$ | Dataloader token akışı, CPU offload, yerel NVMe I/O |
| **InfiniBand (NDR 400G)** | Düğümler arası fabric | $100 \text{ GB/s}$ ($800\text{Gb/s}$ çift yön) | $\approx 1.2 - 2.0\ \mu\text{s}$ | Data Parallelism (DP/FSDP), Pipeline Parallelism (PP) |
| **RoCE v2 (400GbE)** | Düğümler arası Ethernet | $100 \text{ GB/s}$ per port | $\approx 2.5 - 5.0\ \mu\text{s}$ | Scale-out RDMA, lossy fabric yönetimi |

### 1.1. GPU–NIC yakınlığı

Bir DGX H100 sunucusunun içine baktığımızda iki adet x86 CPU, sekiz adet H100 SXM5 GPU, dört adet NVSwitch 3 yongası ve sekiz adet OSFP yuvasında çalışan ConnectX-7 NIC (Ağ Arayüz Kartı) görürüz. Bu bileşenler rastgele birbirine bağlanmamıştır; anakart üzerinde katı bir Non-Uniform Memory Access (NUMA) ve PCIe ağaç simetrisi vardır.

GPU ile NIC arasındaki fiziksel yolun optimizasyonu için dört adımlı zihinsel model:

1. **NUMA Düğümü Ayrımı:** Sunucudaki GPU 0–3 ve NIC 0–3 işlemci NUMA Düğümü 0'a; GPU 4–7 ve NIC 4–7 ise NUMA Düğümü 1'e bağlıdır. İki CPU soketi birbirine UPI (Ultra Path Interconnect, $\approx 64 \text{ GB/s}$) linkleriyle bağlıdır.
2. **GPUDirect RDMA (GDR) Yolu:** İdeal bir dağıtık işlemde, GPU 0 bir tensörü kümedeki başka bir düğüme göndereceği zaman veri doğrudan yerel PCIe switch üzerinden NIC 0'a akar (`GPU HBM -> PCIe Switch -> NIC -> Ağ`). CPU host belleği ve işletim sistemi kernel'ı tamamen atlanır (zero-copy RDMA).
3. **NUMA Çapraz Geçiş Tuzağı (UPI Bottleneck):** Eğer yazılım yapılandırma hatasıyla GPU 0'ın trafiği NIC 4 üzerinden yönlendirilirse, veri PCIe Switch 0 $\to$ CPU 0 $\to$ UPI link $\to$ CPU 1 $\to$ PCIe Switch 1 $\to$ NIC 4 rotasını izler. UPI köprüsü darboğaz oluşturur; gecikme $1.2\ \mu\text{s}$'den $3.8\ \mu\text{s}$'ye fırlar ve bant genişliği yarı yarıya düşer.
4. **NVSwitch ve PCIe Ayrışması:** Sekiz GPU, NVSwitch yongaları sayesinde tam çift yönlü $900 \text{ GB/s}$ hızında tam bağlı (all-to-all) tek bir bellek alanı gibi konuşur. Ancak düğüm dışına çıkacak tüm paketler PCIe Gen5 veri yollarından geçerek NIC'lere ulaşmak zorundadır.



<!-- restored-svg:tr_full:20 -->
<svg viewBox="0 0 720 350" role="img" aria-label="DGX H100&#x27;ün mantıksal topolojisi. Üstte dört NVSwitch çipi vardır; 80&#x27;er GB&#x27;lık sekiz GPU&#x27;nun her biri dördüne 25 GB/s&#x27;lik 18 NVLink ile, GPU başına yön başına 450 GB/s bağlanır. Altta PCIe Gen5 ağacı her GPU&#x27;yu kendi 400 Gb/s (yön başına 50 GB/s) ConnectX-7 NIC&#x27;ine ve oradan compute ağına bağlar. CPU 0, GPU ve NIC 0–3&#x27;e NUMA 0 olarak, CPU 1 ise 4–7&#x27;ye NUMA 1 olarak hizmet eder; CPU&#x27;lar UPI ile bağlıdır. Sunucuda ayrıca depolama ağı için iki çift portlu ConnectX-7, sekiz 3,84 TB NVMe, 2 TB RAM bulunur ve en fazla 10,2 kW çeker." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dx-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dx-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dx-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">DGX H100: mantıksal topoloji</text>
<rect x="96" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="158.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 0</text>
<rect x="229" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="291.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 1</text>
<rect x="362" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="424.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 2</text>
<rect x="495" y="40" width="125" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="557.5" y="59.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px">NVSwitch 3</text>
<line x1="127" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="127" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="127" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="127" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="193" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="259" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="325" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="391" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="457" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="523" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="158.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="291.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="424.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<line x1="589" y1="100" x2="557.5" y2="70" style="stroke:var(--c-accent);stroke-width:0.8;stroke-opacity:.35"/>
<text x="704" y="24" text-anchor="end" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">her GPU: 18 NVLink × 25 GB/s = 450 GB/s/yön</text>
<rect x="96" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="127.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<text x="127.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="162" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="193.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<text x="193.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="228" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="259.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<text x="259.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="294" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="325.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<text x="325.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="360" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="391.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 4</text>
<text x="391.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="426" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="457.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 5</text>
<text x="457.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="492" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="523.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 6</text>
<text x="523.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="558" y="100" width="62" height="44" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="589.0" y="118" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 7</text>
<text x="589.0" y="134" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">80 GB</text>
<rect x="16" y="154" width="70" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="51" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 0</text>
<text x="51" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">56 çekirdek</text>
<rect x="634" y="154" width="70" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="669" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">CPU 1</text>
<text x="669" y="191" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">56 çekirdek</text>
<line x1="86" y1="180" x2="325" y2="180" style="stroke:var(--c-warn);stroke-width:2"/>
<line x1="391" y1="180" x2="634" y2="180" style="stroke:var(--c-warn);stroke-width:2"/>
<line x1="127" y1="144" x2="127" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="127" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="193" y1="144" x2="193" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="193" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="259" y1="144" x2="259" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="259" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="325" y1="144" x2="325" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="325" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="391" y1="144" x2="391" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="391" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="457" y1="144" x2="457" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="457" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="523" y1="144" x2="523" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="523" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<line x1="589" y1="144" x2="589" y2="212" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<circle cx="589" cy="180" r="3.5" style="fill:var(--c-warn);fill-opacity:.9"/>
<rect x="100" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="127.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 0</text>
<line x1="127.0" y1="242" x2="127.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="166" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="193.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 1</text>
<line x1="193.0" y1="242" x2="193.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="232" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="259.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 2</text>
<line x1="259.0" y1="242" x2="259.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="298" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="325.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 3</text>
<line x1="325.0" y1="242" x2="325.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="391.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 4</text>
<line x1="391.0" y1="242" x2="391.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="457.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 5</text>
<line x1="457.0" y1="242" x2="457.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="496" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="523.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 6</text>
<line x1="523.0" y1="242" x2="523.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="562" y="212" width="54" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="589.0" y="230.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">NIC 7</text>
<line x1="589.0" y1="242" x2="589.0" y2="262" marker-end="url(#dx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="358.0" y="278" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">compute ağı: 8 × ConnectX-7 · her biri 400 Gb/s (50 GB/s/yön), her GPU&#x27;ya bir NIC</text>
<rect x="92" y="148" width="268" height="100" rx="8" style="fill:none;stroke:var(--c-warn);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="356" y="148" width="268" height="100" rx="8" style="fill:none;stroke:var(--c-warn);stroke-width:1;stroke-dasharray:4 3"/>
<text x="16" y="222" text-anchor="start" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PCIe Gen5</text>
<text x="16" y="238" text-anchor="start" style="fill:var(--c-warn);font-size:11px;font-weight:600">NUMA 0</text>
<text x="16" y="254" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">UPI ⇄ CPU 1</text>
<text x="704" y="222" text-anchor="end" style="fill:var(--c-warn);font-size:10.5px;font-weight:600">PCIe Gen5</text>
<text x="704" y="238" text-anchor="end" style="fill:var(--c-warn);font-size:11px;font-weight:600">NUMA 1</text>
<rect x="16" y="292" width="688" height="46" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="311" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Sunucuda ayrıca: 2 × çift portlu ConnectX-7 → depolama ağı · 8 × 3,84 TB NVMe · 2 TB RAM · en fazla 10,2 kW</text>
<text x="28" y="328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GPU, diğer GPU&#x27;larla NVSwitch üzerinden, başka sunucularla kendi NIC&#x27;i (GPUDirect RDMA), host ile PCIe üzerinden konuşur.</text>
</svg>

### 1.2. Rail topolojisi

Binlerce GPU'luk bir süper bilgisayarda (örneğin DGX SuperPOD) düğümler rastgele birbirine bağlanırsa, anahtarlarda (switch) devasa paket kuyrukları ve tıkanıklıklar (congestion) oluşur. Bu sorunu çözmek için telekomünikasyondan miras kalan çok katmanlı, tıkanmasız (non-blocking) **Fat-Tree Clos** topolojisi kullanılır. Ancak yapay zekâ iş yüklerinde bu mimari bir adım öteye taşınır: **Rail-Optimized Ağ**.

Rail mimarisinin çalışma prensibi:

1. **Ray (Rail) Ayrımı:** Her DGX sunucusunda 8 adet NIC bulunur. Bu NIC'ler 0'dan 7'ye numaralandırılır. Kümedeki tüm sunucuların 0 numaralı NIC'leri bağımsız bir Leaf switch grubuna (Rail 0), 1 numaralı NIC'leri Rail 1'e, ..., 7 numaralı NIC'leri ise Rail 7'ye bağlanır.
2. **Scalable Unit (SU):** Standart bir DGX H100 SuperPOD SU'su 32 sunucudan (256 GPU) oluşur. Bir SU içinde her ray için bir adet 64 portlu Quantum-2 InfiniBand switch bulunur (toplam 8 leaf switch). 32 sunucunun $r$. NIC'i doğrudan bu leaf switch'e bağlanır (32 downlinks). Kalan 32 port ise yukarıdaki Spine katmanına bağlanır (32 uplinks).
3. **Trafik Yerelleştirmesi:** Data Parallelism veya Tensor Parallelism çalıştıran bir modelde, GPU $r$ daima diğer sunuculardaki GPU $r$ ile konuşur. Rail mimarisi sayesinde bu iletişim tek bir leaf switch içinde yerel kalır; paketler omurga (spine) katmanına tırmanmadan tek atlamada (single-hop) hedefe ulaşır.
4. **Non-blocking Spine Bisection:** Farklı SU'lar arasındaki iletişim için leaf switch'lerin üst portları 16 adet Spine switch'e 1:1 oversubscription (tam simetrik bant genişliği) oranıyla bağlanır. Böylece 1.024 GPU'luk bir kümede herhangi bir GPU çifti arasındaki bisection bant genişliği eksiksiz $400 \text{ Gb/s}$ ($50 \text{ GB/s}$) kalır.



<!-- restored-svg:tr_full:22 -->
<svg viewBox="0 0 720 340" role="img" aria-label="DGX H100 SuperPOD compute ağı. Üstte 16 spine switch vardır. Altta dört SU bulunur; her birinde rail başına bir tane olmak üzere 0–7 numaralı sekiz leaf switch ve 32 DGX H100 vardır. Bir SU&#x27;daki leaf r, 32 sunucunun NIC r&#x27;sini bağlar; böylece SU içinde aynı rail trafiği tek leaf atlamasıyla gider. Her leaf her spine&#x27;a iki 400 Gb/s kabloyla bağlıdır, 32 port yukarı ve 32 port aşağı; ağ tasarımda 1:1&#x27;dir. Vurgulu yol, farklı SU&#x27;lar arasındaki trafiğin leaf, spine, leaf yolunu izlediğini gösterir. Dört SU 128 yuva verir; biri UFM&#x27;e ayrılır, geriye 127 DGX yani 1.016 GPU kalır. Depolama ayrı bir ağda çalışır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ft-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ft-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="ft-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">DGX H100 SuperPOD compute ağı: 32 leaf + 16 spine, rail-optimize</text>
<rect x="27" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="45.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S0</text>
<rect x="69" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="87.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S1</text>
<rect x="111" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="129.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S2</text>
<rect x="153" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="171.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S3</text>
<rect x="195" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="213.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S4</text>
<rect x="237" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="255.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S5</text>
<rect x="279" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="297.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S6</text>
<rect x="321" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="339.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S7</text>
<rect x="363" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="381.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S8</text>
<rect x="405" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="423.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S9</text>
<rect x="447" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="465.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S10</text>
<rect x="489" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="507.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S11</text>
<rect x="531" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="549.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S12</text>
<rect x="573" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="591.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S13</text>
<rect x="615" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="633.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S14</text>
<rect x="657" y="44" width="36" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="675.0" y="60.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">S15</text>
<rect x="17" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="99" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 1</text>
<rect x="23" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="31.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="42.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="51.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="62" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="70.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="81.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="90.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="101" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="109.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="120.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="129.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="140" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="148.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="159.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="168.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="23" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="99.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="31.5" y1="200" x2="29" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="51" y1="200" x2="49" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="70.5" y1="200" x2="69" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="90" y1="200" x2="89" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="109.5" y1="200" x2="109" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="129" y1="200" x2="129" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="148.5" y1="200" x2="149" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="168" y1="200" x2="169" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<rect x="191" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="273" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 2</text>
<rect x="197" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="205.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="216.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="225.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="236" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="244.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="255.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="264.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="275" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="283.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="294.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="303.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="314" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="322.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="333.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="342.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="197" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="273.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="205.5" y1="200" x2="203" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="225" y1="200" x2="223" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="244.5" y1="200" x2="243" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="264" y1="200" x2="263" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="283.5" y1="200" x2="283" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="303" y1="200" x2="303" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="322.5" y1="200" x2="323" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="342" y1="200" x2="343" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<rect x="365" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="447" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 3</text>
<rect x="371" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="379.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="390.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="399.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="410" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="418.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="429.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="449" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="457.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="468.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="477.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="488" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="496.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="507.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="516.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="371" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="447.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="379.5" y1="200" x2="377" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="399" y1="200" x2="397" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="418.5" y1="200" x2="417" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="438" y1="200" x2="437" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="457.5" y1="200" x2="457" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="477" y1="200" x2="477" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="496.5" y1="200" x2="497" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="516" y1="200" x2="517" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<rect x="539" y="150" width="164" height="120" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="621" y="168" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">SU 4</text>
<rect x="545" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="553.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">0</text>
<rect x="564.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="573.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">1</text>
<rect x="584" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="592.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">2</text>
<rect x="603.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="612.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">3</text>
<rect x="623" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="631.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">4</text>
<rect x="642.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="651.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">5</text>
<rect x="662" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="670.5" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">6</text>
<rect x="681.5" y="176" width="17" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="690.0" y="191.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">7</text>
<rect x="545" y="232" width="152" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.14;stroke:var(--c-success);stroke-width:1.2"/>
<text x="621.0" y="250.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">32 × DGX H100</text>
<line x1="553.5" y1="200" x2="551" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="573" y1="200" x2="571" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="592.5" y1="200" x2="591" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="612" y1="200" x2="611" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="631.5" y1="200" x2="631" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="651" y1="200" x2="651" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="670.5" y1="200" x2="671" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="690" y1="200" x2="691" y2="232" style="stroke:var(--c-success);stroke-width:0.8;stroke-opacity:.6"/>
<line x1="31.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="31.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="51" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="70.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="90" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="109.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="129" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="148.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="168" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="205.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="225" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="244.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="264" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="283.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="303" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="322.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="342" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="379.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="399" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="418.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="438" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="457.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="477" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="496.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="516" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="553.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="573" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="592.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="612" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="631.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="651" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="670.5" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="45" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="87" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="129" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="171" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="213" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="255" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="297" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="339" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="381" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="423" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="465" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="507" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="549" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="591" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="633" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<line x1="690" y1="176" x2="675" y2="70" style="stroke:var(--c-text-mute);stroke-width:0.5;stroke-opacity:.18"/>
<path d="M31.5 176 L255 70 L379.5 176" style="fill:none;stroke:var(--c-accent-2);stroke-width:2.2"/>
<text x="704" y="104" text-anchor="end" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">farklı SU → leaf → spine → leaf</text>
<text x="16" y="292" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">Bir SU&#x27;daki leaf r, 32 DGX&#x27;in NIC r&#x27;sini bağlar (rail r) → SU içinde aynı rail = tek leaf atlaması.</text>
<text x="16" y="310" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Her leaf: 32 port aşağı + 32 port yukarı (16 spine&#x27;ın her birine 2 × 400 Gb/s) → tasarımda 1:1, engellemesiz.</text>
<text x="16" y="328" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">4 SU × 32 yuva = 128; bir yuva UFM&#x27;e ayrılır → 127 DGX = 1.016 GPU. Depolama ağı ayrı bir ağdır.</text>
</svg>

<!-- restored-svg:tr_full:23 -->
<svg viewBox="0 0 720 342" role="img" aria-label="Rail-optimize kablolama. A, B ve C sunucularının her birinde GPU başına bir tane olmak üzere 0–7 numaralı sekiz NIC vardır. Her sunucunun NIC i&#x27;si leaf switch i&#x27;ye bağlanır: leaf 0 rail 0&#x27;dır, leaf 1 rail 1&#x27;dir ve böyle devam eder; üstteki spine katmanı leaf&#x27;leri bağlar. Vurgulu: A sunucusunun GPU 0&#x27;ı, B&#x27;nin GPU 0&#x27;ına NIC 0 ve leaf 0 üzerinden aynı SU içinde tek leaf atlamasıyla ulaşır. A&#x27;nın GPU 0&#x27;ından C&#x27;nin GPU 1&#x27;ine gitmek rail değiştirmektir: ya spine&#x27;a çıkılır ya da önce NVLink ile A&#x27;nın GPU 1&#x27;ine geçilip rail 1 kullanılır; NCCL buna PXN der." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rl-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="rl-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Rail optimizasyonu: her sunucunun NIC i&#x27;si leaf i&#x27;de buluşur</text>
<rect x="96" y="40" width="600" height="24" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="396.0" y="56.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">spine katmanı (rail&#x27;ler / SU&#x27;lar arası trafik)</text>
<rect x="100" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="132.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 0</text>
<line x1="132" y1="86" x2="132" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="176" y="86" width="64" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="208.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 1</text>
<line x1="208" y1="86" x2="208" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="252" y="86" width="64" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="284.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 2</text>
<line x1="284" y1="86" x2="284" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="328" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="360.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 3</text>
<line x1="360" y1="86" x2="360" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="404" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="436.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 4</text>
<line x1="436" y1="86" x2="436" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="480" y="86" width="64" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="512.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 5</text>
<line x1="512" y1="86" x2="512" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="556" y="86" width="64" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="588.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 6</text>
<line x1="588" y1="86" x2="588" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="632" y="86" width="64" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="664.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">leaf · rail 7</text>
<line x1="664" y1="86" x2="664" y2="64" style="stroke:var(--c-text-mute);stroke-width:0.8;stroke-opacity:.5"/>
<rect x="16" y="150" width="688" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="24" y="172" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">sunucu A</text>
<rect x="108" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="116.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">0</text>
<line x1="116" y1="156" x2="116" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="184" y="156" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="192.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">1</text>
<line x1="192" y1="156" x2="192" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="260" y="156" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="268.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">2</text>
<line x1="268" y1="156" x2="268" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="336" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="344.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">3</text>
<line x1="344" y1="156" x2="344" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="412" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="420.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">4</text>
<line x1="420" y1="156" x2="420" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="488" y="156" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="496.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">5</text>
<line x1="496" y1="156" x2="496" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="564" y="156" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="572.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">6</text>
<line x1="572" y1="156" x2="572" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="640" y="156" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="648.0" y="171.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">7</text>
<line x1="648" y1="156" x2="648" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="16" y="196" width="688" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="24" y="218" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">sunucu B</text>
<rect x="126" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="134.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">0</text>
<line x1="134" y1="202" x2="134" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="202" y="202" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="210.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">1</text>
<line x1="210" y1="202" x2="210" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="278" y="202" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="286.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">2</text>
<line x1="286" y1="202" x2="286" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="354" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="362.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">3</text>
<line x1="362" y1="202" x2="362" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="430" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">4</text>
<line x1="438" y1="202" x2="438" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="506" y="202" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="514.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">5</text>
<line x1="514" y1="202" x2="514" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="582" y="202" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="590.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">6</text>
<line x1="590" y1="202" x2="590" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="658" y="202" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="666.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">7</text>
<line x1="666" y1="202" x2="666" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="16" y="242" width="688" height="36" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="24" y="264" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">sunucu C</text>
<rect x="144" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="152.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">0</text>
<line x1="152" y1="248" x2="152" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="220" y="248" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="228.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">1</text>
<line x1="228" y1="248" x2="228" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="296" y="248" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="304.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">2</text>
<line x1="304" y1="248" x2="304" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="372" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="380.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">3</text>
<line x1="380" y1="248" x2="380" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<rect x="448" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="456.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">4</text>
<line x1="456" y1="248" x2="456" y2="112" style="stroke:var(--c-accent);stroke-width:1;stroke-opacity:.55"/>
<rect x="524" y="248" width="16" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="532.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">5</text>
<line x1="532" y1="248" x2="532" y2="112" style="stroke:var(--c-success);stroke-width:1;stroke-opacity:.55"/>
<rect x="600" y="248" width="16" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="608.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">6</text>
<line x1="608" y1="248" x2="608" y2="112" style="stroke:var(--c-warn);stroke-width:1;stroke-opacity:.55"/>
<rect x="676" y="248" width="16" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="684.0" y="263.6" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-family:var(--font-mono)">7</text>
<line x1="684" y1="248" x2="684" y2="112" style="stroke:var(--c-accent-2);stroke-width:1;stroke-opacity:.55"/>
<path d="M116 150 V112" style="fill:none;stroke:var(--c-accent);stroke-width:2.6"/>
<path d="M134 196 V112" style="fill:none;stroke:var(--c-accent);stroke-width:2.6"/>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">A·GPU0 → NIC0 → leaf 0 → NIC0 → B·GPU0: aynı rail, tek leaf atlaması (aynı SU içinde).</text>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">A·GPU0 → C·GPU1: farklı rail → spine&#x27;a çıkar ya da önce NVLink ile A·GPU1&#x27;e geçip rail 1&#x27;i kullanır (NCCL PXN).</text>
</svg>

<!-- restored-svg:tr_full:21 -->
<svg viewBox="0 0 720 286" role="img" aria-label="Yön başına bant genişliği, log ölçek. HBM3 tepe değeri yaklaşık 3.350 GB/s; GPU&#x27;nun kendi ağırlık ve aktivasyonlarını okumasıdır. NVLink, NVLink alanında GPU başına yön başına 450 GB/s verir; HBM&#x27;den 7,4 kat azdır. PCIe 5.0 ×16 teorik olarak yön başına yaklaşık 63 GB/s&#x27;dir, host ve PCIe aygıtları içindir; 53 kat azdır. NDR 400 Gb/s port, sunucular arası trafik için yön başına 50 GB/s hat hızı verir; HBM&#x27;den 67 kat azdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="bl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bant genişliği uçurumu: HBM&#x27;den her uzaklaşmada yol birkaç kat daralır</text>
<text x="16" y="62" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">HBM3</text>
<text x="16" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">tepe</text>
<rect x="150" y="48" width="383.579" height="24" rx="3" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="541.5790576486218" y="65" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">3.350 GB/s</text>
<text x="150" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU&#x27;nun kendi ağırlık/aktivasyon okuması</text>
<text x="16" y="116" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NVLink</text>
<text x="16" y="132" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">yön / GPU</text>
<rect x="150" y="102" width="251.139" height="24" rx="3" style="fill:var(--c-success);fill-opacity:.6"/>
<text x="409.1391862669635" y="119" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">450 GB/s</text>
<text x="150" y="142" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">NVLink alanında GPU ↔ GPU</text>
<text x="704" y="119" text-anchor="end" style="fill:var(--c-success);font-size:12px;font-weight:700">÷7,4 (HBM&#x27;e göre)</text>
<text x="16" y="170" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">PCIe 5.0 ×16</text>
<text x="16" y="186" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">yön, teorik</text>
<rect x="150" y="156" width="121.428" height="24" rx="3" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="279.42766490529937" y="173" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">63 GB/s</text>
<text x="150" y="196" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">host ve PCIe aygıtları</text>
<text x="704" y="173" text-anchor="end" style="fill:var(--c-warn);font-size:12px;font-weight:700">÷53 (HBM&#x27;e göre)</text>
<text x="16" y="224" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">NDR 400 Gb/s</text>
<text x="16" y="240" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">yön, hat hızı</text>
<rect x="150" y="210" width="106.18" height="24" rx="3" style="fill:var(--c-danger);fill-opacity:.6"/>
<text x="264.18039523127993" y="227" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">50 GB/s</text>
<text x="150" y="250" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sunucu ↔ sunucu, GPU–NIC trafiği</text>
<text x="704" y="227" text-anchor="end" style="fill:var(--c-danger);font-size:12px;font-weight:700">÷67 (HBM&#x27;e göre)</text>
<line x1="150" y1="258" x2="560" y2="258" style="stroke:var(--c-text-mute);stroke-width:1"/>
<line x1="150" y1="258" x2="150" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="150.0" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">10</text>
<line x1="301.91" y1="258" x2="301.91" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="301.9098023843601" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">100</text>
<line x1="453.82" y1="258" x2="453.82" y2="263" style="stroke:var(--c-text-mute);stroke-width:1"/>
<text x="453.8196047687201" y="276" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">1.000</text>
<text x="560" y="276" text-anchor="end" style="fill:var(--c-text-mute);font-size:10.5px">GB/s (log)</text>
</svg>

### 1.3. Latency ve bandwidth hangi mesajda önemlidir?

Ağ üzerinden tensör taşırken harcanan zamanı modellemek için klasik paralel programlamanın temel taşı olan **Hockney ($\alpha$-$\beta$) Modeli** kullanılır. Bir mesajın aktarım süresi $T$, ağ gecikmesi $\alpha$, aktarılan veri boyutu $S$ ve etkin bant genişliği $\beta$ cinsinden şöyle ifade edilir:

$$
T(S) = \alpha + \frac{S}{\beta}
$$

Burada kritik bir dönüm noktası vardır: **Karakteristik Mesaj Boyutu ($S_{1/2}$)**. Bu, aktarım süresinin yarısının gecikmeye ($\alpha$), yarısının ise bant genişliğine ($S/\beta$) harcandığı mesaj boyutudur:

$$
S_{1/2} = \alpha \times \beta
$$

Fiziksel bağlantı türlerine göre bu değerlerin analizi:

- **NVLink 4:** $\alpha \approx 0.5\ \mu\text{s}$, $\beta = 450 \text{ GB/s}$ (tek yön). $S_{1/2} \approx 225 \text{ KB}$.
- **InfiniBand NDR (400 Gb/s):** $\alpha \approx 1.5\ \mu\text{s}$, $\beta = 50 \text{ GB/s}$. $S_{1/2} \approx 75 \text{ KB}$.
- **Standart Ethernet (TCP/IP 100 Gb/s):** $\alpha \approx 25\ \mu\text{s}$, $\beta = 12.5 \text{ GB/s}$. $S_{1/2} \approx 312 \text{ KB}$.

Bu matematik şu pratik kuralı doğurur:

1. **Gecikme-Baskın Bölge ($S \ll S_{1/2}$):** Pipeline Parallelism aktivasyon sinyalleri, distributed barrier komutları, MoE yönlendirici (router) kararları ve küçük skaler senkronizasyonlar bu bölgeye düşer. Burada bağlantının $400 \text{ Gb/s}$ olması hiçbir işe yaramaz; süreyi belirleyen tek şey mikrosaniye mertebesindeki fiziksel atlama gecikmesidir ($\alpha$).
2. **Bant Genişliği-Baskın Bölge ($S \gg S_{1/2}$):** Büyük Data Parallelism gradient senkronizasyonu ($S \ge 100 \text{ MB}$), FSDP model ağırlığı AllGather operasyonları ve Megatron TP katman çıktıları bu bölgededir. Burada $\alpha$ ihmal edilebilir düzeyde kalır; toplam süreyi linkin doyum kapasitesi ($\beta$) belirler.

---

## 2. Kolektif iletişim ne kadar veri taşır?

Dağıtık derin öğrenmede GPU'lar tekil soket mesajlarıyla değil, tüm grubun aynı anda katıldığı senkron kolektif iletişim (collective communication) primitifleriyle konuşur. NVIDIA Collective Communications Library (NCCL), bu primitifleri donanım topolojisine göre optimize eden endüstri standardı kütüphanedir.

### 2.1. Dört temel kolektif

Tüm dağıtık eğitim kurgusu dört temel kolektif işlem üzerine inşa edilir:

1. **AllReduce:** Her rank $S$ boyutunda yerel bir tensör sunar. Belirlenen matematiksel işlem (genellikle toplama: SUM) tüm rank'lerin elemanları arasında uygulanır ve indirgenmiş sonuç tensörü ($S$) her bir rank'e eksiksiz kopyalanır.
   - *Kullanım:* Tensor Parallelism katman çıkışları, Data Parallelism gradient senkronizasyonu.
2. **ReduceScatter:** Her rank $S$ boyutunda tensör sunar. İndirgeme işlemi yapılır, ancak sonuç tüm GPU'lara dağıtılmaz; her biri $S/N$ boyutundaki eşit birer parçayı (shard) alır.
   - *Kullanım:* ZeRO-2 ve ZeRO-3 gradient birikimi, Sequence Parallelism.
3. **AllGather:** Her rank elindeki küçük parçayı ($S/N$) ortaya koyar. İletişim sonunda tüm parçalar sıralı olarak birleştirilir ve her rank tam tensöre ($S$) sahip olur. ReduceScatter'ın tam tersidir.
   - *Kullanım:* FSDP (ZeRO-3) forward ve backward geçişlerinde katman ağırlıklarını anlık olarak toplama.
4. **AllToAll:** Her rank, gruptaki diğer her bir rank için farklı ve bağımsız bir veri parçası hazırlar. $N \times N$ matris transpozesi gibidir; her rank diğer tüm rank'lere kişiselleştirilmiş bir dilim gönderir ve alır.
   - *Kullanım:* Mixture of Experts (MoE) token yönlendirmesi (dispatch ve combine) ve Context Parallelism ring geçişleri.



<!-- restored-svg:tr_full:25 -->
<svg viewBox="0 0 720 428" role="img" aria-label="Dört GPU&#x27;da dört panel. AllReduce (toplam): GPU&#x27;lar 10, 20, 30 ve 40 ile başlar, hepsi 100 ile biter. AllGather: her GPU bir değerle başlar, hepsi dört parçanın tamamıyla biter. ReduceScatter (toplam): her GPU dört dilimle başlar; GPU i, i. dilimin tüm GPU&#x27;lardaki toplamıyla biter. AllToAll: her GPU dört parçayla başlar; GPU i&#x27;nin j. parçası GPU j&#x27;ye gider, matrisin transpozu gibi. AllReduce, ReduceScatter ve ardından AllGather&#x27;a eşittir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="cl-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Dört GPU&#x27;da dört kolektif (önce → sonra)</text>
<rect x="16" y="40" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllReduce(sum)</text>
<text x="110" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="178" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="246" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="314" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="28" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="28" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="208.0" y="140" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="86" y="86" width="48" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="110.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">10</text>
<rect x="86" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="110.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<rect x="154" y="86" width="48" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="178.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">20</text>
<rect x="154" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="178.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<rect x="222" y="86" width="48" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="246.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">30</text>
<rect x="222" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="246.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<rect x="290" y="86" width="48" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="314.0" y="103.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">40</text>
<rect x="290" y="152" width="48" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="314.0" y="169.32" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">100</text>
<text x="28" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">herkes toplamı bilir</text>
<rect x="364" y="40" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllGather</text>
<text x="458" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="526" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="594" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="662" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="376" y="104" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="376" y="168" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="556.0" y="140" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="446" y="86" width="24" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="458.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">10</text>
<rect x="432" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="445" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="458" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="471" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="514" y="86" width="24" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="526.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">20</text>
<rect x="500" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="513" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="526" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="539" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="582" y="86" width="24" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">30</text>
<rect x="568" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="581" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="594" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="607" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="650" y="86" width="24" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="662.0" y="102.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">40</text>
<rect x="636" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.35;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="649" y="152" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="662" y="152" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="675" y="152" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.35;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="376" y="200" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">her parça herkeste olur</text>
<rect x="16" y="224" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="244" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">ReduceScatter(sum)</text>
<text x="110" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="178" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="246" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="314" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="28" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="28" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="208.0" y="324" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="84" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="90.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="97" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="103.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="110" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="116.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="123" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="129.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="92" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="110.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ0</text>
<rect x="152" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="158.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="165" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="171.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="178" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="184.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="191" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="197.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="160" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="178.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ1</text>
<rect x="220" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="226.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="233" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="239.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="246" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="252.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="259" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="265.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="228" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="246.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ2</text>
<rect x="288" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="294.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="301" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="307.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="314" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="320.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="327" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="333.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="296" y="336" width="36" height="26" rx="4" style="fill:var(--c-text);fill-opacity:.08;stroke:var(--c-text);stroke-width:1.2"/>
<text x="314.0" y="352.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Σ3</text>
<text x="28" y="384" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU i, i. dilimin toplamını tutar</text>
<rect x="364" y="224" width="340" height="172" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="244" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700;font-family:var(--font-mono)">AllToAll</text>
<text x="458" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU0</text>
<text x="526" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU1</text>
<text x="594" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU2</text>
<text x="662" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px;font-family:var(--font-mono)">GPU3</text>
<text x="376" y="288" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">önce</text>
<text x="376" y="352" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sonra</text>
<text x="556.0" y="324" text-anchor="middle" style="fill:var(--c-text-mute);font-size:14px">↓</text>
<rect x="432" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="445" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="451.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="458" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="464.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="471" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="477.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="432" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="438.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="445" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="451.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="458" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="464.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="471" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="477.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="500" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="506.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="513" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="519.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="526" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="532.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="539" y="270" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="545.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="500" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="506.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="513" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="519.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="526" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="532.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="539" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="545.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="568" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="574.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="581" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="587.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="594" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="600.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="607" y="270" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="613.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="568" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="574.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="581" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="587.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="594" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="600.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="607" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="613.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="636" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="642.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">0</text>
<rect x="649" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="655.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">1</text>
<rect x="662" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="668.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">2</text>
<rect x="675" y="270" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="681.0" y="286.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="636" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="642.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="649" y="336" width="12" height="26" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="655.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="662" y="336" width="12" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="668.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<rect x="675" y="336" width="12" height="26" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="681.0" y="352.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">3</text>
<text x="376" y="384" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">GPU i&#x27;nin j. parçası GPU j&#x27;ye gider</text>
<text x="16" y="416" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Renk = kaynak GPU · rakam = dilim no · Σi = i. dilimin tüm GPU&#x27;lardaki toplamı. AllReduce = ReduceScatter + AllGather.</text>
</svg>

### 2.2. Ring AllReduce hesabı

AllReduce işlemini safça yürütmek (örneğin her GPU'nun diğer tüm GPU'lara doğrudan veri göndermesi veya merkezi bir parametre sunucusu kullanması) $O(N)$ iletişim karmaşıklığı yaratır ve GPU sayısı arttığında ağı kilitler. Baidu ve sonrasında NCCL tarafından benimsenen **Ring AllReduce Algoritması**, taşınan veri hacmini düğüm sayısından neredeyse bağımsız hale getirir.

Ring AllReduce'un 4 adımlı mantıksal işleyişi:

1. **Halka Kurulumu ve Dilimleme:** $N$ adet GPU mantıksal bir halkaya dizilir ($0 \to 1 \to 2 \to \dots \to N-1 \to 0$). Taşınacak $S$ baytlık tensör, boyutu $S/N$ olan $N$ eşit dilime (chunk) ayrılır.
2. **Reduce-Scatter Aşaması ($N-1$ Adım):**
   - Her adımda her GPU, elindeki bir dilimi saat yönündeki komşusuna gönderirken, ters yöndeki komşusundan gelen dilimi alır ve elindeki ilgili indeksle toplar.
   - $N-1$ transfer adımının ardından, her bir GPU tensörün tam toplanmış bir dilimine (1 chunk, $S/N$ bayt) sahip olur.
   - Bu fazda GPU başına aktarılan toplam veri:
     $$V_{\mathrm{RS}} = (N-1) \times \frac{S}{N} = \frac{N-1}{N} S$$
3. **All-Gather Aşaması ($N-1$ Adım):**
   - Şimdi amaç, her GPU'da hazır olan toplanmış nihai dilimleri halkanın tamamına yaymaktır.
   - Yine $N-1$ adım boyunca her GPU, elindeki toplanmış dilimi komşusuna kopyalar; komşu bunu doğrudan belleğine yazar ve sonraki adıma iletir.
   - Bu fazda GPU başına aktarılan veri:
     $$V_{\mathrm{AG}} = (N-1) \times \frac{S}{N} = \frac{N-1}{N} S$$
4. **Toplam Veri Hacmi ve Süre:**
   Her bir GPU'nun gönderdiği toplam veri hacmi iki fazın toplamıdır:
   $$V_{\mathrm{ring}} = V_{\mathrm{RS}} + V_{\mathrm{AG}} = 2 \left(\frac{N-1}{N}\right) S$$
   Hockney modeline göre toplam Ring AllReduce süresi:
   $$T_{\mathrm{ring}} = 2(N-1)\alpha + 2\left(\frac{N-1}{N}\right)\frac{S}{\beta}$$

**Sayısal Doğrulama:**
8 GPU'lu bir DGX H100 sisteminde ($N=8$), $S = 8\text{ GB}$'lık bir gradient tensörünü Ring AllReduce ile senkronize edelim:
- GPU başına gönderilen veri:
  $$V_{\mathrm{ring}} = 2 \times \frac{8-1}{8} \times 8\text{ GB} = 2 \times \frac{7}{8} \times 8 = 14\text{ GB}$$
- $N \to \infty$ limitinde $\frac{N-1}{N} \to 1$ olur ve transfer edilen veri kesinlikle $2S = 16\text{ GB}$ sınırını aşmaz! GPU sayısı 8'den 1.024'e çıksa bile GPU başına düşen veri yükü sabit kalır.
- Ancak gecikme terimi $2(N-1)\alpha$ doğrusal büyür. $N=1.024$ olduğunda 2.046 gecikme adımı oluşur; bu yüzden devasa kümelerde saf halka yerine hiyerarşik iki seviyeli halkalar (node-local ring + inter-node ring) veya ağaç algoritmaları tercih edilir.



<!-- restored-svg:tr_full:26 -->
<svg viewBox="0 0 720 342" role="img" aria-label="Halka biçiminde bağlı dört GPU ile ring AllReduce: GPU 0 → 1 → 2 → 3 → 0. Veri S/N boyutunda dört parçaya bölünür. İlk üç adımda (reduce-scatter) her GPU komşusuna bir parça gönderir, komşu kendi payını ekler; kısmi toplamlar halkada dolaşır ve sonunda her GPU bir bitmiş toplamı tutar. Sonraki üç adımda (all-gather) bitmiş toplamlar dolaştırılır ve her GPU hepsine sahip olur. Her GPU 2(N−1)/N · S byte gönderir; N = 8, S = 8 GB için GPU başına 14 GB eder. Süre kabaca 2(N−1) gecikme adımı artı bu hacmin bant genişliğine bölümüdür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="rg-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="rg-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="rg-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">N = 4&#x27;te ring AllReduce: 2(N−1) = 6 adım, adım başına S/N&#x27;lik bir parça</text>
<path d="M136 102 Q 190 90 178 144" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M178 176 Q 190 230 136 218" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M104 218 Q 50 230 62 176" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M62 144 Q 50 90 104 102" marker-end="url(#rg-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="120" cy="86" r="22" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.4"/>
<text x="120" y="90" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU0</text>
<circle cx="194" cy="160" r="22" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.4"/>
<text x="194" y="164" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU1</text>
<circle cx="120" cy="234" r="22" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.4"/>
<text x="120" y="238" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU2</text>
<circle cx="46" cy="160" r="22" style="fill:var(--c-accent-2);fill-opacity:.18;stroke:var(--c-accent-2);stroke-width:1.4"/>
<text x="46" y="164" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:700;font-family:var(--font-mono)">GPU3</text>
<text x="120" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">halka</text>
<text x="297.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 1</text>
<text x="367.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 2</text>
<text x="437.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 3</text>
<text x="507.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 4</text>
<text x="577.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 5</text>
<text x="647.0" y="62" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">adım 6</text>
<text x="254" y="93" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU0</text>
<rect x="266" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +0</text>
<rect x="336" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +3</text>
<rect x="406" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +2</text>
<rect x="476" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ1</text>
<rect x="546" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ0</text>
<rect x="616" y="78" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="95.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ3</text>
<text x="254" y="133" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU1</text>
<rect x="266" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +1</text>
<rect x="336" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +0</text>
<rect x="406" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +3</text>
<rect x="476" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ2</text>
<rect x="546" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ1</text>
<rect x="616" y="118" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="135.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ0</text>
<text x="254" y="173" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU2</text>
<rect x="266" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +2</text>
<rect x="336" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +1</text>
<rect x="406" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +0</text>
<rect x="476" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ3</text>
<rect x="546" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ2</text>
<rect x="616" y="158" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="175.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ1</text>
<text x="254" y="213" text-anchor="end" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">GPU3</text>
<rect x="266" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="297.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +3</text>
<rect x="336" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="367.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +2</text>
<rect x="406" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.14;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="437.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder +1</text>
<rect x="476" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="507.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ0</text>
<rect x="546" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="577.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ3</text>
<rect x="616" y="198" width="62" height="28" rx="4" style="fill:var(--c-accent-2);fill-opacity:.14;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="647.0" y="215.78" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px">gönder Σ2</text>
<line x1="266" y1="238" x2="468" y2="238" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="476" y1="238" x2="678" y2="238" style="stroke:var(--c-accent-2);stroke-width:2"/>
<text x="367.0" y="254" text-anchor="middle" style="fill:var(--c-accent);font-size:11px;font-weight:600">N−1 adım: kısmi toplamlar dolaşır</text>
<text x="577.0" y="254" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">N−1 adım: bitmiş toplamlar yayılır</text>
<rect x="16" y="278" width="688" height="52" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="298" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her GPU 2(N−1)/N · S byte gönderir. N = 8, S = 8 GB → 2 · 7/8 · 8 = GPU başına 14 GB (toplam trafik değil).</text>
<text x="28" y="318" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Süre ≈ 2(N−1)·L + 2(N−1)/N · S/W: gecikme terimi N ile büyür, bant genişliği terimi neredeyse sabit kalır.</text>
</svg>

### 2.3. Ring mi tree mi?

NCCL, mesaj boyutu ve küme büyüklüğüne göre otomatik olarak Ring ve Tree algoritmaları arasında geçiş yapar:

1. **Tree-Based AllReduce (Binomial Tree / Double Binary Tree):**
   - İndirgeme işlemi bir ikili ağaç üzerinden tepeye (root) toplanır ve tekrar yapraklara yayınlanır.
   - Gecikme adımı $2(N-1)$ yerine yalnızca $2 \log_2 N$ adımdır. Örneğin 1.024 GPU için halkada 2.046 adım gerekirken, ağaçta yalnızca $2 \times 10 = 20$ adım yeterlidir!
   - Ancak ağaç algoritmasında ara düğümlerdeki bant genişliği kullanımı halka kadar pürüzsüz paylaştırılamaz. Dolayısıyla küçük mesajlarda Tree, devasa tensörlerde Ring kazanır.
2. **SHARP (Scalable Hierarchical Aggregation and Reduction Protocol):**
   - NVIDIA Quantum-2 InfiniBand switch yongalarında gömülü donanımsal aritmetik mantık birimleri (ALU) bulunur. Bu teknolojiye **In-Network Computing** denir.
   - Geleneksel mimaride GPU'lar veriyi switch üzerinden birbirine aktarır ve toplama işlemini GPU Tensor Core/CUDA çekirdekleri yapar; veri ağdan defalarca geçer.
   - SHARP ile her GPU gradient paketini switch'e gönderir. Switch donanım seviyesinde FP16/BF16/FP32 toplamayı switch portları üzerinde yapar ve nihai toplamı tek bir aşağı yayınla (multicast) GPU'lara geri verir.
   - Ağ kablolarındaki toplam veri trafiği anında yarıya iner ($S$ kadar yukarı, $S$ kadar aşağı). Bu, devasa kümelerde all-reduce gecikmesini %50'ye varan oranda azaltır.



<!-- restored-svg:tr_full:27 -->
<svg viewBox="0 0 720 334" role="img" aria-label="İki panel. Geleneksel: A, B ve C GPU&#x27;ları veriyi yalnızca paket ileten bir switch üzerinden değiş tokuş eder ve toplamalar GPU&#x27;larda yapılır; veri ring adımları boyunca switch&#x27;ten defalarca geçer. SHARP: switch desteklenen azaltma işlemlerini donanımda yapar; her GPU girdisini bir kez yukarı yollar ve sonucu bir kez aşağı alır. Kazanç mesaj boyutuna, azaltma türüne ve trafiğe bağlıdır; sabit %50 tasarruf değildir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="sh-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="sh-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="sh-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Toplama nerede yapılıyor?</text>
<rect x="16" y="40" width="336" height="262" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">Geleneksel: GPU&#x27;larda reduce</text>
<rect x="134" y="84" width="100" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="184" y="102" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">switch</text>
<text x="184" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">yalnızca iletir</text>
<rect x="46" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="82" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU A</text>
<text x="82" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">toplar</text>
<path d="M72 196 L164 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M174 126 L92 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="186" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU B</text>
<text x="186" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">toplar</text>
<path d="M176 196 L184 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M194 126 L196 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="254" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="290" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU C</text>
<text x="290" y="229" text-anchor="middle" style="fill:var(--c-accent);font-size:10.5px">toplar</text>
<path d="M280 196 L204 126" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M214 126 L300 196" marker-end="url(#sh-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="28" y="262" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">veri switch&#x27;ten defalarca geçer (ring adımları)</text>
<text x="28" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">aritmetik GPU çekirdeklerinde yapılır</text>
<rect x="368" y="40" width="336" height="262" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="380" y="60" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">SHARP: switch içinde reduce</text>
<rect x="486" y="84" width="100" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="536" y="102" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">switch</text>
<text x="536" y="117" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px;font-weight:600">donanımda Σ</text>
<rect x="398" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="434" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU A</text>
<text x="434" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir kez yollar</text>
<path d="M424 196 L516 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M526 126 L444 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="502" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="538" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU B</text>
<text x="538" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir kez yollar</text>
<path d="M528 196 L536 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M546 126 L548 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<rect x="606" y="196" width="72" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="642" y="214" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:600">GPU C</text>
<text x="642" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir kez yollar</text>
<path d="M632 196 L556 126" marker-end="url(#sh-a)" style="fill:none;stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M566 126 L652 196" marker-end="url(#sh-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="380" y="262" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">bir kez yukarı: girdiler</text>
<text x="551.944" y="262" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">· bir kez aşağı: sonuç</text>
<text x="380" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">desteklenen azaltma ağda yapılır</text>
<text x="16" y="322" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Kazanç mesaj boyutuna, azaltma türüne ve trafiğe bağlıdır; sabit %50 tasarruf değildir.</text>
</svg>

### 2.4. Overlap ne kazandırır?

Derin öğrenme eğitiminde en büyük performans kaybı, GPU'nun hesaplama yaparken ağı beklemesi veya ağ aktarımı yapılırken GPU çekirdeklerinin boş beklemesidir. Modern kütüphaneler (PyTorch DDP, Megatron-LM) **Compute–Communication Overlap** stratejisi uygular.

Bu mekanizma CUDA Streams (eşzamanlı kuyruklar) ve **Gradient Bucketing** ile çalışır:

1. **Katman Sırası:** Geriye yayılım (backward pass) modelin son katmanından (örneğin Katman 80) ilk katmanına (Katman 1) doğru geriye doğru ilerler.
2. **Kovalara Ayırma (Bucketing):** PyTorch DDP, üretilen gradient'leri tek tek göndermez; örneğin $25 \text{ MB}$'lık kovalar (buckets) oluşturur.
3. **Eşzamanlı Yürütme:** Katman 80'in gradient kovası dolar dolmaz, arka plandaki NCCL iletişim akışında (communication stream) AllReduce tetiklenir. O sırada GPU hesaplama akışı (compute stream) Katman 79, 78 ve 77'nin backward GEMM matrislerini çarpmaya devam eder.
4. **Gizlenen Zaman:** Eğer kovanın ağ üzerinden aktarılma süresi ($T_{\mathrm{comm}}$), takip eden katmanların hesaplama süresinden ($T_{\mathrm{comp}}$) kısa veya eşitse ($T_{\mathrm{comm}} \le T_{\mathrm{comp}}$), iletişim tamamen hesaplamanın altına gizlenir. Dışarıdan bakan bir profiler'da ağ maliyeti sıfır ms gibi görünür.



<!-- restored-svg:tr_full:28 -->
<svg viewBox="0 0 720 274" role="img" aria-label="Hesap ve ağ stream&#x27;leri olan iki zaman çizelgesi. Kötü zamanlama: hesap 1–2 çalışır, ardından hesap boşta beklerken ağ aktarımı yapılır, sonra hesap 3–4 gelir. Örtüşmeli: hesap 1–4 kesintisiz çalışırken ağ aktarımının bağımsız parçası altında yürür; adım, aktarım süresi kadar erken biter. Bu yalnızca aktarım yanındaki hesaba bağımlı değilse işe yarar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="os-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="os-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-danger)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">İki CUDA stream&#x27;inde hesap ve iletişim</text>
<text x="16" y="58" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">kötü zamanlama</text>
<text x="140" y="82" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">hesap</text>
<text x="140" y="116" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">ağ</text>
<line x1="150" y1="96" x2="704" y2="96" style="stroke:var(--c-border);stroke-width:0.6;stroke-dasharray:4 3"/>
<rect x="150" y="66" width="140" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="220.0" y="82.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap 1–2</text>
<rect x="290" y="100" width="140" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360.0" y="116.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ</text>
<rect x="290" y="66" width="140" height="26" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="360.0" y="82.96" text-anchor="middle" style="fill:var(--c-danger);font-size:11px">boşta</text>
<rect x="430" y="66" width="140" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="500.0" y="82.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap 3–4</text>
<line x1="570" y1="56" x2="570" y2="132" style="stroke:var(--c-text);stroke-width:1.4"/>
<text x="576" y="138" text-anchor="start" style="fill:var(--c-text);font-size:11px">bitiş</text>
<text x="16" y="162" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">örtüşmeli</text>
<text x="140" y="186" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">hesap</text>
<text x="140" y="220" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">ağ</text>
<line x1="150" y1="200" x2="704" y2="200" style="stroke:var(--c-border);stroke-width:0.6;stroke-dasharray:4 3"/>
<rect x="150" y="170" width="280" height="26" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="290.0" y="186.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">hesap 1–4</text>
<rect x="220" y="204" width="170" height="26" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="305.0" y="220.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">ağ (bağımsız parça)</text>
<line x1="430" y1="160" x2="430" y2="236" style="stroke:var(--c-text);stroke-width:1.4"/>
<text x="436" y="242" text-anchor="start" style="fill:var(--c-text);font-size:11px">bitiş</text>
<path d="M434 183 H566" marker-start="url(#os-b)" marker-end="url(#os-b)" style="fill:none;stroke:var(--c-danger);stroke-width:1.5"/>
<text x="500" y="176" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:600">kazanılan süre</text>
<text x="16" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Yalnızca aktarım yanındaki hesaptan bağımsızsa işe yarar; gerçek bir bağımlılık kritik yolda kalır.</text>
</svg>

### 2.5. Kolektif süresi neden yalnızca byte/bandwidth değildir?

Bir kümede 1.024 GPU ile AllReduce çalıştırdığınızda, matematiksel formül aktarımın 5 ms sürmesi gerektiğini söylese bile profiler'da 18 ms görebilirsiniz. Bunun nedeni kolektif iletişimin katı bir **dağıtık bariyer (distributed barrier)** olmasıdır.

$$
T_{\mathrm{collective}} = \max_{i \in [1, N]} (T_{\mathrm{compute}, i}) + \max_{i \in [1, N]} (T_{\mathrm{comm}, i}) + T_{\mathrm{skew}}
$$

Bu gecikmenin ardındaki üç temel fiziksel etken:

1. **Straggler (Aylak) Etkisi:** 1.024 GPU'dan yalnızca bir tanesi termal kısıtlama (thermal throttling) nedeniyle çekirdek saat hızını düşürürse veya bellek ECC hatası düzeltmeye çalışırsa, backward hesaplamasını 10 ms geç tamamlar. Kalan 1.023 GPU AllReduce bariyerinde kilitli bekler.
2. **Kuyruk Gecikmesi ve Paket Kaybı:** RoCE v2 gibi kayıplı Ethernet ağlarında PFC (Priority Flow Control) devreye girip duraklatma (pause) çerçeveleri ürettiğinde, tek bir switch portundaki tıkanıklık geri teperek (head-of-line blocking) tüm raydaki iletişimi durdurabilir.
3. **P-99 Kuyruk Dağılımı:** Düğüm sayısı arttıkça, en az bir bileşenin P-99 veya P-99.9 gecikme anomalisi yaşaması istatistiksel bir kesinliğe dönüşür. Kolektif zamanı medyana (P-50) değil, daima grubun en yavaş üyesine (P-100) kilitlenir.

---

## 3. Model hangi eksenlerden bölünür?

Büyük bir yapay zekâ modelini yüzlerce veya binlerce GPU'ya yaymak tek bir yöntemle yapılamaz. Modelin farklı tensör boyutları (batch, katman, gizli boyut, dizi uzunluğu, uzmanlar) farklı paralellik eksenlerine bölünür. Bu yaklaşımların tamamının bir arada kullanılmasına **Çok Boyutlu (3D / 4D) Paralellik** denir.

| Paralellik Ekseni | Hangi Tensör Boyutu Bölünür? | İletişim Tipi | İletişim Frekansı | Minimum Donanım Gereksinimi |
|---|---|---|---|---|
| **Data Parallelism (DP)** | Batch Size ($B$) | AllReduce | Adım başına 1 kez (backward sonu) | InfiniBand / RoCE |
| **ZeRO-3 / FSDP** | Parametreler, Grads, Opt State | AllGather + ReduceScatter | Her katmanda (forward & backward) | Düşük gecikmeli Fabric |
| **Tensor Parallelism (TP)** | Hidden Dim ($H$), FFN ($4H$) | AllReduce | Her Transformer katmanında 2 kez | Düğüm içi NVLink ($900\text{ GB/s}$) |
| **Pipeline Parallelism (PP)** | Katman Derinliği ($L$) | P2P Send/Recv | Microbatch başına stage sınırında | Düğümler arası Fabric |
| **Context Parallelism (CP)** | Dizi Uzunluğu ($S$) | AllToAll / P2P Ring | Her Attention katmanında | Düğüm içi NVLink / Yüksek hızlı Leaf |
| **Expert Parallelism (EP)** | MoE Uzmanları ($E$) | AllToAll (Dispatch & Combine) | Her MoE katmanında 2 kez | Ray-optimize Fabric (Düşük oversubscription) |

### 3.1. Data Parallelism

Klasik Veri Paralelliğinde (DDP) her GPU modelin parametrelerinin, optimizer durumlarının ve gradient'lerinin tam bir kopyasını kendi yerel HBM belleğinde tutar.

1. **İşleyiş:** Global batch $B$, gruptaki GPU sayısı $N_{\mathrm{DP}}$'ye bölünür. Her GPU kendine düşen bağımsız $B / N_{\mathrm{DP}}$ boyutundaki veri dilimini (microbatch) forward ve backward geçişlerinden geçirir.
2. **Senkronizasyon:** Backward sonunda her GPU kendi yerel gradient'ini üretmiştir. Tüm GPU'lar arasında bir AllReduce çalıştırılarak gradient'lerin ortalaması alınır. Her GPU aynı ortalama gradient ile kendi optimizer adımını uygular.
3. **Temel Darboğaz:** Model büyüdükçe (örneğin 70B parametreli bir LLM), parametrelerin ve optimizer durumlarının tek bir GPU'nun belleğine ($80 \text{ GB}$) sığması imkânsız hale gelir. Bu noktada saf DP yetersiz kalır ve FSDP devreye girer.



<!-- restored-svg:tr_full:30 -->
<svg viewBox="0 0 720 296" role="img" aria-label="Veri paralelliği. Global batch A, B ve C parçalarına bölünür. Her parça tam model kopyası tutan bir GPU grubuna gider; grup ileri ve geri geçişi çalıştırıp A, B ve C gradyanlarını üretir. AllReduce gradyanların ortalamasını alır ve her kopya aynı optimizer adımını uygular. Veriyle ölçeklemek kolaydır, ama her kopya tüm model durumunu saklar ve gradyanlar her adımda senkronize edilmelidir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="dp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="dp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="dp-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Veri paralelliği: aynı model, farklı veri, ortak güncelleme</text>
<rect x="16" y="104" width="96" height="60" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="64.0" y="131.75" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Global batch</text>
<text x="64.0" y="146.75" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">3&#x27;e bölünür</text>
<path d="M112 134 C 130 134, 130 70, 150 70" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="56" width="58" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.18;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="179.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">parça A</text>
<line x1="208" y1="70" x2="224" y2="70" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="46" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU grubu A</text>
<text x="319" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tam model kopyası · ileri + geri</text>
<line x1="412" y1="70" x2="428" y2="70" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="56" width="70" height="28" rx="4" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="465.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad A</text>
<path d="M500 70 C 520 70, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M112 134 C 130 134, 130 132, 150 132" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="118" width="58" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.18;stroke:var(--c-success);stroke-width:1.2"/>
<text x="179.0" y="135.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">parça B</text>
<line x1="208" y1="132" x2="224" y2="132" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="108" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="128" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU grubu B</text>
<text x="319" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tam model kopyası · ileri + geri</text>
<line x1="412" y1="132" x2="428" y2="132" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="118" width="70" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<text x="465.0" y="135.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad B</text>
<path d="M500 132 C 520 132, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M112 134 C 130 134, 130 194, 150 194" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="150" y="180" width="58" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="179.0" y="197.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">parça C</text>
<line x1="208" y1="194" x2="224" y2="194" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="226" y="170" width="186" height="48" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="319" y="190" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU grubu C</text>
<text x="319" y="206" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">tam model kopyası · ileri + geri</text>
<line x1="412" y1="194" x2="428" y2="194" marker-end="url(#dp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="430" y="180" width="70" height="28" rx="4" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="465.0" y="197.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px">grad C</text>
<path d="M500 194 C 520 194, 520 134, 538 134" marker-end="url(#dp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="540" y="106" width="164" height="56" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="622.0" y="132.0" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">AllReduce</text>
<text x="622.0" y="147.0" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">gradyan ortalaması</text>
<path d="M622 162 V240 H319 V220" marker-end="url(#dp-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="470" y="256" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:600">her kopyada aynı optimizer adımı</text>
<text x="16" y="284" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">+ veriyle ölçeklemek kolay   − her kopya tüm model durumunu saklar; her adımda gradyan senkronu</text>
</svg>

### 3.2. FSDP ve ZeRO

DeepSpeed tarafından ortaya konan ve PyTorch'ta **Fully Sharded Data Parallel (FSDP)** olarak standartlaşan **ZeRO (Zero Redundancy Optimizer)** mimarisi, klasik DP'deki bellek israfını ortadan kaldırır.

AdamW optimizer ile FP16/BF16 eğitim yaparken parametre başına bellek muhasebesi:
- Model Parametreleri (FP16/BF16): $2\text{ bayt}$
- Gradient'ler (FP16/BF16): $2\text{ bayt}$
- Optimizer Durumları (FP32 master ağırlıklar, momentum, varyans): $4 + 4 + 4 = 12\text{ bayt}$
- **Toplam Statik Bellek:** Parametre başına tam **$16\text{ bayt}$**!

70 milyar parametreli bir model için statik durum:
$$M_{\mathrm{static}} = 70 \times 10^9 \times 16\text{ bayt} \approx 1.120\text{ GB} = 1.12\text{ TB}$$
Bu modeli tek bir 80 GB GPU'da barındırmak imkânsızdır. ZeRO bu belleği üç aşamada dağıtır:

1. **ZeRO-1 (Optimizer State Partitioning):**
   - $12\text{ bayt}$'lık optimizer durumları $N$ GPU arasında paylaştırılır ($12 / N$).
   - GPU başına bellek: $2 + 2 + \frac{12}{N}$ bayt. 4 GPU'da parametre başına bellek $16$'dan $7\text{ bayt}$'a iner.
   - *Ek İletişim Maliyeti:* Sıfır! Klasik DP'deki AllReduce yerine ReduceScatter yapılır; her GPU sadece kendi payına düşen optimizer dilimini günceller.
2. **ZeRO-2 (Gradient Partitioning):**
   - Gradient'ler de $N$ GPU arasında paylaştırılır ($2 / N$).
   - GPU başına bellek: $2 + \frac{2}{N} + \frac{12}{N}$ bayt. 4 GPU'da parametre başına $5.5\text{ bayt}$.
   - *Ek İletişim Maliyeti:* Sıfır!
3. **ZeRO-3 / FSDP (Parameter Partitioning):**
   - Modelin asıl parametreleri de $N$ GPU arasında shard edilir ($2 / N$).
   - GPU başına bellek: $\frac{2 + 2 + 12}{N} = \frac{16}{N}$ bayt. Bellek doğrusal olarak küçülür; 4 GPU'da parametre başına $4\text{ bayt}$'a düşer.
   - *İletişim Bedeli:* Bellek kazanılır ama iletişim artar! Bir katmanın forward hesaplaması başlamadan önce bir `AllGather` çağrılarak ağırlıklar toplanır, forward biter bitmez silinir. Backward sırasında katmanın ağırlıkları tekrar `AllGather` ile toplanır, backward hesaplanır, ağırlıklar silinir ve gradient'ler `ReduceScatter` ile paylaştırılır.
   - FSDP, standart DP'ye göre toplam iletişim hacmini %50 artırır ($2S \to 3S$), ancak gigabaytlarca bellek tasarrufu sağlar.



<!-- restored-svg:tr_full:33 -->
<svg viewBox="0 0 720 298" role="img" aria-label="4 yönlü veri paralelliğinde Adam ile GPU başına model durumu, parametre başına byte. Düz DP her şeyi her GPU&#x27;da tutar: 2 byte parametre, 2 byte gradyan, 12 byte optimizer durumu, toplam 16 byte. ZeRO-1 optimizer&#x27;ı böler: 2 + 2 + 3 = 7 byte, ReduceScatter ile. ZeRO-2 gradyanları da böler: 2 + 0,5 + 3 = 5,5 byte. ZeRO-3 / FSDP parametreleri de böler: 0,5 + 0,5 + 3 = 4 byte; AllGather ve ReduceScatter gerekir. Kazanılan bellek iletişimle ödenir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="zr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Adam ile GPU başına model durumu (parametre başına byte, DP = 4)</text>
<text x="16" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">düz DP</text>
<text x="16" y="83" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">hiçbir şey bölünmez</text>
<rect x="130" y="54" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="54" width="43" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="196.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="218" y="54" width="263" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="350.0" y="72" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">12</text>
<text x="490" y="72" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">16 B</text>
<text x="704" y="72" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">—</text>
<text x="16" y="118" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-1</text>
<text x="16" y="133" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">optimizer bölünür</text>
<rect x="130" y="104" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="104" width="43" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="196.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="218" y="104" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="251.0" y="122" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="292" y="122" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">7 B</text>
<text x="704" y="122" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">ReduceScatter</text>
<text x="16" y="168" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-2</text>
<text x="16" y="183" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">+ gradyanlar bölünür</text>
<rect x="130" y="154" width="43" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="152.0" y="172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">2</text>
<rect x="174" y="154" width="10" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="185" y="154" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="218.0" y="172" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="259.0" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">5,5 B</text>
<text x="704" y="172" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">ReduceScatter</text>
<text x="16" y="218" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">ZeRO-3 / FSDP</text>
<text x="16" y="233" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">+ parametreler bölünür</text>
<rect x="130" y="204" width="10" height="26" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="204" width="10" height="26" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="152" y="204" width="65" height="26" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="185.0" y="222" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">3</text>
<text x="226.0" y="222" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:700">4 B</text>
<text x="704" y="222" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">AllGather + ReduceScatter</text>
<rect x="16" y="254" width="12" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="34" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">parametre</text>
<rect x="120" y="254" width="12" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="138" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyan</text>
<rect x="222" y="254" width="12" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="240" y="264" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">optimizer (FP32 master + 2 moment)</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Kazanılan bellek iletişimle ödenir: bölünmüş durum gerektiğinde toplanmalı veya dağıtılmalıdır.</text>
</svg>

### 3.3. Tensor Parallelism

FSDP bellek sorununu çözse de, aşırı büyük modellerde tek bir katmanın aktivasyon matrisleri veya ara tensörleri dahi tek bir GPU'ya sığmayabilir ya da hesaplama gecikmesi kabul edilemez seviyelere çıkar. Megatron-LM tarafından geliştirilen **Tensor Parallelism (TP)**, bir Transformer katmanının içindeki ağırlık matrislerini satır veya sütun bazında birden çok GPU'ya böler.

Megatron-LM MLP bloğunun 4 adımlı paralelleştirme mimarisi:

1. **Giriş Yayını:** Giriş tensörü $X$, gruptaki tüm GPU'lara özdeş olarak verilir.
2. **Column Parallel Linear (İlk Projeksiyon - $W_1$):**
   - İlk lineer katmanın ağırlık matrisi sütunları boyunca $N_{\mathrm{TP}}$ parçaya bölünür: $W_1 = [W_{1,1} \mid W_{1,2} \mid \dots \mid W_{1,k}]$.
   - Her GPU bağımsız olarak kendi sütun dilimiyle çarpar: $Y_i = \text{GELU}(X \cdot W_{1,i})$.
   - Hiçbir iletişim gerekmez! Ara aktivasyon boyutu $4H / N_{\mathrm{TP}}$ olur.
3. **Row Parallel Linear (İkinci Projeksiyon - $W_2$):**
   - İkinci lineer katmanın ağırlık matrisi satırları boyunca bölünür:
     $$W_2 = \begin{bmatrix} W_{2,1} \\ W_{2,2} \\ \vdots \\ W_{2,k} \end{bmatrix}$$
   - Her GPU kendi yerel aktivasyonu ile kendi satır dilimini çarpar: $Z_i = Y_i \cdot W_{2,i}$.
   - Matematiksel olarak nihai sonuç bu çarpımların toplamıdır:
     $$Z = X W_1 W_2 = \sum_{i=1}^{k} Z_i$$
4. **AllReduce Bariyeri:**
   - Elde edilen $Z_i$ parçaları henüz nihai çıktı değildir. Katmanın çıkışında tam bir **AllReduce (SUM)** çağrılır ve toplanmış $Z$ tensörü tüm GPU'lara yazılır.

**Kritik Kural: TP Neden Düğüm Dışına Çıkamaz?**
Standart bir Transformer bloğunda hem Multi-Head Attention (Self-Attention) çıkışında hem de MLP çıkışında birer AllReduce bulunur. 80 katmanlı bir LLM'de tek bir forward geçişinde $80 \times 2 = 160$ AllReduce, backward ile birlikte toplam **320 AllReduce** çalışır! Bu işlem her katmanın kritik yolundadır; hesaplama devam edemez. Bu frekanstaki bir iletişim ancak $900 \text{ GB/s}$'lik NVSwitch alanında çalışabilir; InfiniBand veya RoCE üzerinden yürütülürse model zamanının %80'ini ağ kuyruklarında harcar.



<!-- restored-svg:tr_full:31 -->
<svg viewBox="0 0 720 352" role="img" aria-label="Megatron tarzı MLP&#x27;de dört GPU ile tensor paralelliği. Her GPU aynı X girdisini alır. GPU i, W1&#x27;in i. sütun bloğunu tutar ve iletişim olmadan act(X·W1ᵢ)&#x27;yi hesaplar; ardından W2&#x27;nin i. satır bloğuyla çarpıp kısmi sonuç Zᵢ&#x27;yi üretir. AllReduce dört kısmi sonucu toplayıp Z&#x27;yi verir. Bu AllReduce her katmanın kritik yolundadır; bu yüzden TP genellikle NVLink alanında tutulur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="tp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="tp-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="tp-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Megatron tarzı MLP&#x27;de tensor paralelliği (TP = 4)</text>
<rect x="16" y="140" width="70" height="60" rx="4" style="fill:var(--c-text-mute);fill-opacity:.08;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="51.0" y="175.76" text-anchor="middle" style="fill:var(--c-text);font-size:16px;font-family:var(--font-mono)">X</text>
<text x="51" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">X her GPU&#x27;da</text>
<text x="51" y="229" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">aynı</text>
<path d="M86 170 C 100 170, 100 70, 116 70" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="44" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="130" y="62" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 0</text>
<rect x="196" y="54" width="82" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="74.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 0]</text>
<line x1="278" y1="70" x2="292" y2="70" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="54" width="82" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="335.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="70" x2="390" y2="70" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="54" width="116" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="450.0" y="73.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[0, :] = Zᵢ</text>
<path d="M518 70 C 540 70, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M86 170 C 100 170, 100 134, 116 134" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="108" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="130" y="126" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 1</text>
<rect x="196" y="118" width="82" height="32" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="237.0" y="138.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 1]</text>
<line x1="278" y1="134" x2="292" y2="134" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="118" width="82" height="32" rx="4" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="335.0" y="137.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="134" x2="390" y2="134" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="118" width="116" height="32" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="450.0" y="137.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[1, :] = Zᵢ</text>
<path d="M518 134 C 540 134, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M86 170 C 100 170, 100 198, 116 198" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="172" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="130" y="190" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 2</text>
<rect x="196" y="182" width="82" height="32" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="237.0" y="202.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 2]</text>
<line x1="278" y1="198" x2="292" y2="198" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="182" width="82" height="32" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="335.0" y="201.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="198" x2="390" y2="198" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="182" width="116" height="32" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="450.0" y="201.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[2, :] = Zᵢ</text>
<path d="M518 198 C 540 198, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M86 170 C 100 170, 100 262, 116 262" marker-end="url(#tp-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="118" y="236" width="400" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="130" y="254" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">GPU 3</text>
<rect x="196" y="246" width="82" height="32" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="237.0" y="266.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">W1[:, 3]</text>
<line x1="278" y1="262" x2="292" y2="262" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="294" y="246" width="82" height="32" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="335.0" y="265.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">act(XW1ᵢ)</text>
<line x1="376" y1="262" x2="390" y2="262" marker-end="url(#tp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="392" y="246" width="116" height="32" rx="4" style="fill:var(--c-accent-2);fill-opacity:.2;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="450.0" y="265.96" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">·W2[3, :] = Zᵢ</text>
<path d="M518 262 C 540 262, 540 170, 560 170" marker-end="url(#tp-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="196" y="302" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">sütun paralel (iletişim yok)</text>
<text x="392" y="302" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">satır paralel (kısmi toplam)</text>
<rect x="562" y="140" width="142" height="60" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="633.0" y="167.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">AllReduce</text>
<text x="633.0" y="183.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">Z = Σ Zᵢ</text>
<text x="633" y="222" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">kritik yolda,</text>
<text x="633" y="236" text-anchor="middle" style="fill:var(--c-warn);font-size:11px;font-weight:600">her katmanda → NVLink</text>
<text x="16" y="324" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Her GPU W1 ve W2&#x27;nin ¼&#x27;ünü tutar. MLP ileri geçişi başına bir AllReduce: küçük, sık, gecikmeye duyarlı;</text>
<text x="16" y="340" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">bu yüzden mümkünse TP&#x27;yi NVLink alanında tutun.</text>
</svg>

### 3.4. Pipeline Parallelism

Model o kadar büyüktür ki katmanları bile tek bir düğüme sığmaz (örneğin 405 milyar parametreli bir model 128 Transformer katmanına sahiptir). Bu durumda modelin katmanları gruplanarak ardışık sunuculara dağıtılır. Buna **Pipeline Parallelism (PP)** denir.

Örneğin 4 pipeline stage'i ($p=4$) olan bir sistemde:
- GPU 0: Katman 1–20
- GPU 1: Katman 21–40
- GPU 2: Katman 41–60
- GPU 3: Katman 61–80

#### Pipeline Bubble ve 1F1B Çizelgelemesi

Basit bir ardışık çalıştırmada GPU 0 işini yaparken diğer 3 GPU beklemek zorundadır. Bu boşta geçen zamana **Pipeline Bubble** (kabarcık) denir. Kabarcık süresini minimize etmek için global batch çok sayıda küçük **microbatch** ($m$) dilimine ayrılır ve **1F1B (One Forward, One Backward)** zamanlama algoritması uygulanır.

1F1B modunda her GPU, bir microbatch'in forward işlemini tamamladıktan hemen sonra hazır olan bir önceki microbatch'in backward işlemini yürütür. Böylece bellek aktivasyon patlaması sınırlandırılır.

İdeal bir 1F1B çizelgelemesinde bubble oranı matematiksel olarak şöyledir:

$$
F_{\mathrm{bubble}} = \frac{p - 1}{m + p - 1}
$$

Burada $p$ pipeline stage sayısı, $m$ ise microbatch sayısıdır.

**Sayısal Analiz:**
16 pipeline stage'li devasa bir kümede ($p=16$) microbatch sayısının bubble üzerindeki etkisini inceleyelim:
- $m = 16 \implies F_{\mathrm{bubble}} = \frac{15}{16 + 15} = \frac{15}{31} \approx 48.4\%$ (Korkunç bir verimsizlik! Hesaplama kapasitesinin yarısı boşa gider.)
- $m = 32 \implies F_{\mathrm{bubble}} = \frac{15}{32 + 15} = \frac{15}{47} \approx 31.9\%$
- $m = 64 \implies F_{\mathrm{bubble}} = \frac{15}{64 + 15} = \frac{15}{79} \approx 19.0\%$
- $m = 128 \implies F_{\mathrm{bubble}} = \frac{15}{128 + 15} = \frac{15}{143} \approx 10.5\%$

Görüldüğü gibi kabarcığı %10'un altına indirmek için $m \ge 8p$ microbatch derinliği gerekir. Ancak microbatch arttıkça bellekte saklanan ara aktivasyon tensörleri katlanarak artar. Bu ikilemi çözmek için modern mimariler **Interleaved 1F1B** (her GPU'ya ardışık olmayan birden fazla sanal katman vermek) veya **DeepSeek-V3 DualPipe** (çift yönlü boru hattı ile forward ve backward iletişimlerini birbiriyle örtüştürme) tekniklerini kullanır.



<!-- restored-svg:tr_full:32 -->
<svg viewBox="0 0 720 302" role="img" aria-label="4 stage ve 6 microbatch&#x27;li, yalnızca ileri geçişi gösteren 9 zaman dilimlik pipeline çizelgesi. Stage 0, 1–6. dilimlerde microbatch 1–6&#x27;yı çalıştırır; sonraki her stage bir dilim geç başlar, stage 3 onları 4–9. dilimlerde çalıştırır. Sonraki stage&#x27;lerin başındaki ve önceki stage&#x27;lerin sonundaki boş dilimler pipeline bubble&#x27;dır: stage başına 9 dilimin 3&#x27;ü, yaklaşık %33; (p − 1)/(m + p − 1) ile aynıdır. 16 stage&#x27;de ideal bubble 16 microbatch için %48,4, 32 için %31,9, 128 için %10,5&#x27;tir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="pp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Pipeline bubble: p = 4 stage, m = 6 microbatch (ileri geçiş)</text>
<text x="142.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t1</text>
<text x="206.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t2</text>
<text x="270.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t3</text>
<text x="334.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t4</text>
<text x="398.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t5</text>
<text x="462.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t6</text>
<text x="526.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t7</text>
<text x="590.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t8</text>
<text x="654.0" y="44" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">t9</text>
<text x="16" y="74" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 0</text>
<rect x="112" y="56" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="142.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="176" y="56" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="206.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="240" y="56" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="270.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="304" y="56" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="334.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="368" y="56" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="398.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="432" y="56" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="462.0" y="75.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<rect x="496" y="56" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="560" y="56" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="624" y="56" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="16" y="112" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 1</text>
<rect x="112" y="94" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="176" y="94" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="206.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="240" y="94" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="304" y="94" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="334.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="368" y="94" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="398.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="432" y="94" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="462.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="496" y="94" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="526.0" y="113.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<rect x="560" y="94" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="624" y="94" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="16" y="150" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 2</text>
<rect x="112" y="132" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="176" y="132" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="240" y="132" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="270.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="304" y="132" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="334.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="368" y="132" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="398.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="432" y="132" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="462.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="496" y="132" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="526.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="560" y="132" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="590.0" y="151.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<rect x="624" y="132" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="16" y="188" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Stage 3</text>
<rect x="112" y="170" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="176" y="170" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="240" y="170" width="60" height="30" rx="4" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="304" y="170" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="334.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F1</text>
<rect x="368" y="170" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="398.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F2</text>
<rect x="432" y="170" width="60" height="30" rx="4" style="fill:var(--c-warn);fill-opacity:.22;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="462.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F3</text>
<rect x="496" y="170" width="60" height="30" rx="4" style="fill:var(--c-accent-2);fill-opacity:.22;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="526.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F4</text>
<rect x="560" y="170" width="60" height="30" rx="4" style="fill:var(--c-accent);fill-opacity:.22;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="590.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F5</text>
<rect x="624" y="170" width="60" height="30" rx="4" style="fill:var(--c-success);fill-opacity:.22;stroke:var(--c-success);stroke-width:1.2"/>
<text x="654.0" y="189.14" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">F6</text>
<text x="110" y="222" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">kesikli kırmızı = bubble (boş stage)</text>
<rect x="16" y="234" width="688" height="56" rx="6" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="254" text-anchor="start" style="fill:var(--c-text);font-size:12.5px;font-weight:600">Bubble ≈ (p − 1) / (m + p − 1) = burada 3 / 9 ≈ %33.</text>
<text x="28" y="274" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">p = 16: m = 16 → %48,4 · m = 32 → %31,9 · m = 128 → %10,5 (idealize, eşit stage)</text>
</svg>

### 3.5. Context ve Expert Parallelism

Modern yapay zekâ iş yüklerinde iki yeni tensör boyutu daha paralelleştirme gerektirir: devasa bağlam pencereleri (Long Context) ve seyreltik uzman modelleri (MoE).

#### Context Parallelism (CP / RingAttention)
Bağlam uzunluğu $32\text{K}$'dan $128\text{K}$ veya $1\text{M}$ token'a çıktığında, Self-Attention matrisinin bellek ihtiyacı dizinin karesiyle ($O(S^2)$) büyür ve tek bir GPU'nun HBM'sini anında tüketir.
- **Mekanizma:** Giriş dizisi token ekseninde $N_{\mathrm{CP}}$ parçaya bölünür. Her GPU dizinin bir dilimini ($S / N_{\mathrm{CP}}$) alır.
- **RingAttention:** Query ($Q$) yerel GPU'da sabit kalırken, Key ($K$) ve Value ($V$) blokları komşu GPU'lara dairesel bir halka boyunca (P2P Send/Recv) aktarılır. Her GPU gelen $K, V$ bloklarıyla yerel $Q$ arasındaki dikkat puanlarını hesaplar ve yerel softmax istatistiklerini çevrimiçi günceller (Online Softmax).

#### Expert Parallelism (EP / MoE)
Mixture of Experts modellerinde (örneğin Mixtral veya DeepSeek), MLP katmanı çok sayıda bağımsız "uzmana" (expert) bölünür. Her token için bir yönlendirici (gate/router) ağ en uygun $k$ uzmanı seçer.
- **Mekanizma:** Uzmanlar GPU'lara dağıtılır (örneğin 64 uzman, 16 GPU'ya dörder adet).
- **AllToAll Dispatch:** Bir GPU'daki token'ların yönlendirildiği uzmanlar başka GPU'lardaysa, token aktivasyon tensörleri küme genelinde bir `AllToAll` kolektifi ile ilgili uzman GPU'lara postalanır.
- **Hesaplama ve AllToAll Combine:** Uzman GPU'lar kendilerine gelen token'ları hesaplar ve sonuç tensörleri ikinci bir `AllToAll` ile token'ların ait olduğu orijinal GPU'lara geri postalanır.
- **Risk:** Router token'ları homojen dağıtamazsa, belirli popüler uzmanların bulunduğu GPU'lar aşırı yüklenir (hot-spot); diğer GPU'lar AllToAll bariyerinde bekler.



<!-- restored-svg:tr_full:34 -->
<svg viewBox="0 0 720 298" role="img" aria-label="Her biri bir uzman barındıran dört GPU ile expert paralelliği. Her GPU&#x27;daki token&#x27;lar router&#x27;ın seçtiği uzmanın rengindedir. AllToAll dispatch her token&#x27;ı uzmanının bulunduğu GPU&#x27;ya gönderir; E2 uzmanına altı, diğerlerine ikişer token gider ve E2 sıcak noktaya döner. Uzmanlar çalıştıktan sonra ikinci bir AllToAll (combine) sonuçları kaynak GPU&#x27;lara geri yollar. MoE token başına aktif FLOP&#x27;u azaltır ama her katmana iki ağ karışımı ekler; dengesiz router bir darboğaz yaratır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="me-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="me-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Expert paralelliği: token&#x27;lar uzmanlarına gider ve döner (2 × AllToAll)</text>
<rect x="16" y="50" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="75" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 0</text>
<rect x="80" y="59" width="22" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="91.0" y="74.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E0</text>
<rect x="107" y="59" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="118.0" y="74.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="134" y="59" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="145.0" y="74.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="16" y="102" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="127" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 1</text>
<rect x="80" y="111" width="22" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="91.0" y="126.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E1</text>
<rect x="107" y="111" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="118.0" y="126.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="134" y="111" width="22" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="145.0" y="126.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E3</text>
<rect x="16" y="154" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="179" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 2</text>
<rect x="80" y="163" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="91.0" y="178.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="107" y="163" width="22" height="24" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="118.0" y="178.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E0</text>
<rect x="134" y="163" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="145.0" y="178.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="16" y="206" width="150" height="42" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="231" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU 3</text>
<rect x="80" y="215" width="22" height="24" rx="4" style="fill:var(--c-accent-2);fill-opacity:.25;stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="91.0" y="230.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E3</text>
<rect x="107" y="215" width="22" height="24" rx="4" style="fill:var(--c-warn);fill-opacity:.25;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="118.0" y="230.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E2</text>
<rect x="134" y="215" width="22" height="24" rx="4" style="fill:var(--c-success);fill-opacity:.25;stroke:var(--c-success);stroke-width:1.2"/>
<text x="145.0" y="230.42" text-anchor="middle" style="fill:var(--c-text);font-size:9.5px;font-family:var(--font-mono)">E1</text>
<rect x="196" y="50" width="92" height="196" rx="4" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="242" y="140" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">AllToAll</text>
<text x="242" y="156" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">dispatch</text>
<line x1="166" y1="150" x2="194" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<line x1="288" y1="150" x2="316" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="318" y="50" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="330" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E0</text>
<text x="330" y="83" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 0</text>
<rect x="410" y="61" width="12" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="426" y="61" width="12" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.6"/>
<rect x="318" y="102" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="330" y="120" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E1</text>
<text x="330" y="135" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 1</text>
<rect x="410" y="113" width="12" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.6"/>
<rect x="426" y="113" width="12" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.6"/>
<rect x="318" y="154" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="330" y="172" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E2</text>
<text x="330" y="187" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 2</text>
<rect x="410" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="426" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="442" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="458" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="474" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<rect x="490" y="165" width="12" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.6"/>
<text x="368" y="187" text-anchor="start" style="fill:var(--c-danger);font-size:10.5px;font-weight:700">sıcak!</text>
<rect x="318" y="206" width="200" height="42" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="330" y="224" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Expert E3</text>
<text x="330" y="239" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">GPU 3</text>
<rect x="410" y="217" width="12" height="20" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<rect x="426" y="217" width="12" height="20" rx="2" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<line x1="518" y1="150" x2="546" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<rect x="548" y="50" width="92" height="196" rx="4" style="fill:var(--c-warn);fill-opacity:.08;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="594" y="140" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">AllToAll</text>
<text x="594" y="156" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">combine</text>
<line x1="640" y1="150" x2="668" y2="150" marker-end="url(#me-w)" style="stroke:var(--c-warn);stroke-width:2"/>
<text x="686" y="178" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kaynak</text>
<text x="686" y="192" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">GPU&#x27;ya</text>
<text x="16" y="270" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Token başına aktif FLOP azalır, ama her MoE katmanında iki ağ karışımı olur;</text>
<text x="16" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">dengesiz router da bir uzmanı herkesin beklediği darboğaza çevirir.</text>
</svg>

### 3.6. Paralellik planı hangi sırayla seçilir?

Bir modeli eğitirken paralellik eksenlerini seçmenin katı bir mühendislik karar sırası vardır:

1. **Önce TP (Düğüm İçi Sınırı):** Model katmanları tek bir GPU'ya sığmıyorsa veya gecikme çok yüksekse, $TP$ değeri artırılır. Ancak $TP \le 8$ tutulur (asla tek DGX düğümünün dışına, NVSwitch alanının ötesine taşırılmaz).
2. **Sonra CP (Uzun Bağlam Varsa):** Eğer dizi uzunluğu $S \ge 32\text{K}$ ise bellek patlamasını önlemek için $CP$ devreye alınır ($CP = 2, 4, 8$). Genellikle TP ile birlikte düğüm içi veya düşük gecikmeli yerel grupta tutulur.
3. **Ardından PP (Model Hâlâ Sığmıyorsa):** Model parametreleri ve optimizer durumları $TP$ sonrasında hâlâ sığmıyorsa, katmanlar $PP$ ile ardışık düğümlere bölünür ($PP = 4, 8, 16$).
4. **Son Olarak DP / FSDP (Geri Kalan Tüm GPU'lar):** Model donanıma yerleştirildikten sonra geriye kalan tüm küme ölçeği Data Parallelism ($DP$) ile doldurulur. Böylece global batch büyütülür ve eğitim hızı katlanır.
5. **MoE Varsa EP:** Eğer model MoE yapısındaysa, $DP$ boyutu içinde expert'ler $EP$ eksenine dağıtılır ($DP = EP \times DP_{\mathrm{replicate}}$).

---

## 4. Paralellik topolojiye nasıl yerleştirilir?

Paralellik stratejisini belirlemek denklemin sadece yarısıdır; asıl ustalık bu soyut grupları veri merkezindeki fiziksel kablolara, switch'lere ve raylara haritalamaktır.

### 4.1. Yerleştirme önceliği

Kural çok basittir: **En sık ve en yüksek bant genişliğiyle haberleşen paralellik boyutu, en hızlı ve en düşük gecikmeli fiziksel bağlantıya yerleştirilmelidir.**

```
Hızlı Fiziksel Yol  <-------------------------------------------------> Yavaş Fiziksel Yol
NVLink (900 GB/s)      SU İçi Leaf (50 GB/s)      Spine Switch (Clos)      Depolama Ağı
[      TP      ]  ->  [     CP / EP     ]   ->   [     PP / DP     ]   ->  [  Checkpoint  ]
```

Bu hiyerarşiyi ihlal etmenin sonuçları:
- TP'yi ağ üzerinden düğümler arasına yayarsanız, $900 \text{ GB/s}$ yerine $50 \text{ GB/s}$'ye düşersiniz; eğitim anında çöker.
- DP gradient senkronizasyonu ise adımda bir kez çalıştığı ve backward ile örtüşebildiği için spine katmanındaki yüzlerce atlamalı devasa Clos kumaşına rahatlıkla yayılabilir.



<!-- restored-svg:tr_full:35 -->
<svg viewBox="0 0 720 310" role="img" aria-label="H100 için 990 TFLOP/s&#x27;de iki iletişim roofline&#x27;ı. Solda tensor paralelliği: Y derecesi F · W / C&#x27;nin altında kalmalıdır. F = 28.672 olan Llama 3 70B MLP&#x27;si için 450 GB/s&#x27;lik NVLink Y ≈ 13&#x27;e izin verir, TP 8 tek DGX&#x27;e sığar; 50 GB/s&#x27;lik NIC yalnızca ≈ 1,45&#x27;e izin verir. Sağda veri paralelliği: gradyan senkronunu örtmek için GPU başına token batch&#x27;i C / W&#x27;yi aşmalıdır: düğüm içinde 2.200, düğüm başına 400 GB/s ile düğümler arasında 2.475 ve 128 expert&#x27;ten 4&#x27;ü etkin bir MoE için 79.200 token; sınır E/k kat büyür." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="cr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">İletişim roofline&#x27;ları: bölmek ne zaman kazandırmaz?</text>
<rect x="16" y="40" width="340" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="60" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700;font-family:var(--font-mono)">TP: Y ≤ F · W / C</text>
<text x="28" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Llama 3 70B MLP, F = 28.672</text>
<text x="40" y="106" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NVLink 450 GB/s</text>
<rect x="40" y="112" width="236.218" height="22" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="282.21818181818185" y="128" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Y ≤ 13,0</text>
<text x="40" y="168" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NIC 50 GB/s</text>
<rect x="40" y="174" width="26.2465" height="22" rx="2" style="fill:var(--c-danger);fill-opacity:.55"/>
<text x="72.24646464646464" y="190" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">Y ≤ 1,45</text>
<line x1="185" y1="92" x2="185" y2="230" style="stroke:var(--c-accent);stroke-width:1.4;stroke-dasharray:4 3"/>
<text x="185.0" y="244" text-anchor="middle" style="fill:var(--c-accent);font-size:11.5px;font-weight:600">TP = 8 (bir DGX)</text>
<text x="40.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">0</text>
<text x="112.5" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">4</text>
<text x="185.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">8</text>
<text x="257.5" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">12</text>
<text x="330.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">16</text>
<rect x="364" y="40" width="340" height="236" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="376" y="60" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700;font-family:var(--font-mono)">DP: B/X ≥ C / W</text>
<text x="376" y="78" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">gradyan senkronunu örtmek için GPU başına token</text>
<text x="388" y="102" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">düğüm içi (450 GB/s)</text>
<rect x="388" y="108" width="49.9937" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<text x="443.9937114000421" y="123" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2.200</text>
<text x="388" y="150" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">düğümler arası (400 GB/s/düğüm)</text>
<rect x="388" y="156" width="57.462" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="451.4619796773598" y="171" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">2.475</text>
<text x="388" y="198" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">MoE, E = 128, k = 4 (× E/k)</text>
<rect x="388" y="204" width="277.214" height="20" rx="2" style="fill:var(--c-danger);fill-opacity:.55"/>
<text x="659.213876512066" y="219" text-anchor="end" style="fill:var(--c-text);font-size:12px;font-weight:700">79.200</text>
<text x="388.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">1K</text>
<text x="534.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">10K</text>
<text x="680.0" y="262" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-family:var(--font-mono)">100K</text>
<text x="16" y="298" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">C = 990 TFLOP/s (H100 BF16). Sınırlar birinci mertebe (Scaling Book); overlap ve topoloji onları kaydırır, ama 10 kat değil.</text>
</svg>

### 4.2. Paralellik çarpımı

Bir kümedeki toplam GPU sayısı ($N_{\mathrm{total}}$), kullanılan tüm paralellik boyutlarının matematiksel çarpımına eşit olmak zorundadır:

$$
N_{\mathrm{total}} = TP \times CP \times EP \times PP \times DP
$$

#### Gerçek Dünya Vaka Analizi: Meta Llama 3 405B

Meta mühendisleri 405 milyar parametreli Llama 3 modelini **16.384 adet NVIDIA H100 GPU** üzerinde eğitti. Bu devasa süper bilgisayarda uygulanan iki farklı paralellik haritası:

1. **Standart Bağlam (8K Token):**
   - $TP = 8$ (Her 8 GPU'luk grup tek bir DGX H100 sunucusunu doldurur, tüm iletişim NVLink içinde kalır).
   - $CP = 1$ (8K bağlam tek GPU'ya sığdığı için context parallelism kapalı).
   - $PP = 16$ (128 katmanlı model 16 pipeline stage'e bölünür; her stage'de 8 katman bulunur. 16 sunucu zincirleme bağlanır: $8 \times 16 = 128\text{ GPU}$ bir tam model kopyasıdır).
   - $DP = 128$ (Bu 128 GPU'luk devasa modelden kümede tam 128 kopya çalıştırılır).
   - Sağlama: $8 \times 1 \times 16 \times 128 = 16.384\text{ GPU}$.
   - **Sonuç:** Meta bu yapılandırmayla **%41 BF16 Model FLOPs Utilization (MFU)** elde etti.
2. **Uzun Bağlam (128K Token):**
   - Model 128K bağlama geçtiğinde aktivasyon belleği patladığı için CP devreye alındı:
   - $TP = 8, CP = 16, PP = 16, DP = 8$
   - Sağlama: $8 \times 16 \times 16 \times 8 = 16.384\text{ GPU}$.
   - **Sonuç:** İletişim maliyetindeki devasa artışa rağmen **%38 BF16 MFU** korundu.



<!-- restored-svg:tr_full:36 -->
<svg viewBox="0 0 720 324" role="img" aria-label="Llama 3 405B&#x27;nin 16.384 H100&#x27;e yerleşimi. 8 GPU&#x27;luk tensor paralel grup bir DGX&#x27;i doldurur ve NVLink üzerinden konuşur. Bu sunuculardan 16&#x27;sı zincir hâlinde bir model kopyasının 16 pipeline stage&#x27;ini, yani 128 GPU&#x27;yu oluşturur ve aktivasyonları sunucudan sunucuya aktarır. 128 kopya veri paralel boyutu oluşturur ve gradyanları senkronlar. 8K bağlamda TP 8, CP 1, PP 16, DP 128 düzeni %41, 128K bağlamda TP 8, CP 16, PP 16, DP 8 düzeni %38 BF16 MFU&#x27;ya ulaştı. En konuşkan eksen en hızlı bağlantıda kalır; Meta&#x27;nın ağı RoCE idi." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="l4-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="l4-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="l4-o" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker><marker id="l4-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Llama 3 405B, 16.384 H100: 8 × 16 × 128 (TP × PP × DP)</text>
<path d="M620 58 H630 V208 H40" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<path d="M613 51 H623 V201 H33" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<rect x="16" y="44" width="600" height="150" rx="10" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="28" y="64" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">bir model kopyası = 128 GPU</text>
<rect x="28" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="33" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="33" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="33" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="33" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="46" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="45.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">0</text>
<line x1="62" y1="116" x2="64" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="64" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="69" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="69" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="69" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="69" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="82" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="81.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">1</text>
<line x1="98" y1="116" x2="100" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="100" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="105" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="105" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="105" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="105" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="118" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="117.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">2</text>
<line x1="134" y1="116" x2="136" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="136" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="141" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="141" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="154" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="153.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">3</text>
<line x1="170" y1="116" x2="172" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="172" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="177" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="177" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="177" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="177" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="190" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="189.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">4</text>
<line x1="206" y1="116" x2="208" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="208" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="213" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="213" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="213" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="213" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="226" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="225.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">5</text>
<line x1="242" y1="116" x2="244" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="244" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="249" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="249" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="262" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="261.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">6</text>
<line x1="278" y1="116" x2="280" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="280" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="285" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="285" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="285" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="285" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="298" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="297.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">7</text>
<line x1="314" y1="116" x2="316" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="316" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="321" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="321" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="334" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="333.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">8</text>
<line x1="350" y1="116" x2="352" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="352" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="357" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="357" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="357" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="357" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="370" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="369.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">9</text>
<line x1="386" y1="116" x2="388" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="388" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="393" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="393" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="393" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="393" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="406" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="405.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">10</text>
<line x1="422" y1="116" x2="424" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="424" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="429" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="429" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="429" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="429" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="442" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="441.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">11</text>
<line x1="458" y1="116" x2="460" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="460" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="465" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="465" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="465" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="465" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="478" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="477.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">12</text>
<line x1="494" y1="116" x2="496" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="496" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="501" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="501" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="501" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="501" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="514" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="513.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">13</text>
<line x1="530" y1="116" x2="532" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="532" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="537" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="537" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="537" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="537" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="550" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="549.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">14</text>
<line x1="566" y1="116" x2="568" y2="116" style="stroke:var(--c-success);stroke-width:1.2"/>
<rect x="568" y="80" width="34" height="72" rx="4" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="573" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="86" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="573" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="101" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="573" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="116" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="573" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="586" y="131" width="10" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="585.0" y="166" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px;font-family:var(--font-mono)">15</text>
<text x="28" y="184" text-anchor="start" style="fill:var(--c-success);font-size:11px;font-weight:600">PP = 16 stage (her biri bir DGX) → aktivasyon sunucudan sunucuya geçer</text>
<rect x="632" y="44" width="72" height="92" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="668" y="62" text-anchor="middle" style="fill:var(--c-accent);font-size:12px;font-weight:700">TP = 8</text>
<rect x="650" y="70" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="70" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="650" y="85" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="85" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="650" y="100" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="100" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="650" y="115" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="670" y="115" width="16" height="11" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<text x="668" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bir DGX,</text>
<text x="668" y="161" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">NVLink</text>
<text x="632" y="226" text-anchor="end" style="fill:var(--c-warn);font-size:12.5px;font-weight:700">× 128 kopya = DP</text>
<text x="16" y="226" text-anchor="start" style="fill:var(--c-warn);font-size:11px">gradyanlar kopyalar arasında senkronlanır</text>
<rect x="16" y="240" width="688" height="24" rx="4" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">8K bağlam</text>
<text x="160" y="256" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP 8 · CP 1 · PP 16 · DP 128</text>
<text x="692" y="256" text-anchor="end" style="fill:var(--c-accent);font-size:12px;font-weight:700">BF16 MFU %41</text>
<rect x="16" y="268" width="688" height="24" rx="4" style="fill:var(--c-border);fill-opacity:.05;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">128K bağlam</text>
<text x="160" y="284" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP 8 · CP 16 · PP 16 · DP 8</text>
<text x="692" y="284" text-anchor="end" style="fill:var(--c-accent);font-size:12px;font-weight:700">BF16 MFU %38</text>
<text x="16" y="312" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hızlı bağlantı en konuşkan ekseni taşır: TP NVLink içinde kalır, PP ve DP (RoCE) ağı geçer. Meta %38–43 MFU raporlar.</text>
</svg>

### 4.3. Strong ve weak scaling

Dağıtık eğitimde ölçekleme başarısı iki temel rejim altında ölçülür:

1. **Weak Scaling (Zayıf Ölçekleme):** GPU başına düşen iş yükü (batch size) sabit tutulurken GPU sayısı artırılır; dolayısıyla global problem boyutu büyür. Derin öğrenme genellikle weak scaling'i sever; GPU başına batch sabit kaldığı için compute/communication oranı korunur. Ancak global batch aşırı büyürse optimizasyon dinamiği (gradient noise scale) bozulur ve model yakınsamayabilir (loss diverge eder).
2. **Strong Scaling (Güçlü Ölçekleme):** Toplam problem boyutu (sabit global batch) sabit tutulurken GPU sayısı artırılır. GPU sayısı iki katına çıktığında, GPU başına düşen hesaplama işi yarıya iner ($T_{\mathrm{comp}} / 2$). Ancak kolektif iletişim boyutu azalmaz ya da gecikme adımları ($2(N-1)\alpha$) büyür! Bir noktadan sonra iletişim süresi hesaplama süresini aşar ve Amdahl Kanunu devreye girer: sisteme daha fazla GPU eklemek eğitimi hızlandırmaz, aksine yavaşlatır.

Ölçekleme verimliliği formülü:

$$
\eta = \frac{T_1}{k \times T_k} \times 100\%
$$

Burada $T_1$ baz kümedeki süre, $T_k$ ise $k$ kat büyük kümedeki süredir. İdeal bir sistemde $\eta = 100\%$ olmalıdır; üretim kümelerinde %80–85 üzeri mükemmel kabul edilir.

---

## 5. Bir training step’inde veri nereye gider?

Tek bir iterasyon (step) boyunca verinin ve tensörlerin donanım üzerindeki fiziksel yolculuğunu anlamadan bir eğitim sürecini profillemek imkânsızdır.



<!-- restored-svg:tr_full:38 -->
<svg viewBox="0 0 720 342" role="img" aria-label="Bir training step&#x27;in beş yerdeki kulvar görünümü: depolama, CPU ve host RAM, GPU HBM ve SM, NVLink ve ağ. 1, dataloader CPU tarafında depodan token okur. 2, ileri geçiş GEMM&#x27;leri GPU&#x27;da çalışır. 3, tensor paralel all-reduce NVLink üzerinden yapılır. 4, pipeline aktivasyonları ağı geçer. 5, loss ve geri geçiş GPU&#x27;da çalışır. 6, DP veya FSDP gradyan senkronu ağı geçer. 7, optimizer ağırlıkları GPU&#x27;da günceller ve döngü sonraki adım için ileri geçişe döner. 8, checkpoint periyodik olarak depolamaya yazılır. 2–6. adımlar her microbatch&#x27;te tekrarlanır; 3 ve 6 hesapla kısmen örtüşebilir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs><marker id="ts-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker><marker id="ts-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker><marker id="ts-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker></defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.03em">Bir training step: her evre fiziksel olarak nerede gerçekleşir?</text>
<rect x="16" y="44" width="688" height="46" rx="6" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="26" y="71" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Depolama</text>
<rect x="16" y="96" width="688" height="46" rx="6" style="fill:var(--c-text-mute);fill-opacity:.05;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="26" y="123" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">CPU + host RAM</text>
<rect x="16" y="148" width="688" height="46" rx="6" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="26" y="175" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">GPU: HBM + SM</text>
<rect x="16" y="200" width="688" height="46" rx="6" style="fill:var(--c-success);fill-opacity:.05;stroke:var(--c-success);stroke-width:1.2"/>
<text x="26" y="227" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">NVLink (TP)</text>
<rect x="16" y="252" width="688" height="46" rx="6" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="26" y="279" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-weight:600">Ağ (PP, DP)</text>
<rect x="131" y="100" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="164.0" y="115" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">oku</text>
<text x="164.0" y="129" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">1 · dataloader</text>
<rect x="203" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="236.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">forward</text>
<text x="236.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">2 · GEMM</text>
<rect x="275" y="204" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="308.0" y="219" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">TP</text>
<text x="308.0" y="233" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">3 · all-reduce</text>
<rect x="347" y="256" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="380.0" y="271" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">PP</text>
<text x="380.0" y="285" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">4 · aktivasyon</text>
<rect x="419" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="452.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">backward</text>
<text x="452.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">5 · + loss</text>
<rect x="491" y="256" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="524.0" y="271" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">DP/FSDP</text>
<text x="524.0" y="285" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">6 · grad senk.</text>
<rect x="563" y="152" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="596.0" y="167" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">optimizer</text>
<text x="596.0" y="181" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">7 · güncelle</text>
<rect x="635" y="48" width="66" height="38" rx="5" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="668.0" y="63" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-weight:700">checkpoint</text>
<text x="668.0" y="77" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">8 · periyodik</text>
<path d="M196.0 119.0 H236.0 V150" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M268.0 171.0 H308.0 V202" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M340.0 223.0 H380.0 V254" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M412.0 275.0 H452.0 V192" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M484.0 171.0 H524.0 V254" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M556.0 275.0 H596.0 V192" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M628.0 164 H668.0 V88" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<path d="M122 66 H138 V98" marker-end="url(#ts-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M596.0 192 V310 H236.0 V192" marker-end="url(#ts-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="16" y="330" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2–6. adımlar her microbatch&#x27;te tekrarlanır; 3 ve 6 hesapla kısmen örtüşebilir. 8. adım her N adımda bir çalışır (senkron/asenkron).</text>
</svg>

### 5.1. Veri okunur

Eğitim verisi (tokenize edilmiş uint16 dizileri) paralel dosya sisteminden (Lustre, GPFS veya NVMe-over-Fabrics) okunur. CPU RAM'inde barındırılan Dataloader worker süreçleri veriyi microbatch'ler halinde paketler, `pinned memory` (sayfalanamaz bellek) alanına yazar ve `cudaMemcpyAsync` ile PCIe Gen5 üzerinden GPU HBM'sine aktarır. Bu aşamada yaşanan herhangi bir disk I/O gecikmesi veya CPU worker tıkanması GPU'ları doğrudan "Input Stall" durumuna sokar.

### 5.2. Forward

Microbatch GPU HBM'sinde hazır olduğunda:
1. Embedding katmanı token ID'lerini vektörlere dönüştürür.
2. Tensor Parallel grubu Transformer bloklarını işletmeye başlar. Column-parallel GEMM çalışır, ara aktivasyonlar üretilir, Row-parallel GEMM çalışır ve NVLink üzerinden AllReduce çağrılarak katman çıktısı birleştirilir.
3. FlashAttention çekirdekleri $Q, K, V$ tensörlerini çarparak attention çıktılarını oluşturur.
4. Pipeline stage sınırına gelindiğinde, elde edilen aktivasyon tensörü InfiniBand üzerinden P2P Send ile bir sonraki düğümdeki PP stage'ine fırlatılır.
5. Bu sırada ara aktivasyonlar backward'da kullanılmak üzere HBM'de saklanır (Activation Checkpointing / Selective Recomputation kullanılıyorsa sadece attention girişleri tutulur).

### 5.3. Backward

Modelin son stage'inde loss fonksiyonu hesaplanır ve geriye doğru zincir kuralı (autograd) başlar:
1. Adjoint gradyan tensörleri ters yönde hesaplanır.
2. PP stage sınırında gradyanlar P2P ile önceki stage'e geri iletilir.
3. Her katmanın ağırlık gradyanları üretildikçe PyTorch DDP / FSDP kovalara yazar. Kova dolduğu anda arka plan CUDA stream'inde ağ üzerinden AllReduce / ReduceScatter başlatılır.
4. Hesaplama ile iletişim burada yarışır; ideal durumda backward bittiği anda tüm gradyan senkronizasyonu da tamamlanmış olur.

### 5.4. Optimizer ve checkpoint

Tüm rank'lerde gradyanlar eşitlendikten sonra:
1. Optimizer (örneğin FP32 AdamW) çekirdekleri çalışır. Master ağırlıklar, momentum ve varyans tensörleri güncellenir.
2. Yeni master ağırlıklar FP16/BF16 formatına dönüştürülerek model parametrelerine yazılır.
3. Periyodik adımlarda (örneğin her 1.000 adımda bir) tüm küme durumunu dondurarak veya asenkron DMA ile arka planda depolama sistemine Checkpoint yazar.

### 5.5. Bir step’in zaman bütçesi

Bir eğitim adımının toplam süresi şu bileşenlerin toplamıdır:

$$
T_{\mathrm{step}} = T_{\mathrm{compute}} + T_{\mathrm{unhidden\ comm}} + T_{\mathrm{input\ stall}} + T_{\mathrm{checkpoint\ amortized}}
$$

Burada $T_{\mathrm{unhidden\ comm}}$, hesaplamanın altına gizlenemeyen, açıkta kalıp Tensor Core'ları durduran net iletişim süresidir.

### 5.6. Bir step’in kritik yolunu sayıyla kurmak

Gerçek bir profiler çıktısını modelleyelim:
- Forward süresi: $14\text{ ms}$
- Backward süresi: $28\text{ ms}$
- Optimizer adımı: $3\text{ ms}$
- Toplam saf hesaplama süresi: $T_{\mathrm{compute}} = 14 + 28 + 3 = 45\text{ ms}$
- Ağ üzerinde akan toplam kolektif süresi: $22\text{ ms}$ (Gradient AllReduce)
- Pipeline aktivasyon iletimi: $4\text{ ms}$ (Kritik yola giren gecikme)
- Dataloader hazırlık süresi: $10\text{ ms}$

Eğer bu süreleri safça alt alta toplarsanız $45 + 22 + 4 + 10 = 81\text{ ms}$ bulursunuz. Bu analiz **tamamen yanlıştır**!

Kritik yol analizi:
1. Dataloader önceki step devam ederken veriyi hazırladığı için kritik yola etkisi: **$0\text{ ms}$**.
2. $22\text{ ms}$'lik gradient AllReduce'un $18\text{ ms}$'si backward hesaplamasının ($28\text{ ms}$) altına gizlenmiştir. Kritik yolda açıkta kalan süre: $22 - 18 =$ **$4\text{ ms}$**.
3. Pipeline P2P aktarımı bir sonraki stage'i beklettiği için doğrudan kritik yoldadır: **$4\text{ ms}$**.

Gerçek step süresi:
$$T_{\mathrm{step}} = 45\text{ ms (compute)} + 4\text{ ms (unhidden comm)} + 4\text{ ms (pipeline wait)} = 53\text{ ms}$$

Bu sistemde MFU'yu artırmak istiyorsanız ağ kartlarını 800G yapmak yerine, açıkta kalan $4\text{ ms}$'lik pipeline kabarcığını çözmeye odaklanmalısınız!

---

## 6. Scheduler, paylaşım ve doğrulama

Binlerce GPU'luk bir kümeyi donanım olarak kurmak yetmez; bu kaynakların iş zamanlayıcılar (scheduler) tarafından nasıl yönetildiği ve arızaların nasıl izole edildiği sistemin kaderini belirler.

### 6.1. Gang scheduling

Klasik web mikroservislerinde bir pod çökerse veya geç başlarsa sistem çalışmaya devam eder. Dağıtık yapay zekâ eğitiminde ise bu bir felakettir. 32 düğümlü bir eğitim işinde 31 düğüm ayağa kalkıp 1 düğüm eksik kalırsa, ayağa kalkan 248 GPU ilk AllReduce bariyerinde kilitlenir ve sonsuza kadar bekler.

Bu sorunu çözmek için **Gang Scheduling (All-or-Nothing Scheduling)** mekanizması (Slurm veya Kubernetes Kueue / Volcano) zorunludur:
- İş için talep edilen tüm kaynaklar (örneğin 32 düğüm, 256 GPU) aynı anda fiziksel olarak tahsis edilemiyorsa, scheduler işi **asla başlatmaz**.
- Tek bir worker bile çökse, tüm grup anında sonlandırılır veya son kontrol noktasından (checkpoint) temiz bir şekilde topluca yeniden başlatılır.

### 6.2. Topology-aware scheduling

Gelişmiş bir iş zamanlayıcı, kümedeki GPU'ları sadece birer sayı olarak görmez; veri merkezinin fiziksel omurga haritasını (topology tree) okur:
- 8 GPU talep eden bir iş asla iki farklı sunucudaki dörder GPU'ya bölünmez; mutlaka aynı NVLink alanındaki tek bir DGX sunucusuna atanır.
- 64 GPU talep eden bir işe rastgele sunucular verilmez; aynı Scalable Unit (SU) içindeki, aynı Leaf switch'e bağlı düğümler atanarak trafiğin omurga (spine) katmanına tırmanması engellenir.

### 6.3. MIG, MPS ve time-slicing

Tek bir GPU'yu birden fazla iş yükü arasında paylaştırmak gerektiğinde üç farklı teknoloji kullanılır:

| Teknoloji | İzolasyon Seviyesi | Bellek Koruması | Hata İzolasyonu | Dağıtık Eğitime Uygunluk |
|---|---|---|---|---|
| **Multi-Instance GPU (MIG)** | Donanımsal (SM, bellek, crossbar tam ayrık) | Tam (Donanımsal partition) | Mükemmel (Bir instance çökerse diğeri etkilenmez) | Uygun değil (Tekil inference / geliştirme için) |
| **Multi-Process Service (MPS)** | Mantıksal (Aynı donanım üzerinde CUDA context paylaşımı) | Paylaşımlı HBM | Zayıf (Bir süreç bellek hatası verirse context düşer) | Küçük modeller / batch inference için ideal |
| **Time-Slicing** | Zamansal (GPU zamanı süreçlere sırayla verilir) | Paylaşımlı HBM | Yok (Bellek tükenmesi OOM yaratır) | Çok kötü (Büyük bağlam değiştirme gecikmesi) |

Büyük ölçekli senkron eğitimlerde bu sanallaştırma yöntemlerinin hiçbiri kullanılmaz; her GPU tek bir sürece (bare-metal rank) tahsis edilir.

### 6.4. Kabul testi

Yeni kurulan veya bakım gören bir AI kümesi hemen eğitime verilmez. Katı bir **Kabul Testi Piramidi (Acceptance Testing)** uygulanır:

```
          [ Seviye 5: Gerçek LLM Soak Run (24-48 saat) ]
        [ Seviye 4: Tüm Küme NCCL AllReduce Sweep (Bisection) ]
      [ Seviye 3: İki Düğüm RDMA Ping-Pong & NCCL Bandwidth ]
    [ Seviye 2: Tek Düğüm cuBLAS GEMM & NVLink BandwidthTest ]
  [ Seviye 1: Donanım & DCGM Diagnostik (ECC, Termal, PCIe Eye) ]
```

1. **Seviye 1 (DCGM):** NVIDIA Data Center GPU Manager ile bellek ECC hataları, PCIe düşmeleri, güç sınırları ve termal kısıtlamalar taranır.
2. **Seviye 2 (Tek Düğüm):** `bandwidthTest` ile GPU-HBM ($3.35\text{ TB/s}$) ve P2P NVLink ($900\text{ GB/s}$) doğrulanır. `cublasLt-bench` ile saf GEMM TFLOP/s performansı ölçülür.
3. **Seviye 3 (İki Düğüm RDMA):** `ibv_rc_pingpong` ile ağ gecikmesi ($< 1.5\ \mu\text{s}$) ve `nccl-tests` ile çift yönlü tek ray bant genişliği ($> 45\text{ GB/s}$) doğrulanır.
4. **Seviye 4 (Küme Geneli NCCL Sweep):** Tüm düğümler arasında eşzamanlı AllReduce ve AllToAll çalıştırılarak bisection bant genişliği ve link kilitlenmeleri (flapping links) tespit edilir.
5. **Seviye 5 (Soak Test):** Sentetik veriyle tam ölçekli Llama mimarisi 48 saat kesintisiz çalıştırılır; MFU stabilitesi ve loss eğrisi izlenir.

---

## 7. 16 GPU’luk tasarım örneği

Öğrendiklerimizi somut bir kurumsal mühendislik senaryosunda birleştirelim.

**Senaryo:** Elinizde iki adet DGX H100 sunucusu ($N = 16\text{ GPU}$) var. Llama 3 70B modelini fine-tune etmek veya bir ön eğitim dilimi çalıştırmak istiyorsunuz. Donanım ve paralellik planını nasıl kurarsınız?

### 7.1. Fiziksel yerleşim

İlk refleks: Modeli 4 pipeline stage'e ve her stage'i 2 GPU'luk TP grubuna bölmek olabilir ($TP=2, PP=4, DP=2$).
Bu seçim **büyük bir tasarım hatasıdır!**

Neden?
- $TP=2$ seçerseniz, DGX içindeki $900 \text{ GB/s}$'lik devasa NVSwitch bant genişliğinin çoğunu boşa harcarsınız.
- $PP=4$ seçerseniz, iki sunucu arasında 4 ayrı pipeline sınırı oluşur; InfiniBand hatları üzerinden sürekli aktivasyon ve gradyan taşınır; ayrıca $F_{\mathrm{bubble}} = \frac{3}{m + 3}$ kabarcığı oluşur.

**Optimal Tasarım:**
- $TP = 8$: Modelin katman içi matrisleri tek bir DGX sunucusunun içindeki 8 GPU'ya yayılır. Tüm o yoğun, katman başı AllReduce iletişimi $900 \text{ GB/s}$'lik yerel NVLink alanında kalır!
- $PP = 1$: Pipeline tamamen kapatılır; kabarcık maliyeti ($F_{\mathrm{bubble}} = 0$) ve stage gecikmesi sıfırlanır!
- $DP = 2$: İki DGX sunucusu birbirine klasik Data Parallelism (FSDP / ZeRO-3) ile bağlanır.

### 7.2. Bellek sorusu

70B modelin statik durumu $1.120\text{ GB}$'tır.
- $TP=8$ uygulandığında her DGX'te model ağırlıkları 8'e bölünür.
- İki düğüm arasında FSDP (ZeRO-3) çalıştırıldığında toplam bellek 16 GPU'ya bölünür:
  $$M_{\mathrm{per\_gpu}} = \frac{1.120\text{ GB}}{16} = 70\text{ GB}$$
- Her H100 GPU'nun $80\text{ GB}$ HBM belleği vardır. Kalan $10\text{ GB}$ ve aktivasyon optimizasyonları (FlashAttention + selective recomputation) sayesinde model, iki düğüme hiçbir CPU offload'a gerek kalmadan kusursuz bir şekilde sığar!

### 7.3. Performans sorusu

İki düğüm arasındaki tek trafik, her step'in sonunda veya backward sırasında katman katman akan FSDP `AllGather` ve `ReduceScatter` trafiğidir.
- Her DGX sunucusunda 8 adet $400 \text{ Gb/s}$ ConnectX-7 NIC bulunur. Düğümler arası toplam çift yönlü bant genişliği:
  $$8 \times 50\text{ GB/s} = 400\text{ GB/s}$$
- Bu muazzam ağ kapasitesi, FSDP iletişiminin neredeyse tamamını backward GEMM hesaplamalarının altına gizler ($T_{\mathrm{unhidden\ comm}} \approx 0$).

### 7.4. İletişim haritasını nasıl çıkarırsınız?

Tasarımınızın iletişim matrisini şu tabloyla belgeleyin:

| İletişim Türü | Hangi Boyut? | Geçtiği Donanım Yolu | Beklenen Bant Genişliği | Profiler Beklentisi |
|---|---|---|---|---|
| Katman Başı AllReduce | Tensor Parallel ($TP=8$) | Düğüm içi NVSwitch 3 | $900\text{ GB/s}$ per GPU | Tamamı hesaplama adımları arasında net görünür |
| Ağırlık Toplama (AllGather) | FSDP ($DP=2$) | ConnectX-7 InfiniBand | $400\text{ GB/s}$ kümeleme | Forward hesaplamasıyla tamamen örtüşmeli |
| Gradyan Dağıtımı (ReduceScatter) | FSDP ($DP=2$) | ConnectX-7 InfiniBand | $400\text{ GB/s}$ kümeleme | Backward hesaplamasıyla tamamen örtüşmeli |
| Token Girdisi | Dataloader | PCIe Gen5 x16 | $32 - 64\text{ GB/s}$ | Kritik yola girmemeli (asenkron prefetch) |

### 7.5. Ölçekleme sonucunu nasıl yorumlarsınız?

Modeli önce tek DGX (8 GPU) üzerinde $TP=8$ ile çalıştırdınız ve step süresi $1.20\text{ saniye}$ çıktı. Ardından ikinci DGX'i ekleyip 16 GPU'ya çıktınız ve aynı batch size ile step süresi $0.68\text{ saniye}$ ölçüldü.
- Hızlanma: $\frac{1.20}{0.68} = 1.76\times$
- Ölçekleme Verimliliği:
  $$\eta = \frac{1.76}{2.0} \times 100\% = 88.2\%$$
Bu sonuç mükemmeldir! Eksik kalan %11.8'lik pay; ağ üzerindeki FSDP senkronizasyon kuyrukları ve iki sunucu arasındaki P-99 jitter farkıdır.

### 7.6. Tasarım cevabını nasıl tamamlarsınız?

Bir sistem mimarisi tasarımında bu analizi eksiksiz tamamlamak için sadece sayıları vermek yetmez; şu üç güvenceyi eklemelisiniz:
1. **Hata Alanı (Blast Radius):** Düğümlerden biri donanım arızası verirse eğitim durur; bu yüzden her 30 dakikada bir yerel NVMe'lere asenkron checkpoint alınacaktır.
2. **NUMA Hizalaması:** Her GPU'nun kendi yerel ConnectX-7 NIC'iyle eşleştirildiği doğrulanacak (`NCCL_TOPO_DUMP` ile kontrol edilecek).
3. **PFC / Flow Control:** RoCE kullanılıyorsa switch'lerde Lossless Priority Flow Control (PFC) ve ECN yapılandırması kontrol edilecektir.

---

## 8. Sistem tasarımı için yedi temel soru

### 8.1. DP ile FSDP farkı nedir?

Klasik Veri Paralelliğinde (DDP) her GPU modelin parametrelerini, gradyanlarını ve optimizer durumlarını tam kopya olarak saklar; iletişim yalnızca backward sonunda gradient AllReduce'tan ibarettir. Bellek karmaşıklığı $O(\text{Model Boyutu})$ olup model büyüdükçe GPU'ya sığmaz.

FSDP (Fully Sharded Data Parallel) ise ZeRO-3 yaklaşımıyla model parametrelerini, gradyanları ve optimizer durumlarını rank'ler arasında eşit olarak dilimler (shard eder). Bellek ihtiyacı $O(\text{Model Boyutu} / N)$'e iner. Ancak bunun bedeli iletişimdir: her katmanda ağırlıkları anlık toplamak için `AllGather` ve gradyanları paylaştırmak için `ReduceScatter` çalıştırılır; toplam ağ trafiği DDP'ye göre %50 artar.

### 8.2. TP neden düğüm içinde tutulur?

Tensor Parallelism, bir Transformer katmanının ağırlık matrislerini sütun ve satır olarak böler. Bunun sonucunda hem Attention hem de MLP bloklarının çıkışında katman başına iki adet AllReduce çalıştırılması zorunludur.

80 katmanlı modern bir LLM'de tek bir adımda 320 kez AllReduce tetiklenir. Bu işlemler hesaplamanın doğrudan kritik yolundadır (sonraki katman önceki katmanın AllReduce sonucu olmadan başlayamaz). Bu devasa mesaj frekansı ve mikrosaniyenin altındaki gecikme ihtiyacı ancak $900 \text{ GB/s}$ bant genişliğine ve $0.5\ \mu\text{s}$ gecikmeye sahip NVSwitch donanımıyla karşılanabilir. Ağ üzerinden düğüm dışına çıkarılırsa eğitim ağ kuyruklarında donar.

### 8.3. PP bubble nasıl azaltılır?

Pipeline Parallelism'de stage'ler arasındaki bağımlılıktan kaynaklanan kabarcık oranı $F_{\mathrm{bubble}} = \frac{p-1}{m+p-1}$ formülüyle belirlenir. Bu oranı düşürmek için üç temel mühendislik kaldıracı kullanılır:

1. **Microbatch Sayısını Artırmak ($m \gg p$):** Microbatch sayısını stage sayısının en az 4–8 katına çıkarmak bubble oranını %10–15 seviyelerine çeker. Ancak bu durum aktivasyon belleğini katlar.
2. **Interleaved 1F1B:** Her fiziksel GPU'ya tek bir blok yerine birden çok sanal stage (virtual stage, $v$) atanır. Kabarcık formülü $F_{\mathrm{bubble}} \approx \frac{p-1}{v \cdot m}$ haline gelir ve $v$ kat küçülür.
3. **DualPipe (DeepSeek-V3 Yaklaşımı):** İleri ve geri yönlü pipeline akışlarını karşılıklı çalıştırarak hesaplama ile pipeline sınırındaki P2P iletişimleri birbirinin kabarcıkları içine örer.

### 8.4. AllReduce neden yavaşlar?

Ring veya Tree AllReduce operasyonunun teorik bant genişliğinin çok altında kalmasının 5 temel kök nedeni vardır:
1. **Straggler GPU:** Gruptaki tek bir GPU'nun aşırı ısınma (throttling) veya bellek hatası nedeniyle yavaşlaması tüm grubu bariyerde kilitler.
2. **NUMA ve UPI Çapraz Geçişi:** GPU'nun yanlış NUMA düğümündeki NIC üzerinden konuşması veriyi yavaş CPU UPI linkine sokar.
3. **Ağ Tıkanıklığı ve Paket Düşmesi:** RoCE ağlarında PFC duraklatma fırtınaları (pause frames) veya switch arabellek taşmaları.
4. **Oversubscription:** Leaf-spine katmanında 1:1 yerine 2:1 veya 3:1 oversubscription olması durumunda farklı raflar arasındaki bisection bant genişliğinin boğulması.
5. **Küçük Mesaj Boyutu:** Mesaj boyutu $S_{1/2}$ eşiğinin altında kaldığında aktarım süresini bant genişliği değil ağ gecikmesi ($\alpha$) domine eder.

### 8.5. Daha çok GPU neden lineer hız vermez?

Sisteme daha fazla GPU eklendiğinde üç fiziksel yasa lineer hızlanmayı engeller:
1. **Amdahl Kanunu:** Programın paralelleştirilemeyen seri kısımları (veri yükleme, optimizer adımı, senkronizasyon bariyerleri) GPU sayısı arttıkça toplam sürenin baskın bileşeni haline gelir.
2. **İletişim ve Gecikme Adımları:** Ring AllReduce'ta gecikme adımı $2(N-1)\alpha$ doğrusal büyür. Düğüm sayısı arttıkça ağaç derinliği ve paket atlama sayıları artar.
3. **Strong Scaling Darboğazı:** Global batch sabitken GPU sayısı artarsa, GPU başına düşen hesaplama işi ($T_{\mathrm{comp}}$) küçülür; buna karşın iletişim süresi ($T_{\mathrm{comm}}$) sabit kalır veya büyür. Hesaplama/iletişim oranı bozulur.

### 8.6. MoE neden AllToAll üretir?

Mixture of Experts modellerinde her token için yönlendirici ağ (router) dinamik olarak farklı uzmanları seçer. Uzmanlar farklı GPU'lara dağıtıldığından (Expert Parallelism), token tensörlerinin bulundukları GPU'lardan ilgili uzmanların bulunduğu hedef GPU'lara fiziksel olarak taşınması gerekir.

Bu aktarım tam bir kişiselleştirilmiş çoktan-çoğa transferdir: her GPU diğer tüm GPU'lara farklı token paketleri yollar (`AllToAll Dispatch`). Uzman hesaplaması bittikten sonra da üretilen çıktılar orijinal GPU'lara aynı şekilde geri döner (`AllToAll Combine`). Bu durum ağ kumaşında bisection bant genişliğini son damlasına kadar zorlayan devasa bir trafik deseni üretir.

### 8.7. Dağıtık iş neden topluca başlatılmalıdır?

Kolektif iletişim primitifleri (NCCL) tüm rank'lerin aynı anda oturuma katılmasını gerektiren senkronize dağıtık durum makineleridir. Eksik tek bir rank dahi olsa hiçbir kolektif işlem tamamlanamaz.

Eğer iş zamanlayıcı kısmi başlatma yaparsa (örneğin 64 GPU'luk işin 60 GPU'sunu başlatıp 4'ünü kuyrukta bekletirse), başlayan 60 GPU hiçbir iş üretemeden boşta döner, bellek ve elektrik tüketir ve kümedeki diğer işlerin kaynaklarını gasp eder. Bu yüzden Gang Scheduling ile tüm rank'lerin aynı anda ayağa kalkması garanti edilmelidir.

---

### Okuyucu artık hangi sorulara yanıt verebilir?

- Hockney modeliyle bir kolektif iletişimin gecikme mi yoksa bant genişliği mi darboğazında olduğunu nasıl anlarsınız?
- Ring AllReduce algoritmasında $N$ GPU varken neden taşınan toplam veri hacmi $2\frac{N-1}{N}S$ formülüyle sınırlı kalır?
- 70B veya 405B parametreli bir modelde TP, PP, CP, EP ve DP eksenleri hangi sıra ve kurallarla seçilir?
- Bir DGX H100 sunucusunda yanlış NUMA yerleşiminin ağ performansına faturası nedir?
- Profiler zaman çizelgesinde hesaplama ile örtüşen ve örtüşemeyen iletişim süreleri nasıl ayrıştırılır?

---

## Bütün hikâye altı satırda

- Dağıtık eğitim performansı, soyut GPU sayısına değil tensör trafiğinin fiziksel ağ topolojisine yerleşimine bağlıdır.
- Ring AllReduce veri hacmini düğüm sayısından bağımsız $2S$ sınırında tutar; ancak gecikme adımları küme büyüdükçe uzar.
- DP batch'i, TP katmanı, PP model derinliğini, CP dizi uzunluğunu, EP ise MoE uzmanlarını böler.
- En yoğun iletişim üreten TP daima yerel NVLink alanında tutulmalı; DP ve PP ise scale-out kumaşına yayılmalıdır.
- Gerçek bir eğitim step'inde başarı, kolektif iletişimlerin backward hesaplamasının altına kusursuz örtüşmesiyle ölçülür.
- Sağlıklı bir küme tasarımı Gang scheduling, rail-optimize topoloji ve katı kabul testleriyle garanti altına alınır.

---

## Terimler sözlüğü

- **Rank:** Dağıtık eğitimde çalışan ve benzersiz bir kimlik numarasına (ID) sahip her bir bağımsız CUDA süreci.
- **Collective (Kolektif):** Bir iletişim grubundaki tüm rank'lerin aynı anda katıldığı senkron veri takas işlemi (AllReduce, AllGather vb.).
- **NVSwitch:** Düğüm içindeki 8 GPU'yu tam bağlı $900 \text{ GB/s}$ çift yönlü bant genişliğiyle konuşturan donanımsal anahtarlama yongası.
- **Rail-Optimized:** Kümedeki sunucuların aynı indeksli NIC'lerini aynı leaf switch'e bağlayarak ray içi trafiği tek atlamada tutan ağ mimarisi.
- **ZeRO / FSDP:** Model parametrelerini, gradyanları ve optimizer durumlarını rank'ler arasında bölerek bellek tasarrufu sağlayan mimari.
- **Pipeline Bubble:** Pipeline stage'leri arasındaki sıralı bağımlılık yüzünden GPU'ların boşta beklediği zaman dilimi.
- **Straggler:** Termal kısıtlama veya donanım anomalisi nedeniyle gruptan geri kalıp tüm kolektif bariyeri kilitleyen yavaş rank.
- **In-Network Computing (SHARP):** İndirgeme ve toplama işlemlerinin GPU'lar yerine doğrudan InfiniBand switch yongaları üzerinde yapılması.
- **Gang Scheduling:** Dağıtık bir işin tüm worker'ları aynı anda hazır olmadan hiçbir parçayı başlatmayan zamanlama stratejisi.

---

## Kaynaklar ve doğrulama bağlantıları

- Önceki yazı: [Yapay Zekâ Altyapısı (2): LLM GPU'yu Nasıl Kullanır? — Training, Prefill ve Decode](post.html?slug=gpu-is-yukleri-ve-uretim-cikarimi)
- Sonraki yazı: [Yapay Zekâ Altyapısı (4): AI Kümesi Neden Yavaşlar? — Depolama, Arızalar ve MFU](post.html?slug=gpudan-ai-super-bilgisayarina)
- **[K1] Scale-out topolojisi:** NVIDIA, [*DGX SuperPOD Network Fabrics*](https://docs.nvidia.com/dgx-superpod/reference-architecture-scalable-infrastructure-h100/latest/network-fabrics.html).
- **[K2] NCCL:** NVIDIA, [*NCCL User Guide*](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html).
- **[K3] Paralellik stratejileri:** NVIDIA, [*Megatron Core Parallelism Guide*](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) ve PyTorch, [*FSDP2*](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).
- **[K4] Gang scheduling:** Kubernetes SIGs, [*Kueue All-or-Nothing Scheduling*](https://github.com/kubernetes-sigs/kueue/blob/main/site/content/en/v0.20/docs/concepts/all_or_nothing.md).
- **[K5] İletişim testi:** NVIDIA, [*nccl-tests*](https://github.com/NVIDIA/nccl-tests) ve [*DCGM Diagnostics*](https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/dcgm-diagnostics.html).
