Bir yapay zekâ modeline prompt girdiğinizde veya `torchrun` komutuyla 405 milyar parametreli bir modelin eğitimini başlattığınızda, arka planda soyut ve ağırlıksız bir "bulut" çalışmaz.

Tam tersine, modern yapay zekâ tamamen katı fiziğe, mikron düzeyinde silikon geometrisine, nanometre mertebesinde bakır veri yollarına, yüksek frekanslı lazer optiklerine ve megavatlarca elektrik enerjisine bağımlıdır. Bir dil modelinin ürettiği her bir sonraki token (next-token prediction), aslında yüz milyarlarca parametrenin yüksek bant genişlikli belleklerden (HBM) saniyede terabaytlarca hızla çekilip, binlerce Tensor Core üzerinde matris-vektör çarpımına sokulmasıdır.

**Bu yazının temel fikri şu:** Bir yapay zekâ sistemini gerçekten anlamak, yalnızca Transformer denklemlerini veya loss fonksiyonunu bilmek değildir; o tensörlerin fiziksel evrende hangi kablolardan aktığını, hangi donanımsal duvarlara çarptığını ve neden megavatlarca elektrik tükettiğini baştan sona görebilmektir.

> [!TIP]
> **Büyük Resim Posteri (İnteraktif Mimari Şeması):**  
> Bu yazıda adım adım tırmanacağımız ölçek basamaklarının (Warp $\to$ SM $\to$ H100 $\to$ DGX $\to$ SU $\to$ SuperPOD $\to$ Multi-Pod Kümesi) eksiksiz mimari şemasını bağımsız ve yüksek çözünürlüklü bir sayfada inceleyebilirsiniz:  
> [**Yapay Zekâ Süper Bilgisayarı Posteri — Yeni Sayfada Aç**](/assets/diagrams/ai-supercomputer-poster-tr.html)

**Bu yazıda**

- [1. Silikon ve SM anatomisi: Bir tensörün ilk durağı](#1-silikon-ve-sm-anatomisi-bir-tensörün-ilk-durağı)
  - [1.1. Thread, Warp ve SM alt blokları](#11-thread-warp-ve-sm-alt-blokları)
  - [1.2. Tensor Core matris gücü ve matematiksel türetim](#12-tensor-core-matris-gücü-ve-matematiksel-türetim)
  - [1.3. İlk fiziksel sınır: Bellek Duvarı ve Roofline modeli](#13-ilk-fiziksel-sınır-bellek-duvarı-ve-roofline-modeli)
- [2. İki zıt dünya: Prefill ve decode asimetrisi](#2-iki-zıt-dünya-prefill-ve-decode-asimetrisi)
  - [2.1. Prefill: Paralel GEMM ve compute-bound doyum](#21-prefill-paralel-gemm-ve-compute-bound-doyum)
  - [2.2. Decode: Seri GEMV ve memory-bandwidth darboğazı](#22-decode-seri-gemv-ve-memory-bandwidth-darboğazı)
  - [2.3. KV cache patlaması ve PagedAttention bellek sayfalaması](#23-kv-cache-patlaması-ve-pagedattention-bellek-sayfalaması)
- [3. Tek sunucudan kümeye: NVLink ve rail topolojisi](#3-tek-sunucudan-kümeye-nvlink-ve-rail-topolojisi)
  - [3.1. Düğüm içi NVSwitch ve NUMA UPI geçiş cezası](#31-düğüm-içi-nvswitch-ve-numa-upi-geçiş-cezası)
  - [3.2. GPUDirect RDMA ve sıfır-kopya veri yolu](#32-gpudirect-rdma-ve-sıfır-kopya-veri-yolu)
  - [3.3. Rail-optimized Fat-Tree Clos ağı ve 1:1 bisection bant genişliği](#33-rail-optimized-fat-tree-clos-ağı-ve-11-bisection-bant-genişliği)
- [4. Modeli bölme sanatı: 4D paralellik ve kolektif iletişim](#4-modeli-bölme-sanatı-4d-paralellik-ve-kolektif-iletişim)
  - [4.1. 16 bayt kuralı ve AdamW bellek muhasebesi](#41-16-bayt-kuralı-ve-adamw-bellek-muhasebesi)
  - [4.2. Beş eksen: TP, PP, DP/FSDP, CP ve EP](#42-beş-eksen-tp-pp-dpfsdp-cp-ve-ep)
  - [4.3. Ring AllReduce algoritması ve SHARP in-network computing](#43-ring-allreduce-algoritması-ve-sharp-in-network-computing)
  - [4.4. Meta Llama 3 405B'nin 16.384 GPU'luk yerleşimi](#44-meta-llama-3-405bnin-16384-gpuluk-yerleşimi)
- [5. Depolama tsunamisi, MTBF ve Young–Daly kuralı](#5-depolama-tsunamisi-mtbf-ve-youngdaly-kuralı)
  - [5.1. Dataloader okuması ile checkpoint yazması arasındaki zıtlık](#51-dataloader-okuması-ile-checkpoint-yazması-arasındaki-zıtlık)
  - [5.2. 16.384 GPU'da arıza kaçınılmazdır: Küme MTBF'si](#52-16384-gpuda-arıza-kaçınılmazdır-küme-mtbfsi)
  - [5.3. Young–Daly denklemi ile optimum checkpoint aralığı](#53-youngdaly-denklemi-ile-optimum-checkpoint-aralığı)
  - [5.4. Otomatik karantina döngüsü ve Silent Data Corruption](#54-otomatik-karantina-döngüsü-ve-silent-data-corruption)
- [6. O meşhur soru: %38 MFU ve kayıp %62 nerede?](#6-o-meşhur-soru-38-mfu-ve-kayıp-62-nerede)
  - [6.1. nvidia-smi kullanımının aldatıcılığı](#61-nvidia-smi-kullanımının-aldatıcılığı)
  - [6.2. Altı fiziksel kayıp kalemi: Eksik %62'nin muhasebesi](#62-altı-fiziksel-kayıp-kalemi-eksik-62nin-muhasebesi)
  - [6.3. MFU ve HFU arasındaki recomputation ayrımı](#63-mfu-ve-hfu-arasındaki-recomputation-ayrımı)
- [7. Tesis, trafo ve FinOps: Megavatlar ve fatura](#7-tesis-trafo-ve-finops-megavatlar-ve-fatura)
  - [7.1. Raf başına 40.8 kW ısı ve doğrudan yongaya sıvı soğutma (DLC)](#71-raf-başına-408-kw-ısı-ve-doğrudan-yongaya-sıvı-soğutma-dlc)
  - [7.2. Veri merkezi PUE çarpanı ve trafo kapasitesi](#72-veri-merkezi-pue-çarpanı-ve-trafo-kapasitesi)
  - [7.3. Faydalı metrikler: Joule başına token ve dolar başına token](#73-faydalı-metrikler-joule-başına-token-ve-dolar-başına-token)
  - [7.4. 70B model referans tasarımı: Serving, fine-tuning ve pretraining](#74-70b-model-referans-tasarımı-serving-fine-tuning-ve-pretraining)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için: 5 bölümlük altyapı serisi](#daha-derine-inmek-için-5-bölümlük-altyapı-serisi)

---

## 1. Silikon ve SM anatomisi: Bir tensörün ilk durağı

Yapay zekâ altyapısının en temel yapı taşı bir transistör değil, bir **Streaming Multiprocessor (SM)** çekirdeğidir. Bir NVIDIA H100 GPU'sunun kalbinde 132 adet SM bulunur.

<svg viewBox="0 0 720 414" role="img" aria-label="NVIDIA H100 SXM5 blok şeması. 132 Streaming Multiprocessor&#x27;dan oluşan bir ızgara; ilk SM büyütülmüş: her SM&#x27;de dört alt bölüm vardır, her birinin kendi warp scheduler&#x27;ı ve Tensor Core&#x27;u bulunur, dördü 256 KB L1 önbelleği / paylaşımlı belleği ortak kullanır. Bütün SM&#x27;ler 50 MB&#x27;lık ortak L2 önbellek üzerinden okur; L2&#x27;nin altında 3.35 TB/s bant genişliğiyle 80 GB HBM3 bellek durur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="hx-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="8" width="688" height="398" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="34" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">NVIDIA H100 SXM5</text>
<text x="428" y="52" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">132 × Streaming Multiprocessor (SM)</text>
<rect x="428" y="62" width="14" height="14" rx="2" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="446" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="62" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="80" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="98" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="116" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="134" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="152" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="170" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="188" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="206" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="224" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="428" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="446" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="464" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="482" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="500" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="518" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="536" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="554" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="572" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="590" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="608" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<rect x="626" y="242" width="14" height="14" rx="2" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1"/>
<line x1="428" y1="62" x2="362" y2="62" style="stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<line x1="428" y1="76" x2="362" y2="260" style="stroke:var(--c-text-mute);stroke-width:1.3;stroke-dasharray:4 3"/>
<rect x="32" y="62" width="330" height="198" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="48" y="86" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">SM 0 · yakın plan</text>
<rect x="48" y="98" width="145" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="58" y="117" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Warp Scheduler</text>
<rect x="58" y="124" width="125" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="120.5" y="138.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="201" y="98" width="145" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="211" y="117" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Warp Scheduler</text>
<rect x="211" y="124" width="125" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="273.5" y="138.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48" y="158" width="145" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="58" y="177" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Warp Scheduler</text>
<rect x="58" y="184" width="125" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="120.5" y="198.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="201" y="158" width="145" height="54" rx="6" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1.2"/>
<text x="211" y="177" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Warp Scheduler</text>
<rect x="211" y="184" width="125" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="273.5" y="198.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Tensor Core</text>
<rect x="48" y="222" width="298" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="197.0" y="239.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">256 KB L1 / paylaşımlı bellek</text>
<text x="428" y="276" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">her kare bir SM; 132&#x27;si de aynıdır</text>
<line x1="197" y1="284" x2="197" y2="302" marker-end="url(#hx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="534" y1="284" x2="534" y2="302" marker-end="url(#hx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="32" y="306" width="656" height="34" rx="8" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="360" y="328" text-anchor="middle" style="fill:var(--c-text);font-size:13px">50 MB birleşik L2 önbellek · bütün SM&#x27;ler paylaşır</text>
<line x1="197" y1="340" x2="197" y2="356" marker-end="url(#hx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="534" y1="340" x2="534" y2="356" marker-end="url(#hx-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="32" y="360" width="656" height="34" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="360" y="382" text-anchor="middle" style="fill:var(--c-text);font-size:13px">80 GB HBM3 bellek · 3.35 TB/s bant genişliği</text>
</svg>

### 1.1. Thread, Warp ve SM alt blokları

Bir SM içindeki hesaplama hiyerarşisi katı bir askeri disiplinle yürütülür:
1. **Thread:** Tek bir kayan nokta işlemini yürüten en küçük mantıksal iş parçacığı.
2. **Warp (32 Thread):** GPU donanımı thread'leri tek tek çalıştırmaz; 32 thread'lik paketler (**Warp**) halinde kilit adımda (lock-step) SIMT (Single Instruction, Multiple Threads) prensibiyle işletir.
3. **Warp Divergence:** Eğer 32 thread'lik bir warp içindeki iş parçacıklarının bir kısmı `if`, bir kısmı `else` dalına saparsa, donanım iki dalı aynı anda çalıştıramaz! Önce birinci dalı çalıştırıp diğerlerini maskeler, sonra ikinci dalı çalıştırır. Yürütme süresi iki katına çıkar; bu yüzden yüksek performanslı CUDA çekirdekleri dallanmalardan katı bir biçimde arındırılır.
4. **Latency Hiding (Little's Law):** GPU'lar bir thread belleğe takıldığında beklemek yerine, aynı saat döngüsünde hazır olan başka bir warp'a bağlam değiştirme (context-switch) cezası olmadan anında atlar ($0\text{ cycle overhead}$). Bir SM'in aritmetik birimlerini %100 meşgul tutabilmesi için donanımda aynı anda onlarca aktif warp bulunmalıdır.

### 1.2. Tensor Core matris gücü ve matematiksel türetim

Geleneksel CUDA çekirdekleri saat döngüsü başına skaler çarpma-toplama ($a \times b + c$) yaparken, 4. Nesil Tensor Core'lar doğrudan $4 \times 4$ veya $16 \times 8 \times 16$ boyutundaki küçük matrisleri tek bir saat döngüsünde çarpar:

$$
D = A \times B + C
$$

Bir H100 SXM5 GPU'nun tepe BF16 yoğun hesaplama gücünün türetimi:
- SM Sayısı: $132$
- SM Başına 4. Nesil Tensor Core: $4$ (GPU genelinde $528$ Tensor Core)
- Saat Frekansı (Boost Clock): $1.830\text{ MHz} = 1.83\text{ GHz}$
- Tensor Core başına döngüde üretilen FLOP (BF16 MAC): $1.024\text{ FLOP}$

$$
\text{Tepe BF16 Gücü} = 132 \times 4 \times 1.024 \times 1.83\text{ GHz} \approx 989.5 \approx 990\text{ TFLOP/s}
$$

Bu sayı bir mühendislik tavanıdır: Tek bir silikon yonga, her saniye yaklaşık **1 Katrilyon kayan nokta işlemi** gerçekleştirebilir.

### 1.3. İlk fiziksel sınır: Bellek Duvarı ve Roofline modeli

Ancak hesaplama gücü tek başına hiçbir anlam ifade etmez. Tensor Core'ların o 1 katrilyon işlemi yapabilmesi için sayıların önce depolandıkları bellekten (HBM) alınıp çekirdeklere getirilmesi gerekir.

Burada yapay zekânın ilk büyük fiziksel sınırı belirir: **Bellek Duvarı (Memory Wall)**.

$$
\text{Hesaplama Kapasitesi} = 989.5\text{ TFLOP/s} \quad \text{vs.} \quad \text{HBM3 Bant Genişliği} = 3.35\text{ TB/s}
$$

Bu iki fiziksel sınır arasındaki oran, sistemin **Kritik Aritmetik Yoğunluğunu ($\alpha_{\mathrm{crit}}$)** tanımlar:

$$
\alpha_{\mathrm{crit}} = \frac{989.5 \times 10^{12}\text{ FLOP/s}}{3.35 \times 10^{12}\text{ Bayt/s}} \approx 295.4\text{ FLOP/Bayt}
$$

**Roofline Kuralı:** Eğer bir işlem belleğe okuduğu her 1 bayt veri karşılığında en az **295 FLOP** matematik yapmıyorsa, Tensor Core'lar vaktinin çoğunu HBM'den veri gelmesini bekleyerek (idle) geçirir! Sistem aritmetik olarak değil, bellek bant genişliğiyle sınırlanır (Memory-Bound).

Aynı oran log-log eksenlerde çizildiğinde dünyayı ikiye böler; birazdan göreceğimiz iki LLM fazı bu çizginin zıt taraflarına düşer:

<svg viewBox="0 0 720 312" role="img" aria-label="H100 SXM5&#x27;in log-log eksenlerde Roofline modeli. Ulaşılabilir hız, aritmetik yoğunlukla birlikte 3.35 TB/s HBM bant genişliğinin belirlediği eğimle yükselir, sonra 989.5 TFLOP/s BF16 tavanında düzleşir. İki çizgi yaklaşık bayt başına 295 FLOP&#x27;ta, kırılma noktasında buluşur. B=1 decode eğimin üzerinde, bayt başına yaklaşık 1 FLOP&#x27;ta durur ve yalnızca 3.35 TFLOP/s&#x27;e, yani tepe gücün %0.34&#x27;üne ulaşır. Bayt başına 300 FLOP&#x27;un üzerindeki prefill, düz tavanın altındaki compute-bound bölgeye düşer." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="498.6355474189498" y="40" width="181.36445258105022" height="220" rx="0" style="fill:var(--c-accent);fill-opacity:.06"/>
<line x1="90" y1="260" x2="680" y2="260" style="stroke:var(--c-border);stroke-width:1.2"/>
<line x1="90" y1="40" x2="90" y2="260" style="stroke:var(--c-border);stroke-width:1.2"/>
<line x1="134.25" y1="260" x2="134.25" y2="265" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="134.25" y="279" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1</text>
<line x1="281.75" y1="260" x2="281.75" y2="265" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="281.75" y="279" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">10</text>
<line x1="429.25" y1="260" x2="429.25" y2="265" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="429.25" y="279" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100</text>
<line x1="576.75" y1="260" x2="576.75" y2="265" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="576.75" y="279" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">1000</text>
<line x1="85" y1="260.0" x2="90" y2="260.0" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="81" y="264.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">1</text>
<line x1="85" y1="193.33333333333331" x2="90" y2="193.33333333333331" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="81" y="197.33333333333331" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">10</text>
<line x1="85" y1="126.66666666666666" x2="90" y2="126.66666666666666" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="81" y="130.66666666666666" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">100</text>
<line x1="85" y1="59.99999999999997" x2="90" y2="59.99999999999997" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="81" y="63.99999999999997" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">1000</text>
<text x="90" y="28" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">TFLOP/s (log)</text>
<text x="385.0" y="300" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">aritmetik yoğunluk (FLOP/bayt, log)</text>
<path d="M89.8 245.1 L498.6 60.3 L679.8 60.3" style="fill:none;stroke:var(--c-accent);stroke-width:2.5"/>
<line x1="498.6355474189498" y1="60.305613430505645" x2="498.6355474189498" y2="260" style="stroke:var(--c-text-mute);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="504.6355474189498" y="252" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">α_crit ≈ 295</text>
<text x="510.6355474189498" y="53.305613430505645" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">989.5 TFLOP/s tavanı</text>
<text x="512.6355474189497" y="92" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:600">compute-bound</text>
<text x="512.6355474189497" y="110" text-anchor="start" style="fill:var(--c-text);font-size:12px">prefill: &gt; 300 FLOP/bayt</text>
<text x="512.6355474189497" y="126" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Tensor Core&#x27;lar doyar</text>
<text x="104" y="66" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:600">memory-bound</text>
<text x="104" y="84" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">eğim = 3.35 TB/s × yoğunluk</text>
<circle cx="134.2" cy="225.0" r="5.5" style="fill:var(--c-danger);fill-opacity:.35;stroke:var(--c-danger);stroke-width:1.5"/>
<text x="146.25" y="228.99701286421032" text-anchor="start" style="fill:var(--c-text);font-size:12px">decode, B=1: ≈ 1 FLOP/bayt</text>
<text x="146.25" y="244.99701286421032" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">→ 3.35 TFLOP/s, tepe gücün %0.34&#x27;ü</text>
</svg>

---

## 2. İki zıt dünya: Prefill ve decode asimetrisi

Bir LLM'in yaşam döngüsü iki taban tabana zıt fazdan oluşur. Bu fazların donanım talepleri öylesine farklıdır ki, ikisini aynı GPU üzerinde yan yana çalıştırmak verimlilik intiharıdır.

| Karşılaştırma Ekseni | Prefill Fazı (Prompt İşleme) | Decode Fazı (Token Üretimi) |
|---|---|---|
| **Matematiksel Form** | Matris–Matris Çarpımı (GEMM) | Matris–Vektör Çarpımı (GEMV) |
| **Girdi Boyutu** | Tüm prompt aynı anda (örn. 4.096 token) | Adım başına tek bir token ($B=1$) |
| **Aritmetik Yoğunluk** | Yüksek ($> 300\text{ FLOP/Bayt}$) | Çok düşük ($\approx 1 - 2\text{ FLOP/Bayt}$) |
| **Donanım Darboğazı** | **Compute-Bound** (Tensor Core TFLOP/s) | **Memory-Bound** (HBM3 Bandwidth) |
| **Kritik Metrik** | TTFT (Time to First Token) | TPOT (Time Per Output Token) / ITL |
| **GPU Verimliliği (MFU)** | Yüksek (%45 – %60) | Çok Düşük (%0.3 – %5) |

### 2.1. Prefill: Paralel GEMM ve compute-bound doyum

Kullanıcı sisteme 4.000 token'lık bir prompt gönderdiğinde, model tüm bu token'ları aynı anda paralel olarak işler. Matrisler karedir; her ağırlık parametresi binlerce farklı token aktivasyonuyla tekrar tekrar çarpılır.
Aritmetik yoğunluk kritik eşiği ($\alpha_{\mathrm{crit}} \approx 295$) rahatlıkla aşar. Tensor Core'lar %100 kapasiteyle doyar. Prefill fazının hızını belirleyen şey Tensor Core'ların ham TFLOP/s gücüdür.

### 2.2. Decode: Seri GEMV ve memory-bandwidth darboğazı

İlk token üretildikten sonra kâbus başlar. Otoregresif dil modelleri gelecekteki token'ları paralel üretemez; her yeni kelime bir önceki kelimeye bağımlıdır (seri döngü).

Tek bir kullanıcı için ($B=1$) sıradaki kelimeyi tahmin etmek:
- 70 milyar parametreli bir modelde (FP16), modelin tüm **140 GB ağırlık matrisi** HBM'den okunup SM çekirdeklerine çekilir.
- Çekirdekler sadece 1 adet token vektörüyle bu 140 GB'ı çarpar (GEMV).
- H100 SXM5 GPU'nun HBM bant genişliği $3.35\text{ TB/s}$'dir. 140 GB ağırlığın okunması teorik olarak en az şu kadar sürer:

$$
T_{\mathrm{token}} \ge \frac{140\text{ GB}}{3.350\text{ GB/s}} \approx 0.0418\text{ saniye} \approx 41.8\text{ ms}
$$

Teorik maksimum hız saniyede $1 / 0.0418 \approx 24\text{ token/s}$'dir! Bu sırada 990 TFLOP/s gücündeki Tensor Core'ların harcadığı fiili güç yalnızca $3.35\text{ TFLOP/s}$ düzeyindedir; yani Tensor Core'lar kapasitesinin **yalnızca %0.34'ünde** çalışır! Kalan %99.66'lık güç HBM'den verinin akmasını bekleyerek boşa gider.

### 2.3. KV cache patlaması ve PagedAttention bellek sayfalaması

Her yeni token üretilirken geçmiş token'ların Key ($K$) ve Value ($V$) vektörlerinin baştan hesaplanmaması için bellekte saklanmasına **KV Cache** denir.
Ancak bağlam uzadıkça ve eşzamanlı kullanıcı sayısı arttıkça bu arabellek HBM'yi hızla yutar.

Llama 3 70B ($L=80\text{ katman}, K=8\text{ KV kafa}, d=128$) için 1 token'ın KV cache ayak izi:

$$
\text{KV per token} = 2 \times 2\text{ bayt} \times 80 \times 8 \times 128 = 327.680\text{ bayt} \approx 320\text{ KB/token}
$$

$8\text{K}$ bağlam kullanan tek bir kullanıcı için KV cache tam **$2.56\text{ GB}$** HBM tüketir! 64 eşzamanlı kullanıcıda KV cache boyutu **$163.8\text{ GB}$**'a ulaşarak model ağırlıklarını ($140\text{ GB}$) geride bırakır.

**PagedAttention Devrimi:** Geleneksel sistemler bellek parçalanması (fragmentation) yüzünden bu belleğin %60'tan fazlasını boşa harcıyordu. İşletim sistemlerinin sanal bellek sayfalamasını (virtual memory paging) GPU HBM'sine uyarlayan **PagedAttention (vLLM)**, KV cache'i sabit 16 veya 32 token'lık küçük bloklara bölerek bellek israfını %4'ün altına indirdi ve eşzamanlı çıkarım kapasitesini 3–4 katına çıkardı.

---

## 3. Tek sunucudan kümeye: NVLink ve rail topolojisi

Model tek bir GPU'ya sığmadığında veya binlerce GPU ile eğitim yapılmak istendiğinde devreye ara bağlantılar (interconnects) girer.

Tek bir DGX H100 kasasında 8 GPU, 4 NVSwitch, 2 CPU ve 10 ConnectX-7 ağ kartı vardır. Aşağıdaki şema NVIDIA'nın bu kasa için çizdiği sistem topolojisini yeniden çizer; numaralı daireler ardından gelen altı soruyla eşleşir:

<svg viewBox="36 3907 488 418" role="img" aria-label="NVIDIA&#x27;nın şemasından yeniden çizilmiş DGX H100 sistem topolojisi. Her biri Xeon 8480C olan CPU 0 ve CPU 1 UPI ile birleşir. Her CPU, depolama için çift portlu bir ConnectX-7 kartı ve dört U.2 NVMe veri diski taşıyan bir üst PCIe switch&#x27;i besler; CPU 0&#x27;da ayrıca iki M.2 işletim sistemi diski, CPU 1&#x27;de bir 100 GbE kartı vardır. Altta CPU 0, NIC ve GPU 0–3&#x27;ü; CPU 1, NIC ve GPU 4–7&#x27;yi besler; her GPU, PCIe Gen5 üzerinden kendi ConnectX-7 ağ kartıyla eşleşir. Her GPU 18 NVLink&#x27;ini 5, 4, 4 ve 5 olarak dört NVSwitch&#x27;e dağıtır; alttaki bir PCIe switch, NVSwitch&#x27;leri Fabric Manager kontrol yolu olarak CPU 0&#x27;a bağlar. 1&#x27;den 6&#x27;ya numaralı daireler NUMA yarılarını, GPU&#x27;nun kendi NIC&#x27;ini, NVLink demetlerini, depolama PCIe switch&#x27;lerini, UPI hattını ve NVSwitch kontrol yolunu işaretler; şemanın altındaki altı soruyla eşleşir." style="width:100%;max-width:640px;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="bp-mut" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="bp-acc" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
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
</svg>

**Topolojiyi okuma: altı soru**

1. **Kim kimi besler?** Düğüm birbirinin aynası olan iki yarıdan oluşur. CPU 0, GPU 0–3'ü ve NIC 0–3'ü PCIe ile besler; CPU 1 de GPU 4–7'yi ve NIC 4–7'yi. Her yarının kendi belleği vardır; buna NUMA alanı denir. *Pratikte:* GPU 5'e veri hazırlayan süreç CPU 1'de koşmalı ve CPU 1'in belleğini kullanmalıdır; yoksa her batch önce CPU'lar arasındaki hattı geçer.
2. **GPU ağa nasıl çıkar?** Her GPU'nun hemen üstünde kendine ait 400 Gb/s'lik bir NIC durur (GPU i ↔ NIC i). GPUDirect RDMA sayesinde veri GPU belleğinden doğrudan NIC'e geçer, CPU belleğine hiç uğramaz. *Pratikte:* kasadan 8 × 50 = 400 GB/s çıkar. Kümede NIC i, i. leaf switch'e takılır; farklı düğümlerdeki aynı numaralı GPU'lar bu yüzden tek sıçramada buluşur.
3. **GPU'lar birbirleriyle nasıl konuşur?** Her GPU'nun 18 NVLink'i aşağıdaki dört NVSwitch'e iner: ikisine 5'er, ikisine 4'er hat. Herhangi iki GPU tek bir switch sıçramasıyla, her yönde 450 GB/s ile konuşur. *Pratikte:* bu yol ağdan 9 kat geniştir; her katmanda konuşan tensor parallelism (TP) bu yüzden bu sekiz GPU'nun içinde tutulur.
4. **Depolama trafiği nereden geçer?** Üstteki iki PCIe switch, çift portlu CX-7 kartlarını (depolama ağı ve in-band yönetim) ve 4'er U.2 NVMe veri diskini taşır. CPU 0'ın yanında ayrıca iki M.2 işletim sistemi diski durur. *Pratikte:* veri okumak ve checkpoint yazmak, gradyan trafiğinden farklı kartları ve kabloları kullanır; büyük bir checkpoint AllReduce'u yavaşlatmaz.
5. **İki yarıyı ne birleştirir?** İki CPU, NVIDIA'nın “CPU communication” dediği hatla (Xeon'da UPI) bağlıdır. Bu hat CPU'ların kendi trafiği içindir; iki yarıdaki GPU'lar birbirleriyle CPU'lar üzerinden değil, NVLink üzerinden konuşur. *Pratikte:* bir CPU öbür yarının belleğine erişebilir ama daha yavaş; NUMA sabitlemesi bu cezadan kaçınmak içindir.
6. **NVSwitch'leri kim yönetir?** CPU 0'dan inen PCIe hattı, alttaki PCIe switch üzerinden dört NVSwitch'e ulaşır. Bu bir kontrol yoludur: NVIDIA Fabric Manager switch'lerin yönlendirme tablolarını buradan kurar; GPU verisi bu hattan akmaz. *Pratikte:* Fabric Manager çalışmazsa GPU'lar tek tek sağlıklı görünse bile NVLink ağı kurulmaz ve çok GPU'lu işler başlamaz.

### 3.1. Düğüm içi NVSwitch ve NUMA UPI geçiş cezası

Bir DGX H100 sunucusunda 8 adet GPU bulunur. Bu 8 GPU birbirine **NVSwitch 3** yongaları üzerinden çift yönlü $900\text{ GB/s}$ hızında bağlanır. Bu hız öylesine yüksektir ki, 8 GPU tek bir devasa mantıksal GPU gibi çalışabilir.

Ancak sunucunun anakartı üzerinde iki ayrı x86 CPU ve katı bir **NUMA (Non-Uniform Memory Access)** simetrisi vardır:
- GPU 0–3 ve NIC 0–3 $\to$ NUMA Düğümü 0'a bağlıdır.
- GPU 4–7 ve NIC 4–7 $\to$ NUMA Düğümü 1'e bağlıdır.

İki CPU soketi birbirine yalnızca $\approx 64\text{ GB/s}$ bant genişliğine sahip dar bir UPI köprüsüyle bağlıdır. Eğer yazılım hatasıyla GPU 0'ın ağ trafiği NIC 4 üzerinden gönderilirse, veri PCIe $\to$ CPU 0 $\to$ UPI Link $\to$ CPU 1 $\to$ NIC 4 rotasını izler. UPI köprüsü tıkanır; ağ gecikmesi $1.2\ \mu\text{s}$'den $3.8\ \mu\text{s}$'ye fırlar ve bant genişliği yarıya düşer!

### 3.2. GPUDirect RDMA ve sıfır-kopya veri yolu

Sunucular arası iletişimde işletim sistemi kernel'ını ve host CPU belleğini aradan çıkaran teknoloji **GPUDirect RDMA (GDR)**'dır.
Klasik TCP/IP ağında tensör GPU HBM'sinden önce CPU RAM'ine kopyalanır, kernel socket buffer'larından geçer ve NIC'e yazılır. GPUDirect RDMA ile GPU, yerel PCIe switch üzerinden doğrudan ConnectX-7 ağ kartına yazar (`GPU HBM -> PCIe Switch -> NIC -> Fabric`). CPU sıfır yük altındadır ve gecikme mikrosaniyelerin altına iner.

İki veri yolu yan yana:

<svg viewBox="0 0 720 246" role="img" aria-label="Bir tensörü başka bir sunucuya göndermenin iki veri yolu. Klasik TCP/IP: tensör GPU HBM&#x27;sinden CPU RAM&#x27;deki bir kopyaya, oradan kernel socket buffer&#x27;ındaki ikinci kopyaya, sonra NIC&#x27;e ve ağ kumaşına gider; her paketi CPU çekirdekleri taşır. GPUDirect RDMA: GPU, HBM&#x27;den PCIe switch üzerinden doğrudan ConnectX-7 ağ kartına yazar ve veri kumaşa çıkar; host RAM ve kernel yolda değildir, kopya sayısı sıfırdır ve CPU boşta kalır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="gd-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="gd-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<text x="32" y="26" text-anchor="start" style="fill:var(--c-danger);font-size:12px;font-weight:600;font-size:13px">Klasik TCP/IP: host üzerinden iki kopya</text>
<rect x="32" y="38" width="116" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="90.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">GPU HBM</text>
<text x="90.0" y="76.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">tensör</text>
<line x1="148" y1="64" x2="164" y2="64" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="166" y="38" width="116" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="224.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">CPU RAM</text>
<text x="224.0" y="76.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kopya 1</text>
<line x1="282" y1="64" x2="298" y2="64" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="300" y="38" width="116" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="358.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">kernel buffer</text>
<text x="358.0" y="76.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kopya 2 · socket</text>
<line x1="416" y1="64" x2="432" y2="64" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="434" y="38" width="116" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="492.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">NIC</text>
<text x="492.0" y="76.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">ConnectX-7</text>
<line x1="550" y1="64" x2="566" y2="64" marker-end="url(#gd-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="568" y="38" width="116" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="626.0" y="60.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">kumaş</text>
<text x="626.0" y="76.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">diğer düğüme</text>
<text x="32" y="112" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">her paketi CPU çekirdekleri taşır; her kopya gecikme ekler</text>
<text x="32" y="146" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:600;font-size:13px">GPUDirect RDMA: sıfır kopya</text>
<rect x="32" y="158" width="116" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="90.0" y="180.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">GPU HBM</text>
<text x="90.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">tensör</text>
<line x1="148" y1="184" x2="164" y2="184" marker-end="url(#gd-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="166" y="158" width="250" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="291.0" y="180.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">PCIe switch</text>
<text x="291.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">GPU, DMA ile doğrudan NIC&#x27;e yazar</text>
<line x1="416" y1="184" x2="432" y2="184" marker-end="url(#gd-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="434" y="158" width="116" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="492.0" y="180.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">NIC</text>
<text x="492.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">ConnectX-7</text>
<line x1="550" y1="184" x2="566" y2="184" marker-end="url(#gd-a)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="568" y="158" width="116" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="626.0" y="180.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">kumaş</text>
<text x="626.0" y="196.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">diğer düğüme</text>
<text x="32" y="232" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:600;font-weight:400;font-size:11.5px">host RAM ve kernel yolda değil; CPU boşta kalır</text>
</svg>

### 3.3. Rail-optimized Fat-Tree Clos ağı ve 1:1 bisection bant genişliği

16.384 GPU'luk devasa bir süper bilgisayarda kablolar rastgele bağlanmaz. Binlerce sunucunun omurga switch'lerini tıkamasını önlemek için **Rail-Optimized Fat-Tree Clos** mimarisi uygulanır:

1. **Ray Ayrımı:** Her DGX sunucusundaki 8 NIC bağımsız 8 ray oluşturur (Rail 0'dan Rail 7'ye). Kümedeki tüm sunucuların 0 numaralı NIC'leri yalnızca Rail 0 leaf switch'ine bağlanır.
2. **Yerel Trafik:** Bir model kopyasında GPU 0'lar birbiriyle konuşurken trafik daima aynı ray içindeki tek bir Leaf switch üzerinden akar; omurga (spine) katmanına tırmanmadan tek atlamada (single-hop) hedefe varır.
3. **1:1 Non-Blocking Bisection:** Farklı kabinler arası iletişim için Leaf switch'lerin yukarı portları 1:1 simetrik bant genişliğiyle Spine switch'lere bağlanır; kümedeki herhangi iki GPU çifti arasındaki bisection bant genişliği eksiksiz $400\text{ Gb/s}$ ($50\text{ GB/s}$) kalır.

Kablolanmış hâliyle rail'ler şöyle görünür (dört sunucu gösterildi):

<svg viewBox="0 0 720 356" role="img" aria-label="Rail-optimized fat-tree. Dört DGX sunucusunun her birinde 0&#x27;dan 7&#x27;ye numaralı sekiz NIC vardır. Her sunucunun k numaralı NIC&#x27;i yalnızca Rail k leaf switch&#x27;ine bağlanır; böylece bütün NIC 0&#x27;lar vurgulanan Rail 0 leaf&#x27;ini paylaşır. Farklı sunuculardaki GPU 0&#x27;lar bu yüzden tek bir leaf üzerinden, spine&#x27;a hiç çıkmadan tek atlamada konuşur. Sekiz leaf switch&#x27;in her biri dört spine switch&#x27;in hepsine 1:1, non-blocking oranla bağlanır; rail&#x27;ler arası trafiği spine taşır. Her NIC 400 Gb/s, yani 50 GB/s hızındadır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="73" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="73" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="73" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="73" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="155" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="155" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="155" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="155" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="237" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="237" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="237" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="237" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="319" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="319" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="319" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="319" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="401" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="401" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="401" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="401" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="483" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="483" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="483" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="483" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="565" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="565" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="565" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="565" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="647" y1="150" x2="120" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="647" y1="150" x2="280" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="647" y1="150" x2="440" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="647" y1="150" x2="600" y2="66" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="47.0" y1="246" x2="73" y2="190" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="64.5" y1="246" x2="155" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="82.0" y1="246" x2="237" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="99.5" y1="246" x2="319" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="117.0" y1="246" x2="401" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="134.5" y1="246" x2="483" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="152.0" y1="246" x2="565" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="169.5" y1="246" x2="647" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="215.0" y1="246" x2="73" y2="190" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="232.5" y1="246" x2="155" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="250.0" y1="246" x2="237" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="267.5" y1="246" x2="319" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="285.0" y1="246" x2="401" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="302.5" y1="246" x2="483" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="320.0" y1="246" x2="565" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="337.5" y1="246" x2="647" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="383.0" y1="246" x2="73" y2="190" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="400.5" y1="246" x2="155" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="418.0" y1="246" x2="237" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="435.5" y1="246" x2="319" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="453.0" y1="246" x2="401" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="470.5" y1="246" x2="483" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="488.0" y1="246" x2="565" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="505.5" y1="246" x2="647" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="551.0" y1="246" x2="73" y2="190" style="stroke:var(--c-accent);stroke-width:1.8"/>
<line x1="568.5" y1="246" x2="155" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="586.0" y1="246" x2="237" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="603.5" y1="246" x2="319" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="621.0" y1="246" x2="401" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="638.5" y1="246" x2="483" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="656.0" y1="246" x2="565" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<line x1="673.5" y1="246" x2="647" y2="190" style="stroke:var(--c-text-mute);stroke-width:1;stroke-opacity:.35"/>
<rect x="60" y="36" width="120" height="30" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="120.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Spine 1</text>
<rect x="220" y="36" width="120" height="30" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Spine 2</text>
<rect x="380" y="36" width="120" height="30" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="440.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Spine 3</text>
<rect x="540" y="36" width="120" height="30" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="600.0" y="55.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Spine 4</text>
<rect x="36" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="73.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 0</text>
<text x="73.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="118" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="155.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 1</text>
<text x="155.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="200" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="237.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 2</text>
<text x="237.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="282" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="319.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 3</text>
<text x="319.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="364" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="401.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 4</text>
<text x="401.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="446" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="483.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 5</text>
<text x="483.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="528" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="565.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 6</text>
<text x="565.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="610" y="150" width="74" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="647.0" y="166.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Rail 7</text>
<text x="647.0" y="182.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">leaf</text>
<rect x="32" y="240" width="152" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="40.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="47.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<rect x="57.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="65.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="75.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="82.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="92.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="100.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="110.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="117.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="127.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="135.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="145.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="152.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="162.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="170.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<text x="108" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX 1</text>
<rect x="200" y="240" width="152" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="208.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="215.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<rect x="225.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="233.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="243.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="250.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="260.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="268.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="278.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="285.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="295.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="303.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="313.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="320.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="330.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="338.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<text x="276" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX 2</text>
<rect x="368" y="240" width="152" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="376.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="383.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<rect x="393.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="401.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="411.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="418.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="428.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="436.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="446.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="453.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="463.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="471.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="481.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="488.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="498.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="506.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<text x="444" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX 3</text>
<rect x="536" y="240" width="152" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="544.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="551.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">0</text>
<rect x="561.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="569.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="579.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="586.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="596.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="604.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="614.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="621.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="631.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="639.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="649.0" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="656.5" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="666.5" y="246" width="15" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="674.0" y="258.9" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<text x="612" y="286" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-weight:600">DGX n</text>
<text x="32" y="322" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:600;font-weight:400">Rail 0: bütün sunucuların NIC 0&#x27;ı aynı leaf&#x27;e bağlanır → GPU 0&#x27;lar tek atlamada konuşur</text>
<text x="32" y="342" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">leaf → spine uplink&#x27;leri 1:1 non-blocking; rail&#x27;ler arası trafik spine&#x27;a çıkar · her NIC 400 Gb/s = 50 GB/s</text>
</svg>

---

## 4. Modeli bölme sanatı: 4D paralellik ve kolektif iletişim

Bir modeli birden fazla GPU'ya bölmek tek bir eksenden yapılamaz. Modern LLM'ler **4D Paralellik (TP + PP + DP/FSDP + CP/EP)** ile eğitilir.

### 4.1. 16 bayt kuralı ve AdamW bellek muhasebesi

FP16/BF16 karma hassasiyetli AdamW eğitimi sırasında her bir model parametresi ($P$) için gereken statik bellek muhasebesi:
- Model Ağırlıkları (FP16/BF16): $2\text{ bayt}$
- Gradyanlar (FP16/BF16): $2\text{ bayt}$
- FP32 Master Ağırlıklar: $4\text{ bayt}$
- FP32 Adam Momentum: $4\text{ bayt}$
- FP32 Adam Varyans: $4\text{ bayt}$
- **Toplam:** Parametre başına tam **$16\text{ bayt}$**!

405 milyar parametreli bir model için statik durum:
$$M_{\mathrm{static}} = 405 \times 10^9 \times 16\text{ bayt} \approx 6.480\text{ GB} = 6.48\text{ TB}$$
Bu devasa tensör kütlesi tek bir GPU'ya sığamaz; matematiksel eksenlere dilimlenmek zorundadır.

16 baytın dağılımı ve 405B ölçeğinde neye vardığı:

<svg viewBox="0 0 720 190" role="img" aria-label="BF16 karma hassasiyetli AdamW eğitiminde parametre başına statik bellek, 16 baytlık bir çubuk olarak: 2 bayt BF16 ağırlık ve 2 bayt BF16 gradyan, birlikte 4 bayt; ardından 4 bayt FP32 master ağırlık, 4 bayt Adam momentum ve 4 bayt Adam varyans, birlikte 12 baytlık FP32 optimizer durumu. 405 milyar parametrede bu 6.48 TB eder; daha tek bir aktivasyon yokken yalnızca statik durumu tutmak için en az 81 adet 80 GB&#x27;lık H100 gerekir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="32" y="56" width="77" height="44" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="70.5" y="75" text-anchor="middle" style="fill:var(--c-text);font-size:12px">ağırlık</text>
<text x="70.5" y="91" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16 · 2 B</text>
<rect x="112" y="56" width="77" height="44" rx="4" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="150.5" y="75" text-anchor="middle" style="fill:var(--c-text);font-size:12px">gradyan</text>
<text x="150.5" y="91" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">BF16 · 2 B</text>
<rect x="192" y="56" width="157" height="44" rx="4" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="270.5" y="75" text-anchor="middle" style="fill:var(--c-text);font-size:12px">FP32 master ağırlık</text>
<text x="270.5" y="91" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4 B</text>
<rect x="352" y="56" width="157" height="44" rx="4" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="430.5" y="75" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Adam momentum (m)</text>
<text x="430.5" y="91" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4 B</text>
<rect x="512" y="56" width="157" height="44" rx="4" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="590.5" y="75" text-anchor="middle" style="fill:var(--c-text);font-size:12px">Adam varyans (v)</text>
<text x="590.5" y="91" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">4 B</text>
<path d="M32 50 V44 H189 V50" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="110.5" y="36" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">BF16 kopyalar: 4 bayt</text>
<path d="M192 50 V44 H669 V50" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="430.5" y="36" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">FP32 optimizer durumu: 12 bayt</text>
<line x1="32" y1="104" x2="32" y2="110" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="32" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">0</text>
<line x1="112" y1="104" x2="112" y2="110" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="112" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">2</text>
<line x1="192" y1="104" x2="192" y2="110" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="192" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">4</text>
<line x1="352" y1="104" x2="352" y2="110" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="352" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">8</text>
<line x1="512" y1="104" x2="512" y2="110" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="512" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">12</text>
<line x1="672" y1="104" x2="672" y2="110" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="672" y="124" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">16</text>
<text x="680" y="124" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">bayt</text>
<text x="32" y="158" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">405 milyar parametre × 16 bayt = 6.48 TB statik durum</text>
<text x="32" y="178" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">yalnızca bunu tutmak için ≥ 81 × H100 (80 GB); aktivasyonlar henüz hesapta yok</text>
</svg>

### 4.2. Beş eksen: TP, PP, DP/FSDP, CP ve EP

Her eksen farklı bir şeyi böler ve bedelini farklı bir iletişim türüyle öder:

<svg viewBox="0 0 720 356" role="img" aria-label="Beş paralellik ekseni; her biri neyi böldüğü ve nasıl iletişim kurduğuyla. Tensor parallelism bir katmanın içindeki MLP ve attention matrislerini GPU&#x27;lar arasında böler ve her katmanda 2 AllReduce ister; bu yüzden 900 GB/s&#x27;lik NVSwitch zorunludur. Pipeline parallelism katman yığınını sunuculara böler ve stage sınırlarında noktadan noktaya mesaj yollar; 1F1B çizelgesi bubble&#x27;ları küçültür. Context parallelism attention için token dizisini böler ve her attention katmanında ring geçişi yapar; 128K üzeri bağlam için şarttır. Expert parallelism MoE uzmanlarını farklı GPU&#x27;lara yerleştirir; token dispatch ve combine için yoğun AllToAll trafiği ister. Data parallelism her GPU&#x27;ya modelin tam bir kopyasını ve farklı veri verir; adım sonunda gradyan AllReduce ile, FSDP&#x27;de AllGather ve ReduceScatter ile senkronize olur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="32" y="24" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">eksen</text>
<text x="100" y="24" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">ne bölünür</text>
<rect x="380" y="12" width="42" height="16" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="401.0" y="23.7" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">GPU A</text>
<rect x="426" y="12" width="42" height="16" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="447.0" y="23.7" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">GPU B</text>
<text x="484" y="24" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">iletişim</text>
<rect x="16" y="36" width="688" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="30" y="52" width="56" height="24" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="58.0" y="68.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">TP</text>
<text x="100" y="59" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Tensor Parallelism</text>
<text x="100" y="77" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Katman içi: MLP/attention matrisleri bölünür</text>
<text x="484" y="59" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her katmanda 2 AllReduce</text>
<text x="484" y="77" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">NVSwitch (900 GB/s) zorunlu</text>
<rect x="400" y="45" width="19" height="38" rx="2" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="420" y="45" width="19" height="38" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="16" y="100" width="688" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="30" y="116" width="56" height="24" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="58.0" y="132.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">PP</text>
<text x="100" y="123" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Pipeline Parallelism</text>
<text x="100" y="141" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Katman derinliği: katmanlar sunuculara</text>
<text x="484" y="123" text-anchor="start" style="fill:var(--c-text);font-size:12px">Stage sınırında P2P</text>
<text x="484" y="141" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">1F1B çizelgesi bubble&#x27;ı küçültür</text>
<rect x="392" y="110.0" width="56" height="7" rx="2" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="392" y="119.5" width="56" height="7" rx="2" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="392" y="129.0" width="56" height="7" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="392" y="138.5" width="56" height="7" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="16" y="164" width="688" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="30" y="180" width="56" height="24" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="58.0" y="196.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">CP</text>
<text x="100" y="187" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Context Parallelism</text>
<text x="100" y="205" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Dizi uzunluğu: token ekseninde bölünür</text>
<text x="484" y="187" text-anchor="start" style="fill:var(--c-text);font-size:12px">Her attention katmanında ring</text>
<text x="484" y="205" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">128K+ bağlam için şart</text>
<rect x="384" y="186" width="10" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="397" y="186" width="10" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="410" y="186" width="10" height="12" rx="2" style="fill:var(--c-accent);fill-opacity:.3;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="423" y="186" width="10" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="436" y="186" width="10" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="449" y="186" width="10" height="12" rx="2" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="16" y="228" width="688" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="30" y="244" width="56" height="24" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="58.0" y="260.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">EP</text>
<text x="100" y="251" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Expert Parallelism</text>
<text x="100" y="269" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">MoE uzmanları farklı GPU&#x27;lara dağılır</text>
<text x="484" y="251" text-anchor="start" style="fill:var(--c-text);font-size:12px">Token dispatch &amp; combine</text>
<text x="484" y="269" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">yoğun AllToAll kumaş trafiği</text>
<rect x="382" y="246" width="19" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="391.5" y="259.7" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">E1</text>
<rect x="404" y="246" width="19" height="20" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="413.5" y="259.7" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">E2</text>
<rect x="426" y="246" width="19" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="435.5" y="259.7" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">E3</text>
<rect x="448" y="246" width="19" height="20" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="457.5" y="259.7" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">E4</text>
<rect x="16" y="292" width="688" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="30" y="308" width="56" height="24" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="58.0" y="324.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">DP</text>
<text x="100" y="315" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Data Parallelism / FSDP</text>
<text x="100" y="333" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Batch: farklı veri, aynı model</text>
<text x="484" y="315" text-anchor="start" style="fill:var(--c-text);font-size:12px">Adım sonunda gradyan AllReduce</text>
<text x="484" y="333" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">FSDP: AllGather + ReduceScatter</text>
<rect x="394" y="305" width="24" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="406.0" y="324.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">M</text>
<rect x="424" y="305" width="24" height="30" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="436.0" y="324.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">M</text>
</svg>

### 4.3. Ring AllReduce algoritması ve SHARP in-network computing

Data Parallelism sırasında gradyanların ortalamasını almak için kullanılan **Ring AllReduce Algoritması**, taşınan veri hacmini düğüm sayısından ($N$) bağımsız kılar.

Tensör $N$ parçaya bölünür. $N-1$ adım süren Reduce-Scatter ve ardından $N-1$ adım süren All-Gather fazlarının sonunda, GPU başına aktarılan toplam veri miktarı:

$$
V_{\mathrm{ring}} = 2 \left(\frac{N-1}{N}\right) S
$$

$N \to \infty$ iken aktarılan veri $2S$ sınırını asla aşmaz! 8 GPU'da da 1.024 GPU'da da her GPU en fazla $2S$ kadar veri taşır.

**SHARP (In-Network Computing):** NVIDIA Quantum-2 InfiniBand switch'lerin içinde gömülü donanımsal ALU'lar bulunur. GPU'lar gradyanları switch'e gönderir; switch donanım seviyesinde paketleri toplar ve sonucu tek bir yayınla GPU'lara geri verir. Ağ üzerindeki toplam veri yükü yarıya ($S$) iner!

### 4.4. Meta Llama 3 405B'nin 16.384 GPU'luk yerleşimi

Meta mühendislerinin Llama 3 405B modelini **16.384 adet NVIDIA H100 GPU** üzerinde eğittiği paralellik denklemi:

$$
N_{\mathrm{total}} = TP \times CP \times PP \times DP = 8 \times 1 \times 16 \times 128 = 16.384\text{ GPU}
$$

- $TP=8$: Her DGX içindeki 8 GPU tam bir TP grubu oluşturur; tüm katman içi AllReduce'lar $900\text{ GB/s}$ NVLink içinde kalır.
- $PP=16$: 128 katmanlı model 16 sunucuya zincirleme dağıtılır (16 DGX = 128 GPU bir tam model kopyasıdır).
- $DP=128$: Bu 128 GPU'luk modelden kümede tam 128 adet bağımsız kopya çalıştırılır.
- **Elde Edilen Sonuç:** Meta bu devasa kumaş üzerinde **%41 BF16 MFU** elde etmiştir.

Aynı yerleşim, kopyalar ve stage'ler olarak çizildiğinde:

<svg viewBox="0 0 720 330" role="img" aria-label="Llama 3 405B&#x27;nin 16.384 H100 GPU üzerindeki yerleşimi. Her satır 16 pipeline stage&#x27;inden oluşan bir model kopyasıdır; data parallel ekseni olan bu kopyalardan 128 tane vardır. Bir stage bir DGX sunucusudur; sunucunun 8 GPU&#x27;su bir tensor parallel grubu oluşturur, bu yüzden katman içi bütün AllReduce&#x27;lar 900 GB/s&#x27;lik NVLink içinde kalır. TP 8 çarpı CP 1 çarpı PP 16 çarpı DP 128, 16.384 GPU eder; bir kopya 16 DGX, yani 128 GPU&#x27;dur. Meta %41 BF16 MFU bildirir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ll-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="32" y="26" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">DP kopyaları ↓</text>
<line x1="120" y1="22" x2="260" y2="22" marker-end="url(#ll-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="268" y="26" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">PP: aktivasyonlar stage&#x27;den stage&#x27;e akar (P2P)</text>
<text x="32" y="55" text-anchor="start" style="fill:var(--c-text);font-size:12px">kopya 1</text>
<rect x="120.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="135.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">1</text>
<rect x="155.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="171.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">2</text>
<rect x="191.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="206.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">3</text>
<rect x="226.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="242.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">4</text>
<rect x="262.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="277.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">5</text>
<rect x="297.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="313.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">6</text>
<rect x="333.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="348.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">7</text>
<rect x="368.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="384.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">8</text>
<rect x="404.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="419.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">9</text>
<rect x="439.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="455.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">10</text>
<rect x="475.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="490.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">11</text>
<rect x="510.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="526.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">12</text>
<rect x="546.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="561.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">13</text>
<rect x="581.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="597.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">14</text>
<rect x="617.0" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="632.5" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">15</text>
<rect x="652.5" y="40" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.1"/>
<text x="668.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">16</text>
<text x="32" y="81" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">kopya 2</text>
<rect x="120.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="155.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="191.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="226.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="262.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="297.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="333.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="368.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="404.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="439.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="475.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="510.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="546.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="581.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="617.0" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="652.5" y="66" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<text x="32" y="107" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">kopya 3</text>
<rect x="120.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="155.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="191.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="226.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="262.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="297.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="333.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="368.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="404.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="439.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="475.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="510.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="546.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="581.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="617.0" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="652.5" y="92" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<text x="70" y="128" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">⋮</text>
<text x="404.0" y="128" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">⋮</text>
<text x="32" y="151" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">kopya 128</text>
<rect x="120.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-accent);fill-opacity:.45;stroke:var(--c-accent);stroke-width:1.1"/>
<rect x="155.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="191.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="226.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="262.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="297.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="333.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="368.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="404.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="439.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="475.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="510.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="546.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="581.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="617.0" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<rect x="652.5" y="136" width="31" height="20" rx="4" style="fill:var(--c-text-mute);fill-opacity:.10;stroke:var(--c-text-mute);stroke-width:1.1"/>
<text x="683.5" y="172" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">her kutu bir DGX = bir pipeline stage&#x27;i</text>
<line x1="120" y1="156" x2="120" y2="196" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<line x1="151" y1="156" x2="400" y2="196" style="stroke:var(--c-text-mute);stroke-width:1;stroke-dasharray:4 3"/>
<rect x="120" y="196" width="280" height="100" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="136" y="216" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">1 stage = 1 DGX · NVLink 900 GB/s</text>
<rect x="136" y="226" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="165.0" y="242.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 0</text>
<rect x="200" y="226" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="229.0" y="242.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 1</text>
<rect x="264" y="226" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="293.0" y="242.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 2</text>
<rect x="328" y="226" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="357.0" y="242.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 3</text>
<rect x="136" y="256" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="165.0" y="272.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 4</text>
<rect x="200" y="256" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="229.0" y="272.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 5</text>
<rect x="264" y="256" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="293.0" y="272.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 6</text>
<rect x="328" y="256" width="58" height="24" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="357.0" y="272.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">GPU 7</text>
<text x="120" y="316" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">TP = 8: katman içi AllReduce&#x27;lar NVLink&#x27;te kalır</text>
<rect x="428" y="196" width="260" height="116" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="444" y="218" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">TP × CP × PP × DP</text>
<text x="444" y="241" text-anchor="start" style="fill:var(--c-text);font-size:14px;font-weight:700;font-family:var(--font-mono)">8 × 1 × 16 × 128</text>
<text x="444" y="263" text-anchor="start" style="fill:var(--c-text);font-size:12px">= 16.384 GPU; kopya başına PP = 16</text>
<text x="444" y="281" text-anchor="start" style="fill:var(--c-text);font-size:12px">1 kopya = 16 DGX = 128 GPU</text>
<text x="444" y="302" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:600">Sonuç: %41 BF16 MFU</text>
</svg>

---

## 5. Depolama tsunamisi, MTBF ve Young–Daly kuralı

Hesaplama ve ağ kusursuz olsa dahi depolama altyapısı bir kümenin eğitimini kolayca durma noktasına getirebilir.

### 5.1. Dataloader okuması ile checkpoint yazması arasındaki zıtlık

- **Dataloader Okuması:** Saniyede sadece birkaç megabaytlık sürekli token akışıdır. Buradaki darboğaz bant genişliği değil, binlerce sürecin aynı anda dosya açmasıyla oluşan metadata kilitlenmeleridir.
- **Checkpoint Yazması:** 405 milyar parametreli bir modelde tüm ağırlıklar ve AdamW durumları diske dökülürken tek bir snapshot **$7.5\text{ TB}$** büyüklüğe ulaşır! 16.384 GPU aynı saniyede depolamaya yüklendiğinde yüzlerce gigabaytlık ani bir yazma tsunamisi ($> 250\text{ GB/s}$) patlar.

### 5.2. 16.384 GPU'da arıza kaçınılmazdır: Küme MTBF'si

Bir sistemdeki bileşen sayısı arttıkça bağımsız arıza olasılıkları toplanır:

$$
\mathrm{MTBF}_{\mathrm{cluster}} = \frac{\mathrm{MTBF}_{\mathrm{node}}}{N_{\mathrm{nodes}}}
$$

Tek bir DGX sunucusunun yıllık arıza oranı %10 bile olsa, binlerce sunucudan oluşan bir süper bilgisayarda optik alıcı-vericiler (transceivers), kablo bükülmeleri, sıvı soğutma sızıntıları ve bellek hataları nedeniyle ortalama arıza aralığı saatler mertebesine iner. Meta Llama 3 405B raporuna göre 54 günde 419 beklenmedik donanım arızası yaşanmıştır: **Ortalama her 3.09 saatte bir kesinti!**

### 5.3. Young–Daly denklemi ile optimum checkpoint aralığı

Arızalar 3 saatte bir yaşanıyorsa ne sıklıkla checkpoint alınmalıdır? Çözüm **Young–Daly Optimum Checkpoint Formülü**'dür:

$$
\tau_{\mathrm{opt}} \approx \sqrt{2 \times \delta \times M}
$$

Burada $\delta$ checkpoint yazma süresi, $M$ ise küme MTBF'sidir.
- Eğer checkpoint yazmak 5 dakika sürüyorsa ($\delta=5\text{ dk}, M=180\text{ dk}$), optimum checkpoint aralığı $\tau_{\mathrm{opt}} \approx 42.4\text{ dakika}$ olur ve çalışma süresinin **%23.6'sı boşa gider**.
- İki kademeli asenkron checkpointing ile yazma duraklaması 30 saniyeye indirilirse ($\delta=0.5\text{ dk}$), optimum aralık $\tau_{\mathrm{opt}} \approx 13.4\text{ dakikaya}$ iner ve toplam kayıp bütçesi **%7.5'e düşer**!

Kaybı aralığa karşı çizmek, neden bir optimum noktası olduğunu ve δ küçülünce bu noktanın neden kaydığını gösterir:

<svg viewBox="0 0 720 298" role="img" aria-label="Young–Daly dengesi: 180 dakikalık küme MTBF&#x27;sinde, checkpoint aralığı tau&#x27;ya karşı küme zamanının boşa giden oranı. Kayıp yaklaşık delta bölü tau artı tau bölü 2M&#x27;dir: çok sık checkpoint yazma duraklamalarına, çok seyrek checkpoint her arızada kaybolan işe zaman harcatır; bu yüzden her eğrinin bir çukuru vardır. 5 dakikalık senkron yazmada optimum 42.4 dakikadadır ve zamanın %23.6&#x27;sı kaybolur. 0.5 dakikalık asenkron duraklamada optimum 13.4 dakikaya kayar ve kayıp %7.5&#x27;e düşer." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="80" y1="250" x2="660" y2="250" style="stroke:var(--c-border);stroke-width:1.2"/>
<line x1="80" y1="40" x2="80" y2="250" style="stroke:var(--c-border);stroke-width:1.2"/>
<line x1="80.0" y1="250" x2="80.0" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="80.0" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">0</text>
<line x1="176.66666666666666" y1="250" x2="176.66666666666666" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="176.66666666666666" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">20</text>
<line x1="273.3333333333333" y1="250" x2="273.3333333333333" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="273.3333333333333" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">40</text>
<line x1="370.0" y1="250" x2="370.0" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="370.0" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">60</text>
<line x1="466.66666666666663" y1="250" x2="466.66666666666663" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="466.66666666666663" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">80</text>
<line x1="563.3333333333334" y1="250" x2="563.3333333333334" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="563.3333333333334" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">100</text>
<line x1="660.0" y1="250" x2="660.0" y2="255" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="660.0" y="268" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">120</text>
<line x1="75" y1="250.0" x2="80" y2="250.0" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="71" y="254.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">%0</text>
<line x1="75" y1="197.5" x2="80" y2="197.5" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="71" y="201.5" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">%10</text>
<line x1="75" y1="145.0" x2="80" y2="145.0" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="71" y="149.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">%20</text>
<line x1="75" y1="92.50000000000003" x2="80" y2="92.50000000000003" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="71" y="96.50000000000003" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">%30</text>
<line x1="75" y1="40.0" x2="80" y2="40.0" style="stroke:var(--c-border);stroke-width:1.3"/>
<text x="71" y="44.0" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">%40</text>
<text x="80" y="26" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">kayıp oranı ≈ δ/τ + τ/(2M) · M = 180 dk</text>
<text x="370.0" y="288" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">checkpoint aralığı τ (dakika)</text>
<path d="M147.7 42.1 L148.9 45.0 L150.1 47.8 L151.3 50.5 L152.5 53.1 L153.7 55.6 L154.9 58.0 L156.1 60.4 L157.3 62.6 L158.5 64.8 L159.8 66.8 L161.0 68.9 L162.2 70.8 L163.4 72.7 L164.6 74.5 L165.8 76.2 L167.0 77.9 L168.2 79.5 L169.4 81.1 L170.6 82.7 L171.8 84.1 L173.0 85.6 L174.2 86.9 L175.5 88.3 L176.7 89.6 L177.9 90.8 L179.1 92.1 L180.3 93.2 L181.5 94.4 L182.7 95.5 L183.9 96.6 L185.1 97.6 L186.3 98.6 L187.5 99.6 L188.8 100.5 L190.0 101.4 L191.2 102.3 L192.4 103.2 L193.6 104.0 L194.8 104.8 L196.0 105.6 L197.2 106.4 L198.4 107.1 L199.6 107.8 L200.8 108.5 L202.0 109.2 L203.2 109.9 L204.5 110.5 L205.7 111.1 L206.9 111.7 L208.1 112.3 L209.3 112.9 L210.5 113.4 L211.7 113.9 L212.9 114.4 L214.1 114.9 L215.3 115.4 L216.5 115.9 L217.8 116.3 L219.0 116.8 L220.2 117.2 L221.4 117.6 L222.6 118.0 L223.8 118.4 L225.0 118.8 L226.2 119.1 L227.4 119.5 L228.6 119.8 L229.8 120.1 L231.0 120.4 L232.2 120.7 L233.5 121.0 L234.7 121.3 L235.9 121.6 L237.1 121.8 L238.3 122.1 L239.5 122.3 L240.7 122.6 L241.9 122.8 L243.1 123.0 L244.3 123.2 L245.5 123.4 L246.8 123.6 L248.0 123.8 L249.2 124.0 L250.4 124.1 L251.6 124.3 L252.8 124.4 L254.0 124.6 L255.2 124.7 L256.4 124.9 L257.6 125.0 L258.8 125.1 L260.0 125.2 L261.2 125.3 L262.5 125.4 L263.7 125.5 L264.9 125.6 L266.1 125.7 L267.3 125.7 L268.5 125.8 L269.7 125.9 L270.9 125.9 L272.1 126.0 L273.3 126.0 L274.5 126.1 L275.8 126.1 L277.0 126.2 L278.2 126.2 L279.4 126.2 L280.6 126.2 L281.8 126.2 L283.0 126.3 L284.2 126.3 L285.4 126.3 L286.6 126.3 L287.8 126.2 L289.0 126.2 L290.2 126.2 L291.5 126.2 L292.7 126.2 L293.9 126.1 L295.1 126.1 L296.3 126.1 L297.5 126.0 L298.7 126.0 L299.9 126.0 L301.1 125.9 L302.3 125.9 L303.5 125.8 L304.8 125.7 L306.0 125.7 L307.2 125.6 L308.4 125.5 L309.6 125.5 L310.8 125.4 L312.0 125.3 L313.2 125.2 L314.4 125.1 L315.6 125.1 L316.8 125.0 L318.0 124.9 L319.2 124.8 L320.5 124.7 L321.7 124.6 L322.9 124.5 L324.1 124.4 L325.3 124.3 L326.5 124.2 L327.7 124.0 L328.9 123.9 L330.1 123.8 L331.3 123.7 L332.5 123.6 L333.8 123.4 L335.0 123.3 L336.2 123.2 L337.4 123.0 L338.6 122.9 L339.8 122.8 L341.0 122.6 L342.2 122.5 L343.4 122.4 L344.6 122.2 L345.8 122.1 L347.0 121.9 L348.2 121.8 L349.5 121.6 L350.7 121.5 L351.9 121.3 L353.1 121.1 L354.3 121.0 L355.5 120.8 L356.7 120.7 L357.9 120.5 L359.1 120.3 L360.3 120.2 L361.5 120.0 L362.8 119.8 L364.0 119.6 L365.2 119.5 L366.4 119.3 L367.6 119.1 L368.8 118.9 L370.0 118.8 L371.2 118.6 L372.4 118.4 L373.6 118.2 L374.8 118.0 L376.0 117.8 L377.2 117.6 L378.5 117.4 L379.7 117.2 L380.9 117.1 L382.1 116.9 L383.3 116.7 L384.5 116.5 L385.7 116.3 L386.9 116.1 L388.1 115.9 L389.3 115.7 L390.5 115.4 L391.8 115.2 L393.0 115.0 L394.2 114.8 L395.4 114.6 L396.6 114.4 L397.8 114.2 L399.0 114.0 L400.2 113.8 L401.4 113.5 L402.6 113.3 L403.8 113.1 L405.0 112.9 L406.2 112.7 L407.5 112.5 L408.7 112.2 L409.9 112.0 L411.1 111.8 L412.3 111.6 L413.5 111.3 L414.7 111.1 L415.9 110.9 L417.1 110.6 L418.3 110.4 L419.5 110.2 L420.8 110.0 L422.0 109.7 L423.2 109.5 L424.4 109.3 L425.6 109.0 L426.8 108.8 L428.0 108.5 L429.2 108.3 L430.4 108.1 L431.6 107.8 L432.8 107.6 L434.0 107.3 L435.2 107.1 L436.5 106.9 L437.7 106.6 L438.9 106.4 L440.1 106.1 L441.3 105.9 L442.5 105.6 L443.7 105.4 L444.9 105.1 L446.1 104.9 L447.3 104.6 L448.5 104.4 L449.8 104.1 L451.0 103.9 L452.2 103.6 L453.4 103.4 L454.6 103.1 L455.8 102.9 L457.0 102.6 L458.2 102.3 L459.4 102.1 L460.6 101.8 L461.8 101.6 L463.0 101.3 L464.2 101.0 L465.5 100.8 L466.7 100.5 L467.9 100.3 L469.1 100.0 L470.3 99.7 L471.5 99.5 L472.7 99.2 L473.9 98.9 L475.1 98.7 L476.3 98.4 L477.5 98.1 L478.8 97.9 L480.0 97.6 L481.2 97.3 L482.4 97.1 L483.6 96.8 L484.8 96.5 L486.0 96.2 L487.2 96.0 L488.4 95.7 L489.6 95.4 L490.8 95.2 L492.0 94.9 L493.2 94.6 L494.5 94.3 L495.7 94.1 L496.9 93.8 L498.1 93.5 L499.3 93.2 L500.5 93.0 L501.7 92.7 L502.9 92.4 L504.1 92.1 L505.3 91.8 L506.5 91.6 L507.8 91.3 L509.0 91.0 L510.2 90.7 L511.4 90.4 L512.6 90.1 L513.8 89.9 L515.0 89.6 L516.2 89.3 L517.4 89.0 L518.6 88.7 L519.8 88.4 L521.0 88.2 L522.2 87.9 L523.5 87.6 L524.7 87.3 L525.9 87.0 L527.1 86.7 L528.3 86.4 L529.5 86.1 L530.7 85.9 L531.9 85.6 L533.1 85.3 L534.3 85.0 L535.5 84.7 L536.8 84.4 L538.0 84.1 L539.2 83.8 L540.4 83.5 L541.6 83.2 L542.8 82.9 L544.0 82.7 L545.2 82.4 L546.4 82.1 L547.6 81.8 L548.8 81.5 L550.0 81.2 L551.2 80.9 L552.5 80.6 L553.7 80.3 L554.9 80.0 L556.1 79.7 L557.3 79.4 L558.5 79.1 L559.7 78.8 L560.9 78.5 L562.1 78.2 L563.3 77.9 L564.5 77.6 L565.8 77.3 L567.0 77.0 L568.2 76.7 L569.4 76.4 L570.6 76.1 L571.8 75.8 L573.0 75.5 L574.2 75.2 L575.4 74.9 L576.6 74.6 L577.8 74.3 L579.0 74.0 L580.2 73.7 L581.5 73.4 L582.7 73.1 L583.9 72.8 L585.1 72.5 L586.3 72.2 L587.5 71.9 L588.7 71.6 L589.9 71.3 L591.1 71.0 L592.3 70.7 L593.5 70.3 L594.8 70.0 L596.0 69.7 L597.2 69.4 L598.4 69.1 L599.6 68.8 L600.8 68.5 L602.0 68.2 L603.2 67.9 L604.4 67.6 L605.6 67.3 L606.8 67.0 L608.0 66.6 L609.2 66.3 L610.5 66.0 L611.7 65.7 L612.9 65.4 L614.1 65.1 L615.3 64.8 L616.5 64.5 L617.7 64.2 L618.9 63.9 L620.1 63.5 L621.3 63.2 L622.5 62.9 L623.8 62.6 L625.0 62.3 L626.2 62.0 L627.4 61.7 L628.6 61.4 L629.8 61.0 L631.0 60.7 L632.2 60.4 L633.4 60.1 L634.6 59.8 L635.8 59.5 L637.0 59.2 L638.2 58.8 L639.5 58.5 L640.7 58.2 L641.9 57.9 L643.1 57.6 L644.3 57.3 L645.5 56.9 L646.7 56.6 L647.9 56.3 L649.1 56.0 L650.3 55.7 L651.5 55.4 L652.8 55.0 L654.0 54.7 L655.2 54.4 L656.4 54.1 L657.6 53.8 L658.8 53.4 L660.0 53.1" style="fill:none;stroke:var(--c-warn);stroke-width:2.2"/>
<line x1="285.0609665440988" y1="126.2563132923542" x2="285.0609665440988" y2="250" style="stroke:var(--c-warn);stroke-width:1.1;stroke-dasharray:4 3"/>
<circle cx="285.1" cy="126.3" r="5" style="fill:var(--c-warn);fill-opacity:.35;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="294.0609665440988" y="144.2563132923542" text-anchor="start" style="fill:var(--c-warn);font-size:12px;font-weight:600;font-size:11.5px">%23.6</text>
<path d="M87.2 72.8 L88.5 97.4 L89.7 115.8 L90.9 130.1 L92.1 141.4 L93.3 150.5 L94.5 158.1 L95.7 164.5 L96.9 169.9 L98.1 174.5 L99.3 178.5 L100.5 182.0 L101.8 185.1 L103.0 187.8 L104.2 190.2 L105.4 192.3 L106.6 194.3 L107.8 196.0 L109.0 197.5 L110.2 198.9 L111.4 200.1 L112.6 201.3 L113.8 202.3 L115.0 203.2 L116.2 204.1 L117.5 204.8 L118.7 205.5 L119.9 206.2 L121.1 206.7 L122.3 207.2 L123.5 207.7 L124.7 208.1 L125.9 208.5 L127.1 208.9 L128.3 209.2 L129.5 209.4 L130.8 209.7 L132.0 209.9 L133.2 210.1 L134.4 210.3 L135.6 210.4 L136.8 210.5 L138.0 210.6 L139.2 210.7 L140.4 210.8 L141.6 210.8 L142.8 210.8 L144.0 210.9 L145.2 210.9 L146.5 210.9 L147.7 210.8 L148.9 210.8 L150.1 210.8 L151.3 210.7 L152.5 210.6 L153.7 210.5 L154.9 210.5 L156.1 210.4 L157.3 210.3 L158.5 210.1 L159.8 210.0 L161.0 209.9 L162.2 209.8 L163.4 209.6 L164.6 209.5 L165.8 209.3 L167.0 209.2 L168.2 209.0 L169.4 208.8 L170.6 208.7 L171.8 208.5 L173.0 208.3 L174.2 208.1 L175.5 207.9 L176.7 207.7 L177.9 207.5 L179.1 207.3 L180.3 207.1 L181.5 206.9 L182.7 206.7 L183.9 206.4 L185.1 206.2 L186.3 206.0 L187.5 205.8 L188.8 205.5 L190.0 205.3 L191.2 205.0 L192.4 204.8 L193.6 204.6 L194.8 204.3 L196.0 204.1 L197.2 203.8 L198.4 203.6 L199.6 203.3 L200.8 203.0 L202.0 202.8 L203.2 202.5 L204.5 202.3 L205.7 202.0 L206.9 201.7 L208.1 201.4 L209.3 201.2 L210.5 200.9 L211.7 200.6 L212.9 200.4 L214.1 200.1 L215.3 199.8 L216.5 199.5 L217.8 199.2 L219.0 198.9 L220.2 198.7 L221.4 198.4 L222.6 198.1 L223.8 197.8 L225.0 197.5 L226.2 197.2 L227.4 196.9 L228.6 196.6 L229.8 196.3 L231.0 196.0 L232.2 195.7 L233.5 195.4 L234.7 195.1 L235.9 194.8 L237.1 194.5 L238.3 194.2 L239.5 193.9 L240.7 193.6 L241.9 193.3 L243.1 193.0 L244.3 192.7 L245.5 192.4 L246.8 192.1 L248.0 191.8 L249.2 191.5 L250.4 191.1 L251.6 190.8 L252.8 190.5 L254.0 190.2 L255.2 189.9 L256.4 189.6 L257.6 189.3 L258.8 188.9 L260.0 188.6 L261.2 188.3 L262.5 188.0 L263.7 187.7 L264.9 187.4 L266.1 187.0 L267.3 186.7 L268.5 186.4 L269.7 186.1 L270.9 185.8 L272.1 185.4 L273.3 185.1 L274.5 184.8 L275.8 184.5 L277.0 184.1 L278.2 183.8 L279.4 183.5 L280.6 183.2 L281.8 182.8 L283.0 182.5 L284.2 182.2 L285.4 181.8 L286.6 181.5 L287.8 181.2 L289.0 180.9 L290.2 180.5 L291.5 180.2 L292.7 179.9 L293.9 179.5 L295.1 179.2 L296.3 178.9 L297.5 178.5 L298.7 178.2 L299.9 177.9 L301.1 177.5 L302.3 177.2 L303.5 176.9 L304.8 176.5 L306.0 176.2 L307.2 175.9 L308.4 175.5 L309.6 175.2 L310.8 174.9 L312.0 174.5 L313.2 174.2 L314.4 173.9 L315.6 173.5 L316.8 173.2 L318.0 172.8 L319.2 172.5 L320.5 172.2 L321.7 171.8 L322.9 171.5 L324.1 171.2 L325.3 170.8 L326.5 170.5 L327.7 170.1 L328.9 169.8 L330.1 169.5 L331.3 169.1 L332.5 168.8 L333.8 168.4 L335.0 168.1 L336.2 167.8 L337.4 167.4 L338.6 167.1 L339.8 166.7 L341.0 166.4 L342.2 166.0 L343.4 165.7 L344.6 165.4 L345.8 165.0 L347.0 164.7 L348.2 164.3 L349.5 164.0 L350.7 163.6 L351.9 163.3 L353.1 163.0 L354.3 162.6 L355.5 162.3 L356.7 161.9 L357.9 161.6 L359.1 161.2 L360.3 160.9 L361.5 160.5 L362.8 160.2 L364.0 159.9 L365.2 159.5 L366.4 159.2 L367.6 158.8 L368.8 158.5 L370.0 158.1 L371.2 157.8 L372.4 157.4 L373.6 157.1 L374.8 156.7 L376.0 156.4 L377.2 156.0 L378.5 155.7 L379.7 155.3 L380.9 155.0 L382.1 154.7 L383.3 154.3 L384.5 154.0 L385.7 153.6 L386.9 153.3 L388.1 152.9 L389.3 152.6 L390.5 152.2 L391.8 151.9 L393.0 151.5 L394.2 151.2 L395.4 150.8 L396.6 150.5 L397.8 150.1 L399.0 149.8 L400.2 149.4 L401.4 149.1 L402.6 148.7 L403.8 148.4 L405.0 148.0 L406.2 147.7 L407.5 147.3 L408.7 147.0 L409.9 146.6 L411.1 146.3 L412.3 145.9 L413.5 145.6 L414.7 145.2 L415.9 144.9 L417.1 144.5 L418.3 144.2 L419.5 143.8 L420.8 143.5 L422.0 143.1 L423.2 142.8 L424.4 142.4 L425.6 142.1 L426.8 141.7 L428.0 141.4 L429.2 141.0 L430.4 140.7 L431.6 140.3 L432.8 139.9 L434.0 139.6 L435.2 139.2 L436.5 138.9 L437.7 138.5 L438.9 138.2 L440.1 137.8 L441.3 137.5 L442.5 137.1 L443.7 136.8 L444.9 136.4 L446.1 136.1 L447.3 135.7 L448.5 135.4 L449.8 135.0 L451.0 134.7 L452.2 134.3 L453.4 133.9 L454.6 133.6 L455.8 133.2 L457.0 132.9 L458.2 132.5 L459.4 132.2 L460.6 131.8 L461.8 131.5 L463.0 131.1 L464.2 130.8 L465.5 130.4 L466.7 130.1 L467.9 129.7 L469.1 129.3 L470.3 129.0 L471.5 128.6 L472.7 128.3 L473.9 127.9 L475.1 127.6 L476.3 127.2 L477.5 126.9 L478.8 126.5 L480.0 126.2 L481.2 125.8 L482.4 125.4 L483.6 125.1 L484.8 124.7 L486.0 124.4 L487.2 124.0 L488.4 123.7 L489.6 123.3 L490.8 123.0 L492.0 122.6 L493.2 122.2 L494.5 121.9 L495.7 121.5 L496.9 121.2 L498.1 120.8 L499.3 120.5 L500.5 120.1 L501.7 119.8 L502.9 119.4 L504.1 119.0 L505.3 118.7 L506.5 118.3 L507.8 118.0 L509.0 117.6 L510.2 117.3 L511.4 116.9 L512.6 116.5 L513.8 116.2 L515.0 115.8 L516.2 115.5 L517.4 115.1 L518.6 114.8 L519.8 114.4 L521.0 114.1 L522.2 113.7 L523.5 113.3 L524.7 113.0 L525.9 112.6 L527.1 112.3 L528.3 111.9 L529.5 111.6 L530.7 111.2 L531.9 110.8 L533.1 110.5 L534.3 110.1 L535.5 109.8 L536.8 109.4 L538.0 109.1 L539.2 108.7 L540.4 108.3 L541.6 108.0 L542.8 107.6 L544.0 107.3 L545.2 106.9 L546.4 106.6 L547.6 106.2 L548.8 105.8 L550.0 105.5 L551.2 105.1 L552.5 104.8 L553.7 104.4 L554.9 104.0 L556.1 103.7 L557.3 103.3 L558.5 103.0 L559.7 102.6 L560.9 102.3 L562.1 101.9 L563.3 101.5 L564.5 101.2 L565.8 100.8 L567.0 100.5 L568.2 100.1 L569.4 99.8 L570.6 99.4 L571.8 99.0 L573.0 98.7 L574.2 98.3 L575.4 98.0 L576.6 97.6 L577.8 97.2 L579.0 96.9 L580.2 96.5 L581.5 96.2 L582.7 95.8 L583.9 95.5 L585.1 95.1 L586.3 94.7 L587.5 94.4 L588.7 94.0 L589.9 93.7 L591.1 93.3 L592.3 92.9 L593.5 92.6 L594.8 92.2 L596.0 91.9 L597.2 91.5 L598.4 91.1 L599.6 90.8 L600.8 90.4 L602.0 90.1 L603.2 89.7 L604.4 89.4 L605.6 89.0 L606.8 88.6 L608.0 88.3 L609.2 87.9 L610.5 87.6 L611.7 87.2 L612.9 86.8 L614.1 86.5 L615.3 86.1 L616.5 85.8 L617.7 85.4 L618.9 85.0 L620.1 84.7 L621.3 84.3 L622.5 84.0 L623.8 83.6 L625.0 83.2 L626.2 82.9 L627.4 82.5 L628.6 82.2 L629.8 81.8 L631.0 81.4 L632.2 81.1 L633.4 80.7 L634.6 80.4 L635.8 80.0 L637.0 79.6 L638.2 79.3 L639.5 78.9 L640.7 78.6 L641.9 78.2 L643.1 77.9 L644.3 77.5 L645.5 77.1 L646.7 76.8 L647.9 76.4 L649.1 76.1 L650.3 75.7 L651.5 75.3 L652.8 75.0 L654.0 74.6 L655.2 74.3 L656.4 73.9 L657.6 73.5 L658.8 73.2 L660.0 72.8" style="fill:none;stroke:var(--c-success);stroke-width:2.2"/>
<line x1="144.8459713474939" y1="210.86881039375368" x2="144.8459713474939" y2="250" style="stroke:var(--c-success);stroke-width:1.1;stroke-dasharray:4 3"/>
<circle cx="144.8" cy="210.9" r="5" style="fill:var(--c-success);fill-opacity:.35;stroke:var(--c-success);stroke-width:1.5"/>
<text x="153.8459713474939" y="228.86881039375368" text-anchor="start" style="fill:var(--c-success);font-size:12px;font-weight:600;font-size:11.5px">%7.5</text>
<line x1="186" y1="48" x2="206" y2="48" style="stroke:var(--c-warn);stroke-width:2.2"/>
<text x="214" y="52" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">δ = 5 dk (senkron yazma): τ_opt ≈ 42.4 dk</text>
<line x1="186" y1="68" x2="206" y2="68" style="stroke:var(--c-success);stroke-width:2.2"/>
<text x="214" y="72" text-anchor="start" style="fill:var(--c-text);font-size:11.5px">δ = 0.5 dk (asenkron): τ_opt ≈ 13.4 dk</text>
</svg>

Asenkron checkpoint δ'yı bu kadar küçültmeyi, GPU'ları yalnızca bellek içi bir snapshot için durdurarak başarır:

<svg viewBox="0 0 720 246" role="img" aria-label="Senkron ve iki kademeli asenkron checkpoint&#x27;in zaman çizelgesi. Senkron: eğitim adımları, ardından GPU&#x27;ların beklediği yaklaşık 5 dakikalık checkpoint yazımı, sonra yeniden eğitim. Asenkron: eğitim adımları, ardından PCIe DMA ile host RAM&#x27;e kopyalanan kısa bir bellek içi snapshot; eğitim hemen sonraki forward GEMM&#x27;lerle devam ederken bir arka plan süreci snapshot&#x27;ı host RAM&#x27;den Lustre depolamaya paralel olarak yazar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ac-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="ac-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-success)"/></marker>
</defs>
<text x="32" y="30" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Senkron: δ ≈ 5 dk</text>
<text x="32" y="62" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GPU&#x27;lar</text>
<rect x="180" y="44" width="120" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="240" y="62" text-anchor="middle" style="fill:var(--c-text);font-size:12px">eğitim adımları</text>
<rect x="304" y="44" width="216" height="28" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="412" y="62" text-anchor="middle" style="fill:var(--c-text);font-size:12px">checkpoint yazımı · GPU&#x27;lar bekler</text>
<rect x="524" y="44" width="164" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="606" y="62" text-anchor="middle" style="fill:var(--c-text);font-size:12px">eğitim adımları</text>
<text x="32" y="110" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Asenkron, iki kademe: δ ≈ 0.5 dk</text>
<text x="32" y="138" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GPU&#x27;lar</text>
<text x="32" y="194" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">arka plan</text>
<rect x="180" y="120" width="120" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="240" y="138" text-anchor="middle" style="fill:var(--c-text);font-size:12px">eğitim adımları</text>
<rect x="304" y="120" width="34" height="28" rx="5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="321" y="112" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">snapshot</text>
<rect x="342" y="120" width="346" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="515" y="138" text-anchor="middle" style="fill:var(--c-text);font-size:12px">eğitim anında devam eder (forward GEMM)</text>
<path d="M321 148 V190 H366" marker-end="url(#ac-g)" style="fill:none;stroke:var(--c-success);stroke-width:1.5;stroke-dasharray:5 4"/>
<rect x="370" y="176" width="280" height="28" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.2"/>
<text x="510" y="194" text-anchor="middle" style="fill:var(--c-text);font-size:12px">host RAM → Lustre (arka plan süreci)</text>
<line x1="180" y1="226" x2="686" y2="226" marker-end="url(#ac-arr)" style="stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="180" y="241" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">zaman →</text>
</svg>

### 5.4. Otomatik karantina döngüsü ve Silent Data Corruption

Arıza anında tüm kümenin kilitlenmesini önlemek için **Düğüm Karantina Döngüsü (Quarantine Loop)** işletilir:
1. DCGM Xid veya NCCL gecikme anomalisinden arızalı düğüm saniyeler içinde izole edilir.
2. Yedek sıcak sunucu (hot-spare) kümeye sokulur; son temiz checkpoint'ten eğitim 2–3 dakikada devam eder.
3. Arızalı düğüm triage testlerine (DCGM, GEMM, NCCL) sokularak onarıma gönderilir.

Döngünün tamamı:

<svg viewBox="0 0 720 196" role="img" aria-label="Üç adımlı otomatik karantina döngüsü. Bir: DCGM Xid olayları veya NCCL gecikme anomalileri bir düğümü işaretler; düğüm saniyeler içinde izole edilir. İki: yedek sıcak sunucu (hot-spare) işe katılır ve eğitim son temiz checkpoint&#x27;ten 2–3 dakikada devam eder. Üç: izole edilen düğüm DCGM, GEMM ve NCCL triage testlerinden geçer ve onarıma gider. Kesikli dönüş oku, onarılan düğümün yedek havuzuna geri döndüğünü gösterir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="qu-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="qu-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="32" y="30" width="192" height="92" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="128.0" y="56.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">1. Algıla ve izole et</text>
<text x="128.0" y="72.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">DCGM Xid / NCCL gecikme</text>
<text x="128.0" y="88.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">anomalisi → düğüm saniyeler</text>
<text x="128.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">içinde çıkarılır</text>
<line x1="224" y1="76" x2="262" y2="76" marker-end="url(#qu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="264" y="30" width="192" height="92" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360.0" y="56.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">2. Hot-spare devreye girer</text>
<text x="360.0" y="72.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">son temiz checkpoint&#x27;ten</text>
<text x="360.0" y="88.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">eğitim 2–3 dakikada</text>
<text x="360.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">devam eder</text>
<line x1="456" y1="76" x2="494" y2="76" marker-end="url(#qu-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="496" y="30" width="192" height="92" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="592.0" y="56.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">3. Triage ve onarım</text>
<text x="592.0" y="72.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">DCGM, GEMM, NCCL</text>
<text x="592.0" y="88.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">test paketleri</text>
<text x="592.0" y="104.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">→ donanım onarımı</text>
<path d="M592 122 V158 H360 V126" marker-end="url(#qu-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="476" y="176" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">onarılan düğüm yedek havuzuna döner</text>
</svg>

**Silent Data Corruption (SDC):** Bir SM register'ında sessizce bir bit ters döner ($0 \to 1$). Ne CUDA ne işletim sistemi hata verir; ancak loss aniden patlar (NaN loss) veya model saçmalamaya başlar. Düzenli deterministik canary GEMM checksum testleri ile bu hayalet arızalar avlanır.

---

## 6. O meşhur soru: %38 MFU ve kayıp %62 nerede?

1.024 adet H100 GPU ile eğitim yapan bir ekip **%38 Model FLOPs Utilization (MFU)** elde ettiğinde yöneticiler sorar: *"Geriye kalan %62 nerede? GPU'ların %62'si kapalı mı?"*

### 6.1. nvidia-smi kullanımının aldatıcılığı

`nvidia-smi` çıktısındaki %100 utilization değeri bir hız metriği değildir; sadece GPU çekirdeklerinin uykuda olmadığını belirten kaba bir örnekleme varlık sinyalidir. Bellek bekleyen veya verimsiz çalışan bir GPU da %100 görünür. Gerçek algoritmik verimlilik yalnızca **MFU** ile ölçülür.

### 6.2. Altı fiziksel kayıp kalemi: Eksik %62'nin muhasebesi

1.024 H100 GPU'nun teorik $1.014\text{ EFLOP/s}$ tepe gücünden %38 faydalı iş ($0.385\text{ EFLOP/s}$) çıkarıldığında, kalan %62 şu 6 fiziksel kalemde kuruşu kuruşuna muhasebeleştirilir:

<svg viewBox="0 0 720 284" role="img" aria-label="1.024 H100&#x27;de %38 MFU&#x27;da kalan %62&#x27;nin nereye gittiği. %100&#x27;lük bir çubuk %38 faydalı model FLOP&#x27;una ve toplamı %62 olan altı kayıp kalemine bölünür: örtülemeyen TP AllReduce ve DP senkronizasyon bariyerinden %14 kolektif ağ; bellek kazanmak için tekrarlanan aktivasyon GEMM&#x27;lerinden %12 recomputation; Tensor Core&#x27;ların HBM&#x27;yi beklediği LayerNorm, Softmax ve RoPE gibi çekirdeklerden %11 memory-bound kayıp; 1F1B stage doldurma ve boşaltmasından %10 pipeline bubble; checkpoint duraklamaları ve straggler&#x27;lardan %8 jitter ve I/O; matris boyutlarının 16x16x16 Tensor Core karolarına tam uymamasından %7 tiling kaybı." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="32" y="30" width="247.27999999999997" height="32" rx="4" style="fill:var(--c-accent);fill-opacity:.25;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="156.64" y="51" text-anchor="middle" style="fill:var(--c-text);font-size:12px">%38 faydalı model FLOP&#x27;u (MFU)</text>
<rect x="281.28" y="30" width="89.83999999999999" height="32" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="327.2" y="50" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%14</text>
<rect x="373.11999999999995" y="30" width="76.72" height="32" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="412.47999999999996" y="50" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%12</text>
<rect x="451.8399999999999" y="30" width="70.16" height="32" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="487.9199999999999" y="50" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%11</text>
<rect x="523.9999999999999" y="30" width="63.599999999999994" height="32" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="556.7999999999998" y="50" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%10</text>
<rect x="589.5999999999999" y="30" width="50.48" height="32" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="615.8399999999999" y="50" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%8</text>
<rect x="642.0799999999999" y="30" width="43.919999999999995" height="32" rx="4" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="665.04" y="50" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">%7</text>
<path d="M281.3 68 V74 H686.0 V68" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="484.63999999999993" y="90" text-anchor="middle" style="fill:var(--c-danger);font-size:12px;font-weight:600">%62 kayıp: altı fiziksel kalem</text>
<text x="32" y="126" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Kolektif ağ</text>
<rect x="176" y="113" width="112" height="17" rx="3" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="294" y="126" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">%14</text>
<text x="334" y="126" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Örtülemeyen TP AllReduce ve DP senkronizasyon bariyeri</text>
<text x="32" y="154" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Recomputation</text>
<rect x="176" y="141" width="96" height="17" rx="3" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="278" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">%12</text>
<text x="334" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Bellek için tekrarlanan aktivasyon GEMM&#x27;leri</text>
<text x="32" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Memory-bound</text>
<rect x="176" y="169" width="88" height="17" rx="3" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="270" y="182" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">%11</text>
<text x="334" y="182" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">LayerNorm, Softmax, RoPE: Tensor Core HBM&#x27;yi bekler</text>
<text x="32" y="210" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">PP bubble</text>
<rect x="176" y="197" width="80" height="17" rx="3" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="262" y="210" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">%10</text>
<text x="334" y="210" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">1F1B pipeline stage doldurma/boşaltma kabarcığı</text>
<text x="32" y="238" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Jitter ve I/O</text>
<rect x="176" y="225" width="64" height="17" rx="3" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="246" y="238" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">%8</text>
<text x="334" y="238" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Checkpoint yazma duraklamaları ve straggler gecikmesi</text>
<text x="32" y="266" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Tiling kaybı</text>
<rect x="176" y="253" width="56" height="17" rx="3" style="fill:var(--c-danger);fill-opacity:.14;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="238" y="266" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px;font-family:var(--font-mono)">%7</text>
<text x="334" y="266" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">Matrislerin 16x16x16 karolara tam uymaması</text>
</svg>

Bu dağılım görüldüğünde %38 MFU'nun aslında son derece optimize edilmiş bir dünya standardı olduğu anlaşılır.

### 6.3. MFU ve HFU arasındaki recomputation ayrımı

- **MFU (Model FLOPs):** Yalnızca faydalı matematiksel modele dönüşen operasyonları sayar ($6P\text{ FLOP/token}$).
- **HFU (Hardware FLOPs):** Donanımın fiziksel olarak çarptığı tüm matrisleri sayar.
- Eğer bellek kazanmak için aktivasyonları backward sırasında baştan hesaplıyorsanız (Activation Recomputation), donanım $8P$ işlem yapar. HFU %50 görünürken MFU %38'de kalır. Aradaki fark, bellek yetersizliğine ödenen matematiksel vergidir.

---

## 7. Tesis, trafo ve FinOps: Megavatlar ve fatura

Yapay zekâ süper bilgisayarları en sonunda fiziksel veri merkezinin betonuna ve trafosuna dayanır.

### 7.1. Raf başına 40.8 kW ısı ve doğrudan yongaya sıvı soğutma (DLC)

Tek bir DGX H100 sunucusu **$10.2\text{ kW}$** elektrik çeker. 4 sunuculu tek bir kabin:

$$
\text{Kabin Gücü} = 4 \times 10.2\text{ kW} = 40.8\text{ kW/raf}
$$

Geleneksel kurumsal veri merkezleri kabin başına $5 - 10\text{ kW}$ soğutabilecek şekilde tasarlanmıştır. $40\text{ kW}$'lık bir ısı fırtınasını hava ile soğutmak imkânsızdır; fanlar son devirde dönse bile GPU'lar termal kısıtlamaya girer.
Yeni nesil AI platformları, sıvıyı doğrudan silikon blokların üzerinden dolaştıran **Doğrudan Yongaya Sıvı Soğutma (Direct-to-Chip Liquid Cooling - DLC)** kullanmak zorundadır.

### 7.2. Veri merkezi PUE çarpanı ve trafo kapasitesi

1.024 GPU'luk bir küme yalnızca hesaplama için yaklaşık $1.5\text{ MW}$ elektrik çeker.
Veri merkezinin Güç Kullanım Verimliliği (PUE) 1.35 ise:

$$
\text{Tesis Gücü} = 1.5\text{ MW} \times 1.35 = 2.025\text{ MW}
$$

Her 1 MW GPU gücü için 0.35 MW soğutma enerjisi harcanır. 3 yıllık bir kullanım ömründe harcanan elektrik ve tesis faturası, GPU'ların satın alma bedelini (CapEx) aşabilir!

### 7.3. Faydalı metrikler: Joule başına token ve dolar başına token

FinOps ekiplerinin baktığı geleneksel "GPU-Saat" metriği kör bir göstergedir. İki ekip aynı GPU saatini kiralayıp aynı faturayı ödeyebilir; ancak biri %20 MFU ile yarım model eğitirken, diğeri %40 MFU ile tam model eğitir.

Doğru metrikler:
1. **Dolar Başına Faydalı Token:** Harcanan toplam TCO bütçesinin üretilen token sayısına oranı.
2. **Joule Başına Token (Energy Efficiency):** Tüketilen elektrik enerjisi başına üretilen faydalı token sayısı. MFU'yu %25'ten %40'a çıkarmak, şirketin elektrik faturasını doğrudan %37.5 düşürür.

### 7.4. 70B model referans tasarımı: Serving, fine-tuning ve pretraining

Tüm bu sistem katmanları kurumsal ölçekte üç farklı mimari doğurur:

<svg viewBox="0 0 720 292" role="img" aria-label="70B model için üç donanım imzası. Serving: toplam 320 GB&#x27;lık 4 H100 SXM5, NVSwitch üzerinde tensor parallel 4; 140 GB ağırlık ve 160 GB KV cache tutar, düşük TTFT ve token arası gecikme için ayarlıdır. Fine-tuning: 2 DGX düğümünde 16 H100; düğüm içinde TP 8, düğümler arasında FSDP; 1.28 TB HBM içinde 1.12 TB AdamW durumu, sıfır CPU offload, haftalık ince ayar döngüleri. Pretraining: 64 DGX düğümünde 512 H100; TP 8 çarpı PP 4 çarpı DP 16; 8-rail InfiniBand Clos, asenkron checkpoint ve %40&#x27;ın üzerinde MFU hedefi." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="32" y="8" width="208" height="276" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="46" y="32" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">SERVING</text>
<text x="46" y="50" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">çıkarım</text>
<rect x="81" y="74" width="22" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="109" y="74" width="22" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="137" y="74" width="22" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="165" y="74" width="22" height="30" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="46" y="124" width="180" height="12" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="46" y="124" width="78.75" height="12" rx="3" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="124.75" y="124" width="90.0" height="12" rx="3" style="fill:var(--c-warn);fill-opacity:.45"/>
<text x="46" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">140 + 160 GB / 320 GB</text>
<text x="46" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">4 × H100 SXM5 (320 GB)</text>
<text x="46" y="202" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP = 4, NVSwitch</text>
<text x="46" y="222" text-anchor="start" style="fill:var(--c-text);font-size:12px">140 GB model ağırlığı</text>
<text x="46" y="242" text-anchor="start" style="fill:var(--c-text);font-size:12px">160 GB KV cache</text>
<text x="46" y="262" text-anchor="start" style="fill:var(--c-text);font-size:12px">Düşük gecikme (TTFT/ITL)</text>
<rect x="256" y="8" width="208" height="276" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="270" y="32" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">FINE-TUNING</text>
<text x="270" y="50" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ince ayar</text>
<rect x="290" y="68" width="64" height="42" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="297.0" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="310.5" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="324.0" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="337.5" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="297.0" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="310.5" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="324.0" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="337.5" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="366" y="68" width="64" height="42" rx="5" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="373.0" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="386.5" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="400.0" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="413.5" y="75" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="373.0" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="386.5" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="400.0" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="413.5" y="90" width="10" height="12" rx="2" style="fill:var(--c-success);fill-opacity:.3;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="270" y="124" width="180" height="12" rx="3" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="270" y="124" width="157.5" height="12" rx="3" style="fill:var(--c-success);fill-opacity:.45"/>
<text x="270" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1.12 TB / 1.28 TB HBM</text>
<text x="270" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">16 × H100 (2 × DGX)</text>
<text x="270" y="202" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP = 8 (iç) + FSDP (dış)</text>
<text x="270" y="222" text-anchor="start" style="fill:var(--c-text);font-size:12px">1.12 TB AdamW durumu</text>
<text x="270" y="242" text-anchor="start" style="fill:var(--c-text);font-size:12px">Sıfır CPU offload</text>
<text x="270" y="262" text-anchor="start" style="fill:var(--c-text);font-size:12px">Haftalık ince ayar</text>
<rect x="480" y="8" width="208" height="276" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="494" y="32" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700;letter-spacing:.06em">PRETRAINING</text>
<text x="494" y="50" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ön eğitim</text>
<rect x="495.0" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="506.3" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="517.6" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="528.9" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="540.2" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="551.5" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="562.8" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="574.1" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="585.4" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="596.7" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="608.0" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="619.3" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="630.6" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="641.9" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="653.2" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="664.5" y="70" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="495.0" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="506.3" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="517.6" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="528.9" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="540.2" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="551.5" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="562.8" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="574.1" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="585.4" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="596.7" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="608.0" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="619.3" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="630.6" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="641.9" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="653.2" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="664.5" y="81" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="495.0" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="506.3" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="517.6" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="528.9" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="540.2" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="551.5" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="562.8" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="574.1" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="585.4" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="596.7" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="608.0" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="619.3" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="630.6" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="641.9" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="653.2" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="664.5" y="92" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="495.0" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="506.3" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="517.6" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="528.9" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="540.2" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="551.5" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="562.8" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="574.1" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="585.4" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="596.7" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="608.0" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="619.3" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="630.6" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="641.9" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="653.2" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="664.5" y="103" width="8.5" height="8.5" rx="1.5" style="fill:var(--c-warn);fill-opacity:.3;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="494" y="154" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">8 × 4 × 16 = 512 GPU</text>
<text x="494" y="182" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">512 × H100 (64 × DGX)</text>
<text x="494" y="202" text-anchor="start" style="fill:var(--c-text);font-size:12px">TP = 8 + PP = 4 + DP = 16</text>
<text x="494" y="222" text-anchor="start" style="fill:var(--c-text);font-size:12px">8-rail InfiniBand Clos</text>
<text x="494" y="242" text-anchor="start" style="fill:var(--c-text);font-size:12px">Asenkron checkpoint</text>
<text x="494" y="262" text-anchor="start" style="fill:var(--c-text);font-size:12px">%40+ MFU hedefi</text>
</svg>

---

## Bütün hikâye altı satırda

- Bir yapay zekâ modeli soyut bir bulutta değil; silikon SM'ler, HBM istifleri, 400G lazerler ve megavatlık trafolar üzerinde yaşar.
- Tensor Core'lar 990 TFLOP/s hesaplayabilir ama HBM $3.35\text{ TB/s}$ aktarabildiği için decode fazı bellek duvarına toslar.
- Tek GPU'ya sığmayan modeller 16 bayt kuralıyla hesaplanır ve 4D paralellik ile NVLink ve InfiniBand kumaşlarına paylaştırılır.
- Binlerce GPU'luk sistemlerde arıza kaçınılmaz bir çalışma durumudur; Young–Daly formülü ve karantina döngüsüyle yönetilir.
- %38 MFU elde edilen bir kümede kayıp %62; bellek kısıtlı katmanlar, kolektif ağ kuyrukları, pipeline kabarcıkları ve recomputation'dır.
- AI platformlarının nihai sınırı GPU sayısı değil, raf başına 40 kW ısıyı tahliye eden sıvı soğutma ve Joule başına token maliyetidir.

---

## Terimler sözlüğü

- **SM (Streaming Multiprocessor):** GPU üzerindeki bağımsız hesaplama çekirdeği; warp scheduler, register dosyası ve Tensor Core'ları barındırır.
- **Roofline Modeli:** Bir algoritmanın hesaplama gücüyle mi yoksa bellek bant genişliğiyle mi sınırlandığını gösteren teorik grafik.
- **Prefill ve Decode:** LLM çıkarımında prompt'un paralel hesaplandığı (GEMM) ve kelimelerin tek tek seri üretildiği (GEMV) iki zıt faz.
- **KV Cache:** Geçmiş token'ların Key ve Value matrislerini saklayarak tekrarlı hesaplamayı önleyen ancak HBM tüketen dinamik arabellek.
- **NVSwitch:** Sunucu içindeki 8 GPU'yu çift yönlü 900 GB/s hızla tam bağlı bağlayan anahtarlama yongası.
- **Rail-Optimized:** Kümedeki aynı indeksli NIC'leri aynı leaf switch'e bağlayarak iletişimi tek atlamada tutan Clos ağ mimarisi.
- **4D Paralellik:** Modelin Tensor (TP), Pipeline (PP), Context (CP), Expert (EP) ve Data (DP/FSDP) eksenlerinde eşzamanlı bölünmesi.
- **MFU (Model FLOPs Utilization):** Gerçekleşen faydalı algoritmik hesaplamanın donanımın teorik tepe gücüne bölünmesiyle bulunan gerçek verimlilik.
- **Young–Daly Kuralı:** Checkpoint yazma süresi ve küme MTBF'sine bağlı olarak optimum kayıt sıklığını belirleyen denklem.
- **DLC (Direct-to-Chip Liquid Cooling):** Raf başına 40 kW'ı aşan yüksek yoğunluklu AI sunucularını soğutmak için sıvıyı doğrudan yongalara ileten sistem.

---

## Daha derine inmek için: 5 bölümlük altyapı serisi

Bu panoramik yazıda büyük resmini gördüğünüz tüm mimari bileşenlerin kod seviyesinde türetimlerini, NCCL analizlerini ve vaka incelemelerini özel olarak işlediğimiz 5 bölümlük derinlemesine serimiz:

1. [GPU Mimarisi ve LLM Ölçekleme — SM, Tensör Çekirdekleri ve Bellek Duvarı](post.html?slug=gpu-mimarisi-ve-llm-olcekleme)
2. [LLM GPU'yu Nasıl Kullanır? — Training, Prefill ve Decode](post.html?slug=gpu-is-yukleri-ve-uretim-cikarimi)
3. [Çoklu GPU Ağları ve Dağıtık Eğitim — NVLink, InfiniBand ve 4D Paralellik](post.html?slug=coklu-gpu-aglari-ve-dagitik-egitim)
4. [AI Kümesi Neden Yavaşlar? — Depolama, Arızalar ve MFU](post.html?slug=gpudan-ai-super-bilgisayarina)
5. [Kurumsal AI Platform Mimarisi — Donanım, Maliyet ve Operasyon](post.html?slug=kurumsal-ai-platform-mimarisi)
