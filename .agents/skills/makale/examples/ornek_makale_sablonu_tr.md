Giriş paragrafı 1 (Canlı Zihinsel Sahne / "Hayal Edin"): Okuru doğrudan fiziksel bir sahnenin içine alın. Örnek: Bir veri merkezine girdiğinizi hayal edin. Karşınızda sıra sıra sunucular, her birinin içinde binlerce paralel işlemi gerçekleştirebilen GPU'lar var. Toplamda 1.024 NVIDIA H100 satın aldınız... İlk ölçümler geliyor: MFU = %38. Birisi soruyor: "Bu kadar GPU aldık, peki kalan %62 nerede?" Bu çok iyi bir soru. Fakat cevabı "GPU'lar boş" kadar basit değil.

Giriş paragrafı 2 (Ölçülebilir Gerilim ve Sistemik Neden): Mühendislik paradoksunu donanımsal ve sistemsel gerçekle bağlayın. Bir GPU hesaplama yaparken bellekten veri bekleyebilir; bir diğeri ağdan gradyan bekler; depolama checkpoint yazarken tüm küme kilitlenir. Kâğıt üzerindeki tepe güç ile fiili verim arasındaki uçurum bir aksaklık değil, sistemik tasarım sonucudur.

Giriş paragrafı 3 (Temel Fikir / Pusula Cümlesi): Konunun zihinsel omurgasını tek bir net cümleyle kurun: "Bu yazının temel fikri şu: Büyük ölçekli yapay zekâ, yalnızca hesaplama gücü problemi değildir. Aynı zamanda veriyi doğru yere, doğru zamanda, doğru güvenlik sınırları içinde ve en düşük maliyetle ulaştırma problemidir."

Giriş paragrafı 4 (Okuma Pusulası & Birlikte Yolculuk): Okurla yol arkadaşı dili kurun: "Nasıl okuyabilirsiniz? Bir kavram ilk kez geçtiğinde önce ne işe yaradığını ve basit bir benzetmesini vereceğiz. Ardından teknik tanımını, donanımsal mekanizmasını ve küçük bir hesaplamasını göstereceğiz. Tablolar ve sayılar ezberlemek için değil; sistemin davranışını zihinde canlandırabilmek için var."

**Bu yazıda**

- [1. Üç fiziksel duvar: Compute, Network, Storage](#1-üç-fiziksel-duvar-compute-network-storage)
- [2. Donanım mekanizması: SM, Tensor Core ve veri akışı](#2-donanım-mekanizması-sm-tensor-core-ve-veri-akışı)
- [3. Bellek hiyerarşisi ve Roofline sınırı](#3-bellek-hiyerarşisi-ve-roofline-sınırı)
- [4. Üç mühendislik gözü: Training, Prefill ve Decode](#4-üç-mühendislik-gözü-training-prefill-ve-decode)
- [5. PyTorch ile adım adım tam doğrulama](#5-pytorch-ile-adım-adım-tam-doğrulama)
- [Okuyucu artık hangi sorulara kendi başına yanıt verebilir?](#okuyucu-artık-hangi-sorulara-kendi-başına-yanıt-verebilir)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Kaynaklar ve doğrulama bağlantıları](#kaynaklar-ve-doğrulama-bağlantıları)

---

## 1. Üç fiziksel duvar: Compute, Network, Storage

### 1. Sezgi (Gözde Canlanan Sahne)
Bir restoran zinciri düşünelim. Yönetim, üretimi artırmak için her şubeye iki kat fazla aşçı gönderiyor. Ama şubelerdeki malzeme depoları aynı; dağıtım kamyonları aynı; mutfak kapılarından malzeme geçişi de aynı. Acaba iki kat fazla yemek üretilebilir mi? Muhtemelen hayır. Çünkü bu sistemde darboğaz artık aşçı sayısı değil...

### 2. Mekanizma (Fiziksel Donanım / Yazılım Gerçeği)
Bir yapay zekâ kümesini restoran zinciri gibi düşünün:
- **Streaming Multiprocessor (SM):** Yemek hazırlayan küçük mutfak istasyonu.
- **Tensor Core:** Çok sayıda benzer matematiksel işi aynı anda yapan özel makine.
- **Register / Shared Memory:** Aşçının elinin altındaki tezgâh.
- **HBM:** Restoranın hızlı fakat sınırlı ortak deposu.
- **NVLink / NVSwitch:** Aynı binadaki mutfaklar arasındaki geniş servis koridoru.
- **InfiniBand / RoCE:** Farklı binalar arasındaki lojistik yolları.
- **Dağıtık depolama:** Hammaddenin ve işletmenin kayıtlarının tutulduğu merkez ambar.

<!-- Çizimler kesinlikle inline SVG olarak gömülür (Mermaid/ASCII yasaktır): -->
<svg viewBox="0 0 560 200" role="img" aria-label="Donanım hiyerarşisi ve veri yolları" style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
  <!-- svgkit standartlarında SVG içeriği -->
</svg>

### 3. Matematik ve Darboğaz Denklemi
Roofline kırılma noktası:

$$\alpha_{\mathrm{crit}} = \frac{C_{\mathrm{peak}}}{W_{\mathrm{mem}}} = \frac{990\ \mathrm{TFLOP/s}}{3{,}35\ \mathrm{TB/s}} \simeq 295{,}5\ \mathrm{FLOP/byte}$$

Şöyle okuyun: $C_{\mathrm{peak}}$ donanımın teorik hesaplama tavanı, $W_{\mathrm{mem}}$ bellek bant genişliği, $\alpha_{\mathrm{crit}}$ ise hesaplama birimlerini tam kapasite beslemek için gereken kritik aritmetik yoğunluk eşiğidir.

```text
Adım adım veri-hareketi hesabı (Llama 3 70B MLP, D=8192, F=28672):
B = 1   -> FLOP = 2*1*D*F,  Byte = 2*D*F -> alpha = 1,0 FLOP/byte   [Aşırı Memory-bound]
B = 128 -> FLOP = 2*128*D*F,             -> alpha = 125,5 FLOP/byte [HBM sınırında]
B = 310 -> FLOP = 2*310*D*F,             -> alpha = 295,6 FLOP/byte [Compute tavanı mümkün]
```

### 4. Sınır & Gerçeklik Payı
Bu sayı bir doğa kanunu değildir; BF16 Tensor Core ve HBM kesişimi için teorik sınırdır. FP32 veya FP8 kullanıldığında tavan değişir. Ayrıca cache reuse ve kernel fusion gibi yazılımsal optimizasyonlar veri hareketini azaltarak davranışı dönüştürebilir.

> **Öğrenme kontrolü:** Bir kernel'i hızlandırmak için 'daha çok thread' açmak isteyebilirsiniz. Önce register spill, shared memory kullanımı ve occupancy dengesini ölçmeden bunu yapmak performansı kötüleştirebilir.

---

### Ara Durak: Bandwidth ile Latency Aynı Şey Değildir
Bir kamyon aynı anda kaç koli taşıyabilir (Bandwidth - GB/s) ve kamyonun depoya gidip gelmesi kaç dakika sürer (Latency - ns)? Bir yolun sekiz şeritli olması yüksek bant genişliği, o yolun 500 kilometre olması yüksek gecikme sezgisidir. Büyük blok veri taşımada bant genişliği, bağımlı küçük işlemlerde ise gecikme belirleyicidir.

---

## 4. Üç mühendislik gözü: Training, Prefill ve Decode

| Boyut | Eğitim (Training) | Ön-Doldurma / Çıkarım (Prefill) | Kod Çözme / Çıkarım (Decode) |
| :--- | :--- | :--- | :--- |
| **İş Yükü Biçimi** | Büyük batch, GEMM yoğunluklu | Uzun prompt sekansı, GEMM | Teker teker token üretimi, GEMV |
| **Donanım Rejimi** | Compute-bound (Tensor Core) | Compute-bound (Parallel GEMM) | Memory-bandwidth-bound (<1 FLOP/bayt) |
| **Bellek Faturası** | Ağırlık + Gradyan + Optimizer (AdamW 16B/param) | Model Ağırlıkları (FP16/FP8) | Model Ağırlıkları + Büyüyen KV Cache |
| **Kritik Risk** | AllReduce senkronizasyonu, checkpoint durması | İlk Token Gecikmesi (TTFT) | Token Başına Gecikme (TPOT), Throughput |

---

## 5. PyTorch ile adım adım tam doğrulama

```python
import torch

# Elle yapılan hesabın kod seviyesinde teyidi
# Sayısal doğrulamayı çalıştırıp konsol çıktısını ekleyin
```

Konsol çıktısı:

```text
Doğrulandı: ...
```

---

### Okuyucu artık hangi sorulara kendi başına yanıt verebilir?

İlk sayfadaki 1.024 H100 ve %38 MFU örneğine geri dönelim. Bu sayı bize artık "GPU'lar kapalı" demiyor. Bir soru işareti gösteriyor ve şu sorular üzerinden teşhis yapabiliyoruz:
- Bir byte nereden geliyor, kaç kez kullanılıyor, nereye taşınıyor?
- SM içindeki bir hesaplama başka hangi bellek ya da ağ sonucunu bekliyor?
- Sistem genelinde veri, güvenilirlik ve servis hedefleri darboğazı nasıl etkiliyor?

---

## Bütün hikâye altı satırda

- Birinci ilke: Büyük ölçekli yapay zekâ yalnızca hesaplama değil; veriyi yerinde tutma ve gecikmeyi örtme sanatıdır.
- İkinci ilke: SM içindeki warp scheduler'ın yaptığı bekleme gizleme ile kümedeki 1F1B pipeline zamanlaması aynı prensiptir.
- Üçüncü ilke: Aritmetik yoğunluk donanım tavanını aşmadıkça en pahalı hızlandırıcılar bile saat döngülerini HBM kuyruğunda bekleyerek harcar.
- Dördüncü ilke: İletişim hiyerarşisi kasayı aşıp optik ağa çıktığı anda bant genişliği bir mertebe düşer; model bölüşümü bu sınıra uymak zorundadır.
- Beşinci ilke: Depolama katmanı yalnızca token değil checkpoint taşır; yetersiz G/Ç tüm hesaplama omurgasını kilitler.
- Altıncı ilke: Gümüş kurşun yoktur; en yüksek tepe değeri değil, arıza ve yoğunluk altında hedef deneyimi koruyan dengeli mimari kazanır.

---

## Terimler sözlüğü

- **Streaming Multiprocessor (SM)** — GPU içindeki bağımsız hesaplama istasyonu; warp scheduler, Tensor Core ve yerel bellekleri barındırır.
- **Model FLOPs Utilization (MFU)** — donanımın teorik tepe hesap gücüne kıyasla modelin faydalı iş için harcadığı fiili oran.
- **Latency Hiding (Gecikme Gizleme)** — bir işlemin bekleme süresinde hazır başka bir işi yürüterek donanımı dolu tutma tekniği.

---

## Kaynaklar ve doğrulama bağlantıları

- **[K1]** Birincil kaynak / orijinal makale
- **[K2]** Resmi mimari dokümantasyonu (NVIDIA / Google / Meta)
- **[Kn]** Bu blogda: [GPU Mimarisi ve LLM Ölçekleme](post.html?slug=gpu-mimarisi-ve-llm-olcekleme)
