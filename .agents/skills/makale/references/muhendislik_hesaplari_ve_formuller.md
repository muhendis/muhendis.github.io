# Mühendislik Hesapları, Bellek ve Donanım Formülleri Rehberi

Bu rehber, blogdaki teknik makalelerde yer alan VRAM, FLOP, KV Cache, donanım bant genişliği ve eğitim/çıkarım hesaplamalarının temel formüllerini ve doğrulama yöntemlerini içerir.

---

## 1. VRAM ve Bellek Faturası Hesapları

### 1.1. Model Ağırlıkları (Inference)
Bir modelin yalnızca ağırlıklarının ekran kartında kapladığı yer:

$$\text{Ağırlık VRAM} = P \times \text{Bayt/Parametre}$$

- **FP32:** 4 bayt/parametre ($1\text{B} \approx 4\text{ GB}$)
- **FP16 / BF16:** 2 bayt/parametre ($1\text{B} \approx 2\text{ GB}$, $8\text{B} \approx 16\text{ GB}$)
- **FP8 / INT8:** 1 bayt/parametre ($8\text{B} \approx 8\text{ GB}$)
- **INT4 (AWQ / GPTQ):** 0.5 bayt/parametre ($8\text{B} \approx 4\text{ GB}$)

### 1.2. Eğitimde Parametre ve Optimizasyon Belleği (AdamW)
Karma duyarlıklı (Mixed Precision FP16/BF16) eğitimde parametre başına bellek tüketimi:

| Bileşen | Hassasiyet | Bayt / Parametre | 8B Model İçin Bellek |
| :--- | :--- | :--- | :--- |
| **Model Ağırlıkları** | FP16/BF16 | 2 bayt | 16 GB |
| **Gradyanlar** | FP16/BF16 | 2 bayt | 16 GB |
| **AdamW 1. Moment ($m_t$)** | FP32 | 4 bayt | 32 GB |
| **AdamW 2. Moment ($v_t$)** | FP32 | 4 bayt | 32 GB |
| **Master Ağırlıklar (FP32)** | FP32 | 4 bayt | 32 GB |
| **TOPLAM (Statik Bellek)** | - | **16 bayt** (Master ile 20B) | **128 – 160 GB** |

> [!NOTE]
> Bu toplama dinamik aktivasyon belleği (activation memory) ve geçici tamponlar (scratch buffers) dâhil değildir. Aktivasyon belleği dizi uzunluğu ($L$), batch boyutu ve katman sayısıyla doğru orantılıdır; *Activation Checkpointing (Gradient Checkpointing)* ile hesap süresi %30 artırılarak bellek dramatik biçimde düşürülür.

### 1.3. KV Cache Bellek Tüketimi
Auto-regressive çıkarımda KV Cache her token ile büyür:

$$\text{KV Cache (bayt)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_k \times L \times B \times \text{hassasiyet\_bayt}$$

- **Çarpan 2:** $K$ ve $V$ tensörleri için.
- $n_{\text{layers}}$: Katman sayısı (örn. LLaMA 3-8B için 32).
- $n_{\text{kv\_heads}}$: Key-Value kafa sayısı (MHA'da $h$ ile eşit; GQA'da $h$'tan küçüktür, örn. LLaMA 3-8B'de 8).
- $d_k$: Kafa boyutu (head dimension, genelde 128).
- $L$: Context uzunluğu (token sayısı).
- $B$: Batch boyutu.
- $\text{hassasiyet\_bayt}$: FP16 için 2, FP8 için 1.

**Örnek Hesap (LLaMA 3.1-8B, 128k context, FP16, Tek İstek):**
- Katman: 32, KV kafa: 8, $d_k$: 128, $L$: 131.072 ($128 \times 1024$), Bayt: 2
- $\text{Token başına KV bayt} = 2 \times 32 \times 8 \times 128 \times 2 = 131.072 \text{ bayt} = 128\text{ KB/token}$.
- $128\text{k token}$ için KV Cache = $128\text{ KB} \times 131.072 \approx \mathbf{16.384\text{ MB}} = \mathbf{16\text{ GB}}$!
- GQA olmasaydı (MHA, 32 kafa): $16 \times 4 = \mathbf{64\text{ GB}}$ olacaktı.

---

## 2. Prefill vs. Decode: Donanımsal Ayrım (Roofline Modeli)

| Aşama | İşlem Türü | Kısıt (Bottleneck) | Aritmetik Yoğunluk (FLOP / Bayt) | Donanım Davranışı |
| :--- | :--- | :--- | :--- | :--- |
| **Prefill** | Matris-Matris (GEMM) | **Compute-Bound** | Çok Yüksek ($>100$) | GPU Tensör Çekirdekleri %100'e yakın doyar; gecikme prompt uzunluğunun karesiyle/küpüyle orantılıdır. |
| **Decode** | Matris-Vektör (GEMV) | **Memory-Bandwidth-Bound** | Çok Düşük ($\approx 1 - 2$) | Çekirdekler boştadır, GPU bellek bant genişliği (HBM) tıkanır; her token için modelin tüm ağırlıkları belleğe akıtılır. |

### Çıkarım Maliyeti (FLOP)
- **Token başına Forward Pass:** $\approx 2P$ FLOP (Parametre başına 1 çarpma, 1 toplama).
- **8B modelde 1 token decode:** $2 \times 8 \times 10^9 = 16\text{ GFLOP}$.
- H100 (SXM5) FP16 hesap gücü $\approx 1.000\text{ TFLOP/s}$ olduğuna göre teorik olarak 16 GFLOP milisaniyenin yüzde birinde biter. Ancak 8B ağırlık (16 GB) H100 bellek bant genişliğinden (3.35 TB/s) okunurken en az:
  $$\frac{16\text{ GB}}{3.350\text{ GB/s}} \approx \mathbf{4,7\text{ ms}}$$
  sürer. Bu nedenle decode'un efendisi **hesaplama hızı değil, bellek bant genişliğidir**.

---

## 3. Self-Attention Matematiksel Sabitleri

### 3.1. $\sqrt{d_k}$ Normalizasyonunun İspatı
- Bağımsız $q_i, k_i \sim \mathcal{N}(0, 1)$ rastgele değişkenleri için:
  $$\mathbb{E}[q_i k_i] = 0, \quad \text{Var}(q_i k_i) = \text{Var}(q_i)\text{Var}(k_i) + \mathbb{E}[q_i]^2\text{Var}(k_i) + \mathbb{E}[k_i]^2\text{Var}(q_i) = 1 \times 1 + 0 + 0 = 1$$
- $d_k$ boyutlu skalar çarpım:
  $$S = \sum_{i=1}^{d_k} q_i k_i \implies \text{Var}(S) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$
- Standart sapma $\sigma = \sqrt{d_k}$ olur. $d_k = 128$ olduğunda standart sapma $\approx 11,3$ olur.
- Softmax girdileri 11 gibi büyük sayılara ulaştığında, softmax bir tek girdiye $1.0$, diğerlerine $0.0$ verir (doygunluk / saturation).
- $\frac{S}{\sqrt{d_k}}$ bölünmesi varyansı tekrar $\mathbf{1}$'e sabitler; gradyanların sıfırlanmasını (vanishing gradient) önler.

### 3.2. RoPE Göreli Konum Formülü
- İki pozisyon $m$ ve $n$ için $Q$ ve $K$ döner matrislerle çarpılır:
  $$\tilde{q}_m = \mathbf{R}_{\Theta, m} q_m, \quad \tilde{k}_n = \mathbf{R}_{\Theta, n} k_n$$
- İç çarpım:
  $$\langle \tilde{q}_m, \tilde{k}_n \rangle = q_m^T \mathbf{R}_{\Theta, m}^T \mathbf{R}_{\Theta, n} k_n = q_m^T \mathbf{R}_{\Theta, n - m} k_n$$
- Dikkat skoru yalnızca mutlak koordinatlara ($m, n$) değil, aralarındaki **göreli mesafeye ($n - m$)** bağlı kalır!
