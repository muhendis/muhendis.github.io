# İnternette Bilgi Arama ve Kaynak Doğrulama Rehberi

Bu rehber, blogdaki teknik makalelerin ihtiyaç duyduğu birincil kaynak araştırması, mimari teyidi, sayısal doğrulama ve literatür taraması süreçlerini adım adım tanımlar.

---

## 1. Birincil Kaynak Otoritesi ve Hiyerarşi

Blogun temel ilkesi **"sıfır dedikodu, sıfır ikincil aktarım"**dır. İnternette araştırma yaparken aşağıdaki kaynak hiyerarşisi katı biçimde uygulanır:

### Tier 1: Kesin Otorite (Doğrudan Kabul Edilir)
1. **Orijinal Akademik Makaleler (arXiv / Seminal Papers):**
   - Transformer: Vaswani et al. (2017) `arXiv:1706.03762`
   - RoPE: Su et al. (2021) `arXiv:2104.09864`
   - Attention öncüsü: Bahdanau et al. (2014) `arXiv:1409.0473`
   - FlashAttention: Dao et al. (2022/2023)
   - LLaMA / DeepSeek / Mistral / Gemma resmi teknik raporları
2. **Resmi Model Konfigürasyonları ve Ağırlık Dosyaları:**
   - Hugging Face Hub üzerindeki resmi `config.json` dosyaları (ör. `google/gemma-3-1b-pt`, `meta-llama/Llama-3.1-8B`).
   - Gerçek tensör boyutları, katman sayıları, kafa (`num_attention_heads`), KV kafa (`num_key_value_heads`), ara boyut (`intermediate_size`), `rope_theta`, `vocab_size`.
3. **Resmi Çekirdek Kaynak Kodları:**
   - PyTorch (`torch.nn`, `torch.functional`), Hugging Face `transformers/models/*`, vLLM motoru kaynak kodları (`vllm/attention/ops/*`), triton/cuda çekirdekleri.
4. **Laboratuvarların Resmi Mühendislik ve Araştırma Dokümanları:**
   - Anthropic Research & Engineering (`platform.claude.com`, Transformer Circuits thread'leri).
   - OpenAI Research (`developers.openai.com`, official system cards).
   - Google DeepMind / Google Research teknik blogları.

### Tier 2: İkincil Doğrulama (Karşılaştırma İçin Kullanılır)
- Hugging Face Blog (derinlemesine teknik analizler ve rehberler).
- promptingguide.ai (akademik referansları verilen prompt teknikleri).
- Milvus / Pinecone / Qdrant / pgvector resmi mimari dokümanları.

### Tier 3: KESİNLİKLE YASAK (Kaynak Olarak Kullanılmaz)
- Medium / Substack / Dev.to özetleri (genellikle hatalı basitleştirmeler ve halüsinasyon içerir).
- LinkedIn veya X (Twitter) "hype" paylaşımları.
- Üçüncü parti haber siteleri veya SEO odaklı pop-tech blogları.
- Başka bir dil modelinin genel geçer (unverified) yanıtları.

---

## 2. Arama Stratejisi ve Araç Kullanımı

Antigravity araçları (`search_web`, `read_url_content`) kullanılırken izlenecek taktikler:

### Taktik A: arXiv ve Orijinal Makale Taraması
- Bir mekanizma araştırılırken arama sorgusuna doğrudan yazar adı veya arXiv terimi eklenir:
  - `site:arxiv.org "rotary position embedding" Su`
  - `"Attention Is All You Need" "scaled dot-product" Vaswani`
- Makale bulunduğunda `read_url_content` ile arXiv özeti veya HTML sürümü taranarak denklem numarası, tablodaki sayı veya ispat aranır.

### Taktik B: Model Konfigürasyonu Çıkarma
- Makalede tek bir gerçek model (vaka çalışması) üzerinden yürünür (ör. Gemma 3 veya LLaMA 3.1).
- Sayılar ezberden veya tahminden yazılmaz; modelin resmi konfigürasyonu taranır:
  - `site:huggingface.co "google/gemma-3-1b" "config.json"`
  - `site:huggingface.co "meta-llama/Meta-Llama-3.1-8B" "config.json"`
- Taranan parametreler:
  - `hidden_size` ($d_{\text{model}}$)
  - `num_attention_heads` ($h$)
  - `head_dim` ($d_k = d_{\text{model}} / h$)
  - `vocab_size`
  - `tie_word_embeddings` (true/false)
  - `rope_theta` (örn. 10000 vs 500000)

### Taktik C: Donanım ve Sistem Kısıtlarını Sorgulama
- Yalnızca "matematik ne?" sorusu değil, "bu matematik hangi donanım kısıtından doğdu?" sorusu araştırılır:
  - Neden $\sqrt{d_k}$? $\to$ Softmax doygunluğu (vanishing gradient) ve varyans patlaması.
  - Neden FlashAttention? $\to$ GPU HBM (High Bandwidth Memory) ile SRAM arasındaki bellek transfer darboğazı (IO-aware tiling).
  - Neden KV Cache? $\to$ Auto-regressive decode fazında önceki token'ların $K$ ve $V$'lerini her adımda yeniden hesaplamamak için bellek/hesap takası.
  - Neden PagedAttention (vLLM)? $\to$ Bellek parçalanması (fragmentation) ve dinamik dizi tahsisi.

---

## 3. Sayısal İddiaları Doğrulama Tablosu

Bir makaleye giren her sayı şu filtreden geçmelidir:

| Sayı Türü | Doğrulama Yöntemi | Kabul Kriteri |
| :--- | :--- | :--- |
| **Model Boyutları** | `config.json` veya teknik rapor | Tam rakam (örn. Gemma 3-1B: $d_{\text{model}}=1536$, kafa sayısı=8) |
| **Parametre Payı** | Katman ağırlık matrislerinin çarpımı ile formül hesabı | $\%30 \pm 1$ gibi kesin orantı |
| **VRAM Hesapları** | Parametre $\times$ bayt hassasiyeti (FP16: 2B, FP8: 1B, INT4: 0.5B) + optimizer durumları | 1B model eğitimde $\sim 16$ GB, çıkarımda $\sim 2$ GB |
| **Fertility / Token Oranları** | `tiktoken` veya Hugging Face tokenizer ile ölçüm | Belirli Türkçe cümle üzerinde somut oran (ör. 1,4 kat) |
| **Oyuncak Örnek Hesapları** | Python / PyTorch ile adım adım yürütülen bağımsız script | Kuruşuna kadar (virgülden sonraki basamaklar dahil) tutarlı |

---

## 4. Teyit Edilemeyen Bilgilerle İlgili İlke

Eğer bir iddia:
- Birincil kaynakta net değilse: *"Literatürde X olarak belirtilmekle birlikte..."* veya *"Yaklaşık olarak..."* ifadesi kullanılır.
- İki muteber kaynak arasında çelişki varsa: Çelişki gizlenmez, iki kaynak ve aralarındaki yaklaşım farkı doğrudan okura aktarılır (ör. "Orijinal makale $X$ alırken LLaMA uygulaması $Y$ kullanmıştır").
- Tamamen kaynaksızsa: **Makaleye ASLA dahil edilmez.**
