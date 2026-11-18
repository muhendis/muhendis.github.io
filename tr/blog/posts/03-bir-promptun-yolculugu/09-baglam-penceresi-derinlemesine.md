Pazarlamasında 128k bağlam penceresi vadeden 8B parametreli bir modeli sunucuya aldınız; 90.000 token uzunluğunda karmaşık bir hukuk sözleşmesini veya monolitik bir kod tabanını isteme (prompt) verdiniz ve altyapınızın saniyeler içinde kilitlendiğini izlediniz. İlk olarak, ön işleme (prefill) motoru HBM ile SRAM arasındaki yüzlerce gigabaytlık ara aktivasyon transferine boğulduğu için ilk token gecikmesi (TTFT - Time-to-First-Token) 42 saniyenin üzerine fırlar. Hemen ardından, eşzamanlı diğer kullanıcı isteklerine neredeyse hiç GPU belleği bırakmayacak şekilde KV Cache tek başına 23 GB High Bandwidth Memory (HBM) yutar. Cevap nihayet ekrana aktığında ise model, sözleşmenin tam 42. sayfasındaki kritik maddeyi görmezden gelerek bağlam çürümesinin (context rot) kör noktasında kendinden emin şekilde uydurur (hallucination).

Pazarlama iddiaları ile fiziksel donanım gerçekliği arasındaki bu gerilim, modern büyük dil modellerinin en kritik mühendislik cephesidir. Bağlam penceresi sonsuza uzanan ucuz bir metin kaydırma alanı değildir; $O(N^2)$ karesel hesaplama faturası, GPU bellek bant genişliği tavanı ve konumsal kodlama çözülmeleriyle çevrili dinamik bir zihinsel çalışma masasıdır. O masa taştığında veya aşırı kalabalıklaştığında sistem ya doğrudan bir CUDA Out-Of-Memory (OOM) çöküşüyle kapanır ya da mantıksal pusulasını tamamen kaybeder.

Bu yazı; bağlam penceresinin evrimini ve arkasındaki donanım gerçeklerini üç ana dönemde ele alıyor: 2017–2022 aralığındaki mutlak konumsal embedding'ler ve karesel $O(N^2)$ bellek duvarından, 2023–2025'in FlashAttention çekirdek kaynaştırması, RoPE/YaRN frekans ölçeklemesi ve GQA optimizasyonlarına; oradan da 2026'nın hibrit Transformer-SSM (Mamba) topolojilerine, RingAttention paralelizmine ve bilişsel önbellek budamasına uzanan eksiksiz bir mimari yolculuk.

**Bu yazıda**

- [1. Bağlam penceresinin önündeki üç temel duvar](#1-bağlam-penceresinin-önündeki-üç-temel-duvar)
- [2. Sezgisel dönemler: Biletli koltuktan akışkan zihne](#2-sezgisel-dönemler-biletli-koltuktan-akışkan-zihne)
- [3. Üç fazda matematiksel ve donanımsal mimari](#3-üç-fazda-matematiksel-ve-donanımsal-mimari)
- [4. Adım adım elle sayısal hesap: Online Softmax ve KV Cache](#4-adım-adım-elle-sayısal-hesap-online-softmax-ve-kv-cache)
- [5. İki mühendislik gözü: Eğitim ve çıkarım](#5-iki-mühendislik-gözü-eğitim-ve-çıkarım)
- [6. Mühendislik karar matrisi ve sınırları aşma stratejileri](#6-mühendislik-karar-matrisi-ve-sınırları-aşma-stratejileri)
- [7. Tarihsel evrim ve mimari özet tablosu](#7-tarihsel-evrim-ve-mimari-özet-tablosu)
- [8. PyTorch ve Python ile doğrulama: Online Softmax ve hibrit durum güncellemesi](#8-pytorch-ve-python-ile-doğrulama-online-softmax-ve-hibrit-durum-güncellemesi)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Bağlam penceresinin önündeki üç temel duvar

Büyük dil modellerinde bağlam penceresi, modelin **zihinsel çalışma masasıdır**. LLMs doğası gereği durumsuz (*stateless*) yapılardır; HTTP oturumları veya sorgular arasında kendi kendilerine hiçbir şey hatırlayamazlar. Yeni bir soru sorulduğunda geçmiş sohbetin tamamı masaya en baştan serilmek zorundadır. Masanın fiziksel sınırları aşıldığında ise eski dosyalar kenardan yere dökülür (*overflow / discarded tokens*).

Bu masayı büyütmek, üç amansız fiziksel duvara çarpar:

<svg viewBox="0 0 560 260" role="img" aria-label="Bağlam penceresinin büyümesini engelleyen üç temel duvar: Mimari karesel işlem yükü O(N²), HBM ile SRAM arasındaki donanım bellek duvarı ve ortada kaybolma performans çürümesi." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tr-w-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="10" width="528" height="64" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="31" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">1. MİMARİ DARBOĞAZ: O(N²) Karesel İşlem Yükü</text>
<text x="32" y="48" text-anchor="start" style="fill:var(--c-text);font-size:13px">Her token&#x27;ın diğer tüm token&#x27;larla skaler çarpımı zorunluluğu</text>
<text x="32" y="63" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">128k token = 16,4 milyar skor matrisi (FLOP ve bellek patlaması)</text>
<line x1="280" y1="74" x2="280" y2="92" marker-end="url(#tr-w-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="94" width="528" height="64" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="32" y="115" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700;letter-spacing:.06em">2. DONANIM DUVARI: HBM ile SRAM Arasındaki Bellek Duvarı</text>
<text x="32" y="132" text-anchor="start" style="fill:var(--c-text);font-size:13px">Yavaş HBM (3,3 TB/sn) ile hızlı çip içi SRAM (30 TB/sn) dar boğazı</text>
<text x="32" y="147" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Aritmetik yoğunluk düşer; GPU saat çevrimleri veri transferi bekler</text>
<line x1="280" y1="158" x2="280" y2="176" marker-end="url(#tr-w-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="178" width="528" height="64" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="32" y="199" text-anchor="start" style="fill:var(--c-danger);font-size:15px;font-weight:700;letter-spacing:.06em">3. PERFORMANS ÇÜRÜMESİ: Lost in the Middle ve Dikkat Sisi</text>
<text x="32" y="216" text-anchor="start" style="fill:var(--c-text);font-size:13px">Masa devasa bir depoya dönüştüğünde ortadaki bilgiler kaybolur</text>
<text x="32" y="231" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Dikkat katsayıları homojen bir sise dönüşerek halüsinasyona yol açar</text>
</svg>

### Mimari darboğaz: Karesel karmaşıklık O(N²)

Standart self-attention (öz-dikkat) mekanizmasında dizideki her bir token, dizideki diğer tüm token'larla skaler çarpıma girer. $N$ uzunluğundaki bir girdi dizisi için dikkat skor matrisi tam $N \times N = N^2$ elemandan oluşur.

$N = 2.048$ token için $N^2$ yaklaşık $4,19 \times 10^6$ ikili ilişki demektir. $N = 128.000$ token'a çıktığınızda bu sayı **$16,38 \times 10^9$ işleme** fırlar. Dizi uzunluğunu 62 kat artırmak, gereken ikili işlem ve skor matrisi boyutunu 3.900 katına çıkarır. Bağlam penceresini ikiye katlamak, hesaplama yükünü ve skor matrisi tahsisini anında dört katına katlar.

### Donanım duvarı: Yavaş HBM ile hızlı SRAM darboğazı

GPU üzerinde asıl kırılma noktası ham tensör çekirdeği işlem gücü (FLOPs) değil, bellek transfer bant genişliğidir — donanım dünyasındaki adıyla **Bellek Duvarı (Memory Wall)**.

Modern bir NVIDIA H100 GPU iki temel bellek katmanına sahiptir:
1. **High Bandwidth Memory (HBM3):** Yüksek kapasiteli (80 GB), ancak görece mütevazı bant genişliğine sahip ($\sim 3,35\text{ TB/sn}$) ana bellek.
2. **Çip İçi Static RAM (SRAM):** Ultra hızlı ($\sim 30\text{ TB/sn}$ birleşik aktarım hızı), ancak son derece kısıtlı kapasiteye sahip ($\sim 50\text{ MB}$) yerel bellek.

Klasik PyTorch attention uygulamalarında $S = QK^T \in \mathbb{R}^{B \times H \times N \times N}$ ara dikkat matrisi doğrudan çip dışındaki HBM'e yazılır, Softmax için tekrar SRAM'e okunur, normalize edilip HBM'e yazılır ve $V$ ile çarpılmak üzere üçüncü kez okunur. Bellek aktarımı işlemci çekirdeklerinden katbekat yavaş kaldığı için GPU zamanının %85'inden fazlasını aritmetik yapmak yerine HBM transferi bekleyerek harcar.

### Performans çürümesi: Lost in the middle ve bilişsel sis

Sisteme yeterince GPU kümesi ekleyip bellek yükünü dağıtsanız bile, uzun bağlam dikkat mekanizması bu kez bilişsel erozyona uğrar:

<svg viewBox="0 0 560 220" role="img" aria-label="Lost in the Middle bağlam çürümesi U-eğrisi: Döküman başı ve sonunda doğruluk yüksekken ortasında kör nokta oluşur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="8" width="528" height="204" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="28" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Lost in the Middle: Döküman Boyunca Geri Çağırma Başarısı</text>
<line x1="55" y1="165" x2="515" y2="165" style="stroke:var(--c-border);stroke-width:1.2"/>
<line x1="55" y1="46" x2="55" y2="165" style="stroke:var(--c-border);stroke-width:1.2"/>
<text x="46" y="52" text-anchor="end" style="fill:var(--c-text-mute);font-size:10px">100%</text>
<text x="46" y="108" text-anchor="end" style="fill:var(--c-text-mute);font-size:10px">50%</text>
<text x="46" y="165" text-anchor="end" style="fill:var(--c-text-mute);font-size:10px">0%</text>
<path d="M 55 58 C 145 58, 175 148, 285 148 C 395 148, 425 58, 515 58" style="fill:none;stroke:var(--c-danger);stroke-width:2.5"/>
<rect x="210" y="96" width="150" height="48" rx="8" style="fill:var(--c-danger);fill-opacity:0.1;stroke:var(--c-danger);stroke-width:1;stroke-dasharray:4 3"/>
<text x="285" y="114" text-anchor="middle" style="fill:var(--c-danger);font-size:11.5px;font-weight:600">Kör Nokta / Çürüme</text>
<text x="285" y="131" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">Döküman Ortası (~%20)</text>
<rect x="65" y="68" width="110" height="36" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/>
<text x="120" y="83" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Öncelik Etkisi</text>
<text x="120" y="96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Başlangıç (~%95)</text>
<rect x="395" y="68" width="110" height="36" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/>
<text x="450" y="83" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-weight:600">Yakınlık Etkisi</text>
<text x="450" y="96" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px">Son Token&#x27;lar (~%95)</text>
<text x="75" y="184" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Döküman Başı</text>
<text x="285" y="184" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Döküman Ortası</text>
<text x="495" y="184" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Döküman Sonu</text>
</svg>

Empirik araştırmalar (*Liu et al., 2023, Stanford / Berkeley*), otoregresif dil modellerinin U şeklinde bir geri çağırma (retrieval) başarı eğrisi sergilediğini kanıtlamıştır. Modeller istemin ilk %10'luk bölümünü (sistem prompt'unun oluşturduğu *öncelik etkisi*) ve son %10'luk bölümünü (üretim anına en yakın *yakınlık etkisi*) son derece yüksek doğrulukla hatırlar. Ancak aranan kritik bilgi dökümanın tam ortasına bırakıldığında başarı oranı dip yapar. Milyonlarca token arasında dikkat katsayıları homojen bir sise dönüşerek seyrelir.

---

## 2. Sezgisel dönemler: Biletli koltuktan akışkan zihne

Bağlam penceresinin 2.048 token'lık dar bir çerçeveden 10 milyon token'ı aşan sınırlara uzanması, modern yapay zeka mühendisliğinin en heyecan verici evrimlerinden biridir. Bu yolculukta karşılaşılan darboğazları kavramanın en kestirme yolu; soyut denklemlerin arkasındaki fiziksel gerçekliği gözümüzde canlandırmak ve her sezgisel benzetmeyi doğrudan silikon donanımın terimleriyle eşleştirmektir.

### 1. Faz (2017-2022): Statik duvarlar ve masadaki evrak yığını

Transformer mimarisinin ilk yıllarında modeller, hem mimari olarak katı kalıplara hapsedilmişti hem de GPU donanımının acımasız bellek sınırlarına tosluyordu:

**1. Biletli Koltuk vs. Mutlak Pozisyon Matrisi (APE):**
Bir konferans salonu düşünün; salonda tam 2.048 adet sabit koltuk var ve her koltuğun üzerine metal bir plaka çakılmış: *1. Koltuk, 2. Koltuk ... 2.048. Koltuk*. Her dinleyicinin biletinde bu koltuklardan biri yazılıdır. Peki 2.049. dinleyici kapıya geldiğinde ne olur? Salonda ona verilebilecek bir koltuk numarası evrende fiziksel olarak tanımlı değildir. Sistem "biraz sıkışın" diyemez; doğrudan kapıyı kilitler.

Bu benzetmenin arkasındaki teknik gerçeklik **Absolute Positional Embeddings (APE)** mimarisidir. Orijinal Vaswani Transformer'da token'ların konumunu belirlemek için ($L_{\max} \times d$) boyutunda sabit bir pozisyon tablosu kullanılıyordu. Model eğitim sırasında yalnızca 0 ile 2.047 arasındaki pozisyon indislerini öğrenir. Prompt'unuz 2.049 token'a ulaştığında, embedding matrisinde bakılabilecek 2.049 numaralı bir satır yoktur. Sonuç: Model eğitilmediği koordinatlara matematiksel olarak genelleme yapamaz (*extrapolation failure*); ya indis taşması hatası (*index out of bounds*) verir ya da anlamsız çıktılar üretir.

**2. Uğultulu Kokteyl Salonu vs. Karesel İşlem Yükü ($O(N^2)$):**
Küçük bir odada 10 kişi sohbet ederken herkesin birbiriyle göz teması kurup konuşması son derece kolaydır; toplamda $10 \times 10 = 100$ ikili bakışma yeterlidir. Fakat odaya 10.000 kişi girdiğinde 100 milyon, 100.000 kişi girdiğinde ise tam **10 milyar** ikili bakışma gerekir. Salon saniyeler içinde uğultudan ve herkesin birbirini takip etme çabasından felç olur.

Teknik karşılık: **Self-Attention mekanizmasının karesel hesaplama karmaşıklığı ($O(N^2)$ FLOPs ve bellek tahsisi)**.
Modeldeki her bir token ($Query$), dizideki önceki tüm token'larla ($Key$) skaler çarpıma girer ($QK^T$). Girdi dizisi iki katına çıktığında, üretilen ara dikkat matrisinin ($S \in \mathbb{R}^{B \times H \times N \times N}$) boyutu ve gereken tensör işlemleri anında **dört katına** fırlar. Örneğin 128k'lık bir bağlamda tek bir katmanda 16,4 milyar skaler çarpım yapılması gerekir. Bu devasa matris, prefill aşamasında GPU'nun tensör çekirdeklerini aritmetik işlem yüküyle boğar.

**3. Kırılan Ahşap Masa vs. KV Cache Tüketimi (VRAM Duvarı):**
Bir dedektif düşünün (LLM'in token üreten Decoder motoru). Çalışma masasına koyduğu her yeni ipucu (yeni üretilen token) için, masadaki önceki tüm belgelere renkli ipler (dikkat katsayıları) çeker. Ancak masanın (GPU VRAM) taşıyabileceği bir ağırlık sınırı vardır. Dedektif ipuçlarını yığdıkça masanın üzerindeki evrakların ve iplerin ağırlığı masayı çatırdatır ve sonunda masa ortadan ikiye kırılır.

Teknik karşılık: **Çıkarım (inference) sırasındaki KV Cache bellek tüketimi ve CUDA Out-of-Memory (OOM)**.
Dil modelleri yeni bir kelime üretirken (decode aşaması), geçmişteki yüz binlerce token'ı her seferinde baştan hesaplamak istemez. Bu yüzden geçmiş token'ların Key ve Value vektörlerini GPU'nun yüksek hızlı belleğinde hazır tutar; buna **KV Cache** diyoruz. Ancak sorun şudur: 128k bağlam uzunluğunda, tek bir kullanıcının KV Cache faturası 16 ile 32 GB VRAM'e ulaşır! Modelin kendi ağırlıkları zaten onlarca gigabayt tutarken, KV Cache için ayrılan bu devasa alan GPU'nun fiziksel VRAM sınırını (örneğin bir H100'deki 80 GB) aştığı anda işletim sistemi süreci acımasızca sonlandırır: `CUDA Out of Memory`.

### 2. Faz (2023-2025): Donanım zekâsı ve akordeon masa

İkinci fazda mühendisler, hem matematiksel rotasyon hileleriyle pozisyon kısıtını kırdı hem de GPU'nun silikon mimarisini optimize etti:

**1. Pusula Açısı vs. Göreli Koordinatlar (RoPE ve YaRN):**
Masadaki evrakların yerini cetvelle "sol baştan 3. santim, 8. santim" gibi mutlak koordinatlarla etiketlemek yerine; belgelerin birbirine göre duruş açısını kullanmaya başladığınızı düşünün. Örneğin "bu belge diğerinin saat 2 yönünde, aralarında 30 derece açı var". Masayı odanın neresine taşırsanız taşıyın, iki belgenin birbirine göre açısı ve mesafesi asla bozulmaz. Üstelik masayı esnetmek istediğinizde açıları sıkıştırarak (akordeon gibi) aynı alana iki kat daha fazla evrak sığdırma illüzyonu yaratabilirsiniz.

Teknik karşılık: **Rotary Position Embedding (RoPE) ve YaRN frekans enterpolasyonu**.
Mühendisler, token'lara sabit pozisyon vektörleri eklemek ($X + P$) yerine, $Q$ ve $K$ vektörlerini 2 boyutlu dilimler halinde pozisyonla orantılı bir açıyla döndürmeyi akıl etti ($\mathcal{R}_{\Theta, m}$). Böylece dikkat skoru mutlak konuma değil, iki token arasındaki **göreli mesafeye ($m - n$)** bağlı hale geldi. 128k gibi hiç eğitilmediği uzunluklara çıkarken ise YaRN tekniğiyle frekanslar dalga boylarına göre akıllıca sıkıştırıldı ($m/s$); böylece model, uzun bağlamları sanki orijinal eğitim penceresindeymiş gibi algılayabildi.

**2. Aşçı, Kesme Tahtası ve Kiler Trafiği (FlashAttention ve Bellek Duvarı):**
Bir restoran mutfağını gözünüzün önüne getirin: Bir baş aşçı (GPU Tensör Çekirdekleri), aşçının hemen önündeki küçük ama ultra hızlı kesme tahtası (çip içi SRAM belleği) ve mutfağın 50 metre ilerisindeki büyük soğuk hava deposu/kiler (çip dışı ana bellek HBM).
Eski sistemde aşçı her bir soğanı doğramak için kilere kadar yürüyor, soğanı alıp kesme tahtasında doğruyor, sonra doğranmış soğanı saklamak için tekrar kilere yürüyor, ardından bir parça tereyağı için yeniden koridorda koşuyordu. Aşçı zamanının %90'ını yemek pişirmek (aritmetik hesaplama) yerine koridorda yürüyerek (veri transferi) harcıyordu!
FlashAttention ile mutfak devrimi yapıldı: Aşçı artık bir tencereye yetecek tüm malzemeyi tek seferde tezgahına kilitliyor (*Kernel Fusion*), doğrama ve pişirme işlemlerinin tamamını kesme tahtasında bitiriyor ve kilere sadece servise hazır nihai yemeği gönderiyor.

Teknik karşılık: **Memory Wall (Bellek Duvarı), SRAM Tiling ve FlashAttention Online Softmax**.
GPU'nun ana belleği olan HBM ($3,35\text{ TB/sn}$), işlemci çekirdeklerinin hemen yanındaki yerel SRAM'e ($30\text{ TB/sn}$) kıyasla 10 kat daha yavaştır. Standart attention mekanizmasında $N \times N$ boyutundaki devasa ara dikkat matrisi her işlem adımında HBM'e yazılıp tekrar okunuyordu. Tri Dao ve ekibinin geliştirdiği FlashAttention, **Online Softmax** algoritmasını kullanarak matrisi küçük bloklara (*tiles*) böldü. Tüm ara matris işlemleri yalnızca ultra hızlı SRAM içinde tamamlandı ve HBM'e yalnızca nihai dikkat çıktısı yazıldı. Sonuç: GPU çekirdekleri veri transferi beklemekten kurtuldu ve 2 ila 4 kat hızlandı.

**3. Pazarlama vs. Gerçeklik Uçurumu (MCW vs. MECW):**
Bir depolama şirketinin "Bu odaya tam 1 milyon kitap sığdırabilirsiniz!" diye reklam yaptığını düşünün. Gerçekten de 1 milyon kitabı odaya tepeleme yığabilirsiniz; kapı kapanır. Fakat içeri girip "542. kitaptaki üçüncü sayfayı bana bul" dediğinizde saatlerce arayıp bulamıyorsanız, o oda bir kütüphane değil, kaos deposudur.

Teknik karşılık: **Maximum Context Window (MCW) ile Maximum Effective Context Window (MECW) arasındaki uçurum**.
Yapay zeka laboratuvarlarının pazarlamada öne çıkardığı **MCW (Maksimum Bağlam Penceresi)**, modelin mimarisinin ve GPU belleğinin teknik olarak OOM hatası vermeden içine kabul edebildiği token kapasitesidir (örneğin 1 milyon veya 2 milyon token). Ancak bir mühendisin asıl bakması gereken metrik **MECW (Maksimum Etkili Bağlam Penceresi)**'dir: Modelin dikkatini kaybetmeden, dökümanlar arasında halüsinasyon görmeden bilgiyi %100 doğrulukla bulup akıl yürütebildiği gerçek limit. Pratikte 1M token destekleyen pek çok modelin MECW sınırı 64k ile 128k arasında tıkanır.

**4. Romanın Ortasındaki İpucu vs. Lost in the Middle:**
1.000 sayfalık sürükleyici bir polisiye roman okuduğunuzu hayal edin. Kitabın ilk 50 sayfasında tanıtılan ana dedektifi (ilk izlenim / öncelik etkisi) ve son 50 sayfadaki heyecanlı final sahnesini (taze hafıza / yakınlık etkisi) neredeyse hiç unutmazsınız. Ancak cinayeti çözen asıl ipucu 500. sayfadaki sıradan bir yemek sohbetinin arasına sıkıştırılmışsa, zihniniz orayı büyük ihtimalle kaçırır.

Teknik karşılık: **Lost in the Middle, Needle In A Haystack (NIAH) ve Attention Dilution (Dikkat Seyrelmesi)**.
Softmax matematiksel olarak normalize bir olasılık dağılımıdır; tüm dikkat katsayılarının toplamı 1 etmek zorundadır ($\sum = 1$). Girdi dizisi yüz binlerce token'a uzadığında, paydadaki toplam devasa boyutlara ulaşır ve token başına düşen dikkat katsayısı homojen bir sise dönüşerek erir. Dil modelleri doğası gereği sistem prompt'una (*öncelik etkisi / primacy*) ve son üretilen token'lara (*yakınlık etkisi / recency*) çok yüksek dikkat ağırlığı verir; dökümanın tam ortasındaki token'lara düşen ağırlık ise gürültü seviyesine iner. Aradığınız kritik bilgi tam ortadaysa, model o bilgiyi "göremez" ve uydurmaya (halüsinasyona) başlar.

### 3. Faz (2026): Hibrit zihin ve bilişsel unutma

Üçüncü fazda mimariler, insan beyninin seçici dikkat ve dinamik unutma refleksini silikona taşıdı:

**1. Açık Fotoğraf Albümü vs. Canlı Zihinsel Özet (Hibrit Modeller):**
Bir ansiklopediyi okurken her yeni kelimede önceki bütün sayfaları masada açık tutup hepsine aynı anda göz gezdirmeye çalışan bir araştırmacı düşünün (Saf Transformer). Ne kadar çok sayfa açarsa gözleri o kadar yorulur, masada yer kalmaz. Şimdi bir de kitabı akıcı bir şekilde okurken aklında sürekli güncellenen tek bir canlı özet tutan, yalnızca çok karmaşık bir çapraz referans gerektiğinde eski sayfalara büyüteçle dönüp bakan uzman bir araştırmacıyı düşünün (Hibrit Transformer-SSM).

Teknik karşılık: **Transformer + SSM (Mamba-2) Hibrit Mimarileri ve $O(1)$ Decode Belleği**.
Standart Transformer modelleri, her yeni token üretirken geçmişteki tüm KV vektörlerini saklamak zorundadır ($O(N)$ bellek). Oysa SSM (State Space Model) ve Mamba-2 blokları, $h_t = \bar{A} h_{t-1} + \bar{B} x_t$ yinelemeli durum denklemiyle tüm geçmişi sabit boyutlu bir gizli vektörde ($h \in \mathbb{R}^{d_{\text{state}}}$) özetler. Bu sayede bağlam ne kadar uzarsa uzasın, decode aşamasında bellek tüketimi sabit kalır ($O(1)$) ve prefill süresi doğrusal ölçeklenir ($O(N)$). 2026'nın modern modelleri (örneğin Jamba mimarisi), katmanların büyük kısmını (%80) doğrusal SSM yaparak sınırsız akış hızı kazanır; kalan katmanları (%20) çapraz dikkat katmanı olarak bırakarak derin akıl yürütme kabiliyetini korur.

**2. Beynin Evrimsel Refleksi vs. Seçici Unutma (Sparse KV Cache):**
İnsan beyni inanılmaz bir bilgi işleme mucizesidir çünkü neyi hatırlayacağını bildiği kadar **neyi unutacağını** da çok iyi bilir. Dün sabah yediğiniz ekmeğin üzerindeki susam sayısını ya da beş yıl önce giydiğiniz çorabın rengini hafızanızda tutmazsınız; hayatta kalmak için değersiz detayları anında siler, yalnızca kritik kalıpları saklarsınız. Eğer çalışma masanızdaki gereksiz müsveddeleri çöpe atmazsanız, bir süre sonra masada çalışamaz hale gelirsiniz.

Teknik karşılık: **Sparse KV Cache Budaması (StreamingLLM, Heavy Hitter Oracle - H2O, SnapKV)**.
Bellekteki yüz binlerce KV tensörünün hepsi eşit değerde değildir. Yapılan araştırmalar, ilk birkaç token'ın (**Attention Sinks**) modelin dikkat dengesini kurmak için hayati olduğunu, en son token'ların ise yerel bağlamı (**Local Sliding Window**) koruduğunu göstermiştir. Geriye kalan ortadaki token'lardan dikkat skoru önemsiz olanlar çalışma anında bellekten tahliye edilir (*cache eviction*). Bu sayede 128k uzunluğundaki bir döküman, bellekte sanki sadece 4k'lık bir bağlam varmış gibi sabit bir VRAM bütçesiyle işlenebilir.

---

## 3. Üç fazda matematiksel ve donanımsal mimari

Tüm bu sezgisel metaforlar doğrudan matematiksel formüllere ve silikon seviyesindeki işlem adımlarına dayanır.

### 1. Faz: Absolute Positional Embeddings çöküşü ve HBM kilitlenmesi

Orijinal Transformer mimarisinde (*Vaswani et al., 2017*) token'ların dizi içindeki konumu, sözlük temsillerine doğrudan eklenen sabit sinüsoidal veya öğrenilmiş mutlak vektörlerle tanımlanıyordu:

$$X_{\text{input}} = X_{\text{embed}} + P, \quad P \in \mathbb{R}^{L_{\max} \times d}$$

Şöyle okuyun: $X_{\text{embed}} \in \mathbb{R}^{N \times d}$ anlamsal token embedding tensörüdür; $P$ ise eğitim sırasında sabitlenmiş $L_{\max} \in [512, 2.048]$ satırlık mutlak pozisyon matrisidir.

Modelin eğittiği $P$ matrisi yalnızca $[0, L_{\max}-1]$ aralığındaki indisleri tanıdığından, $N > L_{\max}$ için dışdeğerbiçim (*extrapolation*) kabiliyeti matematiksel olarak tanımsızdı.

Aynı dönemde standart self-attention formülü:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Şöyle okuyun: $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_k}$ sorgu, anahtar ve değer tensörleridir ($B$: batch, $H$: kafa sayısı, $d_k$: kafa boyutu).

Buradaki en büyük mimari zaafiyet, ara skor matrisi olan $S = QK^T \in \mathbb{R}^{B \times H \times N \times N}$ tensörünün doğrudan HBM'de ayrıştırılmasıydı (*allocation*). $B=1$, $H=32$ ve $N=32.768$ için bu matrisin FP16 ($2\text{ bayt}$) karşılığı:

$$1 \times 32 \times 32.768 \times 32.768 \times 2\text{ bayt} \approx 68,7\text{ GB VRAM}$$

Bu tek ara tensör, henüz model ağırlıkları veya Softmax hesaplanmadan NVIDIA A100'ün (40 GB/80 GB) tüm HBM kapasitesini tek başına tüketerek sistemi çökertecek büyüklüktedir.

### 2. Faz: RoPE, YaRN enterpolasyonu, FlashAttention tiling ve GQA

İkinci dönem bu krizi üç kritik matematiksel ve donanımsal buluşla aştı:

#### 1. Döner Konumsal Gömme (RoPE) ve YaRN Enterpolasyonu

RoPE (*Su et al., 2021*), mutlak vektör toplamak yerine sorgu ve anahtar vektörlerini 2 boyutlu ortogonal rotasyon bloklarıyla döndürdü:

$$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = \mathbf{x}_m^T \mathbf{W}_q^T \mathbf{R}_{\Theta, n-m}^d \mathbf{W}_k \mathbf{x}_n$$

Şöyle okuyun: $m$ ve $n$ pozisyonlarındaki iki token arasındaki skaler çarpım, yalnızca aralarındaki göreli mesafe olan $(n - m)$ açısına bağlıdır ($\mathbf{R}_{\Theta, n-m}^d$ rotasyon matrisi).

Modeli baştan eğitmeden bağlamı uzatmak için geliştirilen **YaRN** (*Peng et al., 2023*), RoPE frekanslarını üç banda ayırdı:
- Yüksek frekanslar (kısa dalga boyları): Yerel dilbilgisini korumak için dokunulmadan bırakıldı.
- Düşük frekanslar (uzun dalga boyları): $s = L_{\text{yeni}} / L_{\text{eski}}$ ölçeğinde doğrusal enterpole edildi.
- Orta frekanslar: Yumuşak bir rampa fonksiyonuyla pürüzsüzce harmanlandı.

#### 2. FlashAttention ve Çip-İçi Online Softmax

FlashAttention (*Dao et al., 2022, 2023*), GPU darboğazının hesaplama değil bellek transferi olduğunu fark etti. $N \times N$ matrisini HBM'e yazmak yerine, verileri $B_r \times d$ ve $B_c \times d$ boyutlu bloklar (*tiling*) halinde doğrudan çip içi hızlı SRAM'e çekti.

<svg viewBox="0 0 560 240" role="img" aria-label="Standart Attention HBM trafiği ile FlashAttention çip-içi SRAM füzyonu karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tr-f-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="10" width="256" height="218" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="32" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700;letter-spacing:.06em">Standart Attention</text>
<text x="28" y="48" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Ara skorlar sürekli HBM&#x27;e yazılır</text>
<rect x="26" y="58" width="236" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="34" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1. QKᵀ çarpımı yapılır</text>
<text x="34" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM&#x27;e yazılır (O(N²) bellek)</text>
<rect x="26" y="100" width="236" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="34" y="115" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2. Softmax normalizasyonu</text>
<text x="34" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM&#x27;den okunur ve tekrar yazılır</text>
<rect x="26" y="142" width="236" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="34" y="157" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">3. Değer matrisi V ile çarpım</text>
<text x="34" y="170" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">HBM&#x27;den 3. kez okunup çarpılır</text>
<text x="144" y="206" text-anchor="middle" style="fill:var(--c-danger);font-size:11px;font-weight:600">Zamanın %85&#x27;i HBM yolunda harcanır</text>
<rect x="288" y="10" width="256" height="218" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="300" y="32" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">FlashAttention (Füzyon)</text>
<text x="300" y="48" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Tüm hesaplama çip-içi SRAM&#x27;de biter</text>
<rect x="298" y="58" width="236" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="306" y="73" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">1. Girdiler blok blok (tiling)</text>
<text x="306" y="86" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hızlı SRAM&#x27;e tek seferde çekilir</text>
<rect x="298" y="100" width="236" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="306" y="115" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">2. Online Softmax yinelemesi</text>
<text x="306" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">SRAM içinde dinamik ölçeklenir</text>
<rect x="298" y="142" width="236" height="36" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="306" y="157" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">3. HBM&#x27;e doğrudan yazım</text>
<text x="306" y="170" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Yalnızca nihai çıktı tensörü O yazılır</text>
<text x="416" y="206" text-anchor="middle" style="fill:var(--c-success);font-size:11px;font-weight:600">Bellek trafiği O(N)&#x27;e düşürülür</text>
</svg>

Dizinin tamamını görmeden blok blok Softmax hesaplayabilmek için **Online Softmax** yineleme algoritması kullanıldı:

$$m^{(j)} = \max(m^{(j-1)}, \tilde{m}^{(j)}), \quad l^{(j)} = e^{m^{(j-1)} - m^{(j)}} l^{(j-1)} + \sum_{k} e^{S_{ik}^{(j)} - m^{(j)}}$$

Şöyle okuyun: $m^{(j)}$ satır bazında koşan maksimum değeri günceller, $l^{(j)}$ ise yeni maksimuma göre eski eksponansiyel toplamı düzeltip yeni bloğun toplamını ekler. SRAM içindeki ara akümülatörler dinamik ölçeklenerek HBM'e *yalnızca nihai $O \in \mathbb{R}^{N \times d}$ tensörü* yazılır; bellek trafiği $O(N^2)$'den $O(N)$'e iner.

#### 3. Grup Sorgulu Dikkat (GQA - Grouped-Query Attention)

Multi-Head Attention (MHA) mimarisinde her sorgu kafası için ayrı Key-Value tutulması çıkarım sırasında GPU belleğini (KV Cache) bitiriyordu:

$$\text{KV Cache Boyutu} = 2 \times B \times L \times H_{KV} \times d_k \times P_{\text{bytes}}$$

Şöyle okuyun: $B$ batch boyutu, $L$ katman sayısı, $H_{KV}$ anahtar-değer kafa sayısı, $d_k$ kafa boyutu, $P_{\text{bytes}}$ hassasiyet baytıdır (BF16 için $2$).

GQA (*Ainslie et al., 2023*), birden fazla sorgu kafasının tek bir KV kafasını paylaşmasını sağlayarak ($H_{KV} = H_Q / 8$) KV Cache faturasını **%87,5 oranında** küçülttü ve 128k çıkarımı uygulanabilir kıldı.

### 3. Faz: Hibrit Transformer-SSM topolojileri ve seyreltik KV önbellek

2026 yılına gelindiğinde saf attention mimarileri fiziksel verimlilik sınırına dayandı. Modern mimariler Transformer ile Durum Uzayı Modellerini (SSM / Mamba) birleştiren **Hibrit Topolojilere** geçti:

<svg viewBox="0 0 560 240" role="img" aria-label="2026 Hibrit Model Topolojisi: Doğrusal SSM Mamba katmanları, Native Chunking ve periyodik Full Attention." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="tr-h-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<rect x="16" y="10" width="528" height="220" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="30" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">2026 Hibrit Model Topolojisi ve Veri Yönetimi</text>
<rect x="130" y="40" width="300" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="280.0" y="57.0" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">Girdi İstemi (1M - 10M+ Token)</text>
<line x1="280" y1="66" x2="280" y2="80" marker-end="url(#tr-h-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="32" y="82" width="496" height="38" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="46" y="97" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Doğrusal SSM / Mamba Katmanları (%80 Pay)</text>
<text x="46" y="111" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Doğrusal Akış O(N) · Sabit Durum ht: O(1) Bellek Ayak İzi · Düşük TTFT</text>
<line x1="280" y1="120" x2="280" y2="132" marker-end="url(#tr-h-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="32" y="134" width="496" height="38" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="46" y="149" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Native Chunking &amp; Context Parallelism</text>
<text x="46" y="163" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">GPU kümesinde paralel halka topolojisi (RingAttention motoru)</text>
<line x1="280" y1="172" x2="280" y2="184" marker-end="url(#tr-h-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="32" y="186" width="496" height="36" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="46" y="201" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">Periyodik Full Attention (%20 Pay) &amp; Seyreltik Önbellek</text>
<text x="46" y="215" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Çok adımlı mantık · Önemsiz token&#x27;ları silme (Runtime Eviction) · FP4/FP8</text>
</svg>

#### Transformer + SSM / Mamba Hibritleşmesi

Jamba, Zamba ve 2026 kurumsal mimarilerinde katmanların %75–%85'i Mamba-2 SSM bloklarıyla donatılmıştır:

$$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t$$

Şöyle okuyun: $h_t \in \mathbb{R}^{d_{\text{state}}}$ her adımda yinelenen sabit boyutlu gizli durum vektörüdür; $x_t$ mevcut token girdisi, $y_t$ ise katman çıktısıdır.

Bu yapının getirdiği devrim:
- **Sabit Bellek Ayak İzi:** $h_t$ durum tensörünün boyutu dizilim uzunluğundan bağımsızdır; bellek tüketimi kesin olarak **$O(1)$**'dir.
- **Doğrusal Prefill:** Ön işleme süresi $O(N^2)$ değil kesin olarak **$O(N)$** ölçeğindedir.
- Saf attention katmanları yalnızca her 4 ila 8 blokta bir konumlandırılarak karmaşık mantıksal geri çağırma yeteneği korunur.

#### Native Chunking, RingAttention ve Seyreltik Önbellek

- **RingAttention:** Milyonlarca token'lık devasa istemleri işlemek için dizi GPU kümesinde halka topolojisinde bölünür. GPU $i$ kendi bloğunun attention hesabını yaparken, komşu GPU'dan bir sonraki $K, V$ bloklarını NVLink üzerinden çeker; iletişim hesaplamanın arkasına gizlenir.
- **Seyreltik Önbellek (Sparse Eviction):** Dikkat katsayılarının izlenmesi, geçmiş token'ların %80'inden fazlasının nihai çıktıyı etkilemediğini gösterir. Çalışma anında bu önemsiz token'lar KV önbelleğinden atılır; kalanlar FP4/FP8 formatında tutulur.

---

## 4. Adım adım elle sayısal hesap: Online Softmax ve KV Cache

Mekanizmayı somutlaştırmak için iki kritik hesabı elle ve gerçek sayılarla basamak basamak yürütelim:
1. FlashAttention Online Softmax blok birleştirme hesabı.
2. Modern bir 8B modelinde KV Cache bellek faturasının ölçeklenmesi.

### 1. Bölüm: Online Softmax Blok Yürüyüşü

Bir sorgu vektörünün $N = 4$ uzunluğundaki bir diziye dikkat uyguladığını varsayalım. Boyutları $B_c = 2$ olan 2 blok halinde işliyoruz. Değer boyutu $d = 2$ olsun.

```text
Ham Skaler Çarpım Skorları:
S = [2,0,  4,0,  3,0,  5,0]
    └─ Blok 1 ─┘ └─ Blok 2 ─┘

Değer Tensörleri (V):
V_blok1 = [[1,0, 0,0],
           [0,0, 2,0]]

V_blok2 = [[2,0, 1,0],
           [1,0, 3,0]]
```

#### Blok 1 Hesabı (SRAM İçinde):
1. Yerel blok maksimumunu bul:
   $$\tilde{m}_1 = \max(2,0, 4,0) = 4,0$$
   Koşan maksimum: $m_1 = \tilde{m}_1 = 4,0$.
2. Normalize edilmemiş üstel değerleri hesapla:
   $$P_1 = [e^{2,0 - 4,0}, e^{4,0 - 4,0}] = [e^{-2}, e^{0}] = [0,135335, 1,000000]$$
3. Yerel payda toplamını hesapla:
   $$l_1 = 0,135335 + 1,000000 = 1,135335$$
4. Normalize edilmemiş yerel çıktı akümülatörünü hesapla:
   $$O_1 = P_1 \times V_{\text{blok1}} = 0,135335 \times [1,0, 0,0] + 1,000000 \times [0,0, 2,0] = [0,135335, 2,000000]$$

#### Blok 2 Hesabı ve Çevrimiçi Birleştirme:
1. Blok 2 yerel maksimumunu bul:
   $$\tilde{m}_2 = \max(3,0, 5,0) = 5,0$$
2. Global maksimumu güncelle:
   $$m_2 = \max(m_1, \tilde{m}_2) = \max(4,0, 5,0) = 5,0$$
3. Blok 2 üstel değerlerini yeni $m_2 = 5,0$ değerine göre hesapla:
   $$P_2 = [e^{3,0 - 5,0}, e^{5,0 - 5,0}] = [e^{-2}, e^{0}] = [0,135335, 1,000000]$$
   Yerel payda katkısı: $\tilde{l}_2 = 0,135335 + 1,000000 = 1,135335$.
4. **Çevrimiçi Düzeltme Katsayısı:** Maksimum $4,0$'tan $5,0$'a çıktığı için Blok 1'den gelen eski terimlerin tamamı küçültülmelidir:
   $$\alpha = e^{m_1 - m_2} = e^{4,0 - 5,0} = e^{-1} \approx 0,367879$$
5. Global payda toplamını güncelle:
   $$l_2 = \alpha \times l_1 + \tilde{l}_2 = (0,367879 \times 1,135335) + 1,135335 = 0,417667 + 1,135335 = 1,553002$$
6. Global çıktı akümülatörünü güncelle:
   $$O_{2,\text{yerel}} = 0,135335 \times [2,0, 1,0] + 1,000000 \times [1,0, 3,0] = [1,270670, 3,135335]$$
   $$O_2 = \alpha \times O_1 + O_{2,\text{yerel}} = 0,367879 \times [0,135335, 2,000000] + [1,270670, 3,135335]$$
   $$O_2 = [0,049787, 0,735758] + [1,270670, 3,135335] = [1,320457, 3,871093]$$
7. Nihai normalize dikkat vektörünü elde et:
   $$O_{\text{nihai}} = \frac{O_2}{l_2} = \left[ \frac{1,320457}{1,553002}, \frac{3,871093}{1,553002} \right] = [0,850262, 2,492653]$$

#### Standart Softmax İle Sağlama:
Tüm 4 skoru global maksimum $5,0$ ile tek seferde standart Softmax'a soktuğumuzda:
$$S - 5,0 = [-3,0, -1,0, -2,0, 0,0]$$
Üstseller: $[0,049787, 0,367879, 0,135335, 1,000000]$, Toplam $= 1,553002$.
Olasılıklar: $P = [0,032059, 0,236883, 0,087144, 0,643914]$.
$V$ ile matris çarpımı:
$$O = 0,032059[1,0] + 0,236883[0,2] + 0,087144[2,1] + 0,643914[1,3] = [0,850262, 2,492653]$$
Sonuçlar virgülden sonra 16 basamak hassasiyetle birebir aynıdır; üstelik HBM'de hiçbir zaman $4 \times 4$'lük bir matris ayrıştırılmamıştır.

---

### 2. Bölüm: KV Cache Bellek Patlaması Hesabı

Şimdi modern bir açık ağırlıklı üretim modeli olan **LLaMA 3.1 8B** parametreleri üzerinden bellek maliyetini hesaplayalım:
- Katman Sayısı ($L$): $32$
- KV Kafa Sayısı ($H_{KV}$): $8$ (32 sorgu kafasıyla GQA)
- Kafa Boyutu ($d_k$): $128$
- Hassasiyet ($P$): BF16 ($2\text{ bayt}$)

İstemdeki her bir token için model her katmanda bir Key ve bir Value vektörü depolar:

$$\text{Token Başına KV Baytı} = 2 \times L \times H_{KV} \times d_k \times P = 2 \times 32 \times 8 \times 128 \times 2 = 131.072\text{ bayt} = 128\text{ KB}$$

Tek bir kullanıcı ($B = 1$) için bağlam uzunluğu arttıkça faturanın nasıl katlandığına bakın:
- **2.048 token (2020 dönemi):** $2.048 \times 128\text{ KB} = \mathbf{0,25\text{ GB}}$
- **8.192 token:** $8.192 \times 128\text{ KB} = \mathbf{1,00\text{ GB}}$
- **32.768 token:** $32.768 \times 128\text{ KB} = \mathbf{4,00\text{ GB}}$
- **131.072 token (128k):** $131.072 \times 128\text{ KB} = \mathbf{16,00\text{ GB}}$
- **1.048.576 token (1M):** $1.048.576 \times 128\text{ KB} = \mathbf{128,00\text{ GB}}$

128k bağlamda tek bir kullanıcının KV Cache'i **16 GB VRAM** tüketir; bu, 8B modelin kendi ağırlıklarının (16 GB) tamamına eşittir. 1M token'da ise tek bir istek, eşzamanlı hiçbir kullanıcıya yer bırakmaksızın iki adet 80 GB NVIDIA H100 GPU'sunu tamamen yutar.

---

## 5. İki mühendislik gözü: Eğitim ve çıkarım

Bağlam uzadıkça donanımın davranışı eğitim ve çıkarım süreçlerinde taban tabana zıt iki karaktere bürünür:

| Boyut | Eğitim Sistemleri (Pre-training / SFT) | Çıkarım Sistemleri (Serving / Generation) |
| :--- | :--- | :--- |
| **Temel Aşama** | Tüm dizi üzerinde eşzamanlı Prefill | İstemin işlendiği Prefill ve ardından adım adım Decode |
| **Hesap Profili** | Matris-Matris Çarpımı (GEMM, Compute-Bound) | Prefill: GEMM; Decode: Matris-Vektör (GEMV, Bellek Bant Genişliği Darboğazı) |
| **Bellek Ayak İzi** | Ağırlık + Gradyan + Optimizer (AdamW 16B) + Aktivasyonlar | Statik Model Ağırlıkları + Dinamik Büyüyen KV Cache |
| **Ölçekleme Darboğazı** | Düğümler Arası İletişim (All-Reduce / RingAttention) | Bellek Bant Genişliği (TPOT) ve HBM Kapasitesi (KV Cache) |
| **Paralellik Deseni** | Tensör Paralelizmi + Pipeline + Context Parallelism | Tensör Paralelizmi + Continuous Batching + PagedAttention |
| **Kritik Riskler** | Loss Spike, Gradyan NaN patlaması, OOM | Yüksek TTFT gecikmesi, KV Cache taşması, Lost in the Middle körlüğü |

### Prefill ve Decode Asimetrisi

Eğitimde ve çıkarımın ilk istem işleme (prefill) aşamasında tüm token'lar önceden bilinir. GPU tensör çekirdekleri yüksek aritmetik yoğunlukta ($> 100\text{ FLOP/bayt}$) matris-matris çarpımları (GEMM) koşturarak tam kapasiteyle çalışır.

Oysa tek tek kelime üretilen decode aşamasında sistem her adımda yalnızca 1 token üretir. Yeni token $t+1$'i üretebilmek için modelin 16 GB ağırlığının tamamı HBM'den SRAM'e çekilip tek bir vektör $[1 \times d]$ ile çarpılır. Aritmetik yoğunluk $< 1\text{ FLOP/bayt}$ seviyesine çöker; GPU saat çevrimlerinin neredeyse tamamını bellek transfer yolunda bekleyerek harcar. Bağlam penceresini uzatmak, decode aşamasında her adımda okunması gereken KV Cache'i devasa boyutlara ulaştırarak token başına üretim süresini (TPOT) feci şekilde yavaşlatır.

---

## 6. Mühendislik karar matrisi ve sınırları aşma stratejileri

Üretimde maliyet ve gecikmeyi kontrol altında tutabilmek için bağlam boyutu amaca göre seçilmelidir:

<svg viewBox="0 0 560 210" role="img" aria-label="Bağlam boyutu karar ağacı: Küçük bağlam ile büyük bağlam karşılaştırması." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="16" y="20" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Mühendislik Karar Matrisi: Bağlam Boyutu Seçimi</text>
<rect x="16" y="30" width="256" height="170" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="52" text-anchor="start" style="fill:var(--c-success);font-size:15px;font-weight:700;letter-spacing:.06em">KÜÇÜK BAĞLAM (&lt; 32K)</text>
<text x="28" y="74" text-anchor="start" style="fill:var(--c-text);font-size:13px">• Otonom Ajanlar</text>
<text x="28" y="90" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hızlı araç çağırma ve karar döngüleri</text>
<text x="28" y="112" text-anchor="start" style="fill:var(--c-text);font-size:13px">• Odaklı Soru-Cevap</text>
<text x="28" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Hassas kod düzenleme ve düşük gecikme</text>
<text x="28" y="150" text-anchor="start" style="fill:var(--c-text);font-size:13px">• Öngörülebilir Altyapı</text>
<text x="28" y="166" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Düşük TTFT ve kontrollü sunucu maliyeti</text>
<rect x="288" y="30" width="256" height="170" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="300" y="52" text-anchor="start" style="fill:var(--c-warn);font-size:15px;font-weight:700;letter-spacing:.06em">BÜYÜK BAĞLAM (128K - 2M+)</text>
<text x="300" y="74" text-anchor="start" style="fill:var(--c-text);font-size:13px">• Kod Deposu Analizi</text>
<text x="300" y="90" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Tüm mimariyi tek seferde inceleme</text>
<text x="300" y="112" text-anchor="start" style="fill:var(--c-text);font-size:13px">• Belge Sentezi</text>
<text x="300" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Çok ciltli sözleşme ve rapor tarama</text>
<text x="300" y="150" text-anchor="start" style="fill:var(--c-text);font-size:13px">• Kritik Riskler</text>
<text x="300" y="166" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Lost in the middle körlüğü ve yüksek TCO</text>
</svg>

### Sınırları Aşmak İçin Güncel Mimari Stratejiler

Her problemi bağlam penceresini büyüterek çözmeye çalışmak en pahalı ve kırılgan mühendislik hatasıdır. Başarılı sistemler üç filtreleme stratejisini birleştirir:

1. **Hassas Bilgi Erişimi (RAG - Retrieval-Augmented Generation):** Tüm kütüphaneyi isteme yığmak yerine, dökümanları yoğun vektörler ve seyrek BM25 algoritmalarıyla indeksleyip yalnızca ilgili 5 parçayı bağlama vermek; gecikmeyi ve faturayı %90 düşürür.
2. **Dinamik Bağlam Sıkıştırma (LLMLingua / Prompt Pruning):** Bilgi kuramı doğal dilde yüksek bir fazlalık olduğunu gösterir. Küçük bir ön model metindeki anlamsal entropiyi ölçerek anlamsız bağlaç ve dolgu kelimelerini submitted prompt'tan temizler; anlamı koruyarak bağlamı 4 kat sıkıştırır.
3. **Anlamsal ve Ön Ek Önbellekleme (Radix Attention / Prefix Caching):** Çok turlu sohbetlerde veya ajan döngülerinde değişmeyen sistem talimatları ve araç tanımları ağaç tabanlı KV önbelleğinde tutulur; her turda bu ön ekin baştan hesaplanması önlenerek TTFT milisaniyeler seviyesine indirilir.

---

## 7. Tarihsel evrim ve mimari özet tablosu

| Parametre | 1. Faz (2017–2022) | 2. Faz (2023–2025) | 3. Faz (2026+) |
| :--- | :--- | :--- | :--- |
| **Bağlam Kapasitesi** | 512 – 2.048 Token (2k) | 32k – 1 Milyon Token (128k) | 1 Milyon – 10+ Milyon Token (10M+) |
| **Konumsal Yaklaşım** | Mutlak Konum (APE) | RoPE + YaRN Frekans Enterpolasyonu | Hibrit (Attention için RoPE + Örtük SSM Durumu) |
| **Donanım Çözümü** | Standart HBM Matmul (Bellek Kilitlenmesi) | FlashAttention 1/2/3 (Tiling + Çekirdek Füzyonu) | Seyreltik KV Önbellek + RingAttention + FP4 Tensörler |
| **Kritik Mimari Kazanım**| Self-Attention Temsil Gücü | Grouped-Query Attention (GQA) | Hibrit Transformer-Mamba Topolojileri ($O(N)$ Prefill) |
| **Ana Darboğaz** | Anında CUDA OOM Çökmeleri | Lost in the Middle & Prefill TTFT Gecikmesi | Enerji Tüketimi ve Toplam Sahip Olma Maliyeti (TCO) |
| **Sistemik Rolü** | Biletli Koltuk & Evrak Yığını | Akordeon Masa & Tezgahta Pişiren Şef | Akışkan Zihin & Bilişsel Unutma Refleksi |

---

## 8. PyTorch ve Python ile doğrulama: Online Softmax ve hibrit durum güncellemesi

Aşağıdaki çalıştırılabilir Python betiği 4. Bölümde yürütülen iki matematiksel omurgayı modeller:
1. FlashAttention Online Softmax blok birleştirmesini standart Softmax ile karşılaştıran nümerik doğrulama.
2. Seçici Durum Uzayı Modelinin (SSM / Mamba) sabit bellekli $O(1)$ durum güncellemesi.

```python
import math

def verify_context_mechanics():
    print("=== 1. FlashAttention Online Softmax vs Standard Softmax ===")
    # 2'ser blokluk 4 skor
    s_block1 = [2.0, 4.0]
    v_block1 = [[1.0, 0.0], [0.0, 2.0]]

    # 1. Adim: Blok 1 SRAM icinde
    m1 = max(s_block1)
    p1_unnorm = [math.exp(x - m1) for x in s_block1]
    l1 = sum(p1_unnorm)
    o1 = [sum(p1_unnorm[i] * v_block1[i][j] for i in range(2)) for j in range(2)]
    print(f"Blok 1: m={m1:.4f}, l={l1:.6f}, O_acc={[round(x, 6) for x in o1]}")

    # 2. Adim: Blok 2 ve Cevrimici Olcekleme
    s_block2 = [3.0, 5.0]
    v_block2 = [[2.0, 1.0], [1.0, 3.0]]

    m2_tilde = max(s_block2)
    m2 = max(m1, m2_tilde)
    p2_unnorm = [math.exp(x - m2) for x in s_block2]
    l2_tilde = sum(p2_unnorm)

    # Cevrimici duzeltme katsayisi
    scale = math.exp(m1 - m2)
    l2 = scale * l1 + l2_tilde
    o2_part = [sum(p2_unnorm[i] * v_block2[i][j] for i in range(2)) for j in range(2)]
    o2 = [scale * o1[j] + o2_part[j] for j in range(2)]
    final_o = [o2[j] / l2 for j in range(2)]
    print(f"Blok 2: m={m2:.4f}, l={l2:.6f}, final_O={[round(x, 6) for x in final_o]}")

    # Standart Softmax ile saglama
    s_all = [2.0, 4.0, 3.0, 5.0]
    v_all = v_block1 + v_block2
    m_all = max(s_all)
    exp_all = [math.exp(x - m_all) for x in s_all]
    sum_exp = sum(exp_all)
    p_std = [x / sum_exp for x in exp_all]
    std_o = [sum(p_std[i] * v_all[i][j] for i in range(4)) for j in range(2)]
    print(f"Standart Softmax final_O:    {[round(x, 6) for x in std_o]}")

    diff = math.sqrt(sum((final_o[j] - std_o[j])**2 for j in range(2)))
    print(f"Fark (Euclidean Distance):   {diff:.2e}")

    print("\n=== 2. Mamba/SSM Sabit Durum Guncellemesi (O(1) Bellek Ayak Izi) ===")
    # h_t = A_bar * h_{t-1} + B_bar * x_t,  y_t = C * h_t
    A_bar = [[0.8, 0.0], [0.0, 0.5]]
    B_bar = [0.5, 1.0]
    C = [1.0, -0.5]
    h = [0.0, 0.0]
    x_tokens = [1.0, 0.5, 2.0, -1.0]

    for t, x in enumerate(x_tokens):
        h = [
            A_bar[0][0] * h[0] + B_bar[0] * x,
            A_bar[1][1] * h[1] + B_bar[1] * x
        ]
        y = C[0] * h[0] + C[1] * h[1]
        print(f"Adim t={t+1} (x={x:4.1f}) -> h_t={[round(v, 4) for v in h]}, y_t={y:6.4f}")

if __name__ == "__main__":
    verify_context_mechanics()
```

Konsol çıktısı:

```text
=== 1. FlashAttention Online Softmax vs Standard Softmax ===
Blok 1: m=4.0000, l=1.135335, O_acc=[0.135335, 2.0]
Blok 2: m=5.0000, l=1.553002, final_O=[0.850262, 2.492653]
Standart Softmax final_O:    [0.850262, 2.492653]
Fark (Euclidean Distance):   4.97e-16

=== 2. Mamba/SSM Sabit Durum Guncellemesi (O(1) Bellek Ayak Izi) ===
Adim t=1 (x= 1.0) -> h_t=[0.5, 1.0], y_t=0.0000
Adim t=2 (x= 0.5) -> h_t=[0.65, 1.0], y_t=0.1500
Adim t=3 (x= 2.0) -> h_t=[1.52, 2.5], y_t=0.2700
Adim t=4 (x=-1.0) -> h_t=[0.716, 0.25], y_t=0.5910
```

---

## Bütün hikâye altı satırda

- Bağlam penceresi soyut bir sayı değil, modelin durumsuz (*stateless*) çalışma masasıdır; bu masa $O(N^2)$ karesel işlem yükü, GPU bellek duvarı (HBM darboğazı) ve döküman ortasını yutan dikkat sisiyle (Lost in the Middle) sınırlıdır.
- 2017'nin sabit indisli mutlak pozisyon gömmeleri (APE), modeli biletli bir koltuk gibi 2.048 token'a kilitledi; matriste olmayan pozisyonlara genelleme yapamamak bağlamın ilk sert mimari tavanı oldu.
- Asıl darboğaz FLOPs değil bellek bant genişliğiydi; FlashAttention, Online Softmax algoritmasıyla ara matrisleri çip içi SRAM'de bloklayıp (tiling) HBM transferlerini $O(N^2)$'den $O(N)$'e indirerek silikon duvarını yıktı.
- Pozisyonu sabit koordinat yerine 2B açılara (RoPE) çevirmek ve YaRN ile frekans dalga boylarını sıkıştırmak, modelleri sıfırdan eğitmeden bağlamı 128k'ya esneten "akordeon masa" illüzyonunu yarattı.
- Pazarlanan 1M token (MCW), çıkarım anında 16–32 GB VRAM yutan KV Cache faturasına ve dikkat seyrelmesine toslar; üretim sistemlerinde güvenilir muhakeme tavanı (MECW) hâlâ 64k–128k aralığındadır.
- 10 milyon token'a giden yol saf dikkati zorlamak değil; geçmişi $O(1)$ bellekle özetleyen Mamba-2 durum uzayları, seyrek çapraz dikkat ve insan beyni gibi önemsiz detayları tahliye eden (Sparse KV Cache) 2026 hibrit topolojileridir.

---

## Terimler sözlüğü

- **Bağlam Penceresi (Context Window)** — Dil modelinin geçmiş verileri kırpmadan tek bir ileri geçişte eşzamanlı olarak işleyebildiği maksimum token dizilimi kapasitesi.
- **Bellek Duvarı (Memory Wall)** — İşlemci çekirdeklerinin hesaplama hızı ile çip dışı HBM ve SRAM arasındaki veri aktarım bant genişliği arasındaki asimetriden doğan donanım darboğazı.
- **Çekirdek Füzyonu (Kernel Fusion)** — Birden çok tensör işlemini tek bir GPU çekirdeğinde birleştirerek ara tensörlerin HBM'e yazılıp geri okunmasını engelleyen derleyici optimizasyonu.
- **Çevrimiçi Softmax (Online Softmax)** — Tüm diziyi bellekte tutmadan, gelen veri blokları üzerinden satır maksimumunu ve üstel toplamı dinamik güncelleyen akış normalizasyonu algoritması.
- **Grup Sorgulu Dikkat (Grouped-Query Attention - GQA)** — Birden çok sorgu kafasının tek bir anahtar-değer kafasını ortaklaşa kullanmasını sağlayarak KV Cache boyutunu küçülten dikkat mimarisi.
- **YaRN (Yet another RoPE extensioN)** — Döner konumsal gömme frekanslarını dalga boylarına göre farklı ölçekleme bantlarına ayırarak bağlamı esneten enterpolasyon yöntemi.
- **Ortada Kaybolma (Lost in the Middle)** — Dil modellerinin uzun istemlerin başındaki ve sonundaki bilgileri kusursuz hatırlayıp döküman ortasındaki detaylara körleşmesi olgusu.
- **Seçici Durum Uzayı Modelleri (SSM / Mamba)** — Dizileri doğrusal $O(N)$ karmaşıklıkta ve sabit $O(1)$ bellek ayak izine sahip gizli durum vektörleriyle işleyen ardışıl mimari.

---

## Daha derine inmek için

- [Vaswani et al. (NeurIPS 2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ölçeklenmiş skaler çarpım dikkat mekanizmasını kuran orijinal Transformer makalesi.
- [Su et al. (2021): RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Döner konumsal gömme (RoPE) mekanizmasının kurucu temelleri.
- [Dao et al. (NeurIPS 2022): FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — GPU SRAM tiling, çekirdek füzyonu ve Online Softmax'ı sunan öncü çalışma.
- [Peng et al. (ICLR 2024): YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — RoPE frekans enterpolasyonunun en gelişmiş standart formülasyonu.
- [Liu et al. (TACL 2023): Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Uzun bağlamlarda dikkat çürümesini ampirik olarak kanıtlayan seminal araştırma.
- [Gu & Dao (2023): Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — $O(1)$ bellek ayak iziyle çıkarım yapabilen seçici durum uzayı mimarisinin temelleri.
- Bu blogda: [Bir Prompt'un Yolculuğu (4): Self-Attention Derinlemesine](post.html?slug=self-attention-derinlemesine) — Skaler çarpım dikkatinin ve izdüşüm matrislerinin matematiksel temelleri.
- Bu blogda: [Bir Prompt'un Yolculuğu (6): Multi-Head Attention Derinlemesine](post.html?slug=multi-head-attention-derinlemesine) — Çok başlıklı dikkat, Multi-Query Attention ve GQA mekanizmaları.
- Bu blogda: [Bir Prompt'un Yolculuğu (8): Çıktı Katmanı (LM Head) Derinlemesine](post.html?slug=cikti-katmani-derinlemesine) — Kelime dağarcığı izdüşümü, logitler ve donanımsal çıkarım gecikmesi.
- Bu blogda: [Üretimde LLM Sistemleri (2): vLLM Derinlemesine](post.html?slug=vllm-derinlemesine) — PagedAttention, sürekli yığınlama ve ön ek önbellekleme mimarisi.
