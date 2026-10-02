Pazartesi günü kullanıcı asistanınıza *"Geçen hafta Berlin'e taşındım"* der. Cumartesi günü ise evine yakın bir hafta sonu yürüyüş rotası sorduğunda, asistan en ufak bir tereddüt yaşamadan İstanbul'un dışındaki Belgrad Ormanı parkurunu önerir. Sistemde hiçbir bileşen çökmemiştir: Vektör araması kusursuz çalışmış, embedding kosinüs benzerliği 0,88 çıkmış, dil modeli kusursuz bir gramerle akıl yürütmüştür. Hafıza deposu da kendisinden isteneni harfiyen yapmış; hem *"İstanbul'da yaşıyor"* hem de *"Berlin'e taşındı"* kayıtlarını saklamış, ancak geçmiş oturumlarda yüzlerce kez çağrıldığı için eski kaydı öne çıkarmıştır. Ajan unutmamıştır; çok daha tehlikeli bir şey yapmış, **kendinden emin biçimde yanlış hatırlamıştır**.

Bu arıza, yapay zekanın en temel ontolojik gerçeğini yüzümüze çarpar: Büyük dil modelleri doğaları gereği **durumsuzdur (stateless)**. 70 milyar parametreli bir model, Shakespeare kalitesinde felsefe yapabilir, karmaşık kuantum algoritmaları türetebilir; ancak çıkarım (inference) çağrısı bittiği anda GPU çekirdekleri sıfırlanır ve model ağırlıklarında kullanıcıya dair tek bir bit dahi kalmaz. Bir sonraki saniyede gelen kullanıcı, model için evrende ilk kez karşılaştığı bir yabancıdır.

Bu durumsuzluk krizini çözmek için akla gelen iki geleneksel mühendislik refleksi de üretim ortamında duvara toslar:
1. **"Bağlam penceresi 1 milyona çıktı, her şeyi prompt'a dolduralım" refleksi:** LOCOMO benchmark'ında bu saf yaklaşım, soru başına ortalama 26.000 token tüketir ve p95 gecikmesini 17 saniyenin üzerine fırlatır. Daha da vahimi, diyalog geçmişi 115.000 token seviyesine ulaştığında GPT-4o gibi amiral gemisi modeller, doğru kanıt prompt'un tam ortasında dururken bile LongMemEval testinde %87,0 doğruluktan %60,6'ya geriler. Literatür buna **Context Rot (Bağlam Çürümesi)** ve **Lost in the Middle (Ortada Kaybolma)** der; dikkat mekanizması uçsuz bucaksız metin okyanusunda kendi geçmiş gürültüsünün içinde boğulur.
2. **"Her şeyi vektör veritabanına atıp kosinüsle arayalım" refleksi:** Gecikmeyi 1,4 saniyeye indirse de, zaman boyutundan, varlık çözümlemesinden (entity resolution) ve ilişkisel graf mantığından yoksundur. Vektör araması için *"İstanbul'da yaşıyorum"* ile *"Berlin'de yaşıyorum"* aynı semantik uzayda yüzen iki benzer cümleden ibarettir; hangisinin dün, hangisinin bugün geçerli olduğunu ayırt edemez.

Modern yapay zeka sistemlerinde hafıza; basit bir anahtar-değer tablosu, bir vektör veritabanı ya da sonuna ekleme yapılan bir metin kütüğü değildir. Hafıza; durumsuz bir modeli çevreleyen, neyin saklanmaya değer olduğunu süzen bir **yazma yolu (write path)**, sorgu anında doğru olguları cımbızlayan bir **okuma yolu (read path)**, oturumlar arasında çelişkileri ayıklayan bir **rüya / konsolidasyon (dreaming)** mekanizması ve çok ajanlı ortamlarda veri sızıntısını engelleyen bir **güvenlik yönetişim katmanıdır**.

Bu yazı; Anthropic'in *Context Engineering* manifestosundan NVIDIA NeMo Agent Toolkit'in modüler soyutlamalarına, Hugging Face `smolagents` durum yönetiminden Google Scholar ve ICLR/NeurIPS literatürünün en güncel bulgularına (CoALA bilişsel taksonomisi, MemGPT sanal bellek hiyerarşisi, Memory-R1 pekiştirmeli öğrenme hafıza yöneticisi, Zep bi-temporal zaman grafı, HippoRAG hipokampal dizinlemesi ve ACE bağlam evrimi) uzanan eksiksiz bir sistem mühendisliği kılavuzudur. Bağlam penceresinin fiziksel sınırlarını, Ebbinghaus sönümlenme matematiğini, GPU tarafındaki KV cache faturasını ($2 \times L \times H_{kv} \times d_{\text{head}} \times b$), hafıza zehirlenmesi açıklarını (AgentPoison, MINJA) ve Berlin hatasını kökünden çözen referans bir Python motorunu adım adım inceliyoruz.

**Bu yazıda**

- [1. Model unutur, döngü hatırlar](#1-model-unutur-döngü-hatırlar)
- [2. Masa: çalışma belleği olarak bağlam penceresi](#2-masa-çalışma-belleği-olarak-bağlam-penceresi)
  - [FIFO ve token bütçesi](#fifo-ve-token-bütçesi)
  - [Daha büyük bir masa neden çözüm değil](#daha-büyük-bir-masa-neden-çözüm-değil)
  - [Sıkıştırma ve bellek baskısı: MemGPT ve Anthropic Compaction](#sıkıştırma-ve-bellek-baskısı-memgpt-ve-anthropic-compaction)
- [3. Bilişsel taksonomi: Dört hafıza türü](#3-bilişsel-taksonomi-dört-hafıza-türü)
- [4. Yazma yolu: neyin saklanacağına karar vermek](#4-yazma-yolu-neyin-saklanacağına-karar-vermek)
  - [Ayıklama ve durum geçişleri: ADD, UPDATE, DELETE, NOOP](#ayıklama-ve-durum-geçişleri-add-update-delete-noop)
  - [Önem ve reflection](#önem-ve-reflection)
  - [Üzerine yazma, geçersiz kıl: İki zamanlı model](#üzerine-yazma-geçersiz-kıl-i̇ki-zamanlı-model)
  - [Unutmayı tasarlamak: Ebbinghaus sönümlenmesi](#unutmayı-tasarlamak-ebbinghaus-sönümlenmesi)
  - [Yönetişim kapısı ve güvenlik](#yönetişim-kapısı-ve-güvenlik)
  - [Sıcak yol mu, arka plan mı](#sıcak-yol-mu-arka-plan-mı)
- [5. Okuma yolu: erişim bir boru hattıdır](#5-okuma-yolu-erişim-bir-boru-hattıdır)
  - [Hafızaya bakmalı mıyız? İhtiyaç tespiti ve Just-in-Time](#hafızaya-bakmalı-mıyız-i̇htiyaç-tespiti-ve-just-in-time)
  - [Sorguyu yeniden yazmak ve HyDE](#sorguyu-yeniden-yazmak-ve-hyde)
  - [Bir anıyı puanlamak: yakınlık, önem, ilgi](#bir-anıyı-puanlamak-yakınlık-önem-ilgi)
  - [Formülde adım adım yürüyüş](#formülde-adım-adım-yürüyüş)
  - [Hibrit erişim ve RRF](#hibrit-erişim-ve-rrf)
  - [Paketleme: köken, konum ve önbellek](#paketleme-köken-konum-ve-önbellek)
- [6. Oturumlar arası birleştirme: Sleep-time compute ve dreaming](#6-oturumlar-arası-birleştirme-sleep-time-compute-ve-dreaming)
  - [Sleep-time compute ekonomisi](#sleep-time-compute-ekonomisi)
  - [Üretimde dreaming: Anthropic ve OpenAI](#üretimde-dreaming-anthropic-ve-openai)
  - [Bağlam çöküşü tuzağı ve ACE mimarisi](#bağlam-çöküşü-tuzağı-ve-ace-mimarisi)
- [7. Sekiz mimari, sekiz ödünleşim](#7-sekiz-mimari-sekiz-ödünleşim)
- [8. Çok ajanlı hafıza: depodan önce kapsam](#8-çok-ajanlı-hafıza-depodan-önce-kapsam)
  - [Kim nereye yazabilir: Politika matrisi](#kim-nereye-yazabilir-politika-matrisi)
  - [Paylaşılan hafızanın altı arıza biçimi](#paylaşılan-hafızanın-altı-arıza-biçimi)
  - [Hafıza zehirlenmesi ve dolaylı enjeksiyon](#hafıza-zehirlenmesi-ve-dolaylı-enjeksiyon)
  - [Depoya doğrudan erişmeden zehirlemek: MINJA](#depoya-doğrudan-erişmeden-zehirlemek-minja)
- [9. İki mühendislik gözü: eğitim ve çıkarım](#9-i̇ki-mühendislik-gözü-eğitim-ve-çıkarım)
  - [Eğitim Gözü: Parametrik hafıza neden yetersizdir?](#eğitim-gözü-parametrik-hafıza-neden-yetersizdir)
  - [Çıkarım Gözü: Hafızanın bedeli prefill ve KV Cache'te ödenir](#çıkarım-gözü-hafızanın-bedeli-prefill-ve-kv-cachete-ödenir)
- [10. Üretimde hafıza bir sistemdir](#10-üretimde-hafıza-bir-sistemdir)
  - [Asgari API](#asgari-api)
  - [Sıcak, ılık, soğuk katmanlama](#sıcak-ılık-soğuk-katmanlama)
  - [Kiracıları yalıtmanın üç yolu](#kiracıları-yalıtmanın-üç-yolu)
  - [p95 nereye gidiyor: Gecikme bütçesi](#p95-nereye-gidiyor-gecikme-bütçesi)
  - [Embedding modelini değiştirdiğiniz gün](#embedding-modelini-değiştirdiğiniz-gün)
  - [Ölçmek: beş temel yetenek](#ölçmek-beş-temel-yetenek)
  - [Benchmark gerçekliği: LoCoMo vs LongMemEval](#benchmark-gerçekliği-locomo-vs-longmemeval)
- [11. Bütün döngü 60 satır Python'da](#11-bütün-döngü-60-satır-pythonda)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Model unutur, döngü hatırlar

Yapay zeka asistanlarının "hatırlaması", biyolojik bir sinir ağının sinaptik ağırlıklarını güncellemesiyle değil; yazılım mühendisliğinin durumsuz bir fonksiyon etrafında kurduğu **durum koruma döngüsüyle (stateful loop)** gerçekleşir.

Bu mekanizmayı zihnimizde somutlaştırmak için bir çalışma odası hayal edelim:
Masanın başında oturan dahi bir danışman vardır. Bu danışman her sabah masaya tam bir hafıza kaybıyla oturur. Ne dünkü müşterilerini hatırlar ne de tamamlanan projeleri bilir. Danışmanın tüm dünyası, o an masasının üzerine bırakılmış evraklardan ibarettir.
- **Danışman:** Dil modelidir (LLM). Akıl yürütür, sentez yapar, dil üretir; ancak hafızası sıfırdır.
- **Masa:** Bağlam penceresidir (Context Window). Fiziksel bir yüzeydir; alanı sınırlıdır, üzerine çok fazla evrak yığarsanız danışman aradığı kâğıdı bulamaz.
- **Gelen Evrak Tepsisi:** Kullanıcının o an gönderdiği güncel mesajdır ($q_t$).
- **Arşiv Odası:** Modelin dışındaki kalıcı uzun süreli hafıza deposudur ($\mathcal{M}$).
- **Dosya Memuru:** Konuşma akarken neyin kalıcı arşive kaydedileceğini seçen yazma yoludur (`Write`).
- **Kütüphaneci:** Gelen yeni soruya göre arşivden en hayati üç dosyayı seçip masaya koyan okuma yoludur (`Retrieve`).

Eğer danışman dünkü kararları biliyor gibi konuşuyorsa, bu danışmanın dehası değil; kütüphanecinin sabah masaya bıraktığı özet dosyasının başarısıdır.

<svg viewBox="0 0 560 268" role="img" aria-label="Ajan hafızasının çalışma zamanı döngüsü. Kullanıcı mesajı q_t okuma yoluna girer; okuma yolu uzun süreli hafıza deposu M&#x27;yi de okur ve ihtiyaç tespiti, yeniden yazım, hibrit arama, RRF ve filtre adımlarını çalıştırır. Okuma yolu dinamik bir prompt kurar: sistem prompt&#x27;u, ilgili anılar, son turlar ve q_t. Durumsuz model cevabı, a_t&#x27;yi üretir. Model ve cevap yazma yoluna akar; yazma yolu olguları ayıklar, bi-temporal güncelleme yapar, maskeler, PII kapısından geçirir ve depoya geri yazar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="aml-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="aml-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
<marker id="aml-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="18" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:600">Çalışma zamanı döngüsü</text>
<rect x="16" y="30" width="104" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="68.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Kullanıcı</text>
<text x="68.0" y="70.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">mesajı q_t</text>
<rect x="140" y="30" width="194" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="237.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Okuma yolu</text>
<text x="237.0" y="62.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">ihtiyaç · yeniden yazım ·</text>
<text x="237.0" y="78.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">hibrit arama · RRF · filtre</text>
<rect x="354" y="30" width="190" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="449.0" y="46.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Dinamik prompt</text>
<text x="449.0" y="62.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">sistem + ilgili anılar</text>
<text x="449.0" y="78.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+ son turlar + q_t</text>
<rect x="354" y="116" width="190" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="449.0" y="140.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Durumsuz model (LLM)</text>
<rect x="354" y="192" width="190" height="40" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="449.0" y="216.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Cevap a_t</text>
<rect x="140" y="186" width="194" height="56" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="237.0" y="202.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Yazma yolu</text>
<text x="237.0" y="218.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">ayıkla · bi-temporal güncelle</text>
<text x="237.0" y="234.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">maskele · PII kapısı</text>
<path d="M16 111 V161 A52.0 7 0 0 0 120 161 V111" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="68.0" cy="111" rx="52.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="68.0" y="137.5" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Hafıza</text>
<text x="68.0" y="152.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">deposu M</text>
<line x1="120" y1="58" x2="138" y2="58" marker-end="url(#aml-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="334" y1="58" x2="352" y2="58" marker-end="url(#aml-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="449" y1="86" x2="449" y2="114" marker-end="url(#aml-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="449" y1="156" x2="449" y2="190" marker-end="url(#aml-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<path d="M120 136 H237 V88" marker-end="url(#aml-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="354" y1="212" x2="336" y2="212" marker-end="url(#aml-w)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M354 136 H300 V184" marker-end="url(#aml-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<path d="M140 214 H68 V170" marker-end="url(#aml-w)" style="fill:none;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="16" y="260" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Model durumsuz kalır; hafıza depoda ve onu saran iki yolda yaşar.</text>
</svg>

Matematiksel olarak $t$ anındaki bir ajan etkileşimi, prompt'u inşa eden şu erişim fonksiyonuyla başlar:

$$\text{prompt}_t = \text{System} \;\oplus\; \text{Retrieve}(q_t, \mathcal{M}_t) \;\oplus\; \text{History}_{t-k:t} \;\oplus\; q_t$$

Şöyle okuyun: *$t$ anındaki prompt; sabit sistem talimatları, uzun süreli depo $\mathcal{M}_t$'den o anki kullanıcı girdisi $q_t$ için getirilen dinamik anılar, son $k$ turun ham diyaloğu ve güncel kullanıcı mesajının arka arkaya eklenmesiyle ($\oplus$) oluşturulur.*

Model cevabı ($a_t$) ürettikten sonra, sistemin durumunu bir sonraki tura aktaran ikinci durum geçiş fonksiyonu tetiklenir:

$$\mathcal{M}_{t+1} = \text{Write}\big(\mathcal{M}_t,\; q_t,\; a_t\big)$$

Şöyle okuyun: *Bir sonraki durumun hafıza deposu $\mathcal{M}_{t+1}$; eski deponun, son etkileşim çiftinden ($q_t, a_t$) ayıklanan yeni olgular, güncellenen inançlar ve geçersiz kılınan eski kayıtlarla yeniden düzenlenmiş hâlidir.*

Bu iki denklem, ajan mühendisliğinin iki temel sorusunu belirler: `Retrieve` fonksiyonu masaya hangi belgeleri koymalıdır ve `Write` fonksiyonu arşivdeki çelişkileri nasıl yönetmelidir?

---

## 2. Masa: çalışma belleği olarak bağlam penceresi

### FIFO ve token bütçesi

Hafıza sistemi kurmanın en ilkel yolu, hiçbir altyapı kurmamaktır: Tüm konuşma geçmişini körlemesine her yeni prompt'un sonuna eklemek. Konuşma kısa olduğu sürece bu yöntem kusursuz çalışır; çünkü model geçmişteki her kelimeye Self-Attention mekanizmasıyla doğrudan erişebilir. Ancak fiziksel bir sınır vardır: Modelin bağlam penceresi ($T_{\text{pencere}}$).

Bir çıkarım çağrısında konuşma geçmişine kalan gerçek kullanılabilir alan şu bütçeyle sınırlıdır:

$$T_{\text{geçmiş}} = T_{\text{pencere}} - T_{\text{sistem}} - T_{\text{araçlar}} - T_{\text{anılar}} - T_{\text{üretim}}$$

Şöyle okuyun: *Konuşma geçmişine ayrılabilecek azami token sayısı; toplam pencereden sistem prompt'u, araç (tool) şemaları, uzun süreli depodan getirilen anılar ve modelin üreteceği cevap için rezerve edilen pay çıkarıldıktan sonra kalan alandır.*

Somut bir 8.192 token'lık bağlam bütçesi düşünelim:

```text
T_pencere   = 8.192 token (Modelin fiziksel sınırı)
T_sistem    = 1.500 token (Persona, operasyon kuralları)
T_araçlar   = 1.200 token (JSON Schema API tanımları)
T_anılar    =   800 token (Geri getirilen uzun süreli anılar)
T_üretim    = 1.000 token (Modelin üreteceği cevabın tavanı)
-----------------------------------------------------------------
T_geçmiş    = 8.192 - (1.500 + 1.200 + 800 + 1.000) = 3.692 token
```

Konuşma dökümü 3.692 token'ı aştığında sistem bir şeyleri masadan atmak zorundadır. En yaygın naif strateji **FIFO (First-In, First-Out)** politikasıdır: En eski mesaj ilk silinir. Ancak FIFO, insan iletişimi için tasarlanabilecek en talihsiz kuraldır. Kullanıcılar en temel, değişmez kurallarını ve tercihlerini konuşmanın en başında söyler (*"Ben vejetaryenim"*, *"Veritabanımız PostgreSQL"*); bu tercihlere dayanan kritik soruları ise saatler veya günler sonra sorarlar. FIFO politikası masanın arkasından tam da bu en hayati kök bilgileri süpürüp atar.

### Daha büyük bir masa neden çözüm değil

Gemini, Claude ve LLaMA ailelerinin bağlam pencerelerini 128k, 1M hatta 2M token seviyesine çıkarması, mühendislerde *"Artık hafıza mimarisine gerek kalmadı, her şeyi bağlama doldurabiliriz"* yanılsaması yarattı. Laboratuvar ölçümleri bu varsayımın üç yönden çöktüğünü kanıtlamaktadır:

1. **Lost in the Middle (Liu et al., Stanford/Berkeley, 2023):** Transformer mimarisinin Softmax paydası uzadıkça, dikkat mekanizması uç noktalara (en başa ve en sona) aşırı odaklanırken bağlamın ortasında kalan bilgileri gözden kaçırır. Çoklu belge soru-cevaplama testlerinde doğru bilgi ortaya yerleştirildiğinde model başarı oranları 20 ila 30 mutlak puan birden düşmektedir.
2. **Context Rot ve LongMemEval Çöküşü (ICLR 2025):** Wu ve arkadaşlarının geliştirdiği LongMemEval benchmark'ında, 115.000 token uzunluğunda gerçek sohbet geçmişi verildiğinde GPT-4o'nun doğruluk oranı %87,0'dan %60,6'ya gerilemiştir. Kanıt metnin içinde birebir durmasına rağmen, model dikkat dağılması yüzünden %30 bağıl performans kaybetmiştir. Anthropic'in *Context Engineering* raporunda vurguladığı gibi: Girdi uzadıkça ve alakasız geçmiş sohbet turları biriktikçe, geçmişteki her bir detay bir **çeldirici (distractor)** hâline gelir ve akıl yürütme kalitesini çürütür.
3. **KV Cache Donanım Maliyeti:** Bağlam penceresine atılan her token, model tek bir kelime üretmeden önce GPU belleğinde yerleşik KV cache olarak saklanır. LLaMA-3-8B için token başına düşen KV cache faturası 128 KiB'dir (Bölüm 9'da hesaplanmıştır). 100.000 token'lık bir sohbet geçmişi, kullanıcı başına anında **12,8 GiB VRAM** kilitler. Milyonlarca kullanıcıya hizmet veren bir sistemde bu yaklaşım finansal bir iflastır.

Bağlam penceresi ile gerçek bir veritabanı arasındaki ontolojik farkı kavramak şarttır:

| Veritabanı Kabiliyeti | Bağlam Penceresi Gerçeği | Mimari Çözüm |
|---|---|---|
| **Sorgu Optimizasyonu** | Her çıkarımda baştan sona tam tarama (Full Scan) | İhtiyaç tespiti ve hedefli erişim (Bölüm 5) |
| **Dizinleme (Indexing)** | Yalnızca ardışık token konumu | Vektör, BM25 ve zamansal bilgi grafı (Bölüm 5) |
| **Erişim Denetimi (RBAC)** | Prompt'a giren her şey modele tamamen açıktır | Kapsam izolasyonu ve kiracı filtreleri (Bölüm 8) |
| **Yaşam Süresi (TTL)** | Pencereden düşene kadar körlemesine tutulur | Ebbinghaus sönümlenme eğrisi (Bölüm 4) |
| **Çelişki Çözümü** | İki zıt iddia yan yana durur ve modeli kilitler | Bi-temporal geçersiz kılma (Bölüm 4) |

### Sıkıştırma ve bellek baskısı: MemGPT ve Anthropic Compaction

FIFO'nun bağlam kaybı sorununu hafifletmek için literatür iki aşamalı sıkıştırma modellerine yönelmiştir.

**MemGPT (Packer et al., UC Berkeley, 2023):** İşletim sistemlerinin sanal bellek (virtual memory) mimarisini LLM'lere uyarlamıştır. MemGPT bağlam penceresini üç bölüme ayırır:
- **Salt Okunur Sistem:** Sabit kurallar.
- **Çalışma Bağlamı (Working Context):** Modelin özel fonksiyon çağrılarıyla (`core_memory_append`, `core_memory_replace`) doğrudan yazıp düzenleyebildiği kısıtlı bir RAM alanı.
- **FIFO Mesaj Kuyruğu:** Konuşma akışı.

Sistem, token sayacı belirli eşiklere ulaştığında işletim sistemlerindeki sayfa değiştirme (paging) mantığını taklit eden sinyaller üretir:

| Eşik | Tetiklenen Sistem Davranışı |
|---|---|
| **%70 Bellek Basıncı (Memory Pressure)** | Sisteme gizli bir uyarı mesajı enjekte edilir: *"Bağlam doluyor, önemli bilgileri çalışma hafızasına veya kalıcı arşive aktar."* Model bu uyarıyla kendi kendini özetler. |
| **%100 Tahliye Eşiği (Eviction)** | Kuyruk yöneticisi en eski turları (pencerenin ~%50'si) dışarı atar, bunları özetleyip özet bloğuna katar ve ham turları diskteki `recall storage` veritabanına gönderir. |

**Anthropic Compaction:** Claude tabanlı kodlama ve araştırma ajanlarında, bağlam belirli bir doygunluğa ulaştığında sunucu tarafında çalışan otonom bir sıkıştırma adımıdır. Eski diyalog turları yapılandırılmış bir ara duruma indirgenir; ancak bu özetlemenin tehlikeli bir arıza biçimi vardır: **Context Collapse (Bağlam Çöküşü)**. Bir özetin özetini aldığınızda, model genel prensipleri korurken asıl işe yarayan küçük hata kodlarını ve uç durum değişkenlerini kaybeder.

---

## 3. Bilişsel taksonomi: Dört hafıza türü

Bilişsel psikolojinin bir asırlık birikimi, hafızanın tek parça olmadığını kanıtlamıştır. **CoALA çerçevesi (Cognitive Architectures for Language Agents, Sumers et al., TMLR 2024)** bu bilişsel mimariyi yapay zeka ajanlarına uyarlamıştır. Günümüzde Hugging Face `smolagents` ve NVIDIA NeMo Agent Toolkit gibi gelişmiş sistemler de bu dörtlü ayrımı temel alır:

| Hafıza Türü | Tanım ve İçerik | Yazma Politikası | Okuma Mekanizması | Fiziksel Depo |
|---|---|---|---|---|
| **Çalışma Belleği (Working Memory)** | Anlık turdaki görev, modelin düşünce zinciri (scratchpad), çağrılan araçların ham dönüşleri. | Her LLM çağrısında dinamik kurulur. | Doğrudan modelin aktif bağlam penceresindedir. | Aktif Prompt / RAM |
| **Epizodik Hafıza (Episodic Memory)** | *"Ne oldu ve ne zaman oldu?"* Ajanın geçmiş oturumları, izlediği eylem dizileri, başarı ve başarısızlık günlükleri. | Oturum veya görev tamamlandığında eklenir. | Zaman damgası ve benzerlik filtreli vektör araması. | Olay günlüğü (Event Log) + Vektör Dizini |
| **Anlamsal Hafıza (Semantic Memory)** | *"Dünya ve kullanıcı hakkında ne biliyorum?"* Olgu ve kurallar: Tercihler, adresler, teknik mimari kısıtları. | Ayıklama, tekilleştirme ve geçersiz kılma süzgecinden geçerek yazılır. | Kosinüs benzerliği, BM25 ve Bilgi Grafı (KG) yürüyüşü. | pgvector, Qdrant veya Neo4j / Graphiti |
| **Yordamsal Hafıza (Procedural Memory)** | *"İşler nasıl yapılır?"* Ajanın kod yazma stili, runbook'lar, araç kullanma protokolleri ve system prompt kuralları. | Yalnızca geliştirici onayıyla veya çevrimdışı pekiştirmeli öğrenmeyle. | Sistem başlangıcında ve araç çağrılarında yüklenir. | Kod tabanı, Git deposu, Beceri Kütüphanesi |

Bu dört türün tek bir senaryoda nasıl birlikte çalıştığını görelim: Kullanıcı *"Geçen cumartesiki gibi bir rota hazırla"* der.
1. **Epizodik hafıza:** Geçen cumartesi oturumunun günlüğünü bulur (hangi rota yürünmüştü).
2. **Anlamsal hafıza:** Kullanıcının güncel durumunu çeker (*"Berlin'de yaşıyor"*, *"Diz sakatlığı var, dik yokuş istemiyor"*).
3. **Yordamsal hafıza:** Rota API'sinin JSON şemasını ve hangi parametrelerle çağrılacağını sağlar.
4. **Çalışma belleği:** Tüm bu bilgileri prompt masasında harmanlar ve cevabı üretir.

CoALA'nın altını çizdiği en kritik mühendislik ilkesi şudur: **Yordamsal hafızaya yazmak, anlamsal hafızaya yazmaktan katbekat daha risklidir.** Anlamsal hafızadaki bir hata tek bir cevabı bozar (*"Berlin yerine İstanbul demek"*). Yordamsal hafızadaki bir kural bozulması ise ajanın sonraki **tüm görevlerini ve güvenlik sınırlarını felç eder**.

Modern araştırmalar bu ayrımı doğrulamaktadır:
- **Voyager (Wang et al., 2023):** Minecraft ortamında otonom ajan, kazandığı becerileri çalıştırılabilir JavaScript fonksiyonları olarak bir `skill library` (yordamsal hafıza) içinde saklamıştır.
- **Reflexion (Shinn et al., NeurIPS 2023):** Başarısız görevlerden sonra ajana dilsel özetler yazdırmış ve prompt'ta sadece son birkaç epizodik dersi ($\Omega = 1..3$) tutarak HumanEval kodlama skorunu %80'den %91'e sıçratmıştır.
- **ReasoningBank (Google Cloud AI Research, 2025):** Ajanın yalnızca başarılarından değil, başarısızlıklarından da genel çıkarımlar yaparak bir strateji bankası oluşturmasını sağlamış; web gezinme görevlerinde başarı oranını %20 artırırken maliyetli adım sayısını %16 düşürmüştür.

---

## 4. Yazma yolu: neyin saklanacağına karar vermek

Okuma yolunda yaşanan felaketlerin ezici bir çoğunluğu, aslında **yazma yolundaki ihmallerin** sonucudur. Kalitesiz bir yazma yolu; çelişkili kayıtları temizlemez, hassas PII verilerini maskelemez ve her gereksiz ayrıntıyı depoya doldurarak okuma boru hattını çöp yığınına çevirir.

### Ayıklama ve durum geçişleri: ADD, UPDATE, DELETE, NOOP

Kullanıcının ham diyalog satırlarını doğrudan vektör veritabanına gömmek felakettir. *"Tamam, geçen seferki gibi olsun ama bu sefer yanıma arkadaşımı da alacağım"* cümlesi altı ay sonra arandığında hiçbir semantik anlam ifade etmez. 

Yazma yolu önce ham turları **bağımsız, atomik olgulara (atomic facts)** dönüştürür. **Mem0 (Chhikara et al., 2025)** mimarisi bu süreci iki aşamalı bir durum makinesine bağlamıştır:
1. **Ayıklama (Extraction):** Bir yardımcı model, son mesaj çiftini okuyarak bağımsız aday önermeler üretir.
2. **Durum Kararı:** Aday önerme, depodaki en yakın $k$ mevcut anıyla karşılaştırılır ve model dört işlemden birini yürütür:

<svg viewBox="0 0 560 300" role="img" aria-label="Bir yazmanın ne yapacağına nasıl karar verdiği. Ham tur, q_t ve a_t, ayıklamaya gider ve aday bir atomik olgu çıkar. Benzerlik araması onu mevcut en yakın anılarla kıyaslar. Karar kapısı dört işlemden birini seçer. Eşdeğer kayıt yoksa ADD: yeni vektör ve kayıt eklenir. Mevcut bilgiyi zenginleştiriyorsa UPDATE: bilgi genişletilir. Mevcut bilgiyle çelişiyorsa DELETE: eski kayıt geçersiz kılınır, supersede. Bilgi zaten aynen mevcutsa NOOP: değişiklik yok. ADD, UPDATE ve DELETE depoya yazar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="amo-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="18" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">ham tur (q_t, a_t)</text>
<line x1="60" y1="24" x2="60" y2="38" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="40" width="160" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="96.0" y="62.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Ayıklama</text>
<text x="96.0" y="78.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">aday atomik olgu</text>
<rect x="200" y="40" width="160" height="52" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="54.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Benzerlik araması</text>
<text x="280.0" y="70.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">en yakın anılarla</text>
<text x="280.0" y="86.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kıyasla</text>
<rect x="384" y="40" width="160" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="464.0" y="70.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Karar kapısı</text>
<line x1="176" y1="66" x2="198" y2="66" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="360" y1="66" x2="382" y2="66" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<path d="M464 92 V108 H78 V122" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="210" y1="108" x2="210" y2="120" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="342" y1="108" x2="342" y2="120" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="474" y1="108" x2="474" y2="120" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="78" y1="108" x2="78" y2="120" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="122" width="124" height="66" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="78" y="141" text-anchor="middle" style="fill:var(--c-success);font-size:13px;font-weight:700;font-family:var(--font-mono)">ADD</text>
<text x="78" y="158" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">eşdeğer</text>
<text x="78" y="172" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">kayıt yoksa</text>
<text x="78" y="206" text-anchor="middle" style="fill:var(--c-text);font-size:11px">yeni vektör</text>
<text x="78" y="220" text-anchor="middle" style="fill:var(--c-text);font-size:11px">&amp; kayıt</text>
<line x1="78" y1="230" x2="78" y2="248" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="148" y="122" width="124" height="66" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="210" y="141" text-anchor="middle" style="fill:var(--c-accent);font-size:13px;font-weight:700;font-family:var(--font-mono)">UPDATE</text>
<text x="210" y="158" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">mevcut bilgiyi</text>
<text x="210" y="172" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">zenginleştiriyorsa</text>
<text x="210" y="206" text-anchor="middle" style="fill:var(--c-text);font-size:11px">bilgiyi</text>
<text x="210" y="220" text-anchor="middle" style="fill:var(--c-text);font-size:11px">genişlet</text>
<line x1="210" y1="230" x2="210" y2="248" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="280" y="122" width="124" height="66" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<text x="342" y="141" text-anchor="middle" style="fill:var(--c-danger);font-size:13px;font-weight:700;font-family:var(--font-mono)">DELETE</text>
<text x="342" y="158" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">mevcut bilgiyle</text>
<text x="342" y="172" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">çelişiyorsa</text>
<text x="342" y="206" text-anchor="middle" style="fill:var(--c-text);font-size:11px">geçersiz kıl</text>
<text x="342" y="220" text-anchor="middle" style="fill:var(--c-text);font-size:11px">(supersede)</text>
<line x1="342" y1="230" x2="342" y2="248" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="412" y="122" width="124" height="66" rx="8" style="fill:var(--c-surface);stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="474" y="141" text-anchor="middle" style="fill:var(--c-text-mute);font-size:13px;font-weight:700;font-family:var(--font-mono)">NOOP</text>
<text x="474" y="158" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">bilgi zaten</text>
<text x="474" y="172" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">aynen mevcutsa</text>
<text x="474" y="206" text-anchor="middle" style="fill:var(--c-text);font-size:11px">değişiklik yok</text>
<text x="474" y="220" text-anchor="middle" style="fill:var(--c-text);font-size:11px"></text>
<path d="M16 257 V285 A194.0 7 0 0 0 404 285 V257" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="210.0" cy="257" rx="194.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="210.0" y="279.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Hafıza deposu</text>
<circle cx="474" cy="268" r="9" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<circle cx="474" cy="268" r="4.5" style="fill:var(--c-text-mute)"/>
<line x1="474" y1="230" x2="474" y2="256" marker-end="url(#amo-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
</svg>

Dört durumun karar dinamiği son derece hassastır:
- **ADD:** *"Kullanıcı sabahları koşmayı sever."* (Yeni bir ilgi alanı).
- **UPDATE:** *"10 km altı parkurları sever"* kaydının, *"10 km altı, eğimsiz parkurları sever"* olarak zenginleştirilmesi.
- **DELETE (veya Geçersiz Kılma):** *"Artık balık yiyorum"* ifadesi geldiğinde *"Kullanıcı katı vejetaryendir"* kaydının statüsünün düşürülmesi.
- **NOOP:** Zaten bilinen bir gerçeğin sohbette laf arasında tekrarlanması.

Burada saf DELETE işlemi son derece tehlikelidir. Eğer sistem kullanıcının Berlin'e taşındığını görüp İstanbul kaydını fiziksel olarak silerse (hard delete), ajan bir hafta sonra *"Ben Berlin'e taşınmadan önce nerede yaşıyordum?"* sorusuna asla cevap veremez. Doğru mühendislik yaklaşımı fiziksel silme değil, **bi-temporal geçersiz kılmadır**.

### Önem ve reflection

Her konuşma turu hafızada saklanmayı hak etmez. Kullanıcının öğle yemeğinde sandviç yediği bilgisi ile penisiline ölümcül alerjisi olduğu bilgisi aynı depolama önceliğine sahip olamaz.

**Generative Agents (Park et al., Stanford, 2023):** Yazma anında hafif bir modele anının önemini 1 ila 10 arasında puanlatır:
- *"Dişlerimi fırçaladım"* $\to$ Önem: 1
- *"Boşanma davası açtım"* $\to$ Önem: 9
- *"İş teklifini kabul ettim"* $\to$ Önem: 8

Bu puan bir kez hesaplanır ve anının metaverisine kazınır. Böylece her sorguda yeniden model çalıştırma maliyeti ödenmez.

**Reflection (İçgözlem):** Ajanın son olaylarının önem puanları toplamı belirli bir eşiği ($\sum \text{Önem} \ge 150$) aştığında arka planda bir içgözlem görevi tetiklenir. Model, son ham epizotları önüne çeker ve üst düzey anlamsal çıkarımlar üretir: *"Klaus araştırmasına aşırı bağlı, işkolik bir akademisyendir."* Bu çıkarım, dayandığı ham epizotların kimliklerine işaret eden bir graf kenarıyla birlikte anlamsal depoya kaydedilir.

### Üzerine yazma, geçersiz kıl: İki zamanlı model

Zep ekibinin **Graphiti (Rasmussen et al., 2025)** zamansal bilgi grafında modellediği **Bi-temporal (İki Zamanlı)** hafıza yapısı, Berlin hatasının nihai ilacıdır.

Her anı kaydı, iki farklı zaman ekseninde toplam dört zaman damgası taşır:

$$\text{MemoryRecord} = \big\langle \text{Fact},\; t_{\text{valid\_from}},\; t_{\text{valid\_to}},\; t'_{\text{created}},\; t'_{\text{expired}} \big\rangle$$

Şöyle okuyun:
- $t_{\text{valid\_from}}$ ve $t_{\text{valid\_to}}$: **Dünya Zamanı (Valid Time)**. Bu bilginin gerçek dünyada hangi tarihler arasında doğru olduğunu gösterir.
- $t'_{\text{created}}$ ve $t'_{\text{expired}}$: **Kayıt Zamanı (Transaction Time)**. Ajan sisteminin bu bilgiyi ne zaman öğrendiğini ve ne zaman geçersiz ilan ettiğini gösterir.

```text
[Kayıt 1 - Eski Durum]
Fact: "Kullanıcı İstanbul'da yaşıyor."
Dünya Zamanı: [2020-01-01 -> 2026-09-20] (20 Eylül'de geçersizleşti)
Kayıt Zamanı: [2026-01-15 -> 2026-09-21] (Sistem 21 Eylül'de öğrendi)

[Kayıt 2 - Güncel Durum]
Fact: "Kullanıcı Berlin'de yaşıyor."
Dünya Zamanı: [2026-09-21 -> Sonsuz] (Hâlâ geçerli)
Kayıt Zamanı: [2026-09-21 -> Sonsuz]
```

Bu model sayesinde okuma yolu şu sorguları mükemmel bir kesinlikle çözer:
- *"Şu an nerede yaşıyorum?"* $\to$ Filtre: $t_{\text{valid\_from}} \le \text{şimdi} < t_{\text{valid\_to}}$ $\implies$ **Berlin**.
- *"Geçen yıl neredeydim?"* $\to$ Filtre: $t_{\text{valid\_from}} \le 2025 < t_{\text{valid\_to}}$ $\implies$ **İstanbul**.
- *"Ajan geçen ay hangi bilgiye dayanarak o tavsiyeyi vermişti?"* $\to$ Filtre: $t'_{\text{created}} \le \text{geçen\_ay} < t'_{\text{expired}}$ $\implies$ **İstanbul** (Denetim izi / Audit Trail).

### Unutmayı tasarlamak: Ebbinghaus sönümlenmesi

Hafızaya giren her kayıt ömrünün sonuna kadar aynı tazelikte kalamaz. **MemoryBank (Zhong et al., 2023)**, bilişsel psikolojinin ünlü **Hermann Ebbinghaus Unutma Eğrisi** modelini yapay zeka ajanlarına uyarlamıştır:

$$R(t) = e^{-\frac{t}{S}}$$

Şöyle okuyun: *Bir anının hatırlanabilirlik / tazelik skoru $R(t)$, anının en son erişildiği andan itibaren geçen süre ($t$) ile üstel olarak sönümlenir. Paydadaki $S$ parametresi anının gücüdür (Memory Strength).*

Bu formülün mucizesi **Aralıklı Tekrar (Spaced Repetition)** dinamiğindedir:
1. Yeni kaydedilen bir anı $S = 1$ gücüyle başlar.
2. Anı her okunduğunda veya sohbette her doğrulandığında güç katsayısı artırılır ($S \leftarrow S + 1$) ve geçen süre sıfırlanır ($t \leftarrow 0$).
3. Böylece kullanıcının sürekli bahsettiği kritik bilgiler zamanla unutulmaya karşı direnç kazanırken, bir kez bahsedilip unutulan anlamsız detaylar hızla sıfıra yaklaşır.

```text
Anı A (Önemsiz detay, bir daha hiç anılmadı, S=1):
  t = 1 gün sonra:  R = e^(-1/1) = 0,368
  t = 3 gün sonra:  R = e^(-3/1) = 0,050
  t = 7 gün sonra:  R = e^(-7/1) = 0,001 (Fiilen unutuldu)

Anı B (Kritik tercih, 1. ve 3. günlerde tekrar kullanıldı):
  t = 1. gün erişim: S = 2, t sıfırlanır
  t = 3. gün erişim: S = 3, t sıfırlanır
  t = 7. gün (erişimden 4 gün sonra): R = e^(-4/3) = 0,264 (Hâlâ diri)
```

Üretim seviyesinde sönümlenme üç altın kurala bağlıdır:
1. **Sönümlenme silme değildir:** Skoru düşen anı yok edilmez, sıcak arama katmanından soğuk arşive kaydırılır.
2. **Önem sönümlenmeyi ezer:** Yüksek öneme sahip yaşamsal olgular (hastalıklar, alerjiler) sabitlenir ($S \to \infty$).
3. **Sönümlenme çelişkiyi çözmez:** İstanbul/Berlin çelişkisi sönümlenmeyle değil, bi-temporal geçersiz kılmayla çözülür.

### Yönetişim kapısı ve güvenlik

Yazma yolundaki son kontrol noktası, verinin kalıcı diske yazılmadan önce girdiği **yönetişim kapısıdır (Governance Gate)**:
- **Hassas Veri Maskeleme (PII/Secret Redaction):** Kredi kartı numaraları, API token'ları ve parolalar regex ve yerel küçük modellerle yakalanıp depolanmadan önce `[REDACTED_SECRET]` ile maskelenir.
- **Kapsam İzolasyonu (Scope Enforcing):** Verinin hangi kullanıcıya ve hangi projeye ait olduğu üstveriye zorunlu olarak işlenir.
- **Köken Bilgisi (Provenance Tracking):** Bilginin doğrudan kullanıcı tarafından mı söylendiği, yoksa ajanın internetten okuduğu şüpheli bir üçüncü taraf web sayfasından mı devşirildiği kaydedilir.

### Sıcak yol mu, arka plan mı

LangGraph ve CrewAI mimarilerinde sıkça tartışılan ikilem: Hafıza hemen o turda mı yazılmalı, yoksa arka planda mı işlenmeli?
- **Sıcak Yol (Hot Path):** Kullanıcı hemen bir sonraki turda *"Bana az önce söylediğim isimle hitap et"* diyebilir. Bu nedenle anlık çalışma belleği güncellemeleri senkron çalışmalıdır.
- **Arka Plan (Background Worker):** Derin bilgi grafı güncellemeleri, özetleme ve vektör dizinleme asenkron bir kuyruğa (Celery / Redis / SQS) atılmalıdır. Böylece kullanıcının hissettiği Time-to-First-Token (TTFT) gecikmesi şişirilmez.

---

## 5. Okuma yolu: erişim bir boru hattıdır

Üretimde hafıza okuma yolu, tek satırlık bir `db.similarity_search()` çağrısı değildir. Gerçek bir ajan hafızası okuma yolu, çok aşamalı bir sistem mühendisliği boru hattıdır:

<svg viewBox="0 0 560 392" role="img" aria-label="Okuma yolu adım adım. Kullanıcı girdisi q_t bir ihtiyaç tespiti kapısına gelir. Hafıza gerekmiyorsa istek doğrudan LLM&#x27;e gider, ek gecikme sıfırdır. Hafıza gerekiyorsa sorgu HyDE ile yeniden yazılır. Üç arama paralel çalışır: kosinüsle yoğun vektör araması, BM25 ile seyrek anahtar kelime araması ve zamansal graf yürüyüşü. Karşılıklı sıra füzyonu, RRF, listeleri birleştirir. Çok sinyalli puanlama yakınlık, önem ve ilgiyi birleştirir. Kapsam ve zaman geçerlilik filtresi geçersizleri atar. Önbellek dostu prompt paketleme sonucu toplar ve sonuç LLM bağlamı olur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="amr-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="amr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="8" width="130" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="81.0" y="36.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Girdi q_t</text>
<rect x="170" y="8" width="160" height="48" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="250.0" y="36.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">1 · İhtiyaç tespiti</text>
<rect x="410" y="8" width="134" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="477.0" y="28.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Doğrudan LLM</text>
<text x="477.0" y="44.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+0 ms ek gecikme</text>
<line x1="146" y1="32" x2="168" y2="32" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="330" y1="32" x2="408" y2="32" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="370" y="26" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px">gerekmiyor</text>
<line x1="250" y1="56" x2="250" y2="86" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="258" y="75" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">hafıza gerekiyor</text>
<rect x="150" y="88" width="200" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="250.0" y="114.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">2 · Yeniden yazım &amp; HyDE</text>
<path d="M250 132 V146 M100 146 H460" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="100" y1="146" x2="100" y2="160" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="162" width="168" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="100.0" y="182.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Yoğun vektör arama</text>
<text x="100.0" y="198.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">dense cosine</text>
<line x1="280" y1="146" x2="280" y2="160" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="196" y="162" width="168" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="182.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Seyrek arama</text>
<text x="280.0" y="198.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">BM25 keyword</text>
<line x1="460" y1="146" x2="460" y2="160" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="376" y="162" width="168" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="460.0" y="182.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Zamansal graf</text>
<text x="460.0" y="198.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">graph walk</text>
<path d="M280 210 V224 M460 210 V224 H100" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="100" y1="210" x2="100" y2="240" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="242" width="168" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="100.0" y="267.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">4 · RRF</text>
<text x="100.0" y="283.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">karşılıklı sıra füzyonu</text>
<line x1="184" y1="271" x2="194" y2="271" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="196" y="242" width="168" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="259.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">5 · Çok sinyalli puan</text>
<text x="280.0" y="275.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">yakınlık + önem</text>
<text x="280.0" y="291.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">+ ilgi</text>
<line x1="364" y1="271" x2="374" y2="271" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="376" y="242" width="168" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="460.0" y="267.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">6 · Geçerlilik filtresi</text>
<text x="460.0" y="283.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">kapsam + zaman</text>
<line x1="460" y1="300" x2="460" y2="326" marker-end="url(#amr-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="376" y="328" width="168" height="58" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="460.0" y="353.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">7 · Prompt paketleme</text>
<text x="460.0" y="369.5" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11.5px">önbellek dostu</text>
<rect x="196" y="336" width="152" height="42" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="272.0" y="361.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">LLM bağlamı</text>
<line x1="376" y1="357" x2="350" y2="357" marker-end="url(#amr-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
</svg>

### Hafızaya bakmalı mıyız? İhtiyaç tespiti ve Just-in-Time

Her kullanıcı mesajı hafıza araması gerektirmez. *"Python'da liste nasıl ters çevrilir?"* ya da *"Newton'un ikinci yasası nedir?"* gibi genel kültür ve kodlama sorularında vektör veritabanına gitmek, sisteme gereksiz yere 150-300 ms gecikme ve veritabanı maliyeti yükler.

Hafıza hattının başında çok hafif, kural tabanlı veya ultra-küçük bir sınıflandırıcı (Classifier) yer alır. **Anthropic'in Just-in-Time (Tam Zamanında) erişim** ilkesi uyarınca: Ajan bağlamında yalnızca hafif referanslar taşınır; hafıza deposu ise yalnızca süreklilik ve kişiselleştirme gerektiren durumlarda tetiklenir.

### Sorguyu yeniden yazmak ve HyDE

Kullanıcının ham mesajı çoğu zaman sefalet derecesinde kötü bir arama sorgusudur:
- Kullanıcı: *"Geçen seferki gibi ama biraz daha ucuz olsun."*

Bu ham metinle vektör araması yaparsanız hiçbir anlamlı anı bulamazsınız; çünkü cümlenin semantiği konuşmanın önceki turlarında saklıdır.
- **Sorgu Çözümleme (Query Resolution):** Küçük bir model son 2-3 turu okuyarak zamirleri ve göndermeleri çözer: *"Kullanıcının 14 Eylül'de kiraladığı Dağ Evi rezervasyonuna benzer konaklama seçenekleri."*
- **HyDE (Hypothetical Document Embeddings, Gao et al., 2022):** Kullanıcılar soru sorar (*"Hangi şehirde yaşıyorum?"*), anılar ise bildirim cümlesi olarak saklanır (*"Kullanıcı Berlin'de ikamet etmektedir"*). Soru vektörü ile olgu vektörü arasındaki anlamsal açı uyuşmazlığını gidermek için model önce varsayımsal bir cevap üretir ve veritabanında bu varsayımsal cümlenin vektörü aratılır.

### Bir anıyı puanlamak: yakınlık, önem, ilgi

Aday anılar toplandıktan sonra, Stanford Generative Agents mimarisinin ortaya koyduğu üç boyutlu puanlama fonksiyonuyla sıralanır:

1. **Yakınlık (Recency):** Anının en son ne zaman hatırlandığını ölçer:

$$\text{recency}(m) = \gamma^{\,\Delta t_m}, \qquad \gamma = 0{,}995$$

Şöyle okuyun: *$\Delta t_m$, anının son çağrılmasından bu yana geçen saat sayısıdır. $\gamma = 0{,}995$ seçildiğinde yarılanma ömrü yaklaşık 138 saattir (yaklaşık 5,8 gün).*

2. **İlgi (Relevance):** Sorgu vektörü $\mathbf{q}$ ile anı vektörü $\mathbf{m}$ arasındaki kosinüs benzerliğidir:

$$\text{relevance}(m) = \cos(\mathbf{q}, \mathbf{m}) = \frac{\mathbf{q} \cdot \mathbf{m}}{\lVert \mathbf{q} \rVert \, \lVert \mathbf{m} \rVert}$$

3. **Önem (Importance):** Yazma anında modele tayin ettirilen 1 ila 10 arasındaki skordur.

Bu üç sinyal tamamen farklı ölçeklerde yaşadığı için, doğrudan toplanamaz. Aday kümesi üzerinde **Min-Max Normalizasyonu** uygulanır:

$$\hat{x}_m = \frac{x_m - \min_j x_j}{\max_j x_j - \min_j x_j}$$

Nihai anı skoru bu üç normalize değerin ağırlıklı toplamıdır:

$$\text{score}(m) = \alpha_{\text{rec}} \, \widehat{\text{rec}}_m + \alpha_{\text{imp}} \, \widehat{\text{imp}}_m + \alpha_{\text{rel}} \, \widehat{\text{rel}}_m$$

### Formülde adım adım yürüyüş

Sistemin t=72. saatte olduğunu varsayalım. Kullanıcı *"Evime yakın yürüyüş rotası önerir misin?"* diye sorar.
Sorgu vektörümüz normalize edilmiş 2 boyutlu oyuncak uzayda $\mathbf{q} = [0{,}80; \; 0{,}60]$ olsun ($\lVert \mathbf{q} \rVert = 1$).

Veritabanında zaman filtresini geçen 3 aday anı bulunmaktadır (Geçersiz kılınan *"İstanbul'da yaşıyor"* kaydı bi-temporal filtre tarafından peşinen elenmiştir):
- **$m_1$ ("Hafta sonları yürüyüş yapar"):** Vektör $\mathbf{m}_1 = [0{,}90; \; 0{,}30]$, Son erişim: 0. saat ($\Delta t = 72$), Önem: 6
- **$m_2$ ("Berlin'de yaşıyor"):** Vektör $\mathbf{m}_2 = [0{,}50; \; 0{,}80]$, Son erişim: 70. saat ($\Delta t = 2$), Önem: 8
- **$m_3$ ("Öğle yemeğinde sandviç yedi"):** Vektör $\mathbf{m}_3 = [-0{,}30; \; 0{,}40]$, Son erişim: 71. saat ($\Delta t = 1$), Önem: 1

**Adım 1: Ham Sinyalleri Hesaplayalım**

```text
1. Yakınlık:
   m1: 0,995^72 = 0,6970
   m2: 0,995^2  = 0,9900
   m3: 0,995^1  = 0,9950

2. Kosinüs İlgisi:
   m1: (0,80*0,90 + 0,60*0,30) / sqrt(0,81 + 0,09) = 0,9000 / 0,9487 = 0,9487
   m2: (0,80*0,50 + 0,60*0,80) / sqrt(0,25 + 0,64) = 0,8800 / 0,9434 = 0,9328
   m3: (0,80*-0,30 + 0,60*0,40) / sqrt(0,09 + 0,16) = 0,0000 / 0,5000 = 0,0000

3. Önem Skoru:
   m1 = 6,  m2 = 8,  m3 = 1
```

**Adım 2: Min-Max Normalizasyonu ve Nihai Skor ($\alpha = 1$)**

```text
Yakınlık Aralığı: [0,6970 -> 0,9950], Fark = 0,2980
  m1_hat = (0,6970 - 0,6970) / 0,2980 = 0,000
  m2_hat = (0,9900 - 0,6970) / 0,2980 = 0,983
  m3_hat = (0,9950 - 0,6970) / 0,2980 = 1,000

Önem Aralığı: [1 -> 8], Fark = 7
  m1_hat = (6 - 1) / 7 = 0,714
  m2_hat = (8 - 1) / 7 = 1,000
  m3_hat = (1 - 1) / 7 = 0,000

İlgi Aralığı: [0,0000 -> 0,9487], Fark = 0,9487
  m1_hat = 0,9487 / 0,9487 = 1,000
  m2_hat = 0,9328 / 0,9487 = 0,983
  m3_hat = 0,0000 / 0,9487 = 0,000

NİHAİ PUAN TABLOSU:
  m2 ("Berlin'de yaşıyor"):         0,983 + 1,000 + 0,983 = 2,966  --> [1. SIRA]
  m1 ("Hafta sonları yürüyüş"):     0,000 + 0,714 + 1,000 = 1,714  --> [2. SIRA]
  m3 ("Sandviç yedi"):              1,000 + 0,000 + 0,000 = 1,000  --> [3. SIRA - Elendi]
```

Sonuç muazzamdır: Modelin cevabı üretebilmek için ihtiyaç duyduğu iki temel bilgi ($m_2$ ve $m_1$) ilk iki sırayı alarak prompt'a girmeye hak kazanmıştır.

### Hibrit erişim ve RRF

Yoğun embedding modelleri kavramsal benzerlikte harikadır, ancak benzersiz kimlik numaralarında (UUID), hata kodlarında veya tıbbi ilaç isimlerinde çuvallar. Tersine BM25 kelime araması tam eşleşmelerde kusursuzdur ama eşanlamlıları yakalayamaz.

Modern üretim hatları yoğun arama, BM25 ve graf yürüyüşünü paralel koşturur ve sonuçları **Reciprocal Rank Fusion (RRF, Cormack et al., SIGIR 2009)** formülüyle birleştirir:

$$\text{RRF}(m) = \sum_{r \in \mathcal{R}} \frac{1}{k + \text{rank}_r(m)}, \qquad k = 60$$

Şöyle okuyun: *Her arama motoru $r$, bulduğu sonuca sıra numarasına ($\text{rank}$) göre ters orantılı bir puan verir ($k=60$ standart yumuşatma sabitidir). Bir motorun ilk sıraya koyduğu anı $1/61 \approx 0{,}0163$ puan kazanır. Tüm motorlardan gelen puanlar toplanır.*

RRF sayesinde farklı ölçeklerdeki skorlar (0,85 kosinüs benzerliği ile 18,4 BM25 skoru) normalizasyon sancısı çekmeden pürüzsüzce harmanlanır.

### Paketleme: köken, konum ve önbellek

Seçilen anıların prompt içerisine nasıl yerleştirileceği, modelin dikkatini doğrudan yönetir:
1. **Yapılandırılmış Köken Etiketi:** Anılar çıplak metin olarak değil; tarih, kaynak ve güvenilirlik metaverileriyle XML blokları içinde sunulmalıdır:
   ```xml
   <retrieved_memory timestamp="2026-09-27T10:00:00Z">
     <entry id="m2" valid_since="2026-09-21" source="user_statement">Kullanıcı Berlin'de yaşamaktadır.</entry>
     <entry id="m1" valid_since="2026-01-10" source="user_statement">Kullanıcı cumartesi günleri doğa yürüyüşü yapmaktan hoşlanır.</entry>
   </retrieved_memory>
   ```
2. **Önbellek Dostu Konumlandırma (Prompt Caching):** Anthropic API ve vLLM gibi gelişmiş çıkarım motorları, prompt'un başındaki ortak önekleri önbelleğe alarak (Prefix Caching) prefill maliyetini ve gecikmesini %80-90 oranında düşürür. Sürekli değişen anıları prompt'un en başına koyarsanız her turda önbelleği geçersiz kılarsınız (cache bust). **Doğru strateji:** Sabit sistem talimatlarını ve araç şemalarını en başa koyup önbellek sınırını sabitlemek; dinamik anıları ise önbellek kesme noktasının hemen arkasına yerleştirmektir.

---

## 6. Oturumlar arası birleştirme: Sleep-time compute ve dreaming

Kullanıcıyla yapılan diyalog sırasında çalışan yazma yolu doğası gereği acelecidir: Anlık gecikmeyi düşük tutmak zorundadır ve yalnızca o anki turun getirdiği birkaç bilgiyi görür. Zamanla hafıza deposunda kaçınılmaz olarak tekrarlar, küçük tutarsızlıklar ve bayatlamış hipotezler birikir. İnsan beyninin bu soruna bulduğu biyolojik çözüm **uykudur**. Uyku evresinde beyin günün anılarını yeniden oynatır (hippocampal replay), gereksiz detayları budar ve önemli dersleri kortekse uzun süreli bilgi olarak mühürler.

### Sleep-time compute ekonomisi

**Lin et al. (Letta & UC Berkeley, 2025):** Bu süreci yapay zekada **Sleep-time Compute (Uyku Zamanı Hesabı)** olarak formülleştirmiştir. Model, henüz kullanıcıdan bir soru gelmemişken çevrimdışı saatlerde çalışır; geçmiş oturumları okur, olası soruları simüle eder ve hafıza grafını yeniden yapılandırır.

Bu mimarinin maliyet dinamiği bir amortisman denklemidir:

$$\text{Sorgu Başına Maliyet} = \frac{C_{\text{uyku}}}{N} + C_{\text{çıkarım}}$$

Şöyle okuyun: *Gece harcanan çevrimdışı hesaplama maliyeti $C_{\text{uyku}}$, o hafızadan yararlanan sonraki $N$ adet kullanıcı sorgusuna bölünür. Çıkarım anındaki gecikme ve token maliyeti ($C_{\text{çıkarım}}$) dramatik biçimde düşer.*

Berkeley ölçümleri, sleep-time compute sayesinde çıkarım anındaki akıl yürütme maliyetinin 5 kat azaldığını ve karmaşık görev tamamlama başarısının %18 arttığını kanıtlamıştır.

### Üretimde dreaming: Anthropic ve OpenAI

2026 yılı itibarıyla sektörün öncü laboratuvarları bu prensibi ürünleştirdi:

- **Anthropic Claude Managed Agents (Dreams):** Claude platformunda *Dreaming*, ajanın geçmiş 1 ila 100 oturumunu inceleyerek hafıza deposunun yepyeni ve temiz bir sürümünü üreten asenkron bir arka plan sürecidir. Anthropic burada hayati bir emniyet sübabı kurmuştur: **Dreaming asla mevcut deponun üzerine körlemesine yazmaz.** Tıpkı yazılım geliştirmedeki bir **Pull Request (PR)** gibi davranır; silinen çelişkileri, birleştirilen tekrarları ve yeni çıkarımları içeren bir fark (diff) üretir. Geliştirici veya yönetici bu farkı onaylayabilir ya da reddedebilir.
- **OpenAI ChatGPT Dreaming:** Kullanıcının geçmiş sohbetlerini arka planda tarayarak planları geçmiş olgulara dönüştürür. *"Gelecek ay Roma'ya uçuyorum"* notu, seyahat tarihi geçtikten sonra tetiklenen rüya evresiyle *"Eylül 2026'da Roma'yı ziyaret etti"* geçmiş anısına evrilir.

### Bağlam çöküşü tuzağı ve ACE mimarisi

Oturumlar arası birleştirmede modelleri kör bir döngüye sokarsanız çok tehlikeli bir arıza baş gösterir: **Context Collapse (Bağlam Çöküşü)**.

**ACE Makalesi (Agent Context Evolution, Zhang et al., ICLR 2026):** AppWorld ortamında LLM'den her görevden sonra hafızasını serbestçe baştan yazmasını istemiştir. Başlangıçta 18.282 token'lık zengin ve detaylı bilgi birikimi olan hafıza, modelin kontrolsüz kısalık eğilimi (**Brevity Bias**) yüzünden birkaç döngü sonra 122 token'lık içi boş tavsiyelere (*"Hata yapma, dikkatli ol"*) indirgenmiş; ajanın görev başarma oranı %66,7'den %57,1'e çökmüştür.

ACE'nin getirdiği mimari kural şudur: **Model hiçbir zaman hafızayı toptan yeniden yazmamalıdır.** Model yalnızca küçük delta değişiklikleri (`ADD`, `UPDATE`, `INVALIDATE`) önermeli; bu değişiklikleri depoya işleyen kod deterministik ve kural tabanlı bir yazılımla yürütülmelidir.

---

## 7. Sekiz mimari, sekiz ödünleşim

Modern bir kurumsal ajanda tek bir hafıza bileşeni her ihtiyacı karşılayamaz. Sistem mimarları aşağıdaki sekiz tasarım desenini amaca göre birleştirir:

| Mimari Türü | Ölçeklenebilirlik | Yapısal Temsil | Geçersiz Kılma Yeteneği | Denetim İzi (Audit) | Literatür & Endüstri Örneği |
|---|---|---|---|---|---|
| **1. Basit Tampon (Buffer)** | Çok Kısıtlı | Yok (Düz metin) | Yok (FIFO ile düşer) | Doğal (Ham log) | Temel LangChain sohbet botları |
| **2. Kayan Özet (Compaction)** | Kısıtlı | Yok (Özet metin) | Kayıplı (Ayrıntı silinir) | Zayıf | Claude Code / Cursor sıkıştırması |
| **3. Düz Vektör Deposu** | Devasa (Milyonlar) | Geometrik (Kosinüs) | Zor (Metadata politikası) | Zayıf | pgvector, Qdrant, Pinecone |
| **4. Zamansal Bilgi Grafı** | Yüksek | Çizge (Varlık-İlişki) | Kusursuz (Zamansal kenar) | Mükemmel | Zep / Graphiti |
| **5. Hiyerarşik İşletim Sistemi** | Sınırsız | Katmanlı (RAM/Disk) | Fonksiyon Çağrısıyla | Güçlü | MemGPT / Letta |
| **6. Hipokampal Dizinleme** | Yüksek | Graf + Kişiselleştirilmiş PageRank | Graf Budama | Doğal | HippoRAG (NeurIPS 2024) |
| **7. Çağrışımsal Ağ (Zettelkasten)** | Yüksek | Düğümler arası dinamik bağ | Kendi Kendini Düzenleyen | Orta | A-MEM (2025) |
| **8. Dosya Tabanlı Otonom Araç** | Modüler | Dizin & Dosyalar | CRUD Komutlarıyla | Git / Versiyon Tabanlı | Anthropic Memory Tool (`/memories`), NeMo |

Öne çıkan üç mimariye yakından bakalım:
- **HippoRAG (NeurIPS 2024):** Beyindeki hipokampusun neokorteks üzerindeki dizinleme işlevini taklit eder. LLM ile metinden bilgi grafı üçlüleri çıkarılır. Sorgu anında sorudaki varlıklar üzerinde **Personalized PageRank (PPR)** algoritması koşturularak, çok adımlı (multi-hop) mantıksal bağlantılar tek bir erişim adımında bulunur. Standart RAG'e göre %20 daha yüksek doğruluk sağlarken, yinelemeli aramalardan 10-20 kat daha ucuzdur.
- **Anthropic Dosya Tabanlı Memory Tool:** Ajanın işletim sistemindeki `/memories` dizini altına `.txt` veya `.json` formatında dosyalar yazmasını sağlar (`view`, `create`, `str_replace`, `delete`). Ajan her yeni oturuma başlarken bu dizini okur. Kodlama ve dosya yönetimi yapan ajanlar için en esnek ve şeffaf yaklaşımdır.
- **NVIDIA NeMo Agent Toolkit:** Bellek arka ucunu iş mantığından tamamen soyutlayan `MemoryEditor` ve `MemoryManager` arayüzleri sunar. Ajan geliştiricisi altyapıyı değiştirdiğinde (örneğin Redis'ten Qdrant'a geçildiğinde) ajanın düşünce ve karar zinciri bozulmaz.

---

## 8. Çok ajanlı hafıza: depodan önce kapsam

Tek bir ajanın hafızası yalnızca *"Ben ne hatırlıyorum?"* sorusunu cevaplar. Ancak birden fazla uzmanın iş birliği yaptığı çok ajanlı (multi-agent) sistemlerde soru şuna dönüşür: **"Hangi ajan hangi bilgiyi kimin adına hatırlıyor ve bu bilgiyi başka kimler görebilir?"**

Ajanlar arası körlemesine paylaşılan tek bir küresel hafıza deposu, felakete davetiyedir.

<svg viewBox="0 0 560 364" role="img" aria-label="Çok ajanlı bir sistemde hafıza kapsamları. En üstte organizasyon kapsamı, global kuralları ve şirket runbook&#x27;larını organizasyon deposunda tutar. Altında proje kapsamı mimari kararları ve ortak kod standartlarını proje deposunda tutar; proje, organizasyon deposunu salt okunur referans olarak okur. Ajan özel bellekleri araştırmacı ajan, kodlayıcı ajan ve güvenlik denetçisi için geçici çalışma notlarını tutar; ajanlar proje deposuna izin gerektiren kısıtlı yazmalar yapar. En altta kullanıcı profili kapsamı kişisel tercihleri ve iletişim stilini user_id bazlı bir depoda tutar; araştırmacı ve kodlayıcı ajanlar onu katı bir kimlik filtresiyle okur." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="ams-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="ams-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
</defs>
<rect x="16" y="8" width="528" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="30" y="29" text-anchor="start" style="fill:var(--c-warn);font-size:13px;font-weight:700">Organizasyon kapsamı</text>
<text x="30" y="46" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">global kurallar, şirket runbook&#x27;ları</text>
<path d="M330 24 V46 A101.0 7 0 0 0 532 46 V24" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<ellipse cx="431.0" cy="24" rx="101.0" ry="7" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1.5"/>
<text x="431.0" y="43.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Organizasyon deposu</text>
<rect x="16" y="96" width="528" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="30" y="117" text-anchor="start" style="fill:var(--c-accent);font-size:13px;font-weight:700">Proje kapsamı</text>
<text x="30" y="134" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">mimari kararlar, ortak kod standartları</text>
<path d="M330 112 V134 A101.0 7 0 0 0 532 134 V112" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.5"/>
<ellipse cx="431.0" cy="112" rx="101.0" ry="7" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.5"/>
<text x="431.0" y="131.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Proje deposu</text>
<line x1="280" y1="96" x2="280" y2="62" marker-end="url(#ams-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="288" y="84" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">salt okunur referans</text>
<rect x="16" y="184" width="528" height="88" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.2"/>
<text x="30" y="203" text-anchor="start" style="fill:var(--c-accent-2);font-size:13px;font-weight:700">Ajan özel bellekleri (geçici çalışma notları)</text>
<rect x="32" y="214" width="156" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="110.0" y="240.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Araştırmacı ajan</text>
<rect x="202" y="214" width="156" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="280.0" y="240.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Kodlayıcı ajan</text>
<rect x="372" y="214" width="156" height="44" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="450.0" y="240.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Güvenlik denetçisi</text>
<line x1="450" y1="184" x2="450" y2="150" marker-end="url(#ams-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="442" y="172" text-anchor="end" style="fill:var(--c-text-mute);font-size:11px">kısıtlı yazma (izin gerekir)</text>
<rect x="16" y="304" width="528" height="52" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="30" y="325" text-anchor="start" style="fill:var(--c-success);font-size:13px;font-weight:700">Kullanıcı profili kapsamı</text>
<text x="30" y="342" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">kişisel tercihler, iletişim stili</text>
<path d="M330 320 V342 A101.0 7 0 0 0 532 342 V320" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.5"/>
<ellipse cx="431.0" cy="320" rx="101.0" ry="7" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.5"/>
<text x="431.0" y="339.0" text-anchor="middle" style="fill:var(--c-text);font-size:12.5px">Kullanıcı deposu (user_id)</text>
<line x1="110" y1="304" x2="110" y2="262" marker-end="url(#ams-a)" style="stroke:var(--c-accent);stroke-width:3"/>
<line x1="280" y1="304" x2="280" y2="262" marker-end="url(#ams-a)" style="stroke:var(--c-accent);stroke-width:3"/>
<text x="122" y="288" text-anchor="start" style="fill:var(--c-text);font-size:11px">katı kimlik filtresiyle okuma</text>
</svg>

### Kim nereye yazabilir: Politika matrisi

Çok ajanlı hafıza mimarisinde altın kural şudur: **Varsayılan olarak her hafıza yazana özeldir; paylaşım ise açık bir yetkilendirme eylemidir.**

| Ajan Rolü | Ürettiği Bilgi Türü | Hedef Kapsam | İzin Kararı |
|---|---|---|---|
| **Araştırmacı Ajan** | Web tarama karalama notları | Özel (Private) | **İzin Verilir** (Gürültü içeri hapsedilir) |
| **Araştırmacı Ajan** | Doğrulanmış kütüphane sürüm kısıtı | Proje (Project) | **İzin Verilir** (Diğer ajanlara lazımdır) |
| **Kodlayıcı Ajan** | Kod refactor stratejisi | Proje (Project) | **İzin Verilir** |
| **Müşteri Temsilcisi** | Müşterinin ev adresi veya ödeme tercihi | Kullanıcı Profili | **İzin Verilir** (Yalnızca o kullanıcı görür) |
| **Müşteri Temsilcisi** | Müşterinin özel şikayet geçmişi | Proje veya Organizasyon | **KESİNLİKLE REDDEDİLİR** (Veri sızıntısı) |
| **Herhangi Bir Ajan** | Sistem Prompt kuralı güncellemesi | Organizasyon | **Yalnızca İnsan Onayı (Human-in-the-Loop)** |

### Paylaşılan hafızanın altı arıza biçimi

1. **Kiracılar Arası Sızıntı (Cross-Tenant Leakage):** A müşterisinin konuşmasında geçen bir mimari sırrın, paylaşılan hafıza üzerinden B müşterisine verilen cevapta alıntılanması.
2. **Aşırı Paylaşım ve Gürültü Kirliliği:** Bir ajanın tuttuğu yüzlerce geçici ara adımın ortak depoya yazılması ve diğer ajanların semantik arama kalitesini çökertmesi.
3. **Zehir Yayılımı (Poison Cascade):** Bir ajanın internetten okuduğu hatalı bir iddiayı ortak depoya yazması ve diğer tüm ajanların bu yalanı mutlak gerçek sanması.
4. **Çelişkili Kararlar (Write Collision):** İki ajanın aynı anda aynı anahtar üzerinde farklı tercihler kaydetmesi.
5. **Bayatlayan Oyun Kitapları (Stale Playbooks):** Güncelliğini yitirmiş eski bir API kuralının sonsuza kadar yordamsal hafızada kalarak yeni kodları patlatması.
6. **Köken Kaybı (Loss of Attribution):** Bir bilginin hangi ajan tarafından, hangi gerekçeyle ve hangi tarihte yazıldığının bilinememesi.

### Hafıza zehirlenmesi ve dolaylı enjeksiyon

Hafıza deposu kalıcı bir siber saldırı yüzeyidir. Bir web sayfasındaki klasik prompt injection yalnızca o anki oturumu etkiler. Ancak hafızaya sızan bir injection, **o hafızayı okuyan gelecekteki tüm kullanıcıları ve tüm oturumları rehin alır**.

**AgentPoison (Chen et al., NeurIPS 2024):** Araştırmacılar, bir ajanın uzun süreli hafızasının veya RAG bilgi tabanının **%0,1'inden daha azını** zehirleyerek, standart görevlerde performansı hiç düşürmeden %80'in üzerinde saldırı başarı oranına ulaşmışlardır. Model normal zamanlarda kusursuz çalışmakta, ancak saldırganın belirlediği özel bir tetikleyici kelime geldiğinde önceden enjekte edilen kötü niyetli talimatı icra etmektedir.

### Depoya doğrudan erişmeden zehirlemek: MINJA

Saldırganın veritabanına doğrudan yazma yetkisi olmasa bile hafızayı zehirleyebileceği kanıtlanmıştır. **MINJA (Dong et al., 2025)** saldırısında saldırgan, sisteme art arda masum görünen yönlendirici sorular sorar. Ajan, bu konuşmaları kendi hafıza akışına başarıyla tamamlanmış faydalı etkileşimler olarak kaydeder. Birkaç oturum sonra bu zehirli etkileşimler ajanın inanç sistemine yerleşir ve masum bir üçüncü taraf kullanıcı geldiğinde ajan saldırganın istediği manipülatif cevabı üretir.

Bu tehditlere karşı savunma hattı:
- Dış kaynaklı (web, üçüncü taraf API) içeriklerin hafızaya yazılmadan önce karantinaya alınması.
- Hafıza kayıtlarının şifrelenmesi ve içerik özetlerinin (`content_sha256`) doğrulanması.
- Geçmişe dönük silme (`redact`) işlemlerinin tüm vektör ve graf indekslerine anında yayılması.

---

## 9. İki mühendislik gözü: eğitim ve çıkarım

Hafıza sistemini tasarlayan bir mühendis, konuya hem modelin eğitimi hem de çıkarım donanımının kısıtları açısından bakmak zorundadır:

### Eğitim Gözü: Parametrik hafıza neden yetersizdir?

Model ağırlıklarının kendisi de devasa bir hafızadır (**Parametrik Hafıza**). Ancak kullanıcı hafızasını model ağırlıklarına ince ayar (fine-tuning) ile yazmaya çalışmak üç ölümcül engelle karşılaşır:
1. **Yavaş ve Pahalıdır:** Kullanıcı yeni bir adres söylediğinde bunu modele öğretmek için eğitim kümesi hazırlayıp backpropagation koşturamazsınız.
2. **Unlearning İmkânsızlığı (GDPR Çıkmazı):** Kullanıcı *"Benimle ilgili tüm verileri sil"* dediğinde (Unutulma Hakkı), milyarlarca ağırlık arasından o spesifik bilginin sökülüp atılması (Machine Unlearning) günümüz derin öğrenmesinde henüz çözülememiş açık bir problemdir.
3. **Yıkıcı Unutma (Catastrophic Forgetting):** Yeni bir bilgiyi ağırlıklara zorla yazmak, modelin daha önce öğrendiği muhakeme yeteneklerini rastgele bozabilir.

Bu yüzden kullanıcı hafızası ağırlıkların dışında, parametrik olmayan (non-parametric) harici sistemlerde tutulmak zorundadır.

**Memory-R1 (Yan et al., 2025) Devrimi:** Eğitim tarafındaki asıl yenilik, hafıza yöneticisinin kendisini **Pekiştirmeli Öğrenme (Reinforcement Learning)** ile eğitmektir. Memory-R1 çalışmasında, küçük bir model (LLaMA-3.1-8B) PPO ve GRPO algoritmalarıyla eğitilmiştir. Model, hafızaya ne zaman `ADD`, ne zaman `UPDATE`, ne zaman `DELETE` yapacağını deneme-yanılma yoluyla öğrenmiştir. Modele hafıza işlemlerini nasıl yapacağı kural olarak dikte ettirilmemiş; ödül sinyali doğrudan dondurulmuş bir cevaplayıcı modelin son kullanıcı sorularına doğru yanıt verip veremediğinden türetilmiştir. Sonuç: Memory-R1, sabit kurallarla yönlendirilen Mem0 taban çizgisini F1 skorunda %28 geride bırakmıştır!

### Çıkarım Gözü: Hafızanın bedeli prefill ve KV Cache'te ödenir

Çıkarım tarafında her hafıza token'ı iki ayrı donanım faturası keser:
1. **Prefill Aşaması (Compute-Bound):** Prompt'a eklenen her anı token'ı, matris çarpımlarıyla (GEMM) işlenir ve Time-to-First-Token (TTFT) süresini doğrudan uzatır.
2. **Decode Aşaması (Memory-Bandwidth-Bound):** Prompt'taki tüm token'lar üretim boyunca GPU'nun HBM belleğinde KV cache olarak tutulur ve her yeni token üretiminde bellek bant genişliğini tüketir.

Token başına düşen KV cache bellek ayak izinin kesin donanım formülü şudur:

$$\text{KV Bayt / Token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times b$$

Şöyle okuyun:
- $2$: Key ve Value tensörleri.
- $L$: Modeldeki katman sayısı (Layers).
- $H_{kv}$: KV kafa sayısı (Grouped-Query Attention'daki kafa adedi).
- $d_{\text{head}}$: Her bir dikkat kafasının vektör boyutu ($d_{\text{model}} / H$).
- $b$: Sayısal hassasiyet bayt büyüklüğü (FP16 veya BF16 için $b = 2$ bayt).

**LLaMA-3-8B Örneği:**
$L = 32$, $H_{kv} = 8$, $d_{\text{head}} = 128$, $b = 2$:

$$\text{KV Bayt / Token} = 2 \times 32 \times 8 \times 128 \times 2 = 131.072 \text{ bayt} = \mathbf{128 \text{ KiB / token}}$$

Şimdi bu donanım gerçeğini üç farklı hafıza yaklaşımına uygulayalım:

```text
1. Tam Geçmiş (LOCOMO Ortalama: 26.031 token):
   26.031 × 128 KiB ≈ 3,18 GiB VRAM (Kullanıcı başına!)

2. Hedefli Hafıza (Mem0 / Zep Ortalama: 1.764 token):
   1.764 × 128 KiB  ≈ 0,22 GiB VRAM

3. Aşırı Bağlam (LongMemEval: 115.000 token):
   115.000 × 128 KiB ≈ 14,04 GiB VRAM
```

Tek bir kullanıcının tam geçmişini bağlamda tutmak **3,2 GiB VRAM** yerken; hedefli ve süzülmüş bir hafıza katmanı bu faturayı **0,22 GiB** seviyesine indirir. Bu **14 katlık fark**, aynı GPU kümesinde kaç eşzamanlı kullanıcının barındırılabileceğini belirleyen temel finansal parametredir.

| Boyut | Eğitim (Training) Gözü | Çıkarım (Inference) Gözü |
|---|---|---|
| **Hafıza Nerede Yaşar?** | Ağırlıklarda (Parametrik): Yavaş güncellenir, birebir silinemez. | Harici Depoda (Non-parametric): Milisaniyede yazılır, kesin silinir. |
| **Temel Donanım Darboğazı** | GPU kümesi iletişim maliyeti, gradient checkpointing. | GPU HBM bellek kapasitesi ve bellek bant genişliği (KV Cache). |
| **Ölçek Sınırı** | Catastrophic forgetting, fine-tuning maliyeti. | Context Rot, Lost in the Middle, prefill TTFT gecikmesi. |
| **Gelecek Trendi** | RL ile otonom hafıza yöneticisi eğitimi (Memory-R1). | Prefix caching dostu akıllı paketleme ve semantik KV sıkıştırma. |

---

## 10. Üretimde hafıza bir sistemdir

Bir yapay zeka ürününe hafıza kazandırmak, kütüphaneden bir vektör veritabanı import etmekten ibaret değildir. Üretim seviyesinde hafıza; API sözleşmesi, katmanlı depolama hiyerarşisi, kiracı izolasyonu ve gecikme bütçesi olan bağımsız bir mikroservistir.

### Asgari API

Bir kurumsal hafıza servisinin dış dünyaya açması gereken asgari üç temel uç nokta vardır:

```text
POST   /v1/memory/events
  Gövde: { user_id, session_id, role, content, timestamp }
  İşlev: Ham diyaloğu kabul eder, kuyruğa atar. Asenkron ayıklama ve dizinleme başlatır.

POST   /v1/memory/search
  Gövde: { user_id, query, scope, time_window, top_k }
  İşlev: Hibrit erişim boru hattını çalıştırır, bi-temporal filtreleri uygular ve sıralı döner.

DELETE /v1/memory/{memory_id}
  Gövde: { user_id, reason, cascade: true }
  İşlev: İlgili anıyı ve ondan türetilmiş tüm alt graf düğümlerini, özetleri ve önbellekleri siler.
```

### Sıcak, ılık, soğuk katmanlama

Verilerin erişim sıklığına ve gecikme toleransına göre katmanlanması sistem maliyetini minimize eder:

| Katman | İçerik ve Format | Erişim Süresi | Tipik Teknoloji | Maliyet |
|---|---|---|---|---|
| **Sıcak (Hot)** | Aktif oturum değişkenleri, son kullanıcı tercihleri, kullanıcı profili. | < 5 ms | Redis / Memcached / In-Memory KV | Yüksek RAM |
| **Ilık (Warm)** | Geçmiş oturum anıları, olgular, semantik bilgi grafı düğümleri. | 30 - 80 ms | Qdrant / pgvector / Neo4j | Orta (SSD + RAM) |
| **Soğuk (Cold)** | Ham diyalog logları, arşivlenmiş oturum dökümleri, eski denetim izleri. | > 500 ms | AWS S3 / Google Cloud Storage | Çok Düşük (Object Storage) |

### Kiracıları yalıtmanın üç yolu

Multi-tenant (çok kiracılı) kurumsal mimarilerde müşterilerin verilerinin birbirine karışmaması hayati önem taşır:
1. **Kiracı Başına Koleksiyon / Namespace:** En katı izolasyondur. Her şirkete ayrı bir vektör indeksi açılır. Güvenlidir ancak binlerce küçük müşteri olduğunda donanım kaynaklarını israf eder.
2. **Metadata Filtreli Paylaşımlı Koleksiyon:** Tüm veriler tek devasa indekste toplanır; her vektör `tenant_id` etiketi taşır. Sorgu anında filtre uygulanır. Ekonomiktir ancak filtre optimizasyonu yapılmazsa arama gecikmesi fırlar.
3. **Hibrit Katmanlama:** Büyük kurumsal müşterilere özel koleksiyon, küçük KOBİ'lere paylaşımlı filtrelenmiş indeks tahsis edilir.

### p95 nereye gidiyor: Gecikme bütçesi

Kullanıcının katlanabileceği azami ek hafıza arama gecikmesi (p95) **400-500 milisaniye** bandındadır. Aşamaların sıralı (serial) ve paralel koşturulması arasındaki fark bu bütçeyi belirler:

```text
Boru Hattı Aşamaları          Tipik Gecikme (ms)
------------------------------------------------
1. Sorgu Yeniden Yazımı            80 ms
2. Yoğun Vektör Arama             60 ms
3. BM25 Kelime Arama              30 ms
4. Bilgi Grafı Yürüyüşü           50 ms
5. Cross-Encoder Reranker        220 ms
6. Formatlama ve Paketleme        15 ms
------------------------------------------------
SIRALI ÇALIŞTIRMA TOPLAMI:       455 ms  (Kritik sınırda)
PARALEL ÇALIŞTIRMA (2,3,4 eşzamanlı):
  80 + max(60, 30, 50) + 220 + 15 = 375 ms  (Güvenli bölge)
```

En büyük gecikme kazancı ise **İhtiyaç Tespiti (Bölüm 5.1)** filtresinden gelir: Hafıza gerektirmeyen sorgularda bu 375 ms'lik gecikme doğrudan **0 ms**'ye iner.

### Embedding modelini değiştirdiğiniz gün

Üretim sistemlerinin en büyük kâbusu, 6 ay sonra daha başarılı bir embedding modelinin (örneğin yeni bir Cohere veya OpenAI modelinin) çıkmasıdır. İki farklı embedding modelinin ürettiği vektörler matematiksel olarak bambaşka uzaylarda yaşar; aralarında kosinüs hesabı yapılamaz.

Bunun tek çözümü **Çift Yazma (Dual-Writing) ve Arka Plan Doldurma (Backfill)** stratejisidir:
1. Yeni model devreye alındığı gün, gelen tüm yeni anılar hem eski hem yeni indekslere paralel yazılır.
2. Arka plandaki bir işçi, Soğuk Katmandaki (S3) ham diyalogları baştan okuyarak eski anıları yeni modelle yeniden vektörleştirir.
3. Gölge doğrulama testleri tamamlandığında okuma trafiği tek bir bayrakla (feature flag) yeni indekse kaydırılır.

### Ölçmek: beş temel yetenek

Ajan hafızasının başarısı demoda değil, **LongMemEval (ICLR 2025)** benchmark'ının tanımladığı beş yetenek üzerinden ölçülür:
1. **Bilgi Çıkarma (Information Extraction):** Uzun geçmişe gömülmüş tek bir spesifik olguyu bulabilme.
2. **Oturumlar Arası Akıl Yürütme (Multi-Session Reasoning):** 1. oturumdaki bilgi ile 10. oturumdaki bilgiyi sentezleyebilme.
3. **Bilgi Güncelleme (Knowledge Updates):** Değişen bir bilginin güncel değerini eski değerine tercih edebilme (Berlin vs İstanbul).
4. **Zamansal Akıl Yürütme (Temporal Reasoning):** Olayların kronolojik sırasını kavrayabilme (*"Taşınmadan önceki gün ne yaptım?"*).
5. **Cevaptan Kaçınma / Reddetme (Abstention):** Bilmediği veya geçmişte hiç geçmeyen bir bilgi sorulduğunda hayal görmeden (hallucination) açıkça *"Bu konuda bir bilgim yok"* diyebilme.

### Benchmark gerçekliği: LoCoMo vs LongMemEval

Üretici firmaların yayımladığı skorlara daima şüpheyle yaklaşılmalıdır. Örneğin LOCOMO benchmark'ında Zep sistemi için firma kendi testinde %84 doğruluk iddia ederken, rakibi Mem0 ekibinin testinde aynı sistem %58 ile %66 arasında puan almıştır.

Bu farkın sebebi konuşma uzunluklarıdır: LOCOMO'daki diyaloglar ortalama 16.000 - 26.000 token bandındadır ve modern modeller bu uzunluğu tam bağlamda zaten taşımaktadır. Hafıza sistemlerinin asıl sınavı, konuşmaların 100.000 token'ı aştığı ve tam bağlamın çöktüğü **LongMemEval-S** sınıfı testlerdir.

---

## 11. Bütün döngü 60 satır Python'da

Aşağıdaki referans kod; bu yazıda incelediğimiz temel prensipleri (bi-temporal geçersiz kılma, PII maskeleme, Ebbinghaus sönümlenmesi ve çok sinyalli Min-Max puanlamasını) harici hiçbir kütüphaneye ihtiyaç duymadan, saf Python standart kütüphanesiyle çalıştırılabilir biçimde sunmaktadır:

```python
import math
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class MemoryItem:
    text: str
    key: str                # Varlık veya konu başlığı (ör. "home_city")
    vector: Tuple[float, float] # Oyuncak 2D embedding
    importance: int         # 1 - 10 arası önem skoru
    valid_from: int         # Doğruluğun başladığı saat (Dünya Zamanı)
    valid_to: Optional[int] = None # Geçersizleştiği saat (None = Hâlâ geçerli)
    last_access: int = 0    # En son getirilme saati

PII_PATTERN = re.compile(r"\b(?:\d[ -]?){13,16}\b") # Basit kredi kartı regex'i

class AgentMemoryEngine:
    def __init__(self):
        self.storage: List[MemoryItem] = []

    def write(self, item: MemoryItem, current_time: int):
        # 1. Yönetişim Kapısı: PII Maskeleme
        if PII_PATTERN.search(item.text):
            masked_text = PII_PATTERN.sub("[KREDİ_KARTI_MASKEYLENDİ]", item.text)
            print(f"  [t={current_time:>2}] GÜVENLİK KAPISI: {masked_text}")
            return

        # 2. Bi-temporal Geçersiz Kılma (Supersession): Aynı anahtardaki eski kaydı kapa
        for old in self.storage:
            if old.key == item.key and old.valid_to is None:
                old.valid_to = item.valid_from
                print(f"  [t={current_time:>2}] GEÇERSİZ KILINDI: '{old.text}' (valid_to={item.valid_from})")

        item.last_access = current_time
        self.storage.append(item)
        print(f"  [t={current_time:>2}] YAZILDI (ADD): '{item.text}'")

    def search(self, query_vec: Tuple[float, float], current_time: int, top_k: int = 2) -> List[MemoryItem]:
        # 1. Zaman Filtresi: Yalnızca şu an geçerli olanları al
        live = [m for m in self.storage if m.valid_to is None or m.valid_to > current_time]
        if not live:
            return []

        # 2. Sinyalleri Hesapla
        cos_sim = lambda a, b: sum(x*y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))
        rec_scores = [0.995 ** (current_time - m.last_access) for m in live]
        imp_scores = [m.importance for m in live]
        rel_scores = [cos_sim(query_vec, m.vector) for m in live]

        # 3. Min-Max Normalizasyonu
        norm = lambda xs: [(x - min(xs)) / (max(xs) - min(xs)) if max(xs) > min(xs) else 1.0 for x in xs]
        r_hat, i_hat, l_hat = norm(rec_scores), norm(imp_scores), norm(rel_scores)

        # 4. Ağırlıklı Toplam Skor
        scored = []
        for idx, m in enumerate(live):
            total_score = r_hat[idx] + i_hat[idx] + l_hat[idx]
            scored.append((m, total_score, r_hat[idx], i_hat[idx], l_hat[idx]))

        scored.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n[OKUMA YOLU t={current_time}] Sorgu Vektörü: {query_vec}")
        for m, total, r, i, l in scored:
            print(f"  -> Skor: {total:.3f} (Yakınlık: {r:.2f}, Önem: {i:.2f}, İlgi: {l:.2f}) | {m.text}")

        # Başarılı getirilenlerin erişim zamanını tazele
        for m, *_ in scored[:top_k]:
            m.last_access = current_time
        return [m for m, *_ in scored[:top_k]]

if __name__ == "__main__":
    engine = AgentMemoryEngine()
    print("=== 1. YAZMA YOLU ÇALIŞTIRILIYOR ===")
    engine.write(MemoryItem("Kullanıcı İstanbul'da yaşıyor.", "home_city", (0.5, 0.8), importance=8, valid_from=0), current_time=0)
    engine.write(MemoryItem("Kullanıcı hafta sonları yürüyüş yapar.", "hobby", (0.9, 0.3), importance=6, valid_from=0), current_time=0)
    engine.write(MemoryItem("Kullanıcı Berlin'e taşındı.", "home_city", (0.5, 0.8), importance=8, valid_from=70), current_time=70)
    engine.write(MemoryItem("Kredi kartım: 4111 2222 3333 4444", "finance", (0.1, 0.2), importance=9, valid_from=71), current_time=71)
    engine.write(MemoryItem("Öğle yemeğinde sandviç yedi.", "diet", (-0.3, 0.4), importance=1, valid_from=71), current_time=71)

    print("\n=== 2. OKUMA YOLU ÇALIŞTIRILIYOR (t=72, Sorgu: 'Evime yakın yürüyüş rotaları?') ===")
    selected = engine.search(query_vec=(0.8, 0.6), current_time=72, top_k=2)
    print("\nPROMPT İÇİNE ENJEKTE EDİLEN ANILAR:")
    for item in selected:
        print(f"  * {item.text}")
```

Konsol çıktısı:

```text
=== 1. YAZMA YOLU ÇALIŞTIRILIYOR ===
  [t= 0] YAZILDI (ADD): 'Kullanıcı İstanbul'da yaşıyor.'
  [t= 0] YAZILDI (ADD): 'Kullanıcı hafta sonları yürüyüş yapar.'
  [t=70] GEÇERSİZ KILINDI: 'Kullanıcı İstanbul'da yaşıyor.' (valid_to=70)
  [t=70] YAZILDI (ADD): 'Kullanıcı Berlin'e taşındı.'
  [t=71] GÜVENLİK KAPISI: Kredi kartım: [KREDİ_KARTI_MASKEYLENDİ]
  [t=71] YAZILDI (ADD): 'Öğle yemeğinde sandviç yedi.'

=== 2. OKUMA YOLU ÇALIŞTIRILIYOR (t=72, Sorgu: 'Evime yakın yürüyüş rotaları?') ===

[OKUMA YOLU t=72] Sorgu Vektörü: (0.8, 0.6)
  -> Skor: 2.967 (Yakınlık: 0.98, Önem: 1.00, İlgi: 0.98) | Kullanıcı Berlin'e taşındı.
  -> Skor: 1.714 (Yakınlık: 0.00, Önem: 0.71, İlgi: 1.00) | Kullanıcı hafta sonları yürüyüş yapar.
  -> Skor: 1.000 (Yakınlık: 1.00, Önem: 0.00, İlgi: 0.00) | Öğle yemeğinde sandviç yedi.

PROMPT İÇİNE ENJEKTE EDİLEN ANILAR:
  * Kullanıcı Berlin'e taşındı.
  * Kullanıcı hafta sonları yürüyüş yapar.
```

Görüldüğü üzere, İstanbul kaydı zamansal filtreden geçemediği için skorlamaya dahi girememiş; Berlin taşınma bilgisi ise taze, önemli ve ilgili olduğu için birinci sıradan prompt'a girmiştir.

---

## Bütün hikâye altı satırda

- Büyük dil modelleri durumsuz fonksiyonlardır; hafıza modelin biyolojik bir yeteneği değil, mimari döngünün her turda bağlamı dinamik kurma sanatıdır.
- Bağlam penceresini veritabanı sanmak pahalı bir yanılgıdır; 115k token geçmişte modeller Lost-in-the-Middle ve Context Rot yüzünden %30 doğruluk kaybederken fahiş KV cache faturası öder.
- Güvenilir bir yazma yolu; ham diyalogları atomik olgulara böler, ADD/UPDATE/DELETE/NOOP durumlarını işletir ve çelişkileri bi-temporal zaman damgalarıyla geçersiz kılar.
- Okuma yolu tekil bir arama değil; ihtiyaç tespiti, HyDE sorgu genişletmesi, hibrit arama (yoğun + BM25 + graf) ve Reciprocal Rank Fusion ile çalışan çok aşamalı bir boru hattıdır.
- Oturumlar arası konsolidasyon (Sleep-time compute ve dreaming); arka planda çelişkileri temizleyen bir PR sistemi gibi çalışmalı ve bağlam çöküşünü önlemek için delta güncellemeleriyle sınırlandırılmalıdır.
- Çok ajanlı sistemlerde hafıza bir depodan önce yetki ve kapsam yönetimidir; sızıntıyı ve AgentPoison/MINJA zehirlenmelerini önlemek için varsayılan gizlilik ve katı kiracı filtreleri şarttır.

---

## Terimler sözlüğü

- **Working Memory (Çalışma Belleği)** — modelin anlık turda akıl yürütürken kullandığı, sistem prompt'u ve araç dönüşlerini barındıran aktif bağlam penceresi.
- **Episodic Memory (Epizodik Hafıza)** — ajanın geçmiş eylem dizilerini, oturum dökümlerini ve deneme yanılma günlüklerini zaman damgasıyla saklayan olay arşivi.
- **Semantic Memory (Anlamsal Hafıza)** — dünya ve kullanıcı hakkındaki doğrulanmış olguları, tercihleri ve varlık ilişkilerini tutan kalıcı bilgi deposu.
- **Procedural Memory (Yordamsal Hafıza)** — ajanın görevleri nasıl yerine getireceğini belirleyen sistem talimatları, araç şemaları, kod kütüphaneleri ve davranış kuralları.
- **Bi-temporal Validity (İki Zamanlı Geçerlilik)** — bir bilginin dünyada doğru olduğu zaman aralığı ($t_{\text{valid}}$) ile sistem tarafından kaydedildiği işlem zamanını ($t_{\text{transaction}}$) ayıran zaman modeli.
- **Supersession (Geçersiz Kılma)** — eskiyen veya çelişen bir bilgiyi fiziksel olarak silmek yerine, geçerlilik bitiş tarihini güncelleyerek tarihsel denetim izini koruma yöntemi.
- **Context Rot (Bağlam Çürümesi)** — prompt uzadıkça dikkat mekanizmasının alakasız ara token'lar yüzünden dağılması ve modelin kritik detayları hatırlama başarısının aşınması.
- **Compaction (Sıkıştırma)** — bağlam penceresi dolduğunda eski turların anlamsal bir özete indirgenerek yeni turlara yer açılması işlemi.
- **Sleep-time Compute (Uyku Zamanı Hesabı)** — modelin kullanıcı çevrimdışıyken çalışarak geçmiş oturumları sentezlemesi, tekrarları birleştirmesi ve hafıza grafını optimize etmesi.
- **Reciprocal Rank Fusion (RRF)** — farklı arama yöntemlerinden (vektör, BM25, graf) gelen sıralamaları skor büyüklüklerinden bağımsız olarak sıra numaraları üzerinden adilce birleştiren füzyon algoritması.
- **Hippocampal Indexing (HippoRAG)** — beyindeki hipokampus mekanizmasını taklit ederek bilgi grafı üzerinde Personalized PageRank ile çok adımlı çağrışımsal ilişkileri tek adımda çözen hafıza mimarisi.
- **Memory Poisoning (Hafıza Zehirlenmesi)** — saldırganın ajanın uzun süreli deposuna manipülatif kayıtlar yerleştirerek gelecekteki çıkarımları saptırması (AgentPoison, MINJA).
- **Prefix Caching (Önbellek Dostu Konumlandırma)** — hafıza anılarını prompt içinde sabit sistem talimatlarının ardına yerleştirerek GPU prefill maliyetini ve gecikmesini %80-90 oranında düşüren donanım optimizasyonu.

---

## Daha derine inmek için

- Sumers et al., [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427) (TMLR 2024) — Dil ajanları için çalışma, epizodik, anlamsal ve yordamsal hafıza taksonomisi.
- Packer et al., [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (2023) — İşletim sistemleri sanal bellek hiyerarşisi, bellek baskısı ve arşiv depoları.
- Park et al., [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (UIST 2023) — Yakınlık, önem ve ilgi sinyallerine dayalı hafıza akışı ve içgözlem (reflection).
- Yan et al., [Memory-R1: Enhancing Language Agents with Goal-Directed Memory Management via RL](https://arxiv.org/abs/2502.12140) (2025) — Hafıza yöneticisinin (ADD/UPDATE/DELETE/NOOP) pekiştirmeli öğrenmeyle uçtan uca eğitilmesi.
- Gutiérrez et al., [HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models](https://arxiv.org/abs/2405.14831) (NeurIPS 2024) — Bilgi grafı ve Personalized PageRank ile tek adımda hipokampal erişim.
- Rasmussen et al., [Graphiti: A Temporal Knowledge Graph for Dynamic Agent Memory](https://github.com/getzep/graphiti) (Zep, 2025) — Bi-temporal zaman damgalı varlık ve kenar ilişkileriyle hafıza geçersiz kılma.
- Wu et al., [LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory](https://arxiv.org/abs/2410.10813) (ICLR 2025) — 115k token geçmişte 5 temel hafıza yeteneği ve bağlam çürümesi testleri.
- Zhang et al., [ACE: Agent Context Evolution for Continual Self-Improvement](https://arxiv.org/abs/2510.04871) (ICLR 2026) — Bağlam çöküşü (context collapse) ve kısalık yanlılığını (brevity bias) önleyen delta hafıza güncellemeleri.
- Chen et al., [AgentPoison: Red-teaming LLM Agents via Imperceptible Long-term Memory Poisoning](https://arxiv.org/abs/2407.12784) (NeurIPS 2024) — Uzun süreli hafızaya arka kapı yerleştirme ve zehirleme dinamikleri.
- Anthropic, [Context Engineering for AI Agents & Claude Managed Agents Documentation](https://docs.anthropic.com/) (2025-2026) — Kompaksiyon, dosya tabanlı hafıza araçları, versiyonlama ve dreaming.
- NVIDIA, [NeMo Agent Toolkit Documentation](https://developer.nvidia.com/nemo) (2025-2026) — Modüler ve backend-bağımsız MemoryEditor ve MemoryManager mimarisi.
- Bu blogda: [Bir Prompt'un Yolculuğu (4): Self-Attention](post.html?slug=self-attention-derinlemesine) — Query, Key, Value geometrisi —, [LLM Sistemleri (1): Maliyet ve Gecikme](post.html?slug=llm-maliyet-ve-gecikme-optimizasyonu) — TTFT, TPOT ve Prefix Caching mekaniği — ve [RAG ve Bilgi Erişimi (1): Vektör Veritabanı](post.html?slug=vektor-veritabani-derinlemesine) — HNSW ve filtrelenmiş arama dinamikleri.
