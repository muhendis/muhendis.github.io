RAG demonuz her soruyu cevaplıyordu. Üretime çıkalı üç hafta olmuşken
bir kullanıcı "bulaşık makinem alttan su sızdırıyor" yazıyor ve bot —
sakin, akıcı, emin — kapı contasının nasıl değiştirileceğini anlatıyor.
Yanlış sayfa, yanlış parça; üstelik kaynak göstererek.

O anda her çözüm eşit derecede makul görünür: daha büyük parçalar mı?
Bir reranker mı? Bir agent mı? Bu yazının tek iddiası: **RAG hataları
sonsuz görünür, ama yalnızca dört aile vardır — aileyi adlandırdığınız
anda çözüm kendini seçer.** Her aile için: gözlediğiniz belirti, kök
neden, tedavi ve etiketteki fiyat.

**Bu yazıda**

- [1. Elinizdeki makine](#1-elinizdeki-makine)
- [2. Soru, belgelerle aynı dili konuşmuyor](#2-soru-belgelerle-aynı-dili-konuşmuyor)
  - [Boşluğu belge tarafından kapatın: hypothetical questions](#boşluğu-belge-tarafından-kapatın-hypothetical-questions)
  - [Sorgu tarafından kapatın: HyDE](#sorgu-tarafından-kapatın-hyde)
  - [Bölün ya da uzaklaşın: sub-query ve step-back](#bölün-ya-da-uzaklaşın-sub-query-ve-step-back)
- [3. Dizin, elindekini saklıyor](#3-dizin-elindekini-saklıyor)
  - [Küçük bul, büyük oku](#küçük-bul-büyük-oku)
  - [İki arayıcı, bir hakem](#iki-arayıcı-bir-hakem)
  - [Cümleyi göm, pencereyi oku](#cümleyi-göm-pencereyi-oku)
  - [Aramadan önce filtreleyin](#aramadan-önce-filtreleyin)
- [4. Erişim çalıştı — cevap yine de yanlış](#4-erişim-çalıştı-cevap-yine-de-yanlış)
- [5. Boru hattı her soruya aynı emeği harcıyor](#5-boru-hattı-her-soruya-aynı-emeği-harcıyor)
- [6. Karar, tek sayfada](#6-karar-tek-sayfada)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

## 1. Elinizdeki makine

**RAG (erişimle güçlendirilmiş üretim — retrieval-augmented
generation)**, donmuş bir LLM'in yanına kütüphane monte eder (çıplak
LLM'in neden uydurduğu: [halüsinasyon
hikâyesi](post.html?slug=llm-nasil-calisir)). Belgeler **parçalara
(chunk)** bölünür, her parça bir **gömme (embedding)** olarak **vektör
veritabanına (vector database)** yazılır; sorgu aynı haritaya gömülür,
en yakın **top-k** parça prompt'a yapıştırılır ve model cevabı
onlardan kurar.

![Yalın RAG boru hattı: belgeler parçalanıp vektör deposuna yazılır; sorgu en ilgili top-k parçayı getirir, parçalar prompt'a eklenir ve LLM cevabı onlardan üretir](../../assets/img/rag/vanilla_rag.png)

Bu makine tam dört yerde kırılabilir: arama yaptığınız soru
(2. bölüm), içinde arama yaptığınız dizin (3. bölüm), modelin bulunan
parçaları okuma biçimi (4. bölüm) ve kendini hiç denetlemeyen boru
hattı (5. bölüm).

Tedaviden önce tek bir sayı ölçün:

> **erişim isabet oranı (retrieval hit rate)** = doğru pasajı erişilen
> top-k içinde bulunan başarısız sorular ÷ tüm başarısız sorular

Yanlış cevaplanan yirmi soru toplayın, her birinin doğru pasajını elle
bulun, erişilenlerin arasında mıydı diye bakın. **Yoksa → arama sorunu
(2–3. bölümler). Varsa ve cevap yine yanlışsa → okuma sorunu (4–5.
bölümler).** Bu tek test, daha hiçbir şeye dokunmadan kataloğun
yarısını eler.

## 2. Soru, belgelerle aynı dili konuşmuyor

**Belirti:** isabet oranı düşük; ama pasajı elle bulabiliyorsunuz.

**Kök neden:** kullanıcı gündelik belirti yazar ("alttan su
sızdırıyor"); kılavuz resmî çözüm yazar ("tahliye hortumu kaplinini
doğrulayın"). Embedding ikisini de sadakatle haritalar — iki ayrı
mahalleye. Arama, tam olarak sorduğunuz soruyu cevaplar: *benim
cümleme benzeyen ne var?*

### Boşluğu belge tarafından kapatın: hypothetical questions

> **Hypothetical questions (varsayımsal sorular)** = dizinleme anında
> LLM'e her parçanın cevapladığı soruları yazdırıp parça yerine
> *onları* gömmek — arama sorudan soruya döner.

Tahliye hortumu paragrafı için üretilen sorulardan biri "bulaşık
makinemin altında neden su birikiyor?" olacaktır — kullanıcının sorusu
artık kılavuz cümleleriyle değil, sorularla eşleşir. Rafları tedarikçi
kataloğuyla değil müşterinin sorduğu adla etiketleyen dükkâncıdır bu.

![Hypothetical questions: dizinleme anında LLM her parça için sorular üretir ve bunlar ayrı bir vektör deposuna gömülür; kullanıcının sorgusu sorudan-soruya arama yapar, eşleşen soruların parçaları LLM'e gider](../../assets/img/rag/hypothetical_question.png)

**Bedeli:** 6.000 parça × 3 soru ≈ 18.000 üretim — bir kez, yazma
anında; her yeniden dizinlemede tekrar.

### Sorgu tarafından kapatın: HyDE

> **HyDE (hypothetical document embeddings)** = LLM'e sorguya *sahte*
> bir cevap taslağı yazdırıp sahteyi gömmek ve aramayı onunla yapmak —
> yanına düşen gerçek paragraflar geri gelir.

Ayna görüntüsü. Sahte cevap kılavuzun kendi sesiyle yazılır —
"Cihazın altında su birikiyorsa tahliye hortumu kaplini gevşemiş
olabilir…" — olguları yanlış olabilir ama *biçimi* doğrudur ve
embedding'in gördüğü tek şey biçimdir. Polis robot resmi gibi: kimse
şüphelinin kendisi olduğunu söylemez; doğru fotoğrafı işaret etmesi
yeter.

![HyDE: LLM önce sorgudan sahte cevap belgeleri üretir; bunların embedding'leriyle vektör deposunda belgeden-belgeye arama yapılır ve gerçek top-k parçalar LLM'e gider](../../assets/img/rag/hyde.png)

**Bedeli:** *her* sorguda +1 LLM çağrısı ve gecikmesi — dar bir alanda
sahte cevap, kendinden emin biçimde yanlış paragrafları da getirebilir.
İkilinin karar kuralı:

> HyDE sorgu başına öder, sonsuza dek. Hypothetical questions korpus
> başına öder, bir kez. Yoğun trafik, durağan korpus → belgeleri
> zenginleştirin. Sık değişen korpus, hafif trafik → sorguyu
> zenginleştirin.

### Bölün ya da uzaklaşın: sub-query ve step-back

> **Sub-queries (alt sorgular)** = bileşik soruyu basit parçalara
> bölmek, her parça için ayrı erişim yapmak ve birleştirmeyi LLM'e
> bırakmak.

**Belirti:** "eko programla yoğun programı karşılaştır" — o
karşılaştırmayı içeren parça yoktur; eko 41. sayfada, yoğun 57.
sayfadadır. Bölün, parça parça erişin, karşılaştırmayı model yapsın.

![Sub-queries: özgün sorgu sub-query 1 ve sub-query 2'ye ayrıştırılır; her biri vektör deposundan kendi top-k parçalarını getirir ve LLM birleşik sonuçlardan cevabı kurar](../../assets/img/rag/sub_query.png)

> **Step-back prompting** = aşırı özgül soruyu bir kademe
> soyutlamak, genel soruyla erişmek, özgül soruyu genel malzemeyle
> cevaplamak.

**Belirti:** "2019 model X-500'üm elektrik kesintisinden sonra neden
iki kez bip sesi çıkarıyor?" — bu kadar özgül parça yoktur. Önce "bip
kodları ne anlama geliyor?" diye sorun: sokağı aramadan önce haritayı
uzaklaştırıp mahalleyi bulun.

![Step-back prompting: özgün soru soyutlanarak bir step-back sorusuna çevrilir; bu soruyla parçalar erişilir ve bir step-back cevabı üretilir; ikinci bir LLM geçişi onu özgün soruyla birleştirip nihai cevabı verir](../../assets/img/rag/stepback.png)

**Bedeli:** her ayrıştırma fazladan bir çağrı, her sub-query baştan
sona fazladan bir erişim — maliyet ve gecikme parça sayısıyla çarpılır.

## 3. Dizin, elindekini saklıyor

**Belirti:** sorgular iyi ifade edilmişken bile isabet oranı düşük — ve
yanlış gelen sonuçların ortak bir *dokusu* var: bağlamsız kırpıntılar,
benzerler, yanlış yılın sayfası.

**Kök neden:** metnin nasıl kesildiği ve yanına nelerin yazıldığı,
neyin bulunabileceğine çoktan karar verdi. Doku, hangi kararı gözden
geçireceğinizi söyler.

### Küçük bul, büyük oku

**Belirti:** 128 token'lık parça hassas gömülür ama çıplak gelir
("Çeyrek tur sıkın." — *neyi* sıkayım?); 1.024 token'lık parçanın
bağlamı vardır ama embedding'i on konunun bulanığıdır. Çözüm: **arama
birimiyle okuma birimini ayırın** — kitap dizini gibi: işi *bulunmak*
olan işaretçi, sizi işi *okunmak* olan sayfaya gönderir.

> **Parent-child (small-to-big; ebeveyn-çocuk erişimi)** = küçük
> çocuk parçaları gömmek; bir çocuk eşleşince LLM'e onun daha büyük
> ebeveynini vermek.

![Parçaları otomatik birleştirme: küçük çocuk parçalar vektör deposuna gömülür; eşleşen çocuklar ebeveyn parçalarına çözülür ve LLM'e ebeveyn gider](../../assets/img/rag/merge_chunks.png)

> **Hierarchical index (hiyerarşik dizin)** = iki katlı dizin: önce
> belge *özetlerinde* ara, parçaları yalnızca kazanan belgelerin
> içinde ara.

![Hierarchical index: özet vektörleri dizini önce ilgili belgeleri seçer; parça düzeyindeki arama yalnızca seçilen belgelerin içinde koşar](../../assets/img/rag/hierarchical_index.png)

**Bedeli:** dizinde fazladan kayıt tutma (bookkeeping — her çocuk
ebeveynini bilmeli), büyüyen prompt'lar ve belge başına bir üretilmiş
özet.

### İki arayıcı, bir hakem

**Belirti:** "E24 hatası", E24 maddesi yerine *hata kodlarına genel
bakış* bölümünü getiriyor — yoğun (dense) embedding'ler nadir birebir
token'ları (kod, parça numarası, ad) bulanıklaştırır; sözcüksel (lexical)
**BM25** ise birebir token'ı asla kaçırmaz, başka sözcüklerle
söyleneni asla tanımaz. Kör noktalar birbirini tamamlar → ikisini
birden koşturun.

> **Hybrid retrieval (hibrit erişim)** = yoğun (embedding) arayıcıyla
> seyrek (sparse), sözcüksel arayıcıyı (**BM25** ya da **SPLADE**) yan yana
> koşturmak; sıralı listeleri **RRF** ile birleştirmek; finalistleri
> bir **cross-encoder** reranker'a yeniden notlatmak.

İki arayıcının puanları kıyaslanamaz — o yüzden *sırayla* birleştirin:

> RRF(belge) = arayıcılar üzerinden Σ 1 ÷ (60 + o arayıcıdaki sıra)

| belge | yoğun sıra | sözcüksel sıra | RRF puanı |
|---|---|---|---|
| "E24" arıza giderme maddesi | 2 | 2 | 1/62 + 1/62 ≈ **0,0323** |
| hata kodlarına genel bakış | 1 | — | 1/61 ≈ 0,0164 |
| tahliye pompası değişimi | — | 1 | 1/61 ≈ 0,0164 |

**İstikrarlı ikinci, yalnız birinciyi yener** — ve kimse bir kosinüsü
bir BM25 puanıyla kıyaslamak zorunda kalmaz. Sonra hakem: bir
**cross-encoder**, sorguyla adayı *birlikte* okur — ayrı ayrı gömen
**bi-encoder**'dan çok daha isabetli, 6.000 parça için çok yavaş; o
yüzden yalnızca ilk ~20 finalisti yeniden notlar.

![Hybrid retrieval ve reranking: sorgu hem vektör deposuna hem BM25 aramasına paralel gider; her biri kendi top-k parçalarını döndürür, reranker bunları birleştirip yeniden notlar ve kazananlar LLM'e gider](../../assets/img/rag/hybrid_and_rerank.png)

**Bedeli:** eşzamanlı tutulacak ikinci bir dizin; sorgu başına ~100 ms
sınıfı reranker çıkarımı.

### Cümleyi göm, pencereyi oku

> **Sentence window retrieval (cümle penceresi)** = tek bir
> cümleyi gömmek; eşleştiğinde LLM'e çevresindeki cümlelerden oluşan
> daha geniş bir pencereyi vermek.

Parça ikileminin en keskin çözümü, parent-child'dan bir zoom daha
ince: tek cümle var olan en hassas arama birimidir; pencere de ona
eksik kalan bağlamı geri verir.

![Sentence window retrieval: vektör deposu tek bir cümleyi eşler, ama LLM'e o cümlenin çevresindeki daha geniş pencere verilir](../../assets/img/rag/sentence_window.png)

**Bedeli:** pencere bir ayar düğmesidir — fazla dar çıplak kırpıntıyı,
fazla geniş gürültüyü geri getirir.

### Aramadan önce filtreleyin

**Belirti:** anlamca kusursuz, olgusal olarak yanlış — 2021
kılavuzunun değeri 2024 model sahibine servis edilmiş. Embedding yılı,
baskıyı, erişim yetkisini göremez.

> **Metadata filtering (üstveri filtreleme)** = her parçanın yanına
> yapılandırılmış alanlar (ürün, yıl, dil, erişim düzeyi) yazmak;
> *önce* onlarla filtrelemek, benzerlikle sıralamayı yalnızca elemeden
> sağ çıkanlar arasında yapmak.

Yüz tanıma kapısından önceki pasaport kontrolü: yüz ne kadar iyi
eşleşirse eşleşsin, yanlış pasaport sohbeti bitirir.

![Metadata filtering: vektör deposu top-k ilgili parçayı döndürür; parçalar LLM'e ulaşmadan önce üstveriyle filtrelenir](../../assets/img/rag/metadata_filtering.png)

**Bedeli:** sorgu anında neredeyse bedava; asıl fatura, içeri alma
(ingestion) sırasındaki üstveri disiplinidir.

Arama sorunları, tek kartta:

| ne gözlüyorsunuz | örnek | kök neden | tedavi | bedel |
|---|---|---|---|---|
| elle bulunuyor, erişim ıskalıyor | "alttan su sızdırıyor" ↔ hortum paragrafı | sorgu-belge üslup farkı | HyDE / hypothetical questions | sorgu başına / korpus başına LLM çağrısı |
| bileşik ya da aşırı özgül sorular başarısız | "eko ile yoğun programı karşılaştır" | aramanın birimi yanlış | sub-queries / step-back | çağrılar × parça sayısı |
| bağlamsız kırpıntılar, bulanık benzerler | "Çeyrek tur sıkın." — *neyi*? | tek parça boyu iki iş yapıyor | small-to-big / sentence window / hierarchical | dizinde kayıt yükü, büyüyen prompt'lar |
| birebir kod ve kimlikler kaçıyor | "E24 hatası" → genel bakış bölümü | yoğun arama nadir token'ı bulanıklaştırır | hybrid + RRF + reranker | ikinci dizin, rerank gecikmesi |
| doğru içerik, yanlış sürüm ya da kitle | 2021 kılavuzu → 2024 sahibi | embedding üstveriyi göremez | metadata filtering | içeri almada disiplin |

## 4. Erişim çalıştı — cevap yine de yanlış

**Belirti:** 1. bölümün testi *pozitif* — doğru pasaj top-k'nın
içinde — ve cevap yine yanlış. Sorun bulmaktan *okumaya* taşındı. Liu
vd. (2023) cevabı içeren belgeyi yirmi belge arasında konumdan konuma
taşıdı; doğruluk bir U çizdi — kenarlarda güçlü, ortada çöküyor,
çukurun dibinde *hiç belge vermemekten bile kötü*:

<svg viewBox="0 0 480 320" role="img" aria-label="Cevap doğruluğunun, doğru belgenin 20 erişilmiş belge içindeki konumuna göre çizdiği U eğrisi: 1. konumda yaklaşık yüzde 75, ortada yaklaşık yüzde 54'e düşüyor, 20. konumda yaklaşık yüzde 63'e toparlıyor. Yüzde 56 civarındaki kesikli yatay çizgi, hiç belge verilmeyen kapalı-kitap doğruluğunu gösterir" style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<line x1="50" y1="260" x2="460" y2="260" style="stroke:var(--c-border);stroke-width:1.5"/>
<line x1="50" y1="260" x2="50" y2="20" style="stroke:var(--c-border);stroke-width:1.5"/>
<g style="stroke:var(--c-border);stroke-width:1">
<line x1="50" y1="260" x2="50" y2="265"/><line x1="134" y1="260" x2="134" y2="265"/><line x1="239" y1="260" x2="239" y2="265"/><line x1="345" y1="260" x2="345" y2="265"/><line x1="450" y1="260" x2="450" y2="265"/>
<line x1="45" y1="60" x2="50" y2="60"/><line x1="45" y1="140" x2="50" y2="140"/><line x1="45" y1="220" x2="50" y2="220"/>
</g>
<g style="fill:var(--c-text-mute);font-size:11px" text-anchor="middle">
<text x="50" y="277">1</text><text x="134" y="277">5</text><text x="239" y="277">10</text><text x="345" y="277">15</text><text x="450" y="277">20</text>
</g>
<g style="fill:var(--c-text-mute);font-size:11px" text-anchor="end">
<text x="40" y="64">%75</text><text x="40" y="144">%65</text><text x="40" y="224">%55</text>
</g>
<text x="455" y="296" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">doğru belgenin 20 belge içindeki konumu</text>
<text x="18" y="140" transform="rotate(-90 18 140)" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">cevap doğruluğu</text>
<line x1="50" y1="212" x2="450" y2="212" style="stroke:var(--c-accent-2);stroke-width:1.8;stroke-dasharray:6 5"/>
<text x="160" y="204" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">hiç belge yokken (kapalı kitap)</text>
<path d="M 50 60 C 90 130, 110 196, 134 196 C 170 224, 205 228, 239 228 C 280 224, 315 218, 345 212 C 390 198, 425 172, 450 156" fill="none" style="stroke:var(--c-accent);stroke-width:2.5"/>
<circle cx="50" cy="60" r="4.5" style="fill:var(--c-text)"/>
<circle cx="239" cy="228" r="4.5" style="fill:var(--c-text)"/>
<circle cx="450" cy="156" r="4.5" style="fill:var(--c-text)"/>
<text x="62" y="52" text-anchor="start" style="fill:var(--c-text);font-size:13px">≈%75</text>
<text x="239" y="250" text-anchor="middle" style="fill:var(--c-text);font-size:13px">≈%54</text>
<text x="444" y="146" text-anchor="end" style="fill:var(--c-text);font-size:13px">≈%63</text>
</svg>

(Liu vd. 2023'ten — "lost in the middle" — yeniden çizilmiş, yaklaşık
değerler.) Bu, seri konum etkisidir (serial-position effect) — on
özgeçmiş okuyun, ilkiyle sonuncusunu hatırlarsınız. Model okumayı
bizden öğrendi. İki tedavi —
ikincisi bedava:

> **Context compression (bağlam sıkıştırma)** = erişilen parçalardan
> yalnızca *bu* sorguyla ilgili cümleleri tutan bir ayıklama geçişi —
> prompt küçülür.

Yirmi parça ≈ 6.000 token; belki 600'ü soruyla ilgili — gerisi para
ödediğiniz ve modelin dikkatini dağıtan gürültü. Ayıklayın: 6.000
girer, 900 çıkar, girdi maliyetinin ~%85'i gider. Eşyayı alın, kutuyu
bırakın.

![Context compression: erişilen top-k parçalar, LLM'e verilmeden önce yalnızca sorguyla ilgili bilgiye sıkıştırılır](../../assets/img/rag/compress_prompt.png)

**Bedeli:** sorgu başına +1 çağrı — ve gereken cümleyi atabilir;
sıkıştırmayı reranking'den *sonra* yapın, güven sıralaması finalistleri
korusun.

> **Reordering (yeniden dizme)** = en güvenilir parçaları bağlamın
> başına ve sonuna, en zayıfları ortaya koymak.

Yapıştırma sırası sizin elinizde, modelin de kenar yanlılığı var — ve
reranker adayları zaten güven sırasına dizmişti; sıralama kullanılmayı
bekliyor. **Bedeli: hiç.** Ender bedava öğle yemeği — pratikte önce
onu yapın.

![Prompt'ta parça sıralamasını ayarlama: top-k parçalar 1, 2, 3 sırasıyla gelir ve en güvenilir parçalar bağlamın iki ucunda duracak şekilde 1, 3, 2 olarak yeniden dizilir](../../assets/img/rag/adjust_order.png)

## 5. Boru hattı her soruya aynı emeği harcıyor

**Belirti çifti:** "teşekkürler, düzeldi!" mesajı koca bir
göm-ara-yapıştır turu başlatıyor; erişim sessizce çöktüğünde ise model
yine de cevap veriyor — *kaynak gösteren* halüsinasyon.

**Kök neden:** sabit boru. Kimse erişime gerek var mıydı diye sormuyor,
kimse işe yaradı mı diye bakmıyor. Son iki desen boruya muhakeme
kazandırır — biri çıkışta, biri girişte.

### Kaynakları kontrol eden editör: self-reflection

> **Self-reflection (corrective RAG; öz-değerlendirme)** = cevaptan önce
> erişilen parçaları notlandırmak; not düşükse sorguyu yeniden yazıp
> yeniden erişmek ya da web aramasına düşmek.

Notlandırıcı bir LLM ya da küçük bir doğal dil çıkarımı (natural
language inference, NLI) modelidir; **Self-RAG** bir adım öteye
gidip eleştiriyi reflection token'ları olarak
modelin kendisine işler. Muhabirin kaynaklarını yayından önce okuyan —
ve tutmuyorsa muhabiri sahaya geri gönderen — gazete editörü.

![Self-reflection: erişilen top-k parçalar doğru ya da belirsiz olarak notlanır; belirsiz olanlar internet aramasıyla doğrulanır ve LLM'e yalnızca nihai ilgili parçalar ulaşır](../../assets/img/rag/self_reflection.png)

**Bedeli:** döngü iki üç kez dönebilir — en kötü durumda gecikme ×2–3.
Kuyruk gecikmesiyle (tail latency) kaliteye *taban* satın alırsınız:
"asla özgüvenle yanlış olmasın" deseni, "hep hızlı olsun" değil.

### Triyaj hemşiresi: routing

> **Query routing (sorgu yönlendirme)** = boru hattının önünde, her
> mesaj için yolu seçen ucuz bir sınıflandırıcı: doğrudan cevapla,
> eriş, sub-query'lere böl ya da web'de ara.

Yönlendirici bir LLM çağrısı, küçük bir model ya da kurallar olabilir.
Havadan sudan sohbet → doğrudan cevapla; korpus sorusu → RAG; taze
bilgi → web araması. Kâğıt kesiğine kimse MR çekmez.

![Query routing: bir agent önce sorgunun RAG gerektirip gerektirmediğine karar verir; evet erişim hattından geçer, hayır doğrudan LLM'e gider](../../assets/img/rag/query_routing.png)

Aynı yönlendirici stratejiler *arasında* da seçim yapabilir — burada,
önce sub-query'lere bölünsün mü diye:

![Sub-query'li routing: agent sorgunun bölünüp bölünmeyeceğine karar verir; evet sub-query'ler üretir ve her biri ayrı erişim yapar, hayır özgün sorguyu normal yoldan geçirir](../../assets/img/rag/query_routing_with_sub_query.png)

**Bedeli:** sınıflandırıcı *her* istekte çalışır; yanlış yönlendirme
yepyeni bir hata türüdür — kararlarını loglayın.

## 6. Karar, tek sayfada

Broşürden değil, testten başlayın:

**TEST — doğru pasaj erişilen top-k içinde mi?**

- **Hayır — üstelik korpusta da yok.** Desen sorunu değil: kaynak
  ekleyin ya da web aramasına yönlendirin (5. bölüm).
- **Hayır — ama korpusta var.** Dönen yanlış sonuçlar neye benziyor?
  - birebir kod ve kimlikler kaçıyor → hybrid + reranker (3. bölüm)
  - doğru içerik, yanlış sürüm ya da kitle → metadata filtresi (3. bölüm)
  - bağlamsız kırpıntılar → small-to-big (3. bölüm)
  - bileşik ya da aşırı özgül sorgular → sub-queries, step-back (2. bölüm)
  - sorgularla belgeler hiç benzeşmiyor → HyDE, hypothetical questions (2. bölüm)
- **Evet — pasaj oradaydı.** Cevap nerede bozuluyor?
  - ortadaki kanıtı görmezden geliyor → reordering (4. bölüm)
  - gürültüden dağılıyor ya da pahalı → context compression (4. bölüm)
  - kolay sorulara fazla emek → routing (5. bölüm)
  - erişim çökünce özgüvenle yanlış → self-reflection döngüsü (5. bölüm)

Bütün klinik, kart kart. Her kartta **Fikir** satırını, sistem
tasarımı mülakatında vereceğiniz cevap olarak okuyun.

**hypothetical questions**
- **Ne zaman:** kullanıcılar gündelik dille soruyor, belgeleriniz resmî yazılmışsa.
- **Fikir:** çeviri bedelini bir kez, dizinleme anında ödeyin — her parçanın cevapladığı soruları saklayın; arama, soruyu soruyla eşlesin.
- **Örnek:** tahliye hortumu paragrafının yanına "altında neden su birikiyor?" yazılır — gündelik sorgu artık isabet eder.
- **Bedel:** korpusun tamamı üzerinde bir LLM geçişi; her yeniden dizinlemede tekrar.
- **Atla:** korpus her gün değişiyorsa.

**HyDE**
- **Ne zaman:** aynı üslup boşluğu var ama korpus sık değişiyor ya da trafik hafifse.
- **Fikir:** LLM sahte bir cevap yazsın; arama, sorunun kelimeleriyle değil cevabın *biçimiyle* yapılsın.
- **Örnek:** "alttan su sızdırıyor" → hortum kaplini hakkında sahte kılavuz paragrafı → gerçek paragraf bulunur.
- **Bedel:** her sorguda fazladan bir LLM çağrısı.
- **Atla:** durağan korpusta trafik yoğunsa — belgeleri bir kez zenginleştirin.

**sub-queries**
- **Ne zaman:** tek soru aslında birkaç soru içeriyorsa.
- **Fikir:** erişimin birimi tek sorudur — her parça tek soru olana dek sorguyu bölün.
- **Örnek:** "eko programla yoğun programı karşılaştır" → iki özellik araması → karşılaştırmayı LLM yapar.
- **Bedel:** çağrılar ve erişimler parça sayısıyla çarpılır.
- **Atla:** sorular zaten basit ve tek konuluysa.

**step-back**
- **Ne zaman:** sorular, korpusta yazılı her şeyden daha özgülse.
- **Fikir:** korpusun genellik düzeyinde erişin, kullanıcının ayrıntı düzeyinde cevaplayın.
- **Örnek:** "2019 X-500'üm neden iki kez bipliyor?" → "bip kodları ne demek?" erişilir → özgül vaka onunla cevaplanır.
- **Bedel:** sorgu başına fazladan bir çağrı ve bir erişim.
- **Atla:** sorgular korpusun genelliğine zaten uyuyorsa.

**small-to-big / sentence window**
- **Ne zaman:** eşleşmeler isabetli ama bağlamsız geliyorsa.
- **Fikir:** küçük birimle arayın, büyük birimle okutun — dizin kartı işaret eder, sayfa anlatır.
- **Örnek:** "Çeyrek tur sıkın" çocuğu eşleşir → LLM 800 token'lık tamir bölümünü okur.
- **Bedel:** dizinde fazladan kayıt tutma, büyüyen prompt'lar.
- **Atla:** parçalarınız zaten kendi başına anlaşılıyorsa.

**hierarchical index**
- **Ne zaman:** doğru pasaj sürekli yanlış belgeden geliyorsa.
- **Fikir:** önce doğru belgeyi seçin, içindeki pasajı ancak ondan sonra.
- **Örnek:** özet dizini X-500 kılavuzunu seçer → parçalar yalnızca onun içinde aranır.
- **Bedel:** dizinleme anında belge başına bir üretilmiş özet.
- **Atla:** korpus küçükse ya da belge yapısı yoksa.

**hybrid + RRF**
- **Ne zaman:** birebir kod, kimlik ve adlar kaçmaya devam ediyorsa.
- **Fikir:** kör noktaları zıt iki arayıcıyı sırayla birleştirin — istikrarlı ikinci, yalnız birinciyi yener.
- **Örnek:** "E24 hatası" → BM25 kolu birebir maddeyi yakalar → RRF onu tepeye koyar.
- **Bedel:** eşzamanlı tutulması gereken ikinci bir dizin.
- **Atla:** sorgular tamamen kavramsalsa, eşlenecek birebir token yoksa.

**cross-encoder rerank**
- **Ne zaman:** iyi adaylar geliyor ama nihai sıralama vasatsa.
- **Fikir:** ucuz arayıcı adayları ölçekte bulur; pahalı okuyucu yalnızca kısa listeyi yeniden notlar.
- **Örnek:** "E24 hatası" 20 adayla birlikte okunur → E24 maddesi genel bakış bölümünün üstüne çıkar.
- **Bedel:** sorgu başına yaklaşık 100 ms model çıkarımı.
- **Atla:** top-k zaten güvenilir biçimde temizse.

**metadata filtering**
- **Ne zaman:** cevaplar anlamca doğru ama yanlış sürümden ya da kitledense.
- **Fikir:** benzerlik yılı göremez — önce yapılandırılmış alanla filtreleyin, anlamla sıralamayı sonra yapın.
- **Örnek:** `model_yılı = 2024` → 2021 kılavuzu arama koşmadan elenir.
- **Bedel:** sorgu anında neredeyse bedava; asıl iş üstveriyi içeri alırken kaydetmek.
- **Atla:** korpusun tek sürümü ve tek kitlesi varsa.

**reordering**
- **Ne zaman:** model, bağlamın ortasına gömülü kanıtı görmüyorsa.
- **Fikir:** model insan gibi okur — kenarlar akılda kalır, orta kaybolur — en güçlü parçaları kenarlara koyun.
- **Örnek:** en iyi parça başa, ikinci en iyi sona → orta çukuru onları saklayamaz.
- **Bedel:** hiç — bu listenin bedava öğle yemeği.
- **Atla:** zaten çok az parça gönderiyorsanız.

**context compression**
- **Ne zaman:** prompt'lar uzun, gürültülü ve pahalıysa.
- **Fikir:** ilgisiz her token hem para hem *dikkat* harcar — yalnızca bu soruyu cevaplayan cümleleri tutun.
- **Örnek:** yirmi parça (~6.000 token) → yalnızca sızıntı ve hortumdan söz eden ~900 token kalır.
- **Bedel:** fazladan bir çağrı; gereken cümleyi düşürebilir — reranking'den sonra sıkıştırın.
- **Atla:** prompt'lar zaten kısaysa.

**routing**
- **Ne zaman:** kolay mesajlar koca hattı çalıştırıyorsa ya da yanlış araç seçiliyorsa.
- **Fikir:** her soru erişimi hak etmez — triyajı kapıda yapın.
- **Örnek:** "teşekkürler, düzeldi!" → hiç arama koşmadan doğrudan cevaplanır.
- **Bedel:** her istekte bir sınıflandırıcı; yanlış rota yeni bir hata türü.
- **Atla:** her sorgu gerçekten erişim istiyorsa.

**self-reflection döngüsü**
- **Ne zaman:** erişim çöktüğünde bile bot özgüvenle cevap veriyorsa.
- **Fikir:** yayınlamadan önce kaynakları kontrol edin — yavaş doğru cevap, hızlı yanlıştan iyidir.
- **Örnek:** parçalar bip sesinden söz etmiyor → sorgu yeniden yazılır, erişim yeniden koşar.
- **Bedel:** en kötü durumda gecikme ×2–3.
- **Atla:** hız, hata tabanından (error floor) daha önemliyse.

İki desen birbirinin yerine geçebilir göründüğünde, ayırt eden soru
şudur — neden birini seçip *benzerini* seçmediğiniz:

- **Hypothetical questions mı, HyDE mi?** Aynı hastalık, zıt taraflar.
  Durağan korpus ve yoğun trafik → bedeli yazma anında bir kez ödeyin
  (hypothetical questions). Sık değişen korpus ve hafif trafik → sorgu
  başına ödeyin (HyDE).
- **Sub-queries mi, step-back mi?** Soruları sayın. Tek soruda birkaç
  soru gizliyse ("A ile B'yi karşılaştır") → sub-queries. Tek soru
  fazla özgül sorulmuşsa → step-back.
- **Parent-child mi, sentence window mu, hierarchical mi?** Tek fikir,
  üç zoom. Varsayılan → parent-child. Hassasiyet her şeyse → sentence
  window. Karışıklık belge *içinde* değil belgeler *arasında*ysa →
  hierarchical index.
- **Hybrid arama mı, daha iyi embedding modeli mi?** Kaçanlar birebir
  token'larsa — kod, kimlik, ad — hiçbir embedding yükseltmesi birebir
  körlüğü tedavi etmez. Sözcüksel arayıcıyı ekleyin.
- **Cross-encoder rerank mi, daha büyük top-k mı?** k'yi büyütmek
  prompt'u genişletir ve 4. bölümün gürültü sorununu besler. Reranker
  hiçbir şeyi genişletmeden isabet ekler.
- **Metadata filtresi mi, reranker mı?** Reranker *ilgiyi* yargılar;
  *geçerliliği* — yılı, baskıyı, kitleyi — yalnızca yapılandırılmış
  alanlar bilir.
- **Context compression mı, reordering mi?** Rakip değiller. Reordering
  bedava — her zaman yapın. Prompt'lar uzun ya da pahalıysa sıkıştırın.
- **Routing mi, self-reflection döngüsü mü?** Borunun iki ayrı
  ucu: routing girişte para, self-reflection çıkışta kalite
  kurtarır — birlikte de çalışırlar.

Birden çok desen hâlâ uyuyorsa, maliyet merdivenini alttan tırmanın:

> **Bedava:** reordering; metadata filtering (üstveri varsa).
> **Bir kez, dizinleme anında:** small-to-big, hierarchical index, hybrid'in ikinci dizini, hypothetical questions.
> **Sorgu başına:** HyDE, context compression, routing, reranker.
> **Sorgu başına, çarpılarak:** sub-queries, self-reflection döngüleri.

Bağlayıcı kural: **önce ölçün, tanıya uyan en ucuz tedaviyi alın ve
bir sonraki deseni yalnızca isabet oranı "yetmedi" dediğinde ekleyin.**

## Bütün hikâye altı satırda

1. Hiçbir şeye dokunmadan önce tek testi çalıştırın: doğru pasaj
   erişilen top-k içinde mi?
2. Değilse ve kabahat *soruda*ysa: HyDE ya da hypothetical questions üslup
   boşluğunu köprüler; sub-query'ler bileşik soruyu böler; step-back
   aşırı özgül olanı yukarı çeker.
3. Değilse ve kabahat *dizinde*yse: küçüğü gömün, büyüğü okutun; RRF'li
   bir sözcüksel arayıcı ve cross-encoder bir hakem ekleyin; aramadan
   önce üstveriyle filtreleyin.
4. Pasaj bulunduysa ve cevap yine de yanlışsa: en güçlü parçaları
   kenarlara koyun, gürültüyü sıkıştırıp atın.
5. Boruya muhakeme verin: girişe bir router, çıkışa bir
   self-reflection döngüsü.
6. Maliyet merdivenini bedava basamaktan tırmanın; ne zaman duracağınızı
   broşür değil, isabet oranı söylesin.

Sızdıran bulaşık makinesine dönün: pasaj top-k'da yoktu, üslup boşluğu
duyuluyor ("alttan su sızdırıyor" ↔ "hortum kaplini"), reçete 2.
bölümden — hypothetical questions — ve fatura bir dizinleme geçişi. Dört
aile vardı ve artık adlarını biliyorsunuz.

## Terimler sözlüğü

Yazının temel sözcük dağarcığı, her biri tek satırda:

- **token** — modelin okuyup yazdığı en küçük metin birimi; kabaca bir İngilizce kelimenin dörtte üçü.
- **gömme (embedding)** — bir metnin anlamını harita üzerinde bir noktaya yerleştiren sayı listesi; benzer anlamlar birbirine yakın düşer.
- **parça (chunk)** — belgenin birkaç yüz token'lık dilimi; tek birim olarak saklanır ve erişilir.
- **vektör veritabanı (vector database)** — embedding'leri tutan ve sorguya en yakın olanları hızla bulan depo.
- **top-k** — aramanın döndürdüğü en yakın k parça; k'yi siz seçersiniz.
- **prompt (istem)** — tek istekte modele verilen her şey: soru, erişilen parçalar, talimatlar.
- **korpus (corpus)** — sistemin içinde arama yaptığı belge koleksiyonunun tamamı.
- **içeri alma (ingestion)** — yazma yolu: belgeleri bölmek, gömmek ve üstverisiyle birlikte saklamak.
- **yoğun / seyrek arama (dense / sparse)** — yoğun arama embedding'leri (anlamı) karşılaştırır; seyrek — sözcüksel, BM25 gibi — birebir kelimeleri eşler.
- **BM25** — klasik sözcüksel sıralama formülü: birebir kelime eşleşmesini, kelimenin nadirliğiyle ağırlıklandırarak ödüllendirir.
- **bi-encoder / cross-encoder** — bi-encoder sorguyla belgeyi ayrı ayrı gömer (hızlı, ölçeklenir); cross-encoder ikisini birlikte okur (isabetli, yavaş).
- **reranking (yeniden sıralama)** — erişilen kısa listeyi daha güçlü bir modelle yeniden puanlamak.
- **RRF (reciprocal rank fusion)** — birden çok sıralı listeyi ham puanlar yerine sıra konumlarıyla birleştirmek.
- **erişim isabet oranı (retrieval hit rate)** — doğru pasajı erişilen top-k içinde *bulunan* başarısız soruların payı; bu yazının tek tanı testi.
- **halüsinasyon (hallucination)** — doğruya benzeyen ama doğru olmayan akıcı cevap; RAG'in önlemek için var olduğu arıza.

## Daha derine inmek için

- Milvus ekibi, [How to Enhance the Performance of Your RAG Pipeline](https://zilliz.com/learn/how-to-enhance-the-performance-of-your-rag-pipeline) — bu yazıdaki diyagramların kaynağı; teknikleri boru hattı aşamalarına göre sıralayan tarama.
- Gao vd., [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (2022) — HyDE makalesi.
- Liu vd., [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (2023) — U eğrisi onlarındır.
- Asai vd., [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) (2023).
- Yan vd., [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) (2024).
- Formal vd., [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) (2021).
- Cormack, Clarke & Büttcher, [Reciprocal Rank Fusion outperforms Condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114) (SIGIR 2009) — k = 60'ın geldiği yer.

---

*Görsel kaynağı: bu yazıdaki boru hattı çizimlerinin tamamı Milvus
ekibinin [How to Enhance the Performance of Your RAG Pipeline](https://zilliz.com/learn/how-to-enhance-the-performance-of-your-rag-pipeline)
rehberinden alınmıştır ve yazarlarına aittir. Lost-in-the-middle
U eğrisi ise bu yazının kendi çizimidir; Liu vd. 2023'ten yaklaşık
değerlerle yeniden çizilmiştir.*
