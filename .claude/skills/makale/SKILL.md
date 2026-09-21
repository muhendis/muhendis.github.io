---
name: makale
description: Engin'in blog makale yazma akışı — yeni bir makale yazılacağında, mevcut bir makale zenginleştirileceğinde ya da EN+TR içerik üretileceğinde bu skill kullanılır. Ton, iki dilli şablon, kaynak doğrulama ve yayın akışının tamamını tanımlar.
---

# Engin'in makale yazma akışı

Bu blog (muhendis.github.io) statik ve iki dillidir; build adımı yok,
markdown tarayıcıda `marked` ile render edilir. Her makale EN+TR çifti
olarak AYNI oturumda yazılır. Aşağıdaki kurallar pazarlıksızdır.

## 1. Akış (sırayla)

1. **Planla:** Konu geldiğinde açı ve derinlik belirsizse
   AskUserQuestion ile sor (varsayılan: kendi sentezimiz, kaynaklar
   kredilendirilir; birebir çeviri/aktarım asla).
2. **Kaynakları topla:** Kullanıcının verdiği linkleri MUTLAKA oku;
   alt sayfaları da gez. Güvenilir ek kaynaklar: Anthropic resmî
   dokümanları (platform.claude.com), OpenAI resmî dokümanları
   (developers.openai.com), arXiv özetleri, promptingguide.ai,
   Inkeep glossary/rehberleri, Hugging Face blog, Milvus/Pinecone/
   Qdrant/pgvector dokümanları, Salesforce ve IBM pratisyen sayfaları.
3. **Sayıları doğrula:** Her sayısal iddia birincil kaynaktan (arXiv
   özeti, resmî doküman, makale tablosu) teyit edilir. Teyit
   edilemeyen sayı yazılmaz ya da "yaklaşık" diye yumuşatılır.
   Önceki makalelerdeki sayılarla çelişki olmamalı.
4. **EN ve TR'yi birlikte yaz** (önce EN, hemen ardından TR).
5. **İndeksleri güncelle:** posts.json (iki dil) + sitemap.xml.
6. **Doğrula** (bölüm 6'daki komutlar).
7. **Commit/push YOK** — Engin "push" demeden asla. "push" deyince:
   tek commit + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
   trailer'ı, push sonrası CDN cache-buster polling ile canlı doğrulama
   (`curl ...?cb=$(date +%s)` ile 200 + posts.json grep).

## 2. Makale şablonu (sabit)

- YAML frontmatter YOK, `# H1` YOK (başlık posts.json'dan gelir);
  gövde `##` ile başlar. Kaynak ~70 kolonda hard-wrap.
- **Giriş:** başlıksız 2–3 paragraf, belirti-önce açılış (okurun
  yaşadığı somut bir arıza/paradoksla başla), son paragrafın sonu yol
  haritası cümlesi. Açılış okura ÖDEV VERMEZ: "şunu yükleyin, şunu
  çalıştırın, şu farkı göreceksiniz" kalıbı yasak — okur daha neyi
  neden umursayacağını bilmiyorken laboratuvar kurmaz. Bunun yerine
  okurun içinde bulunduğu sahneyi anlat ("X'i boyutlandırıyorsunuz ve
  sayılar tutmuyor", "port ettiniz, hata yok ama çıktı saçma").
  Açılıştaki gerilim mühendislik kaynağıyla ölçülebilir olmalı (VRAM,
  FLOP, gecikme, parametre payı), süslü bir gözlemle değil.
- **TOC:** `**In this article**` / `**Bu yazıda**` altında manuel
  anchor listesi (anchor kuralları bölüm 5'te).
- **Gövde:** numaralı `##` bölümler; kavram önce, örnek sonra; tanımlar
  `> **Terim** = ...` blockquote kalıbıyla; karşılaştırmalar tabloyla;
  bir adet mermaid diyagramı (fenced ```mermaid) ya da CSS
  değişkenli inline SVG. Bölümlerden biri **"eğitim tarafı / çıkarım
  tarafı"** ayrımını taşır (bkz. bölüm 3, iki mühendislik gözü):
  konunun doğasına göre ya ayrı bir `##` bölüm (iki `###` alt
  başlıkla) ya da mevcut bölümlerin içine dağıtılmış iki sütunlu bir
  tablo. Sayılar iki tarafta ayrı ayrı verilir (ör. parametre/gradyan
  boyutu ve optimizasyon durumu ↔ yerleşik VRAM ve token başına FLOP).
- **MATEMATİK: LaTeX kullanılır (KaTeX kurulu).** Blok formül `$$...$$`,
  satır içi `$...$`. `assets/js/vendor/katex/` altında KaTeX 0.18.7
  var ve `blog-post.js` bunu yalnızca sayfada formül varsa tembel
  yüklüyor (mermaid deseni). Notlar:
  - `blog-post.js` içindeki `mathBlock`/`mathInline` marked
    extension'ları formülü inline lexer'dan ÖNCE yakalar; bu şart,
    yoksa `\mathbf{x}_{...}` içindeki alt çizgiler `<em>` olur ve
    LaTeX bozulur. Bu extension'lara dokunulmaz.
  - Formül `<div class="math-block">` / `<span class="math-inline">`
    içine ham metin olarak konur, KaTeX onu yerinde render eder.
    KaTeX yüklenemezse LaTeX kaynağı okunur metin olarak kalır.
  - Düz metindeki `$` (fiyat vb.) korunur: inline tokenizer `$5` gibi
    kalıpları eşleştirmez.
  - Yeni formül eklendiğinde bölüm 6'daki iki test koşulur: marked
    çıktısında `<em>` bozulması olmamalı ve KaTeX `renderToString`
    ile hepsi hatasız render edilmeli.
  - **Her formülün ALTINA sembol açıklaması gelir.** Formülün neden
    var olduğunu anlatmak yetmez; okur `R_{Θ,m}` ya da `PE_{(pos,2i)}`
    görünce hangi harfin ne olduğunu bilmeli. Kalıp: kısa bir "şöyle
    okuyun" cümlesi + her sembolün düz karşılığı, gerekirse okuma
    yönü ("verinin aktığı yön olan sağdan sola okuyun"). Örnek: "`W`
    gerçekten saklanan matris, `α` yanındaki tek eğitilebilir sayı,
    `W̄` ileri geçişin kullandığı çarpım." Sembol açıklaması olmayan
    formül, okurun atladığı bir süs olur.
  - **Sembol açıklamasından sonra ADIM ADIM küçük bir sayısal örnek
    gelir.** Hedef: lise matematiği bilen biri kâğıt kalemle takip
    edebilsin. Kurallar:
    - **Formülün orijinali AYNEN kalır.** Örnek onu sadeleştirmek
      için değil, okunur kılmak için vardır; LaTeX'e dokunulmaz.
    - **Ara adımlar gösterilir, sonuç atlanmaz.** `sin(1/1) = 0,84`
      yetmez; `bölen = 10000^(0/4) = 10000^0 = 1` → `sin(1/1) =
      sin(1) = 0,84` diye her basamak yazılır. Okurun "bu sayı
      nereden çıktı?" diye sorabileceği hiçbir yer kalmamalı.
    - **Oyuncak boyut** kullanılır (`d = 4`, 2 boyut, 30° açı) ve
      sayılar yuvarlak tutulur.
    - **Adımlar ```text bloğunda hizalı** yazılır; düzyazı içinde
      dağıtılmaz.
    - Gereken ön bilgi varsa örnek içinde hatırlatılır ("iki birim
      vektörün nokta çarpımı, aralarındaki açının kosinüsüdür").
    - **En iyi örnek iddiayı KANITLAYANdır.** İki durumu yan yana
      koyup farkın kaybolduğunu göstermek (RoPE'ta m=1,n=3 ve
      m=4,n=6 → ikisi de 60° → ikisi de 0,50), "yalnızca boşluk
      önemli" cümlesinden çok daha ikna edicidir.
    - Sayılar python ile gerçekten hesaplanır, uydurulmaz.
    - **"Bu sayı nereden geliyor?" sorusu cevapsız bırakılmaz.**
      Örnekteki her vektör/katsayı için kaynağı söylenir: öğrenilmiş
      mi (rastgele başlar, backprop şekillendirir), hesaplanmış mı
      (formülden çıkar), yoksa elle mi verilmiş. Aynı kavramın birden
      çok yöntemi anlatılıyorsa bu ayrım hepsinde yapılır ve sonunda
      karşılaştırılır ("A öğrenir, B hesaplayıp toplar, C hesaplayıp
      döndürür"). Okur örnekteki bir sayıyı gökten inmiş sanıyorsa
      örnek işini görmemiştir.
    - **Aynı kavramın birden çok yöntemi varsa hepsi AYNI somut
      örnekle, aynı tablo kalıbıyla anlatılır.** Aynı kelime ("the"),
      aynı oyuncak boyut, aynı satır düzeni; yöntemler arasında
      değişen tek şey mekanizma olsun ki okur tabloları üst üste
      koyup farkı görebilsin. Bir yöntemi tablo ile, diğerini çıplak
      formülle anlatmak karşılaştırmayı öldürür. Örnek bittiğinde bir
      önceki yönteme geri bağlanır ("B'deki tabloyla karşılaştırın:
      orada sayılar itiliyordu, burada vektör yalnızca dönüyor").
    - **Formül sayısı örneğe boğulmaz.** Formül bir kez verilir,
      gerisi tablo ve düzyazıyla yürür; art arda iki hesap bloğu
      koymak yerine tek bir "gerçek kelime" tablosu daha iyidir.
- **Metafor ZORUNLU DEĞİL.** Konu kendiliğinden bir metafora oturuyorsa
  (tokenizer masası gibi) baştan sona TEK metafor taşınır; oturmuyorsa
  metafor KULLANILMAZ. Mühendislik odaklı makalelerde (kaynak, bellek,
  gecikme, üretim mimarisi) düz teknik dil metafordan iyidir; uydurma
  metafor makaleyi yumuşatır ve okuru yavaşlatır. Metafor ile
  ölçülebilir mühendislik ayrıntısı çakışırsa ayrıntı kazanır.
- **Vaka çalışması tercih edilir:** soyut anlatım yerine tek bir gerçek
  model/sistem seçilip (ör. Gemma 3) baştan sona onun üzerinden
  yürünür; sayılar o sistemin config'inden gelir.
- **Örnekler:** güçlü, referanslı kaynaklardan birebir alınır (uydurma
  benchmark yok); gerektiğinde altına kısa "How to use it / Nasıl
  kullanılır" notu (nereye yazılır: sistem prompt'u mu kullanıcı
  mesajı mı; kod döngüsü mü prompt mu).
- **Sabit kapanış üçlüsü:**
  `## The whole story in six lines` / `## Bütün hikâye altı satırda`
  (6 madde; düz ve somut dil, mecaz yığını değil) →
  `## Glossary` / `## Terimler sözlüğü` (jargon + yalın karşılık; her
  madde tanımla YETİNMEZ, arkasına bir sezgi cümlesi ekler: ya bir
  benzetme ("ID sayfa numarası, satır da o sayfadaki madde"), ya
  sonucu ("~302M parametre kazandırır"), ya da neden umursanacağı
  ("tam da kimse şikâyet etmediği için tehlikelidir"). Kuru tanım
  listesi okurda hiçbir şey bırakmaz.) →
  `## Going deeper` / `## Daha derine inmek için` (kaynak linkleri +
  son satırda "On this blog / Bu blogda" çapraz linkleri:
  `post.html?slug=...`, TR'de TR slug'ları).

## 3. Ton ve içerik ilkeleri

- Hedef uzunluk ~6 dk (readingMinutes dürüst yazılır); Engin
  "kapsamlı" derse RAG rehberi ayarına (~11 dk) çıkılır.
- Kısa, düz cümleler; iç içe yan cümle ve üçlü kesme çizgisi yığını
  yok. Jargon KULLANILIR ama ilk geçişte tek cümlelik yalın
  açıklamayla ("forward pass — ağın içinden tek seferlik yolculuk").
- Mülakatta işe yarayacak içerik önceliklidir: doğrulanmış sayılar,
  "X'i Y'den ayıran nedir" soruları, ezberlenmeye değer tek tablo.
- **İKİ MÜHENDİSLİK GÖZÜ (pazarlıksız).** Her teknik makale konusunu
  hem *training* hem *inference* engineering tarafından ele alır; biri
  atlanırsa makale yarımdır. Aynı bileşen iki tarafa farklı fatura
  keser ve okur genelde tek tarafta çalışır:
  - **Training:** gradyan davranışı (yoğun/seyrek), optimizasyon durumu
    ve bellek, ilklendirme, dağıtık eğitimde iletişim maliyeti
    (all-reduce/shard), kararlılık (loss spike, norm dengesi),
    dondurma/ince ayar sonuçları.
  - **Inference:** VRAM'de yerleşik ağırlık ve yükleme süresi, token
    başına FLOP, prefill ile decode ayrımı, batch/KV etkisi, nicemleme
    yapılır mı yapılmaz mı, paralelleştirme (TP/dağarcık parçalama).
  - Hesap ile bellek ayrı ayrı söylenir; "pahalı" demek yetmez, hangi
    kaynağın tükendiği yazılır (FLOP mu, bant genişliği mi, VRAM mi).
  - En az bir bölüm ya da tablo bu iki sütunu yan yana koyar; belirti
    tablolarına "taraf" (eğitim/çıkarım) sütunu eklenir.
  - Mülakat bölümü varsa en az bir soru bu ayrımı test eder ("bu bir
    eğitim sorunu mu, çıkarım sorunu mu?").
  - Konunun tek tarafı varsa bu açıkça yazılır, sessizce atlanmaz.
- Az kullanılan / güncelliğini yitirmiş yöntemlerin detayına GİRME:
  tek kısa "neden geride kaldı" notu yeter. Hâlâ canlı teknikler tam
  detay + örnek alır. (ReAct güncel sayılır: "kazandı, sadece terfi
  etti".)
- Belirti-önce düşün: karar bölümleri "belirti → teknik/araç → neden
  işler" tablosuyla biter.
- Aynı kavram önceki bir makalede derinlemesine anlatıldıysa TEKRAR
  ETME: bir paragraf özet + `post.html?slug=...` çapraz link.

## 4. Türkçe kuralları

- TR birebir çeviri DEĞİL, eşdeğer akıcı metindir.
- **TR, EN bittikten sonra BAĞIMSIZ OLARAK baştan sona okunur.** EN'i
  yazıp TR'ye "çevirmek" anlam kayması üretir; TR kendi başına, hiç
  EN görmemiş bir okurun gözüyle okunmalı. Her revizyon turunda TR de
  yeniden taranır — EN'de düzeltilen bir yerin TR karşılığı otomatik
  düzelmiş sayılmaz. Tipik kaymalar (hepsi bu makalede yakalandı):
  - **Sayı/birim hatası:** "~256 bayt" (doğrusu "~256 girdi"). Çeviri
    sırasında en tehlikeli hata türü; sayılar TR'de ayrıca doğrulanır.
  - **Eksik parça:** EN'de olan bir açıklama TR'de hiç yok (uzunluk
    tanımı gibi). Bölüm bölüm karşılaştırma şart.
  - **Örnek tutarsızlığı:** tablo "the cat" diyor ama iki satırda da
    "the" var; EN'de düzelen örnek TR'de eski hâliyle kalmış.
  - **Devrik/bozuk kalıp:** "Metni kelimelerden bölmek", "Hiçbir şey
    toplamayı bırakın", "verim profilinde asla" (eksik cümle).
  - **Belirsiz zamir/özne:** "İkisini birbirine bağlayan", "İki uç da",
    "iki tarafa fatura keser" — neyin kastedildiği yazılmalı.
  - **Yanlış fiil:** "token 3. yuvaya düştüğünde" (doğrusu "yuvada
    göründüğünde"), "33,94 kat uzamıştır" (doğrusu "boyu 33,94 katına
    çıkar").
  - **Çevrilmemiş mecaz:** "sonda" (probe) TR'de anlaşılmaz; kavram
    açık yazılır ("'ben neyi arıyorum?' sorusunu temsil eden vektör").
- Düzyazıda telgraf cümlesi YASAK; ok işaretleri ve fragmanlar yalnız
  tablo/kart/kod bloklarında.
- Metod/desen/parametre adları İngilizce kalır; İLK geçişte parantez
  içinde Türkçesi: "few-shot (az örnekle gösterim)", "zero-shot
  (örneksiz)", "self-consistency (öz-tutarlılık)", "Tree of Thoughts
  (düşünce ağacı)", "fallback (yedek yöntem)", "scaffold (iskelet)",
  "temperature (sıcaklık)", "test-time compute (test anı hesabı)",
  "emergent (ölçekle beliren)", "reasoning model (akıl yürüten
  model)", "sharding (parçalama)", "eventual consistency (nihai
  tutarlılık)". Gloss'tan ÖNCE terimi çıplak kullanma.
- İngilizce kalanlar: prompt, token, embedding, few-shot, effort,
  benchmark, eval (ilk geçişte "çevrimdışı test koşuları"), top-k,
  upsert, thinking.
- Yerleşik karşılıklar: retrieval=erişim, index=dizin,
  self-reflection=öz-değerlendirme, system prompt=sistem prompt'u,
  instruction-tuned=talimatla ince ayarlı, reinforcement
  learning=pekiştirmeli öğrenme (ilk geçişte İngilizcesiyle),
  agent=ajan, sub-agent=alt ajan. Reranking=yeniden sıralama
  (yeniden dizme DEĞİL).
- Ondalıklar virgülle (%17,7), binlik ayracı noktayla (1.024).

## 5. Anchor / slug mekaniği

- Anchor algoritması (`assets/js/blog-post.js`): textContent →
  lowercase → `[^\p{L}\p{N}\s-]` sil → trim → `\s+` TEK tireye.
  Türkçe karakterler anchor'da KALIR (`#2-klasik-alet-çantası`);
  "İ" → i + birleşik nokta → nokta silinir → düz "i". Kesme
  işaretleri silinir (`Prompt'unuzda` → `promptunuzda`).
- posts.json slug'ları ASCII zorunlu; `translationSlug` iki yönde
  simetrik; yeni entry listenin EN ÜSTÜNE; `updated` alanı bugünün
  tarihi; summary'ler TAM CÜMLE (telgraf yasağı burada da geçerli).
- **summary 5N1K'ya cevap verir:** Ne (konu tek cümlede), Neden önemli
  (ölçülebilir gerilim: VRAM/FLOP/maliyet/gecikme), Nasıl (makalenin
  işlediği mekanizmalar), Nerede (hangi model/sistem vaka çalışması),
  Kim için (hedef mühendis profili), Ne zaman (hangi arıza ya da karar
  anında işe yarar). Etiketler düz metin olarak yazılır
  ("Ne:", "Neden önemli:"); summary `el('p', ...)` ile DÜZ METİN
  basılıp `<meta name="description">` içine de girdiği için
  `**bold**` ve diğer markdown işaretleri KULLANILMAZ — ham görünür.
  EN summary İngilizce, TR summary Türkçe yazılır (çeviri unutulmaz).
- sitemap.xml: makale çifti başına 2 `<url>` bloğu, üçlü hreflang,
  `x-default`=EN, `priority` 0.7, `lastmod` yayın tarihi.

## 6. Doğrulama komutları

```bash
python3 .github/scripts/validate_site.py            # 0 hata şart
node .claude/skills/makale/scripts/check_toc.js \
  en/blog/posts/<slug>.md tr/blog/posts/<slug>.md   # 0 broken şart

# Formül kontrolü: LaTeX bozulmamalı + KaTeX render etmeli
node .claude/skills/makale/scripts/check_math.js \
  en/blog/posts/<slug>.md tr/blog/posts/<slug>.md   # 0 mangled, 0 fail şart
```

IDE'nin MD041 (H1 yok), MD033 (inline SVG), MD036, MD051 (GitHub
slug kuralı) ve MD060 uyarıları bu blog için YANLIŞ ALARMDIR;
otorite validator + check_toc.js'tir.

## 7. Revizyon döngüsü beklentisi

Engin ilk taslaktan sonra iteratif ister: karışık yerleri sadeleştir,
gereksizi at, örnekleri güçlendir, başlık/slug'ı içeriğe göre yeniden
şekillendir. Her revizyonda EN ve TR SİMETRİK tutulur ve doğrulama
yeniden koşulur. Slug değişecekse push'tan önce yapılır (dosya adı,
posts.json, sitemap, çapraz linkler birlikte).
