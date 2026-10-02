Bu cümleyi bitirmeden zihniniz sıradaki kelimeyi tahmin etmeye başladı
bile. Kanıt mı? "Bir varmış, bir ___." Durduramadınız: boşluk, sizden
izin almadan kendini doldurdu.

O refleksin bir adı var — sıradaki-token tahmini (next-token prediction)
— ve bu yazının tek bir iddiası: bir dil modelinin size yazdığı her
cümle, her deneme, her kod parçası, az önce yaptığı bir hata için
dilediği her özür, aynı refleksin milyarlarca kat büyütülmüşüdür.
Telefon klavyeniz "görüşürüz"den sonra *yarın* önerdiğinde bu oyunun
cep boyunu oynuyor. Peki bu kadar basit bir oyun nasıl sınav geçer,
nasıl yazılım yazar? Çünkü talebi acımasızdır: insan metninde sıradaki
kelimeyi *iyi* tahmin etmek için dil bilgisini, olguları, üslubu ve
akıl yürütmenin işleyen bir taklidini özümsemek zorundasınızdır.
Aşağıdaki her şey bu fikrin dipnotudur.

**Bu yazıda**

- [1. Metin sayıya dönüşür](#1-metin-sayıya-dönüşür)
- [2. Anlam taşıyan sayılar](#2-anlam-taşıyan-sayılar)
- [3. Transformer: bir bağlam makinesi](#3-transformer-bir-bağlam-makinesi)
  - [Q, K, V — mekanizma, sayılarla](#q-k-v-mekanizma-sayılarla)
  - [Tek yön ve causal mask](#tek-yön-ve-causal-mask)
  - [Birçok kafa](#birçok-kafa)
  - [Model aileleri: Neden herkes decoder-only?](#model-aileleri-neden-herkes-decoder-only)
- [4. Katmanlar: bilgi nerede yaşıyor](#4-katmanlar-bilgi-nerede-yaşıyor)
- [5. Eğitim ve ölçek](#5-eğitim-ve-ölçek)
- [6. Otomatik tamamlamadan asistana](#6-otomatik-tamamlamadan-asistana)
- [7. Çıkarım ve üretim: iki aşamalı döngü](#7-çıkarım-ve-üretim-iki-aşamalı-döngü)
  - [Prefill: paralel hesaplama ve TTFT](#prefill-paralel-hesaplama-ve-ttft)
  - [Decode: seri döngü ve ITL](#decode-seri-döngü-ve-itl)
  - [KV cache'in gerçek boyutu ve bellek duvarı](#kv-cachein-gerçek-boyutu-ve-bellek-duvarı)
  - [Prefill ve decode'un çakışması: disaggregation](#prefill-ve-decodeun-çakışması-disaggregation)
  - [Çekilişi yöneten üç kadran: temperature, top-k, top-p](#çekilişi-yöneten-üç-kadran-temperature-top-k-top-p)
- [8. Sizi hatırlamaz: bağlam penceresi](#8-sizi-hatırlamaz-bağlam-penceresi)
- [9. Neden uyduruyor](#9-neden-uyduruyor)
- [10. Otoregresyonun ötesi: Difüzyon LLM'leri (dLLM)](#10-otoregresyonun-ötesi-difüzyon-llmleri-dllm)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Metin sayıya dönüşür

Bir sinir ağı matris çarpar; harflerle veya kelimelerle doğrudan işlem
yapamaz. Bu yüzden her girdi, ağa ulaşmadan önce **tokenizasyon**
aşamasından geçer. Bir **token**, bir dil modelinin metni işlemek için
kullandığı en küçük yapı taşıdır: yerine göre bir kelime, bir ek, bir hece
ya da tek bir karakter olabilir.

Her LLM'in eğitim öncesinde dondurulmuş sabit bir kelime dağarcığı
(**vocabulary**) vardır. Bu sözlükteki her bir token, benzersiz bir
tamsayıya (**token ID**) eşlenir. Metin modele girdiğinde harfler hemen
bu tamsayı dizilerine dönüştürülür. Örneğin `The quick brown fox jumps
over the lazy dog.` cümlesi modern bir tokenizer'dan geçtiğinde şu
parçalara ve kimlik numaralarına ayrılır:

- **Token'lar:** `"The"`, `" quick"`, `" brown"`, `" fox"`, `" jumps"`, `" over"`, `" the"`, `" lazy"`, `" dog"`, `"."`
- **Token ID'leri:** `[976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 13]`

Bu parçalama işini standart olarak **Byte-Pair Encoding (BPE)** algoritması
yapar. Tıpkı LEGO gibi: dil, en sık yeniden kullanılan tuğlalarına
ayrılır; yaygın kelimeler tek parça kalırken, nadir kelimeler kök ve
eklerine bölünür. Pratik kural: 100 token kabaca 75 İngilizce kelime
eder.

Model harfleri değil yalnızca bu tamsayı damgalarını gördüğü için
"strawberry" kelimesindeki r'leri saymak meşhur biçimde zordu. Bir akıl
yürütme kusuru gibi görünen bu durumun sebebi basittir: kelime modele
ulaştığında çoktan üç bağımsız tamsayıya (`str`, `aw`, `berry`)
dönüşmüştü ve model harfleri hiç görmedi. Sözlük boyutunun parametre
sayısına ve dizi uzunluğuna etkisini daha ayrıntılı incelemek için
[Tokenizasyon nasıl çalışır](post.html?slug=tokenizasyon-nasil-calisir)
yazısına bakabilirsiniz.

## 2. Anlam taşıyan sayılar

Tamsayılar da tek başına yeterli değildir; `4853` sayısının `68347` ile
anlamsal bir benzerliği matematiksel olarak kurulamaz. Bu yüzden her token
ID, modelin ilk katmanında devasa bir ağırlık tablosundan satır çekilerek
bir **embedding** vektörüne dönüştürülür.

Girdi anında bu işlem tek bir aritmetik çarpma bile gerektirmez; sadece
bellekten satır okumaktır (sıfır FLOP). Detaylarını
[Embedding katmanı derinlemesine](post.html?slug=embedding-katmani-derinlemesine)
yazısında ele aldığımız bu lookup mekanizması, ayrık sayıları binlerce
boyutlu sürekli bir anlam uzayındaki geometrik koordinatlara yerleştirir.

Bu uzayı binlerce kadran olarak düşünebilirsiniz: biri resmiyet derecesi,
biri canlılık, biri zaman kipi için... Bu haritada benzer anlamdaki
kelimeler birbirine yakın durur: *kral (king)*, *kraliçe (queen)*
kelimesinin yakınına, *elektronik tablo (spreadsheet)* kelimesinin ise
uzağına düşer. Daha da önemlisi, uzaydaki **yönler** anlamsal ilişkileri
temsil eder. Bunu sadece iki kadranlı sembolik bir haritada dört kelimeyle
görebiliriz: **(2, 1) konumunda *erkek (man)***, **(5, 4) konumunda *kadın
(woman)***, **(3, 6) konumunda *kral (king)*** ve **(6, 9) konumunda
*kraliçe (queen)***:

<svg viewBox="0 0 480 320" role="img" aria-label="İki kadran üzerinde dört kelime: erkek 2,1; kadın 5,4; kral 3,6; kraliçe 6,9. Düz paralel oklar erkek-kadın ve kral-kraliçe cinsiyet yönünü (+3,+3), kesikli paralel oklar erkek-kral ve kadın-kraliçe kraliyet yönünü (+1,+5) gösterir" style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="emb-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent)"/></marker>
<marker id="emb-arr2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<line x1="40" y1="280" x2="460" y2="280" style="stroke:var(--c-border);stroke-width:1.5"/>
<line x1="40" y1="280" x2="40" y2="20" style="stroke:var(--c-border);stroke-width:1.5"/>
<g style="stroke:var(--c-border);stroke-width:1">
<line x1="120" y1="280" x2="120" y2="285"/><line x1="200" y1="280" x2="200" y2="285"/><line x1="280" y1="280" x2="280" y2="285"/><line x1="360" y1="280" x2="360" y2="285"/><line x1="440" y1="280" x2="440" y2="285"/>
<line x1="35" y1="228" x2="40" y2="228"/><line x1="35" y1="176" x2="40" y2="176"/><line x1="35" y1="124" x2="40" y2="124"/><line x1="35" y1="72" x2="40" y2="72"/><line x1="35" y1="20" x2="40" y2="20"/>
</g>
<g style="fill:var(--c-text-mute);font-size:11px" text-anchor="middle">
<text x="120" y="297">2</text><text x="200" y="297">4</text><text x="280" y="297">6</text><text x="360" y="297">8</text><text x="440" y="297">10</text>
</g>
<g style="fill:var(--c-text-mute);font-size:11px" text-anchor="end">
<text x="30" y="232">2</text><text x="30" y="180">4</text><text x="30" y="128">6</text><text x="30" y="76">8</text><text x="30" y="24">10</text>
</g>
<text x="455" y="311" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">kadran 1</text>
<text x="18" y="150" transform="rotate(-90 18 150)" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">kadran 2</text>
<line x1="120" y1="254" x2="160" y2="124" marker-end="url(#emb-arr2)" style="stroke:var(--c-accent-2);stroke-width:1.8;stroke-dasharray:6 5"/>
<line x1="240" y1="176" x2="280" y2="46" marker-end="url(#emb-arr2)" style="stroke:var(--c-accent-2);stroke-width:1.8;stroke-dasharray:6 5"/>
<line x1="120" y1="254" x2="240" y2="176" marker-end="url(#emb-arr)" style="stroke:var(--c-accent);stroke-width:2"/>
<line x1="160" y1="124" x2="280" y2="46" marker-end="url(#emb-arr)" style="stroke:var(--c-accent);stroke-width:2"/>
<circle cx="120" cy="254" r="4.5" style="fill:var(--c-text)"/>
<circle cx="240" cy="176" r="4.5" style="fill:var(--c-text)"/>
<circle cx="160" cy="124" r="4.5" style="fill:var(--c-text)"/>
<circle cx="280" cy="46" r="4.5" style="fill:var(--c-text)"/>
<text x="120" y="272" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-style:italic">erkek (2, 1)</text>
<text x="252" y="182" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-style:italic">kadın (5, 4)</text>
<text x="148" y="118" text-anchor="end" style="fill:var(--c-text);font-size:13px;font-style:italic">kral (3, 6)</text>
<text x="280" y="32" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-style:italic">kraliçe (6, 9)</text>
<line x1="300" y1="116" x2="318" y2="116" style="stroke:var(--c-accent);stroke-width:2"/>
<text x="324" y="120" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">cinsiyet yönü (+3, +3)</text>
<line x1="300" y1="226" x2="318" y2="226" style="stroke:var(--c-accent-2);stroke-width:1.8;stroke-dasharray:6 5"/>
<text x="324" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">kraliyet yönü (+1, +5)</text>
</svg>

Vektör aritmetiği bu paralelkenarda kendini kanıtlar:

> kadın − erkek = (5, 4) − (2, 1) = **(+3, +3)** — *cinsiyet* yönü  
> kraliçe − kral = (6, 9) − (3, 6) = **(+3, +3)** — aynı ok, haritanın yukarısında  
> kral − erkek = kraliçe − kadın = **(+1, +5)** — *kraliyet* yönü, iki kez  

Okları birleştirdiğinizde meşhur denklem ortaya çıkar:

> kral − erkek + kadın = (3, 6) − (2, 1) + (5, 4) = **(6, 9) = kraliçe**

Gerçek embedding uzayları binlerce boyutta bu anlamsal geometriyi inşa eder
(anlamsal arama ve kosinüs benzerliği ayrıntıları için bkz:
[Embedding'ler derinlemesine](post.html?slug=embeddingler-derinlemesine)).

Ancak transformer mimarisinin temelinde önemli bir kör nokta vardır:
kelime sırasından habersizdir. Vektörleri torbaya atılmış gibi algılar.
"Köpek adamı ısırdı" ile "adam köpeği ısırdı" cümlelerinin aynı anlama
gelmemesi için modele **konum bilgisi (positional encoding)** verilmek
zorundadır. GPT-2 gibi eski modeller öğrenilmiş mutlak pozisyon vektörlerini
kelime embedding'ine eklerken, günümüz modern LLM'leri (Llama 3, Gemma 3)
**RoPE (Rotary Position Embedding)** kullanır; pozisyon bilgisini doğrudan
attention iç çarpımında vektörleri karmaşık düzlemde döndürerek işler.

## 3. Transformer: bir bağlam makinesi

Embedding tek başına "yüz" kelimesinin ne anlama geldiğini bilemez:
çehre olan yüz mü, sayı olan yüz mü, yoksa yüzmek fiili mi? Anlam ancak
bağlamda netleşir ve **transformer** — GPT'deki T — tam olarak bu bağlamı
çözmek için inşa edilmiş bir makinedir.

Modern bir LLM omurgası üç ana parçadan oluşur:
1. **Girdi temsili:** Token ID'lerinin embedding vektörlerine ve RoPE ile
   konumsal koordinatlara dönüştürülmesi.
2. **Transformer katmanları kulesi:** Bilgiyi token'lar arasında harmanlayan
   self-attention mekanizması ile her token'ı tek başına dönüştüren ileri
   beslemeli ağların (FFN) üst üste istiflenmesi.
3. **Çıktı başlığı (`lm_head`):** Son katmandaki vektörlerin sözlük
   boyutuna izdüşürülerek aday token puanlarına (logits) çevrilmesi.

Eski dil modelleri metni soldan sağa, mesafeyle solan dar bir hafıza
hücresinden (RNN/LSTM) geçirirdi. 2017 tarihli *Attention Is All You Need*
makalesi bu kısıtı çöpe attı ve tek bir çekirdek ilkeye dayandı:
**attention (dikkat)**. Her token, dizideki diğer her token'a doğrudan ve
aynı anda bakar; kimin ne kadar önemli olduğuna kendisi karar verir.

Makalenin ünlü örneğinde izleyin:

> Hayvan caddeyi geçmedi, çünkü **o** çok *yorgundu*.  
> Hayvan caddeyi geçmedi, çünkü **o** çok *genişti*.

Tek bir sıfat değişir ve "o" zamiri taraf değiştirir: birinde hayvandır,
ötekinde cadde. Siz bunu anında çözersiniz; attention ise modelin bu kararı
verme mekanizmasıdır: **bir kelimenin yeni anlamı, diğer kelimelerin
ağırlıklı karışımıdır — ve attention'ın tek işi bu ağırlıkları seçmektir.**

### Q, K, V — mekanizma, sayılarla

Her katmanda model, her token'ın embedding'inden öğrenilmiş üç ağırlık
matrisi ($W_Q, W_K, W_V$) çarparak üç ayrı vektör üretir:

- **Sorgu (Query, Q):** "Ben ne tür bir bilgi arıyorum?"
- **Anahtar (Key, K):** "Ben hangi bilgiyi sunuyorum, başkaları beni nasıl bulur?"
- **Değer (Value, V):** "Bana dikkat edilirse, aktaracağım asıl içerik nedir?"

YouTube aramasıyla düşünün: arama çubuğuna yazdığınız metin sorgudur (Q),
videoların başlık ve etiketleri anahtardır (K), videonun asıl içeriği ise
değerdir (V). Ham embedding'ler yetersiz kalırdı; çünkü arama tek bir
özelliğe odaklanmak ister — "k ile başlar" üzerinden değil, "yorgun
olabilir" üzerinden eşleşme gibi.

Attention mekanizması dört adımdan oluşur:

**1. Adım — Puanla:** Hedef token'ın sorgusu ile diğer kelimelerin
anahtarları iç çarpıma girer: $\text{puan}_i = Q \cdot K_i$.  
**2. Adım — Ölçekle:** Puanlar anahtar vektörünün boyutu $\sqrt{d_k}$'ye
bölünür. Bu bölme işlemi, büyük boyutlu vektörlerin iç çarpımlarının aşırı
büyüyüp softmax gradyanlarını sıfırlamasını engeller:
$\text{ölçekli}_i = \text{puan}_i \div \sqrt{d_k}$.  
**3. Adım — Softmax:** Üstel fonksiyon alınarak puanlar toplamı 1 olan
yüzdelere dönüştürülür.  
**4. Adım — Karıştır:** Değer (V) vektörleri bu yüzdelerle çarpılıp
toplanır.

> [!IMPORTANT]
> **Değer (V) vektörleri puanlama aşamasında kesinlikle hiçbir rol oynamaz.**
> Kimin kime ne kadar dikkat edeceği yalnızca Sorgu (Q) ve Anahtar (K)
> vektörlerinin iç çarpımıyla belirlenir. Değer (V), ağırlıklar
> kesinleştikten sonra taşınacak olan yükün ta kendisidir.

Modelin "Hızlı kahverengi tilki" cümlesinde *tilki* kelimesi üzerindeki
puanlamasını izleyin:

| çift | puan | e^puan | pay (ağırlık) |
|---|---|---|---|
| Q(tilki) · K(hızlı) | 2,1 | 8,2 | **%3** |
| Q(tilki) · K(kahverengi) | 4,0 | 54,6 | **%19** |
| Q(tilki) · K(tilki) | 5,4 | 221,4 | **%78** |

Softmax liderleri ödüllendirir: 5,4 puanı 4,0'dan çok büyük görünmez ama
üstel büyüme sayesinde %78, %19'un dört katı ağırlık kapar. Sonuçta yeni
vektör şudur:

$$\text{tilki}_{\text{yeni}} = 0{,}03 \cdot V(\text{hızlı}) + 0{,}19 \cdot V(\text{kahverengi}) + 0{,}78 \cdot V(\text{tilki})$$

Artık o sözlükteki soyut *tilki* değildir; *bu-belirli-hızlı-kahverengi-tilki*dir.
Formülün meşhur tek satırlık özeti:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

```mermaid
flowchart TD
    E["tilki'nin embedding'i"] -->|"× W_Q"| Q["Q — ne arıyorum?"]
    E -->|"× W_K"| K["K — nasıl bulunurum?"]
    E -->|"× W_V"| V["V — ne devrederim?"]
    Q --> S1["1. Adım · puan = Q · K"]
    K --> S1
    S1 --> S2["2. Adım · ölçek ÷ √dₖ"]
    S2 --> S3["3. Adım · softmax → yüzdeler"]
    S3 --> S4["4. Adım · karışım = Σ ağırlık × V"]
    V --> S4
    S4 --> OUT["yeni tilki — bu-belirli-hızlı-kahverengi-tilki"]
```

### Tek yön ve causal mask

Self-attention mimari olarak dizideki bütün token'lara aynı anda bakma
yeteneğine sahiptir. Ancak otoregresif bir üretimde modelin sıradaki
kelimeyi tahmin edebilmesi için **gelecekteki kelimeleri görmemesi şarttır**.
Aksi takdirde sınavda cevabı önceden görmek gibi hile yapar ve üretim
yeteneği kazanamaz.

Bu sınır **causal mask (nedensel maske / look-ahead mask)** ile çizilir.
Dizideki her $t$ pozisyonu için, kendisinden sonra gelen ($> t$) tüm
pozisyonların attention puanları softmax öncesinde eksi sonsuza ($-\infty$)
eşitlenir:

$$e^{-\infty} = 0$$

Böylece gelecekteki token'ların ağırlığı sıfıra kilitlenir; bilgi akışı
sadece geçmişten bugüne doğru tek yönlü akar. Causal mask eğitimde ve
çıkarımın ilk aşamasında (prefill) tüm token'lar paralel işlenirken hayati
bir emniyet kemeridir. Tekil üretimde (decode) ise gelecek token'lar henüz
ortada olmadığı için maske doğası gereği sağlanır. Maskeler aynı zamanda
farklı uzunluktaki cümleleri gruplarken eklenen dolgu (padding) token'larını
gizlemek için de kullanılır.

### Birçok kafa

Tek bir attention gözlüğü kaba kalırdı: bir kelimenin bir komşusundan dil
bilgisi uyumunu, kilometrelerce ötedeki bir isimden ise zamir referansını
aynı anda çekmesi gerekir.

Bu nedenle her katmanda birden fazla **kafa (attention head)** paralel
çalışır (Multi-Head Attention). Vektör boyutu kafa sayısına bölünür; örneğin
4096 boyut 32 kafaya bölündüğünde her kafa 128 boyutluk bir alt uzayda
uzmanlaşır. Biri özne-yüklem ilişkisini izlerken, diğeri zaman kipini, bir
başkası sıfat tamlamasını takip eder.

### Model aileleri: Neden herkes decoder-only?

Transformer mimarisi farklı görevler için üç ana çatallaşma yaşadı:

| Mimari | Temsilciler | Prefill ve Decode Aşamaları | Temel Görev Alanı |
|---|---|---|---|
| **Encoder-only** | BERT, RoBERTa | Yok. Tek bir ileri geçişte tüm girdiyi çift yönlü işler; üretim yapmaz. | Sınıflandırma, arama, embedding modelleri |
| **Encoder-decoder** | T5, FLAN-T5, BART | Kısmen. Encoder girdiyi çift yönlü okur; decoder otoregresif üretir. | Erken dönem çeviri ve özetleme |
| **Decoder-only** | GPT, Llama, Qwen, DeepSeek, Claude | Var. Causal maskeli paralel prefill ve ardından token token otoregresif decode. | Modern üretken yapay zekânın mutlak standardı |

Bugün LLM denildiğinde neredeyse istisnasız **decoder-only** modellerden
bahsedilmesinin üç somut mühendislik nedeni vardır:
1. **Tek bir hedef fonksiyonu:** Tüm veri tipleri (kod, sohbet, akıl yürütme)
   sıradaki token tahminine dönüştürülebilir; mimari ayrımına gerek kalmaz.
2. **Sıfır sürtünmeli KV cache:** Prefill'de hesaplanan tüm anahtar ve
   değerler tek bir tensör bloğunda saklanır ve üretim boyunca kesintisiz
   kullanılır.
3. **Ölçekleme verimliliği:** Milyarlarca parametreye çıkıldığında donanım
   üzerinde en kararlı ve en yüksek FLOPS verimini decoder-only yapılar
   sağlamıştır.

## 4. Katmanlar: bilgi nerede yaşıyor

Bir attention katmanı ile bir ileri beslemeli ağ (FFN), birlikte bir
**transformer bloğunu (layer)** oluşturur. Modern bir dil modeli bu
bloklardan onlarcasının (32, 80, hatta 120 kat) üst üste konmasıyla
kurulur:

- **Self-Attention (Kütüphaneci):** Diğer token'lardan bağlamı toplar,
  vektörleri birbiriyle kaynaştırır.
- **İleri Beslemeli Ağ / FFN (Ambar):** Toplanan bağlamı her token için tek
  başına sindirir ve dönüştürür.
- **Residual Bağlantı (Bina Kuralı):** Katmanın çıktısı girdisinin üstüne
  eklenir ($x + \text{Katman}(x)$); böylece katmanlar derinleştikçe sinyalin
  ve gradyanların sönümlenmesi engellenir.
- **Normalizasyon (RMSNorm / LayerNorm):** Tensörlerin sayısal normunu
  sabitleyerek katmanlar arası patlamaları önler.

Modelin öğrendiği olgusal bilgilerin (örneğin "Fransa'nın başkenti Paris'tir")
neredeyse üçte ikisi bu **ileri beslemeli ağların (FFN)** ağırlıklarında
depolanır. **Mixture of Experts (MoE)** mimarileri (örneğin Mixtral veya
DeepSeek), her kata tek bir devasa ambar koymak yerine onlarca küçük ambar
yerleştirir ve her token için en uygun 1-2 tanesini dinamik olarak seçer.

Kulenin en tepesinde borç ödenir: **Çıktı Başlığı (`lm_head`)**. Son
katmandan çıkan gizli durum vektörü, kelime dağarcığındaki tüm token'larla
çarpılarak her bir aday için ham logit puanı üretir:

$$\text{puan}(\text{aday}) = \text{son\_vektör} \cdot W_{\text{lm\_head}}(\text{aday})$$

Bu matris çoğu zaman girdi aşamasındaki `embed_tokens` tablosuyla aynı
ağırlıkları paylaşır (weight tying).

## 5. Eğitim ve ölçek

Makinedeki tüm sayılar rastgele gürültü olarak başlar. Onları hizalayan
süreç **ön eğitimdir (pretraining)**: modele internetten toplanan trilyonlarca
token metin gösterilir, sıradaki kelime gizlenir ve tahmin etmesi istenir.
Veri kendi kendini notlandırır; doğru token bellidir.

Hata **çapraz entropi kaybı (cross-entropy loss)** ile ölçülür:

$$\text{Loss} = -\log p(\text{doğru token})$$

Doğru cevaba %90 olasılık vermek $\approx 0{,}10$ kayba, %20 olasılık
vermek $\approx 1{,}60$ kayba yol açar. **Gradyan inişi (gradient descent)**
her parametreyi yokuş aşağı minicik bir adım kaydırarak modeli eğitir.

```mermaid
flowchart LR
    A["gerçek metni göster"] --> B["sıradaki token'ı gizle"]
    B --> C["model tahmin eder"]
    C --> D["kayıp = doğrunun −log p'si"]
    D --> E["gradyan inişi — minicik bir adım"]
    E -->|"trilyonlarca kez tekrar"| A
```

Ölçeğin getirisi tahmin edilebilir yasalara tabidir. Kaplan ve Chinchilla
**ölçekleme yasaları (scaling laws)**, model parametresi ile eğitim verisi
arasındaki optimum dengeyi parametre başına kabaca **20 token** olarak
sabitledi. 70 milyar parametreli Llama 2'nin 280 milyarlık Gopher'ı geride
bırakabilmesinin sırrı budur.

## 6. Otomatik tamamlamadan asistana

Ön eğitimin çıktısı bir **taban modeldir (base model)**: metin tamamlama
makinesi, o kadar. "Fransa'nın başkenti nedir?" yazdığınızda "Paris" cevabını
verebileceği gibi, arkasından dokuz farklı quiz sorusu da dizebilir; çünkü
eğitim verisinde soru listeleri de vardır.

Onu güvenilir bir asistana çevirmek iki aşamalı bir tornadan geçirir:
1. **Talimatla İnce Ayar (SFT - Supervised Fine-Tuning):** Yüz binlerce
   özenle yazılmış soru-cevap çiftiyle modele bir yardımcının nasıl
   davranması gerektiği öğretilir.
2. **İnsan Geri Bildirimiyle Pekiştirmeli Öğrenme (RLHF / DPO):** İnsanların
   tercih ettiği cevaplar üzerinden modelin üslubu, güvenliği ve dürüstlüğü
   pekiştirilir.

Parametrelerin tamamını eğitmek yerine kenara küçük adaptör matrisleri
ekleyen **LoRA**, bu ince ayar sürecini tek bir GPU'da bile yapılabilir kılar.

## 7. Çıkarım ve üretim: iki aşamalı döngü

Bir model eğitildikten sonra kullanıcı isteklerine cevap verirken çalışan
sürece **çıkarım (inference)** denir. Çıkarım, donanım kaynaklarını tüketim
biçimi açısından birbirine zıt **iki ayrı aşamadan** oluşur.

```mermaid
flowchart LR
    subgraph P["1. Prefill Aşaması (Compute-bound)"]
        direction TB
        PR["Prompt Token'ları"] --> PAR["Tüm token'lar paralel işlenir"]
        PAR --> KVW["KV Cache doldurulur"]
        KVW --> TTFT["İlk token üretilir (TTFT)"]
    end
    subgraph D["2. Decode Aşaması (Memory-bound)"]
        direction TB
        TTFT --> SEQ["Token t teker teker üretilir"]
        SEQ --> KVR["Geçmiş KV Cache okunur"]
        KVR --> ITL["Token arası gecikme (ITL)"]
        ITL --> STOP{"Durdurma token'ı mı?"}
        STOP -- Hayır --> SEQ
        STOP -- Evet --> END["Metin tamamlandı"]
    end
```

### Prefill: paralel hesaplama ve TTFT

Kullanıcı uzun bir metin gönderdiğinde, girdinin tamamı modelin elindedir.
Model bu token'ların tamamını devasa tensör matrisleriyle **aynı anda
(paralel)** işler.

- **Donanım karakteri:** Bu aşama **hesaplama-bağımlıdır (compute-bound)**.
  GPU'nun tensör çekirdekleri (Tensor Cores) sonuna kadar zorlanır; işlemci
  neredeyse hiç beklemez.
- **Kritik metrik:** **Time to First Token (TTFT)** — Kullanıcının enter'a
  bastığı andan ilk kelimenin ekranda belirdiği ana kadar geçen milisaniye.
- **Hayati görevi:** Prefill, prompt'taki bütün token'ların Anahtar (K) ve
  Değer (V) vektörlerini hesaplar ve bir sonraki aşama için **KV cache**
  alanına kaydeder.

Paralel olmak, her token'ın her token'ı gördüğü anlamına gelmez. Prompt
matrisinin satırları birlikte hesaplanır ama causal mask, $i$. pozisyonun
yalnızca $1 \ldots i$ pozisyonlarına bakmasına izin vermeye devam eder:
hesaplama paraleldir, bilgi akışı değil. İlk çıktı token'ı da ayrı bir adım
değildir; aynı geçişin sonunda, son pozisyonun logit'lerinden düşer:

<svg viewBox="0 0 560 345" role="img" aria-label="Prefill. Gökyüzü neden mavi görünür soru işareti prompt&#x27;unun altı token&#x27;ının hepsi baştan bilinir ve tek bir matris olarak birlikte 1. transformer katmanına girer. N katmanın her biri altı pozisyonun hepsi için aynı anda self-attention ve MLP çalıştırır ve altı token&#x27;ın anahtar ile değerlerini o katmanın KV cache&#x27;ine yazar. Son pozisyonun sözlük üzerindeki logit&#x27;leri ilk çıktı token&#x27;ı Güneş&#x27;i seçer; bu tek geçiş TTFT&#x27;nin büyük kısmıdır." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="pf-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="20" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Prompt — 6 token&#x27;ın hepsi baştan belli</text>
<rect x="16" y="30" width="56" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="44.0" y="47.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">Gök</text>
<line x1="44" y1="58" x2="44" y2="77" marker-end="url(#pf-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="80" y="30" width="56" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="108.0" y="47.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">yüzü</text>
<line x1="108" y1="58" x2="108" y2="77" marker-end="url(#pf-arr)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="144" y="30" width="56" height="26" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="172.0" y="47.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)"> neden</text>
<line x1="172" y1="58" x2="172" y2="77" marker-end="url(#pf-arr)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="208" y="30" width="56" height="26" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/><text x="236.0" y="47.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)"> mavi</text>
<line x1="236" y1="58" x2="236" y2="77" marker-end="url(#pf-arr)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="272" y="30" width="56" height="26" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="300.0" y="47.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)"> görünür</text>
<line x1="300" y1="58" x2="300" y2="77" marker-end="url(#pf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="336" y="30" width="56" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="364.0" y="47.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">?</text>
<line x1="364" y1="58" x2="364" y2="77" marker-end="url(#pf-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<text x="410" y="40" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">hepsi birlikte,</text>
<text x="410" y="56" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">tek geçişte işlenir</text>
<rect x="16" y="80" width="376" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="204" y="101" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Katman 1: self-attention + MLP</text>
<text x="204" y="119" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">6 pozisyon tek matris → GEMM</text>
<line x1="392" y1="105" x2="420" y2="105" marker-end="url(#pf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="80" width="122" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="432" y="96" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">KV · katman 1</text>
<rect x="432" y="104" width="14" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="450" y="104" width="14" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/>
<rect x="468" y="104" width="14" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/>
<rect x="486" y="104" width="14" height="18" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/>
<rect x="504" y="104" width="14" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="522" y="104" width="14" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="16" y="168" width="376" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="204" y="189" text-anchor="middle" style="fill:var(--c-text);font-size:13px">Katman N: self-attention + MLP</text>
<text x="204" y="207" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">causal mask: i. pozisyon yalnız 1…i&#x27;yi görür</text>
<line x1="392" y1="193" x2="420" y2="193" marker-end="url(#pf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="168" width="122" height="50" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="432" y="184" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">KV · katman N</text>
<rect x="432" y="192" width="14" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<rect x="450" y="192" width="14" height="18" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/>
<rect x="468" y="192" width="14" height="18" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/>
<rect x="486" y="192" width="14" height="18" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/>
<rect x="504" y="192" width="14" height="18" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/>
<rect x="522" y="192" width="14" height="18" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="204" y="152" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px">⋮</text>
<text x="216" y="151" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">her katmanda aynısı</text>
<text x="483" y="152" text-anchor="middle" style="fill:var(--c-text-mute);font-size:16px">⋮</text>
<line x1="204" y1="218" x2="204" y2="240" marker-end="url(#pf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="16" y="242" width="376" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="258" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">sözlük üzerinde logit&#x27;ler (yalnız son pozisyon)</text>
<rect x="40" y="286" width="14" height="6" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="64" y="283" width="14" height="9" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="88" y="287" width="14" height="5" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="112" y="280" width="14" height="12" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="136" y="285" width="14" height="7" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="160" y="288" width="14" height="4" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="184" y="282" width="14" height="10" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="208" y="266" width="14" height="26" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<rect x="232" y="284" width="14" height="8" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="256" y="287" width="14" height="5" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="280" y="281" width="14" height="11" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="304" y="286" width="14" height="6" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="328" y="288" width="14" height="4" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="352" y="285" width="14" height="7" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<line x1="392" y1="270" x2="420" y2="270" marker-end="url(#pf-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="422" y="242" width="122" height="56" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="483" y="258" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">ilk çıktı token&#x27;ı</text>
<rect x="438" y="266" width="90" height="24" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="483.0" y="282.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">Güneş</text>
<text x="16" y="324" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">İlk token prefill geçişinin kendisinden çıkar; bu yüzden TTFT&#x27;nin çoğu bu geçiştir:</text>
<text x="16" y="340" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">prompt uzadıkça her matris çarpımında satır sayısı, dolayısıyla bekleme artar.</text>
</svg>

### Decode: seri döngü ve ITL

İlk token ekrana düştükten sonra model **decode** aşamasına geçer. Burada
artık paralel üretim imkânsızdır; çünkü 50. kelimenin ne olacağını bilmek
için 49. kelimenin üretilmiş olması gerekir. Üretim otoregresif olarak,
**teker teker ve seri** akar.

- **Donanım karakteri:** Bu aşama **bellek-bant-genişliği bağımlıdır
  (memory-bandwidth bound)**. Her tek bir token üretilirken, GPU'nun tüm
  ağırlık matrisleri ve büyüyen KV cache belleği (HBM) baştan sona okunmak
  zorundadır. GPU'nun devasa hesaplama birimleri çoğu zaman bellekten verinin
  gelmesini bekleyerek boş yatar.
- **Kritik metrik:** **Inter-Token Latency (ITL)** — Art arda gelen iki token
  arasında geçen süre (kullanıcının hissettiği akış hızı).

Her adımda katman yığınından tam olarak bir yeni token geçer ve prefill'in
(ve önceki her adımın) bıraktığı önbelleği okur:

<svg viewBox="0 0 560 330" role="img" aria-label="Decode. KV cache altı prompt token&#x27;ının anahtar ve değerlerini tutar. Tek yeni token, Güneş, katman yığınına tek başına girer. Her katmanda self-attention yeni token&#x27;ın sorgusunu önbellekteki tüm anahtarlarla karşılaştırıp önbellekteki değerleri karıştırır, ardından yeni token&#x27;ın kendi anahtar ve değerini önbelleğe ekler; MLP yalnız yeni token için çalışır. Logit&#x27;ler sıradaki token&#x27;ı, ışığı, seçer; o da dizi sonu token&#x27;ı gelene kadar bir sonraki girdi olarak döngüye döner." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="dc-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="dc-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="16" width="180" height="256" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="36" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">KV cache</text>
<text x="26" y="52" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">prefill + önceki adımlar</text>
<text x="104" y="72" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">K</text>
<text x="158" y="72" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">V</text>
<text x="26" y="94" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Gök</text>
<rect x="82" y="80" width="44" height="18" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="136" y="80" width="44" height="18" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="26" y="120" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">yüzü</text>
<rect x="82" y="106" width="44" height="18" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<rect x="136" y="106" width="44" height="18" rx="4" style="fill:var(--c-success);fill-opacity:.2;stroke:var(--c-success);stroke-width:1.2"/>
<text x="26" y="146" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">neden</text>
<rect x="82" y="132" width="44" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="136" y="132" width="44" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.2;stroke:var(--c-warn);stroke-width:1.2"/>
<text x="26" y="172" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">mavi</text>
<rect x="82" y="158" width="44" height="18" rx="4" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<rect x="136" y="158" width="44" height="18" rx="4" style="fill:var(--c-danger);fill-opacity:.2;stroke:var(--c-danger);stroke-width:1.2"/>
<text x="26" y="198" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">görünür</text>
<rect x="82" y="184" width="44" height="18" rx="4" style="fill:var(--c-text-mute);fill-opacity:.2;stroke:var(--c-text-mute);stroke-width:1.2"/>
<rect x="136" y="184" width="44" height="18" rx="4" style="fill:var(--c-text-mute);fill-opacity:.2;stroke:var(--c-text-mute);stroke-width:1.2"/>
<text x="26" y="224" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">?</text>
<rect x="82" y="210" width="44" height="18" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="136" y="210" width="44" height="18" rx="4" style="fill:var(--c-accent);fill-opacity:.2;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="26" y="250" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">Güneş</text>
<rect x="82" y="236" width="44" height="18" rx="4" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<rect x="136" y="236" width="44" height="18" rx="4" style="fill:var(--c-accent-2);fill-opacity:.08;stroke:var(--c-accent-2);stroke-width:1.2;stroke-dasharray:4 3"/>
<text x="320" y="14" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">her adımda tek yeni token</text>
<rect x="270" y="20" width="100" height="26" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="320.0" y="37.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">Güneş</text>
<line x1="320" y1="46" x2="320" y2="66" marker-end="url(#dc-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="216" y="68" width="188" height="204" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="310" y="86" text-anchor="middle" style="fill:var(--c-text);font-size:13px">N katmanın her biri</text>
<rect x="228" y="96" width="164" height="80" rx="8" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.2"/>
<text x="310" y="114" text-anchor="middle" style="fill:var(--c-text);font-size:13px">self-attention</text>
<text x="310" y="132" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">yalnız yeni token&#x27;ın q&#x27;su</text>
<text x="310" y="148" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">önbellekteki tüm K&#x27;lerle</text>
<text x="310" y="164" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">→ önbellekteki V karışımı</text>
<line x1="196" y1="120" x2="226" y2="120" marker-end="url(#dc-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<text x="205" y="100" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px" transform="rotate(-90 205 100)">oku</text>
<path d="M228 166 H208 V245 H182" marker-end="url(#dc-arr-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5"/>
<text x="203" y="210" text-anchor="middle" style="fill:var(--c-accent-2);font-size:11px" transform="rotate(-90 203 210)">k, v ekle</text>
<line x1="310" y1="176" x2="310" y2="188" marker-end="url(#dc-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="228" y="190" width="164" height="44" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<text x="310" y="208" text-anchor="middle" style="fill:var(--c-text);font-size:13px">MLP</text>
<text x="310" y="225" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">yalnız yeni token</text>
<text x="310" y="258" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">tam ileri geçiş, 1 pozisyon</text>
<line x1="404" y1="150" x2="432" y2="150" marker-end="url(#dc-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="434" y="68" width="110" height="132" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="489" y="86" text-anchor="middle" style="fill:var(--c-text);font-size:13px">logits</text>
<rect x="444" y="98" width="80" height="12" rx="2" style="fill:var(--c-accent-2);fill-opacity:.85"/>
<rect x="444" y="116" width="52" height="12" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="444" y="134" width="30" height="12" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="444" y="152" width="20" height="12" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<rect x="444" y="170" width="12" height="12" rx="2" style="fill:var(--c-text-mute);fill-opacity:.35"/>
<line x1="489" y1="200" x2="489" y2="222" marker-end="url(#dc-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="434" y="224" width="110" height="48" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="489" y="240" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">sıradaki token</text>
<rect x="446" y="246" width="86" height="20" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="489.0" y="260.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)"> ışığı</text>
<path d="M544 248 H553 V33 H374" marker-end="url(#dc-arr-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="550" y="26" text-anchor="end" style="fill:var(--c-accent-2);font-size:12px">&lt;eos&gt; gelene kadar tekrar</text>
<text x="16" y="298" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Yeni token için hiçbir blok atlanmaz: her attention ve MLP yine çalışır. Önbelleğin kurtardığı şey</text>
<text x="16" y="314" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">eski token&#x27;ları yeniden işlemektir. Her adım tek token için bütün ağırlıkları + KV cache&#x27;i okur.</text>
</svg>

"KV cache hesaplamadan tasarruf ettirir" cümlesi sık sık yanlış okunur:
sanki decode sırasında ağın bir kısmı atlanıyormuş gibi. Atlanmaz. Yeni
token her katmanın attention bloğundan da MLP'sinden de geçer. Önbelleğin
kurtardığı şey *eski* token'ları yeniden işlemektir: onların anahtar ve
değerleri yeniden hesaplanmaz, okunur; MLP de onları bir daha hiç görmez.

Aynı ağırlıklar, aynı katmanlar; ama GPU açısından birbirine hiç benzemeyen
iki iş yükü:

<svg viewBox="0 0 560 290" role="img" aria-label="Yan yana. Prefill: baştan bilinen çok sayıda token modele paralel girer, ağırlıklar bir kez okunup hepsi için kullanılır, ilk token sonda çıkar; matris-matris işi, hesaplama-bağımlı. Decode: her adımda tek yeni token, model KV cache&#x27;i okur ve ona ekler, tek token çıkar ve döngüye döner; matris-vektör işi, bellek-bant-genişliği bağımlı. Çok sayıda isteği gruplamak decode&#x27;a paralelliği geri getirir." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="cmp-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="cmp-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<rect x="16" y="16" width="256" height="226" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<rect x="288" y="16" width="256" height="226" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="32" y="40" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">PREFILL</text>
<rect x="33" y="54" width="32" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<line x1="49" y1="78" x2="49" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="71" y="54" width="32" height="22" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/>
<line x1="87" y1="78" x2="87" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="109" y="54" width="32" height="22" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/>
<line x1="125" y1="78" x2="125" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="147" y="54" width="32" height="22" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/>
<line x1="163" y1="78" x2="163" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-danger);stroke-width:1.5"/>
<rect x="185" y="54" width="32" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/>
<line x1="201" y1="78" x2="201" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="223" y="54" width="32" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/>
<line x1="239" y1="78" x2="239" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="33" y="98" width="222" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="144" y="123" text-anchor="middle" style="fill:var(--c-text);font-size:13px">model · ağırlıklar bir kez</text>
<line x1="144" y1="138" x2="144" y2="156" marker-end="url(#cmp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="99" y="158" width="90" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="144.0" y="173.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ilk token</text>
<text x="32" y="206" text-anchor="start" style="fill:var(--c-text);font-size:12px">S token tek ağırlık okumasını paylaşır</text>
<text x="32" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GEMM (matris × matris) → hesap-bağımlı</text>
<text x="304" y="40" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">DECODE</text>
<rect x="356" y="54" width="32" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/>
<line x1="372" y1="78" x2="372" y2="96" marker-end="url(#cmp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="304" y="98" width="136" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="372" y="123" text-anchor="middle" style="fill:var(--c-text);font-size:13px">model</text>
<rect x="458" y="98" width="72" height="40" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="494" y="123" text-anchor="middle" style="fill:var(--c-text);font-size:12px">KV cache</text>
<line x1="458" y1="112" x2="442" y2="112" marker-end="url(#cmp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<line x1="440" y1="126" x2="456" y2="126" marker-end="url(#cmp-arr-g)" style="stroke:var(--c-accent-2);stroke-width:1.5"/>
<line x1="372" y1="138" x2="372" y2="156" marker-end="url(#cmp-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="332" y="158" width="80" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="372.0" y="173.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">1 token</text>
<path d="M332 169 C 292 169, 292 65, 352 65" marker-end="url(#cmp-arr-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.5;stroke-dasharray:5 4"/>
<text x="296" y="192" text-anchor="start" style="fill:var(--c-accent-2);font-size:11px">tekrar</text>
<text x="304" y="206" text-anchor="start" style="fill:var(--c-text);font-size:12px">her ağırlık okumasına 1 yeni token</text>
<text x="304" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:11.5px">GEMV (matris × vektör) → bellek-bağımlı</text>
<text x="16" y="266" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Tek isteğin decode&#x27;u kesinlikle seridir; paralellik istekler arasından geri gelir.</text>
<text x="16" y="282" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">B decode akışını gruplamak, ağırlık okuması başına 1 token&#x27;ı B&#x27;ye çıkarır.</text>
</svg>

Fark, ağırlıkların her baytının ne kadar iş satın aldığında. Prefill'de bir
ağırlık matrisi bir kez okunur ve prompt'un $S$ satırının hepsiyle çarpılır;
decode'da aynı okuma istek başına tek bir satıra hizmet eder. Çıkarım
motorlarının decode adımlarını çok sayıda kullanıcı arasında gruplamak için
bu kadar uğraşmasının nedeni budur;
[vLLM'deki continuous batching (sürekli gruplama)](post.html?slug=vllm-derinlemesine)
fikrinin özü de bu.

### KV cache'in gerçek boyutu ve bellek duvarı

Attention mekanizmasında her yeni token geçmişteki tüm token'lara bakmak
zorundadır. Eğer bir önbellek olmasaydı, 1000. token üretilirken önceki 999
kelimenin K ve V vektörleri sıfırdan tekrar hesaplanacaktı ($O(n^2)$ işlem
faturası).

**KV Cache**, hesaplanan K ve V vektörlerini GPU belleğinde saklar. Yeni
adımda yalnızca son token için Q, K, V hesaplanır; Q geçmişteki önbellek ile
karşılaştırılır ve yeni K, V önbelleğe eklenir ($O(1)$ hesaplama).

Ancak bu hızın bedeli çok ağırdır: **VRAM tüketimi**. KV cache boyutu şu
formülle hesaplanır:

$$\text{Boyut} = 2 \times \text{katman} \times \text{kafa}_{\text{kv}} \times d_{\text{kafa}} \times \text{bağlam} \times \text{bayt}$$

Somut bir veri merkezi örneğiyle hesaplayalım: **Llama-3-8B (FP16)**:
- Model ağırlıkları GPU'da $\approx 16\text{ GB}$ yer tutar.
- 8.000 token'lık (8K) tek bir sohbetin KV cache'i $\approx 1\text{ GB}$
  tüketir.
- 80 GB'lık bir NVIDIA A100/H100 GPU'da ağırlıklar ve sistem çıktıktan sonra
  geriye yaklaşık 64 GB KV cache alanı kalır.
- **Sonuç:** 80 GB'lık dev bir GPU, 8K bağlam kullanan **en fazla ~60
  eşzamanlı kullanıcıya** hizmet verebilir!

Bir sunucunun kapasite tavanını belirleyen şey model ağırlıkları değil,
**KV cache'in ta kendisidir**. Bu darboğazı aşmak için modern çıkarım
sistemleri üç büyük teknik geliştirmiştir:
1. [vLLM ve PagedAttention](post.html?slug=vllm-derinlemesine): Belleği
   sanal sayfalar halinde yöneterek parçalanmayı (fragmentation) sıfırlar
   ve eşzamanlılığı 2-4 katına çıkarır.
2. **Prefix Caching:** Ortak sistem prompt'larının KV bloklarını hafızada
   tutup yeniden hesaplamayı önler (ayrıntılar için bkz:
   [LLM maliyet ve gecikme optimizasyonu](post.html?slug=llm-maliyet-ve-gecikme-optimizasyonu)).
3. [Quantization (Nicemleme)](post.html?slug=post-training-quantization-llm-cikarim):
   Ağırlıkları ve KV cache'i FP8 veya INT4 formatına indirerek bellek
   ihtiyacını yarıya böler.

### Prefill ve decode'un çakışması: disaggregation

Geleneksel sunucularda prefill ile decode aynı GPU üzerinde yan yana
koşturulur. Ancak ağır bir prefill isteği geldiğinde GPU'nun işlem
çekirdeklerini kilitler; bu sırada milisaniyelerle token yazmakta olan decode
istekleri donar ve kullanıcı tarafında kekeleme (ITL sıçraması) yaşanır.

Modern büyük sistemler bu sorunu **Prefill-Decode Disaggregation (Ayrık
Mimari)** ile çözer: prefill istekleri işlemci gücü yüksek ayrı GPU'larda
yapılır; oluşturulan KV cache blokları ultra hızlı ağlar üzerinden decode
GPU'larına aktarılır. Böylece iki aşama birbirini ezmez.

### Çekilişi yöneten üç kadran: temperature, top-k, top-p

Model son katmanda sözlükteki her kelime için bir logit puanı üretir. Bu
puanların gerçek bir kelimeye dönüşmesini üç parametre yönetir:

- **Temperature (Sıcaklık):** Logit'leri softmax öncesinde $T$'ye böler
  ($z_i \div T$). Düşük sıcaklık ($T \to 0$) dağılımı aşırı sivrileştirir; en
  yüksek puanlı kelimeyi zorunlu kılar (deterministik / greedy — kod ve SQL
  için ideal). Yüksek sıcaklık dağılımı yayvanlaştırır; sürpriz kelimelere şans
  tanır (yaratıcı yazım).
- **Top-k:** En olası $k$ kelimeyi tutar, geriye kalan tüm alternatifleri
  kesip atar.
- **Top-p (Nucleus Sampling):** Kümülatif olasılık toplamı $p$'ye (örneğin
  %90) ulaşana kadar en olası token'ları sepete atar; model kendinden çok
  eminse iki token, kararsızsa seksen token arasından seçim yapar.

## 8. Sizi hatırlamaz: bağlam penceresi

Eğitim bittiğinde modelin ağırlıkları **tamamen donar**. Bir model API'sine
"Benim adım neydi?" diye sorduğunuzda model dünkü konuşmayı hatırlamaz;
çünkü bir iç hafızası yoktur.

Sohbetin sürmesini sağlayan şey, arayüzün her yeni mesajda geçmiş konuşma
dökümünü prompt'un başına ekleyerek modele tekrar göndermesidir. Hafıza
sandığınız şey, **bağlam penceresidir (context window)**. Modelin bir görevi
tek satır kod değiştirmeden prompt içindeki örneklerden öğrenmesine ise
**bağlam içi öğrenme (in-context learning)** denir.

## 9. Neden uyduruyor

2023 yılında *Mata v. Avianca* davasının avukatları, ChatGPT'nin uydurduğu
altı hayali mahkeme kararını yargıca sundular ve 5.000 dolar ceza aldılar.

Bu durum bir hata değil, mimarinin doğasıdır: **LLM bir arama motoru veya
veritabanı değil, bir olasılık motorudur.** Model doğruluğu değil, eğitim
verisine göre *en makul görünen* devamı optimize eder. Verinin bol olduğu
yerde makul olan genellikle doğrudur; bilginin az olduğu yerde ise model
doğruyu bilmediği için değil, cümleyi kurallara uygun tamamlamak zorunda
olduğu için uydurur.

Çözüm adımları bir maliyet merdivenidir:
1. **Prompt Mühendisliği:** Cevap sınırlarını bağlamda çizmek.
2. **RAG (Retrieval-Augmented Generation):** Doğru bilgiyi harici bir
   vektör veritabanından alıp prompt'a eklemek.
3. **Fine-Tuning:** Özel alan terminolojisini modelin kalıcı ağırlıklarına
   işlemek.

## 10. Otoregresyonun ötesi: Difüzyon LLM'leri (dLLM)

Otoregresif decoder-only mimarinin temel bir tavanı vardır: token'ları
teker teker üretmek, kaçınılmaz bir seri hesaplama ve bellek bant genişliği
darboğazı yaratır.

Bu kısıtı kırmak için ortaya çıkan en umut verici yeni paradigma **Difüzyon
LLM'leridir (dLLM - Diffusion LLMs)**. Görsel üretimindeki (Stable
Diffusion) gürültü giderme mantığını metne uyarlarlar:
1. Model, hedef yanıt uzunluğunda rastgele bir gürültü tensörüyle başlar.
2. Birkaç adımda bu gürültüyü paralel olarak rafine eder ve tutarlı metne
   dönüştürür.
3. Yanıt teker teker değil, sisin dağılması gibi **bütün halinde ve aynı anda**
   ortaya çıkar.

Bu paralel süreç token-başına gecikme duvarını yıkar ve modelin üretim
sırasında geriye dönüp kendi kendini düzeltmesine (öz-düzeltme / global
editing) olanak tanır. Inception AI'ın **Mercury** modeli ve Google
DeepMind'ın **Gemini Diffusion** araştırması bu alandaki ilk güçlü
adımlardır. Bugün üretim sistemlerinde otoregresif modeller hâlâ mutlak
hâkimdir; ancak difüzyon mimarileri geleceğin çıkarım sistemleri için en
büyük adaylardan biridir.

## Bütün hikâye altı satırda

1. Metin $\to$ **token** $\to$ **embedding** (ayrık sayılardan sürekli
   geometriye; RoPE ile içe işlenen sıra).
2. **Attention** (Q, K, V) bağlamı toplar (V puanlamaya girmez, sadece
   taşınır); causal mask geleceğe bakışı kilitler; bilgiyi FFN katmanları
   saklar.
3. Mimari standart **decoder-only**'dir; **ön eğitim** ölçekli sıradaki-token
   tahminidir; Chinchilla yasası parametre başına ~20 token önerir.
4. Çıkarım iki fazlıdır: **Prefill** paralel ve hesaplama-bağımlıdır (TTFT);
   **Decode** seri ve bellek-bağımlıdır (ITL).
5. **KV cache** geçmiş K ve V'leri saklayarak çıkarımı mümkün kılar, ancak
   VRAM'i tüketerek sunucunun eşzamanlılık tavanını belirler.
6. Ağırlıklar donuktur, hafıza bağlam penceresidir; model doğruyu değil
   *makul olanı* optimize ettiği için uydurur.

Ve bir dahaki sefere biri bu modellerin nasıl çalıştığını sorduğunda — bir
mülakatçı, bir öğrenci ya da içinizdeki meraklı ses — modelin başladığı
yerden başlayın: sıradaki token'dan.

## Daha derine inmek için

- Vaswani vd., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017) — orijinal Transformer makalesi.
- Modular, [How does an LLM work?](https://handbook.modular.com/llms.txt) — çıkarım fazları ve sistem mimarisi el kitabı.
- Abhinandan, [Prefill & Decode made Crystal Clear](https://x.com/abhibuilds/status/2106006454046216461) (2026) — 7. bölümdeki prefill ve decode diyagramlarına ilham veren, ilk prensiplerden giden bir anlatım.
- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — klasikleşmiş görsel anlatım.
- Andrej Karpathy, [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — bütün makinenin sıfırdan Python ile inşası.
- Bu blogda birbiriyle konuşan diğer derinlemesine incelemeler:
  - [Bir Prompt'un Yolculuğu (1): Tokenizasyon](post.html?slug=tokenizasyon-nasil-calisir) — metinden token ID'sine uzanan BPE masası.
  - [Bir Prompt'un Yolculuğu (2): Embedding Katmanı](post.html?slug=embedding-katmani-derinlemesine) — ayrık sayılardan sürekli geometriye ve $\sqrt{d_{\text{model}}}$ ölçeklemesine.
  - [Bir Prompt'un Yolculuğu (3): Anlamsal Embedding'ler](post.html?slug=embeddingler-derinlemesine) — anlamsal arama ve çok boyutlu vektör uzayları.
  - [vLLM derinlemesine](post.html?slug=vllm-derinlemesine) — PagedAttention, blok tabloları ve KV cache yönetimi.
  - [LLM maliyet ve gecikme optimizasyonu](post.html?slug=llm-maliyet-ve-gecikme-optimizasyonu) — TTFT, ITL, bellek duvarı ve prefix caching.
  - [Post-training quantization](post.html?slug=post-training-quantization-llm-cikarim) — FP16'dan INT4'e ağırlık ve KV cache sıkıştırması.
