En güçlü modele "strawberry" kelimesinde kaç tane r olduğunu sorun,
hâlâ yanlış cevap verebilir. Bir kelimeyi tersten yazmasını isteyin,
bocalar. Aynı model bu sırada çalışan bir derleyici geçişi yazıyordur.
Arıza bir akıl yürütme boşluğu gibi görünür ama değildir: model o
harfleri hiç görmedi. "strawberry" ilk katmana ulaştığında çoktan üç
tamsayıya dönüşmüştü — `str`, `aw`, `berry` — ve girdide hiç
bulunmayan bir şeyi sayamazsınız.

Her prompt, ağa varmadan önce bir pasaport kontrol masasından geçer.
O masada metin, sabit bir listedeki parçalara bölünür ve her parçaya
bir tamsayı damgası vurulur. Damgasız hiçbir şey geçemez ve ağ yalnızca
damgaları görür. Bu yazı o masanın içinden geçiyor: sabit liste neden
var, BPE onu nasıl kuruyor, algoritmalar nerede ayrışıyor, vocabulary
(kelime dağarcığı) boyutunun bedeli ne, Türkçe aynı masada neden daha
fazla ödüyor, masanın yarattığı arızalar neler — ve tamsayılar sonunda
nasıl vektöre dönüşüyor.

**Bu yazıda**

- [1. Sabit liste neden var](#1-sabit-liste-neden-var)
- [2. BPE listeyi nasıl kuruyor](#2-bpe-listeyi-nasıl-kuruyor)
- [3. Dört algoritma, tek tablo](#3-dört-algoritma-tek-tablo)
- [4. Token ID yalnızca bir satır numarasıdır](#4-token-id-yalnızca-bir-satır-numarasıdır)
- [5. Vocabulary boyutu ne kazandırır ne götürür](#5-vocabulary-boyutu-ne-kazandırır-ne-götürür)
- [6. Türkçe aynı masada neden daha fazla ödüyor](#6-türkçe-aynı-masada-neden-daha-fazla-ödüyor)
- [7. Masanın kırdıkları](#7-masanın-kırdıkları)
- [8. Damgadan vektöre](#8-damgadan-vektöre)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

## 1. Sabit liste neden var

Bir sinir ağı matris çarpar. "h" harfini çarpamaz. Dolayısıyla metni
sayılara çeviren bir şey olmak zorundadır ve bu eşleme sonlu bir
listeye bakış olmak zorundadır — çünkü modelin ilk katmanı, listedeki
her girdi için bir satır tutan bir tablodur ve tabloların satır sayısı
sabit olmalıdır.

> **Vocabulary (kelime dağarcığı)** = bir tokenizer'ın bildiği sabit
> ve sonlu metin parçaları listesi. Her girdinin bir satır numarası
> vardır. Bu parçalardan kurulamayan metin modele giremez.

Akla ilk gelen liste *bütün kelimeler* olmaktır. İki yerden çöker.
Tek başına İngilizce bile çekimleri, özel isimleri, kod tanımlayıcılarını
ve yazım hatalarını sayınca yüz binlerce forma çıkar ve her biri
embedding matrisinde kendi satırını ister. Daha kötüsü, liste eğitim
anında kapanır: kullanıcılarınızın yazdığı listede olmayan ilk kelime
— yeni bir ürün adı, bir yazım hatası — tek bir `[UNK]` token'ına
düşer ve anlamı kaybolur.

Karşı uçtaki liste *bütün karakterler* olmaktır. Vocabulary birkaç
yüze iner ve hiçbir şey bilinmez kalmaz. Ama bu kez "tokenizasyon"
iki yerine on iki yuva harcar ve attention maliyeti kabaca dizi
uzunluğunun karesiyle büyür. Tek başına neredeyse hiç anlam taşımayan
parçalar için karesel bir bedel ödersiniz.

Subword (alt kelime) tokenizasyonu ortayı tutar. Sık kelimeler bütün
kalır; seyrek olanlar hâlâ anlamlı parçalara ayrılır.

```text
"tokenization"   →  ["token", "ization"]     2 parça, ikisi de anlamlı
"antidisestablishmentarianism"
                 →  ["anti", "dis", "establish", "ment", "arian", "ism"]
"Zyxwvut"        →  ["Zy", "x", "w", "vut"]  asla bilinmez değil, sadece uzun
```

Anlaşma açıktır: sık metin ucuz, seyrek metin pahalı, hiçbir şey
imkânsız değil. Aşağıdaki her şeyi bu takas belirliyor.

## 2. BPE listeyi nasıl kuruyor

**BPE (byte pair encoding — bayt çifti kodlaması)** vocabulary'yi tek
karakterlerden başlayarak kurar ve en sık görünen komşu çifti tekrar
tekrar birbirine yapıştırır. Bu bir sayma döngüsüdür, bir gramer
değil.

Hugging Face tokenizer özetindeki yürüyüşü izleyelim; beş kelimelik
bir derlem ve frekansları:

```text
("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)

Temel vocabulary: ["b", "g", "h", "n", "p", "s", "u"]
Karakterlere bölünmüş hâli:
("h" "u" "g", 10) ("p" "u" "g", 5) ("p" "u" "n", 12)
("b" "u" "n", 4)  ("h" "u" "g" "s", 5)
```

Şimdi komşu çiftleri sayalım. `"u" "g"` çifti *hug* (10), *pug* (5) ve
*hugs* (5) içinde geçiyor — 20 kez. `"u" "n"` çifti *pun* (12) ve
*bun* (4) içinde geçiyor — 16 kez. Yani ilk birleşmeyi `ug` kazanır:

```text
Birleşme 1:  u + g → "ug"     (20 kez)
("h" "ug", 10) ("p" "ug", 5) ("p" "u" "n", 12)
("b" "u" "n", 4) ("h" "ug" "s", 5)

Birleşme 2:  u + n → "un"     (16 kez)
("h" "ug", 10) ("p" "ug", 5) ("p" "un", 12)
("b" "un", 4)  ("h" "ug" "s", 5)

Vocabulary şimdi: ["b","g","h","n","p","s","u","ug","un"]
```

Vocabulary hedef boyuta ulaşana kadar bu tekrarlanır. Pratikte iki
sonuç önemlidir.

Birincisi, **birleşme listesi tokenizer'ın kendisidir**. Yeni metni
kodlamak, o birleşmeleri öğrenildikleri sırayla yeniden oynatmak
demektir; yani birleşme sırası da nihai kelime listesi kadar
artefaktın parçasıdır.

İkincisi, **vocabulary boyutu bir aritmetiktir**: temel karakterler
artı birleşme sayısı. GPT-1, 478 temel token artı 40.000 birleşmeden
oluşan 40.478 token kullandı. GPT-2 byte-level BPE kullandı: temel
olarak 256 bayt değeri, artı 50.000 birleşme, artı bir metin sonu
token'ı — tam olarak 50.257.

O bayt tabanlı temel, `[UNK]` sorununun sessiz çözümüdür. Atomlarınız
256 olası bayt değeriyse, olası *her* girdi — emoji, Çince, bozuk
baytlar, metin olarak yapıştırılmış bir PNG — temsil edilebilir.
Bilinmeyen token diye bir şey kalmaz. GPT-4 ve Llama 3'ün ikisinin de
byte-level BPE kullanmasının nedeni budur.

## 3. Dört algoritma, tek tablo

BPE en yaygın olanıdır ama tek seçenek değildir. Dördü esas olarak
**neyi birleştireceğine ya da tutacağına nasıl karar verdiğiyle**
ayrışır.

| Algoritma | Başlangıç | Karar kuralı | Kullananlar |
| :--- | :--- | :--- | :--- |
| **BPE** | karakterler | **en sık** komşu çifti birleştir | GPT-2, Llama, Qwen2 |
| **Byte-level BPE** | 256 bayt değeri | aynı kural, ama `[UNK]` imkânsız | GPT-2/4, Llama 3 |
| **WordPiece** | karakterler + `##` | en **şaşırtıcı** çifti birleştir | BERT, DistilBERT |
| **Unigram** | geniş bir aday havuzu | derlemin en az ihtiyaç duyduklarını **buda** | T5, Pegasus |

Anlaşılmaya değer satır WordPiece satırıdır, çünkü "en şaşırtıcı"
kulağa muğlak geliyor ama aslında bir formül:

```text
skor(a, b) = frekans(ab) / ( frekans(a) × frekans(b) )
```

BPE *bu ikisi birlikte ne sıklıkta geçiyor* diye sorar. WordPiece
*rastlantının öngördüğünden ne kadar daha sık geçiyor* diye sorar.
Yukarıdaki derlemde `u`+`g` 20 kez geçer ve 20/(36×20) = 0,028 alır;
`g`+`s` ise yalnızca 5 kez geçmesine rağmen 5/(20×5) = 0,050 alır —
dolayısıyla WordPiece önce `gs`'yi birleştirir. Yaygın iki karakterin
bir birleşmeyi hak etmesi çok daha zordur ve bu, vocabulary'yi
gerçekten birbirine ait olan parçalara doğru iter.

**Unigram** tersten çalışır. Fazlasıyla çok sayıda aday subword ile
başlar ve her biri için şunu sorar: *bunu silsem derlem ne kadar daha
kötü tokenize olurdu?* Sonra en alttaki %10–20'yi siler. `"pu"`yu
silmek neredeyse bedavadır, çünkü `pug` ve `pun` hâlâ `["p","ug"]` ve
`["p","un"]` diye yazılabilir. `"ug"`u silmek pahalıdır, çünkü üç
kelime ona yaslanır. Bir yan etki: Unigram bir kelimeyi birkaç farklı
şekilde yazabilir ve en olasısını seçer; yani BPE deterministikken
Unigram olasılıksaldır.

**SentencePiece** dördüncü bir algoritma değildir — BPE ya da
Unigram'ı ham karakter akışı üzerinde çalıştıran bir kütüphanedir.
Asıl katkısı boşluğu sıradan bir karakter gibi ele almasıdır; `▁`
diye yazılır. Bu, kelimeler arasına boşluk koymayan Çince, Japonca ve
Tayca için önemlidir ve çözümlemeyi tam tersine çevrilebilir kılar:
token'ları yapıştırın, `▁`yi boşluğa çevirin, bitti.

Bütün bunlardan önce bir **pre-tokenizer (ön bölücü)** ham metni
parçalara ayırır — genellikle boşluk ve noktalama üzerinden — ki
birleşmeler bir kelime sınırını aşıp bir kelimenin sonuyla diğerinin
başını yapıştıramasın. GPT ailesinin tokenizer'ları burada, `'s` gibi
kısaltmaları da ayıran ve rakam dizilerini üçle sınırlayan bir regex
kullanır; sayıların aritmetiği sessizce zorlaştıran biçimde
tokenize olmasının nedeni budur.

## 4. Token ID yalnızca bir satır numarasıdır

Vocabulary oluştuktan sonra her girdi bir tamsayı alır. Bu tamsayılar
modelin metninize dair bütün görüşüdür — ve keyfîdirler. Bu bölümü
yazarken `tiktoken` ile ölçtüm:

```python
import tiktoken

gpt4  = tiktoken.get_encoding("cl100k_base")   # GPT-4
gpt4o = tiktoken.get_encoding("o200k_base")    # GPT-4o

gpt4.encode("hello")    # [15339]
gpt4o.encode("hello")   # [24912]
```

Aynı kelime, aynı şirket, farklı tokenizer, birbiriyle ilgisiz
sayılar. ID bir tablodaki satır numarasıdır, fazlası değil. 24912,
15339'dan "daha büyük" ya da "daha sonra" ya da "daha olumlu" değildir
ve model onun üzerinde hiç aritmetik yapmaz.

İki hassasiyet insanları şaşırtır ve ikisi de doğrudan vocabulary'nin
*tam dizgileri* saklamasından gelir:

```python
gpt4.encode("hello")   # [15339]          tek token
gpt4.encode("Hello")   # [9906]           tek token, ilgisiz ID
gpt4.encode("HELLO")   # [51812, 1623]    iki token: "HEL" + "LO"

gpt4.encode(" egg")    # [19151]          tek token — boşluk dâhil
gpt4.encode("egg")     # [29468]          tamamen başka bir token
```

Büyük harf dizgiyi değiştirir, dolayısıyla token'ı da değiştirir.
Bağırmak fazladan tutar, çünkü `HELLO` tek bir satır harcanacak kadar
sık değildir. Baştaki boşluk da *token'ın parçasıdır* — `" egg"` ve
`"egg"` farklı satırlardır. Byte-level BPE o boşluğu `Ġ`, SentencePiece
`▁` diye yazar; ikisi de vocabulary dosyasında görünmez kalacak bir
karakterin görünür vekilidir.

Prompt'unuzun sonundaki bir boşluğun modelin çıktısını değiştirebilmesinin
mekanik nedeni budur. Siz boşluk eklemediniz; başka bir satır kümesi
seçtiniz.

## 5. Vocabulary boyutu ne kazandırır ne götürür

Vocabulary boyutu (`V`), eğitimden önce seçtiğiniz ve sonradan asla
değiştiremeyeceğiniz birkaç mimari sayıdan biridir. İki maliyeti
birbirine karşı takas eder.

```text
Küçük V (32.000)                     Büyük V (256.000)
├─ küçük embedding matrisi           ├─ büyük embedding + çıktı matrisi
├─ ucuz çıktı softmax'ı              ├─ pahalı çıktı softmax'ı
└─ DAHA UZUN diziler                 └─ DAHA KISA diziler
   (cümle başına daha çok token)        (context'e daha çok metin)
```

Embedding matrisinin `V × d` parametresi vardır ve sonraki token'ı
tahmin eden çıktı katmanı da aynı şekildedir. Gizli boyut 4.096 iken
vocabulary'yi 32 binden 256 bine çıkarmak matris başına kabaca 918
milyon parametre ekler — tek bir attention katmanı çalışmadan önce
harcanan gerçek bellek.

Karşılığında aldığınız şey daha kısa dizilerdir. Attention maliyeti
dizi uzunluğunun karesiyle büyüdüğü ve context pencereniz token
cinsinden ölçüldüğü için, cümle başına daha az token daha çok metnin
sığması ve her ileri geçişin ucuzlaması demektir. Sektör bu yüzden
istikrarlı biçimde daha büyük vocabulary'lere doğru yürüdü:

| Kuşak | Vocabulary | Not |
| :--- | :--- | :--- |
| Llama 1 / 2, Mistral 7B | 32.000 | SentencePiece; İngilizce dışı fena parçalanır |
| GPT-2 | 50.257 | 256 bayt + 50.000 birleşme + `<\|endoftext\|>` |
| GPT-4 (`cl100k_base`) | 100.277 | `tiktoken` ile ölçüldü |
| GPT-4o (`o200k_base`) | 200.019 | `tiktoken` ile ölçüldü |
| Gemma | 256.000 | SentencePiece, bayt düzeyinde yedek |

Bu ilerlemenin biçimine dikkat edin: neredeyse tamamen çok dilli metin
ve kod tarafından sürükleniyor — yani küçük vocabulary'nin en çok
canını yaktığı yer. Bu da bizi faturaya getiriyor.

## 6. Türkçe aynı masada neden daha fazla ödüyor

Türkçe **eklemeli (agglutinative)** bir dildir — kelime, bir köke ek
üstüne ek yığarak kurulur, dolayısıyla tek bir Türkçe kelime çoğu
zaman İngilizcenin bir öbeğe yaydığını taşır. Ağırlıklı olarak
İngilizce üzerinde eğitilmiş bir tokenizer, birleşme bütçesini
İngilizce dizgilere harcamıştır; aynı anlam bu yüzden daha çok
token'a mal olur.

> **Fertility (bereket oranı)** = bir tokenizer'ın kelime başına
> harcadığı ortalama token sayısı. Yüksek fertility aynı cümlenin daha
> çok paraya, daha çok gecikmeye ve context pencerenizden daha çok
> yere mal olması demektir.

Bunu folkloru tekrarlamak yerine ölçtüm. Aynı cümle çifti, iki GPT
tokenizer'ından geçirildiğinde:

```text
EN: "Hello world, the weather is very nice today."
TR: "Merhaba dünya, bugün hava çok güzel."

cl100k_base (GPT-4):   EN 10 token  →  TR 14 token   (1,4×)
o200k_base  (GPT-4o):  EN 10 token  →  TR  9 token   (0,9×)
```

Daha uzun bir ikinci çift aynı yönü gösteriyor: `cl100k_base` altında
13 EN'e karşı 23 TR token, `o200k_base` altında ise 13'e karşı 18.
GPT-4 ile GPT-4o arasındaki vocabulary genişlemesi Türkçe için gerçek
bir iş yapmış — ceza ortadan kalkmadı ama kabaca yarıya indi.

Cezanın *neden* var olduğu konusunda net olmakta fayda var, çünkü
yaygın açıklama yanlış. BPE Türkçe morfemleri bulamıyor değil.
`cl100k_base`in *evlerimizden* kelimesine gerçekte ne yaptığına bakın:

```text
cl100k_base:  ev | ler | im | iz | den
o200k_base:   ev | ler | imiz | den

Gerçek morfemler: ev + ler (çoğul) + imiz (iyelik) + den (ayrılma)
```

Bu, hiçbir gramer bilgisi olmadan, saf frekans saymayla bulunmuş
neredeyse kusursuz bir morfolojik bölünme. Sorun BPE'nin Türkçe yapıyı
bulamaması değil; harcayacak yalnızca 32 bin ya da 100 bin satırı
olunca bunların çoğunun İngilizceye gitmesi ve Türkçenin İngilizceye
kıyasla *daha çok, daha küçük* ama doğru parçalara bölünmesi. Bu bir
yetkinlik sorunu değil, bütçe sorunu.

Pratik sonuçlar somut. Bir API çağrısını token başına fiyatlıyorsanız,
Türkçe metin birim anlam başına daha pahalıdır. Bir context penceresi
boyutlandırıyorsanız, içine daha az Türkçe sığar. Türkçe bir iş yükü
için modeller arasında seçim yapıyorsanız, tokenizer fertility'si
gerçek bir seçim ölçütüdür — karar vermeden önce gecikmeyi nasıl
ölçüyorsanız onu da kendi metninizde `tiktoken` ile ölçün.

## 7. Masanın kırdıkları

Modellerin dört meşhur arızası akıl yürütme başarısızlığı değil,
tokenizer artefaktıdır. Hangisinin hangisi olduğunu bilmek, prompt ile
düzeltilip düzeltilemeyeceğini söyler.

**Harf sayamamak.** Model `["str","aw","berry"]` görür, üç satır
numarası. Harfler girdide yoktur; dolayısıyla "kaç tane r var" sorusu,
modelin sahip olmadığı bir veri hakkındadır. Benzer soruların
istatistiksel hafızasından cevaplar, güvenilmez olmasının nedeni
budur. *Çözüm:* karakterleri görünür kılın — önce `s-t-r-a-w-b-e-r-r-y`
isteyin ya da işi koda verin. Ne kadar "dikkatlice düşün" derseniz
deyin, masada atılan bilgi geri gelmez.

**Uzun sayılarla aritmetik.** GPT ailesinin ön bölücüleri rakam
dizilerini üçle sınırlar, yani bir sayı basamak değeriyle hiç ilgisi
olmayan yerlerden bölünür. Hizası kaymış parçalar üzerinde sütun
aritmetiği gerçekten zordur. *Çözüm:* daha iyi bir prompt değil, bir
hesap makinesi aracı.

**Glitch token'lar.** 2023'te Jessica Rumbelow ve Matthew Watkins,
GPT-2/GPT-3 vocabulary'sinde modellere tuhaf çıktı ürettiren ya da
tekrarlamayı reddettiren token'lar buldu — `SolidGoldMagikarp`,
`TheNitromeFan`. Sebep iki veri kümesi arasındaki uyumsuzluk: bu
dizgiler tokenizer'ın Reddit kaynaklı eğitim verisinde bir satırı hak
edecek kadar sıktı ama modelin ön eğitim derleminden ayıklanmışlardı.
Dolayısıyla o embedding satırları ilklendirildi ve neredeyse hiç
güncellenmedi. Böyle bir token'ı prompt'lamak, rastgeleye yakın
sayılardan oluşan bir satırı okumaktır. *Çözüm:* çıkarım anında yok;
bu bir veri hattı hijyeni sorunu.

**Tokenizer kurcalaması.** `tokenizer.json`, bir model deposunda
ağırlıkların yanında gelir ve çıkarımda otomatik yüklenir. İmzalı bir
artefakt değil, bir yapılandırma dosyasıdır. HiddenLayer ve NVIDIA'nın
AI Red Team ekibi, *tek bir* dizgi eşlemesini değiştirmenin — mesela
`://` token'ını — modelin ürettiği her URL'yi, bir aracın aldığı her
argümanı, alt sistemlerin çalıştırdığı her komutu sessizce
değiştirdiği saldırıları belgeledi. Hiçbir ağırlığa dokunulmaz,
dolayısıyla ağırlık sağlama toplamları hiçbir şey kanıtlamaz.
*Çözüm:* tokenizer'ı da ağırlıklar gibi sürümleyip sağlama toplamına
bağlayın ve yükleme anında doğrulayın.

| Belirti | Gerçek sebep | Neye yarar |
| :--- | :--- | :--- |
| Kelimedeki harfleri yanlış sayıyor | karakterler modele hiç girmedi | harf harf yazdırın ya da kod kullanın |
| Uzun çarpmada hata yapıyor | rakamlar üçerli bölünüp basamak değeri bozuluyor | hesap makinesi aracı |
| Tuhaf bir dizgide saçmalıyor | eğitilmemiş embedding satırı | token'ı girdilerden çıkarın |
| İngilizce dışı iki katına mal oluyor | birleşme bütçesi İngilizceye harcanmış | fertility ölçün; daha büyük vocabulary seçin |
| Çıktı sinsice değişmiş | değiştirilmiş `tokenizer.json` | tokenizer dosyasının sağlamasını alın |

## 8. Damgadan vektöre

Masa ağa bir tamsayı listesi verir. Bu tamsayılar bir matris çarpımına
giremez; 4. bölümdeki nedenle: onlar satır numarasıdır ve satır
numaraları üzerinde aritmetik anlamsızdır. 24912 eksi 15339 bir
nicelik değildir.

Köprü, `V × d` boyutundaki **embedding matrisi** `E`'dir — her
vocabulary girdisi için bir satır, her satır `d` tane eğitilebilir
sayıdan oluşan bir vektör. Tokenizasyonun çıktısı ona indeks olarak
kullanılır:

```text
token ID k  →  E'nin k. satırı  →  d sayıdan oluşan bir vektör
```

Bütün işlem bu: bir arama, bir hesap değil. Ama satırlar *öğrenilir*;
eğitim, benzer davranan token'ların satırlarını birbirine yaklaştırır
ve keyfî tamsayı anlamlı bir uzaydaki konuma dönüşür. O uzay,
[embedding yazısının](post.html?slug=embeddingler-derinlemesine) konusu
ve hikâyeyi tam buradan devralıyor.

Transformer katmanlarına geçmeden önce iki şey olur. Konum bilgisi
eklenir — arama sıraya kördür, yoksa "köpek adamı ısırdı" ile "adam
köpeği ısırdı" aynı satır torbası olurdu — ve sonuç attention'a girer.
Oradan sonrası
[LLM'ler nasıl çalışır](post.html?slug=llm-nasil-calisir) yazısının
hikâyesi.

Tam turu kodda görmekte fayda var, çünkü padding (dolgu) ve attention
mask'i genellikle hattın ısırdığı yerlerdir:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("google/gemma-2-2b", use_fast=True)

batch = tok(
    ["Tokenizasyon modelden önceki masadır.",
     "BPE sık geçen çiftleri birleştirir."],
    padding=True,        # kısa olanı uzun olana eşitle
    truncation=True,
    max_length=16,
    return_tensors="pt",
)

print(batch["input_ids"])       # tamsayı satırlar, doldurulmuş
print(batch["attention_mask"])  # 1 = gerçek token, 0 = dolgu

print(tok.decode(batch["input_ids"][0], skip_special_tokens=True))
```

İçselleştirmeye değer kısım `attention_mask`. Padding yalnızca farklı
uzunluktaki cümlelerden dikdörtgen bir tensör çıkarmak için vardır;
mask o konumları `0` işaretler ki attention onları yok saysın. Mask'i
unutursanız model dolguya dikkat eder.

Son olarak **özel token'lar**. Öğrenilmiş parçaların yanında her
vocabulary yapı için satır ayırır: yukarıdaki dolgu için `[PAD]`, dizi
başlangıcını işaretleyen `<bos>`/`[CLS]`, son ya da sınır için
`<eos>`/`[SEP]`, BERT tarzı eğitimin tahmin ettiği boşluklar için
`[MASK]`. Sıradan satırları işgal eder ve sıradan embedding alırlar —
model ne anlama geldiklerini diğer token'lar gibi öğrenir. Bir sohbet
şablonu mesajınızı rol işaretleriyle sardığında, yazdığı şey bu
token'lardır.

## Bütün hikâye altı satırda

- Sinir ağı metin okuyamaz; bu yüzden tokenizer her prompt'u sabit bir
  listedeki parçalara böler ve her parçayı bir tamsayıyla değiştirir.
- Kelimeleri token yapmak çok fazla satır ister ve görülmemiş girdide
  çöker; karakterler az satır ister ama dizileri fazla uzatır.
  Subword'ler uzlaşmadır: sık metin ucuz, seyrek metin pahalı, hiçbir
  şey imkânsız değil.
- BPE listeyi en sık komşu çifti tekrar tekrar birleştirerek kurar;
  WordPiece bunun yerine en şaşırtıcı çifti birleştirir, Unigram ise
  fazla sayıda adaydan başlayıp budar.
- Token ID'si büyüklüğü olmayan bir satır numarasıdır — ve satır tam
  bir dizgidir, yani `"hello"`, `"Hello"` ve `" hello"` üç ayrı
  satırdır.
- Büyük vocabulary iki matriste parametre harcar ve karşılığında daha
  kısa dizi verir; çok dilli metin ve kod önem kazandıkça
  vocabulary'lerin 32 binden 256 bine çıkmasının nedeni budur.
- Harf sayma arızaları, rakam aritmetiği, glitch token'lar ve tokenizer
  kurcalaması bu masanın artefaktlarıdır — dolayısıyla prompt hiçbirini
  düzeltmez.

## Terimler sözlüğü

Yazının temel kelime dağarcığı, birer satır:

- **tokenizasyon** — metni sabit bir listedeki parçalara bölüp her birini bir tamsayıyla değiştirme işlemi.
- **token** — o listedeki tek bir parça; çoğu zaman bütün bir kelime, çoğu zaman bir kırıntı, bazen tek bir bayt.
- **vocabulary (kelime dağarcığı)** — sabit listenin kendisi; `V` boyutu eğitimden önce seçilir ve bir daha değişmez.
- **token ID** — bir token'ın vocabulary'deki satır numarası; keyfîdir, büyüklüğü ya da sırası yoktur.
- **BPE (byte pair encoding)** — en sık komşu çifti tekrar tekrar birleştirerek vocabulary kurma yöntemi.
- **byte-level BPE** — atomları 256 bayt değeri olan BPE; hiçbir girdi bilinmez kalmaz.
- **WordPiece** — BERT'in varyantı; rastlantıya kıyasla en çok birlikte geçen çifti birleştirir.
- **Unigram** — fazla sayıda adayla başlayıp en az işe yarayanları budayan yöntem.
- **SentencePiece** — BPE ya da Unigram'ı ham metin üzerinde çalıştıran, boşluğu karakter sayan (`▁`) kütüphane.
- **pre-tokenizer (ön bölücü)** — birleşmelerin kelime sınırını aşmasını engelleyen, genelde boşluk ve noktalamadaki ilk bölme.
- **OOV / `[UNK]`** — vocabulary'nin temsil edemediği girdi; bayt düzeyinde yedekle ortadan kalkar.
- **fertility (bereket oranı)** — kelime başına harcanan ortalama token; İngilizce dışı metni pahalılaştıran sayı.
- **eklemeli (agglutinative) dil** — Türkçe ya da Fince gibi, köke ek yığarak kelime kuran dil.
- **attention mask** — modele hangi konumların gerçek token hangilerinin dolgu olduğunu söyleyen 1/0 vektörü.
- **embedding matrisi** — `k`. satırı token ID `k` için eğitilebilir vektör olan `V × d` boyutundaki tablo.
- **glitch token** — vocabulary'de satırı olan ama eğitim sinyali neredeyse hiç almamış, embedding'i rastgeleye yakın kalmış token.

## Daha derine inmek için

- Hugging Face, [Tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — bu yazıdaki BPE, WordPiece, Unigram ve SentencePiece yürüyüşlerinin kaynağı.
- Hugging Face, [LLM course, 6. bölüm](https://huggingface.co/learn/llm-course/chapter6/1) — sıfırdan tokenizer eğitimi, adım adım.
- Sennrich vd., [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (2015) — BPE'yi NLP'ye getiren makale.
- Kudo, [Subword Regularization](https://arxiv.org/abs/1804.10959) (2018) — Unigram dil modeli tokenizer'ı.
- Kudo & Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226) (2018) — dilden bağımsız tokenizasyon ve `▁` uzlaşımı.
- OpenAI, [tiktoken](https://github.com/openai/tiktoken) — 4. ve 6. bölümdeki bütün ölçümlerde kullanılan kütüphane; kendi metninizde çalıştırın.
- Rumbelow & Watkins, [SolidGoldMagikarp](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) (2023) — glitch token keşfi ve sebebi.
- HiddenLayer, [Tokenizer tampering](https://www.hiddenlayer.com/research/tokenizer-tampering) ve NVIDIA AI Red Team, [Secure LLM tokenizers](https://developer.nvidia.com/blog/secure-llm-tokenizers-to-maintain-application-integrity/) — `tokenizer.json` üzerindeki saldırı yüzeyi.
- Bu blogda: [Embedding'ler derinlemesine](post.html?slug=embeddingler-derinlemesine) — tamsayıya aramadan sonra ne olduğu — [LLM'ler nasıl çalışır](post.html?slug=llm-nasil-calisir) — o vektörleri tüketen katmanlar — ve [LLM maliyet ve gecikme optimizasyonu](post.html?slug=llm-maliyet-ve-gecikme-optimizasyonu) — faturanızın neden token cinsinden yazıldığı.
