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

Akla ilk gelen seçenek *bütün kelimeleri* listelemektir. İki açıdan
çöker. Tek başına İngilizce bile çekimleri, özel isimleri, kod
tanımlayıcılarını ve yazım hatalarını sayınca yüz binlerce forma çıkar
ve her biri embedding matrisinde kendi satırını ister. Daha kötüsü, liste
eğitim anında kapanır: kullanıcılarınızın yazdığı listede bulunmayan
ilk kelime — yeni bir ürün adı ya da bir yazım hatası — tek bir `[UNK]`
token'ına düşer ve anlamı tamamen kaybolur.

Karşı uçtaki seçenek ise *bütün karakterleri* listelemektir. Vocabulary
birkaç yüze iner ve hiçbir şey bilinmez kalmaz. Ama bu kez "tokenizasyon"
iki yerine on iki yuva harcar ve attention maliyeti kabaca dizi
uzunluğunun karesiyle büyür. Tek başına neredeyse hiç anlam taşımayan
parçalar için karesel bir bedel ödersiniz.

Alt kelime (subword) tokenizasyonu ise orta yolu tutar. Sık kelimeler
bütün kalır; seyrek olanlar ise yine anlamlı parçalara ayrılır.

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

Aynı iki birleşme, resim olarak. Her sütun, bir birleşme daha uygulandıktan sonraki derlemdir; o adımda doğan token mor, önceki birleşmeler turkuaz:

<svg viewBox="0 0 560 268" role="img" aria-label="Beş kelimelik derlem üzerinde BPE, üç anlık görüntü olarak. Solda her kelime karakterlerine ayrılmış: hug 10 kez, pug 5, pun 12, bun 4, hugs 5; temel vocabulary 7 sembol. Ortada 1. birleşme u ile g&#x27;yi ug yapar; hug, pug ve hugs içinde 20 kez geçer. Sağda 2. birleşme u ile n&#x27;yi un yapar; pun ve bun içinde 16 kez. Vocabulary artık 7 karakter artı ug artı un." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="16" width="176" height="196" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="28" y="36" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">karakterler</text>
<text x="28" y="53" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">temel vocabulary: 7 sembol</text>
<rect x="28" y="66" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="39.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="54" y="66" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="65.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="80" y="66" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="91.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g</text>
<text x="180" y="81" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×10</text>
<rect x="28" y="94" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="39.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="54" y="94" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="65.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="80" y="94" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="91.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g</text>
<text x="180" y="109" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×5</text>
<rect x="28" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="39.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="54" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="65.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="80" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="91.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">n</text>
<text x="180" y="137" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×12</text>
<rect x="28" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="39.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">b</text>
<rect x="54" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="65.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="80" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="91.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">n</text>
<text x="180" y="165" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×4</text>
<rect x="28" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="39.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="54" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="65.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="80" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="91.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g</text>
<rect x="106" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="117.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">s</text>
<text x="180" y="193" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×5</text>
<rect x="200" y="16" width="176" height="196" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="212" y="36" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">birleşme 1: u + g → ug</text>
<text x="212" y="53" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">20 kez, kazanır</text>
<rect x="212" y="66" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="223.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="238" y="66" width="34" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="255.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<text x="364" y="81" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×10</text>
<rect x="212" y="94" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="223.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="238" y="94" width="34" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="255.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<text x="364" y="109" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×5</text>
<rect x="212" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="223.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="238" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="249.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="264" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="275.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">n</text>
<text x="364" y="137" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×12</text>
<rect x="212" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="223.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">b</text>
<rect x="238" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="249.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="264" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="275.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">n</text>
<text x="364" y="165" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×4</text>
<rect x="212" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="223.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="238" y="178" width="34" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="255.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<rect x="276" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="287.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">s</text>
<text x="364" y="193" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×5</text>
<rect x="384" y="16" width="176" height="196" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="396" y="36" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">birleşme 2: u + n → un</text>
<text x="396" y="53" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">16 kez</text>
<rect x="396" y="66" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="407.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="422" y="66" width="34" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="439.0" y="81.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<text x="548" y="81" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×10</text>
<rect x="396" y="94" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="407.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="422" y="94" width="34" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="439.0" y="109.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<text x="548" y="109" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×5</text>
<rect x="396" y="122" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="407.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="422" y="122" width="34" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="439.0" y="137.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">un</text>
<text x="548" y="137" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×12</text>
<rect x="396" y="150" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="407.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">b</text>
<rect x="422" y="150" width="34" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="439.0" y="165.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">un</text>
<text x="548" y="165" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×4</text>
<rect x="396" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="407.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="422" y="178" width="34" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="439.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<rect x="460" y="178" width="22" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="471.0" y="193.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">s</text>
<text x="548" y="193" text-anchor="end" style="fill:var(--c-text-mute);font-size:12px">×5</text>
<text x="16" y="236" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">vocabulary:</text>
<rect x="96" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="106.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">b</text>
<rect x="119" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="129.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">g</text>
<rect x="142" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="152.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">h</text>
<rect x="165" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="175.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">n</text>
<rect x="188" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="198.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">p</text>
<rect x="211" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="221.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">s</text>
<rect x="234" y="222" width="20" height="22" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="244.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">u</text>
<rect x="261" y="222" width="32" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="277.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">ug</text>
<rect x="297" y="222" width="32" height="22" rx="5" style="fill:var(--c-accent-2);fill-opacity:.16;stroke:var(--c-accent-2);stroke-width:1.3"/><text x="313.0" y="237.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">un</text>
<text x="16" y="262" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Yeni metni kodlayan şey nihai liste değil, sıralı birleşme listesinin kendisidir.</text>
</svg>

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

Gözden kaçan iki ayrıntı kafa karıştırabilir; ikisi de doğrudan
vocabulary'nin *tam karakter dizilerini* (exact strings) saklamasından
kaynaklanır:

```python
gpt4.encode("hello")   # [15339]          tek token
gpt4.encode("Hello")   # [9906]           tek token, ilgisiz ID
gpt4.encode("HELLO")   # [51812, 1623]    iki token: "HEL" + "LO"

gpt4.encode(" egg")    # [19151]          tek token — boşluk dâhil
gpt4.encode("egg")     # [29468]          tamamen başka bir token
```

Büyük harf karakter dizisini değiştirir, dolayısıyla seçilen token da
değişir. Büyük harfle yazmak fazladan ücrete mal olur; çünkü `HELLO`
tek bir satır ayrılacak kadar sık geçmez. Baştaki boşluk da *token'ın
bir parçasıdır* — `" egg"` ve `"egg"` iki farklı satırdır. Byte-level
BPE o boşluğu `Ġ`, SentencePiece ise `▁` sembolüyle gösterir; ikisi de
vocabulary dosyasında görünmez kalacak bir karakterin görünür
temsilcisidir.

Prompt'unuzun sonundaki tek bir boşluğun modelin çıktısını kökten
değiştirebilmesinin mekanik nedeni budur. Siz sadece boşluk eklemediniz;
tamamen başka bir satır kümesi seçtiniz.

## 5. Vocabulary boyutu ne kazandırır ne götürür

Vocabulary boyutu (`V`), eğitimden önce seçtiğiniz ve sonradan asla
değiştiremeyeceğiniz birkaç mimari sayıdan biridir. İki maliyeti
birbirine karşı takas eder.

<svg viewBox="0 0 560 222" role="img" aria-label="Vocabulary boyutunun takası. Küçük vocabulary, 32.000 girdi: 4.096 gizli boyutta embedding ve çıktı matrislerinin her biri 131M parametre tutar ve çıktı softmax&#x27;ı ucuzdur, ama cümleler daha çok token&#x27;a bölünür. Büyük vocabulary, 256.000 girdi: her matris 1,05B parametre tutar ve softmax pahalıdır, ama cümleler daha az token tutar, bağlam penceresine daha çok metin sığar." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<rect x="16" y="8" width="256" height="168" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="30" y="32" text-anchor="start" style="fill:var(--c-accent);font-size:15px;font-weight:700;letter-spacing:.06em">Küçük V (32.000)</text>
<text x="30" y="56" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">embedding ve çıktı matrisi (d = 4.096)</text>
<rect x="30" y="64" width="26" height="16" rx="3" style="fill:var(--c-accent);fill-opacity:.6"/>
<text x="64" y="76" text-anchor="start" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">131M</text>
<text x="30" y="100" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">V üzerinde ucuz çıktı softmax&#x27;ı</text>
<text x="30" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">aynı cümle, token olarak (şematik)</text>
<rect x="30" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="55" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="80" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="105" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="130" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="155" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="180" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="205" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="30" y="170" text-anchor="start" style="fill:var(--c-accent);font-size:12px;font-weight:600">DAHA UZUN diziler</text>
<rect x="288" y="8" width="256" height="168" rx="8" style="fill:none;stroke:var(--c-border);stroke-width:1.2"/>
<text x="302" y="32" text-anchor="start" style="fill:var(--c-accent-2);font-size:15px;font-weight:700;letter-spacing:.06em">Büyük V (256.000)</text>
<text x="302" y="56" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">embedding ve çıktı matrisi (d = 4.096)</text>
<rect x="302" y="64" width="208" height="16" rx="3" style="fill:var(--c-accent-2);fill-opacity:.6"/>
<text x="504" y="76" text-anchor="end" style="fill:var(--c-text);font-size:11.5px;font-family:var(--font-mono)">matris başına 1,05B</text>
<text x="302" y="100" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">V üzerinde pahalı çıktı softmax&#x27;ı</text>
<text x="302" y="128" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">aynı cümle, token olarak (şematik)</text>
<rect x="302" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="327" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="352" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="377" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<rect x="402" y="136" width="22" height="18" rx="3" style="fill:var(--c-text-mute);fill-opacity:.25;stroke:var(--c-text-mute);stroke-width:1"/>
<text x="302" y="170" text-anchor="start" style="fill:var(--c-accent-2);font-size:12px;font-weight:600">DAHA KISA diziler</text>
<text x="16" y="196" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her matris 8 kat büyür (+918M parametre); karşılığında her cümle daha az token tutar.</text>
<text x="16" y="210" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Matris boyutları d = 4.096 için kesin; token sayıları şematik.</text>
</svg>

Embedding matrisinin `V × d` parametresi vardır ve sonraki token'ı
tahmin eden çıktı katmanı da aynı şekildedir. Gizli boyut 4.096 iken
vocabulary'yi 32 binden 256 bine çıkarmak matris başına kabaca 918
milyon parametre ekler — tek bir attention katmanı çalışmadan önce
harcanan gerçek bellek.

Karşılığında aldığınız şey daha kısa dizilerdir. Attention maliyeti
dizi uzunluğunun karesiyle büyüdüğü ve bağlam pencereniz (context
window) token cinsinden ölçüldüğü için, cümle başına daha az token
harcamak, aynı pencereye daha çok metin sığması ve her ileri geçişin
ucuzlaması demektir. Sektör bu yüzden istikrarlı biçimde daha büyük
vocabulary'lere doğru yürüdü:

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
> harcadığı ortalama token sayısı. Yüksek fertility, aynı cümlenin
> daha yüksek maliyete, daha fazla gecikmeye ve bağlam pencerenizden
> daha çok yer harcanmasına yol açması demektir.

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

Parçaları harf harf hizalayıp gerçek token ID'lerini de yazınca sınırlar üst üste biniyor:

<svg viewBox="0 0 560 262" role="img" aria-label="Türkçe evlerimizden kelimesi iki tokenizer altında, gerçek morfemleriyle harf harf hizalanmış. cl100k_base, yani GPT-4, 5 token harcar: ev, ler, im, iz, den; ID&#x27;leri 5230, 1565, 318, 450, 5294. o200k_base, yani GPT-4o, 4 token harcar: ev, ler, imiz, den; ID&#x27;leri 6923, 1639, 25978, 1660. Morfemler: ev, ler çoğul, imiz iyelik, den ayrılma. Her token sınırı bir morfem sınırına düşer; GPT-4 yalnızca imiz&#x27;i ikiye böler." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<text x="140" y="20" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">evlerimizden — tiktoken ile ölçüldü</text>
<text x="16" y="51" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">cl100k_base</text>
<text x="16" y="67" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GPT-4</text>
<rect x="140" y="34" width="50" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="165.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">ev</text>
<text x="165.0" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">5230</text>
<rect x="194" y="34" width="77" height="28" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="232.5" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">ler</text>
<text x="232.5" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1565</text>
<rect x="275" y="34" width="50" height="28" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="300.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">im</text>
<text x="300.0" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">318</text>
<rect x="329" y="34" width="50" height="28" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="354.0" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">iz</text>
<text x="354.0" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">450</text>
<rect x="383" y="34" width="77" height="28" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/><text x="421.5" y="52.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">den</text>
<text x="421.5" y="78" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">5294</text>
<text x="544" y="53" text-anchor="end" style="fill:var(--c-text);font-size:13px">5 token</text>
<text x="16" y="117" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">o200k_base</text>
<text x="16" y="133" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">GPT-4o</text>
<rect x="140" y="100" width="50" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="165.0" y="118.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">ev</text>
<text x="165.0" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">6923</text>
<rect x="194" y="100" width="77" height="28" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="232.5" y="118.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">ler</text>
<text x="232.5" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1639</text>
<rect x="275" y="100" width="104" height="28" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="327.0" y="118.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">imiz</text>
<text x="327.0" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">25978</text>
<rect x="383" y="100" width="77" height="28" rx="5" style="fill:var(--c-danger);fill-opacity:.16;stroke:var(--c-danger);stroke-width:1.3"/><text x="421.5" y="118.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">den</text>
<text x="421.5" y="144" text-anchor="middle" style="fill:var(--c-text-mute);font-size:11px;font-family:var(--font-mono)">1660</text>
<text x="544" y="119" text-anchor="end" style="fill:var(--c-text);font-size:13px">4 token</text>
<text x="16" y="189" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:600">morfemler</text>
<text x="16" y="205" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">dilbilgisi</text>
<rect x="140" y="172" width="50" height="28" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.3;stroke-dasharray:4 3"/><text x="165.0" y="190.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">ev</text>
<text x="165.0" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">kök</text>
<rect x="194" y="172" width="77" height="28" rx="5" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.3;stroke-dasharray:4 3"/><text x="232.5" y="190.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">ler</text>
<text x="232.5" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">çoğul</text>
<rect x="275" y="172" width="104" height="28" rx="5" style="fill:var(--c-warn);fill-opacity:.06;stroke:var(--c-warn);stroke-width:1.3;stroke-dasharray:4 3"/><text x="327.0" y="190.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">imiz</text>
<text x="327.0" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">iyelik</text>
<rect x="383" y="172" width="77" height="28" rx="5" style="fill:var(--c-danger);fill-opacity:.06;stroke:var(--c-danger);stroke-width:1.3;stroke-dasharray:4 3"/><text x="421.5" y="190.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">den</text>
<text x="421.5" y="216" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">ayrılma</text>
<line x1="192" y1="30" x2="192" y2="204" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:2 3"/>
<line x1="273" y1="30" x2="273" y2="204" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:2 3"/>
<line x1="381" y1="30" x2="381" y2="204" style="stroke:var(--c-border);stroke-width:1;stroke-dasharray:2 3"/>
<text x="16" y="252" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">Her sınır bir morfem kenarına düşer. Fark bütçede: GPT-4 imiz&#x27;i ikiye böler.</text>
</svg>

Bu, hiçbir gramer bilgisi olmadan, saf frekans saymayla bulunmuş
neredeyse kusursuz bir morfolojik bölünme. Sorun BPE'nin Türkçe yapıyı
bulamaması değil; harcayacak yalnızca 32 bin ya da 100 bin satırı
olunca bunların çoğunun İngilizceye gitmesi ve Türkçenin İngilizceye
kıyasla *daha çok, daha küçük* ama doğru parçalara bölünmesi. Bu bir
yetkinlik sorunu değil, bütçe sorunu.

Pratik sonuçlar somut. Bir API çağrısını token başına fiyatlıyorsanız,
Türkçe metin birim anlam başına daha pahalıdır. Bir bağlam penceresi
boyutlandırıyorsanız, içine daha az Türkçe metin sığar. Türkçe bir iş yükü
için modeller arasında seçim yapıyorsanız, tokenizer fertility'si
gerçek bir seçim ölçütüdür — karar vermeden önce gecikmeyi nasıl
ölçüyorsanız, fertility'yi de kendi metninizde `tiktoken` ile mutlaka
ölçün.

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

Somut bir kelimeyle bütün masa, metinden satırlara, gerçek `cl100k_base` ID'leriyle:

<svg viewBox="0 0 560 290" role="img" aria-label="Metinden vektöre. Başında boşluk olmayan strawberry kelimesini cl100k_base üç token&#x27;a böler: str, aw ve berry; ID&#x27;leri 496, 675 ve 15717. Her ID, 100.277 vocabulary girdisinin her biri için bir satırı olan embedding matrisi E&#x27;nin bir satırını seçer; o satır token&#x27;ın d tane öğrenilmiş sayıdan oluşan vektörüdür. Hesaplama değil, bir lookup." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="dk-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="30" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">metin</text>
<text x="142" y="30" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">token</text>
<text x="234" y="30" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ID</text>
<text x="374" y="30" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">E (V × d)</text>
<text x="436" y="30" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">satır = vektör</text>
<rect x="16" y="122" width="96" height="30" rx="5" style="fill:var(--c-text-mute);fill-opacity:.16;stroke:var(--c-text-mute);stroke-width:1.3"/><text x="64.0" y="141.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">strawberry</text>
<path d="M112 137 C 124 137, 124 83, 140 83" marker-end="url(#dk-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="142" y="70" width="64" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="174.0" y="87.5" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">str</text>
<line x1="206" y1="83" x2="232" y2="83" marker-end="url(#dk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="234" y="70" width="70" height="26" rx="5" style="fill:var(--c-accent);fill-opacity:.06;stroke:var(--c-accent);stroke-width:1.3"/><text x="269.0" y="87.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">496</text>
<line x1="304" y1="83" x2="340" y2="83" marker-end="url(#dk-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<path d="M112 137 C 124 137, 124 148, 140 148" marker-end="url(#dk-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="142" y="135" width="64" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.16;stroke:var(--c-success);stroke-width:1.3"/><text x="174.0" y="152.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">aw</text>
<line x1="206" y1="148" x2="232" y2="148" marker-end="url(#dk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="234" y="135" width="70" height="26" rx="5" style="fill:var(--c-success);fill-opacity:.06;stroke:var(--c-success);stroke-width:1.3"/><text x="269.0" y="152.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">675</text>
<line x1="304" y1="148" x2="340" y2="148" marker-end="url(#dk-arr)" style="stroke:var(--c-success);stroke-width:1.5"/>
<path d="M112 137 C 124 137, 124 213, 140 213" marker-end="url(#dk-arr)" style="fill:none;stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="142" y="200" width="64" height="26" rx="5" style="fill:var(--c-warn);fill-opacity:.16;stroke:var(--c-warn);stroke-width:1.3"/><text x="174.0" y="217.6" text-anchor="middle" style="fill:var(--c-text);font-size:13px;font-family:var(--font-mono)">berry</text>
<line x1="206" y1="213" x2="232" y2="213" marker-end="url(#dk-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="234" y="200" width="70" height="26" rx="5" style="fill:var(--c-warn);fill-opacity:.06;stroke:var(--c-warn);stroke-width:1.3"/><text x="269.0" y="217.2" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-family:var(--font-mono)">15717</text>
<line x1="304" y1="213" x2="340" y2="213" marker-end="url(#dk-arr)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="342" y="46" width="64" height="196" rx="4" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<line x1="348" y1="54" x2="400" y2="54" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="63" x2="400" y2="63" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="72" x2="400" y2="72" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="81" x2="400" y2="81" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="90" x2="400" y2="90" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="99" x2="400" y2="99" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="108" x2="400" y2="108" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="117" x2="400" y2="117" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="126" x2="400" y2="126" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="135" x2="400" y2="135" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="144" x2="400" y2="144" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="153" x2="400" y2="153" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="162" x2="400" y2="162" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="171" x2="400" y2="171" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="180" x2="400" y2="180" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="189" x2="400" y2="189" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="198" x2="400" y2="198" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="207" x2="400" y2="207" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="216" x2="400" y2="216" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="225" x2="400" y2="225" style="stroke:var(--c-border);stroke-width:1"/>
<line x1="348" y1="234" x2="400" y2="234" style="stroke:var(--c-border);stroke-width:1"/>
<text x="374" y="257" text-anchor="middle" style="fill:var(--c-text-mute);font-size:12px">100.277 satır</text>
<rect x="344" y="78" width="60" height="10" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<line x1="406" y1="83" x2="432" y2="83" marker-end="url(#dk-arr)" style="stroke:var(--c-accent);stroke-width:1.5"/>
<rect x="436" y="73" width="15" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.65"/>
<rect x="454" y="73" width="15" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.45"/>
<rect x="472" y="73" width="15" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.75"/>
<rect x="490" y="73" width="15" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.35"/>
<rect x="508" y="73" width="15" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.55"/>
<rect x="526" y="73" width="15" height="20" rx="2" style="fill:var(--c-accent);fill-opacity:.2"/>
<rect x="344" y="143" width="60" height="10" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<line x1="406" y1="148" x2="432" y2="148" marker-end="url(#dk-arr)" style="stroke:var(--c-success);stroke-width:1.5"/>
<rect x="436" y="138" width="15" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.2"/>
<rect x="454" y="138" width="15" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.65"/>
<rect x="472" y="138" width="15" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.45"/>
<rect x="490" y="138" width="15" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.75"/>
<rect x="508" y="138" width="15" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.35"/>
<rect x="526" y="138" width="15" height="20" rx="2" style="fill:var(--c-success);fill-opacity:.55"/>
<rect x="344" y="208" width="60" height="10" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<line x1="406" y1="213" x2="432" y2="213" marker-end="url(#dk-arr)" style="stroke:var(--c-warn);stroke-width:1.5"/>
<rect x="436" y="203" width="15" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.2"/>
<rect x="454" y="203" width="15" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.65"/>
<rect x="472" y="203" width="15" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.45"/>
<rect x="490" y="203" width="15" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.75"/>
<rect x="508" y="203" width="15" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.35"/>
<rect x="526" y="203" width="15" height="20" rx="2" style="fill:var(--c-warn);fill-opacity:.55"/>
<text x="16" y="280" text-anchor="start" style="fill:var(--c-text-mute);font-size:12px">ID k, k. satırı seçer. Satırlar öğrenilir; ID&#x27;ler yalnızca adrestir (cl100k_base).</text>
</svg>

Bütün işlem bundan ibaret: bir bellek araması (lookup), hesaplama
değil. Ama satırlar *öğrenilir*; eğitim, benzer davranan token'ların
satırlarını birbirine yaklaştırır ve keyfî tamsayı anlamlı bir uzaydaki
konuma dönüşür. O uzay,
[Embedding'ler derinlemesine](post.html?slug=embeddingler-derinlemesine)
yazısının konusu ve hikâyeyi tam buradan devralıyor. Ayrık token'ların
ilk katmandaki ağırlıklar üzerinden nasıl sürekli bir geometriye
dönüştüğünü ise [Embedding katmanı derinlemesine](post.html?slug=embedding-katmani-derinlemesine)
incelemesinde bulabilirsiniz.

Transformer katmanlarına geçmeden önce iki şey olur. Konum bilgisi
eklenir — arama sıraya kördür, yoksa "köpek adamı ısırdı" ile "adam
köpeği ısırdı" aynı satır torbası olurdu — ve sonuç attention'a girer.
Oradan sonrası [LLM'ler nasıl çalışır](post.html?slug=llm-nasil-calisir)
yazısının hikâyesi.

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
- Bu blogda: [Bir Prompt'un Yolculuğu (2): Embedding Katmanı](post.html?slug=embedding-katmani-derinlemesine) — ayrık token'lardan sürekli geometriye —, [Bir Prompt'un Yolculuğu (3): Anlamsal Embedding'ler](post.html?slug=embeddingler-derinlemesine) — tamsayıya aramadan sonra ne olduğu —, [Bir Prompt'un Yolculuğu (4): Self-Attention](post.html?slug=self-attention-derinlemesine) — transformer'ın dikkat motoru ve bağlam vektörleri —, [Büyük Resim (1): Baştan Sona Bir LLM](post.html?slug=llm-nasil-calisir) — o vektörleri tüketen katmanlar — ve [LLM maliyet ve gecikme optimizasyonu](post.html?slug=llm-maliyet-ve-gecikme-optimizasyonu) — faturanızın neden token cinsinden yazıldığı.
