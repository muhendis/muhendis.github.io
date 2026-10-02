---
name: cizim
description: Blog makalelerine tema uyumlu (açık/koyu) inline SVG diyagram çizme akışı — bir mekanizmayı (prefill/decode, KV cache, attention, pipeline, karşılaştırma) EN+TR figür olarak tasarlar, svgkit ile üretir, ekran görüntüsü + otomatik taşma/çakışma kontrolüyle doğrular ve markdown'a gömer. "Şekil/diyagram/görsel çiz", "şu görsellerden faydalanarak ekle", "bunu çizimle anlat" isteklerinde kullan.
---

# Blog Diyagramları: Tema Uyumlu Inline SVG

Bu blogda figürler PNG veya Mermaid değil, **markdown'a gömülü inline SVG**'dir. Renkler `var(--c-*)`
tema token'larından gelir; aynı figür açık ve koyu temada kendiliğinden doğru görünür. Referans
örnek: `how-llms-work` / `llm-nasil-calisir` 7. bölüm (prefill, decode, karşılaştırma) —
kaynağı `examples/prefill_decode.py`.

Makale metni yazılıyorsa önce `makale` skill'i geçerlidir; bu skill onun figür alt akışıdır.

---

## 1. Ne zaman çizilir, ne çizilir

- Figür **mekanizmayı** gösterir: veri neyin içinden, hangi sırayla geçiyor, ne nerede birikiyor,
  ne tekrar ediyor. Süs, logo, ikon kalabalığı yok.
- Kavram **önce metinde** kurulur; figür hemen ardından gelir ve metne yeni bir şey ekler
  (sıra, paralellik, döngü, karşılaştırma). Figürden önceki cümle iki noktayla biter.
- Bir bölüme en fazla 3 figür.
- **Blogda tek çizim biçimi inline SVG'dir.** Yeni Mermaid veya ASCII çizim eklenmez. Mevcut
  olanlar bu skill'le SVG'ye çevrilir (bkz. §3.1). Seri seri gidilir; ilk çevrilen seri
  Prompt'un Yolculuğu'dur.
- **Dış görselleri (X, blog, makale) kopyalama.** İçeriği anla, kendi stilinde yeniden çiz;
  kaynağı `## Going deeper` / `## Daha derine inmek için` listesine "diyagramlara ilham veren"
  notuyla ekle. X/Medium vb. olgu kaynağı sayılmaz (makale skill §1.1).
- X gönderisi/makalesi okumak için: `curl -s https://api.fxtwitter.com/<kullanıcı>/status/<id>`
  → `tweet.article.content.blocks` (metin), `entityMap` (MEDIA caption'ları),
  `media_entities[].media_info.original_img_url` (görseller; `?name=orig` ile indir, Read ile bak).
- **Sahte veri yok:** uydurma token ID'si, uydurma ölçüm, uydurma olasılık sayısı yazma.
  Siluet (etiketsiz çubuklar) kullan ya da sayıyı birincil kaynaktan doğrula.

## 2. Görsel dil (sözleşme)

| Öğe | Kural |
|---|---|
| Tuval | `viewBox="0 0 560 H"`; telefonda ~0,6x küçülür, bu yüzden metin **≥ 11px** (açıklama 12, başlık 13, panel başlığı 15 bold) |
| Renk | Sadece token: `--c-accent` (teal, birincil), `--c-accent-2` (mor = **model üretti / yeni**), `--c-success`, `--c-warn`, `--c-danger`, `--c-text`, `--c-text-mute`, `--c-surface`, `--c-surface-2`, `--c-border`. Hex yazma. |
| Kimlik rengi | Aynı öğe figür boyunca aynı rengi taşır (token → embedding → KV yuvası). Palet: `svgkit.PAL` |
| Chip | Renkli kontur + `fill-opacity:.16` + `--c-text` yazı. Dolu renk üstüne beyaz yazı yok (iki temada kontrast bozulur) |
| Kutu | `BOX` (surface-2 + border); iç vurgu kutusu `inner(renk)` (attention = accent, MLP = success) |
| Ok | Normal akış `--c-text-mute`; üretim/ekleme/döngü `GEN` (mor); tekrar eden döngü kesikli |
| Token/kod | `font-family:var(--font-mono)` |
| Erişilebilirlik | `role="img"` + `aria-label`: figürü görmeyen okura her şeyi anlatan tam cümleler, dile göre ayrı |
| Marker id | Sayfa genelinde benzersiz, figür önekli: `pf-arr`, `dc-arr-g`, `cmp-arr` |

## 3. Akış

```text
1. Kaynağı anla (metin + varsa görseller) → figürün tek cümlelik iddiasını yaz
2. Kâğıtta yerleşim: koordinat ızgarası (16px kenar, 560 genişlik), panel/kutu/ok listesi
3. Üreteci yaz: scratchpad/<konu>_figs.py
     - svgkit'i import et; çizim fonksiyonları dil bağımsız
     - TÜM metin EN/TR sözlüklerinde (L["..."]) — TR etiketler genelde %15-25 daha uzun
     - çıktı: {"en": {"ad": svg}, "tr": {...}} JSON; assert_no_blank_lines
4. Önizle + otomatik kontrol:
     python3 .claude/skills/cizim/scripts/preview.py figs.json --out <scratchpad>/cizim
   → "0 sorun" olana kadar düzelt; shot_<lang>_<theme>.png dosyalarına Read ile GÖZLE de bak
     (döndürülmüş metin, ok kesişmeleri, denge — script bunları yakalamaz)
5. Markdown'a göm (Python ile tam string ekleme; her çapa count==1 assert)
     - figürden önce boş satır, sonra boş satır; SVG içinde boş satır YOK
     - EN ve TR'ye aynı noktaya, kısa açıklayıcı paragrafla
6. Gerçek sayfa kontrolü:
     python3 .claude/skills/cizim/scripts/preview.py --page en/<slug> --page tr/<slug>
   → figür sayısı doğru, bozuk=0
7. makale doğrulaması: ./.agents/skills/makale/scripts/validate_post.sh <en-slug>
8. Commit/push YOK — Engin "push" demeden.
```

### 3.1. Mermaid ve ASCII çizimleri SVG'ye çevirmek

- **Ne çevrilir:** her ` ```mermaid ` bloğu ve kutu/ok/ağaç karakterleriyle (─ │ ┌ ▼ → ──> ●)
  çizilmiş ` ```text ` ya da etiketsiz bloklar. **Ne çevrilmez:** sayısal yürüyüşler, matrisler,
  kod çıktısı, 2–3 satırlık düz karşılaştırmalar; bunlar veri, çizim değil.
- **İçerik korunur, biçim değişir.** Kutudaki metinler ve oklar aynı bilgiyi taşır. Mermaid'deki
  emoji ve `<br>` gibi süsler atılır. Bilgi eklemek serbest, ama çıkarmak değil.
- **Kopya figürleri sil.** Yeni bir SVG aynı şeyi zaten anlatıyorsa eski ASCII/Mermaid bloğunu
  kaldır, iki kez anlatma.
- **Yerinde değiştir:** blok yalnızca SVG ile değiştirilir; hemen önceki tanıtım cümlesi
  okumaya devam ediyorsa olduğu gibi kalır.
- Akış diyagramları için `svgkit.node()` + `svgkit.link()` kullan; düzeni elle ver (Mermaid'in
  otomatik düzenini taklit etme, mekanizmanın okuma yönünü seç: soldan sağa ya da yukarıdan aşağı).
- Değiştirmeden önce bloğun EN ve TR'deki karşılığını eşleştir (sıra ve sayı aynı olmalı); TR
  metinleri TR bloğundan alınır, çeviri yapılmaz.

Sonradan değişiklik: üreteci düzenle → JSON'u yeniden üret → makaledeki **eski SVG string'ini
yenisiyle** değiştir (eski JSON'u sakla ki tam eşleşme yapılabilsin).

## 4. Tuzaklar (yaşanmış)

- **Figüre giren her sayıyı yeniden hesapla (numpy).** Makaledeki elle yürüyüşü kopyalama.
  Prompt'un Yolculuğu Part 4'te `v_2` yanlıştı ve hata Part 7'ye kadar taşınmıştı; figür
  üreteci sayıları yeniden hesaplayınca ortaya çıktı. Çelişki bulursan metni de düzelt ve
  `grep -rn` ile serinin diğer bölümlerini tara.
- **Mevcut mermaid'i okurken geçerliliğine bak.** Birleştirme artığı (aynı `subgraph` id'si
  iki kez, kapanmamış blok) sessizce render olmaz; böyle bir bloğu SVG ile değiştir.

- **marked HTML bloğu boş satırda biter.** SVG içinde tek bir boş satır figürü yarıdan keser,
  geri kalanı `<p>` içinde kaçışlı metin olarak basılır. `--page` kontrolü bunu `bozuk>0` diye yakalar.
- **TR taşmaları:** `KV cache · katman 1`, `görünür` gibi etiketler EN'de sığıp TR'de taşar.
  Kısalt (`KV · katman 1`) ya da chip içi fontu 12'ye indir; kutuyu genişletmek son çare.
- **Çakışan etiketler:** üst kenardaki iki etiket (ör. "her adımda tek yeni token" ile döngü
  etiketi) gözle "değiyor" gibi görünür; checker bunu yakalar — raporu ciddiye al.
- **Döngü okları** diğer okları kesmesin: döngüyü boş kenardan (sağ şerit x≈553 ya da panelin
  boş sol tarafı) dolaştır.
- Yerel sunucuyu `pkill -f http.server` ile öldürme — komut satırı eşleşip kendi kabuğunu da
  öldürür (exit 144). `preview.py --page` sunucuyu kendisi açıp kapatır.
- Chrome: playwright cache'indeki chromium (`~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome`)
  kullanılır; Python `playwright` modülü kurulu değil, gerek de yok.
- `$` içeren etiket yazma (KaTeX blog genelinde `$`'ı yakalar); matematik figürde değil metinde.

## 5. Dosyalar

- `scripts/svgkit.py` — `svg_open`, `marker`, `box`, `text`, `chip`, `arrow`, `path`, `node` (başlık + alt satırlı akış kutusu), `link` (düz/kırık/eğri ok + etiket),
  `bars_v`, `bars_h`, `text_w` (genişlik tahmini), stil sabitleri (`PAL`, `GEN`, `MUTE`, `TXT`, `BOX`, `inner`).
- `scripts/preview.py` — açık/koyu × EN/TR ekran görüntüsü; getBBox ile viewBox taşması,
  kutu taşması ve metin çakışması raporu; `--page` ile gerçek post sayfasında render kontrolü.
- `examples/prefill_decode.py` — üç figürlük tam örnek (pipeline, döngü, yan yana karşılaştırma).
- `examples/flow_conversions.py` — Mermaid/ASCII'den çevrilmiş dokuz figür: iki panelli karşılaştırma,
  kartlar, fan-out/fan-in, iki satırlı yılan akış, iki sütunlu tensör-şekli akışı, atlama yaylı blok akışı.
