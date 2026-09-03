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
- **Giriş:** başlıksız 2 paragraf, belirti-önce açılış (okurun
  yaşadığı somut bir arıza/paradoksla başla), ikinci paragrafın sonu
  yol haritası cümlesi.
- **TOC:** `**In this article**` / `**Bu yazıda**` altında manuel
  anchor listesi (anchor kuralları bölüm 5'te).
- **Gövde:** numaralı `##` bölümler; kavram önce, örnek sonra; her
  makalede baştan sona taşınan TEK canlı çapa metaforu (müsvedde
  kâğıdı, kütüphane binası, bina/kat gibi); tanımlar
  `> **Terim** = ...` blockquote kalıbıyla; karşılaştırmalar tabloyla;
  bir adet mermaid diyagramı (fenced ```mermaid) ya da CSS
  değişkenli inline SVG.
- **Örnekler:** güçlü, referanslı kaynaklardan birebir alınır (uydurma
  benchmark yok); gerektiğinde altına kısa "How to use it / Nasıl
  kullanılır" notu (nereye yazılır: sistem prompt'u mu kullanıcı
  mesajı mı; kod döngüsü mü prompt mu).
- **Sabit kapanış üçlüsü:**
  `## The whole story in six lines` / `## Bütün hikâye altı satırda`
  (6 madde; düz ve somut dil, mecaz yığını değil) →
  `## Glossary` / `## Terimler sözlüğü` (birer satır, jargon + yalın
  karşılık) →
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
- sitemap.xml: makale çifti başına 2 `<url>` bloğu, üçlü hreflang,
  `x-default`=EN, `priority` 0.7, `lastmod` yayın tarihi.

## 6. Doğrulama komutları

```bash
python3 .github/scripts/validate_site.py            # 0 hata şart
node .claude/skills/makale/scripts/check_toc.js \
  en/blog/posts/<slug>.md tr/blog/posts/<slug>.md   # 0 broken şart
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
