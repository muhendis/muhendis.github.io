Bir salı sabahı bir müşteri destek hattını aradı: Kendisine gönderilmiş, bir iadeyle ilgili ama *başka* bir müşterinin siparişini anlatan bir e-posta almıştı. Olay ekibi logları açtı. O vaka için `send_email called` diyen bir satır buldu. E-postayı hangi model sürümünün hazırladığını, hangi politika sürümünün onayladığını, bağlamda hangi getirilmiş parçaların bulunduğunu ya da "onayla"ya kimin tıkladığını bulamadı. Trace'ler %1 örneklemle tutuluyordu ve bu istek örneklemde değildi. Kaydedilmiş *olan* prompt ve yanıt içeriği bir gözlemlenebilirlik sağlayıcısına aktarılmıştı; bu yüzden soruşturmanın kendisi, diğer müşterinin verisinin artık üçüncü taraf bir sistemde de durduğunu ortaya çıkardı. Bir önceki haftanın trafik zirvesinde de denetim işlem hattı olayları sessizce düşürmüştü; dolayısıyla başka e-postaların yanlış kişilere gidip gitmediğini kimse söyleyemiyordu.

Sorunu kontrol altına almak dokuz gün sürdü; sürenin çoğu olanları yeniden kurmakla geçti. Beşinci ile Dokuzuncu bölümler arasındaki kontrollerin hepsi, tek bir uç durum dışında çalışmış olabilirdi. Ama kayıtlar olmadan ekip bunu kanıtlayamadı, uç durumu bulamadı ve etkilenen müşterilere kim olduklarını söyleyemedi.

Bu yazı destek asistanı için kayıtları ve müdahaleyi tasarlıyor. Trace'leri, güvenlik denetim olaylarını, servis loglarını ve metrikleri birbirinden ayıracak, her kararın sürümleriyle birlikte yeniden kurulabilmesi için neyin kaydedilmesi gerektiğine karar verecek, kayıtların hassas verinin ikinci bir kopyasına dönüşmesini engelleyecek, denetim teslimatını güvenilecek kadar kalıcı hâle getirecek, olayları şansa bağlı olmayan sinyallerle tespit edecek ve NIST'in olay müdahalesi rehberini AI'a özgü kontrol altına alma kaldıraçlarına dönüştüreceğiz. Pratik çıktı bir olay şeması, adli adımları içeren bir Python denetim yürütücüsü ve bir olay müdahale runbook'u.

> **Önceki yazıdan devraldığımız fikir:** [Onuncu bölüm](post.html?slug=ai-guvenligini-olcmek) sistemi yayından önce ölçtü. Bu bölüm onu üretimde ölçüyor ve bir ölçümün olaya dönüştüğü güne hazırlanıyor. [Dördüncü bölümün](post.html?slug=ai-tedarik-zinciri-guvenligi) sürüm disiplinine, [Beşinci bölümün](post.html?slug=prompt-injection-guven-siniri) kaynak etiketlerine ve [Sekizinci bölümün](post.html?slug=ajanlarda-en-az-yetki) yürütücüsüne dayanıyor.

> **Bu yazının sonunda**
>
> - Trace'lerin, denetim olaylarının, servis loglarının ve metriklerin farklı işlerini ve neden yalnızca bazılarının örneklenebileceğini açıklayabileceksiniz.
> - Her AI kararını modeline, prompt'una, politikasına, kaynaklarına ve onayına bağlayan bir olay şeması tanımlayabileceksiniz.
> - Gözlemlenebilirliğin yeni bir sızıntı yaratmaması için loglananları tuzlanmış HMAC özetleriyle maskeleyebileceksiniz.
> - Güvenli biçimde başarısız olan bir denetim teslimatı tasarlayabilecek ve sonucu bilinmeyen işlemleri ele alabileceksiniz.
> - AI'a özgü 4 katmanlı kontrol altına alma kaldıraçlarını ve adli yeniden oynatma (deterministic replay) motorunu çalıştırabileceksiniz.

**Bu yazıda**

- [1. Dört tür kayıt](#1-dört-tür-kayıt)
- [2. Bir kararı yeniden kurmak: Ne kaydedilmeli?](#2-bir-kararı-yeniden-kurmak-ne-kaydedilmeli)
- [3. İkinci bir sızıntı olmadan gözlemlenebilirlik](#3-ikinci-bir-sızıntı-olmadan-gözlemlenebilirlik)
- [4. Kalıcı teslimat ve örneklemenin matematiği](#4-kalıcı-teslimat-ve-örneklemenin-matematiği)
  - [4.1. Denetim olayları neden örneklenmemeli?](#41-denetim-olayları-neden-örneklenmemeli)
  - [4.2. Güvenli biçimde başarısız olan teslimat](#42-güvenli-biçimde-başarısız-olan-teslimat)
  - [4.3. Sonucu bilinmeyen işlemler](#43-sonucu-bilinmeyen-işlemler)
- [5. Olayları tespit etmek](#5-olayları-tespit-etmek)
- [6. Müdahale: Kontrol altına al, geri al, toparlan, öğren](#6-müdahale-kontrol-altına-al-geri-al-toparlan-öğren)
- [7. Pratik çıktı: Olay şeması, denetim motoru ve runbook](#7-pratik-çıktı-olay-şeması-denetim-motoru-ve-runbook)
  - [7.1. Güvenlik denetim olayı şeması (örnek)](#71-güvenlik-denetim-olayı-şeması-örnek)
  - [7.2. Adli yeniden oynatma ve giden kutusu denetim motoru](#72-adli-yeniden-oynatma-ve-giden-kutusu-denetim-motoru)
  - [7.3. AI güvenlik olayı müdahale runbook'u (kesit)](#73-ai-güvenlik-olayı-müdahale-runbooku-kesit)
  - [7.4. Mühendisliğin iki gözü: Kurulum ve çalışma zamanı](#74-mühendisliğin-iki-gözü-kurulum-ve-çalışma-zamanı)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Dört tür kayıt

Bir uçak iki farklı kayıt cihazı taşır. Uçuş veri kaydedicisi yüzlerce parametreyi sürekli kaydeder ve bir kazadan sağ çıkmak zorundadır; bakım defteri ise mürettebatın ve mühendislerin yazmayı seçtiği şeyleri kaydeder. Kimse bir kazayı yalnızca bakım defterinden incelemez; kimse de uçuş kaydedicisini kahve makinesi arızalarıyla doldurmaz.

Bir AI sistemi de aynı ayrıma ihtiyaç duyar. Dört tür kayıt çoğu zaman "log" adı altında bir araya toplanır; onlara aynı muameleyi yapmak giriş olayının yaşanma biçimidir:

| Kayıt | Amaç | Eksiksizlik | İçerik | Saklama | Okuyanlar |
| --- | --- | --- | --- | --- | --- |
| **Trace'ler** | Servisler arasında performans ve davranış hatalarını ayıklamak | Örnekleme kabul edilebilir | Zamanlama, öznitelikler; içerik yalnızca isteğe bağlı | Günlerden haftalara | Mühendisler |
| **Güvenlik denetim olayları** | Kararları yeniden kurmak; kanıt | **Eksiksiz, asla örneklenmez** | Kimlikler, sürümler, hash'ler, kararlar; ham içerik yok | Politikaya göre, çoğu zaman bir yıl ya da daha uzun | Güvenlik, denetim, olay ekibi |
| **Servis logları** | Servisi işletmek (hatalar, yeniden başlatmalar, yapılandırma) | En iyi çaba | Teknik mesajlar | Haftalar | Nöbetçi mühendisler |
| **Metrikler** | Eğilimler ve alarm | Toplulaştırılmış | Sayılar, oranlar, gecikmeler | Aylar | Panolar, alarm |

Temel ayrım trace'ler ile denetim olayları arasındadır. Bir trace *"bu istek neden yavaştı?"* sorusunu yanıtlar ve trace'lerin %99'unu kaybetmek sorun değildir. Bir denetim olayı *"bu e-postayı kim, hangi politikayla onayladı?"* sorusunu yanıtlar ve tek birini kaybetmek bir olayın kapsamını belirlemeyi imkânsız kılabilir. Aynı enstrümantasyonu paylaşabilirler, ama farklı işlem hatlarına, depolamaya, erişim kontrolüne ve saklama süresine ihtiyaç duyarlar.

OpenTelemetry'nin üretken yapay zekâ için anlamsal kuralları (semantic conventions), trace'ler için ortak bir dil sunar: `chat`, `invoke_agent`, `execute_tool` ve `retrieval` gibi işlem adları ile `gen_ai.request.model`, `gen_ai.tool.name`, `gen_ai.conversation.id` ve `gen_ai.data_source.id` gibi öznitelikler. Kurallar hâlâ *Development* (geliştirme aşamasında) olarak işaretli; yani adlar değişebilir. Gerçekleştirdiğiniz sürümü sabitleyin.

---

## 2. Bir kararı yeniden kurmak: Ne kaydedilmeli?

Bir denetim tasarımının testi basittir: Sistemin yaptığı herhangi bir işlem için *kimin istediğini, modelin ne gördüğünü, ne önerdiğini, hangi kontrollerin karar verdiğini, kimin onayladığını ve gerçekte ne olduğunu*, işin içindeki her bileşenin sürümüyle birlikte yanıtlayabiliyor musunuz? Her soru alanlara karşılık gelir:

| Soru | Denetim olayındaki alanlar |
| --- | --- |
| Kim istedi? | `user_id` (takma adlı), `session_id`, `task`, istemci kimliği |
| Model ne gördü? | Kaynak listesi: parça kimlikleri, koleksiyonlar, güven düzeyleri $\tau$ (Bölüm 5), anı kimlikleri; kurulan prompt'un **hash'i** |
| Hangi model ve yapılandırma? | `model` ve revizyonu, adaptör, sohbet şablonu, sistem prompt'u sürümü |
| Ne önerdi? | Araç adı, **kanonik parametrelerin hash'i** (Bölüm 8), hedef sınıfı |
| Hangi kontroller karar verdi? | Politika sürümü, dedektör sürümleri ve puanları (bulgular), karar, kural kimliği |
| Kim onayladı? | Onay token'ı kimliği, onaylayan, zaman damgası, gösterilen parametrelerin hash'i |
| Ne oldu? | Çalıştırma sonucu, alt akış istek kimliği, idempotency anahtarı, hata |
| Hangi yoldan? | Gateway yolu, birincil ya da yedek, guard kipi (Bölüm 9) |

Tek bir kullanıcı isteğine ait bütün olaylar aynı `trace_id`'yi taşır ve her alt akış çağrısı (model, guard, araç, MCP sunucusu) onu iletir. Bu da giriş hikâyesindeki soruyu, *"bu politika sürümü geçen hafta hangi e-postaları onayladı?"*, bir soruşturma olmaktan çıkarıp bir sorguya dönüştürür.

İki tasarım ayrıntısı göründüğünden daha önemlidir:

1. **İçerik yerine hash.** Denetim olayı metni değil, prompt'un ve kanonik araç parametrelerinin SHA-256 özetini saklar. İçeriğe daha sonra ihtiyaç duyulursa, kendi erişim kontrolü ve saklama süresi olan kaynak sistemlerden alınıp hash'e karşı kontrol edilebilir. Denetim izi, kopyası olmaya dönüşmeden *neyin* görüldüğünü kanıtlar.
2. **Tanım değil, kimlik olarak sürümler.** "En son politika" bir olayda işe yaramaz. Her bileşen, Dördüncü bölümdeki yayın akışının imzaladığı aynı değiştirilemez sürüm kimliğini raporlar.

<!-- fig:p11_records -->
<svg viewBox="0 0 720 408" role="img" aria-label="Bir istek gateway, orkestratör, yürütücü, model ve araçlardan geçer; trace_id değişmeden iletilir. Dört kayıt hattı üretilir: örneklenen trace, asla örneklenmeyen denetim olayı, en iyi çaba servis logu ve toplulaştırılmış metrik." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p11-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p11-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Tek İstek, Dört Farklı Kayıt Hattı: Amaç, Örnekleme ve Güvenlik Ayrımı</text>
<!-- Component chain -->
<rect x="16" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="77" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">① Gateway</text>
<text x="77" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yol, kota</text>
<line x1="138" y1="70" x2="154" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="156" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="217" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">② Orkestratör</text>
<text x="217" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">bağlam, τ</text>
<line x1="278" y1="70" x2="294" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="296" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="357" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">③ Yürütücü (PEP)</text>
<text x="357" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">yetki, onay</text>
<line x1="418" y1="70" x2="434" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="436" y="48" width="122" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="497" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">④ Model + Guard</text>
<text x="497" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">çıkarım, puan</text>
<line x1="558" y1="70" x2="574" y2="70" marker-end="url(#p11-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="576" y="48" width="128" height="44" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="640" y="66" text-anchor="middle" style="fill:var(--c-text);font-size:12px;font-weight:700">⑤ Araç / MCP</text>
<text x="640" y="82" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10.5px">eylem, sonuç</text>
<!-- Trace Context bar -->
<rect x="16" y="100" width="688" height="22" rx="5" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="360" y="115" text-anchor="middle" style="fill:var(--c-text);font-size:11px;font-family:var(--font-mono)">trace_id tüm servis zincirine W3C TraceContext ile değişmeden aktarılır ve kayıtları birbirine bağlar</text>
<!-- 4 Stream Cards -->
<!-- 1. Traces -->
<rect x="16" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Trace'ler (APM)</text>
<rect x="26" y="162" width="144" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="98" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:600">Örneklenir (%1)</text>
<text x="26" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Amaç: Gecikme ve darboğaz</text>
<text x="26" y="214" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Kapsam: Span süreleri, meta veri</text>
<text x="26" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Ham metin: Varsayılan kapalı</text>
<text x="26" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Saklama: 7–14 gün · Mühendisler</text>
<rect x="24" y="274" width="148" height="28" rx="4" style="fill:var(--c-danger);fill-opacity:.08;stroke:var(--c-danger);stroke-width:1"/>
<text x="98" y="292" text-anchor="middle" style="fill:var(--c-danger);font-size:9.5px;font-weight:600">⚠️ Örnekleme saldırıyı kaçırır</text>
<!-- 2. Audit Events -->
<rect x="190" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.6"/>
<text x="200" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Denetim Olayları</text>
<rect x="200" y="162" width="144" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.18;stroke:var(--c-warn);stroke-width:1"/>
<text x="272" y="175" text-anchor="middle" style="fill:var(--c-text);font-size:10px;font-weight:700">%100 ASLA ÖRNEKLENMEZ</text>
<text x="200" y="198" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Amaç: Yasal kanıt &amp; adli iz</text>
<text x="200" y="214" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Kapsam: Kimlik, sürüm, karar</text>
<text x="200" y="230" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Tuzlu Hash: Sıfır PII sızıntısı</text>
<text x="200" y="246" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Saklama: 1+ yıl · Uyum &amp; Denetim</text>
<rect x="198" y="274" width="148" height="28" rx="4" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1.2"/>
<text x="272" y="292" text-anchor="middle" style="fill:var(--c-success);font-size:9px;font-weight:700">🛡️ Outbox + Sıra No + Boşluk Alarmı</text>
<!-- 3. Service Logs -->
<rect x="364" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="374" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Servis Logları</text>
<rect x="374" y="162" width="144" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="446" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:600">En İyi Çaba</text>
<text x="374" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Amaç: Çökme, yeniden başlatma</text>
<text x="374" y="214" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Kapsam: Hata &amp; uyarı mesajı</text>
<text x="374" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Format: Yapılandırılmış JSON</text>
<text x="374" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Saklama: 30 gün · Nöbetçi SRE</text>
<rect x="372" y="274" width="148" height="28" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="446" y="292" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">İçerik filtreli, hata odaklı</text>
<!-- 4. Metrics -->
<rect x="538" y="132" width="164" height="182" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="548" y="154" text-anchor="start" style="fill:var(--c-text);font-size:13px;font-weight:700">Metrikler</text>
<rect x="548" y="162" width="144" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="620" y="175" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:600">Toplulaştırılmış</text>
<text x="548" y="198" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Amaç: SLA, kota, bütçe takibi</text>
<text x="548" y="214" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Kapsam: Sayaç ve histogramlar</text>
<text x="548" y="230" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Boyut: token/sn, p99 ms, hata %</text>
<text x="548" y="246" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Saklama: 90+ gün · Panolar</text>
<rect x="546" y="274" width="148" height="28" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="620" y="292" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Sayısal özet, sıfır serbest metin</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="326" width="688" height="68" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="346" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 TEMEL MÜHENDİSLİK KURALI: LOG KANIT DEĞİLDİR, DENETİM OLAYI KANITTIR</text>
<text x="28" y="365" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Trace ve servis logları performans ve hata ayıklama içindir; örnekleme nedeniyle hedeflenmiş saldırıları kaçırır.</text>
<text x="28" y="381" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Güvenlik denetim olayları ise adli kanıttır: %100 kaydedilmeli, asla örneklenmemeli, PII yerine tuzlu hash taşımalı ve Transactional Outbox ile sıra garantisi vermelidir.</text>
</svg>

---

## 3. İkinci bir sızıntı olmadan gözlemlenebilirlik

Giriş olayının en kolay önlenebilecek zararı ikinci sızıntıydı: Soruşturma, prompt'lar ve yanıtlar trace'lere kaydedilip dışa aktarıldığı için müşteri verisini gözlemlenebilirlik sağlayıcısının platformunda buldu. İkinci bölüm log aktarımını TB6 güven sınırı olarak işaretlemişti; acısı burada çıkıyor.

OpenTelemetry GenAI kuralları da aynı görüşte: Talimatlar, kullanıcı mesajları ve model çıktıları hassastır ve çoğu zaman büyüktür; bu yüzden enstrümantasyonlar onları varsayılan olarak kaydetmemelidir. `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` yapılandırmasının varsayılanı `NO_CONTENT`'tir. Kurallar, üretim için içeriği span'lerin üzerine koymak yerine harici bir depoda tutmayı ve referansları kaydetmeyi önerir.

Destek asistanı için kurallar:

1. **Gerçek veri gören bütün ortamlarda içerik kaydı varsayılan olarak kapalı.** Yalnızca sentetik veri kullanan üretim öncesi ortamda ya da maskeleme uygulanmış, kısa ve onaylı bir hata ayıklama penceresinde açın.
2. **Dışa aktarmadan önce maskeleyin, sonra değil.** Altıncı bölümdeki maskeleme politikası (yapılandırılmış PII için sağlama toplamı, isimler için NER), herhangi bir şey güven sınırını terk etmeden *önce* toplayıcıda (collector) çalışır. Maskeleme başarısızlığı içeriği düşürmek demektir, asla geçirmek değil.
3. **Kopya değil, referans.** Span'ler, kimliklerin gösterdiği metni değil, parça kimliklerini, vaka kimliklerini ve takma adları (Bölüm 3) taşır.
4. **Ayrı erişim.** Denetim olayları, kısıtlı erişime ve değiştirilemezliğe (bir kez yazılan ya da bütünlük kontrollü yalnızca ekleme yapılan) sahip bir depoda yaşar. Performans hatası ayıklayan mühendislerin hangi e-postayı kimin onayladığına erişmesi gerekmez.
5. **Türe göre saklama.** Trace'ler günler içinde sona erer; denetim olayları mahremiyet ve hukuk birimiyle üzerinde anlaşılan saklama politikasını izler; içerik, hiç kaydediliyorsa, en hızlı sona erendir.

Kaba bir hacim tahmini, içeriğin yalnızca risk için değil maliyet için de neden önemli olduğunu gösterir. Günde 100.000 istek, istek başına altı span ve span başına yaklaşık 2 KB ile trace'ler günde yaklaşık 1,2 GB üretir. İstek başına yaklaşık 10 KB'lık prompt ve yanıt içeriğini kaydetmek günde 1 GB daha ekler ve hacmi neredeyse ikiye katlar. Üstelik bunun her baytı korunması, saklanması ve sonunda silinmesi gereken kişisel veridir (Üçüncü bölümdeki silme matematiği gözlemlenebilirlik depoları için de geçerlidir).

---

## 4. Kalıcı teslimat ve örneklemenin matematiği

### 4.1. Denetim olayları neden örneklenmemeli?

Örnekleme, performans trace'leri için iyi, güvenlik kanıtı için kötü bir araçtır. Bir saldırının $k$ şüpheli istek ürettiğini ve işlem hattının her isteği birbirinden bağımsız olarak $s$ olasılıkla tuttuğunu varsayalım. Bunlardan **en az birinin** örneklemeden sağ çıkma olasılığı şudur:

$$
P(\text{görüldü}) = 1 - (1 - s)^{k}
$$

Şöyle okuyun: $(1-s)$ bir isteğin düşürülme olasılığıdır, $(1-s)^k$ $k$ isteğin hepsinin düşürülme olasılığıdır, tümleyeni de en az birinin tutulma olasılığıdır.

Giriş hikâyesindeki %1'lik trace örneklemesiyle **yürüyüş**:

```text
s = 0,01, tek kötü niyetli istek (k = 1):      1 − 0,99^1   = 0,01    →  herhangi bir kayıt olma şansı %1
s = 0,01, 50 istekten oluşan bir saldırı:      1 − 0,99^50  = 0,395   → %40
s = 0,01, 300 istekten oluşan bir saldırı:     1 − 0,99^300 = 0,951   → %95
```

Yalnızca bir avuç isteğe ihtiyaç duyan dikkatli bir saldırgan, örneklenmiş veride neredeyse görünmezdir. Bir istek tutulsa bile, olayın kapsamını belirlemek için gereken diğerleri tutulmamıştır. Bu yüzden güvenlik kararlarına ait denetim olayları (eşiğin üzerindeki her dedektör bulgusu, riski $\ge 1$ olan bir araç için her politika kararı, her onay ve her çalıştırma) **her** istek için kaydedilir. Hacimleri küçüktür, çünkü içerik değil, kimlik ve hash içerirler.

### 4.2. Güvenli biçimde başarısız olan teslimat

Kaybolan bir denetim olayı, hiç yazılmamış bir olay kadar kötüdür. Üç mekanizma teslimatı kalıcı kılar:

1. **İşlemden önce yaz.** Yüksek etkili araçlarda yürütücü, aracı çağırmadan *önce* "karar" olayını kalıcı depoya, sonra da "sonuç" olayını yazar. Karar olayı yazılamıyorsa işlem gerçekleşmez: Birinci bölümün yüksek etkili işlemler için karar verdiği gibi, kapalı kalarak başarısız olur.
2. **İşlemsel giden kutusu (transactional outbox).** Olaylar durum değişikliğiyle aynı işlem (transaction) içinde yerel ve kalıcı bir giden kutusuna yazılır; ayrı bir süreç onları yeniden denemelerle denetim deposuna iletir. Bir ağ kesintisi olayları geciktirir, kaybettirmez.
3. **Boşluk tespiti.** Olaylar üretici başına sıra numaraları taşır. Denetim deposu boşluklarda alarm verir; böylece "işlem hattı geçen hafta olayları sessizce düşürdü", "işlem hattı saat 14.02'de 7 numaralı üreticinin 312 olayı atladığı alarmını verdi"ye dönüşür.

### 4.3. Sonucu bilinmeyen işlemler

En zor kayıtlar sonucu bilinmeyen işlemlere aittir: E-posta API'si zaman aşımına uğradı, yürütücü e-postanın gönderilip gönderilmediğini bilmiyor. Sekizinci bölüm yeniden denemeleri güvenli kılmak için idempotency anahtarları kullandı. Denetim izi için yürütücü sonucu `unknown` (bilinmiyor) olarak kaydeder; bir mutabakat işi daha sonra alt akış sistemini idempotency anahtarıyla sorgular ve nihai sonucu yazar. Bir olay sırasında `unknown` sonuçlar, mutabakat yapılana kadar kapsamın bir parçasıdır.

---

## 5. Olayları tespit etmek

Giriş olayını bir müşteri tespit etti. Bu en kötü dedektördür: yavaş, eksik ve herkesin gözü önünde. Daha iyi sinyaller yukarıda tasarlanan kayıtlardan gelir.

Görev başına taban çizgileriyle karşılaştırılan, denetim olaylarından ve metriklerden gelen **davranışsal sinyaller**:

- Onay isteklerinde ya da reddedilme oranlarında artış (kullanıcıların şüpheli önerileri reddetmesi bir belirtidir).
- Tek bir kaynaktan ya da göndericiden getirilen içerikte dedektör puanlarının artışı.
- Yürütücünün doğrulamasının reddettiği araç çağrıları (yanlış alıcı, sınırın üzerindeki tutar): Her biri ucuz atlatılmış bir kazadır (near miss).
- Bütçe tükenmesi, olağan dışı token kullanımı ya da döngüler.
- Karantinaya gönderilen bellek yazmaları (Bölüm 7).

**Kanarya kayıtları.** Dizine ve CRM'e, hiçbir meşru görevin getirmemesi ya da anmaması gereken, her biri benzersiz ve tahmin edilemez bir işaret taşıyan birkaç sentetik müşteri kaydı yerleştirin. Bir işaret bir model çıktısında, giden bir e-postada, bir log aktarımında ya da bir ağ isteğinde bir kez bile görünürse, bir şey okumaması gereken veriyi okuyordur. Kanaryalar, veri sızdırmayı hiç fark etmeyebileceğiniz bir şeyden yüksek kesinlikli bir alarma dönüştürür.

**SOC ile eşleme.** Alarmlar tehdit kaydı kimliğini ve uygun olduğunda MITRE ATLAS tekniğini (örneğin dolaylı injection için `AML.T0051.001` ya da araç çağrısıyla sızdırma için `AML.T0086`) taşır; böylece güvenlik operasyon merkezi AI olaylarını diğerleriyle aynı triyaj süreciyle ele alabilir.

---

## 6. Müdahale: Kontrol altına al, geri al, toparlan, öğren

NIST SP 800-61 Revizyon 3 (Nisan 2025), olay müdahalesini Siber Güvenlik Çerçevesi 2.0'ın (CSF 2.0) bir Topluluk Profili olarak yeniden çerçeveler. Hazırlık, CSF'nin *Govern* (yönet), *Identify* (tanımla) ve *Protect* (koru) işlevlerine; bir olayın ele alınması ise *Detect* (tespit et), *Respond* (müdahale et) ve *Recover* (toparlan) işlevlerine karşılık gelir. Yapı AI sistemlerine iyi uyar; AI'ın eklediği şey yeni bir kontrol altına alma kaldıraçları setidir.

<!-- fig:p11_runbook -->
<svg viewBox="0 0 720 410" role="img" aria-label="Bir döngü hâlinde altı müdahale adımı: tespit, çevreleme, kapsam, kökünü kazıma, toparlanma ve öğrenme. Sağda dar etkiden geniş etkiye 4 seviyeli 8 çevreleme kaldıracı." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p11b-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p11b-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-accent-2)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">AI Olay Müdahale Yaşam Döngüsü ve Kademeli Çevreleme Kaldıraçları</text>
<!-- Left: 6-stage lifecycle -->
<!-- 1. Detect -->
<rect x="16" y="48" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent);stroke-width:1.3"/>
<text x="26" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">① TESPİT</text>
<text x="178" y="68" text-anchor="end" style="fill:var(--c-accent);font-size:10px;font-family:var(--font-mono)">Detect</text>
<text x="26" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">kanarya, sapma, kıl payı</text>
<line x1="102" y1="100" x2="102" y2="114" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 2. Contain -->
<rect x="16" y="116" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.3"/>
<text x="26" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">② ÇEVRELEME</text>
<text x="178" y="136" text-anchor="end" style="fill:var(--c-danger);font-size:10px;font-family:var(--font-mono)">Respond</text>
<text x="26" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">en dar etkili kaldıraç</text>
<line x1="102" y1="168" x2="102" y2="182" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 3. Scope -->
<rect x="16" y="184" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.3"/>
<text x="26" y="204" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">③ KAPSAM</text>
<text x="178" y="204" text-anchor="end" style="fill:var(--c-danger);font-size:10px;font-family:var(--font-mono)">Respond</text>
<text x="26" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">denetim sorgusu, sızıntı</text>
<line x1="188" y1="210" x2="202" y2="210" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 4. Eradicate -->
<rect x="204" y="184" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.3"/>
<text x="214" y="204" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">④ KÖKÜNÜ KAZI</text>
<text x="366" y="204" text-anchor="end" style="fill:var(--c-warn);font-size:10px;font-family:var(--font-mono)">Respond</text>
<text x="214" y="224" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">karantina, politika düzelt</text>
<line x1="290" y1="184" x2="290" y2="170" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 5. Recover -->
<rect x="204" y="116" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.3"/>
<text x="214" y="136" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">⑤ TOPARLANMA</text>
<text x="366" y="136" text-anchor="end" style="fill:var(--c-success);font-size:10px;font-family:var(--font-mono)">Recover</text>
<text x="214" y="156" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">salt-oku → onay → normal</text>
<line x1="290" y1="116" x2="290" y2="102" marker-end="url(#p11b-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<!-- 6. Govern -->
<rect x="204" y="48" width="172" height="52" rx="6" style="fill:var(--c-surface);stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="214" y="68" text-anchor="start" style="fill:var(--c-text);font-size:12px;font-weight:700">⑥ ÖĞRENME</text>
<text x="366" y="68" text-anchor="end" style="fill:var(--c-accent-2);font-size:10px;font-family:var(--font-mono)">Govern</text>
<text x="214" y="88" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">test ekle, tehdit güncelle</text>
<!-- Loop back path -->
<path d="M290 48 V38 H102 V46" marker-end="url(#p11b-g)" style="fill:none;stroke:var(--c-accent-2);stroke-width:1.3;stroke-dasharray:4 3"/>
<text x="196" y="34" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:600">hazırlık döngüsü</text>
<!-- Left summary status -->
<rect x="16" y="248" width="360" height="74" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="26" y="268" text-anchor="start" style="fill:var(--c-text);font-size:11px;font-weight:700">NIST SP 800-61r3 / CSF 2.0 Hizalaması</text>
<text x="26" y="286" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Adli Replay: Deterministic seed ve DB snapshot ile adli canlandırma</text>
<text x="26" y="304" text-anchor="start" style="fill:var(--c-text-mute);font-size:10.5px">• Aşamalı İyileşme: Yalnızca okuma → Zorunlu onay → Normal üretim</text>
<!-- Right: 8 Containment Levers -->
<rect x="392" y="48" width="312" height="274" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="404" y="68" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:700;letter-spacing:.02em">ÇEVRELEME KALDIRAÇLARI (Dar → Geniş)</text>
<!-- Tier 1 -->
<rect x="402" y="78" width="292" height="52" rx="4" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1"/>
<text x="410" y="94" text-anchor="start" style="fill:var(--c-danger);font-size:10px;font-weight:700">SEVİYE 1: İSTEK &amp; ARAÇ (Mikro)</text>
<text x="410" y="109" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Tek bir aracı / MCP fonksiyonunu kapat</text>
<text x="410" y="123" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">• Tüm işlemlerde insan onayını (HITL) zorunlu kıl</text>
<!-- Tier 2 -->
<rect x="402" y="136" width="292" height="52" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1"/>
<text x="410" y="152" text-anchor="start" style="fill:var(--c-warn);font-size:10px;font-weight:700">SEVİYE 2: BELLEK &amp; VERİ (Orta)</text>
<text x="410" y="167" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Şüpheli veri kaynağını karantinaya al</text>
<text x="410" y="181" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">• Bellek (episodic memory) yazmalarını dondur</text>
<!-- Tier 3 -->
<rect x="402" y="194" width="292" height="52" rx="4" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1"/>
<text x="410" y="210" text-anchor="start" style="fill:var(--c-warn);font-size:10px;font-weight:700">SEVİYE 3: KİMLİK &amp; AJAN (Makro)</text>
<text x="410" y="225" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">• Devredilmiş belirteçleri ve oturumları iptal et</text>
<text x="410" y="239" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">• Kill-Switch: Ajanı / ajanları anında dondur</text>
<!-- Tier 4 -->
<rect x="402" y="252" width="292" height="58" rx="4" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1"/>
<text x="410" y="268" text-anchor="start" style="fill:var(--c-danger);font-size:10px;font-weight:700">SEVİYE 4: SİSTEM &amp; MODEL (Kökten)</text>
<text x="410" y="283" text-anchor="start" style="fill:var(--c-text);font-size:10px">• Modeli, prompt'u ya da guardrail politikasını geri al</text>
<text x="410" y="297" text-anchor="start" style="fill:var(--c-text-mute);font-size:9.5px">• Sızmış veri tutan KV ve prefix önbellekleri temizle</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="332" width="688" height="66" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="351" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 OPERASYONEL AKSİYOM: KRİZ ANINDA MİMARİ İCAT EDİLMEZ</text>
<text x="28" y="370" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Her müdahale kaldıracı olaydan önce tanımlanmış, yetkilendirilmiş ve tatbikatı yapılmış olmalıdır.</text>
<text x="28" y="386" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Müdahale daima en dar etkili kaldıraçla başlar (Seviye 1) ve gerekirse genişletilir. Yeni bir kabul testi eklenmeden olay toparlanmış sayılmaz.</text>
</svg>

En dardan en genişe, destek asistanı için **kontrol altına alma kaldıraçları**:

| Kaldıraç | Etki | Tipik kullanım |
| --- | --- | --- |
| Tek bir aracı kapat | `send_email` ya da `issue_refund` durur; özetler devam eder | Tek bir işlemin kötüye kullanıldığından şüphe |
| Bütün işlemler için onayı zorunlu kıl | Her yan etki bir insan ister | Kapsam belirsiz, hizmet sürsün |
| Bir kaynağı ya da koleksiyonu karantinaya al | Retrieval onu kullanmayı bırakır (Bölüm 7) | Zehirlenmiş belgeler ya da kötü niyetli bir gönderici |
| Bellek yazmalarını dondur | Yeni anı yok; mevcutlar incelenir | Bellek zehirlemesinden şüphe |
| Kimlik bilgilerini iptal et | Devredilmiş token'lar ve MCP sunucusu kimlik bilgileri geçersiz kılınır | Kimlik bilgisinin kötüye kullanıldığından şüphe |
| Bir ajanı ya da bütün ajanları durdur (kill switch) | Yürütücü o ajandan gelen bütün önerileri reddeder | Etkin ve yayılan bir olay |
| Modeli, prompt'u ya da politikayı geri al | Önceki onaylı set yüklenir (Bölüm 4) | Olaya bir değişiklik yol açtı |
| Önbellekleri temizle | Sızmış veri içerebilecek önbellek yanıtları silinir | Önbellek yoluyla veri ifşası |

Her kaldıraç olaydan *önce* var olmalı, test edilmiş olmalı ve nöbetçi ekip tarafından dakikalar içinde kullanılabilmelidir. Bir kod dağıtımı gerektiren kill switch, kill switch değildir.

**Kapsam belirleme** denetim izini kullanır: *Etkilenen sürümlerle hangi istekler çalıştı, bağlamlarında hangi kaynaklar vardı, hangi işlemler önerildi, onaylandı ve çalıştırıldı, sonuçları (`unknown` dahil) neydi?* Bu, giriş hikâyesindeki ekibin çalıştıramadığı sorgudur.

**Toparlanma**, alt akıştaki doğru durumu geri getirmeyi (mümkünse yanlış iadeleri geri almak, mahremiyet biriminin gerektirdiği şekilde etkilenen müşterileri bilgilendirmek), `unknown` sonuçların mutabakatını yapmayı ve sistemi adım adım geri getirmeyi kapsar: önce yalnızca okuma, sonra zorunlu onayla, sonra normal.

**Öğrenme**, döngüyü önceki bölümlerle kapatır: İşe yarayan saldırı geliştirme setinde ve bir sonraki ayrılmış sette test vakalarına dönüşür (Bölüm 10), tehdit kaydı yeni ya da güncellenmiş bir satır alır (Bölüm 2) ve başarısız olan kontrol bir gerileme testi kazanır. Bir testi ya da bir tehdit kaydı satırını değiştirmeyen bir olay tekrar yaşanır.

---

## 7. Pratik çıktı: Olay şeması, denetim motoru ve runbook

### 7.1. Güvenlik denetim olayı şeması (örnek)

```json
{
  "event": "tool_decision",
  "event_id": "01J…",
  "producer": "executor-3", "seq": 884213,
  "time": "2026-11-03T09:14:22.418Z",
  "trace_id": "4f2c…",
  "user": "emp-7f3a", "session": "s-91c", "task": "handle_complaint",
  "versions": {"model": "support-7b@a1b2c3", "adapter": "v3", "template": "t2",
               "system_prompt": "p14", "policy": "2026-10-29.3",
               "detectors": {"injection": "inj-clf-v7", "content": "guard-slm-v3"}},
  "context": {"prompt_sha256": "9e1d…", "sources": [
      {"id": "case-8812/email-1", "collection": "complaints", "trust": 0},
      {"id": "crm/cust-4711", "collection": "crm", "trust": 1}], "context_trust": 0},
  "proposal": {"tool": "send_email", "params_sha256": "a168…", "destination_class": "record_email"},
  "findings": [{"detector": "injection", "score": 0.07, "span": "case-8812/email-1"}],
  "decision": {"result": "require_approval", "rule": "send_email.context_trust_lt_3"},
  "approval": {"token_id": "n-81", "approver": "emp-7f3a", "displayed_params_sha256": "a168…"},
  "execution": {"status": "unknown", "idempotency_key": "n-81", "downstream_id": null},
  "route": {"gateway": "g9", "path": "primary", "guards": ["input-policy", "stream-policy"]}
}
```

### 7.2. Adli yeniden oynatma ve giden kutusu denetim motoru

Aşağıdaki sadeleştirilmiş Python motoru üç iş yapar: denetim olaylarını üretici başına sıra numarasıyla bir giden kutusuna (outbox) ekler, bir olay anında geçmiş bir politika kararını deterministik olarak yeniden doğrular ve denetim deposuna ulaşanlardaki boşlukları bulur:

```python
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

@dataclass(frozen=True)
class AuditDecisionEvent:
    event_id: str
    producer_id: str
    seq: int
    trace_id: str
    model_version: str
    policy_version: str
    prompt_hash: str
    proposed_tool: str
    params_hash: str
    decision: str

class ForensicReplayEngine:
    def __init__(self, policy_rules: Dict[str, Any]):
        self.policy_rules = policy_rules

    @staticmethod
    def hash_canonical_json(data: Dict[str, Any], salt: str = "") -> str:
        """Anahtarları sıralı, boşluksuz JSON üzerinde SHA-256 (RFC 8785 JCS'ye bir yaklaşım)."""
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256((canonical + salt).encode("utf-8")).hexdigest()

    def replay_decision(
        self,
        event: AuditDecisionEvent,
        raw_params: Dict[str, Any],
        context_trust: int
    ) -> Tuple[bool, str]:
        """Geçmiş bir politika kararının doğruluğunu deterministik olarak yeniden hesaplar."""
        computed_hash = self.hash_canonical_json(raw_params)
        if computed_hash != event.params_hash:
            return (False, "Parametre bütünlük ihlali: Hash eşleşmiyor!")

        rule_key = f"{event.proposed_tool}.min_trust"
        min_required_trust = self.policy_rules.get(rule_key, 2)

        expected_decision = "allow" if context_trust >= min_required_trust else "require_approval"
        is_consistent = (expected_decision == event.decision)
        return (is_consistent, f"Beklenen: {expected_decision}, Kaydedilen: {event.decision}")


class AuditOutbox:
    """Üretici başına giden kutusu: üretimde append(), durum değişikliğiyle aynı transaction içinde çalışır."""

    def __init__(self, producer_id: str):
        self.producer_id = producer_id
        self.seq = 0
        self.pending: List[AuditDecisionEvent] = []

    def append(self, **fields) -> AuditDecisionEvent:
        self.seq += 1
        event = AuditDecisionEvent(producer_id=self.producer_id, seq=self.seq, **fields)
        self.pending.append(event)   # ayrı bir iletici bunları yeniden denemelerle denetim deposuna taşır
        return event

def find_gaps(received_seqs: List[int]) -> List[int]:
    """Alınan en küçük ve en büyük sıra numarası arasında eksik olanlar."""
    expected = set(range(min(received_seqs), max(received_seqs) + 1))
    return sorted(expected - set(received_seqs))


# 7.1'deki tool_decision olayıyla gösterim
params = {"to": "ayse@example.com", "case_id": 8812}
outbox = AuditOutbox("executor-3")
event = outbox.append(
    event_id="01J-demo", trace_id="4f2c", model_version="support-7b@a1b2c3",
    policy_version="2026-10-29.3", prompt_hash="9e1d",
    proposed_tool="send_email",
    params_hash=ForensicReplayEngine.hash_canonical_json(params),
    decision="require_approval",
)
engine = ForensicReplayEngine({"send_email.min_trust": 3})
print("replay, context trust 0 :", engine.replay_decision(event, params, context_trust=0))
print("replay, edited params   :", engine.replay_decision(event, dict(params, to="x@outside.example"), context_trust=0))

# Denetim deposunun executor-3'ten aldıkları üzerinde boşluk tespiti
print("eksik sıra numaraları    :", find_gaps([884210, 884211, 884213, 884216]))
```

Konsol çıktısı:

```text
replay, context trust 0 : (True, 'Beklenen: require_approval, Kaydedilen: require_approval')
replay, edited params   : (False, 'Parametre bütünlük ihlali: Hash eşleşmiyor!')
eksik sıra numaraları    : [884212, 884214, 884215]
```

### 7.3. AI güvenlik olayı müdahale runbook'u (kesit)

1. **Tespit ve triyaj (Detect).** Kanaryadan, davranışsal bir sinyalden ya da bir bildirimden alarm geldi. Önem derecesini varlığa (Bölüm 1) ve etki alanına göre ata. Tehdit kaydı kimliği ve biliniyorsa ATLAS tekniğiyle bir olay kaydı aç.
2. **Kontrol altına al (Respond).** Tablodaki en dar etkili kaldıracı uygula; kapsam 30 dakika içinde netleşmezse daha geniş kaldıraçlara tırman. Kullanılan her kaldıracı zamanı ve uygulayan kişiyle kaydet.
3. **Kapsamı belirle (Respond).** Denetim olaylarını etkilenen sürümlere, kaynaklara, araçlara ve zaman aralığına göre sorgula. Çalıştırılan işlemleri ve `unknown` sonuçları listele. Veri ifşası için kanaryaları ve log aktarımlarını kontrol et.
4. **Kökünü kazı (Respond).** Nedeni ortadan kaldır: kaynakları karantinaya al, politikayı düzelt, modeli ya da prompt'u geri al, kimlik bilgilerini yenile.
5. **Toparlan (Recover).** Sonuçların mutabakatını yap, alt akıştaki durumu düzelt, gerektiği gibi bilgilendir, hizmeti aşamalı geri getir (yalnızca okuma → zorunlu onay → normal).
6. **Öğren (Govern / Identify).** Beş iş günü içinde olay sonrası inceleme: denetim olaylarından zaman çizelgesi, kök neden, yeni test vakaları, tehdit kaydı güncellemesi, kalan risk için sahibin imzası.

### 7.4. Mühendisliğin iki gözü: Kurulum ve çalışma zamanı

**Kurulum zamanı:** Olay şeması, trace yayılımıyla enstrümantasyon, içerik kaydı varsayılanları ve toplayıcıda maskeleme, değiştirilemezliği ve erişim kontrolü olan ayrı denetim deposu, giden kutusu, kanarya kayıtları, alarm kuralları ve inşa edilip tatbikatı yapılmış kontrol altına alma kaldıraçları.

**Çalışma zamanı:** Her güvenlik kararı için eksiksiz denetim olayları, performans için örneklenmiş trace'ler, boşluk tespiti, bilinmeyen sonuçların mutabakatı ve herhangi bir kaldıracı dakikalar içinde çekebilen bir nöbetçi ekip.

---

## Bütün hikâye altı satırda

- Trace'lerin, güvenlik denetim olaylarının, servis loglarının ve metriklerin işleri farklıdır; trace'ler örneklenebilir, denetim olayları asla.
- Bir denetim olayı; kimin istediğini, modelin ne gördüğünü, ne önerdiğini, hangi kontrollerin karar verdiğini, kimin onayladığını ve ne olduğunu her sürümle birlikte yeniden kurmanızı sağlamalıdır.
- İçerik yerine hash, kimlik ve sürüm saklayın; içerik kaydını varsayılan olarak kapalı tutun ve dışa aktarmadan önce maskeleyin, yoksa gözlemlenebilirlik ikinci bir sızıntıya dönüşür.
- %1 örneklemede tek bir kötü niyetli istek %1 olasılıkla kayıt bırakır; işlemden önce kararı yazın, giden kutusu kullanın, boşlukları tespit edin ve bilinmeyen sonuçların mutabakatını yapın.
- Kanarya kayıtları, yürütücüdeki ucuz atlatılmış kazalar ve davranışsal taban çizgileri olayları müşterilerden daha hızlı tespit eder.
- Müdahale, önceden hazırlanmış AI kaldıraçlarıyla NIST SP 800-61r3 ve CSF 2.0'ı izler: araçları kapat, onayı zorunlu kıl, kaynakları karantinaya al, belleği dondur, iptal et, durdur, geri al, temizle; sonra olayı testlere dönüştür.

---

## Terimler sözlüğü

- **Trace:** Bir isteğin servisler boyunca izlediği yolun kaydı; hata ayıklamada kullanılır, örnekleme kabul edilebilir.
- **Güvenlik denetim olayı (security audit event):** Bir güvenlik kararının eksiksiz ve kurcalamaya karşı izli kaydı; kanıt olarak kullanılır, asla örneklenmez.
- **Trace kimliği yayılımı (trace ID propagation):** Bir isteğin yaptığı her çağrıya tek bir kimlik geçirmek; böylece o isteğin bütün kayıtları birleştirilebilir.
- **İçerik kaydı (content capture):** Prompt'ları ve yanıtları telemetride kaydetmek; hassas veriyi kopyaladığı için varsayılan olarak kapalıdır.
- **İşlemsel giden kutusu (transactional outbox):** Olayları durum değişikliğiyle aynı işlemde yerel kalıcı depoya yazıp ardından yeniden denemelerle iletmek.
- **Boşluk tespiti (gap detection):** Kaybolan olayları fark etmek için üretici başına sıra numaralarını kullanmak.
- **Bilinmeyen sonuç (unknown outcome):** Sonucu doğrulanmamış (örneğin bir zaman aşımından sonra) ve mutabakat yapılana kadar kapsamda tutulan işlem.
- **Kanarya kaydı (canary record):** Hiçbir meşru görevin dokunmaması gereken sentetik kayıt; herhangi bir yerde görünmesi veri erişimine ya da sızdırmaya işaret eder.
- **Kontrol altına alma kaldıracı (containment lever):** Bir olayın yayılmasını sınırlayan, önceden hazırlanmış ve test edilmiş kontrol; bir aracı kapatmak ya da onayı zorunlu kılmak gibi.
- **Kill switch:** Bir ajanın işlemlerini kod dağıtımı gerektirmeden anında durduran kaldıraç.

---

## Daha derine inmek için

- Önceki bölüm: [Secure AI by Design (10): AI Güvenliğini Ölçmek, Red Teaming ve Türkçe Değerlendirme](post.html?slug=ai-guvenligini-olcmek)
- Sonraki bölüm: [Secure AI by Design (12): Standarttan Kanıta, AI Güvenliğini Denetlenebilir Hâle Getirmek](post.html?slug=standarttan-kanita)
- NIST, [*SP 800-61 Rev. 3: Incident Response Recommendations and Considerations for Cybersecurity Risk Management, a CSF 2.0 Community Profile*](https://csrc.nist.gov/pubs/sp/800/61/r3/final) (Nisan 2025).
- NIST, [*Cybersecurity Framework (CSF) 2.0*](https://www.nist.gov/cyberframework).
- OpenTelemetry, [*Semantic conventions for generative AI*](https://github.com/open-telemetry/semantic-conventions-genai) (durum: Development; içerik kaydı rehberi).
- OWASP, [*AISVS 1.0*](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/), C12 bölümü (izleme, loglama ve anomali tespiti).
- AI olaylarını teknik kimlikleriyle etiketlemek için MITRE [*ATLAS*](https://atlas.mitre.org/).
- Bu blogdan: Altyapı katmanındaki operasyonel telemetri için [AI kümesi neden yavaşlar?](post.html?slug=gpudan-ai-super-bilgisayarina)