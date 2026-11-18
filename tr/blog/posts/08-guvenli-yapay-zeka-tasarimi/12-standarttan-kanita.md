Canlıya geçişe iki hafta kala büyük bir müşteri bir soru formu gönderdi: *"Asistan ISO/IEC 42001 ile uyumlu mu? LLM Uygulamaları için OWASP Top 10'u izliyor musunuz? Güvenli mi?"* Ekip ciddi bir iş çıkarmıştı: bir tehdit kaydı, imzalı artefaktlar, her araç çağrısında bir güven kuralı, parametrelere bağlı onaylarla çalışan bir yürütücü, korumalı bir gateway, bir değerlendirme raporu ve bir olay runbook'u. Yine de yanıtın ilk taslağı şöyleydi: *"Evet, ISO 42001 ve OWASP ile tamamen uyumluyuz ve asistan güvenlidir."* Bu cümlenin her parçası yanlıştı. Şirket ISO/IEC 42001 belgesine sahip değildi. OWASP Top 10 bir farkındalık listesidir, "uyum sağlanan" bir şey değildir. "Güvenli" kelimesi ise kapsam, yöntem ve sürüm olmadan hiçbir kanıtın destekleyemeyeceği bir iddiadır.

Sorun eksik kontroller değildi. Eksik olan bir **zincirdi**: bir belgedeki bir gereksinimden onu karşılayan kontrole, kontrolü gerçekleştiren özelliğe ya da politikaya, onu sınayan teste, testin belirli bir sürümde geçtiğini gösteren kanıta ve geriye kalan riske uzanan bir zincir. Bu zincirle dürüst yanıt hem daha kısa hem daha güçlüdür: *"Kontrollerimiz ISO/IEC 27090, OWASP AISVS Seviye 2 ve OWASP LLM ile Ajan Top 10 kategorileriyle eşlenmiştir; 2026-11 sürümü için kanıt dizini ekte. Uygulanabilir AISVS gereksinimlerinin %87,5'i doğrulanmıştır, kalan açıklar sahipleri ve tarihleriyle listelenmiştir."*

Serinin bu son bölümü o kanıt zincirini kuruyor. Standart ve kılavuz belgelerini yapısal niteliklerine, temsil ettikleri kanıt türüne ve ne kadarlık bir güvence sağladıklarına göre sınıflandıracak; ISO/IEC 42001, 23894, 42005 ve 27090'ı mimarideki rollerine yerleştirecek, kanıt zincirini ve kapsama oranını tanımlayacak, destek asistanının on iki bölümden oluşan birleşik referans mimarisini kuracak ve bir eşleştirme matrisi, bir kanıt dizini ve riske dayalı bir canlıya geçiş kontrol listesiyle seriyi tamamlayacağız.

> **Önceki yazılardan devraldığımız fikir:** Her şey. [Birinci bölümün](post.html?slug=yapay-zeka-guvenligi-tasarimda-baslar) varlıkları ve sahipleri, [İkinci bölümün](post.html?slug=ai-tehdit-modelleme-ve-guven-sinirlari) tehdit kaydı, Üçüncü ile Dokuzuncu bölümler arasındaki kontroller, [Onuncu bölümün](post.html?slug=ai-guvenligini-olcmek) değerlendirme raporu ve [On birinci bölümün](post.html?slug=ai-izlenebilirlik-ve-olay-mudahalesi) denetim izi tek bir kanıt zincirinin halkalarına dönüşüyor.

> **Bu yazının sonunda**
>
> - Bir yönetim sistemi standardını, bir rehber belgeyi, bir çerçeveyi, bir taksonomiyi, bir farkındalık listesini ve bir doğrulama standardını birbirinden ve her birinin neyi iddia etmenize izin verdiğini ayırabileceksiniz.
> - ISO/IEC 42001, 23894, 42005 ve 27090'ın rollerini açıklayabileceksiniz.
> - Gereksinimden kalan riske uzanan bir kanıt zinciri kurup kapsama oranını hesaplayabileceksiniz.
> - Geriye kalanların açıkça kabul edildiği, riske dayalı programatik bir canlıya geçiş kapısı yürütebileceksiniz.

**Bu yazıda**

- [1. Bütün belgeler aynı türden değildir](#1-bütün-belgeler-aynı-türden-değildir)
- [2. ISO ailesi: Yönetim, risk, etki ve güvenlik](#2-iso-ailesi-yönetim-risk-etki-ve-güvenlik)
- [3. Dürüstçe ne iddia edebilirsiniz?](#3-dürüstçe-ne-iddia-edebilirsiniz)
- [4. Kanıt zinciri ve kapsama oranı](#4-kanıt-zinciri-ve-kapsama-oranı)
- [5. Destek asistanı, bir araya getirilmiş hâliyle](#5-destek-asistanı-bir-araya-getirilmiş-hâliyle)
- [6. Riske dayalı bir yayın kapısı](#6-riske-dayalı-bir-yayın-kapısı)
- [7. Pratik çıktı: Eşleştirme matrisi, kanıt dizini ve canlıya geçiş kontrol listesi](#7-pratik-çıktı-eşleştirme-matrisi-kanıt-dizini-ve-canlıya-geçiş-kontrol-listesi)
- [Bütün hikâye altı satırda](#bütün-hikâye-altı-satırda)
- [Terimler sözlüğü](#terimler-sözlüğü)
- [Daha derine inmek için](#daha-derine-inmek-için)

---

## 1. Bütün belgeler aynı türden değildir

Bir yapı denetçisi, bir yangın güvenliği rehberi, ev yangınlarının en yaygın on nedeninin listesi ve bir kundaklama teknikleri kataloğu; bir ev inşa ederken hepsi işe yarar, ama farklı türden belgelerdir ve onları farklı biçimlerde kullanırsınız. Kimse evinin "en yaygın on yangın nedeni listesiyle uyumlu" olduğunu söylemez.

Bu serinin kaynak gösterdiği belgeler yedi türe ayrılır:

| Tür | Örnekler | Amaç | Belgelendirme mümkün mü? | Tipik dürüst iddia |
| --- | --- | --- | --- | --- |
| **Yönetim sistemi standardı** | ISO/IEC 42001:2023 (yapay zekâ), ISO/IEC 27001 (bilgi güvenliği), ISO/IEC 27701:2025 (mahremiyet) | Bir kurumun nasıl yönettiğine, rolleri nasıl atadığına, riski nasıl yönettiğine ve nasıl iyileştiğine dair gereksinimler | Evet, akredite bir kuruluş tarafından, tanımlı bir kapsam için | "[Kuruluş] tarafından ISO/IEC 42001 belgesi verildi, kapsam [X], sertifika [no]" |
| **Rehber standart** | ISO/IEC 23894:2023 (AI riski), ISO/IEC 42005:2025 (etki değerlendirmesi), ISO/IEC 27090:2026 (AI güvenlik tehditleri) | Bir şeyin nasıl yapılacağına dair öneriler | Hayır | "Sürecimiz ISO/IEC 23894 rehberini izler" |
| **Avrupa / ulusal standart** | ETSI EN 304 223 (Ocak 2026) | Temel gereksinimler; burada beş yaşam döngüsü evresine yayılan 13 ilke | Üzerine kurulan programa bağlı | "EN 304 223'e göre değerlendirildi, sonuçlar [raporda]" |
| **Çerçeve** | NIST AI RMF 1.0, NIST AI 600-1, NIST CSF 2.0, NIST SP 800-61r3, SP 800-218A | İşi düzenlemek için işlevler ve çıktılardan oluşan bir yapı | Hayır | "NIST AI RMF işlevlerine göre düzenlenmiştir" |
| **Taksonomi / bilgi tabanı** | NIST AI 100-2 E2025, MITRE ATLAS | Saldırılar ve önlemler için ortak adlar | Hayır | "Tehditler ATLAS teknik kimlikleriyle belirlenmiştir" |
| **Farkındalık listesi** | OWASP Top 10 for LLM Applications 2026, OWASP Top 10 for Agentic Applications | Önceliklendirme için en önemli risk kategorileri | Hayır | "Her Top 10 kategorisi tehdit kaydında ele alınmıştır" |
| **Doğrulama standardı / spesifikasyon** | OWASP AISVS 1.0 (seviye 1–3), OWASP ASVS 5.0, SLSA 1.2, MCP spesifikasyonu | Test edilebilir gereksinimler ya da protokol kuralları | Belgelendirme değil, ama doğrulanabilir | "AISVS Seviye 2'ye karşı doğrulandı; kanıt dizini ekte" |

Bu ayrım ardından gelen her şeyi belirler. Bir **yönetim sistemi** standardı *"kurumun işleyen bir süreci var mı?"*, bir **doğrulama standardı** *"bu sistem bu test edilebilir gereksinimleri karşılıyor mu?"*, bir **taksonomi** ise *"tehditleri başkalarının anlayacağı biçimde adlandırdınız mı?"* diye sorar. Bunları karıştırmak kulağa güçlü gelen ama hiçbir şey ifade etmeyen iddialara yol açar.

---

## 2. ISO ailesi: Yönetim, risk, etki ve güvenlik

Dört ISO/IEC belgesi, yapay zekâ etrafındaki işi Birinci bölümdeki rollerle örtüşen bir biçimde bölüşür:

- **ISO/IEC 42001:2023: Yapay zekâ yönetim sistemi.** Bir yapay zekâ yönetim sisteminin kurulması, uygulanması, sürdürülmesi ve sürekli iyileştirilmesi için gereksinimleri belirler: politika, roller ve sorumluluklar, risk ve etki süreçleri, kontroller, izleme ve iyileştirme. Belgelendirilebilir. Birinci bölümdeki RACI matrisi ve aşağıdaki canlıya geçiş imzası, onun en küçük pratik ifadeleridir.
- **ISO/IEC 23894:2023: Yapay zekâda risk yönetimi rehberi** (Şubat 2023'te yayımlandı). ISO 31000:2018'in genel risk yönetimi sürecini yapay zekâya uygular ve yapay zekâya özgü hususlar ekler. Rehberdir, belgelendirilemez. İkinci bölümdeki tehdit kaydı ve puanlama, onun sürecini yürütmenin bir yoludur.
- **ISO/IEC 42005:2025: AI sistemi etki değerlendirmesi** (Mayıs 2025'te yayımlandı). Bir yapay zekâ sisteminin bireyler, gruplar ve toplum üzerindeki etkilerinin değerlendirilmesine dair rehberlik sunar. Bu, güvenlik riskinden farklı bir sorudur: *Kim etkilenebilir ve nasıl?* Destek asistanı için bu soru, şikâyetleri özetlenen müşterileri ve işi değişen çalışanları kapsar.
- **ISO/IEC 27090:2026: Yapay zekâ için siber güvenlik rehberi.** BSI, Ekim 2026'da onu *Addressing security threats and compromises to AI systems* (AI sistemlerine yönelik güvenlik tehditlerini ve ihlalleri ele almak) başlığıyla duyurdu. Veri zehirleme, prompt injection, model hırsızlığı ve eğitim verisinin ya da girdilerin sızması gibi saldırıların yaşam döngüsü boyunca AI verilerini, modellerini, çıktılarını ve hizmetlerini nasıl etkilediğini ve geleneksel ve AI'a özgü kontrollerin bunlara nasıl karşılık verdiğini açıklar. Ayrı bir yönetim sistemi yaratmadan ISO/IEC 27001, 27002 ve 42001 ile birlikte çalışmak üzere tasarlanmış bir rehberdir.

Bu serinin planındaki bir dürüstlük kuralı burada özellikle güçlü biçimde geçerlidir: ISO metinleri ücretli belgelerdir; herkese açık özetler ve basın duyuruları maddelerini değil, kapsamlarını anlatır. **Madde düzeyinde bir uyum ifadesi ancak asıl metin okunduktan sonra yazılabilir.** O zamana kadar dürüst iddia "ISO/IEC 27090'ın X maddesiyle uyumlu" değil, "yayımlanmış özetinde tarif edilen ISO/IEC 27090 kapsamıyla eşlenmiştir" olur.

---

## 3. Dürüstçe ne iddia edebilirsiniz?

Bir standardın adını içeren herhangi bir cümle yazmadan önce onu üç soruyla kontrol edin: *Belgenin kurumsal ve teknik niteliği nedir? Tam olarak hangi sistem, hangi sürüme karşı, kim tarafından değerlendirildi? Kapsam ve yasal dayanak nedir?*

| Kaçının | Bunun yerine yazın |
| --- | --- |
| "ISO 42001 ile tamamen uyumluyuz" | "[Kuruluş] tarafından ISO/IEC 42001:2023 belgesi verildi; kapsam: AI müşteri destek hizmetleri, sertifika [no], [tarih]e kadar geçerli". Belge yoksa: "AI yönetim uygulamalarımız ISO/IEC 42001 esas alınarak tasarlandı; belgelendirme [tarih] için planlandı" |
| "OWASP uyumlu" | "OWASP LLM Top 10 (2026) ve Ajan Top 10'un her kategorisi kaydımızdaki tehditler ve kontrollerle eşlendi" |
| "Güvenli", "sıfır sızıntı", "tam koruma" | "Değerlendirme seti v4'te (1.200 saldırı vakası, EN/TR) zarar oranı 0, %95 üst sınırı %0,25; model, politika ve veri seti sürümleri listelendi" |
| "NIST'i izliyoruz" | "Risk sürecimiz NIST AI RMF'nin Govern, Map, Measure ve Manage işlevlerine göre düzenlendi" |
| "AISVS'yi karşılıyoruz" | "OWASP AISVS 1.0 Seviye 2'ye karşı doğrulandı: Uygulanabilir 128 gereksinimden 112'si kanıtlı; sahibi belli 16 açık madde" |

Üçüncü ve beşinci satırlardaki sayılar bir iddianın *biçimini* gösteren örneklerdir. Gerçek raporlar yalnızca ölçülmüş değerler içerir (Bölüm 10).

---

## 4. Kanıt zinciri ve kapsama oranı

Kanıt zinciri, her biri kalıcı bir kimliğe sahip altı şeyi birbirine bağlar:

1. **Kaynak gereksinim:** Bir belgeden bir madde, ilke, kategori ya da gereksinim; örneğin bir AISVS gereksinimi, bir EN 304 223 ilkesi, bir OWASP kategorisi ya da bir ATLAS tekniği.
2. **Kontrol:** Onu karşılayan tasarım kararı; bu serinin kontrollerinden biri (örneğin "C-08 alıcı vaka kaydına bağlıdır").
3. **Gerçekleştirme:** Onu hayata geçiren özellik, yapılandırma ya da politika, sürümüyle birlikte (yürütücü kuralı `send_email.recipient_in_record`, politika `2026-10-29.3`).
4. **Test:** Gerçekleştirmenin çalıştığını gösteren, geçme ölçütü olan kontrol (PI-1: "200 varyant, müşteri dışı adreslere 0 gönderim").
5. **Kanıt:** O testin belirli bir sistem sürümündeki sonucu, değiştirilemez biçimde saklanmış (rapor kimliği, çalıştırma kimliği, artefakt hash'leri).
6. **Kalan risk:** Geriye kalan, puanı ve onu kabul eden sahibi (İkinci bölümdeki $R'$, Birinci bölümdeki RACI'ye göre imzalanmış).

<!-- fig:p12_chain -->
<svg viewBox="0 0 720 236" role="img" aria-label="Birbirine bağlı altı halka: kaynak standart gereksinimi, kontrol tanımı, kod içi gerçekleştirme, doğrulama testi, imzalı çalıştırma kanıtı ve iş birimi sahibinin onayladığı kalan risk." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p12c-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Kriptografik Kanıt Zinciri: Standarttan Kalan Riske 6 Kırılmaz Halka</text>
<!-- 6 Chain Links -->
<!-- 1 -->
<rect x="16" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="22" y="54" width="92" height="18" rx="4" style="fill:var(--c-surface);stroke:var(--c-border);stroke-width:1"/>
<text x="68" y="67" text-anchor="middle" style="fill:var(--c-text-mute);font-size:10px;font-weight:700">① KAYNAK</text>
<text x="68" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Gereksinim</text>
<text x="68" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">AISVS L2 (v1.0)</text>
<text x="68" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">OWASP LLM01</text>
<text x="68" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">ATLAS AML.T0051</text>
<rect x="24" y="136" width="88" height="14" rx="3" style="fill:var(--c-accent);fill-opacity:.12"/>
<text x="68" y="146.5" text-anchor="middle" style="fill:var(--c-accent);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">STANDART</text>
<line x1="120" y1="101" x2="127" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 2 -->
<rect x="132" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<rect x="138" y="54" width="92" height="18" rx="4" style="fill:var(--c-danger);fill-opacity:.12;stroke:var(--c-danger);stroke-width:1"/>
<text x="184" y="67" text-anchor="middle" style="fill:var(--c-danger);font-size:10px;font-weight:700">② KONTROL</text>
<text x="184" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Politika</text>
<text x="184" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">C-08: Alıcı vaka</text>
<text x="184" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">kaydındaki müşteriyle</text>
<text x="184" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">aynı olmalıdır</text>
<rect x="140" y="136" width="88" height="14" rx="3" style="fill:var(--c-danger);fill-opacity:.12"/>
<text x="184" y="146.5" text-anchor="middle" style="fill:var(--c-danger);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">KURAL</text>
<line x1="236" y1="101" x2="243" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 3 -->
<rect x="248" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent);stroke-width:1.2"/>
<rect x="254" y="54" width="92" height="18" rx="4" style="fill:var(--c-accent);fill-opacity:.12;stroke:var(--c-accent);stroke-width:1"/>
<text x="300" y="67" text-anchor="middle" style="fill:var(--c-accent);font-size:10px;font-weight:700">③ UYGULAMA</text>
<text x="300" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">PEP Kodu</text>
<text x="300" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Yürütücü kuralı</text>
<text x="300" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">send_email allowlist</text>
<text x="300" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Sürüm: 10-29.3</text>
<rect x="256" y="136" width="88" height="14" rx="3" style="fill:var(--c-accent);fill-opacity:.12"/>
<text x="300" y="146.5" text-anchor="middle" style="fill:var(--c-accent);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">KOD</text>
<line x1="352" y1="101" x2="359" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 4 -->
<rect x="364" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface-2);stroke:var(--c-accent-2);stroke-width:1.2"/>
<rect x="370" y="54" width="92" height="18" rx="4" style="fill:var(--c-accent-2);fill-opacity:.12;stroke:var(--c-accent-2);stroke-width:1"/>
<text x="416" y="67" text-anchor="middle" style="fill:var(--c-accent-2);font-size:10px;font-weight:700">④ TEST</text>
<text x="416" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Regresyon</text>
<text x="416" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Test PI-1 (ART)</text>
<text x="416" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">200 TR/EN varyant</text>
<text x="416" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Müşteri dışına 0 sızıntı</text>
<rect x="372" y="136" width="88" height="14" rx="3" style="fill:var(--c-accent-2);fill-opacity:.12"/>
<text x="416" y="146.5" text-anchor="middle" style="fill:var(--c-accent-2);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">CI/CD</text>
<line x1="468" y1="101" x2="475" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 5 -->
<rect x="480" y="48" width="104" height="106" rx="8" style="fill:var(--c-surface);stroke:var(--c-success);stroke-width:1.2"/>
<rect x="486" y="54" width="92" height="18" rx="4" style="fill:var(--c-success);fill-opacity:.12;stroke:var(--c-success);stroke-width:1"/>
<text x="532" y="67" text-anchor="middle" style="fill:var(--c-success);font-size:10px;font-weight:700">⑤ KANIT</text>
<text x="532" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">İmzalı Log</text>
<text x="532" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">rc-2026-11.2 adayı</text>
<text x="532" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Çalıştırma #7731</text>
<text x="532" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Sigstore SHA-256</text>
<rect x="488" y="136" width="88" height="14" rx="3" style="fill:var(--c-success);fill-opacity:.12"/>
<text x="532" y="146.5" text-anchor="middle" style="fill:var(--c-success);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">MERKLE</text>
<line x1="584" y1="101" x2="591" y2="101" marker-end="url(#p12c-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>

<!-- 6 -->
<rect x="596" y="48" width="108" height="106" rx="8" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<rect x="602" y="54" width="96" height="18" rx="4" style="fill:var(--c-warn);fill-opacity:.12;stroke:var(--c-warn);stroke-width:1"/>
<text x="650" y="67" text-anchor="middle" style="fill:var(--c-warn);font-size:10px;font-weight:700">⑥ RİSK</text>
<text x="650" y="88" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px;font-weight:700">Kalan Risk</text>
<text x="650" y="104" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">Tehdit T1: R 20→4</text>
<text x="650" y="117" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">İştah: A=6 (Uygun)</text>
<text x="650" y="130" text-anchor="middle" style="fill:var(--c-text-mute);font-size:9.5px">İş sahibi onayladı</text>
<rect x="606" y="136" width="88" height="14" rx="3" style="fill:var(--c-warn);fill-opacity:.12"/>
<text x="650" y="146.5" text-anchor="middle" style="fill:var(--c-warn);font-size:8.5px;font-weight:700;font-family:var(--font-mono)">ONAY</text>

<!-- Bottom Educational Banner -->
<rect x="16" y="164" width="688" height="62" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="183" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 KANIT ZİNCİRİ İLKESİ: DOĞRULANMIŞ İDDİA = ALTI HALKANIN TAMAMI</text>
<text x="28" y="201" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Bir güvenlik iddiası ancak altı halkanın tamamı aynı yayın sürümü (release candidate) için eksiksiz ise geçerlidir.</text>
<text x="28" y="217" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">Model ağırlığı, prompt, guardrail politikası veya araç şeması değiştiğinde eski sürüme ait kanıtlar anında geçersiz kalır.</text>
</svg>

Bir gereksinim ancak altı halkanın hepsi mevcutsa ve kanıt yayımlanan sürüme aitse **doğrulanmış** sayılır. Bir doğrulama hedefinin **kapsama oranı** şudur:

$$
\text{Kapsama} = \frac{\left|\, V \cap A \,\right|}{\left|\, A \,\right|}
$$

Şöyle okuyun: $A$, sisteme uygulanabilen gereksinimlerin kümesidir (belgelenmiş dışlamalardan sonra). $V$, bu sürüm için zinciri eksiksiz olan gereksinimlerin kümesidir. $|\cdot|$ öğeleri sayar. Kapsama oranı, uygulanabilir gereksinimlerin doğrulanmış olan payıdır.

OWASP AISVS 1.0 Seviye 2 hedefi için **yürüyüş**. AISVS 1.0, 191 gereksiniminin her birini bir seviyeye atar: Seviye 1'de 51, Seviye 2'de 95 ve Seviye 3'te 45. Seviye 2'de doğrulama yapmak, Seviye 1 ve Seviye 2 gereksinimlerini doğrulamak demektir:

```text
Seviye 1 + Seviye 2 gereksinimleri:       51 + 95 = 146
Uygulanamaz (belgelenmiş gerekçe,         ör. temel model eğitmiyoruz
  ör. kendi ön eğitimimiz yok):            − 18
Uygulanabilir |A|:                         146 − 18 = 128
Bu sürüm için eksiksiz zincir |V ∩ A|:     112
Kapsama = 112 / 128 = 0,875                → %87,5
Açık maddeler: 128 − 112 = 16              → her birinin bir sahibi, bir planı ya da imzalı kabulü olmalı
```

İki özellik kapsama oranını kozmetik olmaktan çıkarıp işe yarar kılar. Birincisi, **dışlamalar belgelenir**: "Uygulanamaz" bir inceleyicinin kontrol edebileceği bir gerekçe ister; yoksa açıkları gizlemenin bir yoluna dönüşür. İkincisi, **kapsama sürüme bağlıdır**: Yeni bir model ya da politika sürümü eski sürüme dayanan kanıtı geçersiz kılar. Bu yüzden gerileme testleri yeniden çalıştırılmadıysa kapsama yayın günü düşebilir. İstenen davranış da tam olarak budur.

---

## 5. Destek asistanı, bir araya getirilmiş hâliyle

On iki bölüm tek bir sisteme katmanlar ekledi. Bir araya getirildiğinde referans mimari şöyle görünüyor:

<!-- fig:p12_arch -->
<svg viewBox="0 0 720 536" role="img" aria-label="Beş bant hâlinde eksiksiz referans mimari. Yönetişim: Bölüm 1&#x27;den varlıklar ve RACI, risk kabulü, Bölüm 12&#x27;den kanıt dizini ve canlıya geçiş kapısı. Kurulum zamanı: Bölüm 2&#x27;den tehdit kaydı, Bölüm 3&#x27;ten veri yaşam döngüsü, Bölüm 4&#x27;ten tedarik zinciri ve yayın akışı, Bölüm 10&#x27;dan değerlendirme ve red teaming. Çalışma zamanı istek yolu: sohbet arayüzü, Bölüm 9&#x27;dan gateway, Bölüm 5&#x27;ten bağlam etiketleri ve güven, Bölüm 6&#x27;dan guardrail&#x27;ler ve politika motoru, Bölüm 7&#x27;den yetkiye duyarlı retrieval ve bellek, Bölüm 9&#x27;dan izole model sunucusu. Çalışma zamanı işlem yolu: Bölüm 8&#x27;den yetki kesişimi, parametrelere bağlı onaylar, bütçeler ve idempotency ile araçları ve MCP sunucularını çağıran yürütücü. Operasyon: Bölüm 11&#x27;den denetim olayları, maskelenmiş telemetri, kanaryalar ve olay runbook&#x27;u." style="max-width:100%;height:auto;display:block;margin:var(--sp-5) auto;font-family:var(--font-sans)">
<defs>
<marker id="p12a-arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-text-mute)"/></marker>
<marker id="p12a-w" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" style="fill:var(--c-warn)"/></marker>
</defs>
<text x="16" y="24" text-anchor="start" style="fill:var(--c-accent);font-size:14px;font-weight:700;letter-spacing:.02em">Secure AI by Design: destek asistanı, bir araya getirilmiş hâliyle</text>
<rect x="16" y="44" width="688" height="64" rx="8" style="fill:var(--c-accent);fill-opacity:.05;stroke:var(--c-accent);stroke-width:1.3"/>
<text x="28" y="61" text-anchor="start" style="fill:var(--c-accent);font-size:11.5px;font-weight:700">YÖNETİŞİM</text>
<rect x="28" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="156" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B1</text>
<text x="38" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Varlık, sahip, RACI</text>
<rect x="198" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B2</text>
<text x="208" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Risk kabulü</text>
<rect x="368" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B12</text>
<text x="378" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Kanıt dizini</text>
<rect x="538" y="64" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="68" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="79.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B12</text>
<text x="548" y="95" text-anchor="start" style="fill:var(--c-text);font-size:12px">Canlıya geçiş kapısı</text>
<rect x="16" y="118" width="688" height="64" rx="8" style="fill:var(--c-accent-2);fill-opacity:.05;stroke:var(--c-accent-2);stroke-width:1.3"/>
<text x="28" y="135" text-anchor="start" style="fill:var(--c-accent-2);font-size:11.5px;font-weight:700">KURULUM ZAMANI</text>
<rect x="28" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="156" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B2</text>
<text x="38" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tehdit kaydı</text>
<rect x="198" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B3</text>
<text x="208" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Veri yaşam döngüsü</text>
<rect x="368" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B4</text>
<text x="378" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Tedarik, yayın</text>
<rect x="538" y="138" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="142" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="153.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B10</text>
<text x="548" y="169" text-anchor="start" style="fill:var(--c-text);font-size:12px">Değerlendirme</text>
<rect x="16" y="192" width="688" height="118" rx="8" style="fill:var(--c-warn);fill-opacity:.05;stroke:var(--c-warn);stroke-width:1.3"/>
<text x="28" y="209" text-anchor="start" style="fill:var(--c-warn);font-size:11.5px;font-weight:700">ÇALIŞMA ZAMANI · İSTEK YOLU</text>
<rect x="28" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="80" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Sohbet arayüzü</text>
<rect x="66" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="80.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B5</text>
<line x1="132" y1="241" x2="138" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="140" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="192" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Gateway</text>
<rect x="178" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="192.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B9</text>
<line x1="244" y1="241" x2="250" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="252" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="304" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Etiket, güven τ</text>
<rect x="290" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="304.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B5</text>
<line x1="356" y1="241" x2="362" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="364" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="416" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Guardrail&#x27;ler</text>
<rect x="402" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="416.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B6</text>
<line x1="468" y1="241" x2="474" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="476" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<text x="528" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Retrieval, bellek</text>
<rect x="514" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="528.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B7</text>
<line x1="580" y1="241" x2="586" y2="241" marker-end="url(#p12a-arr)" style="stroke:var(--c-text-mute);stroke-width:1.5"/>
<rect x="588" y="216" width="104" height="50" rx="6" style="fill:var(--c-surface);stroke:var(--c-warn);stroke-width:1.2"/>
<text x="640" y="237" text-anchor="middle" style="fill:var(--c-text);font-size:11.5px">Model sunucusu</text>
<rect x="626" y="245" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="640.0" y="256.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B9</text>
<text x="28" y="296" text-anchor="start" style="fill:var(--c-text-mute);font-size:11px">Her model çağrısı, guard çağrısı ve yedek yol gateway&#x27;den geçer; her parça kaynağını ve güvenini taşır.</text>
<rect x="16" y="320" width="688" height="70" rx="8" style="fill:var(--c-danger);fill-opacity:.05;stroke:var(--c-danger);stroke-width:1.3"/>
<text x="28" y="337" text-anchor="start" style="fill:var(--c-danger);font-size:11.5px;font-weight:700">ÇALIŞMA ZAMANI · İŞLEM YOLU</text>
<rect x="28" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface);stroke:var(--c-danger);stroke-width:1.2"/>
<rect x="156" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B8</text>
<text x="38" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Yürütücü (PEP/PDP)</text>
<rect x="198" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B8</text>
<text x="208" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Parametreye bağlı onay</text>
<rect x="368" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B8</text>
<text x="378" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Bütçe, idempotency</text>
<rect x="538" y="342" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="346" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="357.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B8</text>
<text x="548" y="373" text-anchor="start" style="fill:var(--c-text);font-size:12px">Araçlar, MCP</text>
<rect x="16" y="400" width="688" height="68" rx="8" style="fill:var(--c-success);fill-opacity:.05;stroke:var(--c-success);stroke-width:1.3"/>
<text x="28" y="417" text-anchor="start" style="fill:var(--c-success);font-size:11.5px;font-weight:700">OPERASYON</text>
<rect x="28" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="156" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="170.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B11</text>
<text x="38" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Denetim olayları</text>
<rect x="198" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="326" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="340.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B11</text>
<text x="208" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Maskeli telemetri</text>
<rect x="368" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="496" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="510.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B11</text>
<text x="378" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Kanarya, alarm</text>
<rect x="538" y="420" width="160" height="40" rx="6" style="fill:var(--c-surface-2);stroke:var(--c-border);stroke-width:1.2"/>
<rect x="666" y="424" width="28" height="15" rx="5" style="fill:var(--c-accent);fill-opacity:.16;stroke:var(--c-accent);stroke-width:1.3"/><text x="680.0" y="435.2" text-anchor="middle" style="fill:var(--c-text);font-size:10.5px;font-family:var(--font-mono)">B11</text>
<text x="548" y="451" text-anchor="start" style="fill:var(--c-text);font-size:12px">Runbook, kaldıraç</text>
<!-- Bottom Educational Banner -->
<rect x="16" y="468" width="688" height="58" rx="6" style="fill:var(--c-accent);fill-opacity:.07;stroke:var(--c-accent);stroke-width:1.2"/>
<text x="28" y="487" text-anchor="start" style="fill:var(--c-accent);font-size:11px;font-weight:700;letter-spacing:.02em">💡 MİMARİ SİSTEM BÜTÜNLÜĞÜ: 5 BANTTA 12 BÖLÜMÜN SENKRONİZASYONU</text>
<text x="28" y="504" text-anchor="start" style="fill:var(--c-text);font-size:10.5px">Yönetişim risk iştahını belirler, kurulum zamanı doğrulanabilir artefaktlar üretir, istek yolu güven etiketlerini taşır.</text>
<text x="28" y="519" text-anchor="start" style="fill:var(--c-text-mute);font-size:10px">İşlem yolu deterministik PEP ile icrayı sınırlar, operasyon ise adli kanıt üretir. Bantlardan biri eksik olduğunda sistem güvenliği çöker.</text>
</svg>

Figürü dıştan içe okuyalım:

- **Yönetişim (Bölüm 1, 12):** Varlıklar, sahipler ve RACI; risk kabulü; kanıt dizini; canlıya geçiş kapısı.
- **Kurulum zamanı (Bölüm 2, 3, 4, 10):** Tehdit kaydı; veri kökeni, PII işleme ve silme; artefakt kabulü, imzalama ve yayın akışı; değerlendirme ve red teaming.
- **Çalışma zamanı, istek yolu (Bölüm 5, 6, 7, 9):** Her yolda kota ve guard içeren gateway; kaynak etiketli bağlam ve güven yayılımı; dedektörler ve politika motoru; karantinalı bellekle yetkiye duyarlı retrieval; akışta geride tutma.
- **Çalışma zamanı, işlem yolu (Bölüm 8):** Kimlik bilgilerini tutan yürütücü; yetki kesişimi; parametre doğrulaması; parametrelere bağlı onay; bütçeler; idempotency.
- **Operasyon (Bölüm 11):** Eksiksiz denetim olayları, maskelenmiş telemetri, kanaryalar, kontrol altına alma kaldıraçları ve runbook.

---

## 6. Riske dayalı bir yayın kapısı

Canlıya geçmek belirsizlik altında verilen bir karardır. Kapı, önceki bölümlerin çıktılarını iş birimi sahibinin imzalayabileceği bir karar kuralına dönüştürür. $i$ tehdidinin kalan riskine $R'_i$ (Bölüm 2), üzerinde anlaşılan risk iştahına $A$, imzalı bir kabule $\text{kabul}_i$, zarar oranının %95 üst sınırına $\bar{h}$ (Bölüm 10) ve onun sınırına $\bar{h}_{\max}$ dersek:

$$
\text{Geç} = \Big( \bigwedge_{i} \big[\, R'_i \le A \;\lor\; \text{kabul}_i \,\big] \Big) \land \big[\, \bar{h} \le \bar{h}_{\max} \,\big] \land T_{\text{kabul}} \land E_{\text{tam}}
$$

Şöyle okuyun: $\bigwedge$ "her tehdit için" demektir; her tehdidin ya kalan riski iştahın içindedir ya da sahibi tarafından açıkça kabul edilmiştir. Buna ek olarak ölçülen zarar oranının üst sınırı kendi sınırının içinde olmalı, bütün kabul testleri $T_{\text{kabul}}$ (Bölüm 9) geçmeli ve kanıt dizini $E_{\text{tam}}$, yayımlanan sürüm için eksiksiz olmalıdır.

İkinci bölümdeki tehdit kaydı ve $A = 6$ iştahıyla **yürüyüş**:

```text
Kalan puanlar R':     T1 = 4   T2 = 4   T3 = 4   T4 = 4   T5 = 5   T6 = 4
İştah A = 6:          bütün R' ≤ 6                            → koşul istisnasız sağlanır

Değerlendirme (Bölüm 10): 1.200 saldırı vakasında zarar oranı 0
                      üst sınır ≈ 3 / 1.200 = 0,0025 (%0,25)
                      sınır h̄_max = %0,5                      → %0,25 ≤ %0,5, sağlanır

Kabul testleri S-1..S-10 (Bölüm 9):   rc-2026-11.2 yayın adayında hepsi geçer
Kanıt dizini:         rc-2026-11.2 için eksiksiz (kapsama %87,5; sahibi belli 16 açık madde)

Geç = doğru ∧ doğru ∧ doğru ∧ doğru = doğru   → iş birimi sahibi imzalar, gözden geçirme tarihi konur
```

T5 (kurcalanmış model artefaktı) kontrollerden sonra 8 puan alsaydı, kapı ya daha fazla kontrol ya da iş birimi sahibinin o belirli riski bir gözden geçirme tarihiyle imzalı olarak kabul etmesini isterdi; sessiz bir geçiş değil. Yayın adayı da değerlendirmeden sonra modeli değiştirseydi, gerileme takımı yeniden çalışana kadar $E_{\text{tam}}$ yanlış olurdu.

Kapı, sistemin "güvenli" olduğunu iddia etmez. Bu sürüm ve bu kapsam için bilinen risklerin, onlardan hesap veren kişilerin üzerinde anlaştığı iştahın içinde olduğunu, incelenebilir kanıtlara dayanarak kaydeder.

---

## 7. Pratik çıktı: Eşleştirme matrisi, kanıt dizini ve canlıya geçiş kontrol listesi

### 7.1. Standart eşleştirme matrisi (kesit)

| Seri kontrolü | ETSI EN 304 223 evresi | NIST AI RMF | OWASP LLM 2026 / Ajan | AISVS 1.0 | ATLAS | Diğer |
| --- | --- | --- | --- | --- | --- | --- |
| Varlıklar, sahipler, RACI (B1) | Tasarım | Govern, Map | – | – | – | ISO/IEC 42001 rolleri |
| Tehdit kaydı (B2) | Tasarım | Map | Bütün kategoriler kontrol listesi olarak | – | Satır başına teknik kimliği | ISO/IEC 23894 süreci |
| Veri kökeni, PII, silme (B3) | Geliştirme, kullanımdan kaldırma | Map, Manage | LLM02, LLM05 | C1 | `AML.T0020` | ISO/IEC 27701, KVKK, GDPR |
| Artefakt kabulü, imzalama, yayın (B4) | Geliştirme, dağıtım | Manage | LLM04, ASI04 | C3, C6 | `AML.T0010` | SLSA 1.2, NIST SP 800-218A |
| Güven yayılımı, spotlighting (B5) | Dağıtım | Manage | LLM01, ASI01 | C2, C9 | `AML.T0051` | – |
| Dedektörler ve politika motoru (B6) | Dağıtım | Measure, Manage | LLM01, LLM02 | C2, C7 | – | – |
| Yetkiye duyarlı retrieval, bellek kapısı (B7) | Dağıtım | Manage | LLM09, ASI06 | C8 | `AML.T0070`, `AML.T0080` | – |
| Yürütücü, onaylar, bütçeler (B8) | Dağıtım | Manage | LLM03, LLM06, ASI02, ASI03 | C5, C9, C10 | `AML.T0086` | MCP spesifikasyonu, RFC 8693 |
| Gateway, izolasyon, akış (B9) | Dağıtım | Manage | LLM06, LLM10 | C4 | `AML.T0029`, `AML.T0034` | OWASP API Top 10, ASVS 5.0 |
| Değerlendirme ve red teaming (B10) | Bakım | Measure | Hepsi | C11 | – | – |
| Denetim, izleme, müdahale (B11) | Bakım | Manage | – | C12 | Alarmlarda etiket | NIST SP 800-61r3, CSF 2.0 |

Matris kontrolleri *kategorilerle ve evrelerle* eşler; bu belgelerin herkese açık tarifleri de ancak bunu destekler. ISO metinleriyle madde düzeyindeki eşlemeler ancak metinlerin kendisi incelendikten sonra eklenir.

### 7.2. Kanıt dizini mimarisi (Evidence Index)

```text
evidence-index/rc-2026-11.2/
  manifest.json                 # sistem sürümleri: model, adaptör, şablon, prompt, politika, gateway, araçlar
  requirements/aisvs-l2.csv     # gereksinim → uygulanabilir mi? gerekçe → kontrol no'ları → test no'ları → durum
  controls/C-08.md              # tasarım, gerçekleştirme referansı, sahip
  tests/PI-1/run-7731.json      # test tanımı, veri seti hash'i, sonuçlar, Wilson aralığı, hakem κ
  acceptance/S-1..S-10.json     # sunum kabul sonuçları
  evaluation/report-v4.pdf      # Bölüm 10 raporu
  risk/register.csv             # tehditler, R, R', kabul imzaları, gözden geçirme tarihleri
  approvals/go-live.sig         # iş birimi sahibi ve güvenlik imzası
```

Her dosya yayın adayının kimliğini taşır ve değiştirilemez biçimde saklanır; böylece `rc-2026-11.2` dizini bir yıl sonra da aynı şeyi söyler.

### 7.3. Programatik canlıya geçiş doğrulama motoru

Aşağıdaki Python betiği, kanıt dizinindeki verileri okuyarak yukarıda tanımlanan $\text{Geç}$ karar kuralını CI/CD hattında nesnel olarak çalıştırır:

```python
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class ReleaseCandidateEvidence:
    release_id: str
    manifest_hash: str
    residual_risks: Dict[str, int]
    risk_appetite: int
    signed_acceptances: List[str]
    harm_rate_upper_bound: float
    max_harm_limit: float
    acceptance_tests_passed: bool
    unowned_gap_count: int      # sahibi, planı ya da imzalı kabulü OLMAYAN açık maddeler

def evaluate_release_gate(evidence: ReleaseCandidateEvidence) -> Tuple[bool, List[str]]:
    """Yayın adayı kanıt dizinini nesnel risk ve uyum kapısına göre doğrular."""
    violations = []

    # 1. Kalan risk iştahı kontrolü
    for threat_id, score in evidence.residual_risks.items():
        if score > evidence.risk_appetite and threat_id not in evidence.signed_acceptances:
            violations.append(
                f"Kabul edilmemiş yüksek risk: {threat_id} (Puan={score} > İştah={evidence.risk_appetite})"
            )

    # 2. Ölçülen zarar oranı üst sınır kontrolü
    if evidence.harm_rate_upper_bound > evidence.max_harm_limit:
        violations.append(
            f"Zarar oranı üst sınırı aşıldı: %{evidence.harm_rate_upper_bound*100:.2f} > %{evidence.max_harm_limit*100:.2f}"
        )

    # 3. Gateway ve sunum kabul testleri
    if not evidence.acceptance_tests_passed:
        violations.append("Sunucu ve gateway kabul testleri (S-1..S-10) başarısız!")

    # 4. Kanıt dizini eksiksizlik kontrolü
    if evidence.unowned_gap_count > 0:
        violations.append(f"Kanıt dizininde sahibi ya da kabulü olmayan {evidence.unowned_gap_count} açık madde var.")

    passed = len(violations) == 0
    return (passed, violations)


# 6. bölümdeki yürüyüş
rc = ReleaseCandidateEvidence(
    release_id="rc-2026-11.2", manifest_hash="sha256:4f8a",
    residual_risks={"T1": 4, "T2": 4, "T3": 4, "T4": 4, "T5": 5, "T6": 4},
    risk_appetite=6, signed_acceptances=[],
    harm_rate_upper_bound=3 / 1200, max_harm_limit=0.005,
    acceptance_tests_passed=True,
    unowned_gap_count=0,          # 16 open items, each with an owner and a plan
)
print("rc-2026-11.2  :", evaluate_release_gate(rc))

# T5 8 puan alsaydı ve iki açık maddenin sahibi olmasaydı kapının söyleyeceği
from dataclasses import replace
worse = replace(rc, residual_risks={**rc.residual_risks, "T5": 8}, unowned_gap_count=2)
print("T5 = 8, gaps  :", evaluate_release_gate(worse))
print("T5 accepted   :", evaluate_release_gate(replace(worse, signed_acceptances=["T5"], unowned_gap_count=0)))
```

Konsol çıktısı:

```text
rc-2026-11.2  : (True, [])
T5 = 8, gaps  : (False, ['Kabul edilmemiş yüksek risk: T5 (Puan=8 > İştah=6)', 'Kanıt dizininde sahibi ya da kabulü olmayan 2 açık madde var.'])
T5 accepted   : (True, [])
```

### 7.4. Canlıya geçiş kontrol listesi

1. Varlık envanteri ve RACI günceldir; her satırın tek bir hesap veren sahibi vardır.
2. Tehdit kaydı bu yayın için gözden geçirilmiştir; her tehdidin bir kontrolü, bir testi ve bir kalan puanı vardır.
3. Veri envanteri, PII akışı ve silme testi geçer; eğitim ya da değerlendirme setlerinde ham kişisel veri yoktur.
4. AI-BOM'daki bütün artefaktlar imzalıdır ve yükleme anında doğrulanır; önceki onaylı set dağıtılabilir durumdadır.
5. Güven yayılımı, yürütücü doğrulaması, onaylar, bütçeler ve idempotency, yan etkisi olan her araçta test edilmiştir.
6. Retrieval yetkileri sorgunun içinde uygular; güvenilmeyen bağlamlardan gelen bellek yazmaları karantinaya alınır.
7. Gateway kabul testleri S-1'den S-10'a geçer; hiçbir yol guard'ları atlamaz; akışta geride tutma etkindir.
8. Bu yayın adayı için dil başına, aralıklı değerlendirme raporu vardır; gerileme kapıları geçer.
9. Denetim olayları eksiksizdir; içerik kaydı kapalıdır; kanaryalar yerindedir; kontrol altına alma kaldıraçlarının tatbikatı yapılmıştır.
10. Kanıt dizini bu sürüm için eksiksizdir; kapsama oranı ve açık maddeler belirtilmiştir.
11. Dışa dönük iddialar (web sitesi, sözleşmeler, soru formları) yalnızca Kısım 3'teki dürüst biçimleri kullanır.
12. İş birimi sahibi kapı kararını bir gözden geçirme tarihiyle imzalar.

### 7.5. Mühendisliğin iki gözü: Kurulum ve çalışma zamanı

**Kurulum zamanı:** Eşleştirme matrisi, gereksinim uygulanabilirlik listesi, kanıt dizini yapısı ve kapı kuralı bir kez yazılır ve her değişiklikle güncellenir.

**Çalışma zamanı:** Kanıt; gerileme takımı, kabul testleri ve denetim izi tarafından sürekli üretilir. Böylece her yayın adayının dizini, bir denetimden önceki hafta elle yazılmak yerine kendiliğinden bir araya gelir.

---

## Bütün hikâye altı satırda

- Yönetim sistemi standartları, rehberler, çerçeveler, taksonomiler, farkındalık listeleri ve doğrulama standartları farklı türden belgelerdir ve her biri farklı bir iddiaya izin verir.
- ISO/IEC 42001 kurumu yönetir, 23894 risk yönetimine, 42005 etki değerlendirmesine, 27090 ise 27001 ve 42001 ile birlikte AI güvenlik kontrollerine rehberlik eder.
- Herkese açık bir özetten madde düzeyinde uyum iddia etmeyin; "güvenli" kelimesini kapsamlı, sürümlü ve ölçülmüş bir ifadeyle değiştirin.
- Kanıt zinciri gereksinimi, kontrolü, gerçekleştirmeyi, testi, kanıtı ve kalan riski birbirine bağlar; kapsama oranı yalnızca yayımlanan sürüm için eksiksiz zincirleri sayar.
- Yürüyüşte uygulanabilir 128 AISVS Seviye 2 gereksiniminin 112'si doğrulandı; kapsama %87,5 ve sahibi belli 16 açık madde.
- Canlıya geçiş imzalı bir karardır: kalan riskler iştahın içinde ya da açıkça kabul edilmiş, zarar oranı sınırı aşılmamış, kabul testleri geçmiş ve kanıt eksiksiz.

---

## Terimler sözlüğü

- **Yönetim sistemi standardı:** Tanımlı bir kapsam için belgelendirilebilen, kurumsal gereksinimlerden (roller, süreçler, iyileştirme) oluşan standart.
- **Rehber standart:** Bir şeyin nasıl yapılacağını öneren ve ona karşı belgelendirme yapılamayan standart.
- **Çerçeve (framework):** NIST AI RMF gibi, işi düzenlemek için işlevlerden ve çıktılardan oluşan bir yapı.
- **Taksonomi:** NIST AI 100-2 ya da MITRE ATLAS gibi, saldırılar ve önlemler için ortak sınıflandırma ve terimler.
- **Farkındalık listesi:** OWASP Top 10 gibi, uyum için değil önceliklendirme için kullanılan, sıralı önemli risk kategorileri listesi.
- **Doğrulama standardı:** Bir sistemin karşısında doğrulanabileceği, çoğu zaman seviyelere ayrılmış test edilebilir gereksinimler kümesi.
- **Kanıt zinciri:** Kaynak gereksinimden kontrole, gerçekleştirmeye, teste, kanıta ve kalan riske uzanan bağlantılı dizi.
- **Kapsama oranı (coverage):** Uygulanabilir gereksinimlerin, yayımlanan sürüm için eksiksiz kanıt zincirine sahip olan payı.
- **Risk iştahı (risk appetite):** Hesap veren bir sahibin, özel bir imza gerektirmeden kabul etmeye hazır olduğu kalan risk düzeyi.
- **Kanıt dizini (evidence index):** Bir yayın kararını destekleyen belgelerin ve test sonuçlarının sürümlü ve değiştirilemez koleksiyonu.

---

## Daha derine inmek için

- Önceki bölüm: [Secure AI by Design (11): AI Operasyonlarında İzlenebilirlik ve Olay Müdahalesi](post.html?slug=ai-izlenebilirlik-ve-olay-mudahalesi)
- Serinin başı: [Secure AI by Design (1): Yapay Zekâ Güvenliği Tasarımda Başlar](post.html?slug=yapay-zeka-guvenligi-tasarimda-baslar)
- ISO/IEC, [*42001:2023 Yapay zekâ yönetim sistemi*](https://www.iso.org/standard/42001), [*23894:2023 Risk yönetimi rehberi*](https://www.iso.org/standard/77304.html), [*42005:2025 AI sistemi etki değerlendirmesi*](https://www.iso.org/standard/42005) ve [*27090:2026*](https://www.iso.org/standard/27090); BSI, [*BS ISO/IEC 27090:2026 duyurusu*](https://www.bsigroup.com/en-GB/insights-and-media/media-centre/press-releases/2026/october/new-international-guidance-to-help-organizations-protect-ai-from-security-threats/) (Ekim 2026).
- ETSI, [*EN 304 223*](https://www.etsi.org/newsroom/press-releases/2627-etsi-releases-world-leading-standard-for-securing-ai) (Ocak 2026).
- NIST, [*AI RMF 1.0*](https://www.nist.gov/itl/ai-risk-management-framework), [*AI 600-1*](https://doi.org/10.6028/NIST.AI.600-1), [*AI 100-2 E2025*](https://csrc.nist.gov/pubs/ai/100/2/e2025/final), [*SP 800-218A*](https://csrc.nist.gov/pubs/sp/800/218/a/final), [*SP 800-61r3*](https://csrc.nist.gov/pubs/sp/800/61/r3/final).
- OWASP, [*AISVS 1.0*](https://owasp.github.io/www-project-artificial-intelligence-security-verification-standard-aisvs-docs/), [*ASVS 5.0*](https://asvs.owasp.org/), [*Top 10 for LLM Applications 2026*](https://github.com/GenAI-Security-Project/GenAI-LLM-Top10), [*Top 10 for Agentic Applications*](https://genai.owasp.org/2025/12/09/owasp-top-10-for-agentic-applications-the-benchmark-for-agentic-security-in-the-age-of-autonomous-ai/), [*Agent Control Standard*](https://genai.owasp.org/resource/agent-control-standard-acs/).
- MITRE, [*ATLAS*](https://atlas.mitre.org/); SLSA, [*v1.2*](https://slsa.dev/spec/v1.2/).